//! Tile-bounded work for the resident numerical producer.
//!
//! Implicit domains visit only the current rectangle, in bounded candidate
//! chunks. Explicit occurrences are indexed once by four separate integer keys;
//! a tile locates its own interval by binary search, without rescanning M pairs.
use std::{mem::size_of, sync::Arc};

use cudarc::driver::{
    CudaFunction, CudaModule, CudaSlice, CudaStream, DeviceRepr, DriverError, LaunchConfig,
    PushKernelArg,
};
use seqblaze_core::{
    cancel::{self, Site},
    validate::EngineError,
};

use crate::{
    device::{guard, Device},
    interop::{allocate, scan, DeviceColumn, DeviceDType, DeviceLease, DeviceOutput},
    resident::{
        ResidentChunk, ResidentField, ResidentFieldSpec, ResidentProducer, ResidentSpace,
        ResidentWork,
    },
    resident_indices::lexicographic_order,
    resident_producer::NativeProducer,
    session::{failure_ordinal, ptr},
};

const THREADS: usize = 256;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TileDomain {
    Zip,
    Cross,
    SymmetricWithin { self_pairs: bool },
    WithinAsGiven { self_pairs: bool },
    WithinBoth { self_pairs: bool },
    Explicit,
}
impl TileDomain {
    fn abi(self) -> (u64, u64) {
        match self {
            Self::Zip => (0, 0),
            Self::Cross => (1, 0),
            Self::SymmetricWithin { self_pairs } => (2, u64::from(self_pairs)),
            Self::WithinAsGiven { self_pairs } => (3, u64::from(self_pairs)),
            Self::WithinBoth { self_pairs } => (4, u64::from(self_pairs)),
            Self::Explicit => (5, 0),
        }
    }
    fn work_kind(self) -> u64 {
        match self {
            Self::Zip => 0,
            Self::Explicit => 1,
            Self::Cross => 2,
            Self::SymmetricWithin { .. } | Self::WithinAsGiven { .. } => 3,
            Self::WithinBoth { .. } => 4,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
struct Rect {
    row: u64,
    col: u64,
    rows: u64,
    cols: u64,
    domain: u64,
    self_pairs: u64,
}
// SAFETY: six repr(C) integer words, matching SbTileRect exactly.
unsafe impl DeviceRepr for Rect {}
const _: () = assert!(size_of::<Rect>() == 48);

pub(crate) struct Kernels {
    seed: CudaFunction,
    order: CudaFunction,
    bounds: CudaFunction,
    classify: CudaFunction,
    count: CudaFunction,
    compact: CudaFunction,
    copy_flags: CudaFunction,
    gather: CudaFunction,
    fixed_block_supported: bool,
}
impl Kernels {
    pub(crate) fn load(module: &Arc<CudaModule>) -> Result<Self, DriverError> {
        let mut result = Self {
            seed: module.load_function("sb_tile_seed")?,
            order: module.load_function("sb_tile_order")?,
            bounds: module.load_function("sb_tile_bounds")?,
            classify: module.load_function("sb_tile_classify")?,
            count: module.load_function("sb_tile_count")?,
            compact: module.load_function("sb_tile_compact")?,
            copy_flags: module.load_function("sb_tile_copy_flags")?,
            gather: module.load_function("sb_tile_gather")?,
            fixed_block_supported: true,
        };
        for function in [
            &result.seed,
            &result.order,
            &result.bounds,
            &result.classify,
            &result.count,
            &result.compact,
            &result.copy_flags,
            &result.gather,
        ] {
            result.fixed_block_supported &= function.max_threads_per_block()? >= THREADS as i32;
        }
        Ok(result)
    }
}
fn bad(text: &str) -> EngineError {
    EngineError::Device(format!("resident tiles: {text}"))
}
fn failed(lease: &DeviceLease, what: &str, error: DriverError) -> EngineError {
    failure_ordinal(lease.device(), what, error)
}
fn grid(kernels: &Kernels, n: usize) -> Result<LaunchConfig, EngineError> {
    if !kernels.fixed_block_supported {
        return Err(bad("the loaded kernels cannot launch a 256-thread block"));
    }
    Ok(LaunchConfig {
        grid_dim: (n.div_ceil(THREADS).clamp(1, 65_535) as u32, 1, 1),
        block_dim: (THREADS as u32, 1, 1),
        shared_mem_bytes: 0,
    })
}
/// Scratch addresses in the tile ABI bypass typed borrow events. Retain all
/// allocations through completion, including early errors and failed syncs.
struct Pending<T> {
    value: Option<T>,
    stream: Arc<CudaStream>,
    device: Device,
}
impl<T> Pending<T> {
    fn new(lease: &DeviceLease, value: T) -> Self {
        Self {
            value: Some(value),
            stream: Arc::clone(lease.stream()),
            device: lease.device().clone(),
        }
    }
    fn get(&self) -> &T {
        self.value.as_ref().expect("retained tile scratch")
    }
    fn get_mut(&mut self) -> &mut T {
        self.value.as_mut().expect("retained tile scratch")
    }
}
impl<T> Drop for Pending<T> {
    fn drop(&mut self) {
        if guard(|| {
            crate::interop::execution_stream::retire_in_drop(&self.stream)
                .map_err(|e| failure_ordinal(&self.device, "retiring resident tile raw owners", e))
        })
        .is_err()
        {
            if let Some(value) = self.value.take() {
                std::mem::forget(value);
            }
        }
    }
}
fn control<const N: usize>(
    lease: &DeviceLease,
    values: &CudaSlice<u64>,
) -> Result<[u64; N], EngineError> {
    let mut out = [0u64; N];
    lease
        .execution()
        .memcpy_dtoh(&values.slice(0..N), &mut out[..])
        .map_err(|e| failed(lease, "reading fixed-size tile control metadata", e))?;
    lease.synchronize()?;
    Ok(out)
}
fn publish(output: DeviceOutput) -> Result<DeviceColumn, EngineError> {
    // SAFETY: every call queues complete initialization on the allocation stream.
    unsafe { output.finish() }
}
fn zeros(
    lease: &DeviceLease,
    words: usize,
) -> Result<Pending<crate::arena::Scratch<u64>>, EngineError> {
    let mut value = Pending::new(lease, allocate(lease, words)?);
    match words {
        1 | 2 => lease.execution().fill(&mut **value.get_mut(), words, 0),
        _ => return Err(bad("invalid fixed-size tile control record")),
    }
    .map_err(|e| failed(lease, "initializing tile control metadata", e))?;
    Ok(value)
}
fn checked_u64(value: usize) -> Result<u64, EngineError> {
    u64::try_from(value).map_err(|_| EngineError::Overflow("tile geometry exceeds u64".into()))
}

struct ExplicitIndex {
    tile_rows: DeviceColumn,
    tile_cols: DeviceColumn,
    i: DeviceColumn,
    j: DeviceColumn,
    first: DeviceColumn,
    n: usize,
    duplicates: bool,
}
/// One immutable planning index per iterator. It never owns a dense matrix or
/// expanded implicit domain; only explicit caller-provided indices scale as M.
pub struct TilePlan {
    shape: [usize; 2],
    tile_shape: [usize; 2],
    domain: TileDomain,
    explicit: Option<ExplicitIndex>,
    stream: u64,
    ordinal: usize,
    chunk: usize,
}
impl TilePlan {
    pub fn new(
        lease: &mut DeviceLease,
        work: &ResidentWork,
        domain: TileDomain,
        tile_shape: [usize; 2],
    ) -> Result<Self, EngineError> {
        guard(|| {
            if tile_shape.contains(&0) {
                return Err(EngineError::Parameter(
                    "tile dimensions and threads must be positive".into(),
                ));
            }
            for value in [work.n_a, work.n_b, tile_shape[0], tile_shape[1]] {
                checked_u64(value)?;
            }
            if work.space.kind != domain.work_kind() {
                return Err(bad("tile domain disagrees with the validated work kind"));
            }
            if matches!(
                domain,
                TileDomain::SymmetricWithin { .. }
                    | TileDomain::WithinAsGiven { .. }
                    | TileDomain::WithinBoth { .. }
            ) && work.n_a != work.n_b
            {
                return Err(EngineError::Pairing(
                    "oriented within requires a square table shape".into(),
                ));
            }
            let explicit = if domain == TileDomain::Explicit {
                Some(build_index(lease, work, tile_shape)?)
            } else {
                None
            };
            Ok(Self {
                shape: [work.n_a, work.n_b],
                tile_shape,
                domain,
                explicit,
                stream: lease.stream_handle(),
                ordinal: lease.device().info().ordinal,
                chunk: crate::launch::settings().pairs_per_chunk.clamp(1, 65536) as usize,
            })
        })
    }
    pub fn shape(&self) -> [usize; 2] {
        self.shape
    }
    pub fn tile_shape(&self) -> [usize; 2] {
        self.tile_shape
    }

    /// Bind one complete clipped grid tile. NativeProducer owns the numerical
    /// pack for the iterator; this short borrow calls only its emit_space path.
    pub fn producer<'p, 'b, 'r>(
        &'p self,
        lease: &DeviceLease,
        region: [usize; 4],
        native: &'b mut NativeProducer<'r>,
    ) -> Result<TileProducer<'p, 'b, 'r>, EngineError> {
        guard(|| {
            self.check_lease(lease)?;
            let [row, col, rows, cols] = region;
            if row >= self.shape[0]
                || col >= self.shape[1]
                || row % self.tile_shape[0] != 0
                || col % self.tile_shape[1] != 0
                || rows != self.tile_shape[0].min(self.shape[0] - row)
                || cols != self.tile_shape[1].min(self.shape[1] - col)
            {
                return Err(EngineError::Parameter(
                    "dense tile geometry exceeds pair-space shape or grid".into(),
                ));
            }
            let (domain, self_pairs) = self.domain.abi();
            let rect = Rect {
                row: checked_u64(row)?,
                col: checked_u64(col)?,
                rows: checked_u64(rows)?,
                cols: checked_u64(cols)?,
                domain,
                self_pairs,
            };
            let (start, end) = if let Some(index) = &self.explicit {
                let mut bounds = zeros(lease, 2)?;
                if index.n != 0 {
                    let kernels = &lease.device().0.resident_tiles;
                    let (i, j, n) = (
                        index.tile_rows.address(),
                        index.tile_cols.address(),
                        checked_u64(index.n)?,
                    );
                    let (tr, tc) = (
                        checked_u64(row / self.tile_shape[0])?,
                        checked_u64(col / self.tile_shape[1])?,
                    );
                    let output = ptr(bounds.get(), lease.stream());
                    let mut launch = lease.execution().launch_builder(&kernels.bounds);
                    launch
                        .arg(&i)
                        .arg(&j)
                        .arg(&n)
                        .arg(&tr)
                        .arg(&tc)
                        .arg(&output);
                    // SAFETY: sorted immutable coarse keys cover n entries; only
                    // the two retained control words are written by one thread.
                    unsafe { launch.launch(grid(kernels, 1)?) }
                        .map_err(|e| failed(lease, "locating an explicit tile interval", e))?;
                }
                let range = control::<2>(lease, bounds.get_mut())?;
                let start =
                    usize::try_from(range[0]).map_err(|_| bad("tile interval exceeds platform"))?;
                let end =
                    usize::try_from(range[1]).map_err(|_| bad("tile interval exceeds platform"))?;
                if start > end || end > index.n {
                    return Err(bad("invalid explicit tile interval"));
                }
                (start, end)
            } else if self.domain == TileDomain::Zip {
                (
                    0,
                    row.saturating_add(rows)
                        .min(col.saturating_add(cols))
                        .saturating_sub(row.max(col)),
                )
            } else {
                (
                    0,
                    rows.checked_mul(cols).ok_or_else(|| {
                        EngineError::Overflow("tile shape product exceeds platform limit".into())
                    })?,
                )
            };
            Ok(TileProducer {
                plan: self,
                native,
                rect,
                start,
                end,
                cursor: start,
                counts: TileCounts::default(),
                coordinates: None,
            })
        })
    }
    fn check_lease(&self, lease: &DeviceLease) -> Result<(), EngineError> {
        if self.stream != lease.stream_handle() || self.ordinal != lease.device().info().ordinal {
            return Err(bad("a tile plan must use its original device and stream"));
        }
        Ok(())
    }
}

fn build_index(
    lease: &mut DeviceLease,
    work: &ResidentWork,
    tile_shape: [usize; 2],
) -> Result<ExplicitIndex, EngineError> {
    let n = usize::try_from(work.space.n).map_err(|_| bad("explicit count exceeds platform"))?;
    if n != work.original_items {
        return Err(bad(
            "Tiles needs original explicit occurrences, before sink deduplication",
        ));
    }
    let end = work
        .space
        .k0
        .checked_add(work.space.n)
        .ok_or_else(|| bad("explicit range overflows"))?;
    for address in [work.space.li, work.space.lj] {
        let owner = work
            .owners
            .iter()
            .find(|owner| {
                owner.address() == address
                    && owner.dtype() == DeviceDType::I64
                    && owner.shape().len() == 1
                    && owner.shape()[0] as u128 >= end as u128
            })
            .ok_or_else(|| bad("explicit tile indices need retained normalized I64 owners"))?;
        lease.retain(owner.internal_array())?;
    }
    let left = lease.output(DeviceDType::I64, vec![n])?;
    let right = lease.output(DeviceDType::I64, vec![n])?;
    let row = lease.output(DeviceDType::U64, vec![n])?;
    let col = lease.output(DeviceDType::U64, vec![n])?;
    if n != 0 {
        let kernels = &lease.device().0.resident_tiles;
        let (li, lj, first, count) = (work.space.li, work.space.lj, work.space.k0, work.space.n);
        let (height, width) = (checked_u64(tile_shape[0])?, checked_u64(tile_shape[1])?);
        let (oi, oj, tr, tc) = (
            left.address(),
            right.address(),
            row.address(),
            col.address(),
        );
        let mut launch = lease.execution().launch_builder(&kernels.seed);
        launch
            .arg(&li)
            .arg(&lj)
            .arg(&first)
            .arg(&count)
            .arg(&height)
            .arg(&width)
            .arg(&oi)
            .arg(&oj)
            .arg(&tr)
            .arg(&tc);
        // SAFETY: validated retained normalized source indices cover first..end;
        // all four independent outputs contain n eight-byte slots.
        unsafe { launch.launch(grid(kernels, n)?) }
            .map_err(|e| failed(lease, "building explicit tile keys", e))?;
    }
    let (left, right, row, col) = (
        publish(left)?,
        publish(right)?,
        publish(row)?,
        publish(col)?,
    );
    let order = lexicographic_order(
        lease,
        &[row.clone(), col.clone(), left.clone(), right.clone()],
    )?;
    let out_i = lease.output(DeviceDType::I64, vec![n])?;
    let out_j = lease.output(DeviceDType::I64, vec![n])?;
    let out_row = lease.output(DeviceDType::U64, vec![n])?;
    let out_col = lease.output(DeviceDType::U64, vec![n])?;
    let first = lease.output(DeviceDType::U64, vec![n])?;
    let duplicate_count = zeros(lease, 1)?;
    if n != 0 {
        let kernels = &lease.device().0.resident_tiles;
        let (perm, count, li, lj, tr, tc) = (
            order.address(),
            n as u64,
            left.address(),
            right.address(),
            row.address(),
            col.address(),
        );
        let (oi, oj, orow, ocol, flags, duplicates) = (
            out_i.address(),
            out_j.address(),
            out_row.address(),
            out_col.address(),
            first.address(),
            ptr(duplicate_count.get(), lease.stream()),
        );
        let mut launch = lease.execution().launch_builder(&kernels.order);
        launch
            .arg(&perm)
            .arg(&count)
            .arg(&li)
            .arg(&lj)
            .arg(&tr)
            .arg(&tc)
            .arg(&oi)
            .arg(&oj)
            .arg(&orow)
            .arg(&ocol)
            .arg(&flags)
            .arg(&duplicates);
        // SAFETY: order is a complete permutation of n retained source entries;
        // every output is initialized exactly once and the summary has one word.
        unsafe { launch.launch(grid(kernels, n)?) }
            .map_err(|e| failed(lease, "materializing the explicit tile index", e))?;
    }
    let duplicates = control::<1>(lease, duplicate_count.get())?[0] != 0;
    Ok(ExplicitIndex {
        tile_rows: publish(out_row)?,
        tile_cols: publish(out_col)?,
        i: publish(out_i)?,
        j: publish(out_j)?,
        first: publish(first)?,
        n,
        duplicates,
    })
}

/// Same meanings as core Tiles: full counts canonical tasks, mirrored counts
/// actual lower-triangle recomputations whose reverse is outside this tile,
/// and evaluations is their sum. In-tile mirrored writes are not evaluations.
#[derive(Clone, Copy, Default, Debug, Eq, PartialEq)]
pub struct TileCounts {
    pub full: usize,
    pub mirrored: usize,
    pub evaluations: usize,
}
impl TileCounts {
    fn add(&mut self, full: usize, mirrored: usize) -> Result<(), EngineError> {
        let overflow = || EngineError::Overflow("tile work counter exceeds platform limit".into());
        let evaluations = full.checked_add(mirrored).ok_or_else(overflow)?;
        let next = Self {
            full: self.full.checked_add(full).ok_or_else(overflow)?,
            mirrored: self.mirrored.checked_add(mirrored).ok_or_else(overflow)?,
            evaluations: self
                .evaluations
                .checked_add(evaluations)
                .ok_or_else(overflow)?,
        };
        *self = next;
        Ok(())
    }
}
struct ChunkScratch {
    flags: crate::arena::Scratch<u64>,
    summary: crate::arena::Scratch<u64>,
}
fn chunk_scratch(lease: &DeviceLease, n: usize) -> Result<Pending<ChunkScratch>, EngineError> {
    let mut scratch = Pending::new(
        lease,
        ChunkScratch {
            flags: allocate(lease, n)?,
            summary: allocate(lease, 2)?,
        },
    );
    lease
        .execution()
        .fill(&mut *scratch.get_mut().summary, 2, 0)
        .map_err(|e| failed(lease, "initializing tile chunk counters", e))?;
    Ok(scratch)
}
fn summarize(
    lease: &DeviceLease,
    kernels: &Kernels,
    scratch: &mut ChunkScratch,
    n: usize,
) -> Result<(Pending<crate::arena::Scratch<u64>>, [u64; 2]), EngineError> {
    let prefix = Pending::new(lease, scan(lease, &scratch.flags, n)?);
    let (address, count, summary) = (
        ptr(prefix.get(), lease.stream()),
        n as u64,
        ptr(&scratch.summary, lease.stream()),
    );
    let mut launch = lease.execution().launch_builder(&kernels.count);
    launch.arg(&address).arg(&count).arg(&summary);
    // SAFETY: scan initialized n+1 entries; one thread writes count word zero
    // after any mirrored-count atomic updates in the previous kernel complete.
    unsafe { launch.launch(grid(kernels, 1)?) }
        .map_err(|e| failed(lease, "reducing tile chunk counters", e))?;
    let summary = control::<2>(lease, &scratch.summary)?;
    if summary[0] > n as u64 || summary[1] > summary[0] {
        return Err(bad("invalid tile chunk counts"));
    }
    Ok((prefix, summary))
}

pub struct TileProducer<'p, 'b, 'r> {
    plan: &'p TilePlan,
    native: &'b mut NativeProducer<'r>,
    rect: Rect,
    start: usize,
    end: usize,
    cursor: usize,
    counts: TileCounts,
    /// Chunk spaces carry raw i/j addresses. Keep generated coordinates until
    /// the sink finishes consuming that chunk; DeviceColumn retirement orders
    /// their last consumer even when the next chunk replaces these owners.
    coordinates: Option<[DeviceColumn; 2]>,
}
impl TileProducer<'_, '_, '_> {
    pub fn counts(&self) -> TileCounts {
        self.counts
    }
    /// Candidate work items of this tile, the total of its progress stage.
    pub fn candidates(&self) -> usize {
        self.end - self.start
    }
    fn implicit(&mut self, lease: &DeviceLease) -> Result<Option<ResidentChunk>, EngineError> {
        let kernels = &lease.device().0.resident_tiles;
        while self.cursor < self.end {
            let n = (self.end - self.cursor).min(self.plan.chunk);
            let mut scratch = chunk_scratch(lease, n)?;
            let coordinates = Pending::new(
                lease,
                (allocate::<i64>(lease, n)?, allocate::<i64>(lease, n)?),
            );
            let (left, right) = (
                ptr(&coordinates.get().0, lease.stream()),
                ptr(&coordinates.get().1, lease.stream()),
            );
            let (first, count, flags, summary) = (
                self.cursor as u64,
                n as u64,
                ptr(&scratch.get().flags, lease.stream()),
                ptr(&scratch.get().summary, lease.stream()),
            );
            let mut launch = lease.execution().launch_builder(&kernels.classify);
            launch
                .arg(&self.rect)
                .arg(&first)
                .arg(&count)
                .arg(&left)
                .arg(&right)
                .arg(&flags)
                .arg(&summary);
            // SAFETY: candidates are restricted to this checked rectangle and
            // <=65536 slots; the complete flags/coordinate buffers cover n.
            unsafe { launch.launch(grid(kernels, n)?) }
                .map_err(|e| failed(lease, "planning bounded implicit tile work", e))?;
            let (prefix, summary) = summarize(lease, kernels, scratch.get_mut(), n)?;
            self.cursor += n;
            // A resident tile bind (cancellation-progress.md §3.5).
            cancel::boundary(Site::Tile, n)?;
            let kept = summary[0] as usize;
            let mirrored = summary[1] as usize;
            if kept == 0 {
                continue;
            }
            let out_i = lease.output(DeviceDType::I64, vec![kept])?;
            let out_j = lease.output(DeviceDType::I64, vec![kept])?;
            let (prefix_address, oi, oj) = (
                ptr(prefix.get(), lease.stream()),
                out_i.address(),
                out_j.address(),
            );
            let mut launch = lease.execution().launch_builder(&kernels.compact);
            launch
                .arg(&left)
                .arg(&right)
                .arg(&flags)
                .arg(&prefix_address)
                .arg(&count)
                .arg(&oi)
                .arg(&oj);
            // SAFETY: every kept candidate owns one distinct prefix slot in the
            // exact-size coordinate outputs; rejected candidates do not write.
            unsafe { launch.launch(grid(kernels, n)?) }
                .map_err(|e| failed(lease, "compacting bounded tile coordinates", e))?;
            let (left, right) = (publish(out_i)?, publish(out_j)?);
            let space = ResidentSpace {
                kind: 1,
                n: kept as u64,
                li: left.address(),
                lj: right.address(),
                ..ResidentSpace::default()
            };
            self.coordinates = Some([left, right]);
            let chunk = self.native.emit_space(lease, space)?;
            self.counts.add(kept - mirrored, mirrored)?;
            return Ok(Some(chunk));
        }
        Ok(None)
    }
    fn explicit(&mut self, lease: &DeviceLease) -> Result<Option<ResidentChunk>, EngineError> {
        let index = self.plan.explicit.as_ref().expect("explicit tile index");
        let kernels = &lease.device().0.resident_tiles;
        while self.cursor < self.end {
            let n = (self.end - self.cursor).min(self.plan.chunk);
            let start = self.cursor;
            let space = ResidentSpace {
                kind: 1,
                k0: start as u64,
                n: n as u64,
                li: index.i.address(),
                lj: index.j.address(),
                ..ResidentSpace::default()
            };
            // Compute every original occurrence, matching core Tiles Tasks and
            // its actual evaluation counts, even for duplicate-only chunks.
            let chunk = self.native.emit_space(lease, space)?;
            self.cursor += n;
            self.counts.add(n, 0)?;
            // A resident tile bind (cancellation-progress.md §3.5).
            cancel::boundary(Site::Tile, n)?;
            if !index.duplicates {
                return Ok(Some(chunk));
            }
            let mut scratch = chunk_scratch(lease, n)?;
            let (source, first, count, flags) = (
                index.first.address(),
                start as u64,
                n as u64,
                ptr(&scratch.get().flags, lease.stream()),
            );
            let mut launch = lease.execution().launch_builder(&kernels.copy_flags);
            launch.arg(&source).arg(&first).arg(&count).arg(&flags);
            // SAFETY: this tile's sorted subrange covers start..start+n, and the
            // chunk flag buffer has exactly n initialized destinations.
            unsafe { launch.launch(grid(kernels, n)?) }
                .map_err(|e| failed(lease, "selecting first explicit tile occurrences", e))?;
            let (prefix, summary) = summarize(lease, kernels, scratch.get_mut(), n)?;
            let kept = summary[0] as usize;
            if kept == 0 {
                continue;
            }
            if kept == n {
                return Ok(Some(chunk));
            }
            let left = gather(
                lease,
                kernels,
                &index.i,
                start,
                n,
                kept,
                flags,
                ptr(prefix.get(), lease.stream()),
            )?;
            let right = gather(
                lease,
                kernels,
                &index.j,
                start,
                n,
                kept,
                flags,
                ptr(prefix.get(), lease.stream()),
            )?;
            let keys = gather(
                lease,
                kernels,
                &chunk.keys,
                0,
                n,
                kept,
                flags,
                ptr(prefix.get(), lease.stream()),
            )?;
            let mut fields = Vec::new();
            fields
                .try_reserve_exact(chunk.fields.len())
                .map_err(|_| bad("unable to reserve compact tile fields"))?;
            for field in &chunk.fields {
                let values = if field.values.address() == chunk.keys.address()
                    && field.values.dtype() == chunk.keys.dtype()
                {
                    keys.clone()
                } else {
                    gather(
                        lease,
                        kernels,
                        &field.values,
                        0,
                        n,
                        kept,
                        flags,
                        ptr(prefix.get(), lease.stream()),
                    )?
                };
                fields.push(ResidentField {
                    name: field.name.clone(),
                    values,
                    mask_bits: field.mask_bits,
                });
            }
            let space = ResidentSpace {
                kind: 1,
                n: kept as u64,
                li: left.address(),
                lj: right.address(),
                ..ResidentSpace::default()
            };
            self.coordinates = Some([left, right]);
            return Ok(Some(ResidentChunk {
                space,
                keys,
                fields,
            }));
        }
        Ok(None)
    }
}
impl ResidentProducer for TileProducer<'_, '_, '_> {
    fn key_dtype(&self) -> DeviceDType {
        self.native.key_dtype()
    }
    fn fields(&self) -> &[ResidentFieldSpec] {
        self.native.fields()
    }
    fn next_chunk(&mut self, lease: &DeviceLease) -> Result<Option<ResidentChunk>, EngineError> {
        guard(|| {
            self.plan.check_lease(lease)?;
            if self.plan.domain == TileDomain::Explicit {
                self.explicit(lease)
            } else {
                self.implicit(lease)
            }
        })
    }
    fn reset(&mut self) -> Result<(), EngineError> {
        self.cursor = self.start;
        self.counts = TileCounts::default();
        self.coordinates = None;
        Ok(())
    }
}
#[allow(clippy::too_many_arguments)]
fn gather(
    lease: &DeviceLease,
    kernels: &Kernels,
    source: &DeviceColumn,
    start: usize,
    n: usize,
    kept: usize,
    flags: u64,
    prefix: u64,
) -> Result<DeviceColumn, EngineError> {
    if source.shape().len() != 1
        || start
            .checked_add(n)
            .is_none_or(|end| end > source.shape()[0])
    {
        return Err(bad("compact tile source range exceeds its column"));
    }
    let output = lease.output(source.dtype(), vec![kept])?;
    let (address, width, first, count, destination) = (
        source.address(),
        source.dtype().width() as u64,
        start as u64,
        n as u64,
        output.address(),
    );
    let mut launch = lease.execution().launch_builder(&kernels.gather);
    launch
        .arg(&address)
        .arg(&width)
        .arg(&first)
        .arg(&count)
        .arg(&flags)
        .arg(&prefix)
        .arg(&destination);
    // SAFETY: initialized flags and exact prefixes select kept source elements;
    // the source remains owned through the queued D2D byte copies.
    unsafe { launch.launch(grid(kernels, n)?) }
        .map_err(|e| failed(lease, "compacting duplicate tile numeric columns", e))?;
    publish(output)
}
