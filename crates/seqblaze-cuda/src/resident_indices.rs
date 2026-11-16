//! Device index normalization and canonical explicit-pair ordering.
//!
//! Producer storage is retained and ordered through the lease. No index array
//! is downloaded: only a first-error record and the distinct-pair count cross
//! to the host. Each consumes normalize's original order, including duplicates;
//! other sinks may consume canonical_pairs after validate_pair_indices.
use std::{mem::size_of, sync::Arc};

use cudarc::driver::{
    CudaFunction, CudaModule, CudaSlice, CudaStream, DeviceRepr, DriverError, LaunchConfig,
    PushKernelArg,
};
use seqblaze_core::validate::EngineError;

use crate::{
    device::{guard, Device},
    interop::{allocate, DeviceArray, DeviceColumn, DeviceDType, DeviceLease},
    session::{failure_ordinal, ptr},
};

const THREADS: usize = 256;
const NONE: u64 = u64::MAX;

#[repr(C)]
#[derive(Clone, Copy)]
struct IndexView {
    address: u64,
    n: u64,
    stride: i64,
    width: u64,
    signed: u64,
}
#[repr(C)]
#[derive(Clone, Copy)]
struct Pair {
    i: i64,
    j: i64,
}
// SAFETY: plain integer repr(C) records, matching resident_indices.cuh.
unsafe impl DeviceRepr for IndexView {}
unsafe impl DeviceRepr for Pair {}
const _: () = {
    assert!(size_of::<IndexView>() == 40);
    assert!(std::mem::offset_of!(IndexView, stride) == 16);
    assert!(size_of::<Pair>() == 16);
};

pub(crate) struct Kernels {
    normalize: CudaFunction,
    validate: CudaFunction,
    error: CudaFunction,
    pack: CudaFunction,
    histogram: CudaFunction,
    scatter: CudaFunction,
    unique: CudaFunction,
    compact: CudaFunction,
    order_init: CudaFunction,
    order_histogram: CudaFunction,
    order_scatter: CudaFunction,
    scan_block: CudaFunction,
    scan_add: CudaFunction,
    fixed_block_supported: bool,
}
impl Kernels {
    pub(crate) fn load(module: &Arc<CudaModule>) -> Result<Self, DriverError> {
        let mut result = Self {
            normalize: module.load_function("sb_index_normalize")?,
            validate: module.load_function("sb_index_validate")?,
            error: module.load_function("sb_index_error")?,
            pack: module.load_function("sb_index_pack")?,
            histogram: module.load_function("sb_index_radix_hist")?,
            scatter: module.load_function("sb_index_radix_scatter")?,
            unique: module.load_function("sb_index_unique")?,
            compact: module.load_function("sb_index_compact")?,
            order_init: module.load_function("sb_index_order_init")?,
            order_histogram: module.load_function("sb_index_order_hist")?,
            order_scatter: module.load_function("sb_index_order_scatter")?,
            scan_block: module.load_function("sb_input_scan_u64")?,
            scan_add: module.load_function("sb_input_scan_add")?,
            fixed_block_supported: true,
        };
        for function in [
            &result.normalize,
            &result.validate,
            &result.error,
            &result.pack,
            &result.histogram,
            &result.scatter,
            &result.unique,
            &result.compact,
            &result.order_init,
            &result.order_histogram,
            &result.order_scatter,
            &result.scan_block,
            &result.scan_add,
        ] {
            result.fixed_block_supported &= function.max_threads_per_block()? >= THREADS as i32;
        }
        Ok(result)
    }
}

/// Raw-pointer launches bypass cudarc's typed borrow events. Keep every scratch
/// allocation until stream completion, and leak it if completion cannot be proved.
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
        self.value
            .as_ref()
            .expect("retained resident index scratch")
    }
    fn get_mut(&mut self) -> &mut T {
        self.value
            .as_mut()
            .expect("retained resident index scratch")
    }
}
impl<T> Drop for Pending<T> {
    fn drop(&mut self) {
        if guard(|| {
            crate::interop::execution_stream::retire_in_drop(&self.stream)
                .map_err(|e| failure_ordinal(&self.device, "retiring resident index raw owners", e))
        })
        .is_err()
        {
            if let Some(value) = self.value.take() {
                std::mem::forget(value);
            }
        }
    }
}
fn failed(lease: &DeviceLease, what: &str, error: DriverError) -> EngineError {
    failure_ordinal(lease.device(), what, error)
}
fn bad(text: &str) -> EngineError {
    EngineError::Device(format!("resident indices: {text}"))
}
fn exact_grid(kernels: &Kernels, blocks: usize) -> Result<LaunchConfig, EngineError> {
    if !kernels.fixed_block_supported {
        return Err(bad("the loaded kernels cannot launch a 256-thread block"));
    }
    let blocks = u32::try_from(blocks.max(1))
        .ok()
        .filter(|&blocks| blocks <= i32::MAX as u32)
        .ok_or_else(|| bad("the index grid exceeds CUDA limits"))?;
    Ok(LaunchConfig {
        grid_dim: (blocks, 1, 1),
        block_dim: (THREADS as u32, 1, 1),
        shared_mem_bytes: 0,
    })
}
fn grid(kernels: &Kernels, n: usize) -> Result<LaunchConfig, EngineError> {
    exact_grid(kernels, n.div_ceil(THREADS).clamp(1, 65_535))
}
fn control<const N: usize>(
    lease: &DeviceLease,
    input: &CudaSlice<u64>,
    at: usize,
) -> Result<[u64; N], EngineError> {
    let mut out = [0u64; N];
    lease
        .execution()
        .memcpy_dtoh(&input.slice(at..at + N), &mut out[..])
        .map_err(|e| failed(lease, "reading resident index control metadata", e))?;
    lease.synchronize()?;
    Ok(out)
}

/// Normalize one native-endian integer view to a contiguous, immutable I64
/// column. Negative/zero strides and unaligned integer views are read on device.
/// Unsigned values outside I64 are rejected before any pair bounds validation.
pub fn normalize(
    lease: &mut DeviceLease,
    input: DeviceArray,
    label: &str,
) -> Result<DeviceColumn, EngineError> {
    guard(|| {
        let descriptor = input.descriptor().clone();
        let signed = match descriptor.dtype {
            DeviceDType::I8 | DeviceDType::I16 | DeviceDType::I32 | DeviceDType::I64 => true,
            DeviceDType::U8 | DeviceDType::U16 | DeviceDType::U32 | DeviceDType::U64 => false,
            _ => {
                return Err(EngineError::Pairing(format!(
                    "{label} must be one-dimensional integers; use np.array([0], dtype='int64')"
                )))
            }
        };
        if descriptor.shape.len() != 1 {
            return Err(EngineError::Pairing(format!(
                "{label} must be one-dimensional integers; use np.array([0], dtype='int64')"
            )));
        }
        let n = descriptor.shape[0];
        // Validate the actual allocation/context and establish producer ordering
        // before any raw address is used; retain the original producer owner.
        lease.retain(input)?;
        let output = lease.output(DeviceDType::I64, vec![n])?;
        if n != 0 {
            let kernels = &lease.device().0.resident_indices;
            let mut overflow = if descriptor.dtype == DeviceDType::U64 {
                Some(Pending::new(lease, allocate::<u64>(lease, 1)?))
            } else {
                None
            };
            if let Some(overflow) = &mut overflow {
                lease
                    .execution()
                    .fill(&mut **overflow.get_mut(), 1, 0xFF)
                    .map_err(|e| failed(lease, "initializing device index overflow control", e))?;
            }
            let view = IndexView {
                address: descriptor.address,
                n: n as u64,
                stride: descriptor.strides[0],
                width: descriptor.dtype.width() as u64,
                signed: u64::from(signed),
            };
            let destination = output.address();
            let first_bad = overflow
                .as_ref()
                .map_or(0, |value| ptr(value.get(), lease.stream()));
            let mut launch = lease.execution().launch_builder(&kernels.normalize);
            launch.arg(&view).arg(&destination).arg(&first_bad);
            // SAFETY: retain validated the complete signed-stride span; output
            // covers n I64 words and the optional control allocation has one word.
            unsafe { launch.launch(grid(kernels, n)?) }
                .map_err(|e| failed(lease, "normalizing device indices", e))?;
            if let Some(overflow) = &overflow {
                if control::<1>(lease, overflow.get(), 0)?[0] != NONE {
                    return Err(EngineError::Pairing(format!(
                        "{label} contains an index above int64 max; indices must fit int64"
                    )));
                }
            }
        }
        // SAFETY: each logical I64 word was initialized on this lease's stream;
        // no writable alias to the new allocation is retained after publication.
        unsafe { output.finish() }
    })
}

/// Normalize an immutable native column without exporting a writable borrow or
/// routing through a DLPack consumer copy. The lease retains its original owner.
pub fn normalize_owned(
    lease: &mut DeviceLease,
    column: &DeviceColumn,
    label: &str,
) -> Result<DeviceColumn, EngineError> {
    normalize(lease, column.internal_array(), label)
}

fn column_length(column: &DeviceColumn, label: &str) -> Result<usize, EngineError> {
    if column.dtype() != DeviceDType::I64 || column.shape().len() != 1 {
        return Err(EngineError::Pairing(format!(
            "{label} must be a one-dimensional int64 device column"
        )));
    }
    Ok(column.shape()[0])
}
fn retain_pair_columns(
    lease: &mut DeviceLease,
    i: &DeviceColumn,
    j: &DeviceColumn,
) -> Result<usize, EngineError> {
    let n = column_length(i, "i")?;
    let m = column_length(j, "j")?;
    if n != m {
        return Err(EngineError::Pairing(format!(
            "index lengths differ: {n} and {m}"
        )));
    }
    lease.retain(i.internal_array())?;
    lease.retain(j.internal_array())?;
    Ok(n)
}

/// Validate original occurrences before sorting or deduplication. The three-word
/// download contains the first bad position and its signed I64 pair, reproducing
/// core WorkSource::new's exact diagnostic and precedence within the pair list.
pub fn validate_pair_indices(
    lease: &mut DeviceLease,
    i: &DeviceColumn,
    j: &DeviceColumn,
    shape: [usize; 2],
) -> Result<(), EngineError> {
    guard(|| {
        let n = retain_pair_columns(lease, i, j)?;
        if n == 0 {
            return Ok(());
        }
        let kernels = &lease.device().0.resident_indices;
        let mut summary = Pending::new(lease, allocate::<u64>(lease, 3)?);
        lease
            .execution()
            .memcpy_htod(&[NONE, 0, 0][..], &mut **summary.get_mut())
            .map_err(|e| failed(lease, "initializing device pair validation", e))?;
        let (left, right, count) = (i.address(), j.address(), n as u64);
        let (rows, columns) = (shape[0] as u64, shape[1] as u64);
        let first_bad = ptr(summary.get(), lease.stream());
        let mut launch = lease.execution().launch_builder(&kernels.validate);
        launch
            .arg(&left)
            .arg(&right)
            .arg(&count)
            .arg(&rows)
            .arg(&columns)
            .arg(&first_bad);
        // SAFETY: retained normalized inputs contain n initialized I64 words;
        // the only raw scratch address names the retained three-word summary.
        unsafe { launch.launch(grid(kernels, n)?) }
            .map_err(|e| failed(lease, "validating original device index pairs", e))?;
        let mut launch = lease.execution().launch_builder(&kernels.error);
        launch.arg(&left).arg(&right).arg(&first_bad);
        // SAFETY: the preceding kernel can only record positions in 0..n; the
        // stream boundary makes the minimum stable before its pair is gathered.
        unsafe { launch.launch(exact_grid(kernels, 1)?) }
            .map_err(|e| failed(lease, "gathering the first invalid device pair", e))?;
        let summary = control::<3>(lease, summary.get(), 0)?;
        if summary[0] != NONE {
            return Err(EngineError::Pairing(format!(
                "index pair {} ({}, {}) is out of bounds for ({}, {})",
                summary[0], summary[1] as i64, summary[2] as i64, shape[0], shape[1]
            )));
        }
        Ok(())
    })
}

struct ScanLevel {
    capacity: usize,
    used: usize,
    prefix: crate::arena::Scratch<u64>,
    totals: crate::arena::Scratch<u64>,
}
/// Reuse hierarchical scan buffers across all radix passes. Kernel boundaries
/// order each level; there is no host count read or synchronization per pass.
struct ScanWorkspace {
    levels: Vec<ScanLevel>,
}
impl ScanWorkspace {
    fn new(lease: &DeviceLease, mut n: usize) -> Result<Self, EngineError> {
        let mut levels = Vec::new();
        loop {
            let blocks = n.div_ceil(THREADS);
            let prefix_len = n
                .checked_add(1)
                .ok_or_else(|| bad("prefix length overflows"))?;
            levels
                .try_reserve(1)
                .map_err(|_| bad("unable to reserve index scan metadata"))?;
            levels.push(ScanLevel {
                capacity: n,
                used: 0,
                prefix: allocate(lease, prefix_len)?,
                totals: allocate(lease, blocks)?,
            });
            if blocks <= 1 {
                break;
            }
            n = blocks;
        }
        Ok(Self { levels })
    }
    fn run<'a>(
        &'a mut self,
        lease: &DeviceLease,
        kernels: &Kernels,
        input: &CudaSlice<u64>,
        mut n: usize,
    ) -> Result<&'a CudaSlice<u64>, EngineError> {
        if n == 0 || n > self.levels[0].capacity {
            return Err(bad("invalid reusable scan range"));
        }
        let mut active = 0;
        for index in 0..self.levels.len() {
            let (before, after) = self.levels.split_at_mut(index);
            let level = &mut after[0];
            let source = if index == 0 {
                input
            } else {
                &before[index - 1].totals
            };
            let count = n as u64;
            let blocks = n.div_ceil(THREADS);
            level.used = n;
            let mut launch = lease.execution().launch_builder(&kernels.scan_block);
            launch
                .slice_arg(source)
                .slice_arg_mut(&mut level.prefix)
                .slice_arg_mut(&mut level.totals)
                .arg(&count);
            // SAFETY: every level covers its actual logical count and one total
            // per exact 256-element block; the previous level initialized source.
            unsafe { launch.launch(exact_grid(kernels, blocks)?) }
                .map_err(|e| failed(lease, "scanning resident index buckets", e))?;
            active = index + 1;
            if blocks == 1 {
                break;
            }
            n = blocks;
        }
        for index in (0..active - 1).rev() {
            let (before, after) = self.levels.split_at_mut(index + 1);
            let level = &mut before[index];
            let count = level.used as u64;
            let mut launch = lease.execution().launch_builder(&kernels.scan_add);
            launch
                .slice_arg_mut(&mut level.prefix)
                .slice_arg(&after[0].prefix)
                .arg(&count);
            // SAFETY: after[0]'s complete prefix supplies one offset per block
            // of this level; all accesses are bounded by the recorded used count.
            unsafe { launch.launch(exact_grid(kernels, level.used.div_ceil(THREADS))?) }
                .map_err(|e| failed(lease, "combining resident index scan levels", e))?;
        }
        Ok(&self.levels[0].prefix)
    }
}
struct SortScratch {
    first: crate::arena::Scratch<Pair>,
    second: crate::arena::Scratch<Pair>,
    histogram: crate::arena::Scratch<u64>,
    scan: ScanWorkspace,
}

/// GPU lexicographic sort and adjacent deduplication of explicit pairs. The
/// original columns remain immutable for Each and execution-count provenance.
/// Scratch scales with the explicit occurrence count, never the table product.
pub fn canonical_pairs(
    lease: &mut DeviceLease,
    i: &DeviceColumn,
    j: &DeviceColumn,
) -> Result<(DeviceColumn, DeviceColumn), EngineError> {
    let (i, j) = canonical_columns(lease, i, j, false)?;
    Ok((i, j.expect("pair canonicalization publishes both columns")))
}

/// Canonical ascending positions for selecting sinks. Selection-specific bounds
/// checks and Python negative-index normalization belong to the caller. This
/// shares the pair radix engine, with eight passes and one output column.
pub fn canonical_indices(
    lease: &mut DeviceLease,
    positions: &DeviceColumn,
) -> Result<DeviceColumn, EngineError> {
    canonical_columns(lease, positions, positions, true).map(|(positions, _)| positions)
}

fn canonical_columns(
    lease: &mut DeviceLease,
    i: &DeviceColumn,
    j: &DeviceColumn,
    single: bool,
) -> Result<(DeviceColumn, Option<DeviceColumn>), EngineError> {
    guard(|| {
        let n = retain_pair_columns(lease, i, j)?;
        if n <= 1 {
            return Ok((i.clone(), (!single).then(|| j.clone())));
        }
        let kernels = &lease.device().0.resident_indices;
        let blocks = n.div_ceil(THREADS);
        let sort_grid = exact_grid(kernels, blocks)?;
        let histogram_len = blocks
            .checked_mul(256)
            .ok_or_else(|| bad("radix histogram length overflows"))?;
        // All scratch is allocated before the first launch, then one Pending
        // owner retains every raw allocation through success and all early errors.
        let mut scratch = Pending::new(
            lease,
            SortScratch {
                first: allocate(lease, n)?,
                second: allocate(lease, n)?,
                histogram: allocate(lease, histogram_len)?,
                scan: ScanWorkspace::new(lease, histogram_len)?,
            },
        );
        let SortScratch {
            first,
            second,
            histogram,
            scan,
        } = scratch.get_mut();
        let (left, right, count) = (i.address(), j.address(), n as u64);
        let mut launch = lease.execution().launch_builder(&kernels.pack);
        launch
            .arg(&left)
            .arg(&right)
            .arg(&count)
            .slice_arg_mut(&mut *first);
        // SAFETY: retained inputs and both pair buffers contain n pair slots.
        unsafe { launch.launch(grid(kernels, n)?) }
            .map_err(|e| failed(lease, "packing resident index pairs", e))?;
        let mut in_first = true;
        // LSD stability makes j secondary and i primary. Sign-bit flipping
        // defines signed lexicographic order even before the bounds preflight.
        for field in 0u64..if single { 1 } else { 2 } {
            for shift in (0u64..64).step_by(8) {
                let (source, target): (&CudaSlice<Pair>, &mut CudaSlice<Pair>) = if in_first {
                    (&*first, &mut *second)
                } else {
                    (&*second, &mut *first)
                };
                let mut launch = lease.execution().launch_builder(&kernels.histogram);
                launch
                    .slice_arg(source)
                    .arg(&count)
                    .arg(&field)
                    .arg(&shift)
                    .slice_arg_mut(&mut *histogram);
                // SAFETY: each exact block emits 256 initialized bucket counts;
                // inactive lanes participate in barriers but do not read source.
                unsafe { launch.launch(sort_grid) }
                    .map_err(|e| failed(lease, "counting resident pair radix buckets", e))?;
                let prefix = scan.run(lease, kernels, histogram, histogram_len)?;
                let mut launch = lease.execution().launch_builder(&kernels.scatter);
                launch
                    .slice_arg(source)
                    .arg(&count)
                    .arg(&field)
                    .arg(&shift)
                    .slice_arg(prefix)
                    .slice_arg(target);
                // SAFETY: exact bucket prefixes plus stable in-block ranks
                // assign every initialized pair a distinct destination in 0..n.
                unsafe { launch.launch(sort_grid) }
                    .map_err(|e| failed(lease, "ordering resident pair radix buckets", e))?;
                in_first = !in_first;
            }
        }
        let sorted = if in_first { &*first } else { &*second };
        let mut launch = lease.execution().launch_builder(&kernels.unique);
        launch
            .slice_arg(sorted)
            .arg(&count)
            .slice_arg_mut(&mut *histogram);
        // SAFETY: stable radix passes initialized all n sorted pairs;
        // histogram storage has at least n entries and is reused as unique flags.
        unsafe { launch.launch(grid(kernels, n)?) }
            .map_err(|e| failed(lease, "marking distinct resident index pairs", e))?;
        let prefix = scan.run(lease, kernels, histogram, n)?;
        let unique = control::<1>(lease, prefix, n)?[0];
        let unique = usize::try_from(unique)
            .ok()
            .filter(|&unique| unique <= n)
            .ok_or_else(|| bad("distinct-pair count exceeds the input length"))?;
        let out_i = lease.output(DeviceDType::I64, vec![unique])?;
        let out_j = if single {
            None
        } else {
            Some(lease.output(DeviceDType::I64, vec![unique])?)
        };
        let (left, right) = (
            out_i.address(),
            out_j.as_ref().map_or(0, |column| column.address()),
        );
        let mut launch = lease.execution().launch_builder(&kernels.compact);
        launch
            .slice_arg(sorted)
            .slice_arg(&*histogram)
            .slice_arg(prefix)
            .arg(&count)
            .arg(&left)
            .arg(&right);
        // SAFETY: only first occurrences write; their exclusive prefixes cover
        // exactly unique output positions, with no alias to original columns.
        unsafe { launch.launch(grid(kernels, n)?) }
            .map_err(|e| failed(lease, "compacting canonical resident pairs", e))?;
        lease.synchronize()?;
        // SAFETY: complete same-stream compaction initialized both immutable
        // output columns; no writable address survives publication.
        unsafe {
            Ok((
                out_i.finish()?,
                out_j.map(|column| column.finish()).transpose()?,
            ))
        }
    })
}

struct OrderScratch {
    alternate: crate::arena::Scratch<u64>,
    histogram: crate::arena::Scratch<u64>,
    scan: ScanWorkspace,
}

/// Stable GPU lexicographic permutation, retaining every original occurrence.
/// Keys are supplied from primary to secondary, each contiguous I64/U64[n].
/// Equal full keys retain original order; no key or permutation payload is read
/// on the host. Tiles uses separate row/column keys rather than a flattened ID.
pub(crate) fn lexicographic_order(
    lease: &mut DeviceLease,
    keys: &[DeviceColumn],
) -> Result<DeviceColumn, EngineError> {
    guard(|| {
        let first = keys
            .first()
            .ok_or_else(|| bad("lexicographic ordering requires at least one key"))?;
        if first.shape().len() != 1 {
            return Err(bad("lexicographic keys must be one-dimensional"));
        }
        let n = first.shape()[0];
        for key in keys {
            if !matches!(key.dtype(), DeviceDType::I64 | DeviceDType::U64) || key.shape() != [n] {
                return Err(bad(
                    "lexicographic keys must be equal-length I64/U64 columns",
                ));
            }
            lease.retain(key.internal_array())?;
        }
        let output = lease.output(DeviceDType::U64, vec![n])?;
        if n == 0 {
            // SAFETY: the empty logical output has no bytes to initialize.
            return unsafe { output.finish() };
        }
        let kernels = &lease.device().0.resident_indices;
        if n == 1 {
            let (count, address) = (1u64, output.address());
            let mut launch = lease.execution().launch_builder(&kernels.order_init);
            launch.arg(&count).arg(&address);
            // SAFETY: the single logical permutation entry is initialized to zero.
            unsafe { launch.launch(grid(kernels, 1)?) }
                .map_err(|e| failed(lease, "initializing singleton device order", e))?;
            // SAFETY: no sort is needed for the sole initialized index.
            return unsafe { output.finish() };
        }
        let blocks = n.div_ceil(THREADS);
        let histogram_len = blocks
            .checked_mul(256)
            .ok_or_else(|| bad("permutation histogram length overflows"))?;
        let sort_grid = exact_grid(kernels, blocks)?;
        let mut scratch = Pending::new(
            lease,
            OrderScratch {
                alternate: allocate(lease, n)?,
                histogram: allocate(lease, histogram_len)?,
                scan: ScanWorkspace::new(lease, histogram_len)?,
            },
        );
        let OrderScratch {
            alternate,
            histogram,
            scan,
        } = scratch.get_mut();
        let primary = output.address();
        let alternate = ptr(alternate, lease.stream());
        let count = n as u64;
        let mut launch = lease.execution().launch_builder(&kernels.order_init);
        launch.arg(&count).arg(&primary);
        // SAFETY: the allocated output covers n U64 permutation entries.
        unsafe { launch.launch(grid(kernels, n)?) }
            .map_err(|e| failed(lease, "initializing resident lexicographic order", e))?;
        let mut in_primary = true;
        for key in keys.iter().rev() {
            let address = key.address();
            let signed = u64::from(key.dtype() == DeviceDType::I64);
            for shift in (0u64..64).step_by(8) {
                let (source, target) = if in_primary {
                    (primary, alternate)
                } else {
                    (alternate, primary)
                };
                let mut launch = lease.execution().launch_builder(&kernels.order_histogram);
                launch
                    .arg(&source)
                    .arg(&count)
                    .arg(&address)
                    .arg(&signed)
                    .arg(&shift)
                    .slice_arg_mut(&mut *histogram);
                // SAFETY: every source permutation entry indexes a retained n-word
                // key, and every exact block initializes all 256 bucket counts.
                unsafe { launch.launch(sort_grid) }
                    .map_err(|e| failed(lease, "counting resident lexicographic buckets", e))?;
                let prefix = scan.run(lease, kernels, histogram, histogram_len)?;
                let mut launch = lease.execution().launch_builder(&kernels.order_scatter);
                launch
                    .arg(&source)
                    .arg(&count)
                    .arg(&address)
                    .arg(&signed)
                    .arg(&shift)
                    .slice_arg(prefix)
                    .arg(&target);
                // SAFETY: stable bucket ranks produce a complete permutation in
                // the disjoint target buffer, with no source or key mutation.
                unsafe { launch.launch(sort_grid) }
                    .map_err(|e| failed(lease, "scattering resident lexicographic buckets", e))?;
                in_primary = !in_primary;
            }
        }
        // Eight passes per key leave the final permutation in the first output.
        debug_assert!(in_primary);
        lease.synchronize()?;
        // SAFETY: the complete immutable U64 permutation was initialized on the
        // same stream, and every construction-only writable address is retired.
        unsafe { output.finish() }
    })
}

/// Borrow a complete contiguous metadata view of an immutable result column.
/// Public proxy reshapes preserve byte count but need not match its native rank.
pub fn owned_flat_view(column: &DeviceColumn, n: usize) -> Result<DeviceArray, EngineError> {
    if n.checked_mul(column.dtype().width()) != Some(column.nbytes()) {
        return Err(EngineError::Pairing(
            "device index view does not cover its original allocation".into(),
        ));
    }
    let mut descriptor = column.internal_array().descriptor().clone();
    descriptor.shape = vec![n];
    descriptor.strides = vec![column.dtype().width() as i64];
    // SAFETY: the readonly complete span and strong allocation owner are the
    // same as the immutable published column; only logical rank is rewritten.
    unsafe { DeviceArray::from_external(descriptor, Arc::new(column.clone())) }
}
pub fn normalize_owned_flat(
    lease: &mut DeviceLease,
    column: &DeviceColumn,
    n: usize,
    label: &str,
) -> Result<DeviceColumn, EngineError> {
    normalize(lease, owned_flat_view(column, n)?, label)
}
