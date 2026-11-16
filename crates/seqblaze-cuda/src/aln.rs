//! Align distance stage of a CUDA session (docs/wiki/gpu/align-cuda.md §4-§6,
//! §8): the scoring pack, the closed-form and tier launches, the routing pass
//! and the host computation of routed pairs; chunk planning and launch
//! geometry are in `plan.rs`.
//!
//! The device computes one exact integer per pair, the engine key (-score_fx,
//! or INT32_MAX - 1 for water's sentinel); every detail field comes from the
//! kernel's host `details` port for kept pairs (QAG1), so `delivers_details`
//! stays false and the sinks take their host-details paths. The tier of a pair
//! is one predicate (`sb_aln_tier` in kernels/align.cuh, `aln_tier` in
//! plan.rs). Routed pairs run on the host CPU kernel with no cutoff while the
//! device runs the rest, and their keys are scattered into the chunk's
//! buffer before any sink kernel reads it. Later sweeps over the same chunks
//! (RowStats, TopK) reuse the routed positions and host results, with the
//! levenshtein stage's rules (levenshtein-cuda.md §6).
use std::{collections::HashMap, sync::Arc};

use cudarc::driver::{CudaSlice, CudaStream, DeviceRepr, LaunchConfig, PushKernelArg};
use seqblaze_core::{
    detail::try_reserve_exact,
    sched,
    validate::EngineError,
    work::{Orientation, PairKind},
};

use crate::{
    device::Device,
    launch::Settings,
    plan::{
        aln_factor, aln_launches, aln_plan_chunk, AlnGeometry, AlnLengths, AlnPlan, AlnProfile,
        Request, SLOT_BYTES,
    },
    session::{
        dev_alloc, failure_ordinal, ptr, Chunk, Host, SbSpace, Scratch, Session, Table, BOTH, LIST,
        RECT, TRI, ZIP,
    },
    AlignCall, AlignKind,
};

// Kinds; must match SB_ALN_* in kernels/align.cuh.
const KIND_LOCAL: u64 = 0;
const KIND_GLOBAL: u64 = 1;
const KIND_WATER: u64 = 2;
const KIND_NEEDLE: u64 = 3;
/// Entries the routed cache holds at most, as levenshtein's (about 64 MiB,
/// estimated). Past it the cache stops growing and later sweeps recompute.
const CACHE_ITEMS: usize = 1 << 20;

/// #[repr(C)] mirror of `SbAln` in kernels/align.cuh; all fields are 8 bytes.
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct SbAln {
    kind: u64,
    end_gaps: u64,
    go: u64,
    ge: u64,
    mm: u64,
    matched: i64,
    mismatched: i64,
    n_classes: u64,
    cls: u64,
    table: u64,
    thread_rows: u64,
    lane_rows: u64,
    pair_cells: u64,
    slots: u64,
    slot_cols: u64,
}

// SAFETY: plain #[repr(C)] struct of 8-byte integer fields in the CUDA
// struct's order; passed by value as a kernel parameter.
unsafe impl DeviceRepr for SbAln {}

/// Per-call align state of a session.
pub(crate) struct Aln {
    /// Kernel parameters without the per-launch slot fields.
    params: SbAln,
    /// Kind index of the tier kernels (`Kernels::aln_thread` order).
    kind: usize,
    /// Closed form below this length on either side (§5.1): native 1, water
    /// 2, needle 0 (no closed-form pair, so `sb_aln_closed` is not launched).
    closed_below: u64,
    /// The uploaded class map and class table (read through `params`).
    _cls: CudaSlice<u8>,
    _table: CudaSlice<i32>,
    geometry: AlnGeometry,
    profile: AlnProfile,
    slots: Scratch<i32>,
    route_counts: Scratch<u64>,
    route_offsets: Scratch<u64>,
    route_pos: Scratch<u64>,
    scatter_pos: Scratch<u64>,
    scatter_d: Scratch<i32>,
    /// Distinct work items routed to the host so far (`execution.host_pairs`).
    pub(crate) host_pairs: usize,
    /// Range requests: work IDs below this were already counted.
    watermark: usize,
    /// Routed results kept for later sweeps; None in a tile stream.
    cache: Option<Cache>,
}

/// Routed chunk positions of one chunk, shared between the cache and a stage.
/// Routed chunk positions: the fallibly reserved vector itself, shared by a
/// fixed-size `Arc` (no second, input-scaled copy; F4 round 3).
type Positions = Arc<Vec<u64>>;

/// Routed work kept for later sweeps (levenshtein-cuda.md §6).
#[derive(Default)]
struct Cache {
    /// Range requests: (chunk start, request end) -> (chunk length, routed
    /// chunk positions).
    routes: HashMap<(usize, usize), (usize, Positions)>,
    /// (i, j) -> key of a routed pair.
    pairs: HashMap<(usize, usize), i32>,
    /// Entries held, against `CACHE_ITEMS`.
    items: usize,
}

impl Cache {
    fn room(&self, entries: usize) -> bool {
        self.items.saturating_add(entries) <= CACHE_ITEMS
    }
}

impl Aln {
    /// Upload the scoring pack (§4); no scan of the tables. `cache`: keep
    /// routed results for later sweeps (not in a tile stream).
    pub(crate) fn new(
        (stream, device): (&Arc<CudaStream>, &Device),
        settings: &Settings,
        call: &AlignCall,
        b: &Table,
        cache: bool,
    ) -> Result<Self, EngineError> {
        let fail = |what: &str, error| failure_ordinal(device, what, error);
        if call.table.len() != call.n_classes * call.n_classes || call.n_classes > 256 {
            return Err(EngineError::Device(
                "the align scoring pack is inconsistent; use device='cpu'".into(),
            ));
        }
        // SAFETY (both): fully written by the copies below; never cuMemAlloc(0).
        let mut cls =
            unsafe { dev_alloc::<u8>(stream, 256) }.map_err(|error| fail("allocation", error))?;
        crate::session::htod(stream, &call.class[..], &mut cls)
            .map_err(|error| fail("uploading the class map", error))?;
        let mut table = unsafe { dev_alloc::<i32>(stream, call.table.len().max(1)) }
            .map_err(|error| fail("allocation", error))?;
        if !call.table.is_empty() {
            crate::session::htod(stream, &call.table[..], &mut table)
                .map_err(|error| fail("uploading the class table", error))?;
        }
        let (kind, closed_below) = match call.kind {
            AlignKind::Local => (KIND_LOCAL, 1),
            AlignKind::Global => (KIND_GLOBAL, 1),
            AlignKind::Water => (KIND_WATER, 2),
            AlignKind::Needle => (KIND_NEEDLE, 0),
        };
        let (matched, mismatched) = call.matched.unwrap_or((0, 0));
        let params = SbAln {
            kind,
            end_gaps: u64::from(call.end_gaps),
            go: u64::try_from(call.go).unwrap_or(0),
            ge: u64::try_from(call.ge).unwrap_or(0),
            mm: u64::from(call.matched.is_some()),
            matched: i64::from(matched),
            mismatched: i64::from(mismatched),
            n_classes: call.n_classes as u64,
            cls: ptr(&cls, stream),
            table: ptr(&table, stream),
            thread_rows: settings.thread_rows,
            lane_rows: settings.lane_rows,
            pair_cells: settings.pair_cells,
            slots: 0,
            slot_cols: 0,
        };
        Ok(Self {
            params,
            kind: kind as usize,
            closed_below,
            _cls: cls,
            _table: table,
            geometry: AlnGeometry {
                thread_block: u64::from(settings.threads_per_block.min(device.0.aln_block[0])),
                warp_block: u64::from(settings.threads_per_block.min(device.0.aln_block[1])),
                resident_threads: device.0.resident_threads,
                slot_bytes: SLOT_BYTES,
            },
            profile: AlnProfile::new(b.offsets(), settings, closed_below)?,
            slots: Scratch::new(),
            route_counts: Scratch::new(),
            route_offsets: Scratch::new(),
            route_pos: Scratch::new(),
            scatter_pos: Scratch::new(),
            scatter_d: Scratch::new(),
            host_pairs: 0,
            watermark: 0,
            cache: cache.then(Cache::default),
        })
    }

    /// Conservative O(1) check: can any pair of a chunk whose rows have
    /// m <= `m_max` be routed? P·q is non-decreasing in m, so the warp depth
    /// of (m_max, max n of b) bounds every pair's; a thread-tier pair too deep
    /// for pair_cells falls through to the warp tier before it is routed.
    fn may_route(&self, m_max: u64, settings: &Settings) -> bool {
        aln_factor(settings, m_max).saturating_mul(self.profile.max_n.saturating_add(31))
            > settings.pair_cells
    }
}

impl Session {
    /// Stage the align chunk starting at item `pos`: keys of every item in
    /// `dist[0..n]`.
    pub(crate) fn aln_stage(
        &mut self,
        host: Host<'_>,
        request: &Request<'_>,
        pos: usize,
    ) -> Result<Chunk, EngineError> {
        let mut aln = self.aln.take().expect("align session");
        let result = self.aln_stage_with(&mut aln, host, request, pos);
        self.aln = Some(aln);
        result
    }

    fn aln_stage_with(
        &mut self,
        aln: &mut Aln,
        host: Host<'_>,
        request: &Request<'_>,
        pos: usize,
    ) -> Result<Chunk, EngineError> {
        let settings = self.settings;
        let mut space = SbSpace {
            k0: pos as u64,
            ..SbSpace::default()
        };
        match request {
            Request::Range { work, .. } => match &work.kind {
                PairKind::Zip => space.kind = ZIP,
                PairKind::Index { .. } => {
                    space.kind = LIST;
                    (space.li, space.lj) = self.index_columns(work)?;
                }
                PairKind::Cross | PairKind::OneToMany => {
                    space.kind = RECT;
                    space.width = work.shape.1 as u64;
                }
                PairKind::Within { .. }
                | PairKind::WithinOriented {
                    orientation: Orientation::AsGiven,
                    ..
                } => space.kind = TRI,
                PairKind::WithinOriented {
                    self_pairs,
                    orientation: Orientation::Both,
                } => {
                    space.kind = BOTH;
                    space.skip = u64::from(!*self_pairs);
                    space.width = (work.shape.0 - usize::from(!*self_pairs)) as u64;
                }
            },
            Request::Pairs(_) => space.kind = LIST,
            Request::Rect { r0, c0, width, .. } => {
                space.kind = RECT;
                space.r0 = *r0 as u64;
                space.c0 = *c0 as u64;
                space.width = *width as u64;
            }
        }
        let lengths = AlnLengths {
            a_off: self.a.offsets(),
            b_off: self.table_b().offsets(),
            profile: &aln.profile,
        };
        let plan = aln_plan_chunk(&lengths, &settings, &aln.geometry, request, pos);
        let n = plan.n;
        space.n = n as u64;
        if let Request::Pairs(pairs) = request {
            self.upload_pairs(&pairs[pos..pos + n], &mut space)?;
        }
        if space.kind == TRI {
            let Request::Range { work, .. } = request else {
                unreachable!("only within ranges use row segments")
            };
            self.upload_segments(work, pos, n, &mut space)?;
        }
        let stream = Arc::clone(&self.stream);
        let alloc = |error| failure_ordinal(&self.device, "allocation", error);
        let dist = ptr(self.dist.ensure(&stream, n).map_err(alloc)?, &stream);
        // Every item is written by the closed-form kernel, its tier kernel or
        // the scatter; -1 survives only in a skipped hole cell, never read.
        self.fill(dist, u32::MAX, n, true)?;
        if aln.closed_below > 0 {
            let tables = self.tables();
            let params = aln.params;
            let config = self.grid(n as u64);
            let mut launch = stream.launch_builder(&self.device.0.kernels.aln_closed);
            launch.arg(&tables).arg(&space).arg(&params).arg(&dist);
            // SAFETY: matches sb_aln_closed; `dist` holds n items.
            unsafe { launch.launch(config) }
                .map_err(|error| self.fail("launching sb_aln_closed", error))?;
        }
        // A Range chunk planned before (same start, end and length) holds the
        // same items: its routed positions are reused, not recounted.
        let key = match request {
            Request::Range { end, .. } => Some((pos, *end)),
            _ => None,
        };
        let cached = key.and_then(|key| {
            let (len, positions) = aln.cache.as_ref()?.routes.get(&key)?;
            (*len == n).then(|| Arc::clone(positions))
        });
        let routed: Positions = match cached {
            Some(positions) => positions,
            None if aln.may_route(plan.m_max, &settings) => {
                let positions: Positions = Arc::new(self.aln_route(aln, &space, n)?);
                if let (Some(key), Some(cache)) = (key, aln.cache.as_mut()) {
                    let entries = positions.len().max(1);
                    if cache.room(entries) {
                        cache.items += entries;
                        cache.routes.insert(key, (n, Arc::clone(&positions)));
                    }
                }
                positions
            }
            None => Arc::new(Vec::new()),
        };
        self.aln_tiers(aln, &space, &plan, dist)?;
        // Host workers compute the routed pairs while the device runs.
        let (positions, keys) = self.aln_routed(aln, host, request, pos, &routed)?;
        if let Request::Range { .. } = request {
            aln.watermark = aln.watermark.max(pos + n);
        }
        if !positions.is_empty() {
            self.aln_scatter(aln, &positions, &keys, dist)?;
        }
        Ok(Chunk { n, space })
    }

    /// Routing (§5.6): count, scan and write of the routed items, as G1's
    /// Hits, so the list is in item order. Returns their chunk positions.
    fn aln_route(&self, aln: &mut Aln, space: &SbSpace, n: usize) -> Result<Vec<u64>, EngineError> {
        let stream = &self.stream;
        let alloc = |error| failure_ordinal(&self.device, "allocation", error);
        let tables = self.tables();
        let params = aln.params;
        let (blocks, config) = self.tiles(n);
        let counts = ptr(
            aln.route_counts.ensure(stream, blocks).map_err(alloc)?,
            stream,
        );
        let kernels = &self.device.0.kernels;
        let mut launch = stream.launch_builder(&kernels.aln_route_count);
        launch.arg(&tables).arg(space).arg(&params).arg(&counts);
        // SAFETY: matches sb_aln_route_count; one count slot per block.
        unsafe { launch.launch(config) }
            .map_err(|error| self.fail("launching sb_aln_route_count", error))?;
        let mut block_counts = seqblaze_core::sink::try_filled(blocks, 0u64, "route block counts")?;
        let mut offsets = Vec::new();
        try_reserve_exact(&mut offsets, blocks, "route block offsets")?;
        self.download(aln.route_counts.get(), blocks, &mut block_counts)?;
        self.sync()?;
        let mut total = 0u64;
        for count in block_counts {
            offsets.push(total);
            total += count;
        }
        if total == 0 {
            return Ok(Vec::new());
        }
        let total = total as usize;
        let offsets = Session::upload(stream, &mut aln.route_offsets, &offsets).map_err(alloc)?;
        let out = ptr(aln.route_pos.ensure(stream, total).map_err(alloc)?, stream);
        let mut launch = stream.launch_builder(&kernels.aln_route_write);
        launch
            .arg(&tables)
            .arg(space)
            .arg(&params)
            .arg(&offsets)
            .arg(&out);
        // SAFETY: matches sb_aln_route_write; `out` holds `total` positions.
        unsafe { launch.launch(config) }
            .map_err(|error| self.fail("launching sb_aln_route_write", error))?;
        let mut positions = seqblaze_core::sink::try_filled(total, 0u64, "routed positions")?;
        self.download(aln.route_pos.get(), total, &mut positions)?;
        self.sync()?;
        Ok(positions)
    }

    /// Launch the tier kernels that may own a pair of the chunk, each with
    /// the grid `plan.rs` sets; multi-strip warp kernels get one slot of
    /// 2·(n + 1) int32 per warp, within the slot budget.
    fn aln_tiers(
        &self,
        aln: &mut Aln,
        space: &SbSpace,
        plan: &AlnPlan,
        dist: u64,
    ) -> Result<(), EngineError> {
        let settings = &self.settings;
        let stream = &self.stream;
        let kernels = &self.device.0.kernels;
        let tables = self.tables();
        for launch in aln_launches(&aln.geometry, plan) {
            let (kernel, name) = match launch.kernel {
                0 => (&kernels.aln_thread[aln.kind], "sb_aln_thread"),
                k => (&kernels.aln_warp[k - 1][aln.kind], "sb_aln_warp"),
            };
            let blocks = launch.blocks(settings, plan);
            let block_threads = if launch.kernel == 0 {
                aln.geometry.thread_block
            } else {
                aln.geometry.warp_block
            };
            let mut params = aln.params;
            if launch.slot_cols > 0 {
                // Written column by column before any read; never zeroed.
                let ints = (blocks * launch.per_block * 2 * launch.slot_cols) as usize;
                let slots = aln
                    .slots
                    .ensure(stream, ints)
                    .map_err(|error| failure_ordinal(&self.device, "allocation", error))?;
                params.slots = ptr(slots, stream);
                params.slot_cols = launch.slot_cols;
            }
            let config = LaunchConfig {
                grid_dim: (blocks as u32, 1, 1),
                block_dim: (block_threads as u32, 1, 1),
                shared_mem_bytes: 0,
            };
            let mut builder = stream.launch_builder(kernel);
            builder.arg(&tables).arg(space).arg(&params).arg(&dist);
            // SAFETY: matches the align tier kernels; `dist` holds n items and
            // the slots blocks·per_block·2·slot_cols int32.
            unsafe { builder.launch(config) }
                .map_err(|error| self.fail(&format!("launching {name}"), error))?;
        }
        Ok(())
    }

    /// Compute the routed items on host workers with the CPU kernel (no
    /// cutoff), counting distinct work items into `host_pairs` as the
    /// levenshtein stage does: work IDs not counted before (Range), every
    /// pair of a Tiles stream (Pairs), canonical cells of a rectangle; routed
    /// hole cells are skipped. Cached keys are reused; a Range request stores
    /// its new ones. Returns (chunk positions, keys).
    fn aln_routed(
        &self,
        aln: &mut Aln,
        host: Host<'_>,
        request: &Request<'_>,
        pos: usize,
        routed: &[u64],
    ) -> Result<(Vec<u64>, Vec<i32>), EngineError> {
        let mut items = Vec::new();
        try_reserve_exact(&mut items, routed.len(), "routed pairs")?;
        let mut counted = 0usize;
        for &t in routed {
            let t = t as usize;
            let (i, j) = match request {
                Request::Range { work, .. } => {
                    counted += usize::from(pos + t >= aln.watermark);
                    work.range(pos + t, pos + t + 1)
                        .next()
                        .expect("a routed item lies in its range")
                }
                Request::Pairs(pairs) => {
                    counted += usize::from(self.count_pairs);
                    pairs[pos + t]
                }
                Request::Rect {
                    r0,
                    c0,
                    width,
                    class,
                    ..
                } => {
                    let cell = pos + t;
                    let (i, j) = (*r0 + cell / *width, *c0 + cell % *width);
                    match (*class).map_or(Some(true), |class| class(i, j)) {
                        None => continue,
                        Some(canonical) => counted += usize::from(canonical),
                    }
                    (i, j)
                }
            };
            items.push((t as u64, i, j));
        }
        aln.host_pairs += counted;
        if items.is_empty() {
            return Ok((Vec::new(), Vec::new()));
        }
        let kernel = host.kernel;
        let (a, b) = (host.a, host.b);
        let mut keys: Vec<Option<i32>> = Vec::new();
        try_reserve_exact(&mut keys, items.len(), "routed pairs")?;
        let mut misses = Vec::new();
        for (k, &(_, i, j)) in items.iter().enumerate() {
            let key = aln
                .cache
                .as_ref()
                .and_then(|cache| cache.pairs.get(&(i, j)).copied());
            if key.is_none() {
                seqblaze_core::detail::try_reserve(&mut misses, 1, "routed pairs")?;
                misses.push(k);
            }
            keys.push(key);
        }
        if !misses.is_empty() {
            // Scheduling only: the kernel's pair cost, or the routed pairs'
            // mean when larger (about four units per cell, as `pair_cost`).
            let cells: u128 = misses
                .iter()
                .map(|&k| {
                    let (_, i, j) = items[k];
                    a.seq_len(i) as u128 * b.seq_len(j) as u128 * 4
                })
                .sum();
            let mean = usize::try_from(cells / misses.len() as u128).unwrap_or(usize::MAX);
            let cost = kernel.pair_cost(a, b, false).max(mean);
            let limit = self.workers.count(misses.len(), cost);
            let parts = sched::map_ranges(misses.len(), limit, &|start, end| {
                let mut out = Vec::new();
                try_reserve_exact(&mut out, end - start, "routed results")?;
                let mut ticker =
                    seqblaze_core::cancel::Ticker::new(seqblaze_core::cancel::Site::Routed);
                for &k in &misses[start..end] {
                    let (_, i, j) = items[k];
                    out.push(kernel.evaluate(a.get(i), b.get(j), None)?.ok_or_else(|| {
                        EngineError::Device("a routed pair has no score without a cutoff".into())
                    })?);
                    ticker.tick()?;
                }
                ticker.finish()?;
                Ok(out)
            })?;
            let mut next = misses.iter();
            for part in parts {
                for key in part {
                    let k = *next.next().ok_or_else(|| {
                        EngineError::Device("routed pair results exceed the routed list".into())
                    })?;
                    keys[k] = Some(key);
                }
            }
            if next.next().is_some() {
                return Err(EngineError::Device(
                    "routed pair results do not match the routed list".into(),
                ));
            }
            if let (Request::Range { .. }, Some(cache)) = (request, aln.cache.as_mut()) {
                for &k in &misses {
                    if !cache.room(1) {
                        break;
                    }
                    let (_, i, j) = items[k];
                    if let Some(key) = keys[k] {
                        if cache.pairs.insert((i, j), key).is_none() {
                            cache.items += 1;
                        }
                    }
                }
            }
        }
        let mut positions = Vec::new();
        try_reserve_exact(&mut positions, items.len(), "routed positions")?;
        positions.extend(items.iter().map(|&(t, _, _)| t));
        let keys = keys
            .into_iter()
            .map(|key| {
                key.ok_or_else(|| EngineError::Device("a routed pair has no host result".into()))
            })
            .collect::<Result<Vec<i32>, EngineError>>()?;
        Ok((positions, keys))
    }

    /// Write the routed keys into the chunk's buffer (stream-ordered after the
    /// tier kernels, before any sink kernel), with the levenshtein scatter
    /// kernel and no field buffers.
    fn aln_scatter(
        &self,
        aln: &mut Aln,
        positions: &[u64],
        keys: &[i32],
        dist: u64,
    ) -> Result<(), EngineError> {
        let stream = &self.stream;
        let uploaded = Session::upload(stream, &mut aln.scatter_pos, positions)
            .and_then(|p| Ok((p, Session::upload(stream, &mut aln.scatter_d, keys)?)));
        let (pos, d) = uploaded.map_err(|error| self.fail("uploading routed results", error))?;
        let count = positions.len() as u64;
        let none = 0u64;
        let config = self.grid(count);
        let mut launch = stream.launch_builder(&self.device.0.kernels.lev_scatter);
        launch
            .arg(&pos)
            .arg(&d)
            .arg(&none)
            .arg(&none)
            .arg(&count)
            .arg(&dist)
            .arg(&none)
            .arg(&none);
        // SAFETY: matches sb_lev_scatter; `pos` and `d` hold `count` entries and
        // the null field pointers are never dereferenced.
        unsafe { launch.launch(config) }
            .map_err(|error| self.fail("launching sb_lev_scatter", error))?;
        Ok(())
    }
}
