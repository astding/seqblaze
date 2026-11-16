//! Levenshtein distance stage of a CUDA session (docs/wiki/gpu/levenshtein-cuda.md
//! §3-§6, §8): the tier launches, the routing pass and the host computation
//! of routed pairs; chunk planning and launch geometry are in `plan.rs`.
//!
//! Every value is an exact integer of (a, b, mode). The tier of a pair is one
//! predicate (`sb_lev_tier` in kernels/levenshtein.cuh): with W = ⌈m/64⌉ a pair
//! goes to the host when W > `device_words` or its serial depth exceeds
//! `pair_steps` (n·W in the thread tier, (n + 31)·⌈W/32⌉ in the warp tier).
//! Routed pairs run on the host CPU kernel with no cutoff while the device runs
//! the rest, and their results are scattered into the chunk's buffers before
//! any sink kernel reads them, so no sink changes. Later sweeps over the same
//! chunks (RowStats, TopK) reuse the routed positions and host results.
use std::{collections::HashMap, sync::Arc};

use cudarc::driver::{CudaSlice, CudaStream, DeviceRepr, LaunchConfig, PushKernelArg};
use seqblaze_core::{
    detail::{try_reserve_exact, DetailColumns, DetailData, DetailRequest},
    sched,
    validate::EngineError,
    work::{Orientation, PairKind},
};

use crate::{
    device::Device,
    launch::Settings,
    plan::{plan_chunk, tier_launches, Geometry, Lengths, Plan, Profile, Request, SLOT_BYTES},
    session::{
        dev_alloc, failure_ordinal, ptr, Chunk, Host, SbSpace, Scratch, Session, Table, BOTH, LIST,
        RECT, TRI, ZIP,
    },
    EditMode,
};

// Modes; must match SB_LEV_* in kernels/levenshtein.cuh.
const MODE_GLOBAL: u64 = 0;
const MODE_PREFIX: u64 = 1;
const MODE_INFIX: u64 = 2;
/// Entries the routed cache holds at most (routed positions, plus one per
/// cached chunk, plus host results); about 64 MiB, estimated. Past it the
/// cache stops growing and later sweeps recompute: the same values.
const CACHE_ITEMS: usize = 1 << 20;

/// #[repr(C)] mirror of `SbLev` in kernels/levenshtein.cuh; all fields are u64.
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct SbLev {
    mode: u64,
    thread_words: u64,
    device_words: u64,
    pair_steps: u64,
    sigma: u64,
    map: u64,
    slots: u64,
    slot_words: u64,
    b_end: u64,
    n_ends: u64,
}

// SAFETY: plain #[repr(C)] struct of u64 fields in the CUDA struct's order;
// passed by value as a kernel parameter.
unsafe impl DeviceRepr for SbLev {}

/// Per-call levenshtein state of a session.
pub(crate) struct Lev {
    mode: u64,
    /// Indices of `b_end` and `n_ends` in the kernel's detail fields.
    pub(crate) b_end_field: usize,
    pub(crate) n_ends_field: usize,
    /// Raw table a: the device byte -> code map (u16[256]); None for a clean
    /// table a (four inline codes). Sigma is in `geometry`.
    raw: Option<CudaSlice<u16>>,
    geometry: Geometry,
    profile: Profile,
    slots: Scratch<u64>,
    b_end: Scratch<i64>,
    n_ends: Scratch<i32>,
    route_counts: Scratch<u64>,
    route_offsets: Scratch<u64>,
    route_pos: Scratch<u64>,
    scatter_pos: Scratch<u64>,
    scatter_d: Scratch<i32>,
    scatter_e: Scratch<i64>,
    scatter_n: Scratch<i32>,
    /// Distinct work items routed to the host so far (`execution.host_pairs`).
    pub(crate) host_pairs: usize,
    /// Range requests: work IDs below this were already counted (RowStats and
    /// TopK sweep the same work IDs twice).
    watermark: usize,
    /// Routed results kept for later sweeps; None in a tile stream.
    cache: Option<Cache>,
}

/// Routed chunk positions of one chunk, shared between the cache and a stage.
/// Routed chunk positions: the fallibly reserved vector itself, shared by a
/// fixed-size `Arc` (no second, input-scaled copy; F4 round 3).
type Positions = Arc<Vec<u64>>;

/// Routed work kept for later sweeps (levenshtein-cuda.md §6): RowStats and
/// TopK plan the same chunks of the same work twice (`plan_chunk` is a
/// function of the request, its position and the settings), and TopK's
/// detail pass reads its final entries once more.
#[derive(Default)]
struct Cache {
    /// Range requests: (chunk start, request end) -> (chunk length, routed
    /// chunk positions).
    routes: HashMap<(usize, usize), (usize, Positions)>,
    /// (i, j) -> host result of a routed pair.
    pairs: HashMap<(usize, usize), Memo>,
    /// Entries held, against `CACHE_ITEMS`.
    items: usize,
}

impl Cache {
    fn room(&self, entries: usize) -> bool {
        self.items.saturating_add(entries) <= CACHE_ITEMS
    }
}

/// Host result of one routed pair; with `fields`, b_end and n_ends hold the
/// call's requested numeric fields (0 where not requested).
#[derive(Clone, Copy)]
struct Memo {
    dist: i32,
    b_end: i64,
    n_ends: i32,
    fields: bool,
}

impl Lev {
    /// `cache`: keep routed results for later sweeps (not in a tile stream).
    pub(crate) fn new(
        (stream, device): (&Arc<CudaStream>, &Device),
        settings: &Settings,
        (mode, (b_end_field, n_ends_field)): (EditMode, (usize, usize)),
        b: &Table,
        bytes: Option<&[bool; 256]>,
        cache: bool,
    ) -> Result<Self, EngineError> {
        let fail = |what: &str, error| failure_ordinal(device, what, error);
        // Raw table a: codes 0..sigma-1 in increasing byte order, absent code sigma.
        let (raw, raw_sigma) = match bytes {
            None => (None, None),
            Some(seen) => {
                let sigma = seen.iter().filter(|&&present| present).count() as u16;
                let mut map = [0u16; 256];
                let mut next = 0u16;
                for (code, &present) in map.iter_mut().zip(seen) {
                    if present {
                        *code = next;
                        next += 1;
                    } else {
                        *code = sigma;
                    }
                }
                // SAFETY: fully written by the copy below.
                let mut slice = unsafe { dev_alloc::<u16>(stream, 256) }
                    .map_err(|error| fail("allocation", error))?;
                crate::session::htod(stream, &map[..], &mut slice)
                    .map_err(|error| fail("uploading the byte map", error))?;
                (Some(slice), Some(u64::from(sigma)))
            }
        };
        Ok(Self {
            mode: match mode {
                EditMode::Global => MODE_GLOBAL,
                EditMode::Prefix => MODE_PREFIX,
                EditMode::Infix => MODE_INFIX,
            },
            b_end_field,
            n_ends_field,
            raw,
            geometry: Geometry {
                threads_per_block: u64::from(settings.threads_per_block),
                raw_sigma,
                resident_threads: device.0.resident_threads,
                slot_bytes: SLOT_BYTES,
            },
            profile: Profile::new(b.offsets(), settings)?,
            slots: Scratch::new(),
            b_end: Scratch::new(),
            n_ends: Scratch::new(),
            route_counts: Scratch::new(),
            route_offsets: Scratch::new(),
            route_pos: Scratch::new(),
            scatter_pos: Scratch::new(),
            scatter_d: Scratch::new(),
            scatter_e: Scratch::new(),
            scatter_n: Scratch::new(),
            host_pairs: 0,
            watermark: 0,
            cache: cache.then(Cache::default),
        })
    }

    /// Whether the device's forward pass fills the whole request: a non-empty
    /// request of `b_end` and `n_ends` only (§3; any other field is heavy).
    pub(crate) fn delivers(&self, request: DetailRequest) -> bool {
        !request.is_empty()
            && (0..32).all(|field| {
                !request.contains(field) || field == self.b_end_field || field == self.n_ends_field
            })
    }

    /// Which of `b_end` and `n_ends` the request holds.
    pub(crate) fn wanted(&self, request: DetailRequest) -> (bool, bool) {
        (
            request.contains(self.b_end_field),
            request.contains(self.n_ends_field),
        )
    }

    /// Device addresses of the field buffers a field stage wrote (0: not requested).
    pub(crate) fn field_ptrs(&self, stream: &CudaStream, request: DetailRequest) -> (u64, u64) {
        let (want_e, want_n) = self.wanted(request);
        (
            if want_e {
                ptr(self.b_end.get(), stream)
            } else {
                0
            },
            if want_n {
                ptr(self.n_ends.get(), stream)
            } else {
                0
            },
        )
    }

    pub(crate) fn b_end_buffer(&self) -> &CudaSlice<i64> {
        self.b_end.get()
    }

    pub(crate) fn n_ends_buffer(&self) -> &CudaSlice<i32> {
        self.n_ends.get()
    }

    /// Conservative O(1) check (table maxima): can any pair of a chunk whose
    /// rows have W <= `w_max` be routed? Bounds every tier's depth from above.
    fn may_route(&self, w_max: u64, settings: &Settings) -> bool {
        let (tw, n) = (settings.thread_words, self.profile.max_n);
        w_max > settings.device_words
            || (tw > 0 && w_max.min(tw).saturating_mul(n) > settings.pair_steps)
            || (w_max > tw && (n + 31).saturating_mul(w_max.div_ceil(32)) > settings.pair_steps)
    }
}

/// The routed items of a chunk, with their host results.
struct Routed {
    /// Chunk positions of the computed items (holes skipped).
    pos: Vec<u64>,
    dist: Vec<i32>,
    b_end: Vec<i64>,
    n_ends: Vec<i32>,
}

impl Session {
    /// Stage the levenshtein chunk starting at item `pos`. With `fields`, the
    /// requested numeric fields (`b_end`, `n_ends`) are computed beside `dist`.
    pub(crate) fn lev_stage(
        &mut self,
        host: Host<'_>,
        request: &Request<'_>,
        pos: usize,
        fields: bool,
    ) -> Result<Chunk, EngineError> {
        let mut lev = self.lev.take().expect("levenshtein session");
        let result = self.lev_stage_with(&mut lev, host, request, pos, fields);
        self.lev = Some(lev);
        result
    }

    fn lev_stage_with(
        &mut self,
        lev: &mut Lev,
        host: Host<'_>,
        request: &Request<'_>,
        pos: usize,
        fields: bool,
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
        let lengths = Lengths {
            a_off: self.a.offsets(),
            b_off: self.table_b().offsets(),
            profile: &lev.profile,
        };
        let plan = plan_chunk(&lengths, &settings, &lev.geometry, request, pos);
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
        // Every item is written by its tier kernel or the scatter; -1 marks a
        // skipped hole cell, which is never read.
        self.fill(dist, u32::MAX, n, true)?;
        let (want_e, want_n) = if fields {
            lev.wanted(host.kernel.detail_request())
        } else {
            (false, false)
        };
        let sigma = lev.geometry.raw_sigma.unwrap_or(4);
        let map = lev.raw.as_ref().map_or(0, |map| ptr(map, &stream));
        let mut params = SbLev {
            mode: lev.mode,
            thread_words: settings.thread_words,
            device_words: settings.device_words,
            pair_steps: settings.pair_steps,
            sigma,
            map,
            ..SbLev::default()
        };
        if want_e {
            params.b_end = ptr(lev.b_end.ensure(&stream, n).map_err(alloc)?, &stream);
        }
        if want_n {
            params.n_ends = ptr(lev.n_ends.ensure(&stream, n).map_err(alloc)?, &stream);
        }
        // A Range chunk planned before (same start, end and length) holds
        // the same items: its routed positions are reused, not recounted.
        let key = match request {
            Request::Range { end, .. } => Some((pos, *end)),
            _ => None,
        };
        let cached = key.and_then(|key| {
            let (len, positions) = lev.cache.as_ref()?.routes.get(&key)?;
            (*len == n).then(|| Arc::clone(positions))
        });
        let routed: Positions = match cached {
            Some(positions) => positions,
            None if lev.may_route(plan.w_max, &settings) => {
                let positions: Positions = Arc::new(self.route(lev, &space, &params, n)?);
                if let (Some(key), Some(cache)) = (key, lev.cache.as_mut()) {
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
        self.launch_tiers(lev, &space, params, &plan, dist)?;
        // Host workers compute the routed pairs while the device runs.
        let results = self.routed(lev, host, request, pos, &routed, (want_e, want_n))?;
        if let Request::Range { .. } = request {
            lev.watermark = lev.watermark.max(pos + n);
        }
        if !results.pos.is_empty() {
            self.scatter(lev, &results, dist, &params)?;
        }
        Ok(Chunk { n, space })
    }

    /// Routing (§5.5): count, scan and write of the routed items, as G1's
    /// Hits, so the list is in item order. Returns their chunk positions.
    fn route(
        &self,
        lev: &mut Lev,
        space: &SbSpace,
        params: &SbLev,
        n: usize,
    ) -> Result<Vec<u64>, EngineError> {
        let stream = &self.stream;
        let alloc = |error| failure_ordinal(&self.device, "allocation", error);
        let tables = self.tables();
        let (blocks, config) = self.tiles(n);
        let counts = ptr(
            lev.route_counts.ensure(stream, blocks).map_err(alloc)?,
            stream,
        );
        let kernels = &self.device.0.kernels;
        let mut launch = stream.launch_builder(&kernels.lev_route_count);
        launch.arg(&tables).arg(space).arg(params).arg(&counts);
        // SAFETY: matches sb_lev_route_count; one count slot per block.
        unsafe { launch.launch(config) }
            .map_err(|error| self.fail("launching sb_lev_route_count", error))?;
        let mut block_counts = seqblaze_core::sink::try_filled(blocks, 0u64, "route block counts")?;
        let mut offsets = Vec::new();
        try_reserve_exact(&mut offsets, blocks, "route block offsets")?;
        self.download(lev.route_counts.get(), blocks, &mut block_counts)?;
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
        let offsets = Session::upload(stream, &mut lev.route_offsets, &offsets).map_err(alloc)?;
        let out = ptr(lev.route_pos.ensure(stream, total).map_err(alloc)?, stream);
        let mut launch = stream.launch_builder(&kernels.lev_route_write);
        launch
            .arg(&tables)
            .arg(space)
            .arg(params)
            .arg(&offsets)
            .arg(&out);
        // SAFETY: matches sb_lev_route_write; `out` holds `total` positions.
        unsafe { launch.launch(config) }
            .map_err(|error| self.fail("launching sb_lev_route_write", error))?;
        let mut positions = seqblaze_core::sink::try_filled(total, 0u64, "routed positions")?;
        self.download(lev.route_pos.get(), total, &mut positions)?;
        self.sync()?;
        Ok(positions)
    }

    /// Launch the thread tier and the warp-tier variants whose W range meets
    /// the chunk's rows (a kernel that cannot own any pair is not launched),
    /// each with the grid `plan.rs` sets: one unit per item it may own, at
    /// least one resident wave, raw grids within the slot budget, never fewer
    /// units than the per-unit rule needs.
    fn launch_tiers(
        &self,
        lev: &mut Lev,
        space: &SbSpace,
        mut params: SbLev,
        plan: &Plan,
        dist: u64,
    ) -> Result<(), EngineError> {
        let settings = &self.settings;
        let stream = &self.stream;
        let kernels = &self.device.0.kernels;
        let tables = self.tables();
        let raw = usize::from(lev.raw.is_some());
        for launch in tier_launches(settings, &lev.geometry, plan.w_min, plan.w_max) {
            let (kernel, name) = match launch.kernel {
                0 => (&kernels.lev_thread[raw], "sb_lev_thread"),
                k => (&kernels.lev_warp[k - 1][raw], "sb_lev_warp"),
            };
            let blocks = launch.blocks(settings, plan);
            if launch.slot_words > 0 {
                // One slot of sigma·top words per thread (thread tier) or warp;
                // slots are zero between pairs.
                let words = (blocks * launch.per_block * launch.slot_words) as usize;
                let (slots, fresh) = lev
                    .slots
                    .ensure_fresh(stream, words)
                    .map_err(|error| failure_ordinal(&self.device, "allocation", error))?;
                let slots = ptr(slots, stream);
                if fresh {
                    self.fill(slots, 0, words * 2, true)?;
                }
                params.slots = slots;
                params.slot_words = launch.slot_words;
            }
            let config = LaunchConfig {
                grid_dim: (blocks as u32, 1, 1),
                block_dim: (settings.threads_per_block, 1, 1),
                shared_mem_bytes: 0,
            };
            let mut launch = stream.launch_builder(kernel);
            launch.arg(&tables).arg(space).arg(&params).arg(&dist);
            // SAFETY: matches the levenshtein tier kernels; `dist` and the field
            // buffers hold n items, the slots blocks·per_block·slot_words words.
            unsafe { launch.launch(config) }
                .map_err(|error| self.fail(&format!("launching {name}"), error))?;
        }
        Ok(())
    }

    /// Compute the routed items on host workers with the CPU kernel (no
    /// cutoff), counting distinct work items into `host_pairs`: work IDs not
    /// counted before (Range), every pair of a Tiles stream (Pairs), and
    /// canonical cells of a rectangle; its routed hole cells are skipped.
    /// Results already in the cache are reused; a Range request stores its new
    /// ones. Whenever the device can deliver the call's numeric fields they
    /// are computed beside dist (the same forward pass), so a later detail
    /// pass over the same pairs reuses them.
    fn routed(
        &self,
        lev: &mut Lev,
        host: Host<'_>,
        request: &Request<'_>,
        pos: usize,
        routed: &[u64],
        (want_e, want_n): (bool, bool),
    ) -> Result<Routed, EngineError> {
        let mut items = Vec::new();
        try_reserve_exact(&mut items, routed.len(), "routed pairs")?;
        let mut counted = 0usize;
        for &t in routed {
            let t = t as usize;
            let (i, j) = match request {
                Request::Range { work, .. } => {
                    counted += usize::from(pos + t >= lev.watermark);
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
        lev.host_pairs += counted;
        let mut results = Routed {
            pos: Vec::new(),
            dist: Vec::new(),
            b_end: Vec::new(),
            n_ends: Vec::new(),
        };
        if items.is_empty() {
            return Ok(results);
        }
        let kernel = host.kernel;
        let (a, b) = (host.a, host.b);
        let numeric = want_e || want_n;
        let detailed = numeric || lev.delivers(kernel.detail_request());
        let mut memos: Vec<Option<Memo>> = Vec::new();
        try_reserve_exact(&mut memos, items.len(), "routed pairs")?;
        let mut misses = Vec::new();
        for (k, &(_, i, j)) in items.iter().enumerate() {
            let memo = lev
                .cache
                .as_ref()
                .and_then(|cache| cache.pairs.get(&(i, j)).copied())
                .filter(|memo| memo.fields || !numeric);
            if memo.is_none() {
                seqblaze_core::detail::try_reserve(&mut misses, 1, "routed pairs")?;
                misses.push(k);
            }
            memos.push(memo);
        }
        if !misses.is_empty() {
            let request_fields = if detailed {
                kernel.detail_request()
            } else {
                DetailRequest::NONE
            };
            // Scheduling only: the kernel's pair cost, or the mean word steps
            // of the routed pairs when larger (they are the long ones).
            let steps: u128 = misses
                .iter()
                .map(|&k| {
                    let (_, i, j) = items[k];
                    (a.seq_len(i).div_ceil(64) as u128) * b.seq_len(j) as u128
                })
                .sum();
            let mean = usize::try_from(steps / misses.len() as u128).unwrap_or(usize::MAX);
            let cost = kernel.pair_cost(a, b, detailed).max(mean);
            let limit = self.workers.count(misses.len(), cost);
            let parts = sched::map_ranges(misses.len(), limit, &|start, end| {
                let mut dist = Vec::new();
                try_reserve_exact(&mut dist, end - start, "routed results")?;
                let mut rows = DetailColumns::new(kernel.detail_fields(), request_fields);
                let mut ticker =
                    seqblaze_core::cancel::Ticker::new(seqblaze_core::cancel::Site::Routed);
                for &k in &misses[start..end] {
                    let (_, i, j) = items[k];
                    let score = if detailed {
                        kernel.evaluate_detailed(a.get(i), b.get(j), None, None, &mut rows)?
                    } else {
                        kernel.evaluate(a.get(i), b.get(j), None)?
                    };
                    dist.push(score.ok_or_else(|| {
                        EngineError::Device("a routed pair has no score without a cutoff".into())
                    })?);
                    ticker.tick()?;
                }
                ticker.finish()?;
                Ok((dist, rows))
            })?;
            let mut next = misses.iter();
            for (dist, rows) in parts {
                let (mut b_end, mut n_ends) = (None, None);
                for column in rows.into_columns() {
                    match column.data {
                        DetailData::Int64(values) if column.field == lev.b_end_field => {
                            b_end = Some(values);
                        }
                        DetailData::Int32(values) if column.field == lev.n_ends_field => {
                            n_ends = Some(values);
                        }
                        _ => {}
                    }
                }
                let field_rows = |len: Option<usize>| len.is_none_or(|len| len == dist.len());
                if !field_rows(b_end.as_ref().map(Vec::len))
                    || !field_rows(n_ends.as_ref().map(Vec::len))
                    || (want_e && b_end.is_none())
                    || (want_n && n_ends.is_none())
                {
                    return Err(EngineError::Device(
                        "routed pair results do not match the routed list".into(),
                    ));
                }
                for (row, &value) in dist.iter().enumerate() {
                    let k = *next.next().ok_or_else(|| {
                        EngineError::Device("routed pair results exceed the routed list".into())
                    })?;
                    memos[k] = Some(Memo {
                        dist: value,
                        b_end: b_end.as_ref().map_or(0, |values| values[row]),
                        n_ends: n_ends.as_ref().map_or(0, |values| values[row]),
                        fields: detailed,
                    });
                }
            }
            if next.next().is_some() {
                return Err(EngineError::Device(
                    "routed pair results do not match the routed list".into(),
                ));
            }
            if let (Request::Range { .. }, Some(cache)) = (request, lev.cache.as_mut()) {
                for &k in &misses {
                    if !cache.room(1) {
                        break;
                    }
                    let (_, i, j) = items[k];
                    if let Some(memo) = memos[k] {
                        if cache.pairs.insert((i, j), memo).is_none() {
                            cache.items += 1;
                        }
                    }
                }
            }
        }
        try_reserve_exact(&mut results.pos, items.len(), "routed positions")?;
        results.pos.extend(items.iter().map(|&(t, _, _)| t));
        try_reserve_exact(&mut results.dist, memos.len(), "routed results")?;
        if want_e {
            try_reserve_exact(&mut results.b_end, memos.len(), "routed results")?;
        }
        if want_n {
            try_reserve_exact(&mut results.n_ends, memos.len(), "routed results")?;
        }
        for memo in memos {
            let memo =
                memo.ok_or_else(|| EngineError::Device("a routed pair has no host result".into()))?;
            results.dist.push(memo.dist);
            if want_e {
                results.b_end.push(memo.b_end);
            }
            if want_n {
                results.n_ends.push(memo.n_ends);
            }
        }
        Ok(results)
    }

    /// Write the routed results into the chunk's buffers (stream-ordered
    /// after the tier kernels, before any sink kernel).
    fn scatter(
        &self,
        lev: &mut Lev,
        results: &Routed,
        dist: u64,
        params: &SbLev,
    ) -> Result<(), EngineError> {
        let stream = &self.stream;
        let uploaded = Session::upload(stream, &mut lev.scatter_pos, &results.pos).and_then(|p| {
            Ok((
                p,
                Session::upload(stream, &mut lev.scatter_d, &results.dist)?,
                Session::upload(stream, &mut lev.scatter_e, &results.b_end)?,
                Session::upload(stream, &mut lev.scatter_n, &results.n_ends)?,
            ))
        });
        let (pos, d, e, c) =
            uploaded.map_err(|error| self.fail("uploading routed results", error))?;
        let count = results.pos.len() as u64;
        let config = self.grid(count);
        let mut launch = stream.launch_builder(&self.device.0.kernels.lev_scatter);
        launch
            .arg(&pos)
            .arg(&d)
            .arg(&e)
            .arg(&c)
            .arg(&count)
            .arg(&dist)
            .arg(&params.b_end)
            .arg(&params.n_ends);
        // SAFETY: matches sb_lev_scatter; the inputs hold `count` entries (the
        // field inputs whenever the field pointers are non-null).
        unsafe { launch.launch(config) }
            .map_err(|error| self.fail("launching sb_lev_scatter", error))?;
        Ok(())
    }
}
