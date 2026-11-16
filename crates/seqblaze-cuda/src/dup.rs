//! Duplex stage of a CUDA session (docs/wiki/gpu/duplex-cuda.md §4-§6, §8):
//! the device pack, the closed-form, DP and routing launches, and the host
//! computation of routed pairs; chunk planning, the tier predicate and launch
//! geometry are in `plan.rs`.
//!
//! A stage is a score pass (the engine key of every item, int32 or int64 band
//! keys by the call's width) or a detail pass (full keys, the key and the three
//! float64 fields of every item). The tier of a pair is one predicate
//! (`sb_dup_tier` in kernels/duplex.cuh, `dup_tier` in plan.rs). Routed pairs
//! run on the host CPU kernel while the device runs the rest; their results
//! are scattered into the chunk's buffers before any sink kernel reads them.
//! Later sweeps reuse routed positions and host results with the levenshtein
//! stage's rules (levenshtein-cuda.md §6).
use std::{collections::HashMap, sync::Arc};

use cudarc::driver::{CudaSlice, CudaStream, DeviceRepr, LaunchConfig, PushKernelArg};
use seqblaze_core::{
    detail::{try_reserve_exact, DetailColumns, DetailData, DetailRequest},
    sched,
    sink::Columns,
    validate::EngineError,
    work::{Orientation, PairKind},
};

use crate::{
    device::Device,
    launch::Settings,
    plan::{
        dup_c, dup_launches, dup_plan_chunk, dup_settings, DupGeometry, DupLengths, DupLimits,
        DupPlan, DupProfile, DupWidth, Request, DUP_CODES, DUP_CODES_STATES, DUP_GROUPS,
        SLOT_BYTES,
    },
    session::{
        dev_alloc, failure_ordinal, ptr, Chunk, Host, SbSpace, Scratch, Session, Table, BOTH, LIST,
        RECT, TRI, ZIP,
    },
    DuplexCall,
};

/// Entries the routed cache holds at most, as levenshtein's.
const CACHE_ITEMS: usize = 1 << 20;
/// `SbDup::tier` of the DP launches; must match SB_DUP_* in kernels/duplex.cuh.
const TIER_SHARED: u64 = 1;
const TIER_GLOBAL: u64 = 2;
/// `SbDup::poison` when no poison byte is set.
const NO_POISON: u64 = 256;

/// #[repr(C)] mirror of `SbDup` in kernels/duplex.cuh; all fields are 8 bytes.
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct SbDup {
    score: u64,
    full: u64,
    lam: u64,
    end_l: u64,
    end_r: u64,
    stack: u64,
    bulge: u64,
    i11: u64,
    i12: u64,
    i21: u64,
    i22: u64,
    gen: u64,
    loop_out: u64,
    loop_in: u64,
    s: u64,
    kappa4: u64,
    kappa1: u64,
    group_lanes: u64,
    band_bytes: u64,
    pair_cells: u64,
    s_max: u64,
    s_block: u64,
    tier: u64,
    stride: u64,
    slots: u64,
    slot: u64,
    dh: u64,
    ds: u64,
    tm: u64,
    poison: u64,
    /// Pack format 2 (nndb2024-dna.md §5.4.4): N_P, N_C, the device address of CLS, the
    /// start of STATES, whether the pack has STATES, and q.
    n_p: u64,
    n_c: u64,
    cls: u64,
    states: u64,
    has_states: u64,
    codes: u64,
    /// C1 bounded resident stage: u8* skip flags of the launch's items, or 0.
    skip: u64,
    #[cfg(feature = "testing")]
    resident_stats: u64,
}

// SAFETY: plain #[repr(C)] struct of u64 fields in the CUDA struct's order;
// passed by value as a kernel parameter.
unsafe impl DeviceRepr for SbDup {}

/// The score keys on the device, in the call's width; held for the session's
/// lifetime and read by the kernels through `SbDup::score`.
#[allow(dead_code)]
enum ScoreKeys {
    Narrow(CudaSlice<i32>),
    Wide(CudaSlice<i64>),
}

/// A routed pair's host result: its key and, from a detail pass, the bits of
/// `dh_kcal`, `ds_cal_per_k` and `tm_c`.
#[derive(Clone, Copy)]
struct Memo {
    key: i32,
    fields: Option<[u64; 3]>,
}

/// Routed chunk positions of one chunk, shared between the cache and a stage.
/// Routed chunk positions: the fallibly reserved vector itself, shared by a
/// fixed-size `Arc` (no second, input-scaled copy; F4 round 3).
type Positions = Arc<Vec<u64>>;

#[derive(Default)]
struct Cache {
    /// Range requests: (chunk start, request end) -> (chunk length, routed positions).
    routes: HashMap<(usize, usize), (usize, Positions)>,
    pairs: HashMap<(usize, usize), Memo>,
    items: usize,
}

impl Cache {
    fn room(&self, entries: usize) -> bool {
        self.items.saturating_add(entries) <= CACHE_ITEMS
    }
}

/// Per-call duplex state of a session.
pub(crate) struct Dup {
    call: Arc<DuplexCall>,
    /// Kernel parameters without the per-launch fields.
    params: SbDup,
    /// The score pass's band width (narrow or wide).
    score_width: DupWidth,
    _score: ScoreKeys,
    /// The CLS map (u8), read through `SbDup::cls`.
    _cls: CudaSlice<u8>,
    /// The full keys, uploaded on the first detail pass.
    full: Option<CudaSlice<i64>>,
    profile: DupProfile,
    /// Largest n of table a (the O(1) routing skip).
    max_n: u64,
    /// Block caps of the narrow, wide and detail kernels; shared memory limits.
    blocks: [u32; 3],
    limits: DupLimits,
    resident_threads: u64,
    slots: Scratch<u64>,
    fields: [Scratch<u64>; 3],
    route_counts: Scratch<u64>,
    route_offsets: Scratch<u64>,
    route_pos: Scratch<u64>,
    scatter_pos: Scratch<u64>,
    scatter_d: Scratch<i32>,
    scatter_f: Scratch<u64>,
    /// Indices of `dh_kcal`, `ds_cal_per_k` and `tm_c` in the kernel's fields.
    pub(crate) field_index: [usize; 3],
    /// Distinct work items routed to the host so far (`execution.host_pairs`).
    pub(crate) host_pairs: usize,
    watermark: usize,
    cache: Option<Cache>,
}

fn width_index(width: DupWidth) -> usize {
    match width {
        DupWidth::Narrow => 0,
        DupWidth::Wide => 1,
        DupWidth::Detail => 2,
    }
}

impl Dup {
    /// Upload the score keys (§4); the full keys wait for a detail pass.
    pub(crate) fn new(
        (stream, device): (&Arc<CudaStream>, &Device),
        settings: &Settings,
        call: &Arc<DuplexCall>,
        (a, b): (&Table, &Table),
        cache: bool,
    ) -> Result<Self, EngineError> {
        let fail = |what: &str, error| failure_ordinal(device, what, error);
        let settings = &dup_settings(settings);
        let total = call.offsets[14];
        if !call.layout_ok() || call.score.len() != total || call.full.len() != total || total < 2 {
            return Err(EngineError::Device(
                "the duplex device pack is inconsistent; use device='cpu'".into(),
            ));
        }
        let (score, score_ptr) = if call.wide {
            // SAFETY: fully written by the copy below; total >= 2.
            let mut keys = unsafe { dev_alloc::<i64>(stream, total) }
                .map_err(|error| fail("allocation", error))?;
            crate::session::htod(stream, &call.score[..], &mut keys)
                .map_err(|error| fail("uploading the duplex pack", error))?;
            let at = ptr(&keys, stream);
            (ScoreKeys::Wide(keys), at)
        } else {
            let narrow: Vec<i32> = call
                .score
                .iter()
                .map(|&key| i32::try_from(key))
                .collect::<Result<_, _>>()
                .map_err(|_| {
                    EngineError::Device(
                        "the duplex int32 pack holds a key outside int32; use device='cpu'".into(),
                    )
                })?;
            // SAFETY: fully written by the copy below; total >= 2.
            let mut keys = unsafe { dev_alloc::<i32>(stream, total) }
                .map_err(|error| fail("allocation", error))?;
            crate::session::htod(stream, &narrow[..], &mut keys)
                .map_err(|error| fail("uploading the duplex pack", error))?;
            let at = ptr(&keys, stream);
            (ScoreKeys::Narrow(keys), at)
        };
        // CLS: (Λ + 1)² ≥ 1 bytes (nndb2024-dna.md §5.4.2).
        // SAFETY: fully written by the copy below; never zero elements.
        let mut cls = unsafe { dev_alloc::<u8>(stream, call.cls.len()) }
            .map_err(|error| fail("allocation", error))?;
        crate::session::htod(stream, &call.cls[..], &mut cls)
            .map_err(|error| fail("uploading the duplex pack", error))?;
        let o = &call.offsets;
        let has_states = call.has_states();
        let limits = DupLimits {
            s_max: device.0.dup_s_max,
            s_block: device.0.dup_s_block,
            codes: if has_states {
                DUP_CODES_STATES
            } else {
                DUP_CODES
            },
        };
        let params = SbDup {
            score: score_ptr,
            lam: call.loop_max as u64,
            end_l: o[2] as u64,
            end_r: o[3] as u64,
            stack: o[4] as u64,
            bulge: o[5] as u64,
            i11: o[6] as u64,
            i12: o[7] as u64,
            i21: o[8] as u64,
            i22: o[9] as u64,
            gen: o[10] as u64,
            loop_out: o[11] as u64,
            loop_in: o[12] as u64,
            s: call.s_bits,
            kappa4: call.kappa4_bits,
            kappa1: call.kappa1_bits,
            group_lanes: settings.group_lanes,
            band_bytes: settings.band_bytes,
            pair_cells: settings.pair_cells,
            s_max: limits.s_max,
            s_block: limits.s_block,
            poison: poison(),
            n_p: call.n_p as u64,
            n_c: call.n_c as u64,
            cls: ptr(&cls, stream),
            states: o[13] as u64,
            has_states: u64::from(has_states),
            codes: limits.codes,
            ..SbDup::default()
        };
        let lam = call.loop_max as u64;
        let max_n = a
            .offsets()
            .windows(2)
            .map(|pair| (pair[1] - pair[0]) as u64)
            .max()
            .unwrap_or(0);
        Ok(Self {
            call: Arc::clone(call),
            params,
            score_width: if call.wide {
                DupWidth::Wide
            } else {
                DupWidth::Narrow
            },
            _score: score,
            _cls: cls,
            full: None,
            profile: DupProfile::new(b.offsets(), settings, lam)?,
            max_n,
            blocks: device.0.dup_block,
            limits,
            resident_threads: device.0.resident_threads,
            slots: Scratch::new(),
            fields: [Scratch::new(), Scratch::new(), Scratch::new()],
            route_counts: Scratch::new(),
            route_offsets: Scratch::new(),
            route_pos: Scratch::new(),
            scatter_pos: Scratch::new(),
            scatter_d: Scratch::new(),
            scatter_f: Scratch::new(),
            field_index: call.fields,
            host_pairs: 0,
            watermark: 0,
            cache: cache.then(Cache::default),
        })
    }

    /// Whether the device fills every requested detail field: all three
    /// duplex fields are device fields.
    pub(crate) fn delivers(&self, request: DetailRequest) -> bool {
        !request.is_empty()
            && (0..32).all(|field| !request.contains(field) || self.field_index.contains(&field))
    }

    /// Which of the three device fields the request holds.
    pub(crate) fn wanted(&self, request: DetailRequest) -> [bool; 3] {
        self.field_index.map(|field| request.contains(field))
    }

    pub(crate) fn field_buffer(&self, k: usize) -> &CudaSlice<u64> {
        self.fields[k].get()
    }

    fn geometry(&self, settings: &Settings, width: DupWidth) -> DupGeometry {
        DupGeometry {
            threads_per_block: u64::from(
                settings
                    .threads_per_block
                    .min(self.blocks[width_index(width)]),
            ),
            resident_threads: self.resident_threads,
            limits: self.limits,
            slot_bytes: SLOT_BYTES,
        }
    }

    /// Conservative O(1) check: can any pair be routed? The depth grows with
    /// n and m and shrinks with g, so the longest a against the longest b at
    /// the smallest group size in use bounds every pair's.
    fn may_route(&self, settings: &Settings) -> bool {
        let lam = self.profile.lam;
        let (n, m) = (self.max_n, self.profile.max_m());
        let g = if settings.group_lanes > 0 {
            settings.group_lanes
        } else {
            DUP_GROUPS[0]
        };
        n.saturating_mul(m.div_ceil(g))
            .saturating_mul(dup_c(n, m, lam))
            > settings.pair_cells
    }

    /// Upload the full keys, once per call.
    fn ensure_full(
        &mut self,
        stream: &Arc<CudaStream>,
        device: &Device,
    ) -> Result<(), EngineError> {
        if self.full.is_none() {
            let fail = |what: &str, error| failure_ordinal(device, what, error);
            let mut flat = Vec::new();
            try_reserve_exact(&mut flat, 4 * self.call.full.len(), "duplex device pack")?;
            flat.extend(self.call.full.iter().flatten().copied());
            // SAFETY: fully written by the copy below; never zero elements.
            let mut keys = unsafe { dev_alloc::<i64>(stream, flat.len().max(1)) }
                .map_err(|error| fail("allocation", error))?;
            crate::session::htod(stream, &flat[..], &mut keys)
                .map_err(|error| fail("uploading the duplex pack", error))?;
            self.params.full = ptr(&keys, stream);
            self.full = Some(keys);
        }
        Ok(())
    }
}

/// The poison byte of a testing build (`set_cuda_poison`), else none.
fn poison() -> u64 {
    #[cfg(feature = "testing")]
    if let Some(byte) = crate::launch::poison() {
        return u64::from(byte);
    }
    NO_POISON
}

impl Session {
    /// Hits with fields (duplex-cuda.md §6, QKG7): a device detail pass over
    /// the final hit list in its order. Each detail key must equal the hit's
    /// selection key (QKG9); otherwise nothing is published.
    pub(crate) fn dup_hit_details(
        &mut self,
        host: Host<'_>,
        columns: &Columns,
    ) -> Result<DetailColumns, EngineError> {
        let (i, j) = (
            columns.i.as_deref().unwrap_or(&[]),
            columns.j.as_deref().unwrap_or(&[]),
        );
        let mut pairs = Vec::new();
        try_reserve_exact(&mut pairs, i.len(), "duplex hit list")?;
        pairs.extend(i.iter().zip(j).map(|(&i, &j)| (i as usize, j as usize)));
        let mut rows =
            DetailColumns::new(host.kernel.detail_fields(), host.kernel.detail_request());
        rows.reserve(pairs.len())?;
        let request = Request::Pairs(&pairs);
        let mut keys = Vec::new();
        let mut pos = 0;
        while pos < pairs.len() {
            let chunk = self.stage(host, &request, pos, true)?;
            keys.clear();
            self.fetch_into(chunk.n, &mut keys)?;
            for (t, &key) in keys.iter().enumerate() {
                if key != columns.dist[pos + t] {
                    let (i, j) = pairs[pos + t];
                    return Err(EngineError::Device(format!(
                        "duplex Hits: the detail pass gives key {key} for pair ({i}, {j}) but \
                         the selection pass gave {}; no result was published; use device='cpu'",
                        columns.dist[pos + t]
                    )));
                }
            }
            let mut chunk_rows = self.fetch_details(host, chunk.n)?;
            rows.append(&mut chunk_rows)?;
            pos += chunk.n;
        }
        Ok(rows)
    }

    /// Stage the duplex chunk starting at item `pos`: keys of every item in
    /// `dist[0..n]` and, with `fields`, the three field bit patterns in the
    /// duplex field buffers (a detail pass).
    pub(crate) fn dup_stage(
        &mut self,
        host: Host<'_>,
        request: &Request<'_>,
        pos: usize,
        fields: bool,
    ) -> Result<Chunk, EngineError> {
        let mut dup = self.dup.take().expect("duplex session");
        let result = self.dup_stage_with(&mut dup, host, request, pos, fields);
        self.dup = Some(dup);
        result
    }

    fn dup_stage_with(
        &mut self,
        dup: &mut Dup,
        host: Host<'_>,
        request: &Request<'_>,
        pos: usize,
        fields: bool,
    ) -> Result<Chunk, EngineError> {
        // Duplex plans and predicates with its capped pair_cells (plan.rs).
        let settings = dup_settings(&self.settings);
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
        let stream = Arc::clone(&self.stream);
        let width = if fields {
            dup.ensure_full(&stream, &self.device)?;
            DupWidth::Detail
        } else {
            dup.score_width
        };
        let geometry = dup.geometry(&settings, width);
        let lengths = DupLengths {
            a_off: self.a.offsets(),
            b_off: self.table_b().offsets(),
            profile: &dup.profile,
        };
        let plan = dup_plan_chunk(&lengths, &settings, &geometry, request, pos, width);
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
        let alloc = |error| failure_ordinal(&self.device, "allocation", error);
        let dist = ptr(self.dist.ensure(&stream, n).map_err(alloc)?, &stream);
        // Every item is written by the closed-form kernel, its DP kernel or the
        // scatter; -1 survives only in a skipped hole cell, never read.
        self.fill(dist, u32::MAX, n, true)?;
        let mut params = dup.params;
        if fields {
            let [dh, ds, tm] = &mut dup.fields;
            params.dh = ptr(dh.ensure(&stream, n).map_err(alloc)?, &stream);
            params.ds = ptr(ds.ensure(&stream, n).map_err(alloc)?, &stream);
            params.tm = ptr(tm.ensure(&stream, n).map_err(alloc)?, &stream);
            // Testing builds with a poison byte: refill the field buffers every
            // chunk, so an item no kernel writes shows the poison byte, never a
            // previous chunk's bits (the allocation-time fill covers only the
            // first chunk).
            #[cfg(feature = "testing")]
            if let Some(byte) = crate::launch::poison() {
                for address in [params.dh, params.ds, params.tm] {
                    self.fill(address, u32::from(byte), n * 8, false)?;
                }
                crate::launch::count_poisoned(24 * n as u64);
            }
        }
        {
            let tables = self.tables();
            let config = self.grid(n as u64);
            let mut launch = stream.launch_builder(&self.device.0.kernels.dup_closed);
            launch.arg(&tables).arg(&space).arg(&params).arg(&dist);
            // SAFETY: matches sb_dup_closed; `dist` and the field buffers hold n items.
            unsafe { launch.launch(config) }
                .map_err(|error| self.fail("launching sb_dup_closed", error))?;
        }
        // A Range chunk planned before (same start, end and length) holds the
        // same items: its routed positions are reused, not recounted.
        let key = match request {
            Request::Range { end, .. } => Some((pos, *end)),
            _ => None,
        };
        let cached = key.and_then(|key| {
            let (len, positions) = dup.cache.as_ref()?.routes.get(&key)?;
            (*len == n).then(|| Arc::clone(positions))
        });
        let routed: Positions = match cached {
            Some(positions) => positions,
            None if dup.may_route(&settings) => {
                let positions: Positions = Arc::new(self.dup_route(dup, &space, &params, n)?);
                if let (Some(key), Some(cache)) = (key, dup.cache.as_mut()) {
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
        self.dup_variants(
            dup,
            (&space, &plan),
            (&settings, &geometry),
            &params,
            width,
            dist,
        )?;
        // Host workers compute the routed pairs while the device runs.
        let (positions, memos) = self.dup_routed(dup, host, request, pos, &routed, fields)?;
        if let Request::Range { .. } = request {
            dup.watermark = dup.watermark.max(pos + n);
        }
        if !positions.is_empty() {
            self.dup_scatter(dup, &positions, &memos, &params, dist, fields)?;
        }
        Ok(Chunk { n, space })
    }

    /// Routing (§5.6): count, scan and write of the routed items, as G1's
    /// Hits, so the list is in item order. Returns their chunk positions.
    fn dup_route(
        &self,
        dup: &mut Dup,
        space: &SbSpace,
        params: &SbDup,
        n: usize,
    ) -> Result<Vec<u64>, EngineError> {
        let stream = &self.stream;
        let alloc = |error| failure_ordinal(&self.device, "allocation", error);
        let tables = self.tables();
        let (blocks, config) = self.tiles(n);
        let counts = ptr(
            dup.route_counts.ensure(stream, blocks).map_err(alloc)?,
            stream,
        );
        let kernels = &self.device.0.kernels;
        let mut launch = stream.launch_builder(&kernels.dup_route_count);
        launch.arg(&tables).arg(space).arg(params).arg(&counts);
        // SAFETY: matches sb_dup_route_count; one count slot per block.
        unsafe { launch.launch(config) }
            .map_err(|error| self.fail("launching sb_dup_route_count", error))?;
        let mut block_counts = seqblaze_core::sink::try_filled(blocks, 0u64, "route block counts")?;
        let mut offsets = Vec::new();
        try_reserve_exact(&mut offsets, blocks, "route block offsets")?;
        self.download(dup.route_counts.get(), blocks, &mut block_counts)?;
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
        let offsets = Session::upload(stream, &mut dup.route_offsets, &offsets).map_err(alloc)?;
        let out = ptr(dup.route_pos.ensure(stream, total).map_err(alloc)?, stream);
        let mut launch = stream.launch_builder(&kernels.dup_route_write);
        launch
            .arg(&tables)
            .arg(space)
            .arg(params)
            .arg(&offsets)
            .arg(&out);
        // SAFETY: matches sb_dup_route_write; `out` holds `total` positions.
        unsafe { launch.launch(config) }
            .map_err(|error| self.fail("launching sb_dup_route_write", error))?;
        let mut positions = seqblaze_core::sink::try_filled(total, 0u64, "routed positions")?;
        self.download(dup.route_pos.get(), total, &mut positions)?;
        self.sync()?;
        Ok(positions)
    }

    /// Launch the variants (g, tier) that may own a pair of the chunk, each
    /// with the grid and block `plan.rs` sets; global-tier groups get one band
    /// slot each, within the slot budget.
    fn dup_variants(
        &self,
        dup: &mut Dup,
        (space, plan): (&SbSpace, &DupPlan),
        (settings, geometry): (&Settings, &DupGeometry),
        params: &SbDup,
        width: DupWidth,
        dist: u64,
    ) -> Result<(), EngineError> {
        let stream = &self.stream;
        let tables = self.tables();
        let kernels = &self.device.0.kernels.dup[width_index(width)];
        for launch in dup_launches(settings, geometry, plan) {
            let blocks = launch.blocks(settings, plan);
            let mut params = *params;
            params.stride = launch.stride;
            if launch.global {
                params.tier = TIER_GLOBAL;
                // Written row by row before any read; never zeroed (§5.4).
                let words = (blocks * launch.per_block * launch.slot / 8) as usize;
                let slots = dup
                    .slots
                    .ensure(stream, words)
                    .map_err(|error| failure_ordinal(&self.device, "allocation", error))?;
                params.slots = ptr(slots, stream);
                params.slot = launch.slot;
            } else {
                params.tier = TIER_SHARED;
            }
            let config = LaunchConfig {
                grid_dim: (blocks as u32, 1, 1),
                block_dim: (launch.threads() as u32, 1, 1),
                shared_mem_bytes: launch.shared_bytes() as u32,
            };
            let kernel = &kernels[launch.variant / 2];
            let mut builder = stream.launch_builder(kernel);
            builder.arg(&tables).arg(space).arg(&params).arg(&dist);
            // SAFETY: matches the duplex DP kernels; `dist` and the field
            // buffers hold n items, the slots blocks·per_block·slot bytes, and
            // the dynamic shared memory per_block·stride bytes.
            unsafe { builder.launch(config) }.map_err(|error| {
                self.fail(
                    &format!("launching sb_dup (variant {})", launch.variant),
                    error,
                )
            })?;
        }
        Ok(())
    }

    /// Compute the routed items on host workers with the CPU kernel (no
    /// cutoff; with `fields`, every field), counting distinct work items into
    /// `host_pairs` as the levenshtein stage does. Cached results are reused;
    /// a Range request stores its new ones. Returns (chunk positions, results).
    fn dup_routed(
        &self,
        dup: &mut Dup,
        host: Host<'_>,
        request: &Request<'_>,
        pos: usize,
        routed: &[u64],
        fields: bool,
    ) -> Result<(Vec<u64>, Vec<Memo>), EngineError> {
        let mut items = Vec::new();
        try_reserve_exact(&mut items, routed.len(), "routed pairs")?;
        let mut counted = 0usize;
        for &t in routed {
            let t = t as usize;
            let (i, j) = match request {
                Request::Range { work, .. } => {
                    counted += usize::from(pos + t >= dup.watermark);
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
        dup.host_pairs += counted;
        if items.is_empty() {
            return Ok((Vec::new(), Vec::new()));
        }
        let kernel = host.kernel;
        let (a, b) = (host.a, host.b);
        let mut memos: Vec<Option<Memo>> = Vec::new();
        try_reserve_exact(&mut memos, items.len(), "routed pairs")?;
        let mut misses = Vec::new();
        for (k, &(_, i, j)) in items.iter().enumerate() {
            let memo = dup
                .cache
                .as_ref()
                .and_then(|cache| cache.pairs.get(&(i, j)).copied())
                .filter(|memo| memo.fields.is_some() || !fields);
            if memo.is_none() {
                seqblaze_core::detail::try_reserve(&mut misses, 1, "routed pairs")?;
                misses.push(k);
            }
            memos.push(memo);
        }
        if !misses.is_empty() {
            let request_fields = if fields {
                dup.field_index
                    .iter()
                    .fold(DetailRequest::NONE, |request, &field| request.with(field))
            } else {
                DetailRequest::NONE
            };
            let field_index = dup.field_index;
            let cost = kernel.pair_cost(a, b, fields);
            let limit = self.workers.count(misses.len(), cost);
            let parts = sched::map_ranges(misses.len(), limit, &|start, end| {
                let mut keys = Vec::new();
                try_reserve_exact(&mut keys, end - start, "routed results")?;
                let mut rows = DetailColumns::new(kernel.detail_fields(), request_fields);
                let mut ticker =
                    seqblaze_core::cancel::Ticker::new(seqblaze_core::cancel::Site::Routed);
                for &k in &misses[start..end] {
                    let (_, i, j) = items[k];
                    let key = if fields {
                        kernel.evaluate_detailed(a.get(i), b.get(j), None, None, &mut rows)?
                    } else {
                        kernel.evaluate(a.get(i), b.get(j), None)?
                    };
                    keys.push(key.ok_or_else(|| {
                        EngineError::Device("a routed pair has no score without a cutoff".into())
                    })?);
                    ticker.tick()?;
                }
                ticker.finish()?;
                Ok((keys, rows))
            })?;
            let mut next = misses.iter();
            for (keys, rows) in parts {
                let mut columns: [Option<Vec<f64>>; 3] = [None, None, None];
                for column in rows.into_columns() {
                    if let (DetailData::Float64(values), Some(k)) = (
                        column.data,
                        field_index.iter().position(|&field| field == column.field),
                    ) {
                        columns[k] = Some(values);
                    }
                }
                if fields
                    && columns
                        .iter()
                        .any(|column| column.as_ref().is_none_or(|v| v.len() != keys.len()))
                {
                    return Err(EngineError::Device(
                        "routed pair results do not match the routed list".into(),
                    ));
                }
                for (row, &key) in keys.iter().enumerate() {
                    let k = *next.next().ok_or_else(|| {
                        EngineError::Device("routed pair results exceed the routed list".into())
                    })?;
                    let fields = fields.then(|| {
                        let bits = |c: usize| columns[c].as_ref().map_or(0, |v| v[row].to_bits());
                        [bits(0), bits(1), bits(2)]
                    });
                    memos[k] = Some(Memo { key, fields });
                }
            }
            if next.next().is_some() {
                return Err(EngineError::Device(
                    "routed pair results do not match the routed list".into(),
                ));
            }
            if let (Request::Range { .. }, Some(cache)) = (request, dup.cache.as_mut()) {
                for &k in &misses {
                    let (_, i, j) = items[k];
                    let Some(memo) = memos[k] else { continue };
                    let fresh = !cache.pairs.contains_key(&(i, j));
                    if fresh && !cache.room(1) {
                        break;
                    }
                    cache.pairs.insert((i, j), memo);
                    cache.items += usize::from(fresh);
                }
            }
        }
        let mut positions = Vec::new();
        try_reserve_exact(&mut positions, items.len(), "routed positions")?;
        positions.extend(items.iter().map(|&(t, _, _)| t));
        let memos = memos
            .into_iter()
            .map(|memo| {
                memo.ok_or_else(|| EngineError::Device("a routed pair has no host result".into()))
            })
            .collect::<Result<Vec<Memo>, EngineError>>()?;
        Ok((positions, memos))
    }

    /// Write the routed results into the chunk's buffers (stream-ordered after
    /// the DP kernels, before any sink kernel).
    fn dup_scatter(
        &self,
        dup: &mut Dup,
        positions: &[u64],
        memos: &[Memo],
        params: &SbDup,
        dist: u64,
        fields: bool,
    ) -> Result<(), EngineError> {
        let stream = &self.stream;
        let mut keys: Vec<i32> = Vec::new();
        try_reserve_exact(&mut keys, memos.len(), "routed scatter keys")?;
        keys.extend(memos.iter().map(|memo| memo.key));
        let mut bits: Vec<u64> = Vec::new();
        if fields {
            try_reserve_exact(
                &mut bits,
                memos.len().saturating_mul(3),
                "routed scatter fields",
            )?;
            bits.extend(memos.iter().flat_map(|memo| memo.fields.unwrap_or([0; 3])));
        }
        let uploaded = Session::upload(stream, &mut dup.scatter_pos, positions).and_then(|p| {
            let d = Session::upload(stream, &mut dup.scatter_d, &keys)?;
            let f = if fields {
                Session::upload(stream, &mut dup.scatter_f, &bits)?
            } else {
                0
            };
            Ok((p, d, f))
        });
        let (pos, d, f) = uploaded.map_err(|error| self.fail("uploading routed results", error))?;
        let count = positions.len() as u64;
        let config = self.grid(count);
        let mut launch = stream.launch_builder(&self.device.0.kernels.dup_scatter);
        launch
            .arg(&pos)
            .arg(&d)
            .arg(&f)
            .arg(&count)
            .arg(&dist)
            .arg(&params.dh)
            .arg(&params.ds)
            .arg(&params.tm);
        // SAFETY: matches sb_dup_scatter; `pos` and `d` hold `count` entries,
        // `f` three per entry (or null with null field outputs), and the
        // outputs hold the chunk's n items.
        unsafe { launch.launch(config) }
            .map_err(|error| self.fail("launching sb_dup_scatter", error))?;
        Ok(())
    }
}
