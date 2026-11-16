//! K3 native `duplex(model="pf")` stage of a CUDA session
//! (docs/wiki/gpu/duplex-pf-cuda.md §4-§8): the weight upload, the closed-form, DP and
//! routing launches, the host computation of routed pairs, the download of Ẑ and its
//! validity check, and the 64-bit-key batch-distance port. Chunk planning, the tier
//! predicate and launch geometry are in `plan.rs` (`pf_*`).
//!
//! The device computes only Ẑ, one F53 value (M, e) per work item. The host turns each
//! into the term Dyadic {M, e − 52} and the key through the kernel's one term-to-key map
//! (`PairDistanceKernel::key_of_term`, §7.1), and every sink runs the engine core's host
//! code on those keys and terms (`exec::run_batch`, `topk::run_batch`,
//! `matrix::run_batch`, `next_tile_batch`). Routed pairs run on host workers through the
//! CPU kernel's term port (`evaluate_term`) while the device runs the rest; their Ẑ is
//! written into the chunk's host copy before the validity check. Later sweeps reuse
//! routed positions and Ẑ with the levenshtein stage's rules (levenshtein-cuda.md §6).
use std::{collections::HashMap, sync::Arc};

use cudarc::driver::{CudaSlice, DeviceRepr, DriverError, LaunchConfig, PushKernelArg};
use seqblaze_core::{
    detail::{try_reserve_exact, DetailColumns},
    kernel::{BatchDistance, CellClass, DistanceVisit, TermVisit},
    sched,
    sink::rowsum::Dyadic,
    validate::EngineError,
    work::{Orientation, PairKind, WorkSource},
};

use crate::{
    device::{guard, Device},
    launch::Settings,
    plan::{
        dup_c, pf_launches, pf_plan_chunk, pf_settings, DupGeometry, DupLimits, DupPlan, PfLengths,
        PfProfile, Request, DUP_CODES, DUP_GROUPS, MAX_GRID, SLOT_BYTES,
    },
    session::{
        dev_alloc, failure, failure_ordinal, ptr, Host, SbSpace, Scratch, Session, Table, BOTH,
        LIST, RECT, TRI, ZIP,
    },
    PfDeviceCall,
};

/// Entries the routed cache holds at most, as levenshtein's.
const CACHE_ITEMS: usize = 1 << 20;
/// `SbPf::tier` of the DP launches; must match SB_PF_* in kernels/duplex_pf.cuh.
const TIER_SHARED: u64 = 1;
const TIER_GLOBAL: u64 = 2;
/// `SbPf::poison` when no poison byte is set.
const NO_POISON: u64 = 256;

/// #[repr(C)] mirror of `SbPf` in kernels/duplex_pf.cuh; all fields are 8 bytes.
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct SbPf {
    w: u64,
    lam: u64,
    start: u64,
    end: u64,
    stack: u64,
    bulge: u64,
    i11: u64,
    i12: u64,
    i21: u64,
    i22: u64,
    gen: u64,
    loop_out: u64,
    loop_in: u64,
    group_lanes: u64,
    band_bytes: u64,
    pair_cells: u64,
    s_max: u64,
    s_block: u64,
    tier: u64,
    stride: u64,
    slots: u64,
    slot: u64,
    out: u64,
    poison: u64,
    /// Pack format 2 (nndb2024-dna.md §5.4.5): N_P, N_C and the device address of CLS.
    n_p: u64,
    n_c: u64,
    cls: u64,
    /// C2 bounded resident stage: u8* skip flags (0 here: no item is skipped).
    skip: u64,
    #[cfg(feature = "testing")]
    resident_stats: u64,
}

// SAFETY: plain #[repr(C)] struct of u64 fields in the CUDA struct's order; passed by
// value as a kernel parameter.
unsafe impl DeviceRepr for SbPf {}

/// Ẑ = (M, e) from a CPU term Dyadic {M, E} = M·2^E: e = E + 52, zero stays (0, 0).
fn zhat_of(term: Dyadic) -> (u64, i64) {
    if term.m == 0 {
        (0, 0)
    } else {
        (term.m, term.e.wrapping_add(52))
    }
}

/// The term Dyadic {M, e − 52} of Ẑ = (M, e) (§7.1); zero stays {0, 0}.
fn term_of(m: u64, e: i64) -> Dyadic {
    if m == 0 {
        Dyadic::ZERO
    } else {
        Dyadic { m, e: e - 52 }
    }
}

/// Ẑ as (M, e).
type Zhat = (u64, i64);

/// Routed chunk positions of one chunk, shared between the cache and a stage.
/// Routed chunk positions: the fallibly reserved vector itself, shared by a
/// fixed-size `Arc` (no second, input-scaled copy; F4 round 3).
type Positions = Arc<Vec<u64>>;

#[derive(Default)]
struct Cache {
    /// Range requests: (chunk start, request end) -> (chunk length, routed positions).
    routes: HashMap<(usize, usize), (usize, Positions)>,
    /// Ẑ of routed pairs (i, j).
    pairs: HashMap<(usize, usize), (u64, i64)>,
    items: usize,
}

impl Cache {
    fn room(&self, entries: usize) -> bool {
        self.items.saturating_add(entries) <= CACHE_ITEMS
    }
}

/// Per-call K3 state of a session.
pub(crate) struct Pf {
    /// Kernel parameters without the per-launch fields.
    params: SbPf,
    /// The weight entries, (M, e) interleaved; read through `SbPf::w`.
    _weights: CudaSlice<u64>,
    /// The CLS map (u8), read through `SbPf::cls`.
    _cls: CudaSlice<u8>,
    profile: PfProfile,
    /// Largest n of table a (the O(1) routing skip).
    max_n: u64,
    /// Block cap of the DP kernels.
    block: u32,
    limits: DupLimits,
    resident_threads: u64,
    zhat: Scratch<u64>,
    slots: Scratch<u64>,
    route_counts: Scratch<u64>,
    route_offsets: Scratch<u64>,
    route_pos: Scratch<u64>,
    /// The staged chunk's Ẑ on the host, (M, e as bits) interleaved.
    pub(crate) host: Vec<u64>,
    /// Distinct work items routed to the host so far (`execution.host_pairs`).
    pub(crate) host_pairs: usize,
    watermark: usize,
    cache: Option<Cache>,
}

impl Pf {
    /// Upload the weights (§4).
    pub(crate) fn new(
        (stream, device): (&Arc<cudarc::driver::CudaStream>, &Device),
        settings: &Settings,
        call: &PfDeviceCall,
        (a, b): (&Table, &Table),
        cache: bool,
    ) -> Result<Self, EngineError> {
        let fail = |what: &str, error| failure_ordinal(device, what, error);
        let settings = &pf_settings(settings);
        let o = &call.offsets;
        let total = o[12];
        // Shape, CLS and every table's offset (nndb2024-dna.md §5.4.2): START_1 at
        // 25·N_P, END at 50·N_P, ...
        if !call.layout_ok() || call.m.len() != total || call.e.len() != total || total < 2 {
            return Err(EngineError::Device(
                "the duplex pf device pack is inconsistent; use device='cpu'".into(),
            ));
        }
        let mut flat = Vec::new();
        try_reserve_exact(&mut flat, 2 * total, "duplex pf device pack")?;
        for (&m, &e) in call.m.iter().zip(&call.e) {
            flat.push(m);
            flat.push(e as u64);
        }
        // SAFETY: fully written by the copy below; total >= 2.
        let mut weights = unsafe { dev_alloc::<u64>(stream, flat.len()) }
            .map_err(|error| fail("allocation", error))?;
        crate::session::htod(stream, &flat[..], &mut weights)
            .map_err(|error| fail("uploading the duplex pf pack", error))?;
        // SAFETY: fully written by the copy below; (Λ + 1)² ≥ 1 bytes.
        let mut cls = unsafe { dev_alloc::<u8>(stream, call.cls.len()) }
            .map_err(|error| fail("allocation", error))?;
        crate::session::htod(stream, &call.cls[..], &mut cls)
            .map_err(|error| fail("uploading the duplex pf pack", error))?;
        // K3 stages the codes only: q = 400 (no STATES, nndb2024-dna.md §5.4.4).
        let limits = DupLimits {
            s_max: device.0.dup_s_max,
            s_block: device.0.pf_s_block,
            codes: DUP_CODES,
        };
        let params = SbPf {
            w: ptr(&weights, stream),
            lam: call.loop_max as u64,
            start: o[0] as u64,
            end: o[2] as u64,
            stack: o[3] as u64,
            bulge: o[4] as u64,
            i11: o[5] as u64,
            i12: o[6] as u64,
            i21: o[7] as u64,
            i22: o[8] as u64,
            gen: o[9] as u64,
            loop_out: o[10] as u64,
            loop_in: o[11] as u64,
            group_lanes: settings.group_lanes,
            band_bytes: settings.band_bytes,
            pair_cells: settings.pair_cells,
            s_max: limits.s_max,
            s_block: limits.s_block,
            poison: poison(),
            n_p: call.n_p as u64,
            n_c: call.n_c as u64,
            cls: ptr(&cls, stream),
            ..SbPf::default()
        };
        let lam = call.loop_max as u64;
        let max_n = a
            .offsets()
            .windows(2)
            .map(|pair| (pair[1] - pair[0]) as u64)
            .max()
            .unwrap_or(0);
        Ok(Self {
            params,
            _weights: weights,
            _cls: cls,
            profile: PfProfile::new(b.offsets(), lam, call.n_c as u64)?,
            max_n,
            block: device.0.pf_block,
            limits,
            resident_threads: device.0.resident_threads,
            zhat: Scratch::new(),
            slots: Scratch::new(),
            route_counts: Scratch::new(),
            route_offsets: Scratch::new(),
            route_pos: Scratch::new(),
            host: Vec::new(),
            host_pairs: 0,
            watermark: 0,
            cache: cache.then(Cache::default),
        })
    }

    fn geometry(&self, settings: &Settings) -> DupGeometry {
        DupGeometry {
            threads_per_block: u64::from(settings.threads_per_block.min(self.block)),
            resident_threads: self.resident_threads,
            limits: self.limits,
            slot_bytes: SLOT_BYTES,
        }
    }

    /// Conservative O(1) check: can any pair be routed? D_pf grows with n and m and
    /// shrinks with g, so the longest a against the longest b at the smallest group
    /// size in use bounds every pair's depth.
    fn may_route(&self, settings: &Settings) -> bool {
        let lam = self.profile.lam;
        let (n, m) = (self.max_n, self.profile.max_m());
        if n < 2 || m < 2 {
            return false;
        }
        let g = if settings.group_lanes > 0 {
            settings.group_lanes
        } else {
            DUP_GROUPS[0]
        };
        let steps = n
            .saturating_mul(m.div_ceil(g))
            .max(m.saturating_mul(n.div_ceil(g)));
        steps.saturating_mul(dup_c(n, m, lam)) > settings.pair_cells
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

/// The pair (i, j) of chunk item `t` of `request` from position `pos` (error messages
/// and routed pairs), or None for a rectangle cell the caller discards.
fn pair_of(request: &Request<'_>, pos: usize, t: usize) -> Option<(usize, usize)> {
    match request {
        Request::Range { work, .. } => work.range(pos + t, pos + t + 1).next(),
        Request::Pairs(pairs) => Some(pairs[pos + t]),
        Request::Rect { r0, c0, width, .. } => {
            let cell = pos + t;
            Some((*r0 + cell / *width, *c0 + cell % *width))
        }
    }
}

impl Session {
    /// Stage the K3 chunk of `request` starting at item `pos`: Ẑ of every item in
    /// `pf.host[..2n]`, checked (§5.4). Returns n.
    pub(crate) fn pf_stage(
        &mut self,
        host: Host<'_, i64>,
        request: &Request<'_>,
        pos: usize,
    ) -> Result<usize, EngineError> {
        let mut pf = self.pf.take().expect("duplex pf session");
        let result = self.pf_stage_with(&mut pf, host, request, pos);
        self.pf = Some(pf);
        result
    }

    fn pf_space(&mut self, request: &Request<'_>, pos: usize) -> Result<SbSpace, EngineError> {
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
        Ok(space)
    }

    fn pf_stage_with(
        &mut self,
        pf: &mut Pf,
        host: Host<'_, i64>,
        request: &Request<'_>,
        pos: usize,
    ) -> Result<usize, EngineError> {
        let settings = pf_settings(&self.settings);
        let mut space = self.pf_space(request, pos)?;
        let stream = Arc::clone(&self.stream);
        let geometry = pf.geometry(&settings);
        let lengths = PfLengths {
            a_off: self.a.offsets(),
            b_off: self.table_b().offsets(),
            profile: &pf.profile,
        };
        let plan = pf_plan_chunk(&lengths, &settings, &geometry, request, pos);
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
        let out = ptr(pf.zhat.ensure(&stream, 2 * n).map_err(alloc)?, &stream);
        // Testing builds with a poison byte: refill Ẑ every chunk, so an item no kernel
        // writes shows the poison byte (and fails the validity check), never a previous
        // chunk's value (the allocation-time fill covers only the first chunk).
        #[cfg(feature = "testing")]
        if let Some(byte) = crate::launch::poison() {
            self.fill(out, u32::from(byte), 16 * n, false)?;
            crate::launch::count_poisoned(16 * n as u64);
        }
        let mut params = pf.params;
        params.out = out;
        {
            let tables = self.tables();
            let config = self.grid(n as u64);
            let mut launch = stream.launch_builder(&self.device.0.kernels.pf_closed);
            launch.arg(&tables).arg(&space).arg(&params);
            // SAFETY: matches sb_pf_closed; `out` holds n records of 16 bytes.
            unsafe { launch.launch(config) }
                .map_err(|error| self.fail("launching sb_pf_closed", error))?;
        }
        // A Range chunk planned before (same start, end and length) holds the same
        // items: its routed positions are reused, not recounted.
        let key = match request {
            Request::Range { end, .. } => Some((pos, *end)),
            _ => None,
        };
        let cached = key.and_then(|key| {
            let (len, positions) = pf.cache.as_ref()?.routes.get(&key)?;
            (*len == n).then(|| Arc::clone(positions))
        });
        let routed: Positions = match cached {
            Some(positions) => positions,
            None if pf.may_route(&settings) => {
                let positions: Positions = Arc::new(self.pf_route(pf, &space, &params, n)?);
                if let (Some(key), Some(cache)) = (key, pf.cache.as_mut()) {
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
        self.pf_variants(pf, (&space, &plan), (&settings, &geometry), &params)?;
        // Host workers compute the routed pairs while the device runs.
        let routed_values = self.pf_routed(pf, host, request, pos, &routed)?;
        if let Request::Range { .. } = request {
            pf.watermark = pf.watermark.max(pos + n);
        }
        let mut values = std::mem::take(&mut pf.host);
        values.clear();
        try_reserve_exact(&mut values, 2 * n, "duplex pf chunk")?;
        values.resize(2 * n, 0);
        let fetched = self
            .download(pf.zhat.get(), 2 * n, &mut values)
            .and_then(|()| self.sync());
        pf.host = values;
        fetched?;
        for &(t, (m, e)) in &routed_values {
            pf.host[2 * t] = m;
            pf.host[2 * t + 1] = e as u64;
        }
        // The validity check (§5.4, QKP12): every Ẑ of the chunk is an F53 value.
        for t in 0..n {
            let (m, e) = (pf.host[2 * t], pf.host[2 * t + 1] as i64);
            if !crate::f53_canonical(m, e) {
                let (i, j) = pair_of(request, pos, t).unwrap_or((usize::MAX, usize::MAX));
                return Err(EngineError::Device(format!(
                    "the CUDA backend returned an invalid partition-function value for pair \
                     ({i}, {j}); no result was published"
                )));
            }
        }
        Ok(n)
    }

    /// Routing (§6.4): count, scan and write of the routed items, as G1's Hits, so the
    /// list is in item order. Returns their chunk positions.
    fn pf_route(
        &self,
        pf: &mut Pf,
        space: &SbSpace,
        params: &SbPf,
        n: usize,
    ) -> Result<Vec<u64>, EngineError> {
        let stream = &self.stream;
        let alloc = |error| failure_ordinal(&self.device, "allocation", error);
        let tables = self.tables();
        let (blocks, config) = self.tiles(n);
        let counts = ptr(
            pf.route_counts.ensure(stream, blocks).map_err(alloc)?,
            stream,
        );
        let kernels = &self.device.0.kernels;
        let mut launch = stream.launch_builder(&kernels.pf_route_count);
        launch.arg(&tables).arg(space).arg(params).arg(&counts);
        // SAFETY: matches sb_pf_route_count; one count slot per block.
        unsafe { launch.launch(config) }
            .map_err(|error| self.fail("launching sb_pf_route_count", error))?;
        let mut block_counts = seqblaze_core::sink::try_filled(blocks, 0u64, "route block counts")?;
        let mut offsets = Vec::new();
        try_reserve_exact(&mut offsets, blocks, "route block offsets")?;
        self.download(pf.route_counts.get(), blocks, &mut block_counts)?;
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
        let offsets = Session::upload(stream, &mut pf.route_offsets, &offsets).map_err(alloc)?;
        let out = ptr(pf.route_pos.ensure(stream, total).map_err(alloc)?, stream);
        let mut launch = stream.launch_builder(&kernels.pf_route_write);
        launch
            .arg(&tables)
            .arg(space)
            .arg(params)
            .arg(&offsets)
            .arg(&out);
        // SAFETY: matches sb_pf_route_write; `out` holds `total` positions.
        unsafe { launch.launch(config) }
            .map_err(|error| self.fail("launching sb_pf_route_write", error))?;
        let mut positions = seqblaze_core::sink::try_filled(total, 0u64, "routed positions")?;
        self.download(pf.route_pos.get(), total, &mut positions)?;
        self.sync()?;
        Ok(positions)
    }

    /// Launch the variants (g, tier) that may own a pair of the chunk, each with the
    /// grid and block `plan.rs` sets; global-tier groups get one band slot each, within
    /// the slot budget.
    fn pf_variants(
        &self,
        pf: &mut Pf,
        (space, plan): (&SbSpace, &DupPlan),
        (settings, geometry): (&Settings, &DupGeometry),
        params: &SbPf,
    ) -> Result<(), EngineError> {
        let stream = &self.stream;
        let tables = self.tables();
        let kernels = &self.device.0.kernels.pf;
        for launch in pf_launches(settings, geometry, plan) {
            let blocks = launch.blocks(settings, plan);
            debug_assert!(blocks <= MAX_GRID);
            let mut params = *params;
            params.stride = launch.stride;
            if launch.global {
                params.tier = TIER_GLOBAL;
                // Written for each pair before any read; never zeroed (§6.3).
                let words = (blocks * launch.per_block * launch.slot / 8) as usize;
                let slots = pf
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
            builder.arg(&tables).arg(space).arg(&params);
            // SAFETY: matches the K3 DP kernels; `out` holds n records, the slots
            // blocks·per_block·slot bytes, and the dynamic shared memory
            // per_block·stride bytes.
            unsafe { builder.launch(config) }.map_err(|error| {
                self.fail(
                    &format!("launching sb_pf (variant {})", launch.variant),
                    error,
                )
            })?;
        }
        Ok(())
    }

    /// Ẑ of the routed items on host workers with the CPU kernel's term port, counting
    /// distinct work items into `host_pairs` as the levenshtein stage does. Cached
    /// values are reused; a Range request stores its new ones. A routed rectangle cell
    /// the caller discards gets (0, 0), never read. Returns (chunk position, Ẑ).
    fn pf_routed(
        &self,
        pf: &mut Pf,
        host: Host<'_, i64>,
        request: &Request<'_>,
        pos: usize,
        routed: &[u64],
    ) -> Result<Vec<(usize, Zhat)>, EngineError> {
        let mut out = Vec::new();
        try_reserve_exact(&mut out, routed.len(), "routed pairs")?;
        let mut items = Vec::new();
        try_reserve_exact(&mut items, routed.len(), "routed pairs")?;
        let mut counted = 0usize;
        for &t in routed {
            let t = t as usize;
            let (i, j) = match request {
                Request::Range { work, .. } => {
                    counted += usize::from(pos + t >= pf.watermark);
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
                        None => {
                            out.push((t, (0, 0)));
                            continue;
                        }
                        Some(canonical) => counted += usize::from(canonical),
                    }
                    (i, j)
                }
            };
            items.push((t, i, j));
        }
        pf.host_pairs += counted;
        if items.is_empty() {
            return Ok(out);
        }
        let mut values: Vec<Option<(u64, i64)>> = Vec::new();
        try_reserve_exact(&mut values, items.len(), "routed pairs")?;
        let mut misses = Vec::new();
        for (k, &(_, i, j)) in items.iter().enumerate() {
            let value = pf
                .cache
                .as_ref()
                .and_then(|cache| cache.pairs.get(&(i, j)).copied());
            if value.is_none() {
                seqblaze_core::detail::try_reserve(&mut misses, 1, "routed pairs")?;
                misses.push(k);
            }
            values.push(value);
        }
        if !misses.is_empty() {
            let (kernel, a, b) = (host.kernel, host.a, host.b);
            let cost = kernel.pair_cost(a, b, false);
            let limit = self.workers.count(misses.len(), cost);
            let parts = sched::map_ranges(misses.len(), limit, &|start, end| {
                let mut part = Vec::new();
                try_reserve_exact(&mut part, end - start, "routed results")?;
                let mut ticker =
                    seqblaze_core::cancel::Ticker::new(seqblaze_core::cancel::Site::Routed);
                for &k in &misses[start..end] {
                    let (_, i, j) = items[k];
                    let (_, term) = kernel.evaluate_term(a.get(i), b.get(j))?;
                    part.push(zhat_of(term));
                    ticker.tick()?;
                }
                ticker.finish()?;
                Ok(part)
            })?;
            for (&k, value) in misses.iter().zip(parts.into_iter().flatten()) {
                values[k] = Some(value);
            }
            if let (Request::Range { .. }, Some(cache)) = (request, pf.cache.as_mut()) {
                for &k in &misses {
                    let (_, i, j) = items[k];
                    let Some(value) = values[k] else { continue };
                    let fresh = !cache.pairs.contains_key(&(i, j));
                    if fresh && !cache.room(1) {
                        break;
                    }
                    cache.pairs.insert((i, j), value);
                    cache.items += usize::from(fresh);
                }
            }
        }
        for (&(t, _, _), value) in items.iter().zip(values) {
            let value = value
                .ok_or_else(|| EngineError::Device("a routed pair has no host result".into()))?;
            out.push((t, value));
        }
        Ok(out)
    }
}

/// A session bound to one K3 call: the 64-bit-key batch-distance port. Keys come from
/// the kernel's term-to-key map (§7.1); no detail field crosses the bus (QKP16).
pub(crate) struct PfBound<'s> {
    pub(crate) session: &'s mut Session,
    pub(crate) host: Host<'s, i64>,
}

impl PfBound<'_> {
    /// Keys (and terms) of the staged chunk's n items, appended in item order.
    fn convert(
        &self,
        n: usize,
        keys: &mut Vec<i64>,
        mut terms: Option<&mut Vec<Dyadic>>,
    ) -> Result<(), EngineError> {
        let host = &self.session.pf.as_ref().expect("duplex pf session").host;
        let kernel = self.host.kernel;
        try_reserve_exact(keys, n, "duplex pf keys")?;
        if let Some(terms) = terms.as_deref_mut() {
            try_reserve_exact(terms, n, "duplex pf terms")?;
        }
        for t in 0..n {
            let term = term_of(host[2 * t], host[2 * t + 1] as i64);
            keys.push(kernel.key_of_term(term)?);
            if let Some(terms) = terms.as_deref_mut() {
                terms.push(term);
            }
        }
        Ok(())
    }

    /// Ẑ of every item of a request, (M, e) in item order (the testing hook).
    pub(crate) fn zhat(
        &mut self,
        request: &Request<'_>,
    ) -> Result<(Vec<u64>, Vec<i64>), EngineError> {
        let total = request.end();
        let (mut m, mut e) = (Vec::new(), Vec::new());
        try_reserve_exact(&mut m, total, "duplex pf Ẑ")?;
        try_reserve_exact(&mut e, total, "duplex pf Ẑ")?;
        let mut pos = 0;
        while pos < total {
            let n = self.session.pf_stage(self.host, request, pos)?;
            let host = &self.session.pf.as_ref().expect("duplex pf session").host;
            for t in 0..n {
                m.push(host[2 * t]);
                e.push(host[2 * t + 1] as i64);
            }
            pos += n;
        }
        Ok((m, e))
    }

    fn refuse_details() -> EngineError {
        EngineError::Parameter(
            "internal error: the duplex pf device port delivers no detail fields".into(),
        )
    }
}

impl BatchDistance<i64> for PfBound<'_> {
    fn delivers_details(&self) -> bool {
        false
    }

    fn range(
        &mut self,
        work: &WorkSource,
        start: usize,
        end: usize,
        details: bool,
        visit: &mut DistanceVisit<'_, i64>,
    ) -> Result<(), EngineError> {
        if details {
            return Err(Self::refuse_details());
        }
        let request = Request::Range { work, end };
        let mut keys = Vec::new();
        let mut pos = start;
        while pos < end {
            let n = self.session.pf_stage(self.host, &request, pos)?;
            keys.clear();
            self.convert(n, &mut keys, None)?;
            visit(pos, &keys, None)?;
            pos += n;
        }
        Ok(())
    }

    fn pairs(
        &mut self,
        pairs: &[(usize, usize)],
        out: &mut Vec<i64>,
        details: Option<&mut DetailColumns>,
    ) -> Result<(), EngineError> {
        if details.is_some() {
            return Err(Self::refuse_details());
        }
        let request = Request::Pairs(pairs);
        let mut pos = 0;
        while pos < pairs.len() {
            let n = self.session.pf_stage(self.host, &request, pos)?;
            self.convert(n, out, None)?;
            pos += n;
        }
        Ok(())
    }

    fn rect(
        &mut self,
        rows: (usize, usize),
        cols: (usize, usize),
        class: &CellClass<'_>,
        out: &mut Vec<i64>,
        details: Option<&mut DetailColumns>,
    ) -> Result<(), EngineError> {
        if details.is_some() {
            return Err(Self::refuse_details());
        }
        let width = cols.1 - cols.0;
        let cells = (rows.1 - rows.0) * width;
        let request = Request::Rect {
            r0: rows.0,
            c0: cols.0,
            width,
            end: cells,
            class: Some(class),
        };
        let mut pos = 0;
        while pos < cells {
            let n = self.session.pf_stage(self.host, &request, pos)?;
            self.convert(n, out, None)?;
            pos += n;
        }
        Ok(())
    }

    fn range_terms(
        &mut self,
        work: &WorkSource,
        start: usize,
        end: usize,
        visit: &mut TermVisit<'_, i64>,
    ) -> Result<(), EngineError> {
        let request = Request::Range { work, end };
        let (mut keys, mut terms) = (Vec::new(), Vec::new());
        let mut pos = start;
        while pos < end {
            let n = self.session.pf_stage(self.host, &request, pos)?;
            keys.clear();
            terms.clear();
            self.convert(n, &mut keys, Some(&mut terms))?;
            visit(pos, &keys, &terms)?;
            pos += n;
        }
        Ok(())
    }
}

/// Evaluate one F53 operation elementwise on the device (§5.5, §8 item 7); operands are
/// canonical (checked by the caller). Testing only.
pub(crate) fn f53_eval(
    device: &Device,
    mul: bool,
    (xm, xe): (&[u64], &[i64]),
    (ym, ye): (&[u64], &[i64]),
) -> Result<(Vec<u64>, Vec<i64>), EngineError> {
    let count = xm.len();
    if count == 0 {
        return Ok((Vec::new(), Vec::new()));
    }
    guard(|| {
        let label = format!("cuda:{}", device.info().ordinal);
        let fail = |what: &str, error: DriverError| failure(&label, what, error);
        let stream = device
            .0
            .ctx
            .new_stream()
            .map_err(|error| fail("creating a stream", error))?;
        let upload = |host: &[u64]| -> Result<CudaSlice<u64>, DriverError> {
            // SAFETY: fully written by the copy below.
            let mut slice = unsafe { dev_alloc::<u64>(&stream, host.len()) }?;
            crate::session::htod(&stream, host, &mut slice)?;
            Ok(slice)
        };
        let bits = |values: &[i64]| -> Vec<u64> { values.iter().map(|&v| v as u64).collect() };
        let uploaded =
            upload(xm).and_then(|a| Ok((a, upload(&bits(xe))?, upload(ym)?, upload(&bits(ye))?)));
        let (dxm, dxe, dym, dye) = uploaded.map_err(|error| fail("uploading operands", error))?;
        // SAFETY (both): written by sb_f53_eval before the copies read them.
        let om = unsafe { dev_alloc::<u64>(&stream, count) }
            .map_err(|error| fail("allocation", error))?;
        let oe = unsafe { dev_alloc::<u64>(&stream, count) }
            .map_err(|error| fail("allocation", error))?;
        let op: i32 = i32::from(mul);
        let pointers = [&dxm, &dxe, &dym, &dye, &om, &oe].map(|slice| ptr(slice, &stream));
        let n = count as u64;
        let threads = 256u64;
        let config = LaunchConfig {
            grid_dim: (n.div_ceil(threads).clamp(1, MAX_GRID) as u32, 1, 1),
            block_dim: (threads as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut launch = stream.launch_builder(&device.0.kernels.f53_eval);
        launch.arg(&op);
        for pointer in &pointers {
            launch.arg(pointer);
        }
        launch.arg(&n);
        // SAFETY: matches sb_f53_eval; every array holds `count` elements.
        unsafe { launch.launch(config) }.map_err(|error| fail("launching sb_f53_eval", error))?;
        let mut m = vec![0u64; count];
        let mut e = vec![0u64; count];
        crate::session::dtoh(&stream, &om, &mut m[..])
            .and_then(|()| crate::session::dtoh(&stream, &oe, &mut e[..]))
            .and_then(|()| stream.synchronize())
            .map_err(|error| fail("device-to-host copy", error))?;
        Ok((m, e.into_iter().map(|v| v as i64).collect()))
    })
}
