//! The primer3 device stages (docs/wiki/gpu/primer3-cuda.md): the device pack,
//! the group launches of T2 `hairpin` and K2 `duplex` (kernels/primer3.cuh), and
//! K2's int64-key session path (§7): its stage, the routed pairs on host workers,
//! the Each, Hits and RowStats sinks on int64 keys, and the batch-distance port
//! for TopK, Matrix and Tiles.
//!
//! The int64 path is a separate session path (the alternative of §7): the int32
//! session, its sink kernels and their bytes are unchanged. Every item takes the
//! global tier (QPC-3's alternative): its cell table lives in a global slot of its
//! group. The planner (`plan.rs`) places each item: g, depth and K2 routing.
use std::{collections::HashMap, sync::Arc};

use cudarc::driver::{CudaSlice, CudaStream, DeviceRepr, DriverError, LaunchConfig, PushKernelArg};
use seqblaze_core::{
    detail::{
        try_reserve, try_reserve_exact, DetailColumns, DetailData, DetailRequest, CANONICAL_NAN,
    },
    exec::ExecOptions,
    kernel::{BatchDistance, CellClass, DistanceVisit, PairDistanceKernel},
    sched,
    seq::SeqTable,
    sink::{matrix, topk, Columns, Counts, RowState, RowStatsColumns, RunOutput, Sink},
    validate::{EngineError, ValidationReport},
    work::{Orientation, PairKind, WorkSource},
};

use crate::{
    device::{guard, Device},
    launch::Settings,
    plan::{p3_chunk_budget, p3_launches, p3_plan_k2, p3_slot_k2, P3Bucket, Request, P3_GROUPS},
    session::{
        dev_alloc, failure_ordinal, ptr, Chunk, Host, SbSpace, SbTables, Scratch, Session, BOTH,
        LIST, RECT, TRI, ZIP,
    },
    unary::SbUnaryTable,
    DeviceCall, DeviceRun, P3Scalars, Primer3Call,
};

/// `SbP3::poison` when no poison byte is set.
const NO_POISON: u64 = 256;
/// The items one batch of the K2 chunk scan builds (host memory only; results do not
/// depend on it).
const P3_ITEM_BATCH: usize = 4096;
/// Routed items a K2 chunk holds at most. A routed item never charges the
/// device depth budget, so without this bound a chunk of routed items grows to
/// `pairs_per_chunk`, with about 175-330 bytes of host state each (routed
/// list, memos, results, scatter keys; per-item figure derived from the
/// verifier's whole-call growth measurements of 2026-10-19). 2^16 keeps that
/// state near 20 MB (estimated) and leaves every host worker thousands of items
/// per chunk. Chunk boundaries never change a result.
const P3_ROUTED_PER_CHUNK: usize = 1 << 16;
/// Routed host results the cache holds at most (§6.6, §11; levenshtein's 2^20). An
/// entry is a 16 B (i, j) key and a 48 B `Memo`, 64 B plus one control byte. At
/// the map's 7/8 load factor 2^20 entries need 2^21 buckets, so a full cache
/// holds about 130 MiB, and the growth step to 2^21 buckets briefly holds the old
/// and new tables, about 195 MiB (estimated from the layout, not measured). Past
/// it the cache stops growing and later sweeps recompute: the same values. Only
/// work-source ranges fill it; a tile stream keeps no cache.
const CACHE_ITEMS: usize = 1 << 20;

/// #[repr(C)] mirror of `SbP3` in kernels/primer3.cuh; every field is 8 bytes.
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub(crate) struct SbP3 {
    pub tables: u64,
    pub ends: u64,
    pub c: P3Scalars,
    pub items: u64,
    pub count: u64,
    pub slots: u64,
    pub slot: u64,
    pub poison: u64,
    pub key: u64,
    pub found: u64,
    pub dg: u64,
    pub dh: u64,
    pub ds: u64,
    pub tm: u64,
    pub passed: u64,
    pub has_cutoff: u64,
    pub cutoff: u64,
    pub mask_each: u64,
    pub swap: u64,
    pub end1: u64,
}

// SAFETY: plain #[repr(C)] struct of u64 fields (P3Scalars is #[repr(C)] of u64)
// in the CUDA struct's order; passed by value as a kernel parameter.
unsafe impl DeviceRepr for SbP3 {}

const _: () = assert!(std::mem::size_of::<SbP3>() == 8 * 42);

/// The poison byte of a testing build (`set_cuda_poison`), else none.
fn poison() -> u64 {
    #[cfg(feature = "testing")]
    if let Some(byte) = crate::launch::poison() {
        return u64::from(byte);
    }
    NO_POISON
}

/// A call's tables and end terms on the device, with the launch parameters
/// that do not change between launches.
pub(crate) struct P3Pack {
    _pack: Arc<crate::pack_cache::Pack<u64, u64>>,
    pub(crate) params: SbP3,
}

impl P3Pack {
    /// Upload the pack once per call (§4); never zero bytes.
    pub(crate) fn upload(
        stream: &Arc<CudaStream>,
        device: &Device,
        call: &Primer3Call,
    ) -> Result<Self, EngineError> {
        Self::upload_on(stream, device, call, None)
    }
    pub(crate) fn upload_on(
        stream: &Arc<CudaStream>,
        device: &Device,
        call: &Primer3Call,
        execution: Option<&crate::interop::execution_stream::BorrowedExecutionStream>,
    ) -> Result<Self, EngineError> {
        if call.tables.is_empty() || call.ends.is_empty() {
            return Err(EngineError::Device(
                "the primer3 device pack is empty; use device='cpu'".into(),
            ));
        }
        let fail =
            |error: DriverError| failure_ordinal(device, "uploading the primer3 pack", error);
        let upload = |host: &[u64]| -> Result<CudaSlice<u64>, EngineError> {
            crate::interop::execution_stream::upload_on(stream, execution, host).map_err(fail)
        };
        let pack = crate::pack_cache::Pack::get(
            stream,
            device,
            execution,
            crate::pack_cache::Kind::Primer3,
            [
                crate::pack_cache::bytes(&call.tables),
                crate::pack_cache::bytes(&call.ends),
            ],
            || {
                let tables = crate::interop::Retained::on_owned_stream(
                    stream,
                    device,
                    upload(&call.tables)?,
                );
                let ends = upload(&call.ends)?;
                Ok::<_, EngineError>((tables.into_inner(), ends))
            },
        )?;
        let [tables, ends] = pack.addresses();
        let params = SbP3 {
            tables,
            ends,
            c: call.scalars,
            poison: poison(),
            swap: u64::from(call.swap),
            end1: u64::from(call.end1),
            ..SbP3::default()
        };
        Ok(Self {
            _pack: pack,
            params,
        })
    }
}

/// The sequence arguments of a group launch.
pub(crate) enum P3Args {
    /// T2: the uploaded table and the chunk's first sequence.
    Hairpin(SbUnaryTable, u64),
    /// K2: the pair tables and the chunk's pair space.
    Dimer(SbTables, SbSpace),
}

/// The device buffers the group launches reuse: item lists and table slots.
#[derive(Default)]
pub(crate) struct P3Buffers {
    items: Option<CudaSlice<u32>>,
    slots: Option<CudaSlice<u64>>,
    host_items: Vec<u32>,
}

/// `slot`, grown to at least `len` elements (never zero bytes).
fn ensure<'s, T: DeviceRepr>(
    stream: &Arc<CudaStream>,
    slot: &'s mut Option<CudaSlice<T>>,
    len: usize,
) -> Result<&'s mut CudaSlice<T>, DriverError> {
    if slot.as_ref().is_none_or(|slice| slice.len() < len) {
        *slot = None;
        // SAFETY: written (upload or kernel) before any read.
        *slot = Some(unsafe { dev_alloc::<T>(stream, len.max(1)) }?);
    }
    Ok(slot.as_mut().expect("allocated above"))
}

/// Launch every variant of the chunk's buckets (§6.5-§6.6): one item list
/// upload, one slot buffer large enough for every launch, then the launches
/// `p3_launches` gives each variant, all on `stream` in variant order.
#[allow(clippy::too_many_arguments)]
pub(crate) fn launch_buckets(
    (stream, device): (&Arc<CudaStream>, &Device),
    settings: &Settings,
    buckets: &[P3Bucket; 4],
    params: &SbP3,
    args: &P3Args,
    buffers: &mut P3Buffers,
) -> Result<(), EngineError> {
    let fail = |what: &str, error| failure_ordinal(device, what, error);
    let (kernels, cap, what) = match args {
        P3Args::Hairpin(..) => (
            &device.0.kernels.p3_hairpin,
            device.0.p3_block[0],
            "hairpin chunk items",
        ),
        P3Args::Dimer(..) => (
            &device.0.kernels.p3_dimer,
            device.0.p3_block[1],
            "duplex chunk items",
        ),
    };
    let threads = u64::from(settings.threads_per_block.min(cap).max(32));
    let mut plans = Vec::new();
    let mut host = std::mem::take(&mut buffers.host_items);
    host.clear();
    let mut slot_bytes = 0u64;
    for (variant, bucket) in buckets.iter().enumerate() {
        if bucket.items.is_empty() {
            continue;
        }
        let offset = host.len();
        try_reserve(&mut host, bucket.items.len(), what)?;
        host.extend_from_slice(&bucket.items);
        let g = P3_GROUPS[variant];
        let launches = p3_launches(
            settings,
            bucket.items.len(),
            g,
            threads,
            bucket.slot,
            bucket.depth,
        );
        for launch in &launches {
            slot_bytes = slot_bytes.max(launch.blocks * launch.per_block * launch.slot);
        }
        plans.push((variant, offset, launches));
    }
    if plans.is_empty() {
        buffers.host_items = host;
        return Ok(());
    }
    let items = ensure(stream, &mut buffers.items, host.len())
        .map_err(|error| fail("allocation", error))?;
    crate::session::htod(stream, &host[..], &mut items.slice_mut(0..host.len()))
        .map_err(|error| fail("uploading the item lists", error))?;
    let items_ptr = ptr(items, stream);
    buffers.host_items = host;
    // Written by each group before it reads (R2); never cleared between items.
    let slots = ensure(stream, &mut buffers.slots, (slot_bytes / 8) as usize)
        .map_err(|error| fail("allocation", error))?;
    let slots_ptr = ptr(slots, stream);
    for (variant, offset, launches) in plans {
        let g = P3_GROUPS[variant];
        for launch in launches {
            let mut params = *params;
            params.items = items_ptr + 4 * (offset + launch.start) as u64;
            params.count = launch.len as u64;
            params.slots = slots_ptr;
            params.slot = launch.slot;
            let config = LaunchConfig {
                grid_dim: (launch.blocks as u32, 1, 1),
                block_dim: ((launch.per_block * g) as u32, 1, 1),
                shared_mem_bytes: 0,
            };
            let mut builder = stream.launch_builder(&kernels[variant]);
            match args {
                P3Args::Hairpin(table, k0) => builder.arg(table).arg(k0).arg(&params),
                P3Args::Dimer(tables, space) => builder.arg(tables).arg(space).arg(&params),
            };
            // SAFETY: matches sb_p3_hairpin_g*/sb_p3_dimer_g*; the item list holds
            // `count` chunk positions, the slots blocks·per_block·slot bytes, and
            // every output buffer the chunk's items.
            unsafe { builder.launch(config) }
                .map_err(|error| fail(&format!("launching the primer3 kernel (g = {g})"), error))?;
        }
    }
    Ok(())
}

// ------------------------------------------------------------------- K2 session

/// `n` copies of `value` in a fallibly reserved vector: a refused host reservation
/// raises DeviceError instead of aborting (§9 step 6).
fn try_filled<T: Clone>(n: usize, value: T, what: &str) -> Result<Vec<T>, EngineError> {
    let mut out = Vec::new();
    try_reserve_exact(&mut out, n, what)?;
    out.resize(n, value);
    Ok(out)
}

/// The bits of a routed detail column, in a fallibly reserved vector (§9 step 6).
fn try_bits<T: Copy>(values: &[T], bits: impl Fn(T) -> u64) -> Result<Vec<u64>, EngineError> {
    let mut out = Vec::new();
    try_reserve_exact(&mut out, values.len(), "routed results")?;
    out.extend(values.iter().map(|&v| bits(v)));
    Ok(out)
}

/// A routed pair's host result: its key and, from a detail pass, `found` (0/1)
/// and the bits of `dh_kcal`, `ds_cal_per_k` and `tm_c`.
#[derive(Clone, Copy)]
struct Memo {
    key: i64,
    fields: Option<[u64; 4]>,
}

/// Per-call K2 state of a session.
pub(crate) struct P3 {
    pack: P3Pack,
    /// Indices of `found`, `dh_kcal`, `ds_cal_per_k` and `tm_c` in the kernel's fields.
    fields: [usize; 4],
    swap: bool,
    key: Scratch<i64>,
    found: Scratch<i32>,
    floats: [Scratch<u64>; 3],
    buffers: P3Buffers,
    scatter_pos: Scratch<u64>,
    scatter_d: Scratch<i64>,
    scatter_f: Scratch<u64>,
    host_keys: Vec<i64>,
    /// Distinct work items routed to the host so far (`execution.host_pairs`).
    pub(crate) host_pairs: usize,
    watermark: usize,
    /// Routed host results of Range requests, keyed by (i, j), reused by later
    /// sweeps (RowStats, TopK); `None` in a tile stream, which sweeps each pair once.
    /// K2 plans its routing on the host during the chunk scan, so no routed-position
    /// cache is kept (there is no device routing pass to skip).
    cache: Option<HashMap<(usize, usize), Memo>>,
}

impl P3 {
    pub(crate) fn new(
        stream: &Arc<CudaStream>,
        device: &Device,
        call: &Primer3Call,
        cache: bool,
    ) -> Result<Self, EngineError> {
        Ok(Self {
            pack: P3Pack::upload(stream, device, call)?,
            fields: call.fields,
            swap: call.swap,
            key: Scratch::new(),
            found: Scratch::new(),
            floats: [Scratch::new(), Scratch::new(), Scratch::new()],
            buffers: P3Buffers::default(),
            scatter_pos: Scratch::new(),
            scatter_d: Scratch::new(),
            scatter_f: Scratch::new(),
            host_keys: Vec::new(),
            host_pairs: 0,
            watermark: 0,
            cache: cache.then(HashMap::new),
        })
    }
}

/// One item of a chunk: its pair, and whether it is a hole (not computed) or a
/// non-canonical cell (computed, not counted in `host_pairs`).
#[derive(Clone, Copy)]
struct Item {
    i: usize,
    j: usize,
    hole: bool,
    canonical: bool,
}

impl Session {
    fn p3_mut(&mut self) -> &mut P3 {
        self.p3.as_mut().expect("a primer3 session")
    }

    fn p3_ref(&self) -> &P3 {
        self.p3.as_ref().expect("a primer3 session")
    }

    /// The pair space of the chunk starting at item `pos` (G1's kinds; K2 is
    /// asymmetric, so oriented `within` "both" decodes as BOTH).
    fn p3_space(&mut self, request: &Request<'_>, pos: usize) -> Result<SbSpace, EngineError> {
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

    /// The items `[pos, pos + cap)` of `request`, in item order, into `items` (cleared
    /// first).
    fn p3_items(
        request: &Request<'_>,
        pos: usize,
        cap: usize,
        items: &mut Vec<Item>,
    ) -> Result<(), EngineError> {
        items.clear();
        try_reserve_exact(items, cap, "duplex chunk items")?;
        let plain = |(i, j): (usize, usize)| Item {
            i,
            j,
            hole: false,
            canonical: true,
        };
        match request {
            Request::Range { work, .. } => items.extend(work.range(pos, pos + cap).map(plain)),
            Request::Pairs(pairs) => items.extend(pairs[pos..pos + cap].iter().copied().map(plain)),
            Request::Rect {
                r0,
                c0,
                width,
                class,
                ..
            } => {
                for cell in pos..pos + cap {
                    let (i, j) = (*r0 + cell / *width, *c0 + cell % *width);
                    let verdict = class.map_or(Some(true), |class| class(i, j));
                    items.push(Item {
                        i,
                        j,
                        hole: verdict.is_none(),
                        canonical: verdict.unwrap_or(false),
                    });
                }
            }
        }
        Ok(())
    }

    /// Stage the K2 chunk starting at item `pos`: the key of every item in the
    /// key buffer and, with `fields`, `found` and the three float64 fields
    /// (one pass computes everything, §3). The chunk closes at `pairs_per_chunk`
    /// items or the depth budget and holds at least one item.
    pub(crate) fn p3_stage(
        &mut self,
        host: Host<'_, i64>,
        request: &Request<'_>,
        pos: usize,
        fields: bool,
    ) -> Result<Chunk, EngineError> {
        let settings = self.settings;
        let end = pos + (request.end() - pos).min(settings.pairs_per_chunk as usize);
        let budget = p3_chunk_budget(&settings, self.device.0.resident_threads);
        let swap = self.p3_ref().swap;
        let mut buckets: [P3Bucket; 4] = Default::default();
        let mut routed = Vec::new();
        let mut spent = 0u64;
        let mut n = 0usize;
        // The items are built in batches of at most P3_ITEM_BATCH as the scan reaches
        // them. The chunk closes at the depth budget (device items) or at
        // P3_ROUTED_PER_CHUNK routed items, whichever comes first, so the host
        // state of a chunk (the routed list, its memos, results and scatter
        // buffers) stays bounded below pairs_per_chunk (§11). The chunk
        // boundaries are those of one scan over [pos, end); they never change a
        // result.
        let mut items = Vec::new();
        let mut at = pos;
        'scan: while at < end {
            let take = (end - at).min(P3_ITEM_BATCH);
            Self::p3_items(request, at, take, &mut items)?;
            for item in &items {
                if !item.hole {
                    let (la, lb) = (self.a.len(item.i), self.table_b().len(item.j));
                    let (l1, l2) = if swap { (lb, la) } else { (la, lb) };
                    let plan = p3_plan_k2(&settings, l1, l2);
                    // Amortized fallible growth before each push: a refused host
                    // reservation raises DeviceError instead of aborting (§9 step 6).
                    if plan.routed {
                        if n > 0 && routed.len() >= P3_ROUTED_PER_CHUNK {
                            break 'scan;
                        }
                        try_reserve(&mut routed, 1, "routed pairs")?;
                        routed.push((n, *item));
                    } else {
                        let cost = plan.depth.saturating_mul(plan.g);
                        if n > 0 && spent.saturating_add(cost) > budget {
                            break 'scan;
                        }
                        spent = spent.saturating_add(cost);
                        let bucket = &mut buckets[plan.variant()];
                        try_reserve(&mut bucket.items, 1, "duplex chunk items")?;
                        bucket.push(n, &plan, p3_slot_k2(l1, l2));
                    }
                }
                n += 1;
            }
            at += take;
        }
        drop(items);
        let mut space = self.p3_space(request, pos)?;
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
        let device = self.device.clone();
        let alloc = |error| failure_ordinal(&device, "allocation", error);
        let mut p3 = self.p3.take().expect("a primer3 session");
        let result = (|| {
            let mut params = p3.pack.params;
            params.key = ptr(p3.key.ensure(&stream, n).map_err(alloc)?, &stream);
            // A hole keeps -1 and is never read; every other item is written by
            // its group or the scatter.
            self.fill(params.key, u32::MAX, 2 * n, true)?;
            if fields {
                params.found = ptr(p3.found.ensure(&stream, n).map_err(alloc)?, &stream);
                let [dh, ds, tm] = &mut p3.floats;
                params.dh = ptr(dh.ensure(&stream, n).map_err(alloc)?, &stream);
                params.ds = ptr(ds.ensure(&stream, n).map_err(alloc)?, &stream);
                params.tm = ptr(tm.ensure(&stream, n).map_err(alloc)?, &stream);
                // Testing builds with a poison byte: refill the field buffers every
                // chunk, so an item no kernel writes shows the poison (§11).
                #[cfg(feature = "testing")]
                if let Some(byte) = crate::launch::poison() {
                    self.fill(params.found, u32::from(byte), n * 4, false)?;
                    for address in [params.dh, params.ds, params.tm] {
                        self.fill(address, u32::from(byte), n * 8, false)?;
                    }
                    crate::launch::count_poisoned(28 * n as u64);
                }
            }
            let args = P3Args::Dimer(self.tables(), space);
            launch_buckets(
                (&stream, &device),
                &settings,
                &buckets,
                &params,
                &args,
                &mut p3.buffers,
            )?;
            // Host workers compute the routed pairs while the device runs.
            let (positions, memos) =
                self.p3_routed(&mut p3, host, request, pos, &routed, fields)?;
            if let Request::Range { .. } = request {
                p3.watermark = p3.watermark.max(pos + n);
            }
            if !positions.is_empty() {
                self.p3_scatter(&mut p3, &positions, &memos, &params, fields)?;
            }
            Ok(())
        })();
        self.p3 = Some(p3);
        result.map(|()| Chunk { n, space })
    }

    /// The routed pairs' host results: cached ones from an earlier sweep, the others
    /// computed on host workers with the CPU kernel (no cutoff; with `fields`, every
    /// field). Distinct work items count into `host_pairs` with the levenshtein
    /// stage's rules (§6.6), cache hits included (the watermark counts each once).
    fn p3_routed(
        &self,
        p3: &mut P3,
        host: Host<'_, i64>,
        request: &Request<'_>,
        pos: usize,
        routed: &[(usize, Item)],
        fields: bool,
    ) -> Result<(Vec<u64>, Vec<Memo>), EngineError> {
        let mut counted = 0usize;
        for &(t, item) in routed {
            counted += match request {
                Request::Range { .. } => usize::from(pos + t >= p3.watermark),
                Request::Pairs(_) => usize::from(self.count_pairs),
                Request::Rect { .. } => usize::from(item.canonical),
            };
        }
        p3.host_pairs += counted;
        if routed.is_empty() {
            return Ok((Vec::new(), Vec::new()));
        }
        // Results an earlier sweep cached are reused; only the misses run on host
        // workers. A score-only memo does not serve a detail pass.
        let mut memos: Vec<Option<Memo>> = Vec::new();
        try_reserve_exact(&mut memos, routed.len(), "routed pairs")?;
        let mut misses: Vec<usize> = Vec::new();
        try_reserve_exact(&mut misses, routed.len(), "routed pairs")?;
        for (k, &(_, item)) in routed.iter().enumerate() {
            let memo = p3
                .cache
                .as_ref()
                .and_then(|cache| cache.get(&(item.i, item.j)).copied())
                .filter(|memo| memo.fields.is_some() || !fields);
            if memo.is_none() {
                misses.push(k);
            }
            memos.push(memo);
        }
        if !misses.is_empty() {
            self.p3_compute(p3, host, &misses, routed, &mut memos, fields)?;
            if let (Request::Range { .. }, Some(cache)) = (request, p3.cache.as_mut()) {
                for &k in &misses {
                    let item = routed[k].1;
                    let Some(memo) = memos[k] else { continue };
                    let key = (item.i, item.j);
                    if !cache.contains_key(&key)
                        && (cache.len() >= CACHE_ITEMS || cache.try_reserve(1).is_err())
                    {
                        // Full, or the host refused room: stop growing (later sweeps
                        // recompute the same values).
                        break;
                    }
                    cache.insert(key, memo);
                }
            }
        }
        let mut results = Vec::new();
        try_reserve_exact(&mut results, routed.len(), "routed pairs")?;
        for memo in memos {
            results.push(
                memo.ok_or_else(|| EngineError::Device("a routed pair has no host result".into()))?,
            );
        }
        let mut positions = Vec::new();
        try_reserve_exact(&mut positions, routed.len(), "routed pairs")?;
        positions.extend(routed.iter().map(|&(t, _)| t as u64));
        Ok((positions, results))
    }

    /// Run the routed items `misses` (indices into `routed`) on host workers with
    /// the CPU kernel (no cutoff; with `fields`, every field) into `memos`.
    fn p3_compute(
        &self,
        p3: &P3,
        host: Host<'_, i64>,
        misses: &[usize],
        routed: &[(usize, Item)],
        memos: &mut [Option<Memo>],
        fields: bool,
    ) -> Result<(), EngineError> {
        let kernel = host.kernel;
        let (a, b) = (host.a, host.b);
        let field_index = p3.fields;
        let request_fields = if fields {
            field_index
                .iter()
                .fold(DetailRequest::NONE, |request, &field| request.with(field))
        } else {
            DetailRequest::NONE
        };
        let cost = kernel.pair_cost(a, b, fields);
        let limit = self.workers.count(misses.len(), cost);
        let parts = sched::map_ranges(misses.len(), limit, &|start, end| {
            let mut keys = Vec::new();
            try_reserve_exact(&mut keys, end - start, "routed pairs")?;
            let mut rows = DetailColumns::new(kernel.detail_fields(), request_fields);
            let mut ticker =
                seqblaze_core::cancel::Ticker::new(seqblaze_core::cancel::Site::Routed);
            for &k in &misses[start..end] {
                let item = routed[k].1;
                let key = if fields {
                    kernel.evaluate_detailed(a.get(item.i), b.get(item.j), None, None, &mut rows)?
                } else {
                    kernel.evaluate(a.get(item.i), b.get(item.j), None)?
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
            let mut columns: [Option<Vec<u64>>; 4] = [None, None, None, None];
            for column in rows.into_columns() {
                let Some(c) = field_index.iter().position(|&field| field == column.field) else {
                    continue;
                };
                columns[c] = match column.data {
                    DetailData::Int32(values) => Some(try_bits(&values, |v| v as u32 as u64)?),
                    DetailData::Float64(values) => Some(try_bits(&values, f64::to_bits)?),
                    _ => None,
                };
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
                    let bits = |c: usize| columns[c].as_ref().map_or(0, |v| v[row]);
                    [bits(0), bits(1), bits(2), bits(3)]
                });
                memos[k] = Some(Memo { key, fields });
            }
        }
        if next.next().is_some() {
            return Err(EngineError::Device(
                "routed pair results do not match the routed list".into(),
            ));
        }
        Ok(())
    }

    /// Write the routed results into the chunk's buffers (stream-ordered after
    /// the group launches, before any sink kernel).
    fn p3_scatter(
        &self,
        p3: &mut P3,
        positions: &[u64],
        memos: &[Memo],
        params: &SbP3,
        fields: bool,
    ) -> Result<(), EngineError> {
        let stream = &self.stream;
        let mut keys: Vec<i64> = Vec::new();
        try_reserve_exact(&mut keys, memos.len(), "routed results")?;
        keys.extend(memos.iter().map(|memo| memo.key));
        let mut bits: Vec<u64> = Vec::new();
        if fields {
            try_reserve_exact(&mut bits, 4 * memos.len(), "routed results")?;
            bits.extend(memos.iter().flat_map(|memo| memo.fields.unwrap_or([0; 4])));
        }
        let uploaded = Session::upload(stream, &mut p3.scatter_pos, positions).and_then(|p| {
            let d = Session::upload(stream, &mut p3.scatter_d, &keys)?;
            let f = if fields {
                Session::upload(stream, &mut p3.scatter_f, &bits)?
            } else {
                0
            };
            Ok((p, d, f))
        });
        let (pos, d, f) = uploaded.map_err(|error| self.fail("uploading routed results", error))?;
        let count = positions.len() as u64;
        let config = self.grid(count);
        let mut launch = stream.launch_builder(&self.device.0.kernels.p3_scatter);
        launch
            .arg(&pos)
            .arg(&d)
            .arg(&f)
            .arg(&count)
            .arg(&params.key)
            .arg(&params.found)
            .arg(&params.dh)
            .arg(&params.ds)
            .arg(&params.tm);
        // SAFETY: matches sb_p3_scatter; `pos` and `d` hold `count` entries, `f`
        // four per entry (or null with null field outputs), and the outputs hold
        // the chunk's n items.
        unsafe { launch.launch(config) }
            .map_err(|error| self.fail("launching sb_p3_scatter", error))?;
        Ok(())
    }

    fn p3_key_ptr(&self) -> u64 {
        ptr(self.p3_ref().key.get(), &self.stream)
    }

    /// Copy the staged chunk's keys to `host_keys[0..n]`.
    fn p3_fetch(&mut self, n: usize) -> Result<(), EngineError> {
        let mut host = std::mem::take(&mut self.p3_mut().host_keys);
        host.clear();
        let result = try_reserve_exact(&mut host, n, "duplex chunk keys").and_then(|()| {
            host.resize(n, 0);
            self.download(self.p3_ref().key.get(), n, &mut host)
                .and_then(|()| self.sync())
        });
        self.p3_mut().host_keys = host;
        result
    }

    /// Append the staged chunk's keys to `out`.
    fn p3_fetch_into(&self, n: usize, out: &mut Vec<i64>) -> Result<(), EngineError> {
        let base = out.len();
        try_reserve(out, n, "duplex output column")?;
        out.resize(base + n, 0);
        self.download(self.p3_ref().key.get(), n, &mut out[base..])
            .and_then(|()| self.sync())
    }

    /// Copy the staged chunk's four fields into `fields[..][pos..pos + n]`.
    fn p3_download_fields(
        &self,
        fields: &mut Fields,
        pos: usize,
        n: usize,
    ) -> Result<(), EngineError> {
        let p3 = self.p3_ref();
        self.download(p3.found.get(), n, &mut fields.found[pos..pos + n])?;
        for (k, values) in fields.floats.iter_mut().enumerate() {
            self.download(p3.floats[k].get(), n, &mut values[pos..pos + n])?;
        }
        Ok(())
    }

    /// The staged chunk's fields as detail rows (after a sync).
    fn p3_fetch_details(
        &self,
        host: Host<'_, i64>,
        n: usize,
    ) -> Result<DetailColumns, EngineError> {
        let mut fields = Fields::zeroed(n)?;
        self.p3_download_fields(&mut fields, 0, n)?;
        self.sync()?;
        Ok(self.p3_columns(host, fields))
    }

    /// Detail columns of the request from whole field vectors.
    fn p3_columns(&self, host: Host<'_, i64>, fields: Fields) -> DetailColumns {
        let request = host.kernel.detail_request();
        let index = self.p3_ref().fields;
        let mut columns = DetailColumns::new(host.kernel.detail_fields(), request);
        if request.contains(index[0]) {
            columns.set_column(index[0], DetailData::Int32(fields.found));
        }
        for (k, values) in fields.floats.into_iter().enumerate() {
            if request.contains(index[k + 1]) {
                let values = values.into_iter().map(f64::from_bits).collect();
                columns.set_column(index[k + 1], DetailData::Float64(values));
            }
        }
        columns
    }

    /// Each (§8): the key of every item; with a cutoff `sb_each64` (a failing key
    /// becomes i64::MAX) and the host masks the failing rows' fields.
    fn p3_each(
        &mut self,
        host: Host<'_, i64>,
        work: &WorkSource,
        cutoff: Option<i64>,
    ) -> Result<Columns<i64>, EngineError> {
        let n_pairs = work.n_pairs;
        let request = host.kernel.detail_request();
        let with_fields = !request.is_empty();
        let mut dist = try_filled(n_pairs, 0i64, "duplex output column")?;
        let mut passed = match cutoff {
            Some(_) => Some(try_filled(n_pairs, 0u8, "duplex output column")?),
            None => None,
        };
        let mut fields = if with_fields {
            Some(Fields::zeroed(n_pairs)?)
        } else {
            None
        };
        let mut passed_dev = Scratch::<u8>::new();
        let (has_cutoff, limit) = (i32::from(cutoff.is_some()), cutoff.unwrap_or(0));
        let request_range = Request::Range { work, end: n_pairs };
        let mut pos = 0;
        while pos < n_pairs {
            let chunk = self.p3_stage(host, &request_range, pos, with_fields)?;
            let n = chunk.n;
            if passed.is_some() {
                let passed_ptr = ptr(
                    passed_dev
                        .ensure(&self.stream, n)
                        .map_err(|error| self.fail("allocation", error))?,
                    &self.stream,
                );
                let key_ptr = self.p3_key_ptr();
                let count = n as u64;
                let config = self.grid(count);
                let mut launch = self.stream.launch_builder(&self.device.0.kernels.each64);
                launch
                    .arg(&key_ptr)
                    .arg(&passed_ptr)
                    .arg(&count)
                    .arg(&has_cutoff)
                    .arg(&limit);
                // SAFETY: matches sb_each64; both buffers hold at least n elements.
                unsafe { launch.launch(config) }
                    .map_err(|error| self.fail("launching sb_each64", error))?;
            }
            self.download(self.p3_ref().key.get(), n, &mut dist[pos..pos + n])?;
            if let Some(passed) = passed.as_mut() {
                self.download(passed_dev.get(), n, &mut passed[pos..pos + n])?;
            }
            if let Some(fields) = fields.as_mut() {
                self.p3_download_fields(fields, pos, n)?;
            }
            pos += n;
        }
        self.sync()?;
        if let (Some(fields), Some(passed)) = (fields.as_mut(), passed.as_deref()) {
            fields.mask(passed);
        }
        let details = fields.map(|fields| self.p3_columns(host, fields));
        Ok(Columns {
            dist,
            passed,
            details,
            ..Columns::default()
        })
    }

    /// Hits (§8): per-block counts, a host exclusive scan, then ordered writes on
    /// the keys; `index` takes the CPU's sort and dedup. Fields come from a
    /// device detail pass over the final hit list, with the key check (QPC-12).
    fn p3_hits(
        &mut self,
        host: Host<'_, i64>,
        work: &WorkSource,
        cutoff: Option<i64>,
    ) -> Result<Columns<i64>, EngineError> {
        let request = host.kernel.detail_request();
        let (mut ii, mut jj, mut dd) = (Vec::<i64>::new(), Vec::<i64>::new(), Vec::<i64>::new());
        let mut counts_dev = Scratch::<u64>::new();
        let mut offsets_dev = Scratch::<u64>::new();
        let mut out_i = Scratch::<i64>::new();
        let mut out_j = Scratch::<i64>::new();
        let mut out_d = Scratch::<i64>::new();
        let (has_cutoff, limit) = (i32::from(cutoff.is_some()), cutoff.unwrap_or(0));
        let request_range = Request::Range {
            work,
            end: work.n_pairs,
        };
        let mut pos = 0;
        while pos < work.n_pairs {
            let chunk = self.p3_stage(host, &request_range, pos, false)?;
            let (n, count) = (chunk.n, chunk.n as u64);
            let (blocks, config) = self.tiles(n);
            let key_ptr = self.p3_key_ptr();
            let counts_ptr = ptr(
                counts_dev
                    .ensure(&self.stream, blocks)
                    .map_err(|error| self.fail("allocation", error))?,
                &self.stream,
            );
            let mut launch = self
                .stream
                .launch_builder(&self.device.0.kernels.hits_count64);
            launch
                .arg(&key_ptr)
                .arg(&count)
                .arg(&has_cutoff)
                .arg(&limit)
                .arg(&counts_ptr);
            // SAFETY: matches sb_hits_count64; one count slot per block.
            unsafe { launch.launch(config) }
                .map_err(|error| self.fail("launching sb_hits_count64", error))?;
            let mut block_counts = try_filled(blocks, 0u64, "hit counts")?;
            self.download(counts_dev.get(), blocks, &mut block_counts)?;
            self.sync()?;
            let mut offsets = Vec::new();
            try_reserve_exact(&mut offsets, blocks, "hit counts")?;
            let mut total = 0u64;
            for value in block_counts {
                offsets.push(total);
                total += value;
            }
            if total > 0 {
                let hits = total as usize;
                let stream = &self.stream;
                let alloc = |error| failure_ordinal(&self.device, "allocation", error);
                let offsets_ptr =
                    Session::upload(stream, &mut offsets_dev, &offsets).map_err(alloc)?;
                let i_ptr = ptr(out_i.ensure(stream, hits).map_err(alloc)?, stream);
                let j_ptr = ptr(out_j.ensure(stream, hits).map_err(alloc)?, stream);
                let d_ptr = ptr(out_d.ensure(stream, hits).map_err(alloc)?, stream);
                let space = chunk.space;
                let mut launch = self
                    .stream
                    .launch_builder(&self.device.0.kernels.hits_write64);
                launch
                    .arg(&key_ptr)
                    .arg(&count)
                    .arg(&has_cutoff)
                    .arg(&limit)
                    .arg(&offsets_ptr)
                    .arg(&space)
                    .arg(&i_ptr)
                    .arg(&j_ptr)
                    .arg(&d_ptr);
                // SAFETY: matches sb_hits_write64; outputs hold `total` entries.
                unsafe { launch.launch(config) }
                    .map_err(|error| self.fail("launching sb_hits_write64", error))?;
                let base = dd.len();
                for column in [&mut ii, &mut jj, &mut dd] {
                    try_reserve(column, hits, "duplex hit list")?;
                }
                ii.resize(base + hits, 0);
                jj.resize(base + hits, 0);
                dd.resize(base + hits, 0);
                self.download(out_i.get(), hits, &mut ii[base..])?;
                self.download(out_j.get(), hits, &mut jj[base..])?;
                self.download(out_d.get(), hits, &mut dd[base..])?;
            }
            pos += n;
        }
        self.sync()?;
        let mut columns = if matches!(work.kind, PairKind::Index { .. }) {
            let mut hits: Vec<(usize, usize, i64)> = Vec::new();
            try_reserve_exact(&mut hits, dd.len(), "duplex hit list")?;
            hits.extend(
                ii.into_iter()
                    .zip(jj)
                    .zip(dd)
                    .map(|((i, j), d)| (i as usize, j as usize, d)),
            );
            Columns::from_hits(hits)?
        } else {
            Columns {
                dist: dd,
                i: Some(ii),
                j: Some(jj),
                ..Columns::default()
            }
        };
        if !request.is_empty() {
            columns.details = Some(self.p3_hit_details(host, &columns)?);
        }
        Ok(columns)
    }

    /// A device detail pass over the final hit list in its order; each detail
    /// key must equal the hit's selection key (QPC-12), else nothing is published.
    fn p3_hit_details(
        &mut self,
        host: Host<'_, i64>,
        columns: &Columns<i64>,
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
            let chunk = self.p3_stage(host, &request, pos, true)?;
            keys.clear();
            self.p3_fetch_into(chunk.n, &mut keys)?;
            for (t, &key) in keys.iter().enumerate() {
                if key != columns.dist[pos + t] {
                    let (i, j) = pairs[pos + t];
                    return Err(EngineError::Device(format!(
                        "duplex compat='primer3' Hits: the detail pass gives key {key} for pair \
                         ({i}, {j}) but the selection pass gave {}; no result was published; use \
                         device='cpu'",
                        columns.dist[pos + t]
                    )));
                }
            }
            let mut chunk_rows = self.p3_fetch_details(host, chunk.n)?;
            rows.append(&mut chunk_rows)?;
            pos += chunk.n;
        }
        Ok(rows)
    }

    /// RowStats (§7): sweep 1 takes the 64-bit atomic minimum and counts, sweep 2
    /// (keys recomputed) counts ties at the final best and takes the smallest partner.
    fn p3_rowstats(
        &mut self,
        host: Host<'_, i64>,
        work: &WorkSource,
        cutoff: Option<i64>,
    ) -> Result<RowStatsColumns<i64>, EngineError> {
        let rows = work.shape.0;
        let alloc = |error| failure_ordinal(&self.device, "allocation", error);
        // SAFETY (all four): initialised by the upload and fills below before any
        // kernel reads them.
        let mut best = unsafe { dev_alloc::<i64>(&self.stream, rows.max(1)) }.map_err(alloc)?;
        let count = unsafe { dev_alloc::<u64>(&self.stream, rows.max(1)) }.map_err(alloc)?;
        let nbest = unsafe { dev_alloc::<u64>(&self.stream, rows.max(1)) }.map_err(alloc)?;
        let arg = unsafe { dev_alloc::<u64>(&self.stream, rows.max(1)) }.map_err(alloc)?;
        if rows > 0 {
            let init = try_filled(rows, i64::MAX, "row minima")?;
            crate::session::htod(&self.stream, &init[..], &mut best.slice_mut(0..rows))
                .map_err(|error| self.fail("uploading row minima", error))?;
        }
        let first = match work.first_occurrences()? {
            Some(mask) => {
                let mut host_mask: Vec<u8> = Vec::new();
                try_reserve_exact(&mut host_mask, mask.len(), "row mask")?;
                host_mask.extend(mask.into_iter().map(u8::from));
                let mut scratch = Scratch::<u8>::new();
                Session::upload(&self.stream, &mut scratch, &host_mask).map_err(alloc)?;
                Some(scratch)
            }
            None => None,
        };
        let stream = &self.stream;
        let table = SbRows64 {
            mirror: u64::from(work.mirrors_within()),
            first: first.as_ref().map_or(0, |mask| ptr(mask.get(), stream)),
            best: ptr(&best, stream),
            count: ptr(&count, stream),
            nbest: ptr(&nbest, stream),
            arg: ptr(&arg, stream),
        };
        self.fill(table.count, 0, rows * 8, false)?;
        self.fill(table.nbest, 0, rows * 8, false)?;
        self.fill(table.arg, 0xFF, rows * 8, false)?;
        let (has_cutoff, limit) = (i32::from(cutoff.is_some()), cutoff.unwrap_or(0));
        let request = Request::Range {
            work,
            end: work.n_pairs,
        };
        for sweep in [0, 1] {
            let mut pos = 0;
            while pos < work.n_pairs {
                let chunk = self.p3_stage(host, &request, pos, false)?;
                let key_ptr = self.p3_key_ptr();
                let config = self.grid(chunk.n as u64);
                let (kernel, name) = if sweep == 0 {
                    (&self.device.0.kernels.rowstats_min64, "sb_rowstats_min64")
                } else {
                    (&self.device.0.kernels.rowstats_arg64, "sb_rowstats_arg64")
                };
                let mut launch = self.stream.launch_builder(kernel);
                launch
                    .arg(&key_ptr)
                    .arg(&chunk.space)
                    .arg(&has_cutoff)
                    .arg(&limit)
                    .arg(&table);
                // SAFETY: matches the int64 rowstats kernels; row arrays hold `rows` entries.
                unsafe { launch.launch(config) }
                    .map_err(|error| self.fail(&format!("launching {name}"), error))?;
                pos += chunk.n;
            }
        }
        let mut host_best = try_filled(rows, 0i64, "row statistics")?;
        let mut host_count = try_filled(rows, 0u64, "row statistics")?;
        let mut host_nbest = try_filled(rows, 0u64, "row statistics")?;
        let mut host_arg = try_filled(rows, 0u64, "row statistics")?;
        self.download(&best, rows, &mut host_best)?;
        self.download(&count, rows, &mut host_count)?;
        self.download(&nbest, rows, &mut host_nbest)?;
        self.download(&arg, rows, &mut host_arg)?;
        self.sync()?;
        let mut states = Vec::new();
        try_reserve_exact(&mut states, rows, "row statistics")?;
        for row in 0..rows {
            states.push(if host_count[row] == 0 {
                RowState::default()
            } else {
                let partner = i64::try_from(host_arg[row]).map_err(|_| {
                    EngineError::Device(format!("device rowstats argbest is invalid in row {row}"))
                })?;
                RowState::from_summary(host_best[row], partner, host_nbest[row], host_count[row])?
            });
        }
        RowStatsColumns::from_states(states, cutoff.is_some())
    }
}

/// #[repr(C)] mirror of `SbRows` in kernels/hamming.cuh (best as int64 here).
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct SbRows64 {
    mirror: u64,
    first: u64,
    best: u64,
    count: u64,
    nbest: u64,
    arg: u64,
}

// SAFETY: plain #[repr(C)] struct of u64 fields in the CUDA struct's order.
unsafe impl DeviceRepr for SbRows64 {}

/// Host copies of K2's four device fields: `found` and three float64 bit patterns.
struct Fields {
    found: Vec<i32>,
    floats: [Vec<u64>; 3],
}

impl Fields {
    fn zeroed(rows: usize) -> Result<Self, EngineError> {
        let column = || try_filled(rows, 0u64, "duplex field column");
        Ok(Self {
            found: try_filled(rows, 0, "duplex field column")?,
            floats: [column()?, column()?, column()?],
        })
    }

    /// A failing row holds the CPU's masks: INT32_MAX in `found`, the canonical
    /// NaN in every float64 field.
    fn mask(&mut self, passed: &[u8]) {
        for (row, &ok) in passed.iter().enumerate() {
            if ok == 0 {
                self.found[row] = i32::MAX;
                for values in &mut self.floats {
                    values[row] = CANONICAL_NAN.to_bits();
                }
            }
        }
    }
}

/// A session bound to one K2 int64-key call: the batch-distance port (TopK, Matrix, and
/// Tiles through `reg.rs`'s shared `TileBatch64`).
pub(crate) struct Bound64<'s> {
    pub(crate) session: &'s mut Session,
    pub(crate) host: Host<'s, i64>,
}

impl BatchDistance<i64> for Bound64<'_> {
    fn delivers_details(&self) -> bool {
        !self.host.kernel.detail_request().is_empty()
    }

    fn range(
        &mut self,
        work: &WorkSource,
        start: usize,
        end: usize,
        details: bool,
        visit: &mut DistanceVisit<'_, i64>,
    ) -> Result<(), EngineError> {
        let request = Request::Range { work, end };
        let details = details && self.delivers_details();
        let mut pos = start;
        while pos < end {
            let chunk = self.session.p3_stage(self.host, &request, pos, details)?;
            self.session.p3_fetch(chunk.n)?;
            let rows = if details {
                Some(self.session.p3_fetch_details(self.host, chunk.n)?)
            } else {
                None
            };
            let keys = std::mem::take(&mut self.session.p3_mut().host_keys);
            let result = visit(pos, &keys[..chunk.n], rows.as_ref());
            self.session.p3_mut().host_keys = keys;
            result?;
            pos += chunk.n;
        }
        Ok(())
    }

    fn pairs(
        &mut self,
        pairs: &[(usize, usize)],
        out: &mut Vec<i64>,
        mut details: Option<&mut DetailColumns>,
    ) -> Result<(), EngineError> {
        let request = Request::Pairs(pairs);
        let fields = details.is_some() && self.delivers_details();
        try_reserve(out, pairs.len(), "duplex output column")?;
        let mut pos = 0;
        while pos < pairs.len() {
            let chunk = self.session.p3_stage(self.host, &request, pos, fields)?;
            self.session.p3_fetch_into(chunk.n, out)?;
            if let Some(rows) = details.as_deref_mut() {
                let mut chunk_rows = self.chunk_rows(fields, chunk.n)?;
                rows.append(&mut chunk_rows)?;
            }
            pos += chunk.n;
        }
        Ok(())
    }

    fn rect(
        &mut self,
        rows: (usize, usize),
        cols: (usize, usize),
        class: &CellClass<'_>,
        out: &mut Vec<i64>,
        mut details: Option<&mut DetailColumns>,
    ) -> Result<(), EngineError> {
        let width = cols.1 - cols.0;
        let cells = (rows.1 - rows.0) * width;
        let request = Request::Rect {
            r0: rows.0,
            c0: cols.0,
            width,
            end: cells,
            class: Some(class),
        };
        let fields = details.is_some() && self.delivers_details();
        try_reserve(out, cells, "duplex output column")?;
        let mut pos = 0;
        while pos < cells {
            let chunk = self.session.p3_stage(self.host, &request, pos, fields)?;
            self.session.p3_fetch_into(chunk.n, out)?;
            if let Some(target) = details.as_deref_mut() {
                let mut chunk_rows = self.chunk_rows(fields, chunk.n)?;
                target.append(&mut chunk_rows)?;
            }
            pos += chunk.n;
        }
        Ok(())
    }
}

impl Bound64<'_> {
    fn chunk_rows(&self, fields: bool, n: usize) -> Result<DetailColumns, EngineError> {
        if !fields {
            return Err(EngineError::Parameter(
                "this device call cannot deliver the requested detail fields".into(),
            ));
        }
        self.session.p3_fetch_details(self.host, n)
    }
}

/// K2 with the primer3 preset for every non-Tiles sink (`crate::run_i64`, which it
/// shares with `tm` on Pairs, `reg::run`).
pub(crate) fn run(
    device: &Device,
    kernel: &dyn PairDistanceKernel<i64>,
    call: DeviceCall,
    work: &WorkSource,
    (a, b): (&SeqTable, &SeqTable),
    options: &ExecOptions<i64>,
    validation: ValidationReport,
) -> Result<DeviceRun<i64>, EngineError> {
    guard(|| {
        if !matches!(call, DeviceCall::DuplexPrimer3(_)) {
            return Err(EngineError::Device(
                "the int64-key device path serves duplex compat='primer3' only; use device='cpu'"
                    .into(),
            ));
        }
        let mut session = Session::new(device, call, (a, b), options.workers(), false)?;
        let host = Host { kernel, a, b };
        let output = match options.sink {
            Sink::TopK => {
                let mut port = Bound64 {
                    session: &mut session,
                    host,
                };
                topk::run_batch(kernel, (a, b), &mut port, work, options, validation)?
            }
            Sink::Matrix => {
                let mut port = Bound64 {
                    session: &mut session,
                    host,
                };
                matrix::run_batch(kernel, (a, b), &mut port, work, options, validation)?
            }
            sink => {
                let columns = match sink {
                    Sink::Each => session.p3_each(host, work, options.cutoff)?,
                    Sink::Hits => session.p3_hits(host, work, options.cutoff)?,
                    _ => Columns {
                        rowstats: Some(session.p3_rowstats(host, work, options.cutoff)?),
                        ..Columns::default()
                    },
                };
                RunOutput {
                    columns,
                    shape: work.shape,
                    counts: Counts {
                        certified: 0,
                        pruned: 0,
                        full: work.n_pairs,
                    },
                    threads: options.threads,
                    n_illegal_seqs: validation.n_illegal_seqs,
                    validation_scans: validation.scans,
                    topk_replay: None,
                }
            }
        };
        Ok(DeviceRun {
            output,
            host_pairs: session.host_pairs(),
        })
    })
}
