//! Chunk planning and tier launch geometry of the levenshtein distance stage
//! (docs/wiki/gpu/levenshtein-cuda.md §4 "Chunks", §5.2-§5.5, §8). Host
//! integer arithmetic only, compiled in every build so that its properties
//! are unit-tested without a GPU.
//!
//! Grids: a tier kernel grid-strides over all n items of the chunk and skips
//! the pairs it does not own. Its grid has one unit (thread or warp) per item
//! it may own, but at least the units the device keeps resident (so the skips
//! run in parallel) and never more than n: a kernel that owns few of the
//! chunk's items gets a grid of one resident wave (levenshtein-cuda.md §5.4).
//!
//! Per-unit rule (§8, watchdog): a unit whose grid has fewer units than items
//! runs several pairs one after another in one launch. Each kernel's chunk
//! depth D bounds the serial depth of every device pair it owns in the chunk,
//! and a unit may take P = max(1, ⌊pair_steps / D⌋) items. A chunk therefore
//! holds at most U·P items for every tier launch it needs, U the launch's hard
//! unit limit (`MAX_GRID`, and the Peq slot budget of a raw table a), and
//! every launch has at least ⌈n / P⌉ units. No unit runs more than
//! `pair_steps` serial word steps per launch, the bound §5.5 gives one pair.
use seqblaze_core::{
    detail::{try_reserve, try_reserve_exact},
    kernel::CellClass,
    validate::EngineError,
    work::{Orientation, PairKind, WorkSource},
};

use crate::launch::Settings;

/// Grid-stride kernels never launch more blocks than this.
pub(crate) const MAX_GRID: u64 = 1 << 20;
/// Peq slots of a raw table a stay within this many bytes per launch (§5.2).
pub(crate) const SLOT_BYTES: u64 = 256 << 20;
/// Words per lane of the warp-tier kernels, in `Kernels::lev_warp` order.
pub(crate) const WARP_Q: [u64; 4] = [1, 2, 4, 8];
/// Tier kernels: the thread tier (0), then the warp tiers in `WARP_Q` order.
const KERNELS: usize = 1 + WARP_Q.len();
/// Largest `thread_words` and ⌈W/32⌉ (launch.rs bounds the settings).
const THREAD_W: usize = 4;
const WARP_C: usize = 8;

/// Items of one distance request.
pub(crate) enum Request<'r> {
    /// Work IDs `[pos, end)` of a work source.
    Range { work: &'r WorkSource, end: usize },
    /// Explicit pairs `pairs[pos..]`.
    Pairs(&'r [(usize, usize)]),
    /// Cells `[pos, end)` of a rectangle, row-major. `class` tells which cells
    /// the caller keeps (levenshtein: routed hole cells are skipped).
    Rect {
        r0: usize,
        c0: usize,
        width: usize,
        end: usize,
        class: Option<&'r CellClass<'r>>,
    },
}

impl Request<'_> {
    pub(crate) fn end(&self) -> usize {
        match self {
            Self::Range { end, .. } | Self::Rect { end, .. } => *end,
            Self::Pairs(pairs) => pairs.len(),
        }
    }
}

/// Items of a `tm`-on-Pairs chunk whose fields come back to the host as that chunk's
/// detail rows (Hits' detail pass and the batch-distance port: TopK, Matrix, Tiles).
/// Such a chunk also closes at this many items, so the host field copies, the detail
/// rows and the core's per-chunk cell list stay O(`REG_ROW_ITEMS`) beside the output
/// instead of O(`pairs_per_chunk`) (tm-pairs-cuda.md §14, host memory of Matrix).
/// Each downloads into whole-call columns and keeps the plain rule.
pub(crate) const REG_ROW_ITEMS: usize = 1 << 16;

/// Pairs of the `tm`-on-Pairs chunk that starts at a request position with `left`
/// items left (tm-pairs-cuda.md §4, QTP8; hamming's rule): at most `pairs_per_chunk`
/// pairs, closed once the compared bytes (the sum of L over its pairs) would pass
/// `bytes_per_launch`, and at least one pair. With `rows` (a stage whose fields return
/// as the chunk's detail rows) it also holds at most `REG_ROW_ITEMS` items. `uniform`
/// is the one length of an implicit cross or within shape (O(1)); otherwise `len(k)`
/// is the length of the chunk's k-th pair (O(chunk)). Planning never changes a result.
pub(crate) fn reg_chunk(
    left: usize,
    settings: &Settings,
    rows: bool,
    uniform: Option<u64>,
    len: impl Fn(usize) -> u64,
) -> usize {
    let limit = if rows { REG_ROW_ITEMS } else { usize::MAX };
    let cap = left
        .min(settings.pairs_per_chunk.max(1) as usize)
        .min(limit)
        .max(1);
    match uniform {
        Some(length) => {
            let by_bytes = settings
                .bytes_per_launch
                .checked_div(length)
                .map_or(cap, |pairs| pairs.max(1) as usize);
            cap.min(by_bytes)
        }
        None => {
            let (mut n, mut bytes) = (0usize, 0u64);
            while n < cap {
                let length = len(n);
                if n > 0 && bytes.saturating_add(length) > settings.bytes_per_launch {
                    break;
                }
                bytes = bytes.saturating_add(length);
                n += 1;
            }
            n
        }
    }
}

/// Tier of a pair; the host twin of `sb_lev_tier` in kernels/levenshtein.cuh.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Tier {
    Thread,
    Warp,
    Host,
}

/// The tier of a pair with |a| = m and |b| = n (§5.3-§5.5): with W = ⌈m/64⌉,
/// the host when W > `device_words` or the serial depth n·W (thread tier) or
/// (n + 31)·⌈W/32⌉ (warp tier) exceeds `pair_steps`.
pub(crate) fn tier(settings: &Settings, m: u64, n: u64) -> Tier {
    let w = m.div_ceil(64);
    if w > settings.device_words {
        Tier::Host
    } else if settings.thread_words > 0 && w <= settings.thread_words {
        if n.saturating_mul(w) > settings.pair_steps {
            Tier::Host
        } else {
            Tier::Thread
        }
    } else if n.saturating_add(31).saturating_mul(w.div_ceil(32)) > settings.pair_steps {
        Tier::Host
    } else {
        Tier::Warp
    }
}

/// The tier kernel (0: thread, 1 + k: warp `WARP_Q[k]`) that owns a device
/// pair whose a row has W words: warp Q serves ⌈W/32⌉ in (Q/2, Q].
fn kernel_of(settings: &Settings, w: u64) -> usize {
    if settings.thread_words > 0 && w <= settings.thread_words {
        0
    } else {
        1 + w.div_ceil(32).max(1).next_power_of_two().trailing_zeros() as usize
    }
}

/// (kernel, serial depth) of a pair, or None when it is routed to the host.
fn device_depth(settings: &Settings, m: u64, n: u64) -> Option<(usize, u64)> {
    let w = m.div_ceil(64);
    match tier(settings, m, n) {
        Tier::Host => None,
        Tier::Thread => Some((0, n * w)),
        Tier::Warp => Some((kernel_of(settings, w), (n + 31) * w.div_ceil(32))),
    }
}

/// Per-call planning data over table b, built once in O(rows of b).
pub(crate) struct Profile {
    /// `cost_b[j]` = Σ min(|b_j'|, pair_steps) over rows j' < j of table b.
    cost_b: Vec<u64>,
    /// Largest |b| over table b.
    pub max_n: u64,
    /// Largest serial depth of a device pair against table b: `thread[W]` for
    /// a thread-tier row with W words, `warp[c]` for a warp-tier row with
    /// ⌈W/32⌉ = c (0 when every pair of such a row is routed).
    thread: [u64; THREAD_W + 1],
    warp: [u64; WARP_C + 1],
}

impl Profile {
    /// From table b's int64 offsets; the prefix is reserved fallibly.
    pub(crate) fn new(b_off: &[i64], settings: &Settings) -> Result<Self, EngineError> {
        let steps = settings.pair_steps;
        let mut cost_b = Vec::new();
        try_reserve_exact(&mut cost_b, b_off.len(), "levenshtein chunk plan")?;
        cost_b.push(0u64);
        let (mut total, mut max_n) = (0u64, 0u64);
        let mut thread = [0u64; THREAD_W + 1];
        let mut warp = [0u64; WARP_C + 1];
        for pair in b_off.windows(2) {
            let n = (pair[1] - pair[0]) as u64;
            max_n = max_n.max(n);
            total += n.min(steps);
            cost_b.push(total);
            for (w, deepest) in thread.iter_mut().enumerate().skip(1) {
                let depth = n.saturating_mul(w as u64);
                if depth <= steps {
                    *deepest = (*deepest).max(depth);
                }
            }
            for (c, deepest) in warp.iter_mut().enumerate().skip(1) {
                let depth = n.saturating_add(31).saturating_mul(c as u64);
                if depth <= steps {
                    *deepest = (*deepest).max(depth);
                }
            }
        }
        Ok(Self {
            cost_b,
            max_n,
            thread,
            warp,
        })
    }

    /// (kernel, depth bound) of the device pairs of a row with W words
    /// against any row of table b; None when they are all routed.
    fn row_depth(&self, settings: &Settings, w: u64) -> Option<(usize, u64)> {
        if w > settings.device_words {
            return None;
        }
        let kernel = kernel_of(settings, w);
        let depth = if kernel == 0 {
            self.thread[w.min(THREAD_W as u64) as usize]
        } else {
            self.warp[w.div_ceil(32).min(WARP_C as u64) as usize]
        };
        Some((kernel, depth))
    }
}

/// Host lengths of one call: offsets of tables a and b and the profile of b.
pub(crate) struct Lengths<'l> {
    pub a_off: &'l [i64],
    pub b_off: &'l [i64],
    pub profile: &'l Profile,
}

impl Lengths<'_> {
    fn m(&self, i: usize) -> u64 {
        (self.a_off[i + 1] - self.a_off[i]) as u64
    }
    fn n(&self, j: usize) -> u64 {
        (self.b_off[j + 1] - self.b_off[j]) as u64
    }
}

/// Launch geometry of one call.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Geometry {
    pub threads_per_block: u64,
    /// Raw table a: sigma (Peq in slots); None for the four inline codes.
    pub raw_sigma: Option<u64>,
    /// Threads the device keeps resident at once (SMs × threads per SM).
    pub resident_threads: u64,
    /// Budget of all Peq slots of one launch (`SLOT_BYTES`).
    pub slot_bytes: u64,
}

/// One tier kernel launch of a chunk.
#[derive(Clone, Copy, Debug)]
pub(crate) struct TierLaunch {
    /// 0: thread tier; 1 + k: warp tier `WARP_Q[k]`.
    pub kernel: usize,
    /// Units (threads or warps) per block.
    pub per_block: u64,
    /// Grid limit: `MAX_GRID` and, for a raw table a, the slot budget.
    hard_blocks: u64,
    /// Units the device keeps resident at once.
    resident: u64,
    /// Words of one unit's Peq slot: sigma·top (raw table a), else 0.
    pub slot_words: u64,
}

impl TierLaunch {
    /// Pairs one unit may run in a launch over `plan` (the per-unit rule).
    fn per_unit(&self, settings: &Settings, plan: &Plan) -> u64 {
        let depth = plan.depth[self.kernel].min(settings.pair_steps).max(1);
        (settings.pair_steps / depth).max(1)
    }

    /// Largest chunk this launch can serve under the per-unit rule.
    fn item_limit(&self, settings: &Settings, plan: &Plan) -> u64 {
        (self.hard_blocks * self.per_block).saturating_mul(self.per_unit(settings, plan))
    }

    /// Blocks for `plan`'s chunk; the kernel grid-strides over its items. One
    /// unit per item the kernel may own, at least one resident wave and the
    /// units the per-unit rule needs, at most one unit per item, never above
    /// the hard cap (`plan_chunk` keeps the chunk within `item_limit`).
    pub(crate) fn blocks(&self, settings: &Settings, plan: &Plan) -> u64 {
        let n = plan.n as u64;
        let floor = n.div_ceil(self.per_unit(settings, plan));
        plan.owned[self.kernel]
            .max(self.resident)
            .min(n)
            .max(floor)
            .div_ceil(self.per_block)
            .clamp(1, self.hard_blocks)
    }
}

/// The tier kernels a chunk whose rows have W in [`w_min`, `w_max`] needs: a
/// kernel is launched only when that range meets its own (§5.4).
pub(crate) fn tier_launches(
    settings: &Settings,
    geometry: &Geometry,
    w_min: u64,
    w_max: u64,
) -> Vec<TierLaunch> {
    let threads = geometry.threads_per_block;
    let slot = |top: u64| {
        geometry
            .raw_sigma
            .map_or(0, |sigma| sigma.max(1) * top.max(1))
    };
    // All slots of a grid within slot_bytes (at least one block); a slot
    // holds the kernel's largest W, `top`.
    let hard = |slot_words: u64, per_block: u64| {
        (geometry.slot_bytes / 8)
            .checked_div(slot_words)
            .map_or(MAX_GRID, |slots| (slots / per_block).clamp(1, MAX_GRID))
    };
    let mut launches = Vec::new();
    let thread_top = settings.thread_words.min(settings.device_words);
    if settings.thread_words > 0 && w_min <= thread_top {
        let slot_words = slot(thread_top);
        let hard_blocks = hard(slot_words, threads);
        launches.push(TierLaunch {
            kernel: 0,
            per_block: threads,
            hard_blocks,
            resident: geometry.resident_threads,
            slot_words,
        });
    }
    let warp_low = if settings.thread_words > 0 {
        settings.thread_words + 1
    } else {
        0
    };
    let per_block = (threads / 32).max(1);
    for (index, &q) in WARP_Q.iter().enumerate() {
        let low = if q == 1 { 0 } else { 16 * q + 1 }.max(warp_low).max(w_min);
        let high = (32 * q).min(settings.device_words).min(w_max);
        if low <= high {
            let top = (32 * q).min(settings.device_words);
            let slot_words = slot(top);
            let hard_blocks = hard(slot_words, per_block);
            launches.push(TierLaunch {
                kernel: 1 + index,
                per_block,
                hard_blocks,
                resident: geometry.resident_threads / 32,
                slot_words,
            });
        }
    }
    launches
}

/// A planned chunk: `n` items from the request position, the W range of its
/// rows, and per tier kernel a bound on the serial depth of its device pairs
/// and on how many of the items it owns.
pub(crate) struct Plan {
    budget: u64,
    cap: usize,
    spent: u64,
    pub n: usize,
    pub w_min: u64,
    pub w_max: u64,
    depth: [u64; KERNELS],
    owned: [u64; KERNELS],
}

impl Plan {
    fn new(budget: u64, cap: usize) -> Self {
        Self {
            budget,
            cap,
            spent: 0,
            n: 0,
            w_min: u64::MAX,
            w_max: 0,
            depth: [0; KERNELS],
            owned: [0; KERNELS],
        }
    }

    /// `items` items with W = `w` whose device pairs (if any) belong to the
    /// given kernel with at most the given depth.
    fn note(&mut self, w: u64, device: Option<(usize, u64)>, items: usize) {
        self.w_min = self.w_min.min(w);
        self.w_max = self.w_max.max(w);
        if let Some((kernel, depth)) = device {
            self.depth[kernel] = self.depth[kernel].max(depth);
            self.owned[kernel] += items as u64;
        }
    }

    /// One item (per-item loop); false when the chunk is closed.
    fn item(&mut self, w: u64, device: Option<(usize, u64)>, cost: u64) -> bool {
        if self.n >= self.cap || (self.n > 0 && self.spent.saturating_add(cost) > self.budget) {
            return false;
        }
        self.spent = self.spent.saturating_add(cost);
        self.n += 1;
        self.note(w, device, 1);
        true
    }

    /// A run of `len` consecutive items of one row with W = `w`; `cost(k)` is
    /// the (monotone) cost of its first k items. Binary search, so a row
    /// costs O(log len). True when the whole run was taken.
    fn run(
        &mut self,
        w: u64,
        device: Option<(usize, u64)>,
        len: usize,
        cost: impl Fn(usize) -> u64,
    ) -> bool {
        let room = len.min(self.cap - self.n);
        let left = self.budget.saturating_sub(self.spent);
        let (mut lo, mut hi) = (0usize, room);
        while lo < hi {
            let mid = lo + (hi - lo).div_ceil(2);
            if cost(mid) <= left {
                lo = mid;
            } else {
                hi = mid - 1;
            }
        }
        if lo == 0 && self.n == 0 && room > 0 {
            lo = 1;
        }
        if lo > 0 {
            self.spent = self.spent.saturating_add(cost(lo));
            self.n += lo;
            self.note(w, device, lo);
        }
        lo == len
    }
}

/// Plan the chunk of `request` starting at item `pos` (§4 "Chunks"). It
/// closes at `pairs_per_chunk` items or `steps_per_launch` word steps (a pair
/// costs W·min(n, pair_steps)), holds at least one item, and keeps the
/// per-unit rule of the module docs. Implicit shapes use the prefix over table
/// b, one binary search per row run, and the profile's per-W depth bounds;
/// zip, index and explicit pairs use a per-item loop with exact depths.
pub(crate) fn plan_chunk(
    lengths: &Lengths<'_>,
    settings: &Settings,
    geometry: &Geometry,
    request: &Request<'_>,
    pos: usize,
) -> Plan {
    let cap = (request.end() - pos).min(settings.pairs_per_chunk as usize);
    let plan = plan_capped(lengths, settings, request, pos, cap);
    let limit = item_limit(settings, geometry, &plan);
    if plan.n as u64 <= limit {
        return plan;
    }
    // A prefix of the first plan needs a subset of its launches, and its
    // depth bounds are no larger, so each of their limits is at least `limit`.
    let plan = plan_capped(lengths, settings, request, pos, limit as usize);
    debug_assert!(plan.n as u64 <= item_limit(settings, geometry, &plan));
    plan
}

fn item_limit(settings: &Settings, geometry: &Geometry, plan: &Plan) -> u64 {
    tier_launches(settings, geometry, plan.w_min, plan.w_max)
        .iter()
        .map(|launch| launch.item_limit(settings, plan))
        .min()
        .unwrap_or(u64::MAX)
}

fn plan_capped(
    lengths: &Lengths<'_>,
    settings: &Settings,
    request: &Request<'_>,
    pos: usize,
    cap: usize,
) -> Plan {
    let mut plan = Plan::new(settings.steps_per_launch, cap);
    let profile = lengths.profile;
    let cost_b = &profile.cost_b;
    let pair = |plan: &mut Plan, i: usize, j: usize| {
        let (m, n) = (lengths.m(i), lengths.n(j));
        let w = m.div_ceil(64);
        let cost = w.saturating_mul(n.min(settings.pair_steps));
        plan.item(w, device_depth(settings, m, n), cost)
    };
    let row = |i: usize| {
        let w = lengths.m(i).div_ceil(64);
        (w, profile.row_depth(settings, w))
    };
    let rect = |plan: &mut Plan, r0: usize, c0: usize, width: usize| {
        while plan.n < plan.cap {
            let q = pos + plan.n;
            let (i, c) = (r0 + q / width, q % width);
            let (w, device) = row(i);
            let base = cost_b[c0 + c];
            if !plan.run(w, device, width - c, |k| {
                w.saturating_mul(cost_b[c0 + c + k] - base)
            }) {
                break;
            }
        }
    };
    match request {
        Request::Pairs(pairs) => {
            for &(i, j) in &pairs[pos..pos + cap] {
                if !pair(&mut plan, i, j) {
                    break;
                }
            }
        }
        Request::Rect { r0, c0, width, .. } => rect(&mut plan, *r0, *c0, *width),
        Request::Range { work, .. } => match &work.kind {
            PairKind::Zip => {
                for k in pos..pos + cap {
                    if !pair(&mut plan, k, k) {
                        break;
                    }
                }
            }
            PairKind::Index { i, j } => {
                for k in pos..pos + cap {
                    if !pair(&mut plan, i.get(k), j.get(k)) {
                        break;
                    }
                }
            }
            PairKind::Cross | PairKind::OneToMany => rect(&mut plan, 0, 0, work.shape.1),
            PairKind::Within { .. }
            | PairKind::WithinOriented {
                orientation: Orientation::AsGiven,
                ..
            } => {
                let columns = work.shape.1;
                let segments = work.row_segments(pos, pos + cap).expect("within work");
                for (i, j0, k0) in segments {
                    let len = (columns - j0).min(pos + cap - k0);
                    let (w, device) = row(i);
                    let base = cost_b[j0];
                    if !plan.run(w, device, len, |k| w.saturating_mul(cost_b[j0 + k] - base)) {
                        break;
                    }
                }
            }
            PairKind::WithinOriented {
                self_pairs,
                orientation: Orientation::Both,
            } => {
                // Row i has two contiguous column runs: r < i maps to j = r,
                // r >= i to j = r + 1 when the diagonal is skipped.
                let skip = usize::from(!*self_pairs);
                let width = work.shape.0 - skip;
                while plan.n < plan.cap {
                    let q = pos + plan.n;
                    let (i, r) = (q / width, q % width);
                    let (w, device) = row(i);
                    let cost = |k: usize| {
                        let end = r + k;
                        let sum = if skip == 0 {
                            cost_b[end] - cost_b[r]
                        } else {
                            let first = if r < i {
                                cost_b[end.min(i)] - cost_b[r]
                            } else {
                                0
                            };
                            let start = r.max(i);
                            let second = if end > start {
                                cost_b[end + 1] - cost_b[start + 1]
                            } else {
                                0
                            };
                            first + second
                        };
                        w.saturating_mul(sum)
                    };
                    if !plan.run(w, device, width - r, cost) {
                        break;
                    }
                }
            }
        },
    }
    plan
}

/// End of the unary chunk starting at sequence `k0` of `n` (tm-cuda.md §5.2):
/// the chunk closes at `pairs_per_chunk` sequences or `bytes_per_launch`
/// sequence bytes, whichever comes first, and holds at least one sequence.
/// `off` holds the table's n + 1 int64 offsets (ascending).
pub(crate) fn unary_chunk_end(off: &[i64], k0: usize, settings: &Settings) -> usize {
    let n = off.len() - 1;
    let cap_end = n.min(k0.saturating_add(settings.pairs_per_chunk as usize));
    let limit = i128::from(off[k0]) + i128::from(settings.bytes_per_launch);
    let fits = off[k0 + 1..=cap_end].partition_point(|&end| i128::from(end) <= limit);
    k0 + fits.max(1)
}

// ------------------------------------------------------------------- align
//
// Chunk planning and tier launch geometry of the align distance stage
// (docs/wiki/gpu/align-cuda.md §4 "Chunks", §5.4-§5.6, §8). The tier
// predicate below is the host twin of `sb_aln_tier` in kernels/align.cuh.
//
// Closed-form pairs (native m = 0 or n = 0; water m < 2 or n < 2) run no DP
// and are written by one closed-form kernel per chunk, so no tier kernel owns
// them and they are never routed. A thread-tier candidate (m <= thread_rows)
// too deep for `pair_cells` falls through to the warp tier, then to the host
// (QAG3), so one row of a can have pairs in both tier kernels. The per-unit
// rule is the levenshtein one with depth in cell steps; multi-strip warp
// kernels also respect the strip slot budget.

/// Rows per lane of the compiled align warp variants (§5.5, P).
pub(crate) const ALN_WARP_Q: [u64; 4] = [1, 2, 4, 8];
/// Align tier kernels: thread (0), then warp `ALN_WARP_Q[k]` (1 + k).
pub(crate) const ALN_KERNELS: usize = 1 + ALN_WARP_Q.len();

/// Tier of an align pair (§5.6).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum AlnTier {
    Closed,
    Thread,
    Warp,
    Host,
}

/// Closed form below this length on either side: 1 for native mode (m = 0 or
/// n = 0), 2 for water (m < 2 or n < 2), 0 for needle (none).
pub(crate) fn aln_closed(closed_below: u64, m: u64, n: u64) -> bool {
    m < closed_below || n < closed_below
}

/// Rows per lane q = min(lane_rows, Q(m)), Q(m) the smallest of 1, 2, 4, 8
/// with 32Q >= m (8 when m > 256).
pub(crate) fn aln_q(settings: &Settings, m: u64) -> u64 {
    let big = ALN_WARP_Q
        .iter()
        .copied()
        .find(|&q| 32 * q >= m)
        .unwrap_or(8);
    big.min(settings.lane_rows)
}

/// P·q, P = ⌈m / 32q⌉ strips: the warp depth is P·(n + 31)·q.
pub(crate) fn aln_factor(settings: &Settings, m: u64) -> u64 {
    let q = aln_q(settings, m);
    m.div_ceil(32 * q) * q
}

/// Whether a warp-tier pair needs more than one strip (a slot).
pub(crate) fn aln_multi(settings: &Settings, m: u64) -> bool {
    m > 32 * aln_q(settings, m)
}

/// The §5.6 tier predicate: closed form; else the thread tier when
/// thread_rows > 0, m <= thread_rows and m·n <= pair_cells; else the warp
/// tier when P·(n + 31)·q <= pair_cells; else the host.
pub(crate) fn aln_tier(settings: &Settings, closed_below: u64, m: u64, n: u64) -> AlnTier {
    if aln_closed(closed_below, m, n) {
        AlnTier::Closed
    } else if settings.thread_rows > 0
        && m <= settings.thread_rows
        && m.saturating_mul(n) <= settings.pair_cells
    {
        AlnTier::Thread
    } else if aln_factor(settings, m).saturating_mul(n.saturating_add(31)) <= settings.pair_cells {
        AlnTier::Warp
    } else {
        AlnTier::Host
    }
}

/// The warp kernel (1 + k) that owns warp-tier pairs with q = `ALN_WARP_Q[k]`.
fn aln_warp_kernel(settings: &Settings, m: u64) -> usize {
    1 + aln_q(settings, m).trailing_zeros() as usize
}

/// A device pair: its kernel, serial depth, and n when it needs a slot.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct AlnDevice {
    kernel: usize,
    depth: u64,
    slot_n: Option<u64>,
}

fn aln_device(settings: &Settings, closed_below: u64, m: u64, n: u64) -> Option<AlnDevice> {
    match aln_tier(settings, closed_below, m, n) {
        AlnTier::Closed | AlnTier::Host => None,
        AlnTier::Thread => Some(AlnDevice {
            kernel: 0,
            depth: m * n,
            slot_n: None,
        }),
        AlnTier::Warp => Some(AlnDevice {
            kernel: aln_warp_kernel(settings, m),
            depth: aln_factor(settings, m) * (n + 31),
            slot_n: aln_multi(settings, m).then_some(n),
        }),
    }
}

/// Per-call align planning data over table b, built once in O(n_b log n_b).
pub(crate) struct AlnProfile {
    /// `cost_b[j]` = Σ min(|b_j'|, pair_cells) over rows j' < j of table b.
    cost_b: Vec<u64>,
    /// The distinct lengths of table b, ascending.
    lens: Vec<u64>,
    /// Largest |b| over table b.
    pub max_n: u64,
    closed_below: u64,
}

impl AlnProfile {
    pub(crate) fn new(
        b_off: &[i64],
        settings: &Settings,
        closed_below: u64,
    ) -> Result<Self, EngineError> {
        let cells = settings.pair_cells;
        let mut cost_b = Vec::new();
        try_reserve_exact(&mut cost_b, b_off.len(), "align chunk plan")?;
        let mut lens = Vec::new();
        try_reserve_exact(&mut lens, b_off.len(), "align chunk plan")?;
        cost_b.push(0u64);
        let (mut total, mut max_n) = (0u64, 0u64);
        for pair in b_off.windows(2) {
            let n = (pair[1] - pair[0]) as u64;
            max_n = max_n.max(n);
            total += n.min(cells);
            cost_b.push(total);
            lens.push(n);
        }
        lens.sort_unstable();
        lens.dedup();
        Ok(Self {
            cost_b,
            lens,
            max_n,
            closed_below,
        })
    }

    /// The largest length of table b in [low, high], if any.
    fn largest_within(&self, low: u64, high: u64) -> Option<u64> {
        let end = self.lens.partition_point(|&n| n <= high);
        (end > 0 && self.lens[end - 1] >= low).then(|| self.lens[end - 1])
    }

    /// Device pairs of a row of a with length m against any row of b: per
    /// tier kernel, a bound on their depth and on n when they need a slot.
    /// Upper bounds, so a chunk's per-unit rule and slots stay safe.
    fn row(&self, settings: &Settings, m: u64) -> [Option<AlnDevice>; 2] {
        let low = self.closed_below;
        if m < low {
            return [None, None];
        }
        let cells = settings.pair_cells;
        let thread_ok = settings.thread_rows > 0 && m <= settings.thread_rows;
        // Thread-tier pairs: n <= pair_cells / m.
        let thread_limit = cells.checked_div(m).unwrap_or(u64::MAX);
        let thread = if thread_ok {
            self.largest_within(low, thread_limit).map(|n| AlnDevice {
                kernel: 0,
                depth: m * n,
                slot_n: None,
            })
        } else {
            None
        };
        // Warp-tier pairs: P·(n + 31)·q <= pair_cells, beyond the thread tier.
        let factor = aln_factor(settings, m);
        let warp_limit = match cells.checked_div(factor) {
            None => Some(u64::MAX),
            Some(steps) => steps.checked_sub(31),
        };
        let warp = warp_limit
            .and_then(|limit| self.largest_within(low, limit))
            .filter(|&n| !thread_ok || n > thread_limit)
            .map(|n| AlnDevice {
                kernel: aln_warp_kernel(settings, m),
                depth: factor * (n + 31),
                slot_n: aln_multi(settings, m).then_some(n),
            });
        [thread, warp]
    }
}

/// Host lengths of one align call: offsets of tables a and b and b's profile.
pub(crate) struct AlnLengths<'l> {
    pub a_off: &'l [i64],
    pub b_off: &'l [i64],
    pub profile: &'l AlnProfile,
}

impl AlnLengths<'_> {
    fn m(&self, i: usize) -> u64 {
        (self.a_off[i + 1] - self.a_off[i]) as u64
    }
    fn n(&self, j: usize) -> u64 {
        (self.b_off[j + 1] - self.b_off[j]) as u64
    }
}

/// Launch geometry of one align call.
#[derive(Clone, Copy, Debug)]
pub(crate) struct AlnGeometry {
    /// Threads per block of the thread-tier and warp-tier kernels:
    /// `threads_per_block`, capped by what each kernel can launch with (its
    /// register use), a multiple of 32.
    pub thread_block: u64,
    pub warp_block: u64,
    /// Threads the device keeps resident at once (SMs × threads per SM).
    pub resident_threads: u64,
    /// Budget of all strip slots of one launch (`SLOT_BYTES`).
    pub slot_bytes: u64,
}

/// One align tier kernel launch of a chunk.
#[derive(Clone, Copy, Debug)]
pub(crate) struct AlnLaunch {
    /// 0: thread tier; 1 + k: warp tier `ALN_WARP_Q[k]`.
    pub kernel: usize,
    /// Units (threads or warps) per block.
    pub per_block: u64,
    /// Grid limit: `MAX_GRID` and, for multi-strip pairs, the slot budget.
    hard_blocks: u64,
    /// Units the device keeps resident at once.
    resident: u64,
    /// Columns (n + 1) of one warp's strip slot, two int32 per column; 0 when
    /// no owned pair needs one.
    pub slot_cols: u64,
}

impl AlnLaunch {
    /// Pairs one unit may run in a launch over `plan` (the per-unit rule).
    fn per_unit(&self, settings: &Settings, plan: &AlnPlan) -> u64 {
        let depth = plan.depth[self.kernel].min(settings.pair_cells).max(1);
        (settings.pair_cells / depth).max(1)
    }

    /// Largest chunk this launch can serve under the per-unit rule.
    fn item_limit(&self, settings: &Settings, plan: &AlnPlan) -> u64 {
        (self.hard_blocks * self.per_block).saturating_mul(self.per_unit(settings, plan))
    }

    /// Blocks for `plan`'s chunk, as the levenshtein tiers: one unit per item
    /// the kernel may own, at least one resident wave and the units the
    /// per-unit rule needs, at most one unit per item, never above the cap.
    pub(crate) fn blocks(&self, settings: &Settings, plan: &AlnPlan) -> u64 {
        let n = plan.n as u64;
        let floor = n.div_ceil(self.per_unit(settings, plan));
        plan.owned[self.kernel]
            .max(self.resident)
            .min(n)
            .max(floor)
            .div_ceil(self.per_block)
            .clamp(1, self.hard_blocks)
    }
}

/// The tier kernels that may own a pair of `plan`'s chunk.
pub(crate) fn aln_launches(geometry: &AlnGeometry, plan: &AlnPlan) -> Vec<AlnLaunch> {
    let per_warp_block = (geometry.warp_block / 32).max(1);
    (0..ALN_KERNELS)
        .filter(|&kernel| plan.owned[kernel] > 0)
        .map(|kernel| {
            let per_block = if kernel == 0 {
                geometry.thread_block
            } else {
                per_warp_block
            };
            let slot_cols = plan.slot_n[kernel].map_or(0, |n| n + 1);
            // All slots of a grid within slot_bytes (at least one block).
            let hard_blocks = match slot_cols {
                0 => MAX_GRID,
                cols => (geometry.slot_bytes / (8 * cols) / per_block).clamp(1, MAX_GRID),
            };
            AlnLaunch {
                kernel,
                per_block,
                hard_blocks,
                resident: if kernel == 0 {
                    geometry.resident_threads
                } else {
                    geometry.resident_threads / 32
                },
                slot_cols,
            }
        })
        .collect()
}

/// A planned align chunk: `n` items from the request position, the largest m
/// of its rows, and per tier kernel a bound on the depth of its device pairs,
/// on how many items it may own and on n of its multi-strip pairs.
pub(crate) struct AlnPlan {
    budget: u64,
    cap: usize,
    spent: u64,
    pub n: usize,
    pub m_max: u64,
    depth: [u64; ALN_KERNELS],
    owned: [u64; ALN_KERNELS],
    slot_n: [Option<u64>; ALN_KERNELS],
}

impl AlnPlan {
    fn new(budget: u64, cap: usize) -> Self {
        Self {
            budget,
            cap,
            spent: 0,
            n: 0,
            m_max: 0,
            depth: [0; ALN_KERNELS],
            owned: [0; ALN_KERNELS],
            slot_n: [None; ALN_KERNELS],
        }
    }

    fn note(&mut self, m: u64, devices: &[Option<AlnDevice>], items: usize) {
        self.m_max = self.m_max.max(m);
        for device in devices.iter().flatten() {
            let k = device.kernel;
            self.depth[k] = self.depth[k].max(device.depth);
            self.owned[k] += items as u64;
            if let Some(n) = device.slot_n {
                self.slot_n[k] = Some(self.slot_n[k].map_or(n, |old| old.max(n)));
            }
        }
    }

    /// One item (per-item loop); false when the chunk is closed.
    fn item(&mut self, m: u64, device: Option<AlnDevice>, cost: u64) -> bool {
        if self.n >= self.cap || (self.n > 0 && self.spent.saturating_add(cost) > self.budget) {
            return false;
        }
        self.spent = self.spent.saturating_add(cost);
        self.n += 1;
        self.note(m, &[device], 1);
        true
    }

    /// A run of `len` consecutive items of one row; `cost(k)` is the
    /// (monotone) cost of its first k items. True when the run was taken.
    fn run(
        &mut self,
        m: u64,
        devices: [Option<AlnDevice>; 2],
        len: usize,
        cost: impl Fn(usize) -> u64,
    ) -> bool {
        let room = len.min(self.cap - self.n);
        let left = self.budget.saturating_sub(self.spent);
        let (mut lo, mut hi) = (0usize, room);
        while lo < hi {
            let mid = lo + (hi - lo).div_ceil(2);
            if cost(mid) <= left {
                lo = mid;
            } else {
                hi = mid - 1;
            }
        }
        if lo == 0 && self.n == 0 && room > 0 {
            lo = 1;
        }
        if lo > 0 {
            self.spent = self.spent.saturating_add(cost(lo));
            self.n += lo;
            self.note(m, &devices, lo);
        }
        lo == len
    }
}

/// Plan the align chunk of `request` starting at item `pos` (§4 "Chunks"):
/// it closes at `pairs_per_chunk` items or `cells_per_launch` cells (a pair
/// costs m·min(n, pair_cells)), holds at least one item and keeps the
/// per-unit rule. Implicit shapes use the prefix over table b and one binary
/// search per row run; zip, index and explicit pairs a per-item loop.
pub(crate) fn aln_plan_chunk(
    lengths: &AlnLengths<'_>,
    settings: &Settings,
    geometry: &AlnGeometry,
    request: &Request<'_>,
    pos: usize,
) -> AlnPlan {
    let cap = (request.end() - pos).min(settings.pairs_per_chunk as usize);
    let plan = aln_plan_capped(lengths, settings, request, pos, cap);
    let limit = aln_item_limit(settings, geometry, &plan);
    if plan.n as u64 <= limit {
        return plan;
    }
    // A prefix needs a subset of the launches with no larger depth or slot
    // bounds, so each of their limits is at least `limit`.
    let plan = aln_plan_capped(lengths, settings, request, pos, limit as usize);
    debug_assert!(plan.n as u64 <= aln_item_limit(settings, geometry, &plan));
    plan
}

fn aln_item_limit(settings: &Settings, geometry: &AlnGeometry, plan: &AlnPlan) -> u64 {
    aln_launches(geometry, plan)
        .iter()
        .map(|launch| launch.item_limit(settings, plan))
        .min()
        .unwrap_or(u64::MAX)
}

fn aln_plan_capped(
    lengths: &AlnLengths<'_>,
    settings: &Settings,
    request: &Request<'_>,
    pos: usize,
    cap: usize,
) -> AlnPlan {
    let mut plan = AlnPlan::new(settings.cells_per_launch, cap);
    let profile = lengths.profile;
    let cost_b = &profile.cost_b;
    let closed_below = profile.closed_below;
    let pair = |plan: &mut AlnPlan, i: usize, j: usize| {
        let (m, n) = (lengths.m(i), lengths.n(j));
        let cost = m.saturating_mul(n.min(settings.pair_cells));
        plan.item(m, aln_device(settings, closed_below, m, n), cost)
    };
    let row = |i: usize| {
        let m = lengths.m(i);
        (m, profile.row(settings, m))
    };
    let rect = |plan: &mut AlnPlan, r0: usize, c0: usize, width: usize| {
        while plan.n < plan.cap {
            let q = pos + plan.n;
            let (i, c) = (r0 + q / width, q % width);
            let (m, devices) = row(i);
            let base = cost_b[c0 + c];
            if !plan.run(m, devices, width - c, |k| {
                m.saturating_mul(cost_b[c0 + c + k] - base)
            }) {
                break;
            }
        }
    };
    match request {
        Request::Pairs(pairs) => {
            for &(i, j) in &pairs[pos..pos + cap] {
                if !pair(&mut plan, i, j) {
                    break;
                }
            }
        }
        Request::Rect { r0, c0, width, .. } => rect(&mut plan, *r0, *c0, *width),
        Request::Range { work, .. } => match &work.kind {
            PairKind::Zip => {
                for k in pos..pos + cap {
                    if !pair(&mut plan, k, k) {
                        break;
                    }
                }
            }
            PairKind::Index { i, j } => {
                for k in pos..pos + cap {
                    if !pair(&mut plan, i.get(k), j.get(k)) {
                        break;
                    }
                }
            }
            PairKind::Cross | PairKind::OneToMany => rect(&mut plan, 0, 0, work.shape.1),
            // Align within is symmetric (TRI); an oriented both-ways source
            // (never built for align) takes the per-item loop.
            PairKind::Within { .. } | PairKind::WithinOriented { .. } => {
                let columns = work.shape.1;
                let Some(segments) = work.row_segments(pos, pos + cap) else {
                    for (i, j) in work.range(pos, pos + cap) {
                        if !pair(&mut plan, i, j) {
                            break;
                        }
                    }
                    return plan;
                };
                for (i, j0, k0) in segments {
                    let len = (columns - j0).min(pos + cap - k0);
                    let (m, devices) = row(i);
                    let base = cost_b[j0];
                    if !plan.run(m, devices, len, |k| m.saturating_mul(cost_b[j0 + k] - base)) {
                        break;
                    }
                }
            }
        },
    }
    plan
}

// ------------------------------------------------------------------ duplex
//
// K1 duplex (docs/wiki/gpu/duplex-cuda.md §4 "Chunks", §5.3-§5.6, §8). Here n is
// |a| (the rows of a pair's DP) and m is |b| (its columns); the group size g of
// a pair follows m. A pair is closed form (n < 2 or m < 2: depth 0, never
// routed), routed (depth n·⌈m/g⌉·c(n, m, Λ) above `pair_cells`), or a device
// pair of one variant (g, tier): the shared tier when its band of
// R·m·w bytes (R = min(n, Λ + 2)) is at most `band_bytes` and a whole warp of
// such groups fits the block, else the global tier. The device kernels
// evaluate the same predicate (`sb_dup_tier` in kernels/duplex.cuh) with the
// same inputs. Chunks close at `pairs_per_chunk` items or `cells_per_launch`
// candidate evaluations; the per-unit rule is align's with depth in candidate
// evaluations of one lane.

/// Duplex's cap on `pair_cells` (duplex-cuda.md §8 "Watchdog", QKG6). Measured
/// 2026-10-17 on the RTX 5070 Ti (idle, load 2.6-2.7), one pair per call, whole-call
/// wall time, median of five after a warm-up, at depths above the cap: shared-tier
/// score pass (Λ = 100, 200 x 128, depth 4,120,800) 0.209 s; global-tier score pass
/// (Λ = 30, g = 4, 169 x 200, depth 4,191,200) 0.202 s; the same pair's detail pass
/// 0.549 s, above the 0.4 s rule; its int64 score pass 0.253 s. At depth 2,083,200
/// (84 x 200, g = 4) the detail pass ran 0.262 s and the int64 score pass 0.119 s.
/// Both passes route by the capped value (one predicate, §5.6).
pub(crate) const DUP_PAIR_CELLS_CAP: u64 = 1 << 21;

/// The launch settings duplex plans and predicates with: `pair_cells` capped
/// at [`DUP_PAIR_CELLS_CAP`].
pub(crate) fn dup_settings(settings: &Settings) -> Settings {
    Settings {
        pair_cells: settings.pair_cells.min(DUP_PAIR_CELLS_CAP),
        ..*settings
    }
}

/// The compiled group sizes, in variant order.
pub(crate) const DUP_GROUPS: [u64; 4] = [4, 8, 16, 32];
/// Variants: (group size index) · 2 + (0 shared, 1 global).
pub(crate) const DUP_VARIANTS: usize = 2 * DUP_GROUPS.len();
/// Staged letter codes of a and b per group, bytes (q of §5.4): K3's q, and K1's for a
/// pack without STATES; with STATES K1 also stages the run lengths, q = 800
/// (nndb2024-dna.md §5.4.4). The call's q travels in [`DupLimits::codes`].
pub(crate) const DUP_CODES: u64 = 400;
/// K1's q for a pack with STATES: the codes, then ρ_a and ρ_b (nndb2024-dna.md §5.4.4).
pub(crate) const DUP_CODES_STATES: u64 = 800;

/// The key a pass stores in the band: D in int32 or int64 (the score pass) or
/// the full key of four int64 (the detail pass).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum DupWidth {
    Narrow,
    Wide,
    Detail,
}

impl DupWidth {
    pub(crate) fn bytes(self) -> u64 {
        match self {
            Self::Narrow => 4,
            Self::Wide => 8,
            Self::Detail => 32,
        }
    }
}

/// Tier of a duplex pair in one pass (§5.6, §5.4).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum DupTier {
    Closed,
    Shared,
    Global,
    Host,
}

/// Shared memory of one block: the device's opt-in per-block limit S_max and
/// the block's static shared memory s_block (§5.4), and the call's staged bytes per
/// group q (`codes`: [`DUP_CODES`], or [`DUP_CODES_STATES`] for a K1 pack with STATES;
/// the device twin reads `SbDup::codes`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct DupLimits {
    pub s_max: u64,
    pub s_block: u64,
    pub codes: u64,
}

pub(crate) fn dup_closed(n: u64, m: u64) -> bool {
    n < 2 || m < 2
}

/// g of a pair whose b has length m (§5.3, P, QKG3).
pub(crate) fn dup_group(settings: &Settings, m: u64) -> u64 {
    match settings.group_lanes {
        0 if m <= 16 => 4,
        0 if m <= 32 => 8,
        0 if m <= 64 => 16,
        0 => 32,
        lanes => lanes,
    }
}

/// Σ_{u=0}^{U} (min(Λ − u, M) + 1) in closed form: the first terms are M + 1
/// until Λ − u falls below M, then Λ + 1 − u.
fn dup_terms(u_max: u64, m2: u64, lam: u64) -> u64 {
    let u0 = lam.saturating_sub(m2);
    let flat = (u_max + 1).min(u0);
    let mut total = flat * (m2 + 1);
    if u_max >= u0 {
        let count = u_max - u0 + 1;
        // Σ_{u=u0}^{U} (Λ + 1 − u); (U + u0)·count is even.
        total += count * (lam + 1) - (u_max + u0) * count / 2;
    }
    total
}

/// c(n, m, Λ) of §5.6: the largest candidate count of one cell (0 in closed form).
pub(crate) fn dup_c(n: u64, m: u64, lam: u64) -> u64 {
    if dup_closed(n, m) {
        0
    } else {
        dup_terms(lam.min(n - 2), m - 2, lam)
    }
}

/// c∞(m, Λ) of §4 "Chunks": c for any n (0 for m < 2).
pub(crate) fn dup_c_inf(m: u64, lam: u64) -> u64 {
    if m < 2 {
        0
    } else {
        dup_terms(lam, m - 2, lam)
    }
}

/// Candidate evaluations of one lane for the pair: n·⌈m/g⌉·c(n, m, Λ).
pub(crate) fn dup_depth(settings: &Settings, lam: u64, n: u64, m: u64) -> u64 {
    n * m.div_ceil(dup_group(settings, m)) * dup_c(n, m, lam)
}

/// Whether the pair goes to the host CPU kernel (§5.6 item 3).
pub(crate) fn dup_routed(settings: &Settings, lam: u64, n: u64, m: u64) -> bool {
    !dup_closed(n, m) && dup_depth(settings, lam, n, m) > settings.pair_cells
}

/// Band bytes R·m·w, R = min(n, Λ + 2) (§5.4).
pub(crate) fn dup_band(lam: u64, width: DupWidth, n: u64, m: u64) -> u64 {
    n.min(lam + 2) * m * width.bytes()
}

/// Largest band whose group of size g fits a whole warp of groups into the
/// block: (32/g)·(b + q) + s_block <= S_max.
pub(crate) fn dup_fit(limits: &DupLimits, g: u64) -> Option<u64> {
    (limits.s_max.checked_sub(limits.s_block)? / (32 / g)).checked_sub(limits.codes)
}

/// The tier of a pair in a pass of `width` (§5.6, then §5.4).
pub(crate) fn dup_tier(
    settings: &Settings,
    limits: &DupLimits,
    lam: u64,
    width: DupWidth,
    n: u64,
    m: u64,
) -> DupTier {
    if dup_closed(n, m) {
        return DupTier::Closed;
    }
    if dup_routed(settings, lam, n, m) {
        return DupTier::Host;
    }
    let band = dup_band(lam, width, n, m);
    let fits = dup_fit(limits, dup_group(settings, m)).is_some_and(|most| band <= most);
    if band <= settings.band_bytes && fits {
        DupTier::Shared
    } else {
        DupTier::Global
    }
}

fn dup_variant(g: u64, tier: DupTier) -> usize {
    let index = DUP_GROUPS
        .iter()
        .position(|&size| size == g)
        .expect("a compiled group size");
    2 * index + usize::from(tier == DupTier::Global)
}

/// Per-call duplex planning data over table b, built once in O(rows of b).
pub(crate) struct DupProfile {
    /// `cost_b[j]` = Σ m_j'·c∞(m_j', Λ) over rows j' < j of table b.
    cost_b: Vec<u64>,
    /// Per group size of `DUP_GROUPS`: the distinct m >= 2 of table b whose
    /// pairs take that size, ascending.
    class: [Vec<u64>; 4],
    pub lam: u64,
}

impl DupProfile {
    pub(crate) fn new(b_off: &[i64], settings: &Settings, lam: u64) -> Result<Self, EngineError> {
        let mut cost_b = Vec::new();
        try_reserve_exact(&mut cost_b, b_off.len(), "duplex chunk plan")?;
        cost_b.push(0u64);
        let mut class: [Vec<u64>; 4] = Default::default();
        let mut total = 0u64;
        for pair in b_off.windows(2) {
            let m = (pair[1] - pair[0]) as u64;
            total = total.saturating_add(m.saturating_mul(dup_c_inf(m, lam)));
            cost_b.push(total);
            if m >= 2 {
                let k = DUP_GROUPS
                    .iter()
                    .position(|&g| g == dup_group(settings, m))
                    .expect("a compiled group size");
                // Amortized growth through the §3 helper (F4 round 3).
                try_reserve(&mut class[k], 1, "duplex chunk plan")?;
                class[k].push(m);
            }
        }
        for lengths in &mut class {
            lengths.sort_unstable();
            lengths.dedup();
        }
        Ok(Self { cost_b, class, lam })
    }

    /// Largest m >= 2 of table b (0 when every row is shorter).
    pub(crate) fn max_m(&self) -> u64 {
        self.class
            .iter()
            .filter_map(|lengths| lengths.last().copied())
            .max()
            .unwrap_or(0)
    }
}

/// Host lengths of one duplex call: offsets of tables a and b and b's profile.
pub(crate) struct DupLengths<'l> {
    pub a_off: &'l [i64],
    pub b_off: &'l [i64],
    pub profile: &'l DupProfile,
}

/// Launch geometry of one duplex call.
#[derive(Clone, Copy, Debug)]
pub(crate) struct DupGeometry {
    /// `threads_per_block` (a multiple of 32).
    pub threads_per_block: u64,
    /// Threads the device keeps resident at once (SMs × threads per SM).
    pub resident_threads: u64,
    pub limits: DupLimits,
    /// Budget of all global band slots of one launch (`SLOT_BYTES`).
    pub slot_bytes: u64,
}

/// A planned duplex chunk: `n` items from the request position and, per
/// variant, a bound on the depth of its device pairs, on how many items it may
/// own and on their band bytes.
pub(crate) struct DupPlan {
    budget: u64,
    cap: usize,
    spent: u64,
    pub n: usize,
    pub width: DupWidth,
    depth: [u64; DUP_VARIANTS],
    owned: [u64; DUP_VARIANTS],
    band: [u64; DUP_VARIANTS],
}

impl DupPlan {
    fn new(budget: u64, cap: usize, width: DupWidth) -> Self {
        Self {
            budget,
            cap,
            spent: 0,
            n: 0,
            width,
            depth: [0; DUP_VARIANTS],
            owned: [0; DUP_VARIANTS],
            band: [0; DUP_VARIANTS],
        }
    }

    fn note(&mut self, variant: usize, depth: u64, band: u64, items: usize) {
        self.depth[variant] = self.depth[variant].max(depth);
        self.band[variant] = self.band[variant].max(band);
        self.owned[variant] += items as u64;
    }

    /// The exact variant of one explicit pair.
    fn note_pair(&mut self, settings: &Settings, limits: &DupLimits, lam: u64, n: u64, m: u64) {
        let tier = dup_tier(settings, limits, lam, self.width, n, m);
        if matches!(tier, DupTier::Shared | DupTier::Global) {
            let variant = dup_variant(dup_group(settings, m), tier);
            let band = dup_band(lam, self.width, n, m);
            self.note(variant, dup_depth(settings, lam, n, m), band, 1);
        }
    }

    /// Upper bounds for `items` pairs of one row of length n against table b:
    /// depth and band bytes grow with m within a group size, so the class's
    /// smallest m and its largest m that is not routed bound every device
    /// pair of the class.
    fn note_row(
        &mut self,
        settings: &Settings,
        limits: &DupLimits,
        profile: &DupProfile,
        n: u64,
        items: usize,
    ) {
        let lam = profile.lam;
        if n < 2 {
            return;
        }
        for (k, class) in profile.class.iter().enumerate() {
            let device =
                class.partition_point(|&m| dup_depth(settings, lam, n, m) <= settings.pair_cells);
            if device == 0 {
                continue; // no row, or every pair of this class is routed
            }
            let (g, lo, hi) = (DUP_GROUPS[k], class[0], class[device - 1]);
            let depth = dup_depth(settings, lam, n, hi);
            let (small, large) = (
                dup_band(lam, self.width, n, lo),
                dup_band(lam, self.width, n, hi),
            );
            let fit = dup_fit(limits, g);
            let fits = |band: u64| band <= settings.band_bytes && fit.is_some_and(|f| band <= f);
            if fits(small) {
                self.note(dup_variant(g, DupTier::Shared), depth, large, items);
            }
            if !fits(large) {
                self.note(dup_variant(g, DupTier::Global), depth, large, items);
            }
        }
    }

    /// One item (per-item loop); false when the chunk is closed.
    fn item(&mut self, cost: u64) -> bool {
        if self.n >= self.cap || (self.n > 0 && self.spent.saturating_add(cost) > self.budget) {
            return false;
        }
        self.spent = self.spent.saturating_add(cost);
        self.n += 1;
        true
    }

    /// The longest prefix (at least one item when the chunk is empty) of a
    /// run of `len` items whose (monotone) cost of the first k is `cost(k)`.
    fn take(&mut self, len: usize, cost: impl Fn(usize) -> u64) -> usize {
        let room = len.min(self.cap - self.n);
        let left = self.budget.saturating_sub(self.spent);
        let (mut lo, mut hi) = (0usize, room);
        while lo < hi {
            let mid = lo + (hi - lo).div_ceil(2);
            if cost(mid) <= left {
                lo = mid;
            } else {
                hi = mid - 1;
            }
        }
        if lo == 0 && self.n == 0 && room > 0 {
            lo = 1;
        }
        if lo > 0 {
            self.spent = self.spent.saturating_add(cost(lo));
            self.n += lo;
        }
        lo
    }
}

/// One duplex kernel launch of a chunk: a variant (g, tier) of the pass.
#[derive(Clone, Copy, Debug)]
pub(crate) struct DupLaunch {
    pub variant: usize,
    pub g: u64,
    pub global: bool,
    /// Groups per block (whole warps).
    pub per_block: u64,
    /// Dynamic shared bytes of one group: its codes, and in the shared tier
    /// its band slot (a multiple of the key width).
    pub stride: u64,
    /// Global band slot of one group, bytes (global tier only).
    pub slot: u64,
    /// Grid limit: `MAX_GRID` and, for the global tier, the slot budget.
    hard_blocks: u64,
    /// Groups the device keeps resident at once.
    resident: u64,
}

impl DupLaunch {
    /// Threads per block.
    pub(crate) fn threads(&self) -> u64 {
        self.per_block * self.g
    }

    /// Dynamic shared memory per block.
    pub(crate) fn shared_bytes(&self) -> u64 {
        self.per_block * self.stride
    }

    /// Pairs one group may run in a launch over `plan` (the per-unit rule).
    fn per_unit(&self, settings: &Settings, plan: &DupPlan) -> u64 {
        let depth = plan.depth[self.variant].min(settings.pair_cells).max(1);
        (settings.pair_cells / depth).max(1)
    }

    /// Largest chunk this launch can serve under the per-unit rule.
    fn item_limit(&self, settings: &Settings, plan: &DupPlan) -> u64 {
        (self.hard_blocks * self.per_block).saturating_mul(self.per_unit(settings, plan))
    }

    /// Blocks for `plan`'s chunk: one group per item the variant may own, at
    /// least one resident wave and the groups the per-unit rule needs, at most
    /// one group per item, never above the cap.
    pub(crate) fn blocks(&self, settings: &Settings, plan: &DupPlan) -> u64 {
        let n = plan.n as u64;
        let floor = n.div_ceil(self.per_unit(settings, plan));
        plan.owned[self.variant]
            .max(self.resident)
            .min(n)
            .max(floor)
            .div_ceil(self.per_block)
            .clamp(1, self.hard_blocks)
    }
}

/// The variant launches that may own a pair of `plan`'s chunk.
pub(crate) fn dup_launches(
    settings: &Settings,
    geometry: &DupGeometry,
    plan: &DupPlan,
) -> Vec<DupLaunch> {
    let limits = &geometry.limits;
    let w = plan.width.bytes();
    let usable = limits.s_max.saturating_sub(limits.s_block);
    (0..DUP_VARIANTS)
        .filter(|&variant| plan.owned[variant] > 0)
        .map(|variant| {
            let g = DUP_GROUPS[variant / 2];
            let global = variant % 2 == 1;
            let warp = 32 / g;
            let threads = geometry.threads_per_block.max(32);
            if global {
                // Codes in shared memory, the band in a global slot of 8-byte multiples.
                let slot = plan.band[variant].div_ceil(8).max(1) * 8;
                // Whole warps of groups within the block's threads, its shared
                // memory and (for wide slots, down to one warp) the slot budget.
                let by_shared = (usable / limits.codes / warp).max(1) * warp;
                let by_slots = (geometry.slot_bytes / slot / warp).max(1) * warp;
                let per_block = (threads / g).min(by_shared).min(by_slots);
                DupLaunch {
                    variant,
                    g,
                    global,
                    per_block,
                    stride: limits.codes,
                    slot,
                    hard_blocks: (geometry.slot_bytes / slot / per_block).clamp(1, MAX_GRID),
                    resident: (geometry.resident_threads / g).max(1),
                }
            } else {
                // Every shared-tier item has a band of at most this many bytes, a
                // multiple of the key width.
                let most = plan.band[variant]
                    .min(settings.band_bytes)
                    .min(dup_fit(limits, g).unwrap_or(0));
                let stride = limits.codes + most / w * w;
                let warps = (threads / 32).min(usable / (warp * stride)).max(1);
                DupLaunch {
                    variant,
                    g,
                    global,
                    per_block: warps * warp,
                    stride,
                    slot: 0,
                    hard_blocks: MAX_GRID,
                    resident: (geometry.resident_threads / g).max(1),
                }
            }
        })
        .collect()
}

/// Plan the duplex chunk of `request` starting at item `pos` (§4 "Chunks"):
/// it closes at `pairs_per_chunk` items or `cells_per_launch` candidate
/// evaluations (n·m·c(n, m, Λ) for explicit pairs, n·m·c∞(m, Λ) for implicit
/// shapes, 0 in closed form), holds at least one item and keeps the per-unit
/// rule. Implicit shapes use the prefix over table b and one binary search
/// per row run; zip, index and explicit pairs a per-item loop.
pub(crate) fn dup_plan_chunk(
    lengths: &DupLengths<'_>,
    settings: &Settings,
    geometry: &DupGeometry,
    request: &Request<'_>,
    pos: usize,
    width: DupWidth,
) -> DupPlan {
    let cap = (request.end() - pos).min(settings.pairs_per_chunk as usize);
    let plan = dup_plan_capped(lengths, settings, geometry, request, pos, cap, width);
    let limit = dup_item_limit(settings, geometry, &plan);
    if plan.n as u64 <= limit {
        return plan;
    }
    // A prefix has no larger depth or band bounds, so each of its limits is at
    // least `limit`.
    let plan = dup_plan_capped(
        lengths,
        settings,
        geometry,
        request,
        pos,
        limit as usize,
        width,
    );
    debug_assert!(plan.n as u64 <= dup_item_limit(settings, geometry, &plan));
    plan
}

fn dup_item_limit(settings: &Settings, geometry: &DupGeometry, plan: &DupPlan) -> u64 {
    dup_launches(settings, geometry, plan)
        .iter()
        .map(|launch| launch.item_limit(settings, plan))
        .min()
        .unwrap_or(u64::MAX)
}

fn dup_plan_capped(
    lengths: &DupLengths<'_>,
    settings: &Settings,
    geometry: &DupGeometry,
    request: &Request<'_>,
    pos: usize,
    cap: usize,
    width: DupWidth,
) -> DupPlan {
    let mut plan = DupPlan::new(settings.cells_per_launch, cap, width);
    let profile = lengths.profile;
    let lam = profile.lam;
    let limits = &geometry.limits;
    let walk = Walk {
        a_off: lengths.a_off,
        b_off: lengths.b_off,
        cost_b: &profile.cost_b,
        lam,
    };
    walk.fill(
        &mut plan,
        request,
        pos,
        &|plan, n, m| plan.note_pair(settings, limits, lam, n, m),
        &|plan, n, items| plan.note_row(settings, limits, profile, n, items),
    );
    plan
}

/// The item walk shared by the K1 and K3 chunk planners (duplex-cuda.md §4 "Chunks",
/// duplex-pf-cuda.md §4): the cost of an item is n·m·c(n, m, Λ) for explicit pairs and
/// n·m·c∞(m, Λ) for implicit shapes (0 in closed form); each taken item or row run is
/// noted by the planner's own variant rule.
struct Walk<'w> {
    a_off: &'w [i64],
    b_off: &'w [i64],
    /// `cost_b[j]` = Σ m_j'·c∞(m_j', Λ) over rows j' < j of table b.
    cost_b: &'w [u64],
    lam: u64,
}

type NotePair<'n> = dyn Fn(&mut DupPlan, u64, u64) + 'n;
type NoteRow<'n> = dyn Fn(&mut DupPlan, u64, usize) + 'n;

impl Walk<'_> {
    fn n(&self, i: usize) -> u64 {
        (self.a_off[i + 1] - self.a_off[i]) as u64
    }
    fn m(&self, j: usize) -> u64 {
        (self.b_off[j + 1] - self.b_off[j]) as u64
    }

    fn fill(
        &self,
        plan: &mut DupPlan,
        request: &Request<'_>,
        pos: usize,
        note_pair: &NotePair<'_>,
        note_row: &NoteRow<'_>,
    ) {
        let cap = plan.cap;
        let lam = self.lam;
        let cost_b = self.cost_b;
        let pair = |plan: &mut DupPlan, i: usize, j: usize| {
            let (n, m) = (self.n(i), self.m(j));
            let cost = n.saturating_mul(m).saturating_mul(dup_c(n, m, lam));
            if !plan.item(cost) {
                return false;
            }
            note_pair(plan, n, m);
            true
        };
        // A run of `len` items of row i from column j0; true when all were taken.
        let run = |plan: &mut DupPlan, i: usize, j0: usize, len: usize| {
            let n = self.n(i);
            let base = cost_b[j0];
            let per = if n < 2 { 0 } else { n };
            let taken = plan.take(len, |k| per.saturating_mul(cost_b[j0 + k] - base));
            if taken > 0 {
                note_row(plan, n, taken);
            }
            taken == len
        };
        let rect = |plan: &mut DupPlan, r0: usize, c0: usize, width: usize| {
            while plan.n < plan.cap {
                let q = pos + plan.n;
                let (i, c) = (r0 + q / width, q % width);
                if !run(plan, i, c0 + c, width - c) {
                    break;
                }
            }
        };
        match request {
            Request::Pairs(pairs) => {
                for &(i, j) in &pairs[pos..pos + cap] {
                    if !pair(plan, i, j) {
                        break;
                    }
                }
            }
            Request::Rect { r0, c0, width, .. } => rect(plan, *r0, *c0, *width),
            Request::Range { work, .. } => match &work.kind {
                PairKind::Zip => {
                    for k in pos..pos + cap {
                        if !pair(plan, k, k) {
                            break;
                        }
                    }
                }
                PairKind::Index { i, j } => {
                    for k in pos..pos + cap {
                        if !pair(plan, i.get(k), j.get(k)) {
                            break;
                        }
                    }
                }
                PairKind::Cross | PairKind::OneToMany => rect(plan, 0, 0, work.shape.1),
                // Duplex within is symmetric (TRI); an oriented source (never built
                // for duplex) takes the per-item loop.
                PairKind::Within { .. } | PairKind::WithinOriented { .. } => {
                    let columns = work.shape.1;
                    let Some(segments) = work.row_segments(pos, pos + cap) else {
                        for (i, j) in work.range(pos, pos + cap) {
                            if !pair(plan, i, j) {
                                break;
                            }
                        }
                        return;
                    };
                    for (i, j0, k0) in segments {
                        let len = (columns - j0).min(pos + cap - k0);
                        if !run(plan, i, j0, len) {
                            break;
                        }
                    }
                }
            },
        }
    }
}

// ------------------------------------------------------------------ duplex pf (K3)
//
// K3 native `duplex(model="pf")` (docs/wiki/gpu/duplex-pf-cuda.md §4 "Chunks",
// §6.2-§6.4, §8). n = |a| and m = |b|; the device orients each pair canonically, so
// every per-pair quantity the host plans with is symmetric in n and m: the group size
// g by max(n, m), the depth D_pf = c(n, m, Λ)·max(n·⌈m/g⌉, m·⌈n/g⌉) and the band bound
// b_sym = max(b(n, m), b(m, n)), b(r, c) = (2·min(r, Λ + 2) + 1)·c·16. A pair is closed
// form (n < 2 or m < 2), routed (D_pf above min(`pair_cells`, CAP_pf)), or a device pair
// of one variant (g, tier) by K1's shared-tier predicate with b_sym. The device kernels
// evaluate the same predicate (`sb_pf_tier` in kernels/duplex_pf.cuh). Chunks and the
// per-unit rule are K1's, with D_pf.

/// K3's cap on `pair_cells` (CAP_pf, duplex-pf-cuda.md §6.4 item 3 and §8 "Watchdog").
/// Measured 2026-10-18 on the RTX 5070 Ti (idle GPU, CPU load 4.4-4.5), V1 Λ = 30 at
/// 37 °C and 1 M, A^n against T^m, one pair per call, whole-call wall time, median of
/// five after a warm-up, built with the cap at its ceiling 2^21: (a) 200 x 200, global
/// tier, g = 32, depth 694,400: 0.112 s (161 ns per unit of depth); (b) 128 x 128,
/// global tier, g = 4, depth 2,031,616: 0.251 s (124 ns per unit); (c) 94 x 94, shared
/// tier, g = 32, depth 139,872: 0.033 s; (d) pair (a) on RowStats with `lse_dg_kcal`:
/// 0.112 s. At the slower rate, (a)'s, depth 2^21 predicts 0.34 s, within the 0.4 s rule,
/// so CAP_pf is the ceiling 2^21; pair (b), at 97% of it, confirms 0.251 s.
pub(crate) const PF_CAP: u64 = 1 << 21;

/// The launch settings K3 plans and predicates with: `pair_cells` capped at
/// [`PF_CAP`] (an override above it acts as the cap).
pub(crate) fn pf_settings(settings: &Settings) -> Settings {
    Settings {
        pair_cells: settings.pair_cells.min(PF_CAP),
        ..*settings
    }
}

/// Bytes of one F53 value (M, e) in the band.
pub(crate) const PF_VALUE: u64 = 16;

/// g of a pair (§6.2, QKP5): `group_lanes` if set, else by max(n, m).
pub(crate) fn pf_group(settings: &Settings, n: u64, m: u64) -> u64 {
    match settings.group_lanes {
        0 => match n.max(m) {
            0..=16 => 4,
            17..=32 => 8,
            33..=64 => 16,
            _ => 32,
        },
        lanes => lanes,
    }
}

/// D_pf of §6.4 item 2: candidate evaluations of one lane (0 in closed form).
pub(crate) fn pf_depth(settings: &Settings, lam: u64, n: u64, m: u64) -> u64 {
    if dup_closed(n, m) {
        return 0;
    }
    let g = pf_group(settings, n, m);
    dup_c(n, m, lam).saturating_mul(
        n.saturating_mul(m.div_ceil(g))
            .max(m.saturating_mul(n.div_ceil(g))),
    )
}

/// Whether the pair goes to the host CPU kernel (§6.4 item 3).
pub(crate) fn pf_routed(settings: &Settings, lam: u64, n: u64, m: u64) -> bool {
    !dup_closed(n, m) && pf_depth(settings, lam, n, m) > settings.pair_cells
}

/// Band bytes of one orientation with `rows` rows and `cols` columns (§6.3) for N_C =
/// `nc` loop classes: ((1 + N_C)·R + 1)·cols·16, R = min(rows, Λ + 2) (the ZA ring, the
/// ZO ring of N_C values per cell, E; nndb2024-dna.md §5.4.5).
pub(crate) fn pf_band(lam: u64, nc: u64, rows: u64, cols: u64) -> u64 {
    ((1 + nc) * rows.min(lam + 2) + 1) * cols * PF_VALUE
}

/// b_sym of §6.3.
pub(crate) fn pf_band_sym(lam: u64, nc: u64, n: u64, m: u64) -> u64 {
    pf_band(lam, nc, n, m).max(pf_band(lam, nc, m, n))
}

/// The tier of a pair (§6.4, then §6.3).
pub(crate) fn pf_tier(
    settings: &Settings,
    limits: &DupLimits,
    lam: u64,
    nc: u64,
    n: u64,
    m: u64,
) -> DupTier {
    if dup_closed(n, m) {
        return DupTier::Closed;
    }
    if pf_routed(settings, lam, n, m) {
        return DupTier::Host;
    }
    let band = pf_band_sym(lam, nc, n, m);
    let fits = dup_fit(limits, pf_group(settings, n, m)).is_some_and(|most| band <= most);
    if band <= settings.band_bytes && fits {
        DupTier::Shared
    } else {
        DupTier::Global
    }
}

/// Per-call K3 planning data over table b, built once in O(rows of b · log).
pub(crate) struct PfProfile {
    /// As [`DupProfile`]'s: Σ m_j'·c∞(m_j', Λ) over rows j' < j of table b.
    cost_b: Vec<u64>,
    /// The distinct m >= 2 of table b, ascending.
    lengths: Vec<u64>,
    pub lam: u64,
    /// N_C of the call's pack (the band's ZO values per cell).
    pub nc: u64,
}

impl PfProfile {
    pub(crate) fn new(b_off: &[i64], lam: u64, nc: u64) -> Result<Self, EngineError> {
        let mut cost_b = Vec::new();
        try_reserve_exact(&mut cost_b, b_off.len(), "duplex pf chunk plan")?;
        cost_b.push(0u64);
        // One reservation for every row of b, then plain pushes; the sorted distinct
        // lengths keep only their own room.
        let mut lengths = Vec::new();
        try_reserve_exact(
            &mut lengths,
            b_off.len().saturating_sub(1),
            "duplex pf chunk plan",
        )?;
        let mut total = 0u64;
        for pair in b_off.windows(2) {
            let m = (pair[1] - pair[0]) as u64;
            total = total.saturating_add(m.saturating_mul(dup_c_inf(m, lam)));
            cost_b.push(total);
            if m >= 2 {
                lengths.push(m);
            }
        }
        lengths.sort_unstable();
        lengths.dedup();
        lengths.shrink_to_fit();
        Ok(Self {
            cost_b,
            lengths,
            lam,
            nc,
        })
    }

    /// Largest m >= 2 of table b (0 when every row is shorter).
    pub(crate) fn max_m(&self) -> u64 {
        self.lengths.last().copied().unwrap_or(0)
    }
}

/// Host lengths of one K3 call: offsets of tables a and b and b's profile.
pub(crate) struct PfLengths<'l> {
    pub a_off: &'l [i64],
    pub b_off: &'l [i64],
    pub profile: &'l PfProfile,
}

impl DupPlan {
    /// The exact K3 variant of one explicit pair.
    fn pf_note_pair(
        &mut self,
        settings: &Settings,
        limits: &DupLimits,
        (lam, nc): (u64, u64),
        n: u64,
        m: u64,
    ) {
        let tier = pf_tier(settings, limits, lam, nc, n, m);
        if matches!(tier, DupTier::Shared | DupTier::Global) {
            let variant = dup_variant(pf_group(settings, n, m), tier);
            let band = pf_band_sym(lam, nc, n, m);
            self.note(variant, pf_depth(settings, lam, n, m), band, 1);
        }
    }

    /// Upper bounds for `items` pairs of one row of length n against table b. The
    /// group size of (n, m) is nondecreasing in m, so the lengths taking each g form
    /// one run of b's sorted distinct lengths; within it the depth and b_sym grow with
    /// m, so the run's smallest m and its largest m that is not routed bound every
    /// device pair of the run.
    fn pf_note_row(
        &mut self,
        settings: &Settings,
        limits: &DupLimits,
        profile: &PfProfile,
        n: u64,
        items: usize,
    ) {
        let lam = profile.lam;
        if n < 2 {
            return;
        }
        let lengths = &profile.lengths;
        for &g in &DUP_GROUPS {
            let lo = lengths.partition_point(|&m| pf_group(settings, n, m) < g);
            let hi = lengths.partition_point(|&m| pf_group(settings, n, m) <= g);
            let class = &lengths[lo..hi];
            let device =
                class.partition_point(|&m| pf_depth(settings, lam, n, m) <= settings.pair_cells);
            if device == 0 {
                continue; // no length, or every pair of this run is routed
            }
            let (short, long) = (class[0], class[device - 1]);
            let depth = pf_depth(settings, lam, n, long);
            let nc = profile.nc;
            let (small, large) = (
                pf_band_sym(lam, nc, n, short),
                pf_band_sym(lam, nc, n, long),
            );
            let fit = dup_fit(limits, g);
            let fits = |band: u64| band <= settings.band_bytes && fit.is_some_and(|f| band <= f);
            if fits(small) {
                self.note(dup_variant(g, DupTier::Shared), depth, large, items);
            }
            if !fits(large) {
                self.note(dup_variant(g, DupTier::Global), depth, large, items);
            }
        }
    }
}

/// The K3 variant launches that may own a pair of `plan`'s chunk: K1's geometry with
/// 16-byte band values.
pub(crate) fn pf_launches(
    settings: &Settings,
    geometry: &DupGeometry,
    plan: &DupPlan,
) -> Vec<DupLaunch> {
    let limits = &geometry.limits;
    let usable = limits.s_max.saturating_sub(limits.s_block);
    (0..DUP_VARIANTS)
        .filter(|&variant| plan.owned[variant] > 0)
        .map(|variant| {
            let g = DUP_GROUPS[variant / 2];
            let global = variant % 2 == 1;
            let warp = 32 / g;
            let threads = geometry.threads_per_block.max(32);
            if global {
                // Codes in shared memory, the band in a global slot of 16-byte values.
                let slot = plan.band[variant].div_ceil(PF_VALUE).max(1) * PF_VALUE;
                let by_shared = (usable / limits.codes / warp).max(1) * warp;
                let by_slots = (geometry.slot_bytes / slot / warp).max(1) * warp;
                let per_block = (threads / g).min(by_shared).min(by_slots);
                DupLaunch {
                    variant,
                    g,
                    global,
                    per_block,
                    stride: limits.codes,
                    slot,
                    hard_blocks: (geometry.slot_bytes / slot / per_block).clamp(1, MAX_GRID),
                    resident: (geometry.resident_threads / g).max(1),
                }
            } else {
                let most = plan.band[variant]
                    .min(settings.band_bytes)
                    .min(dup_fit(limits, g).unwrap_or(0));
                let stride = limits.codes + most / PF_VALUE * PF_VALUE;
                let warps = (threads / 32).min(usable / (warp * stride)).max(1);
                DupLaunch {
                    variant,
                    g,
                    global,
                    per_block: warps * warp,
                    stride,
                    slot: 0,
                    hard_blocks: MAX_GRID,
                    resident: (geometry.resident_threads / g).max(1),
                }
            }
        })
        .collect()
}

/// Plan the K3 chunk of `request` starting at item `pos`: K1's chunk rule (§4
/// "Chunks") and per-unit rule (§8) with K3's variants and depth. `settings` are
/// [`pf_settings`].
pub(crate) fn pf_plan_chunk(
    lengths: &PfLengths<'_>,
    settings: &Settings,
    geometry: &DupGeometry,
    request: &Request<'_>,
    pos: usize,
) -> DupPlan {
    let cap = (request.end() - pos).min(settings.pairs_per_chunk as usize);
    let plan = pf_plan_capped(lengths, settings, geometry, request, pos, cap);
    let limit = pf_item_limit(settings, geometry, &plan);
    if plan.n as u64 <= limit {
        return plan;
    }
    // A prefix has no larger depth or band bounds, so each of its limits is at
    // least `limit`.
    let plan = pf_plan_capped(lengths, settings, geometry, request, pos, limit as usize);
    debug_assert!(plan.n as u64 <= pf_item_limit(settings, geometry, &plan));
    plan
}

fn pf_item_limit(settings: &Settings, geometry: &DupGeometry, plan: &DupPlan) -> u64 {
    pf_launches(settings, geometry, plan)
        .iter()
        .map(|launch| launch.item_limit(settings, plan))
        .min()
        .unwrap_or(u64::MAX)
}

fn pf_plan_capped(
    lengths: &PfLengths<'_>,
    settings: &Settings,
    geometry: &DupGeometry,
    request: &Request<'_>,
    pos: usize,
    cap: usize,
) -> DupPlan {
    // The width is K1's; K3's band arithmetic never reads it.
    let mut plan = DupPlan::new(settings.cells_per_launch, cap, DupWidth::Detail);
    let profile = lengths.profile;
    let lam = profile.lam;
    let limits = &geometry.limits;
    let walk = Walk {
        a_off: lengths.a_off,
        b_off: lengths.b_off,
        cost_b: &profile.cost_b,
        lam,
    };
    walk.fill(
        &mut plan,
        request,
        pos,
        &|plan, n, m| plan.pf_note_pair(settings, limits, (lam, profile.nc), n, m),
        &|plan, n, items| plan.pf_note_row(settings, limits, profile, n, items),
    );
    plan
}

// ------------------------------------------------------------------- primer3
//
// Planning of the T2 `hairpin` and K2 `duplex` primer3 device stages
// (docs/wiki/gpu/primer3-cuda.md §6.5-§6.6, §11): the candidate count c, both
// depth bounds, the primer3 cap, the g floor, K2 routing, slot bytes and the
// launch split under the per-unit rule. Every item takes the global tier: one
// slot per in-flight group, all slots of a launch within `SLOT_BYTES` (QPC-3,
// the alternative the page allows; tiers never change results).

/// The primer3 cap on an item's depth (§6.6, QPC-9): provisionally 2^17
/// candidate evaluations of one lane, kept by the §11 measurement (see
/// [`P3_LANE_STEPS`]).
pub(crate) const P3_CAP: u64 = 1 << 17;
/// The compiled group sizes, in variant order.
pub(crate) const P3_GROUPS: [u64; 4] = [4, 8, 16, 32];
/// `max_loop` of both presets (QH2, QD3).
const P3_LOOP: u64 = 30;
/// Bytes of one cell: (ΔH, ΔS) bit patterns.
const P3_CELL: u64 = 16;
/// T2 slot header: 64 bytes of codes and 62 prefix cells (kernels/primer3.cuh).
const P3_T2_HEAD: u64 = 64 + 62 * P3_CELL;
/// Depth units per resident lane a chunk may carry (§11 "Rule": a default chunk
/// stays well below 1 s). A chunk closes when Σ depth·g over its device items
/// exceeds `resident_threads` times this (and at `cells_per_launch`). Basis
/// (2026-10-18, RTX 5070 Ti, idle GPU, load 2.6-3.0, synthetic handle, whole
/// call, median of 5): a default chunk of 507 T2 60-mers took 0.066 s and of
/// 591 K2 60 x 60 pairs 0.196 s; 2^16 T2 60-mers took 8.19 s (about 0.063 s
/// per chunk). Lone device items at depth just under P3_CAP at each g (K2
/// A_n against T_m, every cell admissible: 192 x 9 at g = 4, 90 x 22 at g = 8,
/// 70 x 47 at g = 16, 104 x 46 at g = 32) took 0.23-0.31 s, mismatched K2
/// pairs and T2 stems with a mismatch every fifth base at most 0.20 s and
/// 0.052 s, so P3_CAP stays at 2^17 (§11: lowered only above 0.4 s).
/// Per-item throughput was flat from about 500 items per chunk (T2 kernel:
/// 142 registers, 400-byte stack).
pub(crate) const P3_LANE_STEPS: u64 = 1 << 14;

/// c(n, m) of §6.6: K1's per-cell candidate count at Λ = 30.
pub(crate) fn p3_c(n: u64, m: u64) -> u64 {
    dup_c(n, m, P3_LOOP)
}

/// K2 depth (§6.6): L₁·(⌈L₂/g⌉·(c + 1) + L₂) + min(L₁, L₂)·(c + 2).
pub(crate) fn p3_depth_k2(l1: u64, l2: u64, g: u64) -> u64 {
    let c = p3_c(l1, l2);
    l1 * (l2.div_ceil(g) * (c + 1) + l2) + l1.min(l2) * (c + 2)
}

/// T2 depth (§6.6): L·(⌈max(L − 4, 1)/g⌉·(c + 2) + 4L) + ⌈L/2⌉·2(c + 2) + 6L·⌈L/5⌉.
pub(crate) fn p3_depth_t2(l: u64, g: u64) -> u64 {
    let c = p3_c(l, l);
    let w = l.saturating_sub(4).max(1);
    l * (w.div_ceil(g) * (c + 2) + 4 * l) + l.div_ceil(2) * 2 * (c + 2) + 6 * l * l.div_ceil(5)
}

/// The effective cap min(`pair_cells`, P3_CAP).
pub(crate) fn p3_cap(settings: &Settings) -> u64 {
    settings.pair_cells.min(P3_CAP)
}

/// g of §6.5 before the floor: `group_lanes` if set, else by the width w.
fn p3_base_g(settings: &Settings, w: u64) -> u64 {
    match settings.group_lanes {
        0 if w <= 16 => 4,
        0 if w <= 32 => 8,
        0 if w <= 64 => 16,
        0 => 32,
        lanes => lanes,
    }
}

/// Where one item runs (§6.6): a group of `g` lanes at `depth`, or the host.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct P3Item {
    pub g: u64,
    pub depth: u64,
    pub routed: bool,
}

impl P3Item {
    /// The variant (group size index) of a device item.
    pub(crate) fn variant(&self) -> usize {
        P3_GROUPS
            .iter()
            .position(|&g| g == self.g)
            .expect("a planned g is a compiled group size")
    }
}

/// The g floor: the smallest compiled g ≥ the base g whose depth fits the cap.
fn p3_floor(settings: &Settings, w: u64, depth: impl Fn(u64) -> u64) -> Option<(u64, u64)> {
    let base = p3_base_g(settings, w);
    let cap = p3_cap(settings);
    P3_GROUPS
        .iter()
        .filter(|&&g| g >= base)
        .map(|&g| (g, depth(g)))
        .find(|&(_, d)| d <= cap)
}

/// K2 pair of operand lengths (L₁, L₂) after the `end` mapping: routed iff
/// even g = 32 exceeds the effective cap.
pub(crate) fn p3_plan_k2(settings: &Settings, l1: u64, l2: u64) -> P3Item {
    match p3_floor(settings, l2, |g| p3_depth_k2(l1, l2, g)) {
        Some((g, depth)) => P3Item {
            g,
            depth,
            routed: false,
        },
        None => P3Item {
            g: 32,
            depth: p3_depth_k2(l1, l2, 32),
            routed: true,
        },
    }
}

/// T2 sequence of length L: never routed; g = 32 when nothing fits.
pub(crate) fn p3_plan_t2(settings: &Settings, l: u64) -> P3Item {
    let (g, depth) = p3_floor(settings, l.saturating_sub(4).max(1), |g| p3_depth_t2(l, g))
        .unwrap_or((32, p3_depth_t2(l, 32)));
    P3Item {
        g,
        depth,
        routed: false,
    }
}

/// Slot bytes of a T2 item: codes, prefix, (L + 1)² cells.
pub(crate) fn p3_slot_t2(l: u64) -> u64 {
    P3_T2_HEAD + (l + 1) * (l + 1) * P3_CELL
}

/// Slot bytes of a K2 item: both code strings (16-byte aligned), (L₁ + 1)(L₂ + 1) cells.
pub(crate) fn p3_slot_k2(l1: u64, l2: u64) -> u64 {
    (l1 + l2 + 4).div_ceil(16) * 16 + (l1 + 1) * (l2 + 1) * P3_CELL
}

/// The chunk budget in depth units weighted by g (lane steps).
pub(crate) fn p3_chunk_budget(settings: &Settings, resident_threads: u64) -> u64 {
    settings
        .cells_per_launch
        .min(resident_threads.max(1).saturating_mul(P3_LANE_STEPS))
}

/// The items of one variant in a chunk: chunk positions, the largest slot
/// and the largest depth.
#[derive(Clone, Debug, Default)]
pub(crate) struct P3Bucket {
    pub items: Vec<u32>,
    pub slot: u64,
    pub depth: u64,
}

impl P3Bucket {
    pub(crate) fn push(&mut self, t: usize, item: &P3Item, slot: u64) {
        self.items.push(t as u32);
        self.slot = self.slot.max(slot);
        self.depth = self.depth.max(item.depth);
    }
}

/// One launch of a variant: items `start..start + len` of its bucket, run by
/// `blocks` blocks of `per_block` groups, one slot per group.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct P3Launch {
    pub start: usize,
    pub len: usize,
    pub blocks: u64,
    pub per_block: u64,
    pub slot: u64,
}

/// The launches of one variant's bucket (§6.6 per-unit rule): a group takes at
/// most max(1, ⌊cap/D⌋) items per launch; one group per item where the grid
/// (`MAX_GRID` blocks) and the slot budget allow, all slots within
/// `SLOT_BYTES` (at least one block).
pub(crate) fn p3_launches(
    settings: &Settings,
    count: usize,
    g: u64,
    threads: u64,
    slot: u64,
    depth: u64,
) -> Vec<P3Launch> {
    let slot = slot.div_ceil(16).max(1) * 16;
    let per_block = (threads.max(32) / 32 * 32 / g).max(1);
    let by_slots = (SLOT_BYTES / slot / per_block).clamp(1, MAX_GRID);
    let hard_groups = by_slots * per_block;
    let per_unit = (p3_cap(settings) / depth.max(1)).max(1);
    let mut launches = Vec::new();
    let mut start = 0usize;
    while start < count {
        let take = ((count - start) as u64).min(hard_groups.saturating_mul(per_unit));
        let groups = take.min(hard_groups);
        launches.push(P3Launch {
            start,
            len: take as usize,
            blocks: groups.div_ceil(per_block),
            per_block,
            slot,
        });
        start += take as usize;
    }
    launches
}

#[cfg(test)]
mod tests {
    use super::*;
    use seqblaze_core::work::IndexArray;

    /// tm-pairs-cuda.md §4 (QTP8): a chunk closes at `pairs_per_chunk` pairs or at
    /// `bytes_per_launch` compared bytes, whichever comes first, and holds at least one
    /// pair; an implicit shape plans in O(1) from its one length, empty pairs included.
    #[test]
    fn reg_chunks_close_at_pairs_or_bytes_and_hold_a_pair() {
        let with = |pairs_per_chunk: u64, bytes_per_launch: u64| Settings {
            pairs_per_chunk,
            bytes_per_launch,
            ..settings(4, 256, 1 << 22)
        };
        let lengths = [3u64, 0, 200, 1, 2, 2, 0, 0];
        let len = |k: usize| lengths[k];
        assert_eq!(reg_chunk(8, &with(1 << 22, 1 << 30), false, None, len), 8);
        assert_eq!(reg_chunk(8, &with(7, 1 << 30), false, None, len), 7);
        assert_eq!(reg_chunk(8, &with(1, 1 << 30), false, None, len), 1);
        // 3 + 0 fit in 3 bytes; the 200-nt pair would pass them.
        assert_eq!(reg_chunk(8, &with(1 << 22, 3), false, None, len), 2);
        // A first pair above the byte budget still forms a chunk.
        assert_eq!(reg_chunk(8, &with(1 << 22, 1), false, None, |_| 200), 1);
        assert_eq!(reg_chunk(5, &with(1 << 22, 1), false, None, |_| 0), 5);
        assert_eq!(
            reg_chunk(100, &with(1 << 22, 1 << 30), false, Some(3), len),
            100
        );
        assert_eq!(reg_chunk(100, &with(1 << 22, 10), false, Some(3), len), 3);
        assert_eq!(reg_chunk(100, &with(1 << 22, 1), false, Some(3), len), 1);
        assert_eq!(reg_chunk(100, &with(7, 1 << 30), false, Some(0), len), 7);
        assert_eq!(reg_chunk(1, &with(7, 1), false, Some(200), len), 1);
    }

    /// tm-pairs-cuda.md §14 (host memory of Matrix): a stage whose fields return as the
    /// chunk's detail rows also closes at `REG_ROW_ITEMS` items, on implicit shapes and
    /// on explicit lists alike; a key-only or Each stage keeps the plain rule, a smaller
    /// `pairs_per_chunk` or byte budget still closes it first, and it holds a pair.
    #[test]
    fn reg_row_chunks_also_close_at_the_row_cap() {
        let with = |pairs_per_chunk: u64, bytes_per_launch: u64| Settings {
            pairs_per_chunk,
            bytes_per_launch,
            ..settings(4, 256, 1 << 22)
        };
        let (big, wide) = (1usize << 22, with(1 << 22, 1 << 40));
        let (cap, over, under) = (REG_ROW_ITEMS, REG_ROW_ITEMS + 1, REG_ROW_ITEMS - 1);
        for uniform in [Some(20), None] {
            assert_eq!(reg_chunk(big, &wide, false, uniform, |_| 20), big);
            assert_eq!(reg_chunk(big, &wide, true, uniform, |_| 20), cap);
            assert_eq!(reg_chunk(over, &wide, true, uniform, |_| 20), cap);
            assert_eq!(reg_chunk(under, &wide, true, uniform, |_| 20), under);
            assert_eq!(reg_chunk(big, &with(7, 1 << 40), true, uniform, |_| 20), 7);
            assert_eq!(
                reg_chunk(big, &with(1 << 22, 100), true, uniform, |_| 20),
                5
            );
            assert_eq!(reg_chunk(big, &with(1 << 22, 1), true, uniform, |_| 20), 1);
        }
    }

    fn settings(thread_words: u64, device_words: u64, pair_steps: u64) -> Settings {
        Settings {
            threads_per_block: 32,
            piece_bytes: 4096,
            pairs_per_chunk: 1 << 22,
            bytes_per_launch: 1 << 30,
            thread_words,
            device_words,
            pair_steps,
            steps_per_launch: 1 << 34,
            thread_rows: 32,
            lane_rows: 8,
            pair_cells: 1 << 22,
            cells_per_launch: 1 << 34,
            group_lanes: 0,
            band_bytes: 32_768,
        }
    }

    fn offsets(lengths: &[u64]) -> Vec<i64> {
        let mut out = vec![0i64];
        for &len in lengths {
            out.push(out.last().unwrap() + len as i64);
        }
        out
    }

    struct Rng(u64);
    impl Rng {
        fn next(&mut self, bound: u64) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0 % bound
        }
    }

    /// (row of a, row of b) of item `k`, independently of the planner.
    fn item(request: &Request<'_>, k: usize) -> (usize, usize) {
        match request {
            Request::Range { work, .. } => work.range(k, k + 1).next().unwrap(),
            Request::Pairs(pairs) => pairs[k],
            Request::Rect { r0, c0, width, .. } => (r0 + k / width, c0 + k % width),
        }
    }

    /// Whether a tier kernel computes a pair: the predicates of the device
    /// kernels in kernels/levenshtein.cuh (tier, then the warp's W range).
    fn serves(launch: &TierLaunch, pair_tier: Tier, w: u64) -> bool {
        match launch.kernel {
            0 => pair_tier == Tier::Thread,
            k => {
                let q = WARP_Q[k - 1];
                pair_tier == Tier::Warp && w <= 32 * q && (q == 1 || w > 16 * q)
            }
        }
    }

    /// Serial depth of a device pair (§5.3, §5.4).
    fn depth(settings: &Settings, m: u64, n: u64) -> u64 {
        let w = m.div_ceil(64);
        match tier(settings, m, n) {
            Tier::Thread => n * w,
            Tier::Warp => (n + 31) * w.div_ceil(32),
            Tier::Host => unreachable!("routed pairs have no device depth"),
        }
    }

    /// Plans every chunk of `request` and simulates each tier launch's
    /// grid-stride assignment: chunks partition the items, every device pair
    /// is computed by exactly one launched kernel, and no unit runs more than
    /// `pair_steps` serial word steps in one launch. Returns per chunk its
    /// size and the (kernel, blocks) of its launches.
    fn check(
        label: &str,
        settings: &Settings,
        geometry: &Geometry,
        (a, b): (&[u64], &[u64]),
        request: &Request<'_>,
    ) -> Vec<(usize, Vec<(usize, u64)>)> {
        let (a_off, b_off) = (offsets(a), offsets(b));
        let profile = Profile::new(&b_off, settings).unwrap();
        let lengths = Lengths {
            a_off: &a_off,
            b_off: &b_off,
            profile: &profile,
        };
        let mut chunks = Vec::new();
        let mut pos = 0;
        while pos < request.end() {
            let plan = plan_chunk(&lengths, settings, geometry, request, pos);
            assert!(plan.n >= 1 && pos + plan.n <= request.end(), "{label}");
            assert!(plan.n as u64 <= settings.pairs_per_chunk, "{label}");
            let launches = tier_launches(settings, geometry, plan.w_min, plan.w_max);
            let mut units: Vec<Vec<u64>> = launches
                .iter()
                .map(|launch| {
                    let blocks = launch.blocks(settings, &plan);
                    assert!(blocks <= MAX_GRID, "{label}");
                    if launch.slot_words > 0 {
                        // All slots within the budget, or the one-block minimum.
                        assert!(
                            blocks == 1
                                || blocks * launch.per_block * launch.slot_words * 8
                                    <= geometry.slot_bytes,
                            "{label}: {launch:?} exceeds the slot budget"
                        );
                    }
                    vec![0; (blocks * launch.per_block) as usize]
                })
                .collect();
            for t in 0..plan.n {
                let (i, j) = item(request, pos + t);
                let (m, n) = (a[i], b[j]);
                let pair_tier = tier(settings, m, n);
                let w = m.div_ceil(64);
                assert!(w >= plan.w_min && w <= plan.w_max, "{label}");
                let serving: Vec<usize> = (0..launches.len())
                    .filter(|&k| serves(&launches[k], pair_tier, w))
                    .collect();
                if pair_tier == Tier::Host {
                    assert!(serving.is_empty(), "{label}");
                    continue;
                }
                assert_eq!(serving.len(), 1, "{label}: pair (m={m}, n={n})");
                let unit = &mut units[serving[0]];
                let slot = t % unit.len();
                unit[slot] += depth(settings, m, n);
            }
            for (launch, unit) in launches.iter().zip(&units) {
                let worst = unit.iter().copied().max().unwrap_or(0);
                assert!(
                    worst <= settings.pair_steps,
                    "{label}: chunk at {pos}, {launch:?}: a unit runs {worst} serial word steps"
                );
            }
            let grids = launches
                .iter()
                .map(|launch| (launch.kernel, launch.blocks(settings, &plan)))
                .collect();
            chunks.push((plan.n, grids));
            pos += plan.n;
        }
        chunks
    }

    fn range(work: &WorkSource) -> Request<'_> {
        Request::Range {
            work,
            end: work.n_pairs,
        }
    }

    /// The review case (wf_e0787055-2bb correctness.0), scaled down: a raw
    /// table a with sigma = 256 and W = 4 whose slot budget holds 8 thread
    /// slots, which the one-block minimum makes one block of 32 threads; b's
    /// column 0 is a pair of depth exactly pair_steps in every row. Without
    /// the per-unit rule thread 0 runs every row's long pair in one launch.
    #[test]
    fn capped_raw_grid_runs_at_most_pair_steps_per_unit() {
        let settings = settings(4, 256, 4 * 1000);
        let geometry = Geometry {
            threads_per_block: 32,
            raw_sigma: Some(256),
            resident_threads: 512,
            slot_bytes: 8 * 256 * 4 * 8,
        };
        let a = vec![256u64; 24];
        let mut b = vec![3u64; 16];
        b[0] = 1000; // depth 4 * 1000 = pair_steps: on the device, not routed
        let work = WorkSource::new(PairKind::Cross, a.len(), b.len()).unwrap();
        let chunks = check("review case", &settings, &geometry, (&a, &b), &range(&work));
        // 32 units and D = pair_steps: one pair per unit and launch.
        assert!(chunks.iter().all(|(n, _)| *n <= 32), "{chunks:?}");
        // The same table b without the long row keeps one chunk.
        b[0] = 3;
        let chunks = check("short b", &settings, &geometry, (&a, &b), &range(&work));
        assert_eq!(chunks.len(), 1, "{chunks:?}");
    }

    /// A kernel's grid has one unit per item it may own and at least one
    /// resident wave; the per-unit rule raises it when a device pair is deep,
    /// and routed rows of b shrink nothing. Chunks stay whole throughout.
    #[test]
    fn grids_follow_owned_items_and_the_per_unit_rule() {
        let settings = settings(4, 256, 1 << 16);
        let geometry = Geometry {
            threads_per_block: 256,
            raw_sigma: None,
            resident_threads: 64 * 32, // 8 thread blocks or 8 warp blocks
            slot_bytes: SLOT_BYTES,
        };
        let work = WorkSource::new(PairKind::Cross, 40, 50).unwrap();
        // Warp tier only (W = 16, warp Q = 1): one warp per item.
        let mut b = vec![100u64; 50];
        let warp_rows = vec![1000u64; 40];
        let chunks = check(
            "warp",
            &settings,
            &geometry,
            (&warp_rows, &b),
            &range(&work),
        );
        assert_eq!(chunks, vec![(2000, vec![(1, 250)])]);
        // 39 thread-tier rows (W = 2) and one warp-tier row: the warp kernel
        // owns 50 items and gets one resident wave of 64 warps.
        let mut mixed = vec![100u64; 40];
        mixed[39] = 1000;
        let chunks = check("mixed", &settings, &geometry, (&mixed, &b), &range(&work));
        assert_eq!(chunks, vec![(2000, vec![(0, 8), (1, 8)])]);
        // One routed b row (depth > pair_steps) changes nothing.
        b[7] = 1 << 20;
        let chunks = check("routed", &settings, &geometry, (&mixed, &b), &range(&work));
        assert_eq!(chunks, vec![(2000, vec![(0, 8), (1, 8)])]);
        // A warp pair of depth pair_steps: one item per warp, 250 blocks.
        b[7] = (1 << 16) - 31;
        let chunks = check("deep", &settings, &geometry, (&mixed, &b), &range(&work));
        assert_eq!(chunks, vec![(2000, vec![(0, 8), (1, 250)])]);
    }

    /// Random tables, shapes, tier settings and grid caps.
    #[test]
    fn random_plans_partition_cover_and_bound_every_unit() {
        let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
        for round in 0..300 {
            let pair_steps = [0, 7, 64, 300, 5000][rng.next(5) as usize];
            let thread_words = rng.next(5);
            let device_words = [0, 1, 2, 3, 9, 256][rng.next(6) as usize];
            let mut settings = settings(thread_words, device_words, pair_steps);
            settings.pairs_per_chunk = [1, 5, 64, 1 << 22][rng.next(4) as usize];
            settings.steps_per_launch = [1, 50, 1 << 34][rng.next(3) as usize];
            settings.threads_per_block = 32 * (1 + rng.next(2) as u32);
            let geometry = Geometry {
                threads_per_block: u64::from(settings.threads_per_block),
                raw_sigma: [None, Some(1), Some(7), Some(256)][rng.next(4) as usize],
                resident_threads: 32 * (1 + rng.next(8)),
                slot_bytes: 8 * (1 + rng.next(4096)),
            };
            let label = format!("round {round}: {settings:?} {geometry:?}");
            let lens = |rng: &mut Rng, count: u64| -> Vec<u64> {
                (0..count)
                    .map(|_| [0, 1, 63, 64, 65, 129, 300, 700][rng.next(8) as usize] + rng.next(3))
                    .collect()
            };
            let n_a = 1 + rng.next(6);
            let a = lens(&mut rng, n_a);
            let n_b = 1 + rng.next(6);
            let b = lens(&mut rng, n_b);
            let zip_b = lens(&mut rng, n_a);
            let run = |work: &WorkSource, a: &[u64], b: &[u64]| {
                check(&label, &settings, &geometry, (a, b), &range(work));
            };
            run(
                &WorkSource::new(PairKind::Cross, a.len(), b.len()).unwrap(),
                &a,
                &b,
            );
            run(
                &WorkSource::new(PairKind::OneToMany, 1, b.len()).unwrap(),
                &a[..1],
                &b,
            );
            run(
                &WorkSource::new(PairKind::Zip, a.len(), a.len()).unwrap(),
                &a,
                &zip_b,
            );
            let i: Vec<usize> = (0..9).map(|_| rng.next(a.len() as u64) as usize).collect();
            let j: Vec<usize> = (0..9).map(|_| rng.next(b.len() as u64) as usize).collect();
            let index = PairKind::Index {
                i: IndexArray::from(i),
                j: IndexArray::from(j),
            };
            run(&WorkSource::new(index, a.len(), b.len()).unwrap(), &a, &b);
            for self_pairs in [false, true] {
                for kind in [
                    PairKind::Within { self_pairs },
                    PairKind::WithinOriented {
                        self_pairs,
                        orientation: Orientation::AsGiven,
                    },
                    PairKind::WithinOriented {
                        self_pairs,
                        orientation: Orientation::Both,
                    },
                ] {
                    run(&WorkSource::new(kind, a.len(), a.len()).unwrap(), &a, &a);
                }
            }
            let pairs: Vec<(usize, usize)> = (0..11)
                .map(|_| {
                    (
                        rng.next(a.len() as u64) as usize,
                        rng.next(b.len() as u64) as usize,
                    )
                })
                .collect();
            check(
                &label,
                &settings,
                &geometry,
                (&a, &b),
                &Request::Pairs(&pairs),
            );
            let r0 = rng.next(a.len() as u64) as usize;
            let c0 = rng.next(b.len() as u64) as usize;
            let (rows, width) = (a.len() - r0, b.len() - c0);
            let rect = Request::Rect {
                r0,
                c0,
                width,
                end: rows * width,
                class: None,
            };
            check(&label, &settings, &geometry, (&a, &b), &rect);
        }
    }

    // ------------------------------------------------------------- align

    fn aln_settings(thread_rows: u64, lane_rows: u64, pair_cells: u64) -> Settings {
        Settings {
            thread_rows,
            lane_rows,
            pair_cells,
            ..settings(4, 256, 1 << 22)
        }
    }

    /// The §5.6 predicate, independently of `aln_tier`: (tier, depth).
    fn aln_spec(s: &Settings, closed_below: u64, m: u64, n: u64) -> (AlnTier, u64) {
        if m < closed_below || n < closed_below {
            return (AlnTier::Closed, 0);
        }
        let big = if m <= 32 {
            1
        } else if m <= 64 {
            2
        } else if m <= 128 {
            4
        } else {
            8
        };
        let q = s.lane_rows.min(big);
        let warp = m.div_ceil(32 * q) * (n + 31) * q;
        if s.thread_rows > 0 && m <= s.thread_rows && m * n <= s.pair_cells {
            (AlnTier::Thread, m * n)
        } else if warp <= s.pair_cells {
            (AlnTier::Warp, warp)
        } else {
            (AlnTier::Host, warp)
        }
    }

    #[test]
    fn aln_tier_predicate_and_depths_follow_the_spec() {
        // Closed forms are never routed, whatever pair_cells is.
        let none = aln_settings(0, 1, 0);
        for (closed_below, m, n) in [(1, 0, 500), (1, 500, 0), (2, 1, 500), (2, 500, 1)] {
            assert_eq!(aln_tier(&none, closed_below, m, n), AlnTier::Closed);
        }
        assert_eq!(
            aln_tier(&none, 0, 1, 1),
            AlnTier::Host,
            "needle has no closed form"
        );
        // Defaults: thread m <= 32 and m n <= 2^22; a deeper thread pair falls
        // through to the warp tier (QAG3) before it is routed.
        let s = aln_settings(32, 8, 1 << 22);
        assert_eq!(aln_tier(&s, 1, 32, 1 << 17), AlnTier::Thread);
        assert_eq!(aln_tier(&s, 1, 32, (1 << 17) + 1), AlnTier::Warp);
        assert_eq!(aln_tier(&s, 1, 32, (1 << 22) - 31), AlnTier::Warp);
        assert_eq!(aln_tier(&s, 1, 32, (1 << 22) - 30), AlnTier::Host);
        // Single strip, q = 8: m = 256, depth (n + 31)·8.
        assert_eq!((aln_q(&s, 256), aln_factor(&s, 256)), (8, 8));
        assert_eq!(aln_tier(&s, 1, 256, (1 << 19) - 31), AlnTier::Warp);
        assert_eq!(aln_tier(&s, 1, 256, (1 << 19) - 30), AlnTier::Host);
        // Two strips of 256 rows: m = 512, depth 2·(n + 31)·8.
        assert_eq!(aln_factor(&s, 512), 16);
        assert!(aln_multi(&s, 512) && !aln_multi(&s, 256));
        assert_eq!(aln_tier(&s, 1, 512, (1 << 18) - 31), AlnTier::Warp);
        assert_eq!(aln_tier(&s, 1, 512, (1 << 18) - 30), AlnTier::Host);
        // lane_rows caps q: strips of 32 rows.
        let s1 = aln_settings(0, 1, 1 << 22);
        assert_eq!((aln_q(&s1, 257), aln_factor(&s1, 257)), (1, 9));
        assert!(aln_multi(&s1, 33) && !aln_multi(&s1, 32));
        // Exhaustive agreement with the spec form on small grids.
        for thread_rows in [0, 1, 5, 32] {
            for lane_rows in [1, 2, 4, 8] {
                for pair_cells in [0, 1, 31, 64, 100, 1000, 1 << 22] {
                    let s = aln_settings(thread_rows, lane_rows, pair_cells);
                    for closed_below in [0, 1, 2] {
                        for m in (0..70).chain([127, 128, 129, 255, 256, 257, 300, 769]) {
                            for n in [0, 1, 2, 3, 31, 32, 33, 64, 200, 5000] {
                                let (tier, depth) = aln_spec(&s, closed_below, m, n);
                                assert_eq!(aln_tier(&s, closed_below, m, n), tier);
                                let device = aln_device(&s, closed_below, m, n);
                                match tier {
                                    AlnTier::Thread | AlnTier::Warp => {
                                        let device = device.unwrap();
                                        assert_eq!(device.depth, depth);
                                        assert!(device.depth <= pair_cells);
                                    }
                                    _ => assert!(device.is_none()),
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    /// Whether an align tier launch computes a pair (the device kernels'
    /// skips in kernels/align.cuh).
    fn aln_serves(s: &Settings, launch: &AlnLaunch, tier: AlnTier, m: u64) -> bool {
        match launch.kernel {
            0 => tier == AlnTier::Thread,
            k => tier == AlnTier::Warp && aln_q(s, m) == ALN_WARP_Q[k - 1],
        }
    }

    /// Plans every align chunk of `request` and simulates each launch's
    /// grid-stride assignment: chunks partition the items and close at the
    /// cell budget, every device pair is computed by exactly one launched
    /// kernel, slots fit their pairs and the budget, and no unit runs more
    /// than `pair_cells` serial cell steps in one launch. Returns per chunk
    /// its size and the (kernel, blocks) of its launches.
    fn aln_check(
        label: &str,
        s: &Settings,
        geometry: &AlnGeometry,
        closed_below: u64,
        (a, b): (&[u64], &[u64]),
        request: &Request<'_>,
    ) -> Vec<(usize, Vec<(usize, u64)>)> {
        let (a_off, b_off) = (offsets(a), offsets(b));
        let profile = AlnProfile::new(&b_off, s, closed_below).unwrap();
        let lengths = AlnLengths {
            a_off: &a_off,
            b_off: &b_off,
            profile: &profile,
        };
        let mut chunks = Vec::new();
        let mut pos = 0;
        while pos < request.end() {
            let plan = aln_plan_chunk(&lengths, s, geometry, request, pos);
            assert!(plan.n >= 1 && pos + plan.n <= request.end(), "{label}");
            assert!(plan.n as u64 <= s.pairs_per_chunk, "{label}");
            let cost: u64 = (0..plan.n)
                .map(|t| {
                    let (i, j) = item(request, pos + t);
                    a[i] * b[j].min(s.pair_cells)
                })
                .sum();
            // The first item always fits; later ones only within the budget
            // (zero-cost items, closed form or n = 0, may follow a full chunk).
            let first = {
                let (i, j) = item(request, pos);
                a[i] * b[j].min(s.pair_cells)
            };
            assert!(
                cost <= s.cells_per_launch.max(first),
                "{label}: chunk cost {cost}"
            );
            let launches = aln_launches(geometry, &plan);
            let mut units: Vec<Vec<u64>> = launches
                .iter()
                .map(|launch| {
                    let blocks = launch.blocks(s, &plan);
                    assert!(blocks <= MAX_GRID, "{label}");
                    if launch.slot_cols > 0 {
                        assert!(launch.kernel > 0, "{label}");
                        assert!(
                            blocks == 1
                                || blocks * launch.per_block * launch.slot_cols * 8
                                    <= geometry.slot_bytes,
                            "{label}: {launch:?} exceeds the slot budget"
                        );
                    }
                    vec![0; (blocks * launch.per_block) as usize]
                })
                .collect();
            for t in 0..plan.n {
                let (i, j) = item(request, pos + t);
                let (m, n) = (a[i], b[j]);
                let (tier, depth) = aln_spec(s, closed_below, m, n);
                let serving: Vec<usize> = (0..launches.len())
                    .filter(|&k| aln_serves(s, &launches[k], tier, m))
                    .collect();
                if matches!(tier, AlnTier::Closed | AlnTier::Host) {
                    assert!(serving.is_empty(), "{label}");
                    continue;
                }
                assert_eq!(serving.len(), 1, "{label}: pair (m={m}, n={n})");
                let launch = &launches[serving[0]];
                if tier == AlnTier::Warp && aln_multi(s, m) {
                    assert!(
                        launch.slot_cols > n,
                        "{label}: slot of {launch:?} for n={n}"
                    );
                }
                let unit = &mut units[serving[0]];
                let slot = t % unit.len();
                unit[slot] += depth;
            }
            for (launch, unit) in launches.iter().zip(&units) {
                let worst = unit.iter().copied().max().unwrap_or(0);
                assert!(
                    worst <= s.pair_cells,
                    "{label}: chunk at {pos}, {launch:?}: a unit runs {worst} serial cell steps"
                );
            }
            let grids = launches
                .iter()
                .map(|launch| (launch.kernel, launch.blocks(s, &plan)))
                .collect();
            chunks.push((plan.n, grids));
            pos += plan.n;
        }
        chunks
    }

    /// A multi-strip warp kernel whose slots cap the grid at one block: each
    /// unit takes one pair of depth pair_cells per launch, so chunks shrink
    /// to the block's warps; short partners keep one chunk.
    #[test]
    fn aln_slot_capped_multi_strip_runs_at_most_pair_cells_per_unit() {
        // m = 64 with lane_rows 1: two strips of 32 rows, depth 2·(n + 31).
        let s = aln_settings(0, 1, 2 * (1000 + 31));
        let geometry = AlnGeometry {
            thread_block: 64,
            warp_block: 64,
            resident_threads: 1024,
            // Two warps' slots of 1001 columns: one block of 2 warps.
            slot_bytes: 2 * 8 * 1001,
        };
        let a = vec![64u64; 12];
        let mut b = vec![3u64; 9];
        b[0] = 1000; // depth exactly pair_cells: on the device, with a slot
        let work = WorkSource::new(PairKind::Cross, a.len(), b.len()).unwrap();
        let chunks = aln_check("slot cap", &s, &geometry, 1, (&a, &b), &range(&work));
        assert!(chunks.iter().all(|(n, _)| *n <= 2), "{chunks:?}");
        assert!(chunks.iter().all(|(_, g)| g == &vec![(1, 1)]), "{chunks:?}");
        // One deeper partner is routed (depth pair_cells + 2) and needs no slot room.
        b[0] = 1001;
        let chunks = aln_check("routed", &s, &geometry, 1, (&a, &b), &range(&work));
        assert_eq!(chunks.len(), 1, "{chunks:?}");
        // The same table b without the long row keeps one chunk.
        b[0] = 3;
        let chunks = aln_check("short b", &s, &geometry, 1, (&a, &b), &range(&work));
        assert_eq!(chunks.len(), 1, "{chunks:?}");
    }

    /// Chunks close at cells_per_launch (a pair costs m·min(n, pair_cells))
    /// and hold at least one pair; a row of thread-tier candidates whose deep
    /// partners fall through owns items in both the thread and warp kernels.
    #[test]
    fn aln_chunks_close_at_the_cell_budget() {
        let mut s = aln_settings(32, 8, 1000);
        s.cells_per_launch = 100;
        let geometry = AlnGeometry {
            thread_block: 32,
            warp_block: 32,
            resident_threads: 64,
            slot_bytes: SLOT_BYTES,
        };
        let a = vec![10u64, 10, 10, 2];
        let b = vec![5u64, 200, 40, 7];
        let work = WorkSource::new(PairKind::Cross, a.len(), b.len()).unwrap();
        let chunks = aln_check("budget", &s, &geometry, 1, (&a, &b), &range(&work));
        // (10·5) + ... closes at 100 cells: [50], [10·200 alone], [400 alone] ...
        assert_eq!(chunks.iter().map(|c| c.0).sum::<usize>(), 16, "{chunks:?}");
        assert_eq!(chunks[0].0, 1, "{chunks:?}");
        s.cells_per_launch = 1 << 34;
        let chunks = aln_check("mixed", &s, &geometry, 1, (&a, &b), &range(&work));
        // m = 10 against n = 200: depth 2000 > 1000 falls to warp q = 1
        // (231 <= 1000); so both kernels launch.
        assert_eq!(chunks.len(), 1, "{chunks:?}");
        let kernels: Vec<usize> = chunks[0].1.iter().map(|g| g.0).collect();
        assert_eq!(kernels, vec![0, 1], "{chunks:?}");
    }

    /// Random tables, shapes, tier settings, kinds and grid caps.
    #[test]
    fn aln_random_plans_partition_cover_and_bound_every_unit() {
        let mut rng = Rng(0x2545_f491_4f6c_dd1d);
        for round in 0..300 {
            let pair_cells = [0, 7, 64, 300, 5000, 1 << 22][rng.next(6) as usize];
            let thread_rows = [0, 1, 5, 32][rng.next(4) as usize];
            let lane_rows = ALN_WARP_Q[rng.next(4) as usize];
            let mut s = aln_settings(thread_rows, lane_rows, pair_cells);
            s.pairs_per_chunk = [1, 5, 64, 1 << 22][rng.next(4) as usize];
            s.cells_per_launch = [1, 50, 1 << 34][rng.next(3) as usize];
            s.threads_per_block = 32 * (1 + rng.next(2) as u32);
            let geometry = AlnGeometry {
                thread_block: u64::from(s.threads_per_block),
                warp_block: 32 * (1 + rng.next(2)),
                resident_threads: 32 * (1 + rng.next(8)),
                slot_bytes: 8 * (1 + rng.next(4096)),
            };
            let closed_below = rng.next(3);
            let label = format!("round {round}: {s:?} {geometry:?} closed<{closed_below}");
            let lens = |rng: &mut Rng, count: u64| -> Vec<u64> {
                (0..count)
                    .map(|_| {
                        [0, 1, 2, 31, 32, 33, 64, 65, 129, 300][rng.next(10) as usize] + rng.next(2)
                    })
                    .collect()
            };
            let n_a = 1 + rng.next(6);
            let a = lens(&mut rng, n_a);
            let n_b = 1 + rng.next(6);
            let b = lens(&mut rng, n_b);
            let zip_b = lens(&mut rng, n_a);
            let run = |work: &WorkSource, a: &[u64], b: &[u64]| {
                aln_check(&label, &s, &geometry, closed_below, (a, b), &range(work));
            };
            run(
                &WorkSource::new(PairKind::Cross, a.len(), b.len()).unwrap(),
                &a,
                &b,
            );
            run(
                &WorkSource::new(PairKind::OneToMany, 1, b.len()).unwrap(),
                &a[..1],
                &b,
            );
            run(
                &WorkSource::new(PairKind::Zip, a.len(), a.len()).unwrap(),
                &a,
                &zip_b,
            );
            let i: Vec<usize> = (0..9).map(|_| rng.next(a.len() as u64) as usize).collect();
            let j: Vec<usize> = (0..9).map(|_| rng.next(b.len() as u64) as usize).collect();
            let index = PairKind::Index {
                i: IndexArray::from(i),
                j: IndexArray::from(j),
            };
            run(&WorkSource::new(index, a.len(), b.len()).unwrap(), &a, &b);
            for self_pairs in [false, true] {
                run(
                    &WorkSource::new(PairKind::Within { self_pairs }, a.len(), a.len()).unwrap(),
                    &a,
                    &a,
                );
            }
            let pairs: Vec<(usize, usize)> = (0..11)
                .map(|_| {
                    (
                        rng.next(a.len() as u64) as usize,
                        rng.next(b.len() as u64) as usize,
                    )
                })
                .collect();
            aln_check(
                &label,
                &s,
                &geometry,
                closed_below,
                (&a, &b),
                &Request::Pairs(&pairs),
            );
            let r0 = rng.next(a.len() as u64) as usize;
            let c0 = rng.next(b.len() as u64) as usize;
            let (rows, width) = (a.len() - r0, b.len() - c0);
            let rect = Request::Rect {
                r0,
                c0,
                width,
                end: rows * width,
                class: None,
            };
            aln_check(&label, &s, &geometry, closed_below, (&a, &b), &rect);
        }
    }

    #[test]
    fn unary_chunks_close_at_items_or_bytes_and_hold_one_sequence() {
        let settings = |items: u64, bytes: u64| Settings {
            pairs_per_chunk: items,
            bytes_per_launch: bytes,
            ..crate::launch::settings()
        };
        // Lengths 3, 200, 2, 2, 5.
        let off = [10i64, 13, 213, 215, 217, 222];
        let chunks = |s: Settings| {
            let (mut k0, mut ends) = (0, Vec::new());
            while k0 < off.len() - 1 {
                k0 = unary_chunk_end(&off, k0, &s);
                ends.push(k0);
            }
            ends
        };
        assert_eq!(chunks(settings(1 << 22, 1 << 30)), vec![5]);
        assert_eq!(chunks(settings(1, 1 << 30)), vec![1, 2, 3, 4, 5]);
        assert_eq!(chunks(settings(2, 1 << 30)), vec![2, 4, 5]);
        // A 200-mer beyond the byte budget still forms a chunk of one.
        assert_eq!(chunks(settings(1 << 22, 199)), vec![1, 2, 5]);
        assert_eq!(chunks(settings(1 << 22, 1)), vec![1, 2, 3, 4, 5]);
        assert_eq!(chunks(settings(1 << 22, 203)), vec![2, 5]);
    }

    // -------------------------------------------------------------- duplex

    /// sm_120's opt-in per-block shared memory, 99 KB.
    const SM120: DupLimits = DupLimits {
        s_max: 99 * 1024,
        s_block: 0,
        codes: DUP_CODES,
    };

    fn dup_cfg(group_lanes: u64, band_bytes: u64, pair_cells: u64) -> Settings {
        Settings {
            group_lanes,
            band_bytes,
            pair_cells,
            ..settings(4, 256, 1 << 22)
        }
    }

    /// c(n, m, Λ) by the §5.6 sum, term by term.
    fn c_sum(n: u64, m: u64, lam: u64) -> u64 {
        if n < 2 || m < 2 {
            return 0;
        }
        (0..=lam.min(n - 2)).map(|u| (lam - u).min(m - 2) + 1).sum()
    }

    #[test]
    fn dup_candidate_counts_and_depths_follow_the_spec() {
        for lam in 0..=7 {
            for n in 0..=11 {
                for m in 0..=11 {
                    assert_eq!(dup_c(n, m, lam), c_sum(n, m, lam), "c({n}, {m}, {lam})");
                }
            }
        }
        for lam in [0, 1, 2, 7, 30, 31, 100, 199, 200, 396] {
            for n in (0..12).chain([30, 31, 32, 33, 101, 102, 199, 200, 201, 398, 399]) {
                for m in (0..12).chain([16, 17, 32, 33, 64, 65, 128, 129, 200, 398, 1000]) {
                    assert_eq!(dup_c(n, m, lam), c_sum(n, m, lam), "c({n}, {m}, {lam})");
                }
                if n >= 2 {
                    assert_eq!(dup_c_inf(n, lam), c_sum(lam + 2, n, lam), "c∞({n}, {lam})");
                }
            }
        }
        assert_eq!((dup_c_inf(0, 30), dup_c_inf(1, 30)), (0, 0));
        // §5.6 examples.
        let s = dup_cfg(0, 32_768, 1 << 22);
        assert_eq!(dup_c(200, 200, 30), 496);
        assert_eq!(dup_depth(&s, 30, 200, 200), 694_400);
        assert_eq!(dup_depth(&dup_cfg(4, 0, 0), 30, 200, 200), 4_960_000);
        assert_eq!(dup_depth(&s, 100, 200, 200), 7_211_400);
        assert_eq!(dup_c(200, 200, 396), 39_601);
        assert_eq!(dup_depth(&s, 396, 200, 200), 55_441_400);
        // §4 "Chunks": the largest implicit pair cost.
        assert_eq!(200 * 200 * dup_c_inf(200, 396), 2_372_080_000);
        // g by m, or group_lanes.
        for (m, g) in [
            (0, 4),
            (16, 4),
            (17, 8),
            (32, 8),
            (33, 16),
            (64, 16),
            (65, 32),
            (200, 32),
        ] {
            assert_eq!(dup_group(&s, m), g, "m = {m}");
            assert_eq!(dup_group(&dup_cfg(8, 0, 0), m), 8);
        }
        // Band bytes (§5.4 examples).
        assert_eq!(dup_band(30, DupWidth::Narrow, 200, 200), 25_600);
        assert_eq!(dup_band(30, DupWidth::Wide, 200, 200), 51_200);
        assert_eq!(dup_band(30, DupWidth::Detail, 200, 200), 204_800);
        assert_eq!(dup_band(100, DupWidth::Narrow, 120, 120), 48_960);
        assert_eq!(dup_band(30, DupWidth::Narrow, 3, 200), 2_400);
    }

    #[test]
    fn dup_settings_cap_pair_cells() {
        for (pair_cells, capped) in [(0, 0), (64, 64), (1 << 21, 1 << 21), (1 << 22, 1 << 21)] {
            let s = super::dup_settings(&dup_cfg(0, 32_768, pair_cells));
            assert_eq!(s.pair_cells, capped);
            assert_eq!(s.band_bytes, 32_768);
        }
        // The default V1 (Λ = 30) pairs stay on the device: largest depth 694,400.
        let s = super::dup_settings(&dup_cfg(0, 32_768, 1 << 22));
        assert!(!dup_routed(&s, 30, 200, 200));
        // A g = 4 pair of depth 2,083,200 stays, 2,108,000 is routed.
        let four = Settings {
            group_lanes: 4,
            ..s
        };
        assert_eq!(dup_depth(&four, 30, 84, 200), 2_083_200);
        assert!(!dup_routed(&four, 30, 84, 200) && dup_routed(&four, 30, 85, 200));
    }

    #[test]
    fn dup_tiers_and_routing_follow_the_spec() {
        use DupWidth::{Detail, Narrow};
        // Closed forms are never routed, whatever pair_cells is.
        let none = dup_cfg(0, 32_768, 0);
        for (n, m) in [(0, 0), (0, 200), (1, 200), (200, 1), (200, 0)] {
            assert_eq!(dup_tier(&none, &SM120, 30, Narrow, n, m), DupTier::Closed);
            assert!(!dup_routed(&none, 30, n, m));
        }
        assert_eq!(dup_tier(&none, &SM120, 0, Narrow, 2, 2), DupTier::Host);
        // Defaults: no pair of a Λ = 30 pack is routed; 200 × 200 at Λ = 100 is.
        let s = dup_cfg(0, 32_768, 1 << 22);
        assert_eq!(dup_tier(&s, &SM120, 30, Narrow, 200, 200), DupTier::Shared);
        assert_eq!(dup_tier(&s, &SM120, 100, Narrow, 200, 200), DupTier::Host);
        // §8 shared-tier case: Λ = 100, n = 200, m = 128, band 52,224 B.
        let wide = dup_cfg(0, 98_304, 1 << 22);
        assert_eq!(dup_depth(&wide, 100, 200, 128), 4_120_800);
        assert_eq!(dup_band(100, Narrow, 200, 128), 52_224);
        assert_eq!(
            dup_tier(&wide, &SM120, 100, Narrow, 200, 128),
            DupTier::Shared
        );
        assert_eq!(dup_tier(&s, &SM120, 100, Narrow, 200, 128), DupTier::Global);
        assert_eq!(dup_depth(&wide, 100, 200, 129), 5_151_000);
        assert_eq!(
            dup_tier(&wide, &SM120, 100, Narrow, 200, 129),
            DupTier::Host
        );
        // §8 global-tier case: Λ = 30, group_lanes 4, n = 169 or 170, m = 200.
        let four = dup_cfg(4, 32_768, 1 << 22);
        assert_eq!(dup_depth(&four, 30, 169, 200), 4_191_200);
        assert_eq!(dup_depth(&four, 30, 170, 200), 4_216_000);
        assert_eq!(
            dup_tier(&four, &SM120, 30, Narrow, 169, 200),
            DupTier::Global
        );
        assert_eq!(dup_tier(&four, &SM120, 30, Narrow, 170, 200), DupTier::Host);
        // Depth exactly pair_cells stays on the device, one more is routed.
        let edge = dup_cfg(0, 32_768, 694_400);
        assert_eq!(
            dup_tier(&edge, &SM120, 30, Narrow, 200, 200),
            DupTier::Shared
        );
        let edge = dup_cfg(0, 32_768, 694_399);
        assert_eq!(dup_tier(&edge, &SM120, 30, Narrow, 200, 200), DupTier::Host);
        // A band of exactly band_bytes is shared, one key more is global.
        let exact = dup_cfg(0, 25_600, 1 << 22);
        assert_eq!(
            dup_tier(&exact, &SM120, 30, Narrow, 200, 200),
            DupTier::Shared
        );
        let exact = dup_cfg(0, 25_596, 1 << 22);
        assert_eq!(
            dup_tier(&exact, &SM120, 30, Narrow, 200, 200),
            DupTier::Global
        );
        // band_bytes 0 sends every pair to the global tier.
        let zero = dup_cfg(0, 0, 1 << 22);
        assert_eq!(dup_tier(&zero, &SM120, 30, Narrow, 2, 2), DupTier::Global);
        // §5.4: a detail-pass group at g = 4 (m = 16, R = 32) needs 8 · 16,384 B
        // per warp, more than S_max: global even at the largest band_bytes.
        let most = dup_cfg(0, 98_304, 1 << 22);
        assert_eq!(dup_band(30, Detail, 200, 16), 16_384);
        assert_eq!(
            dup_tier(&most, &SM120, 30, Detail, 200, 16),
            DupTier::Global
        );
        // §5.4: the 81,600-byte band of 200 × 200 under Λ = 100 fits band_bytes
        // but not the block once 25.6 KB of tables are staged per block.
        let staged = DupLimits {
            s_max: 99 * 1024,
            s_block: 25_600,
            codes: DUP_CODES,
        };
        let deep = dup_cfg(0, 98_304, 1 << 30);
        assert_eq!(dup_band(100, Narrow, 200, 200), 81_600);
        assert_eq!(
            dup_tier(&deep, &staged, 100, Narrow, 200, 200),
            DupTier::Global
        );
        assert_eq!(
            dup_tier(&deep, &SM120, 100, Narrow, 200, 200),
            DupTier::Shared
        );
        // The fit: (32/g)·(b + q) + s_block <= S_max.
        for g in DUP_GROUPS {
            let fit = dup_fit(&SM120, g).unwrap();
            assert!((32 / g) * (fit + DUP_CODES) <= SM120.s_max);
            assert!((32 / g) * (fit + 1 + DUP_CODES) > SM120.s_max);
            // With STATES the staged bytes are q = 800 (nndb2024-dna.md §5.4.4).
            let states = DupLimits {
                codes: DUP_CODES_STATES,
                ..SM120
            };
            let fit = dup_fit(&states, g).unwrap();
            assert!((32 / g) * (fit + DUP_CODES_STATES) <= SM120.s_max);
            assert!((32 / g) * (fit + 1 + DUP_CODES_STATES) > SM120.s_max);
        }
        assert_eq!(
            dup_fit(
                &DupLimits {
                    s_max: 100,
                    s_block: 0,
                    codes: DUP_CODES,
                },
                32
            ),
            None
        );
    }

    /// The §5.6 predicate for one pair, independently of `dup_tier`:
    /// (variant or None, depth, band).
    fn dup_spec(
        s: &Settings,
        limits: &DupLimits,
        lam: u64,
        width: DupWidth,
        n: u64,
        m: u64,
    ) -> (Option<usize>, u64, u64) {
        if n < 2 || m < 2 {
            return (None, 0, 0);
        }
        let g = if s.group_lanes > 0 {
            s.group_lanes
        } else if m <= 16 {
            4
        } else if m <= 32 {
            8
        } else if m <= 64 {
            16
        } else {
            32
        };
        let depth = n * m.div_ceil(g) * c_sum(n, m, lam);
        if depth > s.pair_cells {
            return (None, depth, 0);
        }
        let band = n.min(lam + 2) * m * width.bytes();
        let shared = band <= s.band_bytes
            && (32 / g) * (band + limits.codes) + limits.s_block <= limits.s_max;
        let k = DUP_GROUPS.iter().position(|&x| x == g).unwrap();
        (Some(2 * k + usize::from(!shared)), depth, band)
    }

    /// Plans every duplex chunk of `request` and simulates each launch's
    /// grid-stride assignment: chunks partition the items and close at the
    /// candidate budget, every device pair is computed by exactly one launch
    /// of its variant, slots and shared memory fit, the slot budget holds, and
    /// no group runs more than `pair_cells` candidate evaluations per launch.
    #[allow(clippy::too_many_arguments)]
    fn dup_check(
        label: &str,
        s: &Settings,
        geometry: &DupGeometry,
        lam: u64,
        width: DupWidth,
        implicit: bool,
        (a, b): (&[u64], &[u64]),
        request: &Request<'_>,
    ) -> Vec<(usize, Vec<(usize, u64)>)> {
        let (a_off, b_off) = (offsets(a), offsets(b));
        let profile = DupProfile::new(&b_off, s, lam).unwrap();
        let lengths = DupLengths {
            a_off: &a_off,
            b_off: &b_off,
            profile: &profile,
        };
        let limits = &geometry.limits;
        let cost = |i: usize, j: usize| {
            let (n, m) = (a[i], b[j]);
            if implicit {
                if n < 2 {
                    0
                } else {
                    n * m * dup_c_inf(m, lam)
                }
            } else {
                n * m * c_sum(n, m, lam)
            }
        };
        let mut chunks = Vec::new();
        let mut pos = 0;
        while pos < request.end() {
            let plan = dup_plan_chunk(&lengths, s, geometry, request, pos, width);
            assert!(plan.n >= 1 && pos + plan.n <= request.end(), "{label}");
            assert!(plan.n as u64 <= s.pairs_per_chunk, "{label}");
            let total: u64 = (0..plan.n)
                .map(|t| {
                    let (i, j) = item(request, pos + t);
                    cost(i, j)
                })
                .sum();
            let first = {
                let (i, j) = item(request, pos);
                cost(i, j)
            };
            assert!(
                total <= s.cells_per_launch.max(first),
                "{label}: chunk cost {total}"
            );
            let launches = dup_launches(s, geometry, &plan);
            let mut units: Vec<Vec<u64>> = launches
                .iter()
                .map(|launch| {
                    let blocks = launch.blocks(s, &plan);
                    assert!((1..=MAX_GRID).contains(&blocks), "{label}");
                    assert_eq!(launch.per_block % (32 / launch.g), 0, "{label}");
                    assert!(
                        launch.threads() <= geometry.threads_per_block.max(32),
                        "{label}: {launch:?}"
                    );
                    assert!(
                        launch.shared_bytes() + limits.s_block <= limits.s_max
                            || launch.per_block == 32 / launch.g,
                        "{label}: {launch:?} exceeds the block's shared memory"
                    );
                    assert_eq!(launch.stride % width.bytes(), limits.codes % width.bytes());
                    if launch.global {
                        assert!(
                            blocks == 1
                                || blocks * launch.per_block * launch.slot <= geometry.slot_bytes,
                            "{label}: {launch:?} exceeds the slot budget"
                        );
                        assert!(
                            launch.per_block == 32 / launch.g
                                || launch.per_block * launch.slot <= geometry.slot_bytes,
                            "{label}: {launch:?}"
                        );
                    } else {
                        assert!(launch.shared_bytes() + limits.s_block <= limits.s_max);
                    }
                    vec![0; (blocks * launch.per_block) as usize]
                })
                .collect();
            for t in 0..plan.n {
                let (i, j) = item(request, pos + t);
                let (n, m) = (a[i], b[j]);
                let (variant, depth, band) = dup_spec(s, limits, lam, width, n, m);
                let serving: Vec<usize> = (0..launches.len())
                    .filter(|&k| Some(launches[k].variant) == variant)
                    .collect();
                let Some(_) = variant else {
                    assert!(serving.is_empty(), "{label}");
                    continue;
                };
                assert_eq!(serving.len(), 1, "{label}: pair (n={n}, m={m})");
                let launch = &launches[serving[0]];
                if launch.global {
                    assert!(
                        launch.slot >= band,
                        "{label}: slot of {launch:?} for {band}"
                    );
                } else {
                    assert!(
                        launch.stride >= band + limits.codes,
                        "{label}: stride of {launch:?} for {band}"
                    );
                }
                let unit = &mut units[serving[0]];
                let slot = t % unit.len();
                unit[slot] += depth;
            }
            for (launch, unit) in launches.iter().zip(&units) {
                let worst = unit.iter().copied().max().unwrap_or(0);
                assert!(
                    worst <= s.pair_cells,
                    "{label}: chunk at {pos}, {launch:?}: a group runs {worst} candidates"
                );
            }
            let grids = launches
                .iter()
                .map(|launch| (launch.variant, launch.blocks(s, &plan)))
                .collect();
            chunks.push((plan.n, grids));
            pos += plan.n;
        }
        chunks
    }

    fn dup_geometry(threads_per_block: u64, resident_threads: u64, slot_bytes: u64) -> DupGeometry {
        DupGeometry {
            threads_per_block,
            resident_threads,
            limits: SM120,
            slot_bytes,
        }
    }

    /// A global-tier variant whose slots cap the grid at one block of one
    /// warp: each group takes one pair of depth pair_cells per launch.
    #[test]
    fn dup_slot_capped_global_tier_runs_at_most_pair_cells_per_group() {
        // g = 32 (m = 100), Λ = 30, n = 40: depth 40·4·c(40, 100, 30).
        let lam = 30;
        let depth = 40 * 4 * c_sum(40, 100, lam);
        let s = dup_cfg(32, 0, depth);
        let band = dup_band(lam, DupWidth::Narrow, 40, 100);
        // Two groups' slots: one block of two groups (g = 32 is one warp).
        let geometry = dup_geometry(256, 1024, 2 * band);
        let a = vec![40u64; 12];
        let mut b = vec![3u64; 9];
        b[0] = 100;
        let work = WorkSource::new(PairKind::Cross, a.len(), b.len()).unwrap();
        let req = range(&work);
        let chunks = dup_check(
            "slot cap",
            &s,
            &geometry,
            lam,
            DupWidth::Narrow,
            true,
            (&a, &b),
            &req,
        );
        assert!(chunks.iter().all(|(n, _)| *n <= 2), "{chunks:?}");
        // A deeper partner (⌈129/32⌉ = 5 column steps) is routed and needs no slot.
        b[0] = 129;
        let chunks = dup_check(
            "routed",
            &s,
            &geometry,
            lam,
            DupWidth::Narrow,
            true,
            (&a, &b),
            &req,
        );
        assert_eq!(chunks.len(), 1, "{chunks:?}");
        b[0] = 3;
        let chunks = dup_check(
            "short b",
            &s,
            &geometry,
            lam,
            DupWidth::Narrow,
            true,
            (&a, &b),
            &req,
        );
        assert_eq!(chunks.len(), 1, "{chunks:?}");
        // A slot larger than the whole budget still runs, one block of one warp.
        let geometry = dup_geometry(256, 1024, 64);
        b[0] = 100;
        dup_check(
            "tiny budget",
            &s,
            &geometry,
            lam,
            DupWidth::Detail,
            true,
            (&a, &b),
            &req,
        );
    }

    /// Chunks close at cells_per_launch (explicit: n·m·c; implicit: n·m·c∞)
    /// and hold at least one pair; one row mixes shared, global and routed
    /// pairs of several g.
    #[test]
    fn dup_chunks_close_at_the_candidate_budget_and_mix_variants() {
        let lam = 30;
        let mut s = dup_cfg(0, 2_400, 1 << 19);
        s.cells_per_launch = 50_000;
        let geometry = dup_geometry(256, 512, SLOT_BYTES);
        let a = vec![3u64, 200, 1, 40];
        let b = vec![5u64, 200, 17, 70, 1, 2];
        let work = WorkSource::new(PairKind::Cross, a.len(), b.len()).unwrap();
        let chunks = dup_check(
            "budget",
            &s,
            &geometry,
            lam,
            DupWidth::Narrow,
            true,
            (&a, &b),
            &range(&work),
        );
        assert_eq!(chunks.iter().map(|c| c.0).sum::<usize>(), 24, "{chunks:?}");
        assert!(chunks.len() > 3, "{chunks:?}");
        s.cells_per_launch = 1 << 34;
        let chunks = dup_check(
            "mixed",
            &s,
            &geometry,
            lam,
            DupWidth::Narrow,
            true,
            (&a, &b),
            &range(&work),
        );
        assert_eq!(chunks.len(), 1, "{chunks:?}");
        let variants: Vec<usize> = chunks[0].1.iter().map(|l| l.0).collect();
        let mut expect: Vec<usize> = a
            .iter()
            .flat_map(|&n| b.iter().map(move |&m| (n, m)))
            .filter_map(|(n, m)| dup_spec(&s, &geometry.limits, lam, DupWidth::Narrow, n, m).0)
            .collect();
        expect.sort_unstable();
        expect.dedup();
        assert_eq!(variants, expect, "{chunks:?}");
        assert!(
            expect.iter().any(|v| v % 2 == 0) && expect.iter().any(|v| v % 2 == 1),
            "{expect:?}"
        );
        assert!(
            expect
                .iter()
                .map(|v| v / 2)
                .collect::<std::collections::BTreeSet<_>>()
                .len()
                >= 3
        );
        // 200 × 200 (depth 694,400) is routed.
        assert!(dup_routed(&s, lam, 200, 200));
        let pairs: Vec<(usize, usize)> = vec![(1, 1), (0, 0), (2, 3), (3, 3), (0, 5)];
        dup_check(
            "pairs",
            &s,
            &geometry,
            lam,
            DupWidth::Wide,
            false,
            (&a, &b),
            &Request::Pairs(&pairs),
        );
    }

    /// Random tables, shapes, widths, settings, kinds and grid caps.
    #[test]
    fn dup_random_plans_partition_cover_and_bound_every_unit() {
        let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
        for round in 0..300 {
            let lam = [0, 1, 7, 30, 100][rng.next(5) as usize];
            let pair_cells = [0, 7, 64, 3000, 100_000, 1 << 22][rng.next(6) as usize];
            let group_lanes = crate::launch::GROUP_LANES[rng.next(5) as usize];
            let band_bytes = [0, 400, 32_768, 98_304][rng.next(4) as usize];
            let mut s = dup_cfg(group_lanes, band_bytes, pair_cells);
            s.pairs_per_chunk = [1, 5, 64, 1 << 22][rng.next(4) as usize];
            s.cells_per_launch = [1, 5000, 1 << 34][rng.next(3) as usize];
            s.threads_per_block = 32 * (1 + rng.next(8) as u32);
            let geometry = DupGeometry {
                threads_per_block: u64::from(s.threads_per_block),
                resident_threads: 32 * (1 + rng.next(8)),
                limits: DupLimits {
                    s_max: [48 * 1024, 99 * 1024][rng.next(2) as usize],
                    s_block: [0, 4000][rng.next(2) as usize],
                    codes: [DUP_CODES, DUP_CODES_STATES][rng.next(2) as usize],
                },
                slot_bytes: 8 * (1 + rng.next(1 << 16)),
            };
            let width = [DupWidth::Narrow, DupWidth::Wide, DupWidth::Detail][rng.next(3) as usize];
            let label = format!("round {round}: Λ={lam} {width:?} {s:?} {geometry:?}");
            let lens = |rng: &mut Rng, count: u64| -> Vec<u64> {
                (0..count)
                    .map(|_| [0, 1, 2, 3, 16, 17, 32, 33, 64, 65, 120, 200][rng.next(12) as usize])
                    .collect()
            };
            let n_a = 1 + rng.next(6);
            let a = lens(&mut rng, n_a);
            let n_b = 1 + rng.next(6);
            let b = lens(&mut rng, n_b);
            let zip_b = lens(&mut rng, n_a);
            let run = |work: &WorkSource, a: &[u64], b: &[u64], implicit: bool| {
                dup_check(
                    &label,
                    &s,
                    &geometry,
                    lam,
                    width,
                    implicit,
                    (a, b),
                    &range(work),
                );
            };
            run(
                &WorkSource::new(PairKind::Cross, a.len(), b.len()).unwrap(),
                &a,
                &b,
                true,
            );
            run(
                &WorkSource::new(PairKind::OneToMany, 1, b.len()).unwrap(),
                &a[..1],
                &b,
                true,
            );
            run(
                &WorkSource::new(PairKind::Zip, a.len(), a.len()).unwrap(),
                &a,
                &zip_b,
                false,
            );
            let i: Vec<usize> = (0..9).map(|_| rng.next(a.len() as u64) as usize).collect();
            let j: Vec<usize> = (0..9).map(|_| rng.next(b.len() as u64) as usize).collect();
            let index = PairKind::Index {
                i: IndexArray::from(i),
                j: IndexArray::from(j),
            };
            run(
                &WorkSource::new(index, a.len(), b.len()).unwrap(),
                &a,
                &b,
                false,
            );
            for self_pairs in [false, true] {
                run(
                    &WorkSource::new(PairKind::Within { self_pairs }, a.len(), a.len()).unwrap(),
                    &a,
                    &a,
                    true,
                );
            }
            let pairs: Vec<(usize, usize)> = (0..11)
                .map(|_| {
                    (
                        rng.next(a.len() as u64) as usize,
                        rng.next(b.len() as u64) as usize,
                    )
                })
                .collect();
            dup_check(
                &label,
                &s,
                &geometry,
                lam,
                width,
                false,
                (&a, &b),
                &Request::Pairs(&pairs),
            );
            let r0 = rng.next(a.len() as u64) as usize;
            let c0 = rng.next(b.len() as u64) as usize;
            let (rows, cols) = (a.len() - r0, b.len() - c0);
            let rect = Request::Rect {
                r0,
                c0,
                width: cols,
                end: rows * cols,
                class: None,
            };
            dup_check(&label, &s, &geometry, lam, width, true, (&a, &b), &rect);
        }
    }

    // -------------------------------------------------------------- duplex pf (K3)

    /// g, D_pf and b_sym by the §6.2-§6.4 formulas, independently of the planner.
    fn pf_spec_group(group_lanes: u64, n: u64, m: u64) -> u64 {
        if group_lanes > 0 {
            return group_lanes;
        }
        let l = n.max(m);
        if l <= 16 {
            4
        } else if l <= 32 {
            8
        } else if l <= 64 {
            16
        } else {
            32
        }
    }

    fn pf_spec_depth(lam: u64, g: u64, n: u64, m: u64) -> u64 {
        c_sum(n, m, lam) * (n * m.div_ceil(g)).max(m * n.div_ceil(g))
    }

    fn pf_spec_band(lam: u64, nc: u64, n: u64, m: u64) -> u64 {
        let b = |rows: u64, cols: u64| ((1 + nc) * rows.min(lam + 2) + 1) * cols * 16;
        b(n, m).max(b(m, n))
    }

    /// (variant or None, depth, b_sym) of one pair (§6.4).
    fn pf_spec(
        s: &Settings,
        limits: &DupLimits,
        lam: u64,
        nc: u64,
        n: u64,
        m: u64,
    ) -> (Option<usize>, u64, u64) {
        if n < 2 || m < 2 {
            return (None, 0, 0);
        }
        let g = pf_spec_group(s.group_lanes, n, m);
        let depth = pf_spec_depth(lam, g, n, m);
        if depth > s.pair_cells {
            return (None, depth, 0);
        }
        let band = pf_spec_band(lam, nc, n, m);
        let shared =
            band <= s.band_bytes && (32 / g) * (band + DUP_CODES) + limits.s_block <= limits.s_max;
        let k = DUP_GROUPS.iter().position(|&x| x == g).unwrap();
        (Some(2 * k + usize::from(!shared)), depth, band)
    }

    /// The worked numbers of duplex-pf-cuda.md §6.3, §6.4 and §8 at Λ = 30.
    #[test]
    fn pf_depths_and_bands_follow_the_spec() {
        let s = dup_cfg(0, 32_768, 1 << 22);
        let depth = |n: u64| pf_depth(&s, 30, n, n);
        assert_eq!(depth(20), 20_400);
        assert_eq!(depth(60), 119_040);
        assert_eq!(depth(128), 253_952);
        assert_eq!(depth(176), 523_776);
        assert_eq!(depth(200), 694_400);
        assert_eq!(pf_depth(&dup_cfg(4, 0, 0), 30, 200, 200), 4_960_000);
        // §8 watchdog pairs (b) and (c).
        assert_eq!(pf_depth(&dup_cfg(4, 0, 0), 30, 128, 128), 2_031_616);
        assert_eq!(pf_depth(&dup_cfg(32, 0, 0), 30, 94, 94), 139_872);
        assert_eq!(pf_depth(&dup_cfg(32, 0, 0), 30, 88, 88), 130_944);
        assert_eq!(pf_band_sym(30, 1, 94, 94), 97_760);
        assert_eq!(pf_band_sym(30, 1, 88, 88), 91_520);
        assert_eq!(pf_band_sym(30, 1, 20, 20), 13_120);
        assert_eq!(pf_band_sym(30, 1, 30, 30), 29_280);
        assert_eq!(pf_band_sym(30, 1, 60, 20), 39_360);
        assert_eq!(pf_band_sym(30, 1, 20, 60), 39_360);
        assert_eq!(pf_band_sym(30, 1, 200, 200), 208_000);
        assert_eq!(pf_band_sym(30, 1, 60, 5), 10_560);
        // Three loop classes (nndb2024-dna.md §5.4.8): ((1 + 3)·32 + 1)·200·16.
        assert_eq!(pf_band_sym(30, 3, 200, 200), 412_800);
        assert_eq!(pf_band(30, 3, 20, 60), ((1 + 3) * 20 + 1) * 60 * 16);
        assert_eq!(pf_group(&s, 60, 5), 16);
        // Closed form: depth 0, never routed, even at pair_cells 0.
        let zero = dup_cfg(0, 32_768, 0);
        for (n, m) in [(0, 0), (1, 200), (200, 1), (0, 5)] {
            assert_eq!(pf_depth(&zero, 30, n, m), 0);
            assert!(!pf_routed(&zero, 30, n, m));
            assert_eq!(pf_tier(&zero, &SM120, 30, 1, n, m), DupTier::Closed);
        }
        assert!(pf_routed(&zero, 30, 2, 2));
        // Symmetry and the formulas over a grid of lengths, Λ and g.
        for lam in [0u64, 1, 2, 5, 30, 100] {
            for group_lanes in crate::launch::GROUP_LANES {
                let s = dup_cfg(group_lanes, 32_768, 1 << 22);
                for n in [0u64, 1, 2, 3, 4, 16, 17, 32, 33, 64, 65, 100, 200] {
                    for m in [0u64, 1, 2, 3, 5, 16, 17, 31, 64, 65, 199] {
                        let g = pf_spec_group(group_lanes, n, m);
                        assert_eq!(pf_group(&s, n, m), g);
                        assert_eq!(pf_group(&s, n, m), pf_group(&s, m, n));
                        let want = if n < 2 || m < 2 {
                            0
                        } else {
                            pf_spec_depth(lam, g, n, m)
                        };
                        assert_eq!(pf_depth(&s, lam, n, m), want, "Λ={lam} g={g} {n}x{m}");
                        assert_eq!(pf_depth(&s, lam, n, m), pf_depth(&s, lam, m, n));
                        for nc in [1, 3] {
                            assert_eq!(pf_band_sym(lam, nc, n, m), pf_spec_band(lam, nc, n, m));
                        }
                        assert_eq!(
                            pf_tier(&s, &SM120, lam, 1, n, m),
                            pf_tier(&s, &SM120, lam, 1, m, n)
                        );
                    }
                }
            }
        }
    }

    /// Routing at min(pair_cells, CAP_pf) and the tier predicate (§6.3, §6.4).
    #[test]
    fn pf_routing_and_tiers_follow_the_spec() {
        // At an effective cap of 2^19, 176 x 176 stays on the device and 177 x 177 is
        // routed (§6.4 item 3).
        let at = dup_cfg(0, 32_768, 1 << 19);
        assert!(!pf_routed(&at, 30, 176, 176));
        assert!(pf_routed(&at, 30, 177, 177));
        // An override above CAP_pf acts as CAP_pf; below it, it is used as given.
        assert_eq!(pf_settings(&dup_cfg(0, 0, 1 << 40)).pair_cells, PF_CAP);
        assert_eq!(pf_settings(&dup_cfg(0, 0, 1 << 22)).pair_cells, PF_CAP);
        assert_eq!(pf_settings(&dup_cfg(0, 0, 7)).pair_cells, 7);
        assert_eq!(pf_settings(&dup_cfg(0, 0, 0)).pair_cells, 0);
        assert!(PF_CAP.is_power_of_two() && PF_CAP <= 1 << 21);
        // §6.3: at Λ = 30 and the defaults (g = 8), square pairs up to 27 nt are
        // shared (staging included), from 28 nt on global.
        let s = dup_cfg(0, 32_768, 1 << 22);
        assert_eq!(pf_tier(&s, &SM120, 30, 1, 20, 20), DupTier::Shared);
        assert_eq!(pf_tier(&s, &SM120, 30, 1, 26, 26), DupTier::Shared);
        assert_eq!(pf_tier(&s, &SM120, 30, 1, 27, 27), DupTier::Shared);
        assert_eq!(pf_tier(&s, &SM120, 30, 1, 28, 28), DupTier::Global);
        assert_eq!(pf_tier(&s, &SM120, 30, 1, 60, 5), DupTier::Shared);
        assert_eq!(pf_tier(&s, &SM120, 30, 1, 5, 60), DupTier::Shared);
        // band_bytes 0 sends every device pair to the global tier.
        let zero = dup_cfg(0, 0, 1 << 22);
        assert_eq!(pf_tier(&zero, &SM120, 30, 1, 2, 2), DupTier::Global);
        // Watchdog pair (c): 94 x 94 with g = 32 fits a block with up to 3,216 B of
        // static shared memory, 88 x 88 up to 9,456 B.
        let c = dup_cfg(32, 98_304, 1 << 21);
        for (s_block, n94, n88) in [
            (0, DupTier::Shared, DupTier::Shared),
            (3_216, DupTier::Shared, DupTier::Shared),
            (3_217, DupTier::Global, DupTier::Shared),
            (9_456, DupTier::Global, DupTier::Shared),
            (9_457, DupTier::Global, DupTier::Global),
        ] {
            let limits = DupLimits {
                s_max: 99 * 1024,
                s_block,
                codes: DUP_CODES,
            };
            assert_eq!(
                pf_tier(&c, &limits, 30, 1, 94, 94),
                n94,
                "s_block {s_block}"
            );
            assert_eq!(
                pf_tier(&c, &limits, 30, 1, 88, 88),
                n88,
                "s_block {s_block}"
            );
        }
        // Against the independent predicate.
        let mut rng = Rng(0x51ed_270b_a3c1_9e57);
        for _ in 0..20_000 {
            let lam = [0, 1, 7, 30, 100, 396][rng.next(6) as usize];
            let group_lanes = crate::launch::GROUP_LANES[rng.next(5) as usize];
            let band_bytes = [0, 400, 13_120, 32_768, 98_304][rng.next(5) as usize];
            let pair_cells = [0, 1, 1000, 1 << 19, 1 << 21][rng.next(5) as usize];
            let s = dup_cfg(group_lanes, band_bytes, pair_cells);
            let limits = DupLimits {
                s_max: [48 * 1024, 99 * 1024][rng.next(2) as usize],
                s_block: [0, 4000][rng.next(2) as usize],
                codes: DUP_CODES,
            };
            let (n, m) = (rng.next(201), rng.next(201));
            let nc = [1, 3][rng.next(2) as usize];
            let (variant, _, _) = pf_spec(&s, &limits, lam, nc, n, m);
            let tier = pf_tier(&s, &limits, lam, nc, n, m);
            let want = match variant {
                None if n < 2 || m < 2 => DupTier::Closed,
                None => DupTier::Host,
                Some(v) if v % 2 == 0 => DupTier::Shared,
                Some(_) => DupTier::Global,
            };
            assert_eq!(tier, want, "Λ={lam} {s:?} {limits:?} {n}x{m}");
        }
    }

    /// The per-unit rule (§8): max(1, ⌊min(pair_cells, CAP_pf)/D⌋) items per group.
    #[test]
    fn pf_per_unit_rule_follows_the_spec() {
        let geometry = dup_geometry(256, 1024, SLOT_BYTES);
        for (pair_cells, depth_n, want) in [
            (1u64 << 19, 20u64, (1u64 << 19) / 20_400),
            (1 << 19, 176, 1),
            (1 << 22, 20, PF_CAP / 20_400),
            (30_000, 20, 1),
            (40_800, 20, 2),
        ] {
            let s = pf_settings(&dup_cfg(0, 32_768, pair_cells));
            let a = vec![depth_n; 3];
            let b = vec![depth_n; 2];
            let (a_off, b_off) = (offsets(&a), offsets(&b));
            let profile = PfProfile::new(&b_off, 30, 1).unwrap();
            let lengths = PfLengths {
                a_off: &a_off,
                b_off: &b_off,
                profile: &profile,
            };
            let work = WorkSource::new(PairKind::Cross, a.len(), b.len()).unwrap();
            let plan = pf_plan_chunk(&lengths, &s, &geometry, &range(&work), 0);
            let launches = pf_launches(&s, &geometry, &plan);
            assert_eq!(launches.len(), 1);
            assert_eq!(
                launches[0].per_unit(&s, &plan),
                want.max(1),
                "{pair_cells} {depth_n}"
            );
        }
    }

    /// Plans every K3 chunk of `request` and simulates each launch's grid-stride
    /// assignment, as `dup_check` does for K1.
    #[allow(clippy::too_many_arguments)]
    fn pf_check(
        label: &str,
        s: &Settings,
        geometry: &DupGeometry,
        lam: u64,
        nc: u64,
        implicit: bool,
        (a, b): (&[u64], &[u64]),
        request: &Request<'_>,
    ) -> Vec<(usize, Vec<(usize, u64)>)> {
        let (a_off, b_off) = (offsets(a), offsets(b));
        let profile = PfProfile::new(&b_off, lam, nc).unwrap();
        let lengths = PfLengths {
            a_off: &a_off,
            b_off: &b_off,
            profile: &profile,
        };
        let limits = &geometry.limits;
        let cost = |i: usize, j: usize| {
            let (n, m) = (a[i], b[j]);
            if implicit {
                if n < 2 {
                    0
                } else {
                    n * m * dup_c_inf(m, lam)
                }
            } else {
                n * m * c_sum(n, m, lam)
            }
        };
        let mut chunks = Vec::new();
        let mut pos = 0;
        while pos < request.end() {
            let plan = pf_plan_chunk(&lengths, s, geometry, request, pos);
            assert!(plan.n >= 1 && pos + plan.n <= request.end(), "{label}");
            assert!(plan.n as u64 <= s.pairs_per_chunk, "{label}");
            let total: u64 = (0..plan.n)
                .map(|t| {
                    let (i, j) = item(request, pos + t);
                    cost(i, j)
                })
                .sum();
            let first = {
                let (i, j) = item(request, pos);
                cost(i, j)
            };
            assert!(
                total <= s.cells_per_launch.max(first),
                "{label}: chunk cost {total}"
            );
            let launches = pf_launches(s, geometry, &plan);
            let mut units: Vec<Vec<u64>> = launches
                .iter()
                .map(|launch| {
                    let blocks = launch.blocks(s, &plan);
                    assert!((1..=MAX_GRID).contains(&blocks), "{label}");
                    assert_eq!(launch.per_block % (32 / launch.g), 0, "{label}");
                    assert!(
                        launch.threads() <= geometry.threads_per_block.max(32),
                        "{label}"
                    );
                    assert_eq!(launch.stride % PF_VALUE, 0, "{label}: {launch:?}");
                    if launch.global {
                        assert_eq!(launch.slot % PF_VALUE, 0, "{label}: {launch:?}");
                        assert!(
                            blocks == 1
                                || blocks * launch.per_block * launch.slot <= geometry.slot_bytes,
                            "{label}: {launch:?} exceeds the slot budget"
                        );
                    } else {
                        assert!(
                            launch.shared_bytes() + limits.s_block <= limits.s_max,
                            "{label}"
                        );
                    }
                    vec![0; (blocks * launch.per_block) as usize]
                })
                .collect();
            for t in 0..plan.n {
                let (i, j) = item(request, pos + t);
                let (n, m) = (a[i], b[j]);
                let (variant, depth, band) = pf_spec(s, limits, lam, nc, n, m);
                let serving: Vec<usize> = (0..launches.len())
                    .filter(|&k| Some(launches[k].variant) == variant)
                    .collect();
                if variant.is_none() {
                    assert!(serving.is_empty(), "{label}");
                    continue;
                }
                assert_eq!(serving.len(), 1, "{label}: pair (n={n}, m={m})");
                let launch = &launches[serving[0]];
                if launch.global {
                    assert!(
                        launch.slot >= band,
                        "{label}: slot of {launch:?} for {band}"
                    );
                } else {
                    assert!(
                        launch.stride >= band + limits.codes,
                        "{label}: stride of {launch:?} for {band}"
                    );
                }
                let unit = &mut units[serving[0]];
                let slot = t % unit.len();
                unit[slot] += depth;
            }
            for (launch, unit) in launches.iter().zip(&units) {
                let worst = unit.iter().copied().max().unwrap_or(0);
                assert!(
                    worst <= s.pair_cells,
                    "{label}: chunk at {pos}, {launch:?}: a group runs {worst} candidates"
                );
            }
            let grids = launches
                .iter()
                .map(|launch| (launch.variant, launch.blocks(s, &plan)))
                .collect();
            chunks.push((plan.n, grids));
            pos += plan.n;
        }
        chunks
    }

    #[test]
    fn pf_random_plans_partition_cover_and_bound_every_unit() {
        let mut rng = Rng(0x2545_f491_4f6c_dd1d);
        for round in 0..300 {
            let lam = [0, 1, 7, 30, 100][rng.next(5) as usize];
            let pair_cells = [0, 7, 64, 3000, 100_000, 1 << 19, 1 << 22][rng.next(7) as usize];
            let group_lanes = crate::launch::GROUP_LANES[rng.next(5) as usize];
            let band_bytes = [0, 400, 32_768, 98_304][rng.next(4) as usize];
            let mut s = pf_settings(&dup_cfg(group_lanes, band_bytes, pair_cells));
            s.pairs_per_chunk = [1, 5, 64, 1 << 22][rng.next(4) as usize];
            s.cells_per_launch = [1, 5000, 1 << 34][rng.next(3) as usize];
            s.threads_per_block = 32 * (1 + rng.next(8) as u32);
            let geometry = DupGeometry {
                threads_per_block: u64::from(s.threads_per_block),
                resident_threads: 32 * (1 + rng.next(8)),
                limits: DupLimits {
                    s_max: [48 * 1024, 99 * 1024][rng.next(2) as usize],
                    s_block: [0, 4000][rng.next(2) as usize],
                    codes: DUP_CODES,
                },
                slot_bytes: 16 * (1 + rng.next(1 << 16)),
            };
            let nc = [1, 3][rng.next(2) as usize];
            let label = format!("round {round}: Λ={lam} N_C={nc} {s:?} {geometry:?}");
            let lens = |rng: &mut Rng, count: u64| -> Vec<u64> {
                (0..count)
                    .map(|_| [0, 1, 2, 3, 16, 17, 32, 33, 64, 65, 120, 200][rng.next(12) as usize])
                    .collect()
            };
            let n_a = 1 + rng.next(6);
            let a = lens(&mut rng, n_a);
            let n_b = 1 + rng.next(6);
            let b = lens(&mut rng, n_b);
            let zip_b = lens(&mut rng, n_a);
            let run = |work: &WorkSource, a: &[u64], b: &[u64], implicit: bool| {
                pf_check(
                    &label,
                    &s,
                    &geometry,
                    lam,
                    nc,
                    implicit,
                    (a, b),
                    &range(work),
                );
            };
            run(
                &WorkSource::new(PairKind::Cross, a.len(), b.len()).unwrap(),
                &a,
                &b,
                true,
            );
            run(
                &WorkSource::new(PairKind::OneToMany, 1, b.len()).unwrap(),
                &a[..1],
                &b,
                true,
            );
            run(
                &WorkSource::new(PairKind::Zip, a.len(), a.len()).unwrap(),
                &a,
                &zip_b,
                false,
            );
            let i: Vec<usize> = (0..9).map(|_| rng.next(a.len() as u64) as usize).collect();
            let j: Vec<usize> = (0..9).map(|_| rng.next(b.len() as u64) as usize).collect();
            let index = PairKind::Index {
                i: IndexArray::from(i),
                j: IndexArray::from(j),
            };
            run(
                &WorkSource::new(index, a.len(), b.len()).unwrap(),
                &a,
                &b,
                false,
            );
            for self_pairs in [false, true] {
                run(
                    &WorkSource::new(PairKind::Within { self_pairs }, a.len(), a.len()).unwrap(),
                    &a,
                    &a,
                    true,
                );
            }
            let pairs: Vec<(usize, usize)> = (0..11)
                .map(|_| {
                    (
                        rng.next(a.len() as u64) as usize,
                        rng.next(b.len() as u64) as usize,
                    )
                })
                .collect();
            pf_check(
                &label,
                &s,
                &geometry,
                lam,
                nc,
                false,
                (&a, &b),
                &Request::Pairs(&pairs),
            );
            let r0 = rng.next(a.len() as u64) as usize;
            let c0 = rng.next(b.len() as u64) as usize;
            let (rows, cols) = (a.len() - r0, b.len() - c0);
            let rect = Request::Rect {
                r0,
                c0,
                width: cols,
                end: rows * cols,
                class: None,
            };
            pf_check(&label, &s, &geometry, lam, nc, true, (&a, &b), &rect);
        }
    }

    /// primer3-cuda.md §6.6 worked numbers at the provisional cap and the defaults.
    #[test]
    fn primer3_worked_numbers() {
        let s = settings(4, 0, 0);
        assert_eq!(s.pair_cells, 1 << 22);
        assert_eq!(p3_c(32, 32), 496);
        assert_eq!(p3_c(1, 60), 0);
        for (l1, l2, g, depth) in [
            (20, 20, 8, 27_700),
            (25, 25, 8, 55_775),
            (44, 44, 16, 89_452),
            (60, 60, 32, 93_120),
            (60, 61, 32, 93_180),
            (61, 60, 32, 94_174),
            (60, 96, 32, 125_100),
        ] {
            let item = p3_plan_k2(&s, l1, l2);
            assert_eq!(
                item,
                P3Item {
                    g,
                    depth,
                    routed: false
                },
                "{l1} x {l2}"
            );
        }
        for (l1, l2, depth) in [
            (60, 120, 156_360),
            (60, 1_000, 1_044_120),
            (60, 10_000, 9_963_540),
        ] {
            let item = p3_plan_k2(&s, l1, l2);
            assert!(item.routed, "{l1} x {l2}");
            assert_eq!(p3_depth_k2(l1, l2, 32), depth);
        }
        for (g, depth) in [(4, 466_920), (8, 257_760), (16, 168_120), (32, 108_360)] {
            assert_eq!(p3_depth_t2(60, g), depth);
        }
        assert_eq!(
            p3_plan_t2(&s, 60),
            P3Item {
                g: 32,
                depth: 108_360,
                routed: false
            }
        );
        assert_eq!(
            p3_plan_t2(&s, 50),
            P3Item {
                g: 16,
                depth: 112_600,
                routed: false
            }
        );
        assert_eq!(
            p3_plan_t2(&s, 30),
            P3Item {
                g: 8,
                depth: 78_480,
                routed: false
            }
        );
        // Every T2 length fits at g = 32 under the cap (no T2 routing).
        assert!((1..=60).all(|l| p3_depth_t2(l, 32) <= P3_CAP));
        // pair_cells = 0 routes every K2 pair and only raises T2's g.
        let zero = Settings { pair_cells: 0, ..s };
        assert!(p3_plan_k2(&zero, 1, 1).routed);
        assert_eq!(p3_plan_t2(&zero, 5).g, 32);
        // A group_lanes below the floor acts as the floor.
        let four = Settings {
            group_lanes: 4,
            ..s
        };
        assert_eq!(p3_plan_k2(&four, 60, 60).g, 32);
        assert_eq!(p3_plan_k2(&four, 20, 20).g, 4);
    }

    /// The launch split keeps the per-unit rule and the slot budget.
    #[test]
    fn primer3_launches_keep_the_rules() {
        let s = settings(4, 0, 0);
        for (count, g, threads, slot, depth) in [
            (1usize, 4u64, 256u64, 1_000u64, 10u64),
            (100_000, 32, 256, 59_536, 108_360),
            (5_000_000, 4, 1024, 1_100, 2_000),
            (7, 16, 32, 320_032, 50_000),
        ] {
            let launches = p3_launches(&s, count, g, threads, slot, depth);
            let per_unit = (P3_CAP / depth).max(1);
            let mut next = 0;
            for launch in &launches {
                assert_eq!(launch.start, next);
                next += launch.len;
                let groups = launch.blocks * launch.per_block;
                assert!(groups >= 1 && launch.blocks <= MAX_GRID);
                assert!((launch.len as u64).div_ceil(groups) <= per_unit);
                assert!(groups * launch.slot <= SLOT_BYTES.max(launch.per_block * launch.slot));
                assert!(launch.per_block * g <= threads.max(32));
            }
            assert_eq!(next, count);
        }
    }
}
