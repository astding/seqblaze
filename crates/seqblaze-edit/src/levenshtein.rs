//! Unit-cost Levenshtein distance in global, prefix and infix modes.
//! Spec: docs/wiki/edit/levenshtein.md.
//!
//! Scores use Myers' (1999) bit-vector algorithm in Hyyrö's (2003) multi-word
//! formulation, all integer. Bounded calls add Ukkonen's (1985) band and the
//! last-active-block cut-off; they only ever certify "greater than the cutoff"
//! (spec §5), so bounded and full outputs are byte-identical. `cigar` follows
//! rule E (spec §4) over stored vertical/horizontal delta vectors.
use std::cell::RefCell;

use seqblaze_core::{
    detail::{
        allocation_error, try_reserve_exact, DetailColumns, DetailField, DetailRequest, FieldKind,
    },
    kernel::PairDistanceKernel,
    plan::BoundAvailability,
    sched::Workers,
    seq::{ByteView, Composition, SeqTable},
    validate::EngineError,
    work::WorkSource,
};

mod lanes;
pub use lanes::{
    isa as lane_isa, set_isa as set_lane_isa, unavailable as lane_isa_unavailable, Isa as LaneIsa,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    Global,
    Prefix,
    Infix,
}

impl Mode {
    pub fn parse(name: &str) -> Option<Self> {
        match name {
            "global" => Some(Self::Global),
            "prefix" => Some(Self::Prefix),
            "infix" => Some(Self::Infix),
            _ => None,
        }
    }
}

/// Detail field indices (D48 §3 order after the primary `dist`).
pub const B_END: usize = 0;
pub const N_ENDS: usize = 1;
pub const B_START: usize = 2;
pub const N_STARTS: usize = 3;
pub const ENDS: usize = 4;
pub const CIGAR: usize = 5;

pub static FIELDS: [DetailField; 6] = [
    DetailField {
        name: "b_end",
        kind: FieldKind::Int64,
    },
    DetailField {
        name: "n_ends",
        kind: FieldKind::Int32,
    },
    DetailField {
        name: "b_start",
        kind: FieldKind::Int64,
    },
    DetailField {
        name: "n_starts",
        kind: FieldKind::Int32,
    },
    DetailField {
        name: "ends",
        kind: FieldKind::ListInt64,
    },
    DetailField {
        name: "cigar",
        kind: FieldKind::Utf8,
    },
];

/// Every sequence: at most 2**31 - 2 bytes, so `dist` <= max(m, n),
/// `n_ends` <= n + 1 and `n_starts` <= b_end + 1 fit int32 (spec §8, §3.1).
pub const MAX_LENGTH: usize = i32::MAX as usize - 1;
/// With `cigar`: every sequence of a, and of b in global mode (spec §8).
pub const CIGAR_MAX_LENGTH: usize = 16_384;

/// One call's configuration: mode and requested detail fields.
#[derive(Debug, Clone, Copy)]
pub struct Levenshtein {
    mode: Mode,
    request: DetailRequest,
}

impl Levenshtein {
    /// `fields` are detail-field names (not `dist`); a field of another mode
    /// (anything but `cigar` in global mode) is a ParameterError.
    pub fn new(mode: Mode, fields: &[&str]) -> Result<Self, EngineError> {
        let mut request = DetailRequest::NONE;
        for &name in fields {
            let index = FIELDS
                .iter()
                .position(|field| field.name == name)
                .ok_or_else(|| {
                    EngineError::Parameter(format!("unknown levenshtein field {name:?}"))
                })?;
            if mode == Mode::Global && index != CIGAR {
                return Err(EngineError::Parameter(format!(
                    "field {name:?} is a prefix/infix field; mode=\"global\" offers dist and cigar"
                )));
            }
            request = request.with(index);
        }
        Ok(Self { mode, request })
    }
    pub fn mode(&self) -> Mode {
        self.mode
    }
}

fn thousands(value: usize) -> String {
    let digits = value.to_string();
    let mut out = String::with_capacity(digits.len() + digits.len() / 3);
    for (k, ch) in digits.chars().enumerate() {
        if k > 0 && (digits.len() - k) % 3 == 0 {
            out.push(',');
        }
        out.push(ch);
    }
    out
}

impl PairDistanceKernel for Levenshtein {
    fn bound_availability(&self) -> BoundAvailability {
        BoundAvailability::Available
    }
    fn name(&self) -> &'static str {
        "levenshtein"
    }
    /// Any lengths pair; only structural length metadata is checked here.
    fn validate_pairs(
        &self,
        a: &SeqTable,
        b: &SeqTable,
        _work: &WorkSource,
    ) -> Result<(), EngineError> {
        a.validate_lengths()?;
        if !std::ptr::eq(a, b) {
            b.validate_lengths()?;
        }
        Ok(())
    }
    fn sequence_length_limit(&self) -> Option<usize> {
        Some(MAX_LENGTH)
    }
    fn length_caps(&self) -> [Option<usize>; 2] {
        if self.request.contains(CIGAR) {
            let b = if self.mode == Mode::Global {
                CIGAR_MAX_LENGTH
            } else {
                MAX_LENGTH
            };
            [Some(CIGAR_MAX_LENGTH), Some(b)]
        } else {
            [Some(MAX_LENGTH), Some(MAX_LENGTH)]
        }
    }
    fn length_error(
        &self,
        input: &'static str,
        index: usize,
        length: usize,
        limit: usize,
    ) -> EngineError {
        let message = if limit == CIGAR_MAX_LENGTH {
            format!(
                "levenshtein(fields=[..., \"cigar\"]): sequence {index} of input {input} has length {}; \
                 with cigar the limit is {} per sequence of a (and of b in global mode). Nothing was computed. \
                 Omit cigar, or move the long sequences to a separate call.",
                thousands(length),
                thousands(limit)
            )
        } else {
            format!(
                "levenshtein: sequence {index} of input {input} has length {}; the limit is {} bytes per \
                 sequence. Nothing was computed.",
                thousands(length),
                thousands(limit)
            )
        };
        EngineError::Sequence {
            message,
            input,
            index,
            position: None,
            byte: None,
            n_bad: 1,
            bad_indices: vec![index],
        }
    }
    fn prepare_bound(
        &self,
        a: &SeqTable,
        b: &SeqTable,
        workers: Workers,
    ) -> Result<(), EngineError> {
        a.prepare_composition(workers)?;
        if !std::ptr::eq(a, b) {
            b.prepare_composition(workers)?;
        }
        Ok(())
    }
    /// Scheduling estimate from the mean lengths m (a) and n (b): the forward
    /// pass takes ⌈m/64⌉ word steps per column of b. With details, infix
    /// b_start or n_starts (also needed by an infix cigar) adds a reverse pass over up to
    /// b_end ≤ n columns, and cigar adds its stored pass over the window (two
    /// words per step) plus the walk of m + |w| moves; the window is b in
    /// global mode and about m long otherwise. Never changes an output.
    fn pair_cost(&self, a: &SeqTable, b: &SeqTable, with_details: bool) -> usize {
        let (m, n) = (a.mean_len(), b.mean_len());
        let words = m.div_ceil(64);
        let mut cost = words.saturating_mul(n);
        if with_details {
            let window = if self.mode == Mode::Global {
                n
            } else {
                m.min(n)
            };
            let cigar = self.request.contains(CIGAR);
            if self.mode == Mode::Infix
                && (cigar || self.request.contains(B_START) || self.request.contains(N_STARTS))
            {
                cost = cost.saturating_add(words.saturating_mul(n));
            }
            if cigar {
                cost = cost
                    .saturating_add(words.saturating_mul(window).saturating_mul(2))
                    .saturating_add(m.saturating_add(window));
            }
        }
        cost
    }
    /// Spec §5: max(P, N) in global mode, P in prefix and infix.
    fn lower_bound(&self, a: &SeqTable, b: &SeqTable, i: usize, j: usize) -> i32 {
        let (Some(left), Some(right)) = (a.composition(), b.composition()) else {
            return 0; // Unprepared: the trivial bound never rejects a pair.
        };
        let (p, n) = match (left, right) {
            (Composition::Acgt(x), Composition::Acgt(y)) => excess(&x[i], &y[j]),
            (Composition::Acgt(x), Composition::Bytes(y)) => {
                let (other, acgt) = split(&y[j], b.seq_len(j));
                let (p, n) = excess(&x[i], &acgt);
                (p, n + other)
            }
            (Composition::Bytes(x), Composition::Acgt(y)) => {
                let (other, acgt) = split(&x[i], a.seq_len(i));
                let (p, n) = excess(&acgt, &y[j]);
                (p + other, n)
            }
            (Composition::Bytes(x), Composition::Bytes(y)) => excess(&x[i], &y[j]),
        };
        let bound = if self.mode == Mode::Global {
            p.max(n)
        } else {
            p
        };
        // Lowering a lower bound keeps it valid, so saturate rather than wrap.
        i32::try_from(bound).unwrap_or(i32::MAX)
    }
    fn evaluate(
        &self,
        a: ByteView<'_>,
        b: ByteView<'_>,
        cutoff: Option<i32>,
    ) -> Result<Option<i32>, EngineError> {
        with_inputs(a, b, |a, b, work| {
            let (a, b) = strip_shared(a, b, self.mode);
            let forward = work.forward(a, b, self.mode, cutoff, false)?;
            Ok(forward.map(|forward| forward.dist as i32))
        })
    }
    /// Batches of 512 pairs for the lane groups (spec §12, "Lane score
    /// pass"); 1 when the testing hook forces the scalar path.
    fn batch_width(&self) -> usize {
        lanes::batch_width()
    }
    /// The lane score pass: each entry equals `evaluate`'s (`lanes.rs`).
    fn evaluate_many(
        &self,
        pairs: &[(ByteView<'_>, ByteView<'_>, Option<i32>)],
        out: &mut Vec<Option<i32>>,
    ) -> Result<(), EngineError> {
        lanes::score_many(self, pairs, out)
    }
    /// Rectangles of 4,096 cells for the shared-pattern lane groups (spec
    /// §12, "Shared-pattern lane form"), for a bounded pass only when a
    /// pattern side's rows fit a lane; 0 when the testing hook forces the
    /// scalar path. `evaluate`'s cutoff test is exact (spec §5): a score at
    /// most the cutoff, else None.
    fn rect_cells(&self, a: &SeqTable, b: &SeqTable, bounded: bool) -> usize {
        lanes::rect_cells(self.mode, a, b, bounded)
    }
    /// The shared-pattern lane groups of a rectangle: each entry equals
    /// `evaluate`'s (`lanes.rs`).
    fn evaluate_rect(
        &self,
        rows: &[ByteView<'_>],
        cols: &[ByteView<'_>],
        cutoff: Option<i32>,
        out: &mut Vec<Option<i32>>,
    ) -> Result<(), EngineError> {
        lanes::score_rect(self, rows, cols, cutoff, out)
    }
    fn detail_fields(&self) -> &'static [DetailField] {
        &FIELDS
    }
    fn detail_request(&self) -> DetailRequest {
        self.request
    }
    /// Prefix and infix details reuse the score's forward pass; a global
    /// pair's only detail (cigar) never did.
    fn fused_details(&self) -> bool {
        self.mode != Mode::Global
    }
    /// One forward pass gives dist, b_end, n_ends and ends; b_start, n_starts
    /// and cigar take their own passes, only for kept pairs.
    fn evaluate_detailed(
        &self,
        a: ByteView<'_>,
        b: ByteView<'_>,
        prune: Option<i32>,
        keep: Option<i32>,
        out: &mut DetailColumns,
    ) -> Result<Option<i32>, EngineError> {
        with_inputs(a, b, |a, b, work| {
            let Some(forward) = work.forward(a, b, self.mode, prune, out.wants(ENDS))? else {
                return Ok(None);
            };
            let dist = forward.dist as i32;
            if keep.is_none_or(|limit| dist <= limit) {
                work.write(self.mode, a, b, &forward, out)?;
            }
            Ok(Some(dist))
        })
    }
    fn details(
        &self,
        a: ByteView<'_>,
        b: ByteView<'_>,
        score: i32,
        out: &mut DetailColumns,
    ) -> Result<(), EngineError> {
        with_inputs(a, b, |a, b, work| {
            let forward = if self.mode == Mode::Global {
                // The only global detail is cigar, over the whole of b.
                Forward {
                    dist: i64::from(score),
                    b_end: b.len(),
                    n_ends: 1,
                    last_end: b.len(),
                }
            } else {
                // The exact score bounds the band; every value needed is <= score.
                work.forward(a, b, self.mode, Some(score), out.wants(ENDS))?
                    .expect("a kept pair's exact score is within its own cutoff")
            };
            debug_assert_eq!(forward.dist, i64::from(score));
            work.write(self.mode, a, b, &forward, out)
        })
    }
}

/// The score-only pair with the bytes no alignment needs removed (spec §12):
/// the common prefix and suffix of a and b in global mode, the common prefix
/// in prefix mode, nothing in infix mode. `dist` is unchanged, so this serves
/// `evaluate` only; detail fields always use the whole pair.
fn strip_shared<'x>(a: &'x [u8], b: &'x [u8], mode: Mode) -> (&'x [u8], &'x [u8]) {
    if mode == Mode::Infix {
        return (a, b);
    }
    let head = common_prefix(a, b);
    let (a, b) = (&a[head..], &b[head..]);
    if mode == Mode::Prefix {
        return (a, b);
    }
    let tail = common_suffix(a, b);
    (&a[..a.len() - tail], &b[..b.len() - tail])
}

/// `evaluate`'s forward pass on a pair already read and reduced by
/// `strip_shared`, with this thread's work buffers (the lane pass's long
/// patterns).
fn score_reduced(
    a: &[u8],
    b: &[u8],
    mode: Mode,
    cutoff: Option<i32>,
) -> Result<Option<i32>, EngineError> {
    SCRATCH.with(|cell| {
        let mut guard = cell.borrow_mut();
        let result = guard.work.forward(a, b, mode, cutoff, false);
        guard.shrink();
        Ok(result?.map(|forward| forward.dist as i32))
    })
}

/// Length of the longest common prefix, eight bytes per comparison.
fn common_prefix(a: &[u8], b: &[u8]) -> usize {
    let n = a.len().min(b.len());
    let mut k = 0;
    while k + 8 <= n {
        let x = word(&a[k..k + 8]) ^ word(&b[k..k + 8]);
        if x != 0 {
            // Little-endian words: the lowest set bit is in the first differing byte.
            return k + (x.trailing_zeros() / 8) as usize;
        }
        k += 8;
    }
    while k < n && a[k] == b[k] {
        k += 1;
    }
    k
}

/// Length of the longest common suffix, eight bytes per comparison.
fn common_suffix(a: &[u8], b: &[u8]) -> usize {
    let (m, n) = (a.len(), b.len());
    let shortest = m.min(n);
    let mut k = 0;
    while k + 8 <= shortest {
        let x = word(&a[m - k - 8..m - k]) ^ word(&b[n - k - 8..n - k]);
        if x != 0 {
            // The highest set bit is in the last differing byte.
            return k + (x.leading_zeros() / 8) as usize;
        }
        k += 8;
    }
    while k < shortest && a[m - 1 - k] == b[n - 1 - k] {
        k += 1;
    }
    k
}

#[inline(always)]
fn word(bytes: &[u8]) -> u64 {
    u64::from_le_bytes(bytes.try_into().expect("eight bytes"))
}

/// The single-word match-mask table (the first 256 words of `peq`).
#[inline]
fn word_table(peq: &mut [u64]) -> &mut [u64; 256] {
    (&mut peq[..256])
        .try_into()
        .expect("the table holds 256 words per query word")
}

/// Composition excess (P, N) over equally indexed count arrays.
fn excess<const K: usize>(x: &[u32; K], y: &[u32; K]) -> (u64, u64) {
    let (mut p, mut n) = (0u64, 0u64);
    for (&cx, &cy) in x.iter().zip(y) {
        if cx > cy {
            p += u64::from(cx - cy);
        } else {
            n += u64::from(cy - cx);
        }
    }
    (p, n)
}

/// A raw row's non-ACGT total and its ACGT counts.
fn split(raw: &[u32; 256], len: usize) -> (u64, [u32; 4]) {
    let acgt = b"ACGT".map(|byte| raw[byte as usize]);
    let total: u64 = acgt.iter().map(|&c| u64::from(c)).sum();
    ((len as u64).saturating_sub(total), acgt)
}

/// Result of one forward pass. For prefix/infix, `b_end` and `last_end` are the
/// smallest and largest optimal ends and `n_ends` their count.
#[derive(Debug, Clone, Copy)]
struct Forward {
    dist: i64,
    b_end: usize,
    n_ends: usize,
    last_end: usize,
}

impl Forward {
    fn start(score: i64) -> Self {
        Self {
            dist: score,
            b_end: 0,
            n_ends: 1,
            last_end: 0,
        }
    }
    /// Column `end` has D(m, end) = `score`.
    #[inline]
    fn record(&mut self, score: i64, end: usize, ends: &mut Option<&mut Vec<i64>>) {
        if score < self.dist {
            *self = Self {
                dist: score,
                b_end: end,
                n_ends: 1,
                last_end: end,
            };
            if let Some(ends) = ends.as_mut() {
                ends.clear();
                ends.push(end as i64);
            }
        } else if score == self.dist {
            self.n_ends += 1;
            self.last_end = end;
            if let Some(ends) = ends.as_mut() {
                ends.push(end as i64);
            }
        }
    }
}

/// One Myers/Hyyrö word step for column state (pv, mv), match mask `eq` and
/// top-row delta `hin`. Returns the horizontal +1/−1 masks before the shift;
/// bit k of them is the horizontal delta at the word's row k + 1.
#[inline(always)]
fn step(pv: &mut u64, mv: &mut u64, eq: u64, hin: i32) -> (u64, u64) {
    let (p, m) = (*pv, *mv);
    let xv = eq | m;
    let eq = if hin < 0 { eq | 1 } else { eq };
    let xh = ((eq & p).wrapping_add(p) ^ p) | eq;
    let ph = m | !(xh | p);
    let mh = p & xh;
    let ph_shifted = (ph << 1) | u64::from(hin > 0);
    let mh_shifted = (mh << 1) | u64::from(hin < 0);
    *pv = mh_shifted | !(xv | ph_shifted);
    *mv = ph_shifted & xv;
    (ph, mh)
}

#[inline(always)]
fn hout(ph: u64, mh: u64) -> i32 {
    (ph >> 63) as i32 - (mh >> 63) as i32
}

#[inline(always)]
fn delta_at(ph: u64, mh: u64, bit: usize) -> i64 {
    ((ph >> bit) & 1) as i64 - ((mh >> bit) & 1) as i64
}

/// Decimal digits of `count`, appended without a temporary allocation.
fn push_count(text: &mut Vec<u8>, mut count: usize) {
    let mut digits = [0u8; 20];
    let mut start = digits.len();
    loop {
        start -= 1;
        digits[start] = b'0' + (count % 10) as u8;
        count /= 10;
        if count == 0 {
            break;
        }
    }
    text.extend_from_slice(&digits[start..]);
}

/// Per-thread buffers reused across pairs. `peq` is all zero between pairs.
#[derive(Default)]
struct Scratch {
    a: Vec<u8>,
    b: Vec<u8>,
    work: Work,
}

#[derive(Default)]
struct Work {
    peq: Vec<u64>,
    /// Set while `peq` may hold bits; a panic mid-pair leaves it set and the
    /// next pair on this thread clears the whole table first.
    peq_dirty: bool,
    pv: Vec<u64>,
    mv: Vec<u64>,
    rev: Vec<u8>,
    ends: Vec<i64>,
    trace: Vec<u64>,
    ops: Vec<u8>,
    text: Vec<u8>,
    /// Byte limit per buffer, below Rust's own (isize::MAX); None outside
    /// tests. Tests lower it to reach the allocation-failure path cheaply.
    budget: Option<usize>,
}

/// Element count `factors` multiplied, checked: an overflow is reported as an
/// allocation failure, like a size the allocator refuses.
fn checked_len(factors: &[usize], what: &str) -> Result<usize, EngineError> {
    factors
        .iter()
        .try_fold(1usize, |total, &factor| total.checked_mul(factor))
        .ok_or_else(|| allocation_error(what, "size overflows usize"))
}

/// Refuse `len` elements of `T` above the injected test budget.
fn check_budget<T>(len: usize, what: &str, budget: Option<usize>) -> Result<(), EngineError> {
    let Some(limit) = budget else {
        return Ok(());
    };
    match len.checked_mul(std::mem::size_of::<T>()) {
        Some(bytes) if bytes <= limit => Ok(()),
        _ => Err(allocation_error(
            what,
            format!("{len} elements exceed the {limit}-byte budget"),
        )),
    }
}

/// Room for `total` elements in a cleared or partly filled scratch vector,
/// reserved exactly and fallibly; later pushes up to `total` never reallocate.
fn reserve_total<T>(
    buffer: &mut Vec<T>,
    total: usize,
    what: &str,
    budget: Option<usize>,
) -> Result<(), EngineError> {
    check_budget::<T>(total, what, budget)?;
    if buffer.capacity() < total {
        try_reserve_exact(buffer, total - buffer.len(), what)?;
    }
    Ok(())
}

/// Grow `buffer` to at least `len` elements, new ones `value`, fallibly.
fn grow<T: Clone>(
    buffer: &mut Vec<T>,
    len: usize,
    value: T,
    what: &str,
    budget: Option<usize>,
) -> Result<(), EngineError> {
    if buffer.len() < len {
        reserve_total(buffer, len, what, budget)?;
        buffer.resize(len, value);
    }
    Ok(())
}

/// Buffers above this size are released after each pair, so an idle thread
/// (including the caller's own thread) does not keep a large traceback alive.
const KEEP_BYTES: usize = 1 << 20;

thread_local! {
    static SCRATCH: RefCell<Scratch> = RefCell::new(Scratch::default());
}

/// Run `body` on the pair's bytes and this thread's work buffers. Buffers are
/// released as usual on success and on error.
fn with_inputs<R>(
    a: ByteView<'_>,
    b: ByteView<'_>,
    body: impl FnOnce(&[u8], &[u8], &mut Work) -> Result<R, EngineError>,
) -> Result<R, EngineError> {
    SCRATCH.with(|cell| {
        let mut guard = cell.borrow_mut();
        let scratch = &mut *guard;
        let budget = scratch.work.budget;
        let result = match (
            bytes(a, &mut scratch.a, budget),
            bytes(b, &mut scratch.b, budget),
        ) {
            (Ok(a), Ok(b)) => body(a, b, &mut scratch.work),
            (Err(error), _) | (_, Err(error)) => Err(error),
        };
        scratch.shrink();
        result
    })
}

/// A row as a slice: immutable storage directly, borrowed NumPy rows copied
/// once (relaxed loads, whole words: `ByteView::copy_padded`) into `buffer`,
/// which keeps its length between pairs (no refill).
fn bytes<'v>(
    view: ByteView<'v>,
    buffer: &'v mut Vec<u8>,
    budget: Option<usize>,
) -> Result<&'v [u8], EngineError> {
    const WHAT: &str = "levenshtein input copy";
    if let Some(slice) = view.as_slice() {
        return Ok(slice);
    }
    let span = view
        .len()
        .checked_next_multiple_of(8)
        .ok_or_else(|| allocation_error(WHAT, "size overflows usize"))?;
    grow(buffer, span, 0, WHAT, budget)?;
    view.copy_padded(0, &mut buffer[..span]);
    Ok(&buffer[..view.len()])
}

impl Scratch {
    fn shrink(&mut self) {
        let work = &mut self.work;
        if work.peq.capacity() * 8 > KEEP_BYTES {
            // A fresh table is all zero, whatever the old one held.
            work.peq = Vec::new();
            work.peq_dirty = false;
        }
        for buffer in [&mut work.pv, &mut work.mv, &mut work.trace] {
            if buffer.capacity() * 8 > KEEP_BYTES {
                *buffer = Vec::new();
            }
        }
        if work.ends.capacity() * 8 > KEEP_BYTES {
            work.ends = Vec::new();
        }
        for buffer in [
            &mut work.ops,
            &mut work.text,
            &mut work.rev,
            &mut self.a,
            &mut self.b,
        ] {
            if buffer.capacity() > KEEP_BYTES {
                *buffer = Vec::new();
            }
        }
    }
}

impl Work {
    /// Load Peq[x] (bit i - 1 set iff a_i = x) for a, or for reversed a.
    fn load_peq(&mut self, a: &[u8], words: usize, reversed: bool) -> Result<(), EngineError> {
        if self.peq_dirty {
            self.peq.fill(0);
            self.peq_dirty = false;
        }
        const WHAT: &str = "levenshtein match-mask table";
        let len = checked_len(&[256, words], WHAT)?;
        grow(&mut self.peq, len, 0, WHAT, self.budget)?;
        self.peq_dirty = true;
        let m = a.len();
        if words == 1 {
            let table = word_table(&mut self.peq);
            for (i, &x) in a.iter().enumerate() {
                let row = if reversed { m - 1 - i } else { i };
                table[x as usize] |= 1u64 << row;
            }
            return Ok(());
        }
        for (i, &x) in a.iter().enumerate() {
            let row = if reversed { m - 1 - i } else { i };
            self.peq[x as usize * words + row / 64] |= 1u64 << (row % 64);
        }
        Ok(())
    }
    /// Clear exactly the words the matching `load_peq` set.
    fn clear_peq(&mut self, a: &[u8], words: usize, reversed: bool) {
        if words == 1 {
            let table = word_table(&mut self.peq);
            for &x in a {
                table[x as usize] = 0;
            }
        } else {
            let m = a.len();
            for (i, &x) in a.iter().enumerate() {
                let row = if reversed { m - 1 - i } else { i };
                self.peq[x as usize * words + row / 64] = 0;
            }
        }
        self.peq_dirty = false;
    }
    fn reserve_columns(&mut self, words: usize) -> Result<(), EngineError> {
        const WHAT: &str = "levenshtein column state";
        grow(&mut self.pv, words, 0, WHAT, self.budget)?;
        grow(&mut self.mv, words, 0, WHAT, self.budget)
    }

    /// Forward pass of `a` (length m) against `b`: global D(m, n), or the
    /// minimum of D(m, e) over ends e with their count. With a cutoff, None
    /// certifies dist > cutoff; every value <= cutoff is exact.
    fn forward(
        &mut self,
        a: &[u8],
        b: &[u8],
        mode: Mode,
        cutoff: Option<i32>,
        want_ends: bool,
    ) -> Result<Option<Forward>, EngineError> {
        let limit = cutoff.map(i64::from);
        if limit.is_some_and(|c| c < 0) {
            return Ok(None);
        }
        let (m, n) = (a.len(), b.len());
        if want_ends {
            // At most one end per column 0..=n; recording never reallocates.
            const WHAT: &str = "levenshtein ends";
            self.ends.clear();
            let len = checked_len(&[n, 1], WHAT)?
                .checked_add(1)
                .ok_or_else(|| allocation_error(WHAT, "size overflows usize"))?;
            reserve_total(&mut self.ends, len, WHAT, self.budget)?;
        }
        if m == 0 {
            let forward = match mode {
                Mode::Global => Forward {
                    dist: n as i64,
                    b_end: n,
                    n_ends: 1,
                    last_end: n,
                },
                Mode::Prefix => Forward::start(0),
                Mode::Infix => Forward {
                    dist: 0,
                    b_end: 0,
                    n_ends: n + 1,
                    last_end: n,
                },
            };
            if limit.is_some_and(|c| forward.dist > c) {
                return Ok(None);
            }
            if want_ends {
                let last = if mode == Mode::Infix { n } else { 0 };
                self.ends.extend(0..=last as i64);
            }
            return Ok(Some(forward));
        }
        if mode == Mode::Global && limit.is_some_and(|c| m.abs_diff(n) as i64 > c) {
            return Ok(None); // D(m, n) >= |m - n|.
        }
        let words = m.div_ceil(64);
        self.reserve_columns(words)?;
        self.load_peq(a, words, false)?;
        let result = match limit {
            None => Some(self.scan_full(words, m, b, mode, want_ends)),
            Some(c) => self.scan_banded(words, m, b, mode, c, want_ends),
        };
        self.clear_peq(a, words, false);
        Ok(result)
    }

    /// Every word of every column; no bound, band or exit (`full`).
    fn scan_full(
        &mut self,
        words: usize,
        m: usize,
        b: &[u8],
        mode: Mode,
        want_ends: bool,
    ) -> Forward {
        let Self {
            peq, pv, mv, ends, ..
        } = self;
        let (pv, mv) = (&mut pv[..words], &mut mv[..words]);
        pv.fill(!0);
        mv.fill(0);
        let last = words - 1;
        let bit = (m - 1) % 64;
        let top = if mode == Mode::Infix { 0 } else { 1 };
        let mut ends = want_ends.then_some(ends);
        if let Some(ends) = ends.as_mut() {
            ends.clear();
            ends.push(0);
        }
        let mut forward = Forward::start(m as i64);
        let mut score = m as i64;
        if words == 1 {
            // A single-word column needs no carry between words. Keep its
            // state local across columns, with the same step and end order.
            let (mut p, mut v) = (!0u64, 0u64);
            for (j, &x) in b.iter().enumerate() {
                let (ph, mh) = step(&mut p, &mut v, peq[x as usize], top);
                score += delta_at(ph, mh, bit);
                if mode != Mode::Global {
                    forward.record(score, j + 1, &mut ends);
                }
            }
            pv[0] = p;
            mv[0] = v;
        } else {
            for (j, &x) in b.iter().enumerate() {
                let eq = &peq[x as usize * words..][..words];
                let mut hin = top;
                for w in 0..last {
                    let (ph, mh) = step(&mut pv[w], &mut mv[w], eq[w], hin);
                    hin = hout(ph, mh);
                }
                let (ph, mh) = step(&mut pv[last], &mut mv[last], eq[last], hin);
                score += delta_at(ph, mh, bit);
                if mode != Mode::Global {
                    forward.record(score, j + 1, &mut ends);
                }
            }
        }
        if mode == Mode::Global {
            forward = Forward {
                dist: score,
                b_end: b.len(),
                n_ends: 1,
                last_end: b.len(),
            };
        }
        forward
    }

    /// Bounded pass with cutoff c >= 0 (spec §5 shortcuts). Words outside
    /// [first, last] are not computed:
    /// - below `last` every cell exceeds c (Myers' last active block: a word is
    ///   dropped once its bottom cell is >= c + rows, activated again, with
    ///   fresh +1 deltas, once the cell above it is <= c);
    /// - above `first` (global/prefix) every cell has D(i, j) >= j - i > c
    ///   (Ukkonen's band); the frozen boundary feeds +1 deltas.
    ///
    /// Both feed over-estimates, so computed values are >= the true ones and
    /// exact wherever the true value is <= c. Prefix/infix tighten c to the
    /// best end found so far.
    fn scan_banded(
        &mut self,
        words: usize,
        m: usize,
        b: &[u8],
        mode: Mode,
        cutoff: i64,
        want_ends: bool,
    ) -> Option<Forward> {
        let Self {
            peq, pv, mv, ends, ..
        } = self;
        let rows = |w: usize| (m - 64 * w).min(64);
        let bottom = |w: usize| 64 * w + rows(w);
        let valid = |w: usize| {
            let r = rows(w);
            if r == 64 {
                !0u64
            } else {
                (1u64 << r) - 1
            }
        };
        let top = if mode == Mode::Infix { 0 } else { 1 };
        let n = b.len();
        let mut ends = want_ends.then_some(ends);
        if let Some(ends) = ends.as_mut() {
            ends.clear();
        }
        let mut c = cutoff;
        let mut first = 0usize;
        // Column 0: D(i, 0) = i, so words whose top row is <= c are active.
        let mut last: isize = if c >= 1 {
            ((c - 1) / 64).min(words as i64 - 1) as isize
        } else {
            -1
        };
        for w in 0..=last {
            pv[w as usize] = !0;
            mv[w as usize] = 0;
        }
        let mut sb: i64 = if last >= 0 {
            bottom(last as usize) as i64
        } else {
            0
        };
        // As after every column: the word below a cell <= c joins (row 0 included).
        if last + 1 < words as isize && sb <= c {
            last += 1;
            let w = last as usize;
            pv[w] = !0;
            mv[w] = 0;
            sb += rows(w) as i64;
        }
        let mut forward = Forward {
            dist: i64::MAX,
            b_end: 0,
            n_ends: 0,
            last_end: 0,
        };
        if mode != Mode::Global && last == words as isize - 1 {
            forward.record(sb, 0, &mut ends);
            c = c.min(forward.dist);
        }
        let mut completed = n == 0;
        for (index, &x) in b.iter().enumerate() {
            let j = index + 1;
            if last < first as isize {
                break;
            }
            let eq = &peq[x as usize * words..][..words];
            let mut hin = if first == 0 { top } else { 1 };
            let lw = last as usize;
            for w in first..lw {
                let (ph, mh) = step(&mut pv[w], &mut mv[w], eq[w], hin);
                hin = hout(ph, mh);
            }
            let (ph, mh) = step(&mut pv[lw], &mut mv[lw], eq[lw], hin);
            sb += delta_at(ph, mh, rows(lw) - 1);
            if mode != Mode::Global && lw == words - 1 {
                forward.record(sb, j, &mut ends);
                c = c.min(forward.dist);
            }
            if j == n {
                completed = true;
                break;
            }
            // Drop bottom words whose cells all exceed c.
            while last >= first as isize && sb >= c + rows(last as usize) as i64 {
                let w = last as usize;
                let mask = valid(w);
                sb -=
                    i64::from((pv[w] & mask).count_ones()) - i64::from((mv[w] & mask).count_ones());
                last -= 1;
            }
            if last < first as isize {
                if first > 0 {
                    break;
                }
                // Row 0 is the region's bottom: D(0, j) = j, or 0 in infix.
                sb = if mode == Mode::Infix { 0 } else { j as i64 };
            }
            // A word below can reach <= c next column only through a cell <= c above it.
            if last + 1 < words as isize && sb <= c {
                last += 1;
                let w = last as usize;
                pv[w] = !0;
                mv[w] = 0;
                sb += rows(w) as i64;
            }
            if mode != Mode::Infix {
                let lo = j as i64 + 1 - c;
                while (first as isize) <= last && (bottom(first) as i64) < lo {
                    first += 1;
                }
            }
        }
        match mode {
            Mode::Global => {
                (completed && last == words as isize - 1 && sb <= cutoff).then_some(Forward {
                    dist: sb,
                    b_end: n,
                    n_ends: 1,
                    last_end: n,
                })
            }
            _ => (forward.dist <= cutoff).then_some(forward),
        }
    }

    /// Append the requested detail fields of a kept pair. Every buffer is
    /// reserved fallibly; on error the caller publishes nothing.
    fn write(
        &mut self,
        mode: Mode,
        a: &[u8],
        b: &[u8],
        forward: &Forward,
        out: &mut DetailColumns,
    ) -> Result<(), EngineError> {
        if mode != Mode::Global {
            if out.wants(B_END) {
                out.push_i64(B_END, forward.b_end as i64)?;
            }
            if out.wants(N_ENDS) {
                out.push_i32(
                    N_ENDS,
                    i32::try_from(forward.n_ends).expect("length cap bounds n_ends"),
                )?;
            }
        }
        // Infix: one reverse pass gives b_start and n_starts (spec §3.1, §6);
        // prefix starts at 0 only, so n_starts is 1.
        let need_start =
            mode == Mode::Infix && (out.wants(B_START) || out.wants(N_STARTS) || out.wants(CIGAR));
        let (b_start, n_starts) = if need_start {
            self.window_start(a, b, forward)?
        } else {
            (0, 1)
        };
        if mode != Mode::Global {
            if out.wants(B_START) {
                out.push_i64(B_START, b_start as i64)?;
            }
            if out.wants(N_STARTS) {
                out.push_i32(
                    N_STARTS,
                    i32::try_from(n_starts).expect("length cap bounds n_starts"),
                )?;
            }
            if out.wants(ENDS) {
                out.push_list(ENDS, &self.ends)?;
            }
        }
        if out.wants(CIGAR) {
            let window = &b[b_start..forward.b_end];
            self.cigar(a, window)?;
            out.push_str(CIGAR, &self.text)?;
        }
        Ok(())
    }

    /// Infix (b_start, n_starts): b_start is the smallest s with
    /// NW(a, b[s:b_end]) = dist and n_starts the number of such s, from one
    /// reverse prefix pass (reversed a against reversed b[0:b_end]): b_end
    /// minus that pass's largest optimal end, and the number of its optimal
    /// ends, end 0 included (spec §3.1, §6). The pass is banded at dist, so
    /// every counted end is exact and every other end exceeds dist.
    fn window_start(
        &mut self,
        a: &[u8],
        b: &[u8],
        forward: &Forward,
    ) -> Result<(usize, usize), EngineError> {
        let m = a.len();
        if m == 0 || forward.b_end == 0 {
            // Only s = b_end can be optimal: b_end = 0 has the one start 0,
            // and m = 0 has b_end = 0 (the smallest end of dist 0).
            return Ok((0, 1));
        }
        let words = m.div_ceil(64);
        self.reserve_columns(words)?;
        self.rev.clear();
        reserve_total(
            &mut self.rev,
            forward.b_end,
            "levenshtein reversed window",
            self.budget,
        )?;
        let mut reversed = std::mem::take(&mut self.rev);
        reversed.extend(b[..forward.b_end].iter().rev());
        let pass = match self.load_peq(a, words, true) {
            Ok(()) => {
                let pass = self.scan_banded(words, m, &reversed, Mode::Prefix, forward.dist, false);
                self.clear_peq(a, words, true);
                Ok(pass.expect("the reverse pass reaches the forward distance"))
            }
            Err(error) => Err(error),
        };
        self.rev = reversed;
        let pass = pass?;
        debug_assert_eq!(pass.dist, forward.dist);
        Ok((forward.b_end - pass.last_end, pass.n_ends))
    }

    /// Rule E CIGAR of a against window w (spec §4) into `self.text`. Every
    /// buffer (traceback, moves, text) is sized with checked arithmetic and
    /// reserved before the DP starts.
    fn cigar(&mut self, a: &[u8], w: &[u8]) -> Result<(), EngineError> {
        const MOVES: &str = "levenshtein cigar moves";
        let (m, n) = (a.len(), w.len());
        // At most m + n moves; a run of r moves prints at most r + 1 <= 2r bytes.
        let moves = m
            .checked_add(n)
            .ok_or_else(|| allocation_error(MOVES, "size overflows usize"))?;
        self.ops.clear();
        self.text.clear();
        reserve_total(&mut self.ops, moves, MOVES, self.budget)?;
        let text = checked_len(&[moves, 2], "levenshtein cigar text")?;
        reserve_total(&mut self.text, text, "levenshtein cigar text", self.budget)?;
        if m > 0 && n > 0 {
            const TRACE: &str = "levenshtein cigar traceback";
            let words = m.div_ceil(64);
            let cells = checked_len(&[2, words, n], TRACE)?;
            self.trace.clear();
            reserve_total(&mut self.trace, cells, TRACE, self.budget)?;
            self.reserve_columns(words)?;
            self.load_peq(a, words, false)?;
            self.fill_trace(words, w);
            debug_assert_eq!(self.trace.len(), cells);
            self.clear_peq(a, words, false);
            self.traceback(words, a, w);
        } else {
            self.ops.extend(std::iter::repeat_n(b'I', m));
            self.ops.extend(std::iter::repeat_n(b'D', n));
        }
        let mut k = 0;
        while k < self.ops.len() {
            let op = self.ops[k];
            let run = self.ops[k..].iter().take_while(|&&x| x == op).count();
            push_count(&mut self.text, run);
            self.text.push(op);
            k += run;
        }
        Ok(())
    }

    /// Full global DP of a against w, storing per column and word the vertical
    /// +1 mask (Pv) then the horizontal +1 mask (Ph): 16 bytes per word per
    /// column. `trace` is already reserved for 2 * words * |w| words, so the
    /// pushes never reallocate, and no cell is written twice (no zero fill).
    fn fill_trace(&mut self, words: usize, w: &[u8]) {
        let Self {
            peq, pv, mv, trace, ..
        } = self;
        let (pv, mv) = (&mut pv[..words], &mut mv[..words]);
        pv.fill(!0);
        mv.fill(0);
        trace.clear();
        for &x in w {
            let eq = &peq[x as usize * words..][..words];
            let mut hin = 1;
            for k in 0..words {
                let (ph, mh) = step(&mut pv[k], &mut mv[k], eq[k], hin);
                hin = hout(ph, mh);
                trace.extend_from_slice(&[pv[k], ph]);
            }
        }
    }

    /// Greedy walk from (m, |w|) to (0, 0): I if D(i, j) = D(i−1, j) + 1, else D
    /// if D(i, j) = D(i, j−1) + 1, else the diagonal (the recurrence's only
    /// remaining witness). Moves are pushed last-first, then reversed.
    fn traceback(&mut self, words: usize, a: &[u8], w: &[u8]) {
        let (mut i, mut j) = (a.len(), w.len());
        while i > 0 || j > 0 {
            let op = if i == 0 {
                b'D'
            } else if j == 0 {
                b'I'
            } else {
                let column = &self.trace[(j - 1) * 2 * words..][..2 * words];
                let (k, bit) = ((i - 1) / 64, (i - 1) % 64);
                if column[2 * k] >> bit & 1 == 1 {
                    b'I'
                } else if column[2 * k + 1] >> bit & 1 == 1 {
                    b'D'
                } else if a[i - 1] == w[j - 1] {
                    b'='
                } else {
                    b'X'
                }
            };
            match op {
                b'I' => i -= 1,
                b'D' => j -= 1,
                _ => {
                    i -= 1;
                    j -= 1;
                }
            }
            self.ops.push(op);
        }
        self.ops.reverse();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use seqblaze_core::batch::SeqBatch;

    /// Literal §2 DP: last row D(m, ·).
    fn dp_last_row(a: &[u8], b: &[u8], mode: Mode) -> Vec<i64> {
        let n = b.len();
        let mut row: Vec<i64> = (0..=n as i64)
            .map(|j| if mode == Mode::Infix { 0 } else { j })
            .collect();
        for (i, &x) in a.iter().enumerate() {
            let mut next = vec![i as i64 + 1; n + 1];
            for j in 1..=n {
                next[j] = (row[j - 1] + i64::from(x != b[j - 1]))
                    .min(row[j] + 1)
                    .min(next[j - 1] + 1);
            }
            row = next;
        }
        row
    }

    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            self.0 >> 33
        }
        fn seq(&mut self, n: usize, alphabet: &[u8]) -> Vec<u8> {
            (0..n)
                .map(|_| alphabet[self.next() as usize % alphabet.len()])
                .collect()
        }
    }

    fn check(work: &mut Work, a: &[u8], b: &[u8]) {
        for mode in [Mode::Global, Mode::Prefix, Mode::Infix] {
            let row = dp_last_row(a, b, mode);
            let (dist, ends): (i64, Vec<i64>) = if mode == Mode::Global {
                (row[b.len()], vec![])
            } else {
                let best = *row.iter().min().unwrap();
                (
                    best,
                    (0..row.len() as i64)
                        .filter(|&e| row[e as usize] == best)
                        .collect(),
                )
            };
            let full = work.forward(a, b, mode, None, true).unwrap().unwrap();
            assert_eq!(full.dist, dist, "{mode:?} {a:?} {b:?}");
            if mode != Mode::Global {
                assert_eq!(work.ends, ends);
                assert_eq!(full.n_ends, ends.len());
                assert_eq!(full.b_end as i64, ends[0]);
                assert_eq!(full.last_end as i64, *ends.last().unwrap());
            }
            for c in [0, 1, 2, dist - 1, dist, dist + 1, dist + 70, 200] {
                if c < 0 {
                    continue;
                }
                let banded = work.forward(a, b, mode, Some(c as i32), true).unwrap();
                if dist <= c {
                    let banded = banded.unwrap_or_else(|| panic!("{mode:?} c={c} {a:?} {b:?}"));
                    assert_eq!(banded.dist, dist);
                    if mode != Mode::Global {
                        assert_eq!(work.ends, ends, "{mode:?} c={c}");
                        assert_eq!(banded.n_ends, ends.len());
                    }
                } else {
                    assert!(banded.is_none(), "{mode:?} c={c} dist={dist}");
                }
            }
        }
    }

    #[test]
    fn forward_passes_match_the_dp_across_word_boundaries() {
        let mut rng = Rng(7);
        let mut work = Work::default();
        for &m in &[1usize, 2, 5, 63, 64, 65, 127, 128, 129, 200] {
            for &n in &[0usize, 1, 3, 63, 64, 65, 130, 210] {
                let a = rng.seq(m, b"ACGT");
                let mut b = rng.seq(n, b"ACGT");
                check(&mut work, &a, &b);
                // planted near-copy
                if n > 4 {
                    let start = rng.next() as usize % n;
                    for (k, byte) in a.iter().enumerate().take(n - start) {
                        if rng.next() % 10 != 0 {
                            b[start + k] = *byte;
                        }
                    }
                    check(&mut work, &a, &b);
                }
                check(&mut work, &vec![b'A'; m], &vec![b'A'; n]);
                check(&mut work, &vec![b'A'; m], &vec![b'C'; n]);
            }
        }
        assert!(work.peq.iter().all(|&x| x == 0));
    }

    /// Literal rule E over a full integer table (global boundaries).
    fn dp_cigar(a: &[u8], w: &[u8]) -> String {
        let (m, n) = (a.len(), w.len());
        let mut t = vec![vec![0i64; n + 1]; m + 1];
        for (j, cell) in t[0].iter_mut().enumerate() {
            *cell = j as i64;
        }
        for i in 1..=m {
            t[i][0] = i as i64;
            for j in 1..=n {
                t[i][j] = (t[i - 1][j - 1] + i64::from(a[i - 1] != w[j - 1]))
                    .min(t[i - 1][j] + 1)
                    .min(t[i][j - 1] + 1);
            }
        }
        let (mut i, mut j) = (m, n);
        let mut ops = Vec::new();
        while i > 0 || j > 0 {
            if i > 0 && t[i][j] == t[i - 1][j] + 1 {
                ops.push('I');
                i -= 1;
            } else if j > 0 && t[i][j] == t[i][j - 1] + 1 {
                ops.push('D');
                j -= 1;
            } else {
                ops.push(if a[i - 1] == w[j - 1] { '=' } else { 'X' });
                i -= 1;
                j -= 1;
            }
        }
        ops.reverse();
        let mut text = String::new();
        let mut k = 0;
        while k < ops.len() {
            let run = ops[k..].iter().take_while(|&&x| x == ops[k]).count();
            text.push_str(&format!("{run}{}", ops[k]));
            k += run;
        }
        text
    }

    fn global(a: &[u8], b: &[u8]) -> i64 {
        dp_last_row(a, b, Mode::Global)[b.len()]
    }

    #[test]
    fn windows_and_cigars_match_the_dp_on_multiword_pairs() {
        let mut rng = Rng(11);
        let table = |x: &[u8]| SeqTable::new(SeqBatch::from_seqs([x]));
        let all = ["b_end", "n_ends", "b_start", "n_starts", "ends", "cigar"];
        for round in 0..60 {
            let m = [1, 7, 63, 64, 65, 129, 150][round % 7];
            let a = rng.seq(m, b"ACGT");
            let n = rng.next() as usize % 220;
            let mut b = rng.seq(n, b"ACGT");
            if round % 2 == 0 && b.len() > m {
                let start = rng.next() as usize % (b.len() - m);
                for k in 0..m {
                    if rng.next() % 8 != 0 {
                        b[start + k] = a[k];
                    }
                }
            }
            for mode in [Mode::Global, Mode::Prefix, Mode::Infix] {
                let row = dp_last_row(&a, &b, mode);
                let (dist, end) = if mode == Mode::Global {
                    (row[b.len()], b.len())
                } else {
                    let best = *row.iter().min().unwrap();
                    (best, row.iter().position(|&v| v == best).unwrap())
                };
                let start = match mode {
                    Mode::Infix => (0..=end).find(|&s| global(&a, &b[s..end]) == dist).unwrap(),
                    _ => 0,
                };
                // Spec §3.1: every start s <= b_end with NW(a, b[s:b_end]) = dist.
                let starts = match mode {
                    Mode::Infix => (0..=end)
                        .filter(|&s| global(&a, &b[s..end]) == dist)
                        .count(),
                    _ => 1,
                };
                let want = dp_cigar(&a, &b[start..end]);
                let fields: &[&str] = if mode == Mode::Global {
                    &["cigar"]
                } else {
                    &all
                };
                let kernel = Levenshtein::new(mode, fields).unwrap();
                let (ta, tb) = (table(&a), table(&b));
                for prune in [None, Some(dist as i32), Some(dist as i32 + 9)] {
                    let mut out = DetailColumns::new(&FIELDS, kernel.request);
                    let score = kernel
                        .evaluate_detailed(ta.get(0), tb.get(0), prune, None, &mut out)
                        .unwrap();
                    assert_eq!(score, Some(dist as i32));
                    for column in out.columns() {
                        match (column.name, &column.data) {
                            ("cigar", seqblaze_core::detail::DetailData::Utf8 { data, .. }) => {
                                assert_eq!(
                                    std::str::from_utf8(data).unwrap(),
                                    want,
                                    "{mode:?} m={m}"
                                )
                            }
                            ("b_start", seqblaze_core::detail::DetailData::Int64(v)) => {
                                assert_eq!(v, &[start as i64], "{mode:?}")
                            }
                            ("n_starts", seqblaze_core::detail::DetailData::Int32(v)) => {
                                assert_eq!(v, &[starts as i32], "{mode:?} m={m}")
                            }
                            ("b_end", seqblaze_core::detail::DetailData::Int64(v)) => {
                                assert_eq!(v, &[end as i64])
                            }
                            _ => {}
                        }
                    }
                }
            }
        }
    }

    /// (a, b, mode, dist, ends, b_start, cigar)
    type Example = (
        &'static [u8],
        &'static [u8],
        Mode,
        i32,
        &'static [i64],
        i64,
        &'static str,
    );

    #[test]
    fn spec_worked_examples() {
        let cases: [Example; 8] = [
            (b"AA", b"AAAA", Mode::Global, 2, &[], 0, "2=2D"),
            (b"AA", b"AAAA", Mode::Prefix, 0, &[2], 0, "2="),
            (b"AA", b"AAAA", Mode::Infix, 0, &[2, 3, 4], 0, "2="),
            (b"AC", b"CA", Mode::Global, 2, &[], 0, "1D1=1I"),
            (b"AC", b"CA", Mode::Infix, 1, &[1, 2], 0, "1I1="),
            (b"A", b"C", Mode::Infix, 1, &[0, 1], 0, "1I"),
            (b"ACGT", b"TGCA", Mode::Prefix, 3, &[1, 2, 3], 0, "3I1="),
            (b"AAAA", b"A", Mode::Infix, 3, &[1], 0, "1=3I"),
        ];
        let all = ["b_end", "n_ends", "b_start", "n_starts", "ends", "cigar"];
        for (a, b, mode, dist, ends, start, cigar) in cases {
            let fields: &[&str] = if mode == Mode::Global {
                &["cigar"]
            } else {
                &all
            };
            let kernel = Levenshtein::new(mode, fields).unwrap();
            let table = |x: &[u8]| SeqTable::new(SeqBatch::from_seqs([x]));
            let (ta, tb) = (table(a), table(b));
            for (prune, keep) in [(None, None), (Some(dist), Some(dist))] {
                let mut out = DetailColumns::new(&FIELDS, kernel.request);
                let score = kernel
                    .evaluate_detailed(ta.get(0), tb.get(0), prune, keep, &mut out)
                    .unwrap();
                assert_eq!(score, Some(dist));
                let mut again = DetailColumns::new(&FIELDS, kernel.request);
                kernel
                    .details(ta.get(0), tb.get(0), dist, &mut again)
                    .unwrap();
                assert_eq!(out, again);
                let columns = out.columns();
                let text = |name: &str| {
                    columns
                        .iter()
                        .find(|c| c.name == name)
                        .map(|c| c.data.clone())
                        .unwrap()
                };
                match text("cigar") {
                    seqblaze_core::detail::DetailData::Utf8 { data, .. } => {
                        assert_eq!(
                            std::str::from_utf8(&data).unwrap(),
                            cigar,
                            "{a:?} {b:?} {mode:?}"
                        )
                    }
                    _ => unreachable!(),
                }
                if mode != Mode::Global {
                    match text("ends") {
                        seqblaze_core::detail::DetailData::ListInt64 { values, .. } => {
                            assert_eq!(values, ends)
                        }
                        _ => unreachable!(),
                    }
                    match text("b_start") {
                        seqblaze_core::detail::DetailData::Int64(values) => {
                            assert_eq!(values, [start])
                        }
                        _ => unreachable!(),
                    }
                }
            }
            assert_eq!(
                kernel
                    .evaluate(ta.get(0), tb.get(0), Some(dist - 1))
                    .unwrap(),
                None,
                "{a:?} {b:?}"
            );
        }
    }

    #[test]
    fn common_prefix_and_suffix_match_a_byte_walk() {
        for len in 0..20usize {
            let a: Vec<u8> = (0..len as u8).map(|k| k.wrapping_mul(37)).collect();
            for other in [len.saturating_sub(1), len, len + 1, len + 9] {
                for differ in 0..=other {
                    let mut b: Vec<u8> = (0..other as u8).map(|k| k.wrapping_mul(37)).collect();
                    let mut c = b.clone();
                    if differ < other {
                        b[differ] ^= 0x80;
                        c[other - 1 - differ] ^= 0x80;
                    }
                    let head = a.iter().zip(&b).take_while(|(x, y)| x == y).count();
                    assert_eq!(common_prefix(&a, &b), head, "{len} {other} {differ}");
                    let tail = a
                        .iter()
                        .rev()
                        .zip(c.iter().rev())
                        .take_while(|(x, y)| x == y)
                        .count();
                    assert_eq!(common_suffix(&a, &c), tail, "{len} {other} {differ}");
                }
            }
        }
    }

    /// `evaluate` removes the shared head (global, prefix) and tail (global)
    /// before its forward pass; its value and its cutoff answers equal the
    /// DP over the whole pair, across word boundaries of what remains.
    #[test]
    fn score_only_evaluation_matches_the_dp_with_shared_heads_and_tails() {
        let mut rng = Rng(11);
        let mut pairs = Vec::new();
        for &(head, tail) in &[(0, 0), (1, 0), (0, 1), (7, 9), (8, 8), (63, 1), (64, 65)] {
            for &(m, n) in &[(0, 0), (0, 3), (3, 0), (1, 1), (5, 9), (64, 63), (70, 140)] {
                let (shared_head, shared_tail) = (rng.seq(head, b"ACGT"), rng.seq(tail, b"ACGT"));
                let a = [&shared_head[..], &rng.seq(m, b"ACGT"), &shared_tail].concat();
                let b = [&shared_head[..], &rng.seq(n, b"ACGT"), &shared_tail].concat();
                pairs.push((a.clone(), b.clone()));
                // a prefix of b, b prefix of a, equal rows.
                pairs.push((a.clone(), [&a[..], &b[..]].concat()));
                pairs.push(([&b[..], &a[..]].concat(), b.clone()));
                pairs.push((a.clone(), a.clone()));
            }
        }
        pairs.push((vec![b'A'; 100], vec![b'A'; 130]));
        pairs.push((vec![b'A'; 130], vec![b'A'; 100]));
        for (a, b) in &pairs {
            let table = SeqTable::new(SeqBatch::from_seqs([&a[..], &b[..]]));
            let (x, y) = (table.get(0), table.get(1));
            for mode in [Mode::Global, Mode::Prefix, Mode::Infix] {
                let row = dp_last_row(a, b, mode);
                let dist = if mode == Mode::Global {
                    row[b.len()]
                } else {
                    *row.iter().min().unwrap()
                } as i32;
                let kernel = Levenshtein::new(mode, &[]).unwrap();
                assert_eq!(kernel.evaluate(x, y, None).unwrap(), Some(dist), "{mode:?}");
                for c in [-1, 0, dist - 1, dist, dist + 1, i32::MAX] {
                    let expected = (dist <= c).then_some(dist);
                    assert_eq!(
                        kernel.evaluate(x, y, Some(c)).unwrap(),
                        expected,
                        "{mode:?} c={c}"
                    );
                }
            }
        }
    }

    #[test]
    fn scratch_above_one_mib_is_released_after_the_pair() {
        // 40,000 rows: 625 words, so Peq is 1.28 MB (> KEEP_BYTES); one base of b.
        let a = SeqTable::new(SeqBatch::from_seqs([&vec![b'C'; 40_000][..]]));
        let b = SeqTable::new(SeqBatch::from_seqs([&b"A"[..]]));
        let kernel = Levenshtein::new(Mode::Infix, &[]).unwrap();
        std::thread::spawn(move || {
            assert_eq!(
                kernel.evaluate(a.get(0), b.get(0), None).unwrap(),
                Some(40_000)
            );
            SCRATCH.with(|cell| {
                let work = &cell.borrow().work;
                assert_eq!(work.peq.capacity(), 0);
                assert!(!work.peq_dirty);
                // Column state (625 words) stays: 5,000 bytes each.
                assert!(work.pv.capacity() * 8 <= KEEP_BYTES);
            });
        })
        .join()
        .unwrap();
    }

    #[test]
    fn pair_cost_follows_the_word_steps() {
        let table = |n: usize| SeqTable::new(SeqBatch::from_seqs([&vec![b'A'; n][..]]));
        let (a, b) = (table(20), table(50_000));
        // Hamming keeps the linear default: a's mean length.
        assert_eq!(crate::Hamming.pair_cost(&a, &b, false), 20);
        let infix = Levenshtein::new(Mode::Infix, &["b_end"]).unwrap();
        assert_eq!(infix.pair_cost(&a, &b, false), 50_000);
        assert_eq!(infix.pair_cost(&a, &b, true), 50_000);
        let infix = Levenshtein::new(Mode::Infix, &["cigar"]).unwrap();
        // Reverse pass over b, stored pass 2 * 1 * 20, walk 20 + 20.
        assert_eq!(infix.pair_cost(&a, &b, true), 50_000 + 50_000 + 40 + 40);
        let global = Levenshtein::new(Mode::Global, &["cigar"]).unwrap();
        let (a, b) = (table(130), table(1_000));
        assert_eq!(global.pair_cost(&a, &b, false), 3 * 1_000);
        assert_eq!(global.pair_cost(&a, &b, true), 3_000 + 6_000 + 1_130);
    }

    #[test]
    fn length_limits_are_pinned() {
        use seqblaze_core::{
            exec::{run, ExecOptions},
            plan::Strategy,
            sched::Grain,
            sink::Sink,
            work::{PairKind, WorkSource},
        };
        // Spec §8: one below Hamming's 2**31 - 1, so n_ends <= n + 1 fits int32.
        const GENERAL: usize = (1 << 31) - 2;
        const CIGAR_CAP: usize = 16_384;
        assert_eq!((MAX_LENGTH, CIGAR_MAX_LENGTH), (GENERAL, CIGAR_CAP));
        let located: &[&str] = &["b_end", "n_ends", "b_start", "ends"];
        for mode in [Mode::Global, Mode::Prefix, Mode::Infix] {
            let others: &[&str] = if mode == Mode::Global { &[] } else { located };
            let mut with_cigar = others.to_vec();
            with_cigar.push("cigar");
            for fields in [&[][..], others] {
                let kernel = Levenshtein::new(mode, fields).unwrap();
                assert_eq!(kernel.sequence_length_limit(), Some(GENERAL));
                assert_eq!(
                    kernel.length_caps(),
                    [Some(GENERAL); 2],
                    "{mode:?} {fields:?}"
                );
            }
            for fields in [&["cigar"][..], &with_cigar] {
                let kernel = Levenshtein::new(mode, fields).unwrap();
                assert_eq!(kernel.sequence_length_limit(), Some(GENERAL));
                // b is capped by cigar only in global mode (spec §8).
                let b = if mode == Mode::Global {
                    CIGAR_CAP
                } else {
                    GENERAL
                };
                assert_eq!(kernel.length_caps(), [Some(CIGAR_CAP), Some(b)], "{mode:?}");
            }
        }
        // The error at limit + 1 names the input, index, length and limit.
        let kernel = Levenshtein::new(Mode::Prefix, &["cigar"]).unwrap();
        for (limit, length, cap, cigar) in [
            (GENERAL, "2,147,483,647", "2,147,483,646", false),
            (CIGAR_CAP, "16,385", "16,384", true),
        ] {
            match kernel.length_error("b", 3, limit + 1, limit) {
                EngineError::Sequence {
                    message,
                    input,
                    index,
                    position,
                    byte,
                    n_bad,
                    bad_indices,
                } => {
                    assert_eq!((input, index, position, byte), ("b", 3, None, None));
                    assert_eq!((n_bad, bad_indices), (1, vec![3]));
                    assert!(message.contains(&format!("sequence 3 of input b has length {length}")));
                    assert!(message.contains(&format!("limit is {cap}")), "{message}");
                    assert!(message.contains("Nothing was computed"), "{message}");
                    assert_eq!(message.contains("cigar"), cigar, "{message}");
                }
                other => panic!("expected a SequenceError, got {other:?}"),
            }
        }
        // Through validation, at the cigar cap (2 GiB rows for the general cap
        // are out of reach): a length equal to the cap runs, one more is
        // refused for row 1 before anything is computed. The partner is one
        // base, so the DP stays tiny.
        let options = ExecOptions {
            strategy: Strategy::Full,
            threads: 1,
            grain: Grain::Exact,
            sink: Sink::Each,
            cutoff: None,
            check_bases: true,
            k: None,
        };
        let work = WorkSource::new(PairKind::Zip, 2, 2).unwrap();
        let short = SeqTable::new(SeqBatch::from_seqs([&b"A"[..], &b"C"[..]]));
        for (mode, long_input) in [(Mode::Prefix, "a"), (Mode::Infix, "a"), (Mode::Global, "b")] {
            let kernel = Levenshtein::new(mode, &["cigar"]).unwrap();
            for length in [CIGAR_CAP, CIGAR_CAP + 1] {
                let long = vec![b'A'; length];
                let long = SeqTable::new(SeqBatch::from_seqs([&b"A"[..], &long[..]]));
                let (a, b) = if long_input == "a" {
                    (&long, &short)
                } else {
                    (&short, &long)
                };
                match run(&kernel, &work, a, b, &options) {
                    Ok(output) if length == CIGAR_CAP => {
                        assert_eq!(output.columns.dist.len(), 2, "{mode:?}")
                    }
                    Err(EngineError::Sequence { input, index, .. }) if length > CIGAR_CAP => {
                        assert_eq!((input, index), (long_input, 1), "{mode:?}")
                    }
                    other => panic!("{mode:?} length {length}: {:?}", other.map(|_| ())),
                }
            }
        }
        // Without cigar the same rows pass in every mode, as a and as b.
        let long = vec![b'A'; CIGAR_CAP + 1];
        let long = SeqTable::new(SeqBatch::from_seqs([&b"A"[..], &long[..]]));
        for mode in [Mode::Global, Mode::Prefix, Mode::Infix] {
            let kernel = Levenshtein::new(mode, &[]).unwrap();
            assert!(
                run(&kernel, &work, &long, &short, &options).is_ok(),
                "{mode:?}"
            );
            assert!(
                run(&kernel, &work, &short, &long, &options).is_ok(),
                "{mode:?}"
            );
        }
    }

    #[test]
    fn sizes_are_checked_and_reservations_fallible() {
        assert!(matches!(
            checked_len(&[usize::MAX, 2], "test buffer"),
            Err(EngineError::Device(message)) if message.contains("test buffer")
        ));
        assert_eq!(checked_len(&[2, 3, 4], "test buffer").unwrap(), 24);
        // More than isize::MAX bytes: refused without attempting an allocation.
        let mut buffer = vec![7u64];
        assert!(matches!(
            reserve_total(&mut buffer, usize::MAX / 8, "test buffer", None),
            Err(EngineError::Device(_))
        ));
        assert!(matches!(
            grow(&mut buffer, usize::MAX / 4, 0, "test buffer", None),
            Err(EngineError::Device(_))
        ));
        assert_eq!(buffer, [7]);
        // The injected budget refuses exactly above its byte count.
        assert!(reserve_total(&mut buffer, 4, "test buffer", Some(32)).is_ok());
        assert!(matches!(
            reserve_total(&mut buffer, 5, "test buffer", Some(32)),
            Err(EngineError::Device(_))
        ));
    }

    /// Run `body` with this thread's scratch limited to `bytes` per buffer.
    fn with_budget<R>(bytes: usize, body: impl FnOnce() -> R) -> R {
        SCRATCH.with(|cell| cell.borrow_mut().work.budget = Some(bytes));
        let result = body();
        SCRATCH.with(|cell| cell.borrow_mut().work.budget = None);
        result
    }

    #[test]
    fn failed_allocations_are_device_errors_and_publish_nothing() {
        use seqblaze_core::{
            exec::{run, ExecOptions},
            plan::Strategy,
            sched::Grain,
            sink::Sink,
            work::{PairKind, WorkSource},
        };
        let mut rng = Rng(5);
        let (a, b) = (rng.seq(200, b"ACGT"), rng.seq(200, b"ACGT"));
        let (ta, tb) = (
            SeqTable::new(SeqBatch::from_seqs([&a[..]])),
            SeqTable::new(SeqBatch::from_seqs([&b[..]])),
        );
        let expected = dp_last_row(&a, &b, Mode::Global)[200] as i32;
        let kernel = Levenshtein::new(Mode::Global, &["cigar"]).unwrap();
        // 200 rows: 4 words, so Peq is 8,192 bytes and the cigar traceback
        // 2 * 4 * 200 words = 12,800 bytes. A 10,000-byte budget admits the
        // score and refuses the traceback.
        with_budget(10_000, || {
            assert_eq!(
                kernel.evaluate(ta.get(0), tb.get(0), None).unwrap(),
                Some(expected)
            );
            let mut out = DetailColumns::new(&FIELDS, kernel.request);
            match kernel.evaluate_detailed(ta.get(0), tb.get(0), None, None, &mut out) {
                Err(EngineError::Device(message)) => {
                    assert!(message.contains("cigar traceback"), "{message}")
                }
                other => panic!("expected DeviceError, got {other:?}"),
            }
            assert_eq!(out.rows(), 0);
            // The engine stops the call: nothing is published.
            let options = ExecOptions {
                strategy: Strategy::Full,
                threads: 1, // runs inline on this thread, under its budget
                grain: Grain::Exact,
                sink: Sink::Each,
                cutoff: None,
                check_bases: true,
                k: None,
            };
            let work = WorkSource::new(PairKind::Zip, 1, 1).unwrap();
            assert!(matches!(
                run(&kernel, &work, &ta, &tb, &options),
                Err(EngineError::Device(_))
            ));
        });
        // A budget below Peq refuses the score itself (a fresh thread, whose
        // scratch has not grown yet; the budget only limits new allocations).
        let (a2, b2) = (a.clone(), b.clone());
        std::thread::spawn(move || {
            let (ta, tb) = (
                SeqTable::new(SeqBatch::from_seqs([&a2[..]])),
                SeqTable::new(SeqBatch::from_seqs([&b2[..]])),
            );
            with_budget(4_096, || {
                assert!(matches!(
                    kernel.evaluate(ta.get(0), tb.get(0), None),
                    Err(EngineError::Device(message)) if message.contains("match-mask")
                ));
            });
        })
        .join()
        .unwrap();
        // The thread's scratch is clean afterwards: the same pair succeeds.
        let mut out = DetailColumns::new(&FIELDS, kernel.request);
        assert_eq!(
            kernel
                .evaluate_detailed(ta.get(0), tb.get(0), None, None, &mut out)
                .unwrap(),
            Some(expected)
        );
        assert_eq!(out.rows(), 1);
        SCRATCH.with(|cell| assert!(cell.borrow().work.peq.iter().all(|&x| x == 0)));
    }
}
