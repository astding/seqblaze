//! Inter-pair score pass (spec §7, "later SIMD paths"): several pairs at once,
//! one per vector lane.
//!
//! Each lane runs the linear-memory score fill of `dp.rs` for its own pair,
//! cell by cell with the same operations, in exact i32 arithmetic instead of
//! i64; so a lane's score is the scalar fill's score (spec §7: byte-identical)
//! and the lane width never changes an output. A group of lanes walks the rows
//! and columns of its longest pair. Cells past a lane's own lengths are
//! padding: they never feed a cell of that lane's pair, and the lane's result
//! reads only its own cells.
//!
//! **Range.** With M and N the group's longest lengths and padding scored like
//! a symbol of the scoring, every cell of the M x N fill is a cell of an
//! ordinary fill, so its value lies within +-2^30 when
//! sigma (min(M, N) + 1) + 2 go + ge (M + N) <= 2^30 (spec §9.1). That also
//! gives go + ge <= 2^29 (M, N >= 2). -infinity is NEG32 = -2^30 - 1, below
//! every finite value. No fill adds to it, and no fill subtracts more than
//! go + ge from it before a finite value dominates, so no i32 operation wraps.
//! A group that fails the test is scored by the scalar fill.
use std::{
    cell::RefCell,
    ops::ControlFlow,
    sync::atomic::{AtomicUsize, Ordering},
};

use seqblaze_core::{seq::ByteView, validate::EngineError};

use crate::dp::Gaps;
use crate::scratch::{filled, KEEP_BYTES};

/// -infinity of the lane fills.
const NEG32: i32 = -(1 << 30) - 1;
const VALUE_RANGE: i64 = 1 << 30;
/// Lane buffers above this many bytes in total are not used: the group is
/// scored by the scalar fill (about 16 bytes per lane per row and column).
const MAX_GROUP_BYTES: usize = 32 << 20;
const ROWS: &str = "align lane rows";
/// Groups per `evaluate_many` call: pairs are grouped by length within it.
pub(crate) const GROUPS_PER_BATCH: usize = 4;

// ------------------------------------------------------------------ width

const AUTO: usize = usize::MAX;
static FORCED: AtomicUsize = AtomicUsize::new(AUTO);

/// Lanes per group on this CPU: 16 with AVX-512 (F, BW, VL; when compiled by
/// rustc 1.89 or later, see `x86::avx512`), 8 with AVX2 or on aarch64 (NEON),
/// else 0 (no lane path: every pair takes the scalar fill).
fn detected() -> usize {
    #[cfg(target_arch = "x86_64")]
    {
        #[cfg(seqblaze_avx512)]
        if x86::Avx512::new().is_some() {
            return 16;
        }
        if x86::Avx2::new().is_some() {
            8
        } else {
            0
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        8
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        0
    }
}

/// The lane width in use: the detected one unless `set_width` forced another.
pub fn width() -> usize {
    match FORCED.load(Ordering::Relaxed) {
        AUTO => detected(),
        forced => forced,
    }
}

/// Testing: force the lane width (0 = scalar fills only, or a width this build
/// and CPU support: 8, or 16 with AVX-512); None restores detection. Returns the
/// previous forced width. Outputs never depend on it.
pub fn set_width(width: Option<usize>) -> Result<Option<usize>, String> {
    let value = match width {
        None => AUTO,
        Some(0) => 0,
        Some(8) if detected() >= 8 => 8,
        Some(16) if detected() == 16 => 16,
        Some(other) => {
            return Err(format!(
                "align lane width {other} is not available on this CPU (detected {})",
                detected()
            ))
        }
    };
    let previous = FORCED.swap(value, Ordering::Relaxed);
    Ok((previous != AUTO).then_some(previous))
}

// ------------------------------------------------------------------ scoring

/// Substitution lookup of one call, by symbol code: a byte maps to an a code
/// and a b code, and s(x, y) = `score(code_a(x), code_b(y))`.
pub(crate) enum Lookup {
    /// `match` if the bytes are equal, else `mismatch`; the codes are the bytes.
    Equal { matched: i32, mismatched: i32 },
    /// At most 16 classes: a code 16 class(x), b code class(y), table index
    /// a code + b code.
    Small {
        class: [u8; 256],
        table: Box<[i32; 256]>,
    },
    /// Up to 256 classes, stride 256.
    Large {
        class: [u8; 256],
        table: Box<[i32; 1 << 16]>,
    },
}

impl Lookup {
    /// From the call's matched scoring or its symmetric class table (`row_classes`).
    pub fn new(
        matched: Option<(i32, i32)>,
        class: [u8; 256],
        n_classes: usize,
        table: &[i32],
    ) -> Self {
        if let Some((matched, mismatched)) = matched {
            return Self::Equal {
                matched,
                mismatched,
            };
        }
        if n_classes <= 16 {
            let mut small = Box::new([0i32; 256]);
            for x in 0..n_classes {
                for y in 0..n_classes {
                    small[x * 16 + y] = table[x * n_classes + y];
                }
            }
            Self::Small {
                class,
                table: small,
            }
        } else {
            let mut large: Box<[i32; 1 << 16]> = vec![0i32; 1 << 16]
                .into_boxed_slice()
                .try_into()
                .expect("65,536 entries");
            for x in 0..n_classes {
                for y in 0..n_classes {
                    large[x * 256 + y] = table[x * n_classes + y];
                }
            }
            Self::Large {
                class,
                table: large,
            }
        }
    }

    /// A byte's symbol in the lane codes: the byte itself with equality
    /// scoring, else its class.
    #[inline]
    fn symbol(&self, byte: u8) -> i32 {
        match self {
            Self::Equal { .. } => i32::from(byte),
            Self::Small { class, .. } | Self::Large { class, .. } => {
                i32::from(class[byte as usize])
            }
        }
    }
}

/// Padding symbols of a and b: no byte or class has them, so padding scores
/// as a mismatch under equality scoring; the table forms recode them to class
/// 0. Either way padding scores like a symbol of the scoring (spec §9.1 then
/// bounds the padded cells too).
const PAD_A: i32 = 256;
const PAD_B: i32 = 257;

/// A set of classes (or bytes).
#[derive(Default)]
struct Classes([u64; 4]);

impl Classes {
    fn insert(&mut self, x: i32) {
        self.0[(x >> 6) as usize] |= 1 << (x & 63);
    }

    fn len(&self) -> u32 {
        self.0.iter().map(|w| w.count_ones()).sum()
    }

    fn iter(&self) -> impl Iterator<Item = usize> + '_ {
        (0..256).filter(|&x| self.0[x >> 6] >> (x & 63) & 1 == 1)
    }
}

/// The (match, mismatch) scores when a class table, restricted to the classes
/// of a group's a rows and b rows, scores every equal pair of classes alike
/// and every unequal pair alike; then the group runs the equality form on
/// class symbols, whose cells score the same table entries. None otherwise,
/// or when the check would cost more than it saves.
fn equal_form(a: &Classes, b: &Classes, score: impl Fn(usize, usize) -> i32) -> Option<(i32, i32)> {
    if a.len() * b.len() > 256 {
        return None;
    }
    let (mut matched, mut mismatched) = (None, None);
    for x in a.iter() {
        for y in b.iter() {
            let slot = if x == y {
                &mut matched
            } else {
                &mut mismatched
            };
            let s = score(x, y);
            if *slot.get_or_insert(s) != s {
                return None;
            }
        }
    }
    let mismatched = mismatched.or(matched)?;
    Some((matched.unwrap_or(mismatched), mismatched))
}

// ------------------------------------------------------------------ groups

/// The fill a group runs (spec §2, §5.5, §5.6).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Fill {
    Local,
    Global { end_gaps: bool },
    Water,
    Needle { end_gaps: bool },
}

/// One group's shape and per-lane lengths (0 for an unused lane).
struct Shape<const L: usize> {
    m: [usize; L],
    n: [usize; L],
    rows: usize,
    cols: usize,
    /// Smallest lengths over the used lanes: every lane is real below them.
    m_min: usize,
    n_min: usize,
}

/// Per-thread lane buffers of one width.
struct Buffers<const L: usize> {
    at: Vec<[i32; L]>,
    bt: Vec<[i32; L]>,
    r0: Vec<[i32; L]>,
    r1: Vec<[i32; L]>,
    r2: Vec<[i32; L]>,
    r3: Vec<[i32; L]>,
    lim: Vec<[i32; L]>,
    bytes: Vec<u8>,
}

impl<const L: usize> Default for Buffers<L> {
    fn default() -> Self {
        Self {
            at: Vec::new(),
            bt: Vec::new(),
            r0: Vec::new(),
            r1: Vec::new(),
            r2: Vec::new(),
            r3: Vec::new(),
            lim: Vec::new(),
            bytes: Vec::new(),
        }
    }
}

impl<const L: usize> Buffers<L> {
    fn shrink(&mut self) {
        for buffer in [
            &mut self.at,
            &mut self.bt,
            &mut self.r0,
            &mut self.r1,
            &mut self.r2,
            &mut self.r3,
            &mut self.lim,
        ] {
            if buffer.capacity() * std::mem::size_of::<[i32; L]>() > KEEP_BYTES {
                *buffer = Vec::new();
            }
        }
        if self.bytes.capacity() > KEEP_BYTES {
            self.bytes = Vec::new();
        }
    }
}

thread_local! {
    #[cfg(all(target_arch = "x86_64", seqblaze_avx512))]
    static BUFFERS16: RefCell<Buffers<16>> = RefCell::new(Buffers::default());
    static BUFFERS8: RefCell<Buffers<8>> = RefCell::new(Buffers::default());
}

/// The call-level constants of a group fill.
pub(crate) struct Call<'c> {
    pub fill: Fill,
    pub gaps: Gaps,
    pub sigma: i64,
    pub lookup: &'c Lookup,
}

impl Call<'_> {
    /// The §9.1 test for a group of longest lengths M, N (module doc).
    fn in_range(&self, rows: usize, cols: usize) -> bool {
        let (m, n) = (rows as i128, cols as i128);
        let need = i128::from(self.sigma) * (m.min(n) + 1)
            + 2 * i128::from(self.gaps.go)
            + i128::from(self.gaps.ge) * (m + n);
        need <= i128::from(VALUE_RANGE)
    }
}

/// Whether a pair of these lengths may be scored in a group: both lengths at
/// least 2 (the fills' edge cases, water's sentinel and needle's R-m1 stay
/// with the scalar fill).
pub(crate) fn eligible(m: usize, n: usize) -> bool {
    m >= 2 && n >= 2
}

/// Score the pairs `members` (indices into `views`, at most the lane width,
/// each `eligible`) in one group, or return Ok(false) when the group should
/// take the scalar fill (outside the range test, too large, or not worth it).
/// On success `scores[k]` holds the fixed-point score of every member k.
pub(crate) fn score_group(
    call: &Call<'_>,
    views: &[(ByteView<'_>, ByteView<'_>, Option<i32>)],
    members: &[usize],
    scores: &mut [i64],
) -> Result<bool, EngineError> {
    match width() {
        #[cfg(all(target_arch = "x86_64", seqblaze_avx512))]
        16 => match x86::Avx512::new() {
            Some(s) => BUFFERS16
                .with(|cell| group(s, call, views, members, scores, &mut cell.borrow_mut())),
            None => Ok(false),
        },
        #[cfg(target_arch = "x86_64")]
        8 => match x86::Avx2::new() {
            Some(s) => {
                BUFFERS8.with(|cell| group(s, call, views, members, scores, &mut cell.borrow_mut()))
            }
            None => Ok(false),
        },
        #[cfg(target_arch = "aarch64")]
        8 => BUFFERS8.with(|cell| {
            group(
                Portable,
                call,
                views,
                members,
                scores,
                &mut cell.borrow_mut(),
            )
        }),
        _ => Ok(false),
    }
}

fn group<S: Simd<L>, const L: usize>(
    simd: S,
    call: &Call<'_>,
    views: &[(ByteView<'_>, ByteView<'_>, Option<i32>)],
    members: &[usize],
    scores: &mut [i64],
    buffers: &mut Buffers<L>,
) -> Result<bool, EngineError> {
    debug_assert!(!members.is_empty() && members.len() <= L);
    let mut shape = Shape {
        m: [0; L],
        n: [0; L],
        rows: 0,
        cols: 0,
        m_min: usize::MAX,
        n_min: usize::MAX,
    };
    let mut cells = 0u128;
    for (lane, &k) in members.iter().enumerate() {
        let (m, n) = (views[k].0.len(), views[k].1.len());
        debug_assert!(eligible(m, n));
        shape.m[lane] = m;
        shape.n[lane] = n;
        shape.rows = shape.rows.max(m);
        shape.cols = shape.cols.max(n);
        shape.m_min = shape.m_min.min(m);
        shape.n_min = shape.n_min.min(n);
        cells += m as u128 * n as u128;
    }
    let (rows, cols) = (shape.rows, shape.cols);
    // One lane step costs about one scalar cell: a group is worth it when its
    // pairs hold at least one full lane of cells.
    let bytes = (rows + 6 * (cols + 1)).saturating_mul(std::mem::size_of::<[i32; L]>());
    if !call.in_range(rows, cols) || cells < rows as u128 * cols as u128 || bytes > MAX_GROUP_BYTES
    {
        return Ok(false);
    }
    let result = run(simd, call, views, members, &shape, buffers);
    buffers.shrink();
    let lanes = result?;
    for (lane, &k) in members.iter().enumerate() {
        scores[k] = lanes[lane];
    }
    Ok(true)
}

/// Transpose the group's sequences into lane codes and run the fill.
fn run<S: Simd<L>, const L: usize>(
    simd: S,
    call: &Call<'_>,
    views: &[(ByteView<'_>, ByteView<'_>, Option<i32>)],
    members: &[usize],
    shape: &Shape<L>,
    buffers: &mut Buffers<L>,
) -> Result<[i64; L], EngineError> {
    let lookup = call.lookup;
    filled(&mut buffers.at, shape.rows, [PAD_A; L], ROWS, None)?;
    filled(&mut buffers.bt, shape.cols, [PAD_B; L], ROWS, None)?;
    let (mut seen_a, mut seen_b) = (Classes::default(), Classes::default());
    for (lane, &k) in members.iter().enumerate() {
        let (a, b, _) = views[k];
        let Buffers { at, bt, bytes, .. } = &mut *buffers;
        for_bytes(a, bytes, |i, x| {
            let symbol = lookup.symbol(x);
            seen_a.insert(symbol);
            at[i][lane] = symbol;
        })?;
        for_bytes(b, bytes, |j, y| {
            let symbol = lookup.symbol(y);
            seen_b.insert(symbol);
            bt[j][lane] = symbol;
        })?;
    }
    // The group's form: equality on symbols, or a table gather on a codes
    // stride x class(a) and b codes class(b), with padding as class 0.
    let form = match lookup {
        Lookup::Equal {
            matched,
            mismatched,
        } => Form::Equal(Equal(*matched, *mismatched)),
        Lookup::Small { table, .. } => {
            match equal_form(&seen_a, &seen_b, |x, y| table[16 * x + y]) {
                Some((matched, mismatched)) => Form::Equal(Equal(matched, mismatched)),
                None => Form::Small(Small(table)),
            }
        }
        Lookup::Large { table, .. } => {
            match equal_form(&seen_a, &seen_b, |x, y| table[256 * x + y]) {
                Some((matched, mismatched)) => Form::Equal(Equal(matched, mismatched)),
                None => Form::Large(Large(table)),
            }
        }
    };
    let stride = match form {
        Form::Equal(_) => None,
        Form::Small(_) => Some(16),
        Form::Large(_) => Some(256),
    };
    if let Some(stride) = stride {
        let code = |symbol: i32, pad: i32| if symbol == pad { 0 } else { symbol };
        for row in &mut buffers.at[..shape.rows] {
            for x in row.iter_mut() {
                *x = stride * code(*x, PAD_A);
            }
        }
        for row in &mut buffers.bt[..shape.cols] {
            for y in row.iter_mut() {
                *y = code(*y, PAD_B);
            }
        }
    }
    let width = shape.cols + 1;
    filled(&mut buffers.r0, width, [0; L], ROWS, None)?;
    filled(&mut buffers.r1, width, [0; L], ROWS, None)?;
    filled(&mut buffers.r2, width, [0; L], ROWS, None)?;
    filled(&mut buffers.r3, width, [0; L], ROWS, None)?;
    filled(&mut buffers.lim, width, [0; L], ROWS, None)?;
    let g = Gaps32 {
        go: i32::try_from(call.gaps.go).expect("range test"),
        ge: i32::try_from(call.gaps.ge).expect("range test"),
    };
    let rows = Rows {
        at: &buffers.at,
        bt: &buffers.bt,
        r0: &mut buffers.r0,
        r1: &mut buffers.r1,
        r2: &mut buffers.r2,
        r3: &mut buffers.r3,
        lim: &mut buffers.lim,
    };
    let fill = call.fill;
    Ok(match &form {
        Form::Equal(t) => simd.run(fill, t, g, shape, rows),
        Form::Small(t) => simd.run(fill, t, g, shape, rows),
        Form::Large(t) => simd.run(fill, t, g, shape, rows),
    })
}

/// The scoring form a group runs.
enum Form<'t> {
    Equal(Equal),
    Small(Small<'t>),
    Large(Large<'t>),
}

/// Every byte of a row with its index, read once (borrowed NumPy rows through
/// `scratch`, with relaxed loads).
fn for_bytes(
    view: ByteView<'_>,
    scratch: &mut Vec<u8>,
    mut each: impl FnMut(usize, u8),
) -> Result<(), EngineError> {
    if let Some(slice) = view.as_slice() {
        for (k, &x) in slice.iter().enumerate() {
            each(k, x);
        }
        return Ok(());
    }
    scratch.clear();
    crate::scratch::reserve_total(scratch, view.len(), "align input copy", None)?;
    let _ = view.for_each_block(|_, block| {
        scratch.extend_from_slice(block);
        ControlFlow::<()>::Continue(())
    });
    for (k, &x) in scratch.iter().enumerate() {
        each(k, x);
    }
    Ok(())
}

#[derive(Clone, Copy)]
struct Gaps32 {
    go: i32,
    ge: i32,
}

impl Gaps32 {
    /// gamma(L) = go + (L - 1) ge, 0 for L = 0 (the range test keeps it in i32).
    #[inline]
    fn cost(self, len: usize) -> i64 {
        if len == 0 {
            0
        } else {
            i64::from(self.go) + (len as i64 - 1) * i64::from(self.ge)
        }
    }
}

struct Rows<'r, const L: usize> {
    at: &'r [[i32; L]],
    bt: &'r [[i32; L]],
    r0: &'r mut [[i32; L]],
    r1: &'r mut [[i32; L]],
    r2: &'r mut [[i32; L]],
    r3: &'r mut [[i32; L]],
    lim: &'r mut [[i32; L]],
}

// ------------------------------------------------------------------ vectors

/// One lane width's vector of i32 lanes and its operations; arithmetic wraps.
/// `run` enters the fills compiled for the implementation's instruction set.
trait Simd<const L: usize>: Copy {
    type V: Copy;
    fn splat(self, x: i32) -> Self::V;
    fn load(self, x: &[i32; L]) -> Self::V;
    fn store(self, out: &mut [i32; L], v: Self::V);
    fn add(self, x: Self::V, y: Self::V) -> Self::V;
    fn sub(self, x: Self::V, y: Self::V) -> Self::V;
    fn max(self, x: Self::V, y: Self::V) -> Self::V;
    fn min(self, x: Self::V, y: Self::V) -> Self::V;
    fn and(self, x: Self::V, y: Self::V) -> Self::V;
    /// `x` in the lanes where a = b, else `y`.
    fn select_eq(self, a: Self::V, b: Self::V, x: Self::V, y: Self::V) -> Self::V;
    /// `table[idx]` per lane; every index must lie within `table`.
    fn gather(self, table: &[i32], idx: Self::V) -> Self::V;
    fn run<T: Sub>(
        self,
        fill: Fill,
        t: &T,
        g: Gaps32,
        shape: &Shape<L>,
        rows: Rows<'_, L>,
    ) -> [i64; L];
}

/// Plain arrays: any width, any CPU (the compiler may vectorize them). The
/// lane width of aarch64 (NEON); x86_64 uses the explicit forms below.
#[derive(Clone, Copy)]
#[cfg_attr(not(target_arch = "aarch64"), allow(dead_code))]
struct Portable;

impl<const L: usize> Simd<L> for Portable {
    type V = [i32; L];
    #[inline(always)]
    fn splat(self, x: i32) -> [i32; L] {
        [x; L]
    }
    #[inline(always)]
    fn load(self, x: &[i32; L]) -> [i32; L] {
        *x
    }
    #[inline(always)]
    fn store(self, out: &mut [i32; L], v: [i32; L]) {
        *out = v;
    }
    #[inline(always)]
    fn add(self, mut x: [i32; L], y: [i32; L]) -> [i32; L] {
        for l in 0..L {
            x[l] = x[l].wrapping_add(y[l]);
        }
        x
    }
    #[inline(always)]
    fn sub(self, mut x: [i32; L], y: [i32; L]) -> [i32; L] {
        for l in 0..L {
            x[l] = x[l].wrapping_sub(y[l]);
        }
        x
    }
    #[inline(always)]
    fn max(self, mut x: [i32; L], y: [i32; L]) -> [i32; L] {
        for l in 0..L {
            x[l] = x[l].max(y[l]);
        }
        x
    }
    #[inline(always)]
    fn min(self, mut x: [i32; L], y: [i32; L]) -> [i32; L] {
        for l in 0..L {
            x[l] = x[l].min(y[l]);
        }
        x
    }
    #[inline(always)]
    fn and(self, mut x: [i32; L], y: [i32; L]) -> [i32; L] {
        for l in 0..L {
            x[l] &= y[l];
        }
        x
    }
    #[inline(always)]
    fn select_eq(self, a: [i32; L], b: [i32; L], x: [i32; L], y: [i32; L]) -> [i32; L] {
        let mut out = y;
        for l in 0..L {
            if a[l] == b[l] {
                out[l] = x[l];
            }
        }
        out
    }
    #[inline(always)]
    fn gather(self, table: &[i32], idx: [i32; L]) -> [i32; L] {
        let mut out = [0; L];
        for l in 0..L {
            out[l] = table[idx[l] as usize];
        }
        out
    }
    fn run<T: Sub>(
        self,
        fill: Fill,
        t: &T,
        g: Gaps32,
        shape: &Shape<L>,
        rows: Rows<'_, L>,
    ) -> [i64; L] {
        fills(self, fill, t, g, shape, rows)
    }
}

#[cfg(target_arch = "x86_64")]
mod x86 {
    use super::{fills, Fill, Gaps32, Rows, Shape, Simd, Sub};
    use std::arch::x86_64::*;

    /// AVX2, 8 lanes. Only `new` makes one, after detection.
    #[derive(Clone, Copy)]
    pub(super) struct Avx2(());

    impl Avx2 {
        /// None unless the CPU has AVX2.
        pub(super) fn new() -> Option<Self> {
            std::is_x86_feature_detected!("avx2").then_some(Self(()))
        }
    }

    // SAFETY (every `unsafe` block of this impl): an `Avx2` exists only after
    // AVX2 was detected; loads and stores go through references to 8 i32s;
    // `gather`'s indices lie within its table (its contract).
    impl Simd<8> for Avx2 {
        type V = __m256i;
        #[inline(always)]
        fn splat(self, x: i32) -> __m256i {
            unsafe { _mm256_set1_epi32(x) }
        }
        #[inline(always)]
        fn load(self, x: &[i32; 8]) -> __m256i {
            unsafe { _mm256_loadu_si256(x.as_ptr().cast()) }
        }
        #[inline(always)]
        fn store(self, out: &mut [i32; 8], v: __m256i) {
            unsafe { _mm256_storeu_si256(out.as_mut_ptr().cast(), v) }
        }
        #[inline(always)]
        fn add(self, x: __m256i, y: __m256i) -> __m256i {
            unsafe { _mm256_add_epi32(x, y) }
        }
        #[inline(always)]
        fn sub(self, x: __m256i, y: __m256i) -> __m256i {
            unsafe { _mm256_sub_epi32(x, y) }
        }
        #[inline(always)]
        fn max(self, x: __m256i, y: __m256i) -> __m256i {
            unsafe { _mm256_max_epi32(x, y) }
        }
        #[inline(always)]
        fn min(self, x: __m256i, y: __m256i) -> __m256i {
            unsafe { _mm256_min_epi32(x, y) }
        }
        #[inline(always)]
        fn and(self, x: __m256i, y: __m256i) -> __m256i {
            unsafe { _mm256_and_si256(x, y) }
        }
        #[inline(always)]
        fn select_eq(self, a: __m256i, b: __m256i, x: __m256i, y: __m256i) -> __m256i {
            unsafe { _mm256_blendv_epi8(y, x, _mm256_cmpeq_epi32(a, b)) }
        }
        #[inline(always)]
        fn gather(self, table: &[i32], idx: __m256i) -> __m256i {
            unsafe { _mm256_i32gather_epi32::<4>(table.as_ptr(), idx) }
        }
        fn run<T: Sub>(
            self,
            fill: Fill,
            t: &T,
            g: Gaps32,
            shape: &Shape<8>,
            rows: Rows<'_, 8>,
        ) -> [i64; 8] {
            // SAFETY: AVX2 was detected (see the impl).
            unsafe { avx2(self, fill, t, g, shape, rows) }
        }
    }

    #[target_feature(enable = "avx2")]
    unsafe fn avx2<T: Sub>(
        s: Avx2,
        fill: Fill,
        t: &T,
        g: Gaps32,
        shape: &Shape<8>,
        rows: Rows<'_, 8>,
    ) -> [i64; 8] {
        fills(s, fill, t, g, shape, rows)
    }

    /// Compiled only by a rustc with the AVX-512 intrinsics and target
    /// features (stable since 1.89; `build.rs` sets `seqblaze_avx512`). Older
    /// compilers within the workspace's rust-version use the AVX2 fills.
    #[cfg(seqblaze_avx512)]
    #[allow(clippy::incompatible_msrv)]
    mod avx512 {
        use super::super::{fills, Fill, Gaps32, Rows, Shape, Simd, Sub};
        use std::arch::x86_64::*;

        /// AVX-512 F, BW and VL, 16 lanes. Only `new` makes one, after detection.
        #[derive(Clone, Copy)]
        pub(crate) struct Avx512(());

        impl Avx512 {
            /// None unless the CPU has AVX-512 F, BW and VL.
            pub(crate) fn new() -> Option<Self> {
                (std::is_x86_feature_detected!("avx512f")
                    && std::is_x86_feature_detected!("avx512bw")
                    && std::is_x86_feature_detected!("avx512vl"))
                .then_some(Self(()))
            }
        }

        // SAFETY (every `unsafe` block of this impl): as for `Avx2`, with AVX-512.
        impl Simd<16> for Avx512 {
            type V = __m512i;
            #[inline(always)]
            fn splat(self, x: i32) -> __m512i {
                unsafe { _mm512_set1_epi32(x) }
            }
            #[inline(always)]
            fn load(self, x: &[i32; 16]) -> __m512i {
                unsafe { _mm512_loadu_si512(x.as_ptr().cast()) }
            }
            #[inline(always)]
            fn store(self, out: &mut [i32; 16], v: __m512i) {
                unsafe { _mm512_storeu_si512(out.as_mut_ptr().cast(), v) }
            }
            #[inline(always)]
            fn add(self, x: __m512i, y: __m512i) -> __m512i {
                unsafe { _mm512_add_epi32(x, y) }
            }
            #[inline(always)]
            fn sub(self, x: __m512i, y: __m512i) -> __m512i {
                unsafe { _mm512_sub_epi32(x, y) }
            }
            #[inline(always)]
            fn max(self, x: __m512i, y: __m512i) -> __m512i {
                unsafe { _mm512_max_epi32(x, y) }
            }
            #[inline(always)]
            fn min(self, x: __m512i, y: __m512i) -> __m512i {
                unsafe { _mm512_min_epi32(x, y) }
            }
            #[inline(always)]
            fn and(self, x: __m512i, y: __m512i) -> __m512i {
                unsafe { _mm512_and_si512(x, y) }
            }
            #[inline(always)]
            fn select_eq(self, a: __m512i, b: __m512i, x: __m512i, y: __m512i) -> __m512i {
                unsafe { _mm512_mask_blend_epi32(_mm512_cmpeq_epi32_mask(a, b), y, x) }
            }
            #[inline(always)]
            fn gather(self, table: &[i32], idx: __m512i) -> __m512i {
                unsafe { _mm512_i32gather_epi32::<4>(idx, table.as_ptr()) }
            }
            fn run<T: Sub>(
                self,
                fill: Fill,
                t: &T,
                g: Gaps32,
                shape: &Shape<16>,
                rows: Rows<'_, 16>,
            ) -> [i64; 16] {
                // SAFETY: AVX-512 F, BW and VL were detected (see the impl).
                unsafe { avx512(self, fill, t, g, shape, rows) }
            }
        }

        #[target_feature(enable = "avx512f,avx512bw,avx512vl")]
        unsafe fn avx512<T: Sub>(
            s: Avx512,
            fill: Fill,
            t: &T,
            g: Gaps32,
            shape: &Shape<16>,
            rows: Rows<'_, 16>,
        ) -> [i64; 16] {
            fills(s, fill, t, g, shape, rows)
        }
    }

    #[cfg(seqblaze_avx512)]
    pub(super) use avx512::Avx512;
}

// ------------------------------------------------------------------ scoring forms

/// One lookup form of the fills (monomorphized per form).
trait Sub {
    fn score<S: Simd<L>, const L: usize>(&self, s: S, a: S::V, b: S::V) -> S::V;
}

struct Equal(i32, i32);
struct Small<'t>(&'t [i32; 256]);
struct Large<'t>(&'t [i32; 1 << 16]);

impl Sub for Equal {
    #[inline(always)]
    fn score<S: Simd<L>, const L: usize>(&self, s: S, a: S::V, b: S::V) -> S::V {
        s.select_eq(a, b, s.splat(self.0), s.splat(self.1))
    }
}

impl Sub for Small<'_> {
    #[inline(always)]
    fn score<S: Simd<L>, const L: usize>(&self, s: S, a: S::V, b: S::V) -> S::V {
        // Codes keep the sum below 256; the mask keeps every index in the table.
        s.gather(&self.0[..], s.and(s.add(a, b), s.splat(0xff)))
    }
}

impl Sub for Large<'_> {
    #[inline(always)]
    fn score<S: Simd<L>, const L: usize>(&self, s: S, a: S::V, b: S::V) -> S::V {
        s.gather(&self.0[..], s.and(s.add(a, b), s.splat(0xffff)))
    }
}
// ------------------------------------------------------------------ fills

#[inline(always)]
fn fills<S: Simd<L>, T: Sub, const L: usize>(
    s: S,
    fill: Fill,
    t: &T,
    g: Gaps32,
    shape: &Shape<L>,
    rows: Rows<'_, L>,
) -> [i64; L] {
    match fill {
        Fill::Local => local(s, t, g, shape, rows).map(i64::from),
        Fill::Global { end_gaps } => global(s, t, g, end_gaps, shape, rows),
        Fill::Water => water(s, t, g, shape, rows).map(i64::from),
        Fill::Needle { end_gaps } => needle(s, t, g, end_gaps, shape, rows),
    }
}

/// Per-column lane limits: `inside` where column j0 is a cell of the lane's
/// pair, `outside` past its length.
fn column_limits<const L: usize>(
    lim: &mut [[i32; L]],
    shape: &Shape<L>,
    inside: i32,
    outside: i32,
) {
    for (j0, limits) in lim.iter_mut().enumerate() {
        for (limit, &n) in limits.iter_mut().zip(&shape.n) {
            *limit = if j0 < n { inside } else { outside };
        }
    }
}

/// Lane limits of row i0: `inside` where it is a row of the lane's pair.
fn row_limits<const L: usize>(shape: &Shape<L>, i0: usize, inside: i32, outside: i32) -> [i32; L] {
    let mut limits = [outside; L];
    for (limit, &m) in limits.iter_mut().zip(&shape.m) {
        if i0 < m {
            *limit = inside;
        }
    }
    limits
}

// ------------------------------------------------------------------ native

/// `dp::local_fill` without counting, per lane: the largest V of the lane's
/// own cells (at least 0).
#[inline(always)]
fn local<S: Simd<L>, T: Sub, const L: usize>(
    s: S,
    t: &T,
    g: Gaps32,
    shape: &Shape<L>,
    rows: Rows<'_, L>,
) -> [i32; L] {
    let n = shape.cols;
    let (vq, pp, lim) = (&mut rows.r0[..n], &mut rows.r1[..n], &mut rows.lim[..n]);
    vq.fill([NEG32; L]);
    pp.fill([NEG32; L]);
    // A cell outside the lane's pair is capped at -1: it never raises best >= 0.
    column_limits(lim, shape, i32::MAX, -1);
    let (go, ge, zero, neg) = (s.splat(g.go), s.splat(g.ge), s.splat(0), s.splat(NEG32));
    let mut best = zero;
    for (i0, arow) in rows.at.iter().enumerate() {
        let a = s.load(arow);
        let (mut diag, mut vp_left, mut q_left) = (neg, neg, neg);
        // One cell of every lane; with a cap, the cell offers min(V, cap) to best.
        macro_rules! cell {
            ($j0:expr $(, $cap:expr)?) => {{
                let j0 = $j0;
                let v = s.add(t.score(s, a, s.load(&rows.bt[j0])), s.max(diag, zero));
                let (up_vq, up_p) = (s.load(&vq[j0]), s.load(&pp[j0]));
                let p = s.max(s.sub(up_vq, go), s.sub(up_p, ge));
                let q = s.max(s.sub(vp_left, go), s.sub(q_left, ge));
                diag = s.max(up_vq, up_p);
                s.store(&mut vq[j0], s.max(v, q));
                s.store(&mut pp[j0], p);
                vp_left = s.max(v, p);
                q_left = q;
                let offered = v;
                $(let offered = s.min(offered, $cap);)?
                best = s.max(best, offered);
            }};
        }
        let split = if i0 < shape.m_min { shape.n_min } else { 0 };
        for j0 in 0..split {
            cell!(j0);
        }
        let row = s.load(&row_limits(shape, i0, i32::MAX, -1));
        for (j0, cap) in lim.iter().enumerate().skip(split) {
            cell!(j0, s.min(s.load(cap), row));
        }
    }
    let mut out = [0; L];
    s.store(&mut out, best);
    out
}

/// `dp::global_score` per lane: max(V(m, n), T_I(L), T_D(L)) of the lane's
/// own lengths, read from row m's values and column n's.
#[inline(always)]
fn global<S: Simd<L>, T: Sub, const L: usize>(
    s: S,
    t: &T,
    g: Gaps32,
    end_gaps: bool,
    shape: &Shape<L>,
    rows: Rows<'_, L>,
) -> [i64; L] {
    let n = shape.cols;
    let lam = |len: usize| if end_gaps { g.cost(len) } else { 0 };
    let (vq, pp) = (&mut rows.r0[..=n], &mut rows.r1[..=n]);
    // Row m's max(V, P) and V, kept for the lanes whose last row it is.
    let (vp_row, v_row) = (&mut rows.r2[..=n], &mut rows.r3[..=n]);
    vq[0] = [0; L];
    for (j, value) in vq.iter_mut().enumerate().skip(1) {
        *value = [-lam(j) as i32; L];
    }
    pp.fill([NEG32; L]);
    let (go, ge, neg) = (s.splat(g.go), s.splat(g.ge), s.splat(NEG32));
    let mut ti = [i64::MIN / 4; L];
    let mut score = [0i64; L];
    for (i0, arow) in rows.at.iter().enumerate() {
        let i = i0 + 1;
        for (l, ti) in ti.iter_mut().enumerate() {
            if i0 < shape.m[l] {
                *ti = (*ti).max(i64::from(vq[shape.n[l]][l]) - lam(shape.m[l] - i0));
            }
        }
        let a = s.load(arow);
        let mut diag = s.max(s.load(&vq[0]), s.load(&pp[0]));
        let p0 = s.splat(-lam(i) as i32);
        vq[0] = [NEG32; L];
        s.store(&mut pp[0], p0);
        let (mut vp_left, mut q_left) = (p0, neg);
        // One cell of every lane; with `true`, row m's values are kept too.
        macro_rules! cell {
            ($j:expr, $keep:expr) => {{
                let j = $j;
                let v = s.add(t.score(s, a, s.load(&rows.bt[j - 1])), diag);
                let (up_vq, up_p) = (s.load(&vq[j]), s.load(&pp[j]));
                let p = s.max(s.sub(up_vq, go), s.sub(up_p, ge));
                let q = s.max(s.sub(vp_left, go), s.sub(q_left, ge));
                diag = s.max(up_vq, up_p);
                s.store(&mut vq[j], s.max(v, q));
                s.store(&mut pp[j], p);
                vp_left = s.max(v, p);
                q_left = q;
                if $keep {
                    s.store(&mut vp_row[j], vp_left);
                    s.store(&mut v_row[j], v);
                }
            }};
        }
        if !shape.m.contains(&i) {
            for j in 1..=n {
                cell!(j, false);
            }
            continue;
        }
        for j in 1..=n {
            cell!(j, true);
        }
        for l in 0..L {
            if shape.m[l] != i {
                continue;
            }
            let nl = shape.n[l];
            // T_D(n - j) from max(Z, P, V)(m, j) for j < n; at j = 0 that is P(m, 0).
            let mut td = -lam(i) - lam(nl);
            for (j, values) in vp_row.iter().enumerate().take(nl).skip(1) {
                td = td.max(i64::from(values[l]) - lam(nl - j));
            }
            let corner = i64::from(v_row[nl][l]);
            score[l] = corner.max(ti[l]).max(td);
        }
    }
    score
}

// ------------------------------------------------------------------ compat

/// `dp::water_fill` without the trace, per lane: the largest R over the
/// lane's interior cells (i, j >= 2 in 1-based terms: i0, j0 >= 1).
#[inline(always)]
fn water<S: Simd<L>, T: Sub, const L: usize>(
    s: S,
    t: &T,
    g: Gaps32,
    shape: &Shape<L>,
    rows: Rows<'_, L>,
) -> [i32; L] {
    let n = shape.cols;
    let (h, f, lim) = (&mut rows.r0[..n], &mut rows.r1[..n], &mut rows.lim[..n]);
    h.fill([0; L]);
    f.fill([NEG32; L]);
    // Real values are above NEG32; a cell outside the lane's pair is capped at it.
    column_limits(lim, shape, i32::MAX, NEG32);
    let (go, ge, zero, neg) = (s.splat(g.go), s.splat(g.ge), s.splat(0), s.splat(NEG32));
    let mut rmax = neg;
    for (i0, arow) in rows.at.iter().enumerate() {
        let a = s.load(arow);
        let (mut diag, mut e, mut hl) = (zero, neg, zero);
        // One cell of every lane, returning R; the caller offers it to rmax.
        macro_rules! cell {
            ($j0:expr) => {{
                let j0 = $j0;
                let mm = s.add(diag, t.score(s, a, s.load(&rows.bt[j0])));
                let uh = s.load(&h[j0]);
                e = s.max(s.sub(hl, go), s.sub(e, ge));
                let fv = s.max(s.sub(uh, go), s.sub(s.load(&f[j0]), ge));
                let mf = s.max(mm, fv);
                let r = s.max(mf, e);
                diag = uh;
                s.store(&mut h[j0], s.max(r, zero));
                s.store(&mut f[j0], fv);
                hl = s.max(mf, zero);
                r
            }};
        }
        if i0 == 0 {
            for j0 in 0..n {
                cell!(j0);
            }
            continue;
        }
        cell!(0);
        let split = if i0 < shape.m_min { shape.n_min } else { 1 };
        for j0 in 1..split {
            rmax = s.max(rmax, cell!(j0));
        }
        let row = s.load(&row_limits(shape, i0, i32::MAX, NEG32));
        for (j0, cap) in lim.iter().enumerate().skip(split) {
            let r = cell!(j0);
            rmax = s.max(rmax, s.min(r, s.min(s.load(cap), row)));
        }
    }
    let mut out = [0; L];
    s.store(&mut out, rmax);
    out
}

/// `dp::needle_fill` without the trace, per lane: the largest candidate of
/// the lane's own lengths (both at least 2).
#[inline(always)]
fn needle<S: Simd<L>, T: Sub, const L: usize>(
    s: S,
    t: &T,
    g: Gaps32,
    end_gaps: bool,
    shape: &Shape<L>,
    rows: Rows<'_, L>,
) -> [i64; L] {
    let n = shape.cols;
    let lead = |len: usize| if end_gaps { g.cost(len) } else { 0 };
    let (h, f, mm_row) = (&mut rows.r0[..=n], &mut rows.r1[..=n], &mut rows.r2[..=n]);
    h[0] = [0; L];
    for (j, value) in h.iter_mut().enumerate().skip(1) {
        *value = [-lead(j) as i32; L];
    }
    f.fill([NEG32; L]);
    let (go, ge, neg) = (s.splat(g.go), s.splat(g.ge), s.splat(NEG32));
    // Candidates other than C0: with m, n >= 2, H(0, n) - lead(m) and
    // -lead(m) - lead(n) (equal).
    let mut cand = [0i64; L];
    for (l, cand) in cand.iter_mut().enumerate() {
        *cand = -lead(shape.n[l]) - lead(shape.m[l]);
    }
    let mut corner = [i64::MIN / 4; L];
    for (i0, arow) in rows.at.iter().enumerate() {
        let i = i0 + 1;
        let a = s.load(arow);
        let mut diag = s.load(&h[0]);
        let h0 = s.splat(-lead(i) as i32);
        s.store(&mut h[0], h0);
        let (mut e, mut hl) = (neg, h0);
        for j in 1..=n {
            let mm = s.add(diag, t.score(s, a, s.load(&rows.bt[j - 1])));
            let uh = s.load(&h[j]);
            let en = s.max(s.sub(hl, go), s.sub(e, ge));
            let fv = s.max(s.sub(uh, go), s.sub(s.load(&f[j]), ge));
            diag = uh;
            s.store(&mut h[j], s.max(s.max(mm, en), fv));
            s.store(&mut f[j], fv);
            s.store(&mut mm_row[j], mm);
            e = en;
            hl = s.max(mm, fv);
        }
        for l in 0..L {
            let (m, nl) = (shape.m[l], shape.n[l]);
            if i > m {
                continue;
            }
            if i == m {
                corner[l] = i64::from(mm_row[nl][l]);
                for (j, values) in mm_row.iter().enumerate().take(nl).skip(1) {
                    cand[l] = cand[l].max(i64::from(values[l]) - lead(nl - j));
                }
            } else {
                cand[l] = cand[l].max(i64::from(mm_row[nl][l]) - lead(m - i));
            }
        }
    }
    let mut out = [0i64; L];
    for (l, out) in out.iter_mut().enumerate() {
        *out = corner[l].max(cand[l]);
    }
    out
}
