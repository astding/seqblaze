//! The K1 recurrence (docs/wiki/thermo/duplex-mfe.md §5) over per-call cell keys:
//! min-plus dynamic programming in the lexicographic order of the key, with a
//! rolling band of min(n, Λ + 2) rows, where n is the top-strand length. Integer
//! arithmetic only; any evaluation order that respects the row dependence gives
//! the same integers, so the fixed loop order below is one of many equivalent ones.
//!
//! On pack format 2 (docs/wiki/thermo/nndb2024-dna.md §5.3.3) a pair is any opposition
//! with a pair type of the pack (PT ≥ 0), tables are indexed by the pair type P, the
//! generic step reads the loop-mismatch tables of its shape class c = CLS[u][v] (one
//! OUT_c band per class), and a bulge(1) step adds K(STATES[ρ]) of the bulged base's run
//! length ρ as its own addend (K1 only; K3 never reads STATES).
use std::{cell::RefCell, sync::Arc};

use seqblaze_core::{detail::try_reserve_exact, validate::EngineError};

use super::pack::{Pack, Tables, NONE, NO_PAIR};

/// A structure key: D alone (the score pass) or (D, H, G, Q) (the detail pass).
/// `INF` is +∞; a sum that contains +∞ stays above `LIMIT` and is normalised back
/// to `INF` before it is stored, so it never overflows and never reaches a finite
/// value (finite |D| < 2^31, finite |H|, |G|, Q < 2^53; at most four terms add).
pub(crate) trait Key: Copy + Ord + std::fmt::Debug {
    const INF: Self;
    fn plus(self, other: Self) -> Self;
    fn finite(self) -> bool;
    #[inline]
    fn normal(self) -> Self {
        if self.finite() {
            self
        } else {
            Self::INF
        }
    }
}

const INF_D: i64 = 1 << 60;
const LIMIT_D: i64 = 1 << 59;

impl Key for i64 {
    const INF: Self = INF_D;
    #[inline]
    fn plus(self, other: Self) -> Self {
        self + other
    }
    #[inline]
    fn finite(self) -> bool {
        self < LIMIT_D
    }
}

/// K = (D, H, G, Q), ordered lexicographically (field order) (§4.4, §5): the reported
/// key is the lexicographic minimum, value first (provisional: QK10). Structures of
/// one pair that tie are not counted (provisional: QK11).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Default)]
pub struct Full {
    pub d: i64,
    pub h: i64,
    pub g: i64,
    pub q: i64,
}

impl Key for Full {
    const INF: Self = Full {
        d: INF_D,
        h: 0,
        g: 0,
        q: 0,
    };
    #[inline]
    fn plus(self, o: Self) -> Self {
        Full {
            d: self.d + o.d,
            h: self.h + o.h,
            g: self.g + o.g,
            q: self.q + o.q,
        }
    }
    #[inline]
    fn finite(self) -> bool {
        self.d < LIMIT_D
    }
}

/// The cell keys of one call in flat arrays, indexed as the tables of
/// nndb2024-dna.md §5.3.1 (by the pair type P; the last index varies fastest); ⊥ is
/// `K::INF`. A format-1 pack has N_P = 4, N_C = 1 and empty STATES cells, so every index
/// equals the format-1 one (P = the top letter of a Watson–Crick pair, c = 0).
pub(crate) struct Cells<K> {
    pub loop_max: usize,
    /// N_P and N_C.
    pub n_p: usize,
    pub n_c: usize,
    /// PT[x·4 + y], [`NO_PAIR`] for no pair.
    pub pt: [u8; 16],
    /// CLS dense over (Λ + 1)² (0 outside the GEN shapes), shared with the pack.
    pub cls: Arc<[u8]>,
    /// K(INIT), and K(INIT) + K(SYM).
    pub init: K,
    pub init_sym: K,
    pub end_l: Vec<K>,
    pub end_r: Vec<K>,
    pub stack: Vec<K>,
    pub bulge: Vec<K>,
    pub i11: Vec<K>,
    pub i12: Vec<K>,
    pub i21: Vec<K>,
    pub i22: Vec<K>,
    pub gen: Vec<K>,
    pub loop_out: Vec<K>,
    pub loop_in: Vec<K>,
    /// K(STATES[σ]) at σ − 1, σ = 1 … 200.
    pub states: Vec<K>,
    /// Every STATES cell is empty (every format-1 pack): the bulge(1) term is 0 and the
    /// run-length prologue is skipped (nndb2024-dna.md §5.3.3).
    pub no_states: bool,
}

/// The run length ρ(p) of strand `s` at `p` (nndb2024-dna.md §3.6, §5.3.3): the length of
/// the maximal run of identical letters containing p, by a direct scan both ways.
#[inline]
pub(crate) fn run_at(s: &[u8], p: usize) -> usize {
    let x = s[p];
    let lo = s[..p].iter().rev().take_while(|&&c| c == x).count();
    let hi = s[p + 1..].iter().take_while(|&&c| c == x).count();
    lo + hi + 1
}

/// ρ of every position of `s` into `out` (one O(n) pass); ρ ≤ n ≤ 200 fits a byte.
pub(crate) fn run_lengths(s: &[u8], out: &mut [u8; MAX_RUN]) {
    let n = s.len();
    let mut start = 0;
    while start < n {
        let mut end = start + 1;
        while end < n && s[end] == s[start] {
            end += 1;
        }
        let len = (end - start) as u8;
        out[start..end].fill(len);
        start = end;
    }
}

/// The longest strand (D48), so every run length fits [`MAX_RUN`] and indexes STATES.
pub(crate) const MAX_RUN: usize = 200;

impl<K: Key> Cells<K> {
    pub(crate) fn new(pack: &Pack, keys: Tables<K>) -> Self {
        let no_states = pack
            .tables
            .states
            .iter()
            .all(|cell| matches!(cell.as_deref(), Some([])));
        Self {
            loop_max: pack.loop_max,
            n_p: pack.n_p,
            n_c: pack.n_c,
            pt: pack.pt,
            cls: Arc::clone(&pack.cls),
            init: keys.init,
            init_sym: keys.init.plus(keys.sym).normal(),
            end_l: keys.end_l,
            end_r: keys.end_r,
            stack: keys.stack,
            bulge: keys.bulge,
            i11: keys.i11,
            i12: keys.i12,
            i21: keys.i21,
            i22: keys.i22,
            gen: keys.gen,
            loop_out: keys.loop_out,
            loop_in: keys.loop_in,
            states: keys.states,
            no_states,
        }
    }

    /// P of the opposition (x, y), or [`NO_PAIR`].
    #[inline(always)]
    pub(crate) fn pair(&self, x: u8, y: u8) -> u8 {
        self.pt[usize::from(x) * 4 + usize::from(y)]
    }

    /// K of the step (k, l) → (i, j) with u = i − k − 1, v = l − j − 1 (§5 STEP,
    /// nndb2024-dna.md §5.3.3), STATES included at a bulge(1), for the evaluator and the
    /// register walk; the recurrence inlines the same lookups. Both pairs must be pairs.
    pub(crate) fn step(
        &self,
        a: &[u8],
        b: &[u8],
        (k, l): (usize, usize),
        (i, j): (usize, usize),
    ) -> K {
        let (u, v) = (i - k - 1, l - j - 1);
        let np = self.n_p;
        let x = usize::from(self.pair(a[k], b[l]));
        let y = usize::from(self.pair(a[i], b[j]));
        let lam = self.loop_max;
        let at = |p: usize| usize::from(a[p]);
        let bt = |p: usize| usize::from(b[p]);
        let states = |key: K, strand: &[u8], p: usize| {
            if self.no_states {
                key
            } else {
                key.plus(self.states[run_at(strand, p) - 1])
            }
        };
        match (u, v) {
            (0, 0) => self.stack[x * np + y],
            (u, 0) => {
                let key = self.bulge[((u - 1) * np + x) * np + y];
                if u == 1 {
                    states(key, a, k + 1)
                } else {
                    key
                }
            }
            (0, v) => {
                let key = self.bulge[((lam + v - 1) * np + x) * np + y];
                if v == 1 {
                    states(key, b, l - 1)
                } else {
                    key
                }
            }
            (1, 1) => self.i11[((x * 4 + at(k + 1)) * 4 + bt(l - 1)) * np + y],
            (1, 2) => self.i12[(((x * 4 + at(k + 1)) * 4 + bt(l - 1)) * 4 + bt(l - 2)) * np + y],
            (2, 1) => self.i21[(((x * 4 + at(k + 1)) * 4 + at(k + 2)) * 4 + bt(l - 1)) * np + y],
            (2, 2) => {
                self.i22[((((x * 4 + at(k + 1)) * 4 + at(k + 2)) * 4 + bt(l - 1)) * 4 + bt(l - 2))
                    * np
                    + y]
            }
            (u, v) => {
                let c = usize::from(self.cls[u * (lam + 1) + v]);
                self.gen[u * (lam + 1) + v]
                    .plus(self.loop_out[((c * np + x) * 4 + at(k + 1)) * 4 + bt(l - 1)])
                    .plus(self.loop_in[((c * 4 + at(i - 1)) * 4 + bt(j + 1)) * np + y])
            }
        }
    }

    /// K(END_L[φ(i−1)][P_ij][ψ(j+1)]).
    #[inline]
    pub(crate) fn left_end(&self, a: &[u8], b: &[u8], i: usize, j: usize) -> K {
        let alpha = if i >= 1 { usize::from(a[i - 1]) } else { NONE };
        let beta = if j + 1 < b.len() {
            usize::from(b[j + 1])
        } else {
            NONE
        };
        let p = usize::from(self.pair(a[i], b[j]));
        self.end_l[(alpha * self.n_p + p) * 5 + beta]
    }

    /// K(END_R[P_ij][φ(i+1)][ψ(j−1)]).
    #[inline]
    pub(crate) fn right_end(&self, a: &[u8], b: &[u8], i: usize, j: usize) -> K {
        let alpha = if i + 1 < a.len() {
            usize::from(a[i + 1])
        } else {
            NONE
        };
        let beta = if j >= 1 { usize::from(b[j - 1]) } else { NONE };
        let p = usize::from(self.pair(a[i], b[j]));
        self.end_r[(p * 5 + alpha) * 5 + beta]
    }
}

/// Per-thread band of one key type: A and A + K(LOOP_OUT) for Λ + 2 rows.
pub(crate) struct Band<K> {
    a: Vec<K>,
    out: Vec<K>,
}

impl<K> Default for Band<K> {
    fn default() -> Self {
        Self {
            a: Vec::new(),
            out: Vec::new(),
        }
    }
}

/// Buffers above this size are released after a pair.
const KEEP_BYTES: usize = 1 << 20;

#[derive(Default)]
pub(crate) struct Scratch {
    pub score: Band<i64>,
    pub full: Band<Full>,
}

thread_local! {
    pub(crate) static SCRATCH: RefCell<Scratch> = RefCell::new(Scratch::default());
}

impl<K: Key> Band<K> {
    fn prepare(&mut self, len: usize, classes: usize) -> Result<(), EngineError> {
        for (buffer, len) in [(&mut self.a, len), (&mut self.out, len * classes)] {
            buffer.clear();
            try_reserve_exact(buffer, len, "duplex DP band")?;
            buffer.resize(len, K::INF);
        }
        Ok(())
    }

    fn release(&mut self) {
        for buffer in [&mut self.a, &mut self.out] {
            if buffer.capacity() * std::mem::size_of::<K>() > KEEP_BYTES {
                *buffer = Vec::new();
            }
        }
    }
}

/// K* of §5 for the letter codes `a`, `b` (0..4, at most 200 nt each), with `start` =
/// K(INIT) or K(INIT) + K(SYM); `K::INF` when the structure space is empty.
pub(crate) fn solve<K: Key>(
    cells: &Cells<K>,
    start: K,
    a: &[u8],
    b: &[u8],
    band: &mut Band<K>,
) -> Result<K, EngineError> {
    let (n, m) = (a.len(), b.len());
    if n < 2 || m < 2 {
        return Ok(K::INF); // k >= 2 needs two pairs (QK8)
    }
    let lam = cells.loop_max;
    let rows = n.min(lam + 2);
    band.prepare(rows * m, cells.n_c)?;
    let result = fill(cells, start, a, b, band, rows);
    band.release();
    Ok(result)
}

/// The largest number of loop classes (nndb2024-dna.md §4.4: N_C is 1 or 3).
const MAX_CLASSES: usize = 3;

#[inline(always)]
fn fill<K: Key>(
    cells: &Cells<K>,
    start: K,
    a: &[u8],
    b: &[u8],
    band: &mut Band<K>,
    rows: usize,
) -> K {
    let (n, m) = (a.len(), b.len());
    let lam = cells.loop_max;
    let width = lam + 1;
    let (np, nc) = (cells.n_p, cells.n_c);
    debug_assert!(nc <= MAX_CLASSES && n <= MAX_RUN && m <= MAX_RUN);
    // Run lengths of both strands (nndb2024-dna.md §5.3.3), skipped without STATES.
    let (mut rho_a, mut rho_b) = ([0u8; MAX_RUN], [0u8; MAX_RUN]);
    let states = !cells.no_states;
    if states {
        run_lengths(a, &mut rho_a);
        run_lengths(b, &mut rho_b);
    }
    let mut best = K::INF;
    let mut cur = 0;
    for i in 0..n {
        let ai = usize::from(a[i]);
        let base = cur * m;
        for j in 0..m {
            // pair(i, j) iff PT[a_i][b_j] >= 0 (nndb2024-dna.md §5.3.1).
            let pi = cells.pt[ai * 4 + usize::from(b[j])];
            if pi == NO_PAIR {
                band.a[base + j] = K::INF;
                band.out[(base + j) * nc..(base + j + 1) * nc].fill(K::INF);
                continue;
            }
            let pi = usize::from(pi);
            // C(i, j): predecessors (k, l) with u = i − k − 1, v = l − j − 1, u + v <= Λ.
            // A finite A(k, l) implies that (k, l) is a pair, so P_kl is read only then.
            let mut c = K::INF;
            if i >= 1 && j + 1 < m {
                // IN_c(i, j) once per class (§5.3.3, per-class reuse).
                let mut inner = [K::INF; MAX_CLASSES];
                let in_base = usize::from(a[i - 1]) * 4 + usize::from(b[j + 1]);
                for (cl, slot) in inner.iter_mut().enumerate().take(nc) {
                    *slot = cells.loop_in[(cl * 16 + in_base) * np + pi];
                }
                let umax = lam.min(i - 1);
                // Ring slot of k = i - 1 - u, walked backwards without division.
                let mut slot = if cur == 0 { rows - 1 } else { cur - 1 };
                for u in 0..=umax {
                    let k = i - 1 - u;
                    let ak = usize::from(a[k]) * 4;
                    let row = slot * m;
                    let vmax = (lam - u).min(m - 2 - j);
                    // v = 0: a stack or a top bulge.
                    let prev = band.a[row + j + 1];
                    if prev.finite() {
                        let pk = usize::from(cells.pt[ak + usize::from(b[j + 1])]);
                        let cand = if u == 0 {
                            prev.plus(cells.stack[pk * np + pi])
                        } else {
                            let step = prev.plus(cells.bulge[((u - 1) * np + pk) * np + pi]);
                            if u == 1 && states {
                                step.plus(cells.states[usize::from(rho_a[k + 1]) - 1])
                            } else {
                                step
                            }
                        };
                        if cand < c {
                            c = cand;
                        }
                    }
                    for v in 1..=vmax {
                        let l = j + 1 + v;
                        let step = match u {
                            0 => {
                                let prev = band.a[row + l];
                                if !prev.finite() {
                                    continue;
                                }
                                let pk = usize::from(cells.pt[ak + usize::from(b[l])]);
                                let step =
                                    prev.plus(cells.bulge[((lam + v - 1) * np + pk) * np + pi]);
                                if v == 1 && states {
                                    step.plus(cells.states[usize::from(rho_b[l - 1]) - 1])
                                } else {
                                    step
                                }
                            }
                            1 | 2 if v <= 2 => {
                                let prev = band.a[row + l];
                                if !prev.finite() {
                                    continue;
                                }
                                let pk = usize::from(cells.pt[ak + usize::from(b[l])]);
                                let p1 = usize::from(a[k + 1]);
                                let q1 = usize::from(b[l - 1]);
                                let cell = match (u, v) {
                                    (1, 1) => cells.i11[((pk * 4 + p1) * 4 + q1) * np + pi],
                                    (1, _) => {
                                        let q2 = usize::from(b[l - 2]);
                                        cells.i12[(((pk * 4 + p1) * 4 + q1) * 4 + q2) * np + pi]
                                    }
                                    (_, 1) => {
                                        let p2 = usize::from(a[k + 2]);
                                        cells.i21[(((pk * 4 + p1) * 4 + p2) * 4 + q1) * np + pi]
                                    }
                                    _ => {
                                        let p2 = usize::from(a[k + 2]);
                                        let q2 = usize::from(b[l - 2]);
                                        cells.i22[((((pk * 4 + p1) * 4 + p2) * 4 + q1) * 4 + q2)
                                            * np
                                            + pi]
                                    }
                                };
                                prev.plus(cell)
                            }
                            _ => {
                                let cl = usize::from(cells.cls[u * width + v]);
                                let prev = band.out[(row + l) * nc + cl];
                                if !prev.finite() {
                                    continue;
                                }
                                prev.plus(cells.gen[u * width + v]).plus(inner[cl])
                            }
                        };
                        if step < c {
                            c = step;
                        }
                    }
                    slot = if slot == 0 { rows - 1 } else { slot - 1 };
                }
            }
            let c = c.normal();
            let first = start.plus(cells.left_end(a, b, i, j)).normal();
            let value = if c < first { c } else { first };
            band.a[base + j] = value;
            let out = &mut band.out[(base + j) * nc..(base + j + 1) * nc];
            if value.finite() && i + 1 < n && j >= 1 {
                // OUT_c(i, j) = A(i, j) + K(LOOP_OUT[c][P_ij][a_{i+1}][b_{j−1}]).
                let tail = usize::from(a[i + 1]) * 4 + usize::from(b[j - 1]);
                for (cl, slot) in out.iter_mut().enumerate() {
                    *slot = value
                        .plus(cells.loop_out[(cl * np + pi) * 16 + tail])
                        .normal();
                }
            } else {
                out.fill(K::INF);
            }
            if c.finite() {
                let total = c.plus(cells.right_end(a, b, i, j));
                if total < best {
                    best = total;
                }
            }
        }
        cur += 1;
        if cur == rows {
            cur = 0;
        }
    }
    best.normal()
}
