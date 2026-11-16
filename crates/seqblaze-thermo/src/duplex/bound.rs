//! The C1 admissible bound of K1 native `duplex(model="mfe")` under `strategy="bounded"`
//! (docs/wiki/thermo/duplex-mfe-bound.md): a relaxed integer K1 recurrence with an exact
//! window of half-width [`U_E`] and exact bulges, in which every generic loop outside the
//! window is relaxed per loop class over skew-staircase regions with per-call integer
//! constants α, β, γ (§3), and its per-call certificate (§3.3, §4.1, §5).
//!
//! LB ≤ K*.D holds by the certificate's finite local checks L1–L5 plus the row induction
//! of §4.2; the exact-path flag χ (§3.5, (min, ∨) over ties) certifies LB = K*.D (§4.3).
//! Integer arithmetic only; every value is a pure function of the pair's bytes and the
//! per-call D cell table, so it does not depend on threads, backend or evaluation order.
use std::cell::RefCell;

use seqblaze_core::{
    detail::try_reserve_exact,
    kernel::BoundOutcome,
    plan::{bound_fault, BoundFault},
    validate::EngineError,
};

use super::{
    dp::{run_lengths, Cells, Key, MAX_RUN},
    pack::{self, NO_PAIR},
    DuplexCall, SENTINEL_KEY,
};

/// The exact window half-width U_E (§3.1, QB2): a kernel constant, never a parameter.
pub const U_E: usize = 4;
/// The shapes with their own tables (I11, I12, I21, I22), never generic (§3.2).
const SMALL: [(usize, usize); 4] = [(1, 1), (1, 2), (2, 1), (2, 2)];
/// The width limit of L4 (§4.6): W < 2^56.
const W_LIMIT: i128 = 1 << 56;

/// +∞ of the packed recurrence and its normalisation threshold (as `dp.rs`).
const INF: i64 = 1 << 60;
const LIMIT: i64 = 1 << 59;

#[inline(always)]
fn finite(x: i64) -> bool {
    x < LIMIT
}

/// One used region of a class (§3.2): lower L(d0) = {u − v ≥ d0, v ≥ start} or upper
/// U(d0) = {v − u ≥ d0, u ≥ start}, with its constant γ (§3.3).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Region {
    pub lower: bool,
    pub d0: usize,
    pub start: usize,
    pub gamma: i64,
}

/// S_c of one class as (u, v, GEN(u, v)) sorted by (s, d, u); `None` for a class holding
/// a ⊥ GEN (§3.2).
pub(crate) type ShapeList = Option<Vec<(usize, usize, i128)>>;

/// The constants of one loop class: α, β and the used regions, lower d0 = U_E … 0 then
/// upper d0 = U_E … 1. A class with an empty S_c has α = β = 0 and no regions.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ClassBound {
    pub alpha: i64,
    pub beta: i64,
    pub regions: Vec<Region>,
}

/// A relaxed class as the recurrence reads it: the slopes and the region queries.
#[derive(Debug, Clone)]
struct Relaxed {
    class: usize,
    /// α + β and α − β.
    plus: i64,
    minus: i64,
    /// Lower regions: (d0 + start, start, γ); upper regions: (start, d0 + start, γ).
    lower: Vec<(usize, usize, i64)>,
    upper: Vec<(usize, usize, i64)>,
}

/// The per-call certificate (§5): the constants of §3.3, W of §4.6 and the outcome of the
/// checks L1–L5. A pure function of the pack and the per-call D cell table.
#[derive(Debug, Clone)]
pub struct Certificate {
    pub pass: bool,
    /// The failed checks (empty iff `pass`); names and shapes only, never a value.
    pub failures: Vec<String>,
    /// W of §4.6 (unspecified when a constant overflowed).
    pub w: i128,
    pub classes: Vec<ClassBound>,
    relaxed: Vec<Relaxed>,
}

impl Certificate {
    /// The representation §4.6 chooses with flag packing: 32 iff 2W + 1 < 2^26.
    pub fn width(&self) -> u8 {
        if 2 * self.w + 1 < (1 << 26) {
            32
        } else {
            64
        }
    }

    /// Whether any class has relaxed edges (else the bound is exact, §4.4 "Every Λ").
    pub fn relaxes(&self) -> bool {
        !self.relaxed.is_empty()
    }

    /// The relaxed classes as device words (duplex-mfe-bound.md §8; mirrored by
    /// `SB_BND_CLASS_WORDS` in kernels/duplex_bound.cuh): per class 32 words, class,
    /// α + β, α − β, the lower count and five (d0 + start, start, γ), the upper count and
    /// four (start, d0 + start, γ). Empty for a failed certificate.
    pub fn device_words(&self) -> (Vec<i64>, usize) {
        let mut words = Vec::with_capacity(32 * self.relaxed.len());
        for r in &self.relaxed {
            let mut block = [0i64; 32];
            block[0] = r.class as i64;
            block[1] = r.plus;
            block[2] = r.minus;
            block[3] = r.lower.len() as i64;
            for (k, &(depth, start, gamma)) in r.lower.iter().enumerate() {
                block[4 + 3 * k] = depth as i64;
                block[5 + 3 * k] = start as i64;
                block[6 + 3 * k] = gamma;
            }
            block[19] = r.upper.len() as i64;
            for (k, &(start, reach, gamma)) in r.upper.iter().enumerate() {
                block[20 + 3 * k] = start as i64;
                block[21 + 3 * k] = reach as i64;
                block[22 + 3 * k] = gamma;
            }
            words.extend_from_slice(&block);
        }
        (words, self.relaxed.len())
    }

    /// Build and check the certificate of `call` (§3.3, §4.1). The `gamma_plus_one`
    /// testing fault raises every γ by 1 before the check, which must then fail. Every
    /// allocation that grows with Λ is reserved fallibly: a refused reservation returns
    /// `Err` (never an abort) and leaves no certificate; the binding maps it to an
    /// unavailable bound (`no_bound`, §5 *Refused reservation*). It is the only `Err`.
    pub fn build(call: &DuplexCall) -> Result<Self, EngineError> {
        seqblaze_core::plan::count_certificate_build();
        let shapes = Self::shapes(call)?;
        let mut cert = Self::constants(call, &shapes);
        if bound_fault() == BoundFault::GammaPlusOne {
            for class in &mut cert.classes {
                for region in &mut class.regions {
                    region.gamma = region.gamma.saturating_add(1);
                }
            }
        }
        cert.check(call, &shapes)?;
        if cert.pass {
            cert.relaxed = cert
                .classes
                .iter()
                .enumerate()
                .filter(|(_, class)| !class.regions.is_empty())
                .map(|(c, class)| Relaxed {
                    class: c,
                    plus: class.alpha + class.beta,
                    minus: class.alpha - class.beta,
                    lower: class
                        .regions
                        .iter()
                        .filter(|r| r.lower)
                        .map(|r| (r.d0 + r.start, r.start, r.gamma))
                        .collect(),
                    upper: class
                        .regions
                        .iter()
                        .filter(|r| !r.lower)
                        .map(|r| (r.start, r.d0 + r.start, r.gamma))
                        .collect(),
                })
                .collect();
        }
        Ok(cert)
    }

    /// Whether (u, v) is a generic shape outside the window (§3.2).
    #[inline]
    fn generic(lam: usize, u: usize, v: usize) -> bool {
        u >= 1 && v >= 1 && u + v <= lam && !SMALL.contains(&(u, v)) && u.max(v) > U_E
    }

    /// S_c of §3.2 per class, as (u, v, GEN(u, v)); `None` for a class holding a ⊥ GEN.
    /// Each list is reserved exactly (a counting pass first) and sorted by (s, d, u), a
    /// total order, so that §3.3's equal-size pairs are contiguous runs.
    pub(crate) fn shapes(call: &DuplexCall) -> Result<Vec<ShapeList>, EngineError> {
        let cells = &call.score;
        let lam = cells.loop_max;
        let width = lam + 1;
        let mut count = vec![0usize; cells.n_c];
        for &(u, v) in &call.pack.gen_shapes {
            if Self::generic(lam, u, v) {
                if let Some(k) = count.get_mut(usize::from(cells.cls[u * width + v])) {
                    *k += 1;
                }
            }
        }
        let mut out: Vec<ShapeList> = Vec::new();
        try_reserve_exact(&mut out, cells.n_c, "duplex bound certificate")?;
        for &k in &count {
            let mut list = Vec::new();
            try_reserve_exact(&mut list, k, "duplex bound certificate")?;
            out.push(Some(list));
        }
        for &(u, v) in &call.pack.gen_shapes {
            if Self::generic(lam, u, v) {
                let c = usize::from(cells.cls[u * width + v]);
                let g = cells.gen[u * width + v];
                match out.get_mut(c) {
                    Some(Some(list)) if g.finite() => list.push((u, v, i128::from(g))),
                    Some(slot) => *slot = None,
                    None => {}
                }
            }
        }
        for list in out.iter_mut().flatten() {
            list.sort_unstable_by_key(|&(u, v, _)| (u + v, u.abs_diff(v), u));
        }
        Ok(out)
    }

    /// The constants of §3.3 and W of §4.6, unchecked, from the size-sorted S_c.
    fn constants(call: &DuplexCall, shapes: &[ShapeList]) -> Self {
        let n_c = call.score.n_c;
        let mut classes = vec![ClassBound::default(); n_c];
        let mut wide = false;
        let fits = |x: i128| -> Option<i64> { i64::try_from(x).ok() };
        for (c, shapes) in shapes.iter().enumerate() {
            let Some(shapes) = shapes.as_ref().filter(|list| !list.is_empty()) else {
                continue;
            };
            let (alpha, beta) = class_slopes(shapes);
            let v_lo = shapes.iter().map(|x| x.1).min().unwrap_or(0);
            let u_lo = shapes.iter().map(|x| x.0).min().unwrap_or(0);
            let mut regions = Vec::new();
            let candidates = (0..=U_E)
                .rev()
                .map(|d0| (true, d0, (U_E + 1 - d0).max(v_lo)))
                .chain(
                    (1..=U_E)
                        .rev()
                        .map(|d0| (false, d0, (U_E + 1 - d0).max(u_lo))),
                );
            for (lower, d0, start) in candidates {
                let (a, b) = if lower {
                    (alpha + beta, alpha - beta)
                } else {
                    (alpha - beta, alpha + beta)
                };
                let gamma = shapes
                    .iter()
                    .filter(|&&(u, v, _)| contains(lower, d0, start, u, v))
                    .map(|&(u, v, g)| g - a * u as i128 - b * v as i128)
                    .min();
                if let Some(gamma) = gamma {
                    let gamma = fits(gamma).unwrap_or_else(|| {
                        wide = true;
                        0
                    });
                    regions.push(Region {
                        lower,
                        d0,
                        start,
                        gamma,
                    });
                }
            }
            let (Some(alpha), Some(beta)) = (fits(alpha), fits(beta)) else {
                wide = true;
                continue;
            };
            classes[c] = ClassBound {
                alpha,
                beta,
                regions,
            };
        }
        let mut cert = Self {
            pass: true,
            failures: Vec::new(),
            w: 0,
            classes,
            relaxed: Vec::new(),
        };
        if wide {
            cert.fail("int64 overflow in a certificate constant".into());
        }
        cert
    }

    fn fail(&mut self, what: String) {
        self.pass = false;
        self.failures.push(what);
    }

    /// L1–L5 of §4.1 and the int64 range (§5 "Failure") on the constants as they stand
    /// (after any testing fault), and W of §4.6. At most [`MAX_FAILURES`] failures are
    /// named; the rest are counted in one closing entry (the verdict counts them all).
    fn check(&mut self, call: &DuplexCall, shape_lists: &[ShapeList]) -> Result<(), EngineError> {
        let mut named = Failures::default();
        pack_facts(call, self.classes.len(), &mut named)?;
        for (c, list) in shape_lists.iter().enumerate() {
            let (Some(list), Some(class)) = (list, self.classes.get(c)) else {
                continue;
            };
            // L3.
            if class.alpha < 0 || class.beta < 0 {
                named.push(|| format!("L3: class {c}"));
            }
            let (alpha, beta) = (i128::from(class.alpha), i128::from(class.beta));
            for r in &class.regions {
                let floor = (U_E + 1).saturating_sub(r.d0);
                if r.start < floor || r.start < 1 || (!r.lower && r.d0 < 1) {
                    named.push(|| format!("region: class {c} d0 {} meets the window", r.d0));
                }
            }
            for &(u, v, g) in list {
                let mut held = false;
                for r in class
                    .regions
                    .iter()
                    .filter(|r| contains(r.lower, r.d0, r.start, u, v))
                {
                    held = true;
                    let (a, b) = if r.lower {
                        (alpha + beta, alpha - beta)
                    } else {
                        (alpha - beta, alpha + beta)
                    };
                    // L2.
                    if i128::from(r.gamma) + a * u as i128 + b * v as i128 > g {
                        named.push(|| format!("L2: class {c} shape ({u}, {v})"));
                    }
                }
                // L1.
                if !held {
                    named.push(|| format!("L1: class {c} shape ({u}, {v}) in no used region"));
                }
            }
        }
        for what in named.finish() {
            self.fail(what);
        }
        // W of §4.6 and L4.
        let (_, max_step, _) =
            pack::pack_bound_parts(&call.pack, |id| i128::from(call.energies[id]));
        let gamma_max = self
            .classes
            .iter()
            .flat_map(|class| class.regions.iter())
            .map(|r| i128::from(r.gamma).abs())
            .max()
            .unwrap_or(0);
        let a_star = self
            .classes
            .iter()
            .filter(|class| !class.regions.is_empty())
            .map(|class| i128::from(class.alpha) + i128::from(class.beta))
            .max()
            .unwrap_or(0);
        let b_rel = i128::from(call.bound) + 199 * gamma_max + 396 * a_star;
        self.w = b_rel + max_step + 400 * a_star;
        if i64::try_from(self.w).is_err() || i64::try_from(b_rel).is_err() {
            self.fail("int64 overflow in a certificate constant".into());
        }
        if self.w >= W_LIMIT {
            self.fail("L4: W >= 2^56".into());
        }
        Ok(())
    }

    /// (LB, χ) of the pair of letter codes (`a` on top), §3.5–§3.6: `None` for +∞. Only
    /// for a passing certificate. The `lb_plus_one` testing fault adds 1 to a finite LB.
    pub fn evaluate(
        &self,
        call: &DuplexCall,
        a: &[u8],
        b: &[u8],
    ) -> Result<(Option<i64>, bool), EngineError> {
        let (lb, chi) = if a.len() < 2 || b.len() < 2 {
            (None, true)
        } else {
            SCRATCH
                .with(|cell| relaxed(&call.score, &self.relaxed, a, b, &mut cell.borrow_mut()))?
        };
        let lb = match (lb, bound_fault()) {
            (Some(lb), BoundFault::LbPlusOne) => Some(lb + 1),
            (lb, _) => lb,
        };
        Ok((lb, chi))
    }

    /// The outcome of §1 against θ (an engine key), from (LB, χ): pruned iff λ > θ
    /// (strict), else certified with λ iff χ = 1, else open. λ = min(LB, the sentinel key),
    /// the sentinel key for +∞ (see [`outcome_of`]).
    pub fn outcome(
        &self,
        call: &DuplexCall,
        a: &[u8],
        b: &[u8],
        threshold: Option<i32>,
    ) -> Result<BoundOutcome<i32>, EngineError> {
        let (lb, chi) = self.evaluate(call, a, b)?;
        Ok(outcome_of(lb, chi, threshold))
    }
}

/// §1's outcome of (LB, χ) against θ.
///
/// λ = min(LB, INT32_MAX − 1): a pair's exact engine key is either K*.D ≤ B ≤ 2³¹ − 3 or the
/// empty-space key INT32_MAX − 1, never above it, so a finite LB above that key (the int64
/// width of §4.6) bounds the engine key only after the clamp. Without it, a pair with no
/// structure would be pruned against a replay limit equal to the empty-space key, which it
/// ties.
pub fn outcome_of(lb: Option<i64>, chi: bool, threshold: Option<i32>) -> BoundOutcome<i32> {
    let sentinel = i64::from(SENTINEL_KEY);
    let lambda = lb.map_or(sentinel, |lb| lb.min(sentinel));
    if threshold.is_some_and(|theta| lambda > i64::from(theta)) {
        BoundOutcome::Pruned
    } else if chi && lb.is_none_or(|lb| lb <= sentinel) {
        // A certified λ is K*.D (or the sentinel), which the range check keeps in int32. A
        // finite LB above the sentinel cannot be exact (G2, §4.6); such a pair stays open.
        match i32::try_from(lambda) {
            Ok(key) => BoundOutcome::Certified(key),
            Err(_) => BoundOutcome::Open,
        }
    } else {
        BoundOutcome::Open
    }
}

/// α_c and β_c of §3.3 from a nonempty S_c sorted by (s, d, u) (as [`Certificate::shapes`]
/// returns it): β = max(0, ⌊least slope over equal-size pairs⌋) and α = max(0, ⌊chord of
/// base over s⌋). Shared with K3's C2 bound (duplex-pf-bound.md §3.3, QZ3), which uses the
/// same rule on the same S_c.
pub(crate) fn class_slopes(shapes: &[(usize, usize, i128)]) -> (i128, i128) {
    let s_of = |&(u, v, _): &(usize, usize, i128)| u + v;
    let d_of = |&(u, v, _): &(usize, usize, i128)| u.abs_diff(v);
    // β: the smallest floor slope over pairs of equal size, clipped at 0. The floor is
    // monotone, so the minimum of the floors is the floor of the minimum exact slope, and
    // `min_slope` finds that over each size run (sorted by d) from consecutive distinct d
    // only. The pair set is §3.3's, so β is the same number; slopes compare exactly by
    // cross-multiplication.
    let mut slope: Option<(i128, i128)> = None;
    for run in shapes.chunk_by(|x, y| s_of(x) == s_of(y)) {
        let points = run.chunk_by(|x, y| d_of(x) == d_of(y)).map(|group| {
            let (lo, hi) = group.iter().fold((i128::MAX, i128::MIN), |(lo, hi), x| {
                (lo.min(x.2), hi.max(x.2))
            });
            (d_of(&group[0]) as i128, lo, hi)
        });
        if let Some((p, q)) = min_slope(points) {
            if slope.is_none_or(|(bp, bq)| p * bq < bp * q) {
                slope = Some((p, q));
            }
        }
    }
    let beta = slope.map_or(0, |(p, q)| p.div_euclid(q).max(0));
    // base(s) at the extreme sizes and α: the floor slope between them, clipped at 0
    // (only base(s_min) and base(s_max) enter α).
    let base = |run: &[(usize, usize, i128)]| {
        run.iter()
            .map(|x| x.2 - beta * d_of(x) as i128)
            .min()
            .unwrap_or(0)
    };
    let mut runs = shapes.chunk_by(|x, y| s_of(x) == s_of(y));
    let first = runs.next().unwrap_or(&[]);
    let last = runs.next_back().unwrap_or(first);
    if first.is_empty() {
        return (0, 0);
    }
    let (lo, hi) = (
        (s_of(&first[0]) as i128, base(first)),
        (s_of(&last[0]) as i128, base(last)),
    );
    let alpha = if hi.0 == lo.0 {
        0
    } else {
        (hi.1 - lo.1).div_euclid(hi.0 - lo.0).max(0)
    };
    (alpha, beta)
}

/// The least slope (g_min(x) − g_max(y)) / (d(x) − d(y)) over points x, y with
/// d(x) > d(y), as an unreduced fraction (numerator, denominator > 0); `None` with fewer
/// than two points. `points` yields (d, g_min, g_max) with strictly increasing d.
///
/// Consecutive points suffice. For y < k < x (by d), write P_y = (d_y, g_max(y)) and
/// q_x = (d_x, g_min(x)). The slope from P_y to q_x is a weighted mean of the slope from
/// P_y to q_k and the slope from q_k to q_x, and the latter is at least the slope from P_k
/// to q_x because g_max(k) ≥ g_min(k). So every pair is no better than a pair of smaller
/// gap, and by induction on the gap the minimum is attained by consecutive points. It
/// equals the pairwise minimum exactly (`min_slope_pairs`, tests).
fn min_slope(points: impl Iterator<Item = (i128, i128, i128)>) -> Option<(i128, i128)> {
    let mut best: Option<(i128, i128)> = None;
    let mut previous: Option<(i128, i128)> = None;
    for (d, g_min, g_max) in points {
        if let Some((d_lo, g_lo)) = previous {
            let (p, q) = (g_min - g_lo, d - d_lo);
            if best.is_none_or(|(bp, bq)| p * bq < bp * q) {
                best = Some((p, q));
            }
        }
        previous = Some((d, g_max));
    }
    best
}

/// The pairwise reference of [`min_slope`] (tests).
#[cfg(test)]
fn min_slope_pairs(points: &[(i128, i128, i128)]) -> Option<(i128, i128)> {
    let mut best: Option<(i128, i128)> = None;
    for (k, &(d_hi, g_min, _)) in points.iter().enumerate() {
        for &(d_lo, _, g_max) in &points[..k] {
            let (p, q) = (g_min - g_max, d_hi - d_lo);
            if best.is_none_or(|(bp, bq)| p * bq < bp * q) {
                best = Some((p, q));
            }
        }
    }
    best
}

/// L5 of §4.1, the pack facts the proof uses (shared with K3's C2 bound,
/// duplex-pf-bound.md §4.1 L5): N_C is 1 or 3 and `n_classes` constants were built, GEN is
/// finite (never ⊥) on every listed shape, and CLS agrees on mirrored listed shapes.
pub(crate) fn pack_facts(
    call: &DuplexCall,
    n_classes: usize,
    named: &mut Failures,
) -> Result<(), EngineError> {
    let cells = &call.score;
    let width = cells.loop_max + 1;
    if !(cells.n_c == 1 || cells.n_c == 3) || n_classes != cells.n_c {
        named.push(|| "L5: the number of classes".into());
    }
    let shapes = &call.pack.gen_shapes;
    let mut listed = Vec::new();
    try_reserve_exact(&mut listed, width * width, "duplex bound certificate")?;
    listed.resize(width * width, false);
    for &(u, v) in shapes {
        listed[u * width + v] = true;
    }
    for &(u, v) in shapes {
        if !cells.gen[u * width + v].finite() {
            named.push(|| format!("L5: GEN({u}, {v}) is ⊥"));
        }
        if listed[v * width + u] && cells.cls[u * width + v] != cells.cls[v * width + u] {
            named.push(|| format!("L5: CLS({u}, {v}) != CLS({v}, {u})"));
        }
    }
    Ok(())
}

/// The most failures a certificate names (§5 "Testing"); the verdict counts every one.
const MAX_FAILURES: usize = 32;

/// The failures of one check run: the first [`MAX_FAILURES`] named, the rest counted, so
/// that a failing certificate's messages stay bounded at every Λ.
#[derive(Default)]
pub(crate) struct Failures {
    named: Vec<String>,
    more: usize,
}

impl Failures {
    pub(crate) fn push(&mut self, what: impl FnOnce() -> String) {
        if self.named.len() < MAX_FAILURES {
            self.named.push(what());
        } else {
            self.more += 1;
        }
    }

    pub(crate) fn finish(mut self) -> Vec<String> {
        if self.more > 0 {
            self.named.push(format!("and {} more failures", self.more));
        }
        self.named
    }
}

/// The class starts of ℓ = min(n, m) in the cost table of §6.6.1 (the last class runs to
/// 200, the longest strand, D48).
const COST_CLASSES: [usize; 13] = [2, 18, 20, 22, 25, 28, 30, 35, 40, 50, 60, 80, 100];
/// ĉ by class, units of 1/64, for pack format 1 (`santalucia2004`) and format 2
/// (`nndb2024-dna`) (§6.6.1: ⌈80·c_max⌉ of the ci-profile ratios of §12.2, QB11).
const COST_TABLE: [[u32; 13]; 2] = [
    [192, 69, 60, 54, 46, 39, 37, 32, 29, 26, 25, 23, 23],
    [192, 75, 67, 60, 51, 44, 40, 34, 31, 27, 25, 23, 22],
];

/// The device tables of §6.6.3 by class (the classes of [`COST_CLASSES`]), for pack
/// format 1 and format 2: Ĉ_dev in 1/64 (above 64: never used on the device, the length
/// gate declines without a probe) and Q̂_dev in units of one item's `full` time at that
/// length (0: no probe-cost gate). Calibrated on 2026-11-06 by the protocol of §6.6.3
/// (margin 1.25, both ends of each class, the class entry the maximum of the two; the
/// per-length measurements are in §12.3 (G)).
const DEVICE_C_TABLE: [[u32; 13]; 2] = [
    [
        255, 255, 255, 255, 64, 64, 255, 255, 255, 255, 255, 255, 255,
    ],
    [255; 13],
];
const DEVICE_Q_TABLE: [[u32; 13]; 2] = [
    [
        577, 292, 474, 566, 441, 420, 387, 489, 415, 341, 302, 299, 302,
    ],
    [
        973, 2165, 960, 898, 772, 655, 748, 685, 622, 544, 540, 526, 534,
    ],
];

/// c(n, m, Λ) of §6.6 step 2 for n, m ≥ 2: Σ_{u=0}^{min(Λ, n−2)} (min(Λ − u, m − 2) + 1),
/// the largest number of predecessor candidates of one cell.
fn candidates(n: usize, m: usize, lam: usize) -> u64 {
    (0..=lam.min(n - 2))
        .map(|u| (lam - u).min(m - 2) as u64 + 1)
        .sum()
}

/// (e, ĉ) of one work item (§6.6 step 2): e = n·m·c(n, m, Λ), 0 for a strand under 2 nt
/// (closed form), and ĉ from the table of §6.6.1 by pack format (2 for every format other
/// than 1) and the class of min(n, m). A pure function of the lengths and the pack.
pub fn cost(format: u8, lam: usize, n: usize, m: usize) -> (u64, u32) {
    let (weight, class) = weight_class(lam, n, m);
    (weight, COST_TABLE[usize::from(format != 1)][class])
}

/// e of §6.6 step 2 and the class index of min(n, m) in [`COST_CLASSES`].
pub(crate) fn weight_class(lam: usize, n: usize, m: usize) -> (u64, usize) {
    let ell = n.min(m);
    let class = COST_CLASSES.iter().rposition(|&lo| ell >= lo).unwrap_or(0);
    let weight = if n < 2 || m < 2 {
        0
    } else {
        (n as u64) * (m as u64) * candidates(n, m, lam)
    };
    (weight, class)
}

/// The device's per-item costs (§6.6.2 step 3): (e, ĉ_dev, q̂_dev) from the tables of
/// §6.6.3, keyed like [`cost`]. `costs` is a snapshot of the testing override
/// (`seqblaze_core::plan::device_costs_override`, taken once per call), under which
/// every entry reads as the given (ĉ, q̂). The CPU never calls this.
pub fn device_cost(
    format: u8,
    lam: usize,
    n: usize,
    m: usize,
    costs: Option<(u32, u32)>,
) -> (u64, u32, u32) {
    let (weight, class) = weight_class(lam, n, m);
    let column = usize::from(format != 1);
    let (c, q) = costs.unwrap_or((DEVICE_C_TABLE[column][class], DEVICE_Q_TABLE[column][class]));
    (weight, c, q)
}

#[inline]
fn contains(lower: bool, d0: usize, start: usize, u: usize, v: usize) -> bool {
    if lower {
        u >= v + d0 && v >= start
    } else {
        v >= u + d0 && u >= start
    }
}

/// Per-thread buffers of the relaxed recurrence: rings of min(n, Λ + 2) rows of RA
/// (packed), RA + OUT_c per class (packed), and F_c, G_c per relaxed class (values).
#[derive(Default)]
struct Scratch {
    ra: Vec<i64>,
    out: Vec<i64>,
    f: Vec<i64>,
    g: Vec<i64>,
}

thread_local! {
    static SCRATCH: RefCell<Scratch> = RefCell::new(Scratch::default());
}

/// Buffers above this size are released after a pair (as the DP's band).
const KEEP_BYTES: usize = 1 << 20;

impl Scratch {
    fn prepare(&mut self, cells: usize, classes: usize, relaxed: usize) -> Result<(), EngineError> {
        for (buffer, len) in [
            (&mut self.ra, cells),
            (&mut self.out, cells * classes),
            (&mut self.f, cells * relaxed),
            (&mut self.g, cells * relaxed),
        ] {
            buffer.clear();
            try_reserve_exact(buffer, len, "duplex bound band")?;
            buffer.resize(len, INF);
        }
        Ok(())
    }

    fn release(&mut self) {
        for buffer in [&mut self.ra, &mut self.out, &mut self.f, &mut self.g] {
            if buffer.capacity() * std::mem::size_of::<i64>() > KEEP_BYTES {
                *buffer = Vec::new();
            }
        }
    }
}

/// The relaxed recurrence of §3.5 with flag packing (§4.6): a value v with flag f is
/// stored as 2v + (1 − f), so integer min realises (min, ∨), an exact edge adds 2·cost
/// and a relaxed candidate is 2·value + 1. F_c and G_c hold plain values (§3.4).
fn relaxed(
    cells: &Cells<i64>,
    rel: &[Relaxed],
    a: &[u8],
    b: &[u8],
    scratch: &mut Scratch,
) -> Result<(Option<i64>, bool), EngineError> {
    let (n, m) = (a.len(), b.len());
    let lam = cells.loop_max;
    let rows = n.min(lam + 2);
    let (np, nc, nr) = (cells.n_p, cells.n_c, rel.len());
    debug_assert!(n <= MAX_RUN && m <= MAX_RUN && nc <= 3);
    scratch.prepare(rows * m, nc, nr)?;
    let best = fill(cells, rel, a, b, scratch, rows, (np, nc));
    scratch.release();
    Ok(if finite(best) {
        (Some(best >> 1), best & 1 == 0)
    } else {
        // §3.6: an empty relaxed space certifies the empty exact space.
        (None, true)
    })
}

#[inline(always)]
fn fill(
    cells: &Cells<i64>,
    rel: &[Relaxed],
    a: &[u8],
    b: &[u8],
    s: &mut Scratch,
    rows: usize,
    (np, nc): (usize, usize),
) -> i64 {
    let (n, m) = (a.len(), b.len());
    let lam = cells.loop_max;
    let width = lam + 1;
    let (mut rho_a, mut rho_b) = ([0u8; MAX_RUN], [0u8; MAX_RUN]);
    let states = !cells.no_states;
    if states {
        run_lengths(a, &mut rho_a);
        run_lengths(b, &mut rho_b);
    }
    let identical = a == b;
    let palindrome = identical && (0..n).all(|k| a[k] + a[n - 1 - k] == 3);
    let head = if palindrome {
        cells.init_sym
    } else {
        cells.init
    };
    let mut best = INF;
    for i in 0..n {
        let cur = i % rows;
        let prev_slot = if cur == 0 { rows - 1 } else { cur - 1 };
        let ai = usize::from(a[i]);
        let base = cur * m;
        for j in 0..m {
            let pi = cells.pt[ai * 4 + usize::from(b[j])];
            if pi == NO_PAIR {
                s.ra[base + j] = INF;
                s.out[(base + j) * nc..(base + j + 1) * nc].fill(INF);
                continue;
            }
            let pi = usize::from(pi);
            let mut c = INF;
            if i >= 1 && j + 1 < m {
                let mut inner = [INF; 3];
                let in_base = usize::from(a[i - 1]) * 4 + usize::from(b[j + 1]);
                for (cl, slot) in inner.iter_mut().enumerate().take(nc) {
                    *slot = cells.loop_in[(cl * 16 + in_base) * np + pi];
                }
                // Exact edges: the window (u, v ≤ U_E, u + v ≤ Λ) and every bulge.
                let umax = lam.min(i - 1);
                let room = m - 2 - j;
                let mut slot = prev_slot;
                for u in 0..=umax {
                    let k = i - 1 - u;
                    let ak = usize::from(a[k]) * 4;
                    let row = slot * m;
                    let prev = s.ra[row + j + 1];
                    if finite(prev) {
                        let pk = usize::from(cells.pt[ak + usize::from(b[j + 1])]);
                        let mut cost = if u == 0 {
                            cells.stack[pk * np + pi]
                        } else {
                            cells.bulge[((u - 1) * np + pk) * np + pi]
                        };
                        if u == 1 && states && finite(cost) {
                            cost += cells.states[usize::from(rho_a[k + 1]) - 1];
                        }
                        if finite(cost) {
                            c = c.min(prev + 2 * cost);
                        }
                    }
                    let vmax = if u == 0 {
                        lam.min(room)
                    } else if u <= U_E {
                        U_E.min(lam - u).min(room)
                    } else {
                        0
                    };
                    for v in 1..=vmax {
                        let l = j + 1 + v;
                        let cand = if u == 0 {
                            let prev = s.ra[row + l];
                            if !finite(prev) {
                                continue;
                            }
                            let pk = usize::from(cells.pt[ak + usize::from(b[l])]);
                            let mut cost = cells.bulge[((lam + v - 1) * np + pk) * np + pi];
                            if v == 1 && states && finite(cost) {
                                cost += cells.states[usize::from(rho_b[l - 1]) - 1];
                            }
                            if !finite(cost) {
                                continue;
                            }
                            prev + 2 * cost
                        } else if u <= 2 && v <= 2 {
                            let prev = s.ra[row + l];
                            if !finite(prev) {
                                continue;
                            }
                            let pk = usize::from(cells.pt[ak + usize::from(b[l])]);
                            let p1 = usize::from(a[k + 1]);
                            let q1 = usize::from(b[l - 1]);
                            let cost = match (u, v) {
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
                                    cells.i22
                                        [((((pk * 4 + p1) * 4 + p2) * 4 + q1) * 4 + q2) * np + pi]
                                }
                            };
                            if !finite(cost) {
                                continue;
                            }
                            prev + 2 * cost
                        } else {
                            let cl = usize::from(cells.cls[u * width + v]);
                            let prev = s.out[(row + l) * nc + cl];
                            let rest = cells.gen[u * width + v];
                            if !finite(prev) || !finite(rest) || !finite(inner[cl]) {
                                continue;
                            }
                            prev + 2 * (rest + inner[cl])
                        };
                        c = c.min(cand);
                    }
                    slot = if slot == 0 { rows - 1 } else { slot - 1 };
                }
                // Relaxed edges: the region queries of every relaxed class (§3.4).
                let (iq, jq) = ((i - 1) as i64, (j + 1) as i64);
                for (t, r) in rel.iter().enumerate() {
                    let x = inner[r.class];
                    if !finite(x) {
                        continue;
                    }
                    let f = &s.f[t * rows * m..];
                    let g = &s.g[t * rows * m..];
                    let lower_off = x + r.plus * iq - r.minus * jq;
                    for &(depth, start, gamma) in &r.lower {
                        let col = j + 1 + start;
                        if depth > i - 1 || col >= m {
                            continue;
                        }
                        let value = f[((i - 1 - depth) % rows) * m + col];
                        if finite(value) {
                            c = c.min(2 * (value + gamma + lower_off) + 1);
                        }
                    }
                    let upper_off = x + r.minus * iq - r.plus * jq;
                    for &(start, reach, gamma) in &r.upper {
                        let col = j + 1 + reach;
                        if start > i - 1 || col >= m {
                            continue;
                        }
                        let value = g[((i - 1 - start) % rows) * m + col];
                        if finite(value) {
                            c = c.min(2 * (value + gamma + upper_off) + 1);
                        }
                    }
                }
            }
            if !finite(c) {
                c = INF;
            }
            let left = cells.left_end(a, b, i, j);
            let first = if finite(head) && finite(left) {
                2 * (head + left)
            } else {
                INF
            };
            let value = c.min(first);
            s.ra[base + j] = value;
            let out = &mut s.out[(base + j) * nc..(base + j + 1) * nc];
            if finite(value) && i + 1 < n && j >= 1 {
                let tail = usize::from(a[i + 1]) * 4 + usize::from(b[j - 1]);
                for (cl, slot) in out.iter_mut().enumerate() {
                    let cost = cells.loop_out[(cl * np + pi) * 16 + tail];
                    *slot = if finite(cost) { value + 2 * cost } else { INF };
                }
            } else {
                out.fill(INF);
            }
            if finite(c) {
                let right = cells.right_end(a, b, i, j);
                if finite(right) {
                    best = best.min(c + 2 * right);
                }
            }
        }
        // Row i is final: F_c(i, ·) and G_c(i, ·) of every relaxed class (§3.4).
        let ii = i as i64;
        for (t, r) in rel.iter().enumerate() {
            let band = t * rows * m;
            let (here, before) = (band + cur * m, band + prev_slot * m);
            for l in 0..m {
                let y = y_value(cells, a, b, s.ra[base + l], i, l, r.class, (np, n));
                let mut f = if finite(y) {
                    y - r.plus * ii + r.minus * l as i64
                } else {
                    INF
                };
                if i >= 1 {
                    f = f.min(s.f[before + l]);
                    if l + 1 < m {
                        f = f.min(s.f[before + l + 1]);
                    }
                }
                s.f[here + l] = f;
            }
            for l in (0..m).rev() {
                let y = y_value(cells, a, b, s.ra[base + l], i, l, r.class, (np, n));
                let mut g = if finite(y) {
                    y - r.minus * ii + r.plus * l as i64
                } else {
                    INF
                };
                if l + 1 < m {
                    g = g.min(s.g[here + l + 1]);
                    if i >= 1 {
                        g = g.min(s.g[before + l + 1]);
                    }
                }
                s.g[here + l] = g;
            }
        }
    }
    best
}

/// RA(k, l) + OUT_c(k, l) as a plain value, +∞ unless RA is finite, k + 1 < n, l ≥ 1 and
/// OUT_c is finite (§3.4).
#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn y_value(
    cells: &Cells<i64>,
    a: &[u8],
    b: &[u8],
    ra: i64,
    k: usize,
    l: usize,
    class: usize,
    (np, n): (usize, usize),
) -> i64 {
    if !finite(ra) || k + 1 >= n || l == 0 {
        return INF;
    }
    let p = usize::from(cells.pt[usize::from(a[k]) * 4 + usize::from(b[l])]);
    let cost =
        cells.loop_out[((class * np + p) * 4 + usize::from(a[k + 1])) * 4 + usize::from(b[l - 1])];
    if !finite(cost) {
        return INF;
    }
    (ra >> 1) + cost
}

#[cfg(test)]
mod tests {
    use super::{min_slope, min_slope_pairs};

    /// The consecutive-point scan equals the pairwise minimum as an exact rational, on random
    /// point sets (d strictly increasing; g_min ≤ g_max), small and wide value ranges.
    #[test]
    fn min_slope_equals_pairwise() {
        let mut state: u64 = 0x9E37_79B9_7F4A_7C15;
        let mut next = |m: u64| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state % m
        };
        for case in 0..20_000 {
            let k = 1 + next(40) as usize;
            let span = [3, 50, 1 << 20, 1 << 40][case % 4];
            let mut points = Vec::with_capacity(k);
            let mut d = next(3) as i128;
            for _ in 0..k {
                let a = next(2 * span) as i128 - span as i128;
                let b = a + next(span / 2 + 1) as i128;
                points.push((d, a, b));
                d += 1 + next(3) as i128;
            }
            let fast = min_slope(points.iter().copied());
            let slow = min_slope_pairs(&points);
            match (fast, slow) {
                (Some((p, q)), Some((r, s))) => assert_eq!(p * s, r * q, "{points:?}"),
                (x, y) => assert_eq!(x, y, "{points:?}"),
            }
        }
    }
}
