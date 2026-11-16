//! The C2 certified ensemble bound of K3 native `duplex(model="pf")` under
//! `strategy="bounded"` (docs/wiki/thermo/duplex-pf-bound.md): a relaxed F53 recurrence
//! Ẑ_b ≥ Z over K3's full structure space, with C1's exact window U = 4 and exact bulges
//! (§3.1); every generic loop outside the window is relaxed per loop class over a
//! partition into rays T_v, B_u and wedges LO, HI (§3.2), summed by dense accumulators
//! with per-call weights x = W(α + β), z = W(2α), Ω_r = W(γ_r) (§3.3–§3.5). From Ẑ_b the
//! binary64 lower bound L ≤ the computed `dg_ens_kcal` (§4.4) and the outcome against an
//! int64 key threshold (§1).
//!
//! Z ≤ Z_b holds by the per-call certificate's finite checks L1–L5 plus the row induction
//! of §4.2; L ≤ d by the rounding count c_b (§4.3) and K3's own error facts (§4.4). Every
//! value is a pure function of the pair's bytes and the per-call constants: the fold
//! orders are binding, so the result does not depend on threads, backend or chunking.
use std::cell::RefCell;

use seqblaze_core::{
    detail::try_reserve_exact,
    kernel::BoundOutcome,
    plan::{bound_fault, BoundFault},
    validate::EngineError,
};

use super::{
    bound::{self, class_slopes, pack_facts, Certificate as C1, Failures, ShapeList},
    pack::NO_PAIR,
    pf::{key_of, log_map, sigma, PfCall, Weights, F53, MAX_CLASSES},
};
use crate::nat::Nat;

/// The exact window U (§3.1, QZ1): C1's U_E, a kernel constant.
pub const U: usize = bound::U_E;
/// The number of regions of the family T_1 … T_U, B_1 … B_U, LO, HI (§3.2).
pub const N_REGIONS: usize = 2 * U + 2;
/// The exponent limit of a nonzero Ẑ_b under L4 (§4.5): |e| < 2^60 + 2^10.
pub const EXPONENT_LIMIT: u64 = (1 << 60) + (1 << 10);

/// A region of the family (§3.2), in the binding order of §3.5.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RegionKind {
    /// T_v: v fixed (1 … U), u ≥ U + 1.
    T,
    /// B_u: u fixed (1 … U), v ≥ U + 1.
    B,
    /// LO: u ≥ v ≥ U + 1.
    Lo,
    /// HI: v > u ≥ U + 1.
    Hi,
}

impl RegionKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::T => "T",
            Self::B => "B",
            Self::Lo => "LO",
            Self::Hi => "HI",
        }
    }
}

/// The region family in the order T_1 … T_U, B_1 … B_U, LO, HI (index 0 for LO and HI).
fn family() -> impl Iterator<Item = (RegionKind, usize)> {
    (1..=U)
        .map(|v| (RegionKind::T, v))
        .chain((1..=U).map(|u| (RegionKind::B, u)))
        .chain([(RegionKind::Lo, 0), (RegionKind::Hi, 0)])
}

/// Whether (u, v) lies in region (kind, index), by the membership rules of §3.2.
fn member(kind: RegionKind, index: usize, u: usize, v: usize) -> bool {
    match kind {
        RegionKind::T => v == index && u > U,
        RegionKind::B => u == index && v > U,
        RegionKind::Lo => u >= v && v > U,
        RegionKind::Hi => v > u && u > U,
    }
}

/// The linear part ℓ_r(u, v) of §3.2 with slopes a (axis) and d (diagonal), for a member.
fn linear(kind: RegionKind, a: i128, d: i128, u: usize, v: usize) -> i128 {
    let (u, v, w) = (u as i128, v as i128, U as i128 + 1);
    match kind {
        RegionKind::T => a * (u - w),
        RegionKind::B => a * (v - w),
        RegionKind::Lo => a * (u - v) + d * (v - w),
        RegionKind::Hi => a * (v - u - 1) + d * (u - w),
    }
}

/// One used region of a class with its constant γ_r (§3.3).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Region {
    pub kind: RegionKind,
    /// v of T_v, u of B_u, 0 for LO and HI.
    pub index: usize,
    pub gamma: i64,
}

/// The constants of one loop class (§3.3): α, β, a = α + β, d = 2α and the used regions
/// in the family order. A class with an empty S_c has zeros and no region.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ClassBound {
    pub alpha: i64,
    pub beta: i64,
    pub a: i64,
    pub d: i64,
    pub regions: Vec<Region>,
}

/// A relaxed class as the recurrence reads it (§3.4): its index, the weights x and z, which
/// accumulators it needs, and the Ω_r of its used regions (indexed by family position).
#[derive(Debug, Clone)]
struct Relaxed {
    class: usize,
    x: F53,
    z: F53,
    /// Ω of T_v at `t[v - 1]`, of B_u at `b[u - 1]`, of LO and HI; `None` if unused.
    t: [Option<F53>; U],
    b: [Option<F53>; U],
    lo: Option<F53>,
    hi: Option<F53>,
    need_h: bool,
    need_v: bool,
}

/// The per-call certificate of §5: the constants of §3.3, G*, s*, B_b, the L4 verdict, the
/// outcome of the checks L1–L5 and (for a passing certificate) the per-call weights. A pure
/// function of the pack, the per-call D cell table and R·T.
#[derive(Debug, Clone)]
pub struct Certificate {
    pub pass: bool,
    /// The failed checks (empty iff `pass`); names and shapes only, never a value.
    pub failures: Vec<String>,
    pub g_star: i64,
    pub s_star: i64,
    pub b_b: i64,
    /// Whether L4 holds (§4.1, §4.5).
    pub l4: bool,
    pub classes: Vec<ClassBound>,
    relaxed: Vec<Relaxed>,
}

/// The device words of one relaxed class (duplex-pf-bound.md §8; mirrored by
/// `SB_PFB_CLASS_WORDS` in kernels/duplex_pf_bound.cuh): class, x (M, e), z (M, e), then
/// 2U + 2 region slots of (used, M, e) in the family order.
pub const DEVICE_CLASS_WORDS: usize = 5 + 3 * N_REGIONS;

impl Certificate {
    /// Build and check the certificate of a K3 call (§3.3, §4.1, §5). The `gamma_plus_one`
    /// testing fault raises every γ_r by 1 before the check, which must then fail (L2).
    /// Every allocation that grows with Λ is reserved fallibly: a refused reservation
    /// returns `Err` (never an abort) and leaves no certificate; the binding maps it to an
    /// unavailable bound (`no_bound`). It is the only `Err`.
    pub fn build(pf: &PfCall) -> Result<Self, EngineError> {
        seqblaze_core::plan::count_certificate_build();
        let call = &pf.call;
        let shapes = C1::shapes(call)?;
        let mut cert = Self::constants(call.score.n_c, &shapes);
        if bound_fault() == BoundFault::GammaPlusOne {
            for class in &mut cert.classes {
                for region in &mut class.regions {
                    region.gamma = region.gamma.saturating_add(1);
                }
            }
        }
        cert.check(pf, &shapes)?;
        if cert.pass {
            cert.weigh(pf);
        }
        Ok(cert)
    }

    /// The constants of §3.3, unchecked: per class C1's α, β on S_c, a = α + β, d = 2α
    /// and γ_r of every used region; int64 overflow fails the certificate.
    fn constants(n_c: usize, shapes: &[ShapeList]) -> Self {
        let mut classes = vec![ClassBound::default(); n_c];
        let mut wide = false;
        let mut fits = |x: i128| -> i64 {
            i64::try_from(x).unwrap_or_else(|_| {
                wide = true;
                0
            })
        };
        for (c, list) in shapes.iter().enumerate() {
            let Some(list) = list.as_ref().filter(|list| !list.is_empty()) else {
                continue;
            };
            let Some(class) = classes.get_mut(c) else {
                continue;
            };
            let (alpha, beta) = class_slopes(list);
            let (a, d) = (alpha + beta, 2 * alpha);
            let mut regions = Vec::new();
            for (kind, index) in family() {
                let gamma = list
                    .iter()
                    .filter(|&&(u, v, _)| member(kind, index, u, v))
                    .map(|&(u, v, g)| g - linear(kind, a, d, u, v))
                    .min();
                if let Some(gamma) = gamma {
                    regions.push(Region {
                        kind,
                        index,
                        gamma: fits(gamma),
                    });
                }
            }
            *class = ClassBound {
                alpha: fits(alpha),
                beta: fits(beta),
                a: fits(a),
                d: fits(d),
                regions,
            };
        }
        let mut cert = Self {
            pass: true,
            failures: Vec::new(),
            g_star: 0,
            s_star: 0,
            b_b: 0,
            l4: false,
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

    /// L1–L5 of §4.1 on the constants as they stand (after any testing fault), with G*, s*,
    /// B_b and the L4 verdict.
    fn check(&mut self, pf: &PfCall, shape_lists: &[ShapeList]) -> Result<(), EngineError> {
        let call = &pf.call;
        let mut named = Failures::default();
        // L5: the pack facts (C1's L5).
        pack_facts(call, self.classes.len(), &mut named)?;
        for (c, list) in shape_lists.iter().enumerate() {
            let (Some(list), Some(class)) = (list, self.classes.get(c)) else {
                continue;
            };
            // L3.
            if class.alpha < 0 || class.beta < 0 || class.a < 0 || class.d < 0 {
                named.push(|| format!("L3: class {c}"));
            }
            let (a, d) = (i128::from(class.a), i128::from(class.d));
            for &(u, v, g) in list {
                // L1: exactly one region of the family contains the shape, and it is used.
                let mut held = 0usize;
                let mut home = None;
                for (kind, index) in family() {
                    if member(kind, index, u, v) {
                        held += 1;
                        home = Some((kind, index));
                    }
                }
                let used = home.and_then(|(kind, index)| {
                    class
                        .regions
                        .iter()
                        .find(|r| r.kind == kind && r.index == index)
                });
                match (held, used) {
                    (1, Some(r)) => {
                        // L2.
                        if i128::from(r.gamma) + linear(r.kind, a, d, u, v) > g {
                            named.push(|| format!("L2: class {c} shape ({u}, {v})"));
                        }
                    }
                    _ => named.push(|| format!("L1: class {c} shape ({u}, {v})")),
                }
            }
        }
        for what in named.finish() {
            self.fail(what);
        }
        // G*, s*, B_b (§3.3) and L4 (§4.5): exact.
        let g_star = self
            .classes
            .iter()
            .flat_map(|class| class.regions.iter())
            .map(|r| i128::from(r.gamma).abs())
            .max()
            .unwrap_or(0);
        let s_star = self
            .classes
            .iter()
            .filter(|class| !class.regions.is_empty())
            .map(|class| i128::from(class.a.max(class.d)))
            .max()
            .unwrap_or(0);
        let b_b = 2 * i128::from(call.bound) + 200 * g_star + 800 * s_star;
        match (
            i64::try_from(g_star),
            i64::try_from(s_star),
            i64::try_from(b_b),
        ) {
            (Ok(g), Ok(s), Ok(b)) if b >= 0 => {
                self.g_star = g;
                self.s_star = s;
                self.b_b = b;
                // L4: 10^4 · B_b < 693 · 2^60 · R·T, with R·T = num/den > 0.
                let rt = &pf.constants.rt;
                let left = Nat::from_u64(b as u64).mul_u64(10_000).mul(&rt.den);
                let right = rt.num.magnitude().mul_u64(693).shl(60);
                self.l4 = !rt.num.is_negative() && left < right;
                if !self.l4 {
                    self.fail("L4: the bound's exponent range".into());
                }
            }
            _ => self.fail("int64 overflow in a certificate constant".into()),
        }
        Ok(())
    }

    /// The per-call weights x_c = W(a_c), z_c = W(d_c) and Ω_r = W(γ_r) (§3.3) of every
    /// class with a used region, by K3's weight routine. A weight the host routine cannot
    /// round fails the certificate (an internal limit, never reached by a passing L4).
    fn weigh(&mut self, pf: &PfCall) {
        let w = |d: i64| pf.constants.weight(d);
        let mut relaxed = Vec::new();
        let mut failed = false;
        for (c, class) in self.classes.iter().enumerate() {
            if class.regions.is_empty() {
                continue;
            }
            let (Ok(x), Ok(z)) = (w(class.a), w(class.d)) else {
                failed = true;
                continue;
            };
            let mut r = Relaxed {
                class: c,
                x,
                z,
                t: [None; U],
                b: [None; U],
                lo: None,
                hi: None,
                need_h: false,
                need_v: false,
            };
            for region in &class.regions {
                let Ok(omega) = w(region.gamma) else {
                    failed = true;
                    continue;
                };
                match region.kind {
                    RegionKind::T => r.t[region.index - 1] = Some(omega),
                    RegionKind::B => r.b[region.index - 1] = Some(omega),
                    RegionKind::Lo => r.lo = Some(omega),
                    RegionKind::Hi => r.hi = Some(omega),
                }
            }
            r.need_h = r.t.iter().any(Option::is_some) || r.lo.is_some();
            r.need_v = r.b.iter().any(Option::is_some) || r.hi.is_some();
            relaxed.push(r);
        }
        if failed {
            self.fail("internal: a per-call weight could not be rounded".into());
        } else {
            self.relaxed = relaxed;
        }
    }

    /// Whether any class has relaxed edges (else Ẑ_b = Ẑ, §3.5).
    pub fn relaxes(&self) -> bool {
        !self.relaxed.is_empty()
    }

    /// The relaxed classes as device words (§8; [`DEVICE_CLASS_WORDS`] per class): class,
    /// x as (M, e), z as (M, e), then per region slot of the family (used, M, e), each F53
    /// as M · 2^(e − 52) (M = 0 for none). Empty for a failed certificate.
    pub fn device_words(&self) -> (Vec<i64>, usize) {
        let parts = |w: F53| -> (i64, i64) {
            let (m, e) = w.parts();
            if m == 0 {
                (0, 0)
            } else {
                (m as i64, e + 52)
            }
        };
        let mut words = Vec::with_capacity(DEVICE_CLASS_WORDS * self.relaxed.len());
        for r in &self.relaxed {
            words.push(r.class as i64);
            for w in [r.x, r.z] {
                let (m, e) = parts(w);
                words.push(m);
                words.push(e);
            }
            let slots = r.t.iter().chain(r.b.iter()).chain([&r.lo, &r.hi]).copied();
            for slot in slots {
                match slot {
                    Some(w) => {
                        let (m, e) = parts(w);
                        words.extend_from_slice(&[1, m, e]);
                    }
                    None => words.extend_from_slice(&[0, 0, 0]),
                }
            }
        }
        (words, self.relaxed.len())
    }

    /// Ẑ_b of a pair of letter codes in canonical orientation (§3.6), with the
    /// `zb_scale_down` testing fault applied (exponent − 64 on a nonzero value). Only for a
    /// passing certificate.
    pub fn zb(&self, pf: &PfCall, a: &[u8], b: &[u8]) -> Result<F53, EngineError> {
        let (x, y) = if a <= b { (a, b) } else { (b, a) };
        let z = SCRATCH
            .with(|cell| relaxed_zhat(&pf.weights, &self.relaxed, x, y, &mut cell.borrow_mut()))?;
        Ok(scale_fault(z))
    }

    /// (Ẑ_b, L, outcome) of a pair of letter codes against θ (an engine key; `None`:
    /// certify only), §1. Runs in E0 (its callers' guards).
    pub fn evaluate(
        &self,
        pf: &PfCall,
        a: &[u8],
        b: &[u8],
        threshold: Option<i64>,
    ) -> Result<(F53, f64, BoundOutcome<i64>), EngineError> {
        let z = self.zb(pf, a, b)?;
        let l = lower(z, pf.constants.c_t);
        Ok((z, l, outcome(z, l, threshold)))
    }
}

/// The `zb_scale_down` testing fault (§10.2): every nonzero Ẑ_b times 2^−64.
#[inline]
pub fn scale_fault(z: F53) -> F53 {
    if !z.is_zero() && bound_fault() == BoundFault::ZbScaleDown {
        F53 {
            mu: z.mu,
            e: z.e - 64,
        }
    } else {
        z
    }
}

/// Ẑ_b from a device result (M, e) with value M · 2^(e − 52) (duplex-pf-bound.md §4.5
/// *Device result check*): (0, 0) is zero; anything else needs 2^52 ≤ M < 2^53 and
/// |e| < 2^60 + 2^10, else an internal DeviceError (no result is published).
pub fn zb_of_device(m: u64, e: i64) -> Result<F53, EngineError> {
    if m == 0 && e == 0 {
        Ok(F53::ZERO)
    } else if (1 << 52..1 << 53).contains(&m) && e.unsigned_abs() < EXPONENT_LIMIT {
        Ok(F53::from_parts(m, e - 52))
    } else {
        Err(EngineError::Device(format!(
            "internal error: duplex model='pf' received an invalid bound value (M = {m}, \
             E = {e}); no result was published"
        )))
    }
}

/// next_up(x) for a finite x: the next binary64 above x (+0.0's is the least subnormal).
#[inline]
fn next_up(x: f64) -> f64 {
    if x == 0.0 {
        f64::from_bits(1)
    } else if x > 0.0 {
        f64::from_bits(x.to_bits() + 1)
    } else {
        f64::from_bits(x.to_bits() - 1)
    }
}

/// L of §4.4 from Ẑ_b and c_T = RN(−R·T/1000): +∞ for Ẑ_b = 0; else
/// N = 6·|e_b| + 2²⁵, M = f64↑(N)·2⁻⁵³, g_b = g(μ_b, e_b), Γ = next_up(RN(g_b + M)),
/// L = RN(c_T·Γ) with −0.0 read as +0.0. One binary64 operation per step in the order
/// written; runs in E0 (its callers' guards). |e_b| < 2^60 + 2^10 under L4.
pub fn lower(z: F53, c_t: f64) -> f64 {
    if z.is_zero() {
        return f64::INFINITY;
    }
    let n = 6 * u128::from(z.e.unsigned_abs()) + (1 << 25);
    // f64↑(N): the nearest binary64 (`as` rounds to nearest), stepped up if below N. The
    // value is an integer of at most 2^65, so the comparison in u128 is exact.
    let mut f = n as f64;
    if (f as u128) < n {
        f = next_up(f);
    }
    let m = f * f64::from_bits((1023 - 53) << 52); // ·2^−53, exact (f ≥ 2^25)
    let g = log_map(z.mu, z.e);
    let gamma = next_up(g + m);
    let l = c_t * gamma;
    if l == 0.0 {
        0.0
    } else {
        l
    }
}

/// The outcome of §1: pruned iff θ is given and λ > θ (λ = key(L), key(+∞) for Ẑ_b = 0);
/// otherwise certified with key(+∞) iff Ẑ_b = 0; otherwise open.
pub fn outcome(z: F53, l: f64, threshold: Option<i64>) -> BoundOutcome<i64> {
    let inf = key_of(f64::INFINITY);
    let lambda = if z.is_zero() { inf } else { key_of(l) };
    if threshold.is_some_and(|theta| lambda > theta) {
        BoundOutcome::Pruned
    } else if z.is_zero() {
        BoundOutcome::Certified(inf)
    } else {
        BoundOutcome::Open
    }
}

/// The outcome code of the testing hooks and the device (0 open, 1 certified, 2 pruned).
pub fn outcome_code(outcome: BoundOutcome<i64>) -> i8 {
    match outcome {
        BoundOutcome::Open => 0,
        BoundOutcome::Certified(_) => 1,
        BoundOutcome::Pruned => 2,
    }
}

// --------------------------------------------------------------------------- costs (§6.6)

/// Ĉ_pf by class (C1's 13 classes of min(n, m): 2–17, 18–19, 20–21, 22–24, 25–27, 28–29,
/// 30–34, 35–39, 40–49, 50–59, 60–79, 80–99, 100–200), units of 1/64, for pack format 1
/// (`santalucia2004`) and format 2 (`nndb2024-dna`) (§6.6, QZ8): the CPU calibration of §12.6,
/// ĉ = min(255, ⌈80·c_max⌉), measured 2026-11-10 on the ci build (reduced).
const COST_TABLE: [[u32; 13]; 2] = [
    [85, 72, 65, 58, 50, 46, 44, 40, 36, 32, 32, 29, 30],
    [89, 64, 64, 52, 44, 40, 38, 33, 31, 28, 27, 24, 23],
];

/// The device tables Ĉ_pf,dev (1/64) and Q̂_pf,dev (one item's `full` time at that length)
/// by class, for format 1 and format 2 (§6.6, QZ8). Calibrated on 2026-11-11 by the device
/// protocol of duplex-mfe-bound.md §6.6.3 against K3's device `full` (margin 1.25, both ends
/// of each class, the class entry the maximum of the two; the per-length measurements are in
/// §12.7). No length qualifies at any resolved fraction (one open item in 64 already makes the
/// device slower than `full`), so Ĉ_pf,dev = 255 everywhere: the length gate declines every
/// natural device pass without a probe, and the measured Q̂_pf,dev never enters a decision.
const DEVICE_C_TABLE: [[u32; 13]; 2] = [[255; 13], [255; 13]];
const DEVICE_Q_TABLE: [[u32; 13]; 2] = [
    [
        960, 723, 2039, 971, 380, 757, 845, 881, 593, 625, 417, 424, 421,
    ],
    [
        3160, 3054, 2428, 1088, 1472, 1198, 783, 512, 676, 510, 541, 551, 366,
    ],
];

/// (e, ĉ) of one work item (§6.6 step 2 with K3's table): e = n·m·c(n, m, Λ), 0 for a
/// strand under 2 nt, and Ĉ_pf by pack format and the class of min(n, m). Symmetric in
/// (n, m), so the evaluated orientation does not matter.
pub fn cost(format: u8, lam: usize, n: usize, m: usize) -> (u64, u32) {
    let (weight, class) = bound::weight_class(lam, n, m);
    (weight, COST_TABLE[usize::from(format != 1)][class])
}

/// The device's per-item costs (e, ĉ_dev, q̂_dev) with K3's device tables, or the testing
/// override `costs` (taken once per call) for every entry. The CPU never calls this.
pub fn device_cost(
    format: u8,
    lam: usize,
    n: usize,
    m: usize,
    costs: Option<(u32, u32)>,
) -> (u64, u32, u32) {
    let (weight, class) = bound::weight_class(lam, n, m);
    let column = usize::from(format != 1);
    let (c, q) = costs.unwrap_or((DEVICE_C_TABLE[column][class], DEVICE_Q_TABLE[column][class]));
    (weight, c, q)
}

// --------------------------------------------------------------------------- recurrence (§3.4, §3.5)

/// Per-thread buffers: RA in a ring of min(n, Λ + 2) rows (bulges reach Λ + 1 rows back),
/// RO_c in a ring of min(n, U + 2) rows, the accumulators H, V, HD, VD of every relaxed
/// class in rings of U + 3 rows, and the column accumulators E_b.
#[derive(Default)]
struct Scratch {
    ra: Vec<F53>,
    ro: Vec<F53>,
    acc: Vec<F53>,
    col: Vec<F53>,
}

thread_local! {
    static SCRATCH: RefCell<Scratch> = RefCell::new(Scratch::default());
}

/// Buffers above this size are released after a pair.
const KEEP_BYTES: usize = 1 << 20;
/// Rows of each accumulator ring: the queries of row i read row i − U − 2.
const ACC_ROWS: usize = U + 3;

impl Scratch {
    fn prepare(&mut self, sizes: [usize; 4]) -> Result<(), EngineError> {
        for (buffer, len) in [
            (&mut self.ra, sizes[0]),
            (&mut self.ro, sizes[1]),
            (&mut self.acc, sizes[2]),
            (&mut self.col, sizes[3]),
        ] {
            buffer.clear();
            try_reserve_exact(buffer, len, "duplex pf bound band")?;
            buffer.resize(len, F53::ZERO);
        }
        Ok(())
    }

    fn release(&mut self) {
        for buffer in [&mut self.ra, &mut self.ro, &mut self.acc, &mut self.col] {
            if buffer.capacity() * std::mem::size_of::<F53>() > KEEP_BYTES {
                *buffer = Vec::new();
            }
        }
    }
}

/// x ⊕ (w ⊗ y), skipping a zero product (adding 0 is exact).
#[inline(always)]
fn fma53(x: F53, w: F53, y: F53) -> F53 {
    if y.is_zero() {
        x
    } else {
        x.plus(w.times(y))
    }
}

/// Ẑ_b of §3.5 for the canonical top strand `a` and bottom strand `b` (codes 0..4). The
/// special exact terms fold in K3's order O restricted to the exact shapes; the generic
/// exact terms of class c fold in order O into RG_c, then the region terms of its used
/// regions in the order T_1 … T_U, B_1 … B_U, LO, HI; RC folds the special terms, then
/// RG_c ⊗ WIN_c for c = 0 … N_C − 1; RA = Ws ⊕ RC. After row i: RO_c(i, ·) and the
/// accumulators of row i (H and HD left to right, V and VD right to left, §3.4). E_b(j)
/// folds rows ascending and Ẑ_b the columns ascending. Zero terms are skipped.
fn relaxed_zhat(
    w: &Weights,
    rel: &[Relaxed],
    a: &[u8],
    b: &[u8],
    s: &mut Scratch,
) -> Result<F53, EngineError> {
    let (n, m) = (a.len(), b.len());
    if n < 2 || m < 2 {
        return Ok(F53::ZERO);
    }
    let lam = w.loop_max;
    let (np, nc) = (w.n_p, w.n_c);
    debug_assert!(nc <= MAX_CLASSES);
    let rows = n.min(lam + 2);
    let ro_rows = n.min(U + 2);
    // Per relaxed class: H, V, HD, VD, each ACC_ROWS × m.
    let plane = ACC_ROWS * m;
    s.prepare([rows * m, ro_rows * m * nc, rel.len() * 4 * plane, m])?;
    let start = &w.start[sigma(a, b)];
    let width = lam + 1;
    let (mut cur, mut cur_ro) = (0usize, 0usize);
    for i in 0..n {
        let ai = usize::from(a[i]);
        let base = cur * m;
        let base_ro = cur_ro * m;
        // Accumulator slots: row r lives at r mod (U + 3); row i − U − 2 at (i + 1) mod.
        let query_slot = (i + 1) % ACC_ROWS;
        for j in 0..m {
            let pi = w.pt[ai * 4 + usize::from(b[j])];
            if pi == NO_PAIR {
                s.ra[base + j] = F53::ZERO;
                s.ro[(base_ro + j) * nc..(base_ro + j + 1) * nc].fill(F53::ZERO);
                continue;
            }
            let pi = usize::from(pi);
            let mut special = F53::ZERO;
            if i >= 1 && j + 1 < m {
                let mut generic = [F53::ZERO; MAX_CLASSES];
                let umax = lam.min(i - 1);
                let room = m - 2 - j;
                let mut slot = if cur == 0 { rows - 1 } else { cur - 1 };
                let mut slot_ro = if cur_ro == 0 { ro_rows - 1 } else { cur_ro - 1 };
                for u in 0..=umax {
                    let k = i - 1 - u;
                    let ak = usize::from(a[k]) * 4;
                    let row = slot * m;
                    // v = 0: a stack or a top bulge (every size is exact).
                    let prev = s.ra[row + j + 1];
                    if !prev.is_zero() {
                        let pk = usize::from(w.pt[ak + usize::from(b[j + 1])]);
                        let weight = if u == 0 {
                            w.stack[pk * np + pi]
                        } else {
                            w.bulge[((u - 1) * np + pk) * np + pi]
                        };
                        special = special.plus(prev.times(weight));
                    }
                    let vmax = if u == 0 {
                        lam.min(room)
                    } else if u <= U {
                        U.min(lam - u).min(room)
                    } else {
                        0
                    };
                    for v in 1..=vmax {
                        let l = j + 1 + v;
                        match u {
                            0 => {
                                let prev = s.ra[row + l];
                                if !prev.is_zero() {
                                    let pk = usize::from(w.pt[ak + usize::from(b[l])]);
                                    let weight = w.bulge[((lam + v - 1) * np + pk) * np + pi];
                                    special = special.plus(prev.times(weight));
                                }
                            }
                            1 | 2 if v <= 2 => {
                                let prev = s.ra[row + l];
                                if prev.is_zero() {
                                    continue;
                                }
                                let pk = usize::from(w.pt[ak + usize::from(b[l])]);
                                let p1 = usize::from(a[k + 1]);
                                let q1 = usize::from(b[l - 1]);
                                let weight = match (u, v) {
                                    (1, 1) => w.i11[((pk * 4 + p1) * 4 + q1) * np + pi],
                                    (1, _) => {
                                        let q2 = usize::from(b[l - 2]);
                                        w.i12[(((pk * 4 + p1) * 4 + q1) * 4 + q2) * np + pi]
                                    }
                                    (_, 1) => {
                                        let p2 = usize::from(a[k + 2]);
                                        w.i21[(((pk * 4 + p1) * 4 + p2) * 4 + q1) * np + pi]
                                    }
                                    _ => {
                                        let p2 = usize::from(a[k + 2]);
                                        let q2 = usize::from(b[l - 2]);
                                        w.i22[((((pk * 4 + p1) * 4 + p2) * 4 + q1) * 4 + q2) * np
                                            + pi]
                                    }
                                };
                                special = special.plus(prev.times(weight));
                            }
                            _ => {
                                // A generic window shape (1 ≤ u, v ≤ U): K3's term with RO.
                                let cl = usize::from(w.cls[u * width + v]);
                                let prev = s.ro[(slot_ro * m + l) * nc + cl];
                                if !prev.is_zero() {
                                    generic[cl] =
                                        generic[cl].plus(prev.times(w.gen[u * width + v]));
                                }
                            }
                        }
                    }
                    slot = if slot == 0 { rows - 1 } else { slot - 1 };
                    // RO is read only for 1 ≤ u ≤ U, rows i − 2 … i − 1 − U, which the
                    // ring of min(n, U + 2) rows holds.
                    slot_ro = if slot_ro == 0 {
                        ro_rows - 1
                    } else {
                        slot_ro - 1
                    };
                }
                // Region terms (§3.4 queries), appended to their class's generic fold.
                for (t, r) in rel.iter().enumerate() {
                    let planes = &s.acc[t * 4 * plane..(t + 1) * 4 * plane];
                    let (h, v_acc, hd, vd) = (
                        &planes[..plane],
                        &planes[plane..2 * plane],
                        &planes[2 * plane..3 * plane],
                        &planes[3 * plane..],
                    );
                    let mut g = generic[r.class];
                    if i > U + 1 {
                        let row = query_slot * m;
                        for (v, omega) in r.t.iter().enumerate() {
                            if let Some(omega) = omega {
                                let col = j + 2 + v;
                                if col < m {
                                    g = fma53(g, *omega, h[row + col]);
                                }
                            }
                        }
                    }
                    let col_b = j + U + 2;
                    if col_b < m {
                        for (u, omega) in r.b.iter().enumerate() {
                            if let Some(omega) = omega {
                                // Row i − 2 − u (u is the index − 1).
                                if i >= u + 2 {
                                    let row = ((i - 2 - u) % ACC_ROWS) * m;
                                    g = fma53(g, *omega, v_acc[row + col_b]);
                                }
                            }
                        }
                    }
                    if i > U + 1 {
                        let row = query_slot * m;
                        if let Some(omega) = r.lo {
                            if col_b < m {
                                g = fma53(g, omega, hd[row + col_b]);
                            }
                        }
                        if let Some(omega) = r.hi {
                            let col = j + U + 3;
                            if col < m {
                                g = fma53(g, omega, vd[row + col]);
                            }
                        }
                    }
                    generic[r.class] = g;
                }
                // The WIN terms are the last terms of RC, classes in grammar order.
                let in_base = usize::from(a[i - 1]) * 4 + usize::from(b[j + 1]);
                for (cl, zg) in generic.iter().enumerate().take(nc) {
                    let inner = w.loop_in[(cl * 16 + in_base) * np + pi];
                    special = special.plus(zg.times(inner));
                }
            }
            let alpha = if i >= 1 { usize::from(a[i - 1]) } else { 4 };
            let beta = if j + 1 < m { usize::from(b[j + 1]) } else { 4 };
            let ra = start[(alpha * np + pi) * 5 + beta].plus(special);
            s.ra[base + j] = ra;
            let ro = &mut s.ro[(base_ro + j) * nc..(base_ro + j + 1) * nc];
            if i + 1 < n && j >= 1 {
                let tail = usize::from(a[i + 1]) * 4 + usize::from(b[j - 1]);
                for (cl, slot) in ro.iter_mut().enumerate() {
                    *slot = ra.times(w.loop_out[(cl * np + pi) * 16 + tail]);
                }
            } else {
                ro.fill(F53::ZERO);
            }
            if !special.is_zero() {
                let alpha = if i + 1 < n { usize::from(a[i + 1]) } else { 4 };
                let beta = if j >= 1 { usize::from(b[j - 1]) } else { 4 };
                let end = w.end[(pi * 5 + alpha) * 5 + beta];
                s.col[j] = s.col[j].plus(special.times(end));
            }
        }
        // Row i is final: the accumulators of row i (§3.4).
        let here = (i % ACC_ROWS) * m;
        let before = ((i + ACC_ROWS - 1) % ACC_ROWS) * m;
        for (t, r) in rel.iter().enumerate() {
            let planes = &mut s.acc[t * 4 * plane..(t + 1) * 4 * plane];
            let (h, rest) = planes.split_at_mut(plane);
            let (v_acc, rest) = rest.split_at_mut(plane);
            let (hd, vd) = rest.split_at_mut(plane);
            let ro_of = |l: usize| s.ro[(base_ro + l) * nc + r.class];
            if r.need_h {
                for l in 0..m {
                    let prev = if i >= 1 { h[before + l] } else { F53::ZERO };
                    h[here + l] = fma53(ro_of(l), r.x, prev);
                }
                if r.lo.is_some() {
                    for l in 0..m {
                        let prev = if i >= 1 && l + 1 < m {
                            hd[before + l + 1]
                        } else {
                            F53::ZERO
                        };
                        hd[here + l] = fma53(h[here + l], r.z, prev);
                    }
                }
            }
            if r.need_v {
                let mut right = F53::ZERO;
                for l in (0..m).rev() {
                    right = fma53(ro_of(l), r.x, right);
                    v_acc[here + l] = right;
                }
                if r.hi.is_some() {
                    for l in 0..m {
                        let prev = if i >= 1 && l + 1 < m {
                            vd[before + l + 1]
                        } else {
                            F53::ZERO
                        };
                        vd[here + l] = fma53(v_acc[here + l], r.z, prev);
                    }
                }
            }
        }
        cur += 1;
        if cur == rows {
            cur = 0;
        }
        cur_ro += 1;
        if cur_ro == ro_rows {
            cur_ro = 0;
        }
    }
    let total = s.col.iter().fold(F53::ZERO, |acc, &e| acc.plus(e));
    s.release();
    Ok(total)
}
