//! K2 `duplex` `compat="primer3"`: primer3-py 2.3.1 module-level `calc_heterodimer`
//! (`end` ANY), `calc_end_stability` (END1, and END2 on the swapped pair) and, through a
//! pair of identical sequences, `calc_homodimer`, all with `max_loop` 30, replayed in
//! binary64 from the run-time parameter files with a correctly rounded natural logarithm
//! (named reference `primer3-py@2.3.1/cr-log`).
//! Spec: docs/wiki/thermo/dimer-primer3.md (§5 per-call constants, §6 per-pair
//! computation, §7 fields, §8 sequence checks), docs/wiki/thermo/primer3-alignment.md
//! (shared rules) and docs/wiki/thermo/primer3-parameter-loader.md (the tables).
//! Derivation route: a behaviour specification written from a reading of primer3-py
//! 2.3.1's source (no code copied; D94), confirmed against the unmodified tool; this
//! module was written from the wiki pages only.
//!
//! Every formula is a sequence of separate binary64 operations in the grouping the page
//! states; rustc never contracts to FMA and nothing here calls `mul_add`. Additions of
//! an explicit `+0.0` are performed. Comparisons follow IEEE semantics (every comparison
//! with NaN is false) and are written in the direction the page states. The kernel
//! implements the primary readings of §11.6 only (V1a: internal-loop closures use M1;
//! V1b: the end-term mismatch option uses M2; V2–V5 as primary; V6 and V7: the
//! unavailable stack has the ratio Θ_Ω of the pair's Q, and availability is the value
//! test on the diagonal predecessor plus a finite NN entry).
use std::{cell::RefCell, sync::Arc};

use seqblaze_core::{
    detail::{canonical, try_reserve_exact, DetailColumns, DetailField, DetailRequest, FieldKind},
    fpenv,
    kernel::PairDistanceKernel,
    plan::BoundAvailability,
    sched::Workers,
    seq::{ByteView, SeqTable},
    validate::EngineError,
    work::{Orientation, PairKind, WorkSource},
};

use crate::{
    cr::{self, Rational},
    duplex::pf::key_of,
    hairpin::{
        add, approx, clamp, parameter, py_float, Args, HairpinPack, ALPHA1, K0, KCAL_DIV, MAX_LOOP,
        T37,
    },
    p3files::{at, entry_words, idx3, idx4, is_wc, Hs, P3DevicePack, P3Tables, OMEGA},
};

/// The engine's length limit of the thal compat presets: at least one strand of each
/// pair has at most this many nt (§1, QD4).
pub const MAX_SHORT: usize = 60;
/// Every strand has at most this many nt (§1, QD4; primer3's own limit).
pub const MAX_LENGTH: usize = 10_000;
/// Duplex initiation of SantaLucia & Hicks 2004 Table 1 (ledger `sh2004:init`: ΔH
/// 0.2 kcal/mol, exactly 200 cal/mol; ΔS −5.7 cal/(mol·K)), shared page §3.3.
pub const INIT_H: f64 = 200.0;
pub const INIT_S: f64 = -5.7;
/// The gas constant as printed in SantaLucia & Hicks 2004 (ledger
/// `sh2004:gas_constant_r`), cal/(mol·K): the alignment engine's printing (shared page
/// §3.3 conflict note), not the 1.987 of `calc_tm`.
pub const GAS_R: f64 = 1.9872;
/// Strand-concentration divisors D₄ = 4·10⁹ and D₁ = 10⁹ (nM), exact in binary64 (§3.2).
pub const D4: f64 = 4e9;
pub const D1: f64 = 1e9;
/// The selection shift fl(10⁻⁶) on ΔH and ΔS of the right-end term (class C, §3.2).
pub const SHIFT: f64 = 1e-6;
/// The message prefix of the preset's errors.
const LABEL: &str = "duplex compat='primer3'";
/// The reference calls named in the pack errors.
const CALLS: &str = "dimer calls (calc_heterodimer, calc_homodimer, calc_end_stability)";

/// Detail fields of the preset (§7) besides the primary `dg_kcal`, which the engine
/// carries as its int64 key (the K3 key, duplex-pf.md §9.2). `found` is carried as int32
/// 0/1 (mask INT32_MAX) and exposed as bool by the binding.
pub const FOUND: usize = 0;
pub const DH_KCAL: usize = 1;
pub const DS_CAL_PER_K: usize = 2;
pub const TM_C: usize = 3;

pub static FIELDS: [DetailField; 4] = [
    DetailField {
        name: "found",
        kind: FieldKind::Int32,
    },
    DetailField {
        name: "dh_kcal",
        kind: FieldKind::Float64,
    },
    DetailField {
        name: "ds_cal_per_k",
        kind: FieldKind::Float64,
    },
    DetailField {
        name: "tm_c",
        kind: FieldKind::Float64,
    },
];

/// The `end` option (§2, QD2).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum End {
    /// `calc_heterodimer(a, b)`: every cell is scanned.
    Any,
    /// `calc_end_stability(a, b)`: the 3′-terminal base of a must be paired.
    End1,
    /// `calc_end_stability(b, a)`, computed for the pair (a, b).
    End2,
}

impl End {
    pub fn parse(name: &str) -> Option<Self> {
        match name {
            "any" => Some(Self::Any),
            "end1" => Some(Self::End1),
            "end2" => Some(Self::End2),
            _ => None,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Any => "any",
            Self::End1 => "end1",
            Self::End2 => "end2",
        }
    }
}

/// The dimer calls' salt, temperature and DNA-concentration inputs after the §4 mapping
/// (mM, nM, °C).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DimerArgs {
    pub salt: Args,
    pub oligo_nm: f64,
}

/// The per-call constants H1–H8 of §5, built once on the host.
#[derive(Debug, Clone, Copy)]
pub struct DimerPack {
    /// H1–H6: T2's steps unchanged (mv, w, μ, x_μ, λ = CR(ln x_μ), σ, T_u).
    pub salt: HairpinPack,
    /// H7: x₄ = fl(c / D₄) and x₁ = fl(c / D₁).
    pub x4: f64,
    pub x1: f64,
    /// H8: Q₄ = fl(1.9872 · CR(ln x₄)), Q₁ = fl(1.9872 · CR(ln x₁)), cal/(mol·K).
    pub q4: f64,
    pub q1: f64,
    /// H8: Θ_Ω at Q₄ and at Q₁ (§6.1): −∞ below Q = fl(6.7), +∞ from it on.
    pub theta_omega4: f64,
    pub theta_omega1: f64,
}

/// Θ_Ω = fl(fl(+∞ + 200) / fl(fl(−1 + (−5.7)) + Q)), the end ratio of the forbidden
/// value (§6.1). Its denominator is +0 exactly at Q = fl(6.7), so the value is −∞ below
/// and +∞ from there on, whatever Q's last bits; expects E0.
fn theta_omega(q: f64) -> f64 {
    let num = OMEGA.h + INIT_H;
    let den = (OMEGA.s + INIT_S) + q;
    num / den
}

impl DimerPack {
    /// H1–H8 of §5; raises `ParameterError` in step order (H3: μ zero or infinite, H4:
    /// x_μ zero, H7: x₄ zero). Runs in the IEEE default floating-point environment
    /// whatever the caller's state, which is restored on return (D99, §13.1).
    pub fn new(args: &DimerArgs) -> Result<Self, EngineError> {
        fpenv::guarded(|| Self::build(args))
    }

    fn build(args: &DimerArgs) -> Result<Self, EngineError> {
        let salt = HairpinPack::build(&args.salt, LABEL, CALLS)?;
        let c = args.oligo_nm;
        // sb.Conditions admits only a positive finite oligo_nM; a direct binding call
        // is checked too (the logarithm needs a positive argument).
        if !c.is_finite() || c <= 0.0 {
            return Err(parameter(format!(
                "{LABEL}: oligo_nM={} must be finite and > 0",
                py_float(c)
            )));
        }
        let x4 = c / D4; // H7
        let x1 = c / D1;
        if x4 == 0.0 {
            // QD6: data-independent (the self-complementary pairs' x₁ may still be > 0).
            return Err(parameter(format!(
                "{LABEL}: oligo_nM={} underflows to a strand concentration of 0 mol/L \
                 (oligo_nM/4e9); primer3's {CALLS} would take log(0) and return a \
                 non-finite concentration term. Use a larger oligo_nM, e.g. 50",
                py_float(c)
            )));
        }
        let ln_c = |x: f64| {
            cr::ln(&Rational::from_f64(x), &Rational::from_ratio(1, 1)).map_err(|cap| {
                parameter(format!(
                    "internal error: the correctly rounded ln_c needs more than {} bits; \
                     nothing was computed",
                    cap.bits
                ))
            })
        };
        let q4 = GAS_R * ln_c(x4)?; // H8
        let q1 = GAS_R * ln_c(x1)?;
        Ok(Self {
            salt,
            x4,
            x1,
            q4,
            q1,
            theta_omega4: theta_omega(q4),
            theta_omega1: theta_omega(q1),
        })
    }
}

/// One pair's reported values (§6.6, §7).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DimerRow {
    pub found: bool,
    pub dg_kcal: f64,
    pub dh_kcal: f64,
    pub ds_cal_per_k: f64,
    pub tm_c: f64,
}

/// The not-found row (§6.6, QD7): primer3's +0.0 values.
const NOT_FOUND: DimerRow = DimerRow {
    found: false,
    dg_kcal: 0.0,
    dh_kcal: 0.0,
    ds_cal_per_k: 0.0,
    tm_c: 0.0,
};

/// The concentration-dependent constants of one pair: Q, Θ_Ω and the end-term table
/// computed with that Q.
struct PairQ<'k> {
    q: f64,
    theta_omega: f64,
    ends: &'k [Hs],
}

/// Θ_e(o) = fl(fl(o_H + 200) / fl(fl(o_S + (−5.7)) + Q)) (§6.1).
#[inline]
fn theta_e(o: Hs, q: f64) -> f64 {
    let num = o.h + INIT_H;
    let den = (o.s + INIT_S) + q;
    num / den
}

/// G(o) = fl(o_H − fl(T₃₇ · o_S)) (§6.1).
#[inline]
fn g_end(o: Hs) -> f64 {
    o.h - T37 * o.s
}

/// Θ_R(v) = fl(fl(fl(v_H + 200) + R_H) / fl(fl(fl(v_S + (−5.7)) + R_S) + Q)) (§6.1).
#[inline]
fn theta_r(v: Hs, r: Hs, q: f64) -> f64 {
    let num = (v.h + INIT_H) + r.h;
    let den = ((v.s + INIT_S) + r.s) + q;
    num / den
}

/// G_R(v) = fl(fl(v_H + R_H) − fl(T₃₇ · fl(v_S + R_S))) (§6.1).
#[inline]
fn g_r(v: Hs, r: Hs) -> f64 {
    (v.h + r.h) - T37 * (v.s + r.s)
}

/// Γ_R(v) = fl(fl(fl(v_H + fl(R_H + 10⁻⁶)) + 200) − fl(T₃₇ · fl(fl(v_S + fl(R_S + 10⁻⁶))
/// + (−5.7)))) (§6.1), the selection score.
#[inline]
fn gamma(v: Hs, r: Hs) -> f64 {
    let h = (v.h + (r.h + SHIFT)) + INIT_H;
    let s = (v.s + (r.s + SHIFT)) + INIT_S;
    h - T37 * s
}

/// The duplex-end term E of the end pair φ·ψ with neighbours φ′, ψ′ (§6.2), at the
/// concentration term `q`. Expects E0.
fn end_term(t: &P3Tables, phi: u8, psi: u8, phi_n: u8, psi_n: u8, q: f64) -> Hs {
    let bare = at(phi, psi);
    if is_wc(phi_n, psi_n) {
        // No dangle option: E = Bare, even when the mismatch option is favourable.
        return bare;
    }
    let d3 = t.d3[idx3(phi, psi, phi_n)];
    let d5 = t.d5[idx3(phi, psi, psi_n)];
    let dan = match (d3.h.is_finite(), d5.h.is_finite()) {
        (true, true) => add(add(bare, d3), d5),
        (true, false) => add(bare, d3),
        (false, true) => add(bare, d5),
        (false, false) => return bare,
    };
    // V1b primary: the end-term mismatch option uses M2.
    let mis = add(bare, t.m2[idx4(phi, phi_n, psi, psi_n)]);
    let mis_usable = mis.h.is_finite() && g_end(mis) < 0.0;
    let dan_usable = g_end(dan) < 0.0;
    let candidate = match (mis_usable, dan_usable) {
        (true, true) => {
            if theta_e(mis, q) < theta_e(dan, q) {
                dan
            } else {
                mis
            }
        }
        (true, false) => mis,
        (false, true) => dan,
        (false, false) => return bare,
    };
    // V2 primary: the candidate iff Θ_e(candidate) ≥ Θ_e(B).
    if theta_e(candidate, q) >= theta_e(bare, q) {
        candidate
    } else {
        bare
    }
}

/// The end-term table at `q`: E(φ, ψ, φ′, ψ′) at `idx4(φ, ψ, φ′, ψ′)` for every base φ, ψ
/// and every code φ′, ψ′ (code 4 beyond a strand end). Expects E0.
fn end_table(t: &P3Tables, q: f64) -> Vec<Hs> {
    let mut table = vec![OMEGA; 625];
    for phi in 0..4u8 {
        for psi in 0..4u8 {
            for phi_n in 0..5u8 {
                for psi_n in 0..5u8 {
                    table[idx4(phi, psi, phi_n, psi_n)] = end_term(t, phi, psi, phi_n, psi_n, q);
                }
            }
        }
    }
    table
}

/// Letter code of an upper-case ACGT byte (shared page §2.2).
#[inline]
fn code(byte: u8) -> u8 {
    match byte {
        b'A' => 0,
        b'C' => 1,
        b'G' => 2,
        b'T' => 3,
        _ => 4,
    }
}

/// Whether coded bases `s[1..=len]` are self-complementary (§5): even length and every
/// base of the first half pairs with its mirror base.
fn self_complementary(s: &[u8], len: usize) -> bool {
    len % 2 == 0 && (1..=len / 2).all(|i| is_wc(s[i], s[len + 1 - i]))
}

/// Workspace bytes kept by a thread between pairs; a larger table is released after
/// its pair (a 60 × 10,000 pair needs about 10 MB, §13).
const KEEP_BYTES: usize = 1 << 20;

/// The per-thread buffers of the fill: the codes of the two strands and the cell table.
/// Grown fallibly to the largest pair a thread meets and reused across pairs; every
/// buffer is fully rewritten for each pair.
#[derive(Default)]
struct Workspace {
    x: Vec<u8>,
    y: Vec<u8>,
    cell: Vec<Hs>,
}

thread_local! {
    static WORKSPACE: RefCell<Workspace> = RefCell::new(Workspace::default());
}

/// Resize `buffer` to `len` items of `fill`, reserving fallibly (a refused reservation
/// is a DeviceError through the engine's device-error helper, the 286f1f8 rule).
fn resize<T: Copy>(
    buffer: &mut Vec<T>,
    len: usize,
    fill: T,
    what: &str,
) -> Result<(), EngineError> {
    buffer.clear();
    try_reserve_exact(buffer, len, what)?;
    buffer.resize(len, fill);
    Ok(())
}

impl Workspace {
    fn release(&mut self) {
        if self.cell.capacity() * std::mem::size_of::<Hs>() > KEEP_BYTES {
            self.cell = Vec::new();
        }
        if self.x.capacity() > KEEP_BYTES {
            self.x = Vec::new();
        }
        if self.y.capacity() > KEEP_BYTES {
            self.y = Vec::new();
        }
    }
}

/// The fill, selection and backtrace state of one pair (§6). Codes are one-based:
/// `x[1..=L₁]` the first strand 5′→3′, `y[1..=L₂]` the second strand reversed (not
/// complemented), code 4 at index 0 and L + 1.
struct Fold<'w> {
    t: &'w P3Tables,
    pq: PairQ<'w>,
    l1: usize,
    l2: usize,
    stride: usize,
    x: &'w [u8],
    y: &'w [u8],
    /// C(u, z) at `u * stride + z`, stride L₂ + 1; row 0 and column 0 unused.
    cell: &'w mut [Hs],
}

impl Fold<'_> {
    #[inline]
    fn c(&self, u: usize, z: usize) -> Hs {
        self.cell[u * self.stride + z]
    }

    /// R(u, z): the right-end term (§6.2).
    #[inline]
    fn right(&self, u: usize, z: usize) -> Hs {
        self.pq.ends[idx4(self.x[u], self.y[z], self.x[u + 1], self.y[z + 1])]
    }

    /// Lt(u, z): the left-end term (§6.2), the right-end rule with the strand roles
    /// exchanged.
    #[inline]
    fn left(&self, u: usize, z: usize) -> Hs {
        self.pq.ends[idx4(self.y[z], self.x[u], self.y[z - 1], self.x[u - 1])]
    }

    /// The NN entry of the stack of (u − 1, z − 1) on (u, z).
    #[inline]
    fn stack_nn(&self, u: usize, z: usize) -> Hs {
        self.t.nn[idx4(self.x[u - 1], self.x[u], self.y[z - 1], self.y[z])]
    }

    /// The loop candidate X of §6.3 stage 3 from the predecessor (u′, z′) = (u − α − 1,
    /// z − β − 1), whose value `pred` is not Ω; guards included, no clamp. A candidate
    /// whose ΔH is not finite is Ω by the value test.
    #[inline]
    fn loop_candidate(&self, u: usize, z: usize, alpha: usize, beta: usize, pred: Hs) -> Hs {
        let t = self.t;
        let (x, y) = (self.x, self.y);
        let (up, zp) = (u - alpha - 1, z - beta - 1);
        let ell = alpha + beta;
        let value = if alpha == 0 || beta == 0 {
            if ell == 1 {
                // Bulge of one: the stack across the bulge, guarded on the increment.
                let inc = add(t.loop_bu[1], t.nn[idx4(x[up], x[u], y[zp], y[z])]);
                if inc.h > 0.0 || inc.s > 0.0 {
                    return OMEGA;
                }
                return add(inc, pred);
            }
            add(
                add(add(t.loop_bu[ell], at(x[up], y[zp])), at(x[u], y[z])),
                pred,
            )
        } else if alpha == 1 && beta == 1 {
            let outer = t.mm[idx4(x[up], x[up + 1], y[zp], y[zp + 1])];
            let inner = t.mm[idx4(y[z], y[z - 1], x[u], x[u - 1])];
            add(add(outer, inner), pred)
        } else {
            // V1a primary: internal-loop closures use M1.
            let outer = t.m1[idx4(x[up], x[up + 1], y[zp], y[zp + 1])];
            let inner = t.m1[idx4(y[z], y[z - 1], x[u], x[u - 1])];
            let asym = ALPHA1 * (alpha as f64 - beta as f64).abs();
            let lp = t.loop_in[ell];
            Hs {
                h: (((lp.h + outer.h) + inner.h) + 0.0) + pred.h,
                s: (((lp.s + outer.s) + inner.s) + asym) + pred.s,
            }
        };
        if !value.h.is_finite() || (value.h > 0.0 && value.s > 0.0) {
            OMEGA
        } else {
            value
        }
    }

    /// Visit the loop predecessors of (u, z) in the stage-3 order: ℓ = α + β ascending
    /// from 1 to 30, then α ascending; only u′, z′ ≥ 1 and predecessors not holding Ω.
    /// `visit(α, β, C(u′, z′))` returns true to stop.
    #[inline]
    fn scan_loops(&self, u: usize, z: usize, mut visit: impl FnMut(usize, usize, Hs) -> bool) {
        if u < 2 || z < 2 {
            return;
        }
        let (amax, bmax) = (u - 2, z - 2);
        let lmax = MAX_LOOP.min(amax + bmax);
        for ell in 1..=lmax {
            let lo = ell.saturating_sub(bmax);
            let hi = ell.min(amax);
            for alpha in lo..=hi {
                let beta = ell - alpha;
                let pred = self.c(u - alpha - 1, z - beta - 1);
                if pred.h.is_finite() && visit(alpha, beta, pred) {
                    return;
                }
            }
        }
    }

    /// C(u, z) of an admissible cell: stages 1–3 of §6.3.
    fn cell_value(&self, u: usize, z: usize) -> Hs {
        let r = self.right(u, z);
        let q = self.pq.q;
        // Stage 1: initiation.
        let mut v = self.left(u, z);
        if u > 1 && z > 1 {
            // Stage 2: the stack, available iff the diagonal predecessor's value is not
            // Ω and the NN entry is finite; else Θ_Ω of the pair's Q (oracle rule, Y7).
            let pred = self.c(u - 1, z - 1);
            let nn = self.stack_nn(u, z);
            let (stacked, theta) = if pred.h.is_finite() && nn.h.is_finite() {
                let s = add(pred, nn);
                (s, theta_r(s, r, q))
            } else {
                (OMEGA, self.pq.theta_omega)
            };
            if theta > theta_r(v, r, q) {
                v = stacked;
            }
            v = clamp(v);
            // Stage 3: loops on G_R, strict, against the current value.
            let mut gv = g_r(v, r);
            self.scan_loops(u, z, |alpha, beta, pred| {
                let x = self.loop_candidate(u, z, alpha, beta, pred);
                if x.h.is_finite() {
                    let gx = g_r(x, r);
                    if gx < gv {
                        v = clamp(x);
                        gv = g_r(v, r);
                    }
                }
                false
            });
        }
        v
    }

    fn fill(&mut self) {
        for u in 1..=self.l1 {
            for z in 1..=self.l2 {
                let value = if is_wc(self.x[u], self.y[z]) {
                    self.cell_value(u, z)
                } else {
                    OMEGA
                };
                self.cell[u * self.stride + z] = value;
            }
        }
    }

    /// The reported cell (§6.4): the first strict minimum of Γ_R over the scanned cells
    /// (every cell for ANY, the row u = L₁ for END1), starting from (1, 1) with +∞.
    fn select(&self, end1: bool) -> (usize, usize) {
        let mut best = (1, 1);
        let mut score = f64::INFINITY;
        let first = if end1 { self.l1 } else { 1 };
        for u in first..=self.l1 {
            for z in 1..=self.l2 {
                let c = self.c(u, z);
                if !c.h.is_finite() {
                    continue; // Ω scores +∞, never strictly lower
                }
                let g = gamma(c, self.right(u, z));
                if g < score {
                    score = g;
                    best = (u, z);
                }
            }
        }
        best
    }

    /// The backtrace path from (u, z) (§6.5), the reported cell included; `pair` is
    /// called for every cell on it, in trace order.
    fn backtrace(&self, mut u: usize, mut z: usize, pair: &mut impl FnMut(usize, usize)) {
        loop {
            pair(u, z);
            let c = self.c(u, z);
            // Row 1: the duplex starts here.
            if approx(c, self.left(u, z)) {
                return;
            }
            // Row 2: the stack.
            if u > 1 && z > 1 {
                let pred = self.c(u - 1, z - 1);
                if pred.h.is_finite() && approx(c, add(pred, self.stack_nn(u, z))) {
                    u -= 1;
                    z -= 1;
                    continue;
                }
            }
            // Row 3: the first loop predecessor in stage-3 order whose candidate matches.
            let mut next = None;
            self.scan_loops(u, z, |alpha, beta, pred| {
                if approx(c, self.loop_candidate(u, z, alpha, beta, pred)) {
                    next = Some((u - alpha - 1, z - beta - 1));
                    true
                } else {
                    false
                }
            });
            match next {
                Some((up, zp)) => {
                    u = up;
                    z = zp;
                }
                // Row 4: stop.
                None => return,
            }
        }
    }
}

/// The K2 pair kernel of one call: the per-call pack, the parameter set, `end`, the
/// requested detail fields and the end-term tables at Q₄ and Q₁.
pub struct DimerPrimer3 {
    pack: DimerPack,
    tables: Arc<P3Tables>,
    end: End,
    request: DetailRequest,
    /// E(φ, ψ, φ′, ψ′) at Q₄ (`ends[0]`) and Q₁ (`ends[1]`).
    ends: [Vec<Hs>; 2],
}

impl DimerPrimer3 {
    /// `fields` by name, of [`FIELDS`] (the facade checks the public names and passes
    /// the primary `dg_kcal` as the key column). The end-term tables are built here, in
    /// the IEEE default floating-point environment whatever the caller's state (D99).
    pub fn new(
        pack: DimerPack,
        tables: Arc<P3Tables>,
        end: End,
        fields: &[&str],
    ) -> Result<Self, EngineError> {
        let mut request = DetailRequest::NONE;
        for name in fields {
            let index = FIELDS
                .iter()
                .position(|field| field.name == *name)
                .ok_or_else(|| {
                    parameter(format!(
                        "field {name:?} is not available for {LABEL}; use any of \
                         ['found', 'dg_kcal', 'dh_kcal', 'ds_cal_per_k', 'tm_c']"
                    ))
                })?;
            request = request.with(index);
        }
        let ends = fpenv::guarded(|| [end_table(&tables, pack.q4), end_table(&tables, pack.q1)]);
        Ok(Self {
            pack,
            tables,
            end,
            request,
            ends,
        })
    }

    /// The device pack of primer3-cuda.md §4: the handle's tables, both end-term tables
    /// exactly as [`DimerPrimer3::new`] built them (at Q₄, then at Q₁), and the scalars,
    /// as bit patterns. A bit copy, with no floating-point operation; a refused
    /// reservation is a DeviceError.
    pub fn device_pack(&self) -> Result<P3DevicePack, EngineError> {
        let tables = self.tables.device_words()?;
        let mut ends = entry_words(&self.ends[0])?;
        let mut second = entry_words(&self.ends[1])?;
        seqblaze_core::detail::try_reserve_exact(&mut ends, second.len(), "dimer end terms")?;
        ends.append(&mut second);
        let pack = &self.pack;
        Ok(P3DevicePack {
            tables,
            ends,
            scalars: crate::p3files::P3DeviceScalars {
                sigma: pack.salt.sigma.to_bits(),
                t_u: pack.salt.t_u.to_bits(),
                q4: pack.q4.to_bits(),
                q1: pack.q1.to_bits(),
                theta_omega4: pack.theta_omega4.to_bits(),
                theta_omega1: pack.theta_omega1.to_bits(),
                ..crate::hairpin::device_constants()
            },
        })
    }

    pub fn pack(&self) -> &DimerPack {
        &self.pack
    }

    pub fn end(&self) -> End {
        self.end
    }

    /// Fill one pair in the calling thread's workspace and hand the fold to `body`.
    /// `a` and `b` are the operands after the `end` mapping (END2 already swapped);
    /// `copy` writes a strand's ASCII bytes into a slice. Expects E0.
    fn with_fold<R>(
        &self,
        (la, lb): (usize, usize),
        copy: impl FnOnce(&mut [u8], &mut [u8]),
        body: impl FnOnce(&Fold<'_>) -> R,
    ) -> Result<R, EngineError> {
        WORKSPACE.with(|cell| {
            let mut guard = cell.borrow_mut();
            let ws = &mut *guard;
            resize(&mut ws.x, la + 2, 4, "dimer strand codes")?;
            resize(&mut ws.y, lb + 2, 4, "dimer strand codes")?;
            let cells = (la + 1)
                .checked_mul(lb + 1)
                .ok_or_else(|| EngineError::Parameter("dimer cell table size overflows".into()))?;
            resize(&mut ws.cell, cells, OMEGA, "dimer cell table")?;
            copy(&mut ws.x[1..=la], &mut ws.y[1..=lb]);
            for byte in &mut ws.x[1..=la] {
                *byte = code(*byte);
            }
            // y: the second strand reversed, not complemented.
            ws.y[1..=lb].reverse();
            for byte in &mut ws.y[1..=lb] {
                *byte = code(*byte);
            }
            let both = self_complementary(&ws.x, la) && self_complementary(&ws.y, lb);
            let pq = if both {
                PairQ {
                    q: self.pack.q1,
                    theta_omega: self.pack.theta_omega1,
                    ends: &self.ends[1],
                }
            } else {
                PairQ {
                    q: self.pack.q4,
                    theta_omega: self.pack.theta_omega4,
                    ends: &self.ends[0],
                }
            };
            let mut fold = Fold {
                t: &self.tables,
                pq,
                l1: la,
                l2: lb,
                stride: lb + 1,
                x: &ws.x,
                y: &ws.y,
                cell: &mut ws.cell,
            };
            fold.fill();
            let out = body(&fold);
            ws.release();
            Ok(out)
        })
    }

    /// The operands of a pair after the `end` mapping, as views (END2 swaps).
    fn operands<'v>(&self, a: ByteView<'v>, b: ByteView<'v>) -> (ByteView<'v>, ByteView<'v>) {
        if self.end == End::End2 {
            (b, a)
        } else {
            (a, b)
        }
    }

    /// One pair's row (§6.4–§6.6). Expects E0: it runs only inside guarded scheduler
    /// units and carries no guard of its own (§13.1).
    pub fn row(&self, a: ByteView<'_>, b: ByteView<'_>) -> Result<DimerRow, EngineError> {
        let (a, b) = self.operands(a, b);
        let end1 = self.end != End::Any;
        let sigma = self.pack.salt.sigma;
        let t_u = self.pack.salt.t_u;
        self.with_fold(
            (a.len(), b.len()),
            |x, y| {
                a.copy_to(0, x);
                b.copy_to(0, y);
            },
            |fold| {
                let (u, z) = fold.select(end1);
                if !is_wc(fold.x[u], fold.y[z]) {
                    return NOT_FOUND;
                }
                let mut n_bp: usize = 0;
                fold.backtrace(u, z, &mut |_, _| n_bp += 1);
                let nu = (n_bp - 1) as f64;
                let c = fold.c(u, z);
                let r = fold.right(u, z);
                let q = fold.pq.q;
                let dh = (c.h + r.h) + INIT_H;
                let ds = ((c.s + r.s) + INIT_S) + nu * sigma;
                let dg = dh - t_u * ds;
                let tm = dh / (ds + q) - K0;
                // A NaN dg (dh = -inf with fl(T_u * ds) = -inf, or ds = +inf at T_u = +0;
                // synthetic tables only) is canonical before it reaches the key, so the
                // key does not depend on the platform's default NaN: it lies above +inf's
                // key, fails every finite cutoff and ranks last, as on the device.
                DimerRow {
                    found: true,
                    dg_kcal: canonical(dg / KCAL_DIV),
                    dh_kcal: dh / KCAL_DIV,
                    ds_cal_per_k: ds,
                    tm_c: tm,
                }
            },
        )
    }

    /// The backtrace pairs (u, z) of one pair, one-based in the coordinates of §6.1 of
    /// the mapped operands (END2: the swapped pair), the reported cell included; empty
    /// when nothing is found (the testing probe, §13). Expects E0.
    pub fn pairs(&self, a: &[u8], b: &[u8]) -> Result<Vec<(usize, usize)>, EngineError> {
        let (a, b) = if self.end == End::End2 {
            (b, a)
        } else {
            (a, b)
        };
        let end1 = self.end != End::Any;
        self.with_fold(
            (a.len(), b.len()),
            |x, y| {
                x.copy_from_slice(a);
                y.copy_from_slice(b);
            },
            |fold| {
                let (u, z) = fold.select(end1);
                let mut path = Vec::new();
                if is_wc(fold.x[u], fold.y[z]) {
                    // At most min(L₁, L₂) ≤ 60 cells: no reallocation below.
                    try_reserve_exact(&mut path, MAX_SHORT, "dimer probe pairs")?;
                    fold.backtrace(u, z, &mut |u, z| path.push((u, z)));
                }
                Ok(path)
            },
        )?
    }

    fn push(&self, row: &DimerRow, out: &mut DetailColumns) -> Result<(), EngineError> {
        if out.wants(FOUND) {
            out.push_i32(FOUND, i32::from(row.found))?;
        }
        for (field, value) in [
            (DH_KCAL, row.dh_kcal),
            (DS_CAL_PER_K, row.ds_cal_per_k),
            (TM_C, row.tm_c),
        ] {
            if out.wants(field) {
                out.push_f64(field, value)?;
            }
        }
        Ok(())
    }
}

/// The `SequenceError` of a sequence of length 0 or above 10,000 nt (§8 step 4.2).
fn length_error(input: &'static str, index: usize, length: usize) -> EngineError {
    let message = if length == 0 {
        format!(
            "duplex input {input} sequence {index} is empty (length 0, minimum 1 nt): primer3's \
             dimer calls raise for an empty sequence; nothing was computed"
        )
    } else {
        format!(
            "duplex input {input} sequence {index} has length {length}, above the limit \
             {MAX_LENGTH} nt of compat='primer3' (primer3's thermodynamic alignment takes at \
             most {MAX_LENGTH} nt per strand); nothing was computed"
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

/// Exact joint-length diagnostic for device metadata validation. This helper
/// does not read input bytes or change the ordering of the public preflight.
pub fn metadata_pair_error(k: usize, i: usize, j: usize, la: usize, lb: usize) -> EngineError {
    pair_error(k, i, j, la, lb)
}

/// The `SequenceError` of pair k = (i, j) whose two strands are both long (§8 step 4.3).
fn pair_error(k: usize, i: usize, j: usize, la: usize, lb: usize) -> EngineError {
    EngineError::Sequence {
        message: format!(
            "duplex compat='primer3': pair {k} ({i}, {j}) has two strands longer than \
             {MAX_SHORT} nt (lengths {la} and {lb}); at least one strand of each pair must \
             be at most {MAX_SHORT} nt (primer3's thermodynamic alignment); nothing was \
             computed"
        ),
        input: "a",
        index: i,
        position: None,
        byte: None,
        n_bad: 1,
        bad_indices: vec![i],
    }
}

/// The checks of §8 step 4.2 and 4.3 on two byte strings (the testing probe): lengths
/// 1–10,000 (a before b), then at least one strand of at most 60 nt. The alphabet is
/// checked by the caller first.
pub fn check_pair_lengths(a: &[u8], b: &[u8]) -> Result<(), EngineError> {
    for (input, seq) in [("a", a), ("b", b)] {
        if seq.is_empty() || seq.len() > MAX_LENGTH {
            return Err(length_error(input, 0, seq.len()));
        }
    }
    if a.len() > MAX_SHORT && b.len() > MAX_SHORT {
        return Err(pair_error(0, 0, 0, a.len(), b.len()));
    }
    Ok(())
}

impl PairDistanceKernel<i64> for DimerPrimer3 {
    /// No bound (§7): `bounded` resolves to full with `no_bound`.
    fn bound_availability(&self) -> BoundAvailability {
        BoundAvailability::Unavailable
    }
    fn name(&self) -> &'static str {
        "duplex"
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
    /// Lengths are checked by `sequence_preflight` in one pass (L = 0 and L > 10,000,
    /// input a before b, then index; §8 step 4.2).
    fn length_caps(&self) -> [Option<usize>; 2] {
        [None, None]
    }
    /// Metadata-only callers reuse the exact existing diagnostic, including
    /// the preset's empty-input rule, after device validation has found its row.
    fn length_error(
        &self,
        input: &'static str,
        index: usize,
        length: usize,
        _limit: usize,
    ) -> EngineError {
        length_error(input, index, length)
    }
    fn sequence_preflight(&self, a: &SeqTable, b: &SeqTable) -> Result<(), EngineError> {
        let shared = std::ptr::eq(a, b);
        for (input, table) in [("a", a), ("b", b)] {
            if input == "b" && shared {
                continue;
            }
            if let Some(index) = (0..table.len()).find(|&i| {
                let len = table.seq_len(i);
                len == 0 || len > MAX_LENGTH
            }) {
                return Err(length_error(input, index, table.seq_len(index)));
            }
        }
        Ok(())
    }
    /// §8 step 4.3: the first pair, in output order, whose two strands are long, found
    /// per shape without visiting implicit pair lists.
    fn pair_preflight(
        &self,
        a: &SeqTable,
        b: &SeqTable,
        work: &WorkSource,
    ) -> Result<(), EngineError> {
        let long = |table: &SeqTable, i: usize| table.seq_len(i) > MAX_SHORT;
        let first_long =
            |table: &SeqTable, from: usize| (from..table.len()).find(|&i| long(table, i));
        let fail =
            |k: usize, i: usize, j: usize| Err(pair_error(k, i, j, a.seq_len(i), b.seq_len(j)));
        match &work.kind {
            PairKind::Zip | PairKind::Index { .. } => {
                for (k, (i, j)) in work.range(0, work.n_pairs).enumerate() {
                    if long(a, i) && long(b, j) {
                        return fail(k, i, j);
                    }
                }
            }
            PairKind::Cross | PairKind::OneToMany => {
                if let (Some(i), Some(j)) = (first_long(a, 0), first_long(b, 0)) {
                    return fail(i * b.len() + j, i, j);
                }
            }
            PairKind::Within { self_pairs } | PairKind::WithinOriented { self_pairs, .. } => {
                let both = matches!(
                    work.kind,
                    PairKind::WithinOriented {
                        orientation: Orientation::Both,
                        ..
                    }
                );
                let n = a.len();
                let s = usize::from(*self_pairs);
                let Some(r1) = first_long(a, 0) else {
                    return Ok(());
                };
                // Rows before r1 hold n − 1 + s pairs each under "both", n − 1 − r + s
                // under the upper triangle.
                let row_start = if both {
                    r1 * (n - 1 + s)
                } else {
                    r1 * (n - 1 + s) - r1 * r1.saturating_sub(1) / 2
                };
                if *self_pairs {
                    let column = if both { r1 } else { 0 };
                    return fail(row_start + column, r1, r1);
                }
                if let Some(r2) = first_long(a, r1 + 1) {
                    let column = if both { r2 - 1 } else { r2 - r1 - 1 };
                    return fail(row_start + column, r1, r2);
                }
            }
        }
        Ok(())
    }
    fn prepare_bound(
        &self,
        _a: &SeqTable,
        _b: &SeqTable,
        _workers: Workers,
    ) -> Result<(), EngineError> {
        Ok(())
    }
    fn lower_bound(&self, _a: &SeqTable, _b: &SeqTable, _i: usize, _j: usize) -> i64 {
        i64::MIN
    }
    /// About L₁ · L₂ cells, one in four admissible, with up to 495 loop predecessors
    /// each; scheduling only.
    fn pair_cost(&self, a: &SeqTable, b: &SeqTable, _with_details: bool) -> usize {
        a.mean_len()
            .max(1)
            .saturating_mul(b.mean_len().max(1))
            .saturating_mul(124)
            .max(1)
    }
    /// The exact key of `dg_kcal`; never `None` (no pruning).
    fn evaluate(
        &self,
        a: ByteView<'_>,
        b: ByteView<'_>,
        _cutoff: Option<i64>,
    ) -> Result<Option<i64>, EngineError> {
        Ok(Some(key_of(self.row(a, b)?.dg_kcal)))
    }
    fn detail_fields(&self) -> &'static [DetailField] {
        &FIELDS
    }
    fn detail_request(&self) -> DetailRequest {
        self.request
    }
    /// One row gives the key and the fields.
    fn fused_details(&self) -> bool {
        true
    }
    fn evaluate_detailed(
        &self,
        a: ByteView<'_>,
        b: ByteView<'_>,
        _prune: Option<i64>,
        keep: Option<i64>,
        out: &mut DetailColumns,
    ) -> Result<Option<i64>, EngineError> {
        let row = self.row(a, b)?;
        let key = key_of(row.dg_kcal);
        if keep.is_none_or(|limit| key <= limit) {
            self.push(&row, out)?;
        }
        Ok(Some(key))
    }
    /// The detail fields of a kept pair whose key is `score`. The recomputed key must
    /// equal it (primer3-cuda.md QPC-12): a selection pass on another backend that
    /// disagrees with this one is a DeviceError, and nothing is published. With the CPU's
    /// own keys it cannot fire.
    fn details(
        &self,
        a: ByteView<'_>,
        b: ByteView<'_>,
        score: i64,
        out: &mut DetailColumns,
    ) -> Result<(), EngineError> {
        let row = self.row(a, b)?;
        let key = key_of(row.dg_kcal);
        if key != score {
            return Err(EngineError::Device(format!(
                "{LABEL}: the detail pass of the pair of lengths ({}, {}) gives key {key} but \
                 the selection pass gave {score}; no result was published; use device='cpu'",
                a.len(),
                b.len()
            )));
        }
        self.push(&row, out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const DEFAULTS: DimerArgs = DimerArgs {
        salt: Args {
            na: 50.0,
            k: 0.0,
            tris: 0.0,
            mg: 1.5,
            dntp: 0.6,
            temp_c: 37.0,
        },
        oligo_nm: 50.0,
    };

    fn at_conc(oligo_nm: f64) -> DimerPack {
        DimerPack::new(&DimerArgs {
            oligo_nm,
            ..DEFAULTS
        })
        .unwrap()
    }

    /// §5: Q₄ and Q₁ at the preset defaults (oracle-confirmed through every found record
    /// of the three default golden sets, shared §7 X3), σ as T2, both Θ_Ω = −∞.
    #[test]
    fn default_pack() {
        let p = DimerPack::new(&DEFAULTS).unwrap();
        assert_eq!(p.salt.sigma, -0.6656578142138302);
        assert_eq!(p.q4, -36.162145909210544);
        assert_eq!(p.q1, -33.407301754793096);
        assert_eq!(p.theta_omega4, f64::NEG_INFINITY);
        assert_eq!(p.theta_omega1, f64::NEG_INFINITY);
    }

    /// §5 "Where Q reaches fl(6.7)": the four boundary concentrations and Θ_Ω's switch.
    #[test]
    fn stack_threshold_concentrations() {
        let below = 6.699999999999999;
        let q4m = at_conc(116497808417.15082);
        let q4p = at_conc(116497808417.15083);
        assert_eq!((q4m.q4, q4p.q4), (below, 6.7));
        assert_eq!(q4m.theta_omega4, f64::NEG_INFINITY);
        assert_eq!(q4p.theta_omega4, f64::INFINITY);
        let q1m = at_conc(29124452104.287704);
        let q1p = at_conc(29124452104.28771);
        assert_eq!((q1m.q1, q1p.q1), (below, 6.7));
        assert_eq!(q1m.theta_omega1, f64::NEG_INFINITY);
        assert_eq!(q1p.theta_omega1, f64::INFINITY);
        // 2·10¹⁰ nM: 0 < Q₄ < Q₁ < fl(6.7).
        let s20 = at_conc(2e10);
        assert!(0.0 < s20.q4 && s20.q4 < s20.q1 && s20.q1 < 6.7);
        assert_eq!(s20.theta_omega1, f64::NEG_INFINITY);
        // §6.8 example 7: Q₄ at 10¹² nM.
        assert_eq!(at_conc(1e12).q4, 10.972247135975856);
        // fl(−1 + (−5.7)) = −fl(6.7) exactly (§3.2).
        assert_eq!(OMEGA.s + INIT_S, -6.7);
    }

    #[test]
    fn pack_errors_in_step_order() {
        let e = |a: DimerArgs| DimerPack::new(&a).unwrap_err().to_string();
        let no_salt = DimerArgs {
            salt: Args {
                na: 0.0,
                mg: 0.0,
                dntp: 0.0,
                ..DEFAULTS.salt
            },
            oligo_nm: 1e-320,
        };
        assert!(e(no_salt).contains("is zero"));
        assert!(e(DimerArgs {
            salt: Args {
                na: 1e308,
                k: 1e308,
                ..DEFAULTS.salt
            },
            ..DEFAULTS
        })
        .contains("infinite"));
        assert!(e(DimerArgs {
            oligo_nm: 1e-320,
            ..DEFAULTS
        })
        .contains("oligo_nM=1e-320"));
        for bad in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            assert!(e(DimerArgs {
                oligo_nm: bad,
                ..DEFAULTS
            })
            .contains("must be finite and > 0"));
        }
    }

    /// §6.8 example 3, END1 fallback values from published constants alone.
    #[test]
    fn end1_fallback_arithmetic() {
        let p = DimerPack::new(&DEFAULTS).unwrap();
        let bare = at(0, 3);
        let c = bare; // Lt(1, 1)
        let r = bare; // R(1, 1)
        let dh = (c.h + r.h) + INIT_H;
        let ds = ((c.s + r.s) + INIT_S) + 0.0 * p.salt.sigma;
        assert_eq!(dh, 4600.0);
        assert_eq!(ds, 8.100000000000001);
        assert_eq!(dh - p.salt.t_u * ds, 2087.785);
        assert_eq!(dh / (ds + p.q4) - K0, -437.0718901819689);
    }

    #[test]
    fn self_complementarity() {
        let coded = |s: &str| {
            let mut v = vec![4u8];
            v.extend(s.bytes().map(code));
            v.push(4);
            v
        };
        for (s, want) in [
            ("ACGT", true),
            ("GATC", true),
            ("TGCA", true),
            ("ACGTACGT", true),
            ("ACGTACGA", false),
            ("ACG", false),
            ("AA", false),
        ] {
            assert_eq!(self_complementary(&coded(s), s.len()), want, "{s}");
        }
    }

    /// primer3-cuda.md §4 and QPC-12: the K2 device pack is a bit copy of the kernel's
    /// tables, end terms and pack; `details` rejects a key the detail pass does not give.
    #[test]
    fn device_pack_and_detail_key_check() {
        use seqblaze_core::{batch::SeqBatch, kernel::PairDistanceKernel, seq::SeqTable};
        let tables = Arc::new(crate::hairpin::synthetic_tables(0xd1ce));
        let pack = DimerPack::new(&DEFAULTS).unwrap();
        let kernel = DimerPrimer3::new(pack, Arc::clone(&tables), End::Any, &["found"]).unwrap();
        let device = kernel.device_pack().unwrap();
        assert_eq!(device.tables, tables.device_words().unwrap());
        let mut ends = entry_words(&kernel.ends[0]).unwrap();
        ends.extend(entry_words(&kernel.ends[1]).unwrap());
        assert_eq!(device.ends, ends);
        let c = device.scalars;
        assert_eq!((c.q4, c.q1), (pack.q4.to_bits(), pack.q1.to_bits()));
        assert_eq!(
            (c.theta_omega4, c.theta_omega1),
            (pack.theta_omega4.to_bits(), pack.theta_omega1.to_bits())
        );
        assert_eq!(
            (c.init_h, c.init_s, c.shift),
            (INIT_H.to_bits(), INIT_S.to_bits(), SHIFT.to_bits())
        );
        let table = SeqTable::new(SeqBatch::from_seqs(["ACGTACGTTT", "AAACGTACGT"]));
        let (a, b) = (table.get(0), table.get(1));
        let key = key_of(kernel.row(a, b).unwrap().dg_kcal);
        let mut out = DetailColumns::new(&FIELDS, kernel.request);
        kernel.details(a, b, key, &mut out).unwrap();
        let wrong = kernel
            .details(a, b, key ^ 1, &mut out)
            .unwrap_err()
            .to_string();
        assert!(wrong.contains("no result was published"), "{wrong}");
    }
}
