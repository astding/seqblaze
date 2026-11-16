//! T2 `hairpin` `compat="primer3"`: primer3-py 2.3.1 module-level `calc_hairpin`
//! (`max_loop` 30) replayed in binary64 from the run-time parameter files, with a
//! correctly rounded natural logarithm (named reference `primer3-py@2.3.1/cr-log`).
//! Spec: docs/wiki/thermo/hairpin-primer3.md (§4 per-call constants, §5 per-sequence
//! computation, §6 fields), docs/wiki/thermo/primer3-alignment.md (shared rules).
//! Derivation route: a behaviour specification written from a reading of primer3-py
//! 2.3.1's source (no code copied; D94), confirmed against the unmodified tool; this
//! module was written from the wiki pages and those specifications only.
//!
//! Every formula is a sequence of separate binary64 operations in the grouping the page
//! states; rustc never contracts to FMA and nothing here calls `mul_add`. Additions of
//! an explicit `+0.0` are performed (they turn −0 into +0). Comparisons follow IEEE
//! semantics: every comparison with NaN is false, and each is written in the direction
//! the page states. The kernel implements the primary readings of §10.6 only
//! (V1a: internal-loop closures use M1; V1b: the end term and the exterior both-sides
//! option use M2; V2–V6 as primary).
use std::sync::Arc;

use seqblaze_core::{
    detail::{try_reserve_exact, DetailColumns, DetailField, DetailRequest, FieldKind},
    fpenv,
    seq::{ByteView, SeqTable},
    unary::UnaryKernel,
    validate::EngineError,
};

use crate::{
    cr::{self, Rational},
    p3files::{
        at, entry_words, idx3, idx4, is_wc, Hs, P3DevicePack, P3DeviceScalars, P3Tables, AT_H,
        AT_S, LOOP_ROWS, OMEGA, PHI,
    },
    params::PRIMER3_SANTALUCIA,
};

/// Fields of the preset (§6), in output order. `found` is carried as int32 0/1 (mask
/// INT32_MAX) and exposed as bool by the binding.
pub static FIELDS: [DetailField; 5] = [
    DetailField {
        name: "found",
        kind: FieldKind::Int32,
    },
    DetailField {
        name: "dg_kcal",
        kind: FieldKind::Float64,
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

/// The engine's length limit of the thal compat presets (§2.2, API §1.1).
pub const MAX_LENGTH: usize = 60;
/// T₃₇ = fl(310.15), the temperature of every ΔG comparison of the fill (shared §3.3).
pub const T37: f64 = 310.15;
/// Celsius to kelvin (SI).
pub const K0: f64 = 273.15;
/// P2 of the shared mapping: Tris⁺ = Tris/2.
const TRIS_FACTOR: f64 = 0.5;
/// mM → mol/L, by division.
const MILLI_DIV: f64 = 1000.0;
/// cal → kcal of the exposed fields, by division (API §3).
pub(crate) const KCAL_DIV: f64 = 1000.0;
/// α₁ = fl(−300 / T₃₇), the asymmetry entropy per nucleotide (Peyret et al. 1999;
/// SantaLucia & Hicks 2004), as its bits `-0x1.ef3e864b31d04p-1` (shared §3.3).
pub const ALPHA1: f64 = f64::from_bits(0xBFEE_F3E8_64B3_1D04);
/// F = (0, −3224), the fresh or clamped cell (class C).
pub const FRESH: Hs = Hs { h: 0.0, s: -3224.0 };
/// Entropy cutoff of the clamp and of exterior candidates (class C).
pub const S_CUTOFF: f64 = -2500.0;
/// Backtrace tolerance (class C, shared §2.6).
pub const TOLERANCE: f64 = 1e-5;
/// 1×1 ratio margin (class C, §5.3).
pub const MARGIN_1X1: f64 = 1e-6;
/// `max_loop`, fixed at 30 (QH2).
pub const MAX_LOOP: usize = 30;
/// The shortest hairpin loop: an admissible pair encloses at least 3 nucleotides.
const MIN_SPAN: usize = 4;

/// The `calc_hairpin` salt and temperature inputs after the §3 mapping (mM, °C).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Args {
    pub na: f64,
    pub k: f64,
    pub tris: f64,
    pub mg: f64,
    pub dntp: f64,
    pub temp_c: f64,
}

/// The per-call constants H1–H6 of §4, built once on the host.
#[derive(Debug, Clone, Copy)]
pub struct HairpinPack {
    /// H1: monovalent cations, mM.
    pub mv: f64,
    /// H2: 120·√(Mg − dNTP) or +0, mM.
    pub w: f64,
    /// H3: monovalent equivalent μ, mM.
    pub mu: f64,
    /// H4: μ in mol/L.
    pub x_mu: f64,
    /// H5: CR(ln x_μ) and σ = fl(0.368 · λ).
    pub ln_mu: f64,
    pub sigma: f64,
    /// H6: T_u = fl(temp_c + 273.15).
    pub t_u: f64,
}

/// The message prefix and reference call of T2's pack errors.
const HAIRPIN_LABEL: &str = "hairpin compat='primer3'";
const HAIRPIN_CALL: &str = "calc_hairpin";

pub(crate) fn parameter(message: String) -> EngineError {
    EngineError::Parameter(message)
}

/// `x` in the layout of Python's `repr(float)`.
pub(crate) fn py_float(x: f64) -> String {
    if x.is_nan() {
        return "nan".to_owned();
    }
    let debug = format!("{x:?}");
    match debug.split_once('e') {
        None => debug,
        Some((mantissa, exponent)) => {
            let (sign, digits) = match exponent.strip_prefix('-') {
                Some(digits) => ('-', digits),
                None => ('+', exponent),
            };
            format!("{mantissa}e{sign}{digits:0>2}")
        }
    }
}

impl HairpinPack {
    /// H1–H6 of §4; raises `ParameterError` in step order (H3: μ zero or infinite,
    /// H4: x_μ underflowing to zero). Runs in the IEEE default floating-point
    /// environment whatever the caller's state, which is restored on return (D99).
    pub fn new(args: &Args) -> Result<Self, EngineError> {
        fpenv::guarded(|| Self::build(args, HAIRPIN_LABEL, HAIRPIN_CALL))
    }

    /// H1–H6 with the error messages of the preset `label` whose reference calls are
    /// `call` (shared by T2 and K2, dimer-primer3.md §5: "H1–H6 are T2's §4 steps
    /// unchanged"). Expects the IEEE default floating-point environment.
    pub(crate) fn build(args: &Args, label: &str, call: &str) -> Result<Self, EngineError> {
        for (name, value) in [
            ("na_mM", args.na),
            ("k_mM", args.k),
            ("tris_mM", args.tris),
            ("mg_mM", args.mg),
            ("dntp_mM", args.dntp),
        ] {
            // sb.Conditions admits only these; a direct binding call is checked too.
            if !value.is_finite() || value < 0.0 {
                return Err(parameter(format!(
                    "{label}: {name}={} must be finite and >= 0",
                    py_float(value)
                )));
            }
        }
        // sb.Conditions compares with the exact rational -5463/20, so it admits the
        // binary64 literal -273.15 (just above it; T_u = +0 there) and the same
        // predicate is `temp_c >= -fl(273.15)` here (tm.md, as built).
        if !args.temp_c.is_finite() || args.temp_c < -K0 {
            return Err(parameter(format!(
                "{label}: temp_c={} must be finite and not below -273.15",
                py_float(args.temp_c)
            )));
        }
        let conditions = || {
            format!(
                "na_mM={}, k_mM={}, tris_mM={}, mg_mM={}, dntp_mM={}",
                py_float(args.na),
                py_float(args.k),
                py_float(args.tris),
                py_float(args.mg),
                py_float(args.dntp)
            )
        };
        let params = &PRIMER3_SANTALUCIA;
        let m1 = args.na + args.k; // H1
        let m2 = args.tris * TRIS_FACTOR;
        let mv = m1 + m2;
        let w = if args.mg > args.dntp {
            // H2: free Mg2+, von Ahsen 2001 Eq. 7 (strict: at mg = dntp the term is +0).
            let free = args.mg - args.dntp;
            params.c_mg * free.sqrt()
        } else {
            0.0
        };
        let mu = mv + w; // H3 (QH5)
        if mu == 0.0 || mu == f64::INFINITY {
            let what = if mu == 0.0 {
                "zero: no monovalent cation and no free Mg2+ (mg_mM > dntp_mM)"
            } else {
                "infinite (the monovalent sum overflows binary64)"
            };
            return Err(parameter(format!(
                "{label}: the monovalent equivalent of {} is {what}; \
                 primer3's {call} would return a non-finite salt entropy \
                 (sigma = -inf) or no number at all. Give a positive, finite salt, e.g. \
                 sb.Conditions.preset('primer3')",
                conditions()
            )));
        }
        let x_mu = mu / MILLI_DIV; // H4
        if x_mu == 0.0 {
            return Err(parameter(format!(
                "{label}: the monovalent equivalent {} mM of {} underflows \
                 to 0 mol/L (mu/1000); primer3's {call} would take log(0) and return a \
                 non-finite salt entropy (sigma = -inf) or no number at all. Use a larger salt \
                 concentration",
                py_float(mu),
                conditions()
            )));
        }
        // H5: CR64(ln x_mu) (the calc_tm preset's Ziv loop, coefficient 1), then sigma.
        let ln_mu =
            cr::ln(&Rational::from_f64(x_mu), &Rational::from_ratio(1, 1)).map_err(|cap| {
                parameter(format!(
                    "internal error: the correctly rounded ln_mu needs more than {} bits; \
                     nothing was computed",
                    cap.bits
                ))
            })?;
        let sigma = params.salt * ln_mu;
        let t_u = args.temp_c + K0; // H6
        Ok(Self {
            mv,
            w,
            mu,
            x_mu,
            ln_mu,
            sigma,
            t_u,
        })
    }
}

/// One sequence's reported values (§5.7, §6).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HairpinRow {
    pub found: bool,
    pub dg_kcal: f64,
    pub dh_kcal: f64,
    pub ds_cal_per_k: f64,
    pub tm_c: f64,
}

/// The five helix options of the exterior selection (§5.5), in table order after carry.
#[derive(Debug, Clone, Copy, PartialEq)]
enum Opt {
    Blunt,
    Dangle5,
    Dangle3,
    Both,
}

const OPTIONS: [Opt; 4] = [Opt::Blunt, Opt::Dangle5, Opt::Dangle3, Opt::Both];

/// Θ(h, s) = fl(fl(h + 0) / fl(fl(s − φ) + 0)) (§5.1; every addition performed).
#[inline]
fn theta(v: Hs) -> f64 {
    let num = v.h + 0.0;
    let den = (v.s - PHI) + 0.0;
    num / den
}

/// G(h, s) = fl(h − fl(T₃₇ · s)).
#[inline]
fn g37(v: Hs) -> f64 {
    v.h - T37 * v.s
}

/// cl(v) of §5.1 (also K2's clamp, dimer-primer3.md §6.1).
#[inline]
pub(crate) fn clamp(v: Hs) -> Hs {
    if v.s < S_CUTOFF {
        FRESH
    } else {
        v
    }
}

/// The per-component sum of two values, fl per component.
#[inline]
pub(crate) fn add(a: Hs, b: Hs) -> Hs {
    Hs {
        h: a.h + b.h,
        s: a.s + b.s,
    }
}

/// x ≈ y of §5.6: every component finite, and each differs by less than 10⁻⁵
/// (also K2's backtrace equality, dimer-primer3.md §6.1).
#[inline]
pub(crate) fn approx(x: Hs, y: Hs) -> bool {
    x.h.is_finite()
        && y.h.is_finite()
        && x.s.is_finite()
        && y.s.is_finite()
        && (x.h - y.h).abs() < TOLERANCE
        && (x.s - y.s).abs() < TOLERANCE
}

/// The end term E of §5.4 (primary readings V1b, V2, V3) of the closing pair p·q whose
/// inner neighbours are p′ (after p) and q′ (after q), codes 0–4. It depends on the four
/// codes and the call's tables only, so the device reads it from a host table of 625
/// slots built by this same function (primer3-cuda.md §4, QPC-5). Expects E0.
fn end_term(t: &P3Tables, p: u8, pp: u8, q: u8, qq: u8) -> Hs {
    let bare = at(p, q);
    if is_wc(pp, qq) {
        // The neighbours pair: no dangle option, E = Bare even if Mis is kept.
        return bare;
    }
    let d3 = t.d3[idx3(p, q, pp)];
    let d5 = t.d5[idx3(p, q, qq)];
    let dan = match (d3.h.is_finite(), d5.h.is_finite()) {
        (true, true) => add(add(bare, d3), d5),
        (true, false) => add(bare, d3),
        (false, true) => add(bare, d5),
        // Unreachable under content fact 3 (a Watson–Crick pair and a base give a
        // finite 3′ dangle); without a dangle option E = Bare.
        (false, false) => return bare,
    };
    let mis = add(bare, t.m2[idx4(p, pp, q, qq)]);
    let g_mis = g37(mis);
    let g_dan = g37(dan);
    let mis_kept = mis.h.is_finite() && g_mis <= 0.0;
    let dan_kept = dan.h.is_finite() && g_dan <= 0.0;
    let survivor = if mis_kept && g_mis < 0.0 {
        if dan_kept && g_dan < 0.0 && theta(dan) > theta(mis) {
            Some(dan)
        } else {
            Some(mis)
        }
    } else if dan_kept && g_dan < 0.0 {
        Some(dan)
    } else {
        None
    };
    // Primary reading (V2): Bare iff no survivor or Θ(survivor) < Θ(Bare); a NaN
    // Θ(Bare) (G·C and C·G closures) therefore keeps the survivor.
    match survivor {
        Some(surv) => {
            if theta(surv) < theta(bare) {
                bare
            } else {
                surv
            }
        }
        None => bare,
    }
}

const SIDE: usize = MAX_LENGTH + 2;
/// Cells of the workspace: (L + 1)² ≤ 61² (stride L + 1, indices 1..=L used).
const CELLS: usize = (MAX_LENGTH + 1) * (MAX_LENGTH + 1);

/// The cell table of one scheduler unit (§12): reserved fallibly once on the heap and
/// reused across the unit's sequences. Each fold resets the (L + 1)² cells it uses to
/// Ω first, so a row never depends on the sequences folded before it.
pub struct Workspace {
    cell: Vec<Hs>,
}

impl Workspace {
    /// A refused reservation is a DeviceError ("unable to reserve hairpin cell table").
    pub fn new() -> Result<Self, EngineError> {
        let mut cell = Vec::new();
        try_reserve_exact(&mut cell, CELLS, "hairpin cell table")?;
        cell.resize(CELLS, OMEGA);
        Ok(Self { cell })
    }
}

/// The fill and backtrace state of one sequence (§5). Codes are one-based:
/// `b[1..=L]` the bases, `b[L + 1]` code 4 (the virtual position after the 3′ end).
struct Fold<'t> {
    t: &'t P3Tables,
    len: usize,
    /// Row stride of `cell`: L + 1.
    stride: usize,
    b: [u8; SIDE + 1],
    /// C(y, z) at `y * stride + z`, the workspace's first (L + 1)² cells.
    cell: &'t mut [Hs],
    prefix: [Hs; SIDE],
    /// T_u of the call: the acceptance temperature of the exterior selection (§5.5).
    t_u: f64,
}

impl<'t> Fold<'t> {
    fn new(t: &'t P3Tables, workspace: &'t mut Workspace, seq: &[u8], t_u: f64) -> Self {
        let mut b = [4u8; SIDE + 1];
        for (slot, &byte) in b[1..].iter_mut().zip(seq) {
            *slot = match byte {
                b'A' => 0,
                b'C' => 1,
                b'G' => 2,
                b'T' => 3,
                _ => 4,
            };
        }
        let len = seq.len();
        let stride = len + 1;
        let cell = &mut workspace.cell[..stride * stride];
        cell.fill(OMEGA);
        Self {
            t,
            len,
            stride,
            b,
            cell,
            prefix: [OMEGA; SIDE],
            t_u,
        }
    }

    #[inline]
    fn c(&self, y: usize, z: usize) -> Hs {
        self.cell[y * self.stride + z]
    }

    #[inline]
    fn admissible(&self, y: usize, z: usize) -> bool {
        z >= y + MIN_SPAN && is_wc(self.b[y], self.b[z])
    }

    /// The stack key (β(y), β(y+1), β(z), β(z−1)).
    #[inline]
    fn stack_key(&self, y: usize, z: usize) -> usize {
        idx4(self.b[y], self.b[y + 1], self.b[z], self.b[z - 1])
    }

    /// The inner-side key (β(z′), β(z′+1), β(y′), β(y′−1)).
    #[inline]
    fn inner_key(&self, yp: usize, zp: usize) -> usize {
        idx4(self.b[zp], self.b[zp + 1], self.b[yp], self.b[yp - 1])
    }

    /// The loop term ℓ of §5.2 (b) for the closing pair (y, z) and inner pair (y′, z′).
    fn loop_term(&self, y: usize, z: usize, yp: usize, zp: usize) -> Hs {
        let t = self.t;
        let l5 = yp - y - 1;
        let l3 = z - zp - 1;
        let n = l5 + l3;
        if l5 == 0 || l3 == 0 {
            if n == 1 {
                // Bulge of one: the stack across the bulge.
                let nn = t.nn[idx4(self.b[y], self.b[yp], self.b[z], self.b[zp])];
                add(t.loop_bu[1], nn)
            } else {
                let outer = at(self.b[y], self.b[z]);
                let inner = at(self.b[yp], self.b[zp]);
                add(add(t.loop_bu[n], outer), inner)
            }
        } else if l5 == 1 && l3 == 1 {
            add(t.mm[self.stack_key(y, z)], t.mm[self.inner_key(yp, zp)])
        } else {
            // V1a primary: internal-loop closures use M1.
            let outer = t.m1[self.stack_key(y, z)];
            let inner = t.m1[self.inner_key(yp, zp)];
            let asym = (l5 as f64 - l3 as f64).abs();
            let alpha = Hs {
                h: 0.0,
                s: ALPHA1 * asym,
            };
            add(add(add(t.loop_in[n], outer), inner), alpha)
        }
    }

    /// The candidate X = ℓ + C(y′, z′) of §5.2 (b); Ω if its ΔH is not finite.
    #[inline]
    fn loop_candidate(&self, y: usize, z: usize, yp: usize, zp: usize) -> Hs {
        let x = add(self.loop_term(y, z, yp, zp), self.c(yp, zp));
        if x.h.is_finite() {
            x
        } else {
            OMEGA
        }
    }

    /// The loop candidates of (y, z) in the §5.3 scan order: n = λ₅ + λ₃ ascending, then
    /// λ₅ ascending; only admissible inner pairs. Calls `visit(y′, z′, is_1x1)`.
    #[inline]
    fn scan_loops(&self, y: usize, z: usize, mut visit: impl FnMut(usize, usize, bool)) {
        // An admissible inner pair needs z′ − y′ ≥ 4, so n ≤ z − y − 6.
        let span = z - y;
        if span < MIN_SPAN + 3 {
            return;
        }
        let n_max = MAX_LOOP.min(span - MIN_SPAN - 2);
        for n in 1..=n_max {
            for l5 in 0..=n {
                let l3 = n - l5;
                let (yp, zp) = (y + 1 + l5, z - 1 - l3);
                if is_wc(self.b[yp], self.b[zp]) {
                    visit(yp, zp, l5 == 1 && l3 == 1);
                }
            }
        }
    }

    /// The hairpin-loop candidate X_hp of §5.2 (c) closed by (y, z), validity rule
    /// included against `v` (the value reached after stage 2).
    fn hairpin_loop(&self, y: usize, z: usize, v: Hs) -> Hs {
        let t = self.t;
        let n = z - y - 1;
        let x = match n {
            3 => {
                let mut x = add(t.loop_hp[3], at(self.b[y], self.b[z]));
                let bonus = crate::p3files::bonus_index(&self.b[y..y + 5]).map(|i| t.b3[i]);
                if let Some(bonus) = bonus {
                    if let Some(h) = bonus.h {
                        x.h += h;
                    }
                    if let Some(s) = bonus.s {
                        x.s += s;
                    }
                }
                x
            }
            4 => {
                let mut x = add(t.loop_hp[4], t.m2[self.stack_key(y, z)]);
                let bonus = crate::p3files::bonus_index(&self.b[y..y + 6]).map(|i| t.b4[i]);
                if let Some(bonus) = bonus {
                    if let Some(h) = bonus.h {
                        x.h += h;
                    }
                    if let Some(s) = bonus.s {
                        x.s += s;
                    }
                }
                x
            }
            _ => add(t.loop_hp[n.min(LOOP_ROWS)], t.m2[self.stack_key(y, z)]),
        };
        let positive = x.h > 0.0 && x.s > 0.0;
        if !x.h.is_finite() || (positive && (v.h <= 0.0 || v.s <= 0.0)) {
            OMEGA
        } else {
            x
        }
    }

    /// The end term E(y, z) of §5.4 (primary readings V1b, V2, V3).
    #[inline]
    fn end_term(&self, y: usize, z: usize) -> Hs {
        end_term(self.t, self.b[y], self.b[y + 1], self.b[z], self.b[z + 1])
    }

    /// The cell of the admissible pair (y, z): stages 1–3 of §5.3.
    fn fill_cell(&self, y: usize, z: usize) -> Hs {
        let t = self.t;
        // Stage 1: stack on (y + 1, z − 1).
        let mut v = FRESH;
        let stack = add(self.c(y + 1, z - 1), t.nn[self.stack_key(y, z)]);
        if theta(stack) > theta(v) {
            v = clamp(stack);
        }
        // Stage 2: internal loops and bulges, strict, the 1×1 margin.
        let mut tv = theta(v);
        self.scan_loops(y, z, |yp, zp, one_by_one| {
            let x = self.loop_candidate(y, z, yp, zp);
            if !x.h.is_finite() {
                return;
            }
            let tx = theta(x);
            if tx > tv && (!one_by_one || tx - tv >= MARGIN_1X1) {
                v = clamp(x);
                tv = theta(v);
            }
        });
        // Stage 3: the hairpin loop against v on G_E, ties to the hairpin loop.
        let hp = self.hairpin_loop(y, z, v);
        let e = self.end_term(y, z);
        let g_e = |x: Hs| (x.h + e.h) - T37 * (x.s + e.s);
        let chosen = if g_e(v) < g_e(hp) { v } else { hp };
        clamp(chosen)
    }

    fn fill(&mut self) {
        let len = self.len;
        for z in 1..=len {
            for y in (1..z).rev() {
                if self.admissible(y, z) {
                    let value = self.fill_cell(y, z);
                    self.cell[y * self.stride + z] = value;
                }
            }
        }
        for rho in 2..=len {
            let carry = self.prefix[rho - 1];
            let mut winner = None;
            let mut best = theta(carry);
            for opt in OPTIONS {
                if let Some(candidate) = self.option_best(opt, rho) {
                    let t = theta(candidate);
                    if t >= best {
                        best = t;
                        winner = Some(candidate);
                    }
                }
            }
            self.prefix[rho] = match winner {
                Some(x) if x.h - self.t_u * x.s < 0.0 => x,
                _ => carry,
            };
        }
    }

    /// Largest κ and the closing pair (u, w) of an option at ρ (§5.5).
    #[inline]
    fn kappa_max(opt: Opt, rho: usize) -> Option<usize> {
        let offset = match opt {
            Opt::Blunt => 5,
            Opt::Dangle5 | Opt::Dangle3 => 6,
            Opt::Both => 7,
        };
        rho.checked_sub(offset)
    }

    #[inline]
    fn closing(opt: Opt, kappa: usize, rho: usize) -> (usize, usize) {
        match opt {
            Opt::Blunt => (kappa + 1, rho),
            Opt::Dangle5 => (kappa + 2, rho),
            Opt::Dangle3 => (kappa + 1, rho - 1),
            Opt::Both => (kappa + 2, rho - 1),
        }
    }

    /// The helix candidate of an option with prefix value `pre` (§5.5): blunt
    /// fl(fl(P + A) + C), the others fl(fl(fl(P + A) + D) + C), per component.
    #[inline]
    fn helix_sum(&self, opt: Opt, kappa: usize, rho: usize, pre: Hs) -> Hs {
        let t = self.t;
        let b = &self.b;
        let (u, w) = Self::closing(opt, kappa, rho);
        let a = at(b[u], b[w]);
        let c = self.c(u, w);
        let with_a = add(pre, a);
        let d = match opt {
            Opt::Blunt => return add(with_a, c),
            Opt::Dangle5 => t.d5[idx3(b[rho], b[kappa + 2], b[kappa + 1])],
            Opt::Dangle3 => t.d3[idx3(b[rho - 1], b[kappa + 1], b[rho])],
            // V1b primary: the exterior both-sides option uses M2.
            Opt::Both => t.m2[idx4(b[rho - 1], b[rho], b[kappa + 2], b[kappa + 1])],
        };
        add(add(with_a, d), c)
    }

    /// An option's best valid candidate at ρ: κ ascending, strictly larger Θ (§5.5).
    fn option_best(&self, opt: Opt, rho: usize) -> Option<Hs> {
        let kmax = Self::kappa_max(opt, rho)?;
        let mut best: Option<Hs> = None;
        let mut best_theta = f64::NEG_INFINITY;
        for kappa in 0..=kmax {
            let p = self.prefix[kappa];
            let pre = if theta(p) >= -0.0 {
                p
            } else {
                Hs { h: 0.0, s: 0.0 }
            };
            let x = self.helix_sum(opt, kappa, rho, pre);
            let valid = x.h.is_finite() && x.h <= 0.0 && x.s <= 0.0 && x.s > S_CUTOFF;
            if valid {
                let tx = theta(x);
                if tx > best_theta {
                    best_theta = tx;
                    best = Some(x);
                }
            }
        }
        best
    }

    /// The pairs of the backtrace (§5.6), one-based (y, z), in trace order through
    /// `pair` (no allocation).
    fn backtrace(&self, pair: &mut impl FnMut(usize, usize)) {
        let mut rho = self.len;
        loop {
            while rho > 0 && approx(self.prefix[rho], self.prefix[rho - 1]) {
                rho -= 1;
            }
            if rho == 0 {
                break;
            }
            let target = self.prefix[rho];
            // V4a primary: the option whose recomputed best equals P(ρ), in the order
            // blunt, 5′, 3′, both sides.
            let Some(opt) = OPTIONS.into_iter().find(|&opt| {
                self.option_best(opt, rho)
                    .is_some_and(|best| approx(target, best))
            }) else {
                break;
            };
            let kmax = Self::kappa_max(opt, rho).expect("an identified option has a range");
            let mut next = None;
            let mut matched = false;
            for kappa in 0..=kmax {
                let without = self.helix_sum(opt, kappa, rho, Hs { h: 0.0, s: 0.0 });
                let found = if approx(without, target) {
                    Some(None)
                } else if approx(self.helix_sum(opt, kappa, rho, self.prefix[kappa]), target) {
                    Some(Some(kappa))
                } else {
                    None
                };
                if let Some(continue_at) = found {
                    let (u, w) = Self::closing(opt, kappa, rho);
                    self.trace_pair(u, w, pair);
                    next = continue_at;
                    matched = true;
                    break;
                }
            }
            match (matched, next) {
                (true, Some(kappa)) => rho = kappa,
                _ => break,
            }
        }
    }

    /// Record (y, z) and follow its decomposition (§5.6 "inside a pair").
    fn trace_pair(&self, mut y: usize, mut z: usize, pair: &mut impl FnMut(usize, usize)) {
        let t = self.t;
        loop {
            pair(y, z);
            let c = self.c(y, z);
            // Row 1: the stack.
            let stack = add(self.c(y + 1, z - 1), t.nn[self.stack_key(y, z)]);
            if approx(c, stack) {
                y += 1;
                z -= 1;
                continue;
            }
            // Row 2: the hairpin loop, with the §5.2 (c) validity rule (V5) taken against
            // the stored cell C rather than the stage-2 value v; the verdicts agree. The
            // rule bites only when X_hp is positive in both components. If stage 3 kept
            // v, C is v or F, and C = F only when v's dS < -2500, so C and v both have a
            // component <= 0 or neither has. If stage 3 took X_hp, it was valid against
            // v; a both-positive X_hp then implies a both-positive v and is stored
            // unclamped, so C = X_hp is both-positive as well.
            if approx(c, self.hairpin_loop(y, z, c)) {
                return;
            }
            // Row 3 (V4b primary): the last candidate with Θ ≥ Θ(cell) must match.
            let tc = theta(c);
            let mut last: Option<Hs> = None;
            self.scan_loops(y, z, |yp, zp, _| {
                let x = self.loop_candidate(y, z, yp, zp);
                if x.h.is_finite() && theta(x) >= tc {
                    last = Some(x);
                }
            });
            if !last.is_some_and(|x| approx(c, x)) {
                return;
            }
            let mut first = None;
            self.scan_loops(y, z, |yp, zp, _| {
                if first.is_none() && approx(c, add(self.loop_term(y, z, yp, zp), self.c(yp, zp))) {
                    first = Some((yp, zp));
                }
            });
            match first {
                Some((yp, zp)) => {
                    y = yp;
                    z = zp;
                }
                None => return,
            }
        }
    }
}

/// The T2 primer3 kernel of one call: the pack, the parameter set and the requested
/// fields (§6).
pub struct HairpinPrimer3 {
    pack: HairpinPack,
    tables: Arc<P3Tables>,
    request: DetailRequest,
}

impl HairpinPrimer3 {
    /// `fields` by name, any subset of the five (§6); every row is computed in full.
    pub fn new(
        pack: HairpinPack,
        tables: Arc<P3Tables>,
        fields: &[&str],
    ) -> Result<Self, EngineError> {
        let mut request = DetailRequest::NONE;
        for name in fields {
            let Some(index) = FIELDS.iter().position(|f| f.name == *name) else {
                return Err(parameter(format!(
                    "field {name:?} is not available for hairpin compat='primer3'; use any of \
                     ['found', 'dg_kcal', 'dh_kcal', 'ds_cal_per_k', 'tm_c']"
                )));
            };
            request = request.with(index);
        }
        Ok(Self {
            pack,
            tables,
            request,
        })
    }

    /// The fill, exterior selection and backtrace of one upper-case ACGT sequence of
    /// 1–60 nt. Expects the IEEE default floating-point environment: it runs only
    /// inside guarded units (§12.1) and carries no guard of its own.
    fn fold<'t>(&'t self, workspace: &'t mut Workspace, seq: &[u8]) -> Fold<'t> {
        let mut fold = Fold::new(&self.tables, workspace, seq, self.pack.t_u);
        fold.fill();
        fold
    }

    /// One sequence's row (§5.7), folded in `workspace`. Expects E0 (see
    /// [`Self::fold`]). Allocates nothing.
    pub fn row(&self, workspace: &mut Workspace, seq: &[u8]) -> HairpinRow {
        let fold = self.fold(workspace, seq);
        let len = fold.len;
        // P(0) = P(1) = Ω, so L = 1 is never found.
        let p = fold.prefix[len];
        if !p.h.is_finite() {
            return HairpinRow {
                found: false,
                dg_kcal: 0.0,
                dh_kcal: 0.0,
                ds_cal_per_k: 0.0,
                tm_c: 0.0,
            };
        }
        let mut paired = [false; SIDE];
        fold.backtrace(&mut |y, z| {
            paired[y] = true;
            paired[z] = true;
        });
        let n_pair = paired[1..len].iter().filter(|&&x| x).count() as i64;
        let m = n_pair / 2 - 1;
        let salt = m as f64 * self.pack.sigma;
        let ds = p.s + salt;
        let dh = p.h;
        let dg = dh - self.pack.t_u * ds;
        let tm = melting_temperature(dh, ds);
        HairpinRow {
            found: true,
            dg_kcal: dg / KCAL_DIV,
            dh_kcal: dh / KCAL_DIV,
            ds_cal_per_k: ds,
            tm_c: tm,
        }
    }

    /// The device pack of primer3-cuda.md §4: the handle's tables, the 625-slot end-term
    /// table (QPC-5) and the scalars, as bit patterns. The end-term table is built by the
    /// CPU's own end-term rule, in the IEEE default floating-point environment whatever
    /// the caller's state (D99, a guard site of FP environment §7.3 item 2); everything
    /// else is a bit copy. A refused reservation is a DeviceError.
    pub fn device_pack(&self) -> Result<P3DevicePack, EngineError> {
        let tables = self.tables.device_words()?;
        let mut entries = Vec::new();
        try_reserve_exact(&mut entries, 625, "hairpin end terms")?;
        entries.resize(625, OMEGA);
        fpenv::guarded(|| {
            for p in 0..4u8 {
                for pp in 0..5u8 {
                    for q in 0..4u8 {
                        for qq in 0..5u8 {
                            entries[idx4(p, pp, q, qq)] = end_term(&self.tables, p, pp, q, qq);
                        }
                    }
                }
            }
        });
        Ok(P3DevicePack {
            tables,
            ends: entry_words(&entries)?,
            scalars: P3DeviceScalars {
                sigma: self.pack.sigma.to_bits(),
                t_u: self.pack.t_u.to_bits(),
                ..device_constants()
            },
        })
    }

    /// The backtrace pairs of one sequence, zero-based (the testing probe, §12).
    /// Expects E0 (see [`Self::fold`]).
    pub fn pairs(&self, seq: &[u8]) -> Result<Vec<(usize, usize)>, EngineError> {
        let mut workspace = Workspace::new()?;
        let fold = self.fold(&mut workspace, seq);
        let len = fold.len;
        let mut pairs = Vec::new();
        if fold.prefix[len].h.is_finite() {
            // At most L / 2 ≤ 30 pairs: the pushes below never reallocate.
            try_reserve_exact(&mut pairs, MAX_LENGTH / 2, "hairpin probe pairs")?;
            fold.backtrace(&mut |y, z| pairs.push((y - 1, z - 1)));
        }
        Ok(pairs)
    }
}

/// The bit patterns of the class-C constants and published constants the T2 and K2
/// recurrences read (primer3-cuda.md §4, QPC-6); the per-call fields are +0.
pub(crate) fn device_constants() -> P3DeviceScalars {
    P3DeviceScalars {
        t37: T37.to_bits(),
        phi: PHI.to_bits(),
        fresh_h: FRESH.h.to_bits(),
        fresh_s: FRESH.s.to_bits(),
        s_cutoff: S_CUTOFF.to_bits(),
        tolerance: TOLERANCE.to_bits(),
        margin: MARGIN_1X1.to_bits(),
        alpha1: ALPHA1.to_bits(),
        at_h: AT_H.to_bits(),
        at_s: AT_S.to_bits(),
        omega_h: OMEGA.h.to_bits(),
        omega_s: OMEGA.s.to_bits(),
        k0: K0.to_bits(),
        kcal_div: KCAL_DIV.to_bits(),
        init_h: crate::dimer::INIT_H.to_bits(),
        init_s: crate::dimer::INIT_S.to_bits(),
        shift: crate::dimer::SHIFT.to_bits(),
        ..P3DeviceScalars::default()
    }
}

/// The `SequenceError` for a length outside 1..=60 (§7 step 2, QH5).
fn length_error(index: usize, length: usize, limit: usize) -> EngineError {
    let message = if length < limit {
        format!(
            "hairpin input a sequence {index} is empty (length {length}, minimum {limit} nt): \
             primer3's calc_hairpin raises for an empty sequence; nothing was computed"
        )
    } else {
        format!(
            "hairpin input a sequence {index} has length {length}, above the limit {limit} nt \
             of compat='primer3' (primer3's thermodynamic alignment takes at most 60 nt for a \
             hairpin); nothing was computed"
        )
    };
    EngineError::Sequence {
        message,
        input: "a",
        index,
        position: None,
        byte: None,
        n_bad: 1,
        bad_indices: vec![index],
    }
}

impl UnaryKernel for HairpinPrimer3 {
    type Row = HairpinRow;
    type Cutoff = f64;
    type Scratch = Workspace;

    fn scratch(&self) -> Result<Workspace, EngineError> {
        Workspace::new()
    }

    fn name(&self) -> &'static str {
        "hairpin"
    }

    fn length_limits(&self) -> (usize, usize) {
        (1, MAX_LENGTH)
    }

    fn length_error(&self, index: usize, length: usize, limit: usize) -> EngineError {
        length_error(index, length, limit)
    }

    fn item_cost(&self, table: &SeqTable) -> usize {
        // About L² cells with up to ~500 loop candidates each; scheduling only.
        let len = table.mean_len().max(1);
        len * len * len.min(30) * 4
    }

    fn columns(&self) -> Result<DetailColumns, EngineError> {
        DetailColumns::try_new(&FIELDS, self.request)
    }

    #[inline]
    fn evaluate(&self, workspace: &mut Workspace, seq: ByteView<'_>) -> HairpinRow {
        match seq.as_slice() {
            Some(bytes) => self.row(workspace, bytes),
            None => {
                let mut buffer = [0u8; MAX_LENGTH];
                let bytes = &mut buffer[..seq.len()];
                seq.copy_to(0, bytes);
                self.row(workspace, bytes)
            }
        }
    }

    #[inline]
    fn passes(&self, row: &HairpinRow, cutoff: f64) -> bool {
        row.dg_kcal <= cutoff
    }

    fn push(&self, row: &HairpinRow, out: &mut DetailColumns) -> Result<(), EngineError> {
        if out.wants(0) {
            out.push_i32(0, i32::from(row.found))?;
        }
        for (field, value) in [
            (1, row.dg_kcal),
            (2, row.dh_kcal),
            (3, row.ds_cal_per_k),
            (4, row.tm_c),
        ] {
            if out.wants(field) {
                out.push_f64(field, value)?;
            }
        }
        Ok(())
    }
}

/// tm = fl(fl(dh / ds) − 273.15) (§5.7), except that an entropy of exactly zero (either
/// sign) reports tm = +0, as primer3-py 2.3.1 `calc_hairpin` does (black-box oracle,
/// 2026-10-17: `GGGCGCAAAAGCGCCC` at Na 4.151124678493096e51 mM and `ACGCAAG` at
/// 1.6409812743718861e-49 mM give ds = +0 and tm = +0). Only exact zero: ds = ±2⁻⁴⁷
/// follows the formula, ds = −∞ gives −273.15 and a NaN ds stays NaN (§8).
fn melting_temperature(dh: f64, ds: f64) -> f64 {
    if ds == 0.0 {
        0.0
    } else {
        dh / ds - K0
    }
}

/// A synthetic handle for unit tests: seeded values on a 0.1 grid (with a few infinite
/// ΔH entries and absent bonus components), never a parameter-file value.
#[cfg(test)]
pub(crate) fn synthetic_tables(seed: u64) -> P3Tables {
    use crate::p3files::Bonus;
    let mut state = seed | 1;
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    let mut entry = |inf: bool| {
        let r = next();
        let h = if inf && r % 11 == 0 {
            f64::INFINITY
        } else {
            ((r >> 8) % 20_001) as f64 / 10.0 - 1_500.0
        };
        let s = ((r >> 32) % 60_001) as f64 / 10.0 - 50.0;
        Hs { h, s: -s.abs() }
    };
    let mut table = |n: usize| (0..n).map(|_| entry(true)).collect::<Vec<_>>();
    let nn = table(625);
    let mm = table(625);
    let m1 = table(625);
    let m2 = table(625);
    let d3 = table(125);
    let d5 = table(125);
    let mut rows = || {
        let mut out = [OMEGA; LOOP_ROWS + 1];
        for slot in out.iter_mut().skip(1) {
            *slot = entry(false);
        }
        out
    };
    let (loop_in, loop_bu, loop_hp) = (rows(), rows(), rows());
    let mut bonuses = |n: usize| {
        (0..n)
            .map(|k| {
                let v = entry(false);
                Bonus {
                    h: (k % 3 != 0).then_some(v.h),
                    s: (k % 5 != 0).then_some(v.s),
                }
            })
            .collect::<Vec<_>>()
    };
    let b3 = bonuses(1024);
    let b4 = bonuses(4096);
    P3Tables {
        name: "synthetic".into(),
        sha256: String::new(),
        synthetic: true,
        nn,
        mm,
        m1,
        m2,
        d3,
        d5,
        loop_in,
        loop_bu,
        loop_hp,
        b3,
        b4,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `calc_hairpin`'s documented defaults (c00).
    const DEFAULTS: Args = Args {
        na: 50.0,
        k: 0.0,
        tris: 0.0,
        mg: 1.5,
        dntp: 0.6,
        temp_c: 37.0,
    };

    #[test]
    fn constants_have_the_documented_bits() {
        assert_eq!(ALPHA1, -300.0 / T37);
        assert_eq!(T37.to_bits(), 0x4073_6266_6666_6666);
        assert_eq!(theta(FRESH).to_bits(), (-0.0f64).to_bits());
        // Θ of a bare non-A·T terminal term is 0/0 (shared §2.4).
        assert!(theta(at(1, 2)).is_nan());
    }

    /// tm at an exact-zero entropy is +0; every other ds follows the formula (§5.7;
    /// oracle values of the 2026-10-17 black-box calls).
    #[test]
    fn melting_temperature_at_zero_entropy() {
        for ds in [0.0, -0.0] {
            for dh in [-43400.0, -13900.0, 0.0] {
                assert_eq!(melting_temperature(dh, ds).to_bits(), 0);
            }
        }
        let tiny = f64::from_bits(0x3D00_0000_0000_0000); // 2^-47
        assert_eq!(tiny, 2f64.powi(-47));
        for ds in [tiny, -tiny] {
            assert_eq!(melting_temperature(-43400.0, ds), -43400.0 / ds - K0);
            assert!(melting_temperature(-43400.0, ds).is_finite());
        }
        assert_eq!(melting_temperature(-39400.0, f64::NEG_INFINITY), -K0);
        assert!(melting_temperature(-39400.0, f64::NAN).is_nan());
        assert_eq!(
            melting_temperature(-16700.0, -55.061315628427664),
            30.148237780899308
        );
    }

    /// §4 at the preset defaults (oracle-confirmed through every found record of the
    /// golden set primer3_hairpin, shared §7 X2).
    #[test]
    fn default_pack() {
        let p = HairpinPack::new(&DEFAULTS).unwrap();
        assert_eq!(p.w, 120.0 * 0.9486832980505138);
        assert_eq!(p.mu, 163.84199576606164);
        assert_eq!(p.x_mu, 0.16384199576606165);
        assert_eq!(p.ln_mu, -1.808852756015843);
        assert_eq!(p.sigma, -0.6656578142138302);
        assert_eq!(p.t_u, T37);
        let c08 = HairpinPack::new(&Args {
            na: 30.1,
            k: 10.2,
            tris: 19.3,
            ..DEFAULTS
        })
        .unwrap();
        assert_eq!(c08.mv, 49.949999999999996);
        let one = HairpinPack::new(&Args {
            na: 1000.0,
            mg: 0.0,
            dntp: 0.0,
            ..DEFAULTS
        })
        .unwrap();
        assert_eq!((one.ln_mu.to_bits(), one.sigma.to_bits()), (0, 0));
    }

    #[test]
    fn pack_errors_in_step_order() {
        let e = |a: Args| HairpinPack::new(&a).unwrap_err().to_string();
        let no_salt = Args {
            na: 0.0,
            mg: 0.0,
            dntp: 0.0,
            ..DEFAULTS
        };
        assert!(e(no_salt).contains("is zero"));
        assert!(e(Args {
            na: 0.0,
            mg: 0.6,
            dntp: 0.6,
            ..DEFAULTS
        })
        .contains("is zero"));
        assert!(e(Args {
            na: 1e308,
            k: 1e308,
            ..DEFAULTS
        })
        .contains("infinite"));
        assert!(e(Args {
            na: 1e-321,
            mg: 0.0,
            dntp: 0.0,
            ..DEFAULTS
        })
        .contains("underflows to 0 mol/L"));
        assert!(e(Args {
            mg: -1.0,
            ..DEFAULTS
        })
        .contains("mg_mM=-1.0"));
    }

    /// sb.Conditions admits the binary64 literal −273.15 (above the exact −5463/20):
    /// T_u = +0 there, and only the doubles below it are rejected.
    #[test]
    fn absolute_zero_literal_is_accepted() {
        let at_zero = HairpinPack::new(&Args {
            temp_c: -273.15,
            ..DEFAULTS
        })
        .unwrap();
        assert_eq!(at_zero.t_u.to_bits(), 0);
        let below = f64::from_bits((-273.15f64).to_bits() + 1);
        assert!(below < -273.15);
        let message = HairpinPack::new(&Args {
            temp_c: below,
            ..DEFAULTS
        })
        .unwrap_err()
        .to_string();
        assert!(message.contains("not below -273.15"), "{message}");
    }

    /// primer3-cuda.md §4: the device pack exports the handle's tables and the pack
    /// bit for bit, and the T2 end-term table equals the per-cell end term for every
    /// closing pair and neighbour codes (synthetic handle; nothing printed).
    #[test]
    fn device_pack_is_a_bit_copy() {
        use crate::p3files::device_layout as l;
        let tables = Arc::new(synthetic_tables(0x5eed));
        let pack = HairpinPack::new(&DEFAULTS).unwrap();
        let kernel = HairpinPrimer3::new(pack, Arc::clone(&tables), &[]).unwrap();
        let device = kernel.device_pack().unwrap();
        let words = &device.tables;
        assert_eq!(words.len(), l::WORDS);
        let pair = |base: usize, k: usize, v: Hs| {
            assert_eq!(words[base + 2 * k], v.h.to_bits());
            assert_eq!(words[base + 2 * k + 1], v.s.to_bits());
        };
        for (base, table) in [
            (l::NN, &tables.nn),
            (l::MM, &tables.mm),
            (l::M1, &tables.m1),
            (l::M2, &tables.m2),
            (l::D3, &tables.d3),
            (l::D5, &tables.d5),
        ] {
            for (k, &v) in table.iter().enumerate() {
                pair(base, k, v);
            }
        }
        for (base, table) in [
            (l::LOOP_IN, &tables.loop_in),
            (l::LOOP_BU, &tables.loop_bu),
            (l::LOOP_HP, &tables.loop_hp),
        ] {
            for (k, &v) in table.iter().enumerate() {
                pair(base, k, v);
            }
        }
        for (base, table) in [(l::B3, &tables.b3), (l::B4, &tables.b4)] {
            for (k, bonus) in table.iter().enumerate() {
                let w = &words[base + 3 * k..base + 3 * k + 3];
                assert_eq!(w[0], bonus.h.map_or(0, f64::to_bits));
                assert_eq!(w[1], bonus.s.map_or(0, f64::to_bits));
                assert_eq!(
                    w[2],
                    u64::from(bonus.h.is_some()) | u64::from(bonus.s.is_some()) << 1
                );
            }
        }
        assert_eq!(device.ends.len(), 1250);
        for p in 0..5u8 {
            for pp in 0..5u8 {
                for q in 0..5u8 {
                    for qq in 0..5u8 {
                        let k = idx4(p, pp, q, qq);
                        let want = if p < 4 && q < 4 {
                            end_term(&tables, p, pp, q, qq)
                        } else {
                            OMEGA
                        };
                        assert_eq!(device.ends[2 * k], want.h.to_bits());
                        assert_eq!(device.ends[2 * k + 1], want.s.to_bits());
                    }
                }
            }
        }
        let c = device.scalars;
        assert_eq!((c.sigma, c.t_u), (pack.sigma.to_bits(), pack.t_u.to_bits()));
        assert_eq!(
            (c.t37, c.alpha1, c.k0),
            (T37.to_bits(), ALPHA1.to_bits(), K0.to_bits())
        );
        assert_eq!(
            (c.omega_h, c.omega_s),
            (OMEGA.h.to_bits(), OMEGA.s.to_bits())
        );
        assert_eq!(
            (c.fresh_h, c.fresh_s),
            (FRESH.h.to_bits(), FRESH.s.to_bits())
        );
        assert_eq!((c.q4, c.q1), (0, 0));
    }
}
