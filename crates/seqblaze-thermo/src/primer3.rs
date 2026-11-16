//! T1 `tm` `compat="primer3"`: primer3-py 2.3.1 module-level `calc_tm` (default method
//! pair `santalucia`/`santalucia`, `max_nn_length` 60, `dmso_fact` 0.6) replayed from
//! ledger units `sl1998` and `va2001`, with correctly rounded `log` and `log10` (QPT2).
//! Spec: docs/wiki/thermo/tm-primer3.md. Derivation route: a behaviour specification
//! written from a reading of primer3-py 2.3.1's source (no code copied; D94), confirmed
//! against the unmodified tool; this module was written from the spec page only.
//!
//! Per call (host, once): P1-P13 of §4, one binary64 operation per statement, errors in
//! step order (P6, P7, P9, P11), and the four logarithms CR64(ln x_mu), CR64(log10 x_mu),
//! CR64(ln x4), CR64(ln x1) (`cr::ln`, `cr::log10`). Per sequence: the exact integer
//! tallies T1-T4 of §5.1, then F1-F9 (2 ≤ L ≤ 60) or L1-L6 (61 ≤ L ≤ 200) as separate
//! binary64 operations, no FMA (rustc never contracts; no `mul_add` here), one NaN encoding.
use seqblaze_core::{
    detail::{canonical, DetailColumns, DetailField, DetailRequest, FieldKind},
    seq::{ByteView, SeqTable},
    unary::UnaryKernel,
    validate::EngineError,
};

use crate::{
    cr::{self, Rational},
    params::{Primer3Params, Tally},
    tm::{length_error, K0, MAX_LENGTH, MIN_LENGTH},
};

/// The only field of `compat="primer3"` (§6).
pub static FIELDS: [DetailField; 1] = [DetailField {
    name: "tm_c",
    kind: FieldKind::Float64,
}];

/// §2.2 published constants that are not ledger entries, as binary64 bit patterns.
/// P2: Tris+ = Tris/2 (QT12).
pub const TRIS_FACTOR: f64 = 0.5;
/// P7: mM -> mol/L, by division (reader S1.9).
pub const MILLI_DIV: f64 = 1000.0;
/// P9: nM -> mol/L together with C_T/4 (SantaLucia 1998, below Eq. 3; reader S1.7, S1.9).
pub const QUARTER_NANO_DIV: f64 = f64::from_bits(0x41ED_CD65_0000_0000);
/// P9: nM -> mol/L, C_T of a self-complementary duplex (SantaLucia 1998; reader S1.7, S1.9).
pub const NANO_DIV: f64 = f64::from_bits(0x41CD_CD65_0000_0000);
/// F1: the 100 cal/mol tally unit -> cal/mol (reader S1.4).
pub const TALLY_H: f64 = 100.0;
/// F2: the 0.1 cal/(mol·K) tally unit, by multiplication with the binary64 0.1 (S1.4).
pub const TALLY_S: f64 = f64::from_bits(0x3FB9_9999_9999_999A);
/// P12: the default `dmso_fact` 0.6 stored as binary32 (Musielski et al. 1981, as printed
/// in the Primer3 manual §17 and the primer3-py 2.3.1 API page; confirmed by reader S2.2,
/// S4.1 and golden d1–d6).
pub const DMSO_FACT_F32: f32 = f32::from_bits(0x3F19_999A);
/// F9, L1: 0.453 per GC fraction (Blake & Delcourt 1996, as printed in the Primer3 manual
/// §17; confirmed by reader S4.2 and golden f1–f6).
pub const FORMAMIDE_GC: f64 = f64::from_bits(0x3FDC_FDF3_B645_A1CB);
/// F9, L1: 2.88 (same source and confirmation).
pub const FORMAMIDE_C: f64 = f64::from_bits(0x4007_0A3D_70A3_D70A);
/// P13: 81.5 °C (Bolton & McCarthy 1962 via Sambrook et al. 1989, as printed in the Primer3
/// manual §17 and the primer3-py 2.3.1 API page; confirmed by reader S3.2 and golden c01–c13).
pub const LONG_C: f64 = 81.5;
/// P13: 16.6 on log10[M+] (same source and confirmation).
pub const LONG_SALT: f64 = f64::from_bits(0x4030_9999_9999_999A);
/// L4: 41 per GC fraction, the printed 0.41 per %GC × 100 (same source; S3.2, S3.3).
pub const LONG_GC: f64 = 41.0;
/// L6: 600 °C·nt (same source and confirmation).
pub const LONG_LEN: f64 = 600.0;
/// §5.3: the nearest-neighbour path for L ≤ 60, the long path above (reader S3.1, QPT1).
pub const MAX_NN_LENGTH: usize = 60;

/// The per-call values the device row reads (tm-compat-cuda.md §3.4), in that page's
/// field order: integer tallies as int64, every binary64 as its bit pattern. Integer
/// copies and `to_bits` only: building it performs no floating-point operation
/// (text-nulls-fp-environment.md §7.3).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Primer3DevicePack {
    /// (τ_H, τ_S) of the step with top strand XY at k = 4·code(X) + code(Y).
    pub steps: [[i64; 2]; 16],
    /// (τ_H, τ_S) of the A·T terminus, the G·C terminus and the symmetry entry.
    pub ends: [i64; 6],
    /// M: the nearest-neighbour path for L ≤ M, the long path above (§5.3).
    pub max_nn_length: i64,
    /// c_s, λ (P8 ln), γ₄, γ₁ (P10), δ_D (P12), φ_F (P11), B, Q (P13).
    pub salt: u64,
    pub ln_mu: u64,
    pub gamma4: u64,
    pub gamma1: u64,
    pub dmso_t: u64,
    pub phi_formamide: u64,
    pub base: u64,
    pub sig10: u64,
    /// κ_H, κ_S (F1, F2), a_F, b_F (F9, L1), a_G (L4), a_L (L6), K₀ (F7).
    pub tally_h: u64,
    pub tally_s: u64,
    pub formamide_gc: u64,
    pub formamide_c: u64,
    pub long_gc: u64,
    pub long_len: u64,
    pub k0: u64,
}

/// The `calc_tm` inputs of §3 after the facade's mapping (mM, nM, %, mol/L).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Args {
    pub na: f64,
    pub k: f64,
    pub tris: f64,
    pub mg: f64,
    pub dntp: f64,
    /// Total strand concentration C, nM (`dna_conc`).
    pub dna: f64,
    /// `dmso_conc`, %.
    pub dmso: f64,
    /// `formamide_conc`, mol/L.
    pub formamide: f64,
}

/// The per-call pack (§4, §12): P1-P13 and exact single/grouped-step tally lookups.
/// Built once, shared read-only by every worker.
#[derive(Debug, Clone)]
pub struct Primer3Pack {
    pub params: &'static Primer3Params,
    /// P3: monovalent cations, mM.
    pub mv: f64,
    /// P5: monovalent equivalent, mM.
    pub mu: f64,
    /// P7: mu in mol/L.
    pub x_mu: f64,
    /// P8: CR64(ln x_mu), CR64(log10 x_mu).
    pub ln_mu: f64,
    pub log10_mu: f64,
    /// P9: C/4 and C in mol/L.
    pub x4: f64,
    pub x1: f64,
    /// P10: CR64(ln x4), CR64(ln x1).
    pub ln_x4: f64,
    pub ln_x1: f64,
    /// P10: RN(R · ln x4), RN(R · ln x1).
    pub gamma4: f64,
    pub gamma1: f64,
    /// P11: DMSO (%) and formamide (mol/L) stored as binary32, widened exactly.
    pub phi_dmso: f64,
    pub phi_formamide: f64,
    /// P12: DMSO depression, °C.
    pub dmso_t: f64,
    /// P13: 81.5 − dmso_t and 16.6 · log10 x_mu (long path).
    pub base: f64,
    pub sig10: f64,
    /// (τ_H, τ_S) of step XY in canonical A,C,G,T order for the device export.
    steps: [Tally; 16],
    /// CPU lookups in the local ASCII codes A,C,G,T = 0,1,3,2.
    local_steps: [Tally; 16],
    /// Exact sum of three adjacent steps, indexed by four local two-bit codes.
    quadruplets: [Tally; 256],
    /// F3, with its two binary64 multiplications in the specified order.
    salt_by_length: [f64; MAX_NN_LENGTH + 1],
    /// L6's independently rounded division, indexed by the sequence length.
    long_length_term: [f64; MAX_LENGTH + 1],
}

fn parameter(message: String) -> EngineError {
    EngineError::Parameter(message)
}

/// `x` in the layout of Python's `repr(float)`, as in compat Biopython's messages.
fn py_float(x: f64) -> String {
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

fn cap_error(name: &str, bits: u64) -> EngineError {
    parameter(format!(
        "internal error: the correctly rounded {name} needs more than {bits} bits; nothing was \
         computed"
    ))
}

/// CR64(ln x) for a positive finite binary64 x (§4 P8, P10).
fn ln1(x: f64, name: &str) -> Result<f64, EngineError> {
    cr::ln(&Rational::from_f64(x), &Rational::from_ratio(1, 1))
        .map_err(|cap| cap_error(name, cap.bits))
}

/// CR64(log10 x) for a positive finite binary64 x (§4 P8, §9.3).
fn log10_1(x: f64, name: &str) -> Result<f64, EngineError> {
    cr::log10(&Rational::from_f64(x)).map_err(|cap| cap_error(name, cap.bits))
}

/// P11: the binding's binary32 storage, rounded to nearest with ties to even (Rust's
/// `as f32`: ±inf on overflow, subnormals and ±0 on underflow), then widened exactly.
#[inline]
fn binary32(x: f64) -> f64 {
    f64::from(x as f32)
}

impl Primer3Pack {
    /// P1-P13 of §4. Raises `ParameterError` in step order: P6, P7, P9, P11.
    /// Runs in the IEEE default floating-point environment whatever the
    /// caller's state, which is restored on return (text-nulls-fp-environment.md
    /// §7.3 item 2).
    pub fn new(params: &'static Primer3Params, args: &Args) -> Result<Self, EngineError> {
        seqblaze_core::fpenv::guarded(|| Self::build(params, args))
    }

    fn build(params: &'static Primer3Params, args: &Args) -> Result<Self, EngineError> {
        let named = [
            ("na_mM", args.na),
            ("k_mM", args.k),
            ("tris_mM", args.tris),
            ("mg_mM", args.mg),
            ("dntp_mM", args.dntp),
            ("oligo_nM", args.dna),
            ("dmso_pct", args.dmso),
            ("formamide_M", args.formamide),
        ];
        for (name, value) in named {
            // sb.Conditions admits only these; a direct binding call is checked too.
            if !value.is_finite() || value < 0.0 {
                return Err(parameter(format!(
                    "compat='primer3': {name}={} must be finite and >= 0",
                    py_float(value)
                )));
            }
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
        let m1 = args.na + args.k; // P1
        let m2 = args.tris * TRIS_FACTOR; // P2
        let mv = m1 + m2; // P3
        let w = if args.mg > args.dntp {
            // P4: free Mg2+, von Ahsen 2001 Eq. 7 (strict: at mg = dntp the term is +0).
            let free = args.mg - args.dntp;
            params.c_mg * free.sqrt()
        } else {
            0.0
        };
        let mu = mv + w; // P5
        if mu == 0.0 || mu == f64::INFINITY {
            // P6 (QPT4)
            let (what, instead) = if mu == 0.0 {
                (
                    "zero: no monovalent cation and no free Mg2+ (mg_mM > dntp_mM)",
                    "a non-physical Tm for L <= 60 (-273.15 before the DMSO and formamide \
                     corrections) and -inf for L > 60",
                )
            } else {
                (
                    "infinite (the monovalent sum overflows binary64)",
                    "a non-physical Tm (-273.15 before the DMSO and formamide corrections) or \
                     +inf",
                )
            };
            return Err(parameter(format!(
                "compat='primer3': the monovalent equivalent of {} is {what}; primer3's calc_tm \
                 returns {instead}. Give a positive, finite salt, e.g. \
                 sb.Conditions.preset('primer3')",
                conditions()
            )));
        }
        let x_mu = mu / MILLI_DIV; // P7
        if x_mu == 0.0 {
            return Err(parameter(format!(
                "compat='primer3': the monovalent equivalent {} mM of {} underflows to 0 mol/L \
                 (mu/1000); primer3's calc_tm takes log(0) and returns a non-physical Tm for L <= 60 \
                 (-273.15 before the DMSO and formamide corrections) and -inf for L > 60. Use a \
                 larger salt concentration",
                py_float(mu),
                conditions()
            )));
        }
        let ln_mu = ln1(x_mu, "ln_mu")?; // P8
        let log10_mu = log10_1(x_mu, "log10_mu")?;
        let x4 = args.dna / QUARTER_NANO_DIV; // P9
        let x1 = args.dna / NANO_DIV;
        if x4 == 0.0 {
            // Message only: inside the window C/4 = 0 < C primer3 gives self-complementary
            // rows of L <= 60 a finite value; compat still raises for the whole call (QPT4).
            let rows = if x1 == 0.0 {
                "rows of L <= 60 (C underflows too)"
            } else {
                "rows of L <= 60 that are not self-complementary (self-complementary rows get a \
                 finite value, since C stays > 0)"
            };
            return Err(parameter(format!(
                "compat='primer3': oligo_nM={} gives a strand concentration C/4 that underflows \
                 to 0 mol/L; primer3's calc_tm takes log(0) and returns a non-physical Tm \
                 (-273.15 before the DMSO and formamide corrections) for {rows}, and the \
                 ordinary value for L > 60, which ignores oligo_nM; seqblaze raises for the whole call. Use a larger \
                 oligo_nM, e.g. oligo_nM=50",
                py_float(args.dna)
            )));
        }
        let ln_x4 = ln1(x4, "ln_x4")?; // P10
        let ln_x1 = ln1(x1, "ln_x1")?;
        let gamma4 = params.gas_r * ln_x4;
        let gamma1 = params.gas_r * ln_x1;
        let phi_dmso = binary32(args.dmso); // P11
        let kap_dmso = f64::from(DMSO_FACT_F32);
        let phi_formamide = binary32(args.formamide);
        for (name, value, phi) in [
            ("dmso_pct", args.dmso, phi_dmso),
            ("formamide_M", args.formamide, phi_formamide),
        ] {
            if phi.is_infinite() {
                return Err(parameter(format!(
                    "compat='primer3': {name}={} is beyond the binary32 range in which primer3-py \
                     stores it (about 3.4e+38); primer3's calc_tm returns -inf for the same \
                     call. Use a value below 3.4e+38",
                    py_float(value)
                )));
            }
        }
        let dmso_t = phi_dmso * kap_dmso; // P12 (exact)
        let base = LONG_C - dmso_t; // P13
        let sig10 = LONG_SALT * log10_mu;
        let mut steps = [Tally { h: 0, s: 0 }; 16];
        for (index, &class) in params.step_class.iter().enumerate() {
            steps[index] = params.stack[class];
        }
        let local_code = [0usize, 1, 3, 2];
        let local_steps =
            std::array::from_fn(|index| steps[local_code[index >> 2] * 4 + local_code[index & 3]]);
        let quadruplets = std::array::from_fn(|index| {
            let left = local_steps[index >> 4];
            let middle = local_steps[(index >> 2) & 15];
            let right = local_steps[index & 15];
            Tally {
                h: left.h + middle.h + right.h,
                s: left.s + middle.s + right.s,
            }
        });
        let mut salt_by_length = [0.0; MAX_NN_LENGTH + 1];
        for (len, salt) in salt_by_length.iter_mut().enumerate().skip(MIN_LENGTH) {
            let n_pairs = (len - 1) as f64;
            let salt0 = params.salt * n_pairs;
            *salt = salt0 * ln_mu; // F3, independently rounded per length.
        }
        let mut long_length_term = [0.0; MAX_LENGTH + 1];
        for (len, term) in long_length_term
            .iter_mut()
            .enumerate()
            .skip(MAX_NN_LENGTH + 1)
        {
            *term = LONG_LEN / len as f64; // L6: division, never a reciprocal product.
        }
        Ok(Self {
            params,
            mv,
            mu,
            x_mu,
            ln_mu,
            log10_mu,
            x4,
            x1,
            ln_x4,
            ln_x1,
            gamma4,
            gamma1,
            phi_dmso,
            phi_formamide,
            dmso_t,
            base,
            sig10,
            steps,
            local_steps,
            quadruplets,
            salt_by_length,
            long_length_term,
        })
    }

    /// The device pack of tm-compat-cuda.md §3.4: the values this pack's `row` reads.
    pub fn device_pack(&self) -> Primer3DevicePack {
        let p = self.params;
        let mut steps = [[0i64; 2]; 16];
        for (slot, step) in steps.iter_mut().zip(&self.steps) {
            *slot = [i64::from(step.h), i64::from(step.s)];
        }
        let tally = |t: Tally| [i64::from(t.h), i64::from(t.s)];
        let [at, gc, sym] = [tally(p.init_at), tally(p.init_gc), tally(p.symmetry)];
        Primer3DevicePack {
            steps,
            ends: [at[0], at[1], gc[0], gc[1], sym[0], sym[1]],
            max_nn_length: MAX_NN_LENGTH as i64,
            salt: p.salt.to_bits(),
            ln_mu: self.ln_mu.to_bits(),
            gamma4: self.gamma4.to_bits(),
            gamma1: self.gamma1.to_bits(),
            dmso_t: self.dmso_t.to_bits(),
            phi_formamide: self.phi_formamide.to_bits(),
            base: self.base.to_bits(),
            sig10: self.sig10.to_bits(),
            tally_h: TALLY_H.to_bits(),
            tally_s: TALLY_S.to_bits(),
            formamide_gc: FORMAMIDE_GC.to_bits(),
            formamide_c: FORMAMIDE_C.to_bits(),
            long_gc: LONG_GC.to_bits(),
            long_len: LONG_LEN.to_bits(),
            k0: K0.to_bits(),
        }
    }

    /// F9 and L1: the formamide term RN(RN(RN(RN(0.453 · n_GC) / L) − 2.88) · phiF).
    #[inline]
    fn formamide(&self, n_gc: f64, len: f64) -> f64 {
        let a = FORMAMIDE_GC * n_gc;
        let b = a / len;
        let c = b - FORMAMIDE_C;
        c * self.phi_formamide
    }

    /// Exact NN tallies, optionally counting GC in the same byte scan. A
    /// compile-time flag keeps the default zero-formamide scan free of GC work.
    #[inline]
    fn tallies<const COUNT_GC: bool>(&self, seq: &[u8]) -> (i32, i32, usize) {
        let p = self.params;
        let first = (seq[0] >> 1) & 3;
        let last = (seq[seq.len() - 1] >> 1) & 3;
        let terminus = |code: u8| {
            if code & 1 == 0 {
                p.init_at
            } else {
                p.init_gc
            }
        };
        let (t_first, t_last) = (terminus(first), terminus(last));
        let mut th = t_first.h + t_last.h;
        let mut ts = t_first.s + t_last.s;
        let mut n_gc = if COUNT_GC { usize::from(first & 1) } else { 0 };
        let mut previous = first;
        let mut groups = seq[1..].chunks_exact(3);
        for group in &mut groups {
            let c1 = (group[0] >> 1) & 3;
            let c2 = (group[1] >> 1) & 3;
            let c3 = (group[2] >> 1) & 3;
            let index = usize::from(previous << 6 | c1 << 4 | c2 << 2 | c3);
            let tally = self.quadruplets[index];
            th += tally.h;
            ts += tally.s;
            if COUNT_GC {
                n_gc += usize::from((c1 & 1) + (c2 & 1) + (c3 & 1));
            }
            previous = c3;
        }
        for &byte in groups.remainder() {
            let code = (byte >> 1) & 3;
            let step = self.local_steps[usize::from(previous << 2 | code)];
            th += step.h;
            ts += step.s;
            if COUNT_GC {
                n_gc += usize::from(code & 1);
            }
            previous = code;
        }
        (th, ts, n_gc)
    }

    /// `tm_c` of one validated sequence (upper-case ACGT, 2 ≤ L ≤ 200): §5.
    /// Expects the IEEE default floating-point environment: it is called per
    /// item from guarded scheduler units and carries no guard of its own
    /// (text-nulls-fp-environment.md §7.3 item 2).
    #[inline]
    pub fn row(&self, seq: &[u8]) -> f64 {
        let p = self.params;
        let len = seq.len();
        let l = len as f64;
        if len > MAX_NN_LENGTH {
            // C and G both have low bits 11; unlike a CODE table lookup this
            // validated-ASCII predicate permits a vectorized integer count.
            let n_gc = seq.iter().filter(|&&byte| byte & 3 == 3).count() as f64;
            // §5.3 long path: no tallies, no dna_conc, no self-complementarity.
            let eps = self.formamide(n_gc, l); // L1
            let t1 = self.base + eps; // L2
            let t2 = t1 + self.sig10; // L3
            let fraction = n_gc / l; // L4
            let gc = LONG_GC * fraction;
            let t3 = t2 + gc; // L5
            return canonical(t3 - self.long_length_term[len]); // L6
        }
        // T1, T2: exact integer tallies (|Σ| < 2^16, §2.1 rule 8).
        let needs_gc = self.phi_formamide != 0.0;
        let (mut th, mut ts, n_gc) = if needs_gc {
            self.tallies::<true>(seq)
        } else {
            self.tallies::<false>(seq)
        };
        // T3: the local codes of complementary validated bases differ by 2.
        let sc = len % 2 == 0 && (0..len / 2).all(|i| (seq[i] ^ seq[len - 1 - i]) & 6 == 4);
        if sc {
            th += p.symmetry.h;
            ts += p.symmetry.s;
        }
        let dh = f64::from(th) * TALLY_H; // F1 (exact)
        let ds = f64::from(ts) * TALLY_S; // F2
        let ds_salt = ds + self.salt_by_length[len]; // F3, F4
        let gamma = if sc { self.gamma1 } else { self.gamma4 };
        let den = ds_salt + gamma; // F5
        let tk = dh / den; // F6: IEEE quotient (±inf for den = ±0, §5.4)
        let tc = tk - K0; // F7
        let td = tc - self.dmso_t; // F8
        let eps = if needs_gc {
            self.formamide(n_gc as f64, l)
        } else {
            // F9's coefficient is strictly negative for 0 <= n_GC <= L.
            // Its product with ±0 is ∓0; keep the final addition to retain
            // the exact signed-zero behavior of the original expression.
            -self.phi_formamide
        };
        canonical(td + eps) // F9
    }
}

/// The unary kernel of one compat call: the pack, `tm_c` only.
pub struct TmPrimer3<'p> {
    pack: &'p Primer3Pack,
}

impl<'p> TmPrimer3<'p> {
    /// `fields` by name; only `tm_c` exists under compat (§6).
    pub fn new(pack: &'p Primer3Pack, fields: &[&str]) -> Result<Self, EngineError> {
        if let Some(name) = fields.iter().find(|name| **name != "tm_c") {
            return Err(parameter(format!(
                "field {name:?} is not available under compat='primer3': it has fields=['tm_c'] only"
            )));
        }
        Ok(Self { pack })
    }
}

impl UnaryKernel for TmPrimer3<'_> {
    type Row = f64;
    type Cutoff = f64;
    type Scratch = ();

    fn scratch(&self) -> Result<(), EngineError> {
        Ok(())
    }

    fn name(&self) -> &'static str {
        "tm"
    }

    fn length_limits(&self) -> (usize, usize) {
        (MIN_LENGTH, MAX_LENGTH)
    }

    fn length_error(&self, index: usize, length: usize, limit: usize) -> EngineError {
        length_error(index, length, limit)
    }

    fn item_cost(&self, table: &SeqTable) -> usize {
        // Two integer additions per base plus the finalize.
        table.mean_len() + 16
    }

    fn columns(&self) -> Result<DetailColumns, EngineError> {
        DetailColumns::try_new(&FIELDS, DetailRequest::NONE.with(0))
    }

    #[inline]
    fn evaluate(&self, _: &mut (), seq: ByteView<'_>) -> f64 {
        match seq.as_slice() {
            Some(bytes) => self.pack.row(bytes),
            None => {
                let mut buffer = [0u8; MAX_LENGTH];
                let bytes = &mut buffer[..seq.len()];
                seq.copy_to(0, bytes);
                self.pack.row(bytes)
            }
        }
    }

    #[inline]
    fn passes(&self, row: &f64, cutoff: f64) -> bool {
        *row >= cutoff
    }

    #[inline]
    fn push(&self, row: &f64, out: &mut DetailColumns) -> Result<(), EngineError> {
        out.push_f64(0, *row)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::params::PRIMER3_SANTALUCIA;

    /// `calc_tm`'s documented defaults (c00).
    const DEFAULTS: Args = Args {
        na: 50.0,
        k: 0.0,
        tris: 0.0,
        mg: 1.5,
        dntp: 0.6,
        dna: 50.0,
        dmso: 0.0,
        formamide: 0.0,
    };
    const README: &[u8] = b"GTAAAACGACGGCCAGT";
    const LONG61: &[u8] = b"TCGCTCTCGCTAACGTGACACAGTCGGGGGTGATGTTGAGAGACAACCTAAGGAAAATCAC";

    fn pack(args: Args) -> Result<Primer3Pack, EngineError> {
        Primer3Pack::new(&PRIMER3_SANTALUCIA, &args)
    }

    #[test]
    fn literals_are_the_nearest_binary64() {
        assert_eq!(QUARTER_NANO_DIV, 4e9);
        assert_eq!(NANO_DIV, 1e9);
        assert_eq!(TALLY_S, 0.1);
        assert_eq!(DMSO_FACT_F32, 0.6f32);
        assert_eq!(f64::from(DMSO_FACT_F32), 0.6000000238418579);
        assert_eq!(FORMAMIDE_GC, 0.453);
        assert_eq!(FORMAMIDE_C, 2.88);
        assert_eq!(LONG_SALT, 16.6);
        assert_eq!(K0, 273.15);
    }

    /// §5.5 worked examples (the [oracle] rows equal `primer3_tm` golden values; the
    /// others are the page's replay values, checked bit for bit by the golden tests).
    #[test]
    fn worked_examples() {
        let p = pack(DEFAULTS).unwrap();
        assert_eq!(p.mu, 163.84199576606164);
        assert_eq!(p.x_mu, 0.16384199576606165);
        assert_eq!(p.ln_mu, -1.808852756015843);
        assert_eq!(p.log10_mu, -0.7855747705131697);
        assert_eq!(p.x4, 1.25e-08);
        assert_eq!(p.x1, 5e-08);
        assert_eq!(p.gamma4, -36.158506401772016);
        assert_eq!(p.gamma1, -33.40393950622679);
        assert_eq!(p.row(README), 54.695494691112515);
        assert_eq!(p.row(b"TC"), -172.67214850096042);
        assert_eq!(p.row(b"GA"), -172.67214850096042);
        assert_eq!(p.row(b"AG"), -177.61563739773436);
        assert_eq!(p.row(b"CT"), -177.61563739773436);
        assert_eq!(p.row(b"AA"), -208.22025544038274);
        assert_eq!(p.row(b"TA"), -219.6185699812095);
        assert_eq!(p.row(b"TGCA"), -47.542337075818494);
        assert_eq!(p.row(b"CATG"), -51.340411282929665);
        assert_eq!(p.sig10, -13.040541190518619);
        assert_eq!(p.row(LONG61), 79.45945880948138);
        assert_eq!(p.row(&[b'G'; 200]), 106.45945880948138);
        let d = pack(Args {
            dmso: 5.0,
            ..DEFAULTS
        })
        .unwrap();
        assert_eq!(d.dmso_t, 3.0000001192092896);
        assert_eq!(d.row(README), 51.695494571903225);
        assert_eq!(d.row(LONG61), 76.45945869027209);
        let f = pack(Args {
            formamide: 1.0,
            ..DEFAULTS
        })
        .unwrap();
        assert_eq!(f.row(README), 52.05531822052428);
        assert_eq!(f.row(LONG61), 76.80967192423547);
        let t = |dmso: f64| pack(Args { dmso, ..DEFAULTS }).unwrap().row(README);
        assert_eq!(t(1.0 + 2f64.powi(-24)), 54.09549466727066); // d3: ties to even -> 1
        assert_eq!(t(1.0 + 3.0 * 2f64.powi(-24)), 54.095494524219504); // d4 -> 1 + 2^-22
        let manual = pack(Args {
            mg: 0.0,
            ..DEFAULTS
        })
        .unwrap();
        assert_eq!(manual.row(b"CGTGACGTGACGGACT"), 50.553531469275185);
    }

    #[test]
    fn mv_order_and_binary32_storage() {
        let p = pack(Args {
            na: 30.1,
            k: 10.2,
            tris: 19.3,
            ..DEFAULTS
        })
        .unwrap();
        assert_eq!(p.mv, 49.949999999999996);
        let s = pack(Args {
            dmso: 1.3,
            formamide: 0.1,
            ..DEFAULTS
        })
        .unwrap();
        assert_eq!(s.phi_dmso, 1.2999999523162842);
        assert_eq!(s.phi_formamide, 0.10000000149011612);
        let tiny = pack(Args {
            dmso: 1e-50,
            formamide: 1e-50,
            ..DEFAULTS
        })
        .unwrap();
        let plain = pack(DEFAULTS).unwrap();
        for seq in [README, LONG61, b"TGCA".as_slice()] {
            assert_eq!(tiny.row(seq).to_bits(), plain.row(seq).to_bits());
        }
    }

    #[test]
    fn exact_logarithm_arguments() {
        let one = pack(Args {
            na: 1000.0,
            mg: 0.0,
            dntp: 0.0,
            ..DEFAULTS
        })
        .unwrap();
        assert_eq!((one.ln_mu.to_bits(), one.log10_mu.to_bits()), (0, 0));
        let ten = pack(Args {
            na: 10000.0,
            mg: 0.0,
            dntp: 0.0,
            ..DEFAULTS
        })
        .unwrap();
        assert_eq!(ten.log10_mu, 1.0);
        let c4 = pack(Args {
            dna: 4e9,
            ..DEFAULTS
        })
        .unwrap();
        assert_eq!(c4.gamma4.to_bits(), 0);
    }

    #[test]
    fn errors_in_step_order() {
        let e = |a: Args| pack(a).unwrap_err().to_string();
        let zero = Args {
            na: 0.0,
            mg: 0.0,
            dntp: 0.0,
            ..DEFAULTS
        };
        assert!(e(zero).contains("is zero"));
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
            dna: 1e-320,
            ..DEFAULTS
        })
        .contains("oligo_nM=1e-320"));
        assert!(e(Args {
            dmso: 1e39,
            ..DEFAULTS
        })
        .contains("dmso_pct=1e+39"));
        assert!(e(Args {
            formamide: 1e39,
            ..DEFAULTS
        })
        .contains("formamide_M=1e+39"));
        // P6 before P9 before P11.
        let all = e(Args {
            dna: 1e-320,
            dmso: 1e39,
            ..zero
        });
        assert!(all.contains("is zero"), "{all}");
        let late = e(Args {
            dna: 1e-320,
            dmso: 1e39,
            ..DEFAULTS
        });
        assert!(late.contains("oligo_nM"), "{late}");
        assert!(e(Args {
            mg: -1.0,
            ..DEFAULTS
        })
        .contains("mg_mM=-1.0"));
        assert!(e(Args {
            dmso: f64::NAN,
            ..DEFAULTS
        })
        .contains("dmso_pct=nan"));
        // Largest finite binary32 still stores; just beyond the rounding boundary overflows.
        assert!(pack(Args {
            dmso: f64::from(f32::MAX),
            ..DEFAULTS
        })
        .is_ok());
    }

    #[test]
    fn revcomp_rows_are_identical() {
        let rc = |s: &[u8]| -> Vec<u8> {
            s.iter()
                .rev()
                .map(|&b| match b {
                    b'A' => b'T',
                    b'C' => b'G',
                    b'G' => b'C',
                    _ => b'A',
                })
                .collect()
        };
        let p = pack(Args {
            formamide: 0.35,
            dmso: 1.3,
            ..DEFAULTS
        })
        .unwrap();
        for s in [
            README,
            LONG61,
            b"ACGTTGCAAGGTCCATGAGCTTAGCA".as_slice(),
            b"AC",
        ] {
            assert_eq!(p.row(s).to_bits(), p.row(&rc(s)).to_bits());
        }
    }

    #[test]
    fn device_pack_carries_the_row_values() {
        let p = pack(Args {
            dmso: 1.3,
            formamide: 0.1,
            ..DEFAULTS
        })
        .unwrap();
        let d = p.device_pack();
        let q = &PRIMER3_SANTALUCIA;
        for (k, step) in d.steps.iter().enumerate() {
            let t = q.stack[q.step_class[k]];
            assert_eq!(*step, [i64::from(t.h), i64::from(t.s)]);
        }
        let e = [q.init_at, q.init_gc, q.symmetry];
        let ends: Vec<i64> = e
            .iter()
            .flat_map(|t| [i64::from(t.h), i64::from(t.s)])
            .collect();
        assert_eq!(d.ends.to_vec(), ends);
        assert_eq!(d.max_nn_length, 60);
        let bits = [
            d.salt,
            d.ln_mu,
            d.gamma4,
            d.gamma1,
            d.dmso_t,
            d.phi_formamide,
            d.base,
            d.sig10,
        ];
        let want = [
            q.salt,
            p.ln_mu,
            p.gamma4,
            p.gamma1,
            p.dmso_t,
            p.phi_formamide,
            p.base,
            p.sig10,
        ];
        assert_eq!(bits, want.map(f64::to_bits));
        let constants = [
            d.tally_h,
            d.tally_s,
            d.formamide_gc,
            d.formamide_c,
            d.long_gc,
            d.long_len,
            d.k0,
        ];
        let want = [
            TALLY_H,
            TALLY_S,
            FORMAMIDE_GC,
            FORMAMIDE_C,
            LONG_GC,
            LONG_LEN,
            K0,
        ];
        assert_eq!(constants, want.map(f64::to_bits));
    }

    #[test]
    fn fields_other_than_tm_c_are_rejected() {
        let p = pack(DEFAULTS).unwrap();
        assert!(TmPrimer3::new(&p, &["tm_c"]).is_ok());
        assert!(TmPrimer3::new(&p, &["dg_kcal"]).is_err());
    }
}
