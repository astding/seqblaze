//! K3 native `duplex(model="pf")`: the intermolecular hybridization partition function
//! on the CPU (docs/wiki/thermo/duplex-pf.md).
//!
//! Same structure space and compiled pack as K1; only the D component of K1's per-call
//! cell keys is read (§2). Per call (host, once): the §5.4 exponent check, the
//! correctly rounded Boltzmann weights W(D) of every cell (§5.1–§5.2), deduplicated by
//! D, and the finalize constants (§5.3). Per pair: the canonical orientation (§6.1), the
//! F53 sum-product recurrence in its fixed fold orders (§6.2), and the finalize (§7):
//! the table-driven logarithm g, `dg_ens_kcal` = RN(c_T · g) and the upward-rounded
//! bound `dg_ens_err_kcal`. No transcendental function runs per pair.
use std::{
    cell::RefCell,
    collections::HashMap,
    sync::{Arc, OnceLock},
};

use seqblaze_core::{
    detail::{
        try_reserve_exact, DetailColumns, DetailField, DetailRequest, FieldKind, CANONICAL_NAN,
    },
    kernel::{BoundOutcome, PairDistanceKernel, Score},
    plan::BoundAvailability,
    sched::Workers,
    seq::{ByteView, SeqTable},
    sink::rowsum::{Dyadic, ExactSum},
    validate::EngineError,
    work::WorkSource,
};

use super::{codes, dp::Key, pack::NO_PAIR, DuplexCall, MAX_LENGTH};
use crate::{
    cr::{self, Rational},
    nat::{Int, Nat},
    tm::Conditions,
};

// --------------------------------------------------------------------------- F53 (§3)

/// An F53 value (§3.1): zero, or μ · 2^e with a binary64 μ ∈ [1, 2) and an int64 e.
/// Zero is stored as (+0.0, 0). Every operation is the exact result rounded once to 53
/// significant bits, nearest, ties to even (§3.2), computed with binary64 hardware as
/// §3.3 states: every operand and result is a normal binary64 and nothing is contracted.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct F53 {
    pub mu: f64,
    pub e: i64,
}

/// 2^-d for 0 ≤ d ≤ 53, exactly.
#[inline(always)]
fn pow2_neg(d: i64) -> f64 {
    f64::from_bits(((1023 - d) as u64) << 52)
}

impl F53 {
    pub const ZERO: Self = Self { mu: 0.0, e: 0 };
    pub const ONE: Self = Self { mu: 1.0, e: 0 };

    #[inline(always)]
    pub fn is_zero(self) -> bool {
        self.mu == 0.0
    }

    /// From (M, E) with 2^52 ≤ M < 2^53: M · 2^E.
    pub fn from_parts(m: u64, e: i64) -> Self {
        debug_assert!((1 << 52..1 << 53).contains(&m));
        Self {
            mu: m as f64 * pow2_neg(52),
            e: e + 52,
        }
    }

    /// (M, E) with value M · 2^E, M ∈ [2^52, 2^53); (0, 0) for zero.
    pub fn parts(self) -> (u64, i64) {
        if self.is_zero() {
            (0, 0)
        } else {
            (self.mu.to_bits() & ((1 << 52) - 1) | 1 << 52, self.e - 52)
        }
    }

    /// x ⊗ y = RN₅₃(x · y): the binary64 product of the significands lies in [1, 4) and
    /// is the correctly rounded product; a product ≥ 2 is halved exactly.
    #[inline(always)]
    pub fn times(self, other: Self) -> Self {
        if self.is_zero() || other.is_zero() {
            return Self::ZERO;
        }
        let p = self.mu * other.mu;
        let e = self.e + other.e;
        if p >= 2.0 {
            Self {
                mu: p * 0.5,
                e: e + 1,
            }
        } else {
            Self { mu: p, e }
        }
    }

    /// x ⊕ y = RN₅₃(x + y): with e_x ≥ e_y and d = e_x − e_y, x itself when d ≥ 54 (y is
    /// below half an ulp of x), else RN(μ_x + μ_y · 2^−d) ∈ [1, 4), normalised. Adding
    /// zero is exact.
    #[inline(always)]
    pub fn plus(self, other: Self) -> Self {
        if other.is_zero() {
            return self;
        }
        if self.is_zero() {
            return other;
        }
        let (x, y) = if self.e >= other.e {
            (self, other)
        } else {
            (other, self)
        };
        let d = x.e - y.e;
        if d >= 54 {
            return x;
        }
        let p = x.mu + y.mu * pow2_neg(d);
        if p >= 2.0 {
            Self {
                mu: p * 0.5,
                e: x.e + 1,
            }
        } else {
            Self { mu: p, e: x.e }
        }
    }
}

// --------------------------------------------------------------------------- finalize (§5.3, §7)

/// The call-independent constants of §5.3: T_i = CR64(ln(1 + i/128)), L2 = CR64(ln 2),
/// built once per process by the correctly rounded host logarithm (tm.md §3.3).
struct LogTable {
    t: [f64; 128],
    l2: f64,
}

fn log_table() -> &'static LogTable {
    static TABLE: OnceLock<LogTable> = OnceLock::new();
    TABLE.get_or_init(|| {
        let one = Rational::from_ratio(1, 1);
        let ln = |num: i64, den: u64| {
            cr::ln(&Rational::from_ratio(num, den), &one).expect("ln of a small rational")
        };
        let mut t = [0.0; 128];
        for (i, slot) in t.iter_mut().enumerate() {
            *slot = ln(128 + i as i64, 128);
        }
        LogTable { t, l2: ln(2, 1) }
    })
}

/// RN(2/3), RN(2/5), RN(2/7): one correctly rounded division of exact operands each.
const B3: f64 = 2.0 / 3.0;
const B5: f64 = 2.0 / 5.0;
const B7: f64 = 2.0 / 7.0;

/// g(μ, e) ≈ ln(μ · 2^e) of §7.1, G1–G8, one binary64 operation per step in the order
/// written. μ ∈ [1, 2) and |e| < 2^61 (checked by the callers). Per-pair hot path: it
/// carries no FP guard and expects E0, which its callers provide (guarded scheduler
/// units, the guarded pack build and testing hooks; text-nulls-fp-environment.md §7.3).
pub fn log_map(mu: f64, e: i64) -> f64 {
    let table = log_table();
    let i = ((mu - 1.0) * 128.0) as usize; // G1: exact, then ⌊·⌋
    let c = 1.0 + i as f64 / 128.0; // c_i, exact
    let f = mu - c; // G2: exact (Sterbenz)
    let d = mu + c; // G3
    let s = f / d; // G4
    let s2 = s * s; // G5
    let p5 = B5 + s2 * B7; // G6
    let p3 = B3 + s2 * p5;
    let p1 = 2.0 + s2 * p3;
    let q = s * p1;
    let y = table.t[i] + q; // G7
    if e == 0 {
        y // G8
    } else {
        (e as f64) * table.l2 + y // f64(e) = RN(e)
    }
}

/// The next binary64 above a positive finite x (+∞ stays +∞).
#[inline]
fn next_up_positive(x: f64) -> f64 {
    debug_assert!(x > 0.0);
    if x.is_infinite() {
        x
    } else {
        f64::from_bits(x.to_bits() + 1)
    }
}

/// next_up(x) for x > 0, 0 for x = 0 (§7.3 ru on an already rounded value).
#[inline]
fn ru(x: f64) -> f64 {
    if x == 0.0 {
        0.0
    } else {
        next_up_positive(x)
    }
}

/// P_Λ + 6 and the rounding count c_Z of §7.3; 0 when min(n, m) < 2.
pub fn rounding_count(n: usize, m: usize, loop_max: usize) -> u64 {
    let l = n.min(m) as u64;
    if l < 2 {
        return 0;
    }
    let lam = loop_max as u64;
    let p = (lam + 1) * (lam + 2) / 2;
    (l - 1) * (p + 6) + n as u64 + m as u64 + 1
}

/// The §9.2 order-preserving int64 image of an exposed value (no −0.0, no NaN).
#[inline]
pub fn key_of(x: f64) -> i64 {
    let bits = x.to_bits();
    if bits >> 63 == 0 {
        bits as i64
    } else {
        -((bits & 0x7FFF_FFFF_FFFF_FFFF) as i64)
    }
}

/// The exposed value of a key (the inverse of [`key_of`]).
#[inline]
pub fn value_of(key: i64) -> f64 {
    if key >= 0 {
        f64::from_bits(key as u64)
    } else {
        f64::from_bits(key.unsigned_abs() | 1 << 63)
    }
}

/// The exposed column of a key column (§9.2), in the keys' own buffer: the engine's mask
/// key (`i64::MAX`, a pair that did not pass or a hole) gives the canonical NaN, any
/// other key its [`value_of`]. Both are bit maps between same-layout types, so `collect`
/// reuses the allocation (std's in-place iteration, checked by the unit test below) and
/// the export costs no second buffer.
pub fn values_of(keys: Vec<i64>) -> Vec<f64> {
    keys.into_iter()
        .map(|key| {
            if key == <i64 as Score>::MASK {
                CANONICAL_NAN
            } else {
                value_of(key)
            }
        })
        .collect()
}

/// log₂ of the finalize-range limit on R·T in cal/mol (§5.4): calls with R·T ≥ 2^1000
/// raise ParameterError.
const FINALIZE_RT_LOG2: u64 = 1000;

/// Which model's messages the §5.4 range checks raise: K3's own, or K1's RowStats
/// `lse_dg_kcal` (duplex-mfe.md §8.6.2), whose checks are the same inequalities with
/// K1's bound B. The variant changes only the message text.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RangeMessages {
    /// `duplex(model="pf")` (§5.4).
    Pf,
    /// `duplex(model="mfe")` with `lse_dg_kcal` (duplex-mfe.md §8.6.2, LK3).
    MfeLse,
}

/// Per-call finalize constants and the exact R·T (§5.3, §5.4).
#[derive(Debug, Clone)]
pub struct PfConstants {
    /// R·T in cal/mol, exact.
    pub rt: Rational,
    /// RN(−R·T/1000).
    pub c_t: f64,
    /// RU(R·T/1000).
    pub kappa: f64,
}

fn internal(name: &str, error: cr::PrecisionCap) -> EngineError {
    EngineError::Parameter(format!(
        "internal error: the correctly rounded {name} needs more than {} bits; nothing was \
         computed",
        error.bits
    ))
}

impl PfConstants {
    /// R·T, c_T, κ and the §5.4 range checks (after K1's range check, which built
    /// `call`): ParameterError iff 10^4 · B ≥ 693 · 2^60 · R · T (exponent range), or
    /// iff R · T ≥ 2^1000 cal/mol (finalize range). Runs in E0 whatever the caller's
    /// floating-point environment (text-nulls-fp-environment.md §7.3 item 2).
    pub fn new(call: &DuplexCall, cond: &Conditions) -> Result<Self, EngineError> {
        Self::new_for(call, cond, RangeMessages::Pf)
    }

    /// [`Self::new`] with the range messages of `messages` (duplex-mfe.md §8.6.2 for
    /// K1's `lse_dg_kcal`); the checks and every constant are the same.
    pub fn new_for(
        call: &DuplexCall,
        cond: &Conditions,
        messages: RangeMessages,
    ) -> Result<Self, EngineError> {
        seqblaze_core::fpenv::guarded(|| Self::build(call, cond, messages))
    }

    fn build(
        call: &DuplexCall,
        cond: &Conditions,
        messages: RangeMessages,
    ) -> Result<Self, EngineError> {
        let r = call.pack.gas_r;
        let t = Rational::from_f64(cond.temp_c);
        // T = t + 5463/20 = (20 t_num + 5463 t_den) / (20 t_den) > 0 (T1 §3.1).
        let t_num = t.num.mul_i64(20).add(&Int::new(false, t.den.mul_u64(5463)));
        let t_den = t.den.mul_u64(20);
        let rt = Rational::new(t_num.mul_i64(r.num), t_den.mul_u64(r.den));
        // 10^4 · B · den ≥ 693 · 2^60 · num, both sides non-negative.
        let left = Nat::from_u64(call.bound.unsigned_abs())
            .mul_u64(10_000)
            .mul(&rt.den);
        let right = rt.num.magnitude().mul_u64(693).shl(60);
        if left >= right {
            // B is named, never printed (nndb2024-dna.md §5.3.8).
            return Err(EngineError::Parameter(match messages {
                RangeMessages::Pf => format!(
                    "duplex model='pf': temp_c={} is too close to absolute zero for the \
                     partition function's exponent range: the per-call bound B reaches \
                     10^4·B >= 693·2^60·R·T (docs/wiki/thermo/duplex-pf.md §5.4); use a higher \
                     temp_c",
                    cond.temp_c
                ),
                RangeMessages::MfeLse => format!(
                    "duplex model='mfe' lse_dg_kcal: temp_c={} is too close to absolute zero \
                     for the exponent range of lse_dg_kcal's Boltzmann weights: the per-call \
                     bound B reaches 10^4·B >= 693·2^60·R·T \
                     (docs/wiki/thermo/duplex-mfe.md §8.6.2); use a higher temp_c, or drop \
                     lse_dg_kcal",
                    cond.temp_c
                ),
            }));
        }
        // Finalize range: below 2^1000 cal/mol every dg_ens_kcal, lse_dg_kcal and
        // E1-E4 value stays a finite binary64 (duplex-pf.md §5.4, §7.2, §7.3).
        if rt.num.magnitude() >= &rt.den.shl(FINALIZE_RT_LOG2) {
            return Err(EngineError::Parameter(match messages {
                RangeMessages::Pf => format!(
                    "duplex model='pf': temp_c={:?} is too high for the partition function's \
                     finalize range: R·T reaches 2^1000 cal/mol, beyond which dg_ens_kcal and \
                     its error bound overflow binary64 (docs/wiki/thermo/duplex-pf.md §5.4); \
                     use a lower temp_c",
                    cond.temp_c
                ),
                RangeMessages::MfeLse => format!(
                    "duplex model='mfe' lse_dg_kcal: temp_c={:?} is too high for the finalize \
                     range of lse_dg_kcal: R·T reaches 2^1000 cal/mol, beyond which \
                     lse_dg_kcal overflows binary64 (docs/wiki/thermo/duplex-mfe.md §8.6.2); \
                     use a lower temp_c, or drop lse_dg_kcal",
                    cond.temp_c
                ),
            }));
        }
        let thousand = rt.den.mul_u64(1000);
        let c_t = cr::to_f64(&rt.num.neg(), &thousand);
        let mut kappa = cr::to_f64(&rt.num, &thousand);
        // κ = RU: step up when the nearest binary64 lies below the exact rational.
        let exact = Rational::from_f64(kappa);
        if exact.num.magnitude().mul(&thousand) < rt.num.magnitude().mul(&exact.den) {
            kappa = next_up_positive(kappa);
        }
        Ok(Self { rt, c_t, kappa })
    }

    /// W(D) = RN₅₃(exp(−10·D/(R·T))) (§5.1); W(0) = 1 exactly.
    pub fn weight(&self, d: i64) -> Result<F53, EngineError> {
        if d == 0 {
            return Ok(F53::ONE);
        }
        let x = Rational::new(
            Int::from_i64(d).mul_i64(-10).mul_nat(&self.rt.den),
            self.rt.num.magnitude().clone(),
        );
        let (m, e) = cr::exp53(&x).map_err(|error| internal("weight W(D)", error))?;
        Ok(F53::from_parts(m, e))
    }

    /// `dg_ens_kcal` of §7.2: +∞ for Ẑ = 0, else RN(c_T · g) with −0.0 read as +0.0.
    pub fn dg(&self, z: F53) -> f64 {
        if z.is_zero() {
            return f64::INFINITY;
        }
        let dg = self.c_t * log_map(z.mu, z.e);
        if dg == 0.0 {
            0.0
        } else {
            dg
        }
    }

    /// `dg_ens_err_kcal` of §7.3, E1–E4; 0 for an empty space.
    pub fn err(&self, dg: f64, c_z: u64) -> f64 {
        if dg == f64::INFINITY {
            return 0.0;
        }
        let alpha = ru(self.kappa * (c_z + 12) as f64); // E1: c_Z + 12 < 2^53, exact
        let beta = ru(7.25 * dg.abs()); // E2
        let eps = ru(alpha + beta); // E3
        eps * pow2_neg(53) // E4: exact scaling of a normal value
    }
}

// --------------------------------------------------------------------------- weight tables (§5.2)

/// The weight tables of one call, indexed as K1's cell tables (§5.2, by the pair type P
/// and the class c of nndb2024-dna.md §5.3.1). STATES has no weight table (§5.3.4).
pub(crate) struct Weights {
    pub(crate) loop_max: usize,
    /// N_P, N_C, the pair-type map and the class map of the pack.
    pub(crate) n_p: usize,
    pub(crate) n_c: usize,
    pub(crate) pt: [u8; 16],
    pub(crate) cls: Arc<[u8]>,
    /// START_σ (σ = 0, 1): (α·N_P + P)·5 + β.
    pub(crate) start: [Vec<F53>; 2],
    pub(crate) end: Vec<F53>,
    pub(crate) stack: Vec<F53>,
    pub(crate) bulge: Vec<F53>,
    pub(crate) i11: Vec<F53>,
    pub(crate) i12: Vec<F53>,
    pub(crate) i21: Vec<F53>,
    pub(crate) i22: Vec<F53>,
    pub(crate) gen: Vec<F53>,
    pub(crate) loop_out: Vec<F53>,
    pub(crate) loop_in: Vec<F53>,
}

/// Tables of the K3 device pack: START_0, START_1, END, WSTACK, WBULGE, WI11, WI12,
/// WI21, WI22, WGEN, WOUT, WIN.
pub const PF_DEVICE_TABLES: usize = 12;

/// The K3 device pack of one call (duplex-pf-cuda.md §4): Λ, N_P, N_C and the dense
/// CLS map (nndb2024-dna.md §5.4.2), each table's start in the concatenated entries
/// (then the total), and every entry as (M, e) with value M · 2^(e − 52), zero as
/// (0, 0). Tables are indexed by the pair type P and the loop class c (§5.3.4); there
/// is no STATES table (K3 never reads it).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PfDevicePack {
    pub loop_max: usize,
    /// N_P (4 or 6) and N_C (1 or 3).
    pub n_p: usize,
    pub n_c: usize,
    /// CLS dense over (Λ + 1)²: c at u·(Λ + 1) + v, 0 outside the GEN shapes.
    pub cls: Vec<u8>,
    pub offsets: [usize; PF_DEVICE_TABLES + 1],
    pub m: Vec<u64>,
    pub e: Vec<i64>,
}

/// One call's K3 pack: the K1 call, the constants and the weight tables.
pub struct PfCall {
    pub call: Arc<DuplexCall>,
    pub constants: PfConstants,
    pub(crate) weights: Weights,
}

impl PfCall {
    /// The §10 pack step for `model="pf"` after K1's: the §5.4 exponent check, then
    /// the weight tables (every reservation fallible: DeviceError on refusal).
    pub fn new(call: Arc<DuplexCall>, cond: &Conditions) -> Result<Self, EngineError> {
        // E0 for the pack build (text-nulls-fp-environment.md §7.3 item 2).
        seqblaze_core::fpenv::guarded(|| Self::build(call, cond))
    }

    fn build(call: Arc<DuplexCall>, cond: &Conditions) -> Result<Self, EngineError> {
        let constants = PfConstants::new(&call, cond)?;
        let cells = &call.score;
        let mut memo: HashMap<i64, F53> = HashMap::new();
        let mut w = |d: i64| -> Result<F53, EngineError> {
            if !d.finite() {
                return Ok(F53::ZERO); // ⊥
            }
            if let Some(&v) = memo.get(&d) {
                return Ok(v);
            }
            let v = constants.weight(d)?;
            memo.insert(d, v);
            Ok(v)
        };
        let mut table = |keys: &[i64]| -> Result<Vec<F53>, EngineError> {
            let mut out = Vec::new();
            try_reserve_exact(&mut out, keys.len(), "duplex pf weight table")?;
            for &k in keys {
                out.push(w(k)?);
            }
            Ok(out)
        };
        let start = |init: i64| -> Vec<i64> {
            cells
                .end_l
                .iter()
                .map(|&e| {
                    if init.finite() && e.finite() {
                        init + e
                    } else {
                        i64::INF
                    }
                })
                .collect()
        };
        let weights = Weights {
            loop_max: cells.loop_max,
            n_p: cells.n_p,
            n_c: cells.n_c,
            pt: cells.pt,
            cls: Arc::clone(&cells.cls),
            start: [table(&start(cells.init))?, table(&start(cells.init_sym))?],
            end: table(&cells.end_r)?,
            stack: table(&cells.stack)?,
            bulge: table(&cells.bulge)?,
            i11: table(&cells.i11)?,
            i12: table(&cells.i12)?,
            i21: table(&cells.i21)?,
            i22: table(&cells.i22)?,
            gen: table(&cells.gen)?,
            loop_out: table(&cells.loop_out)?,
            loop_in: table(&cells.loop_in)?,
        };
        Ok(Self {
            call,
            constants,
            weights,
        })
    }

    pub fn loop_max(&self) -> usize {
        self.weights.loop_max
    }

    /// The K3 device pack (duplex-pf-cuda.md §4, QKP4): Λ and the weight tables in the
    /// flat layouts of the CPU's tables, concatenated in the order START (σ = 0 then 1),
    /// END, WSTACK, WBULGE, WI11, WI12, WI21, WI22, WGEN, WOUT, WIN. Each entry is the
    /// F53 value as (M, e) with value M · 2^(e − 52): M = (bits(μ) & (2^52 − 1)) | 2^52
    /// and the CPU's e for a nonzero (μ, e), (0, 0) for zero. Bit operations only, no
    /// floating-point operation, so it carries no FP guard (§9); the buffers are
    /// reserved fallibly (DeviceError on refusal, D86).
    pub fn device_pack(&self) -> Result<PfDevicePack, EngineError> {
        let w = &self.weights;
        debug_assert_eq!(
            w.pt,
            super::pack::pair_map(w.n_p == 6),
            "the device derives PT"
        );
        let tables: [&[F53]; PF_DEVICE_TABLES] = [
            &w.start[0],
            &w.start[1],
            &w.end,
            &w.stack,
            &w.bulge,
            &w.i11,
            &w.i12,
            &w.i21,
            &w.i22,
            &w.gen,
            &w.loop_out,
            &w.loop_in,
        ];
        let mut offsets = [0usize; PF_DEVICE_TABLES + 1];
        for (k, table) in tables.iter().enumerate() {
            offsets[k + 1] = offsets[k] + table.len();
        }
        let total = offsets[PF_DEVICE_TABLES];
        let mut m = Vec::new();
        let mut e = Vec::new();
        let mut cls = Vec::new();
        try_reserve_exact(&mut m, total, "duplex pf device pack")?;
        try_reserve_exact(&mut e, total, "duplex pf device pack")?;
        try_reserve_exact(&mut cls, w.cls.len(), "duplex pf device pack")?;
        cls.extend_from_slice(&w.cls);
        for value in tables.iter().flat_map(|table| table.iter()) {
            let bits = value.mu.to_bits();
            if bits == 0 {
                m.push(0);
                e.push(0);
            } else {
                m.push(bits & ((1 << 52) - 1) | 1 << 52);
                e.push(value.e);
            }
        }
        Ok(PfDevicePack {
            loop_max: w.loop_max,
            n_p: w.n_p,
            n_c: w.n_c,
            cls,
            offsets,
            m,
            e,
        })
    }

    /// Ẑ of one pair of letter codes, in canonical orientation (§6.1, §6.2).
    pub fn zhat_codes(&self, a: &[u8], b: &[u8]) -> Result<F53, EngineError> {
        // Codes keep the byte order (A < C < G < T); slices compare lexicographically
        // with a proper prefix smaller, so this is the bytewise order of §6.1.
        let (x, y) = if a <= b { (a, b) } else { (b, a) };
        PF_SCRATCH.with(|cell| zhat(&self.weights, x, y, &mut cell.borrow_mut()))
    }

    /// (Ẑ, dg, err) of one pair of letter codes.
    pub fn pair_codes(&self, a: &[u8], b: &[u8]) -> Result<(F53, f64, f64), EngineError> {
        let z = self.zhat_codes(a, b)?;
        let dg = self.constants.dg(z);
        let c_z = rounding_count(a.len(), b.len(), self.loop_max());
        Ok((z, dg, self.constants.err(dg, c_z)))
    }
}

// --------------------------------------------------------------------------- recurrence (§6.2)

/// Per-thread scratch: ZA and the N_C values ZO_c for Λ + 2 rows, and the column
/// accumulators E(j).
#[derive(Default)]
pub(crate) struct PfScratch {
    za: Vec<F53>,
    zo: Vec<F53>,
    col: Vec<F53>,
}

thread_local! {
    static PF_SCRATCH: RefCell<PfScratch> = RefCell::new(PfScratch::default());
}

/// Buffers above this size are released after a pair.
const KEEP_BYTES: usize = 1 << 20;

impl PfScratch {
    fn prepare(&mut self, band: usize, classes: usize, m: usize) -> Result<(), EngineError> {
        for (buffer, len) in [
            (&mut self.za, band),
            (&mut self.zo, band * classes),
            (&mut self.col, m),
        ] {
            buffer.clear();
            try_reserve_exact(buffer, len, "duplex pf DP band")?;
            buffer.resize(len, F53::ZERO);
        }
        Ok(())
    }

    fn release(&mut self) {
        for buffer in [&mut self.za, &mut self.zo, &mut self.col] {
            if buffer.capacity() * std::mem::size_of::<F53>() > KEEP_BYTES {
                *buffer = Vec::new();
            }
        }
    }
}

/// identical(a, b) and a palindrome (K1 §3.4, QK12; QP1): σ of START.
pub(crate) fn sigma(a: &[u8], b: &[u8]) -> usize {
    let n = a.len();
    usize::from(a == b && (0..n).all(|k| a[k] + a[n - 1 - k] == 3))
}

/// The largest number of loop classes (nndb2024-dna.md §4.4: N_C is 1 or 3).
pub(crate) const MAX_CLASSES: usize = 3;

/// Ẑ of §6.2 for the canonical top strand `a` and bottom strand `b` (codes 0..4), with
/// pair(i, j) iff PT[a_i][b_j] ≥ 0 and the generic part split by class
/// (nndb2024-dna.md §5.2 item 2, §5.3.4). Fixed orders: inside a cell the special terms
/// in order O (u ascending, then v ascending), the generic terms of each class c in order
/// O into ZG_c, and the terms ZG_c ⊗ WIN_c last, classes c = 0 … N_C − 1 in grammar
/// order (`1n`, `23`, `other`); E(j) folds its rows in ascending order; Ẑ folds the
/// columns in ascending order. Zero terms are skipped (adding 0 is exact). With N_C = 1
/// this is §6.2 exactly.
fn zhat(w: &Weights, a: &[u8], b: &[u8], scratch: &mut PfScratch) -> Result<F53, EngineError> {
    let (n, m) = (a.len(), b.len());
    if n < 2 || m < 2 {
        return Ok(F53::ZERO); // k ≥ 2 pairs (QK8)
    }
    let lam = w.loop_max;
    let (np, nc) = (w.n_p, w.n_c);
    debug_assert!(nc <= MAX_CLASSES);
    let rows = n.min(lam + 2);
    scratch.prepare(rows * m, nc, m)?;
    let start = &w.start[sigma(a, b)];
    let width = lam + 1;
    let mut cur = 0;
    for i in 0..n {
        let ai = usize::from(a[i]);
        let base = cur * m;
        for j in 0..m {
            let pi = w.pt[ai * 4 + usize::from(b[j])];
            if pi == NO_PAIR {
                scratch.za[base + j] = F53::ZERO;
                scratch.zo[(base + j) * nc..(base + j + 1) * nc].fill(F53::ZERO);
                continue;
            }
            let pi = usize::from(pi);
            let mut special = F53::ZERO;
            if i >= 1 && j + 1 < m {
                let mut generic = [F53::ZERO; MAX_CLASSES];
                let umax = lam.min(i - 1);
                // Ring slot of k = i - 1 - u, walked backwards without division.
                let mut slot = if cur == 0 { rows - 1 } else { cur - 1 };
                for u in 0..=umax {
                    let k = i - 1 - u;
                    let ak = usize::from(a[k]) * 4;
                    let row = slot * m;
                    let vmax = (lam - u).min(m - 2 - j);
                    // v = 0: a stack or a top bulge. A nonzero ZA(k, l) implies that
                    // (k, l) is a pair, so P_kl is read only then.
                    let prev = scratch.za[row + j + 1];
                    if !prev.is_zero() {
                        let pk = usize::from(w.pt[ak + usize::from(b[j + 1])]);
                        let weight = if u == 0 {
                            w.stack[pk * np + pi]
                        } else {
                            w.bulge[((u - 1) * np + pk) * np + pi]
                        };
                        special = special.plus(prev.times(weight));
                    }
                    for v in 1..=vmax {
                        let l = j + 1 + v;
                        match u {
                            0 => {
                                let prev = scratch.za[row + l];
                                if !prev.is_zero() {
                                    let pk = usize::from(w.pt[ak + usize::from(b[l])]);
                                    let weight = w.bulge[((lam + v - 1) * np + pk) * np + pi];
                                    special = special.plus(prev.times(weight));
                                }
                            }
                            1 | 2 if v <= 2 => {
                                let prev = scratch.za[row + l];
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
                                let cl = usize::from(w.cls[u * width + v]);
                                let prev = scratch.zo[(row + l) * nc + cl];
                                if !prev.is_zero() {
                                    generic[cl] =
                                        generic[cl].plus(prev.times(w.gen[u * width + v]));
                                }
                            }
                        }
                    }
                    slot = if slot == 0 { rows - 1 } else { slot - 1 };
                }
                // The WIN terms are the last terms of ZC, classes in grammar order.
                let in_base = usize::from(a[i - 1]) * 4 + usize::from(b[j + 1]);
                for (cl, zg) in generic.iter().enumerate().take(nc) {
                    let inner = w.loop_in[(cl * 16 + in_base) * np + pi];
                    special = special.plus(zg.times(inner));
                }
            }
            let alpha = if i >= 1 { usize::from(a[i - 1]) } else { 4 };
            let beta = if j + 1 < m { usize::from(b[j + 1]) } else { 4 };
            let za = start[(alpha * np + pi) * 5 + beta].plus(special);
            scratch.za[base + j] = za;
            let zo = &mut scratch.zo[(base + j) * nc..(base + j + 1) * nc];
            if i + 1 < n && j >= 1 {
                // ZO_c(i, j) = ZA(i, j) ⊗ WOUT[c][P_ij][a_{i+1}][b_{j−1}].
                let tail = usize::from(a[i + 1]) * 4 + usize::from(b[j - 1]);
                for (cl, slot) in zo.iter_mut().enumerate() {
                    *slot = za.times(w.loop_out[(cl * np + pi) * 16 + tail]);
                }
            } else {
                zo.fill(F53::ZERO);
            }
            if !special.is_zero() {
                let alpha = if i + 1 < n { usize::from(a[i + 1]) } else { 4 };
                let beta = if j >= 1 { usize::from(b[j - 1]) } else { 4 };
                let end = w.end[(pi * 5 + alpha) * 5 + beta];
                scratch.col[j] = scratch.col[j].plus(special.times(end));
            }
        }
        cur += 1;
        if cur == rows {
            cur = 0;
        }
    }
    let total = scratch.col.iter().fold(F53::ZERO, |acc, &e| acc.plus(e));
    scratch.release();
    Ok(total)
}

/// Letter codes of validated ASCII ACGT bytes.
pub fn codes_of(bytes: &[u8]) -> Vec<u8> {
    bytes
        .iter()
        .map(|&c| crate::tm::CODE[usize::from(c)])
        .collect()
}

/// The testing hooks' sequence check (§14.4): upper-case ACGT, at most 200 nt.
pub fn check_strand(input: &'static str, bytes: &[u8]) -> Result<(), EngineError> {
    if let Some(position) = bytes.iter().position(|c| !b"ACGT".contains(c)) {
        let byte = bytes[position];
        return Err(EngineError::Sequence {
            message: format!(
                "duplex input {input} sequence 0 has an illegal byte {:?} at position {position}; \
                 duplex accepts upper-case A, C, G, T only (D42); nothing was computed",
                char::from(byte)
            ),
            input,
            index: 0,
            position: Some(position),
            byte: Some(byte),
            n_bad: 1,
            bad_indices: vec![0],
        });
    }
    if bytes.len() > MAX_LENGTH {
        return Err(EngineError::Sequence {
            message: format!(
                "duplex input {input} sequence 0 has length {}, above the native thermodynamics \
                 limit {MAX_LENGTH} nt (D48); nothing was computed",
                bytes.len()
            ),
            input,
            index: 0,
            position: None,
            byte: None,
            n_bad: 1,
            bad_indices: vec![0],
        });
    }
    Ok(())
}

// --------------------------------------------------------------------------- kernel (§9, §14.1)

/// The detail field of §9.1 besides the primary (`dg_ens_kcal` comes from the key).
pub const DG_ENS_ERR_KCAL: usize = 0;

pub static FIELDS: [DetailField; 1] = [DetailField {
    name: "dg_ens_err_kcal",
    kind: FieldKind::Float64,
}];

/// The K3 pair kernel of one call: the per-call pack, the requested detail field and
/// whether RowStats sums the pairs' Ẑ for `lse_dg_kcal` (§9.4). The selecting key is
/// the int64 image of the exposed `dg_ens_kcal` ([`key_of`]).
pub struct DuplexPf {
    call: Arc<PfCall>,
    request: DetailRequest,
    lse: bool,
    /// The call's passing C2 certificate (duplex-pf-bound.md §5), if any.
    bound: Option<Arc<super::pf_bound::Certificate>>,
}

impl DuplexPf {
    /// `fields` by name, of [`FIELDS`]; `lse` requests the RowStats row sums.
    pub fn new(call: Arc<PfCall>, fields: &[&str], lse: bool) -> Result<Self, EngineError> {
        let mut request = DetailRequest::NONE;
        for name in fields {
            let index = FIELDS
                .iter()
                .position(|field| field.name == *name)
                .ok_or_else(|| {
                    EngineError::Parameter(format!(
                        "duplex model='pf' has no detail field {name:?}"
                    ))
                })?;
            request = request.with(index);
        }
        Ok(Self {
            call,
            request,
            lse,
            bound: None,
        })
    }

    /// The C2 bound of this call (duplex-pf-bound.md §5, §6): kept only when its
    /// certificate passes, so a failed certificate leaves every path as before.
    pub fn with_bound(mut self, certificate: Arc<super::pf_bound::Certificate>) -> Self {
        self.bound = certificate.pass.then_some(certificate);
        self
    }

    /// The call's passing C2 certificate when the outcome bound applies (no
    /// `lse_dg_kcal` request; duplex-pf-bound.md §6.1), for the device path.
    pub fn certificate(&self) -> Option<&Arc<super::pf_bound::Certificate>> {
        self.bound.as_ref().filter(|_| !self.lse)
    }

    pub fn call(&self) -> &PfCall {
        &self.call
    }

    fn with_codes<R>(a: ByteView<'_>, b: ByteView<'_>, body: impl FnOnce(&[u8], &[u8]) -> R) -> R {
        let (mut x, mut y) = ([0u8; MAX_LENGTH], [0u8; MAX_LENGTH]);
        let n = codes(a, &mut x);
        let m = codes(b, &mut y);
        body(&x[..n], &y[..m])
    }

    /// (Ẑ_b, L, outcome) of the C2 bound for one pair against θ (duplex-pf-bound.md §1,
    /// §10.2; testing hooks and measurements): `None` without a passing certificate.
    pub fn bound_values(
        &self,
        a: ByteView<'_>,
        b: ByteView<'_>,
        threshold: Option<i64>,
    ) -> Result<Option<(F53, f64, BoundOutcome<i64>)>, EngineError> {
        let Some(certificate) = self.bound.as_ref() else {
            return Ok(None);
        };
        Self::with_codes(a, b, |a, b| {
            certificate.evaluate(&self.call, a, b, threshold)
        })
        .map(Some)
    }

    /// Ẑ and the key of a pair of letter codes; the key through [`Self::term_key`], the
    /// one map from Ẑ to the key that the device path uses too (QKP3).
    fn solve(&self, a: &[u8], b: &[u8]) -> Result<(F53, i64), EngineError> {
        let z = self.call.zhat_codes(a, b)?;
        let (m, e) = z.parts();
        Ok((z, self.term_key(Dyadic { m, e })?))
    }

    /// The key of Ẑ = M · 2^E (duplex-pf-cuda.md §7.1): key_of(dg(Ẑ)), +∞'s key for
    /// M = 0. Ẑ is (0, 0) or has 2^52 ≤ M < 2^53 and |E + 52| < 2^60 + 403 (§5.4);
    /// anything else is an internal error. Per-pair hot path in E0 (its callers' guards).
    fn term_key(&self, term: Dyadic) -> Result<i64, EngineError> {
        let z = if term.m == 0 && term.e == 0 {
            F53::ZERO
        } else if (1 << 52..1 << 53).contains(&term.m)
            && term
                .e
                .checked_add(52)
                .is_some_and(|e| e.unsigned_abs() < (1 << 60) + 403)
        {
            F53::from_parts(term.m, term.e)
        } else {
            return Err(EngineError::Device(format!(
                "internal error: duplex model='pf' received an invalid partition-function \
                 value (M = {}, E = {}); no result was published",
                term.m, term.e
            )));
        };
        Ok(key_of(self.call.constants.dg(z)))
    }

    /// `dg_ens_err_kcal` of a pair from its key and lengths (§7.3: no recurrence).
    fn push(
        &self,
        n: usize,
        m: usize,
        key: i64,
        out: &mut DetailColumns,
    ) -> Result<(), EngineError> {
        if out.wants(DG_ENS_ERR_KCAL) {
            let c_z = rounding_count(n, m, self.call.loop_max());
            out.push_f64(DG_ENS_ERR_KCAL, self.call.constants.err(value_of(key), c_z))?;
        }
        Ok(())
    }
}

impl PairDistanceKernel<i64> for DuplexPf {
    /// The C2 bound (§9.6; duplex-pf-bound.md §6.1): available with a passing certificate
    /// unless RowStats `lse_dg_kcal` was requested (QZ7).
    fn bound_availability(&self) -> BoundAvailability {
        if self.outcome_bound() {
            BoundAvailability::Available
        } else {
            BoundAvailability::Unavailable
        }
    }
    fn outcome_bound(&self) -> bool {
        self.bound.is_some() && !self.lse
    }
    /// Pruned (key(L) > θ), certified with key(+∞) (Ẑ_b = 0) or open
    /// (duplex-pf-bound.md §1), in canonical orientation.
    fn bound_outcome(
        &self,
        a: ByteView<'_>,
        b: ByteView<'_>,
        threshold: Option<i64>,
    ) -> Result<BoundOutcome<i64>, EngineError> {
        let Some(certificate) = self.bound.as_ref() else {
            return Ok(BoundOutcome::Open);
        };
        seqblaze_core::plan::count_bound_eval();
        Self::with_codes(a, b, |a, b| {
            certificate
                .evaluate(&self.call, a, b, threshold)
                .map(|(_, _, outcome)| outcome)
        })
    }
    /// (e, Ĉ_pf) of duplex-pf-bound.md §6.6, from the pack's Λ and format.
    fn bound_cost(&self, n_a: usize, n_b: usize) -> (u64, u32) {
        let call = &self.call.call;
        super::pf_bound::cost(call.pack.format, call.pack.loop_max, n_a, n_b)
    }
    fn name(&self) -> &'static str {
        "duplex"
    }
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
    fn length_error(
        &self,
        input: &'static str,
        index: usize,
        length: usize,
        limit: usize,
    ) -> EngineError {
        EngineError::Sequence {
            message: format!(
                "duplex input {input} sequence {index} has length {length}, above the native \
                 thermodynamics limit {limit} nt (D48); nothing was computed"
            ),
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
        _a: &SeqTable,
        _b: &SeqTable,
        _workers: Workers,
    ) -> Result<(), EngineError> {
        Ok(())
    }
    fn lower_bound(&self, _a: &SeqTable, _b: &SeqTable, _i: usize, _j: usize) -> i64 {
        i64::MIN
    }
    /// n · m · (Λ + 1)(Λ + 2)/2 candidate products, about a quarter of them reached.
    fn pair_cost(&self, a: &SeqTable, b: &SeqTable, _with_details: bool) -> usize {
        let lam = self.call.loop_max();
        let cost = a
            .mean_len()
            .max(1)
            .saturating_mul(b.mean_len().max(1))
            .saturating_mul((lam + 1) * (lam + 2) / 2)
            / 4;
        cost.max(1)
    }
    /// The exact key; never `None` (no pruning, §9.6).
    fn evaluate(
        &self,
        a: ByteView<'_>,
        b: ByteView<'_>,
        _cutoff: Option<i64>,
    ) -> Result<Option<i64>, EngineError> {
        Self::with_codes(a, b, |a, b| self.solve(a, b).map(|(_, key)| Some(key)))
    }
    fn detail_fields(&self) -> &'static [DetailField] {
        &FIELDS
    }
    fn detail_request(&self) -> DetailRequest {
        self.request
    }
    fn evaluate_detailed(
        &self,
        a: ByteView<'_>,
        b: ByteView<'_>,
        _prune: Option<i64>,
        keep: Option<i64>,
        out: &mut DetailColumns,
    ) -> Result<Option<i64>, EngineError> {
        let (n, m) = (a.len(), b.len());
        let key = Self::with_codes(a, b, |a, b| self.solve(a, b))?.1;
        if keep.is_none_or(|limit| key <= limit) {
            self.push(n, m, key, out)?;
        }
        Ok(Some(key))
    }
    fn details(
        &self,
        a: ByteView<'_>,
        b: ByteView<'_>,
        score: i64,
        out: &mut DetailColumns,
    ) -> Result<(), EngineError> {
        self.push(a.len(), b.len(), score, out)
    }
    fn row_terms(&self) -> bool {
        self.lse
    }
    /// The key and Ẑ = M · 2^E (§9.4).
    fn evaluate_term(
        &self,
        a: ByteView<'_>,
        b: ByteView<'_>,
    ) -> Result<(i64, Dyadic), EngineError> {
        let (z, key) = Self::with_codes(a, b, |a, b| self.solve(a, b))?;
        let (m, e) = z.parts();
        Ok((key, Dyadic { m, e }))
    }
    /// `lse_dg_kcal` of §9.4: +∞ when the exact sum is 0, else RN(c_T · g(RN₅₃(S))).
    fn row_value(&self, sum: &ExactSum) -> f64 {
        match sum.round53() {
            None => f64::INFINITY,
            Some((m, e)) => self.call.constants.dg(F53::from_parts(m, e)),
        }
    }
    /// The key of Ẑ = M · 2^E, as `evaluate` computes it (duplex-pf-cuda.md §7.1, QKP3).
    fn key_of_term(&self, term: Dyadic) -> Result<i64, EngineError> {
        self.term_key(term)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use seqblaze_core::{
        batch::SeqBatch,
        exec::{self, ExecOptions, Grain, Strategy},
        kernel::{CpuBatch, PairDistanceKernel},
        seq::SeqTable,
        sink::{rowsum::Dyadic, RunOutput, Sink},
        validate::ValidationReport,
        work::{IndexArray, PairKind, WorkSource},
    };

    use super::{key_of, values_of, DuplexPf, PfCall, F53, PF_DEVICE_TABLES};
    use crate::duplex::{
        pack::Pack,
        tests::{cond, pack_text, pack_text2_with},
        DuplexCall,
    };

    fn pf_call(lam: usize, seed: u64) -> PfCall {
        let pack = Pack::parse(&pack_text(lam, seed, 1)).expect("the test pack loads");
        let call = DuplexCall::new(Arc::new(pack), &cond()).expect("the test pack is in range");
        PfCall::new(Arc::new(call), &cond()).expect("the test pack passes the K3 checks")
    }

    /// `PfCall::device_pack` reproduces the CPU's weight tables bit for bit
    /// (duplex-pf-cuda.md §4, §11 item 14): the table order and sizes of §4, M =
    /// (bits(μ) & (2^52 − 1)) | 2^52 with the CPU's e for a nonzero entry, (0, 0) for
    /// zero, at Λ = 0 (no WBULGE entry, one WGEN entry), 30 and the recipe maximum 396.
    #[test]
    fn device_pack_reproduces_the_cpu_weights_bit_for_bit() {
        for (lam, seed) in [(0usize, 11u64), (30, 12), (396, 13)] {
            let call = pf_call(lam, seed);
            let pack = call.device_pack().unwrap();
            let w = &call.weights;
            let tables: [&[F53]; PF_DEVICE_TABLES] = [
                &w.start[0],
                &w.start[1],
                &w.end,
                &w.stack,
                &w.bulge,
                &w.i11,
                &w.i12,
                &w.i21,
                &w.i22,
                &w.gen,
                &w.loop_out,
                &w.loop_in,
            ];
            let sizes = [
                100,
                100,
                100,
                16,
                32 * lam,
                256,
                1024,
                1024,
                4096,
                (lam + 1) * (lam + 1),
                64,
                64,
            ];
            assert_eq!(pack.loop_max, lam);
            assert_eq!(pack.offsets[0], 0);
            for (k, table) in tables.iter().enumerate() {
                assert_eq!(table.len(), sizes[k], "Λ={lam} table {k}");
                assert_eq!(pack.offsets[k + 1] - pack.offsets[k], sizes[k]);
                for (x, value) in table.iter().enumerate() {
                    let at = pack.offsets[k] + x;
                    let (m, e) = (pack.m[at], pack.e[at]);
                    let bits = value.mu.to_bits();
                    if bits == 0 {
                        assert_eq!((m, e, value.e), (0, 0, 0), "Λ={lam} table {k} entry {x}");
                    } else {
                        assert!((1 << 52..1 << 53).contains(&m));
                        assert_eq!(0x3FF0_0000_0000_0000 | (m & ((1 << 52) - 1)), bits);
                        assert_eq!(e, value.e, "Λ={lam} table {k} entry {x}");
                    }
                }
            }
            let total = pack.offsets[PF_DEVICE_TABLES];
            assert_eq!(total, 6844 + 32 * lam + (lam + 1) * (lam + 1));
            assert_eq!((pack.m.len(), pack.e.len()), (total, total));
            // A few entries recomputed from the cell keys: W(D) of a WSTACK cell.
            for x in 0..16 {
                let d = call.call.score.stack[x];
                let expected = if super::Key::finite(d) {
                    call.constants.weight(d).unwrap()
                } else {
                    F53::ZERO
                };
                let at = pack.offsets[3] + x;
                let got = if pack.m[at] == 0 {
                    F53::ZERO
                } else {
                    F53::from_parts(pack.m[at], pack.e[at] - 52)
                };
                assert_eq!(got.mu.to_bits(), expected.mu.to_bits());
                assert_eq!(got.e, expected.e);
            }
        }
    }

    /// nndb2024-dna.md §5.4.11 R1, K3 part: on the format-2 test pack (N_P = 6, N_C = 3,
    /// with and without ⊥ cells) the K3 device pack has §5.4.2's 13 offsets (START_0,
    /// START_1 and END 25·N_P each, START_1 at START_0 + 25·N_P, WOUT and WIN 16·N_C·N_P),
    /// every (M, e) equals W of the CPU's score key of its cell (START from INIT or
    /// INIT + SYM plus END_L; ⊥ as zero), and N_P, N_C and CLS are carried.
    #[test]
    fn device_pack_reproduces_the_cpu_weights_on_format_2() {
        for (lam, seed, bottom) in [
            (0usize, 71u64, false),
            (3, 72, true),
            (30, 73, false),
            (30, 74, true),
        ] {
            let text = pack_text2_with(lam, seed, 1, bottom);
            let pack = Pack::parse(&text).expect("the format-2 test pack loads");
            let call = DuplexCall::new(Arc::new(pack), &cond()).expect("in range");
            let call = PfCall::new(Arc::new(call), &cond()).expect("K3 call");
            let dp = call.device_pack().unwrap();
            let (np, nc) = (6usize, 3usize);
            assert_eq!((dp.n_p, dp.n_c, dp.loop_max), (np, nc, lam));
            assert_eq!(&dp.cls[..], &call.call.pack.cls[..]);
            let sizes = [
                25 * np,
                25 * np,
                25 * np,
                np * np,
                2 * lam * np * np,
                16 * np * np,
                64 * np * np,
                64 * np * np,
                256 * np * np,
                (lam + 1) * (lam + 1),
                16 * nc * np,
                16 * nc * np,
            ];
            assert_eq!(dp.offsets[0], 0);
            for (k, size) in sizes.iter().enumerate() {
                assert_eq!(
                    dp.offsets[k + 1] - dp.offsets[k],
                    *size,
                    "Λ={lam} table {k}"
                );
            }
            assert_eq!(dp.offsets[1], 25 * np, "START_1 at START_0 + 25·N_P");
            let total = dp.offsets[PF_DEVICE_TABLES];
            assert_eq!((dp.m.len(), dp.e.len()), (total, total));
            let c = &call.call.score;
            let start = |init: i64| -> Vec<i64> {
                c.end_l
                    .iter()
                    .map(|&e| {
                        if super::Key::finite(init) && super::Key::finite(e) {
                            init + e
                        } else {
                            <i64 as super::Key>::INF
                        }
                    })
                    .collect()
            };
            let keys: [Vec<i64>; PF_DEVICE_TABLES] = [
                start(c.init),
                start(c.init_sym),
                c.end_r.clone(),
                c.stack.clone(),
                c.bulge.clone(),
                c.i11.clone(),
                c.i12.clone(),
                c.i21.clone(),
                c.i22.clone(),
                c.gen.clone(),
                c.loop_out.clone(),
                c.loop_in.clone(),
            ];
            let mut zeros = 0usize;
            for (k, table) in keys.iter().enumerate() {
                assert_eq!(table.len(), sizes[k]);
                for (x, &d) in table.iter().enumerate() {
                    let want = if super::Key::finite(d) {
                        call.constants.weight(d).unwrap()
                    } else {
                        F53::ZERO
                    };
                    let at = dp.offsets[k] + x;
                    if want.mu.to_bits() == 0 {
                        // GEN's non-shape entries are zero in every pack.
                        zeros += usize::from(k != 9);
                        assert_eq!((dp.m[at], dp.e[at]), (0, 0), "Λ={lam} table {k} entry {x}");
                    } else {
                        let got = F53::from_parts(dp.m[at], dp.e[at] - 52);
                        assert_eq!(
                            (got.mu.to_bits(), got.e),
                            (want.mu.to_bits(), want.e),
                            "Λ={lam} table {k} entry {x}"
                        );
                    }
                }
            }
            assert_eq!(
                bottom,
                zeros > 0,
                "Λ={lam}: zero weights exactly with ⊥ cells"
            );
        }
    }

    /// Strands as letter strings over ACGT from a fixed generator.
    fn strands(count: usize, seed: u64, lengths: std::ops::RangeInclusive<usize>) -> Vec<Vec<u8>> {
        let mut state = seed | 1;
        let mut next = move || {
            state ^= state >> 12;
            state ^= state << 25;
            state ^= state >> 27;
            state.wrapping_mul(0x2545_F491_4F6C_DD1D)
        };
        let span = (lengths.end() - lengths.start() + 1) as u64;
        (0..count)
            .map(|_| {
                let n = lengths.start() + (next() % span) as usize;
                (0..n).map(|_| b"ACGT"[(next() % 4) as usize]).collect()
            })
            .collect()
    }

    /// The term-to-key method equals `evaluate`'s key on random pairs (§11 item 14), and
    /// rejects a non-canonical term.
    #[test]
    fn term_to_key_equals_the_evaluated_key() {
        let call = Arc::new(pf_call(30, 21));
        let kernel = DuplexPf::new(Arc::clone(&call), &[], false).unwrap();
        let rows = strands(60, 5, 0..=24);
        let table = SeqTable::new(SeqBatch::from_seqs(&rows));
        let mut nonzero = 0;
        for i in 0..rows.len() {
            for j in [i, (i * 7 + 3) % rows.len()] {
                let (a, b) = (table.get(i), table.get(j));
                let key = kernel.evaluate(a, b, None).unwrap().unwrap();
                let (term_key, term) = kernel.evaluate_term(a, b).unwrap();
                assert_eq!(term_key, key);
                assert_eq!(kernel.key_of_term(term).unwrap(), key, "pair ({i}, {j})");
                nonzero += usize::from(term.m != 0);
            }
        }
        assert!(nonzero > 10, "the pairs reach nonzero partition functions");
        for bad in [
            Dyadic { m: 0, e: 1 },
            Dyadic { m: 1 << 53, e: 0 },
            Dyadic {
                m: (1 << 52) - 1,
                e: 0,
            },
            Dyadic {
                m: 1 << 52,
                e: (1 << 60) + 403,
            },
        ] {
            assert!(kernel.key_of_term(bad).is_err(), "{bad:?}");
        }
    }

    fn published(output: &RunOutput<i64>) -> Vec<String> {
        let c = &output.columns;
        let mut out = vec![
            format!("dist {:?}", c.dist),
            format!("passed {:?}", c.passed),
            format!("i {:?} j {:?}", c.i, c.j),
            format!(
                "details {:?}",
                c.details.as_ref().map(|details| details
                    .columns()
                    .iter()
                    .map(|column| format!("{:?}", column.data))
                    .collect::<Vec<_>>())
            ),
            format!(
                "counts {} {} {}",
                output.counts.certified, output.counts.pruned, output.counts.full
            ),
        ];
        if let Some(stats) = &c.rowstats {
            out.push(format!(
                "rowstats {:?} {:?} {:?} {:?} {:?}",
                stats.best_dist,
                stats.argbest,
                stats.n_best,
                stats.count,
                stats
                    .row_value
                    .as_ref()
                    .map(|values| values.iter().map(|v| v.to_bits()).collect::<Vec<_>>())
            ));
        }
        out
    }

    /// The core batch driver over `CpuBatch` equals `exec::compute` on K3 for Each, Hits
    /// and RowStats (with and without `lse_dg_kcal`, cutoffs, duplicates and (i, i)
    /// index pairs, within with and without self pairs; §11 item 14).
    #[test]
    fn run_batch_over_the_cpu_port_equals_compute_on_k3() {
        let call = Arc::new(pf_call(30, 31));
        let mut rows = strands(9, 7, 0..=14);
        rows.push(rows[3].clone()); // a duplicate sequence at another index
        let table = SeqTable::new(SeqBatch::from_seqs(&rows));
        let n = rows.len();
        // Cutoffs: none, between keys, and the key of a reported value.
        let probe = DuplexPf::new(Arc::clone(&call), &[], false).unwrap();
        let some_key = probe
            .evaluate(table.get(0), table.get(1), None)
            .unwrap()
            .unwrap();
        let kinds = |k: usize| -> PairKind {
            match k {
                0 => PairKind::Zip,
                1 => PairKind::Cross,
                2 => PairKind::Within { self_pairs: false },
                3 => PairKind::Within { self_pairs: true },
                _ => PairKind::Index {
                    i: IndexArray::from(vec![0, 3, 3, 9, 2, 5, 5, 1]),
                    j: IndexArray::from(vec![1, 9, 9, 3, 2, 5, 0, 1]),
                },
            }
        };
        for kind in 0..5 {
            for (fields, lse) in [
                (&[][..], false),
                (&["dg_ens_err_kcal"][..], false),
                (&[][..], true),
            ] {
                let kernel = DuplexPf::new(Arc::clone(&call), fields, lse).unwrap();
                for sink in [Sink::Each, Sink::Hits, Sink::RowStats] {
                    if lse && sink != Sink::RowStats {
                        continue;
                    }
                    for cutoff in [None, Some(some_key), Some(key_of(-1.0)), Some(i64::MAX - 1)] {
                        let work = WorkSource::new(kinds(kind), n, n).unwrap();
                        let options = ExecOptions {
                            strategy: Strategy::Full,
                            threads: 3,
                            grain: Grain::Exact,
                            sink,
                            cutoff,
                            check_bases: true,
                            k: None,
                        };
                        let report = || ValidationReport {
                            scans: [1, 1],
                            n_illegal_seqs: 0,
                        };
                        let expected =
                            exec::compute(&kernel, &work, &table, &table, &options, report())
                                .unwrap();
                        let mut port = CpuBatch {
                            kernel: &kernel,
                            a: &table,
                            b: &table,
                        };
                        let got = exec::run_batch(
                            &kernel,
                            (&table, &table),
                            &mut port,
                            &work,
                            &options,
                            report(),
                        )
                        .unwrap();
                        assert_eq!(
                            published(&got),
                            published(&expected),
                            "kind {kind} {fields:?} lse={lse} {sink:?} {cutoff:?}"
                        );
                    }
                }
            }
        }
    }

    /// The export of a key column (§9.2): exposed values bit for bit, the mask as the
    /// canonical NaN, and no second buffer.
    #[test]
    fn values_of_inverts_key_of_in_the_key_buffer() {
        let exposed = [
            -1234.5,
            -f64::MIN_POSITIVE,
            0.0,
            f64::MIN_POSITIVE,
            3.25,
            f64::MAX,
            f64::INFINITY,
        ];
        let mut keys: Vec<i64> = exposed.iter().map(|&x| key_of(x)).collect();
        keys.push(i64::MAX);
        let (address, capacity) = (keys.as_ptr() as usize, keys.capacity());
        let values = values_of(keys);
        assert_eq!(
            values.as_ptr() as usize,
            address,
            "the keys' buffer is reused"
        );
        assert_eq!(values.capacity(), capacity);
        let bits: Vec<u64> = values.iter().map(|x| x.to_bits()).collect();
        let mut expected: Vec<u64> = exposed.iter().map(|x| x.to_bits()).collect();
        expected.push(0x7FF8_0000_0000_0000);
        assert_eq!(bits, expected);
    }
}
