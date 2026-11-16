//! T1 native `tm`: two-state Tm, ΔG, ΔH and ΔS of a sequence with its perfect
//! complement. Spec: docs/wiki/thermo/tm.md.
//!
//! Per call (host, once): the sodium equivalent N1-N6 (§3.2), the correctly
//! rounded constants s, κ₄, κ₁ (§3.3), the 13-entry integer table G_e(T) with
//! its int32 range check (§3.4). Per sequence: exact integer sums (§4) and the
//! binary64 finalize F1-F10 in fixed order (§5), no FMA (rustc never contracts;
//! no `mul_add` here), one NaN encoding.
use seqblaze_core::{
    detail::{canonical, DetailColumns, DetailField, DetailRequest, FieldKind},
    seq::{ByteView, SeqTable},
    unary::UnaryKernel,
    validate::EngineError,
};

use crate::{
    cr::{self, Rational},
    nat::{round_half_even, Int, Nat},
    params::{Params, Ratio},
};

/// D48 native thermodynamics limit, nt.
pub const MAX_LENGTH: usize = 200;
/// A duplex needs one stack (tm.md §7, QT7).
pub const MIN_LENGTH: usize = 2;

/// Field indices, API §3 order; `tm_c` is the primary (≥).
pub const TM_C: usize = 0;
pub const DG_KCAL: usize = 1;
pub const DG_DCAL: usize = 2;
pub const DH_KCAL: usize = 3;
pub const DS_CAL_PER_K: usize = 4;

pub static FIELDS: [DetailField; 5] = [
    DetailField {
        name: "tm_c",
        kind: FieldKind::Float64,
    },
    DetailField {
        name: "dg_kcal",
        kind: FieldKind::Float64,
    },
    DetailField {
        name: "dg_dcal",
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
];

/// K₀ = 0x1.1126666666666p+8, the binary64 nearest 273.15 (tm.md §2.2; F10 only).
pub const K0: f64 = f64::from_bits(0x4071_1266_6666_6666);

/// Strand concentration in nM (tm.md §3.3).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Oligo {
    /// Total strand concentration C.
    Total(f64),
    /// Strand a (the input sequence) and strand b (its complement).
    Pair(f64, f64),
}

/// The binary64 values of `sb.Conditions` that native `tm` uses (tm.md §3.1).
#[derive(Debug, Clone, Copy, PartialEq)]
#[allow(non_snake_case)]
pub struct Conditions {
    pub temp_c: f64,
    pub na_mM: f64,
    pub k_mM: f64,
    pub tris_mM: f64,
    pub mg_mM: f64,
    pub dntp_mM: f64,
    pub oligo_nM: Oligo,
}

/// Slot indices of the per-call table: the 10 stack classes, then these three.
const INIT: usize = 10;
const TERMINAL_AT: usize = 11;
const SYMMETRY: usize = 12;

#[derive(Debug, Clone, Copy, Default)]
struct Step {
    h: i64,
    g: i64,
    d: i64,
}

#[derive(Debug, Clone, Copy)]
struct PackedStep {
    hg: u64,
    d: i64,
}

#[derive(Debug, Clone)]
struct StepTables<T> {
    steps: [T; 16],
    quadruplets: [T; 256],
}

impl<T: Copy> StepTables<T> {
    fn new(steps: [T; 16], add: impl Fn(T, T, T) -> T) -> Self {
        let quadruplets = std::array::from_fn(|index| {
            add(
                steps[index >> 4],
                steps[(index >> 2) & 15],
                steps[index & 15],
            )
        });
        Self { steps, quadruplets }
    }

    #[inline]
    fn sum<A: Default>(&self, seq: &[u8], first: u8, mut add: impl FnMut(&mut A, T)) -> (A, u8) {
        let mut sum = A::default();
        let mut previous = first;
        let mut triples = seq[1..].chunks_exact(3);
        for triple in triples.by_ref() {
            let middle0 = validated_code(triple[0]);
            let middle1 = validated_code(triple[1]);
            let code = validated_code(triple[2]);
            let entry =
                self.quadruplets[usize::from(previous << 6 | middle0 << 4 | middle1 << 2 | code)];
            add(&mut sum, entry);
            previous = code;
        }
        for &byte in triples.remainder() {
            let code = validated_code(byte);
            add(&mut sum, self.steps[usize::from(previous << 2 | code)]);
            previous = code;
        }
        (sum, previous)
    }
}

// Keep both bounded table forms inline; boxing the fallback would add an
// allocation solely to equalize variant sizes.
#[allow(clippy::large_enum_variant)]
#[derive(Debug, Clone)]
enum StackTables {
    Packed { tables: StepTables<PackedStep> },
    Wide(StepTables<Step>),
}

impl StackTables {
    fn new(steps: [Step; 16]) -> Self {
        let max_h = steps.iter().map(|s| -i128::from(s.h)).max().unwrap();
        let max_g = steps.iter().map(|s| -i128::from(s.g)).max().unwrap();
        let count = (MAX_LENGTH - 1) as i128;
        if steps.iter().all(|s| s.h <= 0 && s.g <= 0)
            && count * max_h <= i128::from(u32::MAX)
            && count * max_g <= i128::from(u32::MAX)
        {
            // Nonnegative magnitudes and their complete 199-step sums fit u32.
            // Adding these words cannot carry H into G or overflow the word.
            let packed = steps.map(|entry| PackedStep {
                hg: (-i128::from(entry.h)) as u64 | (((-i128::from(entry.g)) as u64) << 32),
                d: entry.d,
            });
            Self::Packed {
                tables: StepTables::new(packed, |left, middle, right| PackedStep {
                    hg: left.hg + middle.hg + right.hg,
                    d: left.d + middle.d + right.d,
                }),
            }
        } else {
            Self::Wide(StepTables::new(steps, |left, middle, right| Step {
                h: left.h + middle.h + right.h,
                g: left.g + middle.g + right.g,
                d: left.d + middle.d + right.d,
            }))
        }
    }
}

/// The per-call pack (tm.md §12): host constants, the integer table and the
/// per-step lookups. Built once, shared read-only by every worker.
#[derive(Debug, Clone)]
pub struct CallPack {
    pub params: &'static Params,
    pub na_eq: f64,
    pub s: f64,
    pub kappa4: f64,
    pub kappa1: f64,
    /// G_e(T) in dcal/mol: 10 stack classes, init, terminal_at, symmetry.
    pub table: [i32; 13],
    /// B of the §3.4 range check.
    pub bound: i64,
    /// Exact stack sums, packed when both nonpositive lanes have bounded magnitudes.
    stacks: StackTables,
    init: Step,
    terminal_at: Step,
    symmetry: Step,
}

fn parameter(message: String) -> EngineError {
    EngineError::Parameter(message)
}

fn rational(r: Ratio) -> Rational {
    Rational::from_ratio(r.num, r.den)
}

/// A non-negative finite binary64 as an exact dyadic m · 2^e.
fn dyadic(value: f64) -> (Nat, i64) {
    let (_, m, e) = cr::decompose(value);
    (Nat::from_u64(m), e)
}

/// The exact rational (num · 2^e) / den for a non-negative dyadic numerator.
fn scaled(num: Nat, e: i64, den: Nat) -> Rational {
    if e >= 0 {
        Rational::new(Int::new(false, num.shl(e as u64)), den)
    } else {
        Rational::new(Int::new(false, num), den.shl(e.unsigned_abs()))
    }
}

/// (c₄, c₁) in mol/L as exact rationals (tm.md §3.3; QT3).
pub(crate) fn concentrations(oligo: Oligo) -> (Rational, Rational) {
    let giga = Nat::pow10(9);
    match oligo {
        Oligo::Total(c) => {
            let (m, e) = dyadic(c);
            (scaled(m.clone(), e, giga.mul_u64(4)), scaled(m, e, giga))
        }
        Oligo::Pair(a, b) => {
            let (hi, lo) = if a >= b { (a, b) } else { (b, a) };
            let ((mh, eh), (ml, el)) = (dyadic(hi), dyadic(lo));
            // 2·max − min and a + b, exactly, over the common exponent.
            let base = eh.min(el);
            let twice_hi = mh.shl((eh + 1 - base) as u64);
            let one_hi = mh.shl((eh - base) as u64);
            let one_lo = ml.shl((el - base) as u64);
            let c4 = scaled(twice_hi.sub(&one_lo), base, giga.mul_u64(2));
            let c1 = scaled(one_hi.add(&one_lo), base, giga);
            (c4, c1)
        }
    }
}

/// N1-N6 (tm.md §3.2): one binary64 operation per step, round to nearest even.
///
/// Provenance: N2 (Tris+ = half the Tris buffer) is Biopython's documented
/// `salt_correction` convention; von Ahsen 2001 does not state it (QT12).
/// N5 clamps free Mg2+ at 0 when Mg <= dNTP; no primary source states it (QT7).
pub(crate) fn sodium_equivalent(cond: &Conditions, c_mg: f64) -> f64 {
    let m1 = cond.na_mM + cond.k_mM; // N1
    let m2 = cond.tris_mM * 0.5; // N2: Tris+ = half the Tris buffer (QT12)
    let mono = m1 + m2; // N3
    let f = cond.mg_mM - cond.dntp_mM; // N4
    let w = if f > 0.0 { c_mg * f.sqrt() } else { 0.0 }; // N5 (free Mg clamped at 0; QT7)
    mono + w // N6
}

pub(crate) fn check_conditions(cond: &Conditions) -> Result<(), EngineError> {
    let mut values = vec![
        ("temp_c", cond.temp_c),
        ("na_mM", cond.na_mM),
        ("k_mM", cond.k_mM),
        ("tris_mM", cond.tris_mM),
        ("mg_mM", cond.mg_mM),
        ("dntp_mM", cond.dntp_mM),
    ];
    match cond.oligo_nM {
        Oligo::Total(c) => values.push(("oligo_nM", c)),
        Oligo::Pair(a, b) => values.extend([("oligo_nM", a), ("oligo_nM", b)]),
    }
    for (name, value) in values {
        if !value.is_finite() {
            return Err(parameter(format!("{name}={value} is not finite")));
        }
        if name != "temp_c" && value < 0.0 {
            return Err(parameter(format!("{name}={value} is negative")));
        }
    }
    // t > −273.15 exactly: −5463/20 is not a binary64 and its upper neighbour
    // is the literal −273.15 (nearest, and above), so t >= −273.15 is the test.
    if cond.temp_c < -273.15 {
        return Err(parameter(format!(
            "temp_c={} is at or below absolute zero (-273.15)",
            cond.temp_c
        )));
    }
    let valid = match cond.oligo_nM {
        Oligo::Total(c) => c > 0.0,
        Oligo::Pair(a, b) => a.max(b) > 0.0,
    };
    if !valid {
        return Err(parameter(
            "oligo_nM must be a total > 0 or a pair (a, b) with a, b >= 0 and max(a, b) > 0".into(),
        ));
    }
    Ok(())
}

/// The per-call host constants shared by `tm` and `duplex` (tm.md §3.1-§3.3):
/// the checked conditions, Na_eq (N1-N6) and the correctly rounded s, κ₄, κ₁.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Host {
    pub na_eq: f64,
    pub s: f64,
    pub kappa4: f64,
    pub kappa1: f64,
}

impl Host {
    /// From the salt coefficient b, R and c_mg of a build (tm.md §2.1 rules 4-6);
    /// `kernel` names the call in the messages.
    pub(crate) fn new(
        kernel: &str,
        cond: &Conditions,
        salt_b: Ratio,
        gas_r: Ratio,
        naeq_mg: Ratio,
    ) -> Result<Self, EngineError> {
        check_conditions(cond)?;
        let c_mg = cr::to_f64(&Int::from_i64(naeq_mg.num), &Nat::from_u64(naeq_mg.den));
        let na_eq = sodium_equivalent(cond, c_mg);
        if na_eq.is_nan() || na_eq <= 0.0 || na_eq.is_infinite() {
            return Err(parameter(format!(
                "{kernel}: the sodium equivalent Na_eq = na_mM + k_mM + tris_mM/2 + {c_mg}*sqrt(mg_mM - dntp_mM) \
                 is {na_eq} mM for na_mM={}, k_mM={}, tris_mM={}, mg_mM={}, dntp_mM={}; it must be positive \
                 and finite. Give a monovalent cation or free Mg2+ (mg_mM > dntp_mM), e.g. \
                 sb.Conditions(temp_c=37, na_mM=50, oligo_nM=250)",
                cond.na_mM, cond.k_mM, cond.tris_mM, cond.mg_mM, cond.dntp_mM
            )));
        }
        let cap = |name: &'static str| {
            move |error: cr::PrecisionCap| {
                EngineError::Parameter(format!(
                "internal error: the correctly rounded {name} needs more than {} bits; nothing was computed",
                error.bits
            ))
            }
        };
        let na_ratio = Rational::from_f64(na_eq);
        let na_molar = Rational::new(na_ratio.num, na_ratio.den.mul_u64(1000));
        let s = cr::ln(&na_molar, &rational(salt_b)).map_err(cap("s"))?;
        let (c4, c1) = concentrations(cond.oligo_nM);
        let r = rational(gas_r);
        let kappa4 = cr::ln(&c4, &r).map_err(cap("kappa4"))?;
        let kappa1 = cr::ln(&c1, &r).map_err(cap("kappa1"))?;
        Ok(Self {
            na_eq,
            s,
            kappa4,
            kappa1,
        })
    }
}

impl CallPack {
    /// Build the per-call pack; every failure is a ParameterError (an option
    /// error, raised before the sequence scan). Runs in the IEEE default
    /// floating-point environment whatever the caller's state, which is
    /// restored on return (text-nulls-fp-environment.md §7.3 item 2).
    pub fn new(params: &'static Params, cond: &Conditions) -> Result<Self, EngineError> {
        seqblaze_core::fpenv::guarded(|| Self::build(params, cond))
    }

    fn build(params: &'static Params, cond: &Conditions) -> Result<Self, EngineError> {
        let Host {
            na_eq,
            s,
            kappa4,
            kappa1,
        } = Host::new("tm", cond, params.salt_b, params.gas_r, params.naeq_mg)?;
        let entries: Vec<(i64, i64, i64)> = (0..10)
            .map(|k| (params.stack_dh[k], params.stack_dg[k], 1))
            .chain([params.init, params.terminal_at, params.symmetry].map(|e| (e.dh, e.dg, 0)))
            .collect();
        let t = Rational::from_f64(cond.temp_c);
        let s_exact = Rational::from_f64(s);
        let mut table = [0i32; 13];
        let mut magnitudes = [0i128; 13];
        for (slot, &(h, g, p)) in entries.iter().enumerate() {
            let value = energy_at(h, g, p, &t, &s_exact)
                .to_i64()
                .filter(|v| v.unsigned_abs() < 1 << 62)
                .ok_or_else(|| range_error(cond, na_eq, None))?;
            magnitudes[slot] = i128::from(value).abs();
            table[slot] = i32::try_from(value).unwrap_or(0);
        }
        let stack_max = magnitudes[..10].iter().copied().max().unwrap_or(0);
        let bound =
            199 * stack_max + magnitudes[INIT] + 2 * magnitudes[TERMINAL_AT] + magnitudes[SYMMETRY];
        if bound > i128::from(i32::MAX) {
            return Err(range_error(cond, na_eq, Some(bound)));
        }
        let step = |slot: usize, h: i64, g: i64| Step {
            h,
            g,
            d: i64::from(table[slot]),
        };
        let mut steps = [Step::default(); 16];
        for (index, &class) in params.step_class.iter().enumerate() {
            steps[index] = step(class, params.stack_dh[class], params.stack_dg[class]);
        }
        let stacks = StackTables::new(steps);
        Ok(Self {
            params,
            na_eq,
            s,
            kappa4,
            kappa1,
            table,
            bound: bound as i64,
            stacks,
            init: step(INIT, params.init.dh, params.init.dg),
            terminal_at: step(TERMINAL_AT, params.terminal_at.dh, params.terminal_at.dg),
            symmetry: step(SYMMETRY, params.symmetry.dh, params.symmetry.dg),
        })
    }

    /// Every field of one validated sequence (upper-case ACGT, 2 ≤ L ≤ 200).
    /// Expects the IEEE default floating-point environment: it is called per
    /// item from guarded scheduler units and carries no guard of its own
    /// (text-nulls-fp-environment.md §7.3 item 2).
    #[inline]
    pub fn row(&self, seq: &[u8]) -> TmRow {
        let len = seq.len();
        let first = validated_code(seq[0]);
        let (mut h, mut g, mut d, previous) = match &self.stacks {
            StackTables::Packed { tables } => {
                let ((hg, d), previous) = tables.sum(seq, first, |sum: &mut (u64, i64), entry| {
                    sum.0 += entry.hg;
                    sum.1 += entry.d;
                });
                (-i64::from(hg as u32), -((hg >> 32) as i64), d, previous)
            }
            StackTables::Wide(tables) => {
                let ((h, g, d), previous) =
                    tables.sum(seq, first, |sum: &mut (i64, i64, i64), entry| {
                        sum.0 += entry.h;
                        sum.1 += entry.g;
                        sum.2 += entry.d;
                    });
                (h, g, d, previous)
            }
        };
        let mut add = |entry: &Step| {
            h += entry.h;
            g += entry.g;
            d += entry.d;
        };
        add(&self.init);
        // A = 0 and T = 3: an A·T end.
        for end in [first, previous] {
            if end == 0 || end == 3 {
                add(&self.terminal_at);
            }
        }
        let sc = self_complementary(seq);
        if sc {
            add(&self.symmetry);
        }
        let kappa = if sc { self.kappa1 } else { self.kappa4 };
        finalize(h, g, len as i64 - 1, d, self.s, kappa)
    }
}

/// G_e(T) = RNE(g + (h − g)(37 − t)·20/6203 − T·p·s/10), exact (tm.md §3.4).
/// The same rational as [`energy_at_q`] with q = 2p, so the same integer.
fn energy_at(h: i64, g: i64, p: i64, t: &Rational, s: &Rational) -> Int {
    energy_at_q(h, g, 2 * p, t, s)
}

/// G_e(T) = RNE(g + (h − g)(37 − t)·20/6203 − T·q·s/20) with q = 2p, exact
/// (tm.md §3.4; duplex-mfe.md §4.2), with t = Tn/Td, s = Sn/Sd and T = t + 5463/20:
/// N = g·D + 8000(h − g)(37·Td − Tn)·Sd − 6203·q·Sn·(20·Tn + 5463·Td),
/// D = 2,481,200·Td·Sd.
pub(crate) fn energy_at_q(h: i64, g: i64, q: i64, t: &Rational, s: &Rational) -> Int {
    let td = Int::new(false, t.den.clone());
    let sd = Int::new(false, s.den.clone());
    let d = t.den.mul(&s.den).mul_u64(2_481_200);
    let first = Int::new(false, d.clone()).mul_i64(g);
    let second = td
        .mul_i64(37)
        .sub(&t.num)
        .mul(&sd)
        .mul_i64(8000)
        .mul_i64(h - g);
    let third = t
        .num
        .mul_i64(20)
        .add(&td.mul_i64(5463))
        .mul(&s.num)
        .mul_i64(6203 * q);
    round_half_even(&first.add(&second).sub(&third), &d)
}

/// Call-wide coefficients of the exact rational in [`energy_at_q`]. Each entry
/// still receives its own RNE; only factors independent of (h, g, q) are shared.
pub(crate) struct EnergyAtQ {
    denominator: Int,
    temperature: Int,
    salt: Int,
}

impl EnergyAtQ {
    pub(crate) fn new(t: &Rational, s: &Rational) -> Self {
        let td = Int::new(false, t.den.clone());
        let sd = Int::new(false, s.den.clone());
        let denominator = Int::new(false, t.den.mul(&s.den).mul_u64(2_481_200));
        let temperature = td.mul_i64(37).sub(&t.num).mul(&sd).mul_i64(8000);
        let salt = t
            .num
            .mul_i64(20)
            .add(&td.mul_i64(5463))
            .mul(&s.num)
            .mul_i64(6203);
        Self {
            denominator,
            temperature,
            salt,
        }
    }

    /// RNE((g D + (h - g) A - q B) / D), with positive D and arbitrary-size
    /// integer coefficients. This is the same numerator as [`energy_at_q`].
    #[inline]
    pub(crate) fn entry(&self, h: i64, g: i64, q: i64) -> Int {
        let first = self.denominator.mul_i64(g);
        let second = self.temperature.mul_i64(h - g);
        let third = self.salt.mul_i64(q);
        round_half_even(
            &first.add(&second).sub(&third),
            self.denominator.magnitude(),
        )
    }
}

fn range_error(cond: &Conditions, na_eq: f64, bound: Option<i128>) -> EngineError {
    let reach = bound.map_or_else(
        || "an entry beyond int64".to_string(),
        |b| format!("{b} dcal/mol"),
    );
    parameter(format!(
        "tm: temp_c={} with Na_eq={na_eq} mM is out of the representable range: the per-call \
         energy table reaches {reach} for a 200-nt duplex, beyond int32 (2147483647); use a \
         temp_c or salt condition closer to the model's range",
        cond.temp_c
    ))
}

/// The `SequenceError` for a length outside 2..=200 (tm.md §7 step 2; also compat).
pub(crate) fn length_error(index: usize, length: usize, limit: usize) -> EngineError {
    let message = if length < limit {
        format!(
            "tm input a sequence {index} has length {length}, below the minimum {limit} nt: \
             tm needs at least 2 nt (a duplex needs one stack); nothing was computed"
        )
    } else {
        format!(
            "tm input a sequence {index} has length {length}, above the native thermodynamics \
             limit {limit} nt (D48); nothing was computed"
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

/// Arithmetic encoding of validated ACGT; the result is always in 0..4.
#[inline]
fn validated_code(byte: u8) -> u8 {
    ((byte >> 1) ^ (byte >> 2)) & 3
}

/// Base code: A = 0, C = 1, G = 2, T = 3 (validated input only).
pub(crate) const CODE: [u8; 256] = {
    let mut table = [0u8; 256];
    table[b'C' as usize] = 1;
    table[b'G' as usize] = 2;
    table[b'T' as usize] = 3;
    table
};

/// s = revcomp(s): even length and code(s[i]) + code(s[L-1-i]) = 3 for all i.
#[inline]
fn self_complementary(seq: &[u8]) -> bool {
    let len = seq.len();
    len % 2 == 0
        && (0..len / 2).all(|i| validated_code(seq[i]) + validated_code(seq[len - 1 - i]) == 3)
}

/// One sequence's fields.
#[derive(Debug, Clone, Copy)]
pub struct TmRow {
    pub tm_c: f64,
    pub dg_kcal: f64,
    pub dg_dcal: i32,
    pub dh_kcal: f64,
    pub ds_cal_per_k: f64,
}

/// F1-F10 (tm.md §5): one binary64 operation per statement, in this order.
#[inline]
pub fn finalize(h: i64, g: i64, p: i64, d: i64, s: f64, kappa: f64) -> TmRow {
    finalize_pairs(h, g, p as f64, d, s, kappa)
}

/// F1-F10 with the phosphate-pair count P as an exact binary64 (duplex-mfe.md §7:
/// P = Q/2, a half-integer for K1; tm passes its integer P = L − 1).
#[inline]
pub fn finalize_pairs(h: i64, g: i64, p: f64, d: i64, s: f64, kappa: f64) -> TmRow {
    let dh_kcal = h as f64 / 100.0; // F1
    let dg_kcal = d as f64 / 100.0; // F2
    let x1 = (200 * (h - g)) as f64; // F3, exact
    let x2 = x1 / 6203.0; // F4
    let x3 = p * s; // F5
    let ds_cal_per_k = x2 + x3; // F6
    let x4 = ds_cal_per_k + kappa; // F7
    let x5 = (10 * h) as f64; // F8, exact
    let x6 = x5 / x4; // F9, IEEE quotient (±inf, NaN for 0/0)
    let tm_c = x6 - K0; // F10
    TmRow {
        tm_c: canonical(tm_c),
        dg_kcal: canonical(dg_kcal),
        dg_dcal: d as i32,
        dh_kcal: canonical(dh_kcal),
        ds_cal_per_k: canonical(ds_cal_per_k),
    }
}

/// The unary kernel of one call: the pack and the requested fields.
pub struct Tm<'p> {
    pack: &'p CallPack,
    request: DetailRequest,
}

impl<'p> Tm<'p> {
    /// `fields` by name; `tm_c`, the primary, is always emitted (tm.md §6).
    pub fn new(pack: &'p CallPack, fields: &[&str]) -> Result<Self, EngineError> {
        let mut request = DetailRequest::NONE.with(TM_C);
        for name in fields {
            let index = FIELDS
                .iter()
                .position(|field| field.name == *name)
                .ok_or_else(|| parameter(format!("tm has no field {name:?}")))?;
            request = request.with(index);
        }
        Ok(Self { pack, request })
    }
}

impl UnaryKernel for Tm<'_> {
    type Row = TmRow;
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
        // About one table lookup per base plus the finalize.
        table.mean_len() + 16
    }

    fn columns(&self) -> Result<DetailColumns, EngineError> {
        DetailColumns::try_new(&FIELDS, self.request)
    }

    #[inline]
    fn evaluate(&self, _: &mut (), seq: ByteView<'_>) -> TmRow {
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
    fn passes(&self, row: &TmRow, cutoff: f64) -> bool {
        row.tm_c >= cutoff
    }

    #[inline]
    fn push(&self, row: &TmRow, out: &mut DetailColumns) -> Result<(), EngineError> {
        out.push_f64(TM_C, row.tm_c)?;
        if out.wants(DG_KCAL) {
            out.push_f64(DG_KCAL, row.dg_kcal)?;
        }
        if out.wants(DG_DCAL) {
            out.push_i32(DG_DCAL, row.dg_dcal)?;
        }
        if out.wants(DH_KCAL) {
            out.push_f64(DH_KCAL, row.dh_kcal)?;
        }
        if out.wants(DS_CAL_PER_K) {
            out.push_f64(DS_CAL_PER_K, row.ds_cal_per_k)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::params::SANTALUCIA2004;

    fn cond(temp_c: f64, na: f64, oligo: Oligo) -> Conditions {
        Conditions {
            temp_c,
            na_mM: na,
            k_mM: 0.0,
            tris_mM: 0.0,
            mg_mM: 0.0,
            dntp_mM: 0.0,
            oligo_nM: oligo,
        }
    }

    #[test]
    fn k0_is_the_nearest_binary64_to_273_15() {
        assert_eq!(K0, 273.15);
        // The literal −273.15 lies above −5463/20, so `t >= -273.15` is `t > −5463/20`.
        let below = Rational::from_f64(-273.15);
        let exact = Int::from_i64(-5463).mul_nat(&below.den);
        assert!(below.num.mul_i64(20).sub(&exact).magnitude() > &Nat::zero());
        assert!(!below.num.mul_i64(20).sub(&exact).is_negative());
    }

    #[test]
    fn at_37c_and_1m_the_table_is_the_published_dg() {
        let pack = CallPack::new(&SANTALUCIA2004, &cond(37.0, 1000.0, Oligo::Total(50.0))).unwrap();
        assert_eq!(pack.s.to_bits(), 0);
        let p = &SANTALUCIA2004;
        for k in 0..10 {
            assert_eq!(i64::from(pack.table[k]), p.stack_dg[k]);
        }
        assert_eq!(i64::from(pack.table[INIT]), p.init.dg);
        assert_eq!(i64::from(pack.table[TERMINAL_AT]), p.terminal_at.dg);
        assert_eq!(i64::from(pack.table[SYMMETRY]), p.symmetry.dg);
        // SantaLucia & Hicks 2004 Eq. 1 worked example: CGTTGA, printed -5.35 kcal/mol.
        assert_eq!(pack.row(b"CGTTGA").dg_dcal, -535);
    }

    #[test]
    fn exact_one_arguments_give_positive_zero() {
        let pack = CallPack::new(&SANTALUCIA2004, &cond(37.0, 1000.0, Oligo::Total(4e9))).unwrap();
        assert_eq!((pack.s.to_bits(), pack.kappa4.to_bits()), (0, 0));
        let pack =
            CallPack::new(&SANTALUCIA2004, &cond(37.0, 50.0, Oligo::Pair(1e9, 0.0))).unwrap();
        assert_eq!((pack.kappa4.to_bits(), pack.kappa1.to_bits()), (0, 0));
        let pack =
            CallPack::new(&SANTALUCIA2004, &cond(37.0, 50.0, Oligo::Pair(1e9, 5e-324))).unwrap();
        assert_eq!(pack.kappa4.to_bits(), 1 << 63); // −0.0 (tm.md §3.3)
    }

    #[test]
    fn revcomp_rows_are_identical() {
        let pack =
            CallPack::new(&SANTALUCIA2004, &cond(60.0, 50.0, Oligo::Pair(30.0, 7.0))).unwrap();
        for seq in [
            &b"ACGTTGCAAGGT"[..],
            b"AT",
            b"CCCCG",
            b"GATTACAGATTACA",
            b"ACGT",
        ] {
            let rc: Vec<u8> = seq
                .iter()
                .rev()
                .map(|&b| match b {
                    b'A' => b'T',
                    b'C' => b'G',
                    b'G' => b'C',
                    _ => b'A',
                })
                .collect();
            let (x, y) = (pack.row(seq), pack.row(&rc));
            assert_eq!(x.tm_c.to_bits(), y.tm_c.to_bits());
            assert_eq!(x.dg_dcal, y.dg_dcal);
            assert_eq!(x.ds_cal_per_k.to_bits(), y.ds_cal_per_k.to_bits());
        }
        assert!(self_complementary(b"ACGT") && !self_complementary(b"ACGA"));
        assert!(!self_complementary(b"ACG"));
    }

    #[test]
    fn packed_stack_domains_check_each_lane_and_sign() {
        const M: i64 = 21_582_750; // floor(u32::MAX / 199)
        for h_lane in [true, false] {
            for (maximum, packed) in [(M, true), (M + 1, false)] {
                let mut steps = [Step::default(); 16];
                if h_lane {
                    steps[0].h = -maximum;
                } else {
                    steps[0].g = -maximum;
                }
                assert_eq!(
                    matches!(StackTables::new(steps), StackTables::Packed { .. }),
                    packed
                );
            }
            let mut mixed = [Step { h: -1, g: -1, d: 0 }; 16];
            if h_lane {
                mixed[0].h = 1;
            } else {
                mixed[0].g = 1;
            }
            assert!(matches!(StackTables::new(mixed), StackTables::Wide(_)));
            // Three such steps still fit i64; their 199-step magnitude does not fit u32.
            let value = -i64::MAX / 4;
            let step = Step {
                h: if h_lane { value } else { 0 },
                g: if h_lane { 0 } else { value },
                d: 0,
            };
            assert!(matches!(StackTables::new([step; 16]), StackTables::Wide(_)));
        }
    }

    #[test]
    fn packed_stack_rows_match_plain_integer_sums() {
        const M: i64 = 21_582_750;
        let cases = [
            std::array::from_fn(|i| Step {
                h: i as i64 - 32,
                g: 3 * i as i64 - 64,
                d: i as i64 - 8,
            }),
            [Step { h: -M, g: 0, d: 7 }; 16],
            [Step { h: 0, g: -M, d: -7 }; 16],
        ];
        let mut pack =
            CallPack::new(&SANTALUCIA2004, &cond(37.0, 1000.0, Oligo::Total(4e9))).unwrap();
        pack.init = Step::default();
        pack.terminal_at = Step::default();
        pack.symmetry = Step::default();
        pack.kappa1 = pack.kappa4;
        let bits = |row: TmRow| {
            [
                row.tm_c.to_bits(),
                row.dg_kcal.to_bits(),
                row.dg_dcal as u64,
                row.dh_kcal.to_bits(),
                row.ds_cal_per_k.to_bits(),
            ]
        };
        for steps in cases {
            pack.stacks = StackTables::new(steps);
            assert!(matches!(&pack.stacks, StackTables::Packed { .. }));
            for len in [2, 3, 4, 5, 199, 200] {
                let seq: Vec<u8> = (0..len).map(|i| b"ACGT"[i % 4]).collect();
                let (h, g, d) = seq.windows(2).fold((0, 0, 0), |(h, g, d), pair| {
                    let step =
                        steps[usize::from(CODE[pair[0] as usize] * 4 + CODE[pair[1] as usize])];
                    (h + step.h, g + step.g, d + step.d)
                });
                let expected = finalize(h, g, len as i64 - 1, d, pack.s, pack.kappa4);
                assert_eq!(bits(pack.row(&seq)), bits(expected), "length {len}");
            }
        }
    }

    #[test]
    fn option_errors() {
        let p = &SANTALUCIA2004;
        let err = |c: Conditions| CallPack::new(p, &c).unwrap_err().to_string();
        assert!(err(cond(37.0, 0.0, Oligo::Total(50.0))).contains("Na_eq"));
        assert!(err(cond(1e15, 50.0, Oligo::Total(50.0))).contains("range"));
        assert!(err(cond(-273.16, 50.0, Oligo::Total(50.0))).contains("temp_c"));
        assert!(err(cond(37.0, 50.0, Oligo::Pair(0.0, 0.0))).contains("oligo_nM"));
        assert!(err(cond(37.0, f64::NAN, Oligo::Total(1.0))).contains("na_mM"));
        assert!(CallPack::new(p, &cond(-273.15, 50.0, Oligo::Total(50.0))).is_ok());
    }
}
