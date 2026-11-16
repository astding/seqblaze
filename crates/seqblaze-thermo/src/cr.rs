//! Correctly rounded host arithmetic (tm.md §3.3): exact rationals of binary64
//! values, round-to-nearest-even conversion of a rational to binary64 (gradual
//! underflow and signed zero included), and CR64(c · ln x) by an integer
//! fixed-point Ziv loop (Ziv 1991).
use crate::nat::{Int, Nat};
use std::{cmp::Ordering, sync::OnceLock};

/// An exact rational `num / den`, `den > 0`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Rational {
    pub num: Int,
    pub den: Nat,
}

impl Rational {
    pub fn new(num: Int, den: Nat) -> Self {
        assert!(!den.is_zero(), "zero denominator");
        Self { num, den }
    }

    pub fn from_ratio(num: i64, den: u64) -> Self {
        Self::new(Int::from_i64(num), Nat::from_u64(den))
    }

    /// The exact value of a finite binary64 (−0.0 is 0).
    pub fn from_f64(value: f64) -> Self {
        let (negative, mantissa, exponent) = decompose(value);
        let mantissa = Nat::from_u64(mantissa);
        if exponent >= 0 {
            Self::new(
                Int::new(negative, mantissa.shl(exponent as u64)),
                Nat::from_u64(1),
            )
        } else {
            Self::new(
                Int::new(negative, mantissa),
                Nat::pow2(exponent.unsigned_abs()),
            )
        }
    }
}

/// A finite binary64 as (negative, m, e) with value ±m · 2^e.
pub fn decompose(value: f64) -> (bool, u64, i64) {
    assert!(value.is_finite(), "decompose needs a finite value");
    let bits = value.to_bits();
    let negative = bits >> 63 == 1;
    let biased = ((bits >> 52) & 0x7ff) as i64;
    let fraction = bits & ((1 << 52) - 1);
    if biased == 0 {
        (negative, fraction, -1074)
    } else {
        (negative, fraction | (1 << 52), biased - 1075)
    }
}

/// RN(num / den): the binary64 nearest to the exact rational, ties to even,
/// with gradual underflow (a negative value that rounds to zero is −0.0) and
/// overflow to ±infinity. Zero is +0.0.
pub fn to_f64(num: &Int, den: &Nat) -> f64 {
    assert!(!den.is_zero(), "zero denominator");
    if num.is_zero() {
        return 0.0;
    }
    let negative = num.is_negative();
    let n = num.magnitude();
    // value = q · 2^e with q in [2^52, 2^53): start where q lies in [2^52, 2^54).
    let mut e = n.bits() as i64 - den.bits() as i64 - 53;
    let divide = |e: i64| -> (Nat, Nat, Nat) {
        let (scaled_n, scaled_d) = if e >= 0 {
            (n.clone(), den.shl(e as u64))
        } else {
            (n.shl(e.unsigned_abs()), den.clone())
        };
        let (q, r) = scaled_n.div_rem(&scaled_d);
        (q, r, scaled_d)
    };
    let (mut q, mut r, mut d) = divide(e);
    if q.bits() > 53 {
        e += 1;
        (q, r, d) = divide(e);
    }
    if e < -1074 {
        e = -1074; // subnormal range: the ulp is fixed at 2^-1074
        (q, r, d) = divide(e);
    }
    let mut q = q.to_u64().expect("quotient below 2^54");
    let up = match r.shl(1).cmp(&d) {
        Ordering::Greater => true,
        Ordering::Less => false,
        Ordering::Equal => q & 1 == 1,
    };
    if up {
        q += 1;
        if q == 1 << 53 {
            q = 1 << 52;
            e += 1;
        }
    }
    let sign = if negative { 1u64 << 63 } else { 0 };
    if q == 0 {
        return f64::from_bits(sign);
    }
    if e > 971 {
        return f64::from_bits(sign | 0x7ff0_0000_0000_0000);
    }
    let bits = if q < 1 << 52 {
        q // subnormal (e == -1074)
    } else {
        (((e + 1075) as u64) << 52) | (q - (1 << 52))
    };
    f64::from_bits(sign | bits)
}

/// The Ziv loop exceeded its precision cap (tm.md §3.3 step 4; never expected).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PrecisionCap {
    pub bits: u64,
}

/// Relative working precision of the first Ziv step, in bits.
const START_BITS: u64 = 96;
/// Guard bits above the relative precision.
const GUARD_BITS: u64 = 32;
/// Cap on the relative precision (tm.md §3.3: "e.g. 4,096 bits").
pub const CAP_BITS: u64 = 4096;

/// atanh(u) · 2^p for u = tn / td in [0, 1/3], truncated fixed point, with a bound
/// on the absolute error in units of 2^-p.
///
/// Series Σ u^(2k+1) / (2k+1). Every product and quotient truncates; with
/// u ≤ 1/3 each power carries < 1.8 units of error, each added term < 2, and the
/// tail after the first zero power < 2, so the total stays below 2N + 4 for N
/// added terms.
fn atanh_fixed(tn: &Nat, td: &Nat, p: u64) -> (Nat, u64) {
    let u = tn.shl(p).div_rem(td).0;
    let u2 = u.mul(&u).shr(p);
    let mut power = u.clone();
    let mut sum = u;
    let mut terms = 0u64;
    loop {
        power = power.mul(&u2).shr(p);
        if power.is_zero() {
            break;
        }
        terms += 1;
        sum = sum.add(&power.div_rem_u64(2 * terms + 1).0);
    }
    (sum, 2 * terms + 4)
}

/// The argument reduction of ln x (tm.md §3.3): x = 2^E · y with y in (2/3, 4/3],
/// ln x = E ln 2 + 2 atanh(t), t = (y − 1)/(y + 1), |t| < 1/5.
struct LnReduction {
    e: i64,
    t_negative: bool,
    tn: Nat,
    td: Nat,
    /// With E = 0, |ln x| >= 2|t| >= 2^-offset: the bits to add for relative precision.
    offset: u64,
}

impl LnReduction {
    /// For a rational x > 0, x ≠ 1.
    fn new(xn: &Nat, xd: &Nat) -> Self {
        let mut e = xn.bits() as i64 - xd.bits() as i64;
        let (mut yn, mut yd) = if e >= 0 {
            (xn.clone(), xd.shl(e as u64))
        } else {
            (xn.shl(e.unsigned_abs()), xd.clone())
        };
        // y in (1/2, 2) now; move it into (2/3, 4/3].
        if yn.mul_u64(3) > yd.mul_u64(4) {
            e += 1;
            yd = yd.shl(1);
        } else if yn.mul_u64(3) <= yd.mul_u64(2) {
            e -= 1;
            yn = yn.shl(1);
        }
        let t_negative = yn < yd;
        let tn = if t_negative { yd.sub(&yn) } else { yn.sub(&yd) };
        let td = yn.add(&yd);
        let offset = if e == 0 && !tn.is_zero() {
            td.bits() - tn.bits()
        } else {
            0
        };
        Self {
            e,
            t_negative,
            tn,
            td,
            offset,
        }
    }

    /// (S, err) with |S − 2^p · ln x| <= err, S truncated fixed point.
    fn fixed(&self, p: u64) -> (Int, Nat) {
        let (a, a_error) = atanh_fixed(&self.tn, &self.td, p);
        let mut sum = Int::new(self.t_negative, a.shl(1));
        let mut error = 2 * a_error + 1;
        if self.e != 0 {
            let (half_ln2, ln2_error) = atanh_fixed(&Nat::from_u64(1), &Nat::from_u64(3), p);
            let multiple = half_ln2.shl(1).mul_u64(self.e.unsigned_abs());
            sum = sum.add(&Int::new(self.e < 0, multiple));
            error += 2 * ln2_error * self.e.unsigned_abs();
        }
        (sum, Nat::from_u64(error))
    }
}

/// CR64(coef · ln x): the binary64 nearest to the exact real (ties cannot occur
/// for x ≠ 1, Lindemann 1882), for a rational x > 0 and a rational coefficient.
/// x = 1 or coef = 0 gives exactly +0.0. A subnormal result is the correctly
/// rounded subnormal, and a negative result that rounds to zero is −0.0.
///
/// Ziv's strategy: x = 2^E · y with y in (2/3, 4/3], ln x = E ln 2 + 2 atanh(t),
/// t = (y − 1)/(y + 1), |t| < 1/5, in fixed point with 2^-p resolution, where p
/// adds the magnitude of t when E = 0 (relative, not absolute, precision near 1).
/// Both ends of the error interval are rounded; equal bits are the answer, else
/// the precision doubles, up to [`CAP_BITS`].
pub fn ln(x: &Rational, coef: &Rational) -> Result<f64, PrecisionCap> {
    assert!(
        !x.num.is_negative() && !x.num.is_zero(),
        "ln needs a positive argument"
    );
    let (xn, xd) = (x.num.magnitude(), &x.den);
    if xn == xd || coef.num.is_zero() {
        return Ok(0.0);
    }
    let reduction = LnReduction::new(xn, xd);
    let mut precision = START_BITS;
    while precision <= CAP_BITS {
        let p = precision + reduction.offset + GUARD_BITS;
        let (sum, error) = reduction.fixed(p);
        let error = Int::new(false, error);
        let den = coef.den.shl(p);
        let low = to_f64(&sum.sub(&error).mul(&coef.num), &den);
        let high = to_f64(&sum.add(&error).mul(&coef.num), &den);
        if low.to_bits() == high.to_bits() {
            return Ok(low);
        }
        precision *= 2;
    }
    Err(PrecisionCap { bits: CAP_BITS })
}

/// 2^p · ln 2 in truncated fixed point and a bound on its absolute error in units of
/// 2^-p (ln 2 = 2 atanh(1/3)). Up to [`LN2_CACHE_BITS`] − 16 bits the value comes from one
/// process-wide evaluation, shifted down (one more unit of error).
fn ln2_fixed(p: u64) -> (Nat, Nat) {
    static CACHE: OnceLock<(Nat, Nat)> = OnceLock::new();
    let direct = |p: u64| {
        let (a, error) = atanh_fixed(&Nat::from_u64(1), &Nat::from_u64(3), p);
        (a.shl(1), Nat::from_u64(2 * error))
    };
    if p + 16 > LN2_CACHE_BITS {
        return direct(p);
    }
    let (value, error) = CACHE.get_or_init(|| direct(LN2_CACHE_BITS));
    let shift = LN2_CACHE_BITS - p;
    (value.shr(shift), error.shr(shift).add(&Nat::from_u64(2)))
}

/// Fraction bits of the cached ln 2 (covers the first Ziv steps of every weight).
const LN2_CACHE_BITS: u64 = 1024;

/// Σ_i t_i with t_0 = 2^p and t_i = ⌊⌊t_{i−1} · r / 2^p⌋ / i⌋ until a term is 0, for a
/// fixed-point r ∈ [0, 3/4 · 2^p); returns the sum and the number of steps n. Every
/// floor only lowers a term, so the sum is at most 2^p · exp(r / 2^p); the deficit of a
/// term obeys d_i ≤ (3/4) d_{i−1} + 2 ≤ 8, and the omitted tail after the first zero term
/// is below 4 · 8, so the sum plus 8n + 32 is at least 2^p · exp(r / 2^p).
fn exp_series(r: &Nat, p: u64) -> (Nat, u64) {
    let mut term = Nat::pow2(p);
    let mut sum = term.clone();
    let mut i = 1u64;
    loop {
        term = term.mul(r).shr(p).div_rem_u64(i).0;
        if term.is_zero() {
            return (sum, i);
        }
        sum = sum.add(&term);
        i += 1;
    }
}

/// RN₅₃(v · 2^scale) for a positive natural v: (M, E) with 2^52 ≤ M < 2^53 and value
/// M · 2^E, ties to the even M; a carry to 2^53 moves to (2^52, E + 1).
pub fn round53(v: &Nat, scale: i64) -> (u64, i64) {
    let bits = v.bits();
    assert!(bits > 0, "round53 needs a positive value");
    if bits <= 53 {
        let shift = 53 - bits;
        let m = v.to_u64().expect("at most 53 bits") << shift;
        return (m, scale - shift as i64);
    }
    let shift = bits - 53;
    let q = v.shr(shift);
    let rest = v.sub(&q.shl(shift));
    let half = Nat::pow2(shift - 1);
    let mut m = q.to_u64().expect("53 bits");
    let up = match rest.cmp(&half) {
        Ordering::Greater => true,
        Ordering::Less => false,
        Ordering::Equal => m & 1 == 1,
    };
    let mut e = scale + shift as i64;
    if up {
        m += 1;
        if m == 1 << 53 {
            m = 1 << 52;
            e += 1;
        }
    }
    (m, e)
}

/// One Ziv step of [`exp53`] at p fraction bits: the decided (M, E), or None.
fn exp53_step(negative: bool, xn: &Nat, xd: &Nat, p: u64) -> Option<(u64, i64)> {
    // |x| · 2^p ∈ [x, x + 1]; x · 2^p ∈ [x_lo, x_hi] (signed).
    let x = xn.shl(p).div_rem(xd).0;
    let one = Nat::from_u64(1);
    let (x_lo, x_hi) = if negative {
        (Int::new(true, x.add(&one)), Int::new(true, x.clone()))
    } else {
        (Int::new(false, x.clone()), Int::new(false, x.add(&one)))
    };
    let (l, l_error) = ln2_fixed(p);
    let (l_lo, l_hi) = (l.sub(&l_error), l.add(&l_error));
    // k ≈ ⌊x / ln 2⌋, so that r = x − k ln 2 lies in [0, ln 2) up to the error terms.
    let q = x.div_rem(&l).0;
    let mut k = if negative {
        Int::new(true, q.add(&one))
    } else {
        Int::new(false, q)
    };
    let limit = Nat::pow2(p).mul_u64(3).shr(2);
    for _ in 0..3 {
        // r · 2^p ∈ [x_lo − k·l_hi, x_hi − k·l_lo] for k ≥ 0, with l_lo and l_hi swapped
        // for k < 0.
        let (k_lo, k_hi) = if k.is_negative() {
            (k.mul_nat(&l_lo), k.mul_nat(&l_hi))
        } else {
            (k.mul_nat(&l_hi), k.mul_nat(&l_lo))
        };
        let r_lo = x_lo.sub(&k_lo);
        let r_hi = x_hi.sub(&k_hi);
        if r_lo.is_negative() {
            k = k.sub(&Int::from_i64(1));
            continue;
        }
        if *r_hi.magnitude() >= limit {
            return None;
        }
        let scale = k.to_i64()?.checked_sub(i64::try_from(p).ok()?)?;
        let (low, _) = exp_series(r_lo.magnitude(), p);
        let (high, steps) = exp_series(r_hi.magnitude(), p);
        let high = high.add(&Nat::from_u64(8 * steps + 32));
        let (low, high) = (round53(&low, scale), round53(&high, scale));
        return (low == high).then_some(low);
    }
    None
}

/// RN₅₃(exp(x)) for a rational x (duplex-pf.md §5.1): (M, E) with 2^52 ≤ M < 2^53 and
/// value M · 2^E, nearest, ties to even (a tie cannot occur for x ≠ 0: exp of a nonzero
/// rational is transcendental, Lindemann 1882). x = 0 gives exactly 1 = (2^52, −52).
///
/// Ziv's strategy in integer fixed point (Ziv 1991; tm.md §3.3): x = k ln 2 + r with an
/// integer k and r ∈ [0, ln 2), ln 2 carried to the working precision plus the integer
/// bits of x; exp(r) by its Taylor series with a bound on the truncation error; both ends
/// of the resulting interval are rounded, equal results are the answer, else the
/// precision doubles, up to [`CAP_BITS`]. A k beyond int64 (excluded by the §5.4 check)
/// also ends at the cap.
pub fn exp53(x: &Rational) -> Result<(u64, i64), PrecisionCap> {
    if x.num.is_zero() {
        return Ok((1 << 52, -52));
    }
    let (xn, xd) = (x.num.magnitude(), &x.den);
    let int_bits = xn.bits().saturating_sub(xd.bits()) + 1;
    let mut precision = START_BITS;
    while precision <= CAP_BITS {
        let p = precision + int_bits + GUARD_BITS;
        if let Some(result) = exp53_step(x.num.is_negative(), xn, xd, p) {
            return Ok(result);
        }
        precision *= 2;
    }
    Err(PrecisionCap { bits: CAP_BITS })
}

/// k if the positive natural n is 10^k, else None.
fn power_of_ten(n: &Nat) -> Option<u64> {
    let mut n = n.clone();
    let mut k = 0;
    while n != Nat::from_u64(1) {
        let (q, r) = n.div_rem_u64(10);
        if r != 0 || q.is_zero() {
            return None;
        }
        n = q;
        k += 1;
    }
    Some(k)
}

/// k if the rational x > 0 equals 10^k exactly for an integer k (tm-primer3.md §9.3
/// step 1; for a binary64 x only 10^0 … 10^22).
fn exact_log10(xn: &Nat, xd: &Nat) -> Option<i64> {
    let (q, r) = xn.div_rem(xd);
    if r.is_zero() {
        return power_of_ten(&q).map(|k| k as i64);
    }
    let (q, r) = xd.div_rem(xn);
    if r.is_zero() {
        return power_of_ten(&q).map(|k| -(k as i64));
    }
    None
}

/// CR64(log10 x) for a rational x > 0 (tm-primer3.md §9.3): the binary64 nearest to the
/// exact log10 x, ties to even. x = 10^k returns k exactly (x = 1 gives +0.0); for every
/// other x, log10 x is irrational, never a binary64 midpoint, and the Ziv loop ends.
///
/// At working precision p the intervals [S ± e_S] ∋ 2^p ln x and [T ± e_T] ∋ 2^p ln 10
/// come from the same fixed-point reduction as [`ln`] (ln 10 = 3 ln 2 + 2 atanh(1/9)),
/// with the relative-precision offset of x near 1. The quotient interval is
/// [(S − e_S)/(T + e_T), (S + e_S)/(T − e_T)] for ln x > 0 and
/// [(S − e_S)/(T − e_T), (S + e_S)/(T + e_T)] for ln x < 0; both ends are rounded by
/// [`to_f64`], equal bits are the answer, else the precision doubles up to [`CAP_BITS`].
/// An S interval that contains 0 also doubles the precision.
pub fn log10(x: &Rational) -> Result<f64, PrecisionCap> {
    assert!(
        !x.num.is_negative() && !x.num.is_zero(),
        "log10 needs a positive argument"
    );
    let (xn, xd) = (x.num.magnitude(), &x.den);
    if let Some(k) = exact_log10(xn, xd) {
        return Ok(k as f64); // exact: |k| < 2^53 for any rational that fits in memory
    }
    let reduction = LnReduction::new(xn, xd);
    let ten = LnReduction::new(&Nat::from_u64(10), &Nat::from_u64(1));
    let mut precision = START_BITS;
    while precision <= CAP_BITS {
        let p = precision + reduction.offset + GUARD_BITS;
        let (s, s_error) = reduction.fixed(p);
        let (t, t_error) = ten.fixed(p);
        let (s_abs, t_abs) = (s.magnitude(), t.magnitude());
        if *s_abs <= s_error {
            precision *= 2; // the ln x interval contains 0
            continue;
        }
        // T > e_T always: 2^p ln 10 > 2^p and e_T is a few hundred units.
        let (t_low, t_high) = (t_abs.sub(&t_error), t_abs.add(&t_error));
        let (s_low, s_high) = (s_abs.sub(&s_error), s_abs.add(&s_error));
        // |log10 x| lies in [|S| − e_S, |S| + e_S] / [T − e_T, T + e_T]; the sign is S's.
        let negative = s.is_negative();
        let small = to_f64(&Int::new(negative, s_low), &t_high);
        let large = to_f64(&Int::new(negative, s_high), &t_low);
        if small.to_bits() == large.to_bits() {
            return Ok(small);
        }
        precision *= 2;
    }
    Err(PrecisionCap { bits: CAP_BITS })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rng(state: &mut u64) -> u64 {
        *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = *state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn ratio(num: i64, den: u64) -> f64 {
        to_f64(&Int::from_i64(num), &Nat::from_u64(den))
    }

    #[test]
    fn integer_and_quotient_rounding_match_ieee() {
        let mut state = 3;
        for _ in 0..5000 {
            let a = (rng(&mut state) >> (rng(&mut state) % 64)) as i64;
            assert_eq!(ratio(a, 1).to_bits(), (a as f64).to_bits(), "{a}");
            assert_eq!(ratio(-a, 1).to_bits(), ((-a) as f64).to_bits(), "{a}");
            // Both operands below 2^53 are exact, so IEEE division is the reference.
            let x = (rng(&mut state) >> 11) as i64;
            let y = (rng(&mut state) >> (11 + rng(&mut state) % 50)) | 1;
            assert_eq!(
                ratio(x, y).to_bits(),
                (x as f64 / y as f64).to_bits(),
                "{x}/{y}"
            );
        }
    }

    #[test]
    fn exact_binary64_round_trip_and_extremes() {
        let mut state = 5;
        for _ in 0..3000 {
            let bits = rng(&mut state) & !(0x7ff << 52) | ((rng(&mut state) % 2047) << 52);
            let value = f64::from_bits(bits);
            let r = Rational::from_f64(value);
            let back = to_f64(&r.num, &r.den);
            if value == 0.0 {
                assert_eq!(back.to_bits(), 0); // −0.0 has the exact value 0
            } else {
                assert_eq!(back.to_bits(), value.to_bits());
            }
        }
        let p2 = |k: u64| Nat::pow2(k);
        let one = Int::from_i64(1);
        assert_eq!(to_f64(&one, &p2(1074)), 5e-324);
        assert_eq!(to_f64(&one, &p2(1075)).to_bits(), 0); // tie to even: 0
        assert_eq!(to_f64(&Int::from_i64(-1), &p2(1075)).to_bits(), 1 << 63); // −0.0
        assert_eq!(to_f64(&Int::from_i64(3), &p2(1076)), 5e-324); // 0.75 ulp
        assert_eq!(to_f64(&Int::from_i64(3), &p2(1075)), 1e-323); // 1.5 ulp: tie to 2
        let max = Int::new(false, Nat::from_u64((1 << 53) - 1).shl(971));
        assert_eq!(to_f64(&max, &Nat::from_u64(1)), f64::MAX);
        let inf = Int::new(false, Nat::pow2(1024));
        assert_eq!(to_f64(&inf, &Nat::from_u64(1)), f64::INFINITY);
        // Just below 2^1024 rounds up past MAX: infinity.
        let near = Int::new(false, Nat::pow2(1024).sub(&Nat::pow2(969)));
        assert_eq!(to_f64(&near, &Nat::from_u64(1)), f64::INFINITY);
        // Smallest normal from a subnormal round-up.
        let n = Int::new(false, Nat::pow2(53).sub(&Nat::from_u64(1)));
        assert_eq!(to_f64(&n, &p2(1075)), f64::MIN_POSITIVE);
    }

    fn ln_of(x: f64) -> f64 {
        ln(&Rational::from_f64(x), &Rational::from_ratio(1, 1)).unwrap()
    }

    #[test]
    fn ln_constants_and_exact_one() {
        assert_eq!(ln_of(2.0), std::f64::consts::LN_2);
        assert_eq!(ln_of(10.0), std::f64::consts::LN_10);
        assert_eq!(ln_of(0.5), -std::f64::consts::LN_2);
        assert_eq!(ln_of(1.0).to_bits(), 0);
        let third = Rational::from_ratio(1, 3);
        assert_eq!(
            ln(&third, &Rational::from_ratio(0, 1)).unwrap().to_bits(),
            0
        );
        let one = Rational::from_ratio(7, 7);
        assert_eq!(ln(&one, &Rational::from_ratio(-5, 3)).unwrap().to_bits(), 0);
    }

    #[test]
    fn ln_is_faithful_against_libm_away_from_one() {
        let mut state = 9;
        for _ in 0..400 {
            let bits = rng(&mut state) & !(0x7ff << 52) | ((1 + rng(&mut state) % 2045) << 52);
            let x = f64::from_bits(bits & !(1 << 63));
            if (0.5..2.0).contains(&x) {
                continue;
            }
            let got = ln_of(x);
            let libm = x.ln();
            let ulp = f64::from_bits(got.abs().to_bits() + 1) - got.abs();
            assert!(
                (got - libm).abs() <= ulp,
                "ln({x:e}): {got:e} vs libm {libm:e}"
            );
        }
    }

    #[test]
    fn ln_near_one_keeps_relative_precision_and_sign() {
        // x = 1 - 2^-k: ln x = -2^-k - 2^-2k/2 - ..., correctly rounded is -2^-k(1 + 2^-k-1)
        for k in [60u64, 300, 1000, 1070, 1074, 1100] {
            let num = Int::new(false, Nat::pow2(k).sub(&Nat::from_u64(1)));
            let x = Rational::new(num, Nat::pow2(k));
            let got = ln(&x, &Rational::from_ratio(1, 1)).unwrap();
            let want = if k <= 1074 {
                -f64::from_bits(
                    ((1023 - k as i64).max(0) as u64) << 52
                        | if k > 1022 { 1 << (1074 - k) } else { 0 },
                )
            } else {
                -0.0
            };
            assert_eq!(got.to_bits(), want.to_bits(), "k={k}: {got:e} vs {want:e}");
        }
        // A positive tiny result keeps its sign; a sub-half-ulp one is +0.0.
        let x = Rational::new(
            Int::new(false, Nat::pow2(1100).add(&Nat::from_u64(1))),
            Nat::pow2(1100),
        );
        assert_eq!(ln(&x, &Rational::from_ratio(1, 1)).unwrap().to_bits(), 0);
    }

    fn log10_of(x: f64) -> f64 {
        log10(&Rational::from_f64(x)).unwrap()
    }

    #[test]
    fn log10_exact_powers_of_ten() {
        let below_one = f64::from_bits(1f64.to_bits() - 1);
        assert_eq!(
            log10_of(below_one).to_bits(),
            (-f64::from_bits(0x3C8B_CB7B_1526_E50F)).to_bits()
        );
        let mut power = 1.0f64;
        for k in 0..=22 {
            let got = log10_of(power);
            assert_eq!(got.to_bits(), (k as f64).to_bits(), "10^{k}");
            power *= 10.0; // exact up to 10^22
        }
        assert_eq!(log10_of(1.0).to_bits(), 0);
        // Exact on rationals in either direction, and unreduced forms.
        assert_eq!(log10(&Rational::from_ratio(1, 1000)).unwrap(), -3.0);
        assert_eq!(log10(&Rational::from_ratio(20, 2)).unwrap(), 1.0);
        assert_eq!(log10(&Rational::from_ratio(4, 40)).unwrap(), -1.0);
        assert_eq!(log10(&Rational::from_ratio(7, 7)).unwrap().to_bits(), 0);
    }

    #[test]
    fn log10_near_powers_of_ten_is_not_exact_but_rounds() {
        // 0.1 etc. are not powers of ten as binary64; their log10 is irrational and
        // rounds to the integer (tm-primer3.md §9.3 step 2).
        for (x, want) in [(0.1, -1.0), (0.01, -2.0), (0.001, -3.0), (1e-6, -6.0)] {
            assert_eq!(log10_of(x), want, "{x}");
        }
        // 10^23 is not a binary64; its nearest neighbour gives 23 after rounding.
        assert_eq!(log10_of(1e23), 23.0);
    }

    #[test]
    fn log10_is_faithful_and_signed() {
        let mut state = 21;
        for _ in 0..400 {
            let bits = rng(&mut state) & !(0x7ff << 52) | ((1 + rng(&mut state) % 2045) << 52);
            let x = f64::from_bits(bits & !(1 << 63));
            let got = log10_of(x);
            let libm = x.log10();
            let ulp = f64::from_bits(got.abs().to_bits() + 1) - got.abs();
            assert!(
                (got - libm).abs() <= ulp,
                "log10({x:e}): {got:e} vs libm {libm:e}"
            );
            assert_eq!(got < 0.0, x < 1.0, "sign of log10({x:e})");
        }
        // Extremes: smallest subnormal and largest finite value.
        let tiny = log10_of(5e-324);
        assert!((tiny - (-323.3062153431158)).abs() <= 1e-13, "{tiny}");
        let big = log10_of(f64::MAX);
        assert!((big - 308.25471555991675).abs() <= 1e-13, "{big}");
    }

    #[test]
    fn log10_near_one_keeps_relative_precision() {
        // x = 1 ± k ulp: CR64(ln x)/ln 10 rounds twice, so it is within 2 ulp of the result;
        // an absolute-precision loop would be off by orders of magnitude here. The exact
        // value at x = 1 - 2^-53 (decimal, 80 digits) is -0x1.bcb7b1526e50fp-55.
        for k in 1..=4u64 {
            for x in [
                f64::from_bits(1f64.to_bits() + k),
                f64::from_bits(1f64.to_bits() - k),
            ] {
                let got = log10_of(x);
                let approx = ln_of(x) / std::f64::consts::LN_10;
                let ulp = f64::from_bits(got.abs().to_bits() + 1) - got.abs();
                assert!(
                    got != 0.0 && (got - approx).abs() <= 2.0 * ulp,
                    "{x:e}: {got:e} vs {approx:e}"
                );
                assert_eq!(got < 0.0, x < 1.0);
            }
        }
    }
}
