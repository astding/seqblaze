//! Minimal arbitrary-precision integers for the host constants of tm.md §3.3-§3.4:
//! exact rationals of binary64 values, fixed-point logarithms and correct rounding.
//!
//! Host only and once per call (a few thousand limb operations); never on a
//! per-sequence path. Schoolbook multiplication and binary long division keep
//! the code short and obviously correct; the operands stay below ~5,000 bits.
use std::cmp::Ordering;

/// Unsigned integer: little-endian u64 limbs with no high zero limb.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Nat(Vec<u64>);

impl Nat {
    pub fn zero() -> Self {
        Self(Vec::new())
    }

    pub fn from_u64(value: u64) -> Self {
        let mut out = Self(vec![value]);
        out.trim();
        out
    }

    pub fn from_u128(value: u128) -> Self {
        let mut out = Self(vec![value as u64, (value >> 64) as u64]);
        out.trim();
        out
    }

    /// 2^k.
    pub fn pow2(k: u64) -> Self {
        Self::from_u64(1).shl(k)
    }

    /// 10^k.
    pub fn pow10(k: u32) -> Self {
        (0..k).fold(Self::from_u64(1), |acc, _| acc.mul_u64(10))
    }

    fn trim(&mut self) {
        while self.0.last() == Some(&0) {
            self.0.pop();
        }
    }

    pub fn is_zero(&self) -> bool {
        self.0.is_empty()
    }

    /// Number of significant bits (0 for zero).
    pub fn bits(&self) -> u64 {
        match self.0.last() {
            None => 0,
            Some(&top) => 64 * (self.0.len() as u64 - 1) + u64::from(64 - top.leading_zeros()),
        }
    }

    pub fn is_odd(&self) -> bool {
        self.0.first().is_some_and(|limb| limb & 1 == 1)
    }

    pub fn to_u64(&self) -> Option<u64> {
        match self.0.len() {
            0 => Some(0),
            1 => Some(self.0[0]),
            _ => None,
        }
    }

    pub fn shl(&self, n: u64) -> Self {
        if self.is_zero() {
            return Self::zero();
        }
        let limbs = (n / 64) as usize;
        let bits = (n % 64) as u32;
        let mut out = Vec::with_capacity(limbs + self.0.len() + 1);
        out.resize(limbs, 0);
        if bits == 0 {
            out.extend_from_slice(&self.0);
        } else {
            let mut carry = 0;
            for &limb in &self.0 {
                out.push((limb << bits) | carry);
                carry = limb >> (64 - bits);
            }
            out.push(carry);
        }
        let mut out = Self(out);
        out.trim();
        out
    }

    pub fn shr(&self, n: u64) -> Self {
        let limbs = (n / 64) as usize;
        if limbs >= self.0.len() {
            return Self::zero();
        }
        let bits = (n % 64) as u32;
        let source = &self.0[limbs..];
        let mut out = Vec::with_capacity(source.len());
        for (k, &limb) in source.iter().enumerate() {
            if bits == 0 {
                out.push(limb);
            } else {
                let high = source.get(k + 1).map_or(0, |&next| next << (64 - bits));
                out.push((limb >> bits) | high);
            }
        }
        let mut out = Self(out);
        out.trim();
        out
    }

    pub fn add(&self, other: &Self) -> Self {
        let (long, short) = if self.0.len() >= other.0.len() {
            (self, other)
        } else {
            (other, self)
        };
        let mut out = Vec::with_capacity(long.0.len() + 1);
        let mut carry = 0u64;
        for (k, &limb) in long.0.iter().enumerate() {
            let (sum, c1) = limb.overflowing_add(short.0.get(k).copied().unwrap_or(0));
            let (sum, c2) = sum.overflowing_add(carry);
            out.push(sum);
            carry = u64::from(c1) + u64::from(c2);
        }
        out.push(carry);
        let mut out = Self(out);
        out.trim();
        out
    }

    /// `self - other`; panics if `other > self` (callers order the operands).
    pub fn sub(&self, other: &Self) -> Self {
        assert!(*self >= *other, "Nat::sub underflow");
        let mut out = self.0.clone();
        sub_in_place(&mut out, &other.0);
        let mut out = Self(out);
        out.trim();
        out
    }

    pub fn mul(&self, other: &Self) -> Self {
        if self.is_zero() || other.is_zero() {
            return Self::zero();
        }
        let mut out = vec![0u64; self.0.len() + other.0.len()];
        for (i, &a) in self.0.iter().enumerate() {
            let mut carry = 0u128;
            for (j, &b) in other.0.iter().enumerate() {
                let t = u128::from(a) * u128::from(b) + u128::from(out[i + j]) + carry;
                out[i + j] = t as u64;
                carry = t >> 64;
            }
            out[i + other.0.len()] = carry as u64;
        }
        let mut out = Self(out);
        out.trim();
        out
    }

    pub fn mul_u64(&self, factor: u64) -> Self {
        self.mul(&Self::from_u64(factor))
    }

    /// (floor(self / divisor), self mod divisor) for a non-zero u64 divisor.
    pub fn div_rem_u64(&self, divisor: u64) -> (Self, u64) {
        assert!(divisor != 0, "division by zero");
        let mut out = vec![0u64; self.0.len()];
        let mut rem = 0u128;
        for k in (0..self.0.len()).rev() {
            let current = (rem << 64) | u128::from(self.0[k]);
            out[k] = (current / u128::from(divisor)) as u64;
            rem = current % u128::from(divisor);
        }
        let mut out = Self(out);
        out.trim();
        (out, rem as u64)
    }

    /// (floor(self / divisor), self mod divisor) for a non-zero divisor, by
    /// binary long division.
    pub fn div_rem(&self, divisor: &Self) -> (Self, Self) {
        assert!(!divisor.is_zero(), "division by zero");
        if *self < *divisor {
            return (Self::zero(), self.clone());
        }
        if let Some(small) = divisor.to_u64() {
            let (q, r) = self.div_rem_u64(small);
            return (q, Self::from_u64(r));
        }
        let mut quotient = vec![0u64; self.0.len()];
        let mut rem: Vec<u64> = Vec::with_capacity(divisor.0.len() + 1);
        for bit in (0..self.bits()).rev() {
            // rem = 2 * rem + bit
            let mut carry = (self.0[(bit / 64) as usize] >> (bit % 64)) & 1;
            for limb in rem.iter_mut() {
                let next = *limb >> 63;
                *limb = (*limb << 1) | carry;
                carry = next;
            }
            if carry != 0 {
                rem.push(carry);
            }
            if cmp_limbs(&rem, &divisor.0) != Ordering::Less {
                sub_in_place(&mut rem, &divisor.0);
                while rem.last() == Some(&0) {
                    rem.pop();
                }
                quotient[(bit / 64) as usize] |= 1 << (bit % 64);
            }
        }
        let mut quotient = Self(quotient);
        quotient.trim();
        (quotient, Self(rem))
    }

    /// Parse a non-negative decimal string (testing hooks only).
    pub fn parse_decimal(text: &str) -> Option<Self> {
        if text.is_empty() || !text.bytes().all(|b| b.is_ascii_digit()) {
            return None;
        }
        let mut out = Self::zero();
        for digit in text.bytes() {
            out = out
                .mul_u64(10)
                .add(&Self::from_u64(u64::from(digit - b'0')));
        }
        Some(out)
    }
}

/// Compare trimmed little-endian limb slices.
fn cmp_limbs(a: &[u64], b: &[u64]) -> Ordering {
    let a_len = a.iter().rposition(|&limb| limb != 0).map_or(0, |k| k + 1);
    let b_len = b.iter().rposition(|&limb| limb != 0).map_or(0, |k| k + 1);
    a_len
        .cmp(&b_len)
        .then_with(|| a[..a_len].iter().rev().cmp(b[..b_len].iter().rev()))
}

/// `a -= b` for `a >= b` (as numbers).
fn sub_in_place(a: &mut [u64], b: &[u64]) {
    let mut borrow = 0u64;
    for (k, limb) in a.iter_mut().enumerate() {
        let (d, b1) = limb.overflowing_sub(b.get(k).copied().unwrap_or(0));
        let (d, b2) = d.overflowing_sub(borrow);
        *limb = d;
        borrow = u64::from(b1) + u64::from(b2);
    }
    debug_assert_eq!(borrow, 0, "sub_in_place underflow");
}

impl PartialOrd for Nat {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Nat {
    fn cmp(&self, other: &Self) -> Ordering {
        cmp_limbs(&self.0, &other.0)
    }
}

/// Signed integer as sign and magnitude; zero is never negative.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Int {
    negative: bool,
    magnitude: Nat,
}

impl Int {
    pub fn new(negative: bool, magnitude: Nat) -> Self {
        Self {
            negative: negative && !magnitude.is_zero(),
            magnitude,
        }
    }

    pub fn zero() -> Self {
        Self::new(false, Nat::zero())
    }

    pub fn from_i64(value: i64) -> Self {
        Self::new(value < 0, Nat::from_u64(value.unsigned_abs()))
    }

    pub fn is_negative(&self) -> bool {
        self.negative
    }

    pub fn is_zero(&self) -> bool {
        self.magnitude.is_zero()
    }

    pub fn magnitude(&self) -> &Nat {
        &self.magnitude
    }

    pub fn neg(&self) -> Self {
        Self::new(!self.negative, self.magnitude.clone())
    }

    pub fn add(&self, other: &Self) -> Self {
        if self.negative == other.negative {
            return Self::new(self.negative, self.magnitude.add(&other.magnitude));
        }
        match self.magnitude.cmp(&other.magnitude) {
            Ordering::Less => Self::new(other.negative, other.magnitude.sub(&self.magnitude)),
            _ => Self::new(self.negative, self.magnitude.sub(&other.magnitude)),
        }
    }

    pub fn sub(&self, other: &Self) -> Self {
        self.add(&other.neg())
    }

    pub fn mul(&self, other: &Self) -> Self {
        Self::new(
            self.negative != other.negative,
            self.magnitude.mul(&other.magnitude),
        )
    }

    pub fn mul_nat(&self, other: &Nat) -> Self {
        Self::new(self.negative, self.magnitude.mul(other))
    }

    pub fn mul_i64(&self, factor: i64) -> Self {
        self.mul(&Self::from_i64(factor))
    }

    /// The value as i64, if it fits.
    pub fn to_i64(&self) -> Option<i64> {
        let magnitude = self.magnitude.to_u64()?;
        if self.negative {
            0i64.checked_sub_unsigned(magnitude)
        } else {
            i64::try_from(magnitude).ok()
        }
    }

    /// Parse an optionally signed decimal string (testing hooks only).
    pub fn parse_decimal(text: &str) -> Option<Self> {
        match text.strip_prefix('-') {
            Some(rest) => Some(Self::new(true, Nat::parse_decimal(rest)?)),
            None => Some(Self::new(false, Nat::parse_decimal(text)?)),
        }
    }
}

/// RNE(n / d), tm.md §3.4: q = floor(n / d), r = n - q d; q + 1 if 2r > d, q if
/// 2r < d, the even one of q and q + 1 if 2r = d.
pub fn round_half_even(n: &Int, d: &Nat) -> Int {
    let (q, r) = n.magnitude().div_rem(d);
    let (q, r) = if !n.is_negative() {
        (Int::new(false, q), r)
    } else if r.is_zero() {
        (Int::new(true, q), r)
    } else {
        (Int::new(true, q.add(&Nat::from_u64(1))), d.sub(&r))
    };
    let up = match r.shl(1).cmp(d) {
        Ordering::Greater => true,
        Ordering::Less => false,
        Ordering::Equal => q.magnitude().is_odd(),
    };
    if up {
        q.add(&Int::from_i64(1))
    } else {
        q
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn n(v: u128) -> Nat {
        Nat::from_u128(v)
    }

    /// Deterministic 64-bit generator (SplitMix64) for the arithmetic cases.
    fn rng(state: &mut u64) -> u64 {
        *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = *state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    #[test]
    fn arithmetic_matches_u128_on_small_values() {
        let mut state = 7;
        for _ in 0..2000 {
            let a = u128::from(rng(&mut state) >> (rng(&mut state) % 64));
            let b = u128::from(rng(&mut state) >> (rng(&mut state) % 64)) | 1;
            assert_eq!(n(a).add(&n(b)), n(a + b));
            assert_eq!(n(a).mul(&n(b)), n(a * b));
            let (big, small) = (a.max(b), a.min(b));
            assert_eq!(n(big).sub(&n(small)), n(big - small));
            let wide = a * b + small;
            let (q, r) = n(wide).div_rem(&n(b));
            assert_eq!((q, r), (n(wide / b), n(wide % b)));
            let s = rng(&mut state) % 100;
            assert_eq!(n(a).shl(s).shr(s), n(a));
            assert_eq!(n(wide).shr(s % 120), n(wide >> (s % 120)));
            assert_eq!(n(a).cmp(&n(b)), a.cmp(&b));
        }
    }

    #[test]
    fn long_division_round_trips_on_wide_values() {
        let mut state = 11;
        for _ in 0..200 {
            let a = (0..(rng(&mut state) % 20 + 1))
                .fold(Nat::from_u64(1), |acc, _| acc.mul_u64(rng(&mut state) | 1));
            let d = (0..(rng(&mut state) % 10 + 2))
                .fold(Nat::from_u64(3), |acc, _| acc.mul_u64(rng(&mut state) | 1));
            let (q, r) = a.div_rem(&d);
            assert!(r < d);
            assert_eq!(q.mul(&d).add(&r), a);
        }
    }

    #[test]
    fn signed_ops_and_half_even() {
        let i = Int::from_i64;
        assert_eq!(i(5).sub(&i(9)), i(-4));
        assert_eq!(i(-5).mul(&i(-3)), i(15));
        assert_eq!(i(-7).add(&i(7)), Int::zero());
        assert!(!i(-7).add(&i(7)).is_negative());
        assert_eq!(i(i64::MIN).to_i64(), Some(i64::MIN));
        for (num, den, want) in [
            (1, 2, 0),
            (3, 2, 2),
            (5, 2, 2),
            (-1, 2, 0),
            (-3, 2, -2),
            (-5, 2, -2),
            (7, 3, 2),
            (-7, 3, -2),
            (-8, 3, -3),
            (0, 5, 0),
            (-4, 1, -4),
        ] {
            let got = round_half_even(&i(num), &Nat::from_u64(den));
            assert_eq!(got, i(want), "{num}/{den}");
        }
        assert_eq!(Int::parse_decimal("-1234"), Some(i(-1234)));
        assert_eq!(Nat::parse_decimal("x"), None);
        assert_eq!(Nat::pow10(3), Nat::from_u64(1000));
    }
}
