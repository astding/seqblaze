//! Exact per-row sums for RowStats (D47; duplex-pf.md §9.4): a growable non-negative
//! integer N with an integer exponent β, value N · 2^β, β the smallest significand weight
//! added so far. Terms are non-negative dyadic numbers M · 2^E. Every addition and every
//! merge of two partial sums is exact, so the result does not depend on how pairs are
//! split among workers, chunks or call shapes. There is no fixed anchor and there are no
//! partial sums selected by exponent: one integer per row, which grows to hold its terms.
//!
//! The growth has one resource limit (§9.4 *Width limit*, QP8b): an addition or a merge
//! whose exact result would be wider than [`MAX_LIMBS`] limbs, counted over the whole
//! exact value at its smallest exponent β, ends the call with the D86 `DeviceError`, as a
//! refused reservation does. Every term is non-negative, so a partial sum is never wider
//! than the row's final sum, and the limit fires iff the row's final width exceeds it,
//! whatever the order of the terms (B1).
use crate::validate::EngineError;

/// The largest width of a row's exact sum: W_max = 2^18 limbs of 64 bits (2^24 bits,
/// 2 MiB; duplex-pf.md §9.4 *Width limit*). A row's sum never holds more than
/// `MAX_LIMBS + 2` limbs of capacity.
pub const MAX_LIMBS: usize = 1 << 18;

/// [`MAX_LIMBS`] in bits.
const MAX_BITS: u128 = 64 * MAX_LIMBS as u128;

/// A non-negative dyadic term M · 2^E (M = 0 is zero).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Dyadic {
    pub m: u64,
    pub e: i64,
}

impl Dyadic {
    pub const ZERO: Self = Self { m: 0, e: 0 };
}

/// An exact sum N · 2^β of dyadic terms; N in little-endian u64 limbs, empty for zero.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ExactSum {
    limbs: Vec<u64>,
    beta: i64,
}

/// Built through [`crate::detail::device_error`]: workers add row terms, and a
/// refused reservation there must be a DeviceError, not an abort (engine-layout §6).
fn refused() -> EngineError {
    crate::detail::device_error(format_args!(
        "unable to reserve the exact row sum (width); no result was published"
    ))
}

/// The width limit (§9.4): the message names neither the row nor the width it would have
/// had, so it does not depend on which row reaches the limit first.
fn width_limit() -> EngineError {
    crate::detail::device_error(format_args!(
        "unable to reserve the exact row sum of lse_dg_kcal (width above {MAX_LIMBS} limbs); \
         no result was published"
    ))
}

/// Bit length of the integer in little-endian `limbs` (0 for zero).
fn bit_len(limbs: &[u64]) -> u128 {
    match limbs.iter().rposition(|&l| l != 0) {
        None => 0,
        Some(top) => 64 * top as u128 + u128::from(64 - limbs[top].leading_zeros()),
    }
}

/// `limbs += value · 2^shift` (bits). The result has at most one bit more than the longer
/// of the two aligned operands, so one exact reservation of that many limbs plus one
/// holds it, the carry included.
fn add_shifted(limbs: &mut Vec<u64>, value: &[u64], shift: u64) -> Result<(), EngineError> {
    let value = &value[..value.iter().rposition(|&l| l != 0).map_or(0, |top| top + 1)];
    if value.is_empty() {
        return Ok(());
    }
    let longer = bit_len(limbs).max(u128::from(shift) + bit_len(value));
    let need = usize::try_from(longer.div_ceil(64) + 1).map_err(|_| refused())?;
    if limbs.len() < need {
        limbs
            .try_reserve_exact(need - limbs.len())
            .map_err(|_| refused())?;
        limbs.resize(need, 0);
    }
    // need ≥ word + value.len() + 1, so the spill limb is in range; the carry stops below
    // limb `need`, because the result is below 2^(longer + 1).
    let word = (shift / 64) as usize;
    let bits = (shift % 64) as u32;
    let mut carry = 0u64;
    let mut k = word;
    let mut spill = 0u64;
    for &v in value {
        let part = if bits == 0 { v } else { (v << bits) | spill };
        spill = if bits == 0 { 0 } else { v >> (64 - bits) };
        let (sum, c1) = limbs[k].overflowing_add(part);
        let (sum, c2) = sum.overflowing_add(carry);
        limbs[k] = sum;
        carry = u64::from(c1) + u64::from(c2);
        k += 1;
    }
    let (sum, c1) = limbs[k].overflowing_add(spill);
    let (sum, c2) = sum.overflowing_add(carry);
    limbs[k] = sum;
    carry = u64::from(c1) + u64::from(c2);
    k += 1;
    while carry != 0 {
        let (sum, c) = limbs[k].overflowing_add(carry);
        limbs[k] = sum;
        carry = u64::from(c);
        k += 1;
    }
    while limbs.last() == Some(&0) {
        limbs.pop();
    }
    Ok(())
}

impl ExactSum {
    pub fn is_zero(&self) -> bool {
        self.limbs.is_empty()
    }

    /// Lower β by `shift` bits (shifting N left), exactly. The shift runs in place, from
    /// the top limb down, so lowering never holds a second copy of N. N is nonzero and the
    /// caller has checked the width and reserved the room.
    fn lower(&mut self, shift: u64) -> Result<(), EngineError> {
        if shift == 0 {
            return Ok(());
        }
        let word = usize::try_from(shift / 64).map_err(|_| refused())?;
        let bits = (shift % 64) as u32;
        let old_len = self.limbs.len();
        let new_len = old_len
            .checked_add(word)
            .and_then(|n| n.checked_add(1))
            .ok_or_else(refused)?;
        self.limbs
            .try_reserve_exact(new_len - old_len)
            .map_err(|_| refused())?;
        self.limbs.resize(new_len, 0);
        let limbs = &mut self.limbs;
        // new[j] = (old[j − word] << bits) | (old[j − word − 1] >> (64 − bits)), j
        // descending: every source index is ≤ j and not yet overwritten.
        let old = |limbs: &[u64], k: usize| if k < old_len { limbs[k] } else { 0 };
        for j in (word..new_len).rev() {
            let k = j - word;
            let high = old(limbs, k);
            let value = if bits == 0 {
                high
            } else {
                let low = if k == 0 { 0 } else { old(limbs, k - 1) };
                (high << bits) | (low >> (64 - bits))
            };
            limbs[j] = value;
        }
        limbs[..word].fill(0);
        while limbs.last() == Some(&0) {
            limbs.pop();
        }
        self.beta -= shift as i64;
        Ok(())
    }

    /// Add a term exactly.
    pub fn add(&mut self, term: Dyadic) -> Result<(), EngineError> {
        if term.m == 0 {
            return Ok(());
        }
        self.merge_parts(&[term.m], term.e)
    }

    /// Add another partial sum exactly.
    pub fn merge(&mut self, other: &ExactSum) -> Result<(), EngineError> {
        if other.is_zero() {
            return Ok(());
        }
        self.merge_parts(&other.limbs, other.beta)
    }

    /// `self += value · 2^beta` for a nonzero `value` without leading zero limbs.
    fn merge_parts(&mut self, limbs: &[u64], beta: i64) -> Result<(), EngineError> {
        if self.is_zero() {
            // The result is the other operand.
            if limbs.len() > MAX_LIMBS {
                return Err(width_limit());
            }
            self.limbs
                .try_reserve_exact(limbs.len())
                .map_err(|_| refused())?;
            self.limbs.extend_from_slice(limbs);
            self.beta = beta;
            return Ok(());
        }
        // The width limit, before any memory is touched (§9.4 *Width limit*). At the
        // common β the exact result is at least as long as the longer aligned operand and
        // at most one bit longer; lowering β alone gives the first operand's length.
        let base = self.beta.min(beta);
        let above = |b: i64| (i128::from(b) - i128::from(base)) as u128; // in [0, 2^64)
        let longer = (bit_len(&self.limbs) + above(self.beta)).max(bit_len(limbs) + above(beta));
        if longer > MAX_BITS {
            return Err(width_limit());
        }
        // From here every length is at most 2^24 bits. One exact reservation of at most
        // MAX_LIMBS + 1 limbs holds the lowered N and the sum, the carry included.
        let need = longer.div_ceil(64) as usize + 1;
        self.limbs
            .try_reserve_exact(need.saturating_sub(self.limbs.len()))
            .map_err(|_| refused())?;
        self.lower(above(self.beta) as u64)?;
        add_shifted(&mut self.limbs, limbs, above(beta) as u64)?;
        // A carry can make the result one bit longer than the longer operand.
        if self.limbs.len() > MAX_LIMBS {
            return Err(width_limit());
        }
        Ok(())
    }

    /// RN₅₃ of the sum: (M, E) with 2^52 ≤ M < 2^53 and value M · 2^E, nearest, ties to
    /// the even M; None for zero.
    pub fn round53(&self) -> Option<(u64, i64)> {
        let top = *self.limbs.last()?;
        let bits = 64 * (self.limbs.len() as u64 - 1) + u64::from(64 - top.leading_zeros());
        let bit = |k: u64| (self.limbs[(k / 64) as usize] >> (k % 64)) & 1;
        if bits <= 53 {
            // N < 2^53 fits in the low limb.
            let shift = 53 - bits;
            return Some((self.limbs[0] << shift, self.beta - shift as i64));
        }
        let shift = bits - 53;
        let mut m = 0u64;
        for k in (shift..bits).rev() {
            m = (m << 1) | bit(k);
        }
        let half = bit(shift - 1) == 1;
        let sticky = shift >= 2 && {
            let below = shift - 1;
            let full = (below / 64) as usize;
            self.limbs[..full].iter().any(|&l| l != 0)
                || (below % 64 != 0 && self.limbs[full] & ((1u64 << (below % 64)) - 1) != 0)
        };
        let mut e = self.beta + shift as i64;
        if half && (sticky || m & 1 == 1) {
            m += 1;
            if m == 1 << 53 {
                m = 1 << 52;
                e += 1;
            }
        }
        Some((m, e))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn value(sum: &ExactSum) -> Option<(u128, i64)> {
        // Small sums only: N < 2^128.
        if sum.is_zero() {
            return None;
        }
        let n = sum
            .limbs
            .iter()
            .rev()
            .fold(0u128, |acc, &l| (acc << 64) | u128::from(l));
        Some((n, sum.beta))
    }

    #[test]
    fn exact_addition_and_merge_are_order_free() {
        let terms = [
            Dyadic { m: 3, e: 5 },
            Dyadic { m: 1 << 52, e: -60 },
            Dyadic::ZERO,
            Dyadic { m: 7, e: 0 },
            Dyadic { m: u64::MAX, e: 3 },
        ];
        let mut forward = ExactSum::default();
        for t in terms {
            forward.add(t).unwrap();
        }
        let mut left = ExactSum::default();
        let mut right = ExactSum::default();
        for (k, t) in terms.iter().rev().enumerate() {
            if k % 2 == 0 {
                left.add(*t).unwrap();
            } else {
                right.add(*t).unwrap();
            }
        }
        left.merge(&right).unwrap();
        assert_eq!(forward, left);
        // 3·2^5 + 2^52·2^-60 + 7 + (2^64 − 1)·2^3 at β = −60.
        let want = (3u128 << 65) + (1u128 << 52) + (7u128 << 60) + (u128::from(u64::MAX) << 63);
        assert_eq!(value(&forward), Some((want, -60)));
    }

    #[test]
    fn round53_ties_to_even_and_carries() {
        let mut s = ExactSum::default();
        s.add(Dyadic { m: 1, e: 0 }).unwrap();
        assert_eq!(s.round53(), Some((1 << 52, -52)));
        // 2^53 + 1: a tie between 2^53 and 2^53 + 2 → even (2^53).
        let mut s = ExactSum::default();
        s.add(Dyadic {
            m: (1 << 53) + 1,
            e: 0,
        })
        .unwrap();
        assert_eq!(s.round53(), Some((1 << 52, 1)));
        // 2^53 + 3: a tie → 2^53 + 4.
        let mut s = ExactSum::default();
        s.add(Dyadic {
            m: (1 << 53) + 3,
            e: 0,
        })
        .unwrap();
        assert_eq!(s.round53(), Some(((1 << 52) + 2, 1)));
        // 2^53 + 1 + 2^-200: above the tie → up.
        s = ExactSum::default();
        s.add(Dyadic {
            m: (1 << 53) + 1,
            e: 0,
        })
        .unwrap();
        s.add(Dyadic { m: 1, e: -200 }).unwrap();
        assert_eq!(s.round53(), Some(((1 << 52) + 1, 1)));
        // 2^54 − 1 rounds up to 2^54 (carry).
        s = ExactSum::default();
        s.add(Dyadic {
            m: (1 << 54) - 1,
            e: 0,
        })
        .unwrap();
        assert_eq!(s.round53(), Some((1 << 52, 2)));
        assert_eq!(ExactSum::default().round53(), None);
    }

    #[test]
    fn in_place_lowering_matches_a_fixed_base_sum() {
        // Terms in an order that lowers β by every shift class (0..64 bits, whole limbs);
        // the reference sums the same terms at a fixed base below all of them.
        const BASE: i64 = -400;
        let mut state = 0x9E37_79B9_7F4A_7C15u64;
        let mut next = || {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            state
        };
        let mut sum = ExactSum::default();
        let mut reference = Vec::new();
        let mut absorbed = ExactSum::default();
        let mut part = ExactSum::default();
        for k in 0..400u64 {
            let m = next() | 1;
            let e = 300 - (k as i64 * 67) % 640 + (next() % 3) as i64;
            sum.add(Dyadic { m, e }).unwrap();
            add_shifted(&mut reference, &[m], (e - BASE) as u64).unwrap();
            part.add(Dyadic { m, e }).unwrap();
            if k % 7 == 6 {
                absorbed.merge(&std::mem::take(&mut part)).unwrap();
            }
        }
        absorbed.merge(&part).unwrap();
        let mut seen = Vec::new();
        add_shifted(&mut seen, &sum.limbs, (sum.beta - BASE) as u64).unwrap();
        assert_eq!(seen, reference);
        assert_eq!(absorbed, sum);
        assert_eq!(absorbed.round53(), sum.round53());
    }

    // ------------------------------------------------------------------ the width limit

    const W: usize = MAX_LIMBS;
    const LIMIT_MESSAGE: &str = "unable to reserve the exact row sum of lse_dg_kcal \
                                 (width above 262144 limbs); no result was published";

    fn assert_limit(result: Result<(), EngineError>, what: &str) {
        let error = result.expect_err(what);
        assert!(matches!(error, EngineError::Device(_)), "{what}: {error:?}");
        assert_eq!(error.to_string(), LIMIT_MESSAGE, "{what}");
    }

    /// The §9.4 bound on the memory a row's sum holds, after any operation.
    fn assert_capacity(sum: &ExactSum, what: &str) {
        assert!(
            sum.limbs.capacity() <= W + 2,
            "{what}: capacity {} limbs above W_max + 2",
            sum.limbs.capacity()
        );
    }

    /// duplex-pf.md §13.5 (A) 13, relative to a base β: t₀ = 2^52·2^β fixes β; t₁ =
    /// (2^53 − 1)·2^(β + 64·W − 53) has the top bit of limb W; t₂ = 2^52·2^(β + 64·W − 53)
    /// carries across that limb boundary when added to t₁; t₁↑ is t₁ one bit higher.
    fn constructed(beta: i64) -> [Dyadic; 4] {
        let high = beta + 64 * W as i64 - 53;
        [
            Dyadic {
                m: 1 << 52,
                e: beta,
            },
            Dyadic {
                m: (1 << 53) - 1,
                e: high,
            },
            Dyadic {
                m: 1 << 52,
                e: high,
            },
            Dyadic {
                m: (1 << 53) - 1,
                e: high + 1,
            },
        ]
    }

    fn permutations(items: &[usize]) -> Vec<Vec<usize>> {
        if items.len() <= 1 {
            return vec![items.to_vec()];
        }
        let mut out = Vec::new();
        for k in 0..items.len() {
            let mut rest = items.to_vec();
            let first = rest.remove(k);
            for mut tail in permutations(&rest) {
                tail.insert(0, first);
                out.push(tail);
            }
        }
        out
    }

    /// Adds `order`'s terms one by one; every strict prefix of the cases below is narrower
    /// than the limit, so only the last addition may raise. Returns the last result.
    fn add_all(
        terms: &[Dyadic],
        order: &[usize],
        what: &str,
    ) -> (ExactSum, Result<(), EngineError>) {
        let mut sum = ExactSum::default();
        for (k, &t) in order.iter().enumerate() {
            let result = sum.add(terms[t]);
            assert_capacity(&sum, what);
            if k + 1 == order.len() {
                return (sum, result);
            }
            result.unwrap_or_else(|e| panic!("{what}: prefix {:?} raised {e:?}", &order[..=k]));
        }
        (sum, Ok(()))
    }

    /// Every order of the terms, and every split into two nonempty partial sums merged both
    /// ways; `width` is the final width, the limit raises iff it exceeds W. Returns the
    /// common sum when it does not.
    fn check_every_order(
        terms: &[Dyadic],
        indices: &[usize],
        width: usize,
        what: &str,
    ) -> Option<ExactSum> {
        let raises = width > W;
        let mut sums = Vec::new();
        for order in permutations(indices) {
            let what = format!("{what}, order {order:?}");
            let (sum, result) = add_all(terms, &order, &what);
            if raises {
                assert_limit(result, &what);
                continue;
            }
            result.unwrap_or_else(|e| panic!("{what}: raised {e:?}"));
            assert_eq!(sum.limbs.len(), width, "{what}");
            sums.push(sum);
        }
        for mask in 1..(1usize << indices.len()) - 1 {
            let (left, right): (Vec<usize>, Vec<usize>) =
                (0..indices.len()).partition(|&k| (mask >> k) & 1 == 1);
            let left: Vec<usize> = left.iter().map(|&k| indices[k]).collect();
            let right: Vec<usize> = right.iter().map(|&k| indices[k]).collect();
            for (into, from) in [(&left, &right), (&right, &left)] {
                let what = format!("{what}, merge of {from:?} into {into:?}");
                let (mut sum, result) = add_all(terms, into, &what);
                result.unwrap();
                let (part, result) = add_all(terms, from, &what);
                result.unwrap();
                let result = sum.merge(&part);
                assert_capacity(&sum, &what);
                if raises {
                    assert_limit(result, &what);
                    continue;
                }
                result.unwrap_or_else(|e| panic!("{what}: raised {e:?}"));
                assert_eq!(sum.limbs.len(), width, "{what}");
                sums.push(sum);
            }
        }
        if raises {
            return None;
        }
        for sum in &sums[1..] {
            assert_eq!(sum, &sums[0], "{what}: order-dependent sum");
        }
        Some(sums.swap_remove(0))
    }

    #[test]
    fn the_width_limit_raises_iff_the_final_width_exceeds_w_max() {
        let bases = [0, -1000, 7, 1 << 40, i64::MIN + 3, i64::MAX - (1 << 25)];
        for beta in bases {
            let terms = constructed(beta);
            let high = beta + 64 * W as i64 - 53;
            let what = |case: &str| format!("β = {beta}, {case}");
            // {t₀, t₁}: exactly W limbs, just below the limit: computes, and the value is
            // the exact sum (limb 0 holds t₀, limb W − 1 the top 53 bits of t₁).
            let sum = check_every_order(&terms, &[0, 1], W, &what("{t0, t1}")).unwrap();
            assert_eq!(sum.beta, beta);
            assert_eq!(sum.limbs[0], 1 << 52);
            assert!(sum.limbs[1..W - 1].iter().all(|&l| l == 0));
            assert_eq!(sum.limbs[W - 1], ((1u64 << 53) - 1) << 11);
            // RN₅₃: t₀ lies far below half an ulp of t₁, so the sum rounds to t₁.
            assert_eq!(sum.round53(), Some(((1 << 53) - 1, high)));
            // {t₀, t₂}: also W limbs.
            let sum = check_every_order(&terms, &[0, 2], W, &what("{t0, t2}")).unwrap();
            assert_eq!(sum.round53(), Some((1 << 52, high)));
            // {t₀, t₁, t₂}: W + 1 limbs through the carry alone: raises.
            check_every_order(&terms, &[0, 1, 2], W + 1, &what("{t0, t1, t2}"));
            // {t₀, t₁↑}: W + 1 limbs without a carry: raises.
            check_every_order(&terms, &[0, 3], W + 1, &what("{t0, t1 up}"));
            // {t₁, t₂} alone: 54 bits, one limb; 2^53 + 2^52 − 1 is a tie → even.
            let sum = check_every_order(&terms, &[1, 2], 1, &what("{t1, t2}")).unwrap();
            assert_eq!(sum.round53(), Some((3 << 51, high + 1)));
        }
    }

    /// A gap that alone makes the result too wide raises before anything is reserved, in
    /// both directions (the new term below β: lowering; above: the addition) and for a
    /// merge. Gaps are either just above 64·W bits or far beyond any memory, never between.
    #[test]
    fn a_too_wide_gap_raises_before_any_reservation() {
        let one = |e: i64| Dyadic { m: 1, e };
        let gap = 64 * W as i64; // 1 + gap bits: W + 1 limbs
        for (first, second) in [
            (one(0), one(-gap)),
            (one(0), one(gap)),
            (one(1 << 62), one(-(1 << 62))),
            (one(-(1 << 62)), one(1 << 62)),
            (one(i64::MAX), one(i64::MIN)),
            (one(i64::MIN), one(i64::MAX)),
        ] {
            let what = format!("{first:?} then {second:?}");
            let mut sum = ExactSum::default();
            sum.add(first).unwrap();
            let before = (sum.clone(), sum.limbs.capacity());
            assert_limit(sum.add(second), &what);
            assert_eq!(
                (sum.clone(), sum.limbs.capacity()),
                before,
                "{what}: touched"
            );
            let mut other = ExactSum::default();
            other.add(second).unwrap();
            assert_limit(sum.merge(&other), &what);
            assert_eq!(
                (sum.clone(), sum.limbs.capacity()),
                before,
                "{what}: merge touched"
            );
        }
        // One bit narrower: exactly W limbs, computes, in both directions.
        for (first, second) in [(one(0), one(1 - gap)), (one(0), one(gap - 1))] {
            let mut sum = ExactSum::default();
            sum.add(first).unwrap();
            sum.add(second).unwrap();
            assert_eq!(sum.limbs.len(), W);
            assert_capacity(&sum, "one bit narrower");
            assert_eq!(sum.beta, first.e.min(second.e));
            assert_eq!(sum.round53(), Some((1 << 52, first.e.max(second.e) - 52)));
        }
    }

    #[test]
    fn a_huge_gap_is_the_width_limit() {
        // Released as the allocator's "(width)" refusal; the width limit now decides first.
        let mut s = ExactSum::default();
        s.add(Dyadic { m: 1, e: 1 << 62 }).unwrap();
        let error = s
            .add(Dyadic {
                m: 1,
                e: -(1 << 62),
            })
            .unwrap_err();
        assert!(matches!(error, EngineError::Device(_)), "{error:?}");
        // The message survives the fallible path (crate::detail::device_error).
        assert_eq!(error.to_string(), LIMIT_MESSAGE);
    }
}
