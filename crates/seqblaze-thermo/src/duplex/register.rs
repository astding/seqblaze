//! `sb.tm` on a `Pairs` input: the register pairing of two equal-length strands
//! (K1 QK25 (a); spec docs/wiki/thermo/tm.md §13).
//!
//! No dynamic programming and no change to K1's recurrence. Per pair: one pass over the
//! columns r = 0 … L − 1, whose opposition is (a_r, b_{L−1−r}) (§13.1); the columns
//! whose opposition has a pair type of the pack (Watson–Crick, plus G·T under `pairs
//! wc+gt`; nndb2024-dna.md §5.3.5) form the structure S(a, b), and its key (D, H, G, Q) is the sum
//! of the same per-call cell keys K1's evaluator reads (duplex-mfe.md §12.4). Outside
//! K1's space (k < 2, or a gap g with 2g > Λ) the key is +∞ and `DuplexCall::row` gives
//! K1's sentinel (QK9, PR2); inside it, T1's finalize F1–F10 (§13.2).
//!
//! The engine selects with `key <= cutoff` on an integer key, while `tm_c` is a binary64
//! compared with `>=`. The key here is the int64 order-preserving image of −`tm_c`, built
//! from the bits (−0.0 reads as +0.0), with every NaN mapped above every other key, so a
//! row passes iff `tm_c >= c` in IEEE comparison and a NaN never passes (§13.5, §13.11).
//! The published columns are the row's own F10 bits, never values rebuilt from the key,
//! with one exception: RowStats' `best_tm_c` (RowStats has no detail rows) is
//! [`tm_of_key`] of the row's least key, by bit operations only. It equals F10's bits of
//! the best partner under the four preconditions of §13.11: F10 runs under
//! roundTiesToEven; K0 ≠ 0 and F10 is the last operation on `tm_c`; every NaN `tm_c` is
//! the canonical NaN; the engine's mask and [`NAN_KEY`] lie outside the image of
//! [`key_of_tm`]. A change that breaks one of them must re-prove that equality or carry
//! the best partner's F10 bits beside its key instead.
use std::sync::Arc;

use seqblaze_core::{
    detail::{DetailColumns, DetailField, DetailRequest, CANONICAL_NAN},
    kernel::{PairDistanceKernel, Score},
    plan::BoundAvailability,
    sched::Workers,
    seq::{ByteView, SeqTable},
    validate::{self, EngineError},
    work::WorkSource,
};

use super::{
    codes,
    dp::{Full, Key},
    symmetry, DuplexCall, DuplexRow, MAX_LENGTH,
};
use crate::tm::{self as t1, DG_DCAL, DG_KCAL, DH_KCAL, DS_CAL_PER_K, TM_C};

/// T1's five fields in the slot order of the device call (docs/wiki/gpu/tm-pairs-cuda.md
/// §4): `tm_c`, `dg_kcal`, `dg_dcal`, `dh_kcal`, `ds_cal_per_k`, as indices of
/// [`TmPairs`]' `detail_fields()` (T1's [`t1::FIELDS`]).
pub const DEVICE_FIELDS: [usize; 5] = [TM_C, DG_KCAL, DG_DCAL, DH_KCAL, DS_CAL_PER_K];

/// The key of a NaN `tm_c` (a sentinel row, or a 0/0 in F9): above the image of every
/// other binary64, so it fails every finite cutoff; distinct from the engine's mask.
pub const NAN_KEY: i64 = i64::MAX - 1;

const SIGN: u64 = 1 << 63;
const INFINITY_BITS: u64 = 0x7FF0_0000_0000_0000;

/// The selecting key of a `tm_c` value: the order-preserving int64 image of −`tm_c`
/// (its bits when the sign bit is clear, minus its magnitude bits otherwise), −0.0 read
/// as +0.0, NaN mapped to [`NAN_KEY`]. Bit operations only. The facade maps a cutoff c
/// the same way, so `key(tm_c) <= key(c)` iff `tm_c >= c`.
#[inline]
pub fn key_of_tm(tm_c: f64) -> i64 {
    let bits = tm_c.to_bits();
    if bits & !SIGN > INFINITY_BITS {
        return NAN_KEY;
    }
    let negated = bits ^ SIGN;
    let magnitude = negated & !SIGN;
    if negated & SIGN == 0 {
        magnitude as i64
    } else {
        // −0.0 and +0.0 both give 0.
        -(magnitude as i64)
    }
}

/// The `tm_c` of a selecting key: the inverse of [`key_of_tm`], by bit operations only.
/// The engine's mask (`i64::MAX`, a row with no passing partner) and [`NAN_KEY`] give the
/// canonical NaN; key 0 gives +0.0. It returns F10's own bits for every key the kernel
/// emits: F10 is `x6 − K0` with K0 ≠ 0, which under round-to-nearest-even is +0.0 when the
/// operands are equal and never −0.0, and its NaN is the canonical NaN; so the one value
/// `key_of_tm` merges (−0.0 with +0.0) never occurs. Used for RowStats' `best_tm_c`, the
/// one `tm` on Pairs column without a detail row (tm.md §13.5).
#[inline]
pub fn tm_of_key(key: i64) -> f64 {
    if key == <i64 as Score>::MASK || key == NAN_KEY {
        CANONICAL_NAN
    } else if key > 0 {
        // −tm_c had its sign bit clear: tm_c is negative (or −∞).
        f64::from_bits(key as u64 | SIGN)
    } else {
        // +0.0 for key 0; a positive tm_c (or +∞) otherwise.
        f64::from_bits(key.unsigned_abs())
    }
}

/// [`tm_of_key`] over a key column, in the keys' buffer (a bit map between same-size
/// types), so a RowStats export needs no second column.
pub fn values_of(keys: Vec<i64>) -> Vec<f64> {
    keys.into_iter().map(tm_of_key).collect()
}

/// A ⊥ cell on the register path: rule F cannot fail by construction (tm.md §13.1), so
/// this is an internal defect, never a sentinel.
fn bottom_cell(len: usize) -> EngineError {
    EngineError::Parameter(format!(
        "internal error: tm on Pairs read a ⊥ cell on the register structure of a pair of \
         length {len} (rule F holds by construction, docs/wiki/thermo/tm.md §13.1); nothing \
         was computed"
    ))
}

impl DuplexCall {
    /// (K, identical) of the register structure of the letter codes `x`, `y` (equal
    /// lengths, tm.md §13.1–§13.2), with K = +∞ when S(a, b) is outside K1's space.
    /// Integer arithmetic only: it performs no FP operation.
    pub(crate) fn register_key(&self, x: &[u8], y: &[u8]) -> Result<(Full, bool), EngineError> {
        let n = x.len();
        debug_assert_eq!(n, y.len());
        let (identical, sym) = symmetry(x, y);
        let cells = &self.full;
        let lam = cells.loop_max;
        let add = |key: Full, part: Full| {
            if part.finite() {
                Ok(key.plus(part))
            } else {
                Err(bottom_cell(n))
            }
        };
        let mut key = add(
            Full::default(),
            if sym { cells.init_sym } else { cells.init },
        )?;
        let mut first = None;
        let mut last = 0;
        for r in 0..n {
            // A pair iff the opposition has a pair type of the pack (nndb2024-dna.md
            // §5.3.5): Watson–Crick, plus G·T under `pairs wc+gt`.
            if cells.pair(x[r], y[n - 1 - r]) == super::pack::NO_PAIR {
                continue;
            }
            if first.is_none() {
                first = Some(r);
            } else {
                let gap = r - last - 1;
                if 2 * gap > lam {
                    return Ok((Full::INF, identical));
                }
                key = add(key, cells.step(x, y, (last, n - 1 - last), (r, n - 1 - r)))?;
            }
            last = r;
        }
        match first {
            // k >= 2 (QK8).
            Some(first) if first < last => {
                key = add(key, cells.left_end(x, y, first, n - 1 - first))?;
                key = add(key, cells.right_end(x, y, last, n - 1 - last))?;
                Ok((key, identical))
            }
            _ => Ok((Full::INF, identical)),
        }
    }
}

/// The pair kernel of `sb.tm` on `Pairs`: K1's per-call pack (built for
/// `Caller::TmPairs`) and the requested fields of T1 (`tm_c` always).
pub struct TmPairs {
    call: Arc<DuplexCall>,
    request: DetailRequest,
}

impl TmPairs {
    /// `fields` by name, of T1's [`t1::FIELDS`]; `tm_c`, the primary, is always emitted.
    pub fn new(call: Arc<DuplexCall>, fields: &[&str]) -> Result<Self, EngineError> {
        let mut request = DetailRequest::NONE.with(TM_C);
        for name in fields {
            let index = t1::FIELDS
                .iter()
                .position(|field| field.name == *name)
                .ok_or_else(|| EngineError::Parameter(format!("tm has no field {name:?}")))?;
            request = request.with(index);
        }
        Ok(Self { call, request })
    }

    pub fn call(&self) -> &DuplexCall {
        &self.call
    }

    /// The five fields of one validated pair (upper-case ACGT, equal lengths, at most
    /// 200 nt). Expects the IEEE default floating-point environment: it runs in the
    /// engine's guarded scheduler units and carries no guard of its own
    /// (text-nulls-fp-environment.md §7.3 item 1), like `DuplexCall::row`.
    fn row(&self, a: ByteView<'_>, b: ByteView<'_>) -> Result<DuplexRow, EngineError> {
        let (mut x, mut y) = ([0u8; MAX_LENGTH], [0u8; MAX_LENGTH]);
        let n = codes(a, &mut x);
        let m = codes(b, &mut y);
        if n != m {
            // Step 2 (equal lengths) ran before any pair is computed.
            return Err(EngineError::Pairing(format!(
                "internal error: tm on Pairs reached a pair of lengths {n} and {m}; nothing \
                 was computed"
            )));
        }
        let (key, identical) = self.call.register_key(&x[..n], &y[..m])?;
        Ok(self.call.row(key, identical))
    }

    fn push(&self, row: &DuplexRow, out: &mut DetailColumns) -> Result<(), EngineError> {
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

impl PairDistanceKernel<i64> for TmPairs {
    /// No bound (tm.md §13.5): `bounded` resolves to full with `no_bound`.
    fn bound_availability(&self) -> BoundAvailability {
        BoundAvailability::Unavailable
    }
    fn name(&self) -> &'static str {
        "tm"
    }
    /// Step 2 (tm.md §13.6, PR3): `hamming`'s equal-length rule with a `tm` message.
    fn validate_pairs(
        &self,
        a: &SeqTable,
        b: &SeqTable,
        work: &WorkSource,
    ) -> Result<(), EngineError> {
        validate::equal_lengths_for(
            a,
            b,
            work,
            "tm on Pairs requires equal lengths (register pairing)",
        )
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
                "tm input {input} sequence {index} has length {length}, above the native \
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
    /// One pass over the columns, O(k) cell lookups and the finalize.
    fn pair_cost(&self, a: &SeqTable, _b: &SeqTable, _with_details: bool) -> usize {
        a.mean_len().saturating_add(16)
    }
    fn evaluate(
        &self,
        a: ByteView<'_>,
        b: ByteView<'_>,
        _cutoff: Option<i64>,
    ) -> Result<Option<i64>, EngineError> {
        Ok(Some(key_of_tm(self.row(a, b)?.tm_c)))
    }
    fn detail_fields(&self) -> &'static [DetailField] {
        &t1::FIELDS
    }
    fn detail_request(&self) -> DetailRequest {
        self.request
    }
    /// Each publishes `tm_c` and the other fields from the detail rows (F10's bits) and
    /// never the key, so the Each sink keeps no key column (tm.md §13.11).
    fn each_keys(&self) -> bool {
        false
    }
    /// One row gives the key and the fields.
    fn fused_details(&self) -> bool {
        true
    }
    /// One row per pair gives the key and the fields; the fields are appended iff the
    /// key passes `keep`.
    fn evaluate_detailed(
        &self,
        a: ByteView<'_>,
        b: ByteView<'_>,
        _prune: Option<i64>,
        keep: Option<i64>,
        out: &mut DetailColumns,
    ) -> Result<Option<i64>, EngineError> {
        let row = self.row(a, b)?;
        let key = key_of_tm(row.tm_c);
        if keep.is_none_or(|limit| key <= limit) {
            self.push(&row, out)?;
        }
        Ok(Some(key))
    }
    fn details(
        &self,
        a: ByteView<'_>,
        b: ByteView<'_>,
        _score: i64,
        out: &mut DetailColumns,
    ) -> Result<(), EngineError> {
        let row = self.row(a, b)?;
        self.push(&row, out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// tm.md §13.5: key(x) <= key(c) iff x >= c in IEEE comparison, for finite c;
    /// −0.0 equals +0.0, +∞ passes and −∞ fails every finite c, NaN never passes.
    #[test]
    fn key_order_is_ieee_ge_and_nan_never_passes() {
        let values = [
            f64::NEG_INFINITY,
            -f64::MAX,
            -1.5,
            -f64::MIN_POSITIVE,
            -5e-324,
            -0.0,
            0.0,
            5e-324,
            f64::MIN_POSITIVE,
            1.5,
            f64::MAX,
            f64::INFINITY,
        ];
        let nans = [
            f64::NAN,
            -f64::NAN,
            f64::from_bits(0x7FF0_0000_0000_0001),
            f64::from_bits(0xFFFF_FFFF_FFFF_FFFF),
        ];
        for &c in values.iter().filter(|c| c.is_finite()) {
            for &x in &values {
                assert_eq!(key_of_tm(x) <= key_of_tm(c), x >= c, "x={x:e} c={c:e}");
            }
            for &x in &nans {
                assert!(key_of_tm(x) > key_of_tm(c), "NaN passes c={c:e}");
            }
        }
        assert_eq!(key_of_tm(0.0), key_of_tm(-0.0));
        assert!(nans.iter().all(|&x| key_of_tm(x) == NAN_KEY));
    }

    /// RowStats' `best_tm_c` (tm.md §13.5): `tm_of_key` returns the bits of every value
    /// F10 can produce (everything but −0.0, NaN only as the canonical NaN), maps the
    /// engine's mask and every NaN key to the canonical NaN, and reuses the key buffer.
    #[test]
    fn tm_of_key_inverts_key_of_tm_on_f10_values() {
        let exposed = [
            f64::NEG_INFINITY,
            -f64::MAX,
            -273.15,
            -1.5,
            -f64::MIN_POSITIVE,
            -5e-324,
            0.0,
            5e-324,
            f64::MIN_POSITIVE,
            1.5,
            61.732_412_5,
            f64::MAX,
            f64::INFINITY,
            CANONICAL_NAN,
        ];
        for &x in &exposed {
            assert_eq!(tm_of_key(key_of_tm(x)).to_bits(), x.to_bits(), "x={x:e}");
        }
        // The merged value reads back as +0.0; F10 never yields −0.0 under RNE: a
        // finalize whose F9 quotient equals K0 exactly gives +0.0.
        assert_eq!(tm_of_key(key_of_tm(-0.0)).to_bits(), 0);
        let near = (1000.0 / t1::K0).to_bits();
        let kappa = (0..64u64)
            .flat_map(|step| [near + step, near - step])
            .map(f64::from_bits)
            .find(|&kappa| 1000.0 / kappa == t1::K0)
            .expect("a kappa with 1000 / kappa == K0");
        // h = g = 100 and P = 0: ΔS = 0, so F9 = 1000 / kappa = K0 and F10 = K0 − K0.
        let row = t1::finalize_pairs(100, 100, 0.0, 0, 1.0, kappa);
        assert_eq!(row.tm_c.to_bits(), 0, "F10 gave {:e}", row.tm_c);
        for x in [f64::NAN, -f64::NAN, f64::from_bits(0x7FF0_0000_0000_0001)] {
            assert_eq!(tm_of_key(key_of_tm(x)).to_bits(), CANONICAL_NAN.to_bits());
        }
        assert_eq!(tm_of_key(i64::MAX).to_bits(), CANONICAL_NAN.to_bits());
        let mut keys: Vec<i64> = exposed.iter().map(|&x| key_of_tm(x)).collect();
        keys.push(i64::MAX);
        let address = keys.as_ptr() as usize;
        let values = values_of(keys);
        assert_eq!(values.as_ptr() as usize, address, "values_of reallocated");
        for (value, &x) in values.iter().zip(&exposed) {
            assert_eq!(value.to_bits(), x.to_bits());
        }
        assert_eq!(values[exposed.len()].to_bits(), CANONICAL_NAN.to_bits());
    }
}
