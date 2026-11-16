//! Edit-distance family: exact byte Hamming and unit-cost Levenshtein.
//! Spec: docs/wiki/engine/engine-layout.md §4, engine/planner-scheduler.md and
//! edit/levenshtein.md.
pub mod levenshtein;
pub use levenshtein::Levenshtein;
mod rect;

use std::ops::ControlFlow;

use seqblaze_core::{
    kernel::PairDistanceKernel,
    plan::BoundAvailability,
    sched::Workers,
    seq::{ByteView, Composition, SeqTable, BLOCK},
    validate::EngineError,
};

pub struct Hamming;

impl PairDistanceKernel for Hamming {
    fn bound_availability(&self) -> BoundAvailability {
        BoundAvailability::Available
    }

    fn prepare_bound(
        &self,
        a: &SeqTable,
        b: &SeqTable,
        workers: Workers,
    ) -> Result<(), EngineError> {
        a.prepare_composition(workers)?;
        if !std::ptr::eq(a, b) {
            b.prepare_composition(workers)?;
        }
        Ok(())
    }

    /// `sum_c |n_c(a) - n_c(b)| / 2` over all 256 byte values. Clean tables keep
    /// only ACGT counts (every other count is zero), which gives the same sum.
    fn lower_bound(&self, a: &SeqTable, b: &SeqTable, i: usize, j: usize) -> i32 {
        let (Some(left), Some(right)) = (a.composition(), b.composition()) else {
            return 0; // Unprepared: the trivial bound never rejects a pair.
        };
        let sum = match (left, right) {
            (Composition::Acgt(x), Composition::Acgt(y)) => acgt_difference(&x[i], &y[j]),
            (Composition::Acgt(x), Composition::Bytes(y)) => mixed(&x[i], &y[j], b.seq_len(j)),
            (Composition::Bytes(x), Composition::Acgt(y)) => mixed(&y[j], &x[i], a.seq_len(i)),
            (Composition::Bytes(x), Composition::Bytes(y)) => x[i]
                .iter()
                .zip(&y[j])
                .map(|(&p, &q)| u64::from(p.abs_diff(q)))
                .sum(),
        };
        // Lowering a lower bound keeps it valid, so saturate rather than wrap.
        i32::try_from(sum / 2).unwrap_or(i32::MAX)
    }

    /// Staged rows are slices: the slice comparison, not the word fold.
    fn stages_rows(&self) -> bool {
        true
    }

    /// Rectangles of [`rect::CELLS`] cells for rows of at most [`rect::MAX_LEN`]
    /// bytes (host-sinks.md, "Implicit shapes on cheap kernels"); longer rows
    /// keep the per-pair path. The cutoff test is exact (`evaluate` returns
    /// the count when it is at most the cutoff, else None), and a rectangle's
    /// count needs no per-pair bound. Scheduling only: never an output.
    fn rect_cells(&self, a: &SeqTable, b: &SeqTable, _bounded: bool) -> usize {
        if a.mean_len().max(b.mean_len()) <= rect::MAX_LEN {
            rect::CELLS
        } else {
            0
        }
    }

    /// Each cell equals `evaluate`'s ([`rect`]).
    fn evaluate_rect(
        &self,
        rows: &[ByteView<'_>],
        cols: &[ByteView<'_>],
        cutoff: Option<i32>,
        out: &mut Vec<Option<i32>>,
    ) -> Result<(), EngineError> {
        rect::score(rows, cols, cutoff, out)
    }

    /// Allocation-free: never an error.
    fn evaluate(
        &self,
        a: ByteView<'_>,
        b: ByteView<'_>,
        cutoff: Option<i32>,
    ) -> Result<Option<i32>, EngineError> {
        let n = a.len().min(b.len());
        let limit = cutoff.map_or(i64::MAX, i64::from);
        let distance = match (a.as_slice(), b.as_slice()) {
            (Some(x), Some(y)) => Compare::slices(&x[..n], &y[..n], limit),
            _ => word_distance(a, b, n, limit),
        };
        // The Hamming length cap (2**31 - 1 bytes) keeps every distance in int32.
        Ok(distance.map(|d| i32::try_from(d).expect("Hamming length cap bounds the distance")))
    }
}

/// How two slices are compared: rows of at least one 64-byte block with
/// AVX-512 (BW) mask comparisons where the build and CPU have them, else (and
/// for shorter rows, before any feature check) the portable blocks. Every form
/// gives every count.
#[derive(Clone, Copy)]
enum Compare {
    Portable,
    #[cfg(all(target_arch = "x86_64", seqblaze_avx512))]
    Avx512,
}

impl Compare {
    #[inline]
    fn detect() -> Self {
        #[cfg(all(target_arch = "x86_64", seqblaze_avx512))]
        if std::is_x86_feature_detected!("avx512f") && std::is_x86_feature_detected!("avx512bw") {
            return Self::Avx512;
        }
        Self::Portable
    }

    /// Mismatches of two equal-length slices, or None above `limit`.
    #[inline(always)]
    fn slices(x: &[u8], y: &[u8], limit: i64) -> Option<u64> {
        if x.len() < 64 {
            return slice_distance(x, y, limit);
        }
        Self::detect().compare(x, y, limit)
    }

    #[inline(always)]
    fn compare(self, x: &[u8], y: &[u8], limit: i64) -> Option<u64> {
        match self {
            Self::Portable => slice_distance(x, y, limit),
            // SAFETY: `detect` returns Avx512 only when the CPU has AVX-512 F and BW.
            #[cfg(all(target_arch = "x86_64", seqblaze_avx512))]
            Self::Avx512 => unsafe { avx512::slice_distance(x, y, limit) },
        }
    }
}

/// Compiled only by a rustc with the AVX-512 intrinsics (stable since 1.89;
/// `build.rs` sets `seqblaze_avx512`).
#[cfg(all(target_arch = "x86_64", seqblaze_avx512))]
#[allow(clippy::incompatible_msrv)]
mod avx512 {
    use std::arch::x86_64::*;

    /// [`super::slice_distance`] with 64-byte mask comparisons: whole blocks,
    /// then the tail with a masked load of its bytes only; the cutoff is
    /// checked once per block, as there.
    ///
    /// # Safety
    /// The CPU must have AVX-512 F and BW.
    #[target_feature(enable = "avx512f,avx512bw")]
    pub(super) unsafe fn slice_distance(x: &[u8], y: &[u8], limit: i64) -> Option<u64> {
        debug_assert_eq!(x.len(), y.len());
        let n = x.len().min(y.len());
        let (p, q) = (x.as_ptr(), y.as_ptr());
        let mut distance = 0u64;
        let mut at = 0;
        while at + 64 <= n {
            // SAFETY: bytes at..at + 64 lie in both slices; loadu permits any
            // alignment.
            let mask = unsafe {
                _mm512_cmpneq_epi8_mask(
                    _mm512_loadu_si512(p.add(at).cast()),
                    _mm512_loadu_si512(q.add(at).cast()),
                )
            };
            distance += u64::from(mask.count_ones());
            if distance as i64 > limit {
                return None;
            }
            at += 64;
        }
        if at < n {
            let keep: __mmask64 = (1u64 << (n - at)) - 1;
            // SAFETY: the mask selects bytes at..n of both slices only; a
            // masked load reads no byte outside the mask.
            let mask = unsafe {
                _mm512_mask_cmpneq_epi8_mask(
                    keep,
                    _mm512_maskz_loadu_epi8(keep, p.add(at).cast()),
                    _mm512_maskz_loadu_epi8(keep, q.add(at).cast()),
                )
            };
            distance += u64::from(mask.count_ones());
        }
        (distance as i64 <= limit).then_some(distance)
    }
}

const BASES: [usize; 4] = [b'A' as usize, b'C' as usize, b'G' as usize, b'T' as usize];

fn acgt_difference(x: &[u32; 4], y: &[u32; 4]) -> u64 {
    x.iter()
        .zip(y)
        .map(|(&p, &q)| u64::from(p.abs_diff(q)))
        .sum()
}

/// Clean row (ACGT counts) against a raw row: the clean row has zero of every
/// other byte, so those bytes contribute the raw row's non-ACGT total.
fn mixed(clean: &[u32; 4], raw: &[u32; 256], raw_len: usize) -> u64 {
    let mut acgt = 0u64;
    let mut sum = 0u64;
    for (&count, &byte) in clean.iter().zip(&BASES) {
        acgt += u64::from(raw[byte]);
        sum += u64::from(count.abs_diff(raw[byte]));
    }
    sum + (raw_len as u64).saturating_sub(acgt)
}

// A block count fits in u8, so per-block sums cannot overflow.
const _: () = assert!(BLOCK < 256);

#[inline]
fn block_mismatches(x: &[u8], y: &[u8]) -> u64 {
    // Wrapping is unreachable (at most BLOCK < 256 terms); it keeps the loop
    // free of overflow branches so it vectorizes in checked builds too.
    let count = x
        .iter()
        .zip(y)
        .fold(0u8, |sum, (p, q)| sum.wrapping_add(u8::from(p != q)));
    u64::from(count)
}

/// Mismatches of two equal-length slices, or None once the running count
/// exceeds `limit`; the cutoff is checked once per block, not per byte.
fn slice_distance(x: &[u8], y: &[u8], limit: i64) -> Option<u64> {
    let mut distance = 0u64;
    let mut left = x.chunks_exact(BLOCK);
    let mut right = y.chunks_exact(BLOCK);
    for (p, q) in (&mut left).zip(&mut right) {
        distance += block_mismatches(p, q);
        if distance as i64 > limit {
            return None;
        }
    }
    distance += tail_mismatches(left.remainder(), right.remainder());
    (distance as i64 <= limit).then_some(distance)
}

const LOW7: u64 = 0x7f7f_7f7f_7f7f_7f7f;
const ONES: u64 = 0x0101_0101_0101_0101;

/// One bit (bit 0 of each byte) per nonzero byte of `d`: bit 7 of each byte
/// of `(d & 0x7f..) + 0x7f.. | d` is set iff that byte of `d` is nonzero, and
/// the low seven bits of no byte carry into the next.
#[inline(always)]
fn nonzero_bytes(d: u64) -> u64 {
    ((((d & LOW7) + LOW7) | d) >> 7) & ONES
}

/// Mismatches of two equal-length slices shorter than a block: whole words
/// first (per-byte counts, at most 7, summed once), then the last few bytes.
#[inline]
fn tail_mismatches(x: &[u8], y: &[u8]) -> u64 {
    debug_assert!(x.len() == y.len() && x.len() < BLOCK);
    let (mut p, mut q) = (x.chunks_exact(8), y.chunks_exact(8));
    let mut lanes = 0u64;
    for (u, v) in (&mut p).zip(&mut q) {
        lanes += nonzero_bytes(
            u64::from_ne_bytes(u.try_into().expect("8 bytes"))
                ^ u64::from_ne_bytes(v.try_into().expect("8 bytes")),
        );
    }
    let rest = p
        .remainder()
        .iter()
        .zip(q.remainder())
        .filter(|(u, v)| u != v)
        .count();
    (lanes.wrapping_mul(ONES) >> 56) + rest as u64
}

/// Same count when either row of length `n` is borrowed NumPy storage: the
/// rows' 8-byte words, zero past byte n in both (`ByteView::fold_words`,
/// relaxed loads for a borrowed row), XORed and counted in registers; the
/// cutoff is checked per word.
fn word_distance(a: ByteView<'_>, b: ByteView<'_>, n: usize, limit: i64) -> Option<u64> {
    let distance = a.fold_words(b, n, 0u64, |distance, x, y| {
        // At most 8 per word: the byte sum of the lanes is its top byte.
        let distance = distance + (nonzero_bytes(x ^ y).wrapping_mul(ONES) >> 56);
        if distance as i64 > limit {
            ControlFlow::Break(distance)
        } else {
            ControlFlow::Continue(distance)
        }
    });
    (distance as i64 <= limit).then_some(distance)
}

#[cfg(test)]
mod tests {
    use super::*;
    use seqblaze_core::{batch::SeqBatch, sched::Workers};

    fn table(rows: &[&[u8]]) -> SeqTable {
        SeqTable::new(SeqBatch::from_seqs(rows.iter().copied()))
    }

    fn naive(x: &[u8], y: &[u8]) -> i32 {
        x.iter().zip(y).filter(|(p, q)| p != q).count() as i32
    }

    fn composition_bound(x: &[u8], y: &[u8]) -> i32 {
        let mut counts = [0i64; 256];
        for &p in x {
            counts[p as usize] += 1;
        }
        for &q in y {
            counts[q as usize] -= 1;
        }
        (counts.iter().map(|c| c.unsigned_abs()).sum::<u64>() / 2) as i32
    }

    #[test]
    fn compact_and_raw_compositions_give_the_full_256_value_bound() {
        let clean: [&[u8]; 3] = [b"AACCGGTT", b"TTTTTTTT", b"ACGTACGT"];
        let raw: [&[u8]; 3] = [b"AAxx\0\xffTT", b"acgtACGT", b"TTTTTTTT"];
        let (c, r) = (table(&clean), table(&raw));
        for t in [&c, &r] {
            t.alphabet_scan(1).unwrap();
            t.prepare_composition(Workers::exact(2)).unwrap();
        }
        assert!(matches!(c.composition(), Some(Composition::Acgt(_))));
        assert!(matches!(r.composition(), Some(Composition::Bytes(_))));
        for (left, lrows) in [(&c, &clean), (&r, &raw)] {
            for (right, rrows) in [(&c, &clean), (&r, &raw)] {
                for (i, x) in lrows.iter().enumerate() {
                    for (j, y) in rrows.iter().enumerate() {
                        let expected = composition_bound(x, y);
                        assert_eq!(Hamming.lower_bound(left, right, i, j), expected);
                        assert!(expected <= naive(x, y));
                    }
                }
            }
        }
    }

    /// Every slice comparison form gives the portable count and the same
    /// inclusive cutoff, across 64-byte block and tail boundaries and raw bytes.
    #[test]
    fn slice_forms_match_the_portable_count() {
        let forms = [Compare::Portable, Compare::detect()];
        for n in 0..=200usize {
            let x: Vec<u8> = (0..n).map(|t| (t * 37 + 11) as u8).collect();
            for flips in [0usize, 1, 3, 64] {
                let mut y = x.clone();
                for t in (0..n).step_by(n / flips.max(1) + 1).take(flips.max(1)) {
                    y[t] ^= [0x80, 1, 0x40][t % 3];
                }
                let d = naive(&x, &y) as i64;
                for limit in [-1, 0, d - 1, d, d + 1, i64::MAX] {
                    let expected = (d <= limit).then_some(d as u64);
                    for form in forms {
                        assert_eq!(form.compare(&x, &y, limit), expected, "n={n} limit={limit}");
                    }
                    assert_eq!(
                        Compare::slices(&x, &y, limit),
                        expected,
                        "n={n} limit={limit}"
                    );
                }
            }
        }
    }

    /// Owned slices and borrowed fixed-width rows (relaxed word loads) agree.
    #[test]
    fn block_evaluation_matches_naive_distance_and_inclusive_cutoff() {
        let lengths = [0usize, 1, 7, 8, 9, 30, 63, 64, 65, 130, 9, 64];
        let width = 130;
        let mut rows: Vec<Vec<u8>> = Vec::new();
        for (k, &len) in lengths.iter().enumerate() {
            if k < 10 {
                rows.push((0..len).map(|t| b"ACGT"[t % 4]).collect());
                rows.push((0..len).map(|t| b"ACGT"[(t * 7 / 3) % 4]).collect());
            } else {
                // Raw bytes (high bit included) differing in every bit position;
                // no row ends in NUL, so the padded copies trim back to them.
                let x: Vec<u8> = (0..len).map(|t| (t * 37 + 11) as u8 | 1).collect();
                let y = x
                    .iter()
                    .enumerate()
                    .map(|(t, &p)| p ^ [0x80, 0, 0x40, 0x7e][t % 4])
                    .collect();
                rows.push(x);
                rows.push(y);
            }
        }
        let refs: Vec<&[u8]> = rows.iter().map(Vec::as_slice).collect();
        let owned = table(&refs);
        // NUL-padded fixed-width rows, as NumPy S<width> stores them.
        let mut padded = vec![0u8; rows.len() * width];
        for (r, row) in rows.iter().enumerate() {
            padded[r * width..r * width + row.len()].copy_from_slice(row);
        }
        let address = padded.as_ptr() as usize;
        // SAFETY: the boxed Vec owns the described rows and is never mutated.
        let fixed = unsafe {
            SeqTable::from_fixed_bytes(address, rows.len(), width, width as isize, Box::new(padded))
        }
        .unwrap();
        for (r, &length) in lengths.iter().enumerate() {
            let (ia, ib) = (2 * r, 2 * r + 1);
            let d = naive(&rows[ia], &rows[ib]);
            for (x, y) in [
                (owned.get(ia), owned.get(ib)),
                (owned.get(ia), fixed.get(ib)),
                (fixed.get(ia), owned.get(ib)),
                (fixed.get(ia), fixed.get(ib)),
            ] {
                assert_eq!(x.len(), length);
                assert_eq!(Hamming.evaluate(x, y, None).unwrap(), Some(d));
                for cutoff in [-1, 0, d - 1, d, d + 1] {
                    let expected = (d <= cutoff).then_some(d);
                    assert_eq!(
                        Hamming.evaluate(x, y, Some(cutoff)).unwrap(),
                        expected,
                        "d={d} c={cutoff}"
                    );
                }
            }
        }
    }
}

