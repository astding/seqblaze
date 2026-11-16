//! Hamming rectangles (`PairDistanceKernel::evaluate_rect`; engine-layout.md
//! §4, host-sinks.md "Implicit shapes on cheap kernels"): every cell of a
//! block of rows × columns of one length, the rows and columns staged once per
//! block into zero-padded rows of a common stride, then each row compared with
//! every column in one tight loop.
//!
//! A cell's count is the number of byte positions below the stride where the
//! two staged rows differ. Both rows are their bytes followed by zeros up to
//! the stride, so positions past the shared length never differ and the count
//! is exactly `evaluate`'s mismatch count over the length. Bytes are compared
//! as bytes (any alphabet, raw bytes included). The cutoff test is inclusive,
//! as `evaluate`'s: `Some(d)` when `d <= cutoff`, else None. A block whose
//! rows and columns do not all share one length, or whose length exceeds
//! [`MAX_LEN`], goes through `evaluate` cell by cell. No form, stride or block
//! size changes a value.
use seqblaze_core::{detail::try_reserve_exact, seq::ByteView, validate::EngineError};

use super::{nonzero_bytes, ONES};

/// Cells per rectangle that the engine hands `evaluate_rect` (`rect_cells`).
pub(crate) const CELLS: usize = 16_384;
/// Longest row (bytes) that rectangles stage; longer rows keep the per-pair path.
pub(crate) const MAX_LEN: usize = 1024;
/// Rows and columns staged at once: at most `(ROW_BLOCK + COL_BLOCK) * MAX_LEN`
/// bytes of staging, whatever the rectangle's shape.
const ROW_BLOCK: usize = 64;
const COL_BLOCK: usize = 256;
const SCORES: &str = "hamming rectangle scores";
const STAGING: &str = "hamming rectangle staging";

/// How a staged block is compared. Every form gives every count.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Form {
    /// 8-byte words: any CPU.
    Portable,
    /// 32-byte AVX2 equality masks.
    #[cfg(target_arch = "x86_64")]
    Avx2,
    /// 64-byte AVX-512 (F, BW) inequality masks; only when compiled by rustc
    /// 1.89 or later (`build.rs` sets `seqblaze_avx512`).
    #[cfg(all(target_arch = "x86_64", seqblaze_avx512))]
    Avx512,
}

impl Form {
    /// The widest available form for rows of `len` bytes: AVX-512 above 32
    /// bytes, else AVX2 (a 32-byte stride), else 8-byte words.
    pub(crate) fn detect(len: usize) -> Self {
        #[cfg(all(target_arch = "x86_64", seqblaze_avx512))]
        if len > 32 && Self::Avx512.available() {
            return Self::Avx512;
        }
        #[cfg(target_arch = "x86_64")]
        if Self::Avx2.available() {
            return Self::Avx2;
        }
        let _ = len;
        Self::Portable
    }

    /// Whether this build and CPU can run the form.
    pub(crate) fn available(self) -> bool {
        match self {
            Self::Portable => true,
            #[cfg(target_arch = "x86_64")]
            Self::Avx2 => {
                std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("popcnt")
            }
            #[cfg(all(target_arch = "x86_64", seqblaze_avx512))]
            Self::Avx512 => {
                std::is_x86_feature_detected!("avx512f")
                    && std::is_x86_feature_detected!("avx512bw")
                    && std::is_x86_feature_detected!("popcnt")
            }
        }
    }

    /// Bytes per comparison step; the stride is a multiple of it.
    fn step(self) -> usize {
        match self {
            Self::Portable => 8,
            #[cfg(target_arch = "x86_64")]
            Self::Avx2 => 32,
            #[cfg(all(target_arch = "x86_64", seqblaze_avx512))]
            Self::Avx512 => 64,
        }
    }

    /// `out[c]` = the inclusive cutoff test of the count of staged row `x`
    /// against staged column `c` of `ys` (`out.len()` columns of `stride` bytes).
    #[inline]
    fn line(self, x: &[u8], ys: &[u8], stride: usize, limit: i64, out: &mut [Option<i32>]) {
        debug_assert!(x.len() == stride && ys.len() == out.len() * stride);
        match self {
            Self::Portable => portable::line(x, ys, stride, limit, out),
            // SAFETY: `available` checked AVX2 and POPCNT on this CPU
            // (`score` only runs available forms); the slices hold whole
            // strides, a multiple of 32 bytes.
            #[cfg(target_arch = "x86_64")]
            Self::Avx2 => unsafe { x86::line_avx2(x, ys, stride, limit, out) },
            // SAFETY: as above, AVX-512 F and BW; strides are multiples of 64.
            #[cfg(all(target_arch = "x86_64", seqblaze_avx512))]
            Self::Avx512 => unsafe { x86::line_avx512(x, ys, stride, limit, out) },
        }
    }
}

/// `evaluate(row, col, cutoff)` of every cell, appended row by row to `out`.
pub(crate) fn score(
    rows: &[ByteView<'_>],
    cols: &[ByteView<'_>],
    cutoff: Option<i32>,
    out: &mut Vec<Option<i32>>,
) -> Result<(), EngineError> {
    let len = rows.first().or(cols.first()).map_or(0, |view| view.len());
    score_with(Form::detect(len), rows, cols, cutoff, out)
}

/// [`score`] with the form `form` (available on this CPU).
pub(crate) fn score_with(
    form: Form,
    rows: &[ByteView<'_>],
    cols: &[ByteView<'_>],
    cutoff: Option<i32>,
    out: &mut Vec<Option<i32>>,
) -> Result<(), EngineError> {
    assert!(form.available(), "an available comparison form");
    let cells = rows.len().checked_mul(cols.len()).ok_or_else(|| {
        seqblaze_core::detail::allocation_error(SCORES, "the cell count overflows usize")
    })?;
    if cells == 0 {
        return Ok(());
    }
    try_reserve_exact(out, cells, SCORES)?;
    let len = rows[0].len();
    if len > MAX_LEN || rows.iter().chain(cols).any(|view| view.len() != len) {
        use seqblaze_core::kernel::PairDistanceKernel;
        for &x in rows {
            for &y in cols {
                out.push(super::Hamming.evaluate(x, y, cutoff)?);
            }
        }
        return Ok(());
    }
    let step = form.step();
    // len <= MAX_LEN: no overflow; an empty row is one step of zeros.
    let stride = len.div_ceil(step).max(1) * step;
    let limit = cutoff.map_or(i64::MAX, i64::from);
    let (row_block, col_block) = (ROW_BLOCK.min(rows.len()), COL_BLOCK.min(cols.len()));
    let mut staged = Vec::new();
    try_reserve_exact(&mut staged, (row_block + col_block) * stride, STAGING)?;
    staged.resize((row_block + col_block) * stride, 0u8);
    let (xs, ys) = staged.split_at_mut(row_block * stride);
    let base = out.len();
    out.resize(base + cells, None);
    let grid = &mut out[base..];
    let width = cols.len();
    for c0 in (0..width).step_by(col_block) {
        let c1 = (c0 + col_block).min(width);
        let ys = &mut ys[..(c1 - c0) * stride];
        stage(&cols[c0..c1], stride, ys);
        for r0 in (0..rows.len()).step_by(row_block) {
            let r1 = (r0 + row_block).min(rows.len());
            let xs = &mut xs[..(r1 - r0) * stride];
            stage(&rows[r0..r1], stride, xs);
            for (r, x) in (r0..r1).zip(xs.chunks_exact(stride)) {
                form.line(
                    x,
                    ys,
                    stride,
                    limit,
                    &mut grid[r * width + c0..r * width + c1],
                );
            }
        }
    }
    Ok(())
}

/// Each view's bytes, zero-padded to `stride` (a multiple of 8), back to back.
fn stage(views: &[ByteView<'_>], stride: usize, out: &mut [u8]) {
    for (view, row) in views.iter().zip(out.chunks_exact_mut(stride)) {
        view.copy_padded(0, row);
    }
}

mod portable {
    use super::{nonzero_bytes, ONES};

    #[inline]
    fn word(bytes: &[u8]) -> u64 {
        u64::from_ne_bytes(bytes.try_into().expect("8 bytes"))
    }

    /// Words of 8 bytes, the cutoff checked once per 64 bytes.
    pub(super) fn line(x: &[u8], ys: &[u8], stride: usize, limit: i64, out: &mut [Option<i32>]) {
        for (y, slot) in ys.chunks_exact(stride).zip(out) {
            let mut distance = 0u64;
            let mut over = false;
            for (p, q) in x.chunks(64).zip(y.chunks(64)) {
                for (u, v) in p.chunks_exact(8).zip(q.chunks_exact(8)) {
                    // At most 8 per word: the byte sum of the lanes is its top byte.
                    distance += nonzero_bytes(word(u) ^ word(v)).wrapping_mul(ONES) >> 56;
                }
                if distance as i64 > limit {
                    over = true;
                    break;
                }
            }
            // distance <= MAX_LEN: it fits i32.
            *slot = (!over).then_some(distance as i32);
        }
    }
}

#[cfg(target_arch = "x86_64")]
mod x86 {
    use std::arch::x86_64::*;

    /// 32-byte equality masks; with more than one step per row, the cutoff
    /// is checked once per 64 bytes.
    ///
    /// # Safety
    /// The CPU must have AVX2 and POPCNT; `stride` is a positive multiple of
    /// 32, `x` holds `stride` bytes and `ys` whole strides, one per `out` slot.
    #[target_feature(enable = "avx2,popcnt")]
    pub(super) unsafe fn line_avx2(
        x: &[u8],
        ys: &[u8],
        stride: usize,
        limit: i64,
        out: &mut [Option<i32>],
    ) {
        debug_assert!(stride % 32 == 0 && x.len() == stride && ys.len() == out.len() * stride);
        let load = |bytes: &[u8], at: usize| {
            // SAFETY: at + 32 <= bytes.len() at every call below; loadu permits
            // any alignment.
            unsafe { _mm256_loadu_si256(bytes.as_ptr().add(at).cast()) }
        };
        let differ = |p: __m256i, q: __m256i| {
            32 - (_mm256_movemask_epi8(_mm256_cmpeq_epi8(p, q)) as u32).count_ones()
        };
        if stride == 32 {
            let p = load(x, 0);
            for (y, slot) in ys.chunks_exact(32).zip(out) {
                let distance = differ(p, load(y, 0));
                *slot = (i64::from(distance) <= limit).then_some(distance as i32);
            }
            return;
        }
        for (y, slot) in ys.chunks_exact(stride).zip(out) {
            let mut distance = 0u32;
            let mut over = false;
            let mut at = 0;
            while at < stride {
                distance += differ(load(x, at), load(y, at));
                at += 32;
                if at % 64 == 0 && i64::from(distance) > limit {
                    over = true;
                    break;
                }
            }
            *slot = (!over && i64::from(distance) <= limit).then_some(distance as i32);
        }
    }

    /// 64-byte inequality masks; rows of one or two steps keep the row's
    /// vectors in registers, longer rows check the cutoff once per step.
    ///
    /// # Safety
    /// The CPU must have AVX-512 F, BW and POPCNT; `stride` is a positive
    /// multiple of 64, `x` holds `stride` bytes and `ys` whole strides, one
    /// per `out` slot.
    #[cfg(seqblaze_avx512)]
    #[allow(clippy::incompatible_msrv)]
    #[target_feature(enable = "avx512f,avx512bw,popcnt")]
    pub(super) unsafe fn line_avx512(
        x: &[u8],
        ys: &[u8],
        stride: usize,
        limit: i64,
        out: &mut [Option<i32>],
    ) {
        debug_assert!(stride % 64 == 0 && x.len() == stride && ys.len() == out.len() * stride);
        let load = |bytes: &[u8], at: usize| {
            // SAFETY: at + 64 <= bytes.len() at every call below; loadu permits
            // any alignment.
            unsafe { _mm512_loadu_si512(bytes.as_ptr().add(at).cast()) }
        };
        let differ = |p: __m512i, q: __m512i| _mm512_cmpneq_epi8_mask(p, q).count_ones();
        let test = |distance: u32| (i64::from(distance) <= limit).then_some(distance as i32);
        match stride {
            64 => {
                let p = load(x, 0);
                for (y, slot) in ys.chunks_exact(64).zip(out) {
                    *slot = test(differ(p, load(y, 0)));
                }
            }
            128 => {
                let (p0, p1) = (load(x, 0), load(x, 64));
                for (y, slot) in ys.chunks_exact(128).zip(out) {
                    *slot = test(differ(p0, load(y, 0)) + differ(p1, load(y, 64)));
                }
            }
            _ => {
                for (y, slot) in ys.chunks_exact(stride).zip(out) {
                    let mut distance = 0u32;
                    let mut over = false;
                    let mut at = 0;
                    while at < stride {
                        distance += differ(load(x, at), load(y, at));
                        at += 64;
                        if i64::from(distance) > limit {
                            over = true;
                            break;
                        }
                    }
                    *slot = if over { None } else { test(distance) };
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Hamming;
    use seqblaze_core::{batch::SeqBatch, kernel::PairDistanceKernel, seq::SeqTable};

    fn forms() -> Vec<Form> {
        let mut all = vec![Form::Portable];
        #[cfg(target_arch = "x86_64")]
        all.push(Form::Avx2);
        #[cfg(all(target_arch = "x86_64", seqblaze_avx512))]
        all.push(Form::Avx512);
        all.into_iter().filter(|form| form.available()).collect()
    }

    /// `rows` raw-byte rows of `len` bytes (high bits included), each a few
    /// flips away from the first, so counts span ties, small and large values.
    fn rows(len: usize, count: usize, seed: usize) -> Vec<Vec<u8>> {
        let base: Vec<u8> = (0..len)
            .map(|t| ((t * 37 + seed) % 255) as u8 | 1)
            .collect();
        (0..count)
            .map(|r| {
                let mut row = base.clone();
                for t in (0..len).filter(|t| (t * 7 + r * 13 + seed) % (r % 5 + 1) == 0) {
                    row[t] ^= [0x80, 0x01, 0x40, 0x7e][t % 4];
                }
                row
            })
            .collect()
    }

    fn table(rows: &[Vec<u8>]) -> SeqTable {
        SeqTable::new(SeqBatch::from_seqs(rows.iter().map(Vec::as_slice)))
    }

    fn check(a: &SeqTable, b: &SeqTable, cutoffs: &[Option<i32>]) {
        let xs: Vec<_> = (0..a.len()).map(|i| a.get(i)).collect();
        let ys: Vec<_> = (0..b.len()).map(|j| b.get(j)).collect();
        for &cutoff in cutoffs {
            let mut expected = Vec::new();
            for &x in &xs {
                for &y in &ys {
                    expected.push(Hamming.evaluate(x, y, cutoff).unwrap());
                }
            }
            for form in forms() {
                // Appends after what `out` already holds.
                let mut out = vec![Some(-7)];
                score_with(form, &xs, &ys, cutoff, &mut out).unwrap();
                assert_eq!(out[0], Some(-7));
                assert_eq!(out[1..], expected[..], "{form:?} cutoff {cutoff:?}");
            }
        }
    }

    /// Every form gives `evaluate`'s cell across stride and step boundaries,
    /// the rectangle length limit and the inclusive cutoff.
    #[test]
    fn rectangles_match_evaluate_in_every_form() {
        let lengths = [
            0usize, 1, 7, 8, 9, 31, 32, 33, 63, 64, 65, 100, 127, 128, 129, 200, 1024, 1025,
        ];
        for len in lengths {
            let (a, b) = (table(&rows(len, 5, 3)), table(&rows(len, 7, 11)));
            let len = len as i32;
            let cutoffs = [
                None,
                Some(-1),
                Some(0),
                Some(1),
                Some(len / 3),
                Some(len - 1),
                Some(len),
            ];
            check(&a, &b, &cutoffs);
        }
    }

    /// Blocks of rows and columns past one staging block each, and a
    /// rectangle whose lengths differ (cell by cell).
    #[test]
    fn rectangles_cross_staging_blocks_and_mixed_lengths() {
        let (a, b) = (
            table(&rows(30, ROW_BLOCK + 3, 5)),
            table(&rows(30, COL_BLOCK + 9, 2)),
        );
        check(&a, &b, &[None, Some(4)]);
        let mut mixed = rows(30, 4, 1);
        mixed.push(rows(29, 1, 1).remove(0));
        check(&table(&mixed), &table(&rows(30, 3, 4)), &[None, Some(9)]);
    }
}
