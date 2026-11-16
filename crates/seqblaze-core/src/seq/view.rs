//! Retained foreign byte descriptors and checked per-call length metadata.
use super::{packed::PackedBytes, segmented::SegmentedBytes};
use crate::validate::EngineError;
use std::{
    any::Any,
    fmt,
    marker::PhantomData,
    ops::ControlFlow,
    sync::atomic::{AtomicU64, AtomicU8, Ordering},
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LengthDescriptor {
    pub address: usize,
    pub rows: usize,
    pub itemsize: usize,
    pub stride: isize,
}
#[derive(Clone, Copy, Debug)]
pub struct MatrixDescriptor {
    pub address: usize,
    pub rows: usize,
    pub columns: usize,
    pub row_stride: isize,
    pub column_stride: isize,
    pub lengths: LengthDescriptor,
}

fn check_span(
    address: usize,
    dimensions: &[(usize, isize)],
    itemsize: usize,
) -> Result<(), EngineError> {
    if dimensions.iter().any(|&(n, _)| n > isize::MAX as usize) {
        return Err(EngineError::Overflow(
            "NumPy dimensions exceed platform limits".into(),
        ));
    }
    if dimensions.iter().any(|&(n, _)| n == 0) || itemsize == 0 {
        return Ok(());
    }
    let (mut low_offset, mut high_offset) = (0i128, 0i128);
    for &(n, stride) in dimensions {
        let extent = (n - 1) as i128 * stride as i128;
        if extent < isize::MIN as i128 || extent > isize::MAX as i128 {
            return Err(EngineError::Overflow(
                "NumPy stride offset exceeds platform limits".into(),
            ));
        }
        low_offset += extent.min(0);
        high_offset += extent.max(0);
    }
    let low = address as i128 + low_offset;
    let high = address as i128 + high_offset + itemsize as i128;
    if address == 0 || low < 0 || high > usize::MAX as i128 || high - low > isize::MAX as i128 {
        return Err(EngineError::Overflow(
            "NumPy sequence stride/span exceeds platform limits".into(),
        ));
    }
    Ok(())
}

#[inline]
pub(super) unsafe fn load_byte(pointer: *const u8) -> u8 {
    // SAFETY: caller retains a valid byte allocation. AtomicU8 is byte-aligned
    // with no invalid bit patterns; no shared Rust slice spans foreign storage.
    unsafe { (&*pointer.cast::<AtomicU8>()).load(Ordering::Relaxed) }
}

/// Relaxed-load `out.len()` bytes starting at `source`: single bytes up to the
/// first 8-byte boundary, aligned `AtomicU64` words while a whole word remains,
/// then single bytes. Every load is a relaxed atomic load inside the range, and
/// no Rust slice spans the foreign storage, as with `load_byte` alone.
///
/// # Safety
/// `source..source + out.len()` must lie in one valid, retained allocation.
#[inline]
pub(super) unsafe fn load_bytes(source: *const u8, out: &mut [u8]) {
    let len = out.len();
    let head = ((source as usize).wrapping_neg() & 7).min(len);
    for (offset, byte) in out[..head].iter_mut().enumerate() {
        // SAFETY: inside the caller's range.
        *byte = unsafe { load_byte(source.wrapping_add(offset)) };
    }
    let words = (len - head) / 8;
    // SAFETY: 8-byte aligned (head reaches the boundary) and every whole word
    // lies inside the caller's range.
    unsafe { load_words(source.wrapping_add(head), &mut out[head..head + 8 * words]) };
    for (offset, byte) in out[head + 8 * words..].iter_mut().enumerate() {
        // SAFETY: inside the caller's range.
        *byte = unsafe { load_byte(source.wrapping_add(head + 8 * words + offset)) };
    }
}

/// Relaxed-load the aligned words at `source` into `out` (a whole number of
/// words), four words per step: one aligned `AtomicU64` load per word, its
/// native-endian bytes restoring memory order.
///
/// # Safety
/// `source` must be 8-byte aligned (unless `out` is empty) and
/// `source..source + out.len()` must lie in one valid, retained allocation.
#[inline(always)]
unsafe fn load_words(source: *const u8, out: &mut [u8]) {
    debug_assert!(out.len() % 8 == 0 && (out.is_empty() || source as usize % 8 == 0));
    // SAFETY (each load): aligned and inside the caller's range; AtomicU64
    // accepts every bit pattern.
    let load = |at: *const u8| unsafe { (&*at.cast::<AtomicU64>()).load(Ordering::Relaxed) };
    let mut at = source;
    let mut steps = out.chunks_exact_mut(32);
    for step in &mut steps {
        let words = [
            load(at),
            load(at.wrapping_add(8)),
            load(at.wrapping_add(16)),
            load(at.wrapping_add(24)),
        ];
        for (bytes, word) in step.chunks_exact_mut(8).zip(words) {
            bytes.copy_from_slice(&word.to_ne_bytes());
        }
        at = at.wrapping_add(32);
    }
    for bytes in steps.into_remainder().chunks_exact_mut(8) {
        bytes.copy_from_slice(&load(at).to_ne_bytes());
        at = at.wrapping_add(8);
    }
}

/// The relaxed load of the aligned word at `pointer`, in memory order.
///
/// # Safety
/// `pointer` must be 8-byte aligned and its 8 bytes must lie in one valid,
/// retained allocation.
#[inline(always)]
unsafe fn load_word(pointer: *const u8) -> [u8; 8] {
    // SAFETY: the caller's contract; AtomicU64 accepts every bit pattern and
    // native-endian bytes restore memory order.
    unsafe { (&*pointer.cast::<AtomicU64>()).load(Ordering::Relaxed) }.to_ne_bytes()
}

/// `load_bytes` with aligned words only: the words that hold the first and
/// the last byte of the range may extend past it, but only inside `window`,
/// a byte range the caller may read whole (the span of a packed `S<n>` view,
/// whose bytes are rows and their NUL padding). A range whose end words leave
/// the window falls back to `load_bytes`. Every load is a relaxed atomic
/// load inside the window; no Rust slice spans the foreign storage.
///
/// # Safety
/// `window` must lie in one valid, retained allocation and contain
/// `source..source + out.len()`.
#[inline]
pub(super) unsafe fn load_bytes_within(source: *const u8, out: &mut [u8], window: (usize, usize)) {
    let len = out.len();
    if len == 0 {
        return;
    }
    let start = source as usize;
    let skip = start & 7;
    let first = start - skip;
    let last = (start + len - 1) & !7;
    if first < window.0 || last + 8 > window.1 {
        // SAFETY: the caller's range lies in the window's allocation.
        return unsafe { load_bytes(source, out) };
    }
    // Keeps `source`'s provenance; the aligned words from `first` to `last`
    // lie in the window.
    let base = source.wrapping_sub(skip);
    // SAFETY: 8-byte aligned, and window.0 <= first (checked above).
    let head = unsafe { load_word(base) };
    if first == last {
        out.copy_from_slice(&head[skip..skip + len]);
        return;
    }
    out[..8 - skip].copy_from_slice(&head[skip..]);
    // Words 1..m - 1 are whole bytes of the range; word m (at `last`) holds
    // its last len - (8 - skip) - 8 (m - 1) bytes.
    let m = (last - first) / 8;
    let (middle, tail) = out[8 - skip..].split_at_mut(8 * (m - 1));
    // SAFETY: 8-byte aligned, and every word lies in [first, last], inside
    // the window (checked above).
    unsafe { load_words(base.wrapping_add(8), middle) };
    // SAFETY: 8-byte aligned, and last + 8 <= window.1 (checked above).
    let end = unsafe { load_word(base.wrapping_add(8 * m)) };
    let rest = tail.len();
    tail.copy_from_slice(&end[..rest]);
}

impl LengthDescriptor {
    pub(crate) fn validate(self) -> Result<(), EngineError> {
        if !matches!(self.itemsize, 4 | 8) {
            return Err(EngineError::Parameter(
                "integer descriptor requires int32/int64 items".into(),
            ));
        }
        check_span(self.address, &[(self.rows, self.stride)], self.itemsize)
    }
    /// # Safety
    /// Geometry must be validated and the native allocation retained for this
    /// read; row must be less than the descriptor count.
    pub(crate) unsafe fn read(self, row: usize) -> i64 {
        debug_assert!(row < self.rows);
        let pointer = (self.address as *const u8).wrapping_offset(row as isize * self.stride);
        let mut bytes = [0u8; 8];
        for (offset, byte) in bytes.iter_mut().take(self.itemsize).enumerate() {
            // SAFETY: the checked native descriptor and retained length export
            // cover each element; reading bytes also supports unaligned inputs.
            *byte = unsafe { load_byte(pointer.wrapping_add(offset)) };
        }
        if self.itemsize == 4 {
            i32::from_ne_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as i64
        } else {
            i64::from_ne_bytes(bytes)
        }
    }
}

pub(super) struct FixedBytes {
    pub(super) address: usize,
    pub(super) rows: usize,
    pub(super) width: usize,
    pub(super) stride: isize,
    column_stride: isize,
    explicit_lengths: Option<LengthDescriptor>,
    /// [`Self::window`], fixed at import.
    span: Option<(usize, usize)>,
    _owner: Box<dyn Any + Send + Sync>,
}
impl fmt::Debug for FixedBytes {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("FixedBytes")
            .field("rows", &self.rows)
            .field("width", &self.width)
            .field("stride", &self.stride)
            .field("column_stride", &self.column_stride)
            .finish_non_exhaustive()
    }
}

impl FixedBytes {
    /// # Safety
    /// The native descriptor must be valid and owner must retain its allocation.
    /// No resize/reallocation while retained or mutation during active execution.
    pub(super) unsafe fn new(
        address: usize,
        rows: usize,
        width: usize,
        stride: isize,
        owner: Box<dyn Any + Send + Sync>,
    ) -> Result<Self, EngineError> {
        check_span(address, &[(rows, stride), (width, 1)], 1)?;
        // Rows back to back: rows * width <= isize::MAX past the address
        // (check_span above).
        let span =
            (usize::try_from(stride) == Ok(width)).then(|| (address, address + rows * width));
        Ok(Self {
            address,
            rows,
            width,
            stride,
            column_stride: 1,
            explicit_lengths: None,
            span,
            _owner: owner,
        })
    }
    /// # Safety
    /// Both native descriptors must be valid; owner retains both allocations.
    /// Neither producer may resize/reallocate while retained or change during
    /// execution/active tile iteration. Length values are checked before use.
    pub(super) unsafe fn matrix(
        descriptor: MatrixDescriptor,
        owner: Box<dyn Any + Send + Sync>,
    ) -> Result<Self, EngineError> {
        let MatrixDescriptor {
            address,
            rows,
            columns,
            row_stride,
            column_stride,
            lengths,
        } = descriptor;
        if lengths.rows != rows || !matches!(lengths.itemsize, 4 | 8) {
            return Err(EngineError::Parameter(
                "matrix lengths need one native int32/int64 value per row".into(),
            ));
        }
        check_span(address, &[(rows, row_stride), (columns, column_stride)], 1)?;
        check_span(lengths.address, &[(rows, lengths.stride)], lengths.itemsize)?;
        Ok(Self {
            address,
            rows,
            width: columns,
            stride: row_stride,
            column_stride,
            explicit_lengths: Some(lengths),
            span: None,
            _owner: owner,
        })
    }
    /// Geometry-only proof of one forward contiguous allocation span. The
    /// caller must additionally prove that every row occupies its full width.
    pub(super) fn contiguous_span(&self) -> Option<(usize, usize)> {
        if self.column_stride != 1 || usize::try_from(self.stride).ok()? != self.width {
            return None;
        }
        let bytes = self.rows.checked_mul(self.width)?;
        if i64::try_from(bytes).is_err() || (bytes != 0 && self.address == 0) {
            return None;
        }
        Some((self.address, bytes))
    }

    #[inline]
    pub(super) fn read(&self, row: usize, position: usize) -> u8 {
        debug_assert!(row < self.rows && position < self.width);
        let pointer = (self.address as *const u8)
            .wrapping_offset(row as isize * self.stride)
            .wrapping_offset(position as isize * self.column_stride);
        // SAFETY: checked native geometry and the retained export cover this byte;
        // position is below the checked effective length and column capacity.
        unsafe { load_byte(pointer) }
    }
    /// Relaxed-load bytes `start..start + out.len()` of `row` into `out`.
    #[inline]
    pub(super) fn copy_row(&self, row: usize, start: usize, out: &mut [u8]) {
        debug_assert!(row < self.rows && start + out.len() <= self.width);
        let base = (self.address as *const u8).wrapping_offset(row as isize * self.stride);
        if let Some(window) = self.window() {
            // SAFETY: as in `read`; the copied range lies in the row, inside
            // the view's whole span.
            unsafe { load_bytes_within(base.wrapping_add(start), out, window) };
        } else if self.column_stride == 1 {
            // SAFETY: as in `read`; the copied range lies in the checked prefix.
            unsafe { load_bytes(base.wrapping_add(start), out) };
        } else {
            for (offset, byte) in out.iter_mut().enumerate() {
                *byte = self.read(row, start + offset);
            }
        }
    }
    /// [`ByteView::copy_padded`] of bytes `start..start + n` of `row` (n <=
    /// out.len(), a multiple of 8). In the window, output word k joins the
    /// aligned source words k and k + 1 (the row's bytes need not be aligned)
    /// and the last needed word is masked to the row's bytes; other layouts
    /// and the window's edge words copy with `copy_row`. Words past
    /// ⌈n / 8⌉ are zero.
    pub(super) fn copy_padded(&self, row: usize, start: usize, n: usize, out: &mut [u8]) {
        debug_assert!(row < self.rows && start + n <= self.width && n <= out.len());
        let base = (self.address as *const u8)
            .wrapping_offset(row as isize * self.stride)
            .wrapping_add(start);
        let from = base as usize;
        let skip = from & 7;
        let first = from - skip;
        let last = (from + n).saturating_sub(1) & !7;
        let needed = n.div_ceil(8);
        let (head, rest) = out.split_at_mut(8 * needed);
        match self.span {
            Some(window) if n > 0 && first >= window.0 && last + 8 <= window.1 => {
                // Pointers keep the view's provenance; every word lies in the window.
                let word = |address: usize| {
                    // SAFETY: 8-byte aligned, first <= address <= last, and the
                    // window (inside the retained export) holds [first, last + 8).
                    let pointer = base.wrapping_sub(skip).wrapping_add(address - first);
                    u64::from_le_bytes(unsafe { load_word(pointer) })
                };
                // Bytes skip.. of the 16 bytes `low`, `high` (shift <= 56).
                let shift = 8 * skip as u32;
                let join = |low: u64, high: u64| {
                    ((u128::from(high) << 64 | u128::from(low)) >> shift) as u64
                };
                let mut words = head.chunks_exact_mut(8);
                let mut low = word(first);
                // Every word but the last needs source words k and k + 1 <= last.
                for k in 1..needed {
                    let high = word(first + 8 * k);
                    words
                        .next()
                        .expect("needed words")
                        .copy_from_slice(&join(low, high).to_le_bytes());
                    low = high;
                }
                let high = if first + 8 * needed <= last {
                    word(first + 8 * needed)
                } else {
                    0
                };
                let mut value = join(low, high);
                if n % 8 != 0 {
                    value &= (1u64 << (8 * (n % 8))) - 1;
                }
                words
                    .next()
                    .expect("needed words")
                    .copy_from_slice(&value.to_le_bytes());
            }
            _ => {
                self.copy_row(row, start, &mut head[..n]);
                head[n..].fill(0);
            }
        }
        if !rest.is_empty() {
            rest.fill(0);
        }
    }
    /// The window reader of bytes `0..n` (n >= 1) of `row` for
    /// [`ByteView::fold_words`]: None unless the aligned words that hold
    /// those bytes all lie in the view's window. Loads the first word.
    #[inline(always)]
    fn window_words(&self, row: usize, n: usize) -> Option<WindowWords<'_>> {
        let window = self.window()?;
        debug_assert!(row < self.rows && 0 < n && n <= self.width);
        let base = (self.address as *const u8).wrapping_add(row * self.width);
        let from = base as usize;
        let skip = from & 7;
        let (first, end) = (from - skip, (from + n - 1) & !7);
        if first < window.0 || end + 8 > window.1 {
            return None;
        }
        let mut words = WindowWords {
            // Keeps the view's provenance; 8-byte aligned, inside the window.
            next: base.wrapping_sub(skip),
            end,
            low: 0,
            shift: 8 * skip as u32,
            _view: PhantomData,
        };
        words.low = words.load();
        Some(words)
    }
    /// Whether rows lie back to back: unit column stride and row stride equal
    /// to the width, so rows `r..r + k` are one span of `k * width` bytes.
    pub(super) fn packed_rows(&self) -> bool {
        self.column_stride == 1 && usize::try_from(self.stride) == Ok(self.width)
    }
    /// Whether bytes past a row's effective length may be read: an `S<n>`
    /// view reads its NUL padding anyway to trim it; a matrix never reads
    /// past `lengths[i]` (engine-layout §9).
    pub(super) fn padding_readable(&self) -> bool {
        self.explicit_lengths.is_none()
    }
    /// The address range `[start, end)` a load of a packed `S<n>` view may
    /// cover whole: its rows lie back to back, so every byte of the span is a
    /// row byte or its NUL padding, and an aligned word that holds a byte of
    /// one row may read its neighbours. None for other layouts, whose loads
    /// stay inside the requested range (a matrix never reads past a row's
    /// length; a strided view's gaps belong to other fields).
    #[inline]
    pub(super) fn window(&self) -> Option<(usize, usize)> {
        self.span
    }
    /// Relaxed-load bytes `start..start + out.len()` of a packed view's span
    /// (its rows back to back, NUL padding included) into `out`, with the
    /// window's aligned words; requires [`Self::window`].
    pub(super) fn copy_span(&self, start: usize, out: &mut [u8]) {
        let window = self.window().expect("a packed view");
        assert!(start
            .checked_add(out.len())
            .is_some_and(|end| end <= window.1 - window.0));
        let pointer = (self.address as *const u8).wrapping_add(start);
        // SAFETY: the range lies in the span, which the retained export covers.
        unsafe { load_bytes_within(pointer, out, window) };
    }
    /// Relaxed-load whole rows `first..first + out.len() / width` into `out`
    /// with one block read; requires [`Self::packed_rows`].
    #[inline]
    pub(super) fn copy_rows(&self, first: usize, out: &mut [u8]) {
        assert!(self.packed_rows() && self.width > 0 && out.len() % self.width == 0);
        assert!(first <= self.rows && out.len() / self.width <= self.rows - first);
        let base = (self.address as *const u8).wrapping_add(first * self.width);
        match self.window() {
            // SAFETY: rows are back to back, so the checked native geometry and
            // the retained export cover these whole rows as one span, inside
            // the view's window.
            Some(window) => unsafe { load_bytes_within(base, out, window) },
            // SAFETY: as above.
            None => unsafe { load_bytes(base, out) },
        }
    }
    /// Effective length of `S<n>` row `row`: the width less its maximal
    /// trailing NUL padding. A full row costs one byte read; padding is
    /// searched backward one aligned word at a time in a packed view, else a
    /// block at a time.
    fn trimmed_len(&self, row: usize) -> usize {
        let mut length = self.width;
        if length == 0 || self.read(row, length - 1) != 0 {
            return length;
        }
        if let Some(window) = self.window() {
            return self.trimmed_len_within(row, window);
        }
        let mut block = [0u8; BLOCK];
        while length > 0 {
            let take = BLOCK.min(length);
            self.copy_row(row, length - take, &mut block[..take]);
            match block[..take].iter().rposition(|&byte| byte != 0) {
                Some(last) => return length - take + last + 1,
                None => length -= take,
            }
        }
        0
    }
    /// [`Self::trimmed_len`] of a row of a packed `S<n>` view (width >= 1):
    /// the window's aligned words from the row's end backward, each masked to
    /// the row's bytes, until one holds a non-NUL byte. A word that leaves the
    /// window (only at the span's ends) is read one byte at a time instead.
    fn trimmed_len_within(&self, row: usize, window: (usize, usize)) -> usize {
        let base = (self.address as *const u8).wrapping_add(row * self.width);
        let start = base as usize;
        // Pointers keep the view's provenance; `address` may precede the row
        // start by up to 7 bytes (wrapping offsets are signed).
        let at = |address: usize| base.wrapping_add(address.wrapping_sub(start));
        let mut end = start + self.width;
        while end > start {
            let word_start = (end - 1) & !7;
            if word_start < window.0 || word_start + 8 > window.1 {
                end -= 1;
                // SAFETY: a byte of this row, inside the checked span.
                if unsafe { load_byte(at(end)) } != 0 {
                    return end + 1 - start;
                }
                continue;
            }
            // The row's bytes in this word: [low, high) of its 8 bytes.
            let low = start.max(word_start) - word_start;
            let high = end - word_start;
            // SAFETY: 8-byte aligned and inside the window, which the view's
            // retained export covers (`window`).
            let word = unsafe { load_word(at(word_start)) };
            let kept =
                u64::from_le_bytes(word) & (u64::MAX >> (64 - 8 * high)) & (u64::MAX << (8 * low));
            if kept != 0 {
                let last = (63 - kept.leading_zeros() as usize) / 8;
                return word_start + last + 1 - start;
            }
            end = word_start + low;
        }
        0
    }
    pub(super) fn lengths(&self) -> Result<Vec<usize>, EngineError> {
        self.lengths_with(1, false)
    }

    /// The original length preflight's one reservation and checks. A stream
    /// proof may replace trimming with a constant fill, but never moves the
    /// reservation. Parallel trimming is optional: a refused start reuses this
    /// buffer inline, so it cannot precede a later pair/alphabet error.
    pub(super) fn lengths_with(
        &self,
        workers: usize,
        full_width: bool,
    ) -> Result<Vec<usize>, EngineError> {
        let mut result = Vec::new();
        crate::detail::try_reserve_exact(&mut result, self.rows, "numpy view lengths")?;
        if let Some(lengths) = self.explicit_lengths {
            for row in 0..self.rows {
                // SAFETY: the matrix retains the checked lengths export, and
                // this row lies in its validated descriptor.
                let length = unsafe { lengths.read(row) };
                if length < 0 || length as u128 > self.width as u128 {
                    return Err(EngineError::Parameter(format!(
                        "lengths[{row}]={length} must be between 0 and matrix column capacity {}",
                        self.width
                    )));
                }
                result.push(length as usize);
            }
            Ok(result)
        } else if full_width {
            result.resize(self.rows, self.width);
            Ok(result)
        } else if workers > 1 {
            result.resize(self.rows, 0);
            let trim = |start: usize, out: &mut [usize]| {
                for (offset, length) in out.iter_mut().enumerate() {
                    *length = self.trimmed_len(start + offset);
                }
                Ok(())
            };
            if !crate::sched::try_fill_chunks(&mut result, workers, &trim)? {
                // Every started worker has joined. Overwrite the partial trim
                // in place instead of repeating its allocation.
                trim(0, &mut result)?;
            }
            Ok(result)
        } else {
            result.extend((0..self.rows).map(|row| self.trimmed_len(row)));
            Ok(result)
        }
    }
    pub(super) fn same_input(&self, other: &Self) -> bool {
        self.address == other.address
            && self.rows == other.rows
            && self.width == other.width
            && self.stride == other.stride
            && self.column_stride == other.column_stride
            && self.explicit_lengths == other.explicit_lengths
    }
    pub(super) fn matrix_info(&self) -> Option<MatrixDescriptor> {
        self.explicit_lengths.map(|lengths| MatrixDescriptor {
            address: self.address,
            rows: self.rows,
            columns: self.width,
            row_stride: self.stride,
            column_stride: self.column_stride,
            lengths,
        })
    }
}

#[derive(Clone, Copy)]
enum Data<'a> {
    /// Owned bytes or a retained immutable Arrow span: a direct shared slice.
    Slice(&'a [u8]),
    /// Borrowed NumPy storage: raw relaxed byte loads only, never a Rust slice.
    Fixed(&'a FixedBytes, usize),
}

/// Fixed block width used by [`ByteView::for_each_block`] and kernel block loops.
pub const BLOCK: usize = 64;

/// Byte access to one sequence row.
///
/// Kernels read rows in one of three ways: [`ByteView::as_slice`] for immutable
/// storage (owned and imported Arrow buffers); [`ByteView::copy_to`] or
/// [`ByteView::for_each_block`] for any storage, which read borrowed NumPy bytes
/// with relaxed loads into a caller-owned block; or [`ByteView::iter`] when a
/// byte-at-a-time walk is simplest. Only the first form shares foreign memory
/// as a Rust slice, and only for buffers specified immutable while retained.
#[derive(Clone, Copy)]
pub struct ByteView<'a> {
    data: Data<'a>,
    len: usize,
}
impl<'a> ByteView<'a> {
    pub(super) fn owned(bytes: &'a [u8]) -> Self {
        Self {
            data: Data::Slice(bytes),
            len: bytes.len(),
        }
    }
    /// A row held in caller-owned immutable bytes, such as a staged copy of
    /// a borrowed row (`Stager`); kernels read it as a slice.
    #[inline]
    pub fn from_slice(bytes: &'a [u8]) -> Self {
        Self::owned(bytes)
    }
    pub(super) fn fixed(source: &'a FixedBytes, row: usize, len: usize) -> Self {
        Self {
            data: Data::Fixed(source, row),
            len,
        }
    }
    /// # Safety
    /// `address..address+len` must lie in an immutable native allocation that
    /// outlives `'a` (an imported Arrow buffer retained by the table's owner).
    pub(super) unsafe fn immutable(address: usize, len: usize) -> Self {
        let bytes: &'a [u8] = if len == 0 {
            // Empty Arrow values may carry a null or dangling base pointer.
            &[]
        } else {
            // SAFETY: the caller guarantees a validated, retained, immutable span.
            unsafe { std::slice::from_raw_parts(address as *const u8, len) }
        };
        Self::owned(bytes)
    }
    pub(super) fn packed(source: &'a PackedBytes, row: usize) -> Self {
        // SAFETY: prepared Arrow row spans lie in buffers the PackedBytes owner
        // retains, and imported Arrow buffers are immutable while retained.
        unsafe { Self::immutable(source.start(row), source.seq_len(row)) }
    }
    pub(super) fn segmented(source: &'a SegmentedBytes, row: usize) -> Self {
        let (address, len) = source.span(row);
        // SAFETY: as for packed rows; every chunk owner lives in the table.
        unsafe { Self::immutable(address, len) }
    }
    pub fn len(self) -> usize {
        self.len
    }
    pub fn is_empty(self) -> bool {
        self.len == 0
    }
    /// Whether `other` reads the same bytes of the same storage: one slice
    /// (same address and length) or one row of one borrowed table. A false
    /// answer says nothing about the bytes.
    #[inline(always)]
    pub fn same_row(self, other: Self) -> bool {
        // (address of the bytes, or of the borrowed table and the row): a
        // slice and a borrowed row never compare equal, since no row index is
        // usize::MAX.
        let key = |view: Self| match view.data {
            Data::Slice(bytes) => (bytes.as_ptr() as usize, usize::MAX),
            Data::Fixed(source, row) => (source as *const FixedBytes as usize, row),
        };
        self.len == other.len && key(self) == key(other)
    }
    /// Direct slice for immutable storage; None for borrowed NumPy rows.
    pub fn as_slice(self) -> Option<&'a [u8]> {
        match self.data {
            Data::Slice(bytes) => Some(bytes),
            Data::Fixed(..) => None,
        }
    }
    /// Copy bytes `start..start + out.len()` of this row into `out`.
    /// Borrowed NumPy bytes use relaxed loads from a base pointer computed
    /// once per call (`FixedBytes::copy_row`). Panics if the range exceeds the
    /// row.
    #[inline]
    pub fn copy_to(self, start: usize, out: &mut [u8]) {
        assert!(start <= self.len && out.len() <= self.len - start);
        match self.data {
            Data::Slice(bytes) => out.copy_from_slice(&bytes[start..start + out.len()]),
            Data::Fixed(source, row) => source.copy_row(row, start, out),
        }
    }
    /// Copy bytes `start..start + n` of this row into `out`, n = min(out.len(),
    /// len - start), and zero the rest of `out`, whose length must be a
    /// multiple of 8. Every write is a whole 8-byte word: no copy of a
    /// variable length (which becomes a library call) for short rows. A
    /// packed `S<n>` row is read as the aligned words of its view's span
    /// (`FixedBytes::copy_padded`). Panics if `start` exceeds the row.
    #[inline]
    pub fn copy_padded(self, start: usize, out: &mut [u8]) {
        assert!(start <= self.len && out.len() % 8 == 0);
        let n = out.len().min(self.len - start);
        match self.data {
            Data::Slice(bytes) => {
                let bytes = &bytes[start..start + n];
                let mut chunks = bytes.chunks_exact(8);
                let mut words = out.chunks_exact_mut(8);
                // `chunks` first: zip then takes no word past the last chunk.
                for (chunk, word) in (&mut chunks).zip(&mut words) {
                    word.copy_from_slice(chunk);
                }
                // The partial word, assembled in a register, then zero words.
                let rest = chunks.remainder();
                if let Some(word) = (!rest.is_empty()).then(|| words.next()).flatten() {
                    let value = rest
                        .iter()
                        .enumerate()
                        .fold(0u64, |value, (k, &byte)| value | u64::from(byte) << (8 * k));
                    word.copy_from_slice(&value.to_le_bytes());
                }
                words.for_each(|word| word.fill(0));
            }
            Data::Fixed(source, row) => source.copy_padded(row, start, n, out),
        }
    }
    /// Fold `step(state, x, y)` over the word pairs of bytes `0..n` of this
    /// row and `other` (n at most either length), in order: word k of a row
    /// is its bytes `8k..8k + 8` as a little-endian u64, and both last words
    /// are zero past byte n. `step` may end the fold early with
    /// `ControlFlow::Break`; the fold returns the final state. Words pass
    /// from the loads to `step` in registers, with no buffer between: a
    /// packed `S<n>` row joins the aligned words of its view's span
    /// (`FixedBytes::window`), each loaded once; a slice reads its bytes;
    /// other storage copies each word with relaxed loads.
    #[inline]
    pub fn fold_words<T>(
        self,
        other: Self,
        n: usize,
        init: T,
        step: impl FnMut(T, u64, u64) -> ControlFlow<T, T>,
    ) -> T {
        assert!(n <= self.len && n <= other.len);
        if n == 0 {
            return init;
        }
        match self.words(n) {
            Words::Slice(x) => fold_with(x, other.words(n), n, init, step),
            Words::Window(x) => fold_with(x, other.words(n), n, init, step),
            Words::Copied(x) => fold_with(x, other.words(n), n, init, step),
        }
    }
    /// The word reader of bytes `0..n` (1 <= n <= len) for [`Self::fold_words`].
    #[inline(always)]
    fn words(self, n: usize) -> Words<'a> {
        match self.data {
            Data::Slice(bytes) => Words::Slice(SliceWords(&bytes[..n])),
            Data::Fixed(source, row) => match source.window_words(row, n) {
                Some(words) => Words::Window(words),
                None => Words::Copied(CopiedWords { view: self, n }),
            },
        }
    }
    /// Visit the row in order as consecutive blocks of at most [`BLOCK`] bytes;
    /// `visit(offset, block)` may stop early with `ControlFlow::Break`.
    #[inline]
    pub fn for_each_block<B>(
        self,
        mut visit: impl FnMut(usize, &[u8]) -> ControlFlow<B>,
    ) -> ControlFlow<B> {
        if let Some(bytes) = self.as_slice() {
            for (index, block) in bytes.chunks(BLOCK).enumerate() {
                visit(index * BLOCK, block)?;
            }
            return ControlFlow::Continue(());
        }
        let mut buffer = [0u8; BLOCK];
        let mut start = 0;
        while start < self.len {
            let width = BLOCK.min(self.len - start);
            self.copy_to(start, &mut buffer[..width]);
            visit(start, &buffer[..width])?;
            start += width;
        }
        ControlFlow::Continue(())
    }
    pub fn iter(self) -> Bytes<'a> {
        Bytes {
            view: self,
            position: 0,
        }
    }
}

pub struct Bytes<'a> {
    view: ByteView<'a>,
    position: usize,
}
impl Iterator for Bytes<'_> {
    type Item = u8;
    #[inline]
    fn next(&mut self) -> Option<u8> {
        if self.position == self.view.len {
            return None;
        }
        let position = self.position;
        self.position += 1;
        Some(match self.view.data {
            Data::Slice(bytes) => bytes[position],
            Data::Fixed(source, row) => source.read(row, position),
        })
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        let n = self.view.len - self.position;
        (n, Some(n))
    }
}
impl ExactSizeIterator for Bytes<'_> {}

/// Little-endian 8-byte words of a row's bytes `0..n` (n >= 1), read in
/// increasing order ([`ByteView::fold_words`]).
trait WordReader {
    /// Word k, whose 8 bytes all lie in `0..n`.
    fn whole(&mut self, k: usize) -> u64;
    /// The last word, k = ⌈n / 8⌉ - 1, after every earlier word; its bytes
    /// past n are arbitrary (the caller masks them).
    fn last(&mut self, k: usize) -> u64;
}

/// The reader of one row, chosen once per fold.
enum Words<'a> {
    Slice(SliceWords<'a>),
    Window(WindowWords<'a>),
    Copied(CopiedWords<'a>),
}

/// Words of immutable bytes (exactly the n bytes).
struct SliceWords<'a>(&'a [u8]);
impl WordReader for SliceWords<'_> {
    #[inline(always)]
    fn whole(&mut self, k: usize) -> u64 {
        u64::from_le_bytes(self.0[8 * k..8 * k + 8].try_into().expect("8 bytes"))
    }
    #[inline(always)]
    fn last(&mut self, k: usize) -> u64 {
        let rest = &self.0[8 * k..];
        rest.iter()
            .rev()
            .fold(0, |word, &byte| word << 8 | u64::from(byte))
    }
}

/// Words of a packed `S<n>` row whose aligned words lie in its view's window
/// ([`FixedBytes::window_words`]): word k joins aligned words k and k + 1 of
/// the row's range, and each aligned word is loaded once.
struct WindowWords<'a> {
    /// The next aligned word to load.
    next: *const u8,
    /// Address of the aligned word that holds byte n - 1.
    end: usize,
    /// The aligned word before `next`.
    low: u64,
    /// 8 × the row start's offset in its aligned word (0 to 56).
    shift: u32,
    _view: PhantomData<&'a FixedBytes>,
}
impl WindowWords<'_> {
    #[inline(always)]
    fn load(&mut self) -> u64 {
        // SAFETY: `next` is 8-byte aligned and at most `end`, whose word lies
        // in the window (`window_words`); the borrow of the view keeps the
        // export retained.
        let word = u64::from_le_bytes(unsafe { load_word(self.next) });
        self.next = self.next.wrapping_add(8);
        word
    }
    /// Bytes shift / 8.. of `low`, then the first bytes of `high`
    /// (`(high << 1) << (63 - shift)` is `high << (64 - shift)`, and 0 for
    /// shift 0, with no shift of 64).
    #[inline(always)]
    fn join(&mut self, high: u64) -> u64 {
        let value = self.low >> self.shift | (high << 1) << (63 - self.shift);
        self.low = high;
        value
    }
}
impl WordReader for WindowWords<'_> {
    #[inline(always)]
    fn whole(&mut self, _: usize) -> u64 {
        // Word k's bytes end at or before byte n - 1's aligned word.
        let high = self.load();
        self.join(high)
    }
    #[inline(always)]
    fn last(&mut self, _: usize) -> u64 {
        let high = if self.next as usize <= self.end {
            self.load()
        } else {
            0
        };
        self.join(high)
    }
}

/// Words of any other row, each copied with relaxed loads.
struct CopiedWords<'a> {
    view: ByteView<'a>,
    n: usize,
}
impl WordReader for CopiedWords<'_> {
    #[inline(always)]
    fn whole(&mut self, k: usize) -> u64 {
        let mut word = [0u8; 8];
        self.view.copy_to(8 * k, &mut word);
        u64::from_le_bytes(word)
    }
    #[inline(always)]
    fn last(&mut self, k: usize) -> u64 {
        let mut word = [0u8; 8];
        self.view.copy_to(8 * k, &mut word[..self.n - 8 * k]);
        u64::from_le_bytes(word)
    }
}

/// [`ByteView::fold_words`] with the first row's reader fixed: dispatch the
/// second's, so each pair of reader kinds is its own loop.
#[inline(always)]
fn fold_with<T>(
    x: impl WordReader,
    y: Words<'_>,
    n: usize,
    init: T,
    step: impl FnMut(T, u64, u64) -> ControlFlow<T, T>,
) -> T {
    match y {
        Words::Slice(y) => fold_pairs(x, y, n, init, step),
        Words::Window(y) => fold_pairs(x, y, n, init, step),
        Words::Copied(y) => fold_pairs(x, y, n, init, step),
    }
}

#[inline(always)]
fn fold_pairs<T>(
    mut x: impl WordReader,
    mut y: impl WordReader,
    n: usize,
    mut state: T,
    mut step: impl FnMut(T, u64, u64) -> ControlFlow<T, T>,
) -> T {
    let last = n.div_ceil(8) - 1;
    for k in 0..last {
        match step(state, x.whole(k), y.whole(k)) {
            ControlFlow::Continue(next) => state = next,
            ControlFlow::Break(stop) => return stop,
        }
    }
    // The last word keeps n - 8 * last bytes, 1 to 8.
    let mask = u64::MAX >> (8 * (8 * (last + 1) - n));
    let (ControlFlow::Continue(state) | ControlFlow::Break(state)) =
        step(state, x.last(last) & mask, y.last(last) & mask);
    state
}

#[cfg(test)]
mod tests {
    use super::{load_bytes, load_bytes_within, ByteView, FixedBytes};
    use std::ops::ControlFlow;

    /// Windowed loads equal a plain copy at every start alignment and length,
    /// for windows that end anywhere (end words outside fall back to bytes).
    #[test]
    fn load_bytes_within_equals_a_plain_copy_at_every_alignment() {
        let source: Vec<u8> = (0..112u32).map(|k| (k * 29 + 3) as u8).collect();
        let base = source.as_ptr() as usize;
        for lo in 0..9 {
            for hi in [source.len() - 9, source.len() - 4, source.len()] {
                for start in lo..lo + 17 {
                    for len in 0..=(hi - start).min(50) {
                        let mut out = vec![0u8; len];
                        // SAFETY: the window and range lie inside `source`.
                        unsafe {
                            load_bytes_within(
                                source.as_ptr().wrapping_add(start),
                                &mut out,
                                (base + lo, base + hi),
                            )
                        };
                        assert_eq!(out, &source[start..start + len], "{lo} {hi} {start} {len}");
                    }
                }
            }
        }
    }

    /// Packed `S<n>` trims and row copies against the plain definitions, for
    /// every width up to 20 and every base alignment: rows with no, some and
    /// only NUL padding, internal NULs, and the span's first and last rows.
    #[test]
    fn packed_trim_and_copies_match_the_definition() {
        for width in 1..=20usize {
            for shift in 0..8 {
                let rows = 9;
                let mut data = vec![0u8; shift + rows * width + 3];
                let mut expected = Vec::new();
                for row in 0..rows {
                    let len = (row * 7 + width) % (width + 1);
                    for k in 0..len {
                        // An internal NUL in some rows; the last byte is non-NUL.
                        let byte = if k + 1 < len && (row + k) % 5 == 0 {
                            0
                        } else {
                            b'A' + (k % 26) as u8
                        };
                        data[shift + row * width + k] = byte;
                    }
                    expected.push(len);
                }
                let address = data.as_ptr() as usize + shift;
                let copy = data.clone();
                // SAFETY: the boxed Vec owns the described rows and is never mutated.
                let fixed = unsafe {
                    FixedBytes::new(address, rows, width, width as isize, Box::new(data))
                }
                .unwrap();
                assert!(fixed.window().is_some());
                assert_eq!(
                    fixed.lengths().unwrap(),
                    expected,
                    "width {width} shift {shift}"
                );
                for (row, &len) in expected.iter().enumerate() {
                    let at = shift + row * width;
                    let bytes = &copy[at..at + len];
                    for start in 0..=len {
                        let mut out = vec![0u8; len - start];
                        fixed.copy_row(row, start, &mut out);
                        assert_eq!(out, &bytes[start..]);
                        // Padded copies of every size, from the view and from a
                        // slice: the bytes, then zeros (the buffer starts dirty).
                        for words in 0..=4 {
                            let n = (8 * words).min(len - start);
                            let mut want = bytes[start..start + n].to_vec();
                            want.resize(8 * words, 0);
                            for view in [ByteView::fixed(&fixed, row, len), ByteView::owned(bytes)]
                            {
                                let mut out = vec![0xa5u8; 8 * words];
                                view.copy_padded(start, &mut out);
                                assert_eq!(
                                    out, want,
                                    "width {width} shift {shift} row {row} start {start}"
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    /// `fold_words` against the definition for every pair of readers (packed
    /// window rows, slices, and rows of a view with no window), every row
    /// alignment, every n up to the width, the span's edge rows, and a stop.
    #[test]
    fn fold_words_match_the_definition_for_every_reader() {
        let words_of = |bytes: &[u8]| -> Vec<u64> {
            let word = |chunk: &[u8]| chunk.iter().rev().fold(0, |w, &b| w << 8 | u64::from(b));
            bytes.chunks(8).map(word).collect()
        };
        for width in [1usize, 7, 8, 9, 20, 33] {
            for shift in 0..8 {
                let rows = 4;
                let bytes: Vec<Vec<u8>> = (0..rows)
                    .map(|row| (0..width).map(|k| (k * 29 + row * 71) as u8 | 1).collect())
                    .collect();
                // Packed rows (a window), and rows one byte apart (none).
                let view = |stride: usize| {
                    let mut data = vec![0u8; shift + rows * stride + 3];
                    for (row, row_bytes) in bytes.iter().enumerate() {
                        let at = shift + row * stride;
                        data[at..at + width].copy_from_slice(row_bytes);
                    }
                    let address = data.as_ptr() as usize + shift;
                    // SAFETY: the boxed Vec owns the described rows and is never mutated.
                    unsafe {
                        FixedBytes::new(address, rows, width, stride as isize, Box::new(data))
                    }
                    .unwrap()
                };
                let (packed, spaced) = (view(width), view(width + 1));
                assert!(packed.window().is_some() && spaced.window().is_none());
                let views = |row: usize| {
                    [
                        ByteView::fixed(&packed, row, width),
                        ByteView::fixed(&spaced, row, width),
                        ByteView::owned(&bytes[row]),
                    ]
                };
                for (r, s) in [(0, 1), (1, 3), (3, 0), (2, 2)] {
                    for n in 0..=width {
                        let (x, y) = (words_of(&bytes[r][..n]), words_of(&bytes[s][..n]));
                        let want: Vec<(u64, u64)> = x.into_iter().zip(y).collect();
                        for p in views(r) {
                            for q in views(s) {
                                let all = p.fold_words(q, n, Vec::new(), |mut seen, u, v| {
                                    seen.push((u, v));
                                    ControlFlow::Continue(seen)
                                });
                                assert_eq!(all, want, "width {width} shift {shift} n {n}");
                                let one = p.fold_words(q, n, Vec::new(), |mut seen, u, v| {
                                    seen.push((u, v));
                                    ControlFlow::Break(seen)
                                });
                                assert_eq!(one, want[..want.len().min(1)]);
                            }
                        }
                    }
                }
            }
        }
    }

    /// Every start alignment and length against a plain copy, including ranges
    /// shorter than the head and ranges that end inside a word.
    #[test]
    fn load_bytes_equals_a_plain_copy_at_every_alignment() {
        let source: Vec<u8> = (0..96u32).map(|k| (k * 37 + 11) as u8).collect();
        for start in 0..17 {
            for len in 0..=(source.len() - start).min(48) {
                let mut out = vec![0u8; len];
                // SAFETY: the range lies inside `source`, which outlives the call.
                unsafe { load_bytes(source.as_ptr().wrapping_add(start), &mut out) };
                assert_eq!(out, &source[start..start + len], "start {start}, len {len}");
            }
        }
    }
}
