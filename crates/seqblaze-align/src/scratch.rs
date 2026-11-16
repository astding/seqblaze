//! Per-thread scratch buffers: reserved fallibly (a DeviceError, never an
//! abort) and released after a pair when above 1 MiB.
use std::{cell::RefCell, ops::ControlFlow};

use seqblaze_core::{
    detail::{allocation_error, try_reserve_exact},
    seq::ByteView,
    validate::EngineError,
};

/// Buffers above this size are released after each pair, so an idle thread
/// (including the caller's own thread) does not keep a large traceback alive.
pub(crate) const KEEP_BYTES: usize = 1 << 20;

/// Per-pair symbol-class tallies for the score bound (spec §6); u64, because
/// no align limit bounds the longer sequence of a pair (§9 limit 1 bounds the
/// shorter one only when gap_extend = 0). `scratch` is all zero between uses.
pub(crate) struct Counts {
    pub scratch: [u64; 256],
    pub present_a: Vec<u8>,
    pub values_a: Vec<u64>,
    pub present_b: Vec<u8>,
    pub values_b: Vec<u64>,
}

impl Default for Counts {
    fn default() -> Self {
        Self {
            scratch: [0; 256],
            present_a: Vec::with_capacity(256),
            values_a: Vec::with_capacity(256),
            present_b: Vec::with_capacity(256),
            values_b: Vec::with_capacity(256),
        }
    }
}

/// Reported alignment of the last traceback: 0-based half-open extents in `a`
/// and `b`; the columns are `Work::ops` in forward order (`M`, `I`, `D`).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct Extent {
    pub a_start: usize,
    pub a_end: usize,
    pub b_start: usize,
    pub b_end: usize,
}

#[derive(Default)]
pub(crate) struct Work {
    /// DP rows over j = 0..=n.
    pub r0: Vec<i64>,
    pub r1: Vec<i64>,
    pub r2: Vec<i64>,
    /// One traceback byte per interior cell, row-major over i, j >= 1.
    pub codes: Vec<u8>,
    /// Per-row values of the last column and per-column values of the last row.
    pub col: Vec<[i64; 3]>,
    pub row: Vec<[i64; 3]>,
    /// Alignment columns: built backwards, then reversed in place.
    pub ops: Vec<u8>,
    pub extent: Extent,
    /// Optimal local end cells (V = score > 0) of the last native local fill
    /// that counts them (spec §3.1 `n_ends`); 0 when the score is 0.
    pub n_ends: u64,
    /// Text of one string detail field.
    pub text: Vec<u8>,
    pub counts: Counts,
    /// Byte limit per buffer, below Rust's own (isize::MAX); None outside
    /// tests. Tests lower it to reach the allocation-failure path cheaply.
    pub budget: Option<usize>,
}

#[derive(Default)]
pub(crate) struct Scratch {
    a: Vec<u8>,
    b: Vec<u8>,
    pub work: Work,
}

/// Element count `factors` multiplied, checked: an overflow is reported as an
/// allocation failure, like a size the allocator refuses.
pub(crate) fn checked_len(factors: &[usize], what: &str) -> Result<usize, EngineError> {
    factors
        .iter()
        .try_fold(1usize, |total, &factor| total.checked_mul(factor))
        .ok_or_else(|| allocation_error(what, "size overflows usize"))
}

/// Refuse `len` elements of `T` above the injected test budget.
fn check_budget<T>(len: usize, what: &str, budget: Option<usize>) -> Result<(), EngineError> {
    let Some(limit) = budget else {
        return Ok(());
    };
    match len.checked_mul(std::mem::size_of::<T>()) {
        Some(bytes) if bytes <= limit => Ok(()),
        _ => Err(allocation_error(
            what,
            format!("{len} elements exceed the {limit}-byte budget"),
        )),
    }
}

/// Room for `total` elements in a cleared or partly filled vector, reserved
/// exactly and fallibly; later pushes up to `total` never reallocate.
pub(crate) fn reserve_total<T>(
    buffer: &mut Vec<T>,
    total: usize,
    what: &str,
    budget: Option<usize>,
) -> Result<(), EngineError> {
    check_budget::<T>(total, what, budget)?;
    if buffer.capacity() < total {
        try_reserve_exact(buffer, total - buffer.len(), what)?;
    }
    Ok(())
}

/// Clear `buffer` and reserve room for `len` elements, exactly and fallibly.
/// For a buffer its user fills completely by pushing in order (the traceback
/// codes): unlike `filled`, no element is written twice.
pub(crate) fn emptied<T>(
    buffer: &mut Vec<T>,
    len: usize,
    what: &str,
    budget: Option<usize>,
) -> Result<(), EngineError> {
    buffer.clear();
    reserve_total(buffer, len, what, budget)
}

/// Resize `buffer` to exactly `len` elements of `value`, fallibly.
pub(crate) fn filled<T: Clone>(
    buffer: &mut Vec<T>,
    len: usize,
    value: T,
    what: &str,
    budget: Option<usize>,
) -> Result<(), EngineError> {
    buffer.clear();
    reserve_total(buffer, len, what, budget)?;
    buffer.resize(len, value);
    Ok(())
}

thread_local! {
    pub(crate) static SCRATCH: RefCell<Scratch> = RefCell::new(Scratch::default());
}

/// Run `body` on this thread's work buffers, without touching the pair's
/// bytes (the per-pair bound reads the rows in place before any copy).
pub(crate) fn with_work<R>(body: impl FnOnce(&mut Work) -> R) -> R {
    SCRATCH.with(|cell| body(&mut cell.borrow_mut().work))
}

/// Run `body` on the pair's bytes and this thread's work buffers. Buffers are
/// released as usual on success and on error.
pub(crate) fn with_inputs<R>(
    a: ByteView<'_>,
    b: ByteView<'_>,
    body: impl FnOnce(&[u8], &[u8], &mut Work) -> Result<R, EngineError>,
) -> Result<R, EngineError> {
    SCRATCH.with(|cell| {
        let mut guard = cell.borrow_mut();
        let scratch = &mut *guard;
        let budget = scratch.work.budget;
        let result = match (
            bytes(a, &mut scratch.a, budget),
            bytes(b, &mut scratch.b, budget),
        ) {
            (Ok(a), Ok(b)) => body(a, b, &mut scratch.work),
            (Err(error), _) | (_, Err(error)) => Err(error),
        };
        scratch.shrink();
        result
    })
}

/// A row as a slice: immutable storage directly, borrowed NumPy rows copied
/// once (relaxed loads, block by block) into `buffer`; each byte is written
/// once, with no zero pre-fill.
fn bytes<'v>(
    view: ByteView<'v>,
    buffer: &'v mut Vec<u8>,
    budget: Option<usize>,
) -> Result<&'v [u8], EngineError> {
    if let Some(slice) = view.as_slice() {
        return Ok(slice);
    }
    emptied(buffer, view.len(), "align input copy", budget)?;
    let _ = view.for_each_block(|_, block| {
        buffer.extend_from_slice(block);
        ControlFlow::<()>::Continue(())
    });
    Ok(buffer)
}

fn release<T>(buffer: &mut Vec<T>) {
    if buffer.capacity() * std::mem::size_of::<T>() > KEEP_BYTES {
        *buffer = Vec::new();
    }
}

impl Scratch {
    fn shrink(&mut self) {
        let work = &mut self.work;
        for buffer in [&mut work.r0, &mut work.r1, &mut work.r2] {
            release(buffer);
        }
        release(&mut work.col);
        release(&mut work.row);
        for buffer in [
            &mut work.codes,
            &mut work.ops,
            &mut work.text,
            &mut self.a,
            &mut self.b,
        ] {
            release(buffer);
        }
    }
}
