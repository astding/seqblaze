//! Deterministic, whole-input validation before any scoring.
use std::fmt;

use crate::sched::Workers;
use crate::seq::{AlphabetScan, SeqTable};
use crate::work::{Orientation, PairKind, WorkSource};

pub const HAMMING_MAX_LENGTH: usize = i32::MAX as usize;

#[derive(Debug, Clone)]
pub enum EngineError {
    Parameter(String),
    Overflow(String),
    Pairing(String),
    Sequence {
        message: String,
        input: &'static str,
        index: usize,
        position: Option<usize>,
        byte: Option<u8>,
        n_bad: usize,
        bad_indices: Vec<usize>,
    },
    Device(String),
    /// The call's cancel scope has its flag set (cancellation-progress.md §3.1).
    /// Internal: the binding raises the stored signal-handler exception instead,
    /// and never maps this variant to a Python class of its own.
    Cancelled,
}

/// Message of a DeviceError whose own message could not be reserved
/// (`crate::detail::device_error`): the error carries an empty String and
/// displays this text.
pub const OUT_OF_MEMORY: &str = "unable to reserve memory; no result was published";

/// DeviceError text of a `Cancelled` that reaches a binding with no stored
/// signal exception (a bug; cancellation-progress.md §3.4 rule 2).
pub const CANCELLED_WITHOUT_SIGNAL: &str =
    "internal error: cancelled without a pending signal exception; no result was published";

impl fmt::Display for EngineError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Device(s) if s.is_empty() => f.write_str(OUT_OF_MEMORY),
            Self::Parameter(s) | Self::Pairing(s) | Self::Device(s) | Self::Overflow(s) => {
                f.write_str(s)
            }
            Self::Sequence { message, .. } => f.write_str(message),
            Self::Cancelled => f.write_str(CANCELLED_WITHOUT_SIGNAL),
        }
    }
}
impl std::error::Error for EngineError {}

/// Step 2 of `hamming` (D48 §1.5): the smallest canonical pair k whose rows
/// are both non-null and differ in byte length (text-nulls-fp-environment.md
/// §4.1); null pairs are skipped and reported by step 3. Implicit shapes keep
/// their O(n_a + n_b) reference-row checks over the non-null rows.
pub fn equal_lengths(a: &SeqTable, b: &SeqTable, work: &WorkSource) -> Result<(), EngineError> {
    equal_lengths_for(a, b, work, "hamming requires equal lengths")
}

/// [`equal_lengths`] with the message's requirement clause named by the caller
/// (`tm` on Pairs, tm.md §13.6 step 2); the rule and the pair it reports are the same.
pub fn equal_lengths_for(
    a: &SeqTable,
    b: &SeqTable,
    work: &WorkSource,
    requirement: &str,
) -> Result<(), EngineError> {
    a.validate_lengths()?;
    if !std::ptr::eq(a, b) {
        b.validate_lengths()?;
    }
    // Tables whose rows all share one length (null rows included) hold no
    // unequal pair in any shape.
    if a.is_empty()
        || b.is_empty()
        || a.uniform_length()
            .is_some_and(|len| b.uniform_length() == Some(len))
    {
        return Ok(());
    }
    let (rows_a, rows_b) = (a.rows(), b.rows());
    let mismatch =
        |i: usize, j: usize, k: usize| {
            (rows_a.seq_len(i) != rows_b.seq_len(j)).then(|| {
                EngineError::Pairing(format!(
            "{requirement}: pair {k} ({i}, {j}) has lengths {} and {}; nothing was computed",
            rows_a.seq_len(i), rows_b.seq_len(j)))
            })
        };
    let first_valid = |table: &SeqTable| (0..table.len()).find(|&r| !table.is_null(r));
    match &work.kind {
        PairKind::Cross | PairKind::OneToMany => {
            // Reference rows i*, j*: the first non-null rows (rows before i*
            // hold only null pairs). Without nulls they are 0 and 0.
            let (Some(i_star), Some(j_star)) = (first_valid(a), first_valid(b)) else {
                return Ok(());
            };
            for j in 0..b.len() {
                if !b.is_null(j) {
                    if let Some(error) = mismatch(i_star, j, i_star * b.len() + j) {
                        return Err(error);
                    }
                }
            }
            for i in i_star + 1..a.len() {
                if !a.is_null(i) {
                    if let Some(error) = mismatch(i, j_star, i * b.len() + j_star) {
                        return Err(error);
                    }
                }
            }
        }
        PairKind::Within { self_pairs } | PairKind::WithinOriented { self_pairs, .. } => {
            let Some(r_star) = first_valid(a) else {
                return Ok(());
            };
            // Row r* in canonical order: the rows before it are null, so for
            // every orientation the pair (r*, j > r*) is the first failing one.
            let n = a.len();
            let s = usize::from(*self_pairs);
            let row_start = match &work.kind {
                PairKind::WithinOriented {
                    orientation: Orientation::Both,
                    ..
                } => r_star * (n - 1 + s),
                // Upper triangle, row-major: rows r < r* hold n - 1 - r + s pairs.
                _ => r_star * (n - 1 + s) - r_star * r_star.saturating_sub(1) / 2,
            };
            for j in r_star + 1..n {
                if !a.is_null(j) {
                    let column = match &work.kind {
                        PairKind::WithinOriented {
                            orientation: Orientation::Both,
                            ..
                        } => j - 1 + s,
                        _ => j - r_star - 1 + s,
                    };
                    if let Some(error) = mismatch(r_star, j, row_start + column) {
                        return Err(error);
                    }
                }
            }
        }
        _ => {
            let nulls = !a.nulls().is_empty() || !b.nulls().is_empty();
            for (k, (i, j)) in work.range(0, work.n_pairs).enumerate() {
                if nulls && (a.is_null(i) || b.is_null(j)) {
                    continue;
                }
                if let Some(error) = mismatch(i, j, k) {
                    return Err(error);
                }
            }
        }
    }
    Ok(())
}

pub struct ValidationReport {
    pub scans: [usize; 2],
    pub n_illegal_seqs: usize,
}

pub fn sequences(
    a: &SeqTable,
    b: &SeqTable,
    check_bases: bool,
) -> Result<ValidationReport, EngineError> {
    sequences_for_kernel(a, b, check_bases, "hamming", Some(HAMMING_MAX_LENGTH))
}

pub fn sequences_for_kernel(
    a: &SeqTable,
    b: &SeqTable,
    check_bases: bool,
    kernel: &str,
    length_limit: Option<usize>,
) -> Result<ValidationReport, EngineError> {
    sequences_for_kernel_with_threads(a, b, check_bases, kernel, length_limit, 1)
}

pub fn sequences_for_kernel_with_threads(
    a: &SeqTable,
    b: &SeqTable,
    check_bases: bool,
    kernel: &str,
    length_limit: Option<usize>,
    threads: usize,
) -> Result<ValidationReport, EngineError> {
    sequences_for_kernel_with_workers(
        a,
        b,
        check_bases,
        kernel,
        length_limit,
        Workers::exact(threads),
    )
}

/// Whole-table validation whose alphabet scans use `workers` (limit and grain).
pub fn sequences_for_kernel_with_workers(
    a: &SeqTable,
    b: &SeqTable,
    check_bases: bool,
    kernel: &str,
    length_limit: Option<usize>,
    workers: Workers,
) -> Result<ValidationReport, EngineError> {
    let error = |input: &'static str, index: usize, length: usize, limit: usize| {
        EngineError::Sequence {
            message: format!("{kernel} input {input} sequence {index} length {length} exceeds length limit {limit}; nothing was computed"),
            input, index, position: None, byte: None, n_bad: 1, bad_indices: vec![index],
        }
    };
    sequences_with_caps(
        a,
        b,
        check_bases,
        kernel,
        [length_limit, length_limit],
        &error,
        workers,
    )
}

/// The default alphabet's description in SequenceError messages.
const ACGT_UPPER: &str = "A, C, G, T (upper case)";

/// Legal bytes of a kernel call (D48 §1.1, §1.5).
#[derive(Debug, Clone, Copy)]
pub enum Alphabet {
    /// Upper-case A, C, G, T. With `check_bases=False` the scan only counts
    /// the other sequences, for the warning.
    Acgt,
    /// Exactly the bytes set in `legal`. Any other byte is a SequenceError,
    /// whatever `check_bases` says; `describe` completes "accepts only …" and
    /// `hint` (empty or whole sentences) follows "Nothing was computed.".
    Exact {
        legal: &'static [bool; 256],
        describe: &'static str,
        hint: &'static str,
    },
    /// Every byte value is legal; no scan is needed.
    Bytes,
}

/// Whole-table validation with a per-input length cap `[a, b]`; `error` builds
/// the kernel's message for the first sequence above its input's cap.
pub fn sequences_with_caps(
    a: &SeqTable,
    b: &SeqTable,
    check_bases: bool,
    kernel: &str,
    caps: [Option<usize>; 2],
    error: &dyn Fn(&'static str, usize, usize, usize) -> EngineError,
    workers: Workers,
) -> Result<ValidationReport, EngineError> {
    sequences_with_alphabet(
        a,
        b,
        check_bases,
        kernel,
        Alphabet::Acgt,
        caps,
        error,
        workers,
    )
}

/// [`sequences_with_caps`] over a kernel-supplied alphabet.
#[allow(clippy::too_many_arguments)]
pub fn sequences_with_alphabet(
    a: &SeqTable,
    b: &SeqTable,
    check_bases: bool,
    kernel: &str,
    alphabet: Alphabet,
    caps: [Option<usize>; 2],
    error: &dyn Fn(&'static str, usize, usize, usize) -> EngineError,
    workers: Workers,
) -> Result<ValidationReport, EngineError> {
    let mut report = ValidationReport {
        scans: [0, 0],
        n_illegal_seqs: 0,
    };
    let shared = std::ptr::eq(a, b);
    // One progress stage over every sequence this call scans (cached scans excluded).
    let to_scan = |table: &SeqTable| match alphabet {
        Alphabet::Acgt if !table.scan_cached() => table.len(),
        Alphabet::Exact { legal, .. } if !table.exact_scan_cached(legal) => table.len(),
        _ => 0,
    };
    let total = to_scan(a) + if shared { 0 } else { to_scan(b) };
    crate::cancel::stage(crate::cancel::Stage::Validate, total);
    for (side, (name, table)) in [("a", a), ("b", b)].into_iter().enumerate() {
        if side == 1 && shared {
            continue;
        }
        let (scan, strict, accepts, hint) = match alphabet {
            Alphabet::Acgt => {
                let (scan, scanned) = table.alphabet_scan_with(workers)?;
                report.scans[side] = usize::from(scanned);
                report.n_illegal_seqs += scan.n_bad;
                (Some(scan.clone()), check_bases, ACGT_UPPER, "")
            }
            Alphabet::Exact {
                legal,
                describe,
                hint,
            } => {
                let (scan, scanned) = table.alphabet_scan_exact(legal, workers)?;
                report.scans[side] = usize::from(scanned);
                (Some(scan), true, describe, hint)
            }
            // Every byte is legal: no scan, but nulls are still invalid.
            Alphabet::Bytes => (None, false, "", ""),
        };
        // Bad = Null ∪ Ill; Ill is empty when the call checks no alphabet.
        let ill = if strict { scan.as_ref() } else { None };
        if let Some(error) = sequence_error(name, kernel, (accepts, hint), table, ill) {
            return Err(error);
        }
    }
    // Alphabet validation on both tables precedes all per-kernel length caps.
    // The cached extrema clear a cap without a row walk; a table above it
    // keeps the index-order walk, so the first reported row is unchanged.
    for ((name, table), cap) in [("a", a), ("b", b)].into_iter().zip(caps) {
        if let Some(limit) = cap {
            if table.length_bounds().is_some_and(|(_, hi)| hi > limit) {
                let rows = table.rows();
                if let Some(index) = (0..table.len()).find(|&i| rows.seq_len(i) > limit) {
                    return Err(error(name, index, rows.seq_len(index), limit));
                }
            }
        }
    }
    Ok(report)
}

/// Step 3 of one table (text-nulls-fp-environment.md §4.2): Bad = Null ∪ Ill,
/// where Ill holds the rows `scan` found illegal (None: the call checks no
/// alphabet). The error names k = min Bad, with |Bad| and the five smallest
/// rows of Bad: the D42 illegal-base report when row k holds an illegal byte,
/// the null report (no position, no byte) when row k is null.
fn sequence_error(
    name: &'static str,
    kernel: &str,
    (accepts, hint): (&str, &str),
    table: &SeqTable,
    scan: Option<&AlphabetScan>,
) -> Option<EngineError> {
    let nulls = table.nulls();
    let ill = scan.filter(|scan| scan.n_bad > 0);
    if nulls.is_empty() && ill.is_none() {
        return None;
    }
    let (n_ill, ill_rows, first_ill) = match ill {
        Some(scan) => (scan.n_bad, scan.bad_indices.as_slice(), scan.first),
        None => (0, &[][..], None),
    };
    // Disjoint sets: null rows hold no bytes. Merge the two ascending prefixes.
    let n_bad = nulls.len() + n_ill;
    let mut bad_indices = Vec::with_capacity(5);
    let (mut x, mut y) = (0, 0);
    while bad_indices.len() < 5 && (x < nulls.len().min(5) || y < ill_rows.len()) {
        let take_null = y >= ill_rows.len() || (x < nulls.len() && nulls[x] < ill_rows[y]);
        if take_null {
            bad_indices.push(nulls[x]);
            x += 1;
        } else {
            bad_indices.push(ill_rows[y]);
            y += 1;
        }
    }
    let n = table.len();
    match first_ill {
        Some((index, position, byte)) if nulls.first().is_none_or(|&null| index < null) => {
            let hint = if hint.is_empty() {
                String::new()
            } else {
                format!(" {hint}")
            };
            Some(EngineError::Sequence {
                message: format!("illegal base in input {name}: sequence {index}, position {position} is byte {byte}; {kernel} accepts only {accepts}. {n_bad} of {n} sequences are invalid (indices {bad_indices:?}). Nothing was computed.{hint}"),
                input: name, index, position: Some(position), byte: Some(byte),
                n_bad, bad_indices,
            })
        }
        _ => {
            let index = nulls[0];
            Some(EngineError::Sequence {
                message: format!("null sequence in input {name}: sequence {index} in input {name} is null; {kernel} cannot compute a null. {n_bad} of {n} sequences are invalid (indices {bad_indices:?}). Nothing was computed."),
                input: name, index, position: None, byte: None,
                n_bad, bad_indices,
            })
        }
    }
}

/// Whole-table validation of a unary kernel's one input `a`: length metadata,
/// then the strict upper-case ACGT scan over every sequence, then the inclusive
/// length range `[min, max]` over every sequence in index order. `error(index,
/// length, limit)` builds the kernel's message for the first sequence outside it.
pub fn unary_sequences(
    table: &SeqTable,
    kernel: &str,
    (min, max): (usize, usize),
    error: &dyn Fn(usize, usize, usize) -> EngineError,
    workers: Workers,
) -> Result<ValidationReport, EngineError> {
    table.validate_lengths()?;
    if !table.scan_cached() {
        crate::cancel::stage(crate::cancel::Stage::Validate, table.len());
    }
    let (scan, scanned) = table.alphabet_scan_with(workers)?;
    if let Some(error) = sequence_error("a", kernel, (ACGT_UPPER, ""), table, Some(scan)) {
        return Err(error);
    }
    // Repeated unary calls on an immutable table need no per-row length walk
    // when cached extrema prove the whole range. On a failure, retain the
    // original index-order scan so the first reported row is unchanged.
    if table
        .length_bounds()
        .is_some_and(|(lo, hi)| lo < min || hi > max)
    {
        for index in 0..table.len() {
            let length = table.seq_len(index);
            if length < min {
                return Err(error(index, length, min));
            }
            if length > max {
                return Err(error(index, length, max));
            }
        }
    }
    Ok(ValidationReport {
        scans: [usize::from(scanned), 0],
        n_illegal_seqs: scan.n_bad,
    })
}
