//! Unary CPU execution: one result row per sequence of one table.
//! Spec: docs/wiki/engine/engine-layout.md ("Unary kernels") and thermo/tm.md §6, §12.
//!
//! Order: options, then whole-table validation (`validate::unary_sequences`), then
//! the rows in fixed contiguous index chunks (`sched::map_ranges`), concatenated in
//! index order into whole columns. A row never depends on another row, the chunk
//! split or the thread count, so outputs are byte-identical for any `threads`.
//! There is no bound: every sequence is computed (counts.full = n).
use crate::{
    cancel::{self, Site, Stage, Ticker},
    detail::{try_reserve, try_reserve_exact, DetailColumns},
    sched::{self, Grain, Workers},
    seq::{ByteView, SeqTable},
    sink::{Counts, Sink},
    validate::{self, EngineError, ValidationReport},
};

/// A kernel with one result row per input sequence.
pub trait UnaryKernel: Sync {
    /// One sequence's complete result.
    type Row;
    /// The threshold type of the primary field's cutoff.
    type Cutoff: Copy + Sync;

    fn name(&self) -> &'static str;
    /// Inclusive length range of every sequence, checked after the alphabet scan.
    fn length_limits(&self) -> (usize, usize);
    /// The error for sequence `index` of length `length` outside `length_limits`
    /// (`limit` is the violated bound).
    fn length_error(&self, index: usize, length: usize, limit: usize) -> EngineError {
        EngineError::Sequence {
            message: format!(
                "{} input a sequence {index} has length {length}, outside the length limit {limit}; nothing was computed",
                self.name()
            ),
            input: "a",
            index,
            position: None,
            byte: None,
            n_bad: 1,
            bad_indices: vec![index],
        }
    }
    /// Estimated work of one sequence, in the units of `sched::Workers::count`.
    /// Scheduling only; it never changes an output.
    fn item_cost(&self, table: &SeqTable) -> usize {
        table.mean_len()
    }
    /// Empty output columns of the requested fields, in output order, built
    /// with fallible allocations (`DetailColumns::try_new`): worker threads call
    /// it, and a refused reservation there must be a DeviceError, not an abort.
    fn columns(&self) -> Result<DetailColumns, EngineError>;
    /// Per-unit kernel scratch (`()` for allocation-free kernels).
    type Scratch;
    /// The scratch of one scheduler unit, reserved fallibly once and reused for every
    /// sequence of the unit; a refused reservation is a DeviceError (engine layout,
    /// "Allocation failures"). A row never depends on what the scratch held before.
    fn scratch(&self) -> Result<Self::Scratch, EngineError>;
    /// The row of one validated sequence.
    fn evaluate(&self, scratch: &mut Self::Scratch, seq: ByteView<'_>) -> Self::Row;
    /// Whether `row` passes `cutoff` (inclusive, on the exposed primary field).
    fn passes(&self, row: &Self::Row, cutoff: Self::Cutoff) -> bool;
    /// Append `row`'s requested fields to `out`.
    fn push(&self, row: &Self::Row, out: &mut DetailColumns) -> Result<(), EngineError>;
}

pub struct UnaryOptions<C> {
    pub threads: usize,
    /// Scheduling grain; never changes outputs or counts.
    pub grain: Grain,
    /// `Sink::Each` or `Sink::Hits`.
    pub sink: Sink,
    pub cutoff: Option<C>,
}

pub struct UnaryOutput {
    /// Each: one row per sequence (masked rows fail the cutoff). Hits: passing rows.
    pub columns: DetailColumns,
    /// Each with a cutoff: 1 for a passing row, else 0.
    pub passed: Option<Vec<u8>>,
    /// Hits: ascending, unique sequence indices of the passing rows.
    pub i: Option<Vec<i64>>,
    pub n: usize,
    pub counts: Counts,
    pub threads: usize,
    pub n_illegal_seqs: usize,
    /// 1 when this call performed the table's alphabet scan, 0 when it was cached.
    pub validation_scan: usize,
}

struct Part {
    columns: DetailColumns,
    passed: Vec<u8>,
    i: Vec<i64>,
}

/// Options, the table's `fresh_view`, then whole-table validation; then
/// `then(table, report)` on the validated table. Shared by the CPU compute
/// (`run`) and the device path (tm-cuda.md §5.4), so both raise the same
/// errors in the same order.
pub fn validated<K: UnaryKernel, T>(
    kernel: &K,
    table: &SeqTable,
    options: &UnaryOptions<K::Cutoff>,
    then: impl FnOnce(&SeqTable, ValidationReport) -> Result<T, EngineError>,
) -> Result<T, EngineError> {
    // E0 for the whole call, whatever the caller's FP state (§7.3 item 2).
    let _guard = crate::fpenv::FpGuard::new();
    if options.threads == 0 {
        return Err(EngineError::Parameter("threads must be positive".into()));
    }
    match options.sink {
        Sink::Each => {}
        Sink::Hits if options.cutoff.is_some() => {}
        Sink::Hits => return Err(EngineError::Parameter("Hits requires a cutoff".into())),
        _ => {
            return Err(EngineError::Parameter(format!(
                "{} returns one row per sequence; use keep=sb.keep.each() or sb.keep.hits(cutoff)",
                kernel.name()
            )))
        }
    }
    let view = table.fresh_view();
    let table = view.as_ref().unwrap_or(table);
    let validation = validate::unary_sequences(
        table,
        kernel.name(),
        kernel.length_limits(),
        &|index, length, limit| kernel.length_error(index, length, limit),
        options.workers(),
    )?;
    then(table, validation)
}

impl<C> UnaryOptions<C> {
    /// Host workers of the call (validation, CPU compute, staging).
    pub fn workers(&self) -> Workers {
        Workers {
            limit: self.threads,
            grain: self.grain,
        }
    }
}

pub fn run<K: UnaryKernel>(
    kernel: &K,
    table: &SeqTable,
    options: &UnaryOptions<K::Cutoff>,
) -> Result<UnaryOutput, EngineError> {
    // E0 for the whole call, whatever the caller's FP state (§7.3 item 2).
    let _guard = crate::fpenv::FpGuard::new();
    validated(kernel, table, options, |table, validation| {
        compute(kernel, table, options, validation)
    })
}

/// The CPU compute over a validated table (also the zero-work output of the
/// device path at n = 0).
pub fn compute<K: UnaryKernel>(
    kernel: &K,
    table: &SeqTable,
    options: &UnaryOptions<K::Cutoff>,
    validation: ValidationReport,
) -> Result<UnaryOutput, EngineError> {
    // E0 for the whole call, whatever the caller's FP state (§7.3 item 2).
    let _guard = crate::fpenv::FpGuard::new();
    let each = options.sink == Sink::Each;
    let workers = options.workers();
    let n = table.len();
    let cutoff = options.cutoff;
    let compute = |start: usize, end: usize| -> Result<Part, EngineError> {
        let mut scratch = kernel.scratch()?;
        let mut part = Part {
            columns: kernel.columns()?,
            passed: Vec::new(),
            i: Vec::new(),
        };
        if each {
            part.columns.reserve(end - start)?;
            if cutoff.is_some() {
                try_reserve_exact(&mut part.passed, end - start, "passed column")?;
            }
        }
        let mut ticker = Ticker::new(Site::Sequences);
        for index in start..end {
            let row = kernel.evaluate(&mut scratch, table.get(index));
            let pass = cutoff.is_none_or(|c| kernel.passes(&row, c));
            if each {
                if pass {
                    kernel.push(&row, &mut part.columns)?;
                } else {
                    part.columns.push_masked()?;
                }
                if cutoff.is_some() {
                    part.passed.push(u8::from(pass));
                }
            } else if pass {
                kernel.push(&row, &mut part.columns)?;
                try_reserve(&mut part.i, 1, "hit indices")?;
                part.i.push(index as i64);
            }
            ticker.tick()?;
        }
        ticker.finish()?;
        Ok(part)
    };
    let limit = workers.count(n, kernel.item_cost(table));
    cancel::stage(Stage::Sequences, n);
    let mut parts = sched::map_ranges(n, limit, &compute)?;
    // Parts arrive in index order: concatenation keeps input order (Each) and
    // ascending unique indices (Hits).
    let (columns, passed, hits) = if parts.len() == 1 {
        // One part (an inline run) already is the whole output: move it, no copy.
        let part = parts.pop().expect("one part");
        (part.columns, part.passed, part.i)
    } else {
        // Reserve the merged outputs once (n rows for Each, the hit count for
        // Hits), so the appends below never reallocate.
        let rows = if each {
            n
        } else {
            parts.iter().map(|part| part.i.len()).sum()
        };
        let mut columns = kernel.columns()?;
        let mut passed = Vec::new();
        let mut hits = Vec::new();
        columns.reserve(rows)?;
        if each && cutoff.is_some() {
            try_reserve_exact(&mut passed, n, "passed column")?;
        }
        if !each {
            try_reserve_exact(&mut hits, rows, "hit indices")?;
        }
        for mut part in parts {
            columns.append(&mut part.columns)?;
            passed.append(&mut part.passed);
            hits.append(&mut part.i);
        }
        (columns, passed, hits)
    };
    Ok(UnaryOutput {
        columns,
        passed: (each && cutoff.is_some()).then_some(passed),
        i: (!each).then_some(hits),
        n,
        counts: Counts {
            certified: 0,
            pruned: 0,
            full: n,
        },
        threads: options.threads,
        n_illegal_seqs: validation.n_illegal_seqs,
        validation_scan: validation.scans[0],
    })
}

#[cfg(test)]
mod tests {
    //! A local observer kernel (the sequence length, primary >= cutoff): outputs
    //! must not depend on the worker split, and the sinks keep index order.
    use super::*;
    use crate::{
        batch::SeqBatch,
        detail::{DetailField, DetailRequest, FieldKind},
    };

    static FIELDS: [DetailField; 2] = [
        DetailField {
            name: "len",
            kind: FieldKind::Float64,
        },
        DetailField {
            name: "half",
            kind: FieldKind::Int32,
        },
    ];

    struct Length;
    impl UnaryKernel for Length {
        type Row = usize;
        type Cutoff = f64;
        type Scratch = ();
        fn scratch(&self) -> Result<(), EngineError> {
            Ok(())
        }
        fn name(&self) -> &'static str {
            "length"
        }
        fn length_limits(&self) -> (usize, usize) {
            (1, 10)
        }
        fn columns(&self) -> Result<DetailColumns, EngineError> {
            DetailColumns::try_new(&FIELDS, DetailRequest::NONE.with(0).with(1))
        }
        fn evaluate(&self, _: &mut (), seq: ByteView<'_>) -> usize {
            seq.len()
        }
        fn passes(&self, row: &usize, cutoff: f64) -> bool {
            *row as f64 >= cutoff
        }
        fn push(&self, row: &usize, out: &mut DetailColumns) -> Result<(), EngineError> {
            out.push_f64(0, *row as f64)?;
            out.push_i32(1, (*row / 2) as i32)
        }
    }

    fn table(lengths: &[usize]) -> SeqTable {
        let rows: Vec<Vec<u8>> = lengths.iter().map(|&n| vec![b'A'; n]).collect();
        SeqTable::new(SeqBatch::from_seqs(&rows))
    }

    fn options(sink: Sink, cutoff: Option<f64>, threads: usize) -> UnaryOptions<f64> {
        UnaryOptions {
            threads,
            grain: Grain::Exact,
            sink,
            cutoff,
        }
    }

    #[test]
    fn every_split_gives_the_same_columns_in_index_order() {
        let t = table(&[3, 1, 7, 2, 9, 4, 4, 10, 5]);
        for (sink, cutoff) in [
            (Sink::Each, None),
            (Sink::Each, Some(4.0)),
            (Sink::Hits, Some(4.0)),
        ] {
            let base = run(&Length, &t, &options(sink, cutoff, 1)).unwrap();
            for threads in [2, 3, 5, 16] {
                let got = run(&Length, &t, &options(sink, cutoff, threads)).unwrap();
                assert_eq!(got.columns, base.columns, "{sink:?} t={threads}");
                assert_eq!(got.passed, base.passed);
                assert_eq!(got.i, base.i);
                assert_eq!(got.counts.full, 9);
            }
            if sink == Sink::Hits {
                assert_eq!(base.i.as_deref(), Some(&[2i64, 4, 5, 6, 7, 8][..]));
                assert_eq!(base.columns.rows(), 6);
            } else {
                assert_eq!(base.columns.rows(), 9);
                assert_eq!(base.passed.is_some(), cutoff.is_some());
            }
        }
    }

    #[test]
    fn failing_rows_hold_masks() {
        let t = table(&[3, 5]);
        let out = run(&Length, &t, &options(Sink::Each, Some(4.0), 2)).unwrap();
        let mut want = Length.columns().unwrap();
        want.push_masked().unwrap();
        Length.push(&5, &mut want).unwrap();
        assert_eq!(out.columns, want);
        assert_eq!(out.passed, Some(vec![0, 1]));
    }

    #[test]
    fn length_errors_name_the_first_index_either_side() {
        let t = table(&[3, 11, 0]);
        match run(&Length, &t, &options(Sink::Each, None, 1)) {
            Err(EngineError::Sequence { index, message, .. }) => {
                assert_eq!(index, 1);
                assert!(
                    message.contains("11") && message.contains("10"),
                    "{message}"
                );
            }
            _ => panic!("expected a length error"),
        }
        let empty = run(&Length, &table(&[]), &options(Sink::Hits, Some(1.0), 4)).unwrap();
        assert_eq!(empty.columns.rows(), 0);
        assert_eq!(empty.i, Some(vec![]));
    }

    #[test]
    fn options_are_checked_first() {
        let t = table(&[0]);
        for (sink, cutoff) in [(Sink::TopK, None), (Sink::Hits, None), (Sink::Matrix, None)] {
            assert!(matches!(
                run(&Length, &t, &options(sink, cutoff, 1)),
                Err(EngineError::Parameter(_))
            ));
        }
    }
}
