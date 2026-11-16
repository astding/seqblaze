//! Explicit dense output with bounded worker scratch and deterministic cell commits.
use std::sync::mpsc;

use crate::{
    cancel::{self, Site, Stage, Ticker},
    detail::{
        dense_request_bytes, try_reserve_exact, DenseRow, DetailColumns, DetailField, DetailRequest,
    },
    exec::ExecOptions,
    fpenv::FpGuard,
    kernel::{
        for_each_scored, for_each_tiled, rect_cells, BatchDistance, Outcome, PairDistanceKernel,
        Plan, Score,
    },
    seq::{SeqTable, TableRows},
    sink::{
        kept::{dense_extras, BatchDetails, DenseCell},
        Columns, Counts, RunOutput, SharedCells,
    },
    validate::{EngineError, ValidationReport},
    work::WorkSource,
};

const BATCH_PAIRS: usize = 4096;

/// Numeric detail values of a computed cell and, for a symmetric within
/// off-diagonal cell, of its mirror computed in the mirror row's orientation.
pub(crate) struct DensePair {
    pub cell: DenseRow,
    pub mirror: Option<DenseRow>,
}
pub(crate) type Record<S = i32> = (usize, usize, S, Option<Box<DensePair>>);
type Records<S> = Vec<Record<S>>;

/// Exact score of cell (i, j) plus its numeric detail values when `scratch`
/// holds a detail request. A mirrored within cell's details are never copied
/// from (i, j): they are computed as zip(seq_j, seq_i).
pub(crate) fn dense_score<S: Score, K: PairDistanceKernel<S>>(
    kernel: &K,
    (a, b): (TableRows<'_>, TableRows<'_>),
    (i, j): (usize, usize),
    mirror: bool,
    scratch: Option<&mut DetailColumns>,
    what: &str,
) -> Result<Record<S>, EngineError> {
    let missing =
        || EngineError::Parameter(format!("{what} requires an exact score without a cutoff"));
    let Some(scratch) = scratch else {
        let value = kernel
            .evaluate(a.get(i), b.get(j), None)?
            .ok_or_else(missing)?;
        return Ok((i, j, value, None));
    };
    scratch.clear();
    let value = kernel
        .evaluate_detailed(a.get(i), b.get(j), None, None, scratch)?
        .ok_or_else(missing)?;
    let cell = scratch.numeric_row(0);
    let mirror = if mirror && i != j {
        scratch.clear();
        kernel.details(a.get(j), b.get(i), value, scratch)?;
        Some(scratch.numeric_row(0))
    } else {
        None
    };
    Ok((i, j, value, Some(Box::new(DensePair { cell, mirror }))))
}

/// The exact score of a planned cell (Matrix and Tiles plan no cutoff).
fn exact<S: Score>(outcome: Outcome<S>) -> Result<S, EngineError> {
    match outcome {
        Outcome::Scored(Some(value)) => Ok(value),
        _ => Err(EngineError::Parameter(
            "matrix requires an exact score without a cutoff".into(),
        )),
    }
}

struct Dense<S: Score> {
    dist: Vec<S>,
    passed: Vec<u8>,
    details: Option<DetailColumns>,
}

impl<S: Score> Dense<S> {
    fn commit(&mut self, width: usize, within: bool, (i, j, value, extra): Record<S>) {
        self.commit_cell(width, within, (i, j, value), extra.as_deref());
    }

    /// `commit` of cells (i, j + k) with exact scores `scores[k]` and no
    /// details, a row run at a time.
    fn commit_run(
        &mut self,
        width: usize,
        within: bool,
        (i, j): (usize, usize),
        scores: &[Option<S>],
    ) -> Result<(), EngineError> {
        let start = i * width + j;
        for (cell, &score) in self.dist[start..start + scores.len()]
            .iter_mut()
            .zip(scores)
        {
            *cell = exact(Outcome::Scored(score))?;
        }
        self.passed[start..start + scores.len()].fill(1);
        if within {
            for (j, k) in (j..).zip(start..start + scores.len()) {
                if i != j {
                    let mirror = j * width + i;
                    self.dist[mirror] = self.dist[k];
                    self.passed[mirror] = 1;
                }
            }
        }
        Ok(())
    }

    /// `commit` with the detail payload borrowed, so a caller that holds it
    /// in place needs no per-cell allocation.
    fn commit_cell(
        &mut self,
        width: usize,
        within: bool,
        (i, j, value): (usize, usize, S),
        extra: Option<&DensePair>,
    ) {
        let cell = i * width + j;
        self.dist[cell] = value;
        self.passed[cell] = 1;
        if let (Some(details), Some(extra)) = (self.details.as_mut(), extra) {
            details.set_numeric_row(cell, &extra.cell);
        }
        if within && i != j {
            let mirror = j * width + i;
            self.dist[mirror] = value;
            self.passed[mirror] = 1;
            if let (Some(details), Some(row)) = (
                self.details.as_mut(),
                extra.and_then(|extra| extra.mirror.as_ref()),
            ) {
                details.set_numeric_row(mirror, row);
            }
        }
    }
}

/// `Dense`'s score and flag columns, written in place by the workers of the
/// threaded rectangle path ([`in_place`]). A worker commits only the cells of
/// its own work items: item (i, j) is cell (i, j) and, for symmetric within,
/// its mirror (j, i). Every item belongs to exactly one worker's work-ID range,
/// and a symmetric within source holds item (i, j) only for i <= j, so no
/// cell has two writers.
struct InPlace<S> {
    dist: SharedCells<S>,
    passed: SharedCells<u8>,
}

impl<S: Score> InPlace<S> {
    fn new(dense: &mut Dense<S>) -> Self {
        Self {
            dist: SharedCells::new(&mut dense.dist),
            passed: SharedCells::new(&mut dense.passed),
        }
    }

    /// `Dense::commit_run` from a worker.
    ///
    /// # Safety
    ///
    /// The run's cells (i, j + k) and, with `within`, their mirrors are the
    /// cells of this caller's own work items, which no other thread reads or
    /// writes while the workers run.
    unsafe fn commit_run(
        &self,
        width: usize,
        within: bool,
        (i, j): (usize, usize),
        scores: &[Option<S>],
    ) -> Result<(), EngineError> {
        {
            // The run's cells (i, j + k) lie back to back in row i.
            // SAFETY: they are this caller's own (the contract above), and
            // both slices end with this block.
            let (dist, passed) = unsafe {
                (
                    self.dist.run(i * width + j, scores.len()),
                    self.passed.run(i * width + j, scores.len()),
                )
            };
            for (cell, &score) in dist.iter_mut().zip(scores) {
                *cell = exact(Outcome::Scored(score))?;
            }
            passed.fill(1);
        }
        if within {
            for (j, &score) in (j..).zip(scores) {
                if i != j {
                    let value = exact(Outcome::Scored(score))?;
                    // SAFETY: the mirror of this caller's own item (the
                    // contract above).
                    unsafe {
                        self.dist.set(j * width + i, value);
                        self.passed.set(j * width + i, 1);
                    }
                }
            }
        }
        Ok(())
    }
}

/// Matrix cells without details over implicit work on `workers` threads, a
/// rectangle at a time (`for_each_tiled`): each worker commits its row runs
/// in place ([`InPlace`]) and sends its item count; returns the full count.
/// Ranges, worker start-up and failures are those of the record path.
fn in_place<S: Score, K: PairDistanceKernel<S>>(
    kernel: &K,
    (a, b): (&SeqTable, &SeqTable),
    work: &WorkSource,
    dense: &mut Dense<S>,
    workers: usize,
    cells: usize,
) -> Result<usize, EngineError> {
    let (width, within) = (work.shape.1, work.mirrors_within());
    let target = InPlace::new(dense);
    let target = &target;
    let scope_handle = cancel::current();
    std::thread::scope(|scope| -> Result<usize, EngineError> {
        let (sender, receiver) = mpsc::sync_channel::<Result<usize, EngineError>>(workers);
        let mut pending = Vec::with_capacity(workers);
        let mut start_failure = None;
        for worker in 0..workers {
            let start = (work.n_pairs as u128 * worker as u128 / workers as u128) as usize;
            let end = (work.n_pairs as u128 * (worker + 1) as u128 / workers as u128) as usize;
            let sender = sender.clone();
            let captured = scope_handle.clone();
            let spawned = std::thread::Builder::new().spawn_scoped(scope, move || {
                let _guard = FpGuard::new();
                let _unit = cancel::unit(captured);
                let mut ticker = Ticker::new(Site::Pairs);
                let mut full = 0usize;
                let walked = for_each_tiled(
                    kernel,
                    (a, b),
                    work,
                    (start, end),
                    cells,
                    |_, _| None,
                    |i, j0, scores| {
                        // SAFETY: the run holds items of this worker's
                        // work-ID range only, so their cells are its own.
                        unsafe { target.commit_run(width, within, (i, j0), scores) }?;
                        full += scores.len();
                        ticker.add(scores.len())
                    },
                )
                .and_then(|()| ticker.finish());
                let _ = sender.send(walked.map(|()| full));
            });
            match spawned {
                Ok(handle) => pending.push(handle),
                Err(error) => {
                    start_failure = Some(EngineError::Device(format!(
                        "cannot start matrix worker: {error}"
                    )));
                    break;
                }
            }
        }
        drop(sender);
        let mut full = 0usize;
        let mut error = start_failure;
        // Drain until every worker has gone; the calling thread polls meanwhile.
        while let Ok(packet) = cancel::recv(&receiver) {
            match packet {
                Err(failure) => {
                    error.get_or_insert(failure);
                }
                Ok(count) => full += count,
            }
        }
        for worker in pending {
            if worker.join().is_err() {
                error.get_or_insert_with(|| EngineError::Device("matrix worker panicked".into()));
            }
        }
        match error {
            Some(error) => Err(error),
            None => cancel::check().map(|()| full),
        }
    })
}

pub(crate) fn run<S: Score, K: PairDistanceKernel<S>>(
    kernel: &K,
    work: &WorkSource,
    a: &SeqTable,
    b: &SeqTable,
    options: &ExecOptions<S>,
    validation: ValidationReport,
) -> Result<RunOutput<S>, EngineError> {
    let request = kernel.detail_request();
    let mut dense = allocate(work, kernel.detail_fields(), request)?;
    let scratch =
        (!request.is_empty()).then(|| DetailColumns::new(kernel.detail_fields(), request));
    let within = work.mirrors_within();
    let workers = options
        .workers()
        .count(work.n_pairs, kernel.pair_cost(a, b, !request.is_empty()));
    let mut counts = Counts::default();
    cancel::stage(Stage::Pairs, work.n_pairs);
    if workers == 1 {
        // The inline unit holds its own guard, like every spawned worker (§7.3 item 1).
        let _guard = FpGuard::new();
        let _unit = cancel::inline_unit();
        let mut scratch = scratch;
        let mut ticker = Ticker::new(Site::Pairs);
        if scratch.is_none() {
            if let Some(cells) = rect_cells(kernel, work, (a, b), false) {
                // Plain scores: a rectangle of cells at a time (`for_each_tiled`).
                for_each_tiled(
                    kernel,
                    (a, b),
                    work,
                    (0, work.n_pairs),
                    cells,
                    |_, _| None,
                    |i, j, scores| {
                        dense.commit_run(work.shape.1, within, (i, j), scores)?;
                        counts.full += scores.len();
                        ticker.add(scores.len())
                    },
                )?;
            } else {
                // Plain scores: a block of cells at a time (`for_each_scored`).
                for_each_scored(
                    kernel,
                    (a, b),
                    work,
                    (0, work.n_pairs),
                    |_, _| Plan::Score(None),
                    |_, (i, j), outcome| {
                        dense.commit(work.shape.1, within, (i, j, exact(outcome)?, None));
                        counts.full += 1;
                        ticker.tick()
                    },
                )?;
            }
        } else {
            let rows = (a.rows(), b.rows());
            for pair in work.range(0, work.n_pairs) {
                let record = dense_score(kernel, rows, pair, within, scratch.as_mut(), "matrix")?;
                dense.commit(work.shape.1, within, record);
                counts.full += 1;
                ticker.tick()?;
            }
        }
        ticker.finish()?;
    } else if let Some(cells) =
        rect_cells(kernel, work, (a, b), false).filter(|_| scratch.is_none())
    {
        // Plain scores: rectangles, each worker committing its runs in place.
        counts.full = in_place(kernel, (a, b), work, &mut dense, workers, cells)?;
    } else {
        let scope_handle = cancel::current();
        counts = std::thread::scope(|scope| -> Result<Counts, EngineError> {
            let (sender, receiver) = mpsc::sync_channel::<Result<Records<S>, EngineError>>(workers);
            let mut pending = Vec::with_capacity(workers);
            let mut start_failure = None;
            for worker in 0..workers {
                let start = (work.n_pairs as u128 * worker as u128 / workers as u128) as usize;
                let end = (work.n_pairs as u128 * (worker + 1) as u128 / workers as u128) as usize;
                let sender = sender.clone();
                let mut scratch = scratch.clone();
                let captured = scope_handle.clone();
                let spawned = std::thread::Builder::new().spawn_scoped(scope, move || {
                    let _guard = FpGuard::new();
                    let _unit = cancel::unit(captured);
                    let capacity = BATCH_PAIRS.min(end - start);
                    let mut records = Vec::with_capacity(capacity);
                    let mut ticker = Ticker::new(Site::Pairs);
                    // A closed channel: the receiver has stopped, so this worker stops.
                    let closed = || EngineError::Parameter("matrix receiver closed".into());
                    let deliver = |record: Record<S>,
                                   records: &mut Records<S>,
                                   ticker: &mut Ticker|
                     -> Result<(), EngineError> {
                        ticker.tick()?;
                        records.push(record);
                        if records.len() == BATCH_PAIRS {
                            let ready = std::mem::replace(records, Vec::with_capacity(capacity));
                            sender.send(Ok(ready)).map_err(|_| closed())?;
                        }
                        Ok(())
                    };
                    let walked = if scratch.is_none() {
                        // Plain scores: a block of cells at a time (`for_each_scored`).
                        for_each_scored(
                            kernel,
                            (a, b),
                            work,
                            (start, end),
                            |_, _| Plan::Score(None),
                            |_, (i, j), outcome| {
                                let record = (i, j, exact(outcome)?, None);
                                deliver(record, &mut records, &mut ticker)
                            },
                        )
                    } else {
                        let rows = (a.rows(), b.rows());
                        work.range(start, end).try_for_each(|pair| {
                            let record = dense_score(
                                kernel,
                                rows,
                                pair,
                                within,
                                scratch.as_mut(),
                                "matrix",
                            )?;
                            deliver(record, &mut records, &mut ticker)
                        })
                    };
                    if let Err(error) = walked {
                        let _ = sender.send(Err(error));
                        return;
                    }
                    if let Err(error) = ticker.finish() {
                        let _ = sender.send(Err(error));
                        return;
                    }
                    if !records.is_empty() {
                        let _ = sender.send(Ok(records));
                    }
                });
                match spawned {
                    Ok(handle) => pending.push(handle),
                    Err(error) => {
                        start_failure = Some(EngineError::Device(format!(
                            "cannot start matrix worker: {error}"
                        )));
                        break;
                    }
                }
            }
            drop(sender);
            let mut counts = Counts::default();
            let mut error = start_failure;
            // Drain until every worker has gone; the calling thread polls meanwhile.
            while let Ok(packet) = cancel::recv(&receiver) {
                match packet {
                    Err(failure) => {
                        error.get_or_insert(failure);
                    }
                    Ok(records) => {
                        counts.full += records.len();
                        if error.is_none() {
                            for record in records {
                                dense.commit(work.shape.1, within, record);
                            }
                        }
                    }
                }
            }
            for worker in pending {
                if worker.join().is_err() {
                    error.get_or_insert_with(|| {
                        EngineError::Device("matrix worker panicked".into())
                    });
                }
            }
            match error {
                Some(error) => Err(error),
                None => cancel::check().map(|()| counts),
            }
        })?;
    }
    Ok(output(work, options, validation, counts, dense))
}

/// Checked dense allocation (scores, flags and the requested numeric detail
/// fields); every reservation completes before any field is touched.
fn allocate<S: Score>(
    work: &WorkSource,
    fields: &[DetailField],
    request: DetailRequest,
) -> Result<Dense<S>, EngineError> {
    let cells = work.shape.0.checked_mul(work.shape.1).ok_or_else(|| {
        EngineError::Overflow("matrix shape product exceeds platform limit".into())
    })?;
    let detail_bytes = dense_request_bytes(fields, request)?;
    let _total_bytes = cells
        .checked_mul(5 + detail_bytes)
        .ok_or_else(|| EngineError::Overflow("matrix byte count exceeds platform limit".into()))?;
    if cells
        .checked_mul(std::mem::size_of::<S>())
        .is_none_or(|bytes| bytes > isize::MAX as usize)
    {
        return Err(EngineError::Overflow(
            "matrix distance buffer exceeds addressable allocation size".into(),
        ));
    }
    let mut dist = Vec::new();
    let mut passed = Vec::new();
    try_reserve_exact(&mut dist, cells, "matrix dist column")?;
    try_reserve_exact(&mut passed, cells, "matrix passed column")?;
    let details = if request.is_empty() {
        None
    } else {
        Some(DetailColumns::masked_numeric(
            fields,
            request,
            cells,
            "matrix detail fields",
        )?)
    };
    // Complete every reservation before touching fields or launching work.
    dist.resize(cells, S::MASK);
    passed.resize(cells, 0u8);
    Ok(Dense {
        dist,
        passed,
        details,
    })
}

fn output<S: Score>(
    work: &WorkSource,
    options: &ExecOptions<S>,
    validation: ValidationReport,
    counts: Counts,
    dense: Dense<S>,
) -> RunOutput<S> {
    RunOutput {
        columns: Columns {
            dist: dense.dist,
            passed: Some(dense.passed),
            details: dense.details,
            ..Columns::default()
        },
        shape: work.shape,
        counts,
        threads: options.threads,
        n_illegal_seqs: validation.n_illegal_seqs,
        validation_scans: validation.scans,
        topk_replay: None,
    }
}

/// Matrix over a batch-distance backend: the same allocation and cell commits
/// as `run`, with each work item scored exactly once in work-ID order. Numeric
/// detail rows come from the port when the backend delivers them, else from the
/// host `details` port; a symmetric-within mirror cell's row is always computed
/// as zip(seq_j, seq_i) on the host, as in `run`.
pub fn run_batch<S: Score, K: PairDistanceKernel<S> + ?Sized>(
    kernel: &K,
    tables: (&SeqTable, &SeqTable),
    distances: &mut dyn BatchDistance<S>,
    work: &WorkSource,
    options: &ExecOptions<S>,
    validation: ValidationReport,
) -> Result<RunOutput<S>, EngineError> {
    let mut dense = allocate(work, kernel.detail_fields(), kernel.detail_request())?;
    let within = work.mirrors_within();
    let mode = BatchDetails::of(kernel, &*distances);
    let workers = options.workers();
    let width = work.shape.1;
    let mut counts = Counts::default();
    let port = mode == BatchDetails::Port;
    cancel::stage(Stage::Pairs, work.n_pairs);
    distances.range(work, 0, work.n_pairs, port, &mut |first, values, rows| {
        let pairs = work.range(first, first + values.len());
        if mode == BatchDetails::None {
            for ((i, j), &value) in pairs.zip(values) {
                dense.commit(width, within, (i, j, value, None));
                counts.full += 1;
            }
            // A host boundary after each delivered chunk (cancellation-progress.md §3.5).
            return cancel::boundary(Site::Pairs, values.len());
        }
        // Fallible reservation, and no per-cell allocation at the commit: a
        // refused host reservation raises DeviceError instead of aborting.
        let mut cells: Vec<DenseCell<S>> = Vec::new();
        try_reserve_exact(&mut cells, values.len(), "dense detail cells")?;
        cells.extend(
            pairs
                .zip(values)
                .enumerate()
                .map(|(k, ((i, j), &value))| DenseCell {
                    i,
                    j,
                    value,
                    mirror: within && i != j,
                    row: rows.is_some().then_some(k),
                }),
        );
        let extras = dense_extras(kernel, tables, &cells, rows, workers)?;
        for (cell, extra) in cells.iter().zip(extras.pairs(&cells)) {
            dense.commit_cell(width, within, (cell.i, cell.j, cell.value), Some(&extra?));
            counts.full += 1;
        }
        cancel::boundary(Site::Pairs, values.len())
    })?;
    Ok(output(work, options, validation, counts, dense))
}

#[cfg(test)]
mod tests {
    //! §11 item 13: the Matrix workers, inline and spawned, compute in E0 when
    //! the caller holds no guard, and leave the caller's state intact.
    use super::*;
    use crate::{
        batch::SeqBatch,
        exec::{Grain, Strategy},
        fpenv::{
            read_state,
            testing::{computes_in_e0, install, Restore, HOSTILE},
        },
        plan::BoundAvailability,
        sched::Workers,
        seq::ByteView,
        sink::Sink,
        work::PairKind,
    };

    /// Scores 1 when the evaluating thread computes in E0, else 0.
    struct Environment;
    impl PairDistanceKernel for Environment {
        fn bound_availability(&self) -> BoundAvailability {
            BoundAvailability::Unavailable
        }
        fn prepare_bound(&self, _: &SeqTable, _: &SeqTable, _: Workers) -> Result<(), EngineError> {
            Ok(())
        }
        fn lower_bound(&self, _: &SeqTable, _: &SeqTable, _: usize, _: usize) -> i32 {
            0
        }
        fn evaluate(
            &self,
            _: ByteView<'_>,
            _: ByteView<'_>,
            _: Option<i32>,
        ) -> Result<Option<i32>, EngineError> {
            Ok(Some(i32::from(computes_in_e0())))
        }
    }

    #[test]
    fn matrix_workers_compute_in_e0_from_an_unguarded_hostile_caller() {
        let table = SeqTable::new(SeqBatch::from_seqs([b"AC", b"GT", b"CA", b"TG"]));
        let _restore = Restore(read_state());
        for hostile in HOSTILE {
            let Some(hostile) = install(hostile) else {
                continue;
            };
            for threads in [1, 2, 3] {
                let work = WorkSource::new(PairKind::Cross, 4, 4).unwrap();
                let options = ExecOptions {
                    strategy: Strategy::Full,
                    threads,
                    grain: Grain::Exact,
                    sink: Sink::Matrix,
                    cutoff: None,
                    check_bases: false,
                    k: None,
                };
                let validation = ValidationReport {
                    scans: [0, 0],
                    n_illegal_seqs: 0,
                };
                let output =
                    run(&Environment, &work, &table, &table, &options, validation).unwrap();
                assert_eq!(output.columns.dist, vec![1; 16], "threads {threads}");
                assert_eq!(read_state(), hostile, "threads {threads}");
            }
        }
    }
}
