//! Lazy dense tiles. Only the current final tile and bounded worker chunks are live.
use std::sync::Arc;

use crate::{
    cancel,
    detail::{dense_request_bytes, try_reserve_exact, DetailColumns, DetailField, DetailRequest},
    exec::Strategy,
    kernel::{BatchDistance, PairDistanceKernel, Score},
    sched::Workers,
    seq::SeqTable,
    sink::{
        kept::{dense_extras, BatchDetails, DenseCell},
        matrix::{dense_score, DensePair},
        Columns, Counts, SharedCells,
    },
    validate::{EngineError, ValidationReport},
    work::{Orientation, PairKind, WorkSource},
};

#[derive(Clone, Copy)]
struct Region {
    r0: usize,
    r1: usize,
    c0: usize,
    c1: usize,
}
impl Region {
    fn shape(self) -> (usize, usize) {
        (self.r1 - self.r0, self.c1 - self.c0)
    }
    fn contains(self, i: usize, j: usize) -> bool {
        (self.r0..self.r1).contains(&i) && (self.c0..self.c1).contains(&j)
    }
    fn cell(self, i: usize, j: usize) -> usize {
        (i - self.r0) * self.shape().1 + j - self.c0
    }
}

struct State {
    a: Arc<SeqTable>,
    b: Arc<SeqTable>,
    work: WorkSource,
    index_order: Vec<usize>,
    index_next: usize,
}

pub struct TileOutput<S: Score = i32> {
    pub columns: Columns<S>,
    pub shape: (usize, usize),
    pub rows: (usize, usize),
    pub cols: (usize, usize),
}

/// Position and counters of a [`TileStream`] (see [`TileStream::mark`]).
#[derive(Debug, Clone, Copy)]
pub struct TileMark {
    row: usize,
    col: usize,
    full: usize,
    replayed: usize,
    evaluations: usize,
    yielded: usize,
    complete: bool,
}

pub struct TileStream {
    state: Option<State>,
    /// The state of a stream that completed or failed inside `next_tile`,
    /// kept until `drop_retired` (or `close`, or the stream's drop): dropping
    /// the last table reference can run a producer's release callback, which
    /// must run in the caller's FP environment, never inside the E0 guard
    /// (text-nulls-fp-environment.md §7.3).
    retired: Option<State>,
    tile_shape: (usize, usize),
    row: usize,
    col: usize,
    pub shape: (usize, usize),
    pub strategy: Strategy,
    pub threads: usize,
    workers: Workers,
    pair_cost: usize,
    /// Cells per rectangle of the CPU tiles without details over implicit
    /// work, for a rectangle kernel (`PairDistanceKernel::rect_cells`).
    rect_cells: Option<usize>,
    /// Per-cell byte width of the requested numeric detail fields.
    detail_bytes: usize,
    /// The creating kernel's detail fields and this call's request.
    detail_fields: &'static [DetailField],
    request: DetailRequest,
    pub validation: ValidationReport,
    pub counts: Counts,
    pub evaluations: usize,
    pub replayed: usize,
    pub yielded: usize,
    pub complete: bool,
    pub closed: bool,
}

fn increment(value: usize, delta: usize) -> Result<usize, EngineError> {
    value
        .checked_add(delta)
        .ok_or_else(|| EngineError::Overflow("tile work counter exceeds platform limit".into()))
}

/// `(i, j, canonical, value)` of a requested cell.
type Record<S> = (usize, usize, bool, S);
/// Scores a tile's tasks and commits each requested record to the tile.
type Scorer<'s, S> = dyn FnMut(&Tasks<'_>, usize, &SeqTable, &SeqTable, &mut Tile<'_, S>) -> Result<(), EngineError>
    + 's;

/// The current tile's columns and counters, which a scorer fills record by
/// record ([`Tile::consume`]) or, on the rectangle path, cell by cell from
/// its workers ([`rect_tile`]).
struct Tile<'t, S: Score> {
    region: Region,
    within: bool,
    dist: &'t mut [S],
    passed: &'t mut [u8],
    details: Option<&'t mut DetailColumns>,
    full: usize,
    replayed: usize,
}

impl<S: Score> Tile<'_, S> {
    /// Commits a record with its details, borrowed: those of the cell and, for
    /// a shared symmetric-within mirror, of the mirror cell (None without a
    /// detail request). Borrowing lets the batch path commit with no per-cell
    /// allocation.
    fn consume(
        &mut self,
        (i, j, canonical, value): Record<S>,
        extra: Option<&DensePair>,
    ) -> Result<(), EngineError> {
        let region = self.region;
        let cell = region.cell(i, j);
        self.dist[cell] = value;
        self.passed[cell] = 1;
        if let (Some(details), Some(extra)) = (self.details.as_deref_mut(), extra) {
            details.set_numeric_row(cell, &extra.cell);
        }
        if self.within && i != j && region.contains(j, i) {
            let mirror = region.cell(j, i);
            self.dist[mirror] = value;
            self.passed[mirror] = 1;
            if let (Some(details), Some(row)) = (
                self.details.as_deref_mut(),
                extra.and_then(|extra| extra.mirror.as_ref()),
            ) {
                details.set_numeric_row(mirror, row);
            }
        }
        if canonical {
            self.full = increment(self.full, 1)?;
        } else {
            self.replayed = increment(self.replayed, 1)?;
        }
        Ok(())
    }
}

struct Tasks<'a> {
    work: &'a WorkSource,
    region: Region,
    indices: &'a [usize],
}
impl Tasks<'_> {
    fn len(&self, cells: usize) -> usize {
        match self.work.kind {
            PairKind::Zip => self
                .region
                .r1
                .min(self.region.c1)
                .saturating_sub(self.region.r0.max(self.region.c0)),
            PairKind::Index { .. } => self.indices.len(),
            _ => cells,
        }
    }
    fn at(&self, k: usize) -> Option<(usize, usize, bool)> {
        match &self.work.kind {
            PairKind::Zip => {
                let i = self.region.r0.max(self.region.c0) + k;
                Some((i, i, true))
            }
            PairKind::Index { i, j } => {
                let id = self.indices[k];
                Some((i.get(id), j.get(id), true))
            }
            _ => {
                let i = self.region.r0 + k / self.region.shape().1;
                let j = self.region.c0 + k % self.region.shape().1;
                self.class(i, j).map(|canonical| (i, j, canonical))
            }
        }
    }
    /// Cell (i, j) of an implicit kind's region: None for a hole, else
    /// whether it is its work item's canonical visit.
    #[inline]
    fn class(&self, i: usize, j: usize) -> Option<bool> {
        match self.work.kind {
            PairKind::Within { self_pairs } => {
                if (i == j && !self_pairs) || (i > j && self.region.contains(j, i)) {
                    None
                } else {
                    Some(i <= j)
                }
            }
            PairKind::WithinOriented {
                self_pairs,
                orientation,
            } => {
                if (i == j && !self_pairs) || (orientation == Orientation::AsGiven && i > j) {
                    None
                } else {
                    Some(true)
                }
            }
            _ => Some(true),
        }
    }
    /// Columns `j0..j1` of region row i without their leading and trailing
    /// holes (`class`), for an implicit kind; holes inside remain. The holes
    /// of a row are its diagonal cell without self pairs, the cells left of
    /// the diagonal for oriented within as given, and for symmetric within,
    /// when i is a column of the region, the columns r0..min(i, r1) (whose
    /// mirrors the region holds).
    fn trim(&self, i: usize, (mut j0, mut j1): (usize, usize)) -> (usize, usize) {
        let region = self.region;
        let diagonal = match self.work.kind {
            PairKind::Within { self_pairs } => {
                if (region.c0..region.c1).contains(&i) && region.r0 <= j0 {
                    j0 = j0.max(i.min(region.r1));
                }
                !self_pairs
            }
            PairKind::WithinOriented {
                self_pairs,
                orientation,
            } => {
                if orientation == Orientation::AsGiven {
                    j0 = j0.max(i);
                }
                !self_pairs
            }
            _ => false,
        };
        j1 = j1.max(j0);
        if diagonal && j0 < j1 && j0 == i {
            j0 += 1;
        }
        if diagonal && j0 < j1 && j1 - 1 == i {
            j1 -= 1;
        }
        (j0, j1)
    }
}

/// The tile's requested cells of implicit work without details, through a
/// rectangle kernel's `evaluate_rect` (`for_each_span_tiled`, rectangles of up
/// to `rect` cells over each unit's task rows): the units, their task ranges,
/// worker start-up, failures and checkpoints (one per 4,096 records) are
/// those of `next_tile`'s stream, and every hole cell is computed and
/// discarded. Each unit writes its records' cells in place, and their
/// in-tile mirrors: a record's cell and a requested cell's in-tile mirror
/// (which is a hole) belong to one task, so to one unit. The counts are
/// summed afterwards; their order is moot.
fn rect_tile<S: Score, K: PairDistanceKernel<S>>(
    kernel: &K,
    tasks: &Tasks<'_>,
    tables: (&SeqTable, &SeqTable),
    tile: &mut Tile<'_, S>,
    (workers, pair_cost): (Workers, usize),
    rect: usize,
) -> Result<(), EngineError> {
    let region = tasks.region;
    let width = region.shape().1;
    let n_tasks = tasks.len(region.shape().0 * width);
    let units = workers.count(n_tasks, pair_cost).min(n_tasks.max(1));
    let within = tile.within;
    let (dist, passed) = (
        SharedCells::new(&mut *tile.dist),
        SharedCells::new(&mut *tile.passed),
    );
    let (dist, passed) = (&dist, &passed);
    let unit = |start: usize, end: usize| -> Result<(usize, usize), EngineError> {
        let mut ticker = cancel::Ticker::new(cancel::Site::Tile);
        let (mut full, mut replayed) = (0usize, 0usize);
        // Task k is cell (r0 + k / width, c0 + k % width); each row's span
        // loses its leading and trailing holes.
        let mut k = start;
        let spans = std::iter::from_fn(|| {
            (k < end).then(|| {
                let (row, col) = (k / width, k % width);
                let count = (width - col).min(end - k);
                k += count;
                let i = region.r0 + row;
                let (j0, j1) = tasks.trim(i, (region.c0 + col, region.c0 + col + count));
                (i, j0, j1)
            })
        })
        .filter(|&(_, j0, j1)| j0 < j1);
        crate::kernel::for_each_span_tiled(
            kernel,
            tables,
            spans,
            false,
            rect,
            |_, _| None,
            |i, j0, scores| {
                let mut records = 0;
                for (j, &score) in (j0..).zip(scores) {
                    let Some(canonical) = tasks.class(i, j) else {
                        continue;
                    };
                    let value = score.ok_or_else(|| {
                        EngineError::Parameter(
                            "Tiles requires an exact score without a cutoff".into(),
                        )
                    })?;
                    // SAFETY: the record's cell and its in-tile mirror belong
                    // to this unit's task (the function's note).
                    unsafe {
                        dist.set(region.cell(i, j), value);
                        passed.set(region.cell(i, j), 1);
                        if within && i != j && region.contains(j, i) {
                            dist.set(region.cell(j, i), value);
                            passed.set(region.cell(j, i), 1);
                        }
                    }
                    if canonical {
                        full = increment(full, 1)?;
                    } else {
                        replayed = increment(replayed, 1)?;
                    }
                    records += 1;
                }
                ticker.add(records)
            },
        )?;
        ticker.finish()?;
        Ok((full, replayed))
    };
    let mut add = |(full, replayed): (usize, usize)| -> Result<(), EngineError> {
        tile.full = increment(tile.full, full)?;
        tile.replayed = increment(tile.replayed, replayed)?;
        Ok(())
    };
    if units == 1 {
        let _guard = crate::fpenv::FpGuard::new();
        let _unit = cancel::inline_unit();
        return add(unit(0, n_tasks)?);
    }
    let scope_handle = cancel::current();
    std::thread::scope(|scope| -> Result<(), EngineError> {
        let (sender, receiver) = std::sync::mpsc::sync_channel(units);
        let mut pending = Vec::with_capacity(units);
        let mut failure = None;
        for worker in 0..units {
            let (start, end) = crate::sched::bounds(n_tasks, worker, units);
            let (sender, captured, unit) = (sender.clone(), scope_handle.clone(), &unit);
            let spawned = std::thread::Builder::new().spawn_scoped(scope, move || {
                let _guard = crate::fpenv::FpGuard::new();
                let _unit = cancel::unit(captured);
                let _ = sender.send(unit(start, end));
            });
            match spawned {
                Ok(handle) => pending.push(handle),
                Err(error) => {
                    failure = Some(EngineError::Device(format!(
                        "cannot start tile worker: {error}"
                    )));
                    break;
                }
            }
        }
        drop(sender);
        // Drain until every worker has gone.
        while let Ok(packet) = cancel::recv(&receiver) {
            match packet {
                Err(error) => {
                    failure.get_or_insert(error);
                }
                Ok(counts) => {
                    if failure.is_none() {
                        if let Err(error) = add(counts) {
                            failure = Some(error);
                        }
                    }
                }
            }
        }
        for worker in pending {
            if worker.join().is_err() {
                failure.get_or_insert_with(|| EngineError::Device("tile worker panicked".into()));
            }
        }
        match failure {
            Some(error) => Err(error),
            None => cancel::check(),
        }
    })
}

impl TileStream {
    /// Exact-grain stream: each tile may start `min(threads, tile work)` workers.
    #[allow(clippy::too_many_arguments)]
    pub fn new<S: Score, K: PairDistanceKernel<S>>(
        kernel: &K,
        a: Arc<SeqTable>,
        b: Arc<SeqTable>,
        work: WorkSource,
        tile_shape: (usize, usize),
        threads: usize,
        strategy: Strategy,
        check_bases: bool,
    ) -> Result<Self, EngineError> {
        // E0 for the whole call, whatever the caller's FP state (§7.3 item 2).
        let _guard = crate::fpenv::FpGuard::new();
        Self::with_workers(
            kernel,
            a,
            b,
            work,
            tile_shape,
            Workers::exact(threads),
            strategy,
            check_bases,
        )
    }

    /// Stream whose per-tile worker count follows `workers` (limit and grain);
    /// under `Grain::Auto` a small tile runs inline on the caller.
    #[allow(clippy::too_many_arguments)]
    pub fn with_workers<S: Score, K: PairDistanceKernel<S>>(
        kernel: &K,
        a: Arc<SeqTable>,
        b: Arc<SeqTable>,
        work: WorkSource,
        tile_shape: (usize, usize),
        workers: Workers,
        strategy: Strategy,
        check_bases: bool,
    ) -> Result<Self, EngineError> {
        // E0 for the whole call, whatever the caller's FP state (§7.3 item 2).
        let _guard = crate::fpenv::FpGuard::new();
        let threads = workers.limit;
        if tile_shape.0 == 0 || tile_shape.1 == 0 || threads == 0 {
            return Err(EngineError::Parameter(
                "tile dimensions and threads must be positive".into(),
            ));
        }
        let detail_bytes = dense_request_bytes(kernel.detail_fields(), kernel.detail_request())?;
        let shared = a.same_input(&b);
        let a = a.for_call();
        let b = if shared { Arc::clone(&a) } else { b.for_call() };
        kernel.validate_pairs(&a, &b, &work)?;
        let validation =
            crate::exec::validate_sequences(kernel, &a, &b, &work, check_bases, workers)?;
        let pair_cost = kernel.pair_cost(&a, &b, !kernel.detail_request().is_empty());
        let rect_cells = if kernel.detail_request().is_empty() {
            crate::kernel::rect_cells(kernel, &work, (&a, &b), false)
        } else {
            None
        };
        let shape = work.shape;
        let complete = shape.0 == 0 || shape.1 == 0;
        let index_order = if let PairKind::Index { i, j } = &work.kind {
            // Reserved fallibly: a refusal fails the creating call, which
            // returns no iterator (host-allocation-failures.md A12, D86 (6)).
            let mut order = Vec::<usize>::new();
            crate::detail::try_reserve_exact(&mut order, i.len(), "tile index order")?;
            order.extend(0..i.len());
            order.sort_unstable_by_key(|&id| {
                (i.get(id) / tile_shape.0, j.get(id) / tile_shape.1, id)
            });
            order
        } else {
            Vec::new()
        };
        Ok(Self {
            state: (!complete).then_some(State {
                a,
                b,
                work,
                index_order,
                index_next: 0,
            }),
            retired: None,
            tile_shape,
            row: 0,
            col: 0,
            shape,
            strategy,
            threads,
            workers,
            pair_cost,
            rect_cells,
            detail_bytes,
            detail_fields: kernel.detail_fields(),
            request: kernel.detail_request(),
            validation,
            counts: Counts::default(),
            evaluations: 0,
            replayed: 0,
            yielded: 0,
            complete,
            closed: false,
        })
    }

    /// The stream's position and counters before a `next()` step.
    pub fn mark(&self) -> TileMark {
        TileMark {
            row: self.row,
            col: self.col,
            full: self.counts.full,
            replayed: self.replayed,
            evaluations: self.evaluations,
            yielded: self.yielded,
            complete: self.complete,
        }
    }

    /// Fails a step that returned a tile the caller will not deliver (a signal
    /// handler's exception seen after the compute, cancellation-progress.md §2
    /// rule 5): the position and counters go back to `mark`, so `yielded`
    /// counts only delivered tiles and `complete` stays as it was, and the
    /// stream is closed.
    pub fn fail_step(&mut self, mark: TileMark) {
        self.row = mark.row;
        self.col = mark.col;
        self.counts.full = mark.full;
        self.replayed = mark.replayed;
        self.evaluations = mark.evaluations;
        self.yielded = mark.yielded;
        self.complete = mark.complete;
        self.close();
    }

    /// Closes the stream and drops its tables now.
    pub fn close(&mut self) {
        self.state = None;
        self.retired = None;
        self.closed = true;
    }

    /// Closes the stream but keeps its tables until `drop_retired`, for a
    /// caller that holds an FP guard (the tables' release must not run in it).
    pub fn retire(&mut self) {
        self.retire_state();
        self.closed = true;
    }

    /// Drops the tables of a stream that completed, failed or was retired.
    /// Call it outside every FP guard: a producer's release callback may run.
    pub fn drop_retired(&mut self) {
        self.retired = None;
    }

    fn retire_state(&mut self) {
        if let Some(state) = self.state.take() {
            self.retired = Some(state);
        }
    }

    pub fn next_tile<S: Score, K: PairDistanceKernel<S>>(
        &mut self,
        kernel: &K,
    ) -> Result<Option<TileOutput<S>>, EngineError> {
        // E0 for the whole call, whatever the caller's FP state (§7.3 item 2).
        let _guard = crate::fpenv::FpGuard::new();
        let (workers, pair_cost, rects) = (self.workers, self.pair_cost, self.rect_cells);
        let request = kernel.detail_request();
        if request != self.request {
            return Err(EngineError::Parameter(
                "next_tile needs the kernel configuration that created the stream".into(),
            ));
        }
        let scratch =
            (!request.is_empty()).then(|| DetailColumns::new(kernel.detail_fields(), request));
        self.next_with(&mut |tasks, cells, a, b, tile| {
            if let Some(rect) = rects {
                return rect_tile(kernel, tasks, (a, b), tile, (workers, pair_cost), rect);
            }
            let within = tasks.work.mirrors_within();
            let region = tasks.region;
            let compute = |start, end| {
                let mut scratch = scratch.clone();
                let rows = (a.rows(), b.rows());
                (start..end)
                    .filter_map(|k| tasks.at(k))
                    .map(move |(i, j, canonical)| {
                        // A mirror inside this tile shares the score; its
                        // details are computed as zip(seq_j, seq_i).
                        let shared_mirror = within && i != j && region.contains(j, i);
                        dense_score(
                            kernel,
                            rows,
                            (i, j),
                            shared_mirror,
                            scratch.as_mut(),
                            "Tiles",
                        )
                        .map(|(i, j, value, extra)| ((i, j, canonical, value), extra))
                    })
            };
            let n_tasks = tasks.len(cells);
            let limit = workers.count(n_tasks, pair_cost);
            crate::sched::stream_ranges(n_tasks, limit, &compute, &mut |(record, extra)| {
                tile.consume(record, extra.as_deref())
            })
        })
    }

    /// The same tile as `next_tile`, scored by a batch-distance backend: the
    /// tile rectangle for implicit shapes (hole cells are computed and
    /// discarded), or the tile's explicit pairs for zip and index. Numeric
    /// detail rows come from the port when the backend delivers them, else
    /// from the host `details` port for every requested cell; a shared mirror
    /// cell's row is computed as zip(seq_j, seq_i), as in `next_tile`.
    pub fn next_tile_batch<S: Score, K: PairDistanceKernel<S> + ?Sized>(
        &mut self,
        kernel: &K,
        distances: &mut dyn BatchDistance<S>,
    ) -> Result<Option<TileOutput<S>>, EngineError> {
        // E0 for the whole call, whatever the caller's FP state (§7.3 item 2).
        let _guard = crate::fpenv::FpGuard::new();
        let request = kernel.detail_request();
        if request != self.request {
            return Err(EngineError::Parameter(
                "next_tile_batch needs the kernel configuration that created the stream".into(),
            ));
        }
        let mode = BatchDetails::of(kernel, &*distances);
        let (fields, workers) = (self.detail_fields, self.workers);
        // Per-tile buffers, freed before returning: live iterator storage stays
        // inputs, metadata and the current tile (host-sinks.md).
        self.next_with(&mut |tasks, cells, a, b, tile| {
            let within = tasks.work.mirrors_within();
            let region = tasks.region;
            let mut values = Vec::new();
            let mut rows =
                (mode == BatchDetails::Port).then(|| DetailColumns::new(fields, request));
            // Requested cells: (i, j, canonical, value, position in `values`).
            // Every buffer of this closure is reserved fallibly: a refused host
            // reservation raises DeviceError and publishes no tile.
            let mut kept = Vec::new();
            if matches!(tasks.work.kind, PairKind::Zip | PairKind::Index { .. }) {
                let mut pairs: Vec<(usize, usize)> = Vec::new();
                try_reserve_exact(&mut pairs, tasks.len(cells), "tile pairs")?;
                pairs.extend(
                    (0..tasks.len(cells))
                        .filter_map(|k| tasks.at(k))
                        .map(|(i, j, _)| (i, j)),
                );
                distances.pairs(&pairs, &mut values, rows.as_mut())?;
                if values.len() != pairs.len() {
                    return Err(EngineError::Device(
                        "tile distance count does not match the tile".into(),
                    ));
                }
                try_reserve_exact(&mut kept, pairs.len(), "tile cells")?;
                kept.extend(
                    pairs
                        .iter()
                        .zip(&values)
                        .enumerate()
                        .map(|(k, (&(i, j), &value))| (i, j, true, value, k)),
                );
            } else {
                let class = |i: usize, j: usize| {
                    tasks
                        .at(region.cell(i, j))
                        .map(|(_, _, canonical)| canonical)
                };
                distances.rect(
                    (region.r0, region.r1),
                    (region.c0, region.c1),
                    &class,
                    &mut values,
                    rows.as_mut(),
                )?;
                if values.len() != cells {
                    return Err(EngineError::Device(
                        "tile distance count does not match the tile".into(),
                    ));
                }
                // At most every cell of the tile is requested.
                try_reserve_exact(&mut kept, values.len(), "tile cells")?;
                kept.extend(values.iter().enumerate().filter_map(|(k, &value)| {
                    tasks
                        .at(k)
                        .map(|(i, j, canonical)| (i, j, canonical, value, k))
                }));
            }
            if rows
                .as_ref()
                .is_some_and(|rows| rows.rows() != values.len())
            {
                return Err(EngineError::Device(
                    "tile detail rows do not match the tile".into(),
                ));
            }
            // A host boundary after the tile's device work (cancellation-progress.md §3.5).
            cancel::boundary(cancel::Site::Tile, kept.len())?;
            if mode == BatchDetails::None {
                for (i, j, canonical, value, _) in kept {
                    tile.consume((i, j, canonical, value), None)?;
                }
                return Ok(());
            }
            let delivered = rows.is_some();
            let mut dense: Vec<DenseCell<S>> = Vec::new();
            try_reserve_exact(&mut dense, kept.len(), "tile detail cells")?;
            dense.extend(kept.iter().map(|&(i, j, _, value, k)| DenseCell {
                i,
                j,
                value,
                mirror: within && i != j && region.contains(j, i),
                row: delivered.then_some(k),
            }));
            // Flat rows sized by the request's numeric field count; each
            // record's payload is built only when it is consumed.
            let extras = dense_extras(kernel, (a, b), &dense, rows.as_ref(), workers)?;
            for ((i, j, canonical, value, _), extra) in kept.into_iter().zip(extras.pairs(&dense)) {
                tile.consume((i, j, canonical, value), Some(&extra?))?;
            }
            Ok(())
        })
    }

    /// The SeqTables and work source of an unfinished stream (for backends
    /// that prepare per-call device state); None once complete or closed.
    pub fn inputs(&self) -> Option<(&Arc<SeqTable>, &Arc<SeqTable>, &WorkSource)> {
        self.state
            .as_ref()
            .map(|state| (&state.a, &state.b, &state.work))
    }

    fn next_with<S: Score>(
        &mut self,
        score: &mut Scorer<'_, S>,
    ) -> Result<Option<TileOutput<S>>, EngineError> {
        if self.state.is_none() {
            return Ok(None);
        }
        match self.compute_tile(score) {
            Ok(tile) => Ok(Some(tile)),
            Err(error) => {
                // Inside the next() guard: the tables drop at `drop_retired`.
                self.retire();
                Err(error)
            }
        }
    }

    fn compute_tile<S: Score>(
        &mut self,
        score: &mut Scorer<'_, S>,
    ) -> Result<TileOutput<S>, EngineError> {
        let region = Region {
            r0: self.row,
            r1: self.row + self.tile_shape.0.min(self.shape.0 - self.row),
            c0: self.col,
            c1: self.col + self.tile_shape.1.min(self.shape.1 - self.col),
        };
        let shape = region.shape();
        let cells = shape.0.checked_mul(shape.1).ok_or_else(|| {
            EngineError::Overflow("tile shape product exceeds platform limit".into())
        })?;
        let _bytes = cells.checked_mul(5 + self.detail_bytes).ok_or_else(|| {
            EngineError::Overflow("tile byte count exceeds platform limit".into())
        })?;
        if cells
            .checked_mul(std::mem::size_of::<S>())
            .is_none_or(|bytes| bytes > isize::MAX as usize)
        {
            return Err(EngineError::Overflow(
                "tile distance buffer exceeds addressable allocation size".into(),
            ));
        }
        // Both tile buffers are reserved through the §3 helper before either is
        // written (F4 round 3, the Matrix M1 pattern).
        let mut dist = Vec::new();
        let mut passed = Vec::new();
        try_reserve_exact(&mut dist, cells, "tile dist column")?;
        try_reserve_exact(&mut passed, cells, "tile passed column")?;
        let mut details = if self.request.is_empty() {
            None
        } else {
            Some(DetailColumns::masked_numeric(
                self.detail_fields,
                self.request,
                cells,
                "tile detail fields",
            )?)
        };
        dist.resize(cells, S::MASK);
        passed.resize(cells, 0u8);
        let state = self.state.as_mut().unwrap();
        let first = state.index_next;
        if let PairKind::Index { i, j } = &state.work.kind {
            while state.index_next < state.index_order.len() {
                let id = state.index_order[state.index_next];
                if !region.contains(i.get(id), j.get(id)) {
                    break;
                }
                state.index_next += 1;
            }
        }
        let tasks = Tasks {
            work: &state.work,
            region,
            indices: &state.index_order[first..state.index_next],
        };
        // The tile's progress stage: its task records (hole cells excluded), the
        // items of its `tile` checkpoints (cancellation-progress.md §4.2).
        let records = (0..tasks.len(cells))
            .filter(|&k| tasks.at(k).is_some())
            .count();
        let grid = self
            .shape
            .0
            .div_ceil(self.tile_shape.0)
            .saturating_mul(self.shape.1.div_ceil(self.tile_shape.1));
        cancel::stage(
            cancel::Stage::Tile {
                index: self.yielded + 1,
                of: grid,
            },
            records,
        );
        let (a, b): (&SeqTable, &SeqTable) = (&state.a, &state.b);
        let mut tile = Tile {
            region,
            within: state.work.mirrors_within(),
            dist: &mut dist,
            passed: &mut passed,
            details: details.as_mut(),
            full: 0,
            replayed: 0,
        };
        score(&tasks, cells, a, b, &mut tile)?;
        let (full, replayed) = (tile.full, tile.replayed);
        let next_full = increment(self.counts.full, full)?;
        let next_replayed = increment(self.replayed, replayed)?;
        let next_evaluations = increment(self.evaluations, increment(full, replayed)?)?;
        let next_yielded = increment(self.yielded, 1)?;
        self.counts.full = next_full;
        self.replayed = next_replayed;
        self.evaluations = next_evaluations;
        self.yielded = next_yielded;
        if region.c1 == self.shape.1 {
            self.col = 0;
            self.row = region.r1;
        } else {
            self.col = region.c1;
        }
        if self.row == self.shape.0 {
            self.complete = true;
            // Inside the next() guard: the tables drop at `drop_retired`.
            self.retire_state();
        }
        Ok(TileOutput {
            columns: Columns {
                dist,
                passed: Some(passed),
                details,
                ..Columns::default()
            },
            shape,
            rows: (region.r0, region.r1),
            cols: (region.c0, region.c1),
        })
    }
}
