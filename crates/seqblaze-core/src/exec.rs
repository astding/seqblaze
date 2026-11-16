//! Batch-only CPU execution, with canonical partition and output order.
use crate::{
    cancel::{self, Site, Stage, Ticker},
    detail::DetailColumns,
    kernel::{
        for_each_scored, for_each_tiled, rect_cells, rect_score, BatchDistance, BoundOutcome,
        Outcome, PairDistanceKernel, Plan, Score,
    },
    plan::BoundPrefilter,
    sched::Workers,
    seq::SeqTable,
    sink::{
        rowsum::{Dyadic, ExactSum},
        try_filled, Columns, Counts, RowSlice, RowState, RowStatsColumns, RunOutput, Sink,
    },
    validate::{self, Alphabet, EngineError, ValidationReport},
    work::WorkSource,
};

use std::sync::Mutex;

// Preserve the direct-engine import path used by sinks and existing callers.
pub use crate::plan::Strategy;
pub use crate::sched::Grain;

pub struct ExecOptions<S: Score = i32> {
    pub strategy: Strategy,
    pub threads: usize,
    /// Scheduling grain; never changes outputs or counts.
    pub grain: Grain,
    pub sink: Sink,
    /// The cutoff in the kernel's key type: a score passes iff `score <= cutoff`.
    pub cutoff: Option<S>,
    pub check_bases: bool,
    pub k: Option<usize>,
}

impl<S: Score> ExecOptions<S> {
    pub fn workers(&self) -> Workers {
        Workers {
            limit: self.threads,
            grain: self.grain,
        }
    }

    /// The same options with the cutoff in key type `T` (ParameterError if it does not
    /// fit, which a facade-converted cutoff never does).
    pub fn with_key<T: Score>(&self) -> Result<ExecOptions<T>, EngineError>
    where
        S: Into<i64>,
    {
        let cutoff = match self.cutoff {
            None => None,
            Some(value) => Some(T::from_i64(value.into()).ok_or_else(|| {
                EngineError::Parameter(format!(
                    "cutoff key {value:?} is outside this kernel's key range"
                ))
            })?),
        };
        Ok(ExecOptions {
            strategy: self.strategy,
            threads: self.threads,
            grain: self.grain,
            sink: self.sink,
            cutoff,
            check_bases: self.check_bases,
            k: self.k,
        })
    }
}

struct Partial<S: Score> {
    dist: Vec<S>,
    passed: Vec<u8>,
    hits: Vec<(usize, usize, S)>,
    /// Detail rows aligned with `dist` (Each) or `hits` (Hits).
    details: DetailColumns,
    counts: Counts,
    rows: Option<RowSlice<RowState<S>>>,
}

/// One exact row-term sum per row, shared by every worker. Exact additions are
/// order-free, so the sums do not depend on the worker split or the lock order, and a
/// row's sum exists once whatever the worker count (duplex-pf.md §9.4).
fn shared_sums(n_rows: usize) -> Result<Vec<Mutex<ExactSum>>, EngineError> {
    let mut sums = Vec::new();
    sums.try_reserve_exact(n_rows).map_err(|_| {
        EngineError::Device("unable to reserve rowstats row sums; no result was published".into())
    })?;
    sums.resize_with(n_rows, || Mutex::new(ExactSum::default()));
    Ok(sums)
}

/// Add a row term under the row's lock (a poisoned lock still holds an exact sum: the
/// panicking worker fails the call anyway).
fn add_term(sum: &Mutex<ExactSum>, term: Dyadic) -> Result<(), EngineError> {
    sum.lock().unwrap_or_else(|e| e.into_inner()).add(term)
}

/// Whole-table alphabet validation, the kernel's per-input length caps, then
/// its own whole-table preflight and its pair-shape preflight.
pub(crate) fn validate_sequences<S: Score, K: PairDistanceKernel<S>>(
    kernel: &K,
    a: &SeqTable,
    b: &SeqTable,
    work: &WorkSource,
    check_bases: bool,
    workers: Workers,
) -> Result<validate::ValidationReport, EngineError> {
    let report = validate::sequences_with_alphabet(
        a,
        b,
        check_bases,
        kernel.name(),
        kernel.alphabet(),
        kernel.length_caps(),
        &|input, index, length, limit| kernel.length_error(input, index, length, limit),
        workers,
    )?;
    kernel.sequence_preflight(a, b)?;
    kernel.pair_preflight(a, b, work)?;
    Ok(report)
}

/// Detail columns of kept entries `(i[k], j[k], dist[k])`, in entry order
/// (TopK entries, a final hit list); shared by every backend.
pub fn entry_details<S: Score, K: PairDistanceKernel<S> + ?Sized>(
    kernel: &K,
    a: &SeqTable,
    b: &SeqTable,
    entries: (&[i64], &[i64], &[S]),
    workers: Workers,
) -> Result<DetailColumns, EngineError> {
    let (ii, jj, dist) = entries;
    let entry = |k: usize| (ii[k] as usize, jj[k] as usize, dist[k]);
    let parts = detail_parts(kernel, a, b, dist.len(), &entry, workers)?;
    let mut result = DetailColumns::new(kernel.detail_fields(), kernel.detail_request());
    for mut part in parts {
        result.append(&mut part)?;
    }
    Ok(result)
}

/// The `details` stage over `n` kept entries, `entry(k)` giving entry k's
/// `(i, j, key)`: one detail part per canonical range, in entry order.
fn detail_parts<S: Score, K: PairDistanceKernel<S> + ?Sized>(
    kernel: &K,
    a: &SeqTable,
    b: &SeqTable,
    n: usize,
    entry: &(dyn Fn(usize) -> (usize, usize, S) + Sync),
    workers: Workers,
) -> Result<Vec<DetailColumns>, EngineError> {
    let empty = DetailColumns::new(kernel.detail_fields(), kernel.detail_request());
    let limit = workers.count(n, kernel.pair_cost(a, b, true));
    cancel::stage(Stage::Details, n);
    let (rows_a, rows_b) = (a.rows(), b.rows());
    crate::sched::map_ranges(n, limit, &|start, end| {
        let mut part = empty.clone();
        let mut ticker = Ticker::new(Site::Details);
        for k in start..end {
            let (i, j, key) = entry(k);
            kernel.details(rows_a.get(i), rows_b.get(j), key, &mut part)?;
            ticker.tick()?;
        }
        ticker.finish()?;
        Ok(part)
    })
}

pub fn run<S: Score, K: PairDistanceKernel<S>>(
    kernel: &K,
    work: &WorkSource,
    a: &SeqTable,
    b: &SeqTable,
    options: &ExecOptions<S>,
) -> Result<RunOutput<S>, EngineError> {
    // E0 for the whole call, whatever the caller's FP state (§7.3 item 2).
    let _guard = crate::fpenv::FpGuard::new();
    validated(kernel, work, a, b, options, |a, b, validation| {
        compute(kernel, work, a, b, options, validation)
    })
}

/// A consumer of the byte streams of a call's tables (the caller's input
/// identity, result-provenance.md), fed by the stream pass
/// ([`SeqTable::stream_certify`]) that `validated_observed` runs before the
/// validation steps. Side 0 is `a`, side 1 is `b`; a table shared by both
/// sides is side 0 only.
pub trait StreamObserver: Sync {
    /// Before side `side`'s pass, which would visit `leaves` leaves: whether
    /// the observer takes them. When it declines (its own reservation was
    /// refused), the side makes no pass and is validated as without an
    /// observer; the observer keeps the refusal and raises it no earlier than
    /// where it would have hashed the side without the pass, so that it never
    /// comes before a validation or computation error.
    fn begin(&self, side: usize, leaves: usize) -> bool;
    /// Leaf `index` of side `side`'s stream: its bytes are
    /// `buffer[STREAM_HEAD..]`, with room for a frame head before them.
    fn leaf(&self, side: usize, index: usize, buffer: &mut [u8]);
    /// After side `side`'s pass: whether it certified the side, so that the
    /// visited leaves are that side's stream; otherwise they are discarded.
    fn end(&self, side: usize, certified: bool);
}

/// Option checks, per-call table views and whole-input validation shared by
/// every backend. `then` receives the validated per-call views; nothing is
/// scored before it runs.
pub fn validated<S: Score, K: PairDistanceKernel<S>, R>(
    kernel: &K,
    work: &WorkSource,
    a: &SeqTable,
    b: &SeqTable,
    options: &ExecOptions<S>,
    then: impl FnOnce(&SeqTable, &SeqTable, ValidationReport) -> Result<R, EngineError>,
) -> Result<R, EngineError> {
    validated_observed(kernel, work, a, b, options, None, then)
}

/// [`validated`] that also hands the per-call tables' byte streams to
/// `observer`. For a kernel with the default ACGT alphabet, each distinct
/// per-call table whose stream is one contiguous span goes through
/// [`SeqTable::stream_certify`] once, before the validation steps: one read of
/// its bytes that hands each leaf to the observer and, when every byte is
/// ACGT, records a full-width proof and primes the alphabet scan the steps
/// then use (engine-layout.md §5). Every length-vector reservation and matrix
/// length check stays at its original `validate_pairs` preflight, in side
/// order, even after certification; only the nonallocating proof moves early.
/// A side the observer declines makes no pass ([`StreamObserver::begin`]);
/// its own allocation/overflow refusal is retained until post-validation
/// digest construction. A refused optional worker start skips the pass
/// ([`crate::sched::try_map_ranges`]) or trims the already allocated vector
/// inline. Thus these optimizations never promote an allocation or start
/// refusal ahead of the base validation error (class and message).
pub fn validated_observed<S: Score, K: PairDistanceKernel<S>, R>(
    kernel: &K,
    work: &WorkSource,
    a: &SeqTable,
    b: &SeqTable,
    options: &ExecOptions<S>,
    observer: Option<&dyn StreamObserver>,
    then: impl FnOnce(&SeqTable, &SeqTable, ValidationReport) -> Result<R, EngineError>,
) -> Result<R, EngineError> {
    // E0 for the whole call, whatever the caller's FP state (§7.3 item 2).
    let _guard = crate::fpenv::FpGuard::new();
    if options.threads == 0 {
        return Err(EngineError::Parameter("threads must be positive".into()));
    }
    if options.sink == Sink::TopK && options.k.is_none_or(|k| k == 0) {
        return Err(EngineError::Parameter("TopK requires positive k".into()));
    }
    if options.sink != Sink::TopK && options.k.is_some() {
        return Err(EngineError::Parameter(
            "k is only supported with TopK".into(),
        ));
    }
    if options.sink == Sink::Matrix && options.cutoff.is_some() {
        return Err(EngineError::Parameter(
            "Matrix does not support cutoff".into(),
        ));
    }
    if options.sink == Sink::Matrix {
        crate::detail::dense_request_bytes(kernel.detail_fields(), kernel.detail_request())?;
    }
    let shared = a.same_input(b);
    let a_view = a.fresh_view();
    let b_view = if shared { None } else { b.fresh_view() };
    let a = a_view.as_ref().unwrap_or(a);
    let b = if shared {
        a
    } else {
        b_view.as_ref().unwrap_or(b)
    };
    let workers = options.workers();
    let sides = if shared { &[a][..] } else { &[a, b][..] };
    if let Some(observer) = observer.filter(|_| matches!(kernel.alphabet(), Alphabet::Acgt)) {
        for (side, table) in sides.iter().enumerate() {
            if table
                .stream_leaves()
                .is_some_and(|leaves| observer.begin(side, leaves))
            {
                let certified = table
                    .stream_certify(workers, &|index, buffer| observer.leaf(side, index, buffer))?;
                observer.end(side, certified);
            }
        }
    }
    // Configure only: no allocation or trim runs ahead of validate_pairs.
    // Each side's original validate_lengths preflight owns that work.
    for table in sides {
        table.prepare_lengths(workers)?;
    }
    kernel.validate_pairs(a, b, work)?;
    let validation = validate_sequences(kernel, a, b, work, options.check_bases, workers)?;
    then(a, b, validation)
}

/// The call-level facts of the per-pair sink step of Each, Hits and RowStats, shared
/// by `compute` and `run_batch` (duplex-pf-cuda.md §8 item 5, QKP3).
struct Step<'c, S: Score> {
    sink: Sink,
    cutoff: Option<S>,
    /// Each keeps the key column.
    each_keys: bool,
    /// Each: a failing pair appends a masked detail row (details computed inline).
    mask_rows: bool,
    /// RowStats: the first-occurrence mask by work ID (index work sources).
    first: Option<&'c [bool]>,
    mirror: bool,
    /// RowStats of a `row_terms` kernel: the shared exact row sums.
    sums: Option<&'c [Mutex<ExactSum>]>,
}

impl<S: Score> Step<'_, S> {
    /// An empty part for work IDs `[start, end)`.
    fn part(
        &self,
        work: &WorkSource,
        start: usize,
        end: usize,
        details: &DetailColumns,
    ) -> Result<Partial<S>, EngineError> {
        let mut part = Partial {
            dist: Vec::new(),
            passed: Vec::new(),
            hits: Vec::new(),
            details: details.clone(),
            counts: Counts::default(),
            rows: None,
        };
        if self.sink == Sink::Each {
            // Fallible, key column first (host-allocation-failures.md A2).
            if self.each_keys {
                crate::detail::try_reserve(&mut part.dist, end - start, "the key column")?;
            }
            if self.cutoff.is_some() {
                crate::detail::try_reserve(&mut part.passed, end - start, "the passed column")?;
            }
        }
        if self.sink == Sink::RowStats {
            part.rows = Some(RowSlice::for_range(
                work,
                start,
                end,
                RowState::default(),
                "rowstats worker state",
            )?);
        }
        Ok(part)
    }

    /// The sink step of the items (i, j0 + k) of a row run of implicit work
    /// (no first-occurrence mask, so the work ID `apply` takes is never read),
    /// with rectangle scores `scores[k]` under the cutoff `prune`
    /// (`rect_score`), as `apply` item by item. Hits takes the run in one
    /// loop, and RowStats without row terms observes the run's passing values
    /// for row i in one merge and then, with mirrors, each off-diagonal one for
    /// its row j: the merge rule does not depend on order, so this equals
    /// observing them in turn.
    fn apply_run(
        &self,
        part: &mut Partial<S>,
        (i, j0): (usize, usize),
        scores: &[Option<S>],
        prune: Option<S>,
    ) -> Result<(), EngineError> {
        if self.sink == Sink::Hits {
            for (j, &score) in (j0..).zip(scores) {
                let Some(value) = rect_score(score, prune) else {
                    part.counts.pruned += 1;
                    continue;
                };
                part.counts.full += 1;
                if self.cutoff.is_none_or(|cutoff| value <= cutoff) {
                    // Amortized growth, reserved fallibly (host-allocation-failures.md A3).
                    crate::detail::try_reserve(&mut part.hits, 1, "the hit list")?;
                    part.hits.push((i, j, value));
                }
            }
            return Ok(());
        }
        if self.sink == Sink::RowStats && self.sums.is_none() {
            let rows = part.rows.as_mut().unwrap();
            let full = rows
                .get_mut(i)
                .observe_run(i, j0, scores, prune, self.cutoff)?;
            part.counts.full += full;
            part.counts.pruned += scores.len() - full;
            if self.mirror {
                // A requested within diagonal is never mirrored (A11, D87).
                for (j, &score) in (j0..).zip(scores) {
                    let passed = rect_score(rect_score(score, prune), self.cutoff);
                    if let Some(value) = passed.filter(|_| j != i) {
                        rows.get_mut(j).observe(j, i, value)?;
                    }
                }
            }
            return Ok(());
        }
        for (j, &score) in (j0..).zip(scores) {
            self.apply(part, 0, (i, j), rect_score(score, prune), false, None)?;
        }
        Ok(())
    }

    /// The sink step of work ID `k` = pair (i, j) with its exact score (None: pruned) and,
    /// for a `row_terms` RowStats call, its row term. `certified`: the score is a bound's
    /// certified exact key (counted as certified, not full). Calls arrive in increasing
    /// work-ID order within a part.
    #[inline]
    fn apply(
        &self,
        part: &mut Partial<S>,
        k: usize,
        (i, j): (usize, usize),
        score: Option<S>,
        certified: bool,
        term: Option<Dyadic>,
    ) -> Result<(), EngineError> {
        match (score.is_some(), certified) {
            (false, _) => part.counts.pruned += 1,
            (true, true) => part.counts.certified += 1,
            (true, false) => part.counts.full += 1,
        }
        let passed = score.is_some_and(|score| self.cutoff.is_none_or(|cutoff| score <= cutoff));
        match self.sink {
            Sink::Each => {
                if self.each_keys {
                    part.dist
                        .push(if passed { score.unwrap() } else { S::MASK });
                }
                if self.cutoff.is_some() {
                    part.passed.push(u8::from(passed));
                }
                if self.mask_rows && !passed {
                    part.details.push_masked()?;
                }
            }
            Sink::Hits if passed => {
                // Amortized growth, reserved fallibly (host-allocation-failures.md A3).
                crate::detail::try_reserve(&mut part.hits, 1, "the hit list")?;
                part.hits.push((i, j, score.unwrap()));
            }
            Sink::Hits => (),
            Sink::RowStats if passed => {
                // Count work IDs for execution, but distinct partners for row statistics.
                let first = self.first.is_none_or(|mask| mask[k]);
                // A requested within diagonal is an ordinary partner of its own
                // row, contributed once and never mirrored (A11, D87).
                if first {
                    let rows = part.rows.as_mut().unwrap();
                    rows.get_mut(i).observe(i, j, score.unwrap())?;
                    if self.mirror && i != j {
                        rows.get_mut(j).observe(j, i, score.unwrap())?;
                    }
                    if let (Some(sums), Some(term)) = (self.sums, term) {
                        add_term(&sums[i], term)?;
                        if self.mirror && i != j {
                            add_term(&sums[j], term)?;
                        }
                    }
                }
            }
            Sink::RowStats => (),
            Sink::TopK => unreachable!("TopK uses the two-pass dispatcher"),
            Sink::Matrix => unreachable!("Matrix uses the bounded-scratch dispatcher"),
        }
        Ok(())
    }
}

/// Move all of a part's column after the merged column. While the merged column is
/// empty and would have to allocate, it takes the part's buffer itself (the single
/// part of `run_batch`, the first part of `compute`): no copy and no second buffer of
/// the column. Otherwise a fallible amortized reservation, so a refused one is a
/// DeviceError that publishes nothing (D86) instead of an abort, then an append.
pub(crate) fn merge_column<T>(
    merged: &mut Vec<T>,
    part: &mut Vec<T>,
    what: &str,
) -> Result<(), EngineError> {
    if merged.is_empty() && merged.capacity() < part.len() {
        std::mem::swap(merged, part);
        return Ok(());
    }
    crate::detail::try_reserve(merged, part.len(), what)?;
    merged.append(part);
    Ok(())
}

/// Merge the parts (in canonical order) into the Each, Hits or RowStats output. With
/// `inline_details` the parts carry the detail rows (`compute`); otherwise the caller
/// adds them afterwards (`run_batch`).
#[allow(clippy::too_many_arguments)]
fn assemble<S: Score, K: PairDistanceKernel<S> + ?Sized>(
    kernel: &K,
    work: &WorkSource,
    options: &ExecOptions<S>,
    validation: ValidationReport,
    parts: Vec<Partial<S>>,
    shared: Option<Vec<Mutex<ExactSum>>>,
    mut details: DetailColumns,
    inline_details: bool,
) -> Result<RunOutput<S>, EngineError> {
    let mut columns = Columns::default();
    let mut counts = Counts::default();
    let mut hits = Vec::new();
    let mut passed = Vec::new();
    let mut rows = if options.sink == Sink::RowStats {
        Some(try_filled(
            work.shape.0,
            RowState::default(),
            "rowstats rows",
        )?)
    } else {
        None
    };
    let sums = match shared {
        Some(shared) => {
            let mut sums = Vec::new();
            sums.try_reserve_exact(shared.len()).map_err(|_| {
                EngineError::Device(
                    "unable to reserve rowstats row sums; no result was published".into(),
                )
            })?;
            sums.extend(
                shared
                    .into_iter()
                    .map(|sum| sum.into_inner().unwrap_or_else(|e| e.into_inner())),
            );
            Some(sums)
        }
        None => None,
    };
    for mut part in parts {
        merge_column(&mut columns.dist, &mut part.dist, "the key column")?;
        merge_column(&mut passed, &mut part.passed, "the passed column")?;
        merge_column(&mut hits, &mut part.hits, "the hit list")?;
        details.append(&mut part.details)?;
        if let Some(part_rows) = part.rows.take() {
            // Parts arrive in canonical order; adjacent intervals share at most
            // their boundary row, and the merge rule is order-independent.
            let target = &mut rows.as_mut().unwrap()[part_rows.first..];
            for (offset, (left, right)) in target.iter_mut().zip(part_rows.rows).enumerate() {
                left.merge(part_rows.first + offset, right)?;
            }
        }
        counts.full += part.counts.full;
        counts.pruned += part.counts.pruned;
        counts.certified += part.counts.certified;
    }
    // Detail fields ride only on the sinks that report per-pair entries.
    let detailed = inline_details
        && matches!(options.sink, Sink::Each | Sink::Hits)
        && !kernel.detail_request().is_empty();
    if options.sink == Sink::Hits {
        if detailed {
            // Detail rows follow their hits through the (i, j) sort and dedup.
            (hits, details) = crate::sink::kept::sort_hits(hits, details)?;
        }
        // Already sorted and unique when detailed, so the rows stay aligned.
        columns = Columns::from_hits(hits)?;
    } else if options.sink == Sink::RowStats {
        let value = |sum: &ExactSum| kernel.row_value(sum);
        columns.rowstats = Some(RowStatsColumns::with_row_values(
            rows.unwrap(),
            options.cutoff.is_some(),
            sums.map(|sums| (sums, &value as &dyn Fn(&ExactSum) -> f64)),
        )?);
    } else if options.cutoff.is_some() {
        columns.passed = Some(passed);
    }
    if detailed {
        columns.details = Some(details);
    }
    Ok(RunOutput {
        columns,
        shape: work.shape,
        counts,
        threads: options.threads,
        n_illegal_seqs: validation.n_illegal_seqs,
        validation_scans: validation.scans,
        topk_replay: None,
    })
}

/// A pass that uses the outcome bound, with the probe's outcomes by work ID (increasing),
/// which the pass takes instead of evaluating those items again (duplex-mfe-bound.md §6.6
/// *Reuse*). An outcome is a pure function of the item and the pass threshold, so reuse
/// changes no byte and no count.
pub(crate) struct PassBound<S> {
    probed: Vec<(usize, BoundOutcome<S>)>,
}

impl<S: Score> PassBound<S> {
    /// Work ID `k`'s outcome: the probe's, or `evaluate()` for an item the probe did not
    /// evaluate. At most 64 probed items, so the search costs a few comparisons.
    #[inline]
    pub(crate) fn outcome(
        &self,
        k: usize,
        evaluate: impl FnOnce() -> Result<BoundOutcome<S>, EngineError>,
    ) -> Result<BoundOutcome<S>, EngineError> {
        match self.probed.binary_search_by_key(&k, |&(id, _)| id) {
            Ok(at) => Ok(self.probed[at].1),
            Err(_) => evaluate(),
        }
    }
}

/// The §6.6 weak-bound decision of one pass (duplex-mfe-bound.md, revision 2) over the
/// pass's work items: `threshold(k, i, j)` is `None` when the pass skips work ID `k`
/// (TopK replay), else its threshold θ (`Some(None)`: certify only). The probe items'
/// weights come from the kernel's `bound_cost` of their strand lengths; their bounds are
/// evaluated only when the length gate passes, in parallel on the call's workers (the
/// pass's canonical ranges over the probe set; each outcome is per item and R an exact
/// integer sum, so the decision is the same for every thread count). Returns the pass's
/// bound with the probe's outcomes when the bound is used, else `None`.
pub(crate) fn use_bound_on_pass<S: Score, K: PairDistanceKernel<S> + ?Sized>(
    kernel: &K,
    work: &WorkSource,
    (a, b): (&SeqTable, &SeqTable),
    workers: Workers,
    threshold: impl Fn(usize, usize, usize) -> Option<Option<S>> + Sync,
) -> Result<Option<PassBound<S>>, EngineError> {
    let item = |k: usize| work.range(k, k + 1).next();
    let mut probed = Vec::new();
    let decision = crate::plan::decide_outcome_bound(
        work.n_pairs,
        |k| {
            let Some((i, j)) = item(k) else {
                return Ok(None);
            };
            Ok::<_, EngineError>(
                // The CPU's table (§6.6.1) with q̂ = 0: no probe-cost gate (§6.6.2).
                threshold(k, i, j).map(|_| {
                    let (e, c) = kernel.bound_cost(a.get(i).len(), b.get(j).len());
                    (e, c, 0)
                }),
            )
        },
        |ids, resolved| {
            // Every probe ID is a work ID the pass evaluates (the cost step dropped the
            // skipped ones); an ID without an item or a threshold is not evaluated, does
            // not resolve and is not reused.
            let mut outcomes: Vec<Option<BoundOutcome<S>>> = vec![None; ids.len()];
            let limit = workers.count(ids.len(), kernel.pair_cost(a, b, false));
            crate::sched::fill_chunks(&mut outcomes, limit, &|start, chunk| {
                for (slot, &k) in chunk.iter_mut().zip(&ids[start..]) {
                    if let Some((i, j)) = item(k) {
                        if let Some(theta) = threshold(k, i, j) {
                            *slot = Some(kernel.bound_outcome(a.get(i), b.get(j), theta)?);
                        }
                    }
                }
                Ok(())
            })?;
            for (flag, outcome) in resolved.iter_mut().zip(&outcomes) {
                *flag = outcome.is_some_and(|outcome| outcome != BoundOutcome::Open);
            }
            probed = ids
                .iter()
                .zip(outcomes)
                .filter_map(|(&k, outcome)| outcome.map(|outcome| (k, outcome)))
                .collect();
            Ok(())
        },
    )?;
    // A forced "use" (testing fault) probes nothing, so nothing is reused.
    Ok(decision.used.then_some(PassBound { probed }))
}

/// CPU compute stage over tables that `validated` has already checked.
pub fn compute<S: Score, K: PairDistanceKernel<S>>(
    kernel: &K,
    work: &WorkSource,
    a: &SeqTable,
    b: &SeqTable,
    options: &ExecOptions<S>,
    validation: ValidationReport,
) -> Result<RunOutput<S>, EngineError> {
    // E0 for the whole call, whatever the caller's FP state (§7.3 item 2).
    let _guard = crate::fpenv::FpGuard::new();
    if options.sink == Sink::TopK {
        return crate::sink::topk::run(kernel, work, a, b, options, validation);
    }
    if options.sink == Sink::Matrix {
        return crate::sink::matrix::run(kernel, work, a, b, options, validation);
    }
    let workers = options.workers();
    let bounded = options.strategy == Strategy::Bounded && options.cutoff.is_some();
    // RowStats of a `row_terms` kernel sums every partner's term, so no outcome bound.
    let terms = options.sink == Sink::RowStats && kernel.row_terms();
    // An outcome bound (duplex-mfe-bound.md §6): every item is pruned against the pass
    // threshold (the cutoff), certified with its exact key, or computed; once per pass
    // the weak-bound rule decides whether to use it at all (§6.6), and the pass reuses
    // the probe's outcomes.
    let pass_bound = if options.strategy == Strategy::Bounded && kernel.outcome_bound() && !terms {
        use_bound_on_pass(kernel, work, (a, b), workers, |_, _, _| {
            Some(options.cutoff)
        })?
    } else {
        None
    };
    let outcomes = pass_bound.is_some();
    // Per-pair bounds only where row reuse amortizes their preparation; the
    // early-exit score rejects exactly the same pairs either way.
    let prefilter = bounded && !outcomes && BoundPrefilter::for_work(&work.kind).per_pair();
    if prefilter {
        kernel.prepare_bound(a, b, workers)?;
    }
    let first_occurrences = if options.sink == Sink::RowStats {
        work.first_occurrences()?
    } else {
        None
    };
    // Detail fields ride only on the sinks that report per-pair entries: inline for
    // Each (every row), and inline for Hits when the kernel's detailed evaluation
    // shares its score pass's work (`fused_details`). Other Hits detail rows come
    // from a `details` stage over the parts' hits before they are merged (the
    // `details` port gives the same row as the inline evaluation, so the bytes are
    // the same; cancellation-progress.md §3.2 `details`, §13). Either way the
    // detail rows follow their hits through the merge, sort and dedup, so the
    // buffers live and are freed in the same order.
    let requested = !kernel.detail_request().is_empty();
    let detailed = requested
        && (options.sink == Sink::Each || (options.sink == Sink::Hits && kernel.fused_details()));
    let hit_details = options.sink == Sink::Hits && requested && !detailed;
    let no_details = DetailColumns::new(kernel.detail_fields(), kernel.detail_request());
    // RowStats of a `row_terms` kernel also sums each distinct passing partner's term.
    let shared = if terms {
        Some(shared_sums(work.shape.0)?)
    } else {
        None
    };
    let step = Step {
        sink: options.sink,
        cutoff: options.cutoff,
        // Each keeps the key column unless the kernel publishes every Each column from
        // its detail rows (`each_keys`; tm on Pairs): no unread 8-byte-per-pair column.
        each_keys: options.sink == Sink::Each && (kernel.each_keys() || !detailed),
        mask_rows: detailed,
        first: first_occurrences.as_deref(),
        mirror: work.mirrors_within(),
        sums: shared.as_deref(),
    };
    // Plain scores go through `for_each_scored`, which hands a kernel that scores
    // several pairs at once (`batch_width`) a block of work items at a time, or,
    // for Hits and RowStats over implicit work, through `for_each_tiled`, which
    // hands a rectangle kernel (`rect_cells`) rectangles of cells. A pass that uses
    // the outcome bound takes the per-item loop, which applies each item's outcome.
    let many = !terms && !detailed && !outcomes;
    let rects =
        rect_cells(kernel, work, (a, b), prefilter).filter(|_| many && options.sink != Sink::Each);
    let (rows_a, rows_b) = (a.rows(), b.rows());
    let compute = |start: usize, end: usize| -> Result<Partial<S>, EngineError> {
        let mut part = step.part(work, start, end, &no_details)?;
        let mut ticker = Ticker::new(Site::Pairs);
        if let Some(cells) = rects {
            // Items arrive a row run at a time, rectangle by rectangle; Hits
            // sorts its entries and RowStats' merge rule is order-independent.
            debug_assert!(step.first.is_none());
            let prune = if bounded { options.cutoff } else { None };
            for_each_tiled(
                kernel,
                (a, b),
                work,
                (start, end),
                cells,
                |_, _| prune,
                |i, j0, scores| {
                    step.apply_run(&mut part, (i, j0), scores, prune)?;
                    ticker.add(scores.len())
                },
            )?;
            ticker.finish()?;
            return Ok(part);
        }
        if many {
            let prune = if bounded { options.cutoff } else { None };
            for_each_scored(
                kernel,
                (a, b),
                work,
                (start, end),
                |_, (i, j)| {
                    if prefilter && kernel.lower_bound(a, b, i, j) > options.cutoff.unwrap() {
                        Plan::Reject
                    } else {
                        Plan::Score(prune)
                    }
                },
                |offset, pair, outcome| {
                    let Outcome::Scored(score) = outcome else {
                        unreachable!("no item is skipped")
                    };
                    step.apply(&mut part, start + offset, pair, score, false, None)?;
                    ticker.tick()
                },
            )?;
            ticker.finish()?;
            return Ok(part);
        }
        for (offset, (i, j)) in work.range(start, end).enumerate() {
            let prune = if bounded { options.cutoff } else { None };
            let mut term = None;
            if let Some(bound) = pass_bound.as_ref() {
                let (x, y) = (a.get(i), b.get(j));
                match bound.outcome(start + offset, || {
                    kernel.bound_outcome(x, y, options.cutoff)
                })? {
                    BoundOutcome::Pruned => {
                        step.apply(&mut part, start + offset, (i, j), None, false, None)?;
                        ticker.tick()?;
                        continue;
                    }
                    BoundOutcome::Certified(score) => {
                        // Values only (§6.4): a kept certified pair's detail row comes from
                        // the details port with the certified key, as inline evaluation
                        // appends it (exactly when the pair passes).
                        if detailed && options.cutoff.is_none_or(|limit| score <= limit) {
                            kernel.details(x, y, score, &mut part.details)?;
                        }
                        step.apply(&mut part, start + offset, (i, j), Some(score), true, None)?;
                        ticker.tick()?;
                        continue;
                    }
                    BoundOutcome::Open => {}
                }
            }
            let score = if prefilter && kernel.lower_bound(a, b, i, j) > options.cutoff.unwrap() {
                None
            } else if terms {
                // An exact score and its row term; no pruning (the term is all-partner).
                let (score, value) = kernel.evaluate_term(rows_a.get(i), rows_b.get(j))?;
                term = Some(value);
                Some(score)
            } else if detailed {
                // Appends one detail row exactly when the pair passes.
                kernel.evaluate_detailed(
                    rows_a.get(i),
                    rows_b.get(j),
                    prune,
                    options.cutoff,
                    &mut part.details,
                )?
            } else {
                kernel.evaluate(rows_a.get(i), rows_b.get(j), prune)?
            };
            step.apply(&mut part, start + offset, (i, j), score, false, term)?;
            ticker.tick()?;
        }
        ticker.finish()?;
        Ok(part)
    };
    // Scheduling only: the pair stage's worker count is the one of the inline
    // evaluation (for every Hits call), so the partition is the same either way.
    let limit = workers.count(
        work.n_pairs,
        kernel.pair_cost(a, b, detailed || hit_details),
    );
    cancel::stage(Stage::Pairs, work.n_pairs);
    let parts = crate::sched::map_ranges(work.n_pairs, limit, &compute)?;
    let mut details = no_details;
    if hit_details {
        // Entry k of the concatenated part hit lists, in canonical order.
        let mut first = Vec::with_capacity(parts.len());
        let mut n_hits = 0usize;
        for part in &parts {
            first.push(n_hits);
            n_hits += part.hits.len();
        }
        let entry = |k: usize| {
            let p = first.partition_point(|&start| start <= k) - 1;
            parts[p].hits[k - first[p]]
        };
        for mut part in detail_parts(kernel, a, b, n_hits, &entry, workers)? {
            details.append(&mut part)?;
        }
    }
    assemble(
        kernel,
        work,
        options,
        validation,
        parts,
        shared,
        details,
        detailed || hit_details,
    )
}

/// Full-strategy compute over a batch-distance backend (duplex-pf-cuda.md §7.2, §8
/// item 5, QKP3): every sink, with Each, Hits and RowStats through the per-pair sink
/// step `compute` uses, applied on the calling thread to the backend's chunks in
/// work-ID order, and TopK and Matrix through their batch drivers. RowStats of a
/// `row_terms` kernel reads the backend's term channel. Detail fields of kept pairs
/// come from the kernel's `details` port on host workers (Each: passing rows, failing
/// rows masked; Hits: the final hit list), as for a backend that does not deliver them.
pub fn run_batch<S: Score, K: PairDistanceKernel<S> + ?Sized>(
    kernel: &K,
    (a, b): (&SeqTable, &SeqTable),
    port: &mut dyn BatchDistance<S>,
    work: &WorkSource,
    options: &ExecOptions<S>,
    validation: ValidationReport,
) -> Result<RunOutput<S>, EngineError> {
    // E0 for the whole call, whatever the caller's FP state (§7.3 item 2).
    let _guard = crate::fpenv::FpGuard::new();
    match options.sink {
        Sink::TopK => {
            return crate::sink::topk::run_batch(kernel, (a, b), port, work, options, validation)
        }
        Sink::Matrix => {
            return crate::sink::matrix::run_batch(kernel, (a, b), port, work, options, validation)
        }
        _ => {}
    }
    let first_occurrences = if options.sink == Sink::RowStats {
        work.first_occurrences()?
    } else {
        None
    };
    let detailed =
        matches!(options.sink, Sink::Each | Sink::Hits) && !kernel.detail_request().is_empty();
    let no_details = DetailColumns::new(kernel.detail_fields(), kernel.detail_request());
    let terms = options.sink == Sink::RowStats && kernel.row_terms();
    let shared = if terms {
        Some(shared_sums(work.shape.0)?)
    } else {
        None
    };
    let step = Step {
        sink: options.sink,
        cutoff: options.cutoff,
        // The keys of passing rows feed the host detail pass, so Each keeps them here.
        each_keys: options.sink == Sink::Each,
        mask_rows: false,
        first: first_occurrences.as_deref(),
        mirror: work.mirrors_within(),
        sums: shared.as_deref(),
    };
    let mut part = step.part(work, 0, work.n_pairs, &no_details)?;
    cancel::stage(Stage::Pairs, work.n_pairs);
    {
        let mut chunk =
            |first: usize, values: &[S], terms: Option<&[Dyadic]>| -> Result<(), EngineError> {
                if terms.is_some_and(|terms| terms.len() != values.len()) {
                    return Err(EngineError::Device(
                        "a batch-distance chunk holds a different number of keys and row terms; \
                     no result was published"
                            .into(),
                    ));
                }
                let pairs = work.range(first, first + values.len());
                for (offset, ((i, j), &value)) in pairs.zip(values).enumerate() {
                    let term = terms.map(|terms| terms[offset]);
                    step.apply(&mut part, first + offset, (i, j), Some(value), false, term)?;
                }
                // A host boundary after each delivered chunk (cancellation-progress.md §3.5).
                cancel::boundary(Site::Pairs, values.len())
            };
        if terms {
            port.range_terms(work, 0, work.n_pairs, &mut |first, values, terms| {
                chunk(first, values, Some(terms))
            })?;
        } else {
            port.range(work, 0, work.n_pairs, false, &mut |first, values, _| {
                chunk(first, values, None)
            })?;
        }
    }
    if part.counts.full != work.n_pairs {
        return Err(EngineError::Device(format!(
            "a batch-distance backend delivered {} of {} pairs; no result was published",
            part.counts.full, work.n_pairs
        )));
    }
    let mut output = assemble(
        kernel,
        work,
        options,
        validation,
        vec![part],
        shared,
        no_details,
        false,
    )?;
    if detailed {
        let columns = &mut output.columns;
        let workers = options.workers();
        columns.details = Some(match options.sink {
            Sink::Each => crate::sink::kept::each_details(
                kernel,
                work,
                (a, b),
                &columns.dist,
                columns.passed.as_deref(),
                workers,
            )?,
            _ => entry_details(
                kernel,
                a,
                b,
                (
                    columns.i.as_deref().unwrap_or(&[]),
                    columns.j.as_deref().unwrap_or(&[]),
                    &columns.dist,
                ),
                workers,
            )?,
        });
        if options.sink == Sink::Each && !kernel.each_keys() {
            // As `compute`: no key column when every Each column comes from the rows.
            columns.dist = Vec::new();
        }
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    //! Row-interval worker state and grain: outputs and counts must not depend on
    //! the worker split. Exact grain forces real multi-worker merges on tiny inputs.
    use super::*;
    use crate::{
        batch::SeqBatch,
        detail::{DetailField, DetailRequest, FieldKind},
        kernel::CpuBatch,
        plan::BoundAvailability,
        seq::ByteView,
        work::{Orientation, PairKind},
    };

    /// Byte mismatches with an A-count lower bound; a local observer only.
    struct Mismatch;
    impl PairDistanceKernel for Mismatch {
        fn bound_availability(&self) -> BoundAvailability {
            BoundAvailability::Available
        }
        fn prepare_bound(&self, _: &SeqTable, _: &SeqTable, _: Workers) -> Result<(), EngineError> {
            Ok(())
        }
        fn lower_bound(&self, a: &SeqTable, b: &SeqTable, i: usize, j: usize) -> i32 {
            let count = |view: ByteView<'_>| view.iter().filter(|&byte| byte == b'A').count();
            count(a.get(i)).abs_diff(count(b.get(j))) as i32
        }
        fn evaluate(
            &self,
            a: ByteView<'_>,
            b: ByteView<'_>,
            cutoff: Option<i32>,
        ) -> Result<Option<i32>, EngineError> {
            let d = a.iter().zip(b.iter()).filter(|(x, y)| x != y).count() as i32;
            Ok(cutoff.is_none_or(|cutoff| d <= cutoff).then_some(d))
        }
    }

    /// Mismatch with an exact row term per pair (`row_terms`): M · 2^E from the bytes,
    /// with exponents spread over `spread` bits. With a narrow spread every term moves
    /// RN₅₃ of the row sum, so a dropped or doubled term changes the row value; a wide
    /// spread makes partial sums of different widths meet in the merges.
    struct Terms {
        spread: i64,
    }
    impl PairDistanceKernel for Terms {
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
            a: ByteView<'_>,
            b: ByteView<'_>,
            cutoff: Option<i32>,
        ) -> Result<Option<i32>, EngineError> {
            Mismatch.evaluate(a, b, cutoff)
        }
        fn row_terms(&self) -> bool {
            true
        }
        fn evaluate_term(
            &self,
            a: ByteView<'_>,
            b: ByteView<'_>,
        ) -> Result<(i32, Dyadic), EngineError> {
            let d = Mismatch.evaluate(a, b, None)?.unwrap();
            let h = a
                .iter()
                .chain(b.iter())
                .fold(0xCBF2_9CE4_8422_2325u64, |h, byte| {
                    (h ^ u64::from(byte)).wrapping_mul(0x0100_0000_01B3)
                });
            let m = (1 << 52) | (h & ((1 << 52) - 1));
            Ok((
                d,
                Dyadic {
                    m,
                    e: (h >> 52) as i64 % self.spread,
                },
            ))
        }
        fn row_value(&self, sum: &ExactSum) -> f64 {
            let (m, e) = sum.round53().unwrap();
            f64::from_bits(m ^ (e as u64).rotate_left(53))
        }
    }

    fn table(n: usize, seed: usize) -> SeqTable {
        let rows: Vec<Vec<u8>> = (0..n)
            .map(|r| {
                (0..4)
                    .map(|t| b"ACGT"[(r * 7 + t * 3 + seed + r * t) % 4])
                    .collect()
            })
            .collect();
        SeqTable::new(SeqBatch::from_seqs(&rows))
    }

    fn kinds() -> Vec<(PairKind, usize, usize)> {
        let mut kinds = vec![
            (PairKind::Zip, 9, 9),
            (PairKind::Cross, 7, 5),
            (PairKind::OneToMany, 1, 11),
        ];
        for self_pairs in [false, true] {
            kinds.push((PairKind::Within { self_pairs }, 8, 8));
            for orientation in [Orientation::AsGiven, Orientation::Both] {
                let kind = PairKind::WithinOriented {
                    self_pairs,
                    orientation,
                };
                kinds.push((kind, 8, 8));
            }
        }
        kinds
    }

    fn flat(output: &RunOutput) -> Vec<Vec<i64>> {
        let c = &output.columns;
        let mut result = vec![c.dist.iter().map(|&v| v.into()).collect()];
        if let Some(stats) = &c.rowstats {
            result.push(stats.best_dist.iter().map(|&v| v.into()).collect());
            result.push(stats.argbest.clone());
            result.push(stats.n_best.iter().map(|&v| v.into()).collect());
            result.push(stats.count.clone().unwrap_or_default());
        }
        if let Some(topk) = &c.topk {
            result.push(topk.i.clone());
            result.push(topk.j.clone());
            result.push(topk.dist.iter().map(|&v| v.into()).collect());
            result.push(topk.rank.iter().map(|&v| v.into()).collect());
            result.push(topk.offsets.clone());
            result.push(topk.tie_at_k.iter().map(|&v| v.into()).collect());
        }
        let counts = &output.counts;
        result.push(vec![
            counts.certified as i64,
            counts.pruned as i64,
            counts.full as i64,
        ]);
        result
    }

    #[test]
    fn row_reducers_match_one_worker_for_every_split_strategy_and_grain() {
        for kind_index in 0..kinds().len() {
            let (kind, n_a, n_b) = kinds().swap_remove(kind_index);
            let within = matches!(
                kind,
                PairKind::Within { .. } | PairKind::WithinOriented { .. }
            );
            let (a, b) = (table(n_a, 0), table(n_b, 1));
            let b = if within { &a } else { &b };
            for (sink, k) in [
                (Sink::RowStats, None),
                (Sink::TopK, Some(1)),
                (Sink::TopK, Some(2)),
            ] {
                for cutoff in [None, Some(1), Some(2)] {
                    let mut columns = None;
                    for strategy in [Strategy::Full, Strategy::Bounded] {
                        let mut counts = None;
                        for threads in [1, 2, 3, 5, 16] {
                            for grain in [Grain::Exact, Grain::Auto] {
                                let work =
                                    WorkSource::new(kinds().swap_remove(kind_index).0, n_a, n_b)
                                        .unwrap();
                                let options = ExecOptions {
                                    strategy,
                                    threads,
                                    grain,
                                    sink,
                                    cutoff,
                                    check_bases: true,
                                    k,
                                };
                                let mut got =
                                    flat(&run(&Mismatch, &work, &a, b, &options).unwrap());
                                let got_counts = got.pop();
                                let label = format!("{kind_index} {sink:?} {cutoff:?} {strategy:?} t={threads} {grain:?}");
                                assert_eq!(
                                    &got,
                                    columns.get_or_insert_with(|| got.clone()),
                                    "{label}"
                                );
                                assert_eq!(
                                    &got_counts,
                                    counts.get_or_insert_with(|| got_counts.clone()),
                                    "{label}"
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    /// Mismatch scored through rectangles of `cells` cells (`rect_cells`),
    /// counting its `evaluate_rect` calls; its cutoff test is exact.
    struct Rect {
        cells: usize,
        rects: std::sync::atomic::AtomicUsize,
    }
    impl PairDistanceKernel for Rect {
        fn bound_availability(&self) -> BoundAvailability {
            BoundAvailability::Available
        }
        fn prepare_bound(&self, _: &SeqTable, _: &SeqTable, _: Workers) -> Result<(), EngineError> {
            Ok(())
        }
        fn lower_bound(&self, a: &SeqTable, b: &SeqTable, i: usize, j: usize) -> i32 {
            Mismatch.lower_bound(a, b, i, j)
        }
        fn evaluate(
            &self,
            a: ByteView<'_>,
            b: ByteView<'_>,
            cutoff: Option<i32>,
        ) -> Result<Option<i32>, EngineError> {
            Mismatch.evaluate(a, b, cutoff)
        }
        fn batch_width(&self) -> usize {
            3
        }
        fn rect_cells(&self, _: &SeqTable, _: &SeqTable, _: bool) -> usize {
            self.cells
        }
        fn evaluate_rect(
            &self,
            rows: &[ByteView<'_>],
            cols: &[ByteView<'_>],
            cutoff: Option<i32>,
            out: &mut Vec<Option<i32>>,
        ) -> Result<(), EngineError> {
            self.rects
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            for &x in rows {
                for &y in cols {
                    out.push(self.evaluate(x, y, cutoff)?);
                }
            }
            Ok(())
        }
    }

    /// Every column, count and replay count of an output.
    fn everything(output: &RunOutput) -> Vec<Vec<i64>> {
        let mut result = flat(output);
        let c = &output.columns;
        result.push(c.i.clone().unwrap_or_default());
        result.push(c.j.clone().unwrap_or_default());
        result.push((c.passed.iter().flatten()).map(|&v| i64::from(v)).collect());
        if let Some(replay) = &output.topk_replay {
            result.push(vec![
                replay.full as i64,
                replay.pruned as i64,
                replay.skipped as i64,
            ]);
        }
        result
    }

    /// Rectangles change no output and no count: every sink over every pair
    /// kind, cutoff, strategy and worker split equals the block path of the
    /// same kernel (`rect_cells` 0), and the implicit kinds' plain-score passes
    /// (all sinks but Each) do take rectangles.
    #[test]
    fn rectangles_equal_item_blocks_for_every_sink_split_and_strategy() {
        let mut kinds = vec![
            (PairKind::Zip, 9, 9),
            (PairKind::Cross, 13, 11),
            (PairKind::Cross, 3, 40),
            (PairKind::OneToMany, 1, 17),
        ];
        for self_pairs in [false, true] {
            kinds.push((PairKind::Within { self_pairs }, 12, 12));
            for orientation in [Orientation::AsGiven, Orientation::Both] {
                let kind = PairKind::WithinOriented {
                    self_pairs,
                    orientation,
                };
                kinds.push((kind, 12, 12));
            }
        }
        let sinks = [
            (Sink::Each, None),
            (Sink::Hits, None),
            (Sink::RowStats, None),
            (Sink::TopK, Some(1)),
            (Sink::TopK, Some(3)),
            (Sink::Matrix, None),
        ];
        for (kind, n_a, n_b) in &kinds {
            let within = matches!(
                kind,
                PairKind::Within { .. } | PairKind::WithinOriented { .. }
            );
            let implicit = !matches!(kind, PairKind::Zip);
            let (a, b) = (table(*n_a, 0), table(*n_b, 1));
            let b = if within { &a } else { &b };
            let work = || {
                let kind = match kind {
                    PairKind::Zip => PairKind::Zip,
                    PairKind::Cross => PairKind::Cross,
                    PairKind::OneToMany => PairKind::OneToMany,
                    PairKind::Within { self_pairs } => PairKind::Within {
                        self_pairs: *self_pairs,
                    },
                    PairKind::WithinOriented {
                        self_pairs,
                        orientation,
                    } => PairKind::WithinOriented {
                        self_pairs: *self_pairs,
                        orientation: *orientation,
                    },
                    PairKind::Index { .. } => unreachable!("no index kind here"),
                };
                WorkSource::new(kind, *n_a, *n_b).unwrap()
            };
            for (sink, k) in sinks {
                let cutoffs: &[Option<i32>] = match sink {
                    Sink::Matrix => &[None],
                    Sink::Hits => &[Some(-1), Some(1), Some(2), Some(4)],
                    _ => &[None, Some(-1), Some(1), Some(2)],
                };
                for &cutoff in cutoffs {
                    for strategy in [Strategy::Full, Strategy::Bounded] {
                        for threads in [1, 2, 3, 5] {
                            let options = ExecOptions {
                                strategy,
                                threads,
                                grain: Grain::Exact,
                                sink,
                                cutoff,
                                check_bases: true,
                                k,
                            };
                            let label = format!(
                                "{kind:?} {sink:?} k={k:?} {cutoff:?} {strategy:?} t={threads}"
                            );
                            let blocks = Rect {
                                cells: 0,
                                rects: Default::default(),
                            };
                            let want = everything(&run(&blocks, &work(), &a, b, &options).unwrap());
                            for cells in [1, 5, 16, 4096] {
                                let rect = Rect {
                                    cells,
                                    rects: Default::default(),
                                };
                                let got =
                                    everything(&run(&rect, &work(), &a, b, &options).unwrap());
                                assert_eq!(got, want, "{label} cells={cells}");
                                let used = rect.rects.into_inner() > 0;
                                assert_eq!(used, implicit && sink != Sink::Each, "{label}");
                            }
                        }
                    }
                }
            }
        }
    }

    /// Tiles over a rectangle kernel: every tile's columns, position and the
    /// stream's counters equal the per-cell path of the same kernel
    /// (`rect_cells` 0), for every implicit kind, tile shape and thread count.
    #[test]
    fn tile_rectangles_equal_per_cell_tiles() {
        use crate::sink::tiles::TileStream;
        use std::sync::Arc;
        let mut kinds = vec![(PairKind::Cross, 13, 11), (PairKind::OneToMany, 1, 17)];
        for self_pairs in [false, true] {
            kinds.push((PairKind::Within { self_pairs }, 12, 12));
            for orientation in [Orientation::AsGiven, Orientation::Both] {
                let kind = PairKind::WithinOriented {
                    self_pairs,
                    orientation,
                };
                kinds.push((kind, 12, 12));
            }
        }
        let tiles = |kernel: &Rect, kind: &PairKind, shape: (usize, usize), tile, threads| {
            let within = matches!(
                kind,
                PairKind::Within { .. } | PairKind::WithinOriented { .. }
            );
            let a = Arc::new(table(shape.0, 0));
            let b = if within {
                Arc::clone(&a)
            } else {
                Arc::new(table(shape.1, 1))
            };
            let kind = match kind {
                PairKind::Cross => PairKind::Cross,
                PairKind::OneToMany => PairKind::OneToMany,
                &PairKind::Within { self_pairs } => PairKind::Within { self_pairs },
                &PairKind::WithinOriented {
                    self_pairs,
                    orientation,
                } => PairKind::WithinOriented {
                    self_pairs,
                    orientation,
                },
                _ => unreachable!("implicit kinds only"),
            };
            let work = WorkSource::new(kind, shape.0, shape.1).unwrap();
            let mut stream =
                TileStream::new(kernel, a, b, work, tile, threads, Strategy::Full, true).unwrap();
            let mut out = Vec::new();
            while let Some(tile) = stream.next_tile(kernel).unwrap() {
                let c = &tile.columns;
                out.push(vec![
                    c.dist.iter().map(|&v| i64::from(v)).collect(),
                    (c.passed.iter().flatten()).map(|&v| i64::from(v)).collect(),
                    vec![tile.rows.0 as i64, tile.rows.1 as i64],
                    vec![tile.cols.0 as i64, tile.cols.1 as i64],
                ]);
            }
            out.push(vec![vec![
                stream.counts.full as i64,
                stream.replayed as i64,
                stream.evaluations as i64,
                stream.yielded as i64,
            ]]);
            out
        };
        for (kind, n_a, n_b) in &kinds {
            for tile in [(3, 1), (4, 5), (5, 12), (12, 12), (64, 64)] {
                for threads in [1, 2, 3, 5] {
                    let blocks = Rect {
                        cells: 0,
                        rects: Default::default(),
                    };
                    let want = tiles(&blocks, kind, (*n_a, *n_b), tile, threads);
                    for cells in [1, 5, 4096] {
                        let rect = Rect {
                            cells,
                            rects: Default::default(),
                        };
                        let got = tiles(&rect, kind, (*n_a, *n_b), tile, threads);
                        let label = format!("{kind:?} tile={tile:?} t={threads} cells={cells}");
                        assert_eq!(got, want, "{label}");
                        assert!(rect.rects.into_inner() > 0, "{label}");
                    }
                }
            }
        }
    }

    #[test]
    fn row_term_sums_match_one_worker_for_every_split_and_grain() {
        // lse-style exact row sums (duplex-pf.md §9.4): Grain::Exact forces several
        // workers on these tiny inputs, so per-worker partial sums are really merged.
        for spread in [24, 400] {
            let kernel = Terms { spread };
            for kind_index in 0..kinds().len() {
                let (kind, n_a, n_b) = kinds().swap_remove(kind_index);
                let within = matches!(
                    kind,
                    PairKind::Within { .. } | PairKind::WithinOriented { .. }
                );
                let (a, b) = (table(n_a, 0), table(n_b, 1));
                let b = if within { &a } else { &b };
                for cutoff in [None, Some(1), Some(2)] {
                    let mut expected = None;
                    for threads in [1, 2, 3, 5, 16] {
                        for grain in [Grain::Exact, Grain::Auto] {
                            let work = WorkSource::new(kinds().swap_remove(kind_index).0, n_a, n_b)
                                .unwrap();
                            let options = ExecOptions {
                                strategy: Strategy::Full,
                                threads,
                                grain,
                                sink: Sink::RowStats,
                                cutoff,
                                check_bases: true,
                                k: None,
                            };
                            let output = run(&kernel, &work, &a, b, &options).unwrap();
                            let mut got = flat(&output);
                            let values = output.columns.rowstats.as_ref().unwrap();
                            let values = values.row_value.as_ref().unwrap();
                            if cutoff.is_none() {
                                assert!(values.iter().any(|v| !v.is_nan()));
                            }
                            got.push(values.iter().map(|v| v.to_bits() as i64).collect());
                            let label = format!(
                                "spread={spread} {kind_index} {cutoff:?} t={threads} {grain:?}"
                            );
                            assert_eq!(
                                &got,
                                expected.get_or_insert_with(|| got.clone()),
                                "{label}"
                            );
                        }
                    }
                }
            }
        }
    }

    const MISMATCH_FIELD: [DetailField; 1] = [DetailField {
        name: "mismatches",
        kind: FieldKind::Int32,
    }];

    /// Mismatch with one int32 detail field (the same count) and a switchable `each_keys`.
    struct Keyless {
        keys: bool,
        request: DetailRequest,
    }
    impl PairDistanceKernel for Keyless {
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
            a: ByteView<'_>,
            b: ByteView<'_>,
            cutoff: Option<i32>,
        ) -> Result<Option<i32>, EngineError> {
            Mismatch.evaluate(a, b, cutoff)
        }
        fn detail_fields(&self) -> &'static [DetailField] {
            &MISMATCH_FIELD
        }
        fn detail_request(&self) -> DetailRequest {
            self.request
        }
        fn details(
            &self,
            _: ByteView<'_>,
            _: ByteView<'_>,
            score: i32,
            out: &mut DetailColumns,
        ) -> Result<(), EngineError> {
            out.push_i32(0, score)
        }
        fn each_keys(&self) -> bool {
            self.keys
        }
    }

    /// Everything an Each, Hits or RowStats output publishes, as comparable text.
    fn published(output: &RunOutput) -> Vec<String> {
        let c = &output.columns;
        let mut out = vec![
            format!("dist {:?}", c.dist),
            format!("passed {:?}", c.passed),
            format!("i {:?} j {:?}", c.i, c.j),
            format!(
                "details {:?}",
                c.details.as_ref().map(|details| details
                    .columns()
                    .iter()
                    .map(|column| format!("{:?}", column.data))
                    .collect::<Vec<_>>())
            ),
            format!(
                "counts {} {} {}",
                output.counts.certified, output.counts.pruned, output.counts.full
            ),
        ];
        if let Some(stats) = &c.rowstats {
            out.push(format!(
                "rowstats {:?} {:?} {:?} {:?} {:?}",
                stats.best_dist,
                stats.argbest,
                stats.n_best,
                stats.count,
                stats
                    .row_value
                    .as_ref()
                    .map(|values| values.iter().map(|v| v.to_bits()).collect::<Vec<_>>())
            ));
        }
        out
    }

    /// A part's column moves into an empty merged column as the same buffer (the single
    /// part of `run_batch` is not copied, so no second copy of the column exists), and
    /// later parts follow it in order.
    #[test]
    fn merge_column_moves_the_first_part_and_appends_the_rest() {
        let mut merged: Vec<i64> = Vec::new();
        let mut first = vec![3i64, 1, 2];
        let buffer = first.as_ptr();
        merge_column(&mut merged, &mut first, "the key column").unwrap();
        assert_eq!(merged.as_ptr(), buffer);
        assert_eq!((merged.as_slice(), first.len()), (&[3, 1, 2][..], 0));
        let mut empty = Vec::new();
        merge_column(&mut merged, &mut empty, "the key column").unwrap();
        let mut second = vec![7i64, 5];
        merge_column(&mut merged, &mut second, "the key column").unwrap();
        assert_eq!((merged.as_slice(), second.len()), (&[3, 1, 2, 7, 5][..], 0));
        // An empty merged column that already holds room keeps it and copies.
        let mut roomy: Vec<i64> = Vec::with_capacity(4);
        let room = roomy.as_ptr();
        let mut part = vec![9i64, 8];
        merge_column(&mut roomy, &mut part, "the key column").unwrap();
        assert_eq!((roomy.as_ptr(), roomy.as_slice()), (room, &[9, 8][..]));
    }

    /// The batch driver over the CPU port equals `compute` for Each, Hits and RowStats
    /// (duplex-pf-cuda.md §11 item 14): one shared per-pair sink step, duplicates and
    /// (i, i) index pairs, within with and without self pairs, cutoffs, detail rows and
    /// the row-term sums.
    #[test]
    fn run_batch_over_the_cpu_port_equals_compute() {
        let cases = kinds().len() + 1;
        let request = DetailRequest::NONE.with(0);
        for case in 0..cases {
            for kernel in 0..3 {
                for sink in [Sink::Each, Sink::Hits, Sink::RowStats] {
                    for cutoff in [None, Some(0), Some(2)] {
                        let (kind, n_a, n_b) = match case {
                            c if c + 1 == cases => (
                                PairKind::Index {
                                    i: crate::work::IndexArray::from(vec![
                                        0, 3, 3, 2, 5, 0, 4, 4, 1,
                                    ]),
                                    j: crate::work::IndexArray::from(vec![
                                        1, 3, 3, 2, 0, 1, 6, 2, 1,
                                    ]),
                                },
                                7,
                                7,
                            ),
                            c => kinds().swap_remove(c),
                        };
                        let within = matches!(
                            kind,
                            PairKind::Within { .. } | PairKind::WithinOriented { .. }
                        );
                        let work = WorkSource::new(kind, n_a, n_b).unwrap();
                        let (a, b) = (table(n_a, 0), table(n_b, 1));
                        let b = if within { &a } else { &b };
                        let options = ExecOptions {
                            strategy: Strategy::Full,
                            threads: 3,
                            grain: Grain::Exact,
                            sink,
                            cutoff,
                            check_bases: true,
                            k: None,
                        };
                        let report = || ValidationReport {
                            scans: [1, 1],
                            n_illegal_seqs: 0,
                        };
                        let label = format!("case {case} kernel {kernel} {sink:?} {cutoff:?}");
                        let (expected, got) = match kernel {
                            0 => {
                                let k = Mismatch;
                                let mut port = CpuBatch {
                                    kernel: &k,
                                    a: &a,
                                    b,
                                };
                                (
                                    compute(&k, &work, &a, b, &options, report()).unwrap(),
                                    run_batch(&k, (&a, b), &mut port, &work, &options, report())
                                        .unwrap(),
                                )
                            }
                            1 => {
                                let k = Terms { spread: 24 };
                                let mut port = CpuBatch {
                                    kernel: &k,
                                    a: &a,
                                    b,
                                };
                                (
                                    compute(&k, &work, &a, b, &options, report()).unwrap(),
                                    run_batch(&k, (&a, b), &mut port, &work, &options, report())
                                        .unwrap(),
                                )
                            }
                            _ => {
                                let k = Keyless {
                                    keys: true,
                                    request,
                                };
                                let mut port = CpuBatch {
                                    kernel: &k,
                                    a: &a,
                                    b,
                                };
                                (
                                    compute(&k, &work, &a, b, &options, report()).unwrap(),
                                    run_batch(&k, (&a, b), &mut port, &work, &options, report())
                                        .unwrap(),
                                )
                            }
                        };
                        assert_eq!(published(&got), published(&expected), "{label}");
                    }
                }
            }
        }
    }

    /// `each_keys() == false` drops only the Each key column (tm on Pairs, tm.md §13.11):
    /// `passed` and the detail rows are unchanged at every worker split, other sinks keep
    /// their keys, and a call without a detail request keeps the column.
    #[test]
    fn each_keys_false_drops_only_the_each_key_column() {
        let (a, b) = (table(9, 0), table(9, 1));
        let detail = |columns: &Columns<i32>| {
            columns.details.as_ref().map(|details| {
                details
                    .columns()
                    .iter()
                    .map(|column| format!("{:?}", column.data))
                    .collect::<Vec<_>>()
            })
        };
        for cutoff in [None, Some(1)] {
            for threads in [1, 3, 5] {
                let options = |sink| ExecOptions {
                    strategy: Strategy::Full,
                    threads,
                    grain: Grain::Exact,
                    sink,
                    cutoff,
                    check_bases: true,
                    k: None,
                };
                let work = WorkSource::new(PairKind::Zip, 9, 9).unwrap();
                let request = DetailRequest::NONE.with(0);
                let run_with = |keys, request, sink| {
                    run(&Keyless { keys, request }, &work, &a, &b, &options(sink)).unwrap()
                };
                let kept = run_with(true, request, Sink::Each);
                let dropped = run_with(false, request, Sink::Each);
                assert_eq!(kept.columns.dist.len(), 9);
                assert!(dropped.columns.dist.is_empty(), "t={threads} {cutoff:?}");
                assert_eq!(kept.columns.passed, dropped.columns.passed);
                assert_eq!(detail(&kept.columns), detail(&dropped.columns));
                assert!(detail(&dropped.columns).is_some());
                // No detail request: the key column is the only output, so it stays.
                let bare = run_with(false, DetailRequest::NONE, Sink::Each);
                assert_eq!(bare.columns.dist, kept.columns.dist);
                // Hits keep their keys.
                let hits = run_with(false, request, Sink::Hits);
                assert_eq!(
                    hits.columns.dist,
                    run_with(true, request, Sink::Hits).columns.dist
                );
            }
        }
    }
}
