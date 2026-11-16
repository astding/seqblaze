//! Two-pass, tie-complete integer-distance TopK. No implicit pair-score array.
use std::collections::BinaryHeap;

use crate::{
    cancel::{self, Site, Stage, Ticker},
    detail::{reservation_error, try_reserve, try_reserve_exact, DetailColumns},
    exec::{use_bound_on_pass, ExecOptions, PassBound, Strategy},
    kernel::{
        for_each_scored, for_each_tiled, rect_cells, rect_score, BatchDistance, BoundOutcome,
        Outcome, PairDistanceKernel, Plan, Score,
    },
    plan::BoundPrefilter,
    sched,
    seq::SeqTable,
    sink::{kept::BatchDetails, try_filled, Columns, Counts, RowSlice, RunOutput},
    validate::{EngineError, ValidationReport},
    work::WorkSource,
};

pub struct TopKColumns<S: Score = i32> {
    pub dist: Vec<S>,
    pub i: Vec<i64>,
    pub j: Vec<i64>,
    pub rank: Vec<i32>,
    pub offsets: Vec<i64>,
    pub tie_at_k: Vec<u8>,
}

#[derive(Default)]
pub struct ReplayCounts {
    pub full: usize,
    pub pruned: usize,
    pub skipped: usize,
    /// Items an outcome bound certified (duplex-mfe-bound.md §6.5, QB4).
    pub certified: usize,
}

/// Keep `value` among the row's k smallest. Heap growth is reserved fallibly
/// (host-allocation-failures.md A6).
fn retain<S: Score>(heap: &mut BinaryHeap<S>, k: usize, value: S) -> Result<(), EngineError> {
    if heap.len() < k {
        let len = heap.len();
        heap.try_reserve(1)
            .map_err(|error| reservation_error::<S>("topk row heap", len as u128 + 1, error))?;
        heap.push(value);
    } else if value < *heap.peek().unwrap() {
        *heap.peek_mut().unwrap() = value;
    }
    Ok(())
}

struct ThresholdPart<S: Score> {
    heaps: RowSlice<BinaryHeap<S>>,
    counts: Counts,
}
struct ReplayPart<S: Score> {
    records: Vec<(usize, usize, S)>,
    counts: ReplayCounts,
}

/// `prefilter` adds the per-pair bound; `bounded` alone uses only early exit.
#[allow(clippy::too_many_arguments)]
fn plan<S: Score, K: PairDistanceKernel<S>>(
    kernel: &K,
    a: &SeqTable,
    b: &SeqTable,
    i: usize,
    j: usize,
    cutoff: Option<S>,
    bounded: bool,
    prefilter: bool,
) -> Plan<S> {
    if prefilter && cutoff.is_some_and(|limit| kernel.lower_bound(a, b, i, j) > limit) {
        Plan::Reject
    } else {
        Plan::Score(if bounded { cutoff } else { None })
    }
}

/// Work ID `k` = item (i, j)'s value through the pass's outcome bound (pruned: None;
/// certified: its exact key, flagged; a probed item takes the probe's outcome); an open
/// item is planned by `plan` and scored as without the bound.
#[allow(clippy::too_many_arguments)]
fn resolve<S: Score, K: PairDistanceKernel<S>>(
    kernel: &K,
    a: &SeqTable,
    b: &SeqTable,
    (k, i, j): (usize, usize, usize),
    cutoff: Option<S>,
    bounded: bool,
    prefilter: bool,
    bound: &PassBound<S>,
) -> Result<(Option<S>, bool), EngineError> {
    match bound.outcome(k, || kernel.bound_outcome(a.get(i), b.get(j), cutoff))? {
        BoundOutcome::Pruned => return Ok((None, false)),
        BoundOutcome::Certified(value) => return Ok((Some(value), true)),
        BoundOutcome::Open => {}
    }
    let value = match plan(kernel, a, b, i, j, cutoff, bounded, prefilter) {
        Plan::Score(prune) => kernel.evaluate(a.get(i), b.get(j), prune)?,
        Plan::Reject | Plan::Skip => None,
    };
    Ok((value, false))
}

/// Per-pair TopK rules shared by the CPU and batch-distance paths.
struct Rules<S: Score> {
    k: usize,
    cutoff: Option<S>,
    mirror: bool,
    first: Option<Vec<bool>>,
}

impl<S: Score> Rules<S> {
    fn new(work: &WorkSource, cutoff: Option<S>, k: usize) -> Result<Self, EngineError> {
        Ok(Self {
            k,
            cutoff,
            mirror: work.mirrors_within(),
            first: work.first_occurrences()?,
        })
    }

    /// A distinct partner contributes; a requested within diagonal is an ordinary
    /// partner of its own row (A11, D87).
    fn eligible(&self, work_id: usize) -> bool {
        self.first.as_ref().is_none_or(|mask| mask[work_id])
    }

    /// Whether the work item also feeds row j: symmetric within off the diagonal only.
    fn mirrored(&self, i: usize, j: usize) -> bool {
        self.mirror && i != j
    }

    /// First pass: an exact value that passes the user cutoff joins its rows' heaps.
    fn threshold(
        &self,
        heaps: &mut RowSlice<BinaryHeap<S>>,
        work_id: usize,
        i: usize,
        j: usize,
        value: S,
    ) -> Result<(), EngineError> {
        if self.cutoff.is_none_or(|limit| value <= limit) && self.eligible(work_id) {
            retain(heaps.get_mut(i), self.k, value)?;
            if self.mirrored(i, j) {
                retain(heaps.get_mut(j), self.k, value)?;
            }
        }
        Ok(())
    }

    /// Replay: a committed exact value within a row threshold becomes a record.
    /// Record growth is reserved fallibly (host-allocation-failures.md A9).
    fn commit(
        &self,
        records: &mut Vec<(usize, usize, S)>,
        thresholds: &[Option<S>],
        i: usize,
        j: usize,
        value: S,
    ) -> Result<(), EngineError> {
        if thresholds[i].is_some_and(|limit| value <= limit) {
            try_reserve(records, 1, REPLAY_RECORDS)?;
            records.push((i, j, value));
        }
        if self.mirrored(i, j) && thresholds[j].is_some_and(|limit| value <= limit) {
            try_reserve(records, 1, REPLAY_RECORDS)?;
            records.push((j, i, value));
        }
        Ok(())
    }
}

const REPLAY_RECORDS: &str = "topk replay records";

impl<S: Score> Rules<S> {
    /// The first pass over a row run of implicit work (no first-occurrence
    /// mask): items (i, j0 + k) with rectangle scores `scores[k]` under the
    /// pass's `cutoff` (`rect_score`), as `threshold` item by item; without
    /// mirrors the row's heap is taken once for the run.
    fn threshold_run(
        &self,
        part: &mut ThresholdPart<S>,
        (i, j0): (usize, usize),
        scores: &[Option<S>],
        cutoff: Option<S>,
    ) -> Result<(), EngineError> {
        if self.mirror {
            for (j, &score) in (j0..).zip(scores) {
                if let Some(value) = rect_score(score, cutoff) {
                    part.counts.full += 1;
                    self.threshold(&mut part.heaps, 0, i, j, value)?;
                } else {
                    part.counts.pruned += 1;
                }
            }
            return Ok(());
        }
        let heap = part.heaps.get_mut(i);
        for &score in scores {
            if let Some(value) = rect_score(score, cutoff) {
                part.counts.full += 1;
                if self.cutoff.is_none_or(|limit| value <= limit) {
                    retain(heap, self.k, value)?;
                }
            } else {
                part.counts.pruned += 1;
            }
        }
        Ok(())
    }

    /// The replay over a row run of implicit work (no first-occurrence mask):
    /// items (i, j0 + k) with rectangle scores `scores[k]` under a cutoff at
    /// least each scored item's limit, as the per-item replay plans and
    /// commits them; without mirrors the row's limit is read once for the run.
    fn replay_run(
        &self,
        part: &mut ReplayPart<S>,
        thresholds: &[Option<S>],
        (i, j0): (usize, usize),
        scores: &[Option<S>],
        bounded: bool,
    ) -> Result<(), EngineError> {
        if self.mirror {
            for (j, &score) in (j0..).zip(scores) {
                let limit = thresholds[i].max(thresholds[j]);
                let score = match limit {
                    _ if !bounded => score,
                    None => {
                        part.counts.skipped += 1;
                        continue;
                    }
                    limit => rect_score(score, limit),
                };
                if let Some(value) = score {
                    part.counts.full += 1;
                    self.commit(&mut part.records, thresholds, i, j, value)?;
                } else {
                    part.counts.pruned += 1;
                }
            }
            return Ok(());
        }
        let limit = thresholds[i];
        if bounded && limit.is_none() {
            part.counts.skipped += scores.len();
            return Ok(());
        }
        let prune = if bounded { limit } else { None };
        for (j, &score) in (j0..).zip(scores) {
            if let Some(value) = rect_score(score, prune) {
                part.counts.full += 1;
                if limit.is_some_and(|limit| value <= limit) {
                    try_reserve(&mut part.records, 1, REPLAY_RECORDS)?;
                    part.records.push((i, j, value));
                }
            } else {
                part.counts.pruned += 1;
            }
        }
        Ok(())
    }
}

/// Each row's threshold: the largest of its kept values (None: no partner).
/// If fewer than k pass, the heap already contains every passing partner. The
/// output is reserved fallibly (host-allocation-failures.md A7, A8).
fn thresholds_of<S: Score>(heaps: &[BinaryHeap<S>]) -> Result<Vec<Option<S>>, EngineError> {
    let mut thresholds = Vec::new();
    try_reserve_exact(&mut thresholds, heaps.len(), "topk thresholds")?;
    thresholds.extend(heaps.iter().map(|heap| heap.peek().copied()));
    Ok(thresholds)
}

/// Merge per-worker row heaps (in canonical order) into per-row thresholds.
fn merge_thresholds<S: Score>(
    n_rows: usize,
    k: usize,
    parts: Vec<RowSlice<BinaryHeap<S>>>,
) -> Result<Vec<Option<S>>, EngineError> {
    let mut heaps = try_filled(n_rows, BinaryHeap::new(), "TopK row heaps")?;
    for part in parts {
        for (target, partial) in heaps[part.first..].iter_mut().zip(part.rows) {
            for value in partial {
                retain(target, k, value)?;
            }
        }
    }
    thresholds_of(&heaps)
}

fn output<S: Score>(
    work: &WorkSource,
    options: &ExecOptions<S>,
    validation: ValidationReport,
    counts: Counts,
    replay: ReplayCounts,
    topk: TopKColumns<S>,
) -> RunOutput<S> {
    RunOutput {
        columns: Columns {
            topk: Some(topk),
            ..Columns::default()
        },
        shape: work.shape,
        counts,
        threads: options.threads,
        n_illegal_seqs: validation.n_illegal_seqs,
        validation_scans: validation.scans,
        topk_replay: Some(replay),
    }
}

pub(crate) fn run<S: Score, K: PairDistanceKernel<S>>(
    kernel: &K,
    work: &WorkSource,
    a: &SeqTable,
    b: &SeqTable,
    options: &ExecOptions<S>,
    validation: ValidationReport,
) -> Result<RunOutput<S>, EngineError> {
    let k = options
        .k
        .ok_or_else(|| EngineError::Parameter("TopK requires k".into()))?;
    let bounded = options.strategy == Strategy::Bounded;
    let workers = options.workers();
    // An outcome bound (duplex-mfe-bound.md §6.2): pass 1 against the cutoff (none:
    // certify only), the replay against each item's replay limit; each pass decides
    // once by the weak-bound rule (§6.6).
    let port = bounded && kernel.outcome_bound();
    let first_outcomes = if port {
        use_bound_on_pass(kernel, work, (a, b), workers, |_, _, _| {
            Some(options.cutoff)
        })?
    } else {
        None
    };
    // Replay thresholds can reject pairs even without a user cutoff.
    let prefilter = bounded && !port && BoundPrefilter::for_work(&work.kind).per_pair();
    if prefilter {
        kernel.prepare_bound(a, b, workers)?;
    }
    // Plain scores over implicit work: rectangles for a rectangle kernel. The
    // first pass tests the per-pair bound only under a user cutoff.
    let rects = rect_cells(kernel, work, (a, b), prefilter && options.cutoff.is_some());
    let replay_rects = rect_cells(kernel, work, (a, b), prefilter);
    // Both passes score only; kept entries' details are costed separately.
    let worker_limit = workers.count(work.n_pairs, kernel.pair_cost(a, b, false));
    let rules = Rules::new(work, options.cutoff, k)?;
    cancel::stage(Stage::TopK1, work.n_pairs);
    let threshold_parts = sched::map_ranges(work.n_pairs, worker_limit, &|start, end| {
        let mut part = ThresholdPart {
            heaps: RowSlice::for_range(work, start, end, BinaryHeap::new(), "TopK worker heaps")?,
            counts: Counts::default(),
        };
        let mut ticker = Ticker::new(Site::TopK1);
        if let Some(bound) = first_outcomes.as_ref() {
            // A pass that uses the outcome bound: item by item through its outcomes.
            for (offset, (i, j)) in work.range(start, end).enumerate() {
                let (value, certified) = resolve(
                    kernel,
                    a,
                    b,
                    (start + offset, i, j),
                    options.cutoff,
                    bounded,
                    prefilter,
                    bound,
                )?;
                if let Some(value) = value {
                    if certified {
                        part.counts.certified += 1;
                    } else {
                        part.counts.full += 1;
                    }
                    rules.threshold(&mut part.heaps, start + offset, i, j, value)?;
                } else {
                    part.counts.pruned += 1;
                }
                ticker.tick()?;
            }
        } else if let Some(cells) = rects {
            // A row run at a time; the heaps keep each row's k smallest
            // whatever the order, and implicit work's items are all eligible.
            debug_assert!(rules.first.is_none());
            let cutoff = if bounded { options.cutoff } else { None };
            for_each_tiled(
                kernel,
                (a, b),
                work,
                (start, end),
                cells,
                |_, _| cutoff,
                |i, j0, scores| {
                    rules.threshold_run(&mut part, (i, j0), scores, cutoff)?;
                    ticker.add(scores.len())
                },
            )?;
        } else {
            let mut take = |work_id: usize, (i, j): (usize, usize), score: Option<S>| {
                if let Some(value) = score {
                    part.counts.full += 1;
                    rules.threshold(&mut part.heaps, work_id, i, j, value)
                } else {
                    part.counts.pruned += 1;
                    Ok(())
                }
            };
            for_each_scored(
                kernel,
                (a, b),
                work,
                (start, end),
                |_, (i, j)| plan(kernel, a, b, i, j, options.cutoff, bounded, prefilter),
                |offset, pair, outcome| {
                    let Outcome::Scored(score) = outcome else {
                        unreachable!("no item is skipped")
                    };
                    take(start + offset, pair, score)?;
                    ticker.tick()
                },
            )?;
        }
        ticker.finish()?;
        Ok(part)
    })?;
    let mut counts = Counts::default();
    let mut heap_parts = Vec::with_capacity(threshold_parts.len());
    for part in threshold_parts {
        counts.full += part.counts.full;
        counts.pruned += part.counts.pruned;
        counts.certified += part.counts.certified;
        heap_parts.push(part.heaps);
    }
    let thresholds = merge_thresholds(work.shape.0, k, heap_parts)?;
    // The replay limit of work ID k, None when the bounded replay skips it.
    let replay_limit = |k: usize, i: usize, j: usize| -> Option<S> {
        if !rules.eligible(k) {
            return None;
        }
        if rules.mirror {
            thresholds[i].max(thresholds[j])
        } else {
            thresholds[i]
        }
    };
    let replay_outcomes = if port {
        use_bound_on_pass(kernel, work, (a, b), workers, |k, i, j| {
            replay_limit(k, i, j).map(Some)
        })?
    } else {
        None
    };
    cancel::stage(Stage::TopK2, work.n_pairs);
    let replay_parts = sched::map_ranges(work.n_pairs, worker_limit, &|start, end| {
        let mut part = ReplayPart {
            records: Vec::new(),
            counts: ReplayCounts::default(),
        };
        let mut ticker = Ticker::new(Site::TopK2);
        let limit = |i: usize, j: usize| {
            if rules.mirror {
                thresholds[i].max(thresholds[j])
            } else {
                thresholds[i]
            }
        };
        if let Some(bound) = replay_outcomes.as_ref() {
            // A pass that uses the outcome bound (bounded): item by item through its
            // outcomes; an item the bounded replay skips is visited and counted.
            for (offset, (i, j)) in work.range(start, end).enumerate() {
                let limit = limit(i, j);
                if !rules.eligible(start + offset) || limit.is_none() {
                    part.counts.skipped += 1;
                    // A skipped item is visited, so it counts (§3.2 `topk2`).
                    ticker.tick()?;
                    continue;
                }
                let (value, certified) = resolve(
                    kernel,
                    a,
                    b,
                    (start + offset, i, j),
                    limit,
                    true,
                    prefilter,
                    bound,
                )?;
                if let Some(value) = value {
                    if certified {
                        part.counts.certified += 1;
                    } else {
                        part.counts.full += 1;
                    }
                    rules.commit(&mut part.records, &thresholds, i, j, value)?;
                } else {
                    part.counts.pruned += 1;
                }
                ticker.tick()?;
            }
        } else if let Some(cells) = replay_rects {
            // A row run at a time; the final sort orders the records, and
            // implicit work's items are all eligible (the work ID is never
            // read). A skipped item's rectangle score is discarded. A
            // rectangle's cutoff is the largest limit of its rows (and, for
            // symmetric within, of its columns, so of every cell): at least
            // each scored item's own; with no limit, every item is skipped.
            let widest = |spans: &[(usize, usize, usize)], (c0, c1): (usize, usize)| {
                if !bounded {
                    return None;
                }
                let rows = spans.iter().map(|&(i, ..)| thresholds[i]);
                let cols = (c0..c1).filter(|_| rules.mirror).map(|j| thresholds[j]);
                Some(rows.chain(cols).max().flatten().unwrap_or(S::MIN))
            };
            for_each_tiled(
                kernel,
                (a, b),
                work,
                (start, end),
                cells,
                widest,
                |i, j0, scores| {
                    rules.replay_run(&mut part, &thresholds, (i, j0), scores, bounded)?;
                    ticker.add(scores.len())
                },
            )?;
        } else {
            let mut take = |work_id: usize, (i, j): (usize, usize), outcome: Outcome<S>| {
                match outcome {
                    // A skipped item is visited, so it counts (§3.2 `topk2`).
                    Outcome::Skipped => part.counts.skipped += 1,
                    Outcome::Scored(Some(value)) => {
                        part.counts.full += 1;
                        if rules.eligible(work_id) {
                            rules.commit(&mut part.records, &thresholds, i, j, value)?;
                        }
                    }
                    Outcome::Scored(None) => part.counts.pruned += 1,
                }
                Ok(())
            };
            for_each_scored(
                kernel,
                (a, b),
                work,
                (start, end),
                |offset, (i, j)| {
                    if !bounded {
                        // Full replay recomputes even duplicate items.
                        return Plan::Score(None);
                    }
                    let limit = limit(i, j);
                    if !rules.eligible(start + offset) || limit.is_none() {
                        return Plan::Skip;
                    }
                    plan(kernel, a, b, i, j, limit, true, prefilter)
                },
                |offset, pair, outcome| {
                    take(start + offset, pair, outcome)?;
                    ticker.tick()
                },
            )?;
        }
        ticker.finish()?;
        Ok(part)
    })?;
    let mut records = Vec::new();
    let mut replay = ReplayCounts::default();
    for mut part in replay_parts {
        // The first part's buffer is taken without allocating; later parts
        // reserve fallibly (host-allocation-failures.md A9).
        crate::exec::merge_column(&mut records, &mut part.records, REPLAY_RECORDS)?;
        replay.full += part.counts.full;
        replay.pruned += part.counts.pruned;
        replay.skipped += part.counts.skipped;
        replay.certified += part.counts.certified;
    }
    let topk = finalize(records, work.shape.0, k)?;
    // Detail fields of kept entries only, each in its row's orientation
    // (a mirrored within entry (j, i) is computed as zip(seq_j, seq_i)).
    let details = if kernel.detail_request().is_empty() {
        None
    } else {
        Some(crate::exec::entry_details(
            kernel,
            a,
            b,
            (&topk.i, &topk.j, &topk.dist),
            workers,
        )?)
    };
    let mut out = output(work, options, validation, counts, replay, topk);
    out.columns.details = details;
    Ok(out)
}

/// Full-strategy TopK over a batch-distance backend: the same two passes and
/// per-pair rules as `run`, with distances read from device chunks. Detail
/// fields of the final entries, each in its row's orientation, come from one
/// port `pairs` pass when the backend delivers them, else from the host
/// `details` port (`exec::entry_details`), as on the CPU.
pub fn run_batch<S: Score, K: PairDistanceKernel<S> + ?Sized>(
    kernel: &K,
    (a, b): (&SeqTable, &SeqTable),
    distances: &mut dyn BatchDistance<S>,
    work: &WorkSource,
    options: &ExecOptions<S>,
    validation: ValidationReport,
) -> Result<RunOutput<S>, EngineError> {
    let k = options
        .k
        .ok_or_else(|| EngineError::Parameter("TopK requires k".into()))?;
    let rules = Rules::new(work, options.cutoff, k)?;
    let mut heaps = RowSlice {
        first: 0,
        rows: try_filled(work.shape.0, BinaryHeap::new(), "TopK row heaps")?,
    };
    let mut counts = Counts::default();
    cancel::stage(Stage::TopK1, work.n_pairs);
    distances.range(work, 0, work.n_pairs, false, &mut |first, values, _| {
        let pairs = work.range(first, first + values.len());
        for (offset, ((i, j), &value)) in pairs.zip(values).enumerate() {
            counts.full += 1;
            rules.threshold(&mut heaps, first + offset, i, j, value)?;
        }
        // A host boundary after each delivered chunk (cancellation-progress.md §3.5).
        cancel::boundary(Site::TopK1, values.len())
    })?;
    // One part: read each row's threshold directly (no merge copy).
    let thresholds = thresholds_of(&heaps.rows)?;
    drop(heaps);
    let mut records = Vec::new();
    let mut replay = ReplayCounts::default();
    cancel::stage(Stage::TopK2, work.n_pairs);
    distances.range(work, 0, work.n_pairs, false, &mut |first, values, _| {
        let pairs = work.range(first, first + values.len());
        for (offset, ((i, j), &value)) in pairs.zip(values).enumerate() {
            replay.full += 1;
            if rules.eligible(first + offset) {
                rules.commit(&mut records, &thresholds, i, j, value)?;
            }
        }
        cancel::boundary(Site::TopK2, values.len())
    })?;
    let topk = finalize(records, work.shape.0, k)?;
    let details = match BatchDetails::of(kernel, &*distances) {
        BatchDetails::None => None,
        BatchDetails::Port => {
            // Reserved fallibly before any element is written
            // (host-allocation-failures.md A11).
            let mut entries: Vec<(usize, usize)> = Vec::new();
            try_reserve_exact(&mut entries, topk.i.len(), "topk detail entries")?;
            entries.extend(
                topk.i
                    .iter()
                    .zip(&topk.j)
                    .map(|(&i, &j)| (i as usize, j as usize)),
            );
            let mut values = Vec::new();
            let mut rows = DetailColumns::new(kernel.detail_fields(), kernel.detail_request());
            distances.pairs(&entries, &mut values, Some(&mut rows))?;
            if values != topk.dist || rows.rows() != entries.len() {
                return Err(EngineError::Device(
                    "TopK detail pass disagrees with the selection pass; no result was published"
                        .into(),
                ));
            }
            Some(rows)
        }
        BatchDetails::Host => Some(crate::exec::entry_details(
            kernel,
            a,
            b,
            (&topk.i, &topk.j, &topk.dist),
            options.workers(),
        )?),
    };
    let mut out = output(work, options, validation, counts, replay, topk);
    out.columns.details = details;
    Ok(out)
}

fn finalize<S: Score>(
    mut records: Vec<(usize, usize, S)>,
    n_rows: usize,
    k: usize,
) -> Result<TopKColumns<S>, EngineError> {
    records.sort_unstable_by_key(|&(i, j, value)| (i, value, j));
    let offsets_len = n_rows
        .checked_add(1)
        .ok_or_else(|| EngineError::Overflow("TopK row offsets exceed platform limit".into()))?;
    // Every output column is reserved fallibly, in this order, before the
    // per-row loop and its dtype checks (host-allocation-failures.md A10).
    let mut result = TopKColumns {
        dist: Vec::new(),
        i: Vec::new(),
        j: Vec::new(),
        rank: Vec::new(),
        offsets: Vec::new(),
        tie_at_k: Vec::new(),
    };
    let n = records.len();
    try_reserve_exact(&mut result.dist, n, "topk key column")?;
    try_reserve_exact(&mut result.i, n, "topk i column")?;
    try_reserve_exact(&mut result.j, n, "topk j column")?;
    try_reserve_exact(&mut result.rank, n, "topk rank column")?;
    try_reserve_exact(&mut result.offsets, offsets_len, "topk offsets column")?;
    try_reserve_exact(&mut result.tie_at_k, n_rows, "topk tie_at_k column")?;
    result.offsets.resize(offsets_len, 0);
    result.tie_at_k.resize(n_rows, 0);
    let mut cursor = 0;
    for row in 0..n_rows {
        let start = cursor;
        let mut previous = None;
        let mut rank = 0;
        while cursor < records.len() && records[cursor].0 == row {
            let (i, j, value) = records[cursor];
            if previous != Some(value) {
                rank = i32::try_from(cursor - start + 1).map_err(|_| {
                    EngineError::Overflow(format!(
                        "TopK competition rank exceeds int32 in row {row}"
                    ))
                })?;
                previous = Some(value);
            }
            result.i.push(
                i64::try_from(i)
                    .map_err(|_| EngineError::Overflow("TopK row index exceeds int64".into()))?,
            );
            result.j.push(
                i64::try_from(j).map_err(|_| {
                    EngineError::Overflow("TopK partner index exceeds int64".into())
                })?,
            );
            result.dist.push(value);
            result.rank.push(rank);
            cursor += 1;
        }
        result.offsets[row + 1] = i64::try_from(cursor)
            .map_err(|_| EngineError::Overflow("TopK offsets exceed int64".into()))?;
        result.tie_at_k[row] = u8::from(cursor - start > k);
    }
    Ok(result)
}
