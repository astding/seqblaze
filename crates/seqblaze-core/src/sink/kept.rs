//! Kept-pair detail helpers shared by the CPU engine and batch-distance
//! backends (docs/wiki/gpu/levenshtein-cuda.md §3, §6): the Hits sort with its
//! detail gather, the Each detail pass, and dense detail rows of committed cells.
//!
//! Every detail row comes from the kernel's `details(a, b, score, out)` port or
//! from a backend that delivers the same values, so it depends only on the pair
//! and its exact score, never on the strategy, the backend or the thread count.
use crate::{
    cancel::{self, Site, Stage, Ticker},
    detail::{dense_row, try_reserve_exact, DetailColumns},
    kernel::{BatchDistance, PairDistanceKernel, Score},
    sched::Workers,
    seq::SeqTable,
    sink::{matrix::DensePair, try_filled},
    validate::EngineError,
    work::WorkSource,
};

type Hit<S> = (usize, usize, S);

/// Sort hits by (i, j) and keep each coordinate's first occurrence, moving the
/// aligned detail rows with them (one row per hit, in hit order).
pub fn sort_hits<S: Score>(
    mut hits: Vec<Hit<S>>,
    mut details: DetailColumns,
) -> Result<(Vec<Hit<S>>, DetailColumns), EngineError> {
    let mut order = try_filled(hits.len(), 0usize, "hit order")?;
    for (k, slot) in order.iter_mut().enumerate() {
        *slot = k;
    }
    order.sort_unstable_by_key(|&k| (hits[k].0, hits[k].1, k));
    order.dedup_by_key(|k| (hits[*k].0, hits[*k].1));
    if order.len() != hits.len() || order.iter().enumerate().any(|(n, &k)| n != k) {
        details = details.gather(&order)?;
        let mut sorted = Vec::new();
        try_reserve_exact(&mut sorted, order.len(), "sorted hits")?;
        sorted.extend(order.iter().map(|&k| hits[k]));
        hits = sorted;
    }
    Ok((hits, details))
}

/// Detail rows of an Each result whose scores were computed elsewhere: rows of
/// passing work IDs through the kernel's `details` port, failing rows masked
/// (INT32_MAX, INT64_MAX or null), in work-ID order. `passed` is None without
/// a cutoff (every row passes). Contiguous ranges run on host workers and are
/// appended in order.
pub fn each_details<S: Score, K: PairDistanceKernel<S> + ?Sized>(
    kernel: &K,
    work: &WorkSource,
    (a, b): (&SeqTable, &SeqTable),
    dist: &[S],
    passed: Option<&[u8]>,
    workers: Workers,
) -> Result<DetailColumns, EngineError> {
    let empty = DetailColumns::new(kernel.detail_fields(), kernel.detail_request());
    // Scheduling only: the kernel's cost of a pair with its detail fields.
    let limit = workers.count(work.n_pairs, kernel.pair_cost(a, b, true));
    cancel::stage(Stage::Details, work.n_pairs);
    let parts = crate::sched::map_ranges(work.n_pairs, limit, &|start, end| {
        let mut part = empty.clone();
        let mut ticker = Ticker::new(Site::Details);
        for (offset, (i, j)) in work.range(start, end).enumerate() {
            let k = start + offset;
            if passed.is_none_or(|passed| passed[k] != 0) {
                kernel.details(a.get(i), b.get(j), dist[k], &mut part)?;
            } else {
                part.push_masked()?;
            }
            ticker.tick()?;
        }
        ticker.finish()?;
        Ok(part)
    })?;
    let mut result = empty;
    for mut part in parts {
        result.append(&mut part)?;
    }
    Ok(result)
}

/// Where a batch-distance sink takes the call's detail fields from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum BatchDetails {
    /// The call requests none.
    None,
    /// The backend delivers them beside each distance.
    Port,
    /// The host computes them for kept pairs through `details`.
    Host,
}

impl BatchDetails {
    pub(crate) fn of<S: Score, K: PairDistanceKernel<S> + ?Sized>(
        kernel: &K,
        distances: &dyn BatchDistance<S>,
    ) -> Self {
        if kernel.detail_request().is_empty() {
            Self::None
        } else if distances.delivers_details() {
            Self::Port
        } else {
            Self::Host
        }
    }
}

/// A committed dense cell: its exact score, the index of its numeric row in
/// the backend's delivered rows (None: compute it on the host) and whether a
/// symmetric-within mirror cell needs its own row, computed as zip(seq_j,
/// seq_i) and never copied from (i, j).
pub(crate) struct DenseCell<S: Score = i32> {
    pub i: usize,
    pub j: usize,
    pub value: S,
    pub mirror: bool,
    pub row: Option<usize>,
}

/// Numeric detail rows of dense cells, flat with stride `width` (the
/// request's numeric field count, not a fixed maximum): per cell its row, then
/// its mirror's row when it has one, in cell order across the worker parts.
pub(crate) struct DenseRows {
    width: usize,
    parts: Vec<Vec<i64>>,
}

impl DenseRows {
    /// The record payloads of `cells` (the cells these rows were computed
    /// for), in order; each is built only when the consumer takes it.
    pub(crate) fn pairs<'r, S: Score>(
        &'r self,
        cells: &'r [DenseCell<S>],
    ) -> impl Iterator<Item = Result<DensePair, EngineError>> + 'r {
        let mut rows = self
            .parts
            .iter()
            .flat_map(move |part| part.chunks_exact(self.width.max(1)));
        let missing =
            || EngineError::Device("dense detail rows do not match the committed cells".into());
        cells.iter().map(move |cell| {
            let row = rows.next().ok_or_else(missing)?;
            let mirror = if cell.mirror {
                Some(dense_row(rows.next().ok_or_else(missing)?))
            } else {
                None
            };
            Ok(DensePair {
                cell: dense_row(row),
                mirror,
            })
        })
    }
}

/// Numeric detail rows of `cells`, flat (`DenseRows`): rows the backend
/// delivered are copied from `delivered`, the others and every mirror row are
/// computed through `details` on host workers. Buffers are reserved fallibly.
pub(crate) fn dense_extras<S: Score, K: PairDistanceKernel<S> + ?Sized>(
    kernel: &K,
    (a, b): (&SeqTable, &SeqTable),
    cells: &[DenseCell<S>],
    delivered: Option<&DetailColumns>,
    workers: Workers,
) -> Result<DenseRows, EngineError> {
    let scratch = DetailColumns::new(kernel.detail_fields(), kernel.detail_request());
    let width = scratch.width();
    let host = cells
        .iter()
        .any(|cell| cell.row.is_none() || delivered.is_none() || cell.mirror);
    let limit = if host {
        workers.count(cells.len(), kernel.pair_cost(a, b, true))
    } else {
        1
    };
    let parts = crate::sched::map_ranges(cells.len(), limit, &|start, end| {
        let mut scratch = scratch.clone();
        let cells = &cells[start..end];
        let rows = cells.len() + cells.iter().filter(|cell| cell.mirror).count();
        let mut part = Vec::new();
        let len = rows.checked_mul(width).ok_or_else(|| {
            EngineError::Overflow("dense detail rows exceed platform limit".into())
        })?;
        try_reserve_exact(&mut part, len, "dense detail rows")?;
        // Detail checkpoints of a chunk's kept cells; they count toward no stage
        // (the enclosing pairs or tile stage stays open).
        let mut ticker = Ticker::new(Site::Details);
        for cell in cells {
            match (cell.row, delivered) {
                (Some(row), Some(delivered)) => delivered.push_numeric_row(row, &mut part),
                _ => {
                    scratch.clear();
                    kernel.details(a.get(cell.i), b.get(cell.j), cell.value, &mut scratch)?;
                    scratch.push_numeric_row(0, &mut part);
                }
            }
            if cell.mirror {
                scratch.clear();
                kernel.details(a.get(cell.j), b.get(cell.i), cell.value, &mut scratch)?;
                scratch.push_numeric_row(0, &mut part);
            }
            ticker.tick()?;
        }
        ticker.finish()?;
        Ok(part)
    })?;
    Ok(DenseRows { width, parts })
}
