//! Numeric column ownership and streaming CPU row reductions.
use crate::{kernel::Score, validate::EngineError};

pub mod kept;
pub mod matrix;
pub mod rowsum;
pub mod tiles;
pub mod topk;
use rowsum::ExactSum;
use topk::{ReplayCounts, TopKColumns};

/// `n` copies of `value`, reserved fallibly: failure maps to DeviceError
/// ("unable to reserve {what} ({bytes} bytes): …; no result was published",
/// host-allocation-failures.md §3). Worker threads call it (row-reduction
/// scratch), so the message goes through [`crate::detail::device_error`] and
/// never aborts (engine-layout §6).
pub fn try_filled<T: Clone>(n: usize, value: T, what: &str) -> Result<Vec<T>, EngineError> {
    let mut result = Vec::new();
    result
        .try_reserve_exact(n)
        .map_err(|error| crate::detail::reservation_error::<T>(what, n as u128, error))?;
    result.resize(n, value);
    Ok(result)
}

/// A dense output column written in place by the workers of one stage (the
/// rectangle paths of Matrix and Tiles), each writing only cells that no other
/// thread reads or writes while the workers run; the column is read only after
/// every worker has been joined.
pub(crate) struct SharedCells<T> {
    cells: *mut T,
    len: usize,
}

// SAFETY: a `SharedCells` addresses a column that outlives the worker scope
// and is not otherwise touched during it; workers write disjoint cells (`set`'s
// contract), and `T` values may move between threads.
unsafe impl<T: Send> Send for SharedCells<T> {}
// SAFETY: as for `Send`: shared use only writes disjoint cells.
unsafe impl<T: Send> Sync for SharedCells<T> {}

impl<T: Copy> SharedCells<T> {
    pub(crate) fn new(column: &mut [T]) -> Self {
        Self {
            cells: column.as_mut_ptr(),
            len: column.len(),
        }
    }

    /// Write `value` to `cell` (bounds checked).
    ///
    /// # Safety
    ///
    /// No other thread reads or writes `cell` while the workers run.
    #[inline]
    pub(crate) unsafe fn set(&self, cell: usize, value: T) {
        assert!(cell < self.len, "a cell inside the column");
        // SAFETY: in bounds (asserted); this caller is the cell's only user
        // while the workers run (the contract above).
        unsafe { self.cells.add(cell).write(value) }
    }

    /// The `len` cells from `start` as a slice (bounds checked once).
    ///
    /// # Safety
    ///
    /// No other thread reads or writes these cells while the returned slice
    /// lives, and the caller holds no other reference to them.
    #[inline]
    #[allow(clippy::mut_from_ref)]
    pub(crate) unsafe fn run(&self, start: usize, len: usize) -> &mut [T] {
        assert!(
            start <= self.len && len <= self.len - start,
            "a run inside the column"
        );
        // SAFETY: in bounds (asserted); the cells are this caller's alone
        // while the slice lives (the contract above).
        unsafe { std::slice::from_raw_parts_mut(self.cells.add(start), len) }
    }
}

/// Per-worker row-reduction scratch: `rows[k]` belongs to source row `first + k`.
pub(crate) struct RowSlice<T> {
    pub first: usize,
    pub rows: Vec<T>,
}

impl<T: Clone> RowSlice<T> {
    /// Rows touched by work IDs `start..end`: only that interval for row-monotone
    /// non-mirrored shapes, otherwise every first-table row.
    pub fn for_range(
        work: &crate::work::WorkSource,
        start: usize,
        end: usize,
        value: T,
        what: &str,
    ) -> Result<Self, EngineError> {
        let (first, n) = work.row_interval(start, end).unwrap_or((0, work.shape.0));
        Ok(Self {
            first,
            rows: try_filled(n, value, what)?,
        })
    }
    #[inline]
    pub fn get_mut(&mut self, row: usize) -> &mut T {
        &mut self.rows[row - self.first]
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Sink {
    Each,
    Hits,
    RowStats,
    TopK,
    Matrix,
}

pub struct Columns<S: Score = i32> {
    pub dist: Vec<S>,
    pub passed: Option<Vec<u8>>,
    pub i: Option<Vec<i64>>,
    pub j: Option<Vec<i64>>,
    pub rowstats: Option<RowStatsColumns<S>>,
    pub topk: Option<TopKColumns<S>>,
    /// Requested detail fields, one row per output entry (Each, Hits, TopK) or
    /// per cell (Matrix, Tiles); None when the kernel call requests none.
    pub details: Option<crate::detail::DetailColumns>,
}

impl<S: Score> Default for Columns<S> {
    fn default() -> Self {
        Self {
            dist: Vec::new(),
            passed: None,
            i: None,
            j: None,
            rowstats: None,
            topk: None,
            details: None,
        }
    }
}

impl<S: Score> Columns<S> {
    /// Hits columns: sorted by (i, j), one entry per distinct coordinate.
    /// Duplicate coordinates carry equal distances, so which one is kept is moot.
    /// The i, j and key columns are reserved fallibly, in that order, before any
    /// is written; a refusal is a DeviceError (host-allocation-failures.md A4).
    pub fn from_hits(mut hits: Vec<(usize, usize, S)>) -> Result<Self, EngineError> {
        use crate::detail::try_reserve_exact;
        hits.sort_unstable_by_key(|&(i, j, _)| (i, j));
        hits.dedup_by_key(|entry| (entry.0, entry.1));
        let (mut ii, mut jj, mut dist) = (Vec::new(), Vec::new(), Vec::new());
        try_reserve_exact(&mut ii, hits.len(), "hits i column")?;
        try_reserve_exact(&mut jj, hits.len(), "hits j column")?;
        try_reserve_exact(&mut dist, hits.len(), "hits key column")?;
        for (i, j, d) in hits {
            ii.push(i as i64);
            jj.push(j as i64);
            dist.push(d);
        }
        Ok(Self {
            dist,
            i: Some(ii),
            j: Some(jj),
            ..Self::default()
        })
    }
}

pub struct RowStatsColumns<S: Score = i32> {
    pub best_dist: Vec<S>,
    pub argbest: Vec<i64>,
    pub n_best: Vec<i32>,
    pub count: Option<Vec<i64>>,
    /// The kernel's value of each row's exact row-term sum (`row_terms` kernels only);
    /// NaN for a row without a (passing) partner.
    pub row_value: Option<Vec<f64>>,
}

/// One constant-size state per row. An absent partner is distinct from MAX distance.
/// The exact row-term sums of a `row_terms` kernel are kept beside it (`ExactSum`).
#[derive(Clone, Copy)]
pub struct RowState<S: Score = i32> {
    best: S,
    argbest: i64,
    n_best: u64,
    count: u64,
}

impl<S: Score> Default for RowState<S> {
    fn default() -> Self {
        Self {
            best: S::MASK,
            argbest: -1,
            n_best: 0,
            count: 0,
        }
    }
}

impl<S: Score> RowState<S> {
    /// Reconstruct a mergeable partial summary. Integer export bounds apply at finalization.
    pub fn from_summary(
        best: S,
        argbest: i64,
        n_best: u64,
        count: u64,
    ) -> Result<Self, EngineError> {
        let valid = if argbest == -1 {
            best == S::MASK && n_best == 0 && count == 0
        } else {
            argbest >= 0 && n_best > 0 && n_best <= count
        };
        if !valid {
            return Err(EngineError::Parameter(
                "invalid rowstats partial summary".into(),
            ));
        }
        Ok(Self {
            best,
            argbest,
            n_best,
            count,
        })
    }

    pub fn observe(&mut self, row: usize, partner: usize, value: S) -> Result<(), EngineError> {
        let partner = i64::try_from(partner).map_err(|_| {
            EngineError::Parameter(format!("rowstats partner index exceeds int64 in row {row}"))
        })?;
        self.count = self.count.checked_add(1).ok_or_else(|| {
            EngineError::Overflow(format!(
                "rowstats intermediate count exceeds uint64 in row {row}"
            ))
        })?;
        if self.argbest == -1 || value < self.best {
            self.best = value;
            self.argbest = partner;
            self.n_best = 1;
        } else if value == self.best {
            self.n_best = self.n_best.checked_add(1).ok_or_else(|| {
                EngineError::Overflow(format!(
                    "rowstats intermediate n_best exceeds uint64 in row {row}"
                ))
            })?;
            self.argbest = self.argbest.min(partner);
        }
        Ok(())
    }

    /// `observe` of partners `first + k` with values `values[k]` that pass the
    /// pruning cutoff `prune` and then the sink cutoff `cutoff` (`rect_score`
    /// of each, None observes nothing), in one merge of the run's own state:
    /// the merge rule makes it equal to observing them in turn. Returns how
    /// many values pass `prune` (the run's `full` items), from the same sweep.
    pub(crate) fn observe_run(
        &mut self,
        row: usize,
        first: usize,
        values: &[Option<S>],
        prune: Option<S>,
        cutoff: Option<S>,
    ) -> Result<usize, EngineError> {
        // The run's partners are increasing: its last one checks them all.
        let last = first + values.len().saturating_sub(1);
        i64::try_from(last).map_err(|_| {
            EngineError::Parameter(format!("rowstats partner index exceeds int64 in row {row}"))
        })?;
        let mut full = 0usize;
        let mut run = Self::default();
        for (partner, &value) in (first..).zip(values) {
            let Some(value) = crate::kernel::rect_score(value, prune) else {
                continue;
            };
            full += 1;
            if cutoff.is_some_and(|limit| value > limit) {
                continue;
            }
            // At most `values.len()` observations: no overflow.
            run.count += 1;
            if run.argbest == -1 || value < run.best {
                run.best = value;
                run.argbest = partner as i64;
                run.n_best = 1;
            } else if value == run.best {
                // The earlier partner stays the smaller one.
                run.n_best += 1;
            }
        }
        self.merge(row, run)?;
        Ok(full)
    }

    /// Inputs have disjoint partner sets; merge order cannot change the answer.
    pub fn merge(&mut self, row: usize, other: Self) -> Result<(), EngineError> {
        self.count = self.count.checked_add(other.count).ok_or_else(|| {
            EngineError::Overflow(format!(
                "rowstats intermediate count exceeds uint64 in row {row}"
            ))
        })?;
        if other.argbest == -1 {
            return Ok(());
        }
        if self.argbest == -1 || other.best < self.best {
            self.best = other.best;
            self.argbest = other.argbest;
            self.n_best = other.n_best;
        } else if other.best == self.best {
            self.n_best = self.n_best.checked_add(other.n_best).ok_or_else(|| {
                EngineError::Overflow(format!(
                    "rowstats intermediate n_best exceeds uint64 in row {row}"
                ))
            })?;
            self.argbest = self.argbest.min(other.argbest);
        }
        Ok(())
    }
}

/// Each row's exact row-term sum with the map to its reported value.
pub type RowValues<'f> = (Vec<ExactSum>, &'f dyn Fn(&ExactSum) -> f64);

impl<S: Score> RowStatsColumns<S> {
    pub fn from_states(states: Vec<RowState<S>>, include_count: bool) -> Result<Self, EngineError> {
        Self::with_row_values(states, include_count, None)
    }

    /// As `from_states`, with each row's exact row-term sum (`sums`, one per state) mapped
    /// to its reported value by `row_value` (NaN for a row without partners).
    pub fn with_row_values(
        states: Vec<RowState<S>>,
        include_count: bool,
        row_value: Option<RowValues<'_>>,
    ) -> Result<Self, EngineError> {
        use crate::detail::try_reserve_exact;
        // Every output column is reserved fallibly, in this order, before the
        // per-row loop and its dtype checks (host-allocation-failures.md A5).
        let rows = states.len();
        let mut result = Self {
            best_dist: Vec::new(),
            argbest: Vec::new(),
            n_best: Vec::new(),
            count: include_count.then(Vec::new),
            row_value: row_value.as_ref().map(|_| Vec::new()),
        };
        try_reserve_exact(&mut result.best_dist, rows, "rowstats best_dist column")?;
        try_reserve_exact(&mut result.argbest, rows, "rowstats argbest column")?;
        try_reserve_exact(&mut result.n_best, rows, "rowstats n_best column")?;
        if let Some(count) = result.count.as_mut() {
            try_reserve_exact(count, rows, "rowstats count column")?;
        }
        if let Some(values) = result.row_value.as_mut() {
            try_reserve_exact(values, rows, "rowstats row_value column")?;
        }
        let (sums, value) = match row_value {
            Some((sums, value)) => (sums, Some(value)),
            None => (Vec::new(), None),
        };
        for (row, state) in states.into_iter().enumerate() {
            let n_best = i32::try_from(state.n_best).map_err(|_| {
                EngineError::Overflow(format!("rowstats n_best exceeds int32 in row {row}"))
            })?;
            if let (Some(values), Some(value)) = (result.row_value.as_mut(), value) {
                values.push(if state.count == 0 {
                    f64::NAN
                } else {
                    value(&sums[row])
                });
            }
            result.best_dist.push(state.best);
            result.argbest.push(state.argbest);
            result.n_best.push(n_best);
            if let Some(count) = result.count.as_mut() {
                count.push(i64::try_from(state.count).map_err(|_| {
                    EngineError::Overflow(format!("rowstats count exceeds int64 in row {row}"))
                })?);
            }
        }
        Ok(result)
    }
}

#[derive(Default, Debug)]
pub struct Counts {
    pub certified: usize,
    pub pruned: usize,
    pub full: usize,
}

pub struct RunOutput<S: Score = i32> {
    pub columns: Columns<S>,
    pub shape: (usize, usize),
    pub counts: Counts,
    pub threads: usize,
    pub n_illegal_seqs: usize,
    pub validation_scans: [usize; 2],
    pub topk_replay: Option<ReplayCounts>,
}
