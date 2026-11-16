//! CPU pair-distance port: one selecting score plus optional detail fields.
use crate::{
    detail::{DetailColumns, DetailField, DetailRequest},
    plan::BoundAvailability,
    sched::Workers,
    seq::{ByteView, SeqTable, Stager, TableRows},
    sink::rowsum::{Dyadic, ExactSum},
    validate::{self, Alphabet, EngineError},
    work::WorkSource,
};

/// The engine's selecting key: a totally ordered integer whose order and equality are
/// those of the call's exposed primary (smaller is better). Integer distances and
/// keys use `i32`; a float64 primary uses `i64`, the order-preserving integer image of
/// the exposed value (duplex-pf.md §9.2). Sinks compare and tie keys only, so ties are
/// bitwise equality of the exposed values.
pub trait Score: Copy + Ord + Default + Send + Sync + std::fmt::Debug + 'static {
    /// Never a real score: the mask of a row that does not pass, and the empty-row value.
    const MASK: Self;
    /// The smallest key (a lower bound that rejects nothing).
    const MIN: Self;
    /// An int64 cutoff in this key type, if it fits.
    fn from_i64(value: i64) -> Option<Self>;
}

impl Score for i32 {
    const MASK: Self = i32::MAX;
    const MIN: Self = i32::MIN;
    fn from_i64(value: i64) -> Option<Self> {
        i32::try_from(value).ok()
    }
}

impl Score for i64 {
    const MASK: Self = i64::MAX;
    const MIN: Self = i64::MIN;
    fn from_i64(value: i64) -> Option<Self> {
        Some(value)
    }
}

/// The outcome of a kernel's per-pair bound against a pass threshold θ
/// (duplex-mfe-bound.md §1): pruned (bound > θ, strict), certified with the exact key,
/// or open (the exact evaluation runs).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BoundOutcome<S> {
    Pruned,
    Certified(S),
    Open,
}

/// A pair kernel with selecting key `S` (`i32` unless stated).
pub trait PairDistanceKernel<S: Score = i32>: Sync {
    /// Declare a usable intrinsic bound explicitly; no pack certificate is inferred.
    fn bound_availability(&self) -> BoundAvailability;

    fn name(&self) -> &'static str {
        "hamming"
    }
    fn validate_pairs(
        &self,
        a: &SeqTable,
        b: &SeqTable,
        work: &WorkSource,
    ) -> Result<(), EngineError> {
        validate::equal_lengths(a, b, work)
    }
    fn sequence_length_limit(&self) -> Option<usize> {
        Some(validate::HAMMING_MAX_LENGTH)
    }
    /// Per-input length caps `[a, b]`, checked over every sequence of every
    /// table after the alphabet scan. Defaults to `sequence_length_limit`.
    fn length_caps(&self) -> [Option<usize>; 2] {
        let limit = self.sequence_length_limit();
        [limit, limit]
    }
    /// The error for sequence `index` of `input` whose `length` exceeds `limit`.
    fn length_error(
        &self,
        input: &'static str,
        index: usize,
        length: usize,
        limit: usize,
    ) -> EngineError {
        EngineError::Sequence {
            message: format!(
                "{} input {input} sequence {index} length {length} exceeds length limit {limit}; nothing was computed",
                self.name()
            ),
            input,
            index,
            position: None,
            byte: None,
            n_bad: 1,
            bad_indices: vec![index],
        }
    }
    /// Legal bytes of this call (D48 §1.1). The default is upper-case ACGT,
    /// where `check_bases=False` turns errors into a warning count.
    fn alphabet(&self) -> Alphabet {
        Alphabet::Acgt
    }
    /// Whole-table sequence checks that run after the alphabet scan and the
    /// per-input caps (validation step 3): minimum lengths, joint limits over
    /// both tables. The default checks nothing.
    fn sequence_preflight(&self, _a: &SeqTable, _b: &SeqTable) -> Result<(), EngineError> {
        Ok(())
    }
    /// Pair-shape sequence checks that run last in validation step 3, after
    /// `sequence_preflight`: rules on the two strands of a pair (for example a
    /// joint length rule) that name the first offending pair of the call shape.
    /// Implicit pair lists must not be visited pair by pair. The default checks
    /// nothing.
    fn pair_preflight(
        &self,
        _a: &SeqTable,
        _b: &SeqTable,
        _work: &WorkSource,
    ) -> Result<(), EngineError> {
        Ok(())
    }
    /// Prepare per-row facts `lower_bound` needs, after full validation. The
    /// engine calls this only when it evaluates per-pair bounds
    /// (`plan::BoundPrefilter::PerPair`); allocation failures are errors.
    fn prepare_bound(
        &self,
        a: &SeqTable,
        b: &SeqTable,
        workers: Workers,
    ) -> Result<(), EngineError>;
    /// A valid lower bound on the score of pair (i, j); it only ever rejects
    /// pairs whose exact score also exceeds the cutoff.
    fn lower_bound(&self, a: &SeqTable, b: &SeqTable, i: usize, j: usize) -> S;
    /// Whether this call carries an outcome bound (prune / certify / open per pair,
    /// duplex-mfe-bound.md §6.7). Consulted only under the bounded strategy.
    fn outcome_bound(&self) -> bool {
        false
    }
    /// The outcome of pair (a, b) against `threshold` (an engine key; `None`: certify
    /// only). A pruned pair's exact key exceeds the threshold; a certified key is the
    /// exact key. The default resolves nothing.
    fn bound_outcome(
        &self,
        _a: ByteView<'_>,
        _b: ByteView<'_>,
        _threshold: Option<S>,
    ) -> Result<BoundOutcome<S>, EngineError> {
        Ok(BoundOutcome::Open)
    }
    /// The weak-bound rule's weight e and cost-table entry ĉ (units of 1/64) of one work
    /// item whose strands have lengths `n_a` and `n_b` in the evaluated orientation
    /// (duplex-mfe-bound.md §6.6 step 2, §6.7). The default (0, 0) never uses a bound.
    fn bound_cost(&self, _n_a: usize, _n_b: usize) -> (u64, u32) {
        (0, 0)
    }
    /// Estimated work of one pair of `a` × `b`, in the byte-comparison units of
    /// `sched::Workers::count`, which sizes `Grain::Auto` stages with it.
    /// `with_details` adds the requested detail fields of a kept pair.
    /// Scheduling only: it never changes an output. The default is the linear
    /// model of a byte-compare kernel such as Hamming: a's mean length.
    fn pair_cost(&self, a: &SeqTable, _b: &SeqTable, _with_details: bool) -> usize {
        a.mean_len()
    }
    /// `Ok(None)` means the cutoff was exceeded; no rejected score is exposed.
    /// An error (a scratch buffer that cannot be allocated) stops the call,
    /// which then publishes nothing.
    fn evaluate(
        &self,
        a: ByteView<'_>,
        b: ByteView<'_>,
        cutoff: Option<S>,
    ) -> Result<Option<S>, EngineError>;
    /// Pairs the engine hands `evaluate_many` at once; 1 (the default) keeps the
    /// pair loops on `evaluate`. Scheduling only: it never changes an output.
    fn batch_width(&self) -> usize {
        1
    }
    /// Whether the pair loop at a batch width of 1 hands `evaluate` the rows
    /// of a borrowed packed `S<n>` table that it reads in order (the row read
    /// last or the next one) as staged copies: slices of the same bytes, read
    /// a run of rows at a time ([`Stager`]). For a kernel that reads slices
    /// faster than borrowed rows. Scheduling only: it never changes an output.
    fn stages_rows(&self) -> bool {
        false
    }
    /// `evaluate(a, b, cutoff)` of every `(a, b, cutoff)`, appended to `out` in
    /// order. A kernel may override it to score several pairs together (for
    /// example one pair per vector lane); each entry must equal its pair's
    /// `evaluate`. An error stops the call, as for `evaluate`.
    fn evaluate_many(
        &self,
        pairs: &[(ByteView<'_>, ByteView<'_>, Option<S>)],
        out: &mut Vec<Option<S>>,
    ) -> Result<(), EngineError> {
        for &(a, b, cutoff) in pairs {
            out.push(self.evaluate(a, b, cutoff)?);
        }
        Ok(())
    }
    /// Cells per rectangle that a plain-score pass of a call over tables `a`
    /// and `b` of an implicit pair kind (Matrix, Tiles, Hits, RowStats and
    /// both TopK passes) hands `evaluate_rect` at once; 0 (the default) keeps
    /// it on `evaluate_many` blocks of work items (and Tiles on `evaluate`).
    /// `bounded`: the blocks would test the per-pair bound (`lower_bound`)
    /// before scoring, which rectangles skip. A kernel that returns more than
    /// 0 also promises an exact cutoff test: `evaluate(a, b, Some(c))` is
    /// `evaluate(a, b, None)` when that score is at most c, and None
    /// otherwise, so the engine applies each item's cutoff to its rectangle
    /// score ([`rect_score`]). Scheduling only: it never changes an output.
    fn rect_cells(&self, _a: &SeqTable, _b: &SeqTable, _bounded: bool) -> usize {
        0
    }
    /// `evaluate(x, y, cutoff)` of every cell of a rectangle, for each row
    /// view `x` of `rows` and column view `y` of `cols`, appended to `out` row
    /// by row (`rows.len() * cols.len()` entries). A kernel may override it to
    /// share work between the cells of a row or a column; each entry must
    /// equal its pair's `evaluate`. The default scores each row of cells
    /// through `evaluate_many`. An error stops the call, as for `evaluate`.
    fn evaluate_rect(
        &self,
        rows: &[ByteView<'_>],
        cols: &[ByteView<'_>],
        cutoff: Option<S>,
        out: &mut Vec<Option<S>>,
    ) -> Result<(), EngineError> {
        let mut pairs = Vec::new();
        crate::detail::try_reserve_exact(&mut pairs, cols.len(), "pair batch")?;
        for &x in rows {
            pairs.clear();
            pairs.extend(cols.iter().map(|&y| (x, y, cutoff)));
            self.evaluate_many(&pairs, out)?;
        }
        Ok(())
    }

    /// Detail fields this kernel can emit besides the selecting score.
    fn detail_fields(&self) -> &'static [DetailField] {
        &[]
    }
    /// Detail fields requested for this call (part of the kernel's call
    /// configuration). An empty request keeps every sink on its score-only path.
    fn detail_request(&self) -> DetailRequest {
        DetailRequest::NONE
    }
    /// Score, as `evaluate(a, b, prune)`. If the score is kept
    /// (`keep.is_none_or(|limit| score <= limit)`), append exactly one row of
    /// the requested detail fields to `out`; otherwise append nothing. A kernel
    /// may override this to share work between the score and the details.
    /// Errors are allocation failures, as for `evaluate`.
    fn evaluate_detailed(
        &self,
        a: ByteView<'_>,
        b: ByteView<'_>,
        prune: Option<S>,
        keep: Option<S>,
        out: &mut DetailColumns,
    ) -> Result<Option<S>, EngineError> {
        let Some(score) = self.evaluate(a, b, prune)? else {
            return Ok(None);
        };
        if keep.is_none_or(|limit| score <= limit) {
            self.details(a, b, score, out)?;
        }
        Ok(Some(score))
    }
    /// Whether this call's `evaluate_detailed` takes a kept pair's details from
    /// the work of its own score pass, so that `evaluate` followed by `details`
    /// would repeat that work. The CPU Hits sink then evaluates details inline
    /// in its pair stage; otherwise it scores first and computes the kept
    /// entries' details in a `details` stage of their own (same rows, same
    /// bytes; cancellation-progress.md §13). Scheduling only: never an output.
    fn fused_details(&self) -> bool {
        false
    }
    /// Append one row of the requested detail fields of a kept pair whose exact
    /// score is `score`. It must not depend on the strategy or thread count;
    /// every buffer it needs is reserved fallibly (a DeviceError, never an abort).
    fn details(
        &self,
        _a: ByteView<'_>,
        _b: ByteView<'_>,
        _score: S,
        _out: &mut DetailColumns,
    ) -> Result<(), EngineError> {
        Ok(())
    }
    /// Whether the CPU Each sink keeps the per-pair key column (`Columns::dist`). A kernel
    /// whose published Each columns all come from its detail rows returns false (`tm` on
    /// Pairs, tm.md §13.11: `tm_c` is F10's bits from the detail row and the selecting key is
    /// never published), and its Each output then holds no `dist`. Honoured only while the
    /// call carries a detail request; it never changes `passed` or the detail rows.
    fn each_keys(&self) -> bool {
        true
    }
    /// Whether RowStats also sums a per-pair row term exactly over each row's distinct
    /// passing partners (an exact per-row superaccumulator, D47; duplex-pf.md §9.4).
    fn row_terms(&self) -> bool {
        false
    }
    /// The exact score and the pair's row term, as `evaluate(a, b, None)`; called only
    /// when `row_terms()` holds.
    fn evaluate_term(
        &self,
        _a: ByteView<'_>,
        _b: ByteView<'_>,
    ) -> Result<(S, Dyadic), EngineError> {
        Err(EngineError::Parameter(format!(
            "{} has no row terms",
            self.name()
        )))
    }
    /// The reported value of a row with at least one summed partner, from the exact sum.
    fn row_value(&self, _sum: &ExactSum) -> f64 {
        f64::NAN
    }
    /// The exact key of a pair from its row term, by the map `evaluate` itself applies to
    /// that term (duplex-pf-cuda.md §7.1, QKP3): a backend that computes only the term (K3
    /// on CUDA computes Ẑ) gets the key from the kernel's one copy of that map. Called
    /// only for kernels whose term determines the key; the default is an internal error.
    fn key_of_term(&self, _term: Dyadic) -> Result<S, EngineError> {
        Err(EngineError::Parameter(format!(
            "internal error: {} has no map from a row term to its key",
            self.name()
        )))
    }
}

/// Receives consecutive chunks of exact distances as `(first, values, details)`,
/// where `first` is the work ID (or request position) of `values[0]` and
/// `details`, when the caller asked for them, holds one detail row per value.
pub type DistanceVisit<'v, S = i32> =
    dyn FnMut(usize, &[S], Option<&DetailColumns>) -> Result<(), EngineError> + 'v;

/// Receives consecutive chunks of exact distances with their row terms (the term
/// channel of [`BatchDistance::range_terms`]): `(first, values, terms)`, one term per
/// value, each equal to `evaluate_term`'s.
pub type TermVisit<'v, S = i32> = dyn FnMut(usize, &[S], &[Dyadic]) -> Result<(), EngineError> + 'v;

/// How a caller uses cell `(i, j)` of a rectangle: `None` if it discards the
/// cell (a hole), else `Some(canonical)`, true when this cell is the work
/// item's canonical visit (false for a replayed mirror). A backend may skip a
/// discarded cell's work; its slot still gets a value, which is never read.
pub type CellClass<'c> = dyn Fn(usize, usize) -> Option<bool> + 'c;

/// Batch-distance port (docs/wiki/gpu/cuda-backend.md, levenshtein-cuda.md
/// §6): exact distances for many pairs at once, from a backend bound to one
/// call's validated tables. TopK, Matrix and Tiles call it instead of the
/// per-pair `evaluate` when the backend is not the CPU; every value equals
/// `evaluate(…, None)`. A backend that `delivers_details` also returns the
/// call's requested detail fields beside each distance, equal to
/// `evaluate_detailed(…, None, None, …)`; otherwise the caller computes them
/// on the host through the kernel's `details` port.
pub trait BatchDistance<S: Score = i32> {
    /// Whether chunks can carry every requested detail field of the call.
    fn delivers_details(&self) -> bool;
    /// Work IDs `[start, end)` of `work`, delivered in increasing, consecutive
    /// chunks of bounded size; with `details` (only when `delivers_details`),
    /// each chunk carries its detail rows.
    fn range(
        &mut self,
        work: &WorkSource,
        start: usize,
        end: usize,
        details: bool,
        visit: &mut DistanceVisit<'_, S>,
    ) -> Result<(), EngineError>;
    /// Explicit `(i, j)` pairs, appended to `out` in input order, and their
    /// detail rows to `details` when given.
    fn pairs(
        &mut self,
        pairs: &[(usize, usize)],
        out: &mut Vec<S>,
        details: Option<&mut DetailColumns>,
    ) -> Result<(), EngineError>;
    /// Every cell of rows `r0..r1` × columns `c0..c1`, appended to `out` (and
    /// detail rows to `details`) in row-major order, for pairs of any lengths.
    fn rect(
        &mut self,
        rows: (usize, usize),
        cols: (usize, usize),
        class: &CellClass<'_>,
        out: &mut Vec<S>,
        details: Option<&mut DetailColumns>,
    ) -> Result<(), EngineError>;
    /// The term channel (duplex-pf-cuda.md §8 item 5): as `range` without detail rows,
    /// each chunk also carrying every pair's row term, equal to `evaluate_term`'s. For
    /// RowStats of a `row_terms` kernel; the default is an internal error.
    fn range_terms(
        &mut self,
        _work: &WorkSource,
        _start: usize,
        _end: usize,
        _visit: &mut TermVisit<'_, S>,
    ) -> Result<(), EngineError> {
        Err(EngineError::Parameter(
            "internal error: this batch-distance backend delivers no row terms".into(),
        ))
    }
}

/// CPU implementation of the batch-distance port through `evaluate(…, None)`,
/// or `evaluate_detailed(…, None, None, …)` when detail rows are asked for.
pub struct CpuBatch<'t, K: ?Sized> {
    pub kernel: &'t K,
    pub a: &'t SeqTable,
    pub b: &'t SeqTable,
}

const CPU_CHUNK: usize = 4096;

impl<K: ?Sized> CpuBatch<'_, K> {
    fn exact<S: Score>(
        &self,
        i: usize,
        j: usize,
        details: Option<&mut DetailColumns>,
    ) -> Result<S, EngineError>
    where
        K: PairDistanceKernel<S>,
    {
        let (a, b) = (self.a.get(i), self.b.get(j));
        match details {
            None => self.kernel.evaluate(a, b, None)?,
            Some(out) => self.kernel.evaluate_detailed(a, b, None, None, out)?,
        }
        .ok_or_else(|| EngineError::Parameter("batch distance requires an exact score".into()))
    }

    fn rows<S: Score>(&self, details: bool) -> Option<DetailColumns>
    where
        K: PairDistanceKernel<S>,
    {
        details
            .then(|| DetailColumns::new(self.kernel.detail_fields(), self.kernel.detail_request()))
    }
}

impl<S: Score, K: PairDistanceKernel<S> + ?Sized> BatchDistance<S> for CpuBatch<'_, K> {
    /// Numeric requests only, as a device backend's: list and string fields
    /// are computed by the caller.
    fn delivers_details(&self) -> bool {
        let request = self.kernel.detail_request();
        !request.is_empty()
            && self
                .kernel
                .detail_fields()
                .iter()
                .enumerate()
                .all(|(index, field)| !request.contains(index) || field.kind.is_numeric())
    }

    fn range(
        &mut self,
        work: &WorkSource,
        start: usize,
        end: usize,
        details: bool,
        visit: &mut DistanceVisit<'_, S>,
    ) -> Result<(), EngineError> {
        let mut values = Vec::with_capacity(CPU_CHUNK.min(end.saturating_sub(start)));
        let mut rows = self.rows::<S>(details);
        let mut first = start;
        for (offset, (i, j)) in work.range(start, end).enumerate() {
            values.push(self.exact(i, j, rows.as_mut())?);
            if values.len() == CPU_CHUNK {
                visit(first, &values, rows.as_ref())?;
                first = start + offset + 1;
                values.clear();
                if let Some(rows) = rows.as_mut() {
                    rows.clear();
                }
            }
        }
        if !values.is_empty() {
            visit(first, &values, rows.as_ref())?;
        }
        Ok(())
    }

    fn pairs(
        &mut self,
        pairs: &[(usize, usize)],
        out: &mut Vec<S>,
        mut details: Option<&mut DetailColumns>,
    ) -> Result<(), EngineError> {
        out.reserve(pairs.len());
        for &(i, j) in pairs {
            out.push(self.exact(i, j, details.as_deref_mut())?);
        }
        Ok(())
    }

    fn rect(
        &mut self,
        rows: (usize, usize),
        cols: (usize, usize),
        _class: &CellClass<'_>,
        out: &mut Vec<S>,
        mut details: Option<&mut DetailColumns>,
    ) -> Result<(), EngineError> {
        for i in rows.0..rows.1 {
            for j in cols.0..cols.1 {
                out.push(self.exact(i, j, details.as_deref_mut())?);
            }
        }
        Ok(())
    }

    /// Through `evaluate_term`, in chunks of at most `CPU_CHUNK` pairs.
    fn range_terms(
        &mut self,
        work: &WorkSource,
        start: usize,
        end: usize,
        visit: &mut TermVisit<'_, S>,
    ) -> Result<(), EngineError> {
        let len = CPU_CHUNK.min(end.saturating_sub(start));
        let (mut values, mut terms) = (Vec::with_capacity(len), Vec::with_capacity(len));
        let mut first = start;
        for (offset, (i, j)) in work.range(start, end).enumerate() {
            let (value, term) = self.kernel.evaluate_term(self.a.get(i), self.b.get(j))?;
            values.push(value);
            terms.push(term);
            if values.len() == CPU_CHUNK {
                visit(first, &values, &terms)?;
                first = start + offset + 1;
                values.clear();
                terms.clear();
            }
        }
        if !values.is_empty() {
            visit(first, &values, &terms)?;
        }
        Ok(())
    }
}

/// What a pair loop does with one work item (see [`for_each_scored`]).
pub(crate) enum Plan<S> {
    /// Not evaluated, not scored (`Outcome::Skipped`).
    Skip,
    /// Rejected without evaluation, as a `None` score (the per-pair bound).
    Reject,
    /// `evaluate(a, b, cutoff)`.
    Score(Option<S>),
}

/// The result of one work item, in work-ID order.
pub(crate) enum Outcome<S> {
    Skipped,
    Scored(Option<S>),
}

/// Run work items `start..end` in work-ID order: `plan` decides each item,
/// the planned evaluations go through `evaluate_many` a `batch_width` block at
/// a time (or `evaluate` one by one when the width is 1), and `visit` receives
/// every item's outcome in order. The outcomes equal those of planning and
/// evaluating item by item.
///
/// In a block, the rows of a borrowed packed `S<n>` table are staged first
/// ([`Stager`]): a run of consecutive rows is one relaxed span copy, a row
/// equal to the previous one reuses its copy, and the kernel reads every
/// staged row as a slice of the same bytes.
pub(crate) fn for_each_scored<S: Score, K: PairDistanceKernel<S> + ?Sized>(
    kernel: &K,
    (a, b): (&SeqTable, &SeqTable),
    work: &WorkSource,
    (start, end): (usize, usize),
    mut plan: impl FnMut(usize, (usize, usize)) -> Plan<S>,
    mut visit: impl FnMut(usize, (usize, usize), Outcome<S>) -> Result<(), EngineError>,
) -> Result<(), EngineError> {
    let width = kernel.batch_width().min(end - start);
    let (a, b) = (a.rows(), b.rows());
    if width <= 1 {
        let items = (work, start, end);
        let stages = kernel.stages_rows();
        // Each side's form is chosen once, so the item loop holds one form
        // per side.
        return match (Ahead::of(stages, a)?, Ahead::of(stages, b)?) {
            (Some(mut x), Some(mut y)) => score_items(kernel, items, &mut x, &mut y, plan, visit),
            (Some(mut x), None) => score_items(kernel, items, &mut x, &mut { b }, plan, visit),
            (None, Some(mut y)) => score_items(kernel, items, &mut { a }, &mut y, plan, visit),
            (None, None) => score_items(kernel, items, &mut { a }, &mut { b }, plan, visit),
        };
    }
    const BATCH: &str = "pair batch";
    // A block's pairs and plans side by side; `scored` holds the block
    // positions of its `Score` items, and stays empty while every item of the
    // block is one (the common case: no per-pair bound).
    let (mut pairs, mut plans, mut scored, mut scores) =
        (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    crate::detail::try_reserve_exact(&mut pairs, width, BATCH)?;
    crate::detail::try_reserve_exact(&mut plans, width, BATCH)?;
    crate::detail::try_reserve_exact(&mut scored, width, BATCH)?;
    crate::detail::try_reserve_exact(&mut scores, width, BATCH)?;
    // At most one whole-row copy per scored item and staged side.
    let stagers = [a.stager(), b.stager()];
    let row_bytes: usize = stagers.iter().flatten().map(|stager| stager.width()).sum();
    let mut arena = Vec::new();
    let mut staged = Vec::new();
    if row_bytes > 0 {
        // row_bytes <= 2 * STAGE_WIDTH, and width is a batch width.
        let size = row_bytes.checked_mul(width).ok_or_else(|| {
            crate::detail::allocation_error("pair batch staging", "size overflows usize")
        })?;
        crate::detail::try_reserve_exact(&mut arena, size, "pair batch staging")?;
        arena.resize(size, 0u8);
        crate::detail::try_reserve_exact(&mut staged, width, "pair batch staging")?;
    }
    let mut range = work.range(start, end);
    let mut base = 0;
    while base < end - start {
        pairs.clear();
        plans.clear();
        scored.clear();
        scores.clear();
        pairs.extend(range.by_ref().take(width));
        let mut every = true;
        for (k, &pair) in pairs.iter().enumerate() {
            let plan = plan(base + k, pair);
            every &= matches!(plan, Plan::Score(_));
            plans.push(plan);
        }
        if !every {
            scored.extend((0..pairs.len()).filter(|&k| matches!(plans[k], Plan::Score(_))));
        }
        let count = if every { pairs.len() } else { scored.len() };
        let position = |n: usize| if every { n } else { scored[n] };
        if row_bytes > 0 {
            staged.clear();
            staged.resize(count, [0usize; 2]);
        }
        let mut used = 0;
        for (side, stager) in stagers.iter().enumerate() {
            if let Some(stager) = stager {
                let rows = (0..count).map(|n| {
                    let pair = pairs[position(n)];
                    if side == 0 {
                        pair.0
                    } else {
                        pair.1
                    }
                });
                used = stage(*stager, rows, side, &mut staged, &mut arena, used);
            }
        }
        // Per block: the views borrow this block's staged copies. Each side's
        // form (staged copy or the table's row) is chosen once per block, so
        // the per-item loop holds one form per side.
        let mut views = Vec::new();
        crate::detail::try_reserve_exact(&mut views, count, BATCH)?;
        let item = |n: usize| {
            let k = position(n);
            let &Plan::Score(cutoff) = &plans[k] else {
                unreachable!("scored items only")
            };
            (pairs[k], cutoff)
        };
        let arena = &arena[..];
        let copy = |stager: Stager<'_>, at: usize, row: usize| {
            ByteView::from_slice(&arena[at..at + stager.len(row)])
        };
        match stagers {
            [Some(x), Some(y)] => views.extend((0..count).map(|n| {
                let ((i, j), cutoff) = item(n);
                (copy(x, staged[n][0], i), copy(y, staged[n][1], j), cutoff)
            })),
            [Some(x), None] => views.extend((0..count).map(|n| {
                let ((i, j), cutoff) = item(n);
                (copy(x, staged[n][0], i), b.get(j), cutoff)
            })),
            [None, Some(y)] => views.extend((0..count).map(|n| {
                let ((i, j), cutoff) = item(n);
                (a.get(i), copy(y, staged[n][1], j), cutoff)
            })),
            [None, None] => views.extend((0..count).map(|n| {
                let ((i, j), cutoff) = item(n);
                (a.get(i), b.get(j), cutoff)
            })),
        }
        kernel.evaluate_many(&views, &mut scores)?;
        drop(views);
        let mut next = scores.iter();
        for (k, (&pair, plan)) in pairs.iter().zip(&plans).enumerate() {
            let outcome = match plan {
                Plan::Skip => Outcome::Skipped,
                Plan::Reject => Outcome::Scored(None),
                Plan::Score(_) => Outcome::Scored(*next.next().expect("one score per plan")),
            };
            visit(base + k, pair, outcome)?;
        }
        base += pairs.len();
    }
    Ok(())
}

/// The item loop of [`for_each_scored`] at a batch width of 1, over rows
/// read through `x` (side a) and `y` (side b).
#[inline(always)]
fn score_items<S: Score, K: PairDistanceKernel<S> + ?Sized>(
    kernel: &K,
    (work, start, end): (&WorkSource, usize, usize),
    x: &mut impl Side,
    y: &mut impl Side,
    mut plan: impl FnMut(usize, (usize, usize)) -> Plan<S>,
    mut visit: impl FnMut(usize, (usize, usize), Outcome<S>) -> Result<(), EngineError>,
) -> Result<(), EngineError> {
    for (offset, pair) in work.range(start, end).enumerate() {
        let outcome = match plan(offset, pair) {
            Plan::Skip => Outcome::Skipped,
            Plan::Reject => Outcome::Scored(None),
            Plan::Score(cutoff) => {
                Outcome::Scored(kernel.evaluate(x.row(pair.0), y.row(pair.1), cutoff)?)
            }
        };
        visit(offset, pair, outcome)?;
    }
    Ok(())
}

/// Row access of one side of [`score_items`].
trait Side {
    fn row(&mut self, row: usize) -> ByteView<'_>;
}

impl Side for TableRows<'_> {
    #[inline(always)]
    fn row(&mut self, row: usize) -> ByteView<'_> {
        self.get(row)
    }
}

/// Bytes of each [`Ahead`] copy (rows of at most [`crate::seq::STAGE_WIDTH`]
/// bytes: at least 4 rows).
const AHEAD_BYTES: usize = 4096;

/// The rows of one borrowed packed side staged ahead for a loop that reads
/// them in order: a row outside the staged run that is the row read last or
/// the next one restarts the run there, copying up to `capacity` rows with
/// one span read ([`Stager::copy_rows`]); any other row is read in place.
struct Ahead<'t> {
    stager: Stager<'t>,
    rows: TableRows<'t>,
    capacity: usize,
    buffer: Vec<u8>,
    /// The staged run: rows first..first + count, at buffer[0..].
    first: usize,
    count: usize,
    /// The row read last (usize::MAX before the first).
    last: usize,
}

impl<'t> Ahead<'t> {
    /// The staged form of a side when the kernel `stages` rows and the side
    /// is a borrowed packed view ([`TableRows::stager`]).
    fn of(stages: bool, rows: TableRows<'t>) -> Result<Option<Self>, EngineError> {
        match rows.stager() {
            Some(stager) if stages => Self::new(stager, rows).map(Some),
            _ => Ok(None),
        }
    }

    fn new(stager: Stager<'t>, rows: TableRows<'t>) -> Result<Self, EngineError> {
        let width = stager.width();
        let capacity = (AHEAD_BYTES / width).max(1);
        let mut buffer = Vec::new();
        // width <= STAGE_WIDTH: no overflow.
        crate::detail::try_reserve_exact(&mut buffer, capacity * width, "pair batch staging")?;
        buffer.resize(capacity * width, 0u8);
        Ok(Self {
            stager,
            rows,
            capacity,
            buffer,
            first: 0,
            count: 0,
            last: usize::MAX,
        })
    }

    /// Restart the run at `row` if the loop reads in order there.
    #[cold]
    #[inline(never)]
    fn restage(&mut self, row: usize) -> bool {
        if row != self.last && row != self.last.wrapping_add(1) {
            return false;
        }
        let width = self.stager.width();
        let count = self.capacity.min(self.stager.rows() - row);
        self.stager
            .copy_rows(row, &mut self.buffer[..count * width]);
        (self.first, self.count) = (row, count);
        true
    }
}

impl Side for Ahead<'_> {
    #[inline(always)]
    fn row(&mut self, row: usize) -> ByteView<'_> {
        let staged = row.wrapping_sub(self.first) < self.count || self.restage(row);
        self.last = row;
        if !staged {
            return self.rows.get(row);
        }
        let at = (row - self.first) * self.stager.width();
        ByteView::from_slice(&self.buffer[at..at + self.stager.len(row)])
    }
}

/// Rows per rectangle of [`for_each_tiled`].
const RECT_ROWS: usize = 64;

/// Cells per rectangle for a plain-score pass of a call when the kernel
/// scores rectangles (`PairDistanceKernel::rect_cells`, with `bounded` as
/// there) and the pair kind is implicit; None keeps it on blocks of work items.
pub(crate) fn rect_cells<S: Score, K: PairDistanceKernel<S> + ?Sized>(
    kernel: &K,
    work: &WorkSource,
    (a, b): (&SeqTable, &SeqTable),
    bounded: bool,
) -> Option<usize> {
    // Implicit kinds only: explicit kinds have no row spans.
    let cells = work
        .row_spans(0, 0)
        .map(|_| kernel.rect_cells(a, b, bounded))?;
    (cells > 0).then_some(cells)
}

/// `evaluate(a, b, cutoff)` of an item from its rectangle score
/// `evaluate(a, b, wider)` under a rectangle cutoff `wider` that is None or at
/// least `cutoff`, by the exact cutoff test of a rectangle kernel
/// (`PairDistanceKernel::rect_cells`): a score at most `cutoff` is in both,
/// and any other is None in the result. The per-pair bound rejects only items
/// whose score is above the cutoff, which this gives the same None, so the
/// rectangle passes do not evaluate it.
#[inline]
pub(crate) fn rect_score<S: Score>(score: Option<S>, cutoff: Option<S>) -> Option<S> {
    score.filter(|&value| cutoff.is_none_or(|limit| value <= limit))
}

/// Run work items `start..end` of an implicit pair kind
/// ([`WorkSource::row_spans`]) a rectangle at a time through
/// `evaluate_rect`, and `visit(i, j, scores)` every item with its score
/// (`evaluate(a_i, b_j, c)`), a run of items of one row at a time: items
/// (i, j + k) get `scores[k]`. Rectangle by rectangle, and in each row by
/// row in column order. A rectangle holds the rows of up to [`RECT_ROWS`]
/// consecutive spans that reach a run of columns, at most `cells` cells; a
/// cell of it that is no item of the range is computed and discarded. Its
/// cutoff c is `cutoff(spans, (c0, c1))` of its row spans (cut to nothing
/// but their own rows' cells) and its columns `c0..c1`.
pub(crate) fn for_each_tiled<S: Score, K: PairDistanceKernel<S> + ?Sized>(
    kernel: &K,
    tables: (&SeqTable, &SeqTable),
    work: &WorkSource,
    (start, end): (usize, usize),
    cells: usize,
    cutoff: impl FnMut(&[(usize, usize, usize)], (usize, usize)) -> Option<S>,
    visit: impl FnMut(usize, usize, &[Option<S>]) -> Result<(), EngineError>,
) -> Result<(), EngineError> {
    let spans = work.row_spans(start, end).expect("an implicit pair kind");
    let skip = work.skips_diagonal();
    for_each_span_tiled(kernel, tables, spans, skip, cells, cutoff, visit)
}

/// [`for_each_tiled`] over row spans `(i, j0, j1)` in increasing row order:
/// the items of row i are its cells (i, j) with j0 <= j < j1, except (i, i)
/// when `skip` (the diagonal) is set.
pub(crate) fn for_each_span_tiled<S: Score, K: PairDistanceKernel<S> + ?Sized>(
    kernel: &K,
    (a, b): (&SeqTable, &SeqTable),
    spans: impl Iterator<Item = (usize, usize, usize)>,
    skip: bool,
    cells: usize,
    mut cutoff: impl FnMut(&[(usize, usize, usize)], (usize, usize)) -> Option<S>,
    mut visit: impl FnMut(usize, usize, &[Option<S>]) -> Result<(), EngineError>,
) -> Result<(), EngineError> {
    const WHAT: &str = "pair rectangle";
    let mut spans = spans.peekable();
    let (a, b) = (a.rows(), b.rows());
    let cells = cells.max(1);
    let (mut block, mut chosen): (Vec<(usize, usize, usize)>, Vec<_>) = (Vec::new(), Vec::new());
    let (mut rows, mut cols, mut scores) = (Vec::new(), Vec::new(), Vec::new());
    crate::detail::try_reserve_exact(&mut block, RECT_ROWS, WHAT)?;
    crate::detail::try_reserve_exact(&mut chosen, RECT_ROWS, WHAT)?;
    crate::detail::try_reserve_exact(&mut rows, RECT_ROWS, WHAT)?;
    crate::detail::try_reserve_exact(&mut cols, cells, WHAT)?;
    crate::detail::try_reserve_exact(&mut scores, cells, WHAT)?;
    while spans.peek().is_some() {
        block.clear();
        block.extend(spans.by_ref().take(RECT_ROWS));
        let first = block.iter().map(|span| span.1).min().unwrap_or(0);
        let last = block.iter().map(|span| span.2).max().unwrap_or(0);
        let width = (cells / block.len()).max(1);
        let mut c0 = first;
        while c0 < last {
            let c1 = c0 + width.min(last - c0);
            chosen.clear();
            chosen.extend(
                block
                    .iter()
                    .copied()
                    .filter(|&(_, j0, j1)| j0 < c1 && c0 < j1),
            );
            rows.clear();
            rows.extend(chosen.iter().map(|&(i, ..)| a.get(i)));
            cols.clear();
            cols.extend((c0..c1).map(|j| b.get(j)));
            scores.clear();
            let cutoff = cutoff(&chosen, (c0, c1));
            kernel.evaluate_rect(&rows, &cols, cutoff, &mut scores)?;
            assert_eq!(scores.len(), rows.len() * cols.len(), "one score per cell");
            for (&(i, j0, j1), line) in chosen.iter().zip(scores.chunks_exact(c1 - c0)) {
                let (j0, j1) = (j0.max(c0), j1.min(c1));
                if skip && (j0..j1).contains(&i) {
                    // The diagonal cell splits the run.
                    visit(i, j0, &line[j0 - c0..i - c0])?;
                    visit(i, i + 1, &line[i + 1 - c0..j1 - c0])?;
                } else {
                    visit(i, j0, &line[j0 - c0..j1 - c0])?;
                }
            }
            c0 = c1;
        }
    }
    Ok(())
}

/// Stage one side's rows (in scored-item order) into `arena[used..]` and set
/// `staged[k][side]` to the offset of item k's copy; returns the new `used`.
/// A row inside the current run (the previous row, or any row already in it)
/// reuses that copy; the next row extends the run; any other row starts a
/// new one. Each run is copied once, when it ends.
fn stage(
    stager: Stager<'_>,
    rows: impl Iterator<Item = usize>,
    side: usize,
    staged: &mut [[usize; 2]],
    arena: &mut [u8],
    mut used: usize,
) -> usize {
    let width = stager.width();
    // The current run: rows first..first + count, copied to arena[at..].
    let mut run: Option<(usize, usize, usize)> = None;
    for (k, row) in rows.enumerate() {
        staged[k][side] = match run {
            Some((first, count, at)) if (first..first + count).contains(&row) => {
                at + (row - first) * width
            }
            Some((first, count, at)) if row == first + count => {
                run = Some((first, count + 1, at));
                used += width;
                at + count * width
            }
            _ => {
                if let Some((first, count, at)) = run {
                    stager.copy_rows(first, &mut arena[at..at + count * width]);
                }
                run = Some((row, 1, used));
                used += width;
                used - width
            }
        };
    }
    if let Some((first, count, at)) = run {
        stager.copy_rows(first, &mut arena[at..at + count * width]);
    }
    used
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        batch::SeqBatch,
        work::{Orientation, PairKind},
    };

    /// Scores `len(a) * 1000 + len(b)`: rows of distinct lengths name their cell.
    struct Lengths;
    impl PairDistanceKernel for Lengths {
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
            _: Option<i32>,
        ) -> Result<Option<i32>, EngineError> {
            Ok(Some((a.len() * 1000 + b.len()) as i32))
        }
    }

    /// Scores a row pair by its bytes (position-weighted sums), and reads
    /// staged rows when `stages` is set.
    struct Content {
        stages: bool,
    }
    impl PairDistanceKernel for Content {
        fn bound_availability(&self) -> BoundAvailability {
            BoundAvailability::Unavailable
        }
        fn prepare_bound(&self, _: &SeqTable, _: &SeqTable, _: Workers) -> Result<(), EngineError> {
            Ok(())
        }
        fn lower_bound(&self, _: &SeqTable, _: &SeqTable, _: usize, _: usize) -> i32 {
            0
        }
        fn stages_rows(&self) -> bool {
            self.stages
        }
        fn evaluate(
            &self,
            a: ByteView<'_>,
            b: ByteView<'_>,
            _: Option<i32>,
        ) -> Result<Option<i32>, EngineError> {
            let sum = |view: ByteView<'_>| {
                (view.iter().enumerate()).fold(view.len() as i32, |sum, (k, byte)| {
                    sum.wrapping_mul(31)
                        .wrapping_add((k as i32 + 1) * i32::from(byte))
                })
            };
            Ok(Some(sum(a).wrapping_mul(7).wrapping_add(sum(b))))
        }
    }

    /// Staged rows ([`Ahead`]) score as the rows read in place, for every pair
    /// kind, range and row width (one row or many per staged run), with
    /// padded rows, repeated rows and out-of-order rows.
    #[test]
    fn staged_rows_score_as_rows_in_place() {
        let fixed = |n: usize, width: usize, seed: usize| {
            let mut data = vec![0u8; n * width];
            for row in 0..n {
                let len = if row % 3 == 1 {
                    width - 1 - row % width.min(5)
                } else {
                    width
                };
                for k in 0..len {
                    data[row * width + k] = b"ACGT"[(row * 7 + k * 3 + seed) % 4];
                }
            }
            let address = data.as_ptr() as usize;
            // SAFETY: the boxed Vec owns the described rows and is never mutated.
            let table = unsafe {
                SeqTable::from_fixed_bytes(address, n, width, width as isize, Box::new(data))
            }
            .unwrap();
            table.validate_lengths().unwrap();
            table
        };
        for width in [6, 1000] {
            let n = 11;
            let (a, b) = (fixed(n, width, 0), fixed(n + 2, width, 1));
            let i = vec![0, 1, 2, 2, 2, 7, 3, 4, 5, 9, 10, 10, 0];
            let j = vec![12, 0, 1, 2, 3, 4, 4, 11, 5, 6, 7, 8, 9];
            let index = PairKind::Index {
                i: crate::work::IndexArray::from(i),
                j: crate::work::IndexArray::from(j),
            };
            let works = [
                WorkSource::new(PairKind::Zip, n, n).unwrap(),
                WorkSource::new(index, n, n + 2).unwrap(),
                WorkSource::new(PairKind::OneToMany, 1, n + 2).unwrap(),
                WorkSource::new(PairKind::Within { self_pairs: true }, n, n).unwrap(),
                WorkSource::new(PairKind::Cross, n, n + 2).unwrap(),
            ];
            for work in &works {
                // One-to-many reads row 0 of `a` only.
                let (x, y) = match work.kind {
                    PairKind::Within { .. } => (&a, &a),
                    _ => (&a, &b),
                };
                for (start, end) in [(0, work.n_pairs), (3, work.n_pairs - 2), (7, 9)] {
                    let scores = |stages: bool| {
                        let mut out = Vec::new();
                        for_each_scored(
                            &Content { stages },
                            (x, y),
                            work,
                            (start, end),
                            |_, _| Plan::Score(None),
                            |offset, pair, outcome| {
                                let Outcome::Scored(score) = outcome else {
                                    unreachable!("no item is skipped")
                                };
                                out.push((offset, pair, score));
                                Ok(())
                            },
                        )
                        .unwrap();
                        out
                    };
                    let staged = scores(true);
                    assert_eq!(staged.len(), end - start);
                    assert_eq!(
                        staged,
                        scores(false),
                        "{:?} {start}..{end} width {width}",
                        work.kind
                    );
                }
            }
        }
    }

    /// The tiled walk visits every item of every range once, with its score,
    /// for every implicit kind and rectangle size.
    #[test]
    fn tiled_walks_visit_every_item_once() {
        let table = |n: usize| {
            let rows: Vec<Vec<u8>> = (0..n).map(|k| vec![b'A'; k]).collect();
            SeqTable::new(SeqBatch::from_seqs(rows.iter().map(|x| &x[..])))
        };
        let n = 9;
        let mut works = vec![
            WorkSource::new(PairKind::Cross, n, n + 4).unwrap(),
            WorkSource::new(PairKind::OneToMany, 1, n).unwrap(),
        ];
        for self_pairs in [false, true] {
            works.push(WorkSource::new(PairKind::Within { self_pairs }, n, n).unwrap());
            for orientation in [Orientation::AsGiven, Orientation::Both] {
                let kind = PairKind::WithinOriented {
                    self_pairs,
                    orientation,
                };
                works.push(WorkSource::new(kind, n, n).unwrap());
            }
        }
        for work in &works {
            let (a, b) = (table(work.shape.0), table(work.shape.1));
            for cells in [1, 5, 16, 4096] {
                for (start, end) in [(0, work.n_pairs), (3, work.n_pairs - 2), (7, 8), (5, 5)] {
                    let mut seen = Vec::new();
                    for_each_tiled(
                        &Lengths,
                        (&a, &b),
                        work,
                        (start, end),
                        cells,
                        |_, _| None,
                        |i, j0, scores| {
                            for (j, &score) in (j0..).zip(scores) {
                                assert_eq!(score, Some((i * 1000 + j) as i32));
                                seen.push((i, j));
                            }
                            Ok(())
                        },
                    )
                    .unwrap();
                    seen.sort_unstable();
                    let mut want: Vec<_> = work.range(start, end).collect();
                    want.sort_unstable();
                    assert_eq!(seen, want, "{:?} cells {cells} {start}..{end}", work.kind);
                }
            }
        }
    }
}
