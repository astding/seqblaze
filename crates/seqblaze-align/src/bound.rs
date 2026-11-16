//! Certified score upper bound (spec §6): every `M` column (a_i, b_j) scores
//! at most r_{a_i}(b) = max(0, max over symbols y of b of s(a_i, y)), each
//! residue is in at most one `M` column and gaps cost >= 0, so
//! UB = min(sum_x c_x(a) r_x(b), sum_y c_y(b) r_y(a)) bounds `score`.
//!
//! A row's symbol-class tally (its distinct classes and their counts) is
//! either counted per pair from the bytes ([`upper_bound`], the `evaluate`
//! path) or cached once per table row ([`TableTally`], the engine's per-pair
//! prefilter). Both feed the same [`combine`], so they give the same UB.
use std::ops::ControlFlow;

use seqblaze_core::{
    cancel,
    detail::{try_reserve, try_reserve_exact},
    sched::{self, Workers},
    seq::{ByteView, SeqTable},
    validate::EngineError,
};

use crate::scoring::Tables;
use crate::scratch::Counts;

/// A row's bytes, visited in order as blocks.
pub(crate) trait Blocks: Copy {
    fn each(self, visit: impl FnMut(&[u8]));
}

impl Blocks for &[u8] {
    fn each(self, mut visit: impl FnMut(&[u8])) {
        visit(self);
    }
}

/// Borrowed NumPy rows are read with relaxed loads into a stack block; no
/// row copy is made.
impl Blocks for ByteView<'_> {
    fn each(self, mut visit: impl FnMut(&[u8])) {
        let _ = self.for_each_block(|_, block| {
            visit(block);
            ControlFlow::<()>::Continue(())
        });
    }
}

/// One row's tally: distinct symbol classes and their counts (each > 0), in
/// parallel. An empty row has an empty tally.
#[derive(Clone, Copy)]
pub(crate) struct Tally<'r> {
    pub classes: &'r [u8],
    pub counts: &'r [u64],
}

/// Count `seq` into `present` (distinct classes, first-occurrence order) and
/// `values` (their counts). `scratch` is all zero on entry and on return.
fn tally(
    seq: impl Blocks,
    class: &[u8; 256],
    scratch: &mut [u64; 256],
    present: &mut Vec<u8>,
    values: &mut Vec<u64>,
) {
    present.clear();
    values.clear();
    seq.each(|block| {
        for &x in block {
            let c = class[x as usize];
            let slot = &mut scratch[c as usize];
            if *slot == 0 {
                // At most 256 classes: `present` never grows past its capacity.
                present.push(c);
            }
            // u64: a slice holds at most isize::MAX bytes, so this cannot wrap.
            *slot += 1;
        }
    });
    for &c in present.iter() {
        values.push(scratch[c as usize]);
        scratch[c as usize] = 0;
    }
}

/// count * r for r >= 0, saturating: a saturated term only raises UB, which
/// keeps it a certified upper bound.
fn term(count: u64, r: i64) -> i64 {
    i64::try_from(count).unwrap_or(i64::MAX).saturating_mul(r)
}

/// UB in fixed point from two tallies, O(|A| |B|) over their distinct classes;
/// 0 if either row is empty. Every term is >= 0, so the saturating sums do not
/// depend on the order of the classes.
pub(crate) fn combine(a: Tally<'_>, b: Tally<'_>, t: &Tables) -> i64 {
    let k = t.n_classes;
    let mut sum_a = 0i64;
    for (&x, &count) in a.classes.iter().zip(a.counts) {
        let row = &t.class_score[x as usize * k..][..k];
        let r = b
            .classes
            .iter()
            .map(|&y| i64::from(row[y as usize]))
            .fold(0, i64::max);
        sum_a = sum_a.saturating_add(term(count, r));
    }
    let mut sum_b = 0i64;
    for (&y, &count) in b.classes.iter().zip(b.counts) {
        let r = a
            .classes
            .iter()
            .map(|&x| i64::from(t.class_score[x as usize * k + y as usize]))
            .fold(0, i64::max);
        sum_b = sum_b.saturating_add(term(count, r));
    }
    sum_a.min(sum_b)
}

/// UB of one pair counted from its bytes: O(m + n + |A| |B|).
pub(crate) fn upper_bound(a: impl Blocks, b: impl Blocks, t: &Tables, counts: &mut Counts) -> i64 {
    let Counts {
        scratch,
        present_a,
        values_a,
        present_b,
        values_b,
    } = counts;
    tally(a, &t.class, scratch, present_a, values_a);
    tally(b, &t.class, scratch, present_b, values_b);
    combine(
        Tally {
            classes: present_a,
            counts: values_a,
        },
        Tally {
            classes: present_b,
            counts: values_b,
        },
        t,
    )
}

/// Per-row tallies of one table, rows in order: row i owns entries
/// `ends[i - 1]..ends[i]` (from 0 for row 0) of `classes` and `counts`.
/// Size O(sum over rows of min(len, classes)), never O(rows * classes).
pub(crate) struct TableTally {
    ends: Vec<usize>,
    classes: Vec<u8>,
    counts: Vec<u64>,
}

const TALLY: &str = "align bound tallies";

impl TableTally {
    fn empty() -> Self {
        Self {
            ends: Vec::new(),
            classes: Vec::new(),
            counts: Vec::new(),
        }
    }

    pub(crate) fn rows(&self) -> usize {
        self.ends.len()
    }

    pub(crate) fn row(&self, i: usize) -> Tally<'_> {
        let start = if i == 0 { 0 } else { self.ends[i - 1] };
        let end = self.ends[i];
        Tally {
            classes: &self.classes[start..end],
            counts: &self.counts[start..end],
        }
    }

    /// Tally every row of `table`, in canonical row ranges over the call's
    /// workers, then concatenate the ranges in order. Storage is reserved
    /// fallibly (DeviceError).
    pub(crate) fn build(
        table: &SeqTable,
        t: &Tables,
        workers: Workers,
    ) -> Result<Self, EngineError> {
        let rows = table.len();
        let limit = workers.count(rows, table.mean_len()).max(1);
        // Bound preparation of this table's rows (cancellation-progress.md §4.2).
        cancel::stage(cancel::Stage::Bound, rows);
        let parts = sched::map_ranges(rows, limit, &|start, end| {
            let mut ticker = cancel::Ticker::new(cancel::Site::Bound);
            let mut part = Self::empty();
            try_reserve_exact(&mut part.ends, end - start, TALLY)?;
            let mut scratch = [0u64; 256];
            let (mut present, mut values) = (Vec::with_capacity(256), Vec::with_capacity(256));
            for i in start..end {
                tally(
                    table.get(i),
                    &t.class,
                    &mut scratch,
                    &mut present,
                    &mut values,
                );
                try_reserve(&mut part.classes, present.len(), TALLY)?;
                try_reserve(&mut part.counts, values.len(), TALLY)?;
                part.classes.extend_from_slice(&present);
                part.counts.extend_from_slice(&values);
                part.ends.push(part.classes.len());
                ticker.tick()?;
            }
            ticker.finish()?;
            Ok(part)
        })?;
        let total: usize = parts.iter().map(|part| part.classes.len()).sum();
        let mut whole = Self::empty();
        try_reserve_exact(&mut whole.ends, rows, TALLY)?;
        try_reserve_exact(&mut whole.classes, total, TALLY)?;
        try_reserve_exact(&mut whole.counts, total, TALLY)?;
        for part in parts {
            let base = whole.classes.len();
            whole.ends.extend(part.ends.iter().map(|&end| base + end));
            whole.classes.extend_from_slice(&part.classes);
            whole.counts.extend_from_slice(&part.counts);
        }
        Ok(whole)
    }
}

/// delta of `global` with end gaps (spec §6): at least d = |m - n| gap columns
/// of one kind in k >= 1 runs cost at least go + (d - 1) min(go, ge).
pub(crate) fn end_gap_delta(m: usize, n: usize, go: i64, ge: i64) -> i64 {
    let d = m.abs_diff(n);
    if d == 0 {
        0
    } else {
        go + (d as i64 - 1) * go.min(ge)
    }
}
