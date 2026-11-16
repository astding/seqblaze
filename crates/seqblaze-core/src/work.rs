//! Canonical pair sources. Implicit spaces never allocate a pair list.
use crate::validate::EngineError;
mod index;
pub use index::{IndexArray, IndexInfo};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Orientation {
    AsGiven,
    Both,
}

#[derive(Debug)]
pub enum PairKind {
    Zip,
    Index {
        i: IndexArray,
        j: IndexArray,
    },
    Within {
        self_pairs: bool,
    },
    WithinOriented {
        self_pairs: bool,
        orientation: Orientation,
    },
    Cross,
    OneToMany,
}

#[derive(Debug)]
pub struct WorkSource {
    pub kind: PairKind,
    pub shape: (usize, usize),
    pub n_pairs: usize,
}

impl WorkSource {
    pub fn new(kind: PairKind, n_a: usize, n_b: usize) -> Result<Self, EngineError> {
        if matches!(kind, PairKind::WithinOriented { .. }) && n_a != n_b {
            return Err(EngineError::Pairing(
                "oriented within requires a square table shape".into(),
            ));
        }
        let count: u128 = match &kind {
            PairKind::Zip => {
                if n_a != n_b {
                    return Err(EngineError::Pairing(format!(
                        "zip lengths differ: {n_a} and {n_b}"
                    )));
                }
                n_a as u128
            }
            PairKind::Index { i, j } => {
                if i.len() != j.len() {
                    return Err(EngineError::Pairing(format!(
                        "index lengths differ: {} and {}",
                        i.len(),
                        j.len()
                    )));
                }
                for k in 0..i.len() {
                    let (i, j) = (i.value(k), j.value(k));
                    if i < 0 || j < 0 || i as u128 >= n_a as u128 || j as u128 >= n_b as u128 {
                        return Err(EngineError::Pairing(format!(
                            "index pair {k} ({i}, {j}) is out of bounds for ({n_a}, {n_b})"
                        )));
                    }
                }
                i.len() as u128
            }
            PairKind::Within { self_pairs }
            | PairKind::WithinOriented {
                self_pairs,
                orientation: Orientation::AsGiven,
            } => {
                let n = n_a as u128;
                if *self_pairs {
                    n * (n + 1) / 2
                } else {
                    n * n.saturating_sub(1) / 2
                }
            }
            PairKind::WithinOriented {
                self_pairs,
                orientation: Orientation::Both,
            } => {
                let n = n_a as u128;
                if *self_pairs {
                    n * n
                } else {
                    n * n.saturating_sub(1)
                }
            }
            PairKind::Cross => n_a as u128 * n_b as u128,
            PairKind::OneToMany => {
                if n_a != 1 {
                    return Err(EngineError::Pairing(format!(
                        "one_to_many needs one query, received {n_a}"
                    )));
                }
                n_b as u128
            }
        };
        let n_pairs = usize::try_from(count)
            .map_err(|_| EngineError::Overflow("pair space exceeds platform index limit".into()))?;
        Ok(Self {
            kind,
            shape: (n_a, n_b),
            n_pairs,
        })
    }

    /// Both symmetric and directed within domains exclude self from row reducers.
    pub fn is_within(&self) -> bool {
        matches!(
            self.kind,
            PairKind::Within { .. } | PairKind::WithinOriented { .. }
        )
    }

    /// Only the original symmetric within source permits reverse-value reuse.
    pub fn mirrors_within(&self) -> bool {
        matches!(self.kind, PairKind::Within { .. })
    }

    /// Explicit-list partner deduplication metadata; implicit spaces need no mask.
    /// A refused reservation is a DeviceError that publishes nothing.
    pub fn first_occurrences(&self) -> Result<Option<Vec<bool>>, EngineError> {
        let PairKind::Index { i, j } = &self.kind else {
            return Ok(None);
        };
        // Both buffers are reserved fallibly, in this order, before either is
        // written (host-allocation-failures.md A1).
        let (mut order, mut first) = (Vec::<usize>::new(), Vec::<bool>::new());
        crate::detail::try_reserve_exact(&mut order, i.len(), "index pair order")?;
        crate::detail::try_reserve_exact(&mut first, i.len(), "index first-occurrence mask")?;
        order.extend(0..i.len());
        order.sort_unstable_by_key(|&k| (i.get(k), j.get(k), k));
        first.resize(i.len(), false);
        let mut previous = None;
        for k in order {
            let pair = (i.get(k), j.get(k));
            if previous != Some(pair) {
                first[k] = true;
                previous = Some(pair);
            }
        }
        Ok(Some(first))
    }

    /// Upper-triangle row-major position of canonical work ID `k < n_pairs`.
    fn triangular_position(&self, k: usize, self_pairs: bool) -> (usize, usize) {
        let n = self.shape.0 as u128;
        // Use an exact formula with no intermediate signed subtraction.
        let prefix = |r: usize| {
            if self_pairs {
                r as u128 * (2 * n - r as u128 + 1) / 2
            } else {
                r as u128 * (2 * n - r as u128 - 1) / 2
            }
        };
        let (mut lo, mut hi) = (0, self.shape.0);
        while lo + 1 < hi {
            let mid = lo + (hi - lo) / 2;
            if prefix(mid) <= k as u128 {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        (
            lo,
            lo + usize::from(!self_pairs) + (k as u128 - prefix(lo)) as usize,
        )
    }

    /// Upper-triangular (within) work IDs `[start, end)` as row segments
    /// `(i, j_first, k_first)`: row `i` starts at column `j_first` with work ID
    /// `k_first`, and continues to the end of the row or of the range. None for
    /// other pair kinds. Starts at `range`'s exact position; O(rows) output.
    pub fn row_segments(&self, start: usize, end: usize) -> Option<Vec<(usize, usize, usize)>> {
        Some(self.row_segment_iter(start, end)?.collect())
    }

    /// The segments of [`Self::row_segments`] as an allocation-free iterator,
    /// for callers that reserve their own buffers fallibly.
    pub fn row_segment_iter(
        &self,
        start: usize,
        end: usize,
    ) -> Option<impl Iterator<Item = (usize, usize, usize)> + '_> {
        let self_pairs = self.triangular()?;
        let end = end.min(self.n_pairs);
        let (mut i, mut j) = if start < end {
            self.triangular_position(start, self_pairs)
        } else {
            (0, 0)
        };
        let mut k = start;
        Some(std::iter::from_fn(move || {
            if k >= end {
                return None;
            }
            let segment = (i, j, k);
            k += self.shape.0 - j;
            i += 1;
            j = i + usize::from(!self_pairs);
            Some(segment)
        }))
    }

    /// Work IDs `[start, end)` of the implicit kinds (cross, one-to-many and
    /// both within kinds) as row spans `(i, j0, j1)`, in work-ID order: the
    /// range's items of row `i` are its cells (i, j) with j0 <= j < j1, except
    /// (i, i) when [`Self::skips_diagonal`]. None for zip and index lists.
    pub fn row_spans(
        &self,
        start: usize,
        end: usize,
    ) -> Option<impl Iterator<Item = (usize, usize, usize)> + '_> {
        // Some(self_pairs) for the triangle, else the row width.
        let layout = match self.triangular() {
            Some(self_pairs) => Ok(self_pairs),
            None => Err(self.row_width()?),
        };
        let skip = self.skips_diagonal();
        let end = end.min(self.n_pairs);
        // The row and column (position in the row) of item k.
        let (mut i, mut c) = match layout {
            _ if start >= end => (0, 0),
            Ok(self_pairs) => self.triangular_position(start, self_pairs),
            Err(width) => (start / width, start % width),
        };
        let mut k = start;
        Some(std::iter::from_fn(move || {
            if k >= end {
                return None;
            }
            let span = match layout {
                Ok(self_pairs) => {
                    let count = (self.shape.0 - c).min(end - k);
                    let span = (i, c, c + count);
                    k += count;
                    c = i + 1 + usize::from(!self_pairs);
                    span
                }
                Err(width) => {
                    let count = (width - c).min(end - k);
                    // Without self pairs, position p of row i is column p + [p >= i].
                    let column = |p: usize| p + usize::from(skip && p >= i);
                    let span = (i, column(c), column(c + count - 1) + 1);
                    k += count;
                    c = 0;
                    span
                }
            };
            i += 1;
            Some(span)
        }))
    }

    /// Whether a row span of [`Self::row_spans`] may cover its diagonal cell,
    /// which is no item: oriented within both ways without self pairs.
    pub fn skips_diagonal(&self) -> bool {
        matches!(
            self.kind,
            PairKind::WithinOriented {
                self_pairs: false,
                orientation: Orientation::Both,
            }
        )
    }

    /// Items per source row of the rectangular kinds (cross, one-to-many,
    /// oriented within both ways), whose work ID k is row k / width and
    /// column k % width; None for the other kinds.
    fn row_width(&self) -> Option<usize> {
        match self.kind {
            PairKind::Cross | PairKind::OneToMany => Some(self.shape.1),
            PairKind::WithinOriented {
                self_pairs,
                orientation: Orientation::Both,
            } => Some(self.shape.0.saturating_sub(usize::from(!self_pairs))),
            _ => None,
        }
    }

    fn triangular(&self) -> Option<bool> {
        match self.kind {
            PairKind::Within { self_pairs }
            | PairKind::WithinOriented {
                self_pairs,
                orientation: Orientation::AsGiven,
            } => Some(self_pairs),
            _ => None,
        }
    }

    /// Source row (first-table index) of canonical work ID `k < n_pairs`.
    pub fn row_of(&self, k: usize) -> usize {
        match &self.kind {
            PairKind::Zip => k,
            PairKind::Index { i, .. } => i.get(k),
            PairKind::Cross | PairKind::OneToMany => k / self.shape.1,
            PairKind::WithinOriented {
                self_pairs,
                orientation: Orientation::Both,
            } => k / (self.shape.0 - usize::from(!*self_pairs)),
            PairKind::Within { self_pairs }
            | PairKind::WithinOriented {
                self_pairs,
                orientation: Orientation::AsGiven,
            } => self.triangular_position(k, *self_pairs).0,
        }
    }

    /// `(first_row, n_rows)` of the source rows that canonical work IDs
    /// `start..end` contribute to, for shapes whose source row never decreases
    /// along work order and never mirrors (zip, cross, one-to-many, oriented
    /// within). None for index lists and symmetric within, whose contributions
    /// may reach any row. An empty range gives `(0, 0)`.
    pub fn row_interval(&self, start: usize, end: usize) -> Option<(usize, usize)> {
        if matches!(self.kind, PairKind::Index { .. } | PairKind::Within { .. }) {
            return None;
        }
        if start >= end {
            return Some((0, 0));
        }
        let first = self.row_of(start);
        Some((first, self.row_of(end - 1) + 1 - first))
    }

    pub fn range(&self, start: usize, end: usize) -> PairIter<'_> {
        let (mut i, mut j) = (0, 0);
        if start < self.n_pairs {
            if let Some(self_pairs) = self.triangular() {
                (i, j) = self.triangular_position(start, self_pairs);
            } else if let Some(width) = self.row_width() {
                // Row and column of the first item; `next` steps them on.
                (i, j) = (start / width, start % width);
            }
        }
        PairIter {
            work: self,
            next: start,
            end,
            i,
            j,
        }
    }
}

/// Pairs of a work-ID range in canonical order. The triangular and
/// rectangular kinds carry the current row and column from item to item.
pub struct PairIter<'a> {
    work: &'a WorkSource,
    next: usize,
    end: usize,
    i: usize,
    j: usize,
}
impl PairIter<'_> {
    /// Next column of a rectangular kind with `width` items per row.
    #[inline]
    fn step_column(&mut self, width: usize) {
        self.j += 1;
        if self.j == width {
            self.i += 1;
            self.j = 0;
        }
    }
}
impl Iterator for PairIter<'_> {
    type Item = (usize, usize);
    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.next >= self.end {
            return None;
        }
        let k = self.next;
        self.next += 1;
        Some(match &self.work.kind {
            PairKind::Zip => (k, k),
            PairKind::Index { i, j } => (i.get(k), j.get(k)),
            PairKind::Cross | PairKind::OneToMany => {
                let result = (self.i, self.j);
                self.step_column(self.work.shape.1);
                result
            }
            PairKind::WithinOriented {
                self_pairs,
                orientation: Orientation::Both,
            } => {
                let (i, column) = (self.i, self.j);
                self.step_column(self.work.shape.0 - usize::from(!*self_pairs));
                // Row i skips its own diagonal without self pairs.
                (i, column + usize::from(!*self_pairs && column >= i))
            }
            PairKind::Within { self_pairs }
            | PairKind::WithinOriented {
                self_pairs,
                orientation: Orientation::AsGiven,
            } => {
                let result = (self.i, self.j);
                self.j += 1;
                if self.j == self.work.shape.1 {
                    self.i += 1;
                    self.j = self.i + usize::from(!*self_pairs);
                }
                result
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn shapes(n: usize) -> Vec<WorkSource> {
        let mut result = vec![
            WorkSource::new(PairKind::Zip, n, n).unwrap(),
            WorkSource::new(PairKind::Cross, n, n + 2).unwrap(),
            WorkSource::new(PairKind::OneToMany, 1, n).unwrap(),
        ];
        for self_pairs in [false, true] {
            result.push(WorkSource::new(PairKind::Within { self_pairs }, n, n).unwrap());
            for orientation in [Orientation::AsGiven, Orientation::Both] {
                let kind = PairKind::WithinOriented {
                    self_pairs,
                    orientation,
                };
                result.push(WorkSource::new(kind, n, n).unwrap());
            }
        }
        result
    }

    /// Every sub-range yields the same pairs as the matching slice of the
    /// whole range, and each pair is the closed form of its work ID.
    #[test]
    fn every_sub_range_equals_the_slice_of_the_whole_range() {
        for n in [0, 1, 2, 5] {
            for work in shapes(n) {
                let pairs: Vec<_> = work.range(0, work.n_pairs).collect();
                assert_eq!(pairs.len(), work.n_pairs);
                for (k, &(i, j)) in pairs.iter().enumerate() {
                    let expected = match work.kind {
                        PairKind::Zip => (k, k),
                        PairKind::Cross | PairKind::OneToMany => {
                            (k / work.shape.1, k % work.shape.1)
                        }
                        PairKind::WithinOriented {
                            self_pairs,
                            orientation: Orientation::Both,
                        } => {
                            let width = work.shape.0 - usize::from(!self_pairs);
                            let (row, column) = (k / width, k % width);
                            (row, column + usize::from(!self_pairs && column >= row))
                        }
                        _ => (i, j),
                    };
                    assert_eq!((i, j), expected, "{:?} k={k}", work.kind);
                }
                for start in 0..=work.n_pairs {
                    for end in start..=work.n_pairs {
                        let part: Vec<_> = work.range(start, end).collect();
                        assert_eq!(part, pairs[start..end], "{:?} {start}..{end}", work.kind);
                    }
                }
            }
        }
    }

    /// Row spans expand to exactly the pairs of every sub-range, in order.
    #[test]
    fn row_spans_cover_every_sub_range() {
        for n in [0, 1, 2, 5] {
            for work in shapes(n) {
                let pairs: Vec<_> = work.range(0, work.n_pairs).collect();
                for start in 0..=work.n_pairs {
                    for end in start..=work.n_pairs {
                        let Some(spans) = work.row_spans(start, end) else {
                            assert!(matches!(work.kind, PairKind::Zip));
                            continue;
                        };
                        let skip = work.skips_diagonal();
                        let cells: Vec<_> = spans
                            .flat_map(|(i, j0, j1)| {
                                (j0..j1)
                                    .filter(move |&j| !(skip && j == i))
                                    .map(move |j| (i, j))
                            })
                            .collect();
                        assert_eq!(cells, pairs[start..end], "{:?} {start}..{end}", work.kind);
                    }
                }
            }
        }
    }

    #[test]
    fn row_of_and_row_interval_follow_canonical_ranges() {
        for n in [0, 1, 2, 5] {
            for work in shapes(n) {
                let pairs: Vec<_> = work.range(0, work.n_pairs).collect();
                for (k, &(i, _)) in pairs.iter().enumerate() {
                    assert_eq!(work.row_of(k), i, "{:?} k={k}", work.kind);
                }
                for start in 0..=work.n_pairs {
                    for end in start..=work.n_pairs {
                        let interval = work.row_interval(start, end);
                        if matches!(work.kind, PairKind::Within { .. }) {
                            assert_eq!(interval, None);
                            continue;
                        }
                        let (first, len) = interval.unwrap();
                        for &(i, _) in &pairs[start..end] {
                            assert!(first <= i && i < first + len, "{:?}", work.kind);
                        }
                        if start == end {
                            assert_eq!((first, len), (0, 0));
                        } else {
                            assert_eq!(first, pairs[start].0);
                            assert_eq!(first + len - 1, pairs[end - 1].0);
                        }
                    }
                }
            }
        }
    }
}
