//! Inter-pair score pass (spec §6 and §12, "Lane score pass"): the score-only
//! forward passes of many pairs at once, one pair per vector lane.
//!
//! Each lane runs the single-word forward pass of `Work::scan_full` for its own
//! pair: the pattern a (rows) in the lane word, b scanned column by column, the
//! same Myers/Hyyrö word step with top-row delta +1 (global, prefix) or 0
//! (infix), the score at row m counted from m, and in prefix and infix its
//! running minimum over the ends 0..=n. Only the word differs: u8, u16, u32 or
//! u64 instead of u64. Carries and shifts run upward, so bits above row m never
//! reach a lower row, and bits 0..m are those of the u64 pass. So a lane's
//! `dist` is `evaluate`'s, and neither the word nor the instruction set ever
//! changes an output.
//!
//! **Words.** A lane record holds one word per lane: 128 bytes in the AVX2
//! and AVX-512 forms (128 lanes of u8, 64 of u16, 32 of u32 or 16 of u64), 64
//! bytes in the portable form. A pair takes the narrowest word with at least m
//! bits whose largest value is at least n. Every count at row m is some
//! D(m, j) <= max(m, n), so the unsigned counter never wraps.
//!
//! **Groups.** A group walks the columns of its longest b. Past its own n a
//! lane's score deltas are masked off, so its score and running minimum keep
//! their values at column n; the state words of those columns are never read.
//!
//! **Match masks.** Lane l of column j is Peq_l(b_l[j]), bit i set iff
//! a_l[i] = b_l[j], for all 256 byte values (any alphabet). The AVX2 and
//! AVX-512 forms compare the pattern with each byte of b; the portable form
//! reads per-lane match-mask rows indexed by byte value.
//!
//! **Shared-pattern groups** (spec §6 and §12, "Shared-pattern lane form").
//! When every lane scans the same text, lane l of column j is
//! Peq_l(text[j]) for one byte value in all lanes: the group forms, once per
//! lane, the record of each distinct byte of its texts (row x: lane l holds
//! Peq_l(x)), and a column is one load of its byte's record. Such a group
//! scans its texts one after another, every lane to the same n, so no column
//! is masked. Its texts come from a run of pairs with one row on a side (that
//! row; in `global`, whose distance is symmetric, the shared row may be `a`,
//! and the other rows are then the patterns) or from a Matrix rectangle (the
//! columns or rows of the rectangle). The step, words and counters are those
//! above, so the `dist` of each lane is again `evaluate`'s.
use std::{
    cell::RefCell,
    sync::atomic::{AtomicU8, Ordering},
};

use seqblaze_core::{
    detail::{allocation_error, try_reserve_exact},
    kernel::PairDistanceKernel,
    seq::{ByteView, SeqTable},
    validate::EngineError,
};

use super::{score_reduced, strip_shared, Levenshtein, Mode};

// ------------------------------------------------------------------ instruction set

/// Instruction set of the lane pass.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Isa {
    /// No lane pass: every pair through `evaluate`, one by one.
    Scalar,
    /// The lane pass on plain arrays: any CPU (the compiler may vectorize it).
    Portable,
    /// 256-bit AVX2 vectors, two per lane record.
    Avx2,
    /// 512-bit AVX-512 (F, BW) vectors; only when compiled by rustc 1.89 or
    /// later (`build.rs` sets `seqblaze_avx512`).
    Avx512,
}

impl Isa {
    const ALL: [Self; 4] = [Self::Scalar, Self::Portable, Self::Avx2, Self::Avx512];

    pub fn name(self) -> &'static str {
        match self {
            Self::Scalar => "scalar",
            Self::Portable => "portable",
            Self::Avx2 => "avx2",
            Self::Avx512 => "avx512",
        }
    }

    /// Whether this build and CPU can run it.
    pub fn available(self) -> bool {
        match self {
            Self::Scalar | Self::Portable => true,
            Self::Avx2 => avx2_detected(),
            Self::Avx512 => avx512_detected(),
        }
    }
}

fn avx2_detected() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        x86::Avx2::new().is_some()
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

fn avx512_detected() -> bool {
    #[cfg(all(target_arch = "x86_64", seqblaze_avx512))]
    {
        x86::Avx512::new().is_some()
    }
    #[cfg(not(all(target_arch = "x86_64", seqblaze_avx512)))]
    {
        false
    }
}

/// The widest available form: AVX-512, else AVX2, else plain arrays (aarch64
/// and other CPUs).
fn detected() -> Isa {
    if avx512_detected() {
        Isa::Avx512
    } else if avx2_detected() {
        Isa::Avx2
    } else {
        Isa::Portable
    }
}

/// 0: detection; k > 0: `Isa::ALL[k - 1]` forced by `set_isa`.
static FORCED: AtomicU8 = AtomicU8::new(0);

/// The instruction set in use: the detected one unless `set_isa` forced another.
pub fn isa() -> Isa {
    match FORCED.load(Ordering::Relaxed) {
        0 => detected(),
        code => Isa::ALL[usize::from(code) - 1],
    }
}

/// Names of the instruction sets this build and CPU can run.
pub fn available() -> impl Iterator<Item = &'static str> {
    Isa::ALL
        .into_iter()
        .filter(|isa| isa.available())
        .map(Isa::name)
}

/// The message for a value `shown` that names no available instruction set.
pub fn unavailable(shown: &str) -> String {
    let names: Vec<String> = available().map(|name| format!("'{name}'")).collect();
    format!(
        "levenshtein lane isa {shown} is not available on this build and CPU; use None or one of {}",
        names.join(", ")
    )
}

/// Testing: force the instruction set by name, or restore detection with
/// None. Returns the previous forced name (None when detecting). An unknown or
/// unavailable name changes nothing. Outputs never depend on it.
pub fn set_isa(name: Option<&str>) -> Result<Option<&'static str>, String> {
    let code = match name {
        None => 0,
        Some(name) => match Isa::ALL
            .iter()
            .position(|isa| isa.name() == name && isa.available())
        {
            Some(index) => index as u8 + 1,
            None => return Err(unavailable(&format!("'{name}'"))),
        },
    };
    let previous = FORCED.swap(code, Ordering::Relaxed);
    Ok((previous != 0).then(|| Isa::ALL[usize::from(previous) - 1].name()))
}

// ------------------------------------------------------------------ batches

/// Bytes of a lane record of the AVX2 and AVX-512 forms: four 256-bit or two
/// 512-bit vectors per step, independent chains for the latency-bound step.
const WIDE: usize = 128;
/// Bytes of a lane record of the portable form: plain arrays keep fewer live
/// bytes (a 128-byte record measured slower there than the scalar pass).
const NARROW: usize = 64;
/// Lane words, narrowest first: u8, u16, u32, u64.
const WORDS: usize = 4;
/// Pairs per `evaluate_many` call: four groups of the narrowest word.
const BATCH: usize = 4 * WIDE;
const BATCH_WHAT: &str = "levenshtein lane batch";
const ARENA_WHAT: &str = "levenshtein lane input copy";
const MASKS_WHAT: &str = "levenshtein lane match masks";

/// Patterns longer than this drop their shared affixes first (`strip_shared`,
/// exact): often into a narrower word than u64, or below 65 bytes into a lane.
const STRIP_ABOVE: usize = 32;

/// Fewest consecutive pairs with one row on a side, and fewest lane pairs of
/// such a run, that run in shared-pattern groups: below it a group's columns cost
/// more per pair than forming each lane's masks per column. Also the fewest
/// patterns of a rectangle that take shared-pattern groups.
const SHARED_RUN: usize = 16;
/// Cells per Matrix rectangle (`rect_cells`).
const RECT: usize = 4096;
/// Texts per shared-pattern group call of a rectangle.
const RECT_TEXTS: usize = 64;

/// The engine's batch width: 1 (no batches) when forced to `Scalar`.
pub(super) fn batch_width() -> usize {
    if isa() == Isa::Scalar {
        1
    } else {
        BATCH
    }
}

/// The lane form of a batch, with its detected instruction set.
#[derive(Clone, Copy)]
enum Form {
    Portable,
    #[cfg(target_arch = "x86_64")]
    Avx2(x86::Avx2),
    #[cfg(all(target_arch = "x86_64", seqblaze_avx512))]
    Avx512(x86::Avx512),
}

impl Form {
    /// The form of `isa` (never `Scalar`); plain arrays when the set is absent.
    fn of(isa: Isa) -> Self {
        #[cfg(all(target_arch = "x86_64", seqblaze_avx512))]
        if isa == Isa::Avx512 {
            if let Some(simd) = x86::Avx512::new() {
                return Self::Avx512(simd);
            }
        }
        #[cfg(target_arch = "x86_64")]
        if isa == Isa::Avx2 {
            if let Some(simd) = x86::Avx2::new() {
                return Self::Avx2(simd);
            }
        }
        let _ = isa;
        Self::Portable
    }

    /// Lanes per group of each word.
    fn lanes(self) -> [usize; WORDS] {
        let bytes = match self {
            Self::Portable => NARROW,
            #[cfg(target_arch = "x86_64")]
            _ => WIDE,
        };
        [bytes, bytes / 2, bytes / 4, bytes / 8]
    }
}

/// A pair waiting for a lane group: its pattern `a` and text `b` (after
/// `strip_shared` in a per-pair group; a shared run's text is its shared
/// row), its word (0..WORDS: u8, u16, u32, u64) and its index `k` in the
/// batch.
#[derive(Clone, Copy)]
struct Item<'v> {
    a: &'v [u8],
    b: &'v [u8],
    word: u8,
    k: usize,
}

/// The narrowest word for pattern length m (1..=64) and text length n.
fn word_of(m: usize, n: usize) -> Option<u8> {
    let n = n as u64;
    if m <= 8 && n <= u64::from(u8::MAX) {
        Some(0)
    } else if m <= 16 && n <= u64::from(u16::MAX) {
        Some(1)
    } else if m <= 32 && n <= u64::from(u32::MAX) {
        Some(2)
    } else if m <= 64 {
        Some(3)
    } else {
        None
    }
}

/// `dist` under the pair's cutoff, as `evaluate` reports it.
fn passes(dist: usize, cutoff: Option<i32>) -> Option<i32> {
    let dist = i32::try_from(dist).expect("the length cap bounds dist");
    cutoff.is_none_or(|limit| dist <= limit).then_some(dist)
}

/// `evaluate_many` of the kernel on the instruction set in use.
pub(super) fn score_many(
    kernel: &Levenshtein,
    pairs: &[(ByteView<'_>, ByteView<'_>, Option<i32>)],
    out: &mut Vec<Option<i32>>,
) -> Result<(), EngineError> {
    score_on(isa(), kernel, pairs, out)
}

/// Every entry equals the pair's `evaluate`:
/// - a negative cutoff gives None, as `evaluate` does before any work;
/// - a run of at least `SHARED_RUN` consecutive pairs with one row on a side
///   (`b` in every mode; `a` in `global`, whose distance is symmetric) is a
///   shared run: that row is the text of every pair, the other row the
///   pattern. m = 0 gives the §2 empty-pattern `dist` (n in global mode, else
///   0), a global pair with |m - n| above the cutoff gives None, and patterns
///   of 1-64 bytes go to shared-pattern groups when at least `SHARED_RUN` do;
/// - every other pair is admitted alone (`admit`): a pattern over
///   `STRIP_ABOVE` bytes is reduced by `strip_shared`, which keeps `dist`
///   (spec §12), then the same three cases, the lane pairs going to per-pair
///   groups, sorted into their words by counting;
/// - longer patterns take `evaluate`'s forward pass on the reduced pair.
fn score_on(
    isa: Isa,
    kernel: &Levenshtein,
    pairs: &[(ByteView<'_>, ByteView<'_>, Option<i32>)],
    out: &mut Vec<Option<i32>>,
) -> Result<(), EngineError> {
    try_reserve_exact(out, pairs.len(), BATCH_WHAT)?;
    if isa == Isa::Scalar {
        for &(a, b, cutoff) in pairs {
            out.push(kernel.evaluate(a, b, cutoff)?);
        }
        return Ok(());
    }
    let mode = kernel.mode();
    let form = Form::of(isa);
    // Borrowed NumPy rows: one relaxed copy per run of one row on a side
    // (`fresh`), in `arena` at 8-byte steps (`span`), copied a whole word at a
    // time (`ByteView::copy_padded`).
    let mut copied = 0usize;
    let mut last = [None; 2];
    for &(a, b, _) in pairs.iter().filter(|pair| live(pair.2)) {
        for (view, last) in [a, b].into_iter().zip(&mut last) {
            if fresh(view, last) {
                copied = view
                    .len()
                    .checked_next_multiple_of(8)
                    .and_then(|span| copied.checked_add(span))
                    .ok_or_else(|| allocation_error(ARENA_WHAT, "size overflows usize"))?;
            }
        }
    }
    let mut arena = Vec::new();
    if copied > 0 {
        try_reserve_exact(&mut arena, copied, ARENA_WHAT)?;
        arena.resize(copied, 0u8);
        let (mut at, mut last) = (0, [None; 2]);
        for &(a, b, _) in pairs.iter().filter(|pair| live(pair.2)) {
            for (view, last) in [a, b].into_iter().zip(&mut last) {
                if fresh(view, last) {
                    let end = at + span(view);
                    view.copy_padded(0, &mut arena[at..end]);
                    at = end;
                }
            }
        }
    }
    let mut rows = Rows {
        arena: &arena,
        at: 0,
        last: [None; 2],
    };
    let mut runs = Vec::new();
    try_reserve_exact(&mut runs, pairs.len() / SHARED_RUN + 1, BATCH_WHAT)?;
    shared_runs(pairs, mode, &mut runs);
    let mut pool = Vec::new();
    let mut order = Vec::new();
    // The lane pairs of the shared runs, run after run (pattern in `a`, the
    // shared text in `b`), and per run: its side, its range of `shared` and
    // the word counts of that range.
    let mut shared = Vec::new();
    let mut ranges = Vec::new();
    try_reserve_exact(&mut pool, pairs.len(), BATCH_WHAT)?;
    try_reserve_exact(&mut order, pairs.len(), BATCH_WHAT)?;
    if !runs.is_empty() {
        try_reserve_exact(&mut shared, pairs.len(), BATCH_WHAT)?;
        try_reserve_exact(&mut ranges, runs.len(), BATCH_WHAT)?;
    }
    let base = out.len();
    let mut pool_counts = [0usize; WORDS];
    let mut k = 0;
    for (first, end, side) in runs.iter().copied().chain([(pairs.len(), 0, 0)]) {
        // Pairs alone, then the run.
        for (k, &(a, b, cutoff)) in (k..).zip(&pairs[k..first]) {
            let value = if live(cutoff) {
                let (a, b) = (rows.get(0, a), rows.get(1, b));
                admit(a, b, k, mode, cutoff, &mut pool, &mut pool_counts)?
            } else {
                None
            };
            out.push(value);
        }
        if first == pairs.len() {
            break;
        }
        let (start, mut counts) = (shared.len(), [0usize; WORDS]);
        for (k, &(a, b, cutoff)) in (first..).zip(&pairs[first..end]) {
            let value = if live(cutoff) {
                let (a, b) = (rows.get(0, a), rows.get(1, b));
                let (pattern, text) = if side == 0 { (b, a) } else { (a, b) };
                let (m, n) = (pattern.len(), text.len());
                if m == 0 {
                    passes(if mode == Mode::Global { n } else { 0 }, cutoff)
                } else if mode == Mode::Global
                    && cutoff.is_some_and(|limit| m.abs_diff(n) > limit as usize)
                {
                    None
                } else if let Some(word) = word_of(m, n) {
                    shared.push(Item {
                        a: pattern,
                        b: text,
                        word,
                        k,
                    });
                    counts[usize::from(word)] += 1;
                    None // set by its group below
                } else {
                    admit(a, b, k, mode, cutoff, &mut pool, &mut pool_counts)?
                }
            } else {
                None
            };
            out.push(value);
        }
        ranges.push((side, start, shared.len(), counts));
        k = end;
    }
    let lanes = form.lanes();
    let mut dists = [0u32; WIDE];
    for (side, start, end, counts) in ranges {
        let run = &shared[start..end];
        if run.len() < SHARED_RUN {
            // Too few for a group of their own: admitted alone.
            for item in run {
                let (a, b) = if side == 0 {
                    (item.b, item.a)
                } else {
                    (item.a, item.b)
                };
                let cutoff = pairs[item.k].2;
                out[base + item.k] =
                    admit(a, b, item.k, mode, cutoff, &mut pool, &mut pool_counts)?;
            }
            continue;
        }
        let text = run[0].b;
        let mut seen = [false; 256];
        let mut bytes = [0u8; 256];
        let mut distinct = 0;
        for &x in text {
            if !std::mem::replace(&mut seen[usize::from(x)], true) {
                bytes[distinct] = x;
                distinct += 1;
            }
        }
        let texts = Shared {
            texts: &[text],
            bytes: &bytes[..distinct],
        };
        grouped(lanes, run, counts, &mut order, |word, members| {
            let job = (mode, run, members, &mut dists[..]);
            match word {
                0 => u8::run(form, job, Some(texts))?,
                1 => u16::run(form, job, Some(texts))?,
                2 => u32::run(form, job, Some(texts))?,
                _ => u64::run(form, job, Some(texts))?,
            }
            for (&member, &dist) in members.iter().zip(&dists) {
                let k = run[member].k;
                out[base + k] = passes(dist as usize, pairs[k].2);
            }
            Ok(())
        })?;
    }
    grouped(lanes, &pool, pool_counts, &mut order, |word, members| {
        let job = (mode, &pool[..], members, &mut dists[..]);
        match word {
            0 => u8::run(form, job, None)?,
            1 => u16::run(form, job, None)?,
            2 => u32::run(form, job, None)?,
            _ => u64::run(form, job, None)?,
        }
        for (&member, &dist) in members.iter().zip(&dists) {
            let k = pool[member].k;
            out[base + k] = passes(dist as usize, pairs[k].2);
        }
        Ok(())
    })
}

/// Cells per rectangle that a plain-score pass over tables `a` and `b` hands
/// `evaluate_rect`: 0 (none) when forced to `Scalar`, or for a `bounded` pass
/// (one whose item blocks test the engine's per-pair bound) when no table that
/// can hold the patterns (`a` in `prefix` and `infix`, either in `global`) has
/// a mean row length that fits a lane: then most cells would be admitted
/// alone, as in the blocks, but without the bound.
pub(super) fn rect_cells(mode: Mode, a: &SeqTable, b: &SeqTable, bounded: bool) -> usize {
    let fits = |table: &SeqTable| table.mean_len() <= 64;
    if isa() == Isa::Scalar || bounded && !(fits(a) || mode == Mode::Global && fits(b)) {
        0
    } else {
        RECT
    }
}

/// `evaluate_rect` of the kernel on the instruction set in use.
pub(super) fn score_rect(
    kernel: &Levenshtein,
    rows: &[ByteView<'_>],
    cols: &[ByteView<'_>],
    cutoff: Option<i32>,
    out: &mut Vec<Option<i32>>,
) -> Result<(), EngineError> {
    rect_on(isa(), kernel, rows, cols, cutoff, out)
}

/// Every entry equals the cell's `evaluate(row, col, cutoff)`:
/// - a negative cutoff gives None, as `evaluate` does before any work;
/// - the patterns are the rows in `prefix` and `infix`, and in `global` the
///   list with more rows of 1-64 bytes, or with as many the longer list (the
///   distance is symmetric); the other list holds the texts;
/// - with fewer than `SHARED_RUN` patterns, or on `Scalar`, the cells go
///   through `score_on` a row at a time;
/// - a pattern of 1-64 bytes goes to a shared-pattern group of the word of the
///   longest text, which scans every text (`RECT_TEXTS` at a time); m = 0
///   gives the §2 empty-pattern `dist` (n in global mode, else 0); a longer
///   pattern's cells are admitted alone (`admit`, under the cutoff);
/// - each lane's `dist` is then tested against the cutoff.
fn rect_on(
    isa: Isa,
    kernel: &Levenshtein,
    rows: &[ByteView<'_>],
    cols: &[ByteView<'_>],
    cutoff: Option<i32>,
    out: &mut Vec<Option<i32>>,
) -> Result<(), EngineError> {
    let cells = rows
        .len()
        .checked_mul(cols.len())
        .ok_or_else(|| allocation_error(BATCH_WHAT, "size overflows usize"))?;
    try_reserve_exact(out, cells, BATCH_WHAT)?;
    if !live(cutoff) {
        out.resize(out.len() + cells, None);
        return Ok(());
    }
    let mode = kernel.mode();
    // Cell (p, t) of pattern p and text t is entry p * width + t, or with
    // the columns as patterns t * width + p.
    let lanes =
        |views: &[ByteView<'_>]| views.iter().filter(|x| (1..=64).contains(&x.len())).count();
    let flip = mode == Mode::Global && (lanes(cols), cols.len()) > (lanes(rows), rows.len());
    let (patterns, texts) = if flip { (cols, rows) } else { (rows, cols) };
    if isa == Isa::Scalar || patterns.len() < SHARED_RUN {
        let mut pairs = Vec::new();
        try_reserve_exact(&mut pairs, cols.len(), BATCH_WHAT)?;
        for &x in rows {
            pairs.clear();
            pairs.extend(cols.iter().map(|&y| (x, y, cutoff)));
            score_on(isa, kernel, &pairs, out)?;
        }
        return Ok(());
    }
    let form = Form::of(isa);
    let width = cols.len();
    let entry = |p: usize, t: usize| if flip { t * width + p } else { p * width + t };
    // Borrowed rows: one relaxed copy each (a repeated row once), as in
    // `score_on`, patterns first.
    let mut copied = 0usize;
    let mut last = None;
    for &view in patterns.iter().chain(texts) {
        if fresh(view, &mut last) {
            copied = view
                .len()
                .checked_next_multiple_of(8)
                .and_then(|span| copied.checked_add(span))
                .ok_or_else(|| allocation_error(ARENA_WHAT, "size overflows usize"))?;
        }
    }
    let mut arena = Vec::new();
    try_reserve_exact(&mut arena, copied, ARENA_WHAT)?;
    arena.resize(copied, 0u8);
    let (mut at, mut last) = (0, None);
    for &view in patterns.iter().chain(texts) {
        if fresh(view, &mut last) {
            let end = at + span(view);
            view.copy_padded(0, &mut arena[at..end]);
            at = end;
        }
    }
    let mut copies = Rows {
        arena: &arena,
        at: 0,
        last: [None; 2],
    };
    let mut sides: [Vec<&[u8]>; 2] = [Vec::new(), Vec::new()];
    for (side, views) in sides.iter_mut().zip([patterns, texts]) {
        try_reserve_exact(side, views.len(), BATCH_WHAT)?;
        // One side of `copies` for both lists: they were copied in this order.
        side.extend(views.iter().map(|&view| copies.get(0, view)));
    }
    let [patterns, texts] = sides;
    let n_max = texts.iter().map(|text| text.len()).max().unwrap_or(0);
    let mut seen = [false; 256];
    let mut bytes = [0u8; 256];
    let mut distinct = 0;
    for &x in texts.iter().copied().flatten() {
        if !std::mem::replace(&mut seen[usize::from(x)], true) {
            bytes[distinct] = x;
            distinct += 1;
        }
    }
    let base = out.len();
    out.resize(base + cells, None);
    // Lane patterns (`k` is the pattern's index), and the cells of longer
    // patterns admitted alone (`k` is the entry).
    let long = patterns.iter().filter(|x| x.len() > 64).count();
    let mut items = Vec::new();
    let mut pool = Vec::new();
    let mut order = Vec::new();
    try_reserve_exact(&mut items, patterns.len(), BATCH_WHAT)?;
    try_reserve_exact(&mut pool, long * texts.len(), BATCH_WHAT)?;
    try_reserve_exact(
        &mut order,
        patterns.len().max(long * texts.len()),
        BATCH_WHAT,
    )?;
    let (mut counts, mut pool_counts) = ([0usize; WORDS], [0usize; WORDS]);
    for (p, &x) in patterns.iter().enumerate() {
        let m = x.len();
        if m == 0 {
            for (t, text) in texts.iter().enumerate() {
                let dist = if mode == Mode::Global { text.len() } else { 0 };
                out[base + entry(p, t)] = passes(dist, cutoff);
            }
        } else if let Some(word) = word_of(m, n_max) {
            items.push(Item {
                a: x,
                b: &[],
                word,
                k: p,
            });
            counts[usize::from(word)] += 1;
        } else {
            for (t, &y) in texts.iter().enumerate() {
                let k = entry(p, t);
                let (a, b) = if flip { (y, x) } else { (x, y) };
                out[base + k] = admit(a, b, k, mode, cutoff, &mut pool, &mut pool_counts)?;
            }
        }
    }
    let lanes = form.lanes();
    let mut dists = Vec::new();
    try_reserve_exact(&mut dists, RECT_TEXTS * WIDE, BATCH_WHAT)?;
    dists.resize(RECT_TEXTS * WIDE, 0u32);
    grouped(lanes, &items, counts, &mut order, |word, members| {
        for (chunk, first) in texts.chunks(RECT_TEXTS).zip((0..).step_by(RECT_TEXTS)) {
            let shared = Shared {
                texts: chunk,
                bytes: &bytes[..distinct],
            };
            let job = (mode, &items[..], members, &mut dists[..]);
            match word {
                0 => u8::run(form, job, Some(shared))?,
                1 => u16::run(form, job, Some(shared))?,
                2 => u32::run(form, job, Some(shared))?,
                _ => u64::run(form, job, Some(shared))?,
            }
            for (t, dists) in (first..)
                .zip(dists.chunks_exact(members.len()))
                .take(chunk.len())
            {
                for (&member, &dist) in members.iter().zip(dists) {
                    out[base + entry(items[member].k, t)] = passes(dist as usize, cutoff);
                }
            }
        }
        Ok(())
    })?;
    let mut pair_dists = [0u32; WIDE];
    grouped(lanes, &pool, pool_counts, &mut order, |word, members| {
        let job = (mode, &pool[..], members, &mut pair_dists[..]);
        match word {
            0 => u8::run(form, job, None)?,
            1 => u16::run(form, job, None)?,
            2 => u32::run(form, job, None)?,
            _ => u64::run(form, job, None)?,
        }
        for (&member, &dist) in members.iter().zip(&pair_dists) {
            out[base + pool[member].k] = passes(dist as usize, cutoff);
        }
        Ok(())
    })
}

/// Whether a pair with this cutoff is scored: a negative cutoff selects nothing.
fn live(cutoff: Option<i32>) -> bool {
    cutoff.is_none_or(|limit| limit >= 0)
}

/// The shared runs of a batch into `runs` (capacity at least
/// `pairs.len() / SHARED_RUN + 1`), in pair order, as `(first, end, side)`:
/// the maximal runs of at least `SHARED_RUN` consecutive pairs with one row on
/// side 0 (`a`, in `global` only), then on side 1 (`b`) among the other pairs.
fn shared_runs(
    pairs: &[(ByteView<'_>, ByteView<'_>, Option<i32>)],
    mode: Mode,
    runs: &mut Vec<(usize, usize, usize)>,
) {
    runs.clear();
    let view = |side: usize, k: usize| if side == 0 { pairs[k].0 } else { pairs[k].1 };
    // A run holds pairs k and k + 1 with k a multiple of `PROBE`, since it is
    // longer than `PROBE` pairs: a batch without such a step has none.
    const PROBE: usize = 8;
    const _: () = assert!(SHARED_RUN > PROBE);
    let sides: &[usize] = if mode == Mode::Global { &[0, 1] } else { &[1] };
    let step = |k: usize| {
        sides
            .iter()
            .any(|&side| view(side, k + 1).same_row(view(side, k)))
    };
    if !(0..pairs.len().saturating_sub(1)).step_by(PROBE).any(step) {
        return;
    }
    // The maximal runs of `side` within pairs `from..to`.
    let find = |side: usize, from: usize, to: usize, runs: &mut Vec<_>| {
        let view = |k: usize| view(side, k);
        if from >= to {
            return;
        }
        let (mut first, mut previous) = (from, view(from));
        for k in from + 1..to {
            let current = view(k);
            if !current.same_row(previous) {
                if k - first >= SHARED_RUN {
                    runs.push((first, k, side));
                }
                first = k;
            }
            previous = current;
        }
        if to - first >= SHARED_RUN {
            runs.push((first, to, side));
        }
    };
    if mode == Mode::Global {
        find(0, 0, pairs.len(), runs);
    }
    let (mut from, found) = (0, runs.len());
    for index in 0..=found {
        let (to, next) = if index < found {
            (runs[index].0, runs[index].1)
        } else {
            (pairs.len(), pairs.len())
        };
        find(1, from, to, runs);
        from = next;
    }
    runs.sort_unstable_by_key(|run| run.0);
}

/// One live pair alone: its entry now, or None with the pair queued in
/// `pool` (its word counted in `counts`) for a per-pair group.
#[inline(always)]
fn admit<'v>(
    a: &'v [u8],
    b: &'v [u8],
    k: usize,
    mode: Mode,
    cutoff: Option<i32>,
    pool: &mut Vec<Item<'v>>,
    counts: &mut [usize; WORDS],
) -> Result<Option<i32>, EngineError> {
    // Patterns that fit a narrow word keep their affixes: the lanes cost less
    // than finding them (data-dependent loop exits).
    let (a, b) = if a.len() > STRIP_ABOVE {
        strip_shared(a, b, mode)
    } else {
        (a, b)
    };
    let (m, n) = (a.len(), b.len());
    Ok(if m == 0 {
        passes(if mode == Mode::Global { n } else { 0 }, cutoff)
    } else if mode == Mode::Global && cutoff.is_some_and(|limit| m.abs_diff(n) > limit as usize) {
        None
    } else if let Some(word) = word_of(m, n) {
        pool.push(Item { a, b, word, k });
        counts[usize::from(word)] += 1;
        None // set by its group
    } else {
        score_reduced(a, b, mode, cutoff)?
    })
}

/// Run lane `items` in groups of `lanes[word]`: sorted into their words by
/// counting (item order within a word) in `order`, whose capacity holds every
/// item; full groups of each word, a word's remainder joining the next wider
/// word (a pair fits every wider word), the widest word taking a partial
/// group. `run(word, members)` runs one group; members index `items`.
fn grouped(
    lanes: [usize; WORDS],
    items: &[Item<'_>],
    counts: [usize; WORDS],
    order: &mut Vec<usize>,
    mut run: impl FnMut(u8, &[usize]) -> Result<(), EngineError>,
) -> Result<(), EngineError> {
    let total: usize = counts.iter().sum();
    if total == 0 {
        return Ok(());
    }
    order.clear();
    order.resize(total, 0usize);
    let mut next = [0usize; WORDS];
    for word in 1..WORDS {
        next[word] = next[word - 1] + counts[word - 1];
    }
    for (k, item) in items.iter().enumerate() {
        let slot = &mut next[usize::from(item.word)];
        order[*slot] = k;
        *slot += 1;
    }
    let (mut start, mut end) = (0, 0);
    for (word, lanes) in lanes.into_iter().enumerate() {
        end += counts[word];
        let stop = if word + 1 == WORDS {
            end
        } else {
            start + (end - start) / lanes * lanes
        };
        for members in order[start..stop].chunks(lanes) {
            run(word as u8, members)?;
        }
        start = stop;
    }
    Ok(())
}

/// Bytes a borrowed row takes in the arena: its length rounded up to whole
/// words (the sum was checked when the arena was sized).
fn span(view: ByteView<'_>) -> usize {
    view.len().next_multiple_of(8)
}

/// Whether borrowed `view` takes a copy of its own: not a slice, and not the
/// row its side copied last (`last`, which it then becomes).
fn fresh<'x>(view: ByteView<'x>, last: &mut Option<ByteView<'x>>) -> bool {
    if view.as_slice().is_some() || last.is_some_and(|last| last.same_row(view)) {
        return false;
    }
    *last = Some(view);
    true
}

/// The batch's rows in pair order: immutable storage directly, a borrowed
/// row from its copy in `arena` (copied in the same order, `fresh`).
struct Rows<'x> {
    arena: &'x [u8],
    at: usize,
    /// Per side, the last copied row and its copy.
    last: [Option<(ByteView<'x>, &'x [u8])>; 2],
}

impl<'x> Rows<'x> {
    fn get(&mut self, side: usize, view: ByteView<'x>) -> &'x [u8] {
        if let Some(bytes) = view.as_slice() {
            return bytes;
        }
        if let Some((last, bytes)) = self.last[side] {
            if last.same_row(view) {
                return bytes;
            }
        }
        let bytes = &self.arena[self.at..self.at + view.len()];
        self.at += span(view);
        self.last[side] = Some((view, bytes));
        bytes
    }
}

// ------------------------------------------------------------------ words and records

/// Columns per tile of match masks.
const TILE: usize = 128;

/// A group's arguments: mode, the batch's items, the members (indices into
/// the items, at most one record of lanes) and their dists, in member order
/// (a shared-pattern group's: text by text, `members.len()` per text).
type Group<'j, 'v> = (Mode, &'j [Item<'v>], &'j [usize], &'j mut [u32]);

/// The texts of a shared-pattern group, which every member scans with its own
/// pattern, and the distinct bytes of all of them.
#[derive(Clone, Copy)]
struct Shared<'j, 'v> {
    texts: &'j [&'v [u8]],
    bytes: &'j [u8],
}

/// One lane word: the pattern bits of a pair, also its score counter.
trait Word: Copy + Ord + 'static {
    /// Longest pattern a lane holds.
    const BITS: usize;
    /// Largest count a lane holds.
    const MAX: u64;
    /// The low bits of `value`.
    fn of(value: u64) -> Self;
    fn get(self) -> u64;
    fn or(self, other: Self) -> Self;
    /// Run one group in `form`, on that form's record of this word: a
    /// shared-pattern group (`shared`) or a per-pair group (`shared` None).
    fn run(
        form: Form,
        group: Group<'_, '_>,
        shared: Option<Shared<'_, '_>>,
    ) -> Result<(), EngineError>;
}

/// A lane record: one word per lane of a group.
trait Record: Copy + AsRef<[Self::W]> + AsMut<[Self::W]> + 'static {
    type W: Word;
    const ZERO: Self;
    /// This thread's buffers of this record type.
    fn with_buffers<T>(body: impl FnOnce(&mut Buffers<Self>) -> T) -> T;
}

/// Per-thread lane buffers of one record type (at most 32 KiB each, kept).
struct Buffers<R: Record> {
    /// Match-mask rows by byte value (portable form only): bit i of lane l of
    /// row x is set iff byte i of lane l's pattern is x. All zero between
    /// groups; `dirty` is set while a group may have left bits (after a panic
    /// mid-group the next group clears all first).
    peq: Vec<R>,
    dirty: bool,
    /// Match masks of one tile of columns: lane l of record t is the match
    /// mask of lane l's pattern for b[c0 + t].
    eq: Vec<R>,
    /// Match masks of a shared-pattern group by byte value: lane l of row x is
    /// the match mask of lane l's pattern for x. A group writes the rows of
    /// its text's bytes and reads no other row.
    shared: Vec<R>,
}

impl<R: Record> Default for Buffers<R> {
    fn default() -> Self {
        Self {
            peq: Vec::new(),
            dirty: false,
            eq: Vec::new(),
            shared: Vec::new(),
        }
    }
}

impl<R: Record> Buffers<R> {
    /// The tile, and the Peq rows unless the form compares directly, at full
    /// size; reserved fallibly on first use.
    fn prepare(&mut self, direct: bool) -> Result<(), EngineError> {
        let rows = if direct { 0 } else { 256 };
        for (buffer, len) in [(&mut self.peq, rows), (&mut self.eq, TILE)] {
            grow(buffer, len)?;
        }
        Ok(())
    }
}

/// `buffer` at `len` records at least; reserved fallibly.
fn grow<R: Record>(buffer: &mut Vec<R>, len: usize) -> Result<(), EngineError> {
    if buffer.len() < len {
        try_reserve_exact(buffer, len - buffer.len(), MASKS_WHAT)?;
        buffer.resize(len, R::ZERO);
    }
    Ok(())
}

/// One group, ready to run: `Group`, a shared-pattern group's texts (None for a
/// per-pair group) and this thread's buffers.
struct Job<'j, 'v, R: Record> {
    mode: Mode,
    items: &'j [Item<'v>],
    members: &'j [usize],
    dists: &'j mut [u32],
    shared: Option<Shared<'j, 'v>>,
    buffers: &'j mut Buffers<R>,
}

/// Run `group` in form `simd` on records `R`: a shared-pattern group when
/// `shared` holds its texts, else a per-pair group.
fn run_on<S: Simd<R>, R: Record>(
    simd: S,
    group: Group<'_, '_>,
    shared: Option<Shared<'_, '_>>,
) -> Result<(), EngineError> {
    let (mode, items, members, dists) = group;
    R::with_buffers(|buffers| {
        match shared {
            Some(_) => grow(&mut buffers.shared, 256)?,
            None => buffers.prepare(S::DIRECT)?,
        }
        simd.run(Job {
            mode,
            items,
            members,
            dists,
            shared,
            buffers,
        });
        Ok(())
    })
}

macro_rules! word {
    ($word:ty, $narrow:ident, $wide:ident) => {
        record!($word, NARROW, $narrow);
        #[cfg(target_arch = "x86_64")]
        record!($word, WIDE, $wide);

        impl Word for $word {
            const BITS: usize = <$word>::BITS as usize;
            const MAX: u64 = <$word>::MAX as u64;
            #[inline(always)]
            fn of(value: u64) -> Self {
                value as $word
            }
            #[inline(always)]
            fn get(self) -> u64 {
                u64::from(self)
            }
            #[inline(always)]
            fn or(self, other: Self) -> Self {
                self | other
            }
            fn run(
                form: Form,
                group: Group<'_, '_>,
                shared: Option<Shared<'_, '_>>,
            ) -> Result<(), EngineError> {
                type Narrow = [$word; NARROW / std::mem::size_of::<$word>()];
                #[cfg(target_arch = "x86_64")]
                type Wide = [$word; WIDE / std::mem::size_of::<$word>()];
                match form {
                    Form::Portable => run_on::<_, Narrow>(Portable, group, shared),
                    #[cfg(target_arch = "x86_64")]
                    Form::Avx2(simd) => run_on::<_, Wide>(simd, group, shared),
                    #[cfg(all(target_arch = "x86_64", seqblaze_avx512))]
                    Form::Avx512(simd) => run_on::<_, Wide>(simd, group, shared),
                }
            }
        }
    };
}

macro_rules! record {
    ($word:ty, $bytes:expr, $local:ident) => {
        thread_local! {
            static $local: RefCell<Buffers<[$word; $bytes / std::mem::size_of::<$word>()]>> =
                RefCell::new(Buffers::default());
        }

        impl Record for [$word; $bytes / std::mem::size_of::<$word>()] {
            type W = $word;
            const ZERO: Self = [0; $bytes / std::mem::size_of::<$word>()];
            fn with_buffers<T>(body: impl FnOnce(&mut Buffers<Self>) -> T) -> T {
                $local.with(|cell| body(&mut cell.borrow_mut()))
            }
        }
    };
}

word!(u8, NARROW8, WIDE8);
word!(u16, NARROW16, WIDE16);
word!(u32, NARROW32, WIDE32);
word!(u64, NARROW64, WIDE64);

// ------------------------------------------------------------------ vectors

/// One form's vector of a lane record `R`; arithmetic wraps in `R::W`.
/// `run` enters the group compiled for the form's instruction set.
trait Simd<R: Record>: Copy {
    type V: Copy;
    fn load(self, record: &R) -> Self::V;
    fn store(self, record: &mut R, v: Self::V);
    fn splat(self, x: R::W) -> Self::V;
    fn and(self, x: Self::V, y: Self::V) -> Self::V;
    fn or(self, x: Self::V, y: Self::V) -> Self::V;
    fn xor(self, x: Self::V, y: Self::V) -> Self::V;
    fn add(self, x: Self::V, y: Self::V) -> Self::V;
    fn sub(self, x: Self::V, y: Self::V) -> Self::V;
    /// All ones in the lanes where x = y, else zero.
    fn eq(self, x: Self::V, y: Self::V) -> Self::V;
    /// Unsigned minimum.
    fn min(self, x: Self::V, y: Self::V) -> Self::V;
    /// Whether `fill` forms match masks by comparing the pattern with each
    /// byte; otherwise the group reads them from its Peq rows.
    const DIRECT: bool = false;
    /// Whether a shared-pattern group scans its texts two at a time, side by
    /// side: independent chains for the latency-bound step, for a form with
    /// few vectors per record and registers to spare.
    const PAIRED: bool = false;
    /// With `DIRECT`: lane `lane` of `eq[t]` for every byte `bytes[t]`, the
    /// match mask of pattern `a` (1..=BITS bytes): bit i set iff a[i] = bytes[t].
    fn fill(self, _a: &[u8], _bytes: &[u8], _lane: usize, _eq: &mut [R]) {
        unreachable!("only direct forms fill by comparison")
    }
    /// Lane `lane` of `rows[x]` for every byte x of `bytes` (whose rows are
    /// zero in this lane), the match mask of pattern `a` (1..=BITS bytes):
    /// bit i set iff a[i] = x. Rows of other bytes may change; a shared-pattern
    /// group never reads them. By default one bit per byte of `a`.
    #[inline(always)]
    fn masks(self, a: &[u8], _bytes: &[u8], lane: usize, rows: &mut [R; 256]) {
        for (i, &x) in a.iter().enumerate() {
            let cell = &mut rows[usize::from(x)].as_mut()[lane];
            *cell = cell.or(R::W::of(1 << i));
        }
    }
    /// Run `job` (`group`) compiled for this form's instruction set.
    fn run(self, job: Job<'_, '_, R>);
}

/// Plain arrays: any word, any CPU (the compiler may vectorize them).
#[derive(Clone, Copy)]
struct Portable;

/// `f` lane by lane.
#[inline(always)]
fn lanewise<R: Record>(mut x: R, y: R, f: impl Fn(R::W, R::W) -> R::W) -> R {
    for (left, &right) in x.as_mut().iter_mut().zip(y.as_ref()) {
        *left = f(*left, right);
    }
    x
}

macro_rules! portable {
    ($($word:ty),*) => {$(
        impl Simd<[$word; NARROW / std::mem::size_of::<$word>()]> for Portable {
            type V = [$word; NARROW / std::mem::size_of::<$word>()];
            #[inline(always)]
            fn load(self, record: &Self::V) -> Self::V {
                *record
            }
            #[inline(always)]
            fn store(self, record: &mut Self::V, v: Self::V) {
                *record = v;
            }
            #[inline(always)]
            fn splat(self, x: $word) -> Self::V {
                [x; NARROW / std::mem::size_of::<$word>()]
            }
            #[inline(always)]
            fn and(self, x: Self::V, y: Self::V) -> Self::V {
                lanewise(x, y, |p, q| p & q)
            }
            #[inline(always)]
            fn or(self, x: Self::V, y: Self::V) -> Self::V {
                lanewise(x, y, |p, q| p | q)
            }
            #[inline(always)]
            fn xor(self, x: Self::V, y: Self::V) -> Self::V {
                lanewise(x, y, |p, q| p ^ q)
            }
            #[inline(always)]
            fn add(self, x: Self::V, y: Self::V) -> Self::V {
                lanewise(x, y, <$word>::wrapping_add)
            }
            #[inline(always)]
            fn sub(self, x: Self::V, y: Self::V) -> Self::V {
                lanewise(x, y, <$word>::wrapping_sub)
            }
            #[inline(always)]
            fn eq(self, x: Self::V, y: Self::V) -> Self::V {
                lanewise(x, y, |p, q| if p == q { <$word>::MAX } else { 0 })
            }
            #[inline(always)]
            fn min(self, x: Self::V, y: Self::V) -> Self::V {
                lanewise(x, y, <$word>::min)
            }
            fn run(self, job: Job<'_, '_, Self::V>) {
                group(self, job)
            }
        }
    )*};
}

portable!(u8, u16, u32, u64);

#[cfg(target_arch = "x86_64")]
mod x86 {
    use super::{group, Job, Record, Simd, Word, WIDE};
    use std::arch::x86_64::*;

    /// Bits 0..m set (m <= 64): the pattern rows of a direct fill.
    #[inline(always)]
    fn rows_mask(m: usize) -> u64 {
        if m >= 64 {
            !0
        } else {
            (1 << m) - 1
        }
    }

    /// AVX2: a 128-byte record as four 256-bit vectors. Only `new` makes one,
    /// after detection.
    #[derive(Clone, Copy)]
    pub(super) struct Avx2(());

    impl Avx2 {
        /// None unless the CPU has AVX2.
        pub(super) fn new() -> Option<Self> {
            std::is_x86_feature_detected!("avx2").then_some(Self(()))
        }
    }

    /// Unsigned minimum of 64-bit lanes below 2^63 (AVX2 has no `min_epu64`;
    /// counts, lengths and column numbers are below 2^31).
    #[inline(always)]
    fn min_epu63(x: __m256i, y: __m256i) -> __m256i {
        // SAFETY: called only inside `Avx2` methods, after AVX2 was detected.
        unsafe { _mm256_blendv_epi8(x, y, _mm256_cmpgt_epi64(x, y)) }
    }

    macro_rules! avx2 {
        ($word:ty, $int:ty, $set1:ident, $add:ident, $sub:ident, $cmpeq:ident, $min:ident) => {
            // SAFETY (every `unsafe` block of this impl): an `Avx2` exists only
            // after AVX2 was detected; loads and stores go through references
            // to one 128-byte record (four 32-byte parts) or to the 64-byte
            // stack copy of the pattern in `fill` and `masks` (two parts).
            impl Simd<[$word; WIDE / std::mem::size_of::<$word>()]> for Avx2 {
                type V = [__m256i; WIDE / 32];
                #[inline(always)]
                fn load(self, record: &[$word; WIDE / std::mem::size_of::<$word>()]) -> Self::V {
                    let part = record.as_ptr().cast::<__m256i>();
                    std::array::from_fn(|k| unsafe { _mm256_loadu_si256(part.add(k)) })
                }
                #[inline(always)]
                fn store(
                    self,
                    record: &mut [$word; WIDE / std::mem::size_of::<$word>()],
                    v: Self::V,
                ) {
                    let part = record.as_mut_ptr().cast::<__m256i>();
                    for (k, v) in v.into_iter().enumerate() {
                        unsafe { _mm256_storeu_si256(part.add(k), v) };
                    }
                }
                #[inline(always)]
                fn splat(self, x: $word) -> Self::V {
                    [unsafe { $set1(x as $int) }; WIDE / 32]
                }
                #[inline(always)]
                fn and(self, x: Self::V, y: Self::V) -> Self::V {
                    std::array::from_fn(|k| unsafe { _mm256_and_si256(x[k], y[k]) })
                }
                #[inline(always)]
                fn or(self, x: Self::V, y: Self::V) -> Self::V {
                    std::array::from_fn(|k| unsafe { _mm256_or_si256(x[k], y[k]) })
                }
                #[inline(always)]
                fn xor(self, x: Self::V, y: Self::V) -> Self::V {
                    std::array::from_fn(|k| unsafe { _mm256_xor_si256(x[k], y[k]) })
                }
                #[inline(always)]
                fn add(self, x: Self::V, y: Self::V) -> Self::V {
                    std::array::from_fn(|k| unsafe { $add(x[k], y[k]) })
                }
                #[inline(always)]
                fn sub(self, x: Self::V, y: Self::V) -> Self::V {
                    std::array::from_fn(|k| unsafe { $sub(x[k], y[k]) })
                }
                #[inline(always)]
                fn eq(self, x: Self::V, y: Self::V) -> Self::V {
                    std::array::from_fn(|k| unsafe { $cmpeq(x[k], y[k]) })
                }
                #[inline(always)]
                #[allow(unused_unsafe)]
                fn min(self, x: Self::V, y: Self::V) -> Self::V {
                    std::array::from_fn(|k| unsafe { $min(x[k], y[k]) })
                }
                const DIRECT: bool = true;
                /// The pattern as two 32-byte halves of a zero-padded copy; one
                /// byte comparison and move mask per half and byte (the upper
                /// half only for patterns over 32 bytes), masked to rows 0..m.
                #[inline(always)]
                fn fill(
                    self,
                    a: &[u8],
                    bytes: &[u8],
                    lane: usize,
                    eq: &mut [[$word; WIDE / std::mem::size_of::<$word>()]],
                ) {
                    let mut padded = [0u8; 64];
                    padded[..a.len()].copy_from_slice(a);
                    let valid = rows_mask(a.len());
                    let half = padded.as_ptr().cast::<__m256i>();
                    let (low, high) =
                        unsafe { (_mm256_loadu_si256(half), _mm256_loadu_si256(half.add(1))) };
                    let hits = |pattern: __m256i, y: __m256i| {
                        u64::from(
                            unsafe { _mm256_movemask_epi8(_mm256_cmpeq_epi8(pattern, y)) } as u32,
                        )
                    };
                    if a.len() <= 32 {
                        for (record, &x) in eq.iter_mut().zip(bytes) {
                            let y = unsafe { _mm256_set1_epi8(x as i8) };
                            record.as_mut()[lane] = <$word>::of(hits(low, y) & valid);
                        }
                    } else {
                        for (record, &x) in eq.iter_mut().zip(bytes) {
                            let y = unsafe { _mm256_set1_epi8(x as i8) };
                            let mask = hits(low, y) | hits(high, y) << 32;
                            record.as_mut()[lane] = <$word>::of(mask & valid);
                        }
                    }
                }
                /// As `fill`, into the rows of the text's bytes.
                #[inline(always)]
                fn masks(
                    self,
                    a: &[u8],
                    bytes: &[u8],
                    lane: usize,
                    rows: &mut [[$word; WIDE / std::mem::size_of::<$word>()]; 256],
                ) {
                    let mut padded = [0u8; 64];
                    padded[..a.len()].copy_from_slice(a);
                    let valid = rows_mask(a.len());
                    let half = padded.as_ptr().cast::<__m256i>();
                    let (low, high) =
                        unsafe { (_mm256_loadu_si256(half), _mm256_loadu_si256(half.add(1))) };
                    let hits = |pattern: __m256i, y: __m256i| {
                        u64::from(
                            unsafe { _mm256_movemask_epi8(_mm256_cmpeq_epi8(pattern, y)) } as u32,
                        )
                    };
                    let wide = a.len() > 32;
                    for &x in bytes {
                        let y = unsafe { _mm256_set1_epi8(x as i8) };
                        let mut mask = hits(low, y);
                        if wide {
                            mask |= hits(high, y) << 32;
                        }
                        rows[usize::from(x)].as_mut()[lane] = <$word>::of(mask & valid);
                    }
                }
                fn run(self, job: Job<'_, '_, [$word; WIDE / std::mem::size_of::<$word>()]>) {
                    // SAFETY: AVX2 was detected (see the impl).
                    unsafe { avx2(self, job) }
                }
            }
        };
    }

    avx2!(
        u8,
        i8,
        _mm256_set1_epi8,
        _mm256_add_epi8,
        _mm256_sub_epi8,
        _mm256_cmpeq_epi8,
        _mm256_min_epu8
    );
    avx2!(
        u16,
        i16,
        _mm256_set1_epi16,
        _mm256_add_epi16,
        _mm256_sub_epi16,
        _mm256_cmpeq_epi16,
        _mm256_min_epu16
    );
    avx2!(
        u32,
        i32,
        _mm256_set1_epi32,
        _mm256_add_epi32,
        _mm256_sub_epi32,
        _mm256_cmpeq_epi32,
        _mm256_min_epu32
    );
    avx2!(
        u64,
        i64,
        _mm256_set1_epi64x,
        _mm256_add_epi64,
        _mm256_sub_epi64,
        _mm256_cmpeq_epi64,
        min_epu63
    );

    #[target_feature(enable = "avx2")]
    unsafe fn avx2<R: Record>(s: Avx2, job: Job<'_, '_, R>)
    where
        Avx2: Simd<R>,
    {
        group(s, job)
    }

    /// Compiled only by a rustc with the AVX-512 intrinsics and target
    /// features (stable since 1.89; `build.rs` sets `seqblaze_avx512`). Older
    /// compilers within the workspace's rust-version use the AVX2 form.
    #[cfg(seqblaze_avx512)]
    #[allow(clippy::incompatible_msrv)]
    mod avx512 {
        use super::super::{group, Job, Record, Simd, Word, WIDE};
        use super::rows_mask;
        use std::arch::x86_64::*;

        /// AVX-512 F and BW: a 128-byte record as two 512-bit vectors. Only
        /// `new` makes one, after detection.
        #[derive(Clone, Copy)]
        pub(crate) struct Avx512(());

        impl Avx512 {
            /// None unless the CPU has AVX-512 F and BW.
            pub(crate) fn new() -> Option<Self> {
                (std::is_x86_feature_detected!("avx512f")
                    && std::is_x86_feature_detected!("avx512bw"))
                .then_some(Self(()))
            }
        }

        // All-ones lanes from a comparison mask (BW for bytes and halves).
        #[inline(always)]
        fn ones8(k: __mmask64) -> __m512i {
            // SAFETY: called only inside `Avx512` methods, after detection.
            unsafe { _mm512_movm_epi8(k) }
        }
        #[inline(always)]
        fn ones16(k: __mmask32) -> __m512i {
            // SAFETY: as for `ones8`.
            unsafe { _mm512_movm_epi16(k) }
        }
        #[inline(always)]
        fn ones32(k: __mmask16) -> __m512i {
            // SAFETY: as for `ones8`.
            unsafe { _mm512_maskz_set1_epi32(k, -1) }
        }
        #[inline(always)]
        fn ones64(k: __mmask8) -> __m512i {
            // SAFETY: as for `ones8`.
            unsafe { _mm512_maskz_set1_epi64(k, -1) }
        }

        macro_rules! avx512 {
            ($word:ty, $int:ty, $set1:ident, $add:ident, $sub:ident, $cmpeq:ident, $ones:ident, $min:ident) => {
                // SAFETY (every `unsafe` block of this impl): an `Avx512` exists
                // only after AVX-512 F and BW were detected; loads and stores go
                // through references to one 128-byte record (two 64-byte parts);
                // the masked load of `fill` and `masks` reads only the m <= 64
                // bytes of the pattern slice its mask selects (masked-off bytes
                // are neither read nor faulted on).
                impl Simd<[$word; WIDE / std::mem::size_of::<$word>()]> for Avx512 {
                    type V = [__m512i; WIDE / 64];
                    #[inline(always)]
                    fn load(
                        self,
                        record: &[$word; WIDE / std::mem::size_of::<$word>()],
                    ) -> Self::V {
                        let part = record.as_ptr().cast::<__m512i>();
                        std::array::from_fn(|k| unsafe { _mm512_loadu_si512(part.add(k)) })
                    }
                    #[inline(always)]
                    fn store(
                        self,
                        record: &mut [$word; WIDE / std::mem::size_of::<$word>()],
                        v: Self::V,
                    ) {
                        let part = record.as_mut_ptr().cast::<__m512i>();
                        for (k, v) in v.into_iter().enumerate() {
                            unsafe { _mm512_storeu_si512(part.add(k), v) };
                        }
                    }
                    #[inline(always)]
                    fn splat(self, x: $word) -> Self::V {
                        [unsafe { $set1(x as $int) }; WIDE / 64]
                    }
                    #[inline(always)]
                    fn and(self, x: Self::V, y: Self::V) -> Self::V {
                        std::array::from_fn(|k| unsafe { _mm512_and_si512(x[k], y[k]) })
                    }
                    #[inline(always)]
                    fn or(self, x: Self::V, y: Self::V) -> Self::V {
                        std::array::from_fn(|k| unsafe { _mm512_or_si512(x[k], y[k]) })
                    }
                    #[inline(always)]
                    fn xor(self, x: Self::V, y: Self::V) -> Self::V {
                        std::array::from_fn(|k| unsafe { _mm512_xor_si512(x[k], y[k]) })
                    }
                    #[inline(always)]
                    fn add(self, x: Self::V, y: Self::V) -> Self::V {
                        std::array::from_fn(|k| unsafe { $add(x[k], y[k]) })
                    }
                    #[inline(always)]
                    fn sub(self, x: Self::V, y: Self::V) -> Self::V {
                        std::array::from_fn(|k| unsafe { $sub(x[k], y[k]) })
                    }
                    #[inline(always)]
                    fn eq(self, x: Self::V, y: Self::V) -> Self::V {
                        std::array::from_fn(|k| $ones(unsafe { $cmpeq(x[k], y[k]) }))
                    }
                    #[inline(always)]
                    fn min(self, x: Self::V, y: Self::V) -> Self::V {
                        std::array::from_fn(|k| unsafe { $min(x[k], y[k]) })
                    }
                    const DIRECT: bool = true;
                    const PAIRED: bool = true;
                    /// The pattern in one vector (a masked load of its m bytes
                    /// only), one masked byte comparison per byte of b.
                    #[inline(always)]
                    fn fill(
                        self,
                        a: &[u8],
                        bytes: &[u8],
                        lane: usize,
                        eq: &mut [[$word; WIDE / std::mem::size_of::<$word>()]],
                    ) {
                        let valid = rows_mask(a.len());
                        let pattern = unsafe { _mm512_maskz_loadu_epi8(valid, a.as_ptr().cast()) };
                        for (record, &x) in eq.iter_mut().zip(bytes) {
                            let y = unsafe { _mm512_set1_epi8(x as i8) };
                            let mask = unsafe { _mm512_mask_cmpeq_epi8_mask(valid, pattern, y) };
                            record.as_mut()[lane] = <$word>::of(mask);
                        }
                    }
                    /// As `fill`, into the rows of the text's bytes.
                    #[inline(always)]
                    fn masks(
                        self,
                        a: &[u8],
                        bytes: &[u8],
                        lane: usize,
                        rows: &mut [[$word; WIDE / std::mem::size_of::<$word>()]; 256],
                    ) {
                        let valid = rows_mask(a.len());
                        let pattern = unsafe { _mm512_maskz_loadu_epi8(valid, a.as_ptr().cast()) };
                        for &x in bytes {
                            let y = unsafe { _mm512_set1_epi8(x as i8) };
                            let mask = unsafe { _mm512_mask_cmpeq_epi8_mask(valid, pattern, y) };
                            rows[usize::from(x)].as_mut()[lane] = <$word>::of(mask);
                        }
                    }
                    fn run(self, job: Job<'_, '_, [$word; WIDE / std::mem::size_of::<$word>()]>) {
                        // SAFETY: AVX-512 F and BW were detected (see the impl).
                        unsafe { avx512(self, job) }
                    }
                }
            };
        }

        avx512!(
            u8,
            i8,
            _mm512_set1_epi8,
            _mm512_add_epi8,
            _mm512_sub_epi8,
            _mm512_cmpeq_epi8_mask,
            ones8,
            _mm512_min_epu8
        );
        avx512!(
            u16,
            i16,
            _mm512_set1_epi16,
            _mm512_add_epi16,
            _mm512_sub_epi16,
            _mm512_cmpeq_epi16_mask,
            ones16,
            _mm512_min_epu16
        );
        avx512!(
            u32,
            i32,
            _mm512_set1_epi32,
            _mm512_add_epi32,
            _mm512_sub_epi32,
            _mm512_cmpeq_epi32_mask,
            ones32,
            _mm512_min_epu32
        );
        avx512!(
            u64,
            i64,
            _mm512_set1_epi64,
            _mm512_add_epi64,
            _mm512_sub_epi64,
            _mm512_cmpeq_epi64_mask,
            ones64,
            _mm512_min_epu64
        );

        #[target_feature(enable = "avx512f,avx512bw")]
        unsafe fn avx512<R: Record>(s: Avx512, job: Job<'_, '_, R>)
        where
            Avx512: Simd<R>,
        {
            group(s, job)
        }
    }

    #[cfg(seqblaze_avx512)]
    pub(super) use avx512::Avx512;
}

// ------------------------------------------------------------------ the group pass

/// One group (module doc), each lane's `dist` into `dists`: a shared-pattern
/// group or a per-pair group.
#[inline(always)]
fn group<S: Simd<R>, R: Record>(s: S, job: Job<'_, '_, R>) {
    match job.shared {
        Some(shared) => shared_group(s, job, shared),
        None => pair_group(s, job),
    }
}

/// One column of every lane: the word step of `Work::scan_full` with hin =
/// `top` (never negative). Returns the masks of the lanes whose row-m delta
/// (`bit`) is +1 and -1, all ones (-1) where set.
#[inline(always)]
fn step<S: Simd<R>, R: Record>(
    s: S,
    x: S::V,
    pv: &mut S::V,
    mv: &mut S::V,
    ones: S::V,
    top: S::V,
    bit: S::V,
) -> (S::V, S::V) {
    let xv = s.or(x, *mv);
    let xh = s.or(s.xor(s.add(s.and(x, *pv), *pv), *pv), x);
    let ph = s.or(*mv, s.xor(s.or(xh, *pv), ones));
    let mh = s.and(*pv, xh);
    let up = s.eq(s.and(ph, bit), bit);
    let down = s.eq(s.and(mh, bit), bit);
    let ph = s.or(s.add(ph, ph), top);
    let mh = s.add(mh, mh);
    *pv = s.or(mh, s.xor(s.or(xv, ph), ones));
    *mv = s.and(ph, xv);
    (up, down)
}

/// The scores of `lanes` lanes into `dists`: the final score, or in `prefix`
/// and `infix` its running minimum.
#[inline(always)]
fn finish<S: Simd<R>, R: Record>(s: S, value: S::V, dists: &mut [u32]) {
    let mut values = R::ZERO;
    s.store(&mut values, value);
    for (dist, value) in dists.iter_mut().zip(values.as_ref()) {
        *dist = u32::try_from(value.get()).expect("dist <= max(m, n) < 2^31");
    }
}

/// The forward passes of a per-pair group: each lane its own pattern and text.
#[inline(always)]
fn pair_group<S: Simd<R>, R: Record>(s: S, job: Job<'_, '_, R>) {
    let Job {
        mode,
        items,
        members,
        dists,
        buffers,
        ..
    } = job;
    let Buffers { peq, dirty, eq, .. } = buffers;
    let lanes = members.len();
    assert!(lanes >= 1 && lanes <= R::ZERO.as_ref().len());
    let eq: &mut [R; TILE] = (&mut eq[..TILE]).try_into().expect("one tile");
    let rows = !S::DIRECT;
    let peq: &mut [R] = if rows { &mut peq[..256] } else { &mut [] };
    if rows {
        if *dirty {
            peq.fill(R::ZERO);
        }
        *dirty = true;
    }
    // Per lane: the score bit (row m), the text length n and the start count m.
    let (mut bit, mut len, mut start) = (R::ZERO, R::ZERO, R::ZERO);
    let (mut n_min, mut n_max) = (usize::MAX, 0);
    for (lane, &k) in members.iter().enumerate() {
        let Item { a, b, .. } = items[k];
        let m = a.len();
        debug_assert!((1..=<R::W>::BITS).contains(&m) && b.len() as u64 <= <R::W>::MAX);
        bit.as_mut()[lane] = R::W::of(1 << (m - 1));
        len.as_mut()[lane] = R::W::of(b.len() as u64);
        start.as_mut()[lane] = R::W::of(m as u64);
        n_min = n_min.min(b.len());
        n_max = n_max.max(b.len());
        if rows {
            let mut row = 1u64;
            for &x in a {
                let cell = &mut peq[usize::from(x)].as_mut()[lane];
                *cell = cell.or(R::W::of(row));
                row <<= 1;
            }
        }
    }
    let ones = s.splat(R::W::of(!0));
    let top = s.splat(R::W::of(u64::from(mode != Mode::Infix)));
    let (bit, len) = (s.load(&bit), s.load(&len));
    let (mut pv, mut mv) = (ones, s.splat(R::W::of(0)));
    let mut score = s.load(&start);
    let mut best = score;
    let minimum = mode != Mode::Global;
    let mut c0 = 0;
    while c0 < n_max {
        let c1 = (c0 + TILE).min(n_max);
        for (lane, &k) in members.iter().enumerate() {
            let Item { a, b, .. } = items[k];
            let bytes = &b[c0.min(b.len())..c1.min(b.len())];
            if rows {
                for (record, &y) in eq.iter_mut().zip(bytes) {
                    record.as_mut()[lane] = peq[usize::from(y)].as_ref()[lane];
                }
            } else {
                s.fill(a, bytes, lane, &mut eq[..]);
            }
        }
        // One column of every lane, the row-m delta added to the score;
        // `masked` keeps the delta only in lanes whose b has column j = c0 + t.
        macro_rules! column {
            ($t:expr, $masked:expr) => {{
                let t = $t;
                let x = s.load(&eq[t]);
                let (mut up, mut down) = step(s, x, &mut pv, &mut mv, ones, top, bit);
                if $masked {
                    // j < n iff min(n, j + 1) = j + 1.
                    let next = s.splat(R::W::of((c0 + t + 1) as u64));
                    let own = s.eq(s.min(len, next), next);
                    up = s.and(up, own);
                    down = s.and(down, own);
                }
                // The masks are all ones (-1) where set: score + [up] - [down].
                score = s.add(s.sub(score, up), down);
                if minimum {
                    best = s.min(best, score);
                }
            }};
        }
        let split = n_min.clamp(c0, c1) - c0;
        for t in 0..split {
            column!(t, false);
        }
        for t in split..c1 - c0 {
            column!(t, true);
        }
        c0 = c1;
    }
    if rows {
        for (lane, &k) in members.iter().enumerate() {
            for &x in items[k].a {
                peq[usize::from(x)].as_mut()[lane] = R::W::of(0);
            }
        }
        *dirty = false;
    }
    finish(s, if minimum { best } else { score }, &mut dists[..lanes]);
}

/// The forward passes of a shared-pattern group: every lane scans each text of
/// `shared` with its own pattern. The match masks of the texts' byte values
/// are formed once per lane for all texts, so a column is one load of its
/// byte's row; the lanes scan one text at a time, so no column is masked.
#[inline(always)]
fn shared_group<S: Simd<R>, R: Record>(s: S, job: Job<'_, '_, R>, shared: Shared<'_, '_>) {
    let Job {
        mode,
        items,
        members,
        dists,
        buffers,
        ..
    } = job;
    let Shared { texts, bytes } = shared;
    let lanes = members.len();
    assert!(lanes >= 1 && lanes <= R::ZERO.as_ref().len());
    assert!(dists.len() >= texts.len() * lanes);
    let rows: &mut [R; 256] = (&mut buffers.shared[..256])
        .try_into()
        .expect("one row per byte value");
    for &x in bytes {
        rows[usize::from(x)] = R::ZERO;
    }
    // Per lane: the score bit (row m) and the start count m.
    let (mut bit, mut start) = (R::ZERO, R::ZERO);
    for (lane, &k) in members.iter().enumerate() {
        let a = items[k].a;
        let m = a.len();
        debug_assert!((1..=<R::W>::BITS).contains(&m));
        bit.as_mut()[lane] = R::W::of(1 << (m - 1));
        start.as_mut()[lane] = R::W::of(m as u64);
        s.masks(a, bytes, lane, rows);
    }
    let rows = &*rows;
    let consts = Consts {
        ones: s.splat(R::W::of(!0)),
        top: s.splat(R::W::of(u64::from(mode != Mode::Infix))),
        bit: s.load(&bit),
        start: s.load(&start),
        minimum: mode != Mode::Global,
    };
    let dists = &mut dists[..texts.len() * lanes];
    if S::PAIRED {
        // Two texts side by side over their common columns, then each alone:
        // every text's own chain of steps, as one at a time.
        for (pair, dists) in texts.chunks(2).zip(dists.chunks_mut(2 * lanes)) {
            debug_assert!(pair.iter().all(|text| text.len() as u64 <= <R::W>::MAX));
            let (mut first, mut second) = (scan_start(s, consts), scan_start(s, consts));
            let (x, y) = (pair[0], pair.get(1).copied().unwrap_or_default());
            let common = x.len().min(y.len());
            for (&u, &v) in x[..common].iter().zip(&y[..common]) {
                scan_column(s, rows, u, &mut first, consts);
                scan_column(s, rows, v, &mut second, consts);
            }
            for &u in &x[common..] {
                scan_column(s, rows, u, &mut first, consts);
            }
            for &v in &y[common..] {
                scan_column(s, rows, v, &mut second, consts);
            }
            let (head, tail) = dists.split_at_mut(lanes);
            scan_done(s, first, head, consts);
            if pair.len() == 2 {
                scan_done(s, second, tail, consts);
            }
        }
        return;
    }
    for (&text, dists) in texts.iter().zip(dists.chunks_exact_mut(lanes)) {
        debug_assert!(text.len() as u64 <= <R::W>::MAX);
        let mut state = scan_start(s, consts);
        for &y in text {
            scan_column(s, rows, y, &mut state, consts);
        }
        scan_done(s, state, dists, consts);
    }
}

/// The constants of a shared-pattern group's scans: all ones, the top-row
/// delta, each lane's row-m bit and start count, and whether the scans keep
/// the running minimum (`prefix` and `infix`).
struct Consts<V> {
    ones: V,
    top: V,
    bit: V,
    start: V,
    minimum: bool,
}

impl<V: Copy> Clone for Consts<V> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<V: Copy> Copy for Consts<V> {}

/// One text's scan state: Pv, Mv, the score and its running minimum.
type Scan<V> = (V, V, V, V);

/// A text's scan state before its first column.
#[inline(always)]
fn scan_start<S: Simd<R>, R: Record>(s: S, c: Consts<S::V>) -> Scan<S::V> {
    (c.ones, s.splat(R::W::of(0)), c.start, c.start)
}

/// One column of a text: byte `y`'s match masks from `rows`, the step, and
/// the row-m delta added to the score.
#[inline(always)]
fn scan_column<S: Simd<R>, R: Record>(
    s: S,
    rows: &[R; 256],
    y: u8,
    (pv, mv, score, best): &mut Scan<S::V>,
    c: Consts<S::V>,
) {
    let x = s.load(&rows[usize::from(y)]);
    let (up, down) = step(s, x, pv, mv, c.ones, c.top, c.bit);
    *score = s.add(s.sub(*score, up), down);
    if c.minimum {
        *best = s.min(*best, *score);
    }
}

/// A text's distances into `dists`: the final score, or its running minimum.
#[inline(always)]
fn scan_done<S: Simd<R>, R: Record>(
    s: S,
    (_, _, score, best): Scan<S::V>,
    dists: &mut [u32],
    c: Consts<S::V>,
) {
    finish(s, if c.minimum { best } else { score }, dists);
}

#[cfg(test)]
mod tests {
    use super::*;
    use seqblaze_core::{batch::SeqBatch, seq::SeqTable};

    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> usize {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (self.0 >> 33) as usize
        }
        fn seq(&mut self, n: usize, alphabet: &[u8]) -> Vec<u8> {
            (0..n)
                .map(|_| alphabet[self.next() % alphabet.len()])
                .collect()
        }
        /// `a` with about one edit in `every` bytes (substitution, insertion, deletion).
        fn edited(&mut self, a: &[u8], every: usize, alphabet: &[u8]) -> Vec<u8> {
            let mut out = Vec::new();
            for &x in a {
                match self.next() % (3 * every) {
                    0 => out.push(alphabet[self.next() % alphabet.len()]),
                    1 => {}
                    2 => out.extend([x, alphabet[self.next() % alphabet.len()]]),
                    _ => out.push(x),
                }
            }
            out
        }
    }

    const ACGT: &[u8] = b"ACGT";

    /// Every available instruction set, in batches of `BATCH` pairs after one
    /// existing entry of `out`, against `evaluate` pair by pair. Per mode the
    /// cutoffs cycle through None, -1, 0, dist - 1, dist, dist + 1, INT32_MAX.
    fn check(a: &SeqTable, b: &SeqTable, pairs: &[(usize, usize)]) {
        for mode in [Mode::Global, Mode::Prefix, Mode::Infix] {
            let kernel = Levenshtein::new(mode, &[]).unwrap();
            let views: Vec<_> = pairs
                .iter()
                .enumerate()
                .map(|(k, &(i, j))| {
                    let (x, y) = (a.get(i), b.get(j));
                    let dist = kernel.evaluate(x, y, None).unwrap().unwrap();
                    let cutoff = match k % 7 {
                        0 => None,
                        1 => Some(-1),
                        2 => Some(0),
                        3 => Some(dist - 1),
                        4 => Some(dist),
                        5 => Some(dist + 1),
                        _ => Some(i32::MAX),
                    };
                    (x, y, cutoff)
                })
                .collect();
            let expected: Vec<_> = views
                .iter()
                .map(|&(x, y, cutoff)| kernel.evaluate(x, y, cutoff).unwrap())
                .collect();
            for isa in Isa::ALL.into_iter().filter(|isa| isa.available()) {
                for (chunk, want) in views.chunks(BATCH).zip(expected.chunks(BATCH)) {
                    let mut out = vec![Some(-7)];
                    score_on(isa, &kernel, chunk, &mut out).unwrap();
                    assert_eq!(out[0], Some(-7));
                    for (k, (got, want)) in out[1..].iter().zip(want).enumerate() {
                        assert_eq!(
                            got,
                            want,
                            "{isa:?} {mode:?} pair {k}: a={:?} b={:?} cutoff {:?}",
                            chunk[k].0.iter().collect::<Vec<_>>(),
                            chunk[k].1.iter().collect::<Vec<_>>(),
                            chunk[k].2
                        );
                    }
                    assert_eq!(out.len(), chunk.len() + 1);
                }
            }
        }
    }

    /// Rows of every word, counter and tile class: patterns across 8, 16, 32
    /// and 64 bytes, texts across 255/256 (the u8 counter) and 256/512 (tiles),
    /// near-copies with shared heads and tails, homopolymers, all 256 bytes.
    fn rows(rng: &mut Rng) -> (Vec<Vec<u8>>, Vec<Vec<u8>>) {
        let (mut a, mut b) = (Vec::new(), Vec::new());
        let lengths = [
            1usize, 2, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65, 70, 100,
        ];
        for (k, &m) in lengths.iter().cycle().take(160).enumerate() {
            let x = rng.seq(m, ACGT);
            let y = match k % 5 {
                0 => rng.edited(&x, 12, ACGT),
                1 => {
                    let n = rng.next() % 40;
                    rng.seq(n, ACGT)
                }
                2 => {
                    let n = rng.next() % 300;
                    [rng.seq(n, ACGT), x.clone()].concat()
                }
                3 => x.clone(),
                _ => rng.edited(&x, 4, ACGT),
            };
            a.push(x);
            b.push(y);
        }
        for &n in &[0usize, 1, 254, 255, 256, 257, 511, 512, 513, 700] {
            for m in [1usize, 5, 8, 9, 16, 30, 64] {
                a.push(rng.seq(m, ACGT));
                b.push(rng.seq(n, ACGT));
                a.push(vec![b'A'; m]);
                b.push(vec![b'A'; n]);
            }
        }
        let all: Vec<u8> = (0..=255).collect();
        for m in [1usize, 3, 8, 20, 40, 64] {
            let x = rng.seq(m, &all);
            b.push(rng.edited(&x, 5, &all));
            a.push(x);
        }
        (a, b)
    }

    #[test]
    fn lane_batches_equal_evaluate_on_owned_rows() {
        let mut rng = Rng(3);
        let (a, b) = rows(&mut rng);
        let ta = SeqTable::new(SeqBatch::from_seqs(a.iter().map(|x| &x[..])));
        let tb = SeqTable::new(SeqBatch::from_seqs(b.iter().map(|x| &x[..])));
        // Zipped rows, then a cross block whose pairs mix every word.
        let mut pairs: Vec<_> = (0..a.len()).map(|k| (k, k)).collect();
        for _ in 0..1500 {
            pairs.push((rng.next() % a.len(), rng.next() % b.len()));
        }
        check(&ta, &tb, &pairs);
    }

    /// NumPy `S<n>` views (borrowed rows copied into the batch arena), with
    /// owned rows on the other side; rows end in a non-NUL byte.
    #[test]
    fn lane_batches_equal_evaluate_on_borrowed_rows() {
        let mut rng = Rng(5);
        let (a, b) = rows(&mut rng);
        let keep = |row: &Vec<u8>| row.last().is_none_or(|&x| x != 0);
        let a: Vec<_> = a.into_iter().filter(keep).collect();
        let b: Vec<_> = b.into_iter().filter(keep).take(a.len()).collect();
        let width = a.iter().map(Vec::len).max().unwrap();
        let mut data = vec![0u8; a.len() * width];
        for (i, row) in a.iter().enumerate() {
            data[i * width..i * width + row.len()].copy_from_slice(row);
        }
        let address = data.as_ptr() as usize;
        // SAFETY: the boxed Vec owns every described row and is not mutated.
        let fixed = unsafe {
            SeqTable::from_fixed_bytes(address, a.len(), width, width as isize, Box::new(data))
        }
        .unwrap();
        let fixed = fixed.fresh_view().unwrap();
        fixed.validate_lengths().unwrap();
        let owned = SeqTable::new(SeqBatch::from_seqs(b.iter().map(|x| &x[..])));
        let mut pairs: Vec<_> = (0..b.len()).map(|k| (k, k)).collect();
        for _ in 0..600 {
            pairs.push((rng.next() % a.len(), rng.next() % b.len()));
        }
        check(&fixed, &owned, &pairs);
        let flipped: Vec<_> = pairs.iter().map(|&(i, j)| (j, i)).collect();
        check(&owned, &fixed, &flipped);
    }

    /// A borrowed NumPy `S<n>` table of `rows`, none of which ends in NUL.
    fn fixed_table(rows: &[Vec<u8>]) -> SeqTable {
        let width = rows.iter().map(Vec::len).max().unwrap().max(1);
        let mut data = vec![0u8; rows.len() * width];
        for (i, row) in rows.iter().enumerate() {
            data[i * width..i * width + row.len()].copy_from_slice(row);
        }
        let address = data.as_ptr() as usize;
        // SAFETY: the boxed Vec owns every described row and is not mutated.
        let fixed = unsafe {
            SeqTable::from_fixed_bytes(address, rows.len(), width, width as isize, Box::new(data))
        }
        .unwrap();
        let fixed = fixed.fresh_view().unwrap();
        fixed.validate_lengths().unwrap();
        fixed
    }

    /// Pairs in runs with one row on a side: whole rows of a cross (`a`
    /// shared) and whole columns (`b` shared), then runs of 1, 15, 16, 17
    /// and 40 pairs from random starts.
    fn runs(rng: &mut Rng, n_a: usize, n_b: usize) -> Vec<(usize, usize)> {
        let mut pairs = Vec::new();
        for _ in 0..6 {
            let i = rng.next() % n_a;
            pairs.extend((0..n_b).map(|j| (i, j)));
            let j = rng.next() % n_b;
            pairs.extend((0..n_a).map(|i| (i, j)));
        }
        for len in [1, SHARED_RUN - 1, SHARED_RUN, SHARED_RUN + 1, 40] {
            for _ in 0..4 {
                let (i, j) = (rng.next() % n_a, rng.next() % n_b);
                pairs.extend((0..len).map(|t| (i, (j + t) % n_b)));
                pairs.extend((0..len).map(|t| ((i + t) % n_a, j)));
            }
        }
        pairs
    }

    /// Shared-pattern groups: runs with one row on a side (`runs`), whose other
    /// rows cover every word, counter and tile class (`rows`, plus near
    /// copies of 1-64 bytes), with empty rows and patterns over 64 bytes
    /// inside runs, on owned rows and with borrowed NumPy rows on either side.
    #[test]
    fn shared_runs_equal_evaluate() {
        let mut rng = Rng(11);
        let (mut a, mut b) = rows(&mut rng);
        for _ in 0..40 {
            let m = 1 + rng.next() % 64;
            let x = rng.seq(m, ACGT);
            b.push(rng.edited(&x, 8, ACGT));
            a.push(x);
        }
        a.push(Vec::new());
        b.push(Vec::new());
        let owned =
            |rows: &[Vec<u8>]| SeqTable::new(SeqBatch::from_seqs(rows.iter().map(|x| &x[..])));
        let pairs = runs(&mut rng, a.len(), b.len());
        check(&owned(&a), &owned(&b), &pairs);
        let keep = |row: &Vec<u8>| row.last().is_none_or(|&x| x != 0);
        let a: Vec<_> = a.into_iter().filter(keep).collect();
        let b: Vec<_> = b.into_iter().filter(keep).collect();
        let pairs = runs(&mut rng, a.len(), b.len());
        check(&fixed_table(&a), &owned(&b), &pairs);
        check(&owned(&a), &fixed_table(&b), &pairs);
    }

    /// Rectangles: every cell against `evaluate`, in every mode and form,
    /// after one existing entry of `out`. Row and column lists of every word
    /// and counter class (`rows`, plus near copies of 1-64 bytes), with empty
    /// rows, patterns over 64 bytes, repeated rows, both orientations in
    /// `global`, too few patterns for shared-pattern groups, and borrowed NumPy
    /// rows on either side.
    #[test]
    fn rectangles_equal_evaluate() {
        let mut rng = Rng(13);
        let (mut a, mut b) = rows(&mut rng);
        for _ in 0..40 {
            let m = 1 + rng.next() % 64;
            let x = rng.seq(m, ACGT);
            b.push(rng.edited(&x, 8, ACGT));
            a.push(x);
        }
        a.push(Vec::new());
        b.push(Vec::new());
        let keep = |row: &Vec<u8>| row.last().is_none_or(|&x| x != 0);
        let a: Vec<_> = a.into_iter().filter(keep).collect();
        let b: Vec<_> = b.into_iter().filter(keep).collect();
        let owned =
            |rows: &[Vec<u8>]| SeqTable::new(SeqBatch::from_seqs(rows.iter().map(|x| &x[..])));
        let (oa, ob, fa, fb) = (owned(&a), owned(&b), fixed_table(&a), fixed_table(&b));
        let pick = |rng: &mut Rng, count: usize, n: usize| -> Vec<usize> {
            (0..count).map(|_| rng.next() % n).collect()
        };
        let mut shapes = Vec::new();
        for (r, c) in [
            (64, 64),
            (1, 300),
            (300, 1),
            (15, 40),
            (16, 40),
            (40, 15),
            (3, 200),
        ] {
            shapes.push((pick(&mut rng, r, a.len()), pick(&mut rng, c, b.len())));
        }
        // Short rows only, so every pattern takes a lane; then one row repeated.
        let short: Vec<_> = (0..a.len())
            .filter(|&i| (1..=64).contains(&a[i].len()))
            .collect();
        shapes.push((
            (0..64).map(|k| short[k % short.len()]).collect(),
            pick(&mut rng, 70, b.len()),
        ));
        shapes.push((vec![short[0]; 20], pick(&mut rng, 30, b.len())));
        for (ta, tb) in [(&oa, &ob), (&fa, &ob), (&oa, &fb)] {
            for mode in [Mode::Global, Mode::Prefix, Mode::Infix] {
                let kernel = Levenshtein::new(mode, &[]).unwrap();
                for (rows, cols) in &shapes {
                    let x: Vec<_> = rows.iter().map(|&i| ta.get(i)).collect();
                    let y: Vec<_> = cols.iter().map(|&j| tb.get(j)).collect();
                    for cutoff in [None, Some(-1), Some(0), Some(3), Some(20), Some(i32::MAX)] {
                        let want: Vec<_> = x
                            .iter()
                            .flat_map(|&x| y.iter().map(move |&y| (x, y)))
                            .map(|(x, y)| kernel.evaluate(x, y, cutoff).unwrap())
                            .collect();
                        for isa in Isa::ALL.into_iter().filter(|isa| isa.available()) {
                            let mut out = vec![Some(-7)];
                            rect_on(isa, &kernel, &x, &y, cutoff, &mut out).unwrap();
                            assert_eq!(out[0], Some(-7));
                            assert_eq!(out.len(), want.len() + 1);
                            for (k, (got, want)) in out[1..].iter().zip(&want).enumerate() {
                                let (i, j) = (rows[k / cols.len()], cols[k % cols.len()]);
                                assert_eq!(
                                    got,
                                    want,
                                    "{isa:?} {mode:?} {cutoff:?} cell ({i}, {j}) of {}x{}",
                                    rows.len(),
                                    cols.len()
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    /// Run detection: `a` is shared in `global` only, `b` in every mode (in
    /// `global` outside the runs of `a`), from `SHARED_RUN` pairs on;
    /// borrowed rows compare by table row, not by bytes.
    #[test]
    fn shared_runs_follow_the_mode_and_the_run_length() {
        let rows: Vec<Vec<u8>> = (0..40u8).map(|k| vec![b'A' + k % 4; 3]).collect();
        let owned = SeqTable::new(SeqBatch::from_seqs(rows.iter().map(|x| &x[..])));
        let fixed = fixed_table(&rows);
        for table in [&owned, &fixed] {
            let runs = |pairs: &[(usize, usize)], mode: Mode| {
                let views: Vec<_> = pairs
                    .iter()
                    .map(|&(i, j)| (table.get(i), table.get(j), None))
                    .collect();
                let mut runs = Vec::new();
                shared_runs(&views, mode, &mut runs);
                runs
            };
            // Row 2 against rows 0..20, then rows 0..20 against row 3.
            let mut pairs: Vec<_> = (0..20).map(|j| (2, j)).collect();
            pairs.extend((0..20).map(|i| (i, 3)));
            assert_eq!(runs(&pairs, Mode::Global), [(0, 20, 0), (20, 40, 1)]);
            assert_eq!(runs(&pairs, Mode::Prefix), [(20, 40, 1)]);
            assert_eq!(runs(&pairs[5..], Mode::Global), [(15, 35, 1)]);
            assert_eq!(runs(&pairs[4..], Mode::Infix), [(16, 36, 1)]);
            assert_eq!(runs(&pairs[4..24], Mode::Global), [(0, 16, 0)]);
            assert_eq!(runs(&pairs[5..24], Mode::Global), []);
            // Equal bytes in different rows are not one row.
            let pairs: Vec<_> = (0..20).map(|j| (4 * j % 40, j)).collect();
            assert_eq!(runs(&pairs, Mode::Global), []);
        }
    }

    #[test]
    fn words_follow_the_pattern_and_the_counter_range() {
        assert_eq!(word_of(1, 0), Some(0));
        assert_eq!(word_of(8, 255), Some(0));
        assert_eq!(word_of(8, 256), Some(1));
        assert_eq!(word_of(9, 0), Some(1));
        assert_eq!(word_of(16, 65_535), Some(1));
        assert_eq!(word_of(16, 65_536), Some(2));
        assert_eq!(word_of(32, 1 << 30), Some(2));
        assert_eq!(word_of(33, 0), Some(3));
        assert_eq!(word_of(64, 1 << 30), Some(3));
        assert_eq!(word_of(65, 0), None);
    }

    /// The process-wide hook: names, availability, the previous value, and an
    /// unchanged setting after a refusal. Outputs never depend on it, so other
    /// tests running meanwhile are unaffected.
    #[test]
    fn the_testing_hook_forces_and_restores() {
        let names: Vec<_> = available().collect();
        assert_eq!(&names[..2], ["scalar", "portable"]);
        let previous = set_isa(Some("scalar")).unwrap();
        assert_eq!(batch_width(), 1);
        assert_eq!(set_isa(Some("portable")).unwrap(), Some("scalar"));
        assert_eq!(batch_width(), BATCH);
        for bad in ["", "AVX2", "neon", "sse2"] {
            let message = set_isa(Some(bad)).unwrap_err();
            assert!(message.contains("'portable'"), "{message}");
        }
        if !Isa::Avx512.available() {
            assert!(set_isa(Some("avx512")).is_err());
        }
        assert_eq!(isa(), Isa::Portable);
        assert_eq!(set_isa(None).unwrap(), Some("portable"));
        assert_eq!(isa(), detected());
        set_isa(previous).unwrap();
    }
}
