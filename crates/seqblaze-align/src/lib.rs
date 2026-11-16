//! Affine-gap pairwise alignment (A1, A2): native local and global Gotoh
//! alignment in fixed point (scale 1000) with rule E ties, and the compat
//! presets emboss-water and emboss-needle.
//! Spec: docs/wiki/align/gotoh-align.md.
//!
//! Engine port (spec §8): `PairDistanceKernel` is <=-oriented over i32, so the
//! kernel reports the key -score_fx, and water's -FLT_MAX sentinel as
//! `SENTINEL_KEY` (INT32_MAX - 1). INT32_MAX stays the engine's own mask.
pub mod nuc44;

mod bound;
mod dp;
mod lanes;
mod scoring;
mod scratch;

use std::sync::{Arc, OnceLock};

use seqblaze_core::{
    detail::{DetailColumns, DetailField, DetailRequest, FieldKind},
    kernel::PairDistanceKernel,
    plan::BoundAvailability,
    sched::Workers,
    seq::{ByteView, SeqTable},
    validate::{Alphabet, EngineError},
    work::WorkSource,
};

pub use dp::Score;
pub use lanes::{set_width as set_lane_width, width as lane_width};
pub use scoring::{Substitution, COMPAT_LEGAL, EDNAFULL_LEGAL, SCALE, VALUE_RANGE};

use bound::{Blocks, TableTally};
use dp::Gaps;
use scoring::Tables;
use scratch::{reserve_total, with_inputs, with_work, Extent, Work};

/// Detail field indices (spec §3 order after the primary `score`).
pub const A_START: usize = 0;
pub const A_END: usize = 1;
pub const B_START: usize = 2;
pub const B_END: usize = 3;
pub const ALN_LEN: usize = 4;
pub const N_IDENTITY: usize = 5;
pub const N_SIMILARITY: usize = 6;
pub const N_GAPS: usize = 7;
pub const CIGAR: usize = 8;
pub const ALIGNED_A: usize = 9;
pub const ALIGNED_B: usize = 10;
/// Native local only: optimal local end cells (spec §3.1). Last in the
/// canonical order and outside the traceback predicate.
pub const N_ENDS: usize = 11;

const fn field_list(aligned: FieldKind) -> [DetailField; 12] {
    let int64 = FieldKind::Int64;
    let int32 = FieldKind::Int32;
    [
        DetailField {
            name: "a_start",
            kind: int64,
        },
        DetailField {
            name: "a_end",
            kind: int64,
        },
        DetailField {
            name: "b_start",
            kind: int64,
        },
        DetailField {
            name: "b_end",
            kind: int64,
        },
        DetailField {
            name: "aln_len",
            kind: int32,
        },
        DetailField {
            name: "n_identity",
            kind: int32,
        },
        DetailField {
            name: "n_similarity",
            kind: int32,
        },
        DetailField {
            name: "n_gaps",
            kind: int32,
        },
        DetailField {
            name: "cigar",
            kind: FieldKind::Utf8,
        },
        DetailField {
            name: "aligned_a",
            kind: aligned,
        },
        DetailField {
            name: "aligned_b",
            kind: aligned,
        },
        DetailField {
            name: "n_ends",
            kind: int64,
        },
    ]
}

/// Detail fields with text `aligned` rows.
pub static FIELDS: [DetailField; 12] = field_list(FieldKind::Utf8);
/// Detail fields with byte-string `aligned` rows (raw bytes: a 256 x 256
/// matrix or `check_bases=False`, spec Q12).
pub static FIELDS_BINARY: [DetailField; 12] = field_list(FieldKind::Binary);

/// Native traceback fields: at most (La + 1)(Lb + 1) DP cells per pair, with
/// La and Lb the longest sequence lengths of tables a and b (spec §9.2, QA6).
pub const TRACEBACK_MAX_CELLS: u64 = 1 << 26;
/// Compat traceback fields: every sequence of a and b, the certified range
/// (spec §9.4).
pub const COMPAT_TRACEBACK_MAX_LENGTH: usize = 8_192;
/// Compat presets, any fields: every sequence (spec §9.3).
pub const COMPAT_MAX_LENGTH: usize = 16_000;
/// What switching a compat call to native mode needs (spec §5.2, §5.3; QA2,
/// QA3, QA14). Shared with the facade's compat-domain message.
pub const NATIVE_SWITCH: &str = "Switching to native mode needs the input upper-cased \
     (U and X mapped to T and N) or a 256x256 matrix, because native matrix=\"EDNAFULL\" \
     accepts only the 15 upper-case letters ATGCSWRYKMBVHDN.";
/// Key of water's -FLT_MAX sentinel (spec §8).
pub const SENTINEL_KEY: i32 = i32::MAX - 1;
/// FLT_MAX as a double: 3.4028234663852886e38.
pub const FLT_MAX: f64 = f32::MAX as f64;

/// score_fx of an engine key, or None for the engine's mask (INT32_MAX):
/// INT32_MIN for the sentinel, else -key (spec §8, masks first).
pub fn score_fx_of_key(key: i32) -> Option<i32> {
    match key {
        i32::MAX => None,
        SENTINEL_KEY => Some(i32::MIN),
        key => Some(-key),
    }
}

/// `score` of a score_fx: one correctly rounded division, or -FLT_MAX for the
/// sentinel's INT32_MIN.
pub fn score_of_fx(fx: i32) -> f64 {
    if fx == i32::MIN {
        -FLT_MAX
    } else {
        f64::from(fx) / 1000.0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    Local,
    Global,
}

impl Mode {
    pub fn parse(name: &str) -> Option<Self> {
        match name {
            "local" => Some(Self::Local),
            "global" => Some(Self::Global),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Compat {
    Water,
    Needle,
}

impl Compat {
    pub fn parse(alias: &str) -> Option<Self> {
        match alias {
            "emboss-water" => Some(Self::Water),
            "emboss-needle" => Some(Self::Needle),
            _ => None,
        }
    }
    pub fn alias(self) -> &'static str {
        match self {
            Self::Water => "emboss-water",
            Self::Needle => "emboss-needle",
        }
    }
}

/// One call's configuration. Penalties and scores are fixed point (k = 1000 x),
/// already checked against the grid by the caller.
#[derive(Debug, Clone)]
pub struct Config<'f> {
    pub mode: Mode,
    pub compat: Option<Compat>,
    pub gap_open: i64,
    pub gap_extend: i64,
    /// Required for global mode; ignored in local mode.
    pub end_gaps: Option<bool>,
    pub substitution: Substitution,
    pub check_bases: bool,
    /// Detail-field names (not `score`).
    pub fields: &'f [&'f str],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Kind {
    Local,
    Global { end_gaps: bool },
    Water,
    Needle { end_gaps: bool },
}

/// The DP a device evaluates (align-cuda.md §4, §5.1).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeviceKind {
    Local,
    Global,
    Water,
    Needle,
}

/// Plain scoring data of one call for a device backend (align-cuda.md §4).
/// s(x, y) = `table[class[x] * n_classes + class[y]]` equals the kernel's
/// table entry for every byte pair; with `matched = Some((match, mismatch))`
/// it is `match` if x = y, else `mismatch`, and no class table is built.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DevicePack {
    pub kind: DeviceKind,
    /// Global and needle end gaps (false for the local kinds).
    pub end_gaps: bool,
    /// Fixed-point gap open and extend penalties (scale 1000), in [0, 2^30].
    pub go: i32,
    pub ge: i32,
    pub matched: Option<(i32, i32)>,
    /// Byte -> class; two bytes share a class iff their table rows are equal.
    /// Classes are numbered in increasing order of their smallest byte. All
    /// zero, with `n_classes` 0 and an empty table, when `matched` is Some.
    pub class: [u8; 256],
    pub n_classes: usize,
    /// Class table, row-major, `n_classes` x `n_classes`.
    pub table: Vec<i32>,
}

/// One call's kernel: scoring tables, mode and requested detail fields.
/// Reuse across calls is safe: the bound prefilter serves only the table
/// objects of the first preparation (see `prepare_bound`).
pub struct Align {
    kind: Kind,
    gaps: Gaps,
    tables: Arc<Tables>,
    alphabet: Alphabet,
    fields: &'static [DetailField],
    request: DetailRequest,
    /// Whether the requested fields need the reported alignment.
    trace: bool,
    /// `match`/`mismatch` scoring (fixed point): the device evaluates it by
    /// byte equality instead of a class table (align-cuda.md §4, QAG5).
    matched: Option<(i32, i32)>,
    /// Per-row bound tallies of the first tables `prepare_bound` saw.
    prepared: OnceLock<Prepared>,
    /// Symbol codes of the lane score pass, built on its first use.
    lookup: OnceLock<lanes::Lookup>,
}

/// Bound tallies of the tables one `prepare_bound` call saw, with their
/// identities: `lower_bound` uses them only for those same table objects.
struct Prepared {
    /// `SeqTable::id` of each table: process-unique and never reused, so a
    /// later table at the same address (a reused kernel) does not match.
    a: u64,
    b: u64,
    left: TableTally,
    /// None when `b` is `a` (within): the rows share one tally.
    right: Option<TableTally>,
}

fn parameter(message: impl Into<String>) -> EngineError {
    EngineError::Parameter(message.into())
}

fn thousands(value: impl std::fmt::Display) -> String {
    let digits = value.to_string();
    let mut out = String::with_capacity(digits.len() + digits.len() / 3);
    for (k, ch) in digits.chars().enumerate() {
        if k > 0 && (digits.len() - k) % 3 == 0 {
            out.push(',');
        }
        out.push(ch);
    }
    out
}

impl Align {
    /// Build a call's kernel. The facade reports user errors first, in the
    /// spec's order; these checks keep the kernel safe on its own.
    pub fn new(config: Config<'_>) -> Result<Self, EngineError> {
        let (go, ge) = (config.gap_open, config.gap_extend);
        if !(0..=VALUE_RANGE).contains(&go) || !(0..=VALUE_RANGE).contains(&ge) {
            return Err(parameter(format!(
                "align gap_open and gap_extend must lie in [0, 2^30] fixed-point units, got {go} and {ge}"
            )));
        }
        let kind = match (config.mode, config.compat) {
            (Mode::Local, None) => Kind::Local,
            (Mode::Global, None) => Kind::Global {
                end_gaps: config.end_gaps.ok_or_else(|| {
                    parameter("align(mode=\"global\") requires scoring.end_gaps (True or False)")
                })?,
            },
            (Mode::Local, Some(Compat::Water)) => Kind::Water,
            (Mode::Global, Some(Compat::Needle)) => Kind::Needle {
                end_gaps: config.end_gaps.unwrap_or(false),
            },
            (mode, Some(compat)) => {
                return Err(parameter(format!(
                    "align mode {mode:?} contradicts compat={:?}",
                    compat.alias()
                )))
            }
        };
        let compat = config.compat.is_some();
        let tables = if compat {
            if config.substitution != Substitution::Ednafull {
                return Err(parameter(format!(
                    "align compat presets are certified for matrix=\"EDNAFULL\" only; use native mode \
                     for other scoring. {NATIVE_SWITCH}"
                )));
            }
            let water = kind == Kind::Water;
            let go_ok = if water {
                (5_000..=100_000).contains(&go)
            } else {
                go > 5_000 && go <= 100_000
            };
            if go % 125 != 0 || ge % 125 != 0 || !go_ok || ge > go.min(10_000) {
                return Err(parameter(format!(
                    "align compat penalties outside the certified domain: gap_open {go}, gap_extend {ge} \
                     (units of 0.001); use native mode for other values. {NATIVE_SWITCH}"
                )));
            }
            Tables::shared_compat()
        } else {
            if let Substitution::MatchMismatch {
                matched,
                mismatched,
            } = config.substitution
            {
                for value in [matched, mismatched] {
                    if i64::from(value).abs() > VALUE_RANGE {
                        return Err(parameter(format!(
                            "align match/mismatch {value} exceeds 2^30 fixed-point units"
                        )));
                    }
                }
            }
            if let Substitution::Table(values) = &config.substitution {
                if let Some(value) = values.iter().find(|v| i64::from(**v).abs() > VALUE_RANGE) {
                    return Err(parameter(format!(
                        "align matrix entry {value} exceeds 2^30 fixed-point units"
                    )));
                }
            }
            Tables::shared_native(&config.substitution)?
        };
        let matrix = !matches!(config.substitution, Substitution::MatchMismatch { .. });
        if matrix && !config.check_bases {
            return Err(parameter(
                "align: check_bases=False applies to match/mismatch scoring only; a matrix defines its own alphabet",
            ));
        }
        let alphabet = if compat {
            Alphabet::Exact {
                legal: &COMPAT_LEGAL,
                describe: "the EDNAFULL letters ATGCSWRYKMBVHDN, U and X, in either case",
                hint: NATIVE_SWITCH,
            }
        } else {
            match config.substitution {
                Substitution::Ednafull => Alphabet::Exact {
                    legal: &EDNAFULL_LEGAL,
                    describe: "the 15 EDNAFULL letters ATGCSWRYKMBVHDN (upper case)",
                    hint: "To score other bytes, upper-case the input (U and X mapped to T and N) \
                           or pass a 256x256 matrix; the compat presets also accept lower case, U and X.",
                },
                Substitution::MatchMismatch { .. } => Alphabet::Acgt,
                Substitution::Table(_) => Alphabet::Bytes,
            }
        };
        let binary = matches!(config.substitution, Substitution::Table(_))
            || (matches!(config.substitution, Substitution::MatchMismatch { .. })
                && !config.check_bases);
        let fields: &'static [DetailField] = if binary { &FIELDS_BINARY } else { &FIELDS };
        let mut request = DetailRequest::NONE;
        for &name in config.fields {
            let index = fields
                .iter()
                .position(|field| field.name == name)
                .ok_or_else(|| parameter(format!("unknown align field {name:?}")))?;
            if index == N_ENDS && kind != Kind::Local {
                // The facade rejects it first, naming the selector (spec §3.1).
                return Err(parameter(
                    "align field \"n_ends\" exists in native mode=\"local\" only",
                ));
            }
            request = request.with(index);
        }
        let local = matches!(kind, Kind::Local | Kind::Water);
        let trace = (ALN_LEN..=ALIGNED_B).any(|field| request.contains(field))
            || (local && (A_START..=B_END).any(|field| request.contains(field)));
        let matched = match config.substitution {
            Substitution::MatchMismatch {
                matched,
                mismatched,
            } if !compat => Some((matched, mismatched)),
            _ => None,
        };
        Ok(Self {
            kind,
            gaps: Gaps { go, ge },
            tables,
            alphabet,
            fields,
            request,
            trace,
            matched,
            prepared: OnceLock::new(),
            lookup: OnceLock::new(),
        })
    }

    /// The device scoring pack of this call (align-cuda.md §4): the kind,
    /// fixed-point gaps, and either the equality form of `match`/`mismatch`
    /// scoring (no class map: `n_classes` 0, empty table) or the
    /// row-equivalence class map and class table built from the kernel's
    /// 256 x 256 table.
    pub fn device_pack(&self) -> DevicePack {
        let (kind, end_gaps) = match self.kind {
            Kind::Local => (DeviceKind::Local, false),
            Kind::Global { end_gaps } => (DeviceKind::Global, end_gaps),
            Kind::Water => (DeviceKind::Water, false),
            Kind::Needle { end_gaps } => (DeviceKind::Needle, end_gaps),
        };
        // Equality scoring reads no table: no class map is built (align-cuda.md §4).
        let (class, n_classes, table) = match self.matched {
            Some(_) => ([0u8; 256], 0, Vec::new()),
            None => scoring::row_classes(&self.tables.sub),
        };
        let narrow = |value: i64| i32::try_from(value).expect("gaps lie in [0, 2^30]");
        DevicePack {
            kind,
            end_gaps,
            go: narrow(self.gaps.go),
            ge: narrow(self.gaps.ge),
            matched: self.matched,
            class,
            n_classes,
            table,
        }
    }

    fn compat(&self) -> Option<Compat> {
        match self.kind {
            Kind::Water => Some(Compat::Water),
            Kind::Needle { .. } => Some(Compat::Needle),
            _ => None,
        }
    }

    /// Whether the requested fields need a traceback.
    pub fn needs_traceback(&self) -> bool {
        self.trace
    }

    fn score(&self, a: &[u8], b: &[u8], w: &mut Work) -> Result<Score, EngineError> {
        let (t, g) = (&self.tables, self.gaps);
        Ok(match self.kind {
            Kind::Local => Score::Fx(dp::local_score(a, b, t, g, w)?),
            Kind::Global { end_gaps } => Score::Fx(dp::global_score(a, b, t, g, end_gaps, w)?),
            Kind::Water => dp::water_score(a, b, t, g, w)?,
            Kind::Needle { end_gaps } => Score::Fx(dp::needle_score(a, b, t, g, end_gaps, w)?),
        })
    }

    /// The scalar `evaluate` of a pair the bound did not reject.
    fn scalar_key(&self, a: ByteView<'_>, b: ByteView<'_>) -> Result<i32, EngineError> {
        with_inputs(a, b, |a, b, w| Self::key(self.score(a, b, w)?))
    }

    /// The lane score pass's view of this call (spec §7).
    fn lane_call(&self) -> lanes::Call<'_> {
        let lookup = self.lookup.get_or_init(|| {
            let t = &self.tables;
            lanes::Lookup::new(self.matched, t.class, t.n_classes, &t.class_score)
        });
        let fill = match self.kind {
            Kind::Local => lanes::Fill::Local,
            Kind::Global { end_gaps } => lanes::Fill::Global { end_gaps },
            Kind::Water => lanes::Fill::Water,
            Kind::Needle { end_gaps } => lanes::Fill::Needle { end_gaps },
        };
        lanes::Call {
            fill,
            gaps: self.gaps,
            sigma: self.tables.sigma,
            lookup,
        }
    }

    /// Score and reported alignment (`w.ops`, `w.extent`).
    fn traceback(&self, a: &[u8], b: &[u8], w: &mut Work) -> Result<Score, EngineError> {
        let (t, g) = (&self.tables, self.gaps);
        Ok(match self.kind {
            Kind::Local => Score::Fx(dp::native_trace(a, b, t, g, true, false, w)?),
            Kind::Global { end_gaps } => {
                Score::Fx(dp::native_trace(a, b, t, g, false, end_gaps, w)?)
            }
            Kind::Water => dp::water_trace(a, b, t, g, w)?,
            Kind::Needle { end_gaps } => Score::Fx(dp::needle_trace(a, b, t, g, end_gaps, w)?),
        })
    }

    /// Engine key of a score (spec §8).
    fn key(score: Score) -> Result<i32, EngineError> {
        match score {
            Score::Fx(fx) => i32::try_from(-fx)
                .ok()
                .filter(|key| key.unsigned_abs() <= 1 << 30)
                .ok_or_else(|| {
                    EngineError::Overflow(format!(
                        "align score_fx {fx} lies outside +-2^30; the value-range preflight should have rejected this call"
                    ))
                }),
            Score::Sentinel => Ok(SENTINEL_KEY),
        }
    }

    /// Lower bound on the key of a pair of lengths m, n whose UB is `ub`:
    /// -(UB - delta) with end gaps, -UB otherwise (spec §6, §8).
    fn key_floor(&self, ub: i64, m: usize, n: usize) -> i64 {
        let mut bound = ub;
        if let Kind::Global { end_gaps: true } | Kind::Needle { end_gaps: true } = self.kind {
            bound -= bound::end_gap_delta(m, n, self.gaps.go, self.gaps.ge);
        }
        -bound
    }

    /// Whether the certified bound (spec §6) shows the key exceeds `limit`.
    /// Reads the rows in place (no copy), counting them per pair.
    fn rejects<S: Blocks>(
        &self,
        a: S,
        b: S,
        (m, n): (usize, usize),
        limit: i32,
        w: &mut Work,
    ) -> bool {
        let ub = bound::upper_bound(a, b, &self.tables, &mut w.counts);
        self.key_floor(ub, m, n) > i64::from(limit)
    }

    /// `rejects` for an engine pair, before its rows are copied.
    fn rejects_views(&self, a: ByteView<'_>, b: ByteView<'_>, limit: i32) -> bool {
        with_work(|w| self.rejects(a, b, (a.len(), b.len()), limit, w))
    }

    /// Native local score and `w.n_ends` from the linear-memory counting fill
    /// (spec §3.1), for `n_ends` without traceback fields.
    fn counted(&self, a: &[u8], b: &[u8], w: &mut Work) -> Result<i32, EngineError> {
        debug_assert_eq!(self.kind, Kind::Local);
        let (best, count) = dp::local_score_counted(a, b, &self.tables, self.gaps, w)?;
        w.n_ends = count;
        Self::key(Score::Fx(best))
    }

    /// Detail row of a kept pair whose key is `key`. `counted` says that
    /// `w.n_ends` already holds this pair's count (its score pass counted).
    fn details_in(
        &self,
        a: &[u8],
        b: &[u8],
        key: i32,
        w: &mut Work,
        out: &mut DetailColumns,
        counted: bool,
    ) -> Result<(), EngineError> {
        if self.trace {
            let traced = Self::key(self.traceback(a, b, w)?)?;
            if traced != key {
                // align-cuda.md §8 item 7 (QAG9): a detail row belongs to its
                // exact key; it cannot fire with the CPU's own keys.
                return Err(EngineError::Device(format!(
                    "align: the details of a pair of lengths {} and {} were requested for key \
                     {key}, but its traceback gives key {traced}; the backend's score and the \
                     traceback disagree, so no result was produced; use device='cpu'",
                    a.len(),
                    b.len()
                )));
            }
        } else {
            if out.wants(N_ENDS) && !counted {
                let traced = self.counted(a, b, w)?;
                if traced != key {
                    return Err(EngineError::Device(format!(
                        "align: n_ends of a pair of lengths {} and {} was requested for key \
                         {key}, but its local fill gives key {traced}; the backend's score and the \
                         detail pass disagree, so no result was produced; use device='cpu'",
                        a.len(),
                        b.len()
                    )));
                }
            }
            // Only global coords can be requested without a traceback.
            w.ops.clear();
            w.extent = Extent {
                a_start: 0,
                a_end: a.len(),
                b_start: 0,
                b_end: b.len(),
            };
        }
        self.write(a, b, w, out)
    }

    /// Append the requested fields of the alignment in `w`.
    fn write(
        &self,
        a: &[u8],
        b: &[u8],
        w: &mut Work,
        out: &mut DetailColumns,
    ) -> Result<(), EngineError> {
        let extent = w.extent;
        for (field, value) in [
            (A_START, extent.a_start),
            (A_END, extent.a_end),
            (B_START, extent.b_start),
            (B_END, extent.b_end),
        ] {
            if out.wants(field) {
                out.push_i64(field, value as i64)?;
            }
        }
        let t = &self.tables;
        let ops = &w.ops;
        let int32 = |value: usize| i32::try_from(value).expect("traceback caps bound the columns");
        if (ALN_LEN..=N_GAPS).any(|field| out.wants(field)) {
            let (mut i, mut j) = (extent.a_start, extent.b_start);
            let (mut identity, mut similarity, mut gaps) = (0usize, 0usize, 0usize);
            for &op in ops {
                match op {
                    b'M' => {
                        identity += usize::from(t.is_identity(a[i], b[j]));
                        similarity += usize::from(t.score(a[i], b[j]) > 0);
                        i += 1;
                        j += 1;
                    }
                    b'I' => {
                        gaps += 1;
                        i += 1;
                    }
                    _ => {
                        gaps += 1;
                        j += 1;
                    }
                }
            }
            for (field, value) in [
                (ALN_LEN, ops.len()),
                (N_IDENTITY, identity),
                (N_SIMILARITY, similarity),
                (N_GAPS, gaps),
            ] {
                if out.wants(field) {
                    out.push_i32(field, int32(value))?;
                }
            }
        }
        let text = &mut w.text;
        if out.wants(CIGAR) {
            text.clear();
            reserve_total(text, 2 * ops.len() + 24, "align cigar text", w.budget)?;
            let (mut i, mut j) = (extent.a_start, extent.b_start);
            let mut run: Option<(u8, usize)> = None;
            for &op in ops {
                let symbol = match op {
                    b'M' => {
                        let equal = t.is_identity(a[i], b[j]);
                        i += 1;
                        j += 1;
                        if equal {
                            b'='
                        } else {
                            b'X'
                        }
                    }
                    b'I' => {
                        i += 1;
                        b'I'
                    }
                    _ => {
                        j += 1;
                        b'D'
                    }
                };
                run = match run {
                    Some((current, count)) if current == symbol => Some((current, count + 1)),
                    Some((current, count)) => {
                        push_count(text, count, current);
                        Some((symbol, 1))
                    }
                    None => Some((symbol, 1)),
                };
            }
            if let Some((current, count)) = run {
                push_count(text, count, current);
            }
            out.push_str(CIGAR, text)?;
        }
        for (field, on_a) in [(ALIGNED_A, true), (ALIGNED_B, false)] {
            if !out.wants(field) {
                continue;
            }
            text.clear();
            reserve_total(text, ops.len(), "align aligned row", w.budget)?;
            let (mut i, mut j) = (extent.a_start, extent.b_start);
            for &op in ops {
                let (from_a, from_b) = match op {
                    b'M' => (true, true),
                    b'I' => (true, false),
                    _ => (false, true),
                };
                let shown = if on_a { from_a } else { from_b };
                text.push(match (shown, on_a) {
                    (false, _) => b'-',
                    (true, true) => t.display[a[i] as usize],
                    (true, false) => t.display[b[j] as usize],
                });
                i += usize::from(from_a);
                j += usize::from(from_b);
            }
            out.push_str(field, text)?;
        }
        if out.wants(N_ENDS) {
            let count = i64::try_from(w.n_ends).expect("n_ends <= m * n fits int64");
            out.push_i64(N_ENDS, count)?;
        }
        Ok(())
    }

    /// Spec §9.3–§9.5: the per-sequence limits of the compat presets (native
    /// mode has none; its traceback cap is the per-pair cell cap of §9.2).
    fn length_problem(&self, input: &'static str, index: usize, len: usize) -> Option<String> {
        let compat = self.compat()?;
        let alias = compat.alias();
        if len == 0 {
            return Some(format!(
                "align(compat=\"{alias}\"): sequence {index} of input {input} is empty; \
                 the {alias} preset needs non-empty sequences (EMBOSS rejects empty input). \
                 Nothing was computed. Use native mode, or remove the empty sequences. {NATIVE_SWITCH}"
            ));
        }
        if self.kind == (Kind::Needle { end_gaps: true }) && len < 2 {
            return Some(format!(
                "align(compat=\"{alias}\", end_gaps=True): sequence {index} of input {input} has length {len}; \
                 the {alias} preset with end gaps is certified only for sequences of at least 2 nt. \
                 Nothing was computed. Use end_gaps=False, native mode, or remove the short sequences. \
                 {NATIVE_SWITCH}"
            ));
        }
        if len > COMPAT_MAX_LENGTH {
            return Some(format!(
                "align(compat=\"{alias}\"): sequence {index} of input {input} has length {}; \
                 the {alias} preset is certified for at most {} nt per sequence. \
                 Nothing was computed. Use native mode, or move the long sequences to a separate call. \
                 {NATIVE_SWITCH}",
                thousands(len),
                thousands(COMPAT_MAX_LENGTH)
            ));
        }
        if self.trace && len > COMPAT_TRACEBACK_MAX_LENGTH {
            return Some(format!(
                "align(compat=\"{alias}\"): sequence {index} of input {input} has length {}; \
                 the {alias} preset's traceback fields (stats, cigar, aligned, and coords in local mode) \
                 are certified for at most {} nt per sequence. Nothing was computed. Omit the traceback \
                 fields, use native mode, or move the long sequences to a separate call. {NATIVE_SWITCH}",
                thousands(len),
                thousands(COMPAT_TRACEBACK_MAX_LENGTH)
            ));
        }
        None
    }

    /// Spec §9.2: native traceback fields need (La + 1)(Lb + 1) <= 2^26 DP
    /// cells, O(1) over the longest sequences (length, index) of a and b. The
    /// error names the first longest sequence of the table with the longer one.
    fn cell_cap_error(&self, longest: [(usize, usize); 2], shared: bool) -> Option<EngineError> {
        if !self.trace || self.compat().is_some() {
            return None;
        }
        let cap = u128::from(TRACEBACK_MAX_CELLS);
        let cells = (longest[0].0 as u128 + 1) * (longest[1].0 as u128 + 1);
        if cells <= cap {
            return None;
        }
        let side = usize::from(longest[1].0 > longest[0].0);
        let ((len, index), other) = (longest[side], longest[1 - side].0);
        let (input, other_input) = if side == 0 { ("a", "b") } else { ("b", "a") };
        let partner = if shared {
            format!(
                "within one table it pairs with itself, giving {} cells",
                thousands(cells)
            )
        } else {
            format!(
                "the longest sequence of input {other_input} has {} nt, giving {} cells",
                thousands(other),
                thousands(cells)
            )
        };
        let allowed = cap / (other as u128 + 1);
        let limit = if shared {
            format!("{} nt per sequence", thousands(cap.isqrt() - 1))
        } else if allowed == 0 {
            format!("no length in input {input} with that input {other_input}")
        } else {
            format!(
                "at most {} nt in input {input} with that input {other_input}",
                thousands(allowed - 1)
            )
        };
        Some(sequence_error(
            format!(
                "align: sequence {index} of input {input} has length {}; traceback fields (stats, cigar, \
                 aligned, and coords in local mode) allow at most (La + 1)(Lb + 1) <= 2^26 = {} DP cells \
                 per pair, where La and Lb are the longest sequence lengths of inputs a and b, and {partner}. \
                 The cap allows {limit}. Nothing was computed. Omit the traceback fields, or move the long \
                 sequences to a separate call.",
                thousands(len),
                thousands(cap)
            ),
            input,
            index,
        ))
    }

    /// Whole-table permitted lengths for metadata-only device validation.
    /// The exact diagnostic still comes from metadata_preflight/length_problem.
    pub fn metadata_length_range(&self) -> (usize, usize) {
        if self.compat().is_none() {
            return (0, isize::MAX as usize);
        }
        let minimum = if self.kind == (Kind::Needle { end_gaps: true }) {
            2
        } else {
            1
        };
        (
            minimum,
            if self.trace {
                COMPAT_TRACEBACK_MAX_LENGTH
            } else {
                COMPAT_MAX_LENGTH
            },
        )
    }

    /// The same sequence preflight over a device validation summary. `first`
    /// names the first row outside metadata_length_range in each original table;
    /// longest keeps the first maximum-length row, including unreferenced rows.
    /// No sequence bytes or synthetic host table are needed by these checks.
    pub fn metadata_preflight(
        &self,
        first: [Option<(usize, usize)>; 2],
        longest: [(usize, usize); 2],
        shared: bool,
    ) -> Result<(), EngineError> {
        for (side, input) in ["a", "b"].into_iter().enumerate() {
            if side == 1 && shared {
                continue;
            }
            if let Some((index, length)) = first[side] {
                if let Some(message) = self.length_problem(input, index, length) {
                    return Err(sequence_error(message, input, index));
                }
            }
        }
        if let Some(error) = self.cell_cap_error(longest, shared) {
            return Err(error);
        }
        let range = i128::from(VALUE_RANGE);
        if self.value_need(longest[0].0, longest[1].0) <= range {
            return Ok(());
        }
        let floor = self.value_need(0, 0);
        if floor > range {
            // Even two empty sequences exceed the range (this is the only way
            // empty tables fail): a scoring problem, so no sequence is named.
            return Err(EngineError::Parameter(format!(
                "align: with this scoring the value range is exceeded for any lengths: \
                 sigma + 2*gap_open = {} > 2^30 = {} (in units of 0.001; sigma is the largest \
                 |score| of the scoring, and the range needs sigma*(min(La, Lb) + 1) + 2*gap_open \
                 + gap_extend*(La + Lb) <= 2^30). Nothing was computed. Use a smaller gap_open \
                 or smaller scores.",
                thousands(floor),
                thousands(range)
            )));
        }
        // Name the first longest sequence of the table with the longer one.
        let side = usize::from(longest[1].0 > longest[0].0);
        let ((len, index), other) = (longest[side], longest[1 - side].0);
        let need = |l: usize| {
            if side == 0 {
                self.value_need(l, other)
            } else {
                self.value_need(other, l)
            }
        };
        let (mut lo, mut hi) = (0usize, len);
        while lo < hi {
            let mid = lo + (hi - lo).div_ceil(2);
            if need(mid) <= range {
                lo = mid;
            } else {
                hi = mid - 1;
            }
        }
        let (input, other_input) = if side == 0 { ("a", "b") } else { ("b", "a") };
        // need(0) > range: the other input's longest sequence alone exceeds
        // the range, so no length of this input is admissible beside it.
        let allows = if need(lo) <= range {
            format!("allows at most {} nt in input {input}", thousands(lo))
        } else {
            format!("admits no length in input {input}")
        };
        Err(sequence_error(
            format!(
                "align: sequence {index} of input {input} has length {}; with this scoring the \
                 value range (sigma*(min(La, Lb) + 1) + 2*gap_open + gap_extend*(La + Lb) <= 2^30, \
                 in units of 0.001) {allows} while the longest sequence of input {other_input} \
                 has {} nt. Nothing was computed. Use smaller scores or penalties, or move the \
                 long sequences to a separate call.",
                thousands(len),
                thousands(other)
            ),
            input,
            index,
        ))
    }

    /// sigma (min(La, Lb) + 1) + 2 go + ge (La + Lb), exact.
    fn value_need(&self, la: usize, lb: usize) -> i128 {
        let (la, lb) = (la as i128, lb as i128);
        i128::from(self.tables.sigma) * (la.min(lb) + 1)
            + 2 * i128::from(self.gaps.go)
            + i128::from(self.gaps.ge) * (la + lb)
    }
}

/// Append the decimal digits of `count`, then `op`, without allocating.
fn push_count(text: &mut Vec<u8>, mut count: usize, op: u8) {
    let mut digits = [0u8; 20]; // usize::MAX has 20 digits
    let mut start = digits.len();
    loop {
        start -= 1;
        digits[start] = b'0' + (count % 10) as u8;
        count /= 10;
        if count == 0 {
            break;
        }
    }
    text.extend_from_slice(&digits[start..]);
    text.push(op);
}

fn sequence_error(message: String, input: &'static str, index: usize) -> EngineError {
    EngineError::Sequence {
        message,
        input,
        index,
        position: None,
        byte: None,
        n_bad: 1,
        bad_indices: vec![index],
    }
}

impl PairDistanceKernel for Align {
    fn bound_availability(&self) -> BoundAvailability {
        BoundAvailability::Available
    }
    fn name(&self) -> &'static str {
        "align"
    }
    /// Any lengths pair; only structural length metadata is checked here.
    fn validate_pairs(
        &self,
        a: &SeqTable,
        b: &SeqTable,
        _work: &WorkSource,
    ) -> Result<(), EngineError> {
        a.validate_lengths()?;
        if !std::ptr::eq(a, b) {
            b.validate_lengths()?;
        }
        Ok(())
    }
    fn sequence_length_limit(&self) -> Option<usize> {
        None
    }
    /// Every align limit is checked by `sequence_preflight`.
    fn length_caps(&self) -> [Option<usize>; 2] {
        [None, None]
    }
    fn alphabet(&self) -> Alphabet {
        self.alphabet
    }
    /// Spec §9: the compat per-sequence limits over a, then b, in index order
    /// (non-empty, needle end gaps >= 2 nt, 16,000 nt, the traceback range);
    /// then, O(1) over the longest sequences, the native traceback cell cap and
    /// the native value range.
    fn sequence_preflight(&self, a: &SeqTable, b: &SeqTable) -> Result<(), EngineError> {
        let shared = std::ptr::eq(a, b);
        let mut longest = [(0usize, 0usize); 2];
        for (side, (input, table)) in [("a", a), ("b", b)].into_iter().enumerate() {
            if side == 1 && shared {
                longest[1] = longest[0];
                continue;
            }
            for index in 0..table.len() {
                let len = table.seq_len(index);
                if let Some(message) = self.length_problem(input, index, len) {
                    return Err(sequence_error(message, input, index));
                }
                if len > longest[side].0 {
                    longest[side] = (len, index);
                }
            }
        }
        self.metadata_preflight([None, None], longest, shared)
    }
    /// Tally every row's symbol classes once (spec §6, §8), so `lower_bound`
    /// costs O(|A| |B|) per pair. The first preparation of this kernel wins;
    /// `lower_bound` uses it only for the same table objects (by
    /// `SeqTable::id`). A kernel reused with other tables gets the trivial
    /// bound there, and `evaluate`'s own bound check prunes the same pairs, so
    /// outputs and counts do not change; only the prefilter's saving is lost.
    fn prepare_bound(
        &self,
        a: &SeqTable,
        b: &SeqTable,
        workers: Workers,
    ) -> Result<(), EngineError> {
        if self.prepared.get().is_some() {
            return Ok(());
        }
        let left = TableTally::build(a, &self.tables, workers)?;
        let right = if std::ptr::eq(a, b) {
            None
        } else {
            Some(TableTally::build(b, &self.tables, workers)?)
        };
        let _ = self.prepared.set(Prepared {
            a: a.id(),
            b: b.id(),
            left,
            right,
        });
        Ok(())
    }
    /// -(UB - delta) clamped to i32 (spec §8), from the prepared tallies: it
    /// rejects exactly the pairs `evaluate`'s own bound check rejects. Without
    /// a preparation for these tables it is trivial (i32::MIN).
    fn lower_bound(&self, a: &SeqTable, b: &SeqTable, i: usize, j: usize) -> i32 {
        let Some(prepared) = self.prepared.get() else {
            return i32::MIN;
        };
        let right = prepared.right.as_ref().unwrap_or(&prepared.left);
        if prepared.a != a.id()
            || prepared.b != b.id()
            || prepared.left.rows() != a.len()
            || right.rows() != b.len()
        {
            return i32::MIN;
        }
        let ub = bound::combine(prepared.left.row(i), right.row(j), &self.tables);
        let floor = self.key_floor(ub, a.seq_len(i), b.seq_len(j));
        // Lowering a lower bound keeps it valid, so clamp rather than wrap.
        floor.clamp(i64::from(i32::MIN), i64::from(i32::MAX)) as i32
    }
    /// About four byte-comparison units per DP cell; a traceback doubles it.
    fn pair_cost(&self, a: &SeqTable, b: &SeqTable, with_details: bool) -> usize {
        let cost = a
            .mean_len()
            .max(1)
            .saturating_mul(b.mean_len().max(1))
            .saturating_mul(4);
        if with_details && self.trace {
            cost.saturating_mul(2)
        } else {
            cost
        }
    }
    /// `None` only when the certified bound rejects the pair (counted as
    /// pruned); a computed key above the cutoff is returned as `Some`.
    fn evaluate(
        &self,
        a: ByteView<'_>,
        b: ByteView<'_>,
        cutoff: Option<i32>,
    ) -> Result<Option<i32>, EngineError> {
        if cutoff.is_some_and(|limit| self.rejects_views(a, b, limit)) {
            return Ok(None);
        }
        with_inputs(a, b, |a, b, w| Ok(Some(Self::key(self.score(a, b, w)?)?)))
    }
    /// A few lane groups per call, so pairs of similar lengths share a group.
    fn batch_width(&self) -> usize {
        match lanes::width() {
            0 => 1,
            width => width * lanes::GROUPS_PER_BATCH,
        }
    }
    /// The lane score pass (spec §7): pairs the bound keeps and whose lengths
    /// are both at least 2 are sorted by length and scored a lane group at a
    /// time; every other pair, and every group the pass declines, takes the
    /// scalar fill. Each entry equals `evaluate`'s.
    fn evaluate_many(
        &self,
        pairs: &[(ByteView<'_>, ByteView<'_>, Option<i32>)],
        out: &mut Vec<Option<i32>>,
    ) -> Result<(), EngineError> {
        let width = lanes::width();
        let base = out.len();
        let mut pending = Vec::new();
        seqblaze_core::detail::try_reserve_exact(&mut pending, pairs.len(), "align lane batch")?;
        seqblaze_core::detail::try_reserve_exact(out, pairs.len(), "align lane batch")?;
        for (k, &(a, b, cutoff)) in pairs.iter().enumerate() {
            if cutoff.is_some_and(|limit| self.rejects_views(a, b, limit)) {
                out.push(None);
            } else if width > 0 && lanes::eligible(a.len(), b.len()) {
                out.push(None);
                pending.push(k);
            } else {
                out.push(Some(self.scalar_key(a, b)?));
            }
        }
        if pending.is_empty() {
            return Ok(());
        }
        pending.sort_by_key(|&k| (pairs[k].1.len(), pairs[k].0.len()));
        let mut scores = Vec::new();
        seqblaze_core::detail::try_reserve_exact(&mut scores, pairs.len(), "align lane batch")?;
        scores.resize(pairs.len(), 0i64);
        let call = self.lane_call();
        for members in pending.chunks(width) {
            if lanes::score_group(&call, pairs, members, &mut scores)? {
                for &k in members {
                    out[base + k] = Some(Self::key(Score::Fx(scores[k]))?);
                }
            } else {
                for &k in members {
                    out[base + k] = Some(self.scalar_key(pairs[k].0, pairs[k].1)?);
                }
            }
        }
        Ok(())
    }
    fn detail_fields(&self) -> &'static [DetailField] {
        self.fields
    }
    fn detail_request(&self) -> DetailRequest {
        self.request
    }
    /// `n_ends` without traceback fields: the counting fill is the score pass.
    fn fused_details(&self) -> bool {
        !self.trace && self.request.contains(N_ENDS)
    }
    /// Without a keep limit every pair is kept, so one traceback fill gives
    /// the score and the details; otherwise a linear-memory score pass decides
    /// and only kept pairs are traced.
    fn evaluate_detailed(
        &self,
        a: ByteView<'_>,
        b: ByteView<'_>,
        prune: Option<i32>,
        keep: Option<i32>,
        out: &mut DetailColumns,
    ) -> Result<Option<i32>, EngineError> {
        if prune.is_some_and(|limit| self.rejects_views(a, b, limit)) {
            return Ok(None);
        }
        with_inputs(a, b, |a, b, w| {
            if self.trace && keep.is_none() {
                let key = Self::key(self.traceback(a, b, w)?)?;
                self.write(a, b, w, out)?;
                return Ok(Some(key));
            }
            // n_ends without traceback fields: the counting fill is the score
            // pass, so a kept pair needs no second fill (spec §3.1).
            let counting = !self.trace && out.wants(N_ENDS);
            let key = if counting {
                self.counted(a, b, w)?
            } else {
                Self::key(self.score(a, b, w)?)?
            };
            if keep.is_none_or(|limit| key <= limit) {
                self.details_in(a, b, key, w, out, counting)?;
            }
            Ok(Some(key))
        })
    }
    fn details(
        &self,
        a: ByteView<'_>,
        b: ByteView<'_>,
        score: i32,
        out: &mut DetailColumns,
    ) -> Result<(), EngineError> {
        with_inputs(a, b, |a, b, w| self.details_in(a, b, score, w, out, false))
    }
}
