//! K1 native `duplex(model="mfe")`: intermolecular hybridization MFE on the CPU.
//! Spec: docs/wiki/thermo/duplex-mfe.md.
//!
//! The kernel evaluates whatever cells the compiled pack supplies (§4.5, §12.1):
//! every data choice (unit routing, small-loop compositions, Eq. 7 cells, salt
//! weights, rule F as ⊥ cells) lives in the pack that `harness/tools/gen_params.py`
//! compiles. Per call (host, once): T1's conditions checks, Na_eq and host constants
//! (`tm::Host`), G_e(T) of every base entry in exact rationals (§4.2), the §4.4 range
//! check and the cell keys. Per pair: the integer recurrence of §5 (D alone for the
//! score, (D, H, G, Q) for the opt-in fields) and T1's binary64 finalize (§7).
pub mod bound;
pub mod dp;
pub mod pack;
pub mod pf;
pub mod pf_bound;
#[path = "../params_k1_santalucia2004.rs"]
mod preset;
#[rustfmt::skip]
#[path = "../params_k1_nndb2024_dna.rs"]
mod preset_nndb2024_dna;
pub mod register;

use std::sync::{Arc, Mutex, RwLock};

use seqblaze_core::{
    detail::{
        canonical, try_reserve_exact, DetailColumns, DetailField, DetailRequest, FieldKind,
        CANONICAL_NAN,
    },
    kernel::{BoundOutcome, PairDistanceKernel},
    plan::BoundAvailability,
    sched::Workers,
    seq::{ByteView, SeqTable},
    sink::rowsum::{Dyadic, ExactSum},
    validate::EngineError,
    work::WorkSource,
};

use crate::{
    cr::Rational,
    tm::{finalize_pairs, Conditions, EnergyAtQ, Host, CODE},
};
use dp::{Band, Cells, Full, Key, SCRATCH};
pub use pack::Pack;
use pf::{PfConstants, F53};
pub use preset::SANTALUCIA2004;
pub use preset_nndb2024_dna::{NNDB2024_DNA, NNDB2024_DNA_PENDING};
pub use register::TmPairs;

/// D48 native thermodynamics limit, nt, on both strands.
pub const MAX_LENGTH: usize = 200;
/// The engine key of an empty structure space (§4.4, §7); INT32_MAX is the mask.
/// Provisional: QK9 (public dg_dcal INT32_MAX, dg_kcal RN(INT32_MAX/100), NaN rest).
pub const SENTINEL_KEY: i32 = i32::MAX - 1;
/// The call raises when B reaches this (§4.4).
pub const RANGE_LIMIT: i128 = (1 << 31) - 2;

/// Detail field indices: the opt-in fields of §7 (`dg_dcal` and `dg_kcal` come
/// from the key).
pub const DH_KCAL: usize = 0;
pub const DS_CAL_PER_K: usize = 1;
pub const TM_C: usize = 2;

pub static FIELDS: [DetailField; 3] = [
    DetailField {
        name: "dh_kcal",
        kind: FieldKind::Float64,
    },
    DetailField {
        name: "ds_cal_per_k",
        kind: FieldKind::Float64,
    },
    DetailField {
        name: "tm_c",
        kind: FieldKind::Float64,
    },
];

/// A generated preset (`params_k1_*.rs`): the gate while a build unit is missing,
/// else the build definition and the compiled pack text.
#[derive(Debug, Clone, Copy)]
pub struct Preset {
    pub name: &'static str,
    pub build_units: &'static [&'static str],
    pub missing_units: &'static [&'static str],
    pub definition: Option<&'static str>,
    pub definition_sha256: Option<&'static str>,
    pub pack: Option<&'static str>,
}

impl Preset {
    /// The loaded pack, parsed once per process at first use. The gate (§3.5): a preset
    /// whose `missing_units` is non-empty raises ParameterError "not yet available"
    /// naming them; that error comes from nowhere else.
    pub fn load(&'static self) -> Result<Arc<Pack>, EngineError> {
        // One parse per preset name and process.
        type Loaded = Vec<(&'static str, Result<Arc<Pack>, String>)>;
        static LOADED: Mutex<Loaded> = Mutex::new(Vec::new());
        if !self.missing_units.is_empty() {
            return Err(EngineError::Parameter(format!(
                "duplex params={:?} is not yet available: its build units {} are not accepted \
                 in the parameter ledger yet (docs/wiki/thermo/duplex-mfe.md §3.5)",
                self.name,
                self.missing_units.join(", ")
            )));
        }
        let Some(text) = self.pack else {
            // A generated file with every unit accepted always carries its pack (§12.2).
            return Err(EngineError::Parameter(format!(
                "duplex params={:?}: the generated preset has no compiled pack; re-run \
                 harness/tools/gen_params.py (docs/wiki/thermo/duplex-mfe.md §12.2)",
                self.name
            )));
        };
        let mut loaded = LOADED.lock().unwrap_or_else(|e| e.into_inner());
        if let Some((_, pack)) = loaded.iter().find(|(name, _)| *name == self.name) {
            return pack.clone().map_err(EngineError::Parameter);
        }
        let pack = Pack::parse(text).map(Arc::new).map_err(|e| e.to_string());
        loaded.push((self.name, pack.clone()));
        pack.map_err(EngineError::Parameter)
    }
}

/// The generated K1 preset `name` and its gate's pending items (nndb2024-dna.md §4.7);
/// `None` for an unknown name.
pub fn preset(name: &str) -> Option<(&'static Preset, &'static [&'static str])> {
    match name {
        "santalucia2004" => Some((&SANTALUCIA2004, &[])),
        "nndb2024-dna" => Some((&NNDB2024_DNA, NNDB2024_DNA_PENDING)),
        _ => None,
    }
}

/// Load a generated K1 preset by name: `None` for an unknown name; else the pack or, while
/// a build unit is missing or a pending item remains, the gate's ParameterError "not yet
/// available" naming both (nndb2024-dna.md §4.7). The "re-run gen_params" error of
/// [`Preset::load`] never fires while a pending item remains.
pub fn load_preset(name: &str) -> Option<Result<Arc<Pack>, EngineError>> {
    let (preset, pending) = preset(name)?;
    if !pending.is_empty() {
        return Some(Err(crate::params::gate_error(
            preset.name,
            preset.missing_units,
            pending,
        )));
    }
    Some(preset.load())
}

/// One call's pack (§12.4): host constants, G_e(T), the range bound and the cell keys.
/// Built once, shared read-only by every worker.
pub struct DuplexCall {
    pub pack: Arc<Pack>,
    pub na_eq: f64,
    pub s: f64,
    pub kappa4: f64,
    pub kappa1: f64,
    /// G_e(T) per base entry, dcal/mol.
    pub energies: Vec<i64>,
    /// B of §4.4.
    pub bound: i64,
    score: Cells<i64>,
    full: Cells<Full>,
}

/// The public call a per-call pack is built for: it names the call in the condition
/// and range messages (tm.md §13.6 step 1.6, PR4). The pack is the same for both.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Caller {
    /// `sb.duplex` (K1 and K3).
    Duplex,
    /// `sb.tm` on a `Pairs` input (the register pairing, tm.md §13).
    TmPairs,
}

impl Caller {
    fn label(self) -> &'static str {
        match self {
            Self::Duplex => "duplex",
            Self::TmPairs => "tm on Pairs",
        }
    }

    fn page(self) -> &'static str {
        match self {
            Self::Duplex => "docs/wiki/thermo/duplex-mfe.md §4.4",
            Self::TmPairs => "docs/wiki/thermo/tm.md §13.6",
        }
    }
}

/// The §4.4 range error. It names the condition and the limit, never the computed bound B
/// (nndb2024-dna.md §5.3.8).
fn range_error(caller: Caller, cond: &Conditions, na_eq: f64, overflow: bool) -> EngineError {
    let reach = if overflow {
        "an entry beyond int64"
    } else {
        "a structure bound B"
    };
    EngineError::Parameter(format!(
        "{}: temp_c={} with Na_eq={na_eq} mM is out of the representable range: the \
         per-call energy table reaches {reach} for a 200-nt structure, at or above \
         2^31 - 2 = 2147483646 ({}); use a temp_c or salt condition closer to the model's \
         range",
        caller.label(),
        cond.temp_c,
        caller.page()
    ))
}

impl DuplexCall {
    /// §9 pack step: T1 §3.1 checks, Na_eq, s, κ₄, κ₁, G_e(T), the range check.
    /// Runs in the IEEE default floating-point environment whatever the
    /// caller's state, which is restored on return; the shared host-constant
    /// step (`Host::new`) runs inside the same guard
    /// (text-nulls-fp-environment.md §7.3 item 2).
    pub fn new(pack: Arc<Pack>, cond: &Conditions) -> Result<Self, EngineError> {
        Self::new_for(Caller::Duplex, pack, cond)
    }

    /// [`Self::new`] for `caller`, which only names the call in the messages.
    pub fn new_for(
        caller: Caller,
        pack: Arc<Pack>,
        cond: &Conditions,
    ) -> Result<Self, EngineError> {
        seqblaze_core::fpenv::guarded(|| Self::build(caller, pack, cond))
    }

    fn build(caller: Caller, pack: Arc<Pack>, cond: &Conditions) -> Result<Self, EngineError> {
        let Host {
            na_eq,
            s,
            kappa4,
            kappa1,
        } = Host::new(caller.label(), cond, pack.salt_b, pack.gas_r, pack.naeq_mg)?;
        let t = Rational::from_f64(cond.temp_c);
        let s_exact = Rational::from_f64(s);
        // Saturated at 2^62: such an entry in any non-⊥ cell puts B above the limit.
        let cap = 1i64 << 62;
        let mut energies: Vec<i64> = Vec::new();
        try_reserve_exact(&mut energies, pack.entries.len(), "duplex entry table")?;
        let energy = EnergyAtQ::new(&t, &s_exact);
        energies.extend(pack.entries.iter().map(|e| {
            energy
                .entry(e.h, e.g, e.q)
                .to_i64()
                .filter(|v| v.unsigned_abs() < cap as u64)
                .unwrap_or(cap)
        }));
        let bound = pack::pack_bound(&pack, |id| i128::from(energies[id]));
        if bound >= RANGE_LIMIT {
            let overflow = energies.contains(&cap);
            return Err(range_error(caller, cond, na_eq, overflow));
        }
        let key = |cell: &pack::Cell| -> Full {
            match cell {
                None => Full::INF,
                Some(items) => items.iter().fold(Full::default(), |acc, &(id, mult)| {
                    let e = &pack.entries[id as usize];
                    let m = i64::from(mult);
                    acc.plus(Full {
                        d: energies[id as usize] * m,
                        h: e.h * m,
                        g: e.g * m,
                        q: e.q * m,
                    })
                }),
            }
        };
        // Per-call cell tables, reserved fallibly like the DP band (§12.4): at large Λ
        // GEN alone is (Λ + 1)² keys, and a failed reservation must raise, not abort.
        let full_tables = pack.tables.try_map(key, "duplex cell table")?;
        let score_tables = full_tables.try_map(
            |k: &Full| if k.finite() { k.d } else { i64::INF },
            "duplex cell table",
        )?;
        Ok(Self {
            score: Cells::new(&pack, score_tables),
            full: Cells::new(&pack, full_tables),
            na_eq,
            s,
            kappa4,
            kappa1,
            energies,
            bound: bound as i64,
            pack,
        })
    }
}

/// The tables of the device pack, in upload order (docs/wiki/gpu/duplex-cuda.md §4;
/// STATES, the 14th, from stage S7, nndb2024-dna.md §5.4.2).
pub const DEVICE_TABLES: [&str; 14] = [
    "INIT", "INIT_SYM", "END_L", "END_R", "STACK", "BULGE", "I11", "I12", "I21", "I22", "GEN",
    "LOOP_OUT", "LOOP_IN", "STATES",
];
/// The score pass runs in int64 from this B on (duplex-cuda.md §5.2, QKG2).
pub const WIDE_FROM: i64 = 1 << 26;
/// +∞ of the int32 score width (I = 2^29) and of the int64 width and the full key (2^60).
pub const INF_32: i64 = 1 << 29;
pub const INF_64: i64 = 1 << 60;

/// The per-call device pack (duplex-cuda.md §4): every cell key of the call in the flat
/// layouts of the CPU (the last index fastest), all tables concatenated in
/// [`DEVICE_TABLES`] order. `score` holds D, ⊥ as +∞ of the score width; `full` holds
/// (D, H, G, Q), ⊥ as (2^60, 0, 0, 0). GEN is dense over (Λ + 1)², +∞ outside its
/// shapes (never read). Float64 constants travel as bit patterns. Every table is
/// indexed by the pair type P (N_P = 4 or 6) and the loop class c (N_C = 1 or 3) as
/// nndb2024-dna.md §5.3.1; STATES holds 200 keys iff a STATES cell of the pack is
/// non-empty, else none (§5.4.2), so a format-1-shaped pack keeps the 13 tables, their
/// offsets and contents of pack format 1.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DevicePack {
    pub loop_max: usize,
    /// The int64 score width (B ≥ 2^26); otherwise int32 with +∞ = 2^29.
    pub wide: bool,
    /// N_P (4 or 6) and N_C (1 or 3).
    pub n_p: usize,
    pub n_c: usize,
    /// CLS dense over (Λ + 1)²: c at u·(Λ + 1) + v, 0 outside the GEN shapes.
    pub cls: Vec<u8>,
    /// Start of each table of [`DEVICE_TABLES`] in `score` and `full`, then the total.
    pub offsets: [usize; 15],
    pub score: Vec<i64>,
    pub full: Vec<[i64; 4]>,
    pub s_bits: u64,
    pub kappa4_bits: u64,
    pub kappa1_bits: u64,
}

impl DuplexCall {
    /// The device pack of this call (duplex-cuda.md §4, §8 item 4). Built only for a
    /// resolved device; reserved fallibly. Performs no floating-point operation (integer
    /// keys, the integer finiteness test, s, κ₄ and κ₁ as bit patterns), so it carries no
    /// guard; its callers hold one (text-nulls-fp-environment.md §7.3 items 2 and 3).
    pub fn device_pack(&self) -> Result<DevicePack, EngineError> {
        let wide = self.bound >= WIDE_FROM;
        let inf = if wide { INF_64 } else { INF_32 };
        let c = &self.full;
        debug_assert_eq!(
            c.pt,
            pack::pair_map(c.n_p == 6),
            "the device derives PT from N_P"
        );
        // STATES only when a STATES cell is non-empty (nndb2024-dna.md §5.4.2).
        let states: &[Full] = if c.no_states { &[] } else { &c.states };
        let tables: [&[Full]; 14] = [
            std::slice::from_ref(&c.init),
            std::slice::from_ref(&c.init_sym),
            &c.end_l,
            &c.end_r,
            &c.stack,
            &c.bulge,
            &c.i11,
            &c.i12,
            &c.i21,
            &c.i22,
            &c.gen,
            &c.loop_out,
            &c.loop_in,
            states,
        ];
        let mut offsets = [0usize; 15];
        for (k, table) in tables.iter().enumerate() {
            offsets[k + 1] = offsets[k] + table.len();
        }
        let total = offsets[14];
        let mut cls = Vec::new();
        try_reserve_exact(&mut cls, c.cls.len(), "duplex device pack")?;
        cls.extend_from_slice(&c.cls);
        let (mut score, mut full) = (Vec::new(), Vec::new());
        try_reserve_exact(&mut score, total, "duplex device pack")?;
        try_reserve_exact(&mut full, total, "duplex device pack")?;
        for key in tables.iter().flat_map(|table| table.iter()) {
            if key.finite() {
                score.push(key.d);
                full.push([key.d, key.h, key.g, key.q]);
            } else {
                score.push(inf);
                full.push([INF_64, 0, 0, 0]);
            }
        }
        Ok(DevicePack {
            loop_max: c.loop_max,
            wide,
            n_p: c.n_p,
            n_c: c.n_c,
            cls,
            offsets,
            score,
            full,
            s_bits: self.s.to_bits(),
            kappa4_bits: self.kappa4.to_bits(),
            kappa1_bits: self.kappa1.to_bits(),
        })
    }
}

/// The five fields of one pair or structure (§7); the sentinel for +∞.
#[derive(Debug, Clone, Copy)]
pub struct DuplexRow {
    pub dg_dcal: i32,
    pub dg_kcal: f64,
    pub dh_kcal: f64,
    pub ds_cal_per_k: f64,
    pub tm_c: f64,
}

/// Letter codes of a validated strand (upper-case ACGT, at most 200 nt).
fn codes(view: ByteView<'_>, out: &mut [u8; MAX_LENGTH]) -> usize {
    let len = view.len();
    debug_assert!(len <= MAX_LENGTH);
    match view.as_slice() {
        Some(bytes) => out[..len].copy_from_slice(bytes),
        None => view.copy_to(0, &mut out[..len]),
    }
    for byte in &mut out[..len] {
        *byte = CODE[usize::from(*byte)];
    }
    len
}

/// Letter codes of a validated strand given as bytes.
fn slice_codes(bytes: &[u8], out: &mut [u8; MAX_LENGTH]) -> usize {
    let len = bytes.len();
    for (slot, &byte) in out.iter_mut().zip(bytes) {
        *slot = CODE[usize::from(byte)];
    }
    len
}

/// identical(a, b) (§3.4, κ₁) and SYM: identical and a palindrome (a = revcomp(a)).
/// Provisional: QK12 (byte-identical strands; SYM only for an identical palindrome).
fn symmetry(a: &[u8], b: &[u8]) -> (bool, bool) {
    let identical = a == b;
    let n = a.len();
    let palindrome = identical && (0..n).all(|k| a[k] + a[n - 1 - k] == 3);
    (identical, palindrome)
}

impl DuplexCall {
    /// The fields of a finite key, or the sentinel (§7). Expects the IEEE default
    /// floating-point environment: it is called per pair from guarded scheduler
    /// units and from the guarded [`Self::pair`] and [`Self::evaluate_structure`],
    /// and carries no guard of its own (text-nulls-fp-environment.md §7.3 item 2).
    pub fn row(&self, key: Full, identical: bool) -> DuplexRow {
        if !key.finite() {
            // Provisional: QK9 (empty structure space; k >= 2 pairs, QK8).
            return DuplexRow {
                dg_dcal: i32::MAX,
                dg_kcal: f64::from(i32::MAX) / 100.0,
                dh_kcal: CANONICAL_NAN,
                ds_cal_per_k: CANONICAL_NAN,
                tm_c: CANONICAL_NAN,
            };
        }
        let kappa = if identical { self.kappa1 } else { self.kappa4 };
        // P = Q/2 is exact: Q < 2^53 (build check).
        let row = finalize_pairs(key.h, key.g, key.q as f64 / 2.0, key.d, self.s, kappa);
        DuplexRow {
            dg_dcal: row.dg_dcal,
            dg_kcal: row.dg_kcal,
            dh_kcal: row.dh_kcal,
            ds_cal_per_k: row.ds_cal_per_k,
            tm_c: row.tm_c,
        }
    }

    fn solve_score(&self, a: &[u8], b: &[u8], band: &mut Band<i64>) -> Result<i32, EngineError> {
        let (_, sym) = symmetry(a, b);
        let start = if sym {
            self.score.init_sym
        } else {
            self.score.init
        };
        let k = dp::solve(&self.score, start, a, b, band)?;
        Ok(if k.finite() { k as i32 } else { SENTINEL_KEY })
    }

    fn solve_full(
        &self,
        a: &[u8],
        b: &[u8],
        band: &mut Band<Full>,
    ) -> Result<(Full, bool), EngineError> {
        let (identical, sym) = symmetry(a, b);
        let start = if sym {
            self.full.init_sym
        } else {
            self.full.init
        };
        Ok((dp::solve(&self.full, start, a, b, band)?, identical))
    }

    /// Every field of one pair of validated strands (ASCII ACGT): the full-key pass.
    /// A public per-call entry point, not on the kernel's hot path: it runs in the
    /// IEEE default floating-point environment whatever the caller's state, which
    /// is restored on return (text-nulls-fp-environment.md §7.3 item 2).
    pub fn pair(&self, a: &[u8], b: &[u8]) -> Result<DuplexRow, EngineError> {
        seqblaze_core::fpenv::guarded(|| self.pair_in_e0(a, b))
    }

    fn pair_in_e0(&self, a: &[u8], b: &[u8]) -> Result<DuplexRow, EngineError> {
        if a.len() > MAX_LENGTH || b.len() > MAX_LENGTH {
            return Err(EngineError::Parameter(format!(
                "duplex strands have at most {MAX_LENGTH} nt"
            )));
        }
        let (mut x, mut y) = ([0u8; MAX_LENGTH], [0u8; MAX_LENGTH]);
        let n = slice_codes(a, &mut x);
        let m = slice_codes(b, &mut y);
        SCRATCH.with(|cell| {
            let (key, identical) =
                self.solve_full(&x[..n], &y[..m], &mut cell.borrow_mut().full)?;
            Ok(self.row(key, identical))
        })
    }

    /// The testing evaluator (§12.4): E(S) of a given structure through the same cell
    /// keys, STATES included at bulge(1) steps (nndb2024-dna.md §5.3.3). A list outside §2
    /// (k < 2, a pair out of range or not a pair of the pack's pair types, crossing,
    /// a step above Λ) or one that reads a ⊥ cell (rule F) raises ParameterError.
    /// Runs in the IEEE default floating-point environment whatever the caller's
    /// state, which is restored on return (text-nulls-fp-environment.md §7.3 item 2).
    pub fn evaluate_structure(
        &self,
        a: &[u8],
        b: &[u8],
        pairs: &[(i64, i64)],
    ) -> Result<DuplexRow, EngineError> {
        seqblaze_core::fpenv::guarded(|| self.evaluate_structure_in_e0(a, b, pairs))
    }

    fn evaluate_structure_in_e0(
        &self,
        a: &[u8],
        b: &[u8],
        pairs: &[(i64, i64)],
    ) -> Result<DuplexRow, EngineError> {
        let bad = |why: String| {
            EngineError::Parameter(format!(
                "duplex_evaluate: the pair list is not a structure of duplex-mfe.md §2: {why}"
            ))
        };
        for (input, seq) in [("a", a), ("b", b)] {
            if seq.len() > MAX_LENGTH || seq.iter().any(|c| !b"ACGT".contains(c)) {
                return Err(bad(format!(
                    "strand {input} must be upper-case ACGT of at most {MAX_LENGTH} nt"
                )));
            }
        }
        let x: Vec<u8> = a.iter().map(|&c| CODE[usize::from(c)]).collect();
        let y: Vec<u8> = b.iter().map(|&c| CODE[usize::from(c)]).collect();
        if pairs.len() < 2 {
            return Err(bad("k < 2".into()));
        }
        let mut list = Vec::with_capacity(pairs.len());
        for &(i, j) in pairs {
            let (Ok(i), Ok(j)) = (usize::try_from(i), usize::try_from(j)) else {
                return Err(bad(format!("pair ({i}, {j}) out of range")));
            };
            if i >= x.len() || j >= y.len() {
                return Err(bad(format!("pair ({i}, {j}) out of range")));
            }
            if self.full.pair(x[i], y[j]) == pack::NO_PAIR {
                // The pack's pair types (nndb2024-dna.md §5.3.1): Watson-Crick, plus G·T
                // under `pairs wc+gt`.
                return Err(bad(format!(
                    "pair ({i}, {j}) is not a pair of the pack's pair types"
                )));
            }
            list.push((i, j));
        }
        let lam = self.pack.loop_max;
        for w in list.windows(2) {
            let ((i, j), (i2, j2)) = (w[0], w[1]);
            if !(i2 > i && j2 < j) {
                return Err(bad(format!(
                    "pairs ({i}, {j}), ({i2}, {j2}) cross or repeat"
                )));
            }
            if (i2 - i - 1) + (j - j2 - 1) > lam {
                return Err(bad(format!(
                    "the loop between ({i}, {j}) and ({i2}, {j2}) exceeds Λ = {lam}"
                )));
            }
        }
        let (identical, sym) = symmetry(&x, &y);
        let cells = &self.full;
        let (first, last) = (list[0], list[list.len() - 1]);
        let mut key = if sym { cells.init_sym } else { cells.init };
        let mut parts = vec![
            cells.left_end(&x, &y, first.0, first.1),
            cells.right_end(&x, &y, last.0, last.1),
        ];
        for w in list.windows(2) {
            parts.push(cells.step(&x, &y, w[0], w[1]));
        }
        for part in parts {
            if !part.finite() {
                return Err(bad(
                    "rule F: a flank pair is complementary (a ⊥ cell)".into()
                ));
            }
            key = key.plus(part);
        }
        Ok(self.row(key, identical))
    }
}

/// The label of the per-call weight cache's reservations (duplex-mfe.md §8.6.5,
/// host-allocation-failures.md §13.2).
pub const LSE_WEIGHT_TABLE: &str = "lse weight table";

/// The per-call cache of W(D) for RowStats `lse_dg_kcal` (duplex-mfe.md §8.6.5): a dense
/// window of 16-byte records over [lo, lo + len), each W(D) as its F53 parts (M, E),
/// value M · 2^E, or (0, 0) while not yet computed (every W has M ≥ 2^52). Filled
/// lazily, one host weight per distinct D looked up; the window grows to the hull of
/// the keys seen and each new window is reserved exactly (16 bytes per entry) through
/// the D117 helper. W is a function of D alone and correctly rounded, so the cache
/// changes no value.
#[derive(Default)]
struct WeightWindow {
    lo: i64,
    entries: Vec<Dyadic>,
}

impl WeightWindow {
    fn get(&self, d: i64) -> Option<Dyadic> {
        let at = usize::try_from(d.checked_sub(self.lo)?).ok()?;
        self.entries.get(at).copied().filter(|term| term.m != 0)
    }

    fn store(&mut self, d: i64, term: Dyadic) -> Result<(), EngineError> {
        if self.entries.is_empty() {
            let mut entries = Vec::new();
            try_reserve_exact(&mut entries, 1, LSE_WEIGHT_TABLE)?;
            entries.push(term);
            *self = Self { lo: d, entries };
            return Ok(());
        }
        let hi = self.lo + self.entries.len() as i64 - 1;
        if (self.lo..=hi).contains(&d) {
            self.entries[(d - self.lo) as usize] = term;
            return Ok(());
        }
        // The hull of the old window and d; |D| < 2^31 (§4.4), so it fits.
        let (lo, top) = (self.lo.min(d), hi.max(d));
        let len = usize::try_from(top - lo + 1).map_err(|_| {
            EngineError::Device(format!(
                "unable to reserve {LSE_WEIGHT_TABLE}: the window [{lo}, {top}] exceeds the \
                 address space; no result was published"
            ))
        })?;
        let mut entries = Vec::new();
        try_reserve_exact(&mut entries, len, LSE_WEIGHT_TABLE)?;
        entries.resize(len, Dyadic::ZERO);
        let at = (self.lo - lo) as usize;
        entries[at..at + self.entries.len()].copy_from_slice(&self.entries);
        entries[(d - lo) as usize] = term;
        *self = Self { lo, entries };
        Ok(())
    }
}

/// K1's RowStats `lse_dg_kcal` state of one call (duplex-mfe.md §8.6): K3's constants
/// (exact R·T, c_T, the weight routine) and the call's weight cache.
struct LseTerms {
    constants: Arc<PfConstants>,
    cache: RwLock<WeightWindow>,
}

impl LseTerms {
    /// tᵢⱼ of §8.6.1 for a pair's engine key: exactly 0 for the sentinel (LK2), else
    /// W(D) as (M, E) (LK1), from the cache or computed once and stored. Concurrent
    /// workers that miss the same D compute the same bits.
    fn term(&self, key: i32) -> Result<Dyadic, EngineError> {
        if key == SENTINEL_KEY {
            return Ok(Dyadic::ZERO);
        }
        let d = i64::from(key);
        let cached = self.cache.read().unwrap_or_else(|e| e.into_inner()).get(d);
        if let Some(term) = cached {
            return Ok(term);
        }
        let (m, e) = self.constants.weight(d)?.parts();
        let term = Dyadic { m, e };
        let mut cache = self.cache.write().unwrap_or_else(|e| e.into_inner());
        if cache.get(d).is_none() {
            cache.store(d, term)?;
        }
        Ok(term)
    }
}

/// The pair kernel of one call: the per-call pack, the requested opt-in fields and,
/// for RowStats `lse_dg_kcal` (§8.6), the row-term state.
pub struct Duplex {
    call: Arc<DuplexCall>,
    request: DetailRequest,
    lse: Option<LseTerms>,
    /// The call's passing C1 certificate (duplex-mfe-bound.md §5), if any.
    bound: Option<Arc<bound::Certificate>>,
}

impl Duplex {
    /// `fields` by name, of [`FIELDS`] (the facade checks the public names first).
    pub fn new(call: Arc<DuplexCall>, fields: &[&str]) -> Result<Self, EngineError> {
        let mut request = DetailRequest::NONE;
        for name in fields {
            let index = FIELDS
                .iter()
                .position(|field| field.name == *name)
                .ok_or_else(|| {
                    EngineError::Parameter(format!("duplex has no detail field {name:?}"))
                })?;
            request = request.with(index);
        }
        Ok(Self {
            call,
            request,
            lse: None,
            bound: None,
        })
    }

    /// The C1 bound of this call (duplex-mfe-bound.md §5, §6): kept only when its
    /// certificate passes, so a failed certificate leaves every path as before.
    pub fn with_bound(mut self, certificate: Arc<bound::Certificate>) -> Self {
        self.bound = certificate.pass.then_some(certificate);
        self
    }

    /// RowStats `lse_dg_kcal` (duplex-mfe.md §8.6.5): with `constants` (built by
    /// [`PfConstants::new_for`] with K1's messages for this call), `row_terms()` holds,
    /// `evaluate_term` gives each pair's MFE Boltzmann weight and `row_value` K3's
    /// finalize. Without it every path and byte is as before.
    pub fn with_lse(mut self, constants: Arc<PfConstants>) -> Self {
        self.lse = Some(LseTerms {
            constants,
            cache: RwLock::new(WeightWindow::default()),
        });
        self
    }

    /// (LB, χ) of the C1 bound for one pair in the given orientation (duplex-mfe-bound.md
    /// §1; testing hooks and measurements): `None` without a passing certificate.
    pub fn bound_values(
        &self,
        a: ByteView<'_>,
        b: ByteView<'_>,
    ) -> Result<Option<(Option<i64>, bool)>, EngineError> {
        let Some(certificate) = self.bound.as_ref() else {
            return Ok(None);
        };
        Self::with_codes(a, b, |a, b| certificate.evaluate(&self.call, a, b)).map(Some)
    }

    /// The call's passing C1 certificate when the outcome bound applies (no
    /// `lse_dg_kcal` request; duplex-mfe-bound.md §6.1), for the device path.
    pub fn certificate(&self) -> Option<&Arc<bound::Certificate>> {
        self.bound.as_ref().filter(|_| self.lse.is_none())
    }

    /// The LSE constants of the call, when `lse_dg_kcal` was requested.
    pub fn lse_constants(&self) -> Option<&Arc<PfConstants>> {
        self.lse.as_ref().map(|lse| &lse.constants)
    }

    pub fn call(&self) -> &DuplexCall {
        &self.call
    }

    fn with_codes<R>(a: ByteView<'_>, b: ByteView<'_>, body: impl FnOnce(&[u8], &[u8]) -> R) -> R {
        let (mut x, mut y) = ([0u8; MAX_LENGTH], [0u8; MAX_LENGTH]);
        let n = codes(a, &mut x);
        let m = codes(b, &mut y);
        body(&x[..n], &y[..m])
    }

    fn push(&self, row: &DuplexRow, out: &mut DetailColumns) -> Result<(), EngineError> {
        for (field, value) in [
            (DH_KCAL, row.dh_kcal),
            (DS_CAL_PER_K, row.ds_cal_per_k),
            (TM_C, row.tm_c),
        ] {
            if out.wants(field) {
                out.push_f64(field, canonical(value))?;
            }
        }
        Ok(())
    }

    fn full_row(&self, a: &[u8], b: &[u8]) -> Result<(i32, DuplexRow), EngineError> {
        SCRATCH.with(|cell| {
            let (key, identical) = self.call.solve_full(a, b, &mut cell.borrow_mut().full)?;
            let score = if key.finite() {
                key.d as i32
            } else {
                SENTINEL_KEY
            };
            Ok((score, self.call.row(key, identical)))
        })
    }
}

impl PairDistanceKernel for Duplex {
    /// The C1 bound (§8.4; duplex-mfe-bound.md §6.1): available with a passing
    /// certificate unless RowStats `lse_dg_kcal` was requested (QB5).
    fn bound_availability(&self) -> BoundAvailability {
        if self.outcome_bound() {
            BoundAvailability::Available
        } else {
            BoundAvailability::Unavailable
        }
    }
    fn outcome_bound(&self) -> bool {
        self.bound.is_some() && self.lse.is_none()
    }
    /// Pruned (λ > θ), certified (χ = 1) or open (duplex-mfe-bound.md §1).
    fn bound_outcome(
        &self,
        a: ByteView<'_>,
        b: ByteView<'_>,
        threshold: Option<i32>,
    ) -> Result<BoundOutcome<i32>, EngineError> {
        let Some(certificate) = self.bound.as_ref() else {
            return Ok(BoundOutcome::Open);
        };
        seqblaze_core::plan::count_bound_eval();
        Self::with_codes(a, b, |a, b| {
            certificate.outcome(&self.call, a, b, threshold)
        })
    }
    /// (e, ĉ) of duplex-mfe-bound.md §6.6 step 2 and §6.6.1, from the pack's Λ and format.
    fn bound_cost(&self, n_a: usize, n_b: usize) -> (u64, u32) {
        bound::cost(self.call.pack.format, self.call.pack.loop_max, n_a, n_b)
    }
    fn name(&self) -> &'static str {
        "duplex"
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
        Some(MAX_LENGTH)
    }
    fn length_error(
        &self,
        input: &'static str,
        index: usize,
        length: usize,
        limit: usize,
    ) -> EngineError {
        EngineError::Sequence {
            message: format!(
                "duplex input {input} sequence {index} has length {length}, above the native \
                 thermodynamics limit {limit} nt (D48); nothing was computed"
            ),
            input,
            index,
            position: None,
            byte: None,
            n_bad: 1,
            bad_indices: vec![index],
        }
    }
    fn prepare_bound(
        &self,
        _a: &SeqTable,
        _b: &SeqTable,
        _workers: Workers,
    ) -> Result<(), EngineError> {
        Ok(())
    }
    fn lower_bound(&self, _a: &SeqTable, _b: &SeqTable, _i: usize, _j: usize) -> i32 {
        i32::MIN
    }
    /// n · m · (Λ + 1)(Λ + 2)/2 candidate evaluations, about a quarter of them
    /// reached (one cell in four pairs); the detail pass doubles it.
    fn pair_cost(&self, a: &SeqTable, b: &SeqTable, with_details: bool) -> usize {
        let lam = self.call.pack.loop_max;
        let cost = a
            .mean_len()
            .max(1)
            .saturating_mul(b.mean_len().max(1))
            .saturating_mul((lam + 1) * (lam + 2) / 2)
            / 4;
        let cost = cost.max(1);
        if with_details && !self.request.is_empty() {
            cost.saturating_mul(2)
        } else {
            cost
        }
    }
    /// The D-only score pass (§5); never `None` (there is no bound to prune with).
    fn evaluate(
        &self,
        a: ByteView<'_>,
        b: ByteView<'_>,
        _cutoff: Option<i32>,
    ) -> Result<Option<i32>, EngineError> {
        Self::with_codes(a, b, |a, b| {
            SCRATCH.with(|cell| {
                self.call
                    .solve_score(a, b, &mut cell.borrow_mut().score)
                    .map(Some)
            })
        })
    }
    fn detail_fields(&self) -> &'static [DetailField] {
        &FIELDS
    }
    fn detail_request(&self) -> DetailRequest {
        self.request
    }
    /// Without a keep limit every pair is kept, so one full-key pass gives the score
    /// and the fields; otherwise the D-only pass decides and kept pairs get the
    /// full-key pass (both give the same D). The fields depend on K* only; ties among
    /// structures of one pair are not counted or flagged (provisional: QK11).
    fn evaluate_detailed(
        &self,
        a: ByteView<'_>,
        b: ByteView<'_>,
        _prune: Option<i32>,
        keep: Option<i32>,
        out: &mut DetailColumns,
    ) -> Result<Option<i32>, EngineError> {
        Self::with_codes(a, b, |a, b| {
            if keep.is_none() {
                let (score, row) = self.full_row(a, b)?;
                self.push(&row, out)?;
                return Ok(Some(score));
            }
            let score =
                SCRATCH.with(|cell| self.call.solve_score(a, b, &mut cell.borrow_mut().score))?;
            if keep.is_none_or(|limit| score <= limit) {
                let (full, row) = self.full_row(a, b)?;
                debug_assert_eq!(full, score, "score and detail passes disagree");
                self.push(&row, out)?;
            }
            Ok(Some(score))
        })
    }
    /// The fields of a kept pair whose key `score` came from elsewhere (a device pass,
    /// or the strand-swapped pair of a symmetric-within mirror row, §6.4). The full-key
    /// pass must reproduce that key (duplex-cuda.md §5.5, QKG9); with the CPU's own keys
    /// the check cannot fire.
    fn details(
        &self,
        a: ByteView<'_>,
        b: ByteView<'_>,
        score: i32,
        out: &mut DetailColumns,
    ) -> Result<(), EngineError> {
        let (n, m) = (a.len(), b.len());
        Self::with_codes(a, b, |a, b| {
            let (key, row) = self.full_row(a, b)?;
            if key != score {
                return Err(EngineError::Device(format!(
                    "duplex: the details of a pair of lengths {n} and {m} were requested for \
                     key {score}, but its full-key pass gives key {key}; the backend's score \
                     and the detail pass disagree, so no result was produced; use device='cpu'"
                )));
            }
            self.push(&row, out)
        })
    }
    /// RowStats sums the pairs' MFE Boltzmann weights only when `lse_dg_kcal` was
    /// requested (§8.6.5).
    fn row_terms(&self) -> bool {
        self.lse.is_some()
    }
    /// The D-only score pass's key and tᵢⱼ of §8.6.1 (0 for the sentinel).
    fn evaluate_term(
        &self,
        a: ByteView<'_>,
        b: ByteView<'_>,
    ) -> Result<(i32, Dyadic), EngineError> {
        let lse = self.lse.as_ref().ok_or_else(|| {
            EngineError::Parameter("internal error: duplex model='mfe' has no row terms".into())
        })?;
        let key = Self::with_codes(a, b, |a, b| {
            SCRATCH.with(|cell| self.call.solve_score(a, b, &mut cell.borrow_mut().score))
        })?;
        Ok((key, lse.term(key)?))
    }
    /// `lse_dg_kcal` of §8.6.1: +∞ for a zero sum, else RN(c_T · g(RN₅₃(S))) with −0.0
    /// read as +0.0 (K3's finalize, duplex-pf.md §7.1, §7.2, §9.4).
    fn row_value(&self, sum: &ExactSum) -> f64 {
        let Some(lse) = self.lse.as_ref() else {
            return f64::NAN;
        };
        match sum.round53() {
            None => f64::INFINITY,
            Some((m, e)) => lse.constants.dg(F53::from_parts(m, e)),
        }
    }
}

/// K1's bound without a configured call: per call since C1 (duplex-mfe-bound.md §6.1),
/// so a call-free plan reports none.
pub const DUPLEX_BOUND: BoundAvailability = BoundAvailability::Unavailable;

#[cfg(test)]
mod preset_tests {
    use super::*;

    fn message(err: EngineError) -> String {
        match err {
            EngineError::Parameter(m) => m,
            other => panic!("expected ParameterError, got {other:?}"),
        }
    }

    /// §11.5 (B) 7: the gate error comes only from a preset whose `missing_units` is
    /// non-empty, through `Preset::load`, and names those units.
    #[test]
    fn gate_error_only_for_missing_units() {
        static GATED: Preset = Preset {
            name: "test-gated",
            build_units: &["unit-a", "unit-b"],
            missing_units: &["unit-b"],
            definition: None,
            definition_sha256: None,
            pack: None,
        };
        let m = message(GATED.load().unwrap_err());
        assert!(
            m.contains("not yet available") && m.contains("unit-b"),
            "{m}"
        );
        assert!(!m.contains("unit-a"), "{m}");

        // Every unit accepted but no pack: an error, never the gate's.
        static NO_PACK: Preset = Preset {
            name: "test-no-pack",
            build_units: &["unit-a"],
            missing_units: &[],
            definition: None,
            definition_sha256: None,
            pack: None,
        };
        let m = message(NO_PACK.load().unwrap_err());
        assert!(!m.contains("not yet available"), "{m}");

        // The real preset: gate open, its pack loads and carries the definition sha256.
        assert!(SANTALUCIA2004.missing_units.is_empty());
        let pack = SANTALUCIA2004.load().expect("santalucia2004 loads");
        assert_eq!(pack.name, "santalucia2004");
        let sha = SANTALUCIA2004.definition_sha256.expect("definition sha256");
        assert_eq!(pack.sha256, sha);
    }

    /// nndb2024-dna.md §5.5.2: the `nndb2024-dna` preset is open (stage S9) and its pack
    /// carries the definition sha256; the gate error stays, checked on a synthetic closed
    /// preset; an unknown name is `None`. Messages name presets, units and items only.
    #[test]
    fn nndb2024_dna_preset_loads() {
        assert!(
            NNDB2024_DNA.missing_units.is_empty(),
            "nndb2024-dna: missing units"
        );
        assert!(
            NNDB2024_DNA_PENDING.is_empty(),
            "nndb2024-dna: pending items"
        );
        match load_preset("nndb2024-dna").expect("known") {
            Ok(pack) => {
                assert!(pack.name == "nndb2024-dna", "nndb2024-dna: wrong pack name");
                let sha = NNDB2024_DNA
                    .definition_sha256
                    .expect("nndb2024-dna: definition sha256");
                assert!(
                    pack.sha256 == sha,
                    "nndb2024-dna: pack sha256 differs from the definition sha256"
                );
            }
            Err(e) => panic!("nndb2024-dna does not load: {}", message(e)),
        }
        let m = message(crate::params::gate_error(
            "test-closed",
            &["unit-b"],
            &["gate-opening"],
        ));
        assert!(
            m.contains("not yet available")
                && m.contains("test-closed")
                && m.contains("unit-b")
                && m.contains("gate-opening"),
            "{m}"
        );
        assert!(load_preset("other").is_none());
    }
}
