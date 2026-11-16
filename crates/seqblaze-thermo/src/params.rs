//! Compiled parameter sets. Values are generated from accepted ledger cells by
//! `harness/tools/gen_params.py`; nothing here is typed by hand.

/// An exact rational `num / den` from a printed decimal.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Ratio {
    pub num: i64,
    pub den: u64,
}

/// ΔH and ΔG37 of one entry, in dcal/mol (tm.md §2.1 rule 1).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Entry {
    pub dh: i64,
    pub dg: i64,
}

/// The T1 slots of a nearest-neighbour build definition (tm.md §2.1).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Params {
    pub name: &'static str,
    /// `model.params.sha256` (tm.md QT17).
    pub sha256: &'static str,
    /// The 10 Watson-Crick stack classes, bytewise order (ledger §4.5).
    pub classes: [&'static str; 10],
    pub stack_dh: [i64; 10],
    pub stack_dg: [i64; 10],
    /// Class of the step with top strand XY at index 4·code(X) + code(Y),
    /// code A=0, C=1, G=2, T=3 (rule 3).
    pub step_class: [usize; 16],
    pub init: Entry,
    pub terminal_at: Entry,
    pub symmetry: Entry,
    /// b: salt entropy per phosphate pair, cal/(mol·K) (rule 4).
    pub salt_b: Ratio,
    /// R, cal/(mol·K) (rule 6).
    pub gas_r: Ratio,
    /// c_mg of the sodium equivalent, mM^(1/2) (rule 5).
    pub naeq_mg: Ratio,
}

#[rustfmt::skip]
#[path = "params_santalucia2004.rs"]
mod santalucia2004;
pub use santalucia2004::{BUILD_DEFINITION, PARAMS_SHA256, SANTALUCIA2004};

/// A generated T1 preset beside `santalucia2004` (docs/wiki/thermo/nndb2024-dna.md §4.7):
/// the gate (build units not yet accepted, pending items) while it is closed, else the
/// build definition and the parameters of the preset's slot map.
#[derive(Debug, Clone, Copy)]
pub struct T1Preset {
    pub name: &'static str,
    pub build_units: &'static [&'static str],
    pub missing_units: &'static [&'static str],
    pub pending_items: &'static [&'static str],
    pub definition: Option<&'static [&'static str]>,
    pub params: Option<Params>,
}

impl T1Preset {
    /// The parameters, or the gate's ParameterError "not yet available" naming the missing
    /// units and the pending items whenever either list is non-empty.
    pub fn load(&'static self) -> Result<&'static Params, seqblaze_core::validate::EngineError> {
        if !self.missing_units.is_empty() || !self.pending_items.is_empty() {
            return Err(gate_error(
                self.name,
                self.missing_units,
                self.pending_items,
            ));
        }
        self.params.as_ref().ok_or_else(|| {
            // A generated file with an open gate always carries its parameters (§4.7).
            seqblaze_core::validate::EngineError::Parameter(format!(
                "params={:?}: the generated T1 preset has no parameters; re-run \
                 harness/tools/gen_params.py (docs/wiki/thermo/nndb2024-dna.md §4.7)",
                self.name
            ))
        })
    }
}

/// The closed gate's error of a preset (docs/wiki/thermo/nndb2024-dna.md §4.7), shared by
/// the T1 and K1 loaders: ParameterError "not yet available" naming the build units not
/// yet accepted and the pending items. Never a cell value.
pub fn gate_error(
    name: &str,
    missing: &[&str],
    pending: &[&str],
) -> seqblaze_core::validate::EngineError {
    let mut parts = Vec::new();
    if !missing.is_empty() {
        parts.push(format!(
            "build units not yet accepted in the parameter ledger: {}",
            missing.join(", ")
        ));
    }
    if !pending.is_empty() {
        parts.push(format!("pending items: {}", pending.join(", ")));
    }
    seqblaze_core::validate::EngineError::Parameter(format!(
        "params={name:?} is not yet available: {} (docs/wiki/thermo/nndb2024-dna.md §4.7)",
        parts.join("; ")
    ))
}

#[rustfmt::skip]
#[path = "params_nndb2024_dna.rs"]
mod nndb2024_dna;
pub use nndb2024_dna::NNDB2024_DNA;

/// The native T1 parameters of a `tm` preset by name: `None` for an unknown name, else the
/// parameters or the gate's error (nndb2024-dna.md §4.7).
pub fn native_params(
    name: &str,
) -> Option<Result<&'static Params, seqblaze_core::validate::EngineError>> {
    match name {
        "santalucia2004" => Some(Ok(&SANTALUCIA2004)),
        "nndb2024-dna" => Some(NNDB2024_DNA.load()),
        _ => None,
    }
}

/// ΔH (kcal/mol) and ΔS (cal/(mol·K)) of one compat slot, binary64.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Pair {
    pub dh: f64,
    pub ds: f64,
}

/// The compat build definition `biopython:DNA_NN3` (tm-biopython.md §2.1): every value
/// the binary64 nearest the printed ledger decimal (rule 1).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BiopythonParams {
    pub name: &'static str,
    /// `model.params.sha256` (§2.1 rule 4).
    pub sha256: &'static str,
    /// The 10 Watson-Crick stack classes, bytewise order (ledger §4.5).
    pub classes: [&'static str; 10],
    pub stack_dh: [f64; 10],
    pub stack_ds: [f64; 10],
    /// Class of the step with top strand XY at index 4·code(X) + code(Y) (rule 3).
    pub step_class: [usize; 16],
    /// Per A·T duplex end (`sl1998:init_at`).
    pub per_at: Pair,
    /// Per G·C duplex end (`sl1998:init_gc`).
    pub per_gc: Pair,
    /// R, cal/(mol·K) (`sl1998:gas_constant_r`).
    pub gas_r: f64,
    /// c_salt, cal/(mol·K) per phosphate pair (`sl1998:salt_ds_coef`).
    pub salt: f64,
    /// c_mg of the free-Mg²⁺ term, applied in mM (`va2001:naeq_mg_coef`).
    pub c_mg: f64,
}

#[rustfmt::skip]
#[path = "params_biopython_nn3.rs"]
mod biopython_nn3;
pub use biopython_nn3::{
    BIOPYTHON_NN3, BUILD_DEFINITION as BIOPYTHON_NN3_DEFINITION,
    PARAMS_SHA256 as BIOPYTHON_NN3_SHA256,
};

/// One compat-primer3 slot as integer tallies (tm-primer3.md §2.1 rule 1): ΔH in
/// 100 cal/mol and ΔS in 0.1 cal/(mol·K), τ = 10 × the printed decimal.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Tally {
    pub h: i32,
    pub s: i32,
}

/// The compat build definition `primer3:santalucia` (tm-primer3.md §2.1): integer
/// tallies for the 13 energy slots, binary64 coefficients nearest the printed decimals.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Primer3Params {
    pub name: &'static str,
    /// `model.params.sha256` (§2.1 rule 7).
    pub sha256: &'static str,
    /// The 10 Watson-Crick stack classes, bytewise order (ledger §4.5).
    pub classes: [&'static str; 10],
    pub stack: [Tally; 10],
    /// Class of the step with top strand XY at index 4·code(X) + code(Y) (rule 4).
    pub step_class: [usize; 16],
    /// Per duplex terminus whose base is A or T (`sl1998:init_at`).
    pub init_at: Tally,
    /// Per duplex terminus whose base is G or C (`sl1998:init_gc`).
    pub init_gc: Tally,
    /// Once for a self-complementary sequence (`sl1998:symmetry`).
    pub symmetry: Tally,
    /// c_salt, cal/(mol·K) per nearest-neighbour pair (`sl1998:salt_ds_coef`).
    pub salt: f64,
    /// R, cal/(mol·K) (`sl1998:gas_constant_r`).
    pub gas_r: f64,
    /// c_mg of the free-Mg²⁺ term, applied in mM (`va2001:naeq_mg_coef`).
    pub c_mg: f64,
}

#[rustfmt::skip]
#[path = "params_primer3.rs"]
mod primer3;
pub use primer3::{
    BUILD_DEFINITION as PRIMER3_DEFINITION, PARAMS_SHA256 as PRIMER3_SHA256, PRIMER3_SANTALUCIA,
};

#[cfg(test)]
mod t1_preset_tests {
    use super::*;
    use seqblaze_core::validate::EngineError;

    fn message(err: EngineError) -> String {
        match err {
            EngineError::Parameter(m) => m,
            other => panic!("expected ParameterError, got {other:?}"),
        }
    }

    /// nndb2024-dna.md §4.7: the gate names the missing units and the pending items, and an
    /// open gate without parameters is the "re-run" error, never the gate's (synthetic
    /// presets); the real `nndb2024-dna` preset is open since stage S9 (§5.5.2).
    #[test]
    fn t1_gate_names_missing_units_and_pending_items() {
        static GATED: T1Preset = T1Preset {
            name: "test-gated",
            build_units: &["unit-a", "unit-b"],
            missing_units: &[],
            pending_items: &["gate-opening"],
            definition: None,
            params: None,
        };
        let m = message(GATED.load().unwrap_err());
        assert!(
            m.contains("not yet available") && m.contains("gate-opening"),
            "{m}"
        );
        assert!(!m.contains("unit-a"), "{m}");
        static NO_PARAMS: T1Preset = T1Preset {
            name: "test-open",
            build_units: &["unit-a"],
            missing_units: &[],
            pending_items: &[],
            definition: None,
            params: None,
        };
        let m = message(NO_PARAMS.load().unwrap_err());
        assert!(!m.contains("not yet available"), "{m}");
        // nndb2024-dna.md §5.5.2: the real preset is open (stage S9). Messages name presets
        // only, never a parameter value.
        match native_params("nndb2024-dna").expect("known") {
            Ok(p) => {
                assert!(p.name == "nndb2024-dna", "nndb2024-dna: wrong preset name");
                assert!(
                    p.sha256.len() == 64 && p.sha256.bytes().all(|b| b.is_ascii_hexdigit()),
                    "nndb2024-dna: sha256 is not 64 hexadecimal characters"
                );
                assert!(
                    p.sha256 != SANTALUCIA2004.sha256,
                    "nndb2024-dna: sha256 equals santalucia2004's"
                );
            }
            Err(e) => panic!("nndb2024-dna does not load: {}", message(e)),
        }
        assert!(
            NNDB2024_DNA.missing_units.is_empty(),
            "nndb2024-dna: missing units"
        );
        assert!(
            NNDB2024_DNA.pending_items.is_empty(),
            "nndb2024-dna: pending items"
        );
        assert!(
            NNDB2024_DNA.definition.is_some(),
            "nndb2024-dna: no definition"
        );
        assert_eq!(
            native_params("santalucia2004")
                .expect("known")
                .unwrap()
                .name,
            "santalucia2004"
        );
        assert!(native_params("other").is_none());
    }
}
