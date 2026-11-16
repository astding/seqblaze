//! T1 `tm` `compat="biopython"`: Biopython 1.88 `Tm_NN(seq)` (bare call, `DNA_NN3`,
//! `saltcorr=5`, `selfcomp=False`) replayed from ledger units `sl1998` and `va2001`,
//! with a correctly rounded logarithm (QB2). Spec: docs/wiki/thermo/tm-biopython.md.
//!
//! Per call (host, once): B1-B10 of §4, one binary64 operation per statement, and the
//! two logarithms CR64(ln mon), CR64(ln kk) by the integer Ziv loop with coefficient 1
//! (`cr::ln(x, 1)`). Per sequence: S1-S6 and F1-F6 of §5 in binary64, in exactly that
//! order (stacks 5'->3'), no FMA (rustc never contracts; no `mul_add` here), one NaN
//! encoding. Operation order, literals and unit conversions follow Biopython's
//! `Bio/SeqUtils/MeltingTemp.py` at tag biopython-188 (BSD-3-Clause, read under D46;
//! Biopython contributors, `MeltingTemp` by Sebastian Bassi and Markus Piotrowski);
//! no value is taken from Biopython's tables.
use seqblaze_core::{
    detail::{canonical, DetailColumns, DetailField, DetailRequest, FieldKind},
    seq::{ByteView, SeqTable},
    unary::UnaryKernel,
    validate::EngineError,
};

use crate::{
    cr::{self, Rational},
    params::BiopythonParams,
    tm::{length_error, K0, MAX_LENGTH, MIN_LENGTH},
};

/// The only field of `compat="biopython"` (§6).
pub static FIELDS: [DetailField; 1] = [DetailField {
    name: "tm_c",
    kind: FieldKind::Float64,
}];

/// The binary64 `1e-3` of B5 (mM -> M, by multiplication; §2.2).
pub const MILLI: f64 = f64::from_bits(0x3F50_624D_D2F1_A9FC);
/// The binary64 `1e-9` of B8 (nM -> M, by multiplication; §2.2).
pub const NANO: f64 = f64::from_bits(0x3E11_2E0B_E826_D695);
/// F3: kcal -> cal (§2.2), a unit conversion read by the row and the device pack.
pub const KCAL_TO_CAL: f64 = 1000.0;

/// The per-call values the device row reads (tm-compat-cuda.md §3.3), in that page's
/// field order, every binary64 as its bit pattern. Integer copies and `to_bits` only:
/// building it performs no floating-point operation (text-nulls-fp-environment.md §7.3).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BiopythonDevicePack {
    /// (H[k], S[k]) of the step with top strand XY at k = 4·code(X) + code(Y).
    pub steps: [[u64; 2]; 16],
    /// (H_A, S_A, H_G, S_G): the per-terminus entries of S4 and S5.
    pub ends: [u64; 4],
    /// c_s (F1), λ_m (F1), ρ (F4), κ (F3), K₀ (F6).
    pub salt: u64,
    pub ln_mon: u64,
    pub rho: u64,
    pub kcal_to_cal: u64,
    pub k0: u64,
}

/// The seven `Tm_NN` keyword arguments of §3 (mM and nM), after the facade's mapping.
#[derive(Debug, Clone, Copy, PartialEq)]
#[allow(non_snake_case)]
pub struct Args {
    pub Na: f64,
    pub K: f64,
    pub Tris: f64,
    pub Mg: f64,
    pub dNTPs: f64,
    pub dnac1: f64,
    pub dnac2: f64,
}

/// The per-call pack (§13): B1-B10 and the 16-step lookup. Built once, shared
/// read-only by every worker.
#[derive(Debug, Clone)]
pub struct BiopythonPack {
    pub params: &'static BiopythonParams,
    /// B5: the Na-equivalent in mol/L.
    pub mon: f64,
    /// B8: dnac1 - dnac2/2 in mol/L.
    pub kk: f64,
    /// B7: CR64(ln mon), +inf for mon = +inf.
    pub ln_mon: f64,
    /// B10: CR64(ln kk).
    pub ln_k: f64,
    /// B10: RN(R * ln_k).
    pub rho: f64,
    /// (dH, dS) of step XY in canonical A,C,G,T order for the device export.
    steps: [(f64, f64); 16],
    /// CPU lookups in the local ASCII codes A,C,G,T = 0,1,3,2.
    local_steps: [(f64, f64); 16],
    /// S1-S5, indexed by the number of A/T termini (0, 1 or 2).
    end_terms: [(f64, f64); 3],
    /// F1, with both binary64 multiplications rounded in the specified order.
    salt_by_length: [f64; MAX_LENGTH + 1],
}

fn parameter(message: String) -> EngineError {
    EngineError::Parameter(message)
}

/// `x` in the layout of Python's `repr(float)` (`50.0`, `5e-324`, `1e+300`, `inf`, `nan`),
/// so that messages quote conditions as the facade does. Rust's `{:?}` already picks the
/// shortest round-trip digits and switches to exponent notation at the same bounds as
/// Python (|x| < 1e-4 or >= 1e16); only the exponent's sign and width differ. When two
/// equally short candidates are exactly equally close to `x`, Rust takes the upper one and
/// Python the even one, so the last digit can differ (e.g. 70368744177664.125 prints as
/// `...64.13`, Python `...64.12`). Both round-trip to `x`; the text is for reading only.
fn py_float(x: f64) -> String {
    if x.is_nan() {
        return "nan".to_owned();
    }
    let debug = format!("{x:?}");
    match debug.split_once('e') {
        None => debug,
        Some((mantissa, exponent)) => {
            let (sign, digits) = match exponent.strip_prefix('-') {
                Some(digits) => ('-', digits),
                None => ('+', exponent),
            };
            format!("{mantissa}e{sign}{digits:0>2}")
        }
    }
}

/// CR64(ln x) for a positive binary64 x (§4 B7, B10): +inf at +inf; `PrecisionCap`
/// is an internal error naming the constant, never a silent rounding.
fn ln1(x: f64, name: &str) -> Result<f64, EngineError> {
    if x == f64::INFINITY {
        return Ok(f64::INFINITY); // IEEE 754 §9.2
    }
    cr::ln(&Rational::from_f64(x), &Rational::from_ratio(1, 1)).map_err(|cap| {
        parameter(format!(
            "internal error: the correctly rounded {name} needs more than {} bits; nothing was \
             computed",
            cap.bits
        ))
    })
}

impl BiopythonPack {
    /// B1-B10 of §4. B6 and B9 raise `ParameterError`, B6 first (Biopython's order).
    /// Runs in the IEEE default floating-point environment whatever the
    /// caller's state, which is restored on return (text-nulls-fp-environment.md
    /// §7.3 item 2).
    pub fn new(params: &'static BiopythonParams, args: &Args) -> Result<Self, EngineError> {
        seqblaze_core::fpenv::guarded(|| Self::build(params, args))
    }

    fn build(params: &'static BiopythonParams, args: &Args) -> Result<Self, EngineError> {
        let named = [
            ("Na", args.Na),
            ("K", args.K),
            ("Tris", args.Tris),
            ("Mg", args.Mg),
            ("dNTPs", args.dNTPs),
            ("dnac1", args.dnac1),
            ("dnac2", args.dnac2),
        ];
        for (name, value) in named {
            // sb.Conditions admits only these; a direct binding call is checked too.
            if !value.is_finite() || value < 0.0 {
                return Err(parameter(format!(
                    "compat='biopython': {name}={} must be finite and >= 0",
                    py_float(value)
                )));
            }
        }
        let m1 = args.Na + args.K; // B1
        let m2 = args.Tris / 2.0; // B2: Tris+ = Tris/2 (QT12)
        let mut mon_mm = m1 + m2; // B3
        if args.dNTPs < args.Mg {
            // B4: free Mg2+, von Ahsen 2001 Eq. 7
            let f = args.Mg - args.dNTPs;
            let w = params.c_mg * f.sqrt();
            mon_mm += w;
        }
        let mon = mon_mm * MILLI; // B5
        if mon == 0.0 {
            // B6
            return Err(parameter(format!(
                "compat='biopython': the total ion concentration is zero (na_mM={}, k_mM={}, \
                 tris_mM={}, mg_mM={}, dntp_mM={} give Mon = {} M); Biopython's Tm_NN raises \
                 ValueError 'Total ion concentration of zero is not allowed in this method.' for \
                 the same call. Give a monovalent cation or free Mg2+ (mg_mM > dntp_mM), e.g. \
                 sb.Conditions(temp_c=37, na_mM=50, oligo_nM=(25, 25))",
                py_float(args.Na),
                py_float(args.K),
                py_float(args.Tris),
                py_float(args.Mg),
                py_float(args.dNTPs),
                py_float(mon)
            )));
        }
        let ln_mon = ln1(mon, "ln_mon")?; // B7
        let d2h = args.dnac2 / 2.0; // B8
        let kd = args.dnac1 - d2h;
        let kk = kd * NANO;
        if kk <= 0.0 {
            // B9 (also -0.0). kd > 0 with kk = 0 is an underflow of the product in M.
            let (why, hint) = if kd > 0.0 {
                (
                    format!(
                        "dnac1 - dnac2/2 = {} nM, and k = (dnac1 - dnac2/2)*1e-9 underflows \
                         to {} M",
                        py_float(kd),
                        py_float(kk)
                    ),
                    "Use larger concentrations, so that (dnac1 - dnac2/2)*1e-9 is a non-zero \
                     binary64, e.g. oligo_nM=(25, 25)",
                )
            } else {
                (
                    format!(
                        "dnac1 - dnac2/2 = {} nM (k = {} M) is not positive",
                        py_float(kd),
                        py_float(kk)
                    ),
                    "Use a pair (a, b) with b < 2a, e.g. oligo_nM=(25, 25)",
                )
            };
            return Err(parameter(format!(
                "compat='biopython': oligo_nM gives dnac1={} and dnac2={} nM, so {why}; \
                 Biopython's Tm_NN raises ValueError 'math domain error' (log of k) for the \
                 same call. {hint}",
                py_float(args.dnac1),
                py_float(args.dnac2)
            )));
        }
        let ln_k = ln1(kk, "ln_k")?; // B10
        let rho = params.gas_r * ln_k;
        let mut steps = [(0.0, 0.0); 16];
        for (index, &class) in params.step_class.iter().enumerate() {
            steps[index] = (params.stack_dh[class], params.stack_ds[class]);
        }
        let local_code = [0usize, 1, 3, 2];
        let local_steps =
            std::array::from_fn(|index| steps[local_code[index >> 2] * 4 + local_code[index & 3]]);
        let mut end_terms = [(0.0, 0.0); 3];
        for (n_at, entry) in end_terms.iter_mut().enumerate() {
            let (at, gc) = (n_at as f64, (2 - n_at) as f64);
            let (mut h, mut sig) = (0.0, 0.0); // S1, S2: the init entries are zero.
            let t = params.per_at.dh * at; // S4
            h += t;
            let t = params.per_at.ds * at;
            sig += t;
            let t = params.per_gc.dh * gc; // S5
            h += t;
            let t = params.per_gc.ds * gc;
            sig += t;
            *entry = (h, sig);
        }
        let mut salt_by_length = [0.0; MAX_LENGTH + 1];
        for (len, salt) in salt_by_length.iter_mut().enumerate().skip(MIN_LENGTH) {
            let n = (len - 1) as f64;
            let c0 = params.salt * n;
            *salt = c0 * ln_mon; // F1, independently rounded per length.
        }
        Ok(Self {
            params,
            mon,
            kk,
            ln_mon,
            ln_k,
            rho,
            steps,
            local_steps,
            end_terms,
            salt_by_length,
        })
    }

    /// The device pack of tm-compat-cuda.md §3.3: the bits this pack's `row` reads.
    pub fn device_pack(&self) -> BiopythonDevicePack {
        let p = self.params;
        let mut steps = [[0u64; 2]; 16];
        for (slot, &(dh, ds)) in steps.iter_mut().zip(&self.steps) {
            *slot = [dh.to_bits(), ds.to_bits()];
        }
        BiopythonDevicePack {
            steps,
            ends: [
                p.per_at.dh.to_bits(),
                p.per_at.ds.to_bits(),
                p.per_gc.dh.to_bits(),
                p.per_gc.ds.to_bits(),
            ],
            salt: p.salt.to_bits(),
            ln_mon: self.ln_mon.to_bits(),
            rho: self.rho.to_bits(),
            kcal_to_cal: KCAL_TO_CAL.to_bits(),
            k0: K0.to_bits(),
        }
    }

    /// `tm_c` of one validated sequence (upper-case ACGT, 2 ≤ L ≤ 200): §5 S1-S6, F1-F6.
    /// Expects the IEEE default floating-point environment: it is called per
    /// item from guarded scheduler units and carries no guard of its own
    /// (text-nulls-fp-environment.md §7.3 item 2).
    #[inline]
    pub fn row(&self, seq: &[u8]) -> f64 {
        let len = seq.len();
        // A,C,G,T map to 0,1,3,2; A/T have an even code. The input has
        // already been validated, and the mask proves every step index < 16.
        let first = (seq[0] >> 1) & 3;
        let last = (seq[len - 1] >> 1) & 3;
        let n_at = u8::from(first & 1 == 0) + u8::from(last & 1 == 0);
        // S1-S5: precomputed in the original operation order.
        let (mut h, mut sig) = self.end_terms[usize::from(n_at)];
        // S6: the stacks 5'->3', in the original floating-point order.
        let mut previous = first;
        for &byte in &seq[1..] {
            let code = (byte >> 1) & 3;
            let (dh, ds) = self.local_steps[usize::from(previous << 2 | code)];
            h += dh;
            sig += ds;
            previous = code;
        }
        sig += self.salt_by_length[len]; // F1, F2
        let a = KCAL_TO_CAL * h; // F3
        let den = sig + self.rho; // F4
        let q = a / den; // F5: IEEE quotient (±inf for den = ±0, P6)
        canonical(q - K0) // F6
    }
}

/// The unary kernel of one compat call: the pack, `tm_c` only.
pub struct TmBiopython<'p> {
    pack: &'p BiopythonPack,
}

impl<'p> TmBiopython<'p> {
    /// `fields` by name; only `tm_c` exists under compat (§6).
    pub fn new(pack: &'p BiopythonPack, fields: &[&str]) -> Result<Self, EngineError> {
        if let Some(name) = fields.iter().find(|name| **name != "tm_c") {
            return Err(parameter(format!(
                "field {name:?} is not available under compat='biopython': it has fields=['tm_c'] only"
            )));
        }
        Ok(Self { pack })
    }
}

impl UnaryKernel for TmBiopython<'_> {
    type Row = f64;
    type Cutoff = f64;
    type Scratch = ();

    fn scratch(&self) -> Result<(), EngineError> {
        Ok(())
    }

    fn name(&self) -> &'static str {
        "tm"
    }

    fn length_limits(&self) -> (usize, usize) {
        (MIN_LENGTH, MAX_LENGTH)
    }

    fn length_error(&self, index: usize, length: usize, limit: usize) -> EngineError {
        length_error(index, length, limit)
    }

    fn item_cost(&self, table: &SeqTable) -> usize {
        // Two dependent binary64 additions per base plus the finalize.
        table.mean_len() + 16
    }

    fn columns(&self) -> Result<DetailColumns, EngineError> {
        DetailColumns::try_new(&FIELDS, DetailRequest::NONE.with(0))
    }

    #[inline]
    fn evaluate(&self, _: &mut (), seq: ByteView<'_>) -> f64 {
        match seq.as_slice() {
            Some(bytes) => self.pack.row(bytes),
            None => {
                let mut buffer = [0u8; MAX_LENGTH];
                let bytes = &mut buffer[..seq.len()];
                seq.copy_to(0, bytes);
                self.pack.row(bytes)
            }
        }
    }

    #[inline]
    fn passes(&self, row: &f64, cutoff: f64) -> bool {
        *row >= cutoff
    }

    #[inline]
    fn push(&self, row: &f64, out: &mut DetailColumns) -> Result<(), EngineError> {
        out.push_f64(0, *row)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::params::BIOPYTHON_NN3;

    const BARE: Args = Args {
        Na: 50.0,
        K: 0.0,
        Tris: 0.0,
        Mg: 0.0,
        dNTPs: 0.0,
        dnac1: 25.0,
        dnac2: 25.0,
    };

    fn pack(args: Args) -> Result<BiopythonPack, EngineError> {
        BiopythonPack::new(&BIOPYTHON_NN3, &args)
    }

    #[test]
    fn literals_are_the_nearest_binary64() {
        assert_eq!(MILLI, 1e-3);
        assert_eq!(NANO, 1e-9);
    }

    #[test]
    fn mon_of_one_molar_gives_positive_zero() {
        let p = pack(Args { Na: 1000.0, ..BARE }).unwrap();
        assert_eq!(p.mon, 1.0);
        assert_eq!(p.ln_mon.to_bits(), 0);
    }

    #[test]
    fn mon_overflow_gives_minus_absolute_zero() {
        let p = pack(Args {
            Na: 1e308,
            K: 1e308,
            ..BARE
        })
        .unwrap();
        assert_eq!(p.ln_mon, f64::INFINITY);
        assert_eq!(p.row(b"ACGTTGCA"), -273.15);
    }

    #[test]
    fn errors_in_biopython_order() {
        let e = |a: Args| pack(a).unwrap_err().to_string();
        assert!(e(Args { Na: 0.0, ..BARE }).contains("Total ion"));
        assert!(e(Args {
            dnac1: 25.0,
            dnac2: 50.0,
            ..BARE
        })
        .contains("domain"));
        let both = e(Args {
            Na: 0.0,
            dnac2: 50.0,
            ..BARE
        });
        assert!(both.contains("Total ion") && !both.contains("domain"));
        // Mg = dNTP: no free Mg2+ (B4), so zero ions raise B6.
        assert!(e(Args {
            Na: 0.0,
            Mg: 2.0,
            dNTPs: 2.0,
            ..BARE
        })
        .contains("Total ion"));
        assert!(pack(Args {
            Na: 0.0,
            Mg: 3.0,
            ..BARE
        })
        .is_ok());
    }

    #[test]
    fn py_float_matches_python_repr() {
        for (x, want) in [
            (50.0, "50.0"),
            (0.0, "0.0"),
            (-0.0, "-0.0"),
            (0.1, "0.1"),
            (1e-4, "0.0001"),
            (1e-5, "1e-05"),
            (5e-324, "5e-324"),
            (2.4e-321, "2.4e-321"),
            (1e15, "1000000000000000.0"),
            (1e16, "1e+16"),
            (1.5e300, "1.5e+300"),
            (f64::INFINITY, "inf"),
            (f64::NEG_INFINITY, "-inf"),
            (f64::NAN, "nan"),
        ] {
            assert_eq!(py_float(x), want);
        }
    }

    #[test]
    fn extreme_condition_messages_stay_compact() {
        let e = |a: Args| pack(a).unwrap_err().to_string();
        // B9 by underflow: kd = 5e-324 nM is positive, only kk in M is 0.
        let under = e(Args {
            dnac1: 5e-324,
            dnac2: 0.0,
            ..BARE
        });
        assert!(under.contains("dnac1=5e-324") && under.contains("underflows to 0.0 M"));
        assert!(
            !under.contains("not positive") && under.len() < 400,
            "{under}"
        );
        // The hint follows the branch: the pair already has b < 2a here.
        assert!(
            under.contains("Use larger concentrations") && !under.contains("b < 2a"),
            "{under}"
        );
        // B9 with kd = 0 exactly.
        let zero = e(Args {
            dnac1: 1e300,
            dnac2: 2e300,
            ..BARE
        });
        assert!(zero.contains("dnac1=1e+300") && zero.contains("not positive"));
        assert!(zero.len() < 400, "{zero}");
        assert!(
            zero.contains("b < 2a") && !zero.contains("larger concentrations"),
            "{zero}"
        );
        // B6 with a subnormal Na: Mon = Na*1e-3 underflows.
        let b6 = e(Args {
            Na: 2.4e-321,
            ..BARE
        });
        assert!(b6.contains("na_mM=2.4e-321") && b6.len() < 400, "{b6}");
    }

    #[test]
    fn stack_order_is_not_revcomp_symmetric() {
        // §5 / E7: the stacks of revcomp(s) are summed in the other order. At least one
        // of these differs in the last bits; all are within 1e-12 °C.
        let p = pack(BARE).unwrap();
        let rc = |s: &[u8]| -> Vec<u8> {
            s.iter()
                .rev()
                .map(|&b| match b {
                    b'A' => b'T',
                    b'C' => b'G',
                    b'G' => b'C',
                    _ => b'A',
                })
                .collect()
        };
        let seqs: [&[u8]; 4] = [
            b"ACGTTGCAAGGTCCATGAGCTTAGCA",
            b"GATTACAGATTACAGGCCTTAAGC",
            b"CGTTGACCAGTAGGCATCGATTCAG",
            b"TTGCAGCATGACCGATAGCTAGGACTTAC",
        ];
        let mut differ = 0;
        for s in seqs {
            let (x, y) = (p.row(s), p.row(&rc(s)));
            assert!((x - y).abs() < 1e-12);
            differ += usize::from(x.to_bits() != y.to_bits());
        }
        assert!(differ >= 1);
    }

    #[test]
    fn device_pack_carries_the_row_bits() {
        let p = pack(Args {
            Na: 1e308,
            K: 1e308,
            ..BARE
        })
        .unwrap();
        let d = p.device_pack();
        let q = &BIOPYTHON_NN3;
        for (k, step) in d.steps.iter().enumerate() {
            let class = q.step_class[k];
            assert_eq!(step[0], q.stack_dh[class].to_bits());
            assert_eq!(step[1], q.stack_ds[class].to_bits());
        }
        let ends = [q.per_at.dh, q.per_at.ds, q.per_gc.dh, q.per_gc.ds];
        assert_eq!(d.ends, ends.map(f64::to_bits));
        assert_eq!(d.salt, q.salt.to_bits());
        assert_eq!(d.ln_mon, f64::INFINITY.to_bits());
        assert_eq!(d.rho, p.rho.to_bits());
        assert_eq!(f64::from_bits(d.kcal_to_cal), 1000.0);
        assert_eq!(d.k0, K0.to_bits());
    }

    #[test]
    fn fields_other_than_tm_c_are_rejected() {
        let p = pack(BARE).unwrap();
        assert!(TmBiopython::new(&p, &["tm_c"]).is_ok());
        assert!(TmBiopython::new(&p, &["dg_kcal"]).is_err());
    }
}
