//! T2 `hairpin` `compat="primer3"` bindings (docs/wiki/thermo/hairpin-primer3.md §12,
//! §12.1; docs/wiki/thermo/primer3-parameter-loader.md §7, §9.1): the opaque parameter
//! handle built from the sixteen verified files, the per-call pack (H1–H6), the unary
//! run, and the testing probes. Every GIL-released region starts with the host
//! floating-point guard (D99).
use std::sync::Arc;

use pyo3::{
    exceptions::PyTypeError,
    prelude::*,
    types::{PyBytes, PyDict},
};
use seqblaze_core::{fpenv::FpGuard, validate::EngineError};
use seqblaze_cuda::{P3Scalars, Primer3Call, UnaryCall};
use seqblaze_thermo::{
    hairpin::Args, p3files::P3DevicePack, HairpinPack, HairpinPrimer3, P3Tables,
};

use super::{
    error,
    tm::{run_unary, DeviceBuild},
    PySeqTable,
};

/// The CUDA backend's primer3 call from a thermodynamics device pack (bit copies).
pub(crate) fn primer3_call(
    pack: P3DevicePack,
    swap: bool,
    end1: bool,
    fields: [usize; 4],
) -> Primer3Call {
    let c = pack.scalars;
    Primer3Call {
        tables: pack.tables,
        ends: pack.ends,
        scalars: P3Scalars {
            t37: c.t37,
            phi: c.phi,
            fresh_h: c.fresh_h,
            fresh_s: c.fresh_s,
            s_cutoff: c.s_cutoff,
            tolerance: c.tolerance,
            margin: c.margin,
            alpha1: c.alpha1,
            at_h: c.at_h,
            at_s: c.at_s,
            omega_h: c.omega_h,
            omega_s: c.omega_s,
            k0: c.k0,
            kcal_div: c.kcal_div,
            init_h: c.init_h,
            init_s: c.init_s,
            shift: c.shift,
            sigma: c.sigma,
            t_u: c.t_u,
            q4: c.q4,
            q1: c.q1,
            theta_omega4: c.theta_omega4,
            theta_omega1: c.theta_omega1,
        },
        swap,
        end1,
        fields,
    }
}

/// A parsed primer3-py parameter set (loader §7): Python sees its name, set digest and
/// whether it is a testing set, never a value. No pickling, no buffer export.
#[pyclass(frozen, module = "seqblaze._core", name = "_Primer3Params")]
pub struct PyPrimer3Params {
    pub(crate) tables: Arc<P3Tables>,
}

#[pymethods]
impl PyPrimer3Params {
    #[getter]
    fn name(&self) -> &str {
        &self.tables.name
    }
    /// The set digest of loader §3.3 (`model.params.sha256`).
    #[getter]
    fn sha256(&self) -> &str {
        &self.tables.sha256
    }
    #[getter]
    fn synthetic(&self) -> bool {
        self.tables.synthetic
    }
    fn __repr__(&self) -> String {
        format!(
            "<primer3 parameter set {} (sha256 {}, 16 roles verified)>",
            self.tables.name, self.tables.sha256
        )
    }
    fn __reduce__(&self) -> PyResult<()> {
        Err(PyTypeError::new_err(
            "a primer3 parameter set cannot be pickled; it is read from the installed \
             primer3-py at run time (D63)",
        ))
    }
}

/// Build the handle from the sixteen files in role order E1…E16 (loader §4–§5). The
/// Python loader has located them and checked their pins (or, in testing builds, the
/// caller supplies synthetic bytes). Parse and content-fact failures raise
/// ParameterError naming the role and line or the fact, never file text.
#[pyfunction(signature=(files, name, sha256, synthetic))]
fn _primer3_params(
    py: Python<'_>,
    files: Vec<Bound<'_, PyBytes>>,
    name: &str,
    sha256: &str,
    synthetic: bool,
) -> PyResult<PyPrimer3Params> {
    let owned: Vec<Vec<u8>> = files.iter().map(|b| b.as_bytes().to_vec()).collect();
    let tables = py
        .detach(|| {
            // E0 for the GIL-released region (§12.1); `P3Tables::parse` holds its own.
            let _guard = FpGuard::new();
            let views: Vec<&[u8]> = owned.iter().map(Vec::as_slice).collect();
            P3Tables::parse(&views, name, sha256, synthetic)
        })
        .map_err(|e| error(py, e))?;
    Ok(PyPrimer3Params {
        tables: Arc::new(tables),
    })
}

/// One call's per-call constants H1–H6 (§4).
#[pyclass(frozen, module = "seqblaze._core", name = "_HairpinPrimer3Pack")]
pub struct PyHairpinPack {
    pack: HairpinPack,
}

#[pymethods]
impl PyHairpinPack {
    #[getter]
    fn mv(&self) -> f64 {
        self.pack.mv
    }
    #[getter]
    fn w(&self) -> f64 {
        self.pack.w
    }
    #[getter]
    fn mu(&self) -> f64 {
        self.pack.mu
    }
    #[getter]
    fn x_mu(&self) -> f64 {
        self.pack.x_mu
    }
    #[getter]
    fn ln_mu(&self) -> f64 {
        self.pack.ln_mu
    }
    #[getter]
    fn sigma(&self) -> f64 {
        self.pack.sigma
    }
    #[getter]
    fn t_u(&self) -> f64 {
        self.pack.t_u
    }
}

/// Build the pack from the §3 mapping's inputs (mM, °C): the mv sum, the Mg term, μ, x_μ,
/// the correctly rounded logarithm, σ and T_u. μ zero or infinite and x_μ = 0 raise
/// ParameterError in that order (§7 step 1.4).
#[pyfunction(signature=(na, k, tris, mg, dntp, temp_c))]
fn _hairpin_primer3_pack(
    py: Python<'_>,
    na: f64,
    k: f64,
    tris: f64,
    mg: f64,
    dntp: f64,
    temp_c: f64,
) -> PyResult<PyHairpinPack> {
    let args = Args {
        na,
        k,
        tris,
        mg,
        dntp,
        temp_c,
    };
    let pack = py
        .detach(|| {
            // E0 for the GIL-released region (§12.1); `HairpinPack::new` holds its own.
            let _guard = FpGuard::new();
            HairpinPack::new(&args)
        })
        .map_err(|e| error(py, e))?;
    Ok(PyHairpinPack { pack })
}

/// Run `hairpin` `compat="primer3"` over every sequence of `table` (Each or Hits;
/// `cutoff` compares `dg_kcal <= cutoff`). Returns `{columns, shape, report}` like
/// `_native_tm`; `found` is a bool column (mask false). `device` is a CUDA ordinal
/// (None for the CPU): the device pack is built after device resolution
/// (primer3-cuda.md §4, §9).
#[pyfunction(signature=(table, pack, handle, keep, cutoff, threads, fields, grain="auto", device=None, progress=None))]
#[allow(clippy::too_many_arguments)]
fn _native_hairpin_primer3<'py>(
    py: Python<'py>,
    table: PyRef<'py, PySeqTable>,
    pack: PyRef<'py, PyHairpinPack>,
    handle: PyRef<'py, PyPrimer3Params>,
    keep: &str,
    cutoff: Option<f64>,
    threads: usize,
    fields: Vec<String>,
    grain: &str,
    device: Option<u64>,
    progress: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
    let names: Vec<&str> = fields.iter().map(String::as_str).collect();
    let kernel = HairpinPrimer3::new(pack.pack, Arc::clone(&handle.tables), &names)
        .map_err(|e| error(py, e))?;
    let build = || -> Result<UnaryCall, EngineError> {
        let pack = kernel.device_pack()?;
        Ok(UnaryCall::Hairpin(Arc::new(primer3_call(
            pack, false, false, [0; 4],
        ))))
    };
    let device = device.map(|ordinal| (ordinal, Box::new(build) as DeviceBuild<'_>));
    run_unary(
        py,
        &kernel,
        &table,
        keep,
        cutoff,
        threads,
        grain,
        device,
        &["found"],
        progress,
    )
}

/// Testing builds: the backtrace pairs of one sequence, zero-based (i, j) (§12).
#[cfg(feature = "testing")]
#[pyfunction]
fn _testing_hairpin_probe(
    py: Python<'_>,
    seq: &[u8],
    pack: PyRef<'_, PyHairpinPack>,
    handle: PyRef<'_, PyPrimer3Params>,
) -> PyResult<Vec<(usize, usize)>> {
    if seq.is_empty()
        || seq.len() > seqblaze_thermo::hairpin::MAX_LENGTH
        || !seq.iter().all(|b| matches!(b, b'A' | b'C' | b'G' | b'T'))
    {
        return Err(error(
            py,
            EngineError::Parameter(
                "hairpin_probe takes one upper-case ACGT sequence of 1-60 nt".into(),
            ),
        ));
    }
    let kernel = HairpinPrimer3::new(pack.pack, Arc::clone(&handle.tables), &[])
        .map_err(|e| error(py, e))?;
    let seq = seq.to_vec();
    py.detach(|| {
        // E0 for the GIL-released region (§12.1): the probe recomputes the fill.
        let _guard = FpGuard::new();
        kernel.pairs(&seq)
    })
    .map_err(|e| error(py, e))
}

/// Testing builds: one entry (ΔH, ΔS) of a synthetic handle (loader §9.1). Tables
/// "nn", "mm", "m1", "m2" take (a, b, c, d), "d3", "d5" (a, c, d), "at" (p, q), codes
/// 0–4; "loop_in", "loop_bu", "loop_hp" take k in 1..=30; "b3", "b4" take the key string
/// and return None for an absent component. A handle built from the pins is refused.
/// Reads stored values only; no floating-point operation.
#[cfg(feature = "testing")]
#[pyfunction]
fn _testing_primer3_entry(
    py: Python<'_>,
    handle: PyRef<'_, PyPrimer3Params>,
    table: &str,
    key: &Bound<'_, PyAny>,
) -> PyResult<(Option<f64>, Option<f64>)> {
    use seqblaze_thermo::p3files::{at, bonus_index, idx3, idx4};
    let t = &handle.tables;
    let fail = |message: String| error(py, EngineError::Parameter(message));
    if !t.synthetic {
        return Err(fail(format!(
            "primer3_params_entry refuses the parameter set {}: only a synthetic set may be \
             read back (D76)",
            t.name
        )));
    }
    let codes = |n: usize| -> PyResult<Vec<u8>> {
        let values: Vec<i64> = key
            .extract()
            .map_err(|_| fail(format!("table {table:?} takes a key of {n} codes 0-4")))?;
        if values.len() != n || values.iter().any(|&c| !(0..=4).contains(&c)) {
            return Err(fail(format!(
                "table {table:?} takes a key of {n} codes 0-4"
            )));
        }
        Ok(values.into_iter().map(|c| c as u8).collect())
    };
    let pair = |v: seqblaze_thermo::p3files::Hs| (Some(v.h), Some(v.s));
    match table {
        "nn" | "mm" | "m1" | "m2" => {
            let k = codes(4)?;
            let source = match table {
                "nn" => &t.nn,
                "mm" => &t.mm,
                "m1" => &t.m1,
                _ => &t.m2,
            };
            Ok(pair(source[idx4(k[0], k[1], k[2], k[3])]))
        }
        "d3" | "d5" => {
            let k = codes(3)?;
            let source = if table == "d3" { &t.d3 } else { &t.d5 };
            Ok(pair(source[idx3(k[0], k[1], k[2])]))
        }
        "at" => {
            let k = codes(2)?;
            Ok(pair(at(k[0], k[1])))
        }
        "loop_in" | "loop_bu" | "loop_hp" => {
            let k: i64 = key
                .extract()
                .map_err(|_| fail(format!("table {table:?} takes k in 1..=30")))?;
            if !(1..=30).contains(&k) {
                return Err(fail(format!("table {table:?} takes k in 1..=30")));
            }
            let source = match table {
                "loop_in" => &t.loop_in,
                "loop_bu" => &t.loop_bu,
                _ => &t.loop_hp,
            };
            Ok(pair(source[k as usize]))
        }
        "b3" | "b4" => {
            let width = if table == "b3" { 5 } else { 6 };
            let text: String = key
                .extract()
                .map_err(|_| fail(format!("table {table:?} takes a key string")))?;
            let codes: Vec<u8> = text
                .bytes()
                .map(|b| match b {
                    b'A' => 0,
                    b'C' => 1,
                    b'G' => 2,
                    b'T' => 3,
                    _ => 4,
                })
                .collect();
            let index = (codes.len() == width)
                .then(|| bonus_index(&codes))
                .flatten()
                .ok_or_else(|| {
                    fail(format!(
                        "table {table:?} takes an upper-case ACGT key of {width} letters"
                    ))
                })?;
            let bonus = if table == "b3" {
                t.b3[index]
            } else {
                t.b4[index]
            };
            Ok((bonus.h, bonus.s))
        }
        _ => Err(fail(format!(
            "unknown table {table:?}; use nn, mm, m1, m2, d3, d5, at, loop_in, loop_bu, \
             loop_hp, b3 or b4"
        ))),
    }
}

#[cfg(feature = "cuda")]
#[pyfunction(signature=(table,pack,handle,keep,cutoff,threads,fields,grain="auto",device=None,stream=1,progress=None))]
#[allow(clippy::too_many_arguments)]
fn _native_resident_hairpin_primer3<'py>(
    py: Python<'py>,
    table: PyRef<'py, PySeqTable>,
    pack: PyRef<'py, PyHairpinPack>,
    handle: PyRef<'py, PyPrimer3Params>,
    keep: &str,
    cutoff: Option<f64>,
    threads: usize,
    fields: Vec<String>,
    grain: &str,
    device: Option<u64>,
    stream: u64,
    progress: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
    let names = fields.iter().map(String::as_str).collect::<Vec<_>>();
    let kernel = HairpinPrimer3::new(pack.pack, Arc::clone(&handle.tables), &names)
        .map_err(|e| error(py, e))?;
    super::device_run::unary_dispatch(
        py,
        &kernel,
        &table,
        keep,
        cutoff,
        threads,
        grain,
        device,
        stream,
        || {
            let pack = kernel.device_pack()?;
            Ok(UnaryCall::Hairpin(Arc::new(primer3_call(
                pack, false, false, [0; 4],
            ))))
        },
        progress,
    )
}

pub fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    #[cfg(feature = "cuda")]
    module.add_function(wrap_pyfunction!(_native_resident_hairpin_primer3, module)?)?;
    module.add_class::<PyPrimer3Params>()?;
    module.add_class::<PyHairpinPack>()?;
    module.add_function(wrap_pyfunction!(_primer3_params, module)?)?;
    module.add_function(wrap_pyfunction!(_hairpin_primer3_pack, module)?)?;
    module.add_function(wrap_pyfunction!(_native_hairpin_primer3, module)?)?;
    #[cfg(feature = "testing")]
    module.add_function(wrap_pyfunction!(_testing_hairpin_probe, module)?)?;
    #[cfg(feature = "testing")]
    module.add_function(wrap_pyfunction!(_testing_primer3_entry, module)?)?;
    Ok(())
}
