//! T1 `tm` bindings: native (docs/wiki/thermo/tm.md §12), `compat="biopython"`
//! (docs/wiki/thermo/tm-biopython.md §13) and `compat="primer3"`
//! (docs/wiki/thermo/tm-primer3.md §12). Each has a per-call pack, built during
//! option validation, and a unary run over one sequence table, on the CPU or on
//! CUDA (docs/wiki/gpu/tm-cuda.md, docs/wiki/gpu/tm-compat-cuda.md).
use std::sync::{Arc, Mutex, OnceLock};

use pyo3::{prelude::*, types::PyDict};
use seqblaze_core::{
    fpenv::{self, FpGuard},
    sched::Grain,
    sink::Sink,
    unary::{self, UnaryKernel, UnaryOptions},
    validate::EngineError,
};
use seqblaze_cuda::{TmBiopythonPack, TmPack, TmPrimer3Pack, TmStep, UnaryCall};
use seqblaze_thermo::{
    biopython::Args,
    params::Params,
    primer3,
    tm::{DG_DCAL, DG_KCAL, DH_KCAL, DS_CAL_PER_K, TM_C},
    BiopythonPack, CallPack, Conditions, Oligo, Primer3Pack, Tm, TmBiopython, TmPrimer3,
    BIOPYTHON_NN3, PRIMER3_SANTALUCIA,
};

use super::{
    cancel, column, error, parse_grain, resolve_device, set_details_as, Numeric, PySeqTable,
};

// Device tm writes its columns by kernel field index (seqblaze_cuda::TM_SLOTS).
const _: () = assert!(
    TM_C == 0 && DG_KCAL == 1 && DG_DCAL == 2 && DH_KCAL == 3 && DS_CAL_PER_K == 4,
    "device tm slots follow tm's field order"
);

/// The device pack of tm-cuda.md §3.1 from the host pack (tm.md §3).
fn device_pack(pack: &CallPack) -> TmPack {
    let params = pack.params;
    let entry = |slot: usize, h: i64, g: i64| TmStep {
        h,
        g,
        d: i64::from(pack.table[slot]),
    };
    let mut step = [TmStep::default(); 16];
    for (index, &class) in params.step_class.iter().enumerate() {
        step[index] = entry(class, params.stack_dh[class], params.stack_dg[class]);
    }
    TmPack {
        step,
        init: entry(10, params.init.dh, params.init.dg),
        terminal_at: entry(11, params.terminal_at.dh, params.terminal_at.dg),
        symmetry: entry(12, params.symmetry.dh, params.symmetry.dg),
        s: pack.s.to_bits(),
        kappa4: pack.kappa4.to_bits(),
        kappa1: pack.kappa1.to_bits(),
    }
}

/// The device pack of tm-compat-cuda.md §3.3 from the host Biopython pack: integer
/// copies of its bit patterns only, so it sets no FP status flag (§9).
fn biopython_device_pack(pack: &BiopythonPack) -> TmBiopythonPack {
    let d = pack.device_pack();
    TmBiopythonPack {
        step: d.steps,
        ends: d.ends,
        salt: d.salt,
        ln_mon: d.ln_mon,
        rho: d.rho,
        kcal_to_cal: d.kcal_to_cal,
        k0: d.k0,
    }
}

/// The device pack of tm-compat-cuda.md §3.4 from the host primer3 pack: integer
/// copies only (§9).
fn primer3_device_pack(pack: &Primer3Pack) -> TmPrimer3Pack {
    let d = pack.device_pack();
    TmPrimer3Pack {
        step: d.steps,
        ends: d.ends,
        max_nn_length: d.max_nn_length,
        salt: d.salt,
        ln_mu: d.ln_mu,
        gamma4: d.gamma4,
        gamma1: d.gamma1,
        dmso_t: d.dmso_t,
        phi_formamide: d.phi_formamide,
        base: d.base,
        sig10: d.sig10,
        tally_h: d.tally_h,
        tally_s: d.tally_s,
        formamide_gc: d.formamide_gc,
        formamide_c: d.formamide_c,
        long_gc: d.long_gc,
        long_len: d.long_len,
        k0: d.k0,
    }
}

/// One call's pack: s, κ₄, κ₁, the integer table and the per-step lookups.
#[pyclass(frozen, module = "seqblaze._core", name = "_TmPack")]
pub struct PyTmPack {
    pack: Arc<CallPack>,
}

#[pymethods]
impl PyTmPack {
    #[getter]
    fn params(&self) -> &'static str {
        self.pack.params.name
    }
    /// `model.params.sha256` (tm.md QT17).
    #[getter]
    fn params_sha256(&self) -> &'static str {
        self.pack.params.sha256
    }
    #[getter]
    fn na_eq(&self) -> f64 {
        self.pack.na_eq
    }
    #[getter]
    fn s(&self) -> f64 {
        self.pack.s
    }
    #[getter]
    fn kappa4(&self) -> f64 {
        self.pack.kappa4
    }
    #[getter]
    fn kappa1(&self) -> f64 {
        self.pack.kappa1
    }
    /// G_e(T) by slot: the 10 stack classes, init, terminal_at, symmetry.
    #[getter]
    fn table<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let result = PyDict::new(py);
        let names =
            self.pack
                .params
                .classes
                .iter()
                .copied()
                .chain(["init", "terminal_at", "symmetry"]);
        for (name, value) in names.zip(self.pack.table) {
            result.set_item(name, value)?;
        }
        Ok(result)
    }
    #[getter]
    fn bound(&self) -> i64 {
        self.pack.bound
    }
}

type TmPackCacheEntry = (&'static str, [u64; 9], Arc<CallPack>);
static TM_PACK_CACHE: OnceLock<Mutex<Option<TmPackCacheEntry>>> = OnceLock::new();

/// One successful native pack, keyed by the preset name and every condition bit and
/// oligo variant (nndb2024-dna.md §5.5.2 item 3: without the name, a second preset at the
/// same conditions would get the cached pack of the first). The caller retains its
/// floating-point guard even on a cache hit. No Python object or device state enters
/// this immutable, single-entry cache.
fn cached_tm_pack(
    params: &'static Params,
    cond: &Conditions,
) -> Result<Arc<CallPack>, EngineError> {
    let (variant, a, b) = match cond.oligo_nM {
        Oligo::Total(a) => (0, a.to_bits(), 0),
        Oligo::Pair(a, b) => (1, a.to_bits(), b.to_bits()),
    };
    let key = [
        cond.temp_c.to_bits(),
        cond.na_mM.to_bits(),
        cond.k_mM.to_bits(),
        cond.tris_mM.to_bits(),
        cond.mg_mM.to_bits(),
        cond.dntp_mM.to_bits(),
        variant,
        a,
        b,
    ];
    let cache = TM_PACK_CACHE.get_or_init(|| Mutex::new(None));
    {
        let cached = cache
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if let Some((_, _, pack)) = cached
            .as_ref()
            .filter(|(name, cached_key, _)| *name == params.name && cached_key == &key)
        {
            return Ok(Arc::clone(pack));
        }
    }
    // Build outside the lock; failed builds leave the successful entry intact.
    // Concurrent misses may duplicate this work without changing any result.
    let pack = Arc::new(CallPack::new(params, cond)?);
    let previous = cache
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .replace((params.name, key, Arc::clone(&pack)));
    drop(previous);
    Ok(pack)
}

/// Build the per-call pack from `Conditions` values (scalar `oligo_b=None`, else
/// the pair) for the native preset `params` (nndb2024-dna.md §5.5.2 item 3). An
/// unknown preset, a closed preset's gate, Na_eq, range and host-constant errors
/// raise ParameterError.
#[pyfunction(signature=(temp_c, na_mM, k_mM, tris_mM, mg_mM, dntp_mM, oligo_a, oligo_b=None, params="santalucia2004"))]
#[allow(non_snake_case, clippy::too_many_arguments)]
pub fn _tm_pack(
    py: Python<'_>,
    temp_c: f64,
    na_mM: f64,
    k_mM: f64,
    tris_mM: f64,
    mg_mM: f64,
    dntp_mM: f64,
    oligo_a: f64,
    oligo_b: Option<f64>,
    params: &str,
) -> PyResult<PyTmPack> {
    let preset: &'static Params = match seqblaze_thermo::native_params(params) {
        Some(Ok(p)) => p,
        Some(Err(e)) => return Err(error(py, e)),
        None => {
            return Err(error(
                py,
                EngineError::Parameter(format!("unknown tm preset {params:?}")),
            ))
        }
    };
    let cond = Conditions {
        temp_c,
        na_mM,
        k_mM,
        tris_mM,
        mg_mM,
        dntp_mM,
        oligo_nM: match oligo_b {
            None => Oligo::Total(oligo_a),
            Some(b) => Oligo::Pair(oligo_a, b),
        },
    };
    let pack = py
        .detach(|| {
            let _guard = FpGuard::new();
            cached_tm_pack(preset, &cond)
        })
        .map_err(|e| error(py, e))?;
    Ok(PyTmPack { pack })
}

/// Run `tm` over every sequence of `table` (Each or Hits; `cutoff` compares
/// `tm_c >= cutoff`). Returns `{columns, shape, report}` like `_native_run`;
/// `shape` is `(n, 1)` (n sequences, one column of the API §5 (n_a, n_b) shape;
/// QT20), the columns stay 1-D. `device` is a CUDA ordinal (None for the CPU):
/// it is resolved after the option checks and before the sequence scan, and
/// the report then carries `host_pairs: 0` (tm-cuda.md §5.4, §7, §8).
#[pyfunction(signature=(table, pack, keep, cutoff, threads, fields, grain="auto", device=None, progress=None))]
#[allow(clippy::too_many_arguments)]
pub fn _native_tm<'py>(
    py: Python<'py>,
    table: PyRef<'py, PySeqTable>,
    pack: PyRef<'py, PyTmPack>,
    keep: &str,
    cutoff: Option<f64>,
    threads: usize,
    fields: Vec<String>,
    grain: &str,
    device: Option<u64>,
    progress: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
    let names: Vec<&str> = fields.iter().map(String::as_str).collect();
    let pack = Arc::clone(&pack.pack);
    let kernel = Tm::new(&pack, &names).map_err(|e| error(py, e))?;
    let call = UnaryCall::Tm(device_pack(&pack));
    let device = device.map(|ordinal| (ordinal, Box::new(move || Ok(call)) as DeviceBuild<'_>));
    run_unary(
        py,
        &kernel,
        &table,
        keep,
        cutoff,
        threads,
        grain,
        device,
        &[],
        progress,
    )
}

/// The builder of a unary kernel's device call, run only for a resolved device,
/// after device resolution and before the sequence scan (primer3-cuda.md §4).
pub(super) type DeviceBuild<'b> = Box<dyn FnOnce() -> Result<UnaryCall, EngineError> + Send + 'b>;

/// The shared unary run of a tm kernel: Each or Hits, `{columns, shape, report}`.
/// `device` is a CUDA ordinal with the builder of its device call (None for
/// the CPU); native tm, both tm compatibility presets, and hairpin share this driver.
/// `bools` names int32 columns exposed as bool (the hairpin `found` field).
#[allow(clippy::too_many_arguments)]
pub(super) fn run_unary<'py, K: UnaryKernel<Cutoff = f64>>(
    py: Python<'py>,
    kernel: &K,
    table: &PyRef<'py, PySeqTable>,
    keep: &str,
    cutoff: Option<f64>,
    threads: usize,
    grain: &str,
    device: Option<(u64, DeviceBuild<'_>)>,
    bools: &[&str],
    progress: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
    let grain: Grain = parse_grain(py, grain)?;
    let sink = match keep {
        "each" => Sink::Each,
        "hits" => Sink::Hits,
        _ => {
            return Err(error(
                py,
                EngineError::Parameter(format!(
                    "{} returns one row per sequence; use keep=sb.keep.each() or \
                 sb.keep.hits(cutoff)",
                    kernel.name()
                )),
            ))
        }
    };
    // LLVM may speculate an Option predicate on the unused None payload. A
    // floating comparison there can set DE before the GIL-released FpGuard;
    // classify by integer bits so None and normal inputs leave flags intact.
    if cutoff.is_some_and(|value| (value.to_bits() & 0x7FFF_FFFF_FFFF_FFFF) > 0x7FF0_0000_0000_0000)
    {
        return Err(error(
            py,
            EngineError::Parameter("cutoff must not be NaN".into()),
        ));
    }
    let options = UnaryOptions {
        threads,
        grain,
        sink,
        cutoff,
    };
    // Options above, then device resolution (no data moves), then sequences.
    let (device, call) = match device {
        None => (None, None),
        Some((ordinal, build)) => (resolve_device(py, Some(ordinal))?, Some(build)),
    };
    let table = table.table.host().map_err(|e| error(py, e))?;
    // One cancel scope for the call (cancellation-progress.md §3.7).
    let scope = cancel::Call::new(progress);
    let computed = scope.detach(py, || {
        match (&device, call) {
            (Some(device), Some(build)) => {
                // The device pack, built after device resolution (a refused host
                // reservation is a DeviceError) and before the sequence checks.
                let call = build()?;
                unary::validated(kernel, &table, &options, |table, report| {
                    if table.is_empty() {
                        // No upload, allocation or launch: the CPU zero-work output.
                        unary::compute(kernel, table, &options, report)
                    } else {
                        seqblaze_cuda::run_unary(
                            device,
                            &call,
                            table,
                            kernel.columns()?,
                            &options,
                            report,
                        )
                    }
                })
            }
            _ => unary::run(kernel, &table, &options),
        }
    });
    let output = scope.finish(py, computed)?;
    let columns = PyDict::new(py);
    set_details_as(py, &columns, Some(output.columns), bools)?;
    if let Some(values) = output.passed {
        columns.set_item("passed", column(py, Numeric::Bool(values))?)?;
    }
    if let Some(values) = output.i {
        columns.set_item("i", column(py, Numeric::I64(values))?)?;
    }
    let report = PyDict::new(py);
    let counts = PyDict::new(py);
    counts.set_item("certified", output.counts.certified)?;
    counts.set_item("pruned", output.counts.pruned)?;
    counts.set_item("full", output.counts.full)?;
    report.set_item("counts", counts)?;
    report.set_item("threads", output.threads)?;
    let flags = PyDict::new(py);
    flags.set_item("n_illegal_seqs", output.n_illegal_seqs)?;
    report.set_item("flags", flags)?;
    let scans = PyDict::new(py);
    scans.set_item("a", output.validation_scan)?;
    report.set_item("validation_scans", scans)?;
    if device.is_some() {
        // tm and hairpin never route a sequence to the host (tm-cuda.md §6, §8;
        // primer3-cuda.md §6.6).
        report.set_item("host_pairs", 0usize)?;
    }
    let response = PyDict::new(py);
    response.set_item("columns", columns)?;
    response.set_item("shape", (output.n, 1usize))?;
    response.set_item("report", report)?;
    Ok(response)
}

/// One `compat="biopython"` call's pack (tm-biopython.md §4, §13): mon, kk, their
/// correctly rounded logarithms, rho and the 16-step lookup.
#[pyclass(frozen, module = "seqblaze._core", name = "_TmBiopythonPack")]
pub struct PyTmBiopythonPack {
    pack: Arc<BiopythonPack>,
}

#[pymethods]
impl PyTmBiopythonPack {
    #[getter]
    fn params(&self) -> &'static str {
        self.pack.params.name
    }
    /// `model.params.sha256` (tm-biopython.md §2.1 rule 4).
    #[getter]
    fn params_sha256(&self) -> &'static str {
        self.pack.params.sha256
    }
    #[getter]
    fn mon(&self) -> f64 {
        self.pack.mon
    }
    #[getter]
    fn kk(&self) -> f64 {
        self.pack.kk
    }
    #[getter]
    fn ln_mon(&self) -> f64 {
        self.pack.ln_mon
    }
    #[getter]
    fn ln_k(&self) -> f64 {
        self.pack.ln_k
    }
    #[getter]
    fn rho(&self) -> f64 {
        self.pack.rho
    }
}

/// Build the compat pack from the `Tm_NN` arguments of §3 (mM, nM): a scalar
/// total C gives dnac1 = dnac2 = C*0.5 (P3), computed here in E0 (a subnormal
/// half is a rounded result); a pair (`d2` given) is used verbatim (P4). B6
/// and B9 raise ParameterError, B6 first.
#[pyfunction(signature=(na, k, tris, mg, dntp, d1, d2=None))]
#[allow(clippy::too_many_arguments)]
pub fn _tm_biopython_pack(
    py: Python<'_>,
    na: f64,
    k: f64,
    tris: f64,
    mg: f64,
    dntp: f64,
    d1: f64,
    d2: Option<f64>,
) -> PyResult<PyTmBiopythonPack> {
    let pack = py
        .detach(|| {
            let _guard = FpGuard::new();
            let (dnac1, dnac2) = match d2 {
                Some(d2) => (d1, d2),
                None => fpenv::guarded(|| (d1 * 0.5, d1 * 0.5)),
            };
            let args = Args {
                Na: na,
                K: k,
                Tris: tris,
                Mg: mg,
                dNTPs: dntp,
                dnac1,
                dnac2,
            };
            BiopythonPack::new(&BIOPYTHON_NN3, &args)
        })
        .map_err(|e| error(py, e))?;
    Ok(PyTmBiopythonPack {
        pack: Arc::new(pack),
    })
}

/// Run `tm` `compat="biopython"` over every sequence of `table`; as `_native_tm`,
/// including `device` (tm-compat-cuda.md §7).
#[pyfunction(signature=(table, pack, keep, cutoff, threads, fields, grain="auto", device=None, progress=None))]
#[allow(clippy::too_many_arguments)]
pub fn _native_tm_biopython<'py>(
    py: Python<'py>,
    table: PyRef<'py, PySeqTable>,
    pack: PyRef<'py, PyTmBiopythonPack>,
    keep: &str,
    cutoff: Option<f64>,
    threads: usize,
    fields: Vec<String>,
    grain: &str,
    device: Option<u64>,
    progress: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
    let names: Vec<&str> = fields.iter().map(String::as_str).collect();
    let pack = Arc::clone(&pack.pack);
    let kernel = TmBiopython::new(&pack, &names).map_err(|e| error(py, e))?;
    let device = device.map(|ordinal| {
        let call = UnaryCall::TmBiopython(biopython_device_pack(&pack));
        (ordinal, Box::new(move || Ok(call)) as DeviceBuild<'_>)
    });
    run_unary(
        py,
        &kernel,
        &table,
        keep,
        cutoff,
        threads,
        grain,
        device,
        &[],
        progress,
    )
}

/// One `compat="primer3"` call's pack (tm-primer3.md §4, §12): the P1-P13 constants,
/// their correctly rounded logarithms and the 16-step tally lookup.
#[pyclass(frozen, module = "seqblaze._core", name = "_TmPrimer3Pack")]
pub struct PyTmPrimer3Pack {
    pack: Arc<Primer3Pack>,
}

#[pymethods]
impl PyTmPrimer3Pack {
    #[getter]
    fn params(&self) -> &'static str {
        self.pack.params.name
    }
    /// `model.params.sha256` (tm-primer3.md §2.1 rule 7).
    #[getter]
    fn params_sha256(&self) -> &'static str {
        self.pack.params.sha256
    }
    #[getter]
    fn mv(&self) -> f64 {
        self.pack.mv
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
    fn x4(&self) -> f64 {
        self.pack.x4
    }
    #[getter]
    fn x1(&self) -> f64 {
        self.pack.x1
    }
    #[getter]
    fn ln_mu(&self) -> f64 {
        self.pack.ln_mu
    }
    #[getter]
    fn log10_mu(&self) -> f64 {
        self.pack.log10_mu
    }
    #[getter]
    fn ln_x4(&self) -> f64 {
        self.pack.ln_x4
    }
    #[getter]
    fn ln_x1(&self) -> f64 {
        self.pack.ln_x1
    }
    #[getter]
    fn gamma4(&self) -> f64 {
        self.pack.gamma4
    }
    #[getter]
    fn gamma1(&self) -> f64 {
        self.pack.gamma1
    }
    #[getter]
    fn phi_dmso(&self) -> f64 {
        self.pack.phi_dmso
    }
    #[getter]
    fn phi_formamide(&self) -> f64 {
        self.pack.phi_formamide
    }
    #[getter]
    fn dmso_t(&self) -> f64 {
        self.pack.dmso_t
    }
}

/// Build the compat pack from the §3 mapping's inputs (mM, nM, %, mol/L). P6, P7, P9
/// and P11 raise ParameterError in that order (tm-primer3.md §7 step 1.6).
#[pyfunction(signature=(na, k, tris, mg, dntp, c, dmso, formamide))]
#[allow(clippy::too_many_arguments)]
pub fn _tm_primer3_pack(
    py: Python<'_>,
    na: f64,
    k: f64,
    tris: f64,
    mg: f64,
    dntp: f64,
    c: f64,
    dmso: f64,
    formamide: f64,
) -> PyResult<PyTmPrimer3Pack> {
    let args = primer3::Args {
        na,
        k,
        tris,
        mg,
        dntp,
        dna: c,
        dmso,
        formamide,
    };
    let pack = py
        .detach(|| {
            // E0 for the GIL-released region (§7.3 item 3); `Primer3Pack::new` holds its own.
            let _guard = FpGuard::new();
            Primer3Pack::new(&PRIMER3_SANTALUCIA, &args)
        })
        .map_err(|e| error(py, e))?;
    Ok(PyTmPrimer3Pack {
        pack: Arc::new(pack),
    })
}

/// Run `tm` `compat="primer3"` over every sequence of `table`; as `_native_tm`,
/// including `device` (tm-compat-cuda.md §7).
#[pyfunction(signature=(table, pack, keep, cutoff, threads, fields, grain="auto", device=None, progress=None))]
#[allow(clippy::too_many_arguments)]
pub fn _native_tm_primer3<'py>(
    py: Python<'py>,
    table: PyRef<'py, PySeqTable>,
    pack: PyRef<'py, PyTmPrimer3Pack>,
    keep: &str,
    cutoff: Option<f64>,
    threads: usize,
    fields: Vec<String>,
    grain: &str,
    device: Option<u64>,
    progress: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
    let names: Vec<&str> = fields.iter().map(String::as_str).collect();
    let pack = Arc::clone(&pack.pack);
    let kernel = TmPrimer3::new(&pack, &names).map_err(|e| error(py, e))?;
    let device = device.map(|ordinal| {
        let call = UnaryCall::TmPrimer3(primer3_device_pack(&pack));
        (ordinal, Box::new(move || Ok(call)) as DeviceBuild<'_>)
    });
    run_unary(
        py,
        &kernel,
        &table,
        keep,
        cutoff,
        threads,
        grain,
        device,
        &[],
        progress,
    )
}

/// Testing builds: CR64(log10 x) of a positive finite binary64 (tm-primer3.md §9.3).
#[cfg(feature = "testing")]
#[pyfunction]
pub fn _native_tm_log10(py: Python<'_>, x: f64) -> PyResult<f64> {
    use seqblaze_thermo::cr::{self, Rational};
    // Positive and finite by the bits (sign 0, not ±0, exponent field not all ones):
    // a float comparison with the GIL held would read a subnormal as zero under DAZ
    // (text-nulls-fp-environment.md §7.3 item 4).
    if !(1..0x7FF0_0000_0000_0000).contains(&x.to_bits()) {
        return Err(error(
            py,
            EngineError::Parameter("tm_log10 needs a positive finite float".into()),
        ));
    }
    py.detach(|| {
        let _guard = FpGuard::new();
        cr::log10(&Rational::from_f64(x))
    })
    .map_err(|cap| {
        error(
            py,
            EngineError::Parameter(format!("precision cap of {} bits reached", cap.bits)),
        )
    })
}

/// Testing builds: CR64(coef · ln(num/den)) for decimal integer strings, the
/// host logarithm of tm.md §3.3 on arbitrary rationals.
#[cfg(feature = "testing")]
#[pyfunction]
pub fn _native_tm_ln(
    py: Python<'_>,
    num: &str,
    den: &str,
    coef_num: &str,
    coef_den: &str,
) -> PyResult<f64> {
    use seqblaze_thermo::{
        cr::{self, Rational},
        nat::{Int, Nat},
    };
    let parse = || -> Option<(Rational, Rational)> {
        let x = Rational::new(Int::parse_decimal(num)?, Nat::parse_decimal(den)?);
        let c = Rational::new(Int::parse_decimal(coef_num)?, Nat::parse_decimal(coef_den)?);
        Some((x, c))
    };
    let invalid = || {
        error(
            py,
            EngineError::Parameter(
                "tm_ln needs decimal integers: num > 0, den > 0, coef_num, coef_den > 0".into(),
            ),
        )
    };
    if den.trim_start_matches('0').is_empty() || coef_den.trim_start_matches('0').is_empty() {
        return Err(invalid());
    }
    let (x, coef) = parse().ok_or_else(invalid)?;
    if x.num.is_negative() || x.num.is_zero() {
        return Err(invalid());
    }
    py.detach(|| {
        let _guard = FpGuard::new();
        cr::ln(&x, &coef)
    })
    .map_err(|cap| {
        error(
            py,
            EngineError::Parameter(format!("precision cap of {} bits reached", cap.bits)),
        )
    })
}

/// Testing builds: one soft-binary64 operation per element on the device
/// (tm-cuda.md §10 item 11). `a` and `b` are native-endian u64 bit patterns
/// (int64 for `from_i64`; the unary ops `from_i64`, `is_finite` and `abs` take
/// `b=None`); returns the result bits, and 0 or 1 per element for the predicates.
#[cfg(feature = "testing")]
#[pyfunction(signature=(op, a, b=None, ordinal=0))]
pub fn _native_cuda_f64<'py>(
    py: Python<'py>,
    op: &str,
    a: &[u8],
    b: Option<&[u8]>,
    ordinal: u64,
) -> PyResult<Bound<'py, pyo3::types::PyBytes>> {
    let op = seqblaze_cuda::F64Op::parse(op).ok_or_else(|| {
        error(
            py,
            EngineError::Parameter(format!(
                "op={op:?}; use add, sub, mul, div, from_i64, lt, le, gt, ge, eq, is_finite or abs"
            )),
        )
    })?;
    let words = |bytes: &[u8]| -> PyResult<Vec<u64>> {
        if bytes.len() % 8 != 0 {
            return Err(error(
                py,
                EngineError::Parameter("operands are whole 8-byte words".into()),
            ));
        }
        Ok(bytes
            .chunks_exact(8)
            .map(|word| u64::from_ne_bytes(word.try_into().expect("8 bytes")))
            .collect())
    };
    let a = words(a)?;
    let b = b.map(words).transpose()?;
    let device = resolve_device(py, Some(ordinal))?.expect("an ordinal was given");
    let result = py
        .detach(|| {
            let _guard = FpGuard::new();
            seqblaze_cuda::f64_eval(&device, op, &a, b.as_deref())
        })
        .map_err(|e| error(py, e))?;
    let bytes: Vec<u8> = result.iter().flat_map(|word| word.to_ne_bytes()).collect();
    Ok(pyo3::types::PyBytes::new(py, &bytes))
}

#[cfg(feature = "cuda")]
#[pyfunction(signature=(table,pack,keep,cutoff,threads,fields,grain="auto",device=None,stream=1,progress=None))]
#[allow(clippy::too_many_arguments)]
fn _native_resident_tm<'py>(
    py: Python<'py>,
    table: PyRef<'py, PySeqTable>,
    pack: PyRef<'py, PyTmPack>,
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
    let pack = Arc::clone(&pack.pack);
    let kernel = Tm::new(&pack, &names).map_err(|e| error(py, e))?;
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
        || Ok(UnaryCall::Tm(device_pack(&pack))),
        progress,
    )
}

#[cfg(feature = "cuda")]
#[pyfunction(signature=(table,pack,keep,cutoff,threads,fields,grain="auto",device=None,stream=1,progress=None))]
#[allow(clippy::too_many_arguments)]
fn _native_resident_tm_biopython<'py>(
    py: Python<'py>,
    table: PyRef<'py, PySeqTable>,
    pack: PyRef<'py, PyTmBiopythonPack>,
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
    let pack = Arc::clone(&pack.pack);
    let kernel = TmBiopython::new(&pack, &names).map_err(|e| error(py, e))?;
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
        || Ok(UnaryCall::TmBiopython(biopython_device_pack(&pack))),
        progress,
    )
}

#[cfg(feature = "cuda")]
#[pyfunction(signature=(table,pack,keep,cutoff,threads,fields,grain="auto",device=None,stream=1,progress=None))]
#[allow(clippy::too_many_arguments)]
fn _native_resident_tm_primer3<'py>(
    py: Python<'py>,
    table: PyRef<'py, PySeqTable>,
    pack: PyRef<'py, PyTmPrimer3Pack>,
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
    let pack = Arc::clone(&pack.pack);
    let kernel = TmPrimer3::new(&pack, &names).map_err(|e| error(py, e))?;
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
        || Ok(UnaryCall::TmPrimer3(primer3_device_pack(&pack))),
        progress,
    )
}

/// Check a native `tm` preset by name (docs/wiki/thermo/nndb2024-dna.md §4.7): the gate's
/// ParameterError "not yet available" naming the missing build units and the pending items
/// while the T1 gate is closed; ParameterError for an unknown name.
#[pyfunction]
fn _tm_params(py: Python<'_>, name: &str) -> PyResult<()> {
    match seqblaze_thermo::native_params(name) {
        Some(Ok(_)) => Ok(()),
        Some(Err(e)) => Err(error(py, e)),
        None => Err(error(
            py,
            EngineError::Parameter(format!("unknown tm preset {name:?}")),
        )),
    }
}

pub fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    #[cfg(feature = "cuda")]
    module.add_function(wrap_pyfunction!(_native_resident_tm, module)?)?;
    #[cfg(feature = "cuda")]
    module.add_function(wrap_pyfunction!(_native_resident_tm_biopython, module)?)?;
    #[cfg(feature = "cuda")]
    module.add_function(wrap_pyfunction!(_native_resident_tm_primer3, module)?)?;
    module.add_class::<PyTmPack>()?;
    module.add_function(wrap_pyfunction!(_tm_pack, module)?)?;
    module.add_function(wrap_pyfunction!(_native_tm, module)?)?;
    module.add_function(wrap_pyfunction!(_tm_params, module)?)?;
    module.add_class::<PyTmBiopythonPack>()?;
    module.add_function(wrap_pyfunction!(_tm_biopython_pack, module)?)?;
    module.add_function(wrap_pyfunction!(_native_tm_biopython, module)?)?;
    module.add_class::<PyTmPrimer3Pack>()?;
    module.add_function(wrap_pyfunction!(_tm_primer3_pack, module)?)?;
    module.add_function(wrap_pyfunction!(_native_tm_primer3, module)?)?;
    #[cfg(feature = "testing")]
    module.add_function(wrap_pyfunction!(_native_tm_log10, module)?)?;
    #[cfg(feature = "testing")]
    module.add_function(wrap_pyfunction!(_native_tm_ln, module)?)?;
    #[cfg(feature = "testing")]
    module.add_function(wrap_pyfunction!(_native_cuda_f64, module)?)?;
    Ok(())
}
