//! K1 `duplex` bindings (docs/wiki/thermo/duplex-mfe.md §12.3-§12.4): the pack handle
//! (a generated preset or, in testing builds, a synthetic pack text), the per-call pack
//! built during option validation, and the testing evaluator. The pair run itself is
//! `_native_run` / `_native_tiles` with `kernel="duplex"` and `duplex=<call>`; `sb.tm` on
//! Pairs (tm.md §13) runs `_native_run` with `kernel="tm"` and a call built for it.
use std::sync::{Arc, Mutex, OnceLock};

use pyo3::prelude::*;
#[cfg(feature = "testing")]
use pyo3::types::{PyBytes, PyDict};
use seqblaze_core::{
    exec::Strategy,
    fpenv::FpGuard,
    plan::{BoundAvailability, ExecutionPlan},
    validate::EngineError,
};
use seqblaze_thermo::{
    dimer::{DimerArgs, End},
    duplex::{
        bound::Certificate,
        load_preset,
        pf::{PfCall, PfConstants, RangeMessages},
        pf_bound::Certificate as PfCertificate,
        Pack,
    },
    hairpin::Args,
    Caller, Conditions, DimerPack, DuplexCall, Oligo, P3Tables,
};

use super::{error, hairpin::PyPrimer3Params};

/// A loaded K1 pack: a preset or a testing pack (§12.3).
#[pyclass(frozen, module = "seqblaze._core", name = "_DuplexPack")]
pub struct PyDuplexPack {
    pub(crate) pack: Arc<Pack>,
    /// Only the two canonical public presets may retain a successful tm call.
    /// A testing pack never acquires this provenance from its name or hash.
    stock_preset: bool,
}

#[pymethods]
impl PyDuplexPack {
    #[getter]
    fn name(&self) -> &str {
        &self.pack.name
    }
    /// `model.params.sha256`: SHA-256 of the build-definition text.
    #[getter]
    fn sha256(&self) -> &str {
        &self.pack.sha256
    }
    #[getter]
    fn loop_max(&self) -> usize {
        self.pack.loop_max
    }
    fn __repr__(&self) -> String {
        format!(
            "<duplex pack {} (sha256 {}, loop_max {})>",
            self.pack.name, self.pack.sha256, self.pack.loop_max
        )
    }
}

/// One call's pack (§12.4): host constants, G_e(T), B and the cell keys; for
/// `model="pf"` also K3's constants and weight tables (duplex-pf.md §5); for a
/// `model="mfe"` RowStats call with `lse_dg_kcal` (duplex-mfe.md §8.6.5) the LSE
/// constants that `_duplex_lse` adds.
#[pyclass(frozen, module = "seqblaze._core", name = "_DuplexCall")]
pub struct PyDuplexCall {
    pub(crate) call: Arc<DuplexCall>,
    pub(crate) pf: Option<Arc<PfCall>>,
    pub(crate) lse: Option<Arc<PfConstants>>,
    /// The public call the pack was built for (`sb.duplex` or `sb.tm` on Pairs).
    pub(crate) caller: Caller,
    /// K1 `sb.duplex(model="mfe")`: the cell of the call's C1 certificate, built only
    /// when the plan can use it (duplex-mfe-bound.md §5 *When*, revision 3); `None` for
    /// every other call.
    pub(crate) bound: Option<BoundCell>,
    /// K3 `sb.duplex(model="pf")` without `lse_dg_kcal`: the cell of the call's C2
    /// certificate (duplex-pf-bound.md §5 *Where*), built only when the plan can use it;
    /// `None` for every other call.
    pub(crate) pf_bound: Option<BoundCell<PfCertificate>>,
}

/// The result of one certificate build for a call's lifetime (duplex-mfe-bound.md §5;
/// duplex-pf-bound.md §5 for K3's C2 certificate).
pub(crate) enum CertState<C = Certificate> {
    /// A passing or failing certificate.
    Built(Arc<C>),
    /// A host reservation was refused while it was built: the bound is unavailable for
    /// this call (`no_bound`); nothing entered the process-wide cache (§5, QB13). The
    /// message is raised only by the testing hooks, which need the certificate.
    Refused(String),
}

/// The once-initialised certificate cell of a K1 `model="mfe"` call (§5 *Where it lives*),
/// or of a K3 `model="pf"` call's C2 certificate.
pub(crate) struct BoundCell<C = Certificate> {
    stock: bool,
    cond: Conditions,
    state: OnceLock<CertState<C>>,
}

impl<C> BoundCell<C> {
    fn new(stock: bool, cond: Conditions) -> Self {
        Self {
            stock,
            cond,
            state: OnceLock::new(),
        }
    }
}

impl PyDuplexCall {
    /// Step 1 of the two-step resolution (duplex-mfe-bound.md §6.1, revision 3;
    /// duplex-pf-bound.md §6.1): a K1 `model="mfe"` call without `lse_dg_kcal`, or a K3
    /// `model="pf"` call without `lse_dg_kcal`, may use the bound, before its certificate
    /// is known. Builds nothing.
    pub(crate) fn bound_eligible(&self) -> bool {
        if self.pf.is_some() {
            return self.pf_bound.is_some();
        }
        self.lse.is_none() && self.bound.is_some()
    }

    /// Take the call's certificate (K1's C1 or K3's C2), building it if needed; for the
    /// private entries' certificate step (§6.1 *Private entries*).
    pub(crate) fn take_certificate(&self, py: Python<'_>) {
        if self.pf.is_some() {
            let _ = self.pf_certificate_state(py);
        } else {
            let _ = self.certificate_state(py);
        }
    }

    /// The C2 certificate if a previous step of this call built it (never builds).
    pub(crate) fn pf_built_certificate(&self) -> Option<&Arc<PfCertificate>> {
        match self.pf_bound.as_ref()?.state.get()? {
            CertState::Built(cert) => Some(cert),
            CertState::Refused(_) => None,
        }
    }

    /// The call's C2 certificate state, built on first use through its own process-wide
    /// cache (QZ14), with the GIL released and under the binding's FP guard (E0).
    /// `None` for a call without a C2 cell.
    pub(crate) fn pf_certificate_state(&self, py: Python<'_>) -> Option<&CertState<PfCertificate>> {
        let cell = self.pf_bound.as_ref()?;
        if let Some(state) = cell.state.get() {
            return Some(state);
        }
        let pf = self.pf.as_ref()?;
        let built = py.detach(|| {
            let _guard = FpGuard::new();
            match cached_pf_certificate(cell.stock, pf, &cell.cond) {
                Ok(cert) => CertState::Built(cert),
                Err(e) => CertState::Refused(e.to_string()),
            }
        });
        Some(cell.state.get_or_init(|| built))
    }

    /// The C2 certificate for a testing hook, built on demand; a refused reservation
    /// raises DeviceError there.
    #[cfg(feature = "testing")]
    fn pf_hook_certificate(&self, py: Python<'_>, what: &str) -> PyResult<Arc<PfCertificate>> {
        match self.pf_certificate_state(py) {
            Some(CertState::Built(cert)) => Ok(Arc::clone(cert)),
            Some(CertState::Refused(message)) => {
                Err(error(py, EngineError::Device(message.clone())))
            }
            None => Err(error(
                py,
                EngineError::Parameter(format!("{what} needs a duplex model='pf' call")),
            )),
        }
    }

    /// The certificate if a previous step of this call built it (never builds).
    pub(crate) fn built_certificate(&self) -> Option<&Arc<Certificate>> {
        match self.bound.as_ref()?.state.get()? {
            CertState::Built(cert) => Some(cert),
            CertState::Refused(_) => None,
        }
    }

    /// The call's certificate state, built on first use through the process-wide cache,
    /// with the GIL released and under the binding's FP guard (E0) (§5 *Where it lives*).
    /// `None` for a call without a certificate cell.
    pub(crate) fn certificate_state(&self, py: Python<'_>) -> Option<&CertState> {
        let cell = self.bound.as_ref()?;
        if let Some(state) = cell.state.get() {
            return Some(state);
        }
        let call = &self.call;
        let built = py.detach(|| {
            let _guard = FpGuard::new();
            match cached_certificate(cell.stock, call, &cell.cond) {
                Ok(cert) => CertState::Built(cert),
                Err(e) => CertState::Refused(e.to_string()),
            }
        });
        Some(cell.state.get_or_init(|| built))
    }

    /// Step 2 of the two-step resolution (§6.1): whether the call's bound is Available,
    /// building the certificate if needed. A failing certificate or a refused reservation
    /// gives false (`no_bound`); nothing is raised.
    pub(crate) fn bound_available(&self, py: Python<'_>) -> bool {
        if !self.bound_eligible() {
            return false;
        }
        if self.pf.is_some() {
            matches!(self.pf_certificate_state(py), Some(CertState::Built(cert)) if cert.pass)
        } else {
            matches!(self.certificate_state(py), Some(CertState::Built(cert)) if cert.pass)
        }
    }

    /// The certificate for a testing hook, built on demand (§10.3 item 12 (c)); a refused
    /// reservation raises DeviceError there.
    #[cfg(feature = "testing")]
    fn hook_certificate(&self, py: Python<'_>, what: &str) -> PyResult<Arc<Certificate>> {
        if self.pf.is_some() {
            return Err(error(
                py,
                EngineError::Parameter(format!("{what} needs a duplex model='mfe' call")),
            ));
        }
        match self.certificate_state(py) {
            Some(CertState::Built(cert)) => Ok(Arc::clone(cert)),
            Some(CertState::Refused(message)) => {
                Err(error(py, EngineError::Device(message.clone())))
            }
            None => Err(error(
                py,
                EngineError::Parameter(format!("{what} needs a duplex model='mfe' call")),
            )),
        }
    }
}

/// Before a private run with `strategy="bounded"` that did not pass through the public
/// plan (duplex-mfe-bound.md §6.1 *Private entries*): take the K1 call's certificate, so
/// that its kernel carries a passing bound, but only when the plan can use it, i.e.
/// `ExecutionPlan::resolve(bounded, Available, demand of keep and cutoff)` is effective
/// `bounded` and the call is bound-eligible. Each without a cutoff, Matrix, Tiles and
/// unknown sinks build nothing. Public calls built it in `_native_plan` already. A run
/// requesting `lse_dg_kcal` builds nothing either (duplex-pf-bound.md §6.1, QZ7): its
/// call object carries no bound, and a K3 call that still carries its C2 cell is refused
/// by `RuntimeKernel::configure` afterwards.
pub(crate) fn prepare_bounded(
    py: Python<'_>,
    duplex: Option<&Bound<'_, PyAny>>,
    strategy: &str,
    keep: &str,
    has_cutoff: bool,
    details: &[String],
) {
    if strategy != "bounded" || details.iter().any(|name| name == "lse_dg_kcal") {
        return;
    }
    let usable = super::output_demand(keep, has_cutoff).is_some_and(|demand| {
        ExecutionPlan::resolve(Strategy::Bounded, BoundAvailability::Available, demand).effective
            == Strategy::Bounded
    });
    if !usable {
        return;
    }
    if let Some(call) = duplex.and_then(|call| call.cast::<PyDuplexCall>().ok()) {
        let call = call.get();
        if call.bound_eligible() {
            call.take_certificate(py);
        }
    }
}

#[pymethods]
impl PyDuplexCall {
    #[getter]
    fn params(&self) -> &str {
        &self.call.pack.name
    }
    #[getter]
    fn params_sha256(&self) -> &str {
        &self.call.pack.sha256
    }
    #[getter]
    fn na_eq(&self) -> f64 {
        self.call.na_eq
    }
    #[getter]
    fn s(&self) -> f64 {
        self.call.s
    }
    #[getter]
    fn kappa4(&self) -> f64 {
        self.call.kappa4
    }
    #[getter]
    fn kappa1(&self) -> f64 {
        self.call.kappa1
    }
    /// B of §4.4.
    #[getter]
    fn bound(&self) -> i64 {
        self.call.bound
    }
    /// The model the call was built for ("mfe" or "pf").
    #[getter]
    fn model(&self) -> &'static str {
        if self.pf.is_some() {
            "pf"
        } else {
            "mfe"
        }
    }
}

/// The preset `name` (§3.5 gate, nndb2024-dna.md §4.7): ParameterError "not yet
/// available" naming the missing build units and the pending items while the gate is closed.
#[pyfunction]
fn _duplex_preset(py: Python<'_>, name: &str) -> PyResult<PyDuplexPack> {
    let pack = py
        .detach(|| {
            // E0 for the GIL-released region (§7.3 item 3): it enters an engine crate.
            let _guard = FpGuard::new();
            load_preset(name).unwrap_or_else(|| {
                Err(EngineError::Parameter(format!(
                    "unknown duplex preset {name:?}"
                )))
            })
        })
        .map_err(|e| error(py, e))?;
    Ok(PyDuplexPack {
        pack,
        stock_preset: matches!(name, "santalucia2004" | "nndb2024-dna"),
    })
}

/// Testing builds: parse a compiled pack text with the preset loader (§12.3).
#[cfg(feature = "testing")]
#[pyfunction]
fn _native_duplex_pack(py: Python<'_>, text: &str) -> PyResult<PyDuplexPack> {
    let pack = py
        .detach(|| {
            // E0 for the GIL-released region (§7.3 item 3).
            let _guard = FpGuard::new();
            Pack::parse(text)
        })
        .map_err(|e| error(py, e))?;
    Ok(PyDuplexPack {
        pack: Arc::new(pack),
        stock_preset: false,
    })
}

#[allow(non_snake_case, clippy::too_many_arguments)]
fn conditions(
    temp_c: f64,
    na_mM: f64,
    k_mM: f64,
    tris_mM: f64,
    mg_mM: f64,
    dntp_mM: f64,
    oligo_a: f64,
    oligo_b: Option<f64>,
) -> Conditions {
    Conditions {
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
    }
}

type TmPairsCallCacheEntry = (Caller, [u64; 9], Arc<DuplexCall>);
static TM_PAIRS_CALL_CACHE: OnceLock<Mutex<Option<TmPairsCallCacheEntry>>> = OnceLock::new();

/// One successful native tm(Pairs) call, entered only for canonical stock packs
/// under the binding's FpGuard. The cached call strongly owns its immutable pack:
/// Arc identity covers the entire pack, and that owner prevents address ABA.
/// Neither a matching name nor a build-definition hash admits a synthetic pack.
fn cached_tm_pairs_call(
    caller: Caller,
    pack: Arc<Pack>,
    cond: &Conditions,
) -> Result<Arc<DuplexCall>, EngineError> {
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
    let cache = TM_PAIRS_CALL_CACHE.get_or_init(|| Mutex::new(None));
    {
        let cached = cache
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if let Some((_, _, call)) = cached.as_ref().filter(|(cached_caller, cached_key, call)| {
            *cached_caller == caller && cached_key == &key && Arc::ptr_eq(&call.pack, &pack)
        }) {
            return Ok(Arc::clone(call));
        }
    }
    // Misses retain the original construction and errors outside the lock. A
    // failed build leaves the previous success intact; racing misses may build
    // independently, and each caller returns its own matching immutable call.
    let call = Arc::new(DuplexCall::new_for(caller, pack, cond)?);
    let previous = cache
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .replace((caller, key, Arc::clone(&call)));
    // Releasing potentially large host tables must not hold the cache mutex.
    drop(previous);
    Ok(call)
}

type BoundCacheEntry = (Arc<Pack>, [u64; 6], Arc<Certificate>);
static BOUND_CACHE: OnceLock<Mutex<Option<BoundCacheEntry>>> = OnceLock::new();

/// The C1 certificate of a K1 call (duplex-mfe-bound.md §5): from the process-wide cache
/// for a canonical stock pack, keyed by the pack (Arc identity, which the entry owns) and
/// the bits of the conditions that determine the per-call D cell table, or built and
/// inserted. One entry, the most recent (QB1). Synthetic packs build per call; a
/// certificate built while a testing fault is active is never cached. A refused
/// reservation during the build returns `Err` and leaves the cache as it was (the
/// caller records it in the call's cell as an unavailable bound, §5 *Refused reservation*).
fn cached_certificate(
    stock: bool,
    call: &DuplexCall,
    cond: &Conditions,
) -> Result<Arc<Certificate>, EngineError> {
    let faulted = seqblaze_core::plan::any_bound_fault();
    if !stock || faulted {
        return Certificate::build(call).map(Arc::new);
    }
    let key = [
        cond.temp_c.to_bits(),
        cond.na_mM.to_bits(),
        cond.k_mM.to_bits(),
        cond.tris_mM.to_bits(),
        cond.mg_mM.to_bits(),
        cond.dntp_mM.to_bits(),
    ];
    let cache = BOUND_CACHE.get_or_init(|| Mutex::new(None));
    {
        let cached = cache.lock().unwrap_or_else(|e| e.into_inner());
        if let Some((_, _, cert)) = cached
            .as_ref()
            .filter(|(pack, cached_key, _)| cached_key == &key && Arc::ptr_eq(pack, &call.pack))
        {
            return Ok(Arc::clone(cert));
        }
    }
    let cert = Arc::new(Certificate::build(call)?);
    let previous = cache.lock().unwrap_or_else(|e| e.into_inner()).replace((
        Arc::clone(&call.pack),
        key,
        Arc::clone(&cert),
    ));
    drop(previous);
    Ok(cert)
}

type PfBoundCacheEntry = (Arc<Pack>, [u64; 6], Arc<PfCertificate>);
static PF_BOUND_CACHE: OnceLock<Mutex<Option<PfBoundCacheEntry>>> = OnceLock::new();

/// The C2 certificate of a K3 call (duplex-pf-bound.md §5 *Cache*, QZ14): C1's cache rule
/// with a one-entry cache of its own, keyed by the pack (Arc identity, which the entry
/// owns) and the bits of the conditions that determine the per-call D cell table and R·T.
/// Synthetic packs build per call; a certificate built while a testing fault is active is
/// never cached; a refused reservation returns `Err` and leaves the cache as it was.
fn cached_pf_certificate(
    stock: bool,
    pf: &PfCall,
    cond: &Conditions,
) -> Result<Arc<PfCertificate>, EngineError> {
    let faulted = seqblaze_core::plan::any_bound_fault();
    if !stock || faulted {
        return PfCertificate::build(pf).map(Arc::new);
    }
    let key = [
        cond.temp_c.to_bits(),
        cond.na_mM.to_bits(),
        cond.k_mM.to_bits(),
        cond.tris_mM.to_bits(),
        cond.mg_mM.to_bits(),
        cond.dntp_mM.to_bits(),
    ];
    let pack = &pf.call.pack;
    let cache = PF_BOUND_CACHE.get_or_init(|| Mutex::new(None));
    {
        let cached = cache.lock().unwrap_or_else(|e| e.into_inner());
        if let Some((_, _, cert)) = cached.as_ref().filter(|(cached_pack, cached_key, _)| {
            cached_key == &key && Arc::ptr_eq(cached_pack, pack)
        }) {
            return Ok(Arc::clone(cert));
        }
    }
    let cert = Arc::new(PfCertificate::build(pf)?);
    let previous = cache.lock().unwrap_or_else(|e| e.into_inner()).replace((
        Arc::clone(pack),
        key,
        Arc::clone(&cert),
    ));
    drop(previous);
    Ok(cert)
}

/// Build the per-call pack from `Conditions` values (scalar `oligo_b=None`, else the
/// pair). Na_eq, host-constant and range errors raise ParameterError (§9); with
/// `model="pf"` the K3 exponent check follows K1's range check, then the weight tables
/// (duplex-pf.md §10). `kernel="tm"` builds the same K1 pack for `sb.tm` on Pairs (the
/// register pairing, tm.md §13.6 step 1.6): its messages name `tm`, and it has no
/// `model="pf"`.
#[pyfunction(signature=(pack, temp_c, na_mM, k_mM, tris_mM, mg_mM, dntp_mM, oligo_a, oligo_b=None, model="mfe", kernel="duplex"))]
#[allow(non_snake_case, clippy::too_many_arguments)]
fn _duplex_call(
    py: Python<'_>,
    pack: PyRef<'_, PyDuplexPack>,
    temp_c: f64,
    na_mM: f64,
    k_mM: f64,
    tris_mM: f64,
    mg_mM: f64,
    dntp_mM: f64,
    oligo_a: f64,
    oligo_b: Option<f64>,
    model: &str,
    kernel: &str,
) -> PyResult<PyDuplexCall> {
    let pf = match model {
        "mfe" => false,
        "pf" => true,
        _ => {
            return Err(error(
                py,
                EngineError::Parameter(format!("model={model:?}; use model='mfe' or 'pf'")),
            ))
        }
    };
    let caller = match kernel {
        "duplex" => Caller::Duplex,
        "tm" if !pf => Caller::TmPairs,
        _ => {
            return Err(error(
                py,
                EngineError::Parameter(format!(
                    "kernel={kernel:?} with model={model:?}; use kernel='duplex', or \
                     kernel='tm' with model='mfe'"
                )),
            ))
        }
    };
    let cond = conditions(
        temp_c, na_mM, k_mM, tris_mM, mg_mM, dntp_mM, oligo_a, oligo_b,
    );
    let cacheable = pack.stock_preset && caller == Caller::TmPairs && !pf;
    let stock = pack.stock_preset;
    let pack = Arc::clone(&pack.pack);
    let (call, pf) = py
        .detach(|| -> Result<_, EngineError> {
            // E0 for the GIL-released region (§7.3 item 3); the builders hold their own.
            let _guard = FpGuard::new();
            let call = if cacheable {
                cached_tm_pairs_call(caller, pack, &cond)?
            } else {
                Arc::new(DuplexCall::new_for(caller, pack, &cond)?)
            };
            let pf = if pf {
                Some(Arc::new(PfCall::new(Arc::clone(&call), &cond)?))
            } else {
                None
            };
            Ok((call, pf))
        })
        .map_err(|e| error(py, e))?;
    // K1 sb.duplex(model="mfe"): the C1 certificate's cell, filled only when the plan
    // can use the bound (duplex-mfe-bound.md §5 *When*, revision 3); K3 model="pf": the
    // C2 certificate's cell (duplex-pf-bound.md §5 *When*).
    let bound = (pf.is_none() && caller == Caller::Duplex).then(|| BoundCell::new(stock, cond));
    let pf_bound = pf.is_some().then(|| BoundCell::new(stock, cond));
    Ok(PyDuplexCall {
        call,
        pf,
        lse: None,
        caller,
        bound,
        pf_bound,
    })
}

/// K1 RowStats `lse_dg_kcal` (duplex-mfe.md §8.6.2, §8.6.5): K3's §5.4 range checks with
/// K1's bound B and the call's R·T, raising the §8.6.2 messages, and the LSE constants.
/// Takes the `model="mfe"`, `kernel="duplex"` call the facade built with the same
/// condition arguments and returns a new call object that shares its `DuplexCall` and
/// carries the constants (`model` still reads "mfe"). Any other call is a caller bug.
#[pyfunction(signature=(call, temp_c, na_mM, k_mM, tris_mM, mg_mM, dntp_mM, oligo_a, oligo_b=None))]
#[allow(non_snake_case, clippy::too_many_arguments)]
fn _duplex_lse(
    py: Python<'_>,
    call: PyRef<'_, PyDuplexCall>,
    temp_c: f64,
    na_mM: f64,
    k_mM: f64,
    tris_mM: f64,
    mg_mM: f64,
    dntp_mM: f64,
    oligo_a: f64,
    oligo_b: Option<f64>,
) -> PyResult<PyDuplexCall> {
    if call.pf.is_some() || call.caller != Caller::Duplex {
        return Err(error(
            py,
            EngineError::Parameter(
                "internal error: _duplex_lse needs a model='mfe' duplex call".into(),
            ),
        ));
    }
    let cond = conditions(
        temp_c, na_mM, k_mM, tris_mM, mg_mM, dntp_mM, oligo_a, oligo_b,
    );
    let shared = Arc::clone(&call.call);
    let lse = py
        .detach(|| {
            // E0 for the GIL-released region (§7.3 item 3); the constructor holds its own.
            let _guard = FpGuard::new();
            PfConstants::new_for(&shared, &cond, RangeMessages::MfeLse)
        })
        .map_err(|e| error(py, e))?;
    Ok(PyDuplexCall {
        call: shared,
        pf: None,
        lse: Some(Arc::new(lse)),
        caller: Caller::Duplex,
        // lse_dg_kcal makes the bound unavailable (§6.1 step 1): no certificate is built.
        bound: None,
        pf_bound: None,
    })
}

/// K3 RowStats `lse_dg_kcal` (duplex-pf-bound.md §6.1, QZ7): the `model="pf"` call without
/// its C2 certificate cell, so that the plan resolves `bounded` to full with `no_bound`
/// and no certificate is built. It shares the call's packs; any other call is a caller bug.
#[pyfunction]
fn _duplex_pf_lse(py: Python<'_>, call: PyRef<'_, PyDuplexCall>) -> PyResult<PyDuplexCall> {
    let Some(pf) = call.pf.as_ref() else {
        return Err(error(
            py,
            EngineError::Parameter(
                "internal error: _duplex_pf_lse needs a model='pf' duplex call".into(),
            ),
        ));
    };
    Ok(PyDuplexCall {
        call: Arc::clone(&call.call),
        pf: Some(Arc::clone(pf)),
        lse: None,
        caller: call.caller,
        bound: None,
        pf_bound: None,
    })
}

/// Testing builds: Ẑ of one pair as (μ, e, c_Z) in canonical orientation, (0.0, 0, c_Z)
/// for an empty space (duplex-pf.md §14.4). The strands are checked as by the public
/// call (upper-case ACGT, at most 200 nt).
#[cfg(feature = "testing")]
#[pyfunction]
fn _native_duplex_pf_scaled(
    py: Python<'_>,
    call: PyRef<'_, PyDuplexCall>,
    a: &[u8],
    b: &[u8],
) -> PyResult<(f64, i64, u64)> {
    use seqblaze_thermo::duplex::pf;
    let Some(pf_call) = call.pf.as_ref() else {
        return Err(error(
            py,
            EngineError::Parameter("duplex_pf_scaled needs a model='pf' call".into()),
        ));
    };
    pf::check_strand("a", a).map_err(|e| error(py, e))?;
    pf::check_strand("b", b).map_err(|e| error(py, e))?;
    let (x, y) = (pf::codes_of(a), pf::codes_of(b));
    let z = py
        .detach(|| {
            // E0 for the GIL-released region (text-nulls-fp-environment.md §7.3 item 3).
            let _guard = FpGuard::new();
            pf_call.zhat_codes(&x, &y)
        })
        .map_err(|e| error(py, e))?;
    let c_z = pf::rounding_count(a.len(), b.len(), pf_call.loop_max());
    Ok((z.mu, z.e, c_z))
}

/// Testing builds: Ẑ of every zipped pair (a[k], b[k]) as (M, e) arrays, M·2^(e − 52),
/// in canonical orientation (duplex-pf-cuda.md §8 item 7): on the CPU from the CPU
/// kernel's term port, on a CUDA device (`device` = ordinal) from the device producer
/// with the session's launch settings and routing. The public call's sequence checks
/// apply (`exec::validated` with the K3 kernel).
#[cfg(feature = "testing")]
#[pyfunction(signature=(call, a, b, device=None, threads=1))]
fn _native_duplex_pf_zhat<'py>(
    py: Python<'py>,
    call: PyRef<'_, PyDuplexCall>,
    a: PyRef<'_, super::PySeqTable>,
    b: PyRef<'_, super::PySeqTable>,
    device: Option<u64>,
    threads: usize,
) -> PyResult<(Bound<'py, PyBytes>, Bound<'py, PyBytes>)> {
    use seqblaze_core::{
        exec::{self, ExecOptions, Grain, Strategy},
        kernel::PairDistanceKernel,
        sink::Sink,
        work::{PairKind, WorkSource},
    };
    let Some(pf_call) = call.pf.as_ref() else {
        return Err(error(
            py,
            EngineError::Parameter("duplex_pf_zhat needs a model='pf' call".into()),
        ));
    };
    let kernel = seqblaze_thermo::DuplexPf::new(Arc::clone(pf_call), &[], false)
        .map_err(|e| error(py, e))?;
    let device = super::resolve_device(py, device)?;
    let work =
        WorkSource::new(PairKind::Zip, a.table.len(), b.table.len()).map_err(|e| error(py, e))?;
    let options: ExecOptions<i64> = ExecOptions {
        strategy: Strategy::Full,
        threads,
        grain: Grain::Auto,
        sink: Sink::Each,
        cutoff: None,
        check_bases: true,
        k: None,
    };
    let (ta, tb) = (
        a.table.host().map_err(|e| error(py, e))?,
        b.table.host().map_err(|e| error(py, e))?,
    );
    let (m, e) = py
        .detach(|| {
            let _guard = FpGuard::new();
            let call = match &device {
                Some(_) => {
                    let pack = kernel.call().device_pack()?;
                    Some(seqblaze_cuda::DeviceCall::DuplexPf(Arc::new(
                        seqblaze_cuda::PfDeviceCall {
                            loop_max: pack.loop_max,
                            n_p: pack.n_p,
                            n_c: pack.n_c,
                            cls: pack.cls,
                            offsets: pack.offsets,
                            m: pack.m,
                            e: pack.e,
                        },
                    )))
                }
                None => None,
            };
            exec::validated(&kernel, &work, &ta, &tb, &options, |a, b, _| {
                match (&device, call) {
                    (Some(device), Some(call)) if work.n_pairs > 0 => seqblaze_cuda::pf_zhat(
                        device,
                        &kernel,
                        call,
                        &work,
                        (a, b),
                        options.workers(),
                    ),
                    _ => {
                        let (mut m, mut e) = (Vec::new(), Vec::new());
                        for (i, j) in work.range(0, work.n_pairs) {
                            let (_, term) = kernel.evaluate_term(a.get(i), b.get(j))?;
                            m.push(term.m);
                            e.push(if term.m == 0 { 0 } else { term.e + 52 });
                        }
                        Ok((m, e))
                    }
                }
            })
        })
        .map_err(|e| error(py, e))?;
    let m: Vec<u8> = m.iter().flat_map(|v| v.to_ne_bytes()).collect();
    let e: Vec<u8> = e.iter().flat_map(|v| v.to_ne_bytes()).collect();
    Ok((PyBytes::new(py, &m), PyBytes::new(py, &e)))
}

/// Testing builds: one F53 operation per element on a CUDA device (duplex-pf-cuda.md
/// §5.5, §8 item 7): `op` "add" or "mul", operands as native-endian u64 (M) and i64 (e)
/// bytes of one length; a non-canonical operand raises ParameterError.
#[cfg(feature = "testing")]
#[pyfunction]
fn _native_cuda_f53<'py>(
    py: Python<'py>,
    op: &str,
    x_m: &[u8],
    x_e: &[u8],
    y_m: &[u8],
    y_e: &[u8],
    ordinal: u64,
) -> PyResult<(Bound<'py, PyBytes>, Bound<'py, PyBytes>)> {
    let op = seqblaze_cuda::F53Op::parse(op).ok_or_else(|| {
        error(
            py,
            EngineError::Parameter(format!("op={op:?}; use add or mul")),
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
    let signed = |words: Vec<u64>| -> Vec<i64> { words.into_iter().map(|w| w as i64).collect() };
    let (xm, xe, ym, ye) = (
        words(x_m)?,
        signed(words(x_e)?),
        words(y_m)?,
        signed(words(y_e)?),
    );
    let device = super::resolve_device(py, Some(ordinal))?.expect("an ordinal was given");
    let (m, e) = py
        .detach(|| {
            let _guard = FpGuard::new();
            seqblaze_cuda::f53_eval(&device, op, (&xm, &xe), (&ym, &ye))
        })
        .map_err(|e| error(py, e))?;
    let m: Vec<u8> = m.iter().flat_map(|v| v.to_ne_bytes()).collect();
    let e: Vec<u8> = e.iter().flat_map(|v| v.to_ne_bytes()).collect();
    Ok((PyBytes::new(py, &m), PyBytes::new(py, &e)))
}

/// Testing builds: W(d) = RN₅₃(exp(−10·d/(R·T))) as (μ, e) for |d| ≤ B of the call
/// (duplex-pf.md §5.1, §14.4), after K1's checks and the §5.4 exponent check.
#[cfg(feature = "testing")]
#[pyfunction(signature=(pack, temp_c, na_mM, k_mM, tris_mM, mg_mM, dntp_mM, oligo_a, oligo_b, d))]
#[allow(non_snake_case, clippy::too_many_arguments)]
fn _native_duplex_pf_weight(
    py: Python<'_>,
    pack: PyRef<'_, PyDuplexPack>,
    temp_c: f64,
    na_mM: f64,
    k_mM: f64,
    tris_mM: f64,
    mg_mM: f64,
    dntp_mM: f64,
    oligo_a: f64,
    oligo_b: Option<f64>,
    d: i64,
) -> PyResult<(f64, i64)> {
    use seqblaze_thermo::duplex::pf::PfConstants;
    let cond = conditions(
        temp_c, na_mM, k_mM, tris_mM, mg_mM, dntp_mM, oligo_a, oligo_b,
    );
    // The hook is called once per d; the per-call constants of the last (pack,
    // conditions) are kept, with the pack (testing only; values do not depend on it).
    type Cached = (Arc<Pack>, [u64; 8], i64, Arc<PfConstants>);
    static LAST: std::sync::Mutex<Option<Cached>> = std::sync::Mutex::new(None);
    let key = [
        temp_c.to_bits(),
        na_mM.to_bits(),
        k_mM.to_bits(),
        tris_mM.to_bits(),
        mg_mM.to_bits(),
        dntp_mM.to_bits(),
        oligo_a.to_bits(),
        oligo_b.map_or(u64::MAX, f64::to_bits),
    ];
    let pack = Arc::clone(&pack.pack);
    let w = py
        .detach(|| -> Result<_, EngineError> {
            // E0 for the GIL-released region (text-nulls-fp-environment.md §7.3 item 3).
            let _guard = FpGuard::new();
            let cached = LAST
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .as_ref()
                .filter(|(p, k, _, _)| Arc::ptr_eq(p, &pack) && *k == key)
                .map(|(_, _, bound, constants)| (*bound, Arc::clone(constants)));
            let (bound, constants) = match cached {
                Some(hit) => hit,
                None => {
                    let call = DuplexCall::new(Arc::clone(&pack), &cond)?;
                    let constants = Arc::new(PfConstants::new(&call, &cond)?);
                    *LAST.lock().unwrap_or_else(|e| e.into_inner()) =
                        Some((Arc::clone(&pack), key, call.bound, Arc::clone(&constants)));
                    (call.bound, constants)
                }
            };
            if d.unsigned_abs() > bound.unsigned_abs() {
                // Neither d nor B is printed (nndb2024-dna.md §5.3.8).
                return Err(EngineError::Parameter(
                    "duplex_pf_weight: |d| exceeds the call's bound B \
                     (docs/wiki/thermo/duplex-pf.md §14.4)"
                        .into(),
                ));
            }
            constants.weight(d)
        })
        .map_err(|e| error(py, e))?;
    Ok((w.mu, w.e))
}

/// Testing builds: the logarithm map g(μ, e) of duplex-pf.md §7.1 for μ ∈ [1, 2) and
/// |e| < 2^61; ParameterError otherwise.
#[cfg(feature = "testing")]
#[pyfunction]
fn _native_duplex_pf_log(py: Python<'_>, mu: f64, e: i128) -> PyResult<f64> {
    // μ ∈ [1, 2) checked by the bits, not by a float comparison (§7.3 item 4).
    if !(0x3FF0_0000_0000_0000..0x4000_0000_0000_0000).contains(&mu.to_bits())
        || e.unsigned_abs() >= 1 << 61
    {
        return Err(error(
            py,
            EngineError::Parameter(format!(
                "duplex_pf_log({mu:?}, {e}): needs 1 <= mu < 2 and |e| < 2^61"
            )),
        ));
    }
    // GIL held: the map runs in a guarded body that calls no Python (§7.3 item 4).
    Ok(seqblaze_core::fpenv::guarded(|| {
        seqblaze_thermo::duplex::pf::log_map(mu, e as i64)
    }))
}

/// Testing builds: the five fields of E(S) for the structure `pairs` of (a, b)
/// through the call's cell keys (§12.4).
#[cfg(feature = "testing")]
#[pyfunction]
fn _native_duplex_evaluate<'py>(
    py: Python<'py>,
    call: PyRef<'py, PyDuplexCall>,
    a: &[u8],
    b: &[u8],
    pairs: Vec<(i64, i64)>,
) -> PyResult<Bound<'py, PyDict>> {
    // The evaluation runs with the GIL held, so in a guarded body that calls no Python
    // code (§7.3 item 4); the dict is built after the guard is dropped.
    let duplex: &DuplexCall = &call.call;
    let row = seqblaze_core::fpenv::guarded(|| duplex.evaluate_structure(a, b, &pairs))
        .map_err(|e| error(py, e))?;
    let out = PyDict::new(py);
    out.set_item("dg_kcal", row.dg_kcal)?;
    out.set_item("dg_dcal", row.dg_dcal)?;
    out.set_item("dh_kcal", row.dh_kcal)?;
    out.set_item("ds_cal_per_k", row.ds_cal_per_k)?;
    out.set_item("tm_c", row.tm_c)?;
    Ok(out)
}

// --------------------------------------------------------------------------- K2, compat="primer3"
// docs/wiki/thermo/dimer-primer3.md §13 and §13.1: the per-call pack H1–H8, the per-call
// object of a run (pack, the loader's parameter handle, `end`) and the testing probe. Every
// GIL-released region starts with the host floating-point guard (D99).

/// One call's per-call constants H1–H8 (§5), Θ_Ω at Q₄ and Q₁ included.
#[pyclass(frozen, module = "seqblaze._core", name = "_DuplexPrimer3Pack")]
pub struct PyDimerPack {
    pub(crate) pack: DimerPack,
}

#[pymethods]
impl PyDimerPack {
    #[getter]
    fn mv(&self) -> f64 {
        self.pack.salt.mv
    }
    #[getter]
    fn w(&self) -> f64 {
        self.pack.salt.w
    }
    #[getter]
    fn mu(&self) -> f64 {
        self.pack.salt.mu
    }
    #[getter]
    fn x_mu(&self) -> f64 {
        self.pack.salt.x_mu
    }
    #[getter]
    fn ln_mu(&self) -> f64 {
        self.pack.salt.ln_mu
    }
    #[getter]
    fn sigma(&self) -> f64 {
        self.pack.salt.sigma
    }
    #[getter]
    fn t_u(&self) -> f64 {
        self.pack.salt.t_u
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
    fn q4(&self) -> f64 {
        self.pack.q4
    }
    #[getter]
    fn q1(&self) -> f64 {
        self.pack.q1
    }
    #[getter]
    fn theta_omega4(&self) -> f64 {
        self.pack.theta_omega4
    }
    #[getter]
    fn theta_omega1(&self) -> f64 {
        self.pack.theta_omega1
    }
}

/// Build the pack from the §4 mapping's inputs (mM, nM, °C): the mv sum, the Mg term, μ,
/// x_μ, λ, σ, T_u, x₄, x₁, Q₄, Q₁ and both Θ_Ω. μ zero or infinite, x_μ = 0 and x₄ = 0
/// raise ParameterError in that order (§8 step 1.4).
#[pyfunction(signature=(na, k, tris, mg, dntp, oligo_nM, temp_c))]
#[allow(non_snake_case, clippy::too_many_arguments)]
fn _duplex_primer3_pack(
    py: Python<'_>,
    na: f64,
    k: f64,
    tris: f64,
    mg: f64,
    dntp: f64,
    oligo_nM: f64,
    temp_c: f64,
) -> PyResult<PyDimerPack> {
    let args = DimerArgs {
        salt: Args {
            na,
            k,
            tris,
            mg,
            dntp,
            temp_c,
        },
        oligo_nm: oligo_nM,
    };
    let pack = py
        .detach(|| {
            // E0 for the GIL-released region (§13.1); `DimerPack::new` holds its own.
            let _guard = FpGuard::new();
            DimerPack::new(&args)
        })
        .map_err(|e| error(py, e))?;
    Ok(PyDimerPack { pack })
}

/// The per-call object of a K2 run: the pack, the parameter handle and `end`; passed to
/// `_native_run` / `_native_tiles` as `duplex=`.
#[pyclass(frozen, module = "seqblaze._core", name = "_DuplexPrimer3Call")]
pub struct PyDimerCall {
    pub(crate) pack: DimerPack,
    pub(crate) tables: Arc<P3Tables>,
    pub(crate) end: End,
}

#[pymethods]
impl PyDimerCall {
    #[getter]
    fn end(&self) -> &'static str {
        self.end.as_str()
    }
    #[getter]
    fn params(&self) -> &str {
        &self.tables.name
    }
    #[getter]
    fn params_sha256(&self) -> &str {
        &self.tables.sha256
    }
}

fn parse_end(py: Python<'_>, end: &str) -> PyResult<End> {
    End::parse(end).ok_or_else(|| {
        error(
            py,
            EngineError::Parameter(format!("end={end:?}; use one of 'any', 'end1', 'end2'")),
        )
    })
}

/// The per-call object of `sb.duplex(..., compat="primer3")` (no floating-point work).
#[pyfunction]
fn _duplex_primer3_call(
    py: Python<'_>,
    pack: PyRef<'_, PyDimerPack>,
    handle: PyRef<'_, PyPrimer3Params>,
    end: &str,
) -> PyResult<PyDimerCall> {
    Ok(PyDimerCall {
        pack: pack.pack,
        tables: Arc::clone(&handle.tables),
        end: parse_end(py, end)?,
    })
}

/// Testing builds: the backtrace pairs (u, z) of one pair, one-based in the coordinates of
/// dimer-primer3.md §6.1 of the mapped operands (END2: the swapped pair), the reported
/// cell included; [] when nothing is found (§13). The strands are checked as by the public
/// call: upper-case ACGT, then lengths 1–10,000 (a before b), then the pair rule.
#[cfg(feature = "testing")]
#[pyfunction]
fn _testing_duplex_primer3_probe(
    py: Python<'_>,
    a: &[u8],
    b: &[u8],
    pack: PyRef<'_, PyDimerPack>,
    handle: PyRef<'_, PyPrimer3Params>,
    end: &str,
) -> PyResult<Vec<(usize, usize)>> {
    use seqblaze_thermo::{dimer, DimerPrimer3};
    let end = parse_end(py, end)?;
    for (input, seq) in [("a", a), ("b", b)] {
        if let Some(position) = seq.iter().position(|c| !b"ACGT".contains(c)) {
            let byte = seq[position];
            return Err(error(
                py,
                EngineError::Sequence {
                    message: format!(
                        "duplex input {input} sequence 0 has an illegal byte {:?} at position \
                         {position}; compat='primer3' accepts upper-case A, C, G, T only (D42); \
                         nothing was computed",
                        char::from(byte)
                    ),
                    input,
                    index: 0,
                    position: Some(position),
                    byte: Some(byte),
                    n_bad: 1,
                    bad_indices: vec![0],
                },
            ));
        }
    }
    dimer::check_pair_lengths(a, b).map_err(|e| error(py, e))?;
    let kernel = DimerPrimer3::new(pack.pack, Arc::clone(&handle.tables), end, &[])
        .map_err(|e| error(py, e))?;
    let (a, b) = (a.to_vec(), b.to_vec());
    py.detach(|| {
        // E0 for the GIL-released region (§13.1): the probe recomputes the fill.
        let _guard = FpGuard::new();
        kernel.pairs(&a, &b)
    })
    .map_err(|e| error(py, e))
}

/// The device constants of K1's C1 bound (duplex-mfe-bound.md §8): the device pack's D
/// keys, CLS and table starts, and the certificate's relaxed classes as words.
#[cfg(any(feature = "testing", feature = "cuda"))]
pub(crate) fn device_bound_call(
    call: &DuplexCall,
    cert: &Certificate,
) -> Result<seqblaze_cuda::DupBoundCall, EngineError> {
    let pack = call.device_pack()?;
    let (relaxed, n_relaxed) = cert.device_words();
    Ok(seqblaze_cuda::DupBoundCall {
        loop_max: pack.loop_max,
        n_p: pack.n_p,
        n_c: pack.n_c,
        has_states: pack.offsets[14] > pack.offsets[13],
        cls: pack.cls.clone(),
        offsets: pack.offsets,
        keys: pack.full.iter().map(|key| key[0]).collect(),
        relaxed,
        n_relaxed,
    })
}

/// Testing builds: the C1 certificate of a K1 call as a dict (duplex-mfe-bound.md §10.2):
/// status, width, W, u_e and per class alpha, beta and the used regions with gamma.
#[cfg(feature = "testing")]
#[pyfunction]
fn _native_duplex_bound_certificate<'py>(
    py: Python<'py>,
    call: PyRef<'_, PyDuplexCall>,
) -> PyResult<Bound<'py, PyDict>> {
    use pyo3::types::PyList;
    let cert = call.hook_certificate(py, "duplex_bound_certificate")?;
    let out = PyDict::new(py);
    out.set_item("status", if cert.pass { "pass" } else { "fail" })?;
    out.set_item("width", cert.width())?;
    out.set_item("W", cert.w)?;
    out.set_item("u_e", seqblaze_thermo::duplex::bound::U_E)?;
    out.set_item("failures", PyList::new(py, &cert.failures)?)?;
    let classes = PyList::empty(py);
    for class in &cert.classes {
        let entry = PyDict::new(py);
        entry.set_item("alpha", class.alpha)?;
        entry.set_item("beta", class.beta)?;
        let regions = PyList::empty(py);
        for r in &class.regions {
            let region = PyDict::new(py);
            region.set_item("kind", if r.lower { "lower" } else { "upper" })?;
            region.set_item("d0", r.d0)?;
            region.set_item("start", r.start)?;
            region.set_item("gamma", r.gamma)?;
            regions.append(region)?;
        }
        entry.set_item("regions", regions)?;
        classes.append(entry)?;
    }
    out.set_item("classes", classes)?;
    Ok(out)
}

/// Testing builds: (LB, χ, outcome) of every zipped pair (a[k], b[k]) in the given
/// orientation (duplex-mfe-bound.md §10.2) as native-endian bytes: int64 `lb_dcal`
/// (INT64_MAX for +∞), bool `exact`, int8 `outcome` (0 open, 1 certified, 2 pruned)
/// against `threshold` (an engine key, or None). A failed certificate reports every
/// outcome open. The public call's sequence checks apply; `threads` workers.
#[cfg(feature = "testing")]
#[pyfunction(signature=(call, a, b, threshold=None, threads=1, device=None))]
fn _native_duplex_bound<'py>(
    py: Python<'py>,
    call: PyRef<'_, PyDuplexCall>,
    a: PyRef<'_, super::PySeqTable>,
    b: PyRef<'_, super::PySeqTable>,
    threshold: Option<i64>,
    threads: usize,
    device: Option<u64>,
) -> PyResult<(
    Bound<'py, PyBytes>,
    Bound<'py, PyBytes>,
    Bound<'py, PyBytes>,
)> {
    use seqblaze_core::{
        exec::{self, ExecOptions, Grain, Strategy},
        kernel::BoundOutcome,
        sink::Sink,
        work::{PairKind, WorkSource},
    };
    let cert = call.hook_certificate(py, "duplex_bound")?;
    let threshold = threshold
        .map(|t| {
            i32::try_from(t).map_err(|_| {
                error(
                    py,
                    EngineError::Parameter("threshold must be an int32 engine key".into()),
                )
            })
        })
        .transpose()?;
    let device = super::resolve_device(py, device)?;
    let duplex_call = Arc::clone(&call.call);
    let kernel = seqblaze_thermo::Duplex::new(Arc::clone(&call.call), &[])
        .map_err(|e| error(py, e))?
        .with_bound(Arc::clone(&cert));
    let work =
        WorkSource::new(PairKind::Zip, a.table.len(), b.table.len()).map_err(|e| error(py, e))?;
    let options: ExecOptions<i32> = ExecOptions {
        strategy: Strategy::Full,
        threads,
        grain: Grain::Auto,
        sink: Sink::Each,
        cutoff: None,
        check_bases: true,
        k: None,
    };
    let (ta, tb) = (
        a.table.host().map_err(|e| error(py, e))?,
        b.table.host().map_err(|e| error(py, e))?,
    );
    type Item = (i64, bool, i8);
    let items: Vec<Item> = py
        .detach(|| {
            let _guard = FpGuard::new();
            exec::validated(&kernel, &work, &ta, &tb, &options, |a, b, _| {
                if let Some(device) = &device {
                    // The device bound (duplex-mfe-bound.md §8): the device pack's D keys
                    // and the certificate's relaxed classes; a failed certificate leaves
                    // every outcome open, as on the CPU.
                    if !cert.pass {
                        return Ok(vec![(i64::MAX, false, 0); work.n_pairs]);
                    }
                    let bound_call = device_bound_call(&duplex_call, &cert)?;
                    let code = |view: seqblaze_core::seq::ByteView<'_>| -> Vec<u8> {
                        let mut bytes = vec![0u8; view.len()];
                        view.copy_to(0, &mut bytes);
                        seqblaze_thermo::duplex::pf::codes_of(&bytes)
                    };
                    let (mut xs, mut ys) = (Vec::new(), Vec::new());
                    for (i, j) in work.range(0, work.n_pairs) {
                        xs.push(code(a.get(i)));
                        ys.push(code(b.get(j)));
                    }
                    let xr: Vec<&[u8]> = xs.iter().map(Vec::as_slice).collect();
                    let yr: Vec<&[u8]> = ys.iter().map(Vec::as_slice).collect();
                    let lb_plus_one = seqblaze_core::plan::bound_fault()
                        == seqblaze_core::plan::BoundFault::LbPlusOne;
                    let (lb, chi, codes) = seqblaze_cuda::duplex_bound_eval(
                        device,
                        &bound_call,
                        &xr,
                        &yr,
                        threshold,
                        lb_plus_one,
                    )?;
                    return Ok(lb
                        .into_iter()
                        .zip(chi)
                        .zip(codes)
                        .map(|((lb, chi), code)| (lb, chi != 0, code as i8))
                        .collect());
                }
                let parts = seqblaze_core::sched::map_ranges(
                    work.n_pairs,
                    threads,
                    &|start, end| -> Result<Vec<Item>, EngineError> {
                        let _guard = FpGuard::new();
                        let mut out = Vec::with_capacity(end - start);
                        for (i, j) in work.range(start, end) {
                            let Some((lb, chi)) = kernel.bound_values(a.get(i), b.get(j))? else {
                                out.push((i64::MAX, false, 0));
                                continue;
                            };
                            let code = match seqblaze_thermo::duplex::bound::outcome_of(
                                lb, chi, threshold,
                            ) {
                                BoundOutcome::Open => 0,
                                BoundOutcome::Certified(_) => 1,
                                BoundOutcome::Pruned => 2,
                            };
                            out.push((lb.unwrap_or(i64::MAX), chi, code));
                        }
                        Ok(out)
                    },
                )?;
                Ok(parts.into_iter().flatten().collect())
            })
        })
        .map_err(|e| error(py, e))?;
    let lb: Vec<u8> = items.iter().flat_map(|x| x.0.to_ne_bytes()).collect();
    let exact: Vec<u8> = items.iter().map(|x| u8::from(x.1)).collect();
    let outcome: Vec<u8> = items.iter().map(|x| x.2 as u8).collect();
    Ok((
        PyBytes::new(py, &lb),
        PyBytes::new(py, &exact),
        PyBytes::new(py, &outcome),
    ))
}

/// Testing builds: the C2 certificate of a K3 call as a dict (duplex-pf-bound.md §10.2):
/// status, failures, u, g_star, s_star, b_b, l4 and per class alpha, beta, a, d and the
/// used regions with kind (T, B, LO, HI), index and gamma.
#[cfg(feature = "testing")]
#[pyfunction]
fn _native_duplex_pf_bound_certificate<'py>(
    py: Python<'py>,
    call: PyRef<'_, PyDuplexCall>,
) -> PyResult<Bound<'py, PyDict>> {
    use pyo3::types::PyList;
    let cert = call.pf_hook_certificate(py, "duplex_pf_bound_certificate")?;
    let out = PyDict::new(py);
    out.set_item("status", if cert.pass { "pass" } else { "fail" })?;
    out.set_item("failures", PyList::new(py, &cert.failures)?)?;
    out.set_item("u", seqblaze_thermo::duplex::pf_bound::U)?;
    out.set_item("g_star", cert.g_star)?;
    out.set_item("s_star", cert.s_star)?;
    out.set_item("b_b", cert.b_b)?;
    out.set_item("l4", cert.l4)?;
    let classes = PyList::empty(py);
    for class in &cert.classes {
        let entry = PyDict::new(py);
        entry.set_item("alpha", class.alpha)?;
        entry.set_item("beta", class.beta)?;
        entry.set_item("a", class.a)?;
        entry.set_item("d", class.d)?;
        let regions = PyList::empty(py);
        for r in &class.regions {
            let region = PyDict::new(py);
            region.set_item("kind", r.kind.as_str())?;
            region.set_item("index", r.index)?;
            region.set_item("gamma", r.gamma)?;
            regions.append(region)?;
        }
        entry.set_item("regions", regions)?;
        classes.append(entry)?;
    }
    out.set_item("classes", classes)?;
    Ok(out)
}

/// Testing builds: (Ẑ_b, L, outcome) of every zipped pair (a[k], b[k]) in canonical
/// orientation (duplex-pf-bound.md §10.2) as native-endian bytes: float64 `zb_mu` (0.0
/// for Ẑ_b = 0), int64 `zb_e` (0 for zero), float64 `lower_kcal` (+∞ for zero) and int8
/// `outcome` (0 open, 1 certified, 2 pruned) against `threshold` (an int64 engine key, or
/// None). A failed certificate reports every outcome open. The public call's sequence
/// checks apply; `threads` workers on the CPU, or the device bound (`device` = ordinal).
#[cfg(feature = "testing")]
#[pyfunction(signature=(call, a, b, threshold=None, threads=1, device=None))]
#[allow(clippy::type_complexity)]
fn _native_duplex_pf_bound<'py>(
    py: Python<'py>,
    call: PyRef<'_, PyDuplexCall>,
    a: PyRef<'_, super::PySeqTable>,
    b: PyRef<'_, super::PySeqTable>,
    threshold: Option<i64>,
    threads: usize,
    device: Option<u64>,
) -> PyResult<(
    Bound<'py, PyBytes>,
    Bound<'py, PyBytes>,
    Bound<'py, PyBytes>,
    Bound<'py, PyBytes>,
)> {
    use seqblaze_core::{
        exec::{self, ExecOptions, Grain, Strategy},
        sink::Sink,
        work::{PairKind, WorkSource},
    };
    use seqblaze_thermo::duplex::pf_bound;
    let cert = call.pf_hook_certificate(py, "duplex_pf_bound")?;
    let Some(pf_call) = call.pf.as_ref() else {
        return Err(error(
            py,
            EngineError::Parameter("duplex_pf_bound needs a model='pf' call".into()),
        ));
    };
    let device = super::resolve_device(py, device)?;
    let kernel = seqblaze_thermo::DuplexPf::new(Arc::clone(pf_call), &[], false)
        .map_err(|e| error(py, e))?
        .with_bound(Arc::clone(&cert));
    let work =
        WorkSource::new(PairKind::Zip, a.table.len(), b.table.len()).map_err(|e| error(py, e))?;
    let options: ExecOptions<i64> = ExecOptions {
        strategy: Strategy::Full,
        threads,
        grain: Grain::Auto,
        sink: Sink::Each,
        cutoff: None,
        check_bases: true,
        k: None,
    };
    let (ta, tb) = (
        a.table.host().map_err(|e| error(py, e))?,
        b.table.host().map_err(|e| error(py, e))?,
    );
    type Item = (f64, i64, f64, i8);
    let open: Item = (0.0, 0, f64::INFINITY, 0);
    let items: Vec<Item> = py
        .detach(|| {
            let _guard = FpGuard::new();
            exec::validated(&kernel, &work, &ta, &tb, &options, |a, b, _| {
                if !cert.pass {
                    return Ok(vec![open; work.n_pairs]);
                }
                if let Some(device) = &device {
                    let code = |view: seqblaze_core::seq::ByteView<'_>| -> Vec<u8> {
                        let mut bytes = vec![0u8; view.len()];
                        view.copy_to(0, &mut bytes);
                        seqblaze_thermo::duplex::pf::codes_of(&bytes)
                    };
                    let (mut xs, mut ys) = (Vec::new(), Vec::new());
                    for (i, j) in work.range(0, work.n_pairs) {
                        xs.push(code(a.get(i)));
                        ys.push(code(b.get(j)));
                    }
                    let xr: Vec<&[u8]> = xs.iter().map(Vec::as_slice).collect();
                    let yr: Vec<&[u8]> = ys.iter().map(Vec::as_slice).collect();
                    // Ẑ_b (with the testing fault), L and the outcome all come from the
                    // device (duplex-pf-bound.md §8, QZ11).
                    let bound_call = device_pf_bound_call(pf_call, &cert)?;
                    let (m, e, l, code) = seqblaze_cuda::duplex_pf_bound_eval(
                        device,
                        &bound_call,
                        &xr,
                        &yr,
                        threshold,
                    )?;
                    return (0..m.len())
                        .map(|t| -> Result<Item, EngineError> {
                            let z = pf_bound::zb_of_device(m[t], e[t])?;
                            Ok((z.mu, z.e, f64::from_bits(l[t]), code[t] as i8))
                        })
                        .collect();
                }
                let parts = seqblaze_core::sched::map_ranges(
                    work.n_pairs,
                    threads,
                    &|start, end| -> Result<Vec<Item>, EngineError> {
                        let _guard = FpGuard::new();
                        let mut out = Vec::with_capacity(end - start);
                        for (i, j) in work.range(start, end) {
                            let Some((z, l, o)) =
                                kernel.bound_values(a.get(i), b.get(j), threshold)?
                            else {
                                out.push(open);
                                continue;
                            };
                            out.push((z.mu, z.e, l, pf_bound::outcome_code(o)));
                        }
                        Ok(out)
                    },
                )?;
                Ok(parts.into_iter().flatten().collect())
            })
        })
        .map_err(|e| error(py, e))?;
    let mu: Vec<u8> = items.iter().flat_map(|x| x.0.to_ne_bytes()).collect();
    let e: Vec<u8> = items.iter().flat_map(|x| x.1.to_ne_bytes()).collect();
    let lower: Vec<u8> = items.iter().flat_map(|x| x.2.to_ne_bytes()).collect();
    let outcome: Vec<u8> = items.iter().map(|x| x.3 as u8).collect();
    Ok((
        PyBytes::new(py, &mu),
        PyBytes::new(py, &e),
        PyBytes::new(py, &lower),
        PyBytes::new(py, &outcome),
    ))
}

/// The device constants of K3's C2 bound (duplex-pf-bound.md §8): K3's device pack (the
/// weight tables) and the certificate's relaxed classes as words.
#[cfg(any(feature = "testing", feature = "cuda"))]
pub(crate) fn device_pf_bound_call(
    pf: &PfCall,
    cert: &PfCertificate,
) -> Result<seqblaze_cuda::PfBoundCall, EngineError> {
    let pack = pf.device_pack()?;
    let (relaxed, n_relaxed) = cert.device_words();
    // K3's finalize constants for L on the device (duplex-pf-bound.md §8, QZ11): the
    // resident finalize's 135 words.
    #[cfg(feature = "cuda")]
    let finalize = {
        let c = super::pf_device_finalize::constants(pf);
        let mut words = c.table.to_vec();
        words.extend_from_slice(&[c.l2, c.c_t, c.kappa, c.b3, c.b5, c.b7, c.loop_max]);
        words
    };
    #[cfg(not(feature = "cuda"))]
    let finalize = Vec::new();
    Ok(seqblaze_cuda::PfBoundCall {
        loop_max: pack.loop_max,
        n_p: pack.n_p,
        n_c: pack.n_c,
        cls: pack.cls,
        offsets: pack.offsets,
        m: pack.m,
        e: pack.e,
        relaxed,
        n_relaxed,
        finalize,
        scale_down: seqblaze_core::plan::bound_fault()
            == seqblaze_core::plan::BoundFault::ZbScaleDown,
    })
}

/// Testing builds: set the bound fault of `kind`'s slot (duplex-mfe-bound.md §10.2) to
/// `kind` ("gamma_plus_one", "lb_plus_one": certificate slot; "probe_use",
/// "probe_skip": decision slot), returning the slot's previous kind. `kind="none"`
/// clears the slot of `slot` (a kind naming the slot), or both slots without it. The
/// two slots compose, so nested faults of different slots both act.
#[cfg(feature = "testing")]
#[pyfunction(signature=(kind, slot=None))]
fn _native_duplex_bound_fault(
    py: Python<'_>,
    kind: &str,
    slot: Option<&str>,
) -> PyResult<&'static str> {
    use seqblaze_core::plan::{set_bound_fault, BoundFault, FaultSlot};
    let parse = |kind: &str| -> PyResult<BoundFault> {
        Ok(match kind {
            "none" => BoundFault::None,
            "gamma_plus_one" => BoundFault::GammaPlusOne,
            "lb_plus_one" => BoundFault::LbPlusOne,
            "zb_scale_down" => BoundFault::ZbScaleDown,
            "probe_use" => BoundFault::ProbeUse,
            "probe_skip" => BoundFault::ProbeSkip,
            _ => {
                return Err(error(
                    py,
                    EngineError::Parameter(format!(
                        "unknown bound fault {kind:?}; use gamma_plus_one, lb_plus_one, \
                         zb_scale_down, probe_use or probe_skip"
                    )),
                ))
            }
        })
    };
    let name = |fault: BoundFault| match fault {
        BoundFault::None => "none",
        BoundFault::GammaPlusOne => "gamma_plus_one",
        BoundFault::LbPlusOne => "lb_plus_one",
        BoundFault::ZbScaleDown => "zb_scale_down",
        BoundFault::ProbeUse => "probe_use",
        BoundFault::ProbeSkip => "probe_skip",
    };
    let fault = parse(kind)?;
    let slots: Vec<FaultSlot> = match (fault.slot(), slot) {
        (Some(slot), _) => vec![slot],
        (None, Some(other)) => parse(other)?.slot().into_iter().collect(),
        (None, None) => vec![FaultSlot::Certificate, FaultSlot::Decision],
    };
    let mut previous = BoundFault::None;
    for slot in slots {
        let old = set_bound_fault(slot, fault);
        if old != BoundFault::None {
            previous = old;
        }
    }
    Ok(name(previous))
}

/// Testing builds: the K1 bound evaluations of `sb.duplex` passes since the last reset
/// (duplex-mfe-bound.md §10.2, probe evaluations included); `reset` zeroes the count
/// and returns the count before it.
#[cfg(feature = "testing")]
#[pyfunction(signature=(reset=false))]
fn _native_duplex_bound_evals(reset: bool) -> u64 {
    seqblaze_core::plan::bound_evals(reset)
}

/// Testing builds: the certificate builds started since the last reset (duplex-mfe-bound.md
/// §10.2, revision 3); `reset` zeroes the count and returns the count before it.
#[cfg(feature = "testing")]
#[pyfunction(signature=(reset=false))]
fn _native_duplex_bound_certificate_builds(reset: bool) -> u64 {
    seqblaze_core::plan::certificate_builds(reset)
}

/// Testing builds: set the device cost override (duplex-mfe-bound.md §10.2
/// `duplex_bound_device_costs`, revision 3) to (c_hat, q_hat), or clear it with None,
/// returning the previous override (None if none was active).
#[cfg(feature = "testing")]
#[pyfunction(signature=(costs=None))]
fn _native_duplex_bound_device_costs(
    py: Python<'_>,
    costs: Option<(u32, u32)>,
) -> PyResult<Option<(u32, u32)>> {
    seqblaze_core::plan::set_device_costs_override(costs).map_err(|e| error(py, e))
}

pub fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyDuplexPack>()?;
    module.add_class::<PyDuplexCall>()?;
    module.add_class::<PyDimerPack>()?;
    module.add_class::<PyDimerCall>()?;
    module.add_function(wrap_pyfunction!(_duplex_preset, module)?)?;
    module.add_function(wrap_pyfunction!(_duplex_call, module)?)?;
    module.add_function(wrap_pyfunction!(_duplex_lse, module)?)?;
    module.add_function(wrap_pyfunction!(_duplex_pf_lse, module)?)?;
    module.add_function(wrap_pyfunction!(_duplex_primer3_pack, module)?)?;
    module.add_function(wrap_pyfunction!(_duplex_primer3_call, module)?)?;
    #[cfg(feature = "testing")]
    module.add_function(wrap_pyfunction!(_testing_duplex_primer3_probe, module)?)?;
    #[cfg(feature = "testing")]
    module.add_function(wrap_pyfunction!(_native_duplex_pack, module)?)?;
    #[cfg(feature = "testing")]
    module.add_function(wrap_pyfunction!(_native_duplex_evaluate, module)?)?;
    #[cfg(feature = "testing")]
    module.add_function(wrap_pyfunction!(_native_duplex_pf_scaled, module)?)?;
    #[cfg(feature = "testing")]
    module.add_function(wrap_pyfunction!(_native_duplex_pf_weight, module)?)?;
    #[cfg(feature = "testing")]
    module.add_function(wrap_pyfunction!(_native_duplex_pf_log, module)?)?;
    #[cfg(feature = "testing")]
    module.add_function(wrap_pyfunction!(_native_duplex_pf_zhat, module)?)?;
    #[cfg(feature = "testing")]
    module.add_function(wrap_pyfunction!(_native_cuda_f53, module)?)?;
    #[cfg(feature = "testing")]
    module.add_function(wrap_pyfunction!(_native_duplex_bound, module)?)?;
    #[cfg(feature = "testing")]
    module.add_function(wrap_pyfunction!(_native_duplex_bound_certificate, module)?)?;
    #[cfg(feature = "testing")]
    module.add_function(wrap_pyfunction!(_native_duplex_pf_bound, module)?)?;
    #[cfg(feature = "testing")]
    module.add_function(wrap_pyfunction!(
        _native_duplex_pf_bound_certificate,
        module
    )?)?;
    #[cfg(feature = "testing")]
    module.add_function(wrap_pyfunction!(_native_duplex_bound_fault, module)?)?;
    #[cfg(feature = "testing")]
    module.add_function(wrap_pyfunction!(_native_duplex_bound_evals, module)?)?;
    #[cfg(feature = "testing")]
    module.add_function(wrap_pyfunction!(
        _native_duplex_bound_certificate_builds,
        module
    )?)?;
    #[cfg(feature = "testing")]
    module.add_function(wrap_pyfunction!(_native_duplex_bound_device_costs, module)?)?;
    Ok(())
}
