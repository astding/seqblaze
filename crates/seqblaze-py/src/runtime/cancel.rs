//! The binding side of Ctrl-C cancellation and progress (cancellation-progress.md
//! §3.3, §3.4, §3.7, §4.4): the per-call scope with its Python watcher, the
//! GIL-released region helper and the outcome precedence; in testing builds the
//! `_testing_cancel_probe*` functions (§7).
use std::sync::{Arc, Mutex};

use pyo3::{exceptions::PyException, intern, prelude::*};
use seqblaze_core::{
    cancel::{PollOutcome, Scope, Watcher},
    fpenv::FpGuard,
    validate::EngineError,
};

/// The watcher of one call: signal checks and progress writes under the GIL.
#[derive(Default)]
struct PyWatcher {
    /// The first exception a handler (or a progress write) raised.
    exception: Mutex<Option<PyErr>>,
}

impl PyWatcher {
    fn store(&self, error: PyErr) {
        let mut slot = self.exception.lock().unwrap_or_else(|e| e.into_inner());
        if slot.is_none() {
            *slot = Some(error);
        }
    }

    fn take(&self) -> Option<PyErr> {
        self.exception
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .take()
    }
}

/// Write one progress line to `sys.stderr` (nothing when it is missing or None).
fn write_line(py: Python<'_>, line: &str) -> PyResult<bool> {
    let sys = py.import(intern!(py, "sys"))?;
    let Ok(stderr) = sys.getattr(intern!(py, "stderr")) else {
        return Ok(false);
    };
    if stderr.is_none() {
        return Ok(false);
    }
    stderr.call_method1(intern!(py, "write"), (line,))?;
    if stderr.hasattr(intern!(py, "flush"))? {
        stderr.call_method0(intern!(py, "flush"))?;
    }
    Ok(true)
}

impl Watcher for PyWatcher {
    fn poll(&self, line: Option<&str>) -> PollOutcome {
        // An interpreter that is finalizing is not attached to: the call continues.
        Python::try_attach(|py| {
            let mut outcome = PollOutcome {
                attached: true,
                ..PollOutcome::default()
            };
            if let Err(error) = py.check_signals() {
                self.store(error);
                outcome.cancel = true;
                return outcome;
            }
            if let Some(line) = line {
                match write_line(py, line) {
                    Ok(written) => outcome.line_written = written,
                    // A failed write is reported once and switches progress off.
                    Err(error) if error.is_instance_of::<PyException>(py) => {
                        error.write_unraisable(py, None);
                        outcome.progress_off = true;
                    }
                    // KeyboardInterrupt, SystemExit, ...: the call is cancelled.
                    Err(error) => {
                        self.store(error);
                        outcome.cancel = true;
                    }
                }
            }
            outcome
        })
        .unwrap_or_default()
    }
}

/// One public compute call's cancel scope (§3.1): created once per call, entered
/// by every GIL-released region of the call, ended when it is dropped.
pub(crate) struct Call {
    scope: Arc<Scope>,
    watcher: Arc<PyWatcher>,
}

impl Call {
    /// `progress` is the public kernel name when progress lines are on.
    pub(crate) fn new(progress: Option<&str>) -> Self {
        let watcher = Arc::new(PyWatcher::default());
        let scope = Scope::new(watcher.clone(), progress);
        Self { scope, watcher }
    }

    /// One GIL-released compute region: E0 under the region's outermost guard,
    /// inside this call's scope (§3.7).
    pub(crate) fn detach<T, F>(&self, py: Python<'_>, body: F) -> T
    where
        F: Send + FnOnce() -> T,
        T: Send,
    {
        let scope = &self.scope;
        py.detach(move || {
            // E0 for the whole GIL-released region (text-nulls-fp-environment.md §7.3 item 3).
            let guard = FpGuard::new();
            let _entered = scope.enter(guard.saved());
            body()
        })
    }

    /// One region of a call made of several regions: [`Self::detach`], then the
    /// precedence of §3.4 for this step. A stored handler exception is raised
    /// even when the region succeeded, so no later region of the call runs.
    /// Only the CUDA entries have calls of several regions.
    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    pub(crate) fn region<T, F>(&self, py: Python<'_>, body: F) -> PyResult<T>
    where
        F: Send + FnOnce() -> Result<T, EngineError>,
        T: Send,
    {
        let result = self.detach(py, body);
        if let Some(raised) = self.watcher.take() {
            drop(result);
            return Err(raised);
        }
        result.map_err(|e| super::error(py, e))
    }

    /// The exception a handler raised during this call, if any (§3.4 rule 1).
    pub(crate) fn exception(&self) -> Option<PyErr> {
        self.scope.end();
        self.watcher.take()
    }

    /// The outcome of the call (§3.4): a stored handler exception wins over any
    /// result or error; otherwise the engine result, mapped as usual.
    pub(crate) fn finish<T>(self, py: Python<'_>, result: Result<T, EngineError>) -> PyResult<T> {
        if let Some(raised) = self.exception() {
            drop(result);
            return Err(raised);
        }
        result.map_err(|e| super::error(py, e))
    }
}

impl Drop for Call {
    fn drop(&mut self) {
        self.scope.end();
    }
}

#[cfg(feature = "testing")]
mod probe {
    use pyo3::{
        prelude::*,
        types::{PyDict, PyList},
    };
    use seqblaze_core::{
        cancel::{probe, Site},
        validate::EngineError,
    };
    use std::time::Duration;

    /// Testing builds: arm, re-arm or disarm the process-wide probe (§7). The
    /// argument checks are in `seqblaze._testing.cancel_probe`.
    #[pyfunction(signature=(stage, nth, timeout_ms, progress_interval_ms))]
    fn _native_cancel_probe(
        py: Python<'_>,
        stage: Option<&str>,
        nth: u64,
        timeout_ms: u64,
        progress_interval_ms: Option<u64>,
    ) -> PyResult<()> {
        let target = match stage {
            None => None,
            Some("any") => Some(None),
            Some(key) => Some(Some(Site::from_key(key).ok_or_else(|| {
                super::super::error(
                    py,
                    EngineError::Parameter(format!("unknown cancel probe stage {key:?}")),
                )
            })?)),
        };
        let interval = if target.is_none() {
            None
        } else {
            progress_interval_ms.map(Duration::from_millis)
        };
        probe::arm(target, nth, timeout_ms, interval);
        Ok(())
    }

    /// Testing builds: a fresh dict of the probe's statistics (§7).
    #[pyfunction]
    fn _native_cancel_probe_state(py: Python<'_>) -> PyResult<Bound<'_, PyDict>> {
        let snap = probe::snapshot();
        let out = PyDict::new(py);
        out.set_item("paused", snap.paused)?;
        out.set_item("pauses", snap.pauses)?;
        let checkpoints = PyDict::new(py);
        let items = PyDict::new(py);
        for (k, site) in Site::ALL.iter().enumerate() {
            checkpoints.set_item(site.key(), snap.checkpoints[k])?;
            items.set_item(site.key(), snap.items[k])?;
        }
        out.set_item("checkpoints", checkpoints)?;
        out.set_item("items", items)?;
        out.set_item("max_gap", snap.max_gap)?;
        out.set_item("polls", snap.polls)?;
        out.set_item("cancelled", snap.cancelled)?;
        out.set_item("live_units", snap.live_units)?;
        out.set_item("progress_lines", snap.progress_lines)?;
        let stages = PyList::empty(py);
        for row in snap.stages {
            stages.append(
                (row.label, row.unit, row.total, row.done)
                    .into_pyobject(py)?
                    .to_list(),
            )?;
        }
        out.set_item("stages", stages)?;
        Ok(out)
    }

    /// Testing builds: release a paused checkpoint (nothing if none is paused).
    #[pyfunction]
    fn _native_cancel_probe_release() {
        probe::release();
    }

    pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
        module.add_function(wrap_pyfunction!(_native_cancel_probe, module)?)?;
        module.add_function(wrap_pyfunction!(_native_cancel_probe_state, module)?)?;
        module.add_function(wrap_pyfunction!(_native_cancel_probe_release, module)?)?;
        Ok(())
    }
}

pub fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    #[cfg(feature = "testing")]
    probe::register(module)?;
    #[cfg(not(feature = "testing"))]
    let _ = module;
    Ok(())
}
