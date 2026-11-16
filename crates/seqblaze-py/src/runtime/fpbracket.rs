//! The caller-state bracket as a private `_core` context manager (D99;
//! docs/wiki/engine/text-nulls-fp-environment.md §7.2 and §7.3 item 5).
//!
//! `with _core._FpBracket(): ...` saves the calling thread's whole
//! floating-point state at entry (x86-64: MXCSR; aarch64: FPCR and FPSR) and
//! writes it back at the `with` block's exit, whatever the exit is: normal end,
//! exception, `KeyboardInterrupt`, or a panic of native code inside (raised as
//! `PanicException`). It installs nothing, so the Python code inside runs in
//! the caller's environment; only what that code leaves in the state is
//! undone. No method accepts a state word: only a word read on the same
//! thread is ever written back.
use pyo3::{exceptions::PyRuntimeError, prelude::*};
use seqblaze_core::fpenv::FpBracket;

/// Single use, bound to the thread that created it (`unsendable`: a use from
/// another thread raises PyO3's `PanicException` before any state is read or
/// written, and a drop on another thread is leaked by PyO3, writing nothing).
#[pyclass(unsendable, module = "seqblaze._core", name = "_FpBracket")]
pub struct PyFpBracket {
    held: Option<FpBracket>,
    used: bool,
}

#[pymethods]
impl PyFpBracket {
    #[new]
    fn new() -> Self {
        Self {
            held: None,
            used: false,
        }
    }

    /// Save the calling thread's state. A second entry raises and reads nothing.
    fn __enter__(&mut self) -> PyResult<()> {
        if self.used {
            return Err(PyRuntimeError::new_err(
                "a floating-point state bracket is used once only",
            ));
        }
        self.used = true;
        self.held = Some(FpBracket::new());
        Ok(())
    }

    /// Write the saved state back; never suppresses the exception. Without a
    /// preceding entry nothing is written.
    fn __exit__(
        &mut self,
        _exc_type: &Bound<'_, PyAny>,
        _exc_value: &Bound<'_, PyAny>,
        _traceback: &Bound<'_, PyAny>,
    ) -> bool {
        drop(self.held.take());
        false
    }
}

impl Drop for PyFpBracket {
    /// An object deallocated while still entered (its `__exit__` never ran)
    /// writes nothing: the thread's state may have moved on since the entry,
    /// and a late write would overwrite it.
    fn drop(&mut self) {
        if let Some(bracket) = self.held.take() {
            std::mem::forget(bracket);
        }
    }
}

pub fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyFpBracket>()
}
