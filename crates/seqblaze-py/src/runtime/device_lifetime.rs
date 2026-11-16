//! Native device owners may synchronize their allocation stream on Drop.
//! Keep that cleanup outside the GIL, including early Python errors and GC of
//! results/iterators from a previous call. Python exporters remain owned until
//! the native guards finish; their own PyO3 handles arrange Python retirement.
use std::ops::{Deref, DerefMut};

use pyo3::Python;

pub(super) struct Detached<T: Send>(Option<T>);
impl<T: Send> Detached<T> {
    pub(super) fn new(value: T) -> Self {
        Self(Some(value))
    }
    pub(super) fn take(mut self) -> T {
        self.0.take().expect("live native owner")
    }
}
impl<T: Send> Deref for Detached<T> {
    type Target = T;
    fn deref(&self) -> &T {
        self.0.as_ref().expect("live native owner")
    }
}
impl<T: Send> DerefMut for Detached<T> {
    fn deref_mut(&mut self) -> &mut T {
        self.0.as_mut().expect("live native owner")
    }
}
impl<T: Send> Drop for Detached<T> {
    fn drop(&mut self) {
        if let Some(value) = self.0.take() {
            let mut value = Some(value);
            if Python::try_attach(|py| py.detach(|| drop(value.take()))).is_none() {
                // Interpreter finalization cannot run exporter callbacks or
                // acquire the GIL. Preserve ownership rather than free DMA
                // storage without its native retirement guard.
                std::mem::forget(value);
            }
        }
    }
}
