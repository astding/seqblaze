//! Lease-local, monotone scratch placement (per-call cost S7). Published
//! columns never use this allocator. Retirement remains host-proved; S4 is off.
use crate::interop::{execution_stream::BorrowedExecutionStream, Retained};
use cudarc::driver::{CudaSlice, DeviceRepr, DriverError};
use std::{
    ops::{Deref, DerefMut},
    sync::{Arc, Mutex},
};

const BYTES: usize = 1 << 20;

pub(crate) struct Arena {
    enabled: bool,
    storage: Option<Arc<Retained<CudaSlice<u8>>>>,
    used: usize,
}
impl Default for Arena {
    fn default() -> Self {
        Self {
            enabled: true,
            storage: None,
            used: 0,
        }
    }
}

/// A typed allocation, or a non-owning typed view retaining its arena. A view
/// must never enqueue a free of its interior pointer or become a Result column.
pub struct Scratch<T> {
    view: Option<CudaSlice<T>>,
    arena: Option<Arc<Retained<CudaSlice<u8>>>>,
}
impl<T> Scratch<T> {
    pub(crate) fn owned(value: CudaSlice<T>) -> Self {
        Self {
            view: Some(value),
            arena: None,
        }
    }
}
impl<T> Deref for Scratch<T> {
    type Target = CudaSlice<T>;
    fn deref(&self) -> &Self::Target {
        self.view.as_ref().expect("live scratch")
    }
}
impl<T> DerefMut for Scratch<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.view.as_mut().expect("live scratch")
    }
}
impl<T> Drop for Scratch<T> {
    fn drop(&mut self) {
        if self.arena.is_some() {
            // Tracking is disabled (S1): leak releases only this view's stream
            // reference, with no driver work. The arena retains the real base
            // allocation and its host-proved retirement guard.
            self.view.take().expect("live scratch").leak();
        }
    }
}

impl Arena {
    pub(crate) fn disable(slot: &Mutex<Self>) {
        let mut arena = slot.lock().unwrap_or_else(|p| p.into_inner());
        assert!(
            arena.storage.is_none(),
            "Tiles must disable the arena before binding"
        );
        arena.enabled = false;
    }
    /// # Safety
    /// As BorrowedExecutionStream::alloc: each read is preceded by initialization.
    pub(crate) unsafe fn alloc<T: DeviceRepr>(
        slot: &Mutex<Self>,
        execution: &BorrowedExecutionStream,
        len: usize,
    ) -> Result<Scratch<T>, DriverError> {
        let bytes = len.saturating_mul(std::mem::size_of::<T>());
        let rounded = bytes.checked_add(255).map(|n| n & !255);
        // No driver operation or polling wait runs under a bookkeeping lock.
        let create = {
            let arena = slot.lock().unwrap_or_else(|p| p.into_inner());
            arena.enabled
                && rounded.is_some_and(|n| n <= BYTES - arena.used)
                && arena.storage.is_none()
        };
        if create {
            // A lease is used on its calling thread. Even a reentrant call has
            // its own arena; the lock is not held during pool retry/waits.
            let value = unsafe { execution.alloc::<u8>(BYTES) }?;
            let storage = Arc::new(Retained::on_owned_stream(
                execution.allocation(),
                execution.device(),
                value,
            ));
            slot.lock().unwrap_or_else(|p| p.into_inner()).storage = Some(storage);
        }
        let span = {
            let mut arena = slot.lock().unwrap_or_else(|p| p.into_inner());
            if arena.enabled && rounded.is_some_and(|n| n <= BYTES - arena.used) {
                let offset = arena.used;
                arena.used += rounded.expect("bounded scratch");
                Some((
                    Arc::clone(arena.storage.as_ref().expect("created arena")),
                    offset,
                ))
            } else {
                None
            }
        };
        let Some((storage, offset)) = span else {
            return unsafe { execution.alloc(len) }.map(Scratch::owned);
        };
        let address = crate::session::ptr(&storage, execution.allocation()) + offset as u64;
        // SAFETY: the disjoint, 256-byte aligned span is inside the retained
        // base allocation. Scratch::drop removes the view without freeing it.
        let view = unsafe { execution.allocation().upgrade_device_ptr::<T>(address, len) };
        let result = Scratch {
            view: Some(view),
            arena: Some(storage),
        };
        #[cfg(feature = "testing")]
        if let Some(byte) = crate::launch::poison() {
            // Poison each sub-allocation before first use, including reused
            // pool pages; count only bytes actually filled here.
            unsafe { execution.memset(address, byte, bytes) }?;
            crate::launch::count_poisoned(bytes as u64);
        }
        Ok(result)
    }
}
