//! Per-device driver resources shared by every call on one ordinal
//! (docs/wiki/gpu/cuda-backend.md, "Per-call host and driver cost", S1 (c) and
//! S2): the free list of ordering events and the seqblaze-owned stream-ordered
//! memory pool. Both live as long as the process; neither is ever destroyed
//! during it (at exit the statics never destruct CUDA objects). Their locks
//! cover host bookkeeping only: no driver wait, copy or poll runs under them
//! (PC5 I12).
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Mutex,
};

use cudarc::driver::{sys, DriverError};

/// Q1 (approved 2026-11-06): the owned pool keeps up to this many bytes mapped
/// between calls. While a call scope is open the threshold is lifted
/// ([`DevicePool::enter`]; amendment Q1a of Q1, D134, 2026-11-11, lane `coldgpu`).
pub(crate) const RELEASE_THRESHOLD: u64 = 256 * 1024 * 1024;

/// Initial capacity of an ordinal's event list: steady-state pushes never
/// allocate unless more than this many `order()` calls run at once.
const EVENT_CAPACITY: usize = 64;

/// A `CU_EVENT_DISABLE_TIMING` event of one ordinal's primary context.
struct Event(sys::CUevent);
// SAFETY: a driver handle; every use binds the owning context first, and each
// event is held by one thread for one record-and-wait pair (S1 (c)).
unsafe impl Send for Event {}

/// The free list of ordering events (S1 (c)): grows on demand, bounded by the
/// peak number of concurrent `order()` calls, never destroyed.
pub(crate) struct EventList(Mutex<Vec<Event>>);

impl EventList {
    pub(crate) fn new() -> Self {
        Self(Mutex::new(Vec::with_capacity(EVENT_CAPACITY)))
    }

    /// Record `from`'s current work into a listed event and make `to` wait on
    /// it. `cuStreamWaitEvent` waits for the record made at the time of the
    /// call, so returning the event right after the wait keeps the dependency.
    /// After a failed record the event is still returned: it is recorded again
    /// before its next wait. The caller has bound the owning context.
    pub(crate) fn order(&self, from: sys::CUstream, to: sys::CUstream) -> Result<(), Created> {
        let taken = self.0.lock().unwrap_or_else(|p| p.into_inner()).pop();
        let event = match taken {
            Some(event) => event,
            None => {
                let mut event = std::ptr::null_mut();
                // SAFETY: the caller bound this ordinal's context.
                unsafe {
                    sys::cuEventCreate(
                        &mut event,
                        sys::CUevent_flags::CU_EVENT_DISABLE_TIMING as u32,
                    )
                }
                .result()
                .map_err(Created::Failed)?;
                Event(event)
            }
        };
        // SAFETY: both streams belong to the bound context of this event.
        let queued = unsafe { sys::cuEventRecord(event.0, from) }
            .result()
            .and_then(|()| unsafe { sys::cuStreamWaitEvent(to, event.0, 0) }.result());
        self.give(event);
        queued.map_err(Created::Queued)
    }

    fn give(&self, event: Event) {
        let mut list = self.0.lock().unwrap_or_else(|p| p.into_inner());
        if list.try_reserve(1).is_ok() {
            list.push(event);
        } else {
            drop(list);
            // A refused host reservation (only past EVENT_CAPACITY concurrent
            // orders) destroys the event instead of growing the list.
            // SAFETY: no enqueued wait depends on the handle staying alive.
            let _ = unsafe { sys::cuEventDestroy_v2(event.0) };
        }
    }
}

/// Which step of [`EventList::order`] failed, so callers keep their existing
/// operation-specific diagnostics.
#[derive(Clone, Copy, Debug)]
pub(crate) enum Created {
    Failed(DriverError),
    Queued(DriverError),
}
impl Created {
    pub(crate) fn error(self) -> DriverError {
        match self {
            Self::Failed(error) | Self::Queued(error) => error,
        }
    }
}

/// The seqblaze-owned pool of one ordinal (S2, Q1): device-pinned memory, no
/// IPC handle types, release threshold [`RELEASE_THRESHOLD`] between calls,
/// internal-dependency reuse off; follow-event-dependencies and opportunistic
/// reuse keep their default (on). The device's default and current pools are
/// never modified.
pub(crate) struct DevicePool {
    pool: sys::CUmemoryPool,
    /// Open call scopes on this ordinal ([`DevicePool::enter`]). The lock
    /// covers the count and the threshold write that goes with its 0 ↔ 1
    /// transitions: no driver wait, copy or poll runs under it (PC5 I12).
    scopes: Mutex<usize>,
}
// SAFETY: a driver handle that is only passed to thread-safe driver calls.
unsafe impl Send for DevicePool {}
unsafe impl Sync for DevicePool {}

static FORCE_FALLBACK: AtomicBool = AtomicBool::new(false);

/// Testing builds: force the S2 fallback mode for ordinals opened later.
#[cfg(feature = "testing")]
pub(crate) fn force_fallback() {
    FORCE_FALLBACK.store(true, Ordering::Relaxed);
}

impl DevicePool {
    /// The ordinal's pool, or None (the fallback mode: today's cudarc
    /// allocation on the current pool, with no new error) when it cannot be
    /// created or configured. A configured pool maps its Q1 bound at once
    /// ([`DevicePool::reserve`]) on `anchor`, the device's otherwise empty
    /// anchor stream. The caller bound the ordinal's context.
    pub(crate) fn create(ordinal: usize, anchor: sys::CUstream) -> Option<Self> {
        if FORCE_FALLBACK.load(Ordering::Relaxed) {
            return None;
        }
        let id = i32::try_from(ordinal).ok()?;
        // SAFETY: an all-zero CUmemPoolProps is the documented default (no
        // handle types, no maximum size, reserved bytes zero).
        let mut props: sys::CUmemPoolProps = unsafe { std::mem::zeroed() };
        props.allocType = sys::CUmemAllocationType::CU_MEM_ALLOCATION_TYPE_PINNED;
        props.handleTypes = sys::CUmemAllocationHandleType::CU_MEM_HANDLE_TYPE_NONE;
        props.location = sys::CUmemLocation {
            type_: sys::CUmemLocationType::CU_MEM_LOCATION_TYPE_DEVICE,
            id,
        };
        let mut pool: sys::CUmemoryPool = std::ptr::null_mut();
        // SAFETY: props is fully initialised; the driver writes one handle.
        unsafe { sys::cuMemPoolCreate(&mut pool, &props) }
            .result()
            .ok()?;
        let mut threshold = RELEASE_THRESHOLD;
        let mut internal: i32 = 0;
        // SAFETY: each attribute takes the pointer to its documented scalar type
        // (cuuint64_t for the threshold, int for the reuse flag).
        let configured = unsafe {
            sys::cuMemPoolSetAttribute(
                pool,
                sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_RELEASE_THRESHOLD,
                (&mut threshold as *mut u64).cast(),
            )
        }
        .result()
        .and_then(|()| {
            unsafe {
                sys::cuMemPoolSetAttribute(
                    pool,
                    sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_REUSE_ALLOW_INTERNAL_DEPENDENCIES,
                    (&mut internal as *mut i32).cast(),
                )
            }
            .result()
        });
        if configured.is_err() {
            // SAFETY: the pool has no allocations yet.
            let _ = unsafe { sys::cuMemPoolDestroy(pool) };
            return None;
        }
        let pool = Self {
            pool,
            scopes: Mutex::new(0),
        };
        pool.reserve(anchor);
        Some(pool)
    }

    /// Map [`RELEASE_THRESHOLD`] bytes once, at device open (amendment
    /// Q1a of Q1, D134, lane `coldgpu`, 2026-11-11): one allocation of the
    /// bound, freed on the same stream and waited for, so that the pool holds
    /// the bound from the start and the threshold keeps it. Growing the pool
    /// later, after the device has run work, stalled for 0.04-0.8 s on the
    /// development machine (WSL2), while this growth costs about a
    /// millisecond. Every step may fail without effect: the pool then grows on
    /// demand as before. The testing refusal hook does not count it.
    fn reserve(&self, anchor: sys::CUstream) {
        let mut address: sys::CUdeviceptr = 0;
        // SAFETY: the caller bound the pool's context; the driver writes one
        // address, valid in stream order on `anchor`.
        let allocated = unsafe {
            sys::cuMemAllocFromPoolAsync(
                &mut address,
                RELEASE_THRESHOLD as usize,
                self.pool,
                anchor,
            )
        }
        .result();
        if allocated.is_err() {
            return;
        }
        // SAFETY: the allocation above, never read or written, freed on the
        // stream it was made on. The wait makes the free complete before any
        // other stream allocates, so every later allocation may reuse it.
        if unsafe { sys::cuMemFreeAsync(address, anchor) }
            .result()
            .is_ok()
        {
            let _ = unsafe { sys::cuStreamSynchronize(anchor) };
        }
    }

    /// `CU_MEMPOOL_ATTR_RELEASE_THRESHOLD` := `bytes`.
    fn set_threshold(&self, bytes: u64) -> Result<(), DriverError> {
        let mut threshold = bytes;
        // SAFETY: the attribute takes a pointer to a cuuint64_t.
        unsafe {
            sys::cuMemPoolSetAttribute(
                self.pool,
                sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_RELEASE_THRESHOLD,
                (&mut threshold as *mut u64).cast(),
            )
        }
        .result()
    }

    /// Open a call scope (cuda-backend.md, "First calls at a new size", lane
    /// `coldgpu`, 2026-11-11). The first open scope of the ordinal lifts the
    /// release threshold, so the call's own synchronizations release nothing
    /// and its chunks reuse what the pool has mapped instead of unmapping and
    /// regrowing it. A refused write leaves Q1's threshold in force: the
    /// scope only changes speed, never a result or an error.
    pub(crate) fn enter(&self) {
        let mut open = self.scopes.lock().unwrap_or_else(|p| p.into_inner());
        if *open == 0 {
            let _ = self.set_threshold(u64::MAX);
        }
        *open += 1;
    }

    /// Close a call scope. The last scope to close restores
    /// [`RELEASE_THRESHOLD`] and then, with no lock held, trims the pool to it:
    /// between calls the pool again keeps at most the Q1 bound of unused
    /// memory. A trim that races a new scope only returns memory early.
    pub(crate) fn leave(&self) {
        let last = {
            let mut open = self.scopes.lock().unwrap_or_else(|p| p.into_inner());
            *open = open.saturating_sub(1);
            if *open == 0 {
                let _ = self.set_threshold(RELEASE_THRESHOLD);
            }
            *open == 0
        };
        if last {
            // SAFETY: a live pool handle; the trim releases unused memory only.
            let _ = unsafe { sys::cuMemPoolTrimTo(self.pool, RELEASE_THRESHOLD as usize) };
        }
    }

    /// `cuMemAllocFromPoolAsync` of `bytes` on `stream` (the caller's owned
    /// stream), honouring the testing refusal hook.
    pub(crate) fn alloc_async(
        &self,
        bytes: usize,
        stream: sys::CUstream,
    ) -> Result<sys::CUdeviceptr, DriverError> {
        if refused() {
            return Err(sys::CUresult::CUDA_ERROR_OUT_OF_MEMORY
                .result()
                .unwrap_err());
        }
        let mut address: sys::CUdeviceptr = 0;
        // SAFETY: the caller bound the pool's context; the driver writes one
        // address that is valid in stream order on `stream`.
        unsafe { sys::cuMemAllocFromPoolAsync(&mut address, bytes, self.pool, stream) }.result()?;
        Ok(address)
    }

    /// `cuMemPoolTrimTo(pool, 0)`: release every unused byte.
    pub(crate) fn trim(&self) -> Result<(), DriverError> {
        // SAFETY: a live pool handle.
        unsafe { sys::cuMemPoolTrimTo(self.pool, 0) }.result()
    }

    /// One `cuuint64_t` pool attribute (testing observations only).
    #[cfg(feature = "testing")]
    pub(crate) fn attribute(
        &self,
        attribute: sys::CUmemPool_attribute,
    ) -> Result<u64, DriverError> {
        let mut value = 0u64;
        // SAFETY: every attribute read here is a cuuint64_t.
        unsafe {
            sys::cuMemPoolGetAttribute(self.pool, attribute, (&mut value as *mut u64).cast())
        }
        .result()?;
        Ok(value)
    }
}

// ---- Testing refusal hook (`cuda_pool_fail`, PC6 testing hooks) ----

#[cfg(feature = "testing")]
struct Fail {
    skip: u64,
    count: u64,
}
#[cfg(feature = "testing")]
static FAIL: Mutex<Option<Fail>> = Mutex::new(None);
#[cfg(feature = "testing")]
static FIRED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Whether the testing hook refuses this pool allocation (always false in
/// builds without `testing`). The trim-and-retry allocation counts as one.
fn refused() -> bool {
    #[cfg(feature = "testing")]
    {
        let mut state = FAIL.lock().unwrap_or_else(|p| p.into_inner());
        if let Some(fail) = state.as_mut() {
            if fail.skip > 0 {
                fail.skip -= 1;
                return false;
            }
            if fail.count > 0 {
                fail.count -= 1;
                if fail.count == 0 {
                    *state = None;
                }
                FIRED.fetch_add(1, Ordering::Relaxed);
                return true;
            }
            *state = None;
        }
    }
    false
}

/// Testing builds: after `skip` successful pool allocations, refuse the next
/// `count` with `CUDA_ERROR_OUT_OF_MEMORY` without calling the driver; None or a
/// zero count disarms.
#[cfg(feature = "testing")]
pub fn set_pool_fail(count: Option<u64>, skip: u64) {
    *FAIL.lock().unwrap_or_else(|p| p.into_inner()) = count
        .filter(|&count| count > 0)
        .map(|count| Fail { skip, count });
}

/// Testing builds: pool allocations refused by [`set_pool_fail`] so far.
#[cfg(feature = "testing")]
pub fn pool_fails_fired() -> u64 {
    FIRED.load(Ordering::Relaxed)
}

/// S8: the largest batched host export that goes through an ordinal's pinned
/// bounce buffer; the buffer has exactly this size.
pub(crate) const BOUNCE_BYTES: usize = 4 * 1024 * 1024;

/// A page-locked host buffer of `BOUNCE_BYTES` bytes, never freed during the
/// process (S8).
pub(crate) struct Pinned(*mut u8);
// SAFETY: a driver-pinned host allocation; one thread holds it at a time
// (checked out of the slot under its lock), and nothing frees it.
unsafe impl Send for Pinned {}
impl Pinned {
    pub(crate) fn as_ptr(&self) -> *mut u8 {
        self.0
    }
    /// The caller has bound the ordinal's context.
    fn allocate() -> Result<Self, DriverError> {
        let mut pointer = std::ptr::null_mut();
        // SAFETY: the driver writes one host pointer of BOUNCE_BYTES bytes.
        unsafe { sys::cuMemHostAlloc(&mut pointer, BOUNCE_BYTES, 0) }.result()?;
        Ok(Self(pointer.cast()))
    }
}

#[derive(Default)]
struct BounceState {
    buffer: Option<Pinned>,
    out: bool,
    failures: u8,
}

/// An ordinal's bounce slot for batched host exports (S8). Under the lock an
/// export only checks the buffer out (`Option::take`) or puts it back; the
/// copies, the wait and the copy-out run with no lock held. An export that
/// finds the buffer checked out takes the per-column path, so exports never
/// wait for each other. After a failed wait the buffer is leaked, never
/// returned (a copy may still target it, I8); a later export allocates one
/// replacement lazily. After a second failure (a failed wait or a failed
/// pinned allocation) the ordinal stays on the per-column path.
pub(crate) struct BounceSlot(Mutex<BounceState>);

impl BounceSlot {
    pub(crate) fn new() -> Self {
        Self(Mutex::new(BounceState::default()))
    }
    fn state(&self) -> std::sync::MutexGuard<'_, BounceState> {
        self.0.lock().unwrap_or_else(|p| p.into_inner())
    }
    /// Check the buffer out, allocating it on first use; None sends the export
    /// to the per-column path. The caller has bound the ordinal's context.
    pub(crate) fn checkout(&self) -> Option<Pinned> {
        let held = {
            let mut state = self.state();
            if state.out || state.failures >= 2 {
                return None;
            }
            state.out = true;
            state.buffer.take()
        };
        if held.is_some() {
            return held;
        }
        // The allocation runs with no lock held; the slot stays checked out.
        match Pinned::allocate() {
            Ok(buffer) => Some(buffer),
            Err(_) => {
                let mut state = self.state();
                state.out = false;
                state.failures = state.failures.saturating_add(1);
                None
            }
        }
    }
    /// Return the buffer after a successful wait and copy-out.
    pub(crate) fn give_back(&self, buffer: Pinned) {
        let mut state = self.state();
        state.buffer = Some(buffer);
        state.out = false;
    }
    /// After a failed wait: the buffer is leaked, never returned (I8).
    pub(crate) fn quarantine(&self, _buffer: Pinned) {
        // `Pinned` has no destructor: letting it go leaks the allocation.
        let mut state = self.state();
        state.out = false;
        state.failures = state.failures.saturating_add(1);
    }
}
