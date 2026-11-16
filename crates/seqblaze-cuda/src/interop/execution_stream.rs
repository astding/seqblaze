//! Borrowed execution handles. Only `allocation` is owned: this module never
//! destroys a caller stream. Driver operations, including the special legacy
//! and PTDS tokens, run with the retained context bound on the calling thread.
//!
//! cudarc 0.19.10's public CudaFunction::cu_function and pointer traits are used
//! as ABI accessors (cudarc, Chelsea Lowman, MIT OR Apache-2.0). The adapter and
//! lifetime/dependency policy here are seqblaze's; no non-owning CudaStream
//! constructor is assumed.
use std::{
    ffi::c_void,
    marker::PhantomData,
    sync::Arc,
    thread::ThreadId,
    time::{Duration, Instant},
};

use cudarc::driver::{
    sys, CudaFunction, CudaSlice, CudaStream, DevicePtr, DevicePtrMut, DeviceRepr, DriverError,
    LaunchConfig, PushKernelArg,
};
use seqblaze_core::validate::EngineError;

use super::cuda::{external_stream, order_streams};
use crate::{
    device::Device,
    session::{check_stream, failure_ordinal},
};

#[cfg(feature = "testing")]
thread_local! {
    static TRACE: std::cell::RefCell<[Vec<u64>; 8]> = const {
        std::cell::RefCell::new([
            Vec::new(), Vec::new(), Vec::new(), Vec::new(),
            Vec::new(), Vec::new(), Vec::new(), Vec::new(),
        ])
    };
}
// Testing builds: the first refused reservation of a trace entry since the
// last reset (F4 round 3). The trace grows with the call's operations, so its
// growth goes through the §3 helper; `record` cannot return an EngineError to
// its DriverError callers, so the refusal is kept here and the resident runner
// raises it before it publishes (`resident_launch_streams_check`).
#[cfg(feature = "testing")]
thread_local! {
    static TRACE_REFUSAL: std::cell::RefCell<Option<EngineError>> =
        const { std::cell::RefCell::new(None) };
}
#[cfg(feature = "testing")]
const TRACE_LABEL: &str = "execution stream records";
#[cfg(feature = "testing")]
pub fn resident_launch_streams() -> [Vec<u64>; 8] {
    TRACE.with(|trace| trace.borrow().clone())
}
#[cfg(feature = "testing")]
pub fn resident_launch_streams_reset() {
    TRACE.with(|trace| *trace.borrow_mut() = std::array::from_fn(|_| Vec::new()));
    TRACE_REFUSAL.with(|slot| *slot.borrow_mut() = None);
}
/// The refused trace reservation since the last reset, as its §3 DeviceError
/// (testing builds; always `Ok` otherwise). Taking it clears it.
pub fn resident_launch_streams_check() -> Result<(), EngineError> {
    #[cfg(feature = "testing")]
    return TRACE_REFUSAL.with(|slot| slot.borrow_mut().take().map_or(Ok(()), Err));
    #[cfg(not(feature = "testing"))]
    Ok(())
}
fn record(category: usize, handle: u64) {
    #[cfg(feature = "testing")]
    TRACE.with(|trace| {
        use seqblaze_core::detail::try_reserve;
        let mut trace = trace.borrow_mut();
        // Detail observations also belong to the existing generic categories.
        let generic = (category >= 4).then_some(if category == 7 { 1 } else { 0 });
        // Both entries are reserved before either is written.
        let reserved = try_reserve(&mut trace[category], 1, TRACE_LABEL).and_then(|()| {
            generic.map_or(Ok(()), |generic| {
                try_reserve(&mut trace[generic], 1, TRACE_LABEL)
            })
        });
        match reserved {
            Ok(()) => {
                trace[category].push(handle);
                if let Some(generic) = generic {
                    trace[generic].push(handle);
                }
            }
            Err(error) => TRACE_REFUSAL.with(|slot| {
                slot.borrow_mut().get_or_insert(error);
            }),
        }
    });
    #[cfg(not(feature = "testing"))]
    let _ = (category, handle);
}

pub(crate) enum DetailOperation {
    Plan = 4,
    Continue = 5,
    Progress = 6,
}

// If both the handoff and the borrowed-stream fallback synchronization fail,
// the owned stream alone cannot prove completion. Keep its context alive and
// make every associated retirement guard retain its storage too. Entries are
// never removed/reused; this is an exceptional failed-context path only.
static FAILED_HANDOFFS: std::sync::Mutex<Vec<Arc<CudaStream>>> = std::sync::Mutex::new(Vec::new());
fn handoff_failed(stream: &CudaStream) -> bool {
    FAILED_HANDOFFS
        .lock()
        .unwrap_or_else(|p| p.into_inner())
        .iter()
        .any(|failed| std::ptr::eq(failed.as_ref(), stream))
}

// ---- Host waits (cuda-backend.md, per-call cost S3; Q7) ----
//
// Every host wait of the resident driver runs one loop: query the stream;
// while it reports CUDA_ERROR_NOT_READY, yield the thread for the first
// millisecond of the wait, then sleep clamp(elapsed / 16, 50 µs, 500 µs). The
// loop ends when the query reports idle or any error; the caller's blocking
// cuStreamSynchronize or copy then runs and reports the error, so a wait is
// never cut short. Inside a cancel scope the loop runs through
// `cancel::wait_while` (polls about every 100 ms, unchanged); outside a scope,
// after the flag is set and in Drop paths it runs without polls, so no Python
// runs in a destructor. Stream-scoped only: no context-wide synchronization,
// no added copy, launch or event, no helper thread.
const YIELD_PHASE: Duration = Duration::from_millis(1);
const SLEEP_MIN: Duration = Duration::from_micros(50);
const SLEEP_MAX: Duration = Duration::from_micros(500);

struct StreamWait {
    stream: sys::CUstream,
    start: Instant,
    idle: bool,
    first_busy: Option<bool>,
}
impl StreamWait {
    fn new(stream: sys::CUstream) -> Self {
        Self {
            stream,
            start: Instant::now(),
            idle: false,
            first_busy: None,
        }
    }
    /// One query; while the stream is busy, the yield or sleep, then true.
    fn busy(&mut self) -> bool {
        if self.idle {
            return false;
        }
        // SAFETY: a live stream handle of this call; the special legacy and
        // PTDS handles are queried with the context bound on their thread.
        let state = unsafe { sys::cuStreamQuery(self.stream) };
        let busy = state == sys::CUresult::CUDA_ERROR_NOT_READY;
        self.first_busy.get_or_insert(busy);
        if !busy {
            self.idle = true;
            return false;
        }
        let elapsed = self.start.elapsed();
        if elapsed < YIELD_PHASE {
            std::thread::yield_now();
        } else {
            std::thread::sleep((elapsed / 16).clamp(SLEEP_MIN, SLEEP_MAX));
        }
        true
    }
    /// The rest of the wait without polls.
    fn finish(mut self) -> bool {
        while self.busy() {}
        self.first_busy == Some(true)
    }
}

/// Testing builds: Drop-path waits made, and those whose first stream query
/// reported CUDA_ERROR_NOT_READY (`cuda_wait_stats`).
#[cfg(feature = "testing")]
static DROP_WAITS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
#[cfg(feature = "testing")]
static DROP_BUSY: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// `cuda_wait_stats()` values (testing builds).
#[cfg(feature = "testing")]
#[derive(Clone, Copy, Debug, Default)]
pub struct WaitStats {
    pub drop_waits: u64,
    pub drop_busy: u64,
}
#[cfg(feature = "testing")]
pub fn wait_stats() -> WaitStats {
    use std::sync::atomic::Ordering::Relaxed;
    WaitStats {
        drop_waits: DROP_WAITS.load(Relaxed),
        drop_busy: DROP_BUSY.load(Relaxed),
    }
}
#[cfg(feature = "testing")]
pub fn wait_stats_reset() {
    use std::sync::atomic::Ordering::Relaxed;
    DROP_WAITS.store(0, Relaxed);
    DROP_BUSY.store(0, Relaxed);
}

/// The owned stream's completion wait without polls, then its blocking
/// synchronize. With the result: whether a wait was made, and whether its
/// first query found the stream busy.
fn retire_quietly(stream: &CudaStream) -> (Option<bool>, Result<(), DriverError>) {
    if handoff_failed(stream) {
        return (None, sys::CUresult::CUDA_ERROR_UNKNOWN.result());
    }
    if let Err(error) = stream.context().bind_to_thread() {
        return (None, Err(error));
    }
    let busy = StreamWait::new(stream.cu_stream()).finish();
    // SAFETY: a live owned stream of the bound context.
    let result = unsafe { sys::cuStreamSynchronize(stream.cu_stream()) }.result();
    (Some(busy), result)
}

/// A plain host wait for the owned stream (no polls): exports and other waits
/// that run outside the call's scope.
pub(crate) fn retire_owned_stream(stream: &CudaStream) -> Result<(), DriverError> {
    retire_quietly(stream).1
}

/// The host-proved retirement of a Drop path: the same wait without polls, so
/// no Python runs in a destructor; counted for `cuda_wait_stats`.
pub(crate) fn retire_in_drop(stream: &CudaStream) -> Result<(), DriverError> {
    let (waited, result) = retire_quietly(stream);
    #[cfg(feature = "testing")]
    if let Some(busy) = waited {
        use std::sync::atomic::Ordering::Relaxed;
        DROP_WAITS.fetch_add(1, Relaxed);
        if busy {
            DROP_BUSY.fetch_add(1, Relaxed);
        }
    }
    #[cfg(not(feature = "testing"))]
    let _ = waited;
    result
}

/// A host wait of a call's driver (cancellation-progress.md §3.5): on the
/// calling thread of a cancel scope, query `stream` and poll whenever a poll
/// is due; then, and always outside a scope or after the flag, the same loop
/// without polls until the stream is idle. Stream-scoped only; no
/// context-wide sync. The caller's blocking copy or synchronize follows, so
/// the wait is never cut short and errors are reported there.
fn poll_while_busy(stream: sys::CUstream) {
    let mut wait = StreamWait::new(stream);
    seqblaze_core::cancel::wait_while(|| wait.busy());
    wait.finish();
}

/// The polling part of a driver's explicit host wait on the owned stream,
/// without its blocking synchronize: inside a cancel scope it polls whenever a
/// poll is due, then waits out the rest without polls, and it returns once the
/// stream is idle or reports any error (T7). A driver calls it on its success
/// path before scratch guards drop, so that their Drop-path wait, which never
/// polls, finds the stream idle. The guards still make their own blocking
/// synchronize and handle its error, so no error, message or raise point moves.
pub(crate) fn idle_owned_stream(stream: &CudaStream) {
    if handoff_failed(stream) || stream.context().bind_to_thread().is_err() {
        return;
    }
    poll_while_busy(stream.cu_stream());
}

/// The owned stream's completion wait for a driver's explicit host wait, with
/// the polls of [`poll_while_busy`]. Destructors use [`retire_in_drop`].
pub(crate) fn wait_owned_stream(stream: &CudaStream) -> Result<(), DriverError> {
    if handoff_failed(stream) {
        return sys::CUresult::CUDA_ERROR_UNKNOWN.result();
    }
    stream.context().bind_to_thread()?;
    poll_while_busy(stream.cu_stream());
    // SAFETY: a live owned stream of the bound context.
    unsafe { sys::cuStreamSynchronize(stream.cu_stream()) }.result()
}

// ---- Pending host staging (cuda-backend.md, per-call cost S5 (b)) ----
//
// A host-to-device copy no longer waits for its own completion. Its staging
// buffer moves into the lease's pending list and is released only after the
// next successful host wait that proves the copy complete: a control read
// (its synchronize of the borrowed stream the copy was enqueued on), an
// explicit synchronize of the owned stream (every enqueue publishes its
// completion dependency there), the final wait, or the Drop-path wait. Once
// the list holds more than `PENDING_LIMIT` bytes, the upload waits at once.
// After a failed wait the list is leaked, never freed (I8): a copy may still
// read it.

/// The most host staging a lease keeps pending between proving waits.
const PENDING_LIMIT: usize = 64 * 1024 * 1024;

/// One staging buffer, type-erased: its allocation and how to free it.
struct HeldStaging {
    pointer: *mut u8,
    capacity: usize,
    capacity_bytes: usize,
    release: unsafe fn(*mut u8, usize),
}
// SAFETY: owned transport allocations or a Send + Sync retained table. Only
// the pending list frees each entry, once, after a proving wait.
unsafe impl Send for HeldStaging {}

/// # Safety
/// `pointer` and `capacity` come from a `Vec<MaybeUninit<T>>` of this `T`.
unsafe fn release_staging<T>(pointer: *mut u8, capacity: usize) {
    drop(unsafe { Vec::<std::mem::MaybeUninit<T>>::from_raw_parts(pointer.cast(), 0, capacity) });
}

impl HeldStaging {
    fn new<T>(staging: Vec<std::mem::MaybeUninit<T>>) -> Self {
        let mut staging = std::mem::ManuallyDrop::new(staging);
        Self {
            pointer: staging.as_mut_ptr().cast(),
            capacity: staging.capacity(),
            capacity_bytes: staging.capacity() * std::mem::size_of::<T>(),
            release: release_staging::<T>,
        }
    }
    fn owner<T: Send + Sync>(owner: Arc<T>) -> Self {
        unsafe fn release<T>(pointer: *mut u8, _: usize) {
            // SAFETY: this entry owns the one Arc reference from into_raw.
            drop(unsafe { Arc::from_raw(pointer.cast::<T>()) });
        }
        Self {
            pointer: Arc::into_raw(owner).cast_mut().cast(),
            capacity: 0,
            capacity_bytes: 0,
            release: release::<T>,
        }
    }
    /// Free after a proving wait. Dropping a `HeldStaging` without this leaks.
    fn free(self) {
        // SAFETY: constructed with its matching vector/Arc release routine.
        unsafe { (self.release)(self.pointer, self.capacity) }
    }
}

#[derive(Default)]
struct PendingStaging {
    held: Vec<HeldStaging>,
    bytes: usize,
}

/// A token and a retained context, with no stream-destruction authority.
/// A raw handle's external owner must outlive every active call/iterator.
pub struct BorrowedExecutionStream {
    handle: u64,
    device: Device,
    allocation: Arc<CudaStream>,
    thread: ThreadId,
    /// Host staging of enqueued uploads awaiting a proving wait (S5 (b)).
    pending: std::sync::Mutex<PendingStaging>,
    arena: std::sync::Mutex<crate::arena::Arena>,
    packs: std::sync::Mutex<Vec<Arc<dyn crate::pack_cache::Candidate>>>,
    /// The call scope on the ordinal's owned pool; declared last, so it closes
    /// after the arena's and the packs' storage has retired.
    _scope: PoolScope,
}

/// An open call scope of the ordinal's owned pool ([`crate::pool::DevicePool::enter`]),
/// held for the life of one lease: a pair or unary call, a Tiles iterator, or a
/// device-side helper operation (device tables, digests, index normalization,
/// selections, DLPack copy exports). Nothing in the S2 fallback mode.
struct PoolScope(Device);
impl PoolScope {
    fn open(device: &Device) -> Self {
        if let Some(pool) = device.0.pool.as_ref() {
            pool.enter();
        }
        Self(device.clone())
    }
}
impl Drop for PoolScope {
    fn drop(&mut self) {
        if let Some(pool) = self.0 .0.pool.as_ref() {
            pool.leave();
        }
    }
}
impl BorrowedExecutionStream {
    pub(crate) fn new(
        device: &Device,
        allocation: Arc<CudaStream>,
        handle: u64,
    ) -> Result<Self, EngineError> {
        let handle = handle.max(1);
        external_stream(device, handle)?;
        Ok(Self {
            handle,
            device: device.clone(),
            allocation,
            thread: std::thread::current().id(),
            pending: std::sync::Mutex::new(PendingStaging::default()),
            arena: std::sync::Mutex::new(crate::arena::Arena::default()),
            packs: std::sync::Mutex::new(Vec::new()),
            _scope: PoolScope::open(device),
        })
    }
    pub(crate) fn allocation(&self) -> &Arc<CudaStream> {
        &self.allocation
    }
    pub(crate) fn device(&self) -> &Device {
        &self.device
    }
    pub fn disable_arena(&self) {
        crate::arena::Arena::disable(&self.arena);
    }
    pub(crate) unsafe fn scratch<T: DeviceRepr>(
        &self,
        len: usize,
    ) -> Result<crate::arena::Scratch<T>, DriverError> {
        unsafe { crate::arena::Arena::alloc(&self.arena, self, len) }
    }
    pub(crate) fn offer_pack(&self, pack: Arc<dyn crate::pack_cache::Candidate>) {
        let mut packs = self.packs.lock().unwrap_or_else(|p| p.into_inner());
        if packs.try_reserve(1).is_ok() {
            packs.push(pack);
        }
    }
    /// Called only after the final successful completion wait and cancellation
    /// and trace checks, before publishing the call/step. No intermediate wait
    /// or Drop path may admit a pack to the cache.
    pub fn publish_packs(&self) {
        let packs = std::mem::take(&mut *self.packs.lock().unwrap_or_else(|p| p.into_inner()));
        for pack in packs {
            pack.insert(&self.device.0.pack_cache);
        }
    }
    fn pending(&self) -> std::sync::MutexGuard<'_, PendingStaging> {
        self.pending.lock().unwrap_or_else(|p| p.into_inner())
    }
    /// Whether host staging still awaits a proving wait (S5 (b)).
    pub(crate) fn has_pending(&self) -> bool {
        !self.pending().held.is_empty()
    }
    /// The pending entries a wait that starts now can prove (S5 (b)).
    pub(crate) fn pending_mark(&self) -> usize {
        self.pending().held.len()
    }
    /// Settle the pending host staging after a host wait that started at
    /// `mark`: free the entries it proved complete, or leak every entry after
    /// a failed wait (I8). Entries held after the wait started stay pending.
    pub(crate) fn settle_from(&self, mark: usize, proved: bool) {
        let mut pending = self.pending();
        let proved_entries = if proved {
            mark.min(pending.held.len())
        } else {
            pending.held.len()
        };
        let PendingStaging { held, bytes } = &mut *pending;
        for entry in held.drain(..proved_entries) {
            *bytes = bytes.saturating_sub(entry.capacity_bytes);
            if proved {
                entry.free();
            }
            // Otherwise dropping `entry` leaks its allocation on purpose.
        }
    }
    /// [`Self::settle_from`] for a wait that started after every hold so far.
    pub(crate) fn settle(&self, proved: bool) {
        self.settle_from(usize::MAX, proved);
    }
    /// Keep `staging` until a proving wait. With the result: whether the list
    /// now holds more than `PENDING_LIMIT` bytes. A refused list slot hands the
    /// staging back, and the caller waits for the copy as before (no new error).
    fn hold(&self, held: HeldStaging) -> Result<bool, HeldStaging> {
        let mut pending = self.pending();
        if pending.held.try_reserve(1).is_err() {
            return Err(held);
        }
        pending.bytes = pending.bytes.saturating_add(held.capacity_bytes);
        pending.held.push(held);
        Ok(pending.bytes > PENDING_LIMIT)
    }
    /// A host wait for the borrowed stream after an enqueue: polls while busy,
    /// then its blocking synchronize; settles the pending staging.
    fn wait_borrowed(&self) -> Result<(), DriverError> {
        let mark = self.pending_mark();
        poll_while_busy(self.raw());
        // SAFETY: the bound, prepared borrowed stream of this call.
        let completed = unsafe { sys::cuStreamSynchronize(self.raw()) }.result();
        self.settle_from(mark, completed.is_ok());
        completed
    }
    pub fn validate(device: &Device, handle: u64) -> Result<(), EngineError> {
        crate::device::guard(|| external_stream(device, handle.max(1)).map(|_| ()))
    }
    pub fn check_thread(&self) -> Result<(), EngineError> {
        if self.handle == 2 && self.thread != std::thread::current().id() {
            return Err(EngineError::Device(
                "PTDS Tiles must be advanced on their creating thread".into(),
            ));
        }
        Ok(())
    }
    pub fn handle(&self) -> u64 {
        self.handle
    }
    fn raw(&self) -> sys::CUstream {
        self.handle as usize as sys::CUstream
    }
    fn prepare(&self) -> Result<(), DriverError> {
        self.device.0.ctx.bind_to_thread()?;
        if self.handle == 2 && self.thread != std::thread::current().id() {
            return sys::CUresult::CUDA_ERROR_INVALID_HANDLE.result();
        }
        Ok(())
    }
    fn order(&self, from: sys::CUstream, to: sys::CUstream) -> Result<(), DriverError> {
        // Preserve DriverError for the existing operation-specific diagnostics.
        if from == to {
            return Ok(());
        }
        // Both streams belong to the bound retained context; the event comes
        // from the ordinal's free list and returns to it after the wait is
        // enqueued (S1 (c)).
        self.device
            .0
            .events
            .order(from, to)
            .map_err(crate::pool::Created::error)
    }
    fn complete(&self, operation: Result<(), DriverError>) -> Result<(), DriverError> {
        // Every enqueue publishes a completion dependency to the owned stream.
        // Scratch destructors, delivered columns and foreign-owner retirement
        // can therefore wait without consulting the borrowed handle, even if a
        // Tiles iterator is closed from another thread (PTDS included).
        let ordered = self.order(self.raw(), self.allocation.cu_stream());
        if ordered.is_err() {
            // The borrow is still valid here. If even this fallback fails, no
            // guard may free storage using an unproven owned-stream completion.
            if let Err(error) = unsafe { sys::cuStreamSynchronize(self.raw()) }.result() {
                FAILED_HANDOFFS
                    .lock()
                    .unwrap_or_else(|p| p.into_inner())
                    .push(Arc::clone(&self.allocation));
                return Err(error);
            }
        }
        operation.and(ordered)
    }
    pub fn synchronize(&self) -> Result<(), DriverError> {
        // All submitted work already has a completion dependency on this owned
        // stream. In particular this never reinterprets PTDS on another thread.
        let mark = self.pending_mark();
        let completed = retire_owned_stream(&self.allocation);
        self.settle_from(mark, completed.is_ok());
        completed
    }
    pub(crate) fn wait_for(&self, producer: Option<u64>) -> Result<(), EngineError> {
        if let Some(handle) = producer {
            let producer = external_stream(&self.device, handle)?;
            order_streams(&self.device, producer, self.raw())?;
            self.complete(Ok(()))
                .map_err(|e| failure_ordinal(&self.device, "retaining producer dependency", e))?;
        }
        Ok(())
    }
    pub(crate) fn launch_builder<'a>(&'a self, function: &'a CudaFunction) -> ExecutionLaunch<'a> {
        self.builder(function, 0)
    }
    pub(crate) fn detail_builder<'a>(
        &'a self,
        function: &'a CudaFunction,
        operation: DetailOperation,
    ) -> ExecutionLaunch<'a> {
        self.builder(function, operation as usize)
    }
    pub(crate) fn validation_builder<'a>(
        &'a self,
        function: &'a CudaFunction,
    ) -> ExecutionLaunch<'a> {
        self.builder(function, 2)
    }
    pub(crate) fn digest_builder<'a>(&'a self, function: &'a CudaFunction) -> ExecutionLaunch<'a> {
        self.builder(function, 3)
    }
    pub(super) fn builder<'a>(
        &'a self,
        function: &'a CudaFunction,
        category: usize,
    ) -> ExecutionLaunch<'a> {
        ExecutionLaunch {
            stream: self,
            function,
            args: Vec::new(),
            category,
            lifetime: PhantomData,
        }
    }
    pub(crate) fn memcpy_htod<T: DeviceRepr, D: DevicePtrMut<T>>(
        &self,
        host: &[T],
        target: &mut D,
    ) -> Result<(), DriverError> {
        assert!(host.len() <= target.len());
        self.prepare()?;
        let staging = host_staging::<T>(host.len())?;
        self.htod_through(host, target, staging)
    }
    /// The host-origin upload with a staging copy that the caller reserved
    /// (`upload_staging`), so a refused staging reservation is a labelled
    /// DeviceError before any copy is enqueued (host-allocation-failures §12.6
    /// rule 4). Driver failures keep their DriverError mapping.
    pub(crate) fn memcpy_htod_staged<T: DeviceRepr, D: DevicePtrMut<T>>(
        &self,
        host: &[T],
        target: &mut D,
        staging: Vec<std::mem::MaybeUninit<T>>,
    ) -> Result<(), DriverError> {
        assert!(host.len() <= target.len());
        self.prepare()?;
        self.htod_through(host, target, staging)
    }
    /// The copy itself, through `staging` (exactly `host.len()` elements), after
    /// `prepare`. It waits for earlier stream work before the copy (a pageable
    /// copy can block), but not for the copy itself: the staging stays in the
    /// pending list until a proving wait (S5 (b)).
    fn htod_through<T: DeviceRepr, D: DevicePtrMut<T>>(
        &self,
        host: &[T],
        target: &mut D,
        mut staging: Vec<std::mem::MaybeUninit<T>>,
    ) -> Result<(), DriverError> {
        assert!(staging.len() == host.len());
        // SAFETY: MaybeUninit holds a byte-for-byte transport representation,
        // including any padding; no cloned T is read or dropped on the host.
        unsafe {
            std::ptr::copy_nonoverlapping(
                host.as_ptr(),
                staging.as_mut_ptr().cast::<T>(),
                host.len(),
            );
        }
        self.htod_retained(
            staging.as_ptr().cast(),
            std::mem::size_of_val(host),
            target,
            HeldStaging::new(staging),
        )
    }
    /// S10(b): transfer a private transport vector, with no staging copy.
    pub(crate) fn memcpy_htod_owned<T: DeviceRepr + Copy, D: DevicePtrMut<u8>>(
        &self,
        host: Vec<T>,
        target: &mut D,
    ) -> Result<(), DriverError> {
        let bytes = std::mem::size_of_val(host.as_slice());
        assert!(bytes <= target.len());
        self.prepare()?;
        let mut host = std::mem::ManuallyDrop::new(host);
        // SAFETY: MaybeUninit<T> has exactly T's layout and allocator; the
        // private vector's ownership moves to the pending list without a copy.
        let staging = unsafe {
            Vec::from_raw_parts(
                host.as_mut_ptr().cast::<std::mem::MaybeUninit<T>>(),
                host.len(),
                host.capacity(),
            )
        };
        self.htod_retained(
            staging.as_ptr().cast(),
            bytes,
            target,
            HeldStaging::new(staging),
        )
    }
    /// S10(a), S13c: DMA from the table's retained storage. `address..bytes`
    /// must be a span covered by `owner` under its no-mutation call contract.
    pub(crate) unsafe fn memcpy_htod_table<D: DevicePtrMut<u8>>(
        &self,
        address: usize,
        bytes: usize,
        owner: Arc<seqblaze_core::seq::SeqTable>,
        target: &mut D,
    ) -> Result<(), DriverError> {
        assert!(bytes <= target.len());
        self.prepare()?;
        self.htod_retained(
            address as *const c_void,
            bytes,
            target,
            HeldStaging::owner(owner),
        )
    }
    /// Enqueue with its host owner. Every proving wait releases the entries
    /// present before it; a failed wait leaks them, including table owners.
    fn htod_retained<T: DeviceRepr, D: DevicePtrMut<T>>(
        &self,
        source: *const c_void,
        bytes: usize,
        target: &mut D,
        held: HeldStaging,
    ) -> Result<(), DriverError> {
        check_stream(target.stream(), &self.allocation);
        let address = target.device_ptr_mut(&self.allocation).0;
        // Pageable DMA can block behind earlier work: poll it first.
        poll_while_busy(self.raw());
        record(1, self.handle);
        // SAFETY: the pending entry owns the complete source span through a
        // proving wait, including early errors and failed completion handoffs.
        let copied =
            unsafe { sys::cuMemcpyHtoDAsync_v2(address, source, bytes, self.raw()) }.result();
        let completed = match self.hold(held) {
            Ok(false) => Ok(()),
            Ok(true) => self.wait_borrowed(),
            Err(held) => {
                let completed = self.wait_borrowed();
                if completed.is_ok() {
                    held.free();
                }
                // Otherwise the type-erased entry leaks its DMA owner.
                completed
            }
        };
        self.complete(copied.and(completed))
    }
    /// A constant fill of the first `count` elements of `target` (S5 (a)): one
    /// memset of `byte` on the borrowed stream, then the completion handoff.
    /// The bytes equal those of an upload of `count` equal values, with no host
    /// staging and no wait.
    pub(crate) fn fill<T: DeviceRepr, D: DevicePtrMut<T>>(
        &self,
        target: &mut D,
        count: usize,
        byte: u8,
    ) -> Result<(), DriverError> {
        assert!(count <= target.len());
        check_stream(target.stream(), &self.allocation);
        let address = target.device_ptr_mut(&self.allocation).0;
        // SAFETY: `target` covers `count` elements of this allocation stream.
        unsafe { self.memset(address, byte, count * std::mem::size_of::<T>()) }
    }
    /// A new allocation of `count` elements filled with `byte` (S5 (a)), in
    /// place of an upload of `count` equal values.
    pub(crate) fn alloc_filled<T: DeviceRepr>(
        &self,
        count: usize,
        byte: u8,
    ) -> Result<CudaSlice<T>, DriverError> {
        // SAFETY: initialized immediately below; owner is the allocation stream.
        let mut target = unsafe { self.alloc(count) }?;
        if let Err(error) = self.fill(&mut target, count, byte) {
            if handoff_failed(&self.allocation) {
                std::mem::forget(target);
            }
            return Err(error);
        }
        Ok(target)
    }
    pub(crate) fn memcpy_dtoh<T: DeviceRepr, S: DevicePtr<T>>(
        &self,
        source: &S,
        host: &mut [T],
    ) -> Result<(), DriverError> {
        self.memcpy_dtoh_category(source, host, 1)
    }
    pub(crate) fn detail_memcpy_dtoh<T: DeviceRepr, S: DevicePtr<T>>(
        &self,
        source: &S,
        host: &mut [T],
    ) -> Result<(), DriverError> {
        self.memcpy_dtoh_category(source, host, 7)
    }
    fn memcpy_dtoh_category<T: DeviceRepr, S: DevicePtr<T>>(
        &self,
        source: &S,
        host: &mut [T],
        category: usize,
    ) -> Result<(), DriverError> {
        assert!(host.len() <= source.len());
        self.prepare()?;
        let mut staging = host_staging::<T>(host.len())?;
        check_stream(source.stream(), &self.allocation);
        let address = source.device_ptr(&self.allocation).0;
        // A copy to pageable staging blocks until earlier work on the stream
        // has finished: wait for that work first, with polls (§3.5).
        poll_while_busy(self.raw());
        let mark = self.pending_mark();
        record(category, self.handle);
        // SAFETY: independently owned staging covers the entire control copy;
        // a failed stream never leaves DMA targeting the caller's stack.
        let copied = unsafe {
            sys::cuMemcpyDtoHAsync_v2(
                staging.as_mut_ptr().cast(),
                address,
                std::mem::size_of_val(host),
                self.raw(),
            )
        }
        .result();
        let completed = unsafe { sys::cuStreamSynchronize(self.raw()) }.result();
        // The synchronize of the stream every earlier upload was enqueued on
        // proves those uploads complete (S5 (b)).
        self.settle_from(mark, completed.is_ok());
        if completed.is_err() {
            std::mem::forget(staging);
        } else if copied.is_ok() {
            // SAFETY: successful copy plus synchronization initialized all T
            // representations. DeviceRepr is the same transport contract used
            // by cudarc's typed host copies.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    staging.as_ptr().cast::<T>(),
                    host.as_mut_ptr(),
                    host.len(),
                );
            }
        }
        self.complete(copied.and(completed))
    }
    /// # Safety
    /// Contents are unset (or testing poison) and must be initialized before use.
    pub(crate) unsafe fn alloc<T: DeviceRepr>(
        &self,
        len: usize,
    ) -> Result<CudaSlice<T>, DriverError> {
        self.prepare()?;
        // Allocation alone belongs to the owned stream. Establish readiness only
        // here; kernel/copy submission never runs an extra allocation-stream job.
        let target = match self.device.0.pool.as_ref() {
            Some(pool) => unsafe { self.pool_alloc::<T>(pool, len) }?,
            // The S2 fallback mode: today's cudarc allocation on the current pool.
            None => unsafe { self.allocation.alloc::<T>(len) }?,
        };
        self.order(self.allocation.cu_stream(), self.raw())?;
        #[cfg(feature = "testing")]
        if let Some(byte) = crate::launch::poison() {
            let bytes = len * std::mem::size_of::<T>();
            if bytes != 0 {
                let address = target.device_ptr(&self.allocation).0;
                if let Err(error) = unsafe { self.memset(address, byte, bytes) } {
                    if handoff_failed(&self.allocation) {
                        std::mem::forget(target);
                    }
                    return Err(error);
                }
                crate::launch::count_poisoned(bytes as u64);
            }
        }
        Ok(target)
    }
    /// An allocation from the ordinal's owned pool on the owned stream (S2):
    /// on CUDA_ERROR_OUT_OF_MEMORY, wait for this lease's owned stream (never
    /// the context), trim the pool to zero and retry once; a second refusal is
    /// reported as the allocation failure it is today.
    ///
    /// # Safety
    /// As [`Self::alloc`].
    unsafe fn pool_alloc<T: DeviceRepr>(
        &self,
        pool: &crate::pool::DevicePool,
        len: usize,
    ) -> Result<CudaSlice<T>, DriverError> {
        let bytes = len * std::mem::size_of::<T>();
        let owned = self.allocation.cu_stream();
        let address = match pool.alloc_async(bytes, owned) {
            Ok(address) => address,
            Err(error) if error.0 == sys::CUresult::CUDA_ERROR_OUT_OF_MEMORY => {
                // The wait binds the context; on a failed wait the refusal stands.
                if wait_owned_stream(&self.allocation).is_err() {
                    return Err(error);
                }
                // A failed trim leaves the retry to report the refusal.
                let _ = pool.trim();
                pool.alloc_async(bytes, owned)?
            }
            Err(error) => return Err(error),
        };
        // SAFETY: a live pool allocation of `bytes` bytes, made on the owned
        // stream; with event tracking off this wraps it without any driver
        // call, and its drop frees it on the owned stream.
        Ok(unsafe { self.allocation.upgrade_device_ptr::<T>(address, len) })
    }
    pub(crate) fn clone_htod<T: DeviceRepr>(
        &self,
        host: &[T],
    ) -> Result<CudaSlice<T>, DriverError> {
        // SAFETY: initialized immediately below; owner is the allocation stream.
        let mut target = unsafe { self.alloc(host.len()) }?;
        if let Err(error) = self.memcpy_htod(host, &mut target) {
            if handoff_failed(&self.allocation) {
                std::mem::forget(target);
            }
            return Err(error);
        }
        Ok(target)
    }
    /// # Safety
    /// The caller retains both complete, non-overlapping same-context spans.
    pub(crate) unsafe fn copy_device(
        &self,
        target: u64,
        source: u64,
        bytes: usize,
    ) -> Result<(), DriverError> {
        unsafe { self.copy_device_category(target, source, bytes, 1) }
    }
    /// # Safety
    /// The caller retains both complete, non-overlapping same-context spans.
    pub(crate) unsafe fn detail_copy_device(
        &self,
        target: u64,
        source: u64,
        bytes: usize,
    ) -> Result<(), DriverError> {
        unsafe { self.copy_device_category(target, source, bytes, 7) }
    }
    unsafe fn copy_device_category(
        &self,
        target: u64,
        source: u64,
        bytes: usize,
        category: usize,
    ) -> Result<(), DriverError> {
        self.prepare()?;
        record(category, self.handle);
        let copied =
            unsafe { sys::cuMemcpyDtoDAsync_v2(target, source, bytes, self.raw()) }.result();
        self.complete(copied)
    }
    /// # Safety
    /// The caller retains a writable device span covering `bytes` bytes.
    pub(crate) unsafe fn memset(
        &self,
        target: u64,
        byte: u8,
        bytes: usize,
    ) -> Result<(), DriverError> {
        self.prepare()?;
        record(1, self.handle);
        let filled = unsafe { sys::cuMemsetD8Async(target, byte, bytes, self.raw()) }.result();
        self.complete(filled)
    }
}

impl Drop for BorrowedExecutionStream {
    fn drop(&mut self) {
        // Pending host staging may still be read by an enqueued copy: free it
        // only after the owned stream's host-proved completion (no polls, no
        // Python), and leak it if that wait fails (S5 (b), I8). A lease that
        // waited in its own Drop has settled the list already.
        if !self.has_pending() {
            return;
        }
        let proved = crate::device::guard(|| {
            retire_in_drop(&self.allocation)
                .map_err(|e| failure_ordinal(&self.device, "retiring host upload staging", e))
        })
        .is_ok();
        self.settle(proved);
    }
}

enum Argument {
    Value(*mut c_void),
    Pointer(u64),
}
pub(crate) struct ExecutionLaunch<'a> {
    stream: &'a BorrowedExecutionStream,
    function: &'a CudaFunction,
    args: Vec<Argument>,
    category: usize,
    lifetime: PhantomData<&'a ()>,
}
// SAFETY: DeviceRepr guarantees a kernel-parameter representation; the builder
// retains the argument's lifetime until launch and passes its address unchanged.
unsafe impl<'a, 'b: 'a, T: DeviceRepr> PushKernelArg<&'b T> for ExecutionLaunch<'a> {
    fn arg(&mut self, value: &'b T) -> &mut Self {
        self.args
            .push(Argument::Value(std::ptr::from_ref(value).cast_mut().cast()));
        self
    }
}
impl<'a> ExecutionLaunch<'a> {
    pub(crate) fn slice_arg<T>(&mut self, value: &'a CudaSlice<T>) -> &mut Self {
        check_stream(value.stream(), &self.stream.allocation);
        self.args.push(Argument::Pointer(
            value.device_ptr(&self.stream.allocation).0,
        ));
        self
    }
    pub(crate) fn slice_arg_mut<T>(&mut self, value: &'a mut CudaSlice<T>) -> &mut Self {
        check_stream(value.stream(), &self.stream.allocation);
        self.args.push(Argument::Pointer(
            value.device_ptr(&self.stream.allocation).0,
        ));
        self
    }
    /// # Safety
    /// Caller verifies argument ABI, launch geometry and all kernel access spans.
    pub(crate) unsafe fn launch(&mut self, config: LaunchConfig) -> Result<(), DriverError> {
        self.stream.prepare()?;
        let mut pointers: Vec<*mut c_void> = self
            .args
            .iter_mut()
            .map(|arg| match arg {
                Argument::Value(pointer) => *pointer,
                Argument::Pointer(address) => std::ptr::from_mut(address).cast(),
            })
            .collect();
        record(self.category, self.stream.handle);
        let queued = unsafe {
            sys::cuLaunchKernel(
                self.function.cu_function(),
                config.grid_dim.0,
                config.grid_dim.1,
                config.grid_dim.2,
                config.block_dim.0,
                config.block_dim.1,
                config.block_dim.2,
                config.shared_mem_bytes,
                self.stream.raw(),
                pointers.as_mut_ptr(),
                std::ptr::null_mut(),
            )
        }
        .result();
        self.stream.complete(queued)
    }
}

// One temporary transfer buffer at a time; never per-pair host objects. A
// fallible reservation maps to the existing DeviceError through DriverError.
fn host_staging<T>(len: usize) -> Result<Vec<std::mem::MaybeUninit<T>>, DriverError> {
    let mut staging = Vec::new();
    if staging.try_reserve_exact(len).is_err() {
        return Err(sys::CUresult::CUDA_ERROR_OUT_OF_MEMORY
            .result()
            .unwrap_err());
    }
    staging.resize_with(len, std::mem::MaybeUninit::uninit);
    Ok(staging)
}

/// The staging copy of a host-origin upload, reserved through the shared helper
/// with the label `host upload staging` (host-allocation-failures §12.3 B4).
pub(crate) fn upload_staging<T>(len: usize) -> Result<Vec<std::mem::MaybeUninit<T>>, EngineError> {
    let mut staging = Vec::new();
    seqblaze_core::detail::try_reserve_exact(&mut staging, len, "host upload staging")?;
    staging.resize_with(len, std::mem::MaybeUninit::uninit);
    Ok(staging)
}

/// Preserve the original owned-stream upload path outside resident calls.
pub(crate) fn upload_on<T: DeviceRepr>(
    stream: &Arc<CudaStream>,
    execution: Option<&BorrowedExecutionStream>,
    host: &[T],
) -> Result<CudaSlice<T>, DriverError> {
    if let Some(execution) = execution {
        return execution.clone_htod(host);
    }
    // SAFETY: initialized by the following owned-stream upload.
    let mut target = unsafe { crate::session::dev_alloc(stream, host.len()) }?;
    crate::session::htod(stream, host, &mut target)?;
    Ok(target)
}
