//! Native DLPack retirement. Only ordinary export calls create workers/nodes.
//! Deleters transfer preallocated nodes, without Python, CUDA or GPU waits.
use std::{
    mem::ManuallyDrop,
    panic::{catch_unwind, AssertUnwindSafe},
    sync::{
        atomic::{AtomicBool, AtomicUsize, Ordering},
        Arc, Condvar, Mutex,
    },
};

use cudarc::driver::{sys, CudaContext};
use seqblaze_core::validate::EngineError;

use super::DeviceColumn;
use crate::device::{guard, Device};

static CLOSING: AtomicBool = AtomicBool::new(false);
// Statics are process-lifetime owners; they never destruct CUDA during exit.
static QUEUES: Mutex<Vec<Arc<ContextQueue>>> = Mutex::new(Vec::new());
// Also bound zero-byte exports and their host nodes. Failed reservations stay
// charged, so repeated driver failures cannot accumulate unlimited exports.
const MAX_TICKETS: usize = 65536;

struct Pending {
    head: Option<Box<Node>>,
    active: usize,
}

struct ContextQueue {
    context: Arc<CudaContext>,
    _device: Device,
    pending: Mutex<Pending>,
    ready: Condvar,
    completed: Condvar,
    reserved_bytes: AtomicUsize,
    tickets: AtomicUsize,
    byte_limit: usize,
}

fn unavailable() -> EngineError {
    EngineError::Device("unable to reserve native DLPack retirement resources".into())
}

// Keep Rust 1.85 support: Atomic::try_update requires Rust 1.95, while its
// predecessor is deprecated by the development toolchain. Reserve with CAS.
fn reserve(counter: &AtomicUsize, amount: usize, limit: usize) -> Result<(), EngineError> {
    let mut current = counter.load(Ordering::Acquire);
    loop {
        let next = current
            .checked_add(amount)
            .filter(|&n| n <= limit)
            .ok_or_else(unavailable)?;
        match counter.compare_exchange_weak(current, next, Ordering::AcqRel, Ordering::Acquire) {
            Ok(_) => return Ok(()),
            Err(actual) => current = actual,
        }
    }
}

fn queue(column: &DeviceColumn) -> Result<Arc<ContextQueue>, EngineError> {
    guard(|| {
        let mut queues = QUEUES.lock().unwrap_or_else(|e| e.into_inner());
        if CLOSING.load(Ordering::Acquire) {
            return Err(unavailable());
        }
        let context = column.export_context();
        if let Some(queue) = queues
            .iter()
            .find(|q| q.context.cu_ctx() == context.cu_ctx())
        {
            return Ok(Arc::clone(queue));
        }
        queues.try_reserve(1).map_err(|_| unavailable())?;
        context.bind_to_thread().map_err(|_| unavailable())?;
        let mut device = 0;
        let mut bytes = 0;
        // SAFETY: this is the live retained allocation context; outputs are
        // correctly typed host scalars. Reservation is bounded by device RAM.
        unsafe {
            sys::cuCtxGetDevice(&mut device)
                .result()
                .map_err(|_| unavailable())?;
            sys::cuDeviceTotalMem_v2(&mut bytes, device)
                .result()
                .map_err(|_| unavailable())?;
        }
        let queue = Arc::new(ContextQueue {
            context: Arc::clone(context),
            _device: column.export_device(),
            pending: Mutex::new(Pending {
                head: None,
                active: 0,
            }),
            ready: Condvar::new(),
            completed: Condvar::new(),
            reserved_bytes: AtomicUsize::new(0),
            tickets: AtomicUsize::new(0),
            byte_limit: bytes,
        });
        let worker = Arc::clone(&queue);
        // std threads inherit the creating thread's CPU affinity. Dropping the
        // JoinHandle detaches it; module shutdown never joins with the GIL held.
        std::thread::Builder::new()
            .name("seqblaze-dlpack-retire".into())
            .spawn(move || worker.run())
            .map_err(|_| unavailable())?;
        queues.push(Arc::clone(&queue));
        Ok(queue)
    })
}

struct Node {
    column: Option<ManuallyDrop<DeviceColumn>>,
    queue: Arc<ContextQueue>,
    next: Option<Box<Node>>,
    bytes: usize,
    #[cfg(feature = "testing")]
    counters: Arc<Counters>,
}

impl Drop for Node {
    fn drop(&mut self) {
        self.queue
            .reserved_bytes
            .fetch_sub(self.bytes, Ordering::AcqRel);
        self.queue.tickets.fetch_sub(1, Ordering::AcqRel);
    }
}

/// Prepared before D2D copying and capsule publication. Dropping an attached
/// job submits it, including a failed capsule-construction path.
pub struct RetireJob {
    node: Option<Box<Node>>,
}

impl RetireJob {
    pub fn prepare(source: &DeviceColumn) -> Result<Self, EngineError> {
        let queue = queue(source)?;
        let bytes = source.nbytes().max(1);
        reserve(&queue.tickets, 1, MAX_TICKETS)?;
        if reserve(&queue.reserved_bytes, bytes, queue.byte_limit).is_err() {
            queue.tickets.fetch_sub(1, Ordering::AcqRel);
            return Err(unavailable());
        }
        let job = Self {
            node: Some(Box::new(Node {
                column: None,
                queue,
                next: None,
                bytes,
                #[cfg(feature = "testing")]
                counters: Arc::new(Counters::default()),
            })),
        };
        #[cfg(feature = "testing")]
        {
            let mut records = RECORDS.lock().unwrap_or_else(|e| e.into_inner());
            records.try_reserve(1).map_err(|_| unavailable())?;
            records.push(Arc::clone(&job.node.as_ref().unwrap().counters));
        }
        Ok(job)
    }

    pub fn attach(&mut self, column: DeviceColumn) {
        let node = self.node.as_mut().expect("prepared retirement ticket");
        assert!(node.column.is_none());
        assert_eq!(
            column.export_context().cu_ctx(),
            node.queue.context.cu_ctx()
        );
        assert_eq!(column.nbytes().max(1), node.bytes);
        node.column = Some(ManuallyDrop::new(column));
        #[cfg(feature = "testing")]
        node.counters.created.store(1, Ordering::Release);
    }

    pub fn submit(&mut self) {
        let Some(mut node) = self.node.take() else {
            return;
        };
        if node.column.is_none() {
            return;
        }
        #[cfg(feature = "testing")]
        node.counters.enqueued.fetch_add(1, Ordering::AcqRel);
        if CLOSING.load(Ordering::Acquire) {
            node.retain();
            return;
        }
        let queue = Arc::clone(&node.queue);
        let mut pending = queue.pending.lock().unwrap_or_else(|e| e.into_inner());
        // The intrusive node is already allocated. This lock is never held by
        // a CUDA operation, so a callback deleter cannot wait for GPU progress.
        node.next = pending.head.take();
        pending.head = Some(node);
        pending.active += 1;
        queue.ready.notify_one();
    }
}

impl Drop for RetireJob {
    fn drop(&mut self) {
        self.submit();
    }
}

impl Node {
    fn retain(self: Box<Self>) {
        #[cfg(feature = "testing")]
        {
            self.counters.retained.fetch_add(1, Ordering::AcqRel);
            self.counters
                .retained_bytes
                .store(self.bytes, Ordering::Release);
        }
        // Keep allocation, exact context, Device and budget ticket together.
        std::mem::forget(self);
    }
}

impl ContextQueue {
    fn run(&self) {
        loop {
            let mut node = {
                let mut pending = self.pending.lock().unwrap_or_else(|e| e.into_inner());
                while pending.head.is_none() {
                    pending = self.ready.wait(pending).unwrap_or_else(|e| e.into_inner());
                }
                let mut node = pending.head.take().unwrap();
                pending.head = node.next.take();
                node
            };
            let released = !CLOSING.load(Ordering::Acquire)
                && matches!(
                    catch_unwind(AssertUnwindSafe(|| {
                        node.column.as_mut().unwrap().retire_export()?;
                        // SAFETY: context barrier and actual allocation free both
                        // succeeded. Drop only the now-empty native column owner.
                        unsafe { ManuallyDrop::drop(node.column.as_mut().unwrap()) };
                        node.column = None;
                        Ok::<(), EngineError>(())
                    })),
                    Ok(Ok(()))
                );
            if released {
                #[cfg(feature = "testing")]
                node.counters.freed.fetch_add(1, Ordering::AcqRel);
                drop(node);
            } else {
                node.retain();
            }
            let mut pending = self.pending.lock().unwrap_or_else(|e| e.into_inner());
            pending.active -= 1;
            self.completed.notify_all();
        }
    }
}

/// Module teardown only sets native state. Never join, attach Python or CUDA.
pub fn close() {
    CLOSING.store(true, Ordering::Release);
}

#[cfg(feature = "testing")]
use std::{
    sync::atomic::AtomicU64,
    time::{Duration, Instant},
};

#[cfg(feature = "testing")]
#[derive(Default)]
struct Counters {
    created: AtomicU64,
    enqueued: AtomicU64,
    freed: AtomicU64,
    retained: AtomicU64,
    retained_bytes: AtomicUsize,
}
#[cfg(feature = "testing")]
static RECORDS: Mutex<Vec<Arc<Counters>>> = Mutex::new(Vec::new());

/// Stable IDs and cumulative records survive successful retirement.
#[cfg(feature = "testing")]
pub fn export_stats() -> Vec<(usize, [u64; 5])> {
    RECORDS
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .iter()
        .enumerate()
        .map(|(id, c)| {
            (
                id + 1,
                [
                    c.created.load(Ordering::Acquire),
                    c.enqueued.load(Ordering::Acquire),
                    c.freed.load(Ordering::Acquire),
                    c.retained.load(Ordering::Acquire),
                    c.retained_bytes.load(Ordering::Acquire) as u64,
                ],
            )
        })
        .collect()
}

/// Ordinary test threads only: the binding detaches Python before waiting.
#[cfg(feature = "testing")]
pub fn drain(timeout: Duration) -> bool {
    let start = Instant::now();
    let queues = QUEUES.lock().unwrap_or_else(|e| e.into_inner()).clone();
    for queue in queues {
        let mut pending = queue.pending.lock().unwrap_or_else(|e| e.into_inner());
        while pending.active != 0 {
            let Some(left) = timeout.checked_sub(start.elapsed()) else {
                return false;
            };
            let (state, _) = queue
                .completed
                .wait_timeout(pending, left)
                .unwrap_or_else(|e| e.into_inner());
            pending = state;
        }
    }
    true
}
