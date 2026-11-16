//! Bounded process-lifetime cache of immutable device pack arrays (S12).
//! An upload belongs to its lease until the *final* successful completion wait.
use crate::interop::{execution_stream::BorrowedExecutionStream, Retained};
use cudarc::driver::{CudaSlice, DeviceRepr};
use std::{
    hash::{Hash, Hasher},
    sync::{Arc, Mutex},
};

const ENTRIES: usize = 16;
const BYTES: usize = 8 << 20;

/// One kind per uploaded content and element type, so that no entry is ever
/// read under another kind's layout.
#[derive(Clone, Copy, Hash, PartialEq, Eq)]
pub(crate) enum Kind {
    /// K1 full keys (i64 × 4 per cell) and CLS.
    K1,
    /// K1 score keys at int64 width (the wide pass) and CLS.
    K1Wide,
    /// K1 score keys narrowed to int32 (the narrow pass) and CLS.
    K1Narrow,
    K3,
    Register,
    Primer3,
}

struct Key {
    kind: Kind,
    hash: u64,
    split: usize,
    image: Vec<u8>,
}
impl Key {
    fn new(kind: Kind, a: &[u8], b: &[u8]) -> Option<Self> {
        let len = a.len().checked_add(b.len())?;
        if len > BYTES {
            return None;
        }
        let mut image = Vec::new();
        image.try_reserve_exact(len).ok()?;
        image.extend_from_slice(a);
        image.extend_from_slice(b);
        let mut hash = std::collections::hash_map::DefaultHasher::new();
        kind.hash(&mut hash);
        a.len().hash(&mut hash);
        image.hash(&mut hash);
        Some(Self {
            kind,
            hash: hash.finish(),
            split: a.len(),
            image,
        })
    }
    fn matches(&self, other: &Self) -> bool {
        self.kind == other.kind
            && self.hash == other.hash
            && self.split == other.split
            && self.image == other.image
    }
}
struct Entry {
    key: Key,
    addresses: [u64; 2],
}
#[derive(Default)]
struct Entries {
    values: Vec<Entry>,
    bytes: usize,
}
#[derive(Default)]
pub(crate) struct Cache(Mutex<Entries>);
impl Cache {
    fn lookup(&self, key: &Key) -> Option<[u64; 2]> {
        self.0
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .values
            .iter()
            .find(|e| e.key.matches(key))
            .map(|e| e.addresses)
    }
    #[cfg(feature = "testing")]
    pub(crate) fn stats(&self) -> (u64, u64) {
        let entries = self.0.lock().unwrap_or_else(|p| p.into_inner());
        (entries.bytes as u64, entries.values.len() as u64)
    }
}

/// The only bytes admitted here are initialized scalar integer pack arrays.
/// No status/defect/control/poison/item/slot buffer passes this interface.
pub(crate) trait Candidate: Send + Sync {
    fn insert(&self, cache: &Cache);
}
enum Storage<T, U> {
    Resident(Retained<(CudaSlice<T>, CudaSlice<U>)>),
    // The private Session path keeps its original ordinary slice retirement;
    // it neither participates in the resident cache nor gains a Drop wait.
    Private(#[allow(dead_code)] (CudaSlice<T>, CudaSlice<U>)),
}
type Uploaded<T, U> = (Option<Key>, Storage<T, U>);
pub(crate) struct Pack<T, U> {
    addresses: [u64; 2],
    uploaded: Mutex<Option<Uploaded<T, U>>>,
}
impl<T, U> Pack<T, U> {
    pub(crate) fn addresses(&self) -> [u64; 2] {
        self.addresses
    }
}
impl<T: Send + Sync, U: Send + Sync> Candidate for Pack<T, U> {
    fn insert(&self, cache: &Cache) {
        let mut uploaded = self.uploaded.lock().unwrap_or_else(|p| p.into_inner());
        let Some((key, _)) = uploaded.as_ref() else {
            return;
        };
        let Some(key) = key.as_ref() else {
            return;
        };
        let mut entries = cache.0.lock().unwrap_or_else(|p| p.into_inner());
        if entries.values.len() >= ENTRIES
            || key.image.len() > BYTES - entries.bytes
            || entries.values.iter().any(|e| e.key.matches(key))
            || entries.values.try_reserve(1).is_err()
        {
            return;
        }
        let (key, buffers) = uploaded.take().expect("offered pack");
        // Host-only with tracking disabled; the lease still retains its stream.
        // Bypass the Retained destructor only after the successful final wait.
        let Storage::Resident(buffers) = buffers else {
            unreachable!("only resident packs are offered")
        };
        let (a, b) = buffers.into_inner();
        let bytes = key.as_ref().expect("cacheable pack").image.len();
        a.leak();
        b.leak();
        entries.bytes += bytes;
        entries.values.push(Entry {
            key: key.expect("cacheable pack"),
            addresses: self.addresses,
        });
    }
}

/// Exact array transport bytes, not a serialized launch configuration.
/// T has no padding at any call site (u8, i32, u64 or i64).
pub(crate) fn bytes<T: Copy>(values: &[T]) -> &[u8] {
    // SAFETY: used only for the initialized integer scalars named above.
    unsafe { std::slice::from_raw_parts(values.as_ptr().cast(), std::mem::size_of_val(values)) }
}

impl<T: DeviceRepr + Send + Sync + 'static, U: DeviceRepr + Send + Sync + 'static> Pack<T, U> {
    pub(crate) fn get<E>(
        stream: &Arc<cudarc::driver::CudaStream>,
        device: &crate::Device,
        execution: Option<&BorrowedExecutionStream>,
        kind: Kind,
        image: [&[u8]; 2],
        upload: impl FnOnce() -> Result<(CudaSlice<T>, CudaSlice<U>), E>,
    ) -> Result<Arc<Self>, E> {
        let key = execution.and_then(|_| Key::new(kind, image[0], image[1]));
        if let Some(addresses) = key.as_ref().and_then(|k| device.0.pack_cache.lookup(k)) {
            return Ok(Arc::new(Self {
                addresses,
                uploaded: Mutex::new(None),
            }));
        }
        // No cache lock spans upload, pool retry, wait or poll. Existing upload
        // reservations and their operation-specific errors are the caller's.
        let (a, b) = upload()?;
        let addresses = [
            crate::session::ptr(&a, stream),
            crate::session::ptr(&b, stream),
        ];
        let pack = Arc::new(Self {
            addresses,
            uploaded: Mutex::new(Some((
                key,
                if execution.is_some() {
                    Storage::Resident(Retained::on_owned_stream(stream, device, (a, b)))
                } else {
                    Storage::Private((a, b))
                },
            ))),
        });
        if let Some(execution) = execution {
            execution.offer_pack(pack.clone());
        }
        Ok(pack)
    }
}
