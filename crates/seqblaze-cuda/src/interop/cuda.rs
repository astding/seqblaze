//! Driver-only dependencies and owned device columns. No CUDA runtime calls.
use std::sync::Arc;

use cudarc::driver::{sys, CudaContext, CudaSlice, CudaStream};
use seqblaze_core::validate::EngineError;

use super::{DeviceArray, DeviceArrayDescriptor, DeviceDType, DeviceRows, DeviceSequenceTable};
use crate::{
    device::{guard, Device},
    session::{failure_ordinal, ptr},
};

// SAFETY: repr(C), fourteen 64-bit integer fields, matching DeviceRows in CUDA.
unsafe impl cudarc::driver::DeviceRepr for DeviceRows {}

fn failed(device: &Device, operation: &str, error: cudarc::driver::DriverError) -> EngineError {
    failure_ordinal(device, operation, error)
}

/// Validate the real device/context and initialized allocation span, not just
/// a producer's self-reported ordinal. Empty views do not query address zero.
fn validate_pointer(device: &Device, array: &DeviceArrayDescriptor) -> Result<(), EngineError> {
    array.validate()?;
    if array.ordinal != device.info().ordinal {
        return Err(EngineError::Device(format!("device input belongs to cuda:{}, requested cuda:{}; cross-device inputs are not supported", array.ordinal, device.info().ordinal)));
    }
    let Some((first, end)) = array.span()? else {
        return Ok(());
    };
    device
        .0
        .ctx
        .bind_to_thread()
        .map_err(|e| failed(device, "binding an imported allocation", e))?;
    let mut ordinal = -1i32;
    let mut context: sys::CUcontext = std::ptr::null_mut();
    let mut base = 0u64;
    let mut bytes = 0usize;
    // SAFETY: the driver writes one correctly typed scalar per attribute; no
    // CPU dereference of the supplied device pointer occurs.
    unsafe {
        sys::cuPointerGetAttribute(
            (&mut context as *mut sys::CUcontext).cast(),
            sys::CUpointer_attribute::CU_POINTER_ATTRIBUTE_CONTEXT,
            array.address,
        )
        .result()
        .map_err(|e| failed(device, "identifying an imported allocation context", e))?;
        sys::cuPointerGetAttribute(
            (&mut ordinal as *mut i32).cast(),
            sys::CUpointer_attribute::CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL,
            array.address,
        )
        .result()
        .map_err(|e| failed(device, "identifying an imported device allocation", e))?;
        sys::cuPointerGetAttribute(
            (&mut base as *mut u64).cast(),
            sys::CUpointer_attribute::CU_POINTER_ATTRIBUTE_RANGE_START_ADDR,
            array.address,
        )
        .result()
        .map_err(|e| failed(device, "reading an imported allocation start", e))?;
        sys::cuPointerGetAttribute(
            (&mut bytes as *mut usize).cast(),
            sys::CUpointer_attribute::CU_POINTER_ATTRIBUTE_RANGE_SIZE,
            array.address,
        )
        .result()
        .map_err(|e| failed(device, "reading an imported allocation size", e))?;
    }
    if !context.is_null() && context != device.0.ctx.cu_ctx() {
        return Err(EngineError::Device(
            "device input allocation belongs to a different CUDA context".into(),
        ));
    }
    if usize::try_from(ordinal).ok() != Some(device.info().ordinal) {
        return Err(EngineError::Device(
            "device input pointer belongs to a different CUDA device".into(),
        ));
    }
    let allocation_end = base
        .checked_add(bytes as u64)
        .ok_or_else(|| EngineError::Parameter("device allocation span overflows".into()))?;
    if first < base || end > allocation_end {
        return Err(EngineError::Parameter(
            "device view extends beyond its retained allocation".into(),
        ));
    }
    Ok(())
}

/// Add producer -> consumer ordering without copying data or synchronizing
/// the whole device. Special CAI stream tokens use the current primary context.
pub(super) fn external_stream(device: &Device, handle: u64) -> Result<sys::CUstream, EngineError> {
    if handle == 0 {
        return Err(EngineError::Parameter(
            "CUDA Array Interface stream 0 is ambiguous".into(),
        ));
    }
    device
        .0
        .ctx
        .bind_to_thread()
        .map_err(|e| failed(device, "binding an imported stream", e))?;
    let stream = handle as usize as sys::CUstream;
    if handle > 2 {
        let mut context: sys::CUcontext = std::ptr::null_mut();
        // SAFETY: the driver validates this opaque stream metadata handle.
        unsafe { sys::cuStreamGetCtx(stream, &mut context) }
            .result()
            .map_err(|e| failed(device, "identifying an external stream", e))?;
        if context != device.0.ctx.cu_ctx() {
            return Err(EngineError::Device(
                "external stream belongs to a different CUDA context/device".into(),
            ));
        }
    }
    Ok(stream)
}
pub(super) fn order_streams(
    device: &Device,
    producer: sys::CUstream,
    consumer: sys::CUstream,
) -> Result<(), EngineError> {
    if producer == consumer {
        return Ok(());
    }
    // The caller bound this context (external_stream). The event comes from the
    // ordinal's free list and returns to it once the wait is enqueued (S1 (c)).
    device
        .0
        .events
        .order(producer, consumer)
        .map_err(|created| match created {
            crate::pool::Created::Failed(e) => failed(device, "creating a producer dependency", e),
            crate::pool::Created::Queued(e) => failed(device, "waiting for the producer stream", e),
        })
}
fn wait_for_stream(
    device: &Device,
    consumer: &CudaStream,
    producer: Option<u64>,
) -> Result<(), EngineError> {
    let Some(producer) = producer else {
        return Ok(());
    };
    order_streams(
        device,
        external_stream(device, producer)?,
        consumer.cu_stream(),
    )
}

/// Retains every foreign owner until the consumer stream has finished, also
/// on early errors. Returning a host error never releases in-flight inputs.
/// Query a real device pointer's allocation ordinal. This reads driver control
/// metadata only and never dereferences or downloads the payload.
pub fn device_pointer_ordinal(address: u64) -> Result<usize, EngineError> {
    guard(|| {
        if address == 0 {
            return Err(EngineError::Parameter(
                "an empty device pointer has no queryable allocation ordinal".into(),
            ));
        }
        // SAFETY: cuInit and pointer attributes are driver metadata queries.
        unsafe { sys::cuInit(0) }.result().map_err(|e| {
            EngineError::Device(format!(
                "initializing CUDA pointer inspection failed: {e:?}"
            ))
        })?;
        let mut ordinal = -1i32;
        unsafe {
            sys::cuPointerGetAttribute(
                (&mut ordinal as *mut i32).cast(),
                sys::CUpointer_attribute::CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL,
                address,
            )
        }
        .result()
        .map_err(|e| {
            EngineError::Device(format!("querying CUDA input allocation failed: {e:?}"))
        })?;
        usize::try_from(ordinal)
            .map_err(|_| EngineError::Device("CUDA input is not a device allocation".into()))
    })
}

pub struct DeviceLease {
    device: Device,
    stream: Arc<CudaStream>,
    execution: super::execution_stream::BorrowedExecutionStream,
    inputs: Vec<DeviceArray>,
}

enum HostUpload {
    Data(Vec<u8>),
    Offsets(Vec<i64>),
    Table { address: usize, bytes: usize },
}

impl DeviceLease {
    fn upload_host_buffer(
        &self,
        dtype: DeviceDType,
        elements: usize,
        buffer: HostUpload,
        owner: &Arc<seqblaze_core::seq::SeqTable>,
    ) -> Result<DeviceColumn, EngineError> {
        let mut output = self.output(dtype, vec![elements])?;
        if elements == 0 {
            // No DMA reads an empty source.
            return unsafe { output.finish() };
        }
        let copied = match buffer {
            HostUpload::Data(data) => self
                .execution
                .memcpy_htod_owned(data, output.storage.get_mut()),
            HostUpload::Offsets(offsets) => self
                .execution
                .memcpy_htod_owned(offsets, output.storage.get_mut()),
            HostUpload::Table { address, bytes } => {
                // SAFETY: bind_host_table obtains these spans from the
                // retained owner's batch or validated fixed-width view.
                unsafe {
                    self.execution.memcpy_htod_table(
                        address,
                        bytes,
                        Arc::clone(owner),
                        output.storage.get_mut(),
                    )
                }
            }
        };
        copied.map_err(|e| failed(&self.device, "uploading host-origin device bytes", e))?;
        // SAFETY: the same-stream copy initialized all logical bytes.
        unsafe { output.finish() }
    }

    pub fn new(device: &Device) -> Result<Self, EngineError> {
        Self::with_stream(device, 1)
    }
    pub fn with_stream(device: &Device, handle: u64) -> Result<Self, EngineError> {
        guard(|| {
            // cudarc otherwise falls back to cuMemAlloc/cuMemFree, whose
            // device-wide barriers can wait for unrelated foreign producers.
            if !device.0.ctx.has_async_alloc() {
                return Err(EngineError::Device(
                    "resident CUDA execution requires stream-ordered allocation support; use device='cpu'".into(),
                ));
            }
            let stream = device
                .0
                .ctx
                .new_stream()
                .map_err(|e| failed(device, "creating an interop stream", e))?;
            Ok(Self {
                device: device.clone(),
                execution: super::execution_stream::BorrowedExecutionStream::new(
                    device,
                    Arc::clone(&stream),
                    handle,
                )?,
                stream,
                inputs: Vec::new(),
            })
        })
    }
    pub(crate) fn output(
        &self,
        dtype: DeviceDType,
        shape: Vec<usize>,
    ) -> Result<DeviceOutput, EngineError> {
        let elements = if shape.contains(&0) {
            0
        } else {
            shape
                .iter()
                .try_fold(1usize, |n, &dim| n.checked_mul(dim))
                .ok_or_else(|| EngineError::Device("device output shape overflows".into()))?
        };
        let bytes = elements
            .checked_mul(dtype.width())
            .filter(|&n| n <= isize::MAX as usize)
            .ok_or_else(|| {
                EngineError::Device("device output bytes exceed platform limits".into())
            })?;
        let storage = guard(|| {
            // SAFETY: the producer initializes every logical byte before finish.
            unsafe { self.execution.alloc::<u8>(bytes.max(1)) }
                .map_err(|e| failed(&self.device, "allocating a device output", e))
        })?;
        Ok(DeviceOutput {
            storage: RetainedStorage::new(&self.device, Arc::clone(&self.stream), storage),
            stream: Arc::clone(&self.stream),
            device: self.device.clone(),
            dtype,
            shape,
        })
    }
    /// Upload host-origin bytes once. Device-origin payloads never use this
    /// path; it is for caller-owned host tables and small immutable call packs.
    pub fn upload_bytes(
        &self,
        dtype: DeviceDType,
        shape: Vec<usize>,
        bytes: &[u8],
    ) -> Result<DeviceColumn, EngineError> {
        guard(|| {
            let mut output = self.output(dtype, shape)?;
            if output.nbytes() != bytes.len() {
                return Err(EngineError::Parameter(
                    "host upload byte count disagrees with device shape/dtype".into(),
                ));
            }
            if !bytes.is_empty() {
                // The staging copy is reserved before the copy is enqueued
                // (host-allocation-failures §12.6 rule 4).
                let staging = super::execution_stream::upload_staging::<u8>(bytes.len())?;
                self.execution
                    .memcpy_htod_staged(
                        bytes,
                        &mut output.storage.get_mut().slice_mut(0..bytes.len()),
                        staging,
                    )
                    .map_err(|e| failed(&self.device, "uploading host-origin device bytes", e))?;
            }
            // SAFETY: every logical byte was initialized by the same-stream copy.
            unsafe { output.finish() }
        })
    }
    pub fn execution(&self) -> &super::execution_stream::BorrowedExecutionStream {
        &self.execution
    }
    pub fn stream(&self) -> &Arc<CudaStream> {
        &self.stream
    }
    pub fn device(&self) -> &Device {
        &self.device
    }
    pub fn retain(&mut self, array: DeviceArray) -> Result<(), EngineError> {
        guard(|| {
            let own = array.descriptor.producer_stream
                == Some(self.stream.cu_stream() as usize as u64)
                && array.descriptor.ordinal == self.device.info().ordinal;
            if own {
                array.descriptor.validate()?;
            } else {
                validate_pointer(&self.device, &array.descriptor)?;
            }
            self.inputs
                .try_reserve(1)
                .map_err(|_| EngineError::Device("unable to retain a device input owner".into()))?;
            let producer = array.descriptor.producer_stream;
            // Retain before queuing anything: a failed producer handoff must
            // still leave the foreign owner under the lease retirement guard.
            self.inputs.push(array);
            if !own {
                self.execution.wait_for(producer)?;
            }
            Ok(())
        })
    }
    pub fn bind_table(
        &mut self,
        table: &DeviceSequenceTable,
    ) -> Result<BoundDeviceRows, EngineError> {
        for input in &table.inputs {
            self.retain(input.clone())?;
        }
        let descriptor = guard(|| {
            self.execution
                .clone_htod(&[table.rows])
                .map_err(|e| failed(&self.device, "binding the device row descriptor", e))
        })?;
        let mut owners = Vec::new();
        owners.try_reserve_exact(table.inputs.len()).map_err(|_| {
            EngineError::Device("unable to retain bound device table owners".into())
        })?;
        owners.extend(table.inputs.iter().cloned());
        Ok(BoundDeviceRows {
            descriptor: Arc::new(descriptor),
            stream: Arc::clone(&self.stream),
            device: self.device.clone(),
            rows: table.rows,
            owners,
            host_total: None,
            host_summary: None,
        })
    }
    /// Upload one host-origin table once, then consume it through exactly the
    /// resident row ABI. Mixed host/device calls use this same lease and stream.
    pub fn bind_host_table(
        &mut self,
        table: &seqblaze_core::seq::SeqTable,
        owner: &Arc<seqblaze_core::seq::SeqTable>,
        workers: seqblaze_core::sched::Workers,
    ) -> Result<BoundDeviceRows, EngineError> {
        guard(|| {
            if !table.same_input(owner) {
                return Err(EngineError::Device(
                    "host upload owner does not retain its call view".into(),
                ));
            }
            table.validate_lengths()?;
            let n = table.len();
            let (data, offsets, total) = if let Some(batch) = table.owned_batch() {
                // S13c: the batch is immutable and retained through a proving
                // wait, including errors/cancellation. Neither buffer needs
                // an offsets/byte/staging copy (Q6).
                (
                    HostUpload::Table {
                        address: batch.data().as_ptr() as usize,
                        bytes: batch.data().len(),
                    },
                    HostUpload::Table {
                        address: batch.offsets().as_ptr() as usize,
                        bytes: std::mem::size_of_val(batch.offsets()),
                    },
                    batch.data().len(),
                )
            } else if let Some((address, bytes)) = table
                .contiguous_fixed_span()
                .filter(|_| table.nulls().is_empty())
            {
                // S10(a): arithmetic offsets, no row staging or second trim.
                let width = bytes / n.max(1);
                let mut offsets = Vec::new();
                seqblaze_core::detail::try_reserve_exact(
                    &mut offsets,
                    n + 1,
                    "host table offsets",
                )?;
                offsets.extend((0..=n).map(|row| (row * width) as i64));
                (
                    HostUpload::Table { address, bytes },
                    HostUpload::Offsets(offsets),
                    bytes,
                )
            } else {
                let (data, offsets) = crate::staging::stage_view(
                    table,
                    workers,
                    crate::staging::PARALLEL_STAGING_BYTES,
                )?;
                let total = data.len();
                (HostUpload::Data(data), HostUpload::Offsets(offsets), total)
            };
            let host_total = table.nulls().is_empty().then_some(total as u64);
            // Reserve every table buffer before the first copy (§12.6 rule 8).
            let mut mask = Vec::new();
            if !table.nulls().is_empty() {
                seqblaze_core::detail::try_reserve_exact(
                    &mut mask,
                    n.div_ceil(8),
                    "host table null bitmap",
                )?;
            }
            let data = self
                .upload_host_buffer(DeviceDType::U8, total, data, owner)?
                .internal_array();
            let offsets = self
                .upload_host_buffer(DeviceDType::I64, n + 1, offsets, owner)?
                .internal_array();
            let validity = if table.nulls().is_empty() {
                None
            } else {
                let bytes = n.div_ceil(8);
                mask.resize(bytes, 255u8);
                for &row in table.nulls() {
                    mask[row / 8] &= !(1u8 << (row % 8));
                }
                Some(
                    self.upload_bytes(DeviceDType::U8, vec![bytes], &mask)?
                        .internal_array(),
                )
            };
            let rows =
                super::DeviceSequenceTable::packed(data, offsets, validity, n, 0, table.is_text())?;
            let mut bound = self.bind_table(&rows)?;
            bound.host_total = host_total;
            Ok(bound)
        })
    }
    pub fn stream_handle(&self) -> u64 {
        self.execution.handle()
    }
    pub fn synchronize(&self) -> Result<(), EngineError> {
        guard(|| {
            let mark = self.execution.pending_mark();
            let completed = crate::interop::execution_stream::wait_owned_stream(&self.stream);
            // The owned stream carries every upload's completion dependency, so
            // a successful wait releases the pending host staging (S5 (b)).
            self.execution.settle_from(mark, completed.is_ok());
            completed.map_err(|e| failed(&self.device, "device input synchronization", e))
        })
    }
}

impl Drop for DeviceLease {
    fn drop(&mut self) {
        // Raw-pointer launches cannot use cudarc's borrow-based event tracking.
        // Keep a producer's Python/RMM/DLPack owner alive through all launches,
        // and pending host staging alive until its copies are proved (S5 (b)).
        if self.inputs.is_empty() && !self.execution.has_pending() {
            return;
        }
        let proved = guard(|| {
            crate::interop::execution_stream::retire_in_drop(&self.stream)
                .map_err(|e| failed(&self.device, "retiring device input owners", e))
        })
        .is_ok();
        self.execution.settle(proved);
        if !proved {
            // A driver failure must not turn into a use-after-free. A leaked
            // owner on this exceptional path is preferable to freeing live DMA.
            for array in self.inputs.drain(..) {
                std::mem::forget(array.owner);
            }
        }
    }
}

/// Raw-pointer launches do not register cudarc's typed borrow events. This
/// allocation owner brackets retirement even when construction/publication
/// fails: sync failure keeps the allocation live rather than freeing DMA data.
struct RetainedStorage {
    slice: Option<CudaSlice<u8>>,
    stream: Arc<CudaStream>,
    device: Device,
}
impl RetainedStorage {
    fn new(device: &Device, stream: Arc<CudaStream>, slice: CudaSlice<u8>) -> Self {
        Self {
            slice: Some(slice),
            stream,
            device: device.clone(),
        }
    }
    fn get_mut(&mut self) -> &mut CudaSlice<u8> {
        self.slice.as_mut().expect("live device construction owner")
    }
}
impl std::ops::Deref for RetainedStorage {
    type Target = CudaSlice<u8>;
    fn deref(&self) -> &Self::Target {
        self.slice.as_ref().expect("live device result owner")
    }
}
impl Drop for RetainedStorage {
    fn drop(&mut self) {
        if self.slice.is_none() {
            return;
        }
        if guard(|| {
            crate::interop::execution_stream::retire_in_drop(&self.stream)
                .map_err(|e| failed(&self.device, "retiring a device result allocation", e))
        })
        .is_err()
        {
            if let Some(slice) = self.slice.take() {
                std::mem::forget(slice);
            }
        }
    }
}

/// An immutable, fully initialized device result allocation. Python exposes a
/// read-only proxy, never this allocation's writable CAI/DLPack borrow.
#[derive(Clone)]
pub struct DeviceColumn {
    storage: Arc<RetainedStorage>,
    stream: Arc<CudaStream>,
    device: Device,
    dtype: DeviceDType,
    shape: Vec<usize>,
    bytes: usize,
}

impl DeviceColumn {
    /// # Safety
    /// All bytes of the logical column must be initialized by work queued on
    /// `stream` before the column is published. It must never be written again.
    pub(crate) unsafe fn publish(
        device: &Device,
        stream: Arc<CudaStream>,
        storage: CudaSlice<u8>,
        dtype: DeviceDType,
        shape: Vec<usize>,
    ) -> Result<Self, EngineError> {
        let storage = RetainedStorage::new(device, Arc::clone(&stream), storage);
        if storage.stream().cu_stream() != stream.cu_stream()
            || storage.context().cu_ctx() != device.0.ctx.cu_ctx()
        {
            // A raw launch may have used the promised consumer stream even on
            // an invalid publication attempt. Retire that work before dropping
            // storage on its actual allocation stream.
            if let Err(error) = stream.synchronize() {
                std::mem::forget(storage);
                return Err(failed(
                    device,
                    "retiring a mismatched device result allocation",
                    error,
                ));
            }
            return Err(EngineError::Parameter(
                "device result storage must be allocated on its publishing context and stream"
                    .into(),
            ));
        }
        let elements = if shape.contains(&0) {
            0
        } else {
            shape
                .iter()
                .try_fold(1usize, |n, &dim| n.checked_mul(dim))
                .ok_or_else(|| EngineError::Parameter("device output shape overflows".into()))?
        };
        let bytes = elements
            .checked_mul(dtype.width())
            .filter(|&n| n <= isize::MAX as usize)
            .ok_or_else(|| {
                EngineError::Parameter("device output byte length exceeds platform limits".into())
            })?;
        if storage.len() < bytes {
            return Err(EngineError::Parameter(
                "device output allocation is shorter than its logical shape".into(),
            ));
        }
        Ok(Self {
            storage: Arc::new(storage),
            stream,
            device: device.clone(),
            dtype,
            shape,
            bytes,
        })
    }
    /// An independent D2D tensor for a consumer that may mutate its storage.
    /// The consumer's requested stream is validated before payload allocation;
    /// completed copying makes the returned buffer immediately visible to it.
    pub fn copy_for_export(
        &self,
        shape: Vec<usize>,
        consumer_stream: Option<u64>,
    ) -> Result<Self, EngineError> {
        guard(|| {
            let bytes = if shape.contains(&0) {
                0
            } else {
                shape
                    .iter()
                    .try_fold(self.dtype.width(), |n, &dim| n.checked_mul(dim))
                    .filter(|&n| n <= isize::MAX as usize)
                    .ok_or_else(|| EngineError::Parameter("DLPack export shape overflows".into()))?
            };
            if bytes != self.bytes {
                return Err(EngineError::Parameter(
                    "DLPack export shape changes the column size".into(),
                ));
            }
            let consumer = external_stream(&self.device, consumer_stream.unwrap_or(1))?;
            let lease = DeviceLease::new(&self.device)?;
            let output = lease.output(self.dtype, shape)?;
            // A copying DLPack exporter performs its actual copy on the
            // requested consumer stream. These dependencies establish both
            // source readiness and allocation readiness there without a borrow.
            order_streams(&self.device, self.stream.cu_stream(), consumer)?;
            order_streams(&self.device, lease.stream().cu_stream(), consumer)?;
            if self.bytes != 0 {
                // SAFETY: independent same-context buffers cover this complete
                // copy; source/target owners are live until consumer completion.
                let copied = unsafe {
                    sys::cuMemcpyDtoDAsync_v2(
                        output.address(),
                        self.address(),
                        self.bytes,
                        consumer,
                    )
                }
                .result();
                let completed =
                    copied.and_then(|()| unsafe { sys::cuStreamSynchronize(consumer) }.result());
                if let Err(error) = completed {
                    // Neither allocation stream can prove a foreign operation
                    // stopped after a failed sync; retain both DMA allocations.
                    std::mem::forget(output);
                    std::mem::forget(self.clone());
                    return Err(failed(
                        &self.device,
                        "copying an independent DLPack export",
                        error,
                    ));
                }
            }
            // SAFETY: the sole consumer-stream D2D copy completed, initializing
            // every logical byte. Its allocation still uses the owner's stream.
            unsafe { output.finish() }
        })
    }
    /// # Safety
    /// The caller must keep this owner alive while the address is used. A
    /// writable external tensor must refer only to an independent export copy,
    /// never the immutable Result's allocation. This is a Rust bridge, not a
    /// Python borrowing protocol.
    pub unsafe fn export_address(&self) -> u64 {
        self.address()
    }
    pub(super) fn export_context(&self) -> &Arc<CudaContext> {
        self.storage.context()
    }
    pub(super) fn export_device(&self) -> Device {
        self.device.clone()
    }
    /// Worker-only retirement of a uniquely owned export copy. Foreign writes
    /// have no cudarc borrow events, so only the allocation's exact context
    /// proves completion, including work in the original caller's PTDS.
    pub fn retire_export(&mut self) -> Result<(), EngineError> {
        guard(|| {
            self.export_context()
                .bind_to_thread()
                .map_err(|e| failed(&self.device, "binding DLPack export retirement", e))?;
            // SAFETY: bind the retained allocation context, never a borrowed
            // consumer stream. No allocation can be dropped before success.
            unsafe { sys::cuCtxSynchronize() }
                .result()
                .map_err(|e| failed(&self.device, "retiring a DLPack export context", e))?;
            let storage = Arc::get_mut(&mut self.storage).ok_or_else(|| {
                EngineError::Device("DLPack export retirement requires unique storage".into())
            })?;
            // Bypass RetainedStorage's fallible, leak-on-error destructor: a
            // worker must distinguish a real release from intentional retention.
            // A panic from cudarc's free is caught with the column/context still
            // protected by the retirement job's ManuallyDrop.
            drop(storage.slice.take());
            crate::interop::execution_stream::retire_owned_stream(&self.stream)
                .map_err(|e| failed(&self.device, "finishing DLPack export release", e))
        })
    }

    pub(crate) fn internal_array(&self) -> DeviceArray {
        let mut strides = vec![0i64; self.shape.len()];
        let mut stride = self.dtype.width() as i64;
        for axis in (0..self.shape.len()).rev() {
            strides[axis] = stride;
            stride = stride.saturating_mul(self.shape[axis] as i64);
        }
        // SAFETY: this immutable internal owner retains the allocation and its
        // producer stream. It is never exported as a writable public protocol.
        unsafe {
            DeviceArray::from_external(
                DeviceArrayDescriptor {
                    address: self.address(),
                    shape: self.shape.clone(),
                    strides,
                    dtype: self.dtype,
                    readonly: true,
                    ordinal: self.ordinal(),
                    producer_stream: Some(self.stream.cu_stream() as usize as u64),
                },
                Arc::new(self.clone()),
            )
        }
        .expect("published device column has valid metadata")
    }
    pub(crate) fn address(&self) -> u64 {
        ptr(&self.storage, &self.stream)
    }
    pub fn dtype(&self) -> DeviceDType {
        self.dtype
    }
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }
    pub fn ordinal(&self) -> usize {
        self.device.info().ordinal
    }
    pub fn nbytes(&self) -> usize {
        self.bytes
    }
    pub fn synchronize(&self) -> Result<(), EngineError> {
        guard(|| {
            crate::interop::execution_stream::wait_owned_stream(&self.stream)
                .map_err(|e| failed(&self.device, "device result synchronization", e))
        })
    }

    /// Write an independent contiguous device allocation supplied by an export
    /// consumer. Both producer work and source copies finish before return, so
    /// no mutable alias to the immutable result is exported.
    pub fn copy_into(&self, target: &DeviceArray) -> Result<(), EngineError> {
        let mut lease = DeviceLease {
            device: self.device.clone(),
            stream: Arc::clone(&self.stream),
            execution: super::execution_stream::BorrowedExecutionStream::new(
                &self.device,
                Arc::clone(&self.stream),
                self.stream.cu_stream() as usize as u64,
            )?,
            inputs: Vec::new(),
        };
        lease.retain(target.clone())?;
        guard(|| {
            validate_pointer(&self.device, &target.descriptor)?;
            let layout = &target.descriptor;
            if layout.readonly
                || layout.dtype != self.dtype
                || layout.shape != self.shape
                || !layout.is_contiguous()
            {
                return Err(EngineError::Parameter("device export requires a writable contiguous target with the original shape and dtype".into()));
            }
            wait_for_stream(&self.device, &self.stream, layout.producer_stream)?;
            if self.bytes != 0 {
                // SAFETY: validated live non-overlapping allocations. The output
                // allocator cannot borrow this column through a public protocol.
                let source = ptr(&self.storage, &self.stream);
                let end = layout
                    .address
                    .checked_add(self.bytes as u64)
                    .ok_or_else(|| {
                        EngineError::Parameter("device export address overflows".into())
                    })?;
                let source_end = source + self.bytes as u64;
                if layout.address < source_end && source < end {
                    return Err(EngineError::Parameter(
                        "device export target aliases immutable result storage".into(),
                    ));
                }
                unsafe {
                    sys::cuMemcpyDtoDAsync_v2(
                        layout.address,
                        source,
                        self.bytes,
                        self.stream.cu_stream(),
                    )
                }
                .result()
                .map_err(|e| failed(&self.device, "copying a device result", e))?;
            }
            crate::interop::execution_stream::retire_owned_stream(&self.stream)
                .map_err(|e| failed(&self.device, "finishing a device result export", e))
        })
    }

    /// The batched host export behind `Result.to_numpy()` (per-call cost S8).
    /// When every column shares one owned stream (one Result's columns), each
    /// column's host vector is reserved first with its bytes (B9, label
    /// `device-to-host export`), then all copies go into the ordinal's pinned
    /// bounce buffer when they fit it, with one wait and the copy-out into the
    /// vectors; otherwise column by column into the vectors' capacity, each
    /// with its wait. Columns of several streams take [`Self::to_host_bytes`]
    /// each. No zero fill; refusals and driver failures keep today's messages.
    pub fn to_host_bytes_batch(columns: &[&DeviceColumn]) -> Result<Vec<Vec<u8>>, EngineError> {
        guard(|| {
            let Some(first) = columns.first() else {
                return Ok(Vec::new());
            };
            let shared = columns.iter().all(|column| {
                Arc::ptr_eq(&column.stream, &first.stream)
                    && column.device.0.ctx.cu_ctx() == first.device.0.ctx.cu_ctx()
            });
            let mut out = Vec::with_capacity(columns.len());
            if !shared {
                for column in columns {
                    out.push(column.to_host_bytes()?);
                }
                return Ok(out);
            }
            for column in columns {
                let mut bytes = Vec::new();
                seqblaze_core::detail::try_reserve_exact(
                    &mut bytes,
                    column.bytes,
                    "device-to-host export",
                )?;
                out.push(bytes);
            }
            let total = columns
                .iter()
                .try_fold(0usize, |sum, column| sum.checked_add(column.bytes))
                .ok_or_else(|| {
                    EngineError::Device("device-to-host export size overflows".into())
                })?;
            let device = &first.device;
            device
                .0
                .ctx
                .bind_to_thread()
                .map_err(|e| failed(device, "explicit device-to-host export", e))?;
            if total == 0 {
                // Nothing to copy: the one completion wait of the export.
                crate::interop::execution_stream::retire_owned_stream(&first.stream)
                    .map_err(|e| failed(device, "finishing explicit host export", e))?;
                return Ok(out);
            }
            let bounce = if total <= crate::pool::BOUNCE_BYTES {
                device.0.bounce.checkout()
            } else {
                None
            };
            match bounce {
                Some(buffer) => Self::export_bounced(first, columns, &mut out, buffer)?,
                None => {
                    for (column, bytes) in columns.iter().zip(&mut out) {
                        column.export_into(bytes)?;
                    }
                }
            }
            Ok(out)
        })
    }
    /// All copies into the checked-out bounce buffer, one wait, the copy-out.
    fn export_bounced(
        first: &DeviceColumn,
        columns: &[&DeviceColumn],
        out: &mut [Vec<u8>],
        buffer: crate::pool::Pinned,
    ) -> Result<(), EngineError> {
        let device = &first.device;
        let stream = first.stream.cu_stream();
        let mut offset = 0usize;
        let mut copied = Ok(());
        for column in columns {
            if column.bytes == 0 {
                continue;
            }
            // SAFETY: the checked-out buffer holds BOUNCE_BYTES >= the total,
            // and the live column covers `bytes` initialized bytes on `stream`.
            copied = unsafe {
                sys::cuMemcpyDtoHAsync_v2(
                    buffer.as_ptr().add(offset).cast(),
                    column.address(),
                    column.bytes,
                    stream,
                )
            }
            .result();
            if copied.is_err() {
                break;
            }
            offset += column.bytes;
        }
        // One wait for every enqueued copy, also after a failed enqueue: none
        // may still target the buffer when it returns to the slot.
        let waited = crate::interop::execution_stream::retire_owned_stream(&first.stream);
        if let Err(error) = waited {
            device.0.bounce.quarantine(buffer);
            return Err(match copied {
                Err(copy) => failed(device, "explicit device-to-host export", copy),
                Ok(()) => failed(device, "finishing explicit host export", error),
            });
        }
        if let Err(copy) = copied {
            device.0.bounce.give_back(buffer);
            return Err(failed(device, "explicit device-to-host export", copy));
        }
        let mut offset = 0usize;
        for (column, bytes) in columns.iter().zip(out.iter_mut()) {
            // SAFETY: the wait proved every copy complete; each vector has
            // capacity for its column's bytes, which the copy fully initializes.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    buffer.as_ptr().add(offset),
                    bytes.as_mut_ptr(),
                    column.bytes,
                );
                bytes.set_len(column.bytes);
            }
            offset += column.bytes;
        }
        device.0.bounce.give_back(buffer);
        Ok(())
    }
    /// One column copied into `bytes`' reserved capacity, then its wait (the
    /// per-column path of [`Self::to_host_bytes_batch`]); no zero fill.
    fn export_into(&self, bytes: &mut Vec<u8>) -> Result<(), EngineError> {
        debug_assert!(bytes.capacity() >= self.bytes);
        if self.bytes != 0 {
            // SAFETY: the reserved capacity covers the column's bytes; the
            // live column covers them on its own stream.
            unsafe {
                sys::cuMemcpyDtoHAsync_v2(
                    bytes.as_mut_ptr().cast(),
                    self.address(),
                    self.bytes,
                    self.stream.cu_stream(),
                )
            }
            .result()
            .map_err(|e| failed(&self.device, "explicit device-to-host export", e))?;
        }
        if let Err(error) = crate::interop::execution_stream::retire_owned_stream(&self.stream) {
            // The copy may still target the vector: leak it (I8).
            std::mem::forget(std::mem::take(bytes));
            return Err(failed(
                &self.device,
                "finishing explicit host export",
                error,
            ));
        }
        // SAFETY: the completed copy initialized exactly `self.bytes` bytes.
        unsafe { bytes.set_len(self.bytes) };
        Ok(())
    }

    /// Explicit host export only; regular kernel output construction never
    /// calls this method. Bit patterns are transported without FP arithmetic.
    pub fn to_host_bytes(&self) -> Result<Vec<u8>, EngineError> {
        guard(|| {
            let mut bytes = Vec::new();
            seqblaze_core::detail::try_reserve_exact(
                &mut bytes,
                self.bytes,
                "device-to-host export",
            )?;
            bytes.resize(self.bytes, 0u8);
            crate::session::dtoh(&self.stream, &self.storage.slice(0..self.bytes), &mut bytes)
                .map_err(|e| failed(&self.device, "explicit device-to-host export", e))?;
            crate::interop::execution_stream::retire_owned_stream(&self.stream)
                .map_err(|e| failed(&self.device, "finishing explicit host export", e))?;
            Ok(bytes)
        })
    }
}

/// The row descriptor uploaded as control metadata. All underlying allocations
/// remain owned by its DeviceLease until the consumer finishes.
pub struct BoundDeviceRows {
    descriptor: Arc<CudaSlice<DeviceRows>>,
    stream: Arc<CudaStream>,
    pub(crate) rows: DeviceRows,
    device: Device,
    owners: Vec<DeviceArray>,
    /// The byte total of a host-origin table without null rows (S6).
    pub(crate) host_total: Option<u64>,
    /// Facts supplied only after whole-call host preflight (S13a).
    pub(crate) host_summary: Option<super::DeviceInputSummary>,
}
impl BoundDeviceRows {
    /// Trust the current call's host preflight instead of revalidating its
    /// uploaded bytes. Device/mixed calls must not use this shortcut.
    pub fn with_host_summary(mut self, summary: super::DeviceInputSummary) -> Self {
        assert_eq!(summary.rows, self.rows.rows);
        self.host_summary = Some(summary);
        self
    }
    pub(crate) fn ensure_lease(&self, lease: &DeviceLease) -> Result<(), EngineError> {
        if self.stream.cu_stream() != lease.stream().cu_stream()
            || self.device.0.ctx.cu_ctx() != lease.device().0.ctx.cu_ctx()
        {
            return Err(EngineError::Device(
                "bound device rows belong to a different consumer stream/context".into(),
            ));
        }
        Ok(())
    }
    pub fn address(&self) -> u64 {
        ptr(&self.descriptor, &self.stream)
    }
    /// The byte total of a host-origin table without null rows, known before
    /// any device read (per-call cost S6); None for device-origin tables.
    pub fn host_total(&self) -> Option<u64> {
        self.host_total
    }
}

impl Drop for BoundDeviceRows {
    fn drop(&mut self) {
        // Bound rows independently retain their input allocations. A caller
        // dropping its lease early cannot invalidate a still-live row token.
        if self.owners.is_empty() {
            return;
        }
        if guard(|| {
            crate::interop::execution_stream::retire_in_drop(&self.stream)
                .map_err(|e| failed(&self.device, "retiring bound device table owners", e))
        })
        .is_err()
        {
            // The descriptor itself is also a kernel input, not merely the
            // allocations named by its addresses.
            std::mem::forget(Arc::clone(&self.descriptor));
            for array in self.owners.drain(..) {
                std::mem::forget(array.owner);
            }
        }
    }
}

/// Mutable construction-only output. Its address is never exposed to Python;
/// finishing moves it into the immutable result owner without any data copy.
pub(crate) struct DeviceOutput {
    storage: RetainedStorage,
    stream: Arc<CudaStream>,
    device: Device,
    dtype: DeviceDType,
    shape: Vec<usize>,
}
impl DeviceOutput {
    pub(crate) fn address(&self) -> u64 {
        ptr(&self.storage, &self.stream)
    }
    pub(crate) fn nbytes(&self) -> usize {
        if self.shape.contains(&0) {
            0
        } else {
            self.shape.iter().product::<usize>() * self.dtype.width()
        }
    }
    /// # Safety
    /// Every logical byte must have been written by work on this exact stream;
    /// no caller may write this allocation again after the returned publication.
    pub(crate) unsafe fn finish(mut self) -> Result<DeviceColumn, EngineError> {
        let storage = self.storage.slice.take().expect("live construction output");
        // SAFETY: forwarded to the caller's initialization/immutability contract.
        unsafe { DeviceColumn::publish(&self.device, self.stream, storage, self.dtype, self.shape) }
    }
}
