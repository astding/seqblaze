//! Device-only Result normalization, bit-copy gather and TopK row metadata.
//! Only bounds/capacity controls and the normalized-index Merkle digest return
//! to the host. Values, offsets, validity and complete indices stay resident.
use crate::{
    device::guard,
    interop::device_column_index_digest,
    interop::{DeviceArray, DeviceColumn, DeviceDType, DeviceLease, DeviceOutput},
    resident::{ResidentField, ResidentRaggedField, ResidentRaggedKind},
    resident_indices::canonical_indices,
    resident_sinks::scan_u64,
    session::{failure_ordinal, ptr},
};
use cudarc::driver::{
    CudaFunction, CudaSlice, CudaStream, DeviceRepr, DriverError, LaunchConfig, PushKernelArg,
};
use seqblaze_core::validate::EngineError;
use std::{mem::size_of, sync::Arc};

#[derive(Debug)]
pub enum SelectionError {
    Bounds { position: u64, rows: usize },
    Engine(EngineError),
}
impl From<EngineError> for SelectionError {
    fn from(error: EngineError) -> Self {
        Self::Engine(error)
    }
}
pub struct NormalizedSelection {
    indices: DeviceColumn,
    parent_rows: usize,
    masked: bool,
    sha256: [u8; 32],
}
impl NormalizedSelection {
    pub fn indices(&self) -> &DeviceColumn {
        &self.indices
    }
    pub fn parent_rows(&self) -> usize {
        self.parent_rows
    }
    pub fn masked(&self) -> bool {
        self.masked
    }
    pub fn sha256(&self) -> &[u8; 32] {
        &self.sha256
    }
}
#[derive(Clone, Copy)]
pub struct TopKSelection {
    pub rows: usize,
    pub k: usize,
}
pub struct SelectionOutput {
    pub numeric: Vec<ResidentField>,
    pub ragged: Vec<ResidentRaggedField>,
    pub tied_rows: Option<u64>,
}
#[repr(C)]
#[derive(Clone, Copy)]
struct IndexView {
    address: u64,
    n: u64,
    stride: i64,
    width: u64,
    is_signed: u64,
}
// SAFETY: five repr(C) eight-byte integer fields matching SbIndexView.
unsafe impl DeviceRepr for IndexView {}
const _: () = {
    assert!(size_of::<IndexView>() == 40);
};
fn parameter(message: &str) -> EngineError {
    EngineError::Parameter(message.into())
}
fn overflow(message: &str) -> EngineError {
    EngineError::Overflow(message.into())
}
fn fail(lease: &DeviceLease, operation: &str, e: DriverError) -> EngineError {
    failure_ordinal(lease.device(), operation, e)
}
struct Buffer<T: DeviceRepr> {
    data: Option<crate::arena::Scratch<T>>,
    stream: Arc<CudaStream>,
}
impl<T: DeviceRepr> Buffer<T> {
    fn allocate(lease: &DeviceLease, n: usize) -> Result<Self, EngineError> {
        n.max(1)
            .checked_mul(size_of::<T>())
            .filter(|&n| n <= isize::MAX as usize)
            .ok_or_else(|| overflow("device selection workspace exceeds platform limits"))?;
        // SAFETY: every logical element is initialized by copies or same-stream
        // kernels before use; zero-length logical spans are never accessed.
        let data = unsafe { lease.execution().scratch(n.max(1)) }
            .map_err(|e| fail(lease, "allocating device selection workspace", e))?;
        Ok(Self::adopt(lease, data))
    }
    fn adopt(lease: &DeviceLease, data: crate::arena::Scratch<T>) -> Self {
        Self {
            data: Some(data),
            stream: Arc::clone(lease.stream()),
        }
    }
    fn data(&self) -> &CudaSlice<T> {
        self.data.as_ref().expect("live selection workspace")
    }
    fn address(&self) -> u64 {
        ptr(self.data(), &self.stream)
    }
}
impl Buffer<u64> {
    fn zeros(lease: &DeviceLease, n: usize) -> Result<Self, EngineError> {
        let output = Self::allocate(lease, n)?;
        // SAFETY: this private allocation covers n.max(1) u64 values. Queue
        // initialization on the execution stream before its counting kernels;
        // allocation-stream writes have no dependency back to that stream.
        unsafe {
            lease
                .execution()
                .memset(output.address(), 0, n.max(1) * size_of::<u64>())
        }
        .map_err(|e| fail(lease, "initializing device selection workspace", e))?;
        Ok(output)
    }
    fn status(lease: &DeviceLease, n: usize) -> Result<Self, EngineError> {
        let mut output = Self::zeros(lease, n)?;
        lease
            .execution()
            .fill(
                &mut output
                    .data
                    .as_mut()
                    .expect("live selection workspace")
                    .slice_mut(0..1),
                1,
                0xFF,
            )
            .map_err(|e| fail(lease, "initializing device selection bounds control", e))?;
        Ok(output)
    }
}
impl<T: DeviceRepr> Drop for Buffer<T> {
    fn drop(&mut self) {
        if self.data.is_some()
            && guard(|| {
                crate::interop::execution_stream::retire_in_drop(&self.stream)
                    .map_err(|_| EngineError::Device(String::new()))
            })
            .is_err()
        {
            if let Some(data) = self.data.take() {
                std::mem::forget(data);
            }
        }
    }
}
fn control<const N: usize>(
    lease: &DeviceLease,
    input: &Buffer<u64>,
) -> Result<[u64; N], EngineError> {
    let mut output = [0u64; N];
    lease
        .execution()
        .memcpy_dtoh(&input.data().slice(0..N), &mut output[..])
        .map_err(|e| fail(lease, "reading device selection control scalars", e))?;
    lease.synchronize()?;
    Ok(output)
}
fn last(lease: &DeviceLease, input: &Buffer<u64>, n: usize) -> Result<u64, EngineError> {
    let mut output = [0u64];
    lease
        .execution()
        .memcpy_dtoh(&input.data().slice(n..n + 1), &mut output[..])
        .map_err(|e| fail(lease, "reading device selection allocation total", e))?;
    lease.synchronize()?;
    Ok(output[0])
}
fn threads(lease: &DeviceLease, function: &CudaFunction) -> Result<u32, EngineError> {
    let max = function
        .max_threads_per_block()
        .map_err(|e| fail(lease, "querying device selection launch limit", e))?;
    let threads = max.clamp(0, 256) as u32 / 32 * 32;
    if threads == 0 {
        return Err(EngineError::Device(
            "device selection cannot launch a full warp".into(),
        ));
    }
    Ok(threads)
}
fn grid(threads: u32, n: usize) -> LaunchConfig {
    LaunchConfig {
        grid_dim: (n.div_ceil(threads as usize).clamp(1, 65535) as u32, 1, 1),
        block_dim: (threads, 1, 1),
        shared_mem_bytes: 0,
    }
}
fn publish(output: DeviceOutput) -> Result<DeviceColumn, EngineError> {
    // SAFETY: logical bytes are fully initialized by same-stream work and this
    // construction-only writable owner is relinquished here.
    unsafe { output.finish() }
}
fn copy(lease: &DeviceLease, output: &DeviceOutput, source: u64) -> Result<(), EngineError> {
    if output.nbytes() == 0 {
        return Ok(());
    }
    // SAFETY: the prefix and output allocations cover the exact byte count,
    // are disjoint, retained and ordered on this same consumer stream.
    unsafe {
        lease
            .execution()
            .copy_device(output.address(), source, output.nbytes())
    }
    .map_err(|e| fail(lease, "copying device selection offsets", e))
}
fn retain(
    lease: &mut DeviceLease,
    column: &DeviceColumn,
    dtype: Option<DeviceDType>,
    shape: &[usize],
) -> Result<(), EngineError> {
    if column.shape() != shape
        || dtype.is_some_and(|kind| kind != column.dtype())
        || column.ordinal() != lease.device().info().ordinal
    {
        return Err(parameter(
            "device selection source dtype, shape or device is inconsistent",
        ));
    }
    lease.retain(column.internal_array())
}

/// Negative indices use Python indexing. Bounds are a separate typed error so
/// the Python adapter preserves IndexError, including unsigned overflow values.
pub fn normalize_selector(
    lease: &mut DeviceLease,
    input: DeviceArray,
    rows: usize,
    allow_mask: bool,
    canonical: bool,
) -> Result<NormalizedSelection, SelectionError> {
    guard(|| Ok(normalize_inner(lease, input, rows, allow_mask, canonical)))
        .map_err(SelectionError::Engine)?
}
fn normalize_inner(
    lease: &mut DeviceLease,
    input: DeviceArray,
    rows: usize,
    allow_mask: bool,
    canonical: bool,
) -> Result<NormalizedSelection, SelectionError> {
    let descriptor = input.descriptor().clone();
    descriptor.validate()?;
    if rows > isize::MAX as usize || descriptor.shape.len() != 1 {
        return Err(parameter("selection requires a 1-D integer array or bool mask").into());
    }
    let masked = descriptor.dtype == DeviceDType::Bool;
    let signed = matches!(
        descriptor.dtype,
        DeviceDType::I8 | DeviceDType::I16 | DeviceDType::I32 | DeviceDType::I64
    );
    let integer = signed
        || matches!(
            descriptor.dtype,
            DeviceDType::U8 | DeviceDType::U16 | DeviceDType::U32 | DeviceDType::U64
        );
    if (masked && !allow_mask) || (!masked && !integer) {
        return Err(
            parameter("take requires 1-D integer indices; use r[mask] for bool masks").into(),
        );
    }
    let n = descriptor.shape[0];
    if masked && n != rows {
        return Err(parameter("selection mask length must match the Result entry count").into());
    }
    lease.retain(input)?;
    let view = IndexView {
        address: descriptor.address,
        n: n as u64,
        stride: descriptor.strides[0],
        width: descriptor.dtype.width() as u64,
        is_signed: u64::from(signed),
    };
    let kernels = &lease.device().0.kernels;
    let indices = if masked {
        let counts = Buffer::<u64>::allocate(lease, n)?;
        let count_address = counts.address();
        let thread_count = threads(lease, &kernels.selection_mask_counts)?;
        if n != 0 {
            let mut launch = lease
                .execution()
                .launch_builder(&kernels.selection_mask_counts);
            launch.arg(&view).arg(&count_address);
            // SAFETY: the retained validated descriptor spans n bool values;
            // this kernel initializes exactly n integer prefix counts.
            unsafe { launch.launch(grid(thread_count, n)) }
                .map_err(|e| fail(lease, "counting device selection mask", e))?;
        }
        let prefix = Buffer::adopt(lease, scan_u64(lease, counts.data(), n)?);
        let selected = usize::try_from(last(lease, &prefix, n)?)
            .map_err(|_| overflow("device selection mask count exceeds platform limits"))?;
        let output = lease.output(DeviceDType::I64, vec![selected])?;
        if n != 0 {
            let prefix_address = prefix.address();
            let output_address = output.address();
            let threads = threads(lease, &kernels.selection_mask_positions)?;
            let mut launch = lease
                .execution()
                .launch_builder(&kernels.selection_mask_positions);
            let n64 = n as u64;
            launch
                .arg(&count_address)
                .arg(&n64)
                .arg(&prefix_address)
                .arg(&output_address);
            // SAFETY: immutable copied bool counts give unique increasing
            // positions, and the checked final total is the exact allocation.
            // The mutable foreign mask is not reread after the count snapshot.
            unsafe { launch.launch(grid(threads, n)) }
                .map_err(|e| fail(lease, "normalizing device selection mask", e))?;
        }
        publish(output)?
    } else {
        let status = Buffer::<u64>::status(lease, 1)?;
        let output = lease.output(DeviceDType::I64, vec![n])?;
        if n != 0 {
            let rows64 = rows as u64;
            let output_address = output.address();
            let status_address = status.address();
            let threads = threads(lease, &kernels.selection_normalize)?;
            let mut launch = lease
                .execution()
                .launch_builder(&kernels.selection_normalize);
            launch
                .arg(&view)
                .arg(&rows64)
                .arg(&output_address)
                .arg(&status_address);
            // SAFETY: raw byte reads honor the checked signed-stride span;
            // every output position is initialized and only bounds controls
            // return to the host before publication.
            unsafe { launch.launch(grid(threads, n)) }
                .map_err(|e| fail(lease, "normalizing device selection indices", e))?;
        }
        let status = control::<1>(lease, &status)?;
        if status[0] != u64::MAX {
            return Err(SelectionError::Bounds {
                position: status[0],
                rows,
            });
        }
        let output = publish(output)?;
        if canonical {
            canonical_indices(lease, &output)?
        } else {
            output
        }
    };
    let sha256 = device_column_index_digest(lease, &indices)?;
    Ok(NormalizedSelection {
        indices,
        parent_rows: rows,
        masked,
        sha256,
    })
}
pub fn normalize_owned(
    lease: &mut DeviceLease,
    input: &DeviceColumn,
    rows: usize,
    allow_mask: bool,
    canonical: bool,
) -> Result<NormalizedSelection, SelectionError> {
    normalize_selector(lease, input.internal_array(), rows, allow_mask, canonical)
}

/// A ReadOnlyDeviceColumn.reshape view changes only metadata. Flatten a full
/// contiguous immutable owner after proving the public element count matches
/// its actual allocation; no foreign readonly flag or claimed span is trusted.
pub fn normalize_owned_flat(
    lease: &mut DeviceLease,
    input: &DeviceColumn,
    elements: usize,
    rows: usize,
    allow_mask: bool,
    canonical: bool,
) -> Result<NormalizedSelection, SelectionError> {
    let width = input.dtype().width();
    if elements.checked_mul(width) != Some(input.nbytes()) {
        return Err(parameter("device selector view does not cover its immutable owner").into());
    }
    let original = input.internal_array();
    let mut descriptor = original.descriptor().clone();
    descriptor.shape = vec![elements];
    descriptor.strides = vec![width as i64];
    // SAFETY: the actual immutable DeviceColumn independently retains exactly
    // the checked contiguous byte allocation. Address, dtype, ordinal and
    // producer stream come from that owner; only full-span shape is changed.
    let view = unsafe { DeviceArray::from_external(descriptor, Arc::new(input.clone())) }?;
    normalize_selector(lease, view, rows, allow_mask, canonical)
}

fn numeric(
    lease: &mut DeviceLease,
    input: &DeviceColumn,
    selection: &NormalizedSelection,
) -> Result<DeviceColumn, EngineError> {
    retain(lease, input, None, &[selection.parent_rows])?;
    let n = selection.indices.shape()[0];
    let output = lease.output(input.dtype(), vec![n])?;
    if n != 0 {
        let source = input.address();
        let indices = selection.indices.address();
        let destination = output.address();
        let n64 = n as u64;
        let width = input.dtype().width() as u64;
        let function = &lease.device().0.kernels.selection_numeric;
        let threads = threads(lease, function)?;
        let mut launch = lease.execution().launch_builder(function);
        launch
            .arg(&source)
            .arg(&indices)
            .arg(&destination)
            .arg(&n64)
            .arg(&width);
        // SAFETY: normalization proved every index is within the immutable
        // source's checked span; per-entry byte copies preserve all f64 bits.
        unsafe { launch.launch(grid(threads, n)) }
            .map_err(|e| fail(lease, "gathering device numeric results", e))?;
    }
    publish(output)
}
fn ragged(
    lease: &mut DeviceLease,
    input: &ResidentRaggedField,
    selection: &NormalizedSelection,
) -> Result<ResidentRaggedField, EngineError> {
    let source_offsets = selection
        .parent_rows
        .checked_add(1)
        .ok_or_else(|| overflow("device ragged source offset length overflows"))?;
    retain(
        lease,
        &input.offsets,
        Some(DeviceDType::I64),
        &[source_offsets],
    )?;
    if input.values.shape().len() != 1 {
        return Err(parameter("device ragged values must be one-dimensional"));
    }
    let values_len = input.values.shape()[0];
    let dtype = if input.kind == ResidentRaggedKind::ListI64 {
        DeviceDType::I64
    } else {
        DeviceDType::U8
    };
    retain(lease, &input.values, Some(dtype), &[values_len])?;
    if let Some(valid) = &input.validity {
        retain(
            lease,
            valid,
            Some(DeviceDType::Bool),
            &[selection.parent_rows],
        )?;
    }
    let n = selection.indices.shape()[0];
    let counts = Buffer::<u64>::allocate(lease, n)?;
    let status = Buffer::<u64>::status(lease, 3)?;
    if n != 0 {
        let source = input.offsets.address();
        let values_len = values_len as u64;
        let indices = selection.indices.address();
        let n64 = n as u64;
        let count_address = counts.address();
        let status_address = status.address();
        let function = &lease.device().0.kernels.selection_ragged_lengths;
        let threads = threads(lease, function)?;
        let mut launch = lease.execution().launch_builder(function);
        launch
            .arg(&source)
            .arg(&values_len)
            .arg(&indices)
            .arg(&n64)
            .arg(&count_address)
            .arg(&status_address);
        // SAFETY: only normalized selected offset pairs are read. Each is
        // checked against the actual retained values span before any payload
        // read; the saturating total proves the later scan cannot wrap.
        unsafe { launch.launch(grid(threads, n)) }
            .map_err(|e| fail(lease, "planning device ragged selection", e))?;
    }
    let status = control::<3>(lease, &status)?;
    if status[0] != u64::MAX {
        return Err(parameter("ragged offsets lie outside their values buffer"));
    }
    if status[2] != 0 {
        return Err(parameter("selected ragged offsets exceed int64 range"));
    }
    let total = usize::try_from(status[1])
        .map_err(|_| overflow("selected ragged payload exceeds platform limits"))?;
    let prefix = Buffer::adopt(lease, scan_u64(lease, counts.data(), n)?);
    let offset_len = n
        .checked_add(1)
        .ok_or_else(|| overflow("selected ragged offset length overflows"))?;
    let offsets = lease.output(DeviceDType::I64, vec![offset_len])?;
    copy(lease, &offsets, prefix.address())?;
    let values = lease.output(dtype, vec![total])?;
    let valid = input
        .validity
        .as_ref()
        .map(|_| lease.output(DeviceDType::Bool, vec![n]))
        .transpose()?;
    if n != 0 {
        let source_offsets = input.offsets.address();
        let source_values = input.values.address();
        let source_valid = input.validity.as_ref().map_or(0, DeviceColumn::address);
        let indices = selection.indices.address();
        let offsets_address = offsets.address();
        let values_address = values.address();
        let valid_address = valid.as_ref().map_or(0, DeviceOutput::address);
        let n64 = n as u64;
        let width = dtype.width() as u64;
        let function = &lease.device().0.kernels.selection_ragged_copy;
        let threads = threads(lease, function)?;
        let average = total.saturating_mul(dtype.width()) / n;
        let group = if average >= 4096 { threads } else { 32 };
        let groups = threads / group;
        let config = LaunchConfig {
            grid_dim: (n.div_ceil(groups as usize).clamp(1, 65535) as u32, 1, 1),
            block_dim: (threads, 1, 1),
            shared_mem_bytes: 0,
        };
        let group = u64::from(group);
        let mut launch = lease.execution().launch_builder(function);
        launch
            .arg(&source_offsets)
            .arg(&source_values)
            .arg(&source_valid)
            .arg(&indices)
            .arg(&offsets_address)
            .arg(&values_address)
            .arg(&valid_address)
            .arg(&n64)
            .arg(&width)
            .arg(&group);
        // SAFETY: the prior selected-range pass checked every immutable source
        // range. Nonnegative prefix sums assign disjoint exact output ranges;
        // no per-byte int64 position array or unselected offset scan exists.
        unsafe { launch.launch(config) }
            .map_err(|e| fail(lease, "gathering device ragged payload", e))?;
    }
    Ok(ResidentRaggedField {
        name: input.name.clone(),
        kind: input.kind,
        offsets: publish(offsets)?,
        values: publish(values)?,
        validity: valid.map(publish).transpose()?,
    })
}
fn topk(
    lease: &mut DeviceLease,
    i: &DeviceColumn,
    options: TopKSelection,
) -> Result<(DeviceColumn, DeviceColumn, u64), EngineError> {
    if options.k == 0
        || options.rows > isize::MAX as usize
        || i.dtype() != DeviceDType::I64
        || i.shape().len() != 1
    {
        return Err(parameter(
            "TopK selection requires original positive k and int64 entry rows",
        ));
    }
    let n = i.shape()[0];
    retain(lease, i, Some(DeviceDType::I64), &[n])?;
    let counts = Buffer::<u64>::zeros(lease, options.rows)?;
    let status = Buffer::<u64>::status(lease, 1)?;
    if n != 0 {
        let indices_i = i.address();
        let n64 = n as u64;
        let rows = options.rows as u64;
        let count_address = counts.address();
        let status_address = status.address();
        let function = &lease.device().0.kernels.selection_topk_count;
        let threads = threads(lease, function)?;
        let mut launch = lease.execution().launch_builder(function);
        launch
            .arg(&indices_i)
            .arg(&n64)
            .arg(&rows)
            .arg(&count_address)
            .arg(&status_address);
        // SAFETY: every selected entry is checked before its row count access;
        // canonical entry selection retains original nondecreasing row order.
        unsafe { launch.launch(grid(threads, n)) }
            .map_err(|e| fail(lease, "counting selected TopK rows", e))?;
    }
    if control::<1>(lease, &status)?[0] != u64::MAX {
        return Err(parameter(
            "selected TopK i rows are outside the original shape or row order",
        ));
    }
    let prefix = Buffer::adopt(lease, scan_u64(lease, counts.data(), options.rows)?);
    let offset_len = options
        .rows
        .checked_add(1)
        .ok_or_else(|| overflow("TopK row offset length overflows"))?;
    let offsets = lease.output(DeviceDType::I64, vec![offset_len])?;
    copy(lease, &offsets, prefix.address())?;
    let flags = lease.output(DeviceDType::Bool, vec![options.rows])?;
    let ties = Buffer::<u64>::zeros(lease, 1)?;
    if options.rows != 0 {
        let count_address = counts.address();
        let rows = options.rows as u64;
        let k = options.k as u64;
        let flags_address = flags.address();
        let ties_address = ties.address();
        let function = &lease.device().0.kernels.selection_topk_flags;
        let threads = threads(lease, function)?;
        let mut launch = lease.execution().launch_builder(function);
        launch
            .arg(&count_address)
            .arg(&rows)
            .arg(&k)
            .arg(&flags_address)
            .arg(&ties_address);
        // SAFETY: row counts sum to n and flags cover exactly original n_a
        // rows, including empty rows. Integer ties are count>k, never reranked.
        unsafe { launch.launch(grid(threads, options.rows)) }
            .map_err(|e| fail(lease, "rebuilding selected TopK tie flags", e))?;
    }
    let tied_rows = control::<1>(lease, &ties)?[0];
    Ok((publish(offsets)?, publish(flags)?, tied_rows))
}

pub fn gather_columns(
    lease: &mut DeviceLease,
    selection: &NormalizedSelection,
    numeric_fields: &[ResidentField],
    ragged_fields: &[ResidentRaggedField],
    topk_options: Option<TopKSelection>,
) -> Result<SelectionOutput, EngineError> {
    guard(|| {
        gather_inner(
            lease,
            selection,
            numeric_fields,
            ragged_fields,
            topk_options,
        )
    })
}
fn gather_inner(
    lease: &mut DeviceLease,
    selection: &NormalizedSelection,
    numeric_fields: &[ResidentField],
    ragged_fields: &[ResidentRaggedField],
    topk_options: Option<TopKSelection>,
) -> Result<SelectionOutput, EngineError> {
    let n = selection
        .indices
        .shape()
        .first()
        .copied()
        .ok_or_else(|| parameter("selection indices must be one-dimensional"))?;
    retain(lease, &selection.indices, Some(DeviceDType::I64), &[n])?;
    if topk_options.is_some()
        && (numeric_fields
            .iter()
            .any(|v| matches!(v.name.as_str(), "offsets" | "tie_at_k"))
            || !numeric_fields.iter().any(|v| v.name == "i"))
    {
        return Err(parameter(
            "TopK gather requires entry i and excludes row metadata from entry columns",
        ));
    }
    let mut output = SelectionOutput {
        numeric: Vec::new(),
        ragged: Vec::new(),
        tied_rows: None,
    };
    output
        .numeric
        .try_reserve_exact(numeric_fields.len() + usize::from(topk_options.is_some()) * 2)
        .map_err(|_| EngineError::Device(String::new()))?;
    output
        .ragged
        .try_reserve_exact(ragged_fields.len())
        .map_err(|_| EngineError::Device(String::new()))?;
    for field in numeric_fields {
        output.numeric.push(ResidentField {
            name: field.name.clone(),
            mask_bits: field.mask_bits,
            values: numeric(lease, &field.values, selection)?,
        });
    }
    for field in ragged_fields {
        output.ragged.push(ragged(lease, field, selection)?);
    }
    if let Some(options) = topk_options {
        let i = &output
            .numeric
            .iter()
            .find(|v| v.name == "i")
            .expect("validated TopK i field")
            .values;
        let (offsets, tie_at_k, tied_rows) = topk(lease, i, options)?;
        output.numeric.push(ResidentField {
            name: "offsets".into(),
            values: offsets,
            mask_bits: 0,
        });
        output.numeric.push(ResidentField {
            name: "tie_at_k".into(),
            values: tie_at_k,
            mask_bits: 0,
        });
        output.tied_rows = Some(tied_rows);
    }
    lease.synchronize()?;
    Ok(output)
}

/// The one-to-many Each mapping needs only a selected-length zero column.
/// zip can reuse the immutable normalized owner for both coordinates.
pub fn zero_indices(lease: &DeviceLease, n: usize) -> Result<DeviceColumn, EngineError> {
    guard(|| zero_inner(lease, n))
}
fn zero_inner(lease: &DeviceLease, n: usize) -> Result<DeviceColumn, EngineError> {
    let output = lease.output(DeviceDType::I64, vec![n])?;
    if output.nbytes() != 0 {
        // SAFETY: output owns exactly this writable byte span on the lease
        // stream and no public alias exists until construction is finished.
        unsafe {
            lease
                .execution()
                .memset(output.address(), 0, output.nbytes())
        }
        .map_err(|e| fail(lease, "constructing selected one-to-many coordinates", e))?;
    }
    publish(output)
}

/// D87 RowStats.pairs: nonnegative argbest rows only, in original row order.
/// Native immutable owners are retained; no scores or row payloads are read back.
pub fn rowstats_pairs(
    lease: &mut DeviceLease,
    argbest: &DeviceColumn,
) -> Result<(DeviceColumn, DeviceColumn), EngineError> {
    guard(|| {
        if argbest.shape().len() != 1 {
            return Err(parameter("RowStats argbest must be one-dimensional int64"));
        }
        let n = argbest.shape()[0];
        retain(lease, argbest, Some(DeviceDType::I64), &[n])?;
        let counts = Buffer::<u64>::allocate(lease, n)?;
        if n != 0 {
            let source = argbest.address();
            let n64 = n as u64;
            let counts_address = counts.address();
            let function = &lease.device().0.kernels.selection_rowstats_counts;
            let threads = threads(lease, function)?;
            let mut launch = lease.execution().launch_builder(function);
            launch.arg(&source).arg(&n64).arg(&counts_address);
            // SAFETY: argbest is a retained immutable I64[n] owner and counts
            // is initialized completely before its positive prefix scan.
            unsafe { launch.launch(grid(threads, n)) }
                .map_err(|e| fail(lease, "counting RowStats pair rows", e))?;
        }
        let prefix = Buffer::adopt(lease, scan_u64(lease, counts.data(), n)?);
        let selected = usize::try_from(last(lease, &prefix, n)?)
            .map_err(|_| overflow("RowStats pair count exceeds platform limits"))?;
        let i = lease.output(DeviceDType::I64, vec![selected])?;
        let j = lease.output(DeviceDType::I64, vec![selected])?;
        if n != 0 {
            let source = argbest.address();
            let n64 = n as u64;
            let prefix_address = prefix.address();
            let output_i = i.address();
            let output_j = j.address();
            let function = &lease.device().0.kernels.selection_rowstats_pairs;
            let threads = threads(lease, function)?;
            let mut launch = lease.execution().launch_builder(function);
            launch
                .arg(&source)
                .arg(&n64)
                .arg(&prefix_address)
                .arg(&output_i)
                .arg(&output_j);
            // SAFETY: the immutable argbest cannot change between passes;
            // stable positive prefixes assign exactly selected output rows.
            unsafe { launch.launch(grid(threads, n)) }
                .map_err(|e| fail(lease, "constructing RowStats device pairs", e))?;
        }
        lease.synchronize()?;
        Ok((publish(i)?, publish(j)?))
    })
}
