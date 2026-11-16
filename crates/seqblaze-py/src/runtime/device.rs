//! CAI v3 and DLPack ownership bridges. These adapters read host metadata only;
//! every payload pointer remains a device address. C layouts follow the public
//! DLPack C API (Apache-2.0): https://dmlc.github.io/dlpack/latest/c_api.html.
use super::device_lifetime::Detached;
use std::{ffi::c_void, mem::align_of, sync::Arc};

use pyo3::{
    prelude::*,
    types::{PyBytes, PyCapsule, PyCapsuleMethods, PyDict, PyTuple},
};
use seqblaze_core::validate::EngineError;
use seqblaze_cuda::interop::{
    DeviceArray, DeviceArrayDescriptor, DeviceColumn, DeviceDType, DeviceLease, DeviceSequenceTable,
};

use super::{error, PySeqTable, TableStorage};

fn parameter(py: Python<'_>, message: impl Into<String>) -> PyErr {
    error(py, EngineError::Parameter(message.into()))
}

fn dtype(py: Python<'_>, code: u8, bits: u8, lanes: u16) -> PyResult<DeviceDType> {
    if lanes != 1 {
        return Err(parameter(
            py,
            "device arrays require scalar elements, not vector lanes",
        ));
    }
    match (code, bits) {
        (0, 8) => Ok(DeviceDType::I8),
        (0, 16) => Ok(DeviceDType::I16),
        (0, 32) => Ok(DeviceDType::I32),
        (0, 64) => Ok(DeviceDType::I64),
        (1, 8) => Ok(DeviceDType::U8),
        (1, 16) => Ok(DeviceDType::U16),
        (1, 32) => Ok(DeviceDType::U32),
        (1, 64) => Ok(DeviceDType::U64),
        (2, 64) => Ok(DeviceDType::F64),
        (6, 8) => Ok(DeviceDType::Bool),
        _ => Err(parameter(
            py,
            "device arrays require byte/integer, float64 or bool elements",
        )),
    }
}

fn contiguous_strides(py: Python<'_>, shape: &[usize], width: usize) -> PyResult<Vec<i64>> {
    let mut strides = vec![0; shape.len()];
    let mut stride = i64::try_from(width).expect("supported dtype width");
    for dimension in (0..shape.len()).rev() {
        strides[dimension] = stride;
        stride = stride
            .checked_mul(
                i64::try_from(shape[dimension])
                    .map_err(|_| parameter(py, "device dimension exceeds int64"))?,
            )
            .ok_or_else(|| parameter(py, "device strides exceed int64"))?;
    }
    Ok(strides)
}

pub(super) fn import_cai(
    py: Python<'_>,
    object: &Bound<'_, PyAny>,
    ordinal: usize,
) -> PyResult<DeviceArray> {
    let interface = object.getattr("__cuda_array_interface__")?;
    let dictionary = interface
        .cast::<PyDict>()
        .map_err(|_| parameter(py, "__cuda_array_interface__ must be a dict"))?;
    let required = |name: &str| {
        dictionary
            .get_item(name)?
            .ok_or_else(|| parameter(py, format!("CUDA Array Interface lacks {name}")))
    };
    if required("version")?.extract::<u32>()? != 3 {
        return Err(parameter(
            py,
            "device input requires CUDA Array Interface version 3",
        ));
    }
    if dictionary
        .get_item("mask")?
        .is_some_and(|mask| !mask.is_none())
    {
        return Err(parameter(
            py,
            "masked CUDA arrays are unsupported; sequence nulls use the strings validity bitmap",
        ));
    }
    let shape_value = required("shape")?;
    let shape_tuple = shape_value
        .cast::<PyTuple>()
        .map_err(|_| parameter(py, "device array shape must be a tuple"))?;
    if shape_tuple.len() > 2 {
        return Err(parameter(
            py,
            "device sequence/index arrays have at most two dimensions",
        ));
    }
    let shape: Vec<usize> = shape_tuple.extract()?;
    let typestr: String = required("typestr")?.extract()?;
    let numpy_dtype = py.import("numpy")?.getattr("dtype")?.call1((typestr,))?;
    if !numpy_dtype.getattr("isnative")?.is_truthy()? {
        return Err(parameter(
            py,
            "device arrays require native-endian elements",
        ));
    }
    let kind: String = numpy_dtype.getattr("kind")?.extract()?;
    let width: usize = numpy_dtype.getattr("itemsize")?.extract()?;
    let code = match kind.as_str() {
        "i" => 0,
        "u" => 1,
        "f" => 2,
        "b" => 6,
        _ => return Err(parameter(py, "unsupported CUDA Array Interface dtype")),
    };
    let bits = width
        .checked_mul(8)
        .and_then(|n| u8::try_from(n).ok())
        .ok_or_else(|| parameter(py, "device dtype is too wide"))?;
    let dtype = dtype(py, code, bits, 1)?;
    let (address, readonly): (u64, bool) = required("data")?.extract()?;
    let strides = match dictionary.get_item("strides")? {
        Some(strides) if !strides.is_none() => {
            let values = strides
                .cast::<PyTuple>()
                .map_err(|_| parameter(py, "device strides must be a tuple or None"))?;
            if values.len() != shape.len() {
                return Err(parameter(py, "device strides must match the array rank"));
            }
            values.extract::<Vec<i64>>()?
        }
        _ => contiguous_strides(py, &shape, dtype.width())?,
    };
    let producer_stream = match dictionary.get_item("stream")? {
        Some(stream) if !stream.is_none() => Some(stream.extract::<u64>()?),
        _ => None,
    };
    let descriptor = DeviceArrayDescriptor {
        address,
        shape,
        strides,
        dtype,
        readonly,
        ordinal,
        producer_stream,
    };
    // SAFETY: CAI's producer retains its described allocation for the lifetime
    // of the exporting object; the caller retains this owner through completion.
    unsafe { DeviceArray::from_external(descriptor, Arc::new(object.clone().unbind())) }
        .map_err(|e| error(py, e))
}

#[repr(C)]
#[derive(Clone, Copy)]
struct DlDevice {
    kind: i32,
    ordinal: i32,
}
#[repr(C)]
#[derive(Clone, Copy)]
struct DlDType {
    code: u8,
    bits: u8,
    lanes: u16,
}
#[repr(C)]
#[derive(Clone, Copy)]
struct DlTensor {
    data: *mut c_void,
    device: DlDevice,
    ndim: i32,
    dtype: DlDType,
    shape: *const i64,
    strides: *const i64,
    byte_offset: u64,
}
#[repr(C)]
struct DlManaged {
    tensor: DlTensor,
    manager: *mut c_void,
    deleter: Option<unsafe extern "C" fn(*mut DlManaged)>,
}
#[repr(C)]
struct DlVersion {
    major: u32,
    minor: u32,
}
#[repr(C)]
struct DlVersioned {
    version: DlVersion,
    manager: *mut c_void,
    deleter: Option<unsafe extern "C" fn(*mut DlVersioned)>,
    flags: u64,
    tensor: DlTensor,
}

enum DlOwner {
    Legacy(*mut DlManaged),
    Versioned(*mut DlVersioned),
}
// SAFETY: only the device payload is read; these managed tensor bases are never
// mutated after capsule consumption, and the unique final deleter attaches Python.
unsafe impl Send for DlOwner {}
unsafe impl Sync for DlOwner {}
impl Drop for DlOwner {
    fn drop(&mut self) {
        // Do not invoke a producer's Python callback after interpreter teardown.
        // SAFETY: Py_IsInitialized is an always-available abi3 query.
        if unsafe { pyo3::ffi::Py_IsInitialized() } == 0 {
            return;
        }
        Python::attach(|_| {
            // SAFETY: the capsule was renamed before ownership moved here;
            // exactly this owner invokes the producer-supplied deleter once.
            unsafe {
                match *self {
                    Self::Legacy(pointer) => {
                        if let Some(release) = (*pointer).deleter {
                            release(pointer);
                        }
                    }
                    Self::Versioned(pointer) => {
                        if let Some(release) = (*pointer).deleter {
                            release(pointer);
                        }
                    }
                }
            }
        });
    }
}

pub(super) fn import_dlpack(
    py: Python<'_>,
    object: &Bound<'_, PyAny>,
    ordinal: usize,
    consumer_stream: u64,
) -> PyResult<DeviceArray> {
    let (kind, device): (i32, i32) = object.call_method0("__dlpack_device__")?.extract()?;
    if !matches!(kind, 2 | 13) || usize::try_from(device).ok() != Some(ordinal) {
        return Err(error(
            py,
            EngineError::Device("DLPack input belongs to another device or backend".into()),
        ));
    }
    // Request the universally supported Python DLPack stream argument. Both
    // legacy and versioned capsule responses are accepted and owned correctly.
    let kwargs = PyDict::new(py);
    kwargs.set_item("stream", consumer_stream)?;
    let exported = object.call_method("__dlpack__", (), Some(&kwargs))?;
    let capsule = exported
        .cast::<PyCapsule>()
        .map_err(|_| parameter(py, "__dlpack__ must return a managed tensor capsule"))?;
    let (owner, tensor, readonly) = if let Ok(pointer) = capsule.pointer_checked(Some(c"dltensor"))
    {
        let pointer = pointer.cast::<DlManaged>().as_ptr();
        if pointer as usize % align_of::<DlManaged>() != 0 {
            return Err(parameter(py, "unaligned DLPack managed tensor"));
        }
        // SAFETY: a named protocol capsule retains this aligned managed tensor.
        let tensor = unsafe { (*pointer).tensor };
        // SAFETY: static name lives for the capsule's lifetime. A successful
        // rename disables its original deleter before DlOwner takes ownership.
        if unsafe { pyo3::ffi::PyCapsule_SetName(capsule.as_ptr(), c"used_dltensor".as_ptr()) } != 0
        {
            return Err(PyErr::fetch(py));
        }
        (DlOwner::Legacy(pointer), tensor, false)
    } else {
        let pointer = capsule
            .pointer_checked(Some(c"dltensor_versioned"))
            .map_err(|_| parameter(py, "DLPack capsule was consumed or has an unsupported name"))?
            .cast::<DlVersioned>()
            .as_ptr();
        if pointer as usize % align_of::<DlVersioned>() != 0 {
            return Err(parameter(py, "unaligned versioned DLPack managed tensor"));
        }
        // SAFETY: version/deleter have a stable prefix across major versions.
        let major = unsafe { (*pointer).version.major };
        if major != 1 {
            return Err(parameter(py, "unsupported DLPack major version"));
        }
        let tensor = unsafe { (*pointer).tensor };
        let flags = unsafe { (*pointer).flags };
        if flags & 2 != 0 {
            return Err(parameter(
                py,
                "DLPack producer copied its input despite the zero-copy device import contract",
            ));
        }
        if unsafe {
            pyo3::ffi::PyCapsule_SetName(capsule.as_ptr(), c"used_dltensor_versioned".as_ptr())
        } != 0
        {
            return Err(PyErr::fetch(py));
        }
        (DlOwner::Versioned(pointer), tensor, flags & 1 != 0)
    };
    let owner = Arc::new((owner, object.clone().unbind()));
    if !matches!(tensor.device.kind, 2 | 13) || tensor.device.ordinal != device {
        return Err(error(
            py,
            EngineError::Device("DLPack capsule disagrees with its producer device".into()),
        ));
    }
    let rank = usize::try_from(tensor.ndim).map_err(|_| parameter(py, "negative DLPack rank"))?;
    if rank > 2 || (rank != 0 && tensor.shape.is_null()) {
        return Err(parameter(
            py,
            "DLPack sequence/index arrays require rank zero to two and a valid shape",
        ));
    }
    if tensor.shape as usize % align_of::<i64>() != 0
        || (!tensor.strides.is_null() && tensor.strides as usize % align_of::<i64>() != 0)
    {
        return Err(parameter(py, "DLPack shape/strides must be int64-aligned"));
    }
    let dtype = dtype(py, tensor.dtype.code, tensor.dtype.bits, tensor.dtype.lanes)?;
    let mut shape = Vec::with_capacity(rank);
    for axis in 0..rank {
        // SAFETY: the managed DLPack tensor retains rank shape entries.
        shape.push(
            usize::try_from(unsafe { *tensor.shape.add(axis) })
                .map_err(|_| parameter(py, "negative DLPack dimension"))?,
        );
    }
    let strides = if tensor.strides.is_null() {
        contiguous_strides(py, &shape, dtype.width())?
    } else {
        let mut strides = Vec::with_capacity(rank);
        for axis in 0..rank {
            let elements = unsafe { *tensor.strides.add(axis) };
            strides.push(
                elements
                    .checked_mul(dtype.width() as i64)
                    .ok_or_else(|| parameter(py, "DLPack byte stride exceeds int64"))?,
            );
        }
        strides
    };
    let address = (tensor.data as usize as u64)
        .checked_add(tensor.byte_offset)
        .ok_or_else(|| parameter(py, "DLPack byte offset exceeds address space"))?;
    let descriptor = DeviceArrayDescriptor {
        address,
        shape,
        strides,
        dtype,
        readonly,
        ordinal,
        producer_stream: None,
    };
    // The producer has already made its work visible on this consumer stream.
    // SAFETY: DlOwner now owns exactly the managed tensor's live storage.
    unsafe { DeviceArray::from_external(descriptor, owner) }.map_err(|e| error(py, e))
}

/// The host bytes of an explicit device-to-host export, handed to NumPy without
/// a second copy (levenshtein-cuda.md §14): NumPy reads them read-only through
/// `__array_interface__` and keeps this owner alive as the array's base. The
/// bytes come from the export's own `device-to-host export` reservation.
#[pyclass(frozen, module = "seqblaze._core", name = "_HostBytes")]
pub(super) struct PyHostBytes {
    data: Vec<u8>,
}

#[pymethods]
impl PyHostBytes {
    #[getter(__array_interface__)]
    fn array_interface<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let interface = PyDict::new(py);
        interface.set_item("version", 3)?;
        interface.set_item("shape", (self.data.len(),))?;
        interface.set_item("typestr", "|u1")?;
        interface.set_item("data", (self.data.as_ptr() as usize, true))?;
        Ok(interface)
    }
}

/// Exported host bytes as a Python object: `bytes` when empty, otherwise the
/// owner of the export's vector itself (no copy).
fn host_object(py: Python<'_>, data: Vec<u8>) -> PyResult<Bound<'_, PyAny>> {
    if data.is_empty() {
        return Ok(PyBytes::new(py, &[]).into_any());
    }
    Ok(Bound::new(py, PyHostBytes { data })?.into_any())
}

#[pyclass(frozen, module = "seqblaze._core", name = "_DeviceColumn")]
pub(super) struct PyDeviceColumn {
    pub(super) column: Detached<DeviceColumn>,
}

#[pymethods]
impl PyDeviceColumn {
    #[getter]
    fn dtype(&self) -> &'static str {
        self.column.dtype().typestr()
    }
    #[getter]
    fn shape(&self) -> Vec<usize> {
        self.column.shape().to_vec()
    }
    #[getter]
    fn ordinal(&self) -> usize {
        self.column.ordinal()
    }
    #[getter]
    fn nbytes(&self) -> usize {
        self.column.nbytes()
    }
    fn _host_bytes<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let data = py
            .detach(|| self.column.to_host_bytes())
            .map_err(|e| error(py, e))?;
        host_object(py, data)
    }
    fn _copy_into(&self, py: Python<'_>, target: &Bound<'_, PyAny>) -> PyResult<()> {
        let target = import_cai(py, target, self.column.ordinal())?;
        py.detach(|| self.column.copy_into(&target))
            .map_err(|e| error(py, e))
    }
}

struct HostLengths {
    index: seqblaze_core::work::IndexArray,
}
impl HostLengths {
    fn same_geometry(&self, other: &Self) -> bool {
        let a = self.index.storage_info();
        let b = other.index.storage_info();
        (a.address, a.len, a.itemsize, a.stride) == (b.address, b.len, b.itemsize, b.stride)
    }
}
enum DeviceSource {
    Matrix {
        data: Py<PyAny>,
        lengths: Py<PyAny>,
        host_lengths: Option<HostLengths>,
    },
    Strings {
        data: Py<PyAny>,
        offsets: Py<PyAny>,
        validity: Option<Py<PyAny>>,
        row_offset: usize,
    },
}

/// Internal imported table. Its ownership is real device ownership, never an
/// empty host table impersonating a device input. Public SeqTable dispatch is
/// connected only once every resident compute path accepts this storage.
#[pyclass(frozen, module = "seqblaze._core", name = "_DeviceSeqTable")]
pub(super) struct PyDeviceSeqTable {
    ordinal: usize,
    imported: Detached<DeviceSequenceTable>,
    source: DeviceSource,
}

// Host length metadata is refreshed per call and uploaded once. Its NumPy
// memoryview stays retained between calls, matching the CPU resize boundary.
fn matrix_lengths(
    py: Python<'_>,
    object: &Bound<'_, PyAny>,
    lease: &mut DeviceLease,
    rows: usize,
    columns: usize,
) -> PyResult<(DeviceArray, Option<HostLengths>)> {
    if object.hasattr("__cuda_array_interface__")? {
        return Ok((array_on_stream(py, object, lease)?, None));
    }
    if object.hasattr("__dlpack_device__")? {
        let (kind, _): (i32, i32) = object.call_method0("__dlpack_device__")?.extract()?;
        if kind == 2 || kind == 13 {
            return Ok((array_on_stream(py, object, lease)?, None));
        }
    }
    let view = super::numpy::native_view(py, object, "lengths")?;
    let dtype = view.getattr("dtype")?;
    let kind: String = dtype.getattr("kind")?.extract()?;
    let width: usize = dtype.getattr("itemsize")?.extract()?;
    let ndim: usize = view.getattr("ndim")?.extract()?;
    if ndim != 1
        || kind != "i"
        || !matches!(width, 4 | 8)
        || !dtype.getattr("isnative")?.is_truthy()?
    {
        return Err(parameter(
            py,
            "lengths must be a one-dimensional native-endian NumPy int32/int64 array",
        ));
    }
    let (n,): (usize,) = view.getattr("shape")?.extract()?;
    if n != rows {
        return Err(parameter(
            py,
            format!("lengths has {n} values; uint8 matrix has {rows} rows"),
        ));
    }
    let (stride,): (isize,) = view.getattr("strides")?.extract()?;
    let (address, _): (usize, bool) = view
        .getattr("__array_interface__")?
        .get_item("data")?
        .extract()?;
    let descriptor = seqblaze_core::seq::LengthDescriptor {
        address,
        rows: n,
        itemsize: width,
        stride,
    };
    let export = py
        .import("builtins")?
        .getattr("memoryview")?
        .call1((&view,))?
        .unbind();
    // SAFETY: exact native ndarray geometry plus its retained memoryview cover
    // all byte-atomic reads, including unaligned, negative and zero strides.
    let index = unsafe {
        seqblaze_core::work::IndexArray::from_borrowed(descriptor, Box::new(export), false)
    }
    .map_err(|e| error(py, e))?;
    let capacity = n
        .checked_mul(8)
        .filter(|bytes| *bytes <= isize::MAX as usize)
        .ok_or_else(|| {
            error(
                py,
                EngineError::Overflow("device host length metadata exceeds platform limits".into()),
            )
        })?;
    let mut bytes = Vec::new();
    seqblaze_core::detail::try_reserve_exact(&mut bytes, capacity, "device host length metadata")
        .map_err(|e| error(py, e))?;
    use std::sync::atomic::{AtomicU8, Ordering};
    for row in 0..n {
        let pointer = (address as *const u8).wrapping_offset(row as isize * stride);
        let mut word = [0u8; 8];
        for (at, byte) in word[..width].iter_mut().enumerate() {
            // SAFETY: IndexArray validated this exact descriptor above and
            // keeps its native export alive throughout this GIL-held scan.
            *byte =
                unsafe { (&*pointer.wrapping_add(at).cast::<AtomicU8>()).load(Ordering::Relaxed) };
        }
        let length = if width == 4 {
            i32::from_ne_bytes(word[..4].try_into().expect("four bytes")) as i64
        } else {
            i64::from_ne_bytes(word)
        };
        if length < 0 || length as u64 > columns as u64 {
            return Err(parameter(py,format!("lengths[{row}]={length} must be between 0 and matrix column capacity {columns}")));
        }
        bytes.extend_from_slice(&length.to_le_bytes());
    }
    let column = py
        .detach(|| {
            let _fp = seqblaze_core::fpenv::FpGuard::new();
            lease.upload_bytes(DeviceDType::I64, vec![n], &bytes)
        })
        .map(Detached::new)
        .map_err(|e| error(py, e))?;
    let array =
        seqblaze_cuda::resident_indices::owned_flat_view(&column, n).map_err(|e| error(py, e))?;
    Ok((array, Some(HostLengths { index })))
}
fn matrix_table(
    py: Python<'_>,
    data: &Bound<'_, PyAny>,
    lengths: &Bound<'_, PyAny>,
    lease: &mut DeviceLease,
) -> PyResult<(DeviceSequenceTable, Option<HostLengths>)> {
    let data = array_on_stream(py, data, lease)?;
    let d = data.descriptor();
    if d.dtype != DeviceDType::U8 || d.shape.len() != 2 {
        return Err(parameter(
            py,
            "device sequence matrices require rank-two uint8 data",
        ));
    }
    let (lengths, host) = matrix_lengths(py, lengths, lease, d.shape[0], d.shape[1])?;
    Ok((
        py.detach(|| DeviceSequenceTable::matrix(data, lengths))
            .map_err(|e| error(py, e))?,
        host,
    ))
}

pub(super) fn array_on_stream(
    py: Python<'_>,
    object: &Bound<'_, PyAny>,
    lease: &DeviceLease,
) -> PyResult<DeviceArray> {
    if object.hasattr("__cuda_array_interface__")? {
        import_cai(py, object, lease.device().info().ordinal)
    } else {
        import_dlpack(
            py,
            object,
            lease.device().info().ordinal,
            lease.stream_handle(),
        )
    }
}

impl PyDeviceSeqTable {
    pub(super) fn bind(
        &self,
        py: Python<'_>,
        lease: &mut DeviceLease,
    ) -> PyResult<seqblaze_cuda::interop::BoundDeviceRows> {
        if self.ordinal != lease.device().info().ordinal {
            return Err(error(
                py,
                EngineError::Device("device tables belong to different CUDA devices".into()),
            ));
        }
        let current = match &self.source {
            DeviceSource::Matrix {
                data,
                lengths,
                host_lengths,
            } => {
                let (current, host) = matrix_table(py, data.bind(py), lengths.bind(py), lease)?;
                let same = match (host_lengths.as_ref(), host.as_ref()) {
                    (None, None) => true,
                    (Some(a), Some(b)) => a.same_geometry(b),
                    _ => false,
                };
                if !same {
                    return Err(parameter(
                        py,
                        "retained host lengths were resized or their layout changed",
                    ));
                }
                current
            }
            DeviceSource::Strings {
                data,
                offsets,
                validity,
                row_offset,
            } => {
                let valid = validity
                    .as_ref()
                    .map(|value| array_on_stream(py, value.bind(py), lease))
                    .transpose()?;
                DeviceSequenceTable::packed(
                    array_on_stream(py, data.bind(py), lease)?,
                    array_on_stream(py, offsets.bind(py), lease)?,
                    valid,
                    self.imported.rows().rows as usize,
                    *row_offset,
                    self.imported.is_text(),
                )
                .map_err(|e| error(py, e))?
            }
        };
        let current = Detached::new(current);
        let mut before = self.imported.rows();
        let mut after = current.rows();
        if matches!(
            &self.source,
            DeviceSource::Matrix {
                host_lengths: Some(_),
                ..
            }
        ) {
            before.lengths = 0;
            after.lengths = 0;
        }
        if after != before {
            return Err(parameter(
                py,
                "retained device sequence buffers were resized or their layout changed",
            ));
        }
        // Descriptor refresh above needs Python; binding may wait for the
        // current producer while uploading its row descriptor. Producer host
        // callbacks must be able to acquire the GIL during that wait.
        py.detach(|| lease.bind_table(&current))
            .map_err(|e| error(py, e))
    }
}

#[pymethods]
impl PyDeviceSeqTable {
    fn __len__(&self) -> usize {
        self.imported.rows().rows as usize
    }
    #[getter]
    fn _source_kind(&self) -> &'static str {
        "device"
    }
    #[getter]
    fn _text(&self) -> bool {
        self.imported.is_text()
    }
    #[getter]
    fn _device_ordinal(&self) -> usize {
        self.ordinal
    }
    fn _device_digest<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let device =
            super::resolve_device(py, Some(self.ordinal as u64))?.expect("ordinal was supplied");
        let mut lease = py
            .detach(|| DeviceLease::new(&device))
            .map(Detached::new)
            .map_err(|e| error(py, e))?;
        let bound = self.bind(py, &mut lease)?;
        let (summary, digest) = py
            .detach(|| {
                let rows = seqblaze_cuda::interop::validate_device_rows(
                    &lease,
                    bound,
                    [u64::MAX; 4],
                    0,
                    isize::MAX as usize,
                )?;
                let digest = seqblaze_cuda::interop::device_sequence_digest(&lease, &rows)?;
                Ok::<_, EngineError>((rows.summary, digest))
            })
            .map_err(|e| error(py, e))?;
        let output = PyDict::new(py);
        let digest: String = digest.iter().map(|byte| format!("{byte:02x}")).collect();
        output.set_item("sha256", digest)?;
        output.set_item("n", summary.rows)?;
        output.set_item("total_nt", summary.total)?;
        output.set_item("min_len", summary.min_length)?;
        output.set_item("max_len", summary.max_length)?;
        output.set_item("text", self.imported.is_text())?;
        Ok(output)
    }
    fn _device_snapshot<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let device =
            super::resolve_device(py, Some(self.ordinal as u64))?.expect("ordinal was supplied");
        let mut lease = py
            .detach(|| DeviceLease::new(&device))
            .map(Detached::new)
            .map_err(|e| error(py, e))?;
        let bound = self.bind(py, &mut lease)?;
        let (summary, snapshot) = py
            .detach(|| {
                let rows = seqblaze_cuda::interop::validate_device_rows(
                    &lease,
                    bound,
                    [u64::MAX; 4],
                    0,
                    isize::MAX as usize,
                )?;
                let snapshot = seqblaze_cuda::interop::device_input_snapshot(&lease, &rows)?;
                Ok::<_, EngineError>((rows.summary, snapshot))
            })
            .map_err(|e| error(py, e))?;
        let output = PyDict::new(py);
        let digest: String = snapshot
            .digest
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect();
        output.set_item("sha256", digest)?;
        output.set_item("n", summary.rows)?;
        output.set_item("total_nt", summary.total)?;
        output.set_item("min_len", summary.min_length)?;
        output.set_item("max_len", summary.max_length)?;
        output.set_item("text", self.imported.is_text())?;
        output.set_item("data", PyBytes::new(py, &snapshot.data))?;
        let offset_bytes = snapshot
            .offsets
            .len()
            .checked_mul(8)
            .ok_or_else(|| parameter(py, "device snapshot offsets exceed platform limits"))?;
        let mut offsets = Vec::new();
        offsets.try_reserve_exact(offset_bytes).map_err(|_| {
            pyo3::exceptions::PyMemoryError::new_err("unable to serialize device snapshot offsets")
        })?;
        for &offset in &snapshot.offsets {
            let offset = i64::try_from(offset)
                .map_err(|_| parameter(py, "device snapshot offset exceeds int64"))?;
            offsets.extend_from_slice(&offset.to_le_bytes());
        }
        output.set_item("offsets", PyBytes::new(py, &offsets))?;
        output.set_item("validity", PyBytes::new(py, &snapshot.validity))?;
        // Repro is the sole explicit payload serialization path. Every byte of
        // this safe Python literal comes from the independently owned snapshot
        // whose digest was computed before its D2H copy, never the live input.
        let capacity = snapshot
            .data
            .len()
            .checked_mul(4)
            .and_then(|n| {
                summary
                    .rows
                    .checked_mul(8)
                    .and_then(|rows| usize::try_from(rows).ok())
                    .and_then(|rows| n.checked_add(rows))
            })
            .and_then(|n| n.checked_add(2))
            .ok_or_else(|| parameter(py, "device repro literal exceeds platform limits"))?;
        let mut literal = Vec::new();
        literal.try_reserve_exact(capacity).map_err(|_| {
            pyo3::exceptions::PyMemoryError::new_err("unable to serialize device repro literal")
        })?;
        literal.push(b'[');
        const HEX: &[u8; 16] = b"0123456789abcdef";
        for row in 0..snapshot.validity.len() {
            if row != 0 {
                literal.extend_from_slice(b", ");
            }
            if snapshot.validity[row] == 0 {
                literal.extend_from_slice(b"None");
                continue;
            }
            literal.extend_from_slice(b"b'");
            let begin = snapshot.offsets[row] as usize;
            let end = snapshot.offsets[row + 1] as usize;
            for &byte in &snapshot.data[begin..end] {
                literal.extend_from_slice(b"\\x");
                literal.push(HEX[(byte >> 4) as usize]);
                literal.push(HEX[(byte & 15) as usize]);
            }
            literal.push(b'\'');
        }
        literal.push(b']');
        output.set_item("python_rows", PyBytes::new(py, &literal))?;
        Ok(output)
    }
    #[getter]
    pub(super) fn storage_info<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let info = PyDict::new(py);
        info.set_item("kind", "device")?;
        info.set_item("device", format!("cuda:{}", self.ordinal))?;
        info.set_item("rows", self.imported.rows().rows)?;
        info.set_item(
            "layout",
            if self.imported.rows().layout == 0 {
                "strings"
            } else {
                "matrix"
            },
        )?;
        Ok(info)
    }
}

#[pyfunction]
fn _native_seqs_device_matrix<'py>(
    py: Python<'py>,
    object: Bound<'py, PyAny>,
    lengths: Bound<'py, PyAny>,
    ordinal: usize,
) -> PyResult<Bound<'py, PySeqTable>> {
    let device = super::resolve_device(py, Some(ordinal as u64))?.expect("ordinal was supplied");
    let mut lease = py
        .detach(|| DeviceLease::new(&device))
        .map(Detached::new)
        .map_err(|e| error(py, e))?;
    let (imported, host_lengths) = matrix_table(py, &object, &lengths, &mut lease)?;
    let imported = Detached::new(imported);
    let _bound = py
        .detach(|| lease.bind_table(&imported))
        .map(Detached::new)
        .map_err(|e| error(py, e))?;
    let rows = imported.rows().rows as usize;
    let text = imported.is_text();
    let owner = Py::new(
        py,
        PyDeviceSeqTable {
            ordinal,
            imported,
            source: DeviceSource::Matrix {
                data: object.unbind(),
                lengths: lengths.unbind(),
                host_lengths,
            },
        },
    )?;
    Bound::new(
        py,
        PySeqTable {
            table: TableStorage::Device { owner, rows, text },
        },
    )
}

#[pyfunction(signature=(data,offsets,validity,rows,row_offset,ordinal,text=true))]
#[allow(clippy::too_many_arguments)]
fn _native_seqs_device_strings<'py>(
    py: Python<'py>,
    data: Bound<'py, PyAny>,
    offsets: Bound<'py, PyAny>,
    validity: Option<Bound<'py, PyAny>>,
    rows: usize,
    row_offset: usize,
    ordinal: usize,
    text: bool,
) -> PyResult<Bound<'py, PySeqTable>> {
    let device = super::resolve_device(py, Some(ordinal as u64))?.expect("ordinal was supplied");
    let mut lease = py
        .detach(|| DeviceLease::new(&device))
        .map(Detached::new)
        .map_err(|e| error(py, e))?;
    let data_array = array_on_stream(py, &data, &lease)?;
    let offsets_array = array_on_stream(py, &offsets, &lease)?;
    let validity_array = validity
        .as_ref()
        .map(|value| array_on_stream(py, value, &lease))
        .transpose()?;
    let imported = DeviceSequenceTable::packed(
        data_array,
        offsets_array,
        validity_array,
        rows,
        row_offset,
        text,
    )
    .map(Detached::new)
    .map_err(|e| error(py, e))?;
    let _bound = py
        .detach(|| lease.bind_table(&imported))
        .map(Detached::new)
        .map_err(|e| error(py, e))?;
    let text = imported.is_text();
    let owner = Py::new(
        py,
        PyDeviceSeqTable {
            ordinal,
            imported,
            source: DeviceSource::Strings {
                data: data.unbind(),
                offsets: offsets.unbind(),
                validity: validity.map(Bound::unbind),
                row_offset,
            },
        },
    )?;
    Bound::new(
        py,
        PySeqTable {
            table: TableStorage::Device { owner, rows, text },
        },
    )
}

#[pyfunction]
fn _native_device_pointer_ordinal(py: Python<'_>, address: u64) -> PyResult<usize> {
    py.detach(|| seqblaze_cuda::interop::device_pointer_ordinal(address))
        .map_err(|e| error(py, e))
}

#[cfg(feature = "cuda")]
#[pyfunction]
fn _native_execution_stream_validate(py: Python<'_>, ordinal: u64, stream: u64) -> PyResult<()> {
    let device = super::resolve_device(py, Some(ordinal))?.expect("explicit ordinal");
    py.detach(|| {
        seqblaze_cuda::interop::execution_stream::BorrowedExecutionStream::validate(&device, stream)
    })
    .map_err(|e| error(py, e))
}

/// The batched host export behind `Result.to_numpy()` (per-call cost S8): the
/// host bytes of every given column, with the GIL released as for one column.
#[pyfunction]
fn _device_host_bytes_batch<'py>(
    py: Python<'py>,
    owners: Vec<PyRef<'py, PyDeviceColumn>>,
) -> PyResult<Vec<Bound<'py, PyAny>>> {
    let columns: Vec<&DeviceColumn> = owners.iter().map(|owner| &*owner.column).collect();
    let data = py
        .detach(|| DeviceColumn::to_host_bytes_batch(&columns))
        .map_err(|e| error(py, e))?;
    // Each host vector moves into its NumPy owner without a second copy, so the
    // peak is the exported total (levenshtein-cuda.md §14.2 item 5).
    data.into_iter()
        .map(|bytes| host_object(py, bytes))
        .collect()
}

pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    #[cfg(feature = "cuda")]
    module.add_function(wrap_pyfunction!(_native_execution_stream_validate, module)?)?;
    module.add_function(wrap_pyfunction!(_device_host_bytes_batch, module)?)?;
    module.add_class::<PyDeviceColumn>()?;
    module.add_class::<PyHostBytes>()?;
    module.add_function(wrap_pyfunction!(_native_device_pointer_ordinal, module)?)?;
    module.add_class::<PyDeviceSeqTable>()?;
    module.add_function(wrap_pyfunction!(_native_seqs_device_matrix, module)?)?;
    module.add_function(wrap_pyfunction!(_native_seqs_device_strings, module)?)?;
    Ok(())
}
