//! Owned DLPack exports of immutable device results. One D2D copy becomes the
//! consumer's writable tensor; no Python array library or host payload is used.
//! ABI/protocol: https://dmlc.github.io/dlpack/latest/c_api.html and
//! https://data-apis.org/array-api/latest/API_specification/generated/array_api.array.__dlpack__.html
use std::{
    ffi::c_void,
    panic::{catch_unwind, AssertUnwindSafe},
};

use super::{device::PyDeviceColumn, error};
use pyo3::{
    exceptions::{PyBufferError, PyMemoryError, PyOverflowError, PyTypeError, PyValueError},
    prelude::*,
    types::{PyBool, PyInt, PyTuple},
};
use seqblaze_core::fpenv::{FpBracket, FpGuard};
use seqblaze_cuda::interop::{
    retirement::{self, RetireJob},
    DeviceColumn, DeviceDType,
};

const CUDA: i32 = 2;
const IS_COPIED: u64 = 2;

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
const _: () = {
    assert!(std::mem::size_of::<DlDevice>() == 8);
    assert!(std::mem::size_of::<DlDType>() == 4);
    assert!(std::mem::size_of::<DlTensor>() == 48);
    assert!(std::mem::size_of::<DlManaged>() == 64);
    assert!(std::mem::size_of::<DlVersioned>() == 80);
    assert!(std::mem::offset_of!(DlTensor, shape) == 24);
    assert!(std::mem::offset_of!(DlManaged, manager) == 48);
    assert!(std::mem::offset_of!(DlVersioned, tensor) == 32);
};

/// No Python owner or callback is needed by the final consumer deleter. The
/// allocation and both metadata arrays remain native-owned after Python exits.
struct ExportOwner {
    retirement: RetireJob,
    shape: Box<[i64]>,
    strides: Box<[i64]>,
}

unsafe fn release_export(manager: *mut c_void) {
    if manager.is_null() {
        return;
    }
    // SAFETY: manager originated from exactly one Box<ExportOwner>; the
    // DLPack consumer/capsule protocol transfers this unique release once.
    let mut owner = unsafe { Box::from_raw(manager.cast::<ExportOwner>()) };
    owner.retirement.submit();
    // Only native queue transfer and metadata deallocation run here. The worker
    // owns the independent copy until its context barrier proves completion.
}
unsafe extern "C" fn delete_legacy(pointer: *mut DlManaged) {
    if pointer.is_null() {
        return;
    }
    // No Rust panic may unwind across the consumer's C ABI. GPU retirement
    // failure follows the intentional-retention branch above.
    let _ = catch_unwind(AssertUnwindSafe(|| {
        // SAFETY: this callback receives its original unique managed allocation.
        let tensor = unsafe { Box::from_raw(pointer) };
        // SAFETY: manager is the ExportOwner paired with this managed tensor.
        unsafe { release_export(tensor.manager) };
    }));
}
unsafe extern "C" fn delete_versioned(pointer: *mut DlVersioned) {
    if pointer.is_null() {
        return;
    }
    let _ = catch_unwind(AssertUnwindSafe(|| {
        // SAFETY: this callback receives its original unique managed allocation.
        let tensor = unsafe { Box::from_raw(pointer) };
        // SAFETY: manager is the ExportOwner paired with this managed tensor.
        unsafe { release_export(tensor.manager) };
    }));
}
unsafe extern "C" fn capsule_destructor(capsule: *mut pyo3::ffi::PyObject) {
    // The normal path invokes no Python callback. Preserve state even for an
    // exceptional PyErr_WriteUnraisable hook, without normalizing its caller.
    let _bracket = FpBracket::new();
    // SAFETY: Python calls this destructor with a live capsule and its GIL.
    // Static names survive every capsule; consumed names must do nothing.
    unsafe {
        if pyo3::ffi::PyCapsule_IsValid(capsule, c"dltensor".as_ptr()) != 0 {
            let pointer = pyo3::ffi::PyCapsule_GetPointer(capsule, c"dltensor".as_ptr());
            if pointer.is_null() {
                pyo3::ffi::PyErr_WriteUnraisable(capsule);
            } else {
                delete_legacy(pointer.cast());
            }
        } else if pyo3::ffi::PyCapsule_IsValid(capsule, c"dltensor_versioned".as_ptr()) != 0 {
            let pointer = pyo3::ffi::PyCapsule_GetPointer(capsule, c"dltensor_versioned".as_ptr());
            if pointer.is_null() {
                pyo3::ffi::PyErr_WriteUnraisable(capsule);
            } else {
                delete_versioned(pointer.cast());
            }
        }
    }
}

fn plain_integer(value: &Bound<'_, PyAny>, name: &str) -> PyResult<i128> {
    if !value.is_exact_instance_of::<PyInt>() {
        return Err(PyTypeError::new_err(format!(
            "DLPack {name} must be a Python integer"
        )));
    }
    value.extract()
}
fn plain_tuple<'py>(value: &Bound<'py, PyAny>, name: &str) -> PyResult<Bound<'py, PyTuple>> {
    if !value.is_exact_instance_of::<PyTuple>() {
        return Err(PyTypeError::new_err(format!(
            "DLPack {name} must be a tuple"
        )));
    }
    Ok(value.cast::<PyTuple>()?.clone())
}
fn pair(value: &Bound<'_, PyAny>, name: &str) -> PyResult<[i128; 2]> {
    let items = plain_tuple(value, name)?;
    if items.len() != 2 {
        return Err(PyTypeError::new_err(format!(
            "DLPack {name} must contain two integers"
        )));
    }
    Ok([
        plain_integer(&items.get_item(0)?, name)?,
        plain_integer(&items.get_item(1)?, name)?,
    ])
}
fn device_pair(value: &Bound<'_, PyAny>) -> PyResult<[i128; 2]> {
    let items = plain_tuple(value, "dl_device")?;
    if items.len() != 2 {
        return Err(PyTypeError::new_err(
            "DLPack dl_device must contain two integers",
        ));
    }
    let kind = items.get_item(0)?;
    // IntEnum device codes are specified by the Python Array API. A PyLong
    // subclass is read directly by the C API; arbitrary __index__ is not run.
    if !kind.is_instance_of::<PyInt>() || kind.is_instance_of::<PyBool>() {
        return Err(PyTypeError::new_err(
            "DLPack device kind must be an integer or IntEnum",
        ));
    }
    Ok([
        kind.extract()?,
        plain_integer(&items.get_item(1)?, "device ordinal")?,
    ])
}
fn dtype(dtype: DeviceDType) -> DlDType {
    let code = match dtype {
        DeviceDType::I8 | DeviceDType::I16 | DeviceDType::I32 | DeviceDType::I64 => 0,
        DeviceDType::U8 | DeviceDType::U16 | DeviceDType::U32 | DeviceDType::U64 => 1,
        DeviceDType::F64 => 2,
        DeviceDType::Bool => 6,
    };
    DlDType {
        code,
        bits: (dtype.width() * 8) as u8,
        lanes: 1,
    }
}
struct ShapeMetadata {
    dimensions: Vec<usize>,
    logical: Box<[i64]>,
    strides: Box<[i64]>,
}

fn shape_metadata(shape: &Bound<'_, PyAny>, column: &DeviceColumn) -> PyResult<ShapeMetadata> {
    let items = plain_tuple(shape, "shape")?;
    i32::try_from(items.len())
        .map_err(|_| PyOverflowError::new_err("DLPack rank exceeds int32"))?;
    let mut dimensions = Vec::new();
    let mut logical = Vec::new();
    let mut strides = Vec::new();
    dimensions
        .try_reserve_exact(items.len())
        .map_err(|_| PyMemoryError::new_err("unable to allocate DLPack shape"))?;
    logical
        .try_reserve_exact(items.len())
        .map_err(|_| PyMemoryError::new_err("unable to allocate DLPack shape"))?;
    strides
        .try_reserve_exact(items.len())
        .map_err(|_| PyMemoryError::new_err("unable to allocate DLPack strides"))?;
    for value in items.iter() {
        let number = plain_integer(&value, "shape dimension")?;
        let dimension = i64::try_from(number)
            .map_err(|_| PyOverflowError::new_err("DLPack dimension exceeds int64"))?;
        if dimension < 0 {
            return Err(PyValueError::new_err(
                "DLPack dimensions must be nonnegative",
            ));
        }
        let dimension = usize::try_from(dimension)
            .map_err(|_| PyOverflowError::new_err("DLPack dimension exceeds platform limits"))?;
        dimensions.push(dimension);
        logical.push(dimension as i64);
        strides.push(0i64);
    }
    let bytes = if dimensions.contains(&0) {
        0
    } else {
        dimensions
            .iter()
            .try_fold(column.dtype().width(), |n, &dim| n.checked_mul(dim))
            .filter(|&n| n <= isize::MAX as usize)
            .ok_or_else(|| PyOverflowError::new_err("DLPack shape size exceeds platform limits"))?
    };
    if bytes != column.nbytes() {
        return Err(PyValueError::new_err(
            "DLPack shape must preserve the column size",
        ));
    }
    // Empty tensors dereference no element. Explicit zero strides are valid
    // there and avoid meaningless products of unused large dimensions.
    if bytes != 0 {
        let mut stride = 1i64;
        for axis in (0..logical.len()).rev() {
            strides[axis] = stride;
            stride = stride
                .checked_mul(logical[axis])
                .ok_or_else(|| PyOverflowError::new_err("DLPack stride exceeds int64"))?;
        }
    }
    Ok(ShapeMetadata {
        dimensions,
        logical: logical.into_boxed_slice(),
        strides: strides.into_boxed_slice(),
    })
}

#[pyfunction]
#[pyo3(signature=(owner,shape,*,stream=None,max_version=None,dl_device=None,copy=None))]
pub(super) fn _device_dlpack(
    py: Python<'_>,
    owner: PyRef<'_, PyDeviceColumn>,
    shape: &Bound<'_, PyAny>,
    stream: Option<&Bound<'_, PyAny>>,
    max_version: Option<&Bound<'_, PyAny>>,
    dl_device: Option<&Bound<'_, PyAny>>,
    copy: Option<&Bound<'_, PyAny>>,
) -> PyResult<Py<PyAny>> {
    let _guard = FpGuard::new();
    // Validate the immutable-borrow policy before any device allocation/copy.
    if let Some(copy) = copy {
        if !copy.is_exact_instance_of::<PyBool>() {
            return Err(PyTypeError::new_err("DLPack copy must be None or bool"));
        }
        if !copy.extract::<bool>()? {
            return Err(PyBufferError::new_err(
                "immutable device result storage requires a DLPack copy",
            ));
        }
    }
    let consumer_stream = match stream {
        None => None,
        Some(value) => {
            let number = plain_integer(value, "stream")?;
            if number == -1 {
                return Err(PyBufferError::new_err("an immutable DLPack export requires a synchronized copy; stream=-1 is unsupported"));
            }
            if number <= 0 {
                return Err(PyValueError::new_err(
                    "DLPack CUDA stream must be 1, 2, or a positive stream address",
                ));
            }
            Some(
                u64::try_from(number)
                    .map_err(|_| PyOverflowError::new_err("DLPack stream exceeds pointer size"))?,
            )
        }
    };
    let versioned = if let Some(version) = max_version {
        let [major, minor] = pair(version, "max_version")?;
        let major = u32::try_from(major).map_err(|_| {
            PyValueError::new_err("DLPack max_version entries must be nonnegative uint32 integers")
        })?;
        u32::try_from(minor).map_err(|_| {
            PyValueError::new_err("DLPack max_version entries must be nonnegative uint32 integers")
        })?;
        major >= 1
    } else {
        false
    };
    let ordinal = i32::try_from(owner.column.ordinal())
        .map_err(|_| PyBufferError::new_err("DLPack CUDA ordinal exceeds int32"))?;
    if let Some(device) = dl_device {
        if device_pair(device)? != [i128::from(CUDA), i128::from(ordinal)] {
            return Err(PyBufferError::new_err("DLPack export supports only the result's CUDA device; use to_numpy() for a host copy"));
        }
    }
    let ShapeMetadata {
        dimensions: shape,
        logical: dimensions,
        strides,
    } = shape_metadata(shape, &owner.column)?;
    let source = owner.column.clone();
    let (copied, mut retirement) = py
        .detach(move || {
            // Prepare worker, budget ticket and queue node before copying and
            // before any capsule can reach a callback/foreign-thread deleter.
            let retirement = RetireJob::prepare(&source)?;
            let copied = source.copy_for_export(shape, consumer_stream)?;
            Ok((copied, retirement))
        })
        .map_err(|e| error(py, e))?;
    // SAFETY: this independent, synchronously initialized copy becomes the
    // capsule's unique allocation owner and outlives every tensor address use.
    let address = if copied.nbytes() == 0 {
        0
    } else {
        unsafe { copied.export_address() }
    };
    let copied_dtype = copied.dtype();
    retirement.attach(copied);
    let mut owner = Box::new(ExportOwner {
        retirement,
        shape: dimensions,
        strides,
    });
    let tensor = DlTensor {
        data: address as usize as *mut c_void,
        device: DlDevice {
            kind: CUDA,
            ordinal,
        },
        ndim: owner.shape.len() as i32,
        dtype: dtype(copied_dtype),
        shape: owner.shape.as_mut_ptr(),
        strides: owner.strides.as_mut_ptr(),
        byte_offset: 0,
    };
    let manager = Box::into_raw(owner).cast::<c_void>();
    let (pointer, name) = if versioned {
        (
            Box::into_raw(Box::new(DlVersioned {
                version: DlVersion { major: 1, minor: 0 },
                manager,
                deleter: Some(delete_versioned),
                flags: IS_COPIED,
                tensor,
            }))
            .cast::<c_void>(),
            c"dltensor_versioned",
        )
    } else {
        (
            Box::into_raw(Box::new(DlManaged {
                tensor,
                manager,
                deleter: Some(delete_legacy),
            }))
            .cast::<c_void>(),
            c"dltensor",
        )
    };
    // SAFETY: the capsule stores the exact selected ABI pointer, owns its
    // destructor, and uses a static protocol name. No Python value is embedded.
    let capsule =
        unsafe { pyo3::ffi::PyCapsule_New(pointer, name.as_ptr(), Some(capsule_destructor)) };
    if capsule.is_null() {
        let failure = PyErr::fetch(py);
        // SAFETY: failed capsule construction did not transfer managed ownership.
        unsafe {
            if versioned {
                delete_versioned(pointer.cast());
            } else {
                delete_legacy(pointer.cast());
            }
        }
        return Err(failure);
    }
    // SAFETY: successful PyCapsule_New returned one owned Python reference.
    Ok(unsafe { Bound::<PyAny>::from_owned_ptr(py, capsule) }.unbind())
}

unsafe extern "C" fn close_retirement(_capsule: *mut pyo3::ffi::PyObject) {
    retirement::close();
}

#[cfg(feature = "testing")]
#[pyfunction]
fn _testing_dlpack_export_stats(py: Python<'_>) -> PyResult<Bound<'_, pyo3::types::PyDict>> {
    use pyo3::types::PyDict;
    let result = PyDict::new(py);
    for (id, values) in retirement::export_stats() {
        let row = PyDict::new(py);
        for (name, value) in ["created", "enqueued", "freed", "retained", "retained_bytes"]
            .into_iter()
            .zip(values)
        {
            row.set_item(name, value)?;
        }
        result.set_item(id, row)?;
    }
    Ok(result)
}

#[cfg(feature = "testing")]
#[pyfunction]
fn _testing_dlpack_retire_drain(py: Python<'_>, timeout_s: f64) -> PyResult<()> {
    let _guard = FpGuard::new();
    let timeout = std::time::Duration::try_from_secs_f64(timeout_s)
        .map_err(|_| PyValueError::new_err("timeout_s must be finite and nonnegative"))?;
    if py.detach(move || retirement::drain(timeout)) {
        Ok(())
    } else {
        Err(pyo3::exceptions::PyTimeoutError::new_err(
            "DLPack retirement is still pending",
        ))
    }
}

pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(_device_dlpack, module)?)?;
    // A static non-null marker needs no owned/Python/CUDA teardown. Its capsule
    // destructor only sets closing, even if other managed tensors outlive it.
    static LIFETIME: u8 = 0;
    let pointer = unsafe {
        pyo3::ffi::PyCapsule_New(
            (&LIFETIME as *const u8).cast_mut().cast(),
            c"seqblaze.dlpack_retirement".as_ptr(),
            Some(close_retirement),
        )
    };
    if pointer.is_null() {
        return Err(PyErr::fetch(module.py()));
    }
    let lifetime = unsafe { Bound::<PyAny>::from_owned_ptr(module.py(), pointer) };
    module.add("_dlpack_retirement_lifetime", lifetime)?;
    #[cfg(feature = "testing")]
    {
        module.add_function(wrap_pyfunction!(_testing_dlpack_export_stats, module)?)?;
        module.add_function(wrap_pyfunction!(_testing_dlpack_retire_drain, module)?)?;
    }
    Ok(())
}
