//! CPU Arrow PyCapsule input, using the stable C Data ABI without Arrow dependencies.
//! C layout declarations adapted from Apache Arrow (Apache-2.0):
//! https://arrow.apache.org/docs/format/CDataInterface.html
//! Capsule/ownership protocol: .../CDataInterface/PyCapsuleInterface.html.
pub(crate) mod export;
mod stream;
use super::{error, PySeqTable};
use pyo3::{
    prelude::*,
    types::{PyCapsule, PyCapsuleMethods, PyDict, PyTuple},
};
use seqblaze_core::{
    batch::SeqBatch,
    detail::try_reserve_exact,
    seq::{ArrowBuffer, ArrowDescriptor, SeqTable},
    validate::EngineError,
};
use std::{
    ffi::{c_char, c_int, c_void, CStr},
    mem::align_of,
    sync::Arc,
};

#[repr(C)]
struct ArrowSchema {
    format: *const c_char,
    name: *const c_char,
    metadata: *const c_char,
    flags: i64,
    n_children: i64,
    children: *mut *mut ArrowSchema,
    dictionary: *mut ArrowSchema,
    release: Option<unsafe extern "C" fn(*mut ArrowSchema)>,
    private_data: *mut c_void,
}
#[repr(C)]
struct ArrowArray {
    length: i64,
    null_count: i64,
    offset: i64,
    n_buffers: i64,
    n_children: i64,
    buffers: *const *const c_void,
    children: *mut *mut ArrowArray,
    dictionary: *mut ArrowArray,
    release: Option<unsafe extern "C" fn(*mut ArrowArray)>,
    private_data: *mut c_void,
}
#[repr(C)]
struct ArrowArrayStream {
    get_schema: Option<unsafe extern "C" fn(*mut ArrowArrayStream, *mut ArrowSchema) -> c_int>,
    get_next: Option<unsafe extern "C" fn(*mut ArrowArrayStream, *mut ArrowArray) -> c_int>,
    get_last_error: Option<unsafe extern "C" fn(*mut ArrowArrayStream) -> *const c_char>,
    release: Option<unsafe extern "C" fn(*mut ArrowArrayStream)>,
    private_data: *mut c_void,
}
struct ArrowOwner {
    schema: ArrowSchema,
    array: ArrowArray,
}
// SAFETY: this owner is exclusively responsible for immutable native exports.
// Workers only read the retained CPU buffers; mutation occurs only in exclusive
// Drop, which attaches Python before invoking producer release callbacks.
unsafe impl Send for ArrowOwner {}
unsafe impl Sync for ArrowOwner {}
impl Drop for ArrowOwner {
    fn drop(&mut self) {
        Python::attach(|_| {
            if let Some(release) = self.array.release {
                // SAFETY: this is the sole moved live base; producer releases children.
                unsafe { release(&mut self.array) };
            }
            if let Some(release) = self.schema.release {
                // SAFETY: same ownership rule; the producer supplied this callback.
                unsafe { release(&mut self.schema) };
            }
        });
    }
}
impl ArrowOwner {
    fn empty() -> Self {
        // SAFETY: these C ABI structures contain only integers, raw pointers and
        // nullable function pointers. A zeroed base is released/not yet live.
        unsafe {
            Self {
                schema: std::mem::zeroed(),
                array: std::mem::zeroed(),
            }
        }
    }
}
fn parameter(py: Python<'_>, message: impl Into<String>) -> PyErr {
    error(py, EngineError::Parameter(message.into()))
}

/// A producer whose export raises ImportError lacks an optional Arrow runtime;
/// pass that through unchanged so the caller may use its documented `__array__`
/// fallback (engine-layout §9). Every other export failure is a ParameterError.
fn export_error(py: Python<'_>, protocol: &str, failure: PyErr) -> PyErr {
    if failure.is_instance_of::<pyo3::exceptions::PyImportError>(py) {
        failure
    } else {
        parameter(py, format!("Arrow CPU {protocol} export failed: {failure}"))
    }
}

#[pyfunction]
pub fn _native_seqs_arrow<'py>(
    py: Python<'py>,
    object: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let export = object
        .call_method0("__arrow_c_array__")
        .map_err(|e| export_error(py, "array", e))?;
    let pair = export.cast::<PyTuple>().map_err(|_| {
        parameter(
            py,
            "__arrow_c_array__ must return (schema capsule, array capsule)",
        )
    })?;
    if pair.len() != 2 {
        return Err(parameter(py, "__arrow_c_array__ must return two capsules"));
    }
    let schema_object = pair.get_item(0)?;
    let array_object = pair.get_item(1)?;
    let schema_capsule = schema_object
        .cast::<PyCapsule>()
        .map_err(|_| parameter(py, "Arrow schema export must be a PyCapsule"))?;
    let array_capsule = array_object
        .cast::<PyCapsule>()
        .map_err(|_| parameter(py, "Arrow array export must be a PyCapsule"))?;
    let schema_pointer = schema_capsule
        .pointer_checked(Some(c"arrow_schema"))
        .map_err(|_| parameter(py, "expected an arrow_schema capsule"))?
        .cast::<ArrowSchema>()
        .as_ptr();
    let array_pointer = array_capsule
        .pointer_checked(Some(c"arrow_array"))
        .map_err(|_| parameter(py, "expected an arrow_array capsule"))?
        .cast::<ArrowArray>()
        .as_ptr();
    if (schema_pointer as usize) % align_of::<ArrowSchema>() != 0
        || (array_pointer as usize) % align_of::<ArrowArray>() != 0
    {
        return Err(parameter(py, "Arrow base structures must be aligned"));
    }
    // SAFETY: standard named capsules contain valid, aligned C Data structures.
    // No arbitrary Python code runs between pointer extraction and this move.
    let owner = unsafe {
        if (*schema_pointer).release.is_none() || (*array_pointer).release.is_none() {
            return Err(parameter(
                py,
                "Arrow capsules were already consumed or released",
            ));
        }
        let schema = schema_pointer.read();
        let array = array_pointer.read();
        (*schema_pointer).release = None;
        (*array_pointer).release = None;
        ArrowOwner { schema, array }
    };
    let (format, width) = schema_format(py, &owner.schema)?;
    if format == "n" {
        let rows = null_array_rows(py, &owner.array)?;
        return table_object(py, all_null_table(py, rows)?);
    }
    let descriptor = array_descriptor(py, &owner.array, format, width)?;
    // SAFETY: the sole moved owner retains the validated immutable CPU export.
    let table = unsafe { SeqTable::from_arrow_bytes(descriptor, Box::new(owner)) }
        .map_err(|e| error(py, e))?;
    table_object(py, table)
}

fn schema_format(py: Python<'_>, schema: &ArrowSchema) -> PyResult<(&'static str, usize)> {
    if schema.release.is_none()
        || schema.format.is_null()
        || schema.n_children != 0
        || !schema.dictionary.is_null()
    {
        return Err(parameter(
            py,
            "Arrow inputs require a live flat binary/string schema without a dictionary",
        ));
    }
    // SAFETY: the live native ArrowSchema retains its NUL-terminated format.
    match unsafe { CStr::from_ptr(schema.format) }.to_bytes() {
        b"z" => Ok(("z", 4)),
        b"Z" => Ok(("Z", 8)),
        b"u" => Ok(("u", 4)),
        b"U" => Ok(("U", 8)),
        b"vu" => Ok(("vu", 0)),
        b"vz" => Ok(("vz", 0)),
        // The Arrow null type: every slot is null (pandas exports an object
        // Series whose items are all missing this way; D86 (1)).
        b"n" => Ok(("n", 0)),
        _ => Err(parameter(
            py,
            "Arrow inputs require binary/large_binary/string/large_string/string_view/binary_view",
        )),
    }
}
/// The length of a live Arrow null-type array (format `n`). The C Data
/// Interface gives the null type no buffers; Polars (and older Arrow C++)
/// export one validity slot that is NULL, which Arrow C++ also accepts, so a
/// single NULL slot is read as "no buffers" (text-nulls-fp-environment.md §12).
fn null_array_rows(py: Python<'_>, array: &ArrowArray) -> PyResult<usize> {
    let buffers_ok = match array.n_buffers {
        0 => true,
        // SAFETY: a live export with n_buffers == 1 points `buffers` at one slot.
        1 => !array.buffers.is_null() && unsafe { (*array.buffers).is_null() },
        _ => false,
    };
    if array.release.is_none()
        || array.n_children != 0
        || !array.dictionary.is_null()
        || array.length < 0
        || array.offset < 0
        || !buffers_ok
    {
        return Err(parameter(py, "invalid Arrow null array metadata"));
    }
    usize::try_from(array.length).map_err(|_| {
        error(
            py,
            EngineError::Overflow("Arrow length exceeds platform limit".into()),
        )
    })
}

/// An owned table of `rows` null rows (empty spans, every row recorded null).
fn all_null_table(py: Python<'_>, rows: usize) -> PyResult<SeqTable> {
    let mut offsets = Vec::new();
    try_reserve_exact(&mut offsets, rows + 1, "null-table offsets").map_err(|e| error(py, e))?;
    offsets.resize(rows + 1, 0i64);
    let batch = SeqBatch::try_from_parts(Vec::new(), offsets)
        .map_err(|message| error(py, EngineError::Parameter(message.into())))?;
    let mut nulls = Vec::new();
    try_reserve_exact(&mut nulls, rows, "null-table rows").map_err(|e| error(py, e))?;
    nulls.extend(0..rows);
    SeqTable::with_nulls(batch, nulls, false).map_err(|e| error(py, e))
}

fn array_descriptor(
    py: Python<'_>,
    array: &ArrowArray,
    format: &'static str,
    width: usize,
) -> PyResult<ArrowDescriptor> {
    let is_view = matches!(format, "vu" | "vz");
    if array.release.is_none()
        || array.n_children != 0
        || !array.dictionary.is_null()
        || array.length < 0
        || array.offset < 0
        || array.null_count < -1
        || array.null_count > array.length
        || (if is_view {
            array.n_buffers < 3
        } else {
            array.n_buffers != 3
        })
        || array.buffers.is_null()
    {
        return Err(parameter(py, "invalid Arrow binary/string array metadata"));
    }
    let rows = usize::try_from(array.length).map_err(|_| {
        error(
            py,
            EngineError::Overflow("Arrow length exceeds platform limit".into()),
        )
    })?;
    let offset = usize::try_from(array.offset).map_err(|_| {
        error(
            py,
            EngineError::Overflow("Arrow offset exceeds platform limit".into()),
        )
    })?;
    if is_view {
        return view_descriptor(py, array, format, rows, offset);
    }
    // SAFETY: the live native binary/string base owns exactly three CPU buffer slots.
    let (validity, offsets, data) =
        unsafe { (*array.buffers, *array.buffers.add(1), *array.buffers.add(2)) };
    // The validity bitmap is authoritative and `null_count` is not consulted;
    // an absent bitmap means no nulls (text-nulls-fp-environment.md N1).
    Ok(ArrowDescriptor {
        address: data as usize,
        rows,
        offsets_address: offsets as usize,
        offsets_itemsize: width,
        offset,
        validity_address: validity as usize,
        format,
        view_buffers: None,
    })
}
fn view_descriptor(
    py: Python<'_>,
    array: &ArrowArray,
    format: &'static str,
    rows: usize,
    offset: usize,
) -> PyResult<ArrowDescriptor> {
    let overflow = || {
        error(
            py,
            EngineError::Overflow("Arrow view buffer metadata exceeds platform limits".into()),
        )
    };
    let slots = usize::try_from(array.n_buffers).map_err(|_| overflow())?;
    let slot_bytes = slots
        .checked_mul(std::mem::size_of::<*const c_void>())
        .ok_or_else(overflow)?;
    checked_span(py, array.buffers as usize, slot_bytes)?;
    let count = slots - 3;
    let size_bytes = count.checked_mul(8).ok_or_else(overflow)?;
    let mut buffers = Vec::new();
    try_reserve_exact(&mut buffers, count, "Arrow variadic buffer metadata")
        .map_err(|e| error(py, e))?;
    // SAFETY: a live native C Data export retains the declared pointer slots.
    let (validity, views, sizes) = unsafe {
        (
            *array.buffers,
            *array.buffers.add(1),
            *array.buffers.add(slots - 1),
        )
    };
    // The validity bitmap is authoritative and `null_count` is not consulted;
    // an absent bitmap means no nulls (text-nulls-fp-environment.md N1).
    checked_span(py, sizes as usize, size_bytes)?;
    for index in 0..count {
        // SAFETY: the native ABI retains count int64 sizes and count data pointers.
        let size = unsafe { (sizes as *const i64).add(index).read_unaligned() };
        let size = usize::try_from(size).map_err(|_| {
            parameter(py, format!("Arrow variadic buffer size at index {index} must be nonnegative and platform-sized"))
        })?;
        let address = unsafe { *array.buffers.add(index + 2) } as usize;
        buffers.push(ArrowBuffer { address, size });
    }
    Ok(ArrowDescriptor {
        address: views as usize,
        rows,
        offsets_address: 0,
        offsets_itemsize: 0,
        offset,
        validity_address: validity as usize,
        format,
        view_buffers: Some(Arc::new(buffers)),
    })
}
fn checked_span(py: Python<'_>, address: usize, size: usize) -> PyResult<()> {
    if size != 0
        && (address == 0 || size > isize::MAX as usize || address.checked_add(size).is_none())
    {
        return Err(error(
            py,
            EngineError::Overflow("Arrow view buffer address/span exceeds platform limits".into()),
        ));
    }
    Ok(())
}
fn table_object<'py>(py: Python<'py>, table: SeqTable) -> PyResult<Bound<'py, PyAny>> {
    Ok(Bound::new(
        py,
        PySeqTable {
            table: super::TableStorage::Host(Arc::new(table)),
        },
    )?
    .into_any())
}
pub(super) fn metadata<'py>(
    py: Python<'py>,
    descriptor: ArrowDescriptor,
) -> PyResult<Bound<'py, PyDict>> {
    let result = PyDict::new(py);
    if let Some(buffers) = &descriptor.view_buffers {
        result.set_item("kind", "arrow_view")?;
        result.set_item("format", descriptor.format)?;
        result.set_item("rows", descriptor.rows)?;
        result.set_item("offset", descriptor.offset)?;
        result.set_item("views_address", descriptor.address)?;
        let mut items = Vec::with_capacity(buffers.len());
        for buffer in buffers.iter() {
            let item = PyDict::new(py);
            item.set_item("address", buffer.address)?;
            item.set_item("size", buffer.size)?;
            items.push(item);
        }
        result.set_item("buffers", PyTuple::new(py, items)?)?;
        return Ok(result);
    }
    result.set_item("kind", "arrow")?;
    result.set_item("address", descriptor.address)?;
    result.set_item("rows", descriptor.rows)?;
    result.set_item("offsets_address", descriptor.offsets_address)?;
    result.set_item("offsets_itemsize", descriptor.offsets_itemsize)?;
    result.set_item("offset", descriptor.offset)?;
    result.set_item("format", descriptor.format)?;
    Ok(result)
}

pub fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(_native_seqs_arrow, module)?)?;
    stream::register(module)?;
    export::register(module)
}
