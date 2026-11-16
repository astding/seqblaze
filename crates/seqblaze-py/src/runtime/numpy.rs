//! Native NumPy byte views and one-time Unicode (UTF-8) / object conversion.
use super::{error, PySeqTable};
use pyo3::{
    prelude::*,
    types::{PyBytes, PyDict},
};
use seqblaze_core::{
    seq::{LengthDescriptor, MatrixDescriptor, SeqTable},
    validate::EngineError,
};

fn parameter(py: Python<'_>, message: impl Into<String>) -> PyErr {
    error(py, EngineError::Parameter(message.into()))
}

pub(super) fn native_view<'py>(
    py: Python<'py>,
    object: &Bound<'py, PyAny>,
    label: &str,
) -> PyResult<Bound<'py, PyAny>> {
    let numpy = py.import("numpy")?;
    let array_type = numpy.getattr("ndarray")?;
    if !object.is_instance(&array_type)? {
        return Err(parameter(
            py,
            format!("{label} must be a NumPy ndarray in this CPU adapter"),
        ));
    }
    if numpy
        .getattr("ma")?
        .getattr("isMaskedArray")?
        .call1((object,))?
        .is_truthy()?
    {
        return Err(parameter(
            py,
            format!("masked NumPy {label} arrays are not supported by this CPU adapter"),
        ));
    }
    let kwargs = PyDict::new(py);
    kwargs.set_item("type", &array_type)?;
    // Native view metadata bypasses ndarray-subclass Python overrides.
    array_type.getattr("view")?.call((object,), Some(&kwargs))
}

fn address(view: &Bound<'_, PyAny>) -> PyResult<usize> {
    let (pointer, _readonly): (usize, bool) = view
        .getattr("__array_interface__")?
        .get_item("data")?
        .extract()?;
    Ok(pointer)
}
fn retain(py: Python<'_>, view: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    Ok(py
        .import("builtins")?
        .getattr("memoryview")?
        .call1((view,))?
        .unbind())
}

#[pyfunction(signature=(object,lengths=None))]
pub fn _native_seqs_numpy<'py>(
    py: Python<'py>,
    object: &Bound<'py, PyAny>,
    lengths: Option<Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyAny>> {
    let view = native_view(py, object, "input")?;
    let ndim: usize = view.getattr("ndim")?.extract()?;
    let dtype = view.getattr("dtype")?;
    let kind: String = dtype.getattr("kind")?.extract()?;
    let itemsize: usize = dtype.getattr("itemsize")?.extract()?;
    if ndim == 1 && kind == "O" {
        if lengths.is_some() {
            return Err(parameter(
                py,
                "lengths is only supported with a uint8 matrix",
            ));
        }
        // Iterate the base ndarray with the GIL held: NumPy owns each Python
        // reference and supplies logical strided order. Never borrow raw object
        // pointers, invoke subclass iterators or manufacture a Python row list.
        // The shared collector retains only the final immutable Rust storage.
        return super::owned_sequence_table(py, view.try_iter()?);
    }
    let table=if ndim==1 && kind=="S" {
        if lengths.is_some() {return Err(parameter(py,"lengths is only supported with a uint8 matrix"));}
        let (rows,):(usize,)=view.getattr("shape")?.extract()?;
        let (stride,):(isize,)=view.getattr("strides")?.extract()?;
        let pointer=address(&view)?;
        let owner=retain(py,&view)?;
        // SAFETY: the native ndarray descriptor is valid and its live memoryview
        // owns the allocation; the contract forbids resize and active mutation.
        unsafe {SeqTable::from_fixed_bytes(pointer,rows,itemsize,stride,Box::new(owner))}
    } else if ndim==1 && kind=="U" {
        if lengths.is_some() {return Err(parameter(py,"lengths is only supported with a uint8 matrix"));}
        let (rows,):(usize,)=view.getattr("shape")?.extract()?;
        let (stride,):(isize,)=view.getattr("strides")?.extract()?;
        let byteorder:String=dtype.getattr("byteorder")?.extract()?;
        let big_endian=match byteorder.as_str() {
            ">"=>true,"<"=>false,"="|"|"=>cfg!(target_endian="big"),
            _=>return Err(parameter(py,"unsupported NumPy Unicode byte order")),
        };
        let pointer=address(&view)?;
        let owner=retain(py,&view)?;
        // SAFETY: the native view/export retains all UCS4 bytes while the GIL
        // is released for one Rust conversion into independent owned storage.
        py.detach(||{let _guard=seqblaze_core::fpenv::FpGuard::new(); unsafe {SeqTable::from_utf32_text(pointer,rows,itemsize,stride,big_endian,Box::new(owner))}})
    } else if ndim==2 && kind=="u" && itemsize==1 {
        let lengths=lengths.ok_or_else(||parameter(py,"uint8 matrices require explicit lengths; call seqs(matrix, lengths=L)"))?;
        let lengths_view=native_view(py,&lengths,"lengths")?;
        let length_dtype=lengths_view.getattr("dtype")?;
        let length_kind:String=length_dtype.getattr("kind")?.extract()?;
        let length_size:usize=length_dtype.getattr("itemsize")?.extract()?;
        let length_ndim:usize=lengths_view.getattr("ndim")?.extract()?;
        if length_ndim!=1 || length_kind!="i" || !matches!(length_size,4|8)
            || !length_dtype.getattr("isnative")?.is_truthy()? {
            return Err(parameter(py,"lengths must be a one-dimensional native-endian NumPy int32/int64 array"));
        }
        let (rows,columns):(usize,usize)=view.getattr("shape")?.extract()?;
        let (length_rows,):(usize,)=lengths_view.getattr("shape")?.extract()?;
        if length_rows!=rows {
            return Err(parameter(py,format!("lengths has {length_rows} values; uint8 matrix has {rows} rows")));
        }
        let (row_stride,column_stride):(isize,isize)=view.getattr("strides")?.extract()?;
        let (length_stride,):(isize,)=lengths_view.getattr("strides")?.extract()?;
        let descriptor=MatrixDescriptor {address:address(&view)?,rows,columns,row_stride,column_stride,
            lengths:LengthDescriptor {address:address(&lengths_view)?,rows:length_rows,itemsize:length_size,stride:length_stride}};
        let owners=(retain(py,&view)?,retain(py,&lengths_view)?);
        // SAFETY: both exact ndarray views provide valid native descriptors and
        // both memoryview exports remain owned by the core. Values are checked
        // before any prefix reads and refreshed on every execution.
        unsafe {SeqTable::from_matrix_bytes(descriptor,Box::new(owners))}
    } else {
        return Err(parameter(py,"NumPy inputs require a one-dimensional bytes/Unicode/object (S/U/O) array or a two-dimensional uint8 matrix with lengths in this CPU adapter"));
    }.map_err(|e|error(py,e))?;
    Ok(Bound::new(
        py,
        PySeqTable {
            table: super::TableStorage::Host(std::sync::Arc::new(table)),
        },
    )?
    .into_any())
}

/// (address, length) of a 1-D C-contiguous native NumPy array of `kind` and
/// `itemsize`; anything else is a ParameterError.
fn flat(
    py: Python<'_>,
    object: &Bound<'_, PyAny>,
    label: &str,
    kind: &str,
    itemsize: usize,
) -> PyResult<(usize, usize)> {
    let view = native_view(py, object, label)?;
    let dtype = view.getattr("dtype")?;
    let ok = view.getattr("ndim")?.extract::<usize>()? == 1
        && dtype.getattr("kind")?.extract::<String>()? == kind
        && dtype.getattr("itemsize")?.extract::<usize>()? == itemsize
        && dtype.getattr("isnative")?.is_truthy()?
        && view
            .getattr("flags")?
            .getattr("c_contiguous")?
            .is_truthy()?;
    if !ok {
        return Err(parameter(
            py,
            format!("{label} must be a 1-D C-contiguous native {kind}{itemsize} NumPy array"),
        ));
    }
    let (len,): (usize,) = view.getattr("shape")?.extract()?;
    Ok((address(&view)?, len))
}

/// The rows `values[starts[k]:ends[k]]`, concatenated in order, as one new
/// `bytes` object: one copy per row into the result, with no per-byte index
/// (`BinaryColumn.take`). Every row is checked against the values buffer.
#[pyfunction]
pub fn _gather_rows<'py>(
    py: Python<'py>,
    values: &Bound<'py, PyAny>,
    starts: &Bound<'py, PyAny>,
    ends: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyBytes>> {
    let (values_at, values_len) = flat(py, values, "values", "u", 1)?;
    let (starts_at, rows) = flat(py, starts, "starts", "i", 8)?;
    let (ends_at, end_rows) = flat(py, ends, "ends", "i", 8)?;
    if end_rows != rows {
        return Err(parameter(py, "starts and ends differ in length"));
    }
    let bounds = |at: usize| -> &[i64] {
        if rows == 0 {
            return &[];
        }
        // SAFETY: `flat` checked a live 1-D C-contiguous int64 array of `rows`
        // elements, kept alive by the caller's arguments for this call.
        unsafe { std::slice::from_raw_parts(at as *const i64, rows) }
    };
    let (starts, ends) = (bounds(starts_at), bounds(ends_at));
    let mut total = 0usize;
    for (k, (&start, &end)) in starts.iter().zip(ends).enumerate() {
        if start < 0 || end < start || end as u64 > values_len as u64 {
            return Err(parameter(
                py,
                format!(
                    "row {k}: [{start}, {end}) lies outside the {values_len}-byte values buffer"
                ),
            ));
        }
        total = total
            .checked_add((end - start) as usize)
            .ok_or_else(|| parameter(py, "the selected rows exceed the address space"))?;
    }
    PyBytes::new_with(py, total, |out| {
        if total == 0 {
            return Ok(());
        }
        py.detach(move || {
            // SAFETY: as above; the values buffer holds `values_len` bytes and
            // every row was checked to lie inside it.
            let (starts, ends, values) = unsafe {
                (
                    std::slice::from_raw_parts(starts_at as *const i64, rows),
                    std::slice::from_raw_parts(ends_at as *const i64, rows),
                    std::slice::from_raw_parts(values_at as *const u8, values_len),
                )
            };
            let mut at = 0;
            for (&start, &end) in starts.iter().zip(ends) {
                let row = &values[start as usize..end as usize];
                out[at..at + row.len()].copy_from_slice(row);
                at += row.len();
            }
        });
        Ok(())
    })
}

pub fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(_native_seqs_numpy, module)?)?;
    module.add_function(wrap_pyfunction!(_gather_rows, module)?)?;
    Ok(())
}
