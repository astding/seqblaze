//! Native index normalization with a retained contiguous exporter.
use super::{column, error, Numeric};
use pyo3::{prelude::*, types::PyDict};
use seqblaze_core::{seq::LengthDescriptor, validate::EngineError, work::IndexArray};
use std::sync::atomic::{AtomicU8, Ordering};

#[pyclass(frozen, module = "seqblaze._core", name = "_IndexColumn")]
pub struct PyIndexColumn {
    pub(super) column: IndexArray,
}

#[pymethods]
impl PyIndexColumn {
    fn __len__(&self) -> usize {
        self.column.len()
    }
    #[getter]
    fn storage_info<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let info = self.column.storage_info();
        let result = PyDict::new(py);
        result.set_item("kind", "numpy_index")?;
        result.set_item("address", info.address)?;
        result.set_item("len", info.len)?;
        result.set_item("itemsize", info.itemsize)?;
        result.set_item("stride", info.stride)?;
        result.set_item("copied", info.copied)?;
        Ok(result)
    }
}

#[pyfunction]
pub fn _native_indices(
    py: Python<'_>,
    object: &Bound<'_, PyAny>,
    label: &str,
) -> PyResult<PyIndexColumn> {
    let fail = || {
        error(
            py,
            EngineError::Pairing(format!(
                "{label} must be a one-dimensional native-endian NumPy int32/int64 array"
            )),
        )
    };
    let numpy = py.import("numpy")?;
    let array_type = numpy.getattr("ndarray")?;
    if !object.is_instance(&array_type)?
        || numpy
            .getattr("ma")?
            .getattr("isMaskedArray")?
            .call1((object,))?
            .is_truthy()?
    {
        return Err(fail());
    }
    let kwargs = PyDict::new(py);
    kwargs.set_item("type", &array_type)?;
    let view = array_type.getattr("view")?.call((object,), Some(&kwargs))?;
    let ndim: usize = view.getattr("ndim")?.extract()?;
    let dtype = view.getattr("dtype")?;
    let kind: String = dtype.getattr("kind")?.extract()?;
    let itemsize: usize = dtype.getattr("itemsize")?.extract()?;
    if ndim != 1
        || kind != "i"
        || !matches!(itemsize, 4 | 8)
        || !dtype.getattr("isnative")?.is_truthy()?
    {
        return Err(fail());
    }
    let copied = !view
        .getattr("flags")?
        .getattr("c_contiguous")?
        .is_truthy()?;
    let view = if copied {
        let kwargs = PyDict::new(py);
        kwargs.set_item("order", "C")?;
        view.call_method("copy", (), Some(&kwargs))?
    } else {
        view
    };
    let (rows,): (usize,) = view.getattr("shape")?.extract()?;
    let (stride,): (isize,) = view.getattr("strides")?.extract()?;
    let (address, _readonly): (usize, bool) = view
        .getattr("__array_interface__")?
        .get_item("data")?
        .extract()?;
    let owner = py
        .import("builtins")?
        .getattr("memoryview")?
        .call1((&view,))?
        .unbind();
    let descriptor = LengthDescriptor {
        address,
        rows,
        itemsize,
        stride,
    };
    // SAFETY: the exact native ndarray view supplies the descriptor, and its
    // memoryview export retains all bytes. Active mutation is outside the contract.
    let column = unsafe { IndexArray::from_borrowed(descriptor, Box::new(owner), copied) }
        .map_err(|e| error(py, e))?;
    Ok(PyIndexColumn { column })
}

/// Project RowStats' non-negative argbest values to immutable int64 pair columns.
/// Keep the GIL and the input export for both integer scans. No Python row
/// callbacks, NumPy ufuncs or floating-point operations run while reading values.
#[pyfunction]
pub fn _native_rowstats_pairs<'py>(
    py: Python<'py>,
    argbest: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyDict>> {
    let fail = || {
        error(
            py,
            EngineError::Parameter(
                "argbest must be a one-dimensional native-endian NumPy int64 array".into(),
            ),
        )
    };
    let numpy = py.import("numpy")?;
    let array_type = numpy.getattr("ndarray")?;
    if !argbest.is_instance(&array_type)?
        || numpy
            .getattr("ma")?
            .getattr("isMaskedArray")?
            .call1((argbest,))?
            .is_truthy()?
    {
        return Err(fail());
    }
    let kwargs = PyDict::new(py);
    kwargs.set_item("type", &array_type)?;
    let view = array_type
        .getattr("view")?
        .call((argbest,), Some(&kwargs))?;
    let dtype = view.getattr("dtype")?;
    if view.getattr("ndim")?.extract::<usize>()? != 1
        || dtype.getattr("kind")?.extract::<String>()? != "i"
        || dtype.getattr("itemsize")?.extract::<usize>()? != 8
        || !dtype.getattr("isnative")?.is_truthy()?
    {
        return Err(fail());
    }
    let (rows,): (usize,) = view.getattr("shape")?.extract()?;
    let (stride,): (isize,) = view.getattr("strides")?.extract()?;
    let (address, _readonly): (usize, bool) = view
        .getattr("__array_interface__")?
        .get_item("data")?
        .extract()?;
    let owner = py
        .import("builtins")?
        .getattr("memoryview")?
        .call1((&view,))?
        .unbind();
    let descriptor = LengthDescriptor {
        address,
        rows,
        itemsize: 8,
        stride,
    };
    // SAFETY: the base ndarray view supplies its native descriptor, and its
    // memoryview retains the allocation. IndexArray validates dimensions and
    // byte spans. The GIL remains held; mutation during a call is forbidden.
    let input = unsafe { IndexArray::from_borrowed(descriptor, Box::new(owner), false) }
        .map_err(|e| error(py, e))?;
    let info = input.storage_info();
    let read = |row: usize| {
        debug_assert!(row < info.len);
        let pointer = (info.address as *const u8).wrapping_offset(row as isize * info.stride);
        let mut bytes = [0u8; 8];
        for (offset, byte) in bytes.iter_mut().enumerate() {
            // SAFETY: validated geometry and the retained export cover this
            // element, even when strided or unaligned. Match LengthDescriptor's
            // relaxed byte reads; no shared Rust slice spans foreign storage.
            *byte = unsafe {
                (&*pointer.wrapping_add(offset).cast::<AtomicU8>()).load(Ordering::Relaxed)
            };
        }
        i64::from_ne_bytes(bytes)
    };
    let count = (0..rows).filter(|&row| read(row) >= 0).count();
    let bytes = count
        .checked_mul(std::mem::size_of::<i64>())
        .and_then(|bytes| bytes.checked_mul(2))
        .ok_or_else(|| {
            error(
                py,
                EngineError::Parameter("RowStats pair index bytes exceed platform limits".into()),
            )
        })?;
    let allocation = |reason| {
        error(
            py,
            seqblaze_core::detail::device_error(format_args!(
                "unable to reserve {bytes} bytes for RowStats pair index columns: {reason}; \
                 no result was published"
            )),
        )
    };
    let mut i = Vec::new();
    let mut j = Vec::new();
    i.try_reserve_exact(count).map_err(allocation)?;
    j.try_reserve_exact(count).map_err(allocation)?;
    for row in 0..rows {
        let partner = read(row);
        if partner >= 0 {
            // Descriptor validation bounds rows by isize::MAX, hence int64.
            i.push(row as i64);
            j.push(partner);
        }
    }
    let columns = PyDict::new(py);
    columns.set_item("i", column(py, Numeric::I64(i))?)?;
    columns.set_item("j", column(py, Numeric::I64(j))?)?;
    Ok(columns)
}

pub fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyIndexColumn>()?;
    module.add_function(wrap_pyfunction!(_native_indices, module)?)?;
    module.add_function(wrap_pyfunction!(_native_rowstats_pairs, module)?)?;
    Ok(())
}
