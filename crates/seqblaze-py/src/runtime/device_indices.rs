//! Retained device index sources. Logical values are normalized on the call's
//! consumer stream; pair construction and replay never stage indices on CPU.
use super::device_lifetime::Detached;
use super::{
    device::{self, PyDeviceColumn},
    error,
    indices::PyIndexColumn,
    PySeqTable,
};
use pyo3::{
    prelude::*,
    types::{PyBytes, PyDict},
};
use seqblaze_core::{fpenv::FpGuard, validate::EngineError};
use seqblaze_cuda::{
    interop::{DeviceArrayDescriptor, DeviceColumn, DeviceDType, DeviceLease},
    resident_indices,
};
use std::sync::atomic::{AtomicU8, Ordering};

pub(super) enum SelectionSource {
    Owned(DeviceColumn),
    External(seqblaze_cuda::interop::DeviceArray),
}

enum Source {
    External {
        object: Py<PyAny>,
        descriptor: DeviceArrayDescriptor,
    },
    Owned {
        column: Detached<DeviceColumn>,
        elements: usize,
    },
}
#[pyclass(frozen, module = "seqblaze._core", name = "_DeviceIndexColumn")]
pub(super) struct PyDeviceIndexColumn {
    source: Source,
    ordinal: usize,
    len: usize,
}
impl PyDeviceIndexColumn {
    /// Selection applies its own negative-index and IndexError rules to this
    /// raw retained source, rather than reusing pair normalization semantics.
    pub(super) fn selection_source(
        &self,
        py: Python<'_>,
        lease: &DeviceLease,
    ) -> PyResult<SelectionSource> {
        if self.ordinal != lease.device().info().ordinal {
            return Err(error(
                py,
                EngineError::Device("device index inputs belong to different CUDA devices".into()),
            ));
        }
        match &self.source {
            Source::Owned { column, elements } => {
                if column.shape() == [*elements] {
                    Ok(SelectionSource::Owned((**column).clone()))
                } else {
                    Ok(SelectionSource::External(
                        resident_indices::owned_flat_view(column, *elements)
                            .map_err(|e| error(py, e))?,
                    ))
                }
            }
            Source::External { object, descriptor } => {
                let current = device::array_on_stream(py, object.bind(py), lease)?;
                let now = current.descriptor();
                if now.address != descriptor.address
                    || now.shape != descriptor.shape
                    || now.strides != descriptor.strides
                    || now.dtype != descriptor.dtype
                    || now.ordinal != descriptor.ordinal
                {
                    return Err(error(
                        py,
                        EngineError::Pairing(
                            "device index buffer was resized or its layout changed".into(),
                        ),
                    ));
                }
                Ok(SelectionSource::External(current))
            }
        }
    }
    pub(super) fn normalize(
        &self,
        py: Python<'_>,
        lease: &mut DeviceLease,
        label: &str,
    ) -> PyResult<DeviceColumn> {
        if self.ordinal != lease.device().info().ordinal {
            return Err(error(
                py,
                EngineError::Device("device index inputs belong to different CUDA devices".into()),
            ));
        }
        match &self.source {
            Source::Owned { column, elements } => py
                .detach(|| resident_indices::normalize_owned_flat(lease, column, *elements, label))
                .map_err(|e| error(py, e)),
            Source::External { object, descriptor } => {
                let current = device::array_on_stream(py, object.bind(py), lease)?;
                let now = current.descriptor();
                if now.address != descriptor.address
                    || now.shape != descriptor.shape
                    || now.strides != descriptor.strides
                    || now.dtype != descriptor.dtype
                    || now.ordinal != descriptor.ordinal
                {
                    return Err(error(
                        py,
                        EngineError::Pairing(format!(
                            "{label} device buffer was resized or its layout changed"
                        )),
                    ));
                }
                py.detach(|| {
                    let _fp = FpGuard::new();
                    resident_indices::normalize(lease, current, label)
                })
                .map_err(|e| error(py, e))
            }
        }
    }
}
#[pymethods]
impl PyDeviceIndexColumn {
    fn __len__(&self) -> usize {
        self.len
    }
    #[getter]
    fn _device_ordinal(&self) -> usize {
        self.ordinal
    }
    #[getter]
    fn _source_kind(&self) -> &'static str {
        "device"
    }
    #[getter]
    fn storage_info<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let info = PyDict::new(py);
        info.set_item("kind", "device_index")?;
        info.set_item("len", self.len)?;
        info.set_item("device", format!("cuda:{}", self.ordinal))?;
        Ok(info)
    }
    fn _device_index_digest<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        self.digest(py, false)
    }
    fn _device_index_snapshot<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        self.digest(py, true)
    }
}
impl PyDeviceIndexColumn {
    fn digest<'py>(&self, py: Python<'py>, snapshot: bool) -> PyResult<Bound<'py, PyDict>> {
        let dev = super::resolve_device(py, Some(self.ordinal as u64))?.expect("ordinal supplied");
        let mut lease = py
            .detach(|| DeviceLease::new(&dev))
            .map(Detached::new)
            .map_err(|e| error(py, e))?;
        // Normalization creates the owned D2D snapshot before any explicit D2H.
        let column = Detached::new(self.normalize(py, &mut lease, "indices")?);
        let digest = py
            .detach(|| seqblaze_cuda::interop::device_column_index_digest(&lease, &column))
            .map_err(|e| error(py, e))?;
        let result = PyDict::new(py);
        result.set_item(
            "sha256",
            digest
                .iter()
                .map(|b| format!("{b:02x}"))
                .collect::<String>(),
        )?;
        result.set_item("n", self.len)?;
        if snapshot {
            let data = py
                .detach(|| column.to_host_bytes())
                .map_err(|e| error(py, e))?;
            result.set_item("data", PyBytes::new(py, &data))?;
        }
        Ok(result)
    }
}
#[pyfunction(signature=(object,label,ordinal,elements=None))]
fn _native_indices_device(
    py: Python<'_>,
    object: &Bound<'_, PyAny>,
    label: &str,
    ordinal: usize,
    elements: Option<usize>,
) -> PyResult<PyDeviceIndexColumn> {
    let dev = super::resolve_device(py, Some(ordinal as u64))?.expect("ordinal supplied");
    let mut lease = py
        .detach(|| DeviceLease::new(&dev))
        .map(Detached::new)
        .map_err(|e| error(py, e))?;
    let (source, len) = if let Ok(owned) = object.extract::<PyRef<'_, PyDeviceColumn>>() {
        let column = owned.column.clone();
        if elements.is_none() && column.shape().len() != 1 {
            return Err(error(
                py,
                EngineError::Pairing(format!(
                    "{label} must be one-dimensional integers; use np.array([0], dtype='int64')"
                )),
            ));
        }
        let elements = elements.unwrap_or(column.nbytes() / column.dtype().width());
        let normalized = py
            .detach(|| resident_indices::normalize_owned_flat(&mut lease, &column, elements, label))
            .map(Detached::new)
            .map_err(|e| error(py, e))?;
        (
            Source::Owned {
                column: Detached::new(column),
                elements,
            },
            normalized.shape()[0],
        )
    } else {
        let array = device::array_on_stream(py, object, &lease)?;
        let descriptor = array.descriptor().clone();
        let normalized = py
            .detach(|| resident_indices::normalize(&mut lease, array, label))
            .map(Detached::new)
            .map_err(|e| error(py, e))?;
        (
            Source::External {
                object: object.clone().unbind(),
                descriptor,
            },
            normalized.shape()[0],
        )
    };
    Ok(PyDeviceIndexColumn {
        source,
        ordinal,
        len,
    })
}

/// Host-origin indices may be uploaded once into a mixed resident call. Use the
/// native descriptor's byte-wise atomic reads, including negative values, before
/// GPU bounds validation; IndexArray::get assumes prevalidated nonnegative input.
pub(super) fn normalized(
    py: Python<'_>,
    lease: &mut DeviceLease,
    object: &Bound<'_, PyAny>,
    label: &str,
) -> PyResult<DeviceColumn> {
    if let Ok(column) = object.extract::<PyRef<'_, PyDeviceIndexColumn>>() {
        return column.normalize(py, lease, label);
    }
    let input = object.extract::<PyRef<'_, PyIndexColumn>>()?;
    let info = input.column.storage_info();
    let capacity = info
        .len
        .checked_mul(8)
        .filter(|n| *n <= isize::MAX as usize)
        .ok_or_else(|| {
            error(
                py,
                EngineError::Overflow("resident index bytes exceed platform limits".into()),
            )
        })?;
    let mut bytes = Vec::new();
    seqblaze_core::detail::try_reserve_exact(&mut bytes, capacity, "host index upload")
        .map_err(|e| error(py, e))?;
    for row in 0..info.len {
        let pointer = (info.address as *const u8).wrapping_offset(row as isize * info.stride);
        let mut word = [0u8; 8];
        for (offset, byte) in word[..info.itemsize].iter_mut().enumerate() {
            // SAFETY: IndexArray retains a validated native exporter throughout
            // this GIL-held scan; byte loads allow unaligned descriptors.
            *byte = unsafe {
                (&*pointer.wrapping_add(offset).cast::<AtomicU8>()).load(Ordering::Relaxed)
            };
        }
        let value = if info.itemsize == 4 {
            i32::from_ne_bytes(word[..4].try_into().expect("four bytes")) as i64
        } else {
            i64::from_ne_bytes(word)
        };
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    py.detach(|| {
        let _fp = FpGuard::new();
        lease.upload_bytes(DeviceDType::I64, vec![info.len], &bytes)
    })
    .map_err(|e| error(py, e))
}
#[pyfunction]
fn _native_pairs_device(
    py: Python<'_>,
    a: PyRef<'_, PySeqTable>,
    b: PyRef<'_, PySeqTable>,
    i: &Bound<'_, PyAny>,
    j: &Bound<'_, PyAny>,
    ordinal: usize,
) -> PyResult<usize> {
    let dev = super::resolve_device(py, Some(ordinal as u64))?.expect("ordinal supplied");
    let mut lease = py
        .detach(|| DeviceLease::new(&dev))
        .map(Detached::new)
        .map_err(|e| error(py, e))?;
    let i = Detached::new(normalized(py, &mut lease, i, "i")?);
    let j = Detached::new(normalized(py, &mut lease, j, "j")?);
    let shape = [a.table.len(), b.table.len()];
    py.detach(|| {
        let _fp = FpGuard::new();
        resident_indices::validate_pair_indices(&mut lease, &i, &j, shape)
    })
    .map_err(|e| error(py, e))?;
    Ok(i.shape()[0])
}
pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyDeviceIndexColumn>()?;
    module.add_function(wrap_pyfunction!(_native_indices_device, module)?)?;
    module.add_function(wrap_pyfunction!(_native_pairs_device, module)?)?;
    Ok(())
}
