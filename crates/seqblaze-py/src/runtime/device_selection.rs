//! Python bridge for device-owned selection. This module is registered only
//! after the resident selection kernels and shared index helpers are present.
use super::device_lifetime::Detached;
use super::{
    device::{self, PyDeviceColumn},
    error,
};
use pyo3::{
    exceptions::PyIndexError,
    prelude::*,
    types::{PyBytes, PyDict, PyTuple},
};
use seqblaze_core::validate::EngineError;
use seqblaze_cuda::{
    interop::{DeviceColumn, DeviceDType, DeviceLease},
    resident::{ResidentField, ResidentRaggedField, ResidentRaggedKind},
    resident_selection::{self, SelectionError, TopKSelection},
};

fn parameter(py: Python<'_>, message: &str) -> PyErr {
    error(py, EngineError::Parameter(message.into()))
}
fn column(py: Python<'_>, object: &Bound<'_, PyAny>) -> PyResult<DeviceColumn> {
    object
        .extract::<PyRef<'_, PyDeviceColumn>>()
        .map(|owner| owner.column.clone())
        .map_err(|_| {
            parameter(
                py,
                "device selection requires immutable native result owners",
            )
        })
}
fn wrap(py: Python<'_>, column: DeviceColumn) -> PyResult<Py<PyDeviceColumn>> {
    Py::new(
        py,
        PyDeviceColumn {
            column: Detached::new(column),
        },
    )
}
fn selection_error(py: Python<'_>, cause: SelectionError) -> PyErr {
    match cause {
        SelectionError::Bounds { position, rows } => PyIndexError::new_err(format!(
            "selection index at position {position} is out of bounds for axis 0 with size {rows}"
        )),
        SelectionError::Engine(cause) => error(py, cause),
    }
}
fn parse_fields(
    py: Python<'_>,
    columns: &Bound<'_, PyDict>,
) -> PyResult<(Vec<ResidentField>, Vec<ResidentRaggedField>, usize)> {
    let mut numeric = Vec::new();
    let mut ragged = Vec::new();
    let mut ordinal = None;
    numeric
        .try_reserve_exact(columns.len())
        .map_err(|_| error(py, EngineError::Device(String::new())))?;
    ragged
        .try_reserve_exact(columns.len())
        .map_err(|_| error(py, EngineError::Device(String::new())))?;
    for (name, value) in columns.iter() {
        let name: String = name.extract()?;
        if name.starts_with("__selection_pair_") {
            return Err(parameter(py, "reserved internal selection column name"));
        }
        if let Ok(parts) = value.cast::<PyTuple>() {
            if parts.len() != 4 {
                return Err(parameter(
                    py,
                    "device ragged selection requires four buffers",
                ));
            }
            let kind: String = parts.get_item(0)?.extract()?;
            let kind = match kind.as_str() {
                "string" => ResidentRaggedKind::Utf8,
                "binary" => ResidentRaggedKind::Binary,
                "list" => ResidentRaggedKind::ListI64,
                _ => return Err(parameter(py, "unknown device ragged selection kind")),
            };
            let offsets = column(py, &parts.get_item(1)?)?;
            let values = column(py, &parts.get_item(2)?)?;
            let valid = parts.get_item(3)?;
            let validity = if valid.is_none() {
                None
            } else {
                Some(column(py, &valid)?)
            };
            ordinal.get_or_insert(offsets.ordinal());
            ragged.push(ResidentRaggedField {
                name,
                kind,
                offsets,
                values,
                validity,
            });
        } else {
            let values = column(py, &value)?;
            ordinal.get_or_insert(values.ordinal());
            numeric.push(ResidentField {
                name,
                values,
                mask_bits: 0,
            });
        }
    }
    let ordinal = ordinal
        .ok_or_else(|| parameter(py, "device selection requires at least one result column"))?;
    Ok((numeric, ragged, ordinal))
}

// Mirror the existing binding's explicit selection controls without changing its ABI.
#[allow(clippy::too_many_arguments)]
#[pyfunction(signature = (columns, selector, parent_rows, *, allow_mask=false,
    canonical=false, topk_rows=None, k=None, host_normalized=false,
    host_masked=false, pair_kind=None, pair_columns=None, selector_rows=None))]
fn _native_device_selection<'py>(
    py: Python<'py>,
    columns: &Bound<'py, PyDict>,
    selector: &Bound<'py, PyAny>,
    parent_rows: usize,
    allow_mask: bool,
    canonical: bool,
    topk_rows: Option<usize>,
    k: Option<usize>,
    host_normalized: bool,
    host_masked: bool,
    pair_kind: Option<&str>,
    pair_columns: Option<&Bound<'py, PyTuple>>,
    selector_rows: Option<usize>,
) -> PyResult<Bound<'py, PyDict>> {
    let (mut numeric, ragged, ordinal) = parse_fields(py, columns)?;
    if selector_rows.is_some()
        && (host_normalized || selector.extract::<PyRef<'_, PyDeviceColumn>>().is_err())
    {
        return Err(parameter(
            py,
            "device selector reshape metadata requires its immutable native owner",
        ));
    }
    let options = match (topk_rows, k) {
        (None, None) => None,
        (Some(rows), Some(k)) if k > 0 => Some(TopKSelection { rows, k }),
        _ => {
            return Err(parameter(
                py,
                "TopK selection requires original rows and positive k together",
            ))
        }
    };
    if host_masked && (!host_normalized || !allow_mask) {
        return Err(parameter(
            py,
            "invalid host selection mask normalization state",
        ));
    }
    match pair_kind {
        None | Some("zip" | "one_to_many") => {
            if pair_columns.is_some() {
                return Err(parameter(
                    py,
                    "implicit selected pairs do not need source index columns",
                ));
            }
        }
        Some("index") => {
            let source = pair_columns.ok_or_else(|| {
                parameter(
                    py,
                    "Each index selection requires retained call-time device pair indices",
                )
            })?;
            if source.len() != 2 {
                return Err(parameter(py, "selected pair mapping requires i and j"));
            }
            numeric
                .try_reserve_exact(2)
                .map_err(|_| error(py, EngineError::Device(String::new())))?;
            for (position, name) in ["__selection_pair_i", "__selection_pair_j"]
                .into_iter()
                .enumerate()
            {
                let values = column(py, &source.get_item(position)?)?;
                if values.dtype() != DeviceDType::I64 {
                    return Err(parameter(
                        py,
                        "selected call-time pair indices must be int64",
                    ));
                }
                numeric.push(ResidentField {
                    name: name.into(),
                    values,
                    mask_bits: 0,
                });
            }
        }
        _ => {
            return Err(parameter(
                py,
                "unknown Each pair mapping for device selection",
            ))
        }
    }
    let device = super::resolve_device(py, Some(ordinal as u64))?.expect("ordinal supplied");
    let mut lease = py
        .detach(|| DeviceLease::new(&device))
        .map(Detached::new)
        .map_err(|e| error(py, e))?;
    let selection = if host_normalized {
        let bytes = selector
            .cast::<PyBytes>()
            .map_err(|_| parameter(py, "normalized host selection requires int64 bytes"))?
            .as_bytes();
        if bytes.len() % 8 != 0 {
            return Err(parameter(
                py,
                "normalized host index bytes require int64 elements",
            ));
        }
        let mut owned = Vec::new();
        owned
            .try_reserve_exact(bytes.len())
            .map_err(|_| error(py, EngineError::Device(String::new())))?;
        owned.extend_from_slice(bytes);
        let rows = owned.len() / 8;
        let indices = py
            .detach(|| lease.upload_bytes(DeviceDType::I64, vec![rows], &owned))
            .map_err(|e| error(py, e))?;
        py.detach(|| {
            resident_selection::normalize_owned(
                &mut lease,
                &indices,
                parent_rows,
                false,
                canonical && !host_masked,
            )
        })
        .map_err(|e| selection_error(py, e))?
    } else if let Ok(owner) = selector.extract::<PyRef<'_, PyDeviceColumn>>() {
        let indices = owner.column.clone();
        py.detach(|| match selector_rows {
            Some(elements) => resident_selection::normalize_owned_flat(
                &mut lease,
                &indices,
                elements,
                parent_rows,
                allow_mask,
                canonical,
            ),
            None => resident_selection::normalize_owned(
                &mut lease,
                &indices,
                parent_rows,
                allow_mask,
                canonical,
            ),
        })
        .map_err(|e| selection_error(py, e))?
    } else if let Ok(indices) =
        selector.extract::<PyRef<'_, super::device_indices::PyDeviceIndexColumn>>()
    {
        match indices.selection_source(py, &lease)? {
            super::device_indices::SelectionSource::Owned(source) => py
                .detach(|| {
                    resident_selection::normalize_owned(
                        &mut lease,
                        &source,
                        parent_rows,
                        allow_mask,
                        canonical,
                    )
                })
                .map_err(|e| selection_error(py, e))?,
            super::device_indices::SelectionSource::External(source) => py
                .detach(|| {
                    resident_selection::normalize_selector(
                        &mut lease,
                        source,
                        parent_rows,
                        allow_mask,
                        canonical,
                    )
                })
                .map_err(|e| selection_error(py, e))?,
        }
    } else {
        let source = device::array_on_stream(py, selector, &lease)?;
        py.detach(|| {
            resident_selection::normalize_selector(
                &mut lease,
                source,
                parent_rows,
                allow_mask,
                canonical,
            )
        })
        .map_err(|e| selection_error(py, e))?
    };
    let selected = selection.indices().shape()[0];
    let masked = host_masked || selection.masked();
    let mut result = py
        .detach(|| {
            resident_selection::gather_columns(&mut lease, &selection, &numeric, &ragged, options)
        })
        .map_err(|e| error(py, e))?;
    let pair_indices = match pair_kind {
        Some("zip") => Some((selection.indices().clone(), selection.indices().clone())),
        Some("one_to_many") => Some((
            py.detach(|| resident_selection::zero_indices(&lease, selected))
                .map_err(|e| error(py, e))?,
            selection.indices().clone(),
        )),
        Some("index") => {
            let mut take = |name: &str| {
                let at = result
                    .numeric
                    .iter()
                    .position(|field| field.name == name)
                    .ok_or_else(|| parameter(py, "missing selected call-time pair mapping"))?;
                Ok::<_, PyErr>(result.numeric.remove(at).values)
            };
            Some((take("__selection_pair_i")?, take("__selection_pair_j")?))
        }
        _ => None,
    };
    let output = PyDict::new(py);
    let columns = PyDict::new(py);
    for field in result.numeric {
        columns.set_item(field.name, wrap(py, field.values)?)?;
    }
    for field in result.ragged {
        let kind = match field.kind {
            ResidentRaggedKind::Utf8 => "string",
            ResidentRaggedKind::Binary => "binary",
            ResidentRaggedKind::ListI64 => "list",
        };
        let offsets = wrap(py, field.offsets)?;
        let values = wrap(py, field.values)?;
        let validity = field.validity.map(|v| wrap(py, v)).transpose()?;
        columns.set_item(field.name, (kind, offsets, values, validity))?;
    }
    output.set_item("columns", columns)?;
    output.set_item("indices", wrap(py, selection.indices().clone())?)?;
    output.set_item(
        "indices_sha256",
        selection
            .sha256()
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect::<String>(),
    )?;
    output.set_item("index_hash_algorithm", "merkle-sha256-v1")?;
    output.set_item("masked", masked)?;
    output.set_item("tie_rows", result.tied_rows)?;
    output.set_item(
        "pair_indices",
        pair_indices
            .map(|(i, j)| Ok::<_, PyErr>((wrap(py, i)?, wrap(py, j)?)))
            .transpose()?,
    )?;
    Ok(output)
}

#[pyfunction]
fn _native_device_rowstats_pairs<'py>(
    py: Python<'py>,
    argbest: &PyDeviceColumn,
) -> PyResult<Bound<'py, PyDict>> {
    let device = super::resolve_device(py, Some(argbest.column.ordinal() as u64))?
        .expect("ordinal supplied");
    let mut lease = py
        .detach(|| DeviceLease::new(&device))
        .map(Detached::new)
        .map_err(|e| error(py, e))?;
    let source = argbest.column.clone();
    let (i, j) = py
        .detach(|| resident_selection::rowstats_pairs(&mut lease, &source))
        .map_err(|e| error(py, e))?;
    let output = PyDict::new(py);
    output.set_item("i", wrap(py, i)?)?;
    output.set_item("j", wrap(py, j)?)?;
    Ok(output)
}

pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(_native_device_selection, module)?)?;
    module.add_function(wrap_pyfunction!(_native_device_rowstats_pairs, module)?)?;
    Ok(())
}
