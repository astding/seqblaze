//! Shared CPU runtime. Only the probe dispatch and PF-RUN aliases need `testing`.
mod arrow;
pub(crate) mod cancel;
#[cfg(feature = "cuda")]
mod device;
#[cfg(feature = "cuda")]
mod device_dlpack;
#[cfg(feature = "cuda")]
mod device_indices;
#[cfg(feature = "cuda")]
mod device_lifetime;
#[cfg(feature = "cuda")]
mod device_run;
#[cfg(feature = "cuda")]
mod device_selection;
#[cfg(feature = "cuda")]
mod device_tiles;
mod duplex;
mod fastx;
mod fpbracket;
mod hairpin;
mod indices;
mod manifest;
mod numpy;
#[cfg(feature = "cuda")]
mod pf_device_finalize;
mod provenance;
mod selection;
use indices::PyIndexColumn;
mod tiles;
mod tm;
use std::sync::Arc;

use pyo3::{
    conversion::FromPyObjectOwned,
    exceptions::{PyMemoryError, PyUnicodeEncodeError},
    prelude::*,
    types::{PyBytes, PyDict, PyList, PyString, PyTuple},
};
use seqblaze_align::{Align, Compat, Config as AlignConfig, Substitution};
use seqblaze_core::{
    batch::SeqBatch,
    detail::{DetailColumns, DetailData, FieldKind},
    exec::{self, ExecOptions, Grain, Strategy},
    fpenv::{self, FpGuard},
    kernel::{PairDistanceKernel, Score},
    plan::{BoundAvailability, ExecutionPlan, OutputDemand},
    seq::SeqTable,
    sink::{RunOutput, Sink},
    validate::EngineError,
    work::{Orientation, PairKind, WorkSource},
};
use seqblaze_cuda::{
    AlignCall, AlignKind, DeviceCall, DuplexCall, EditMode, PfDeviceCall, RegisterCall,
};
use seqblaze_edit::levenshtein::{self, Levenshtein, Mode};

fn error(py: Python<'_>, error: EngineError) -> PyErr {
    let name = match &error {
        EngineError::Parameter(_) => "ParameterError",
        EngineError::Pairing(_) => "_PairingError",
        EngineError::Sequence { .. } => "SequenceError",
        EngineError::Device(_) => "DeviceError",
        EngineError::Overflow(_) => "OverflowError",
        // Never mapped on its own: the call raises the stored handler exception
        // (cancellation-progress.md §3.4); without one it is an internal error.
        EngineError::Cancelled => "DeviceError",
    };
    let make = || -> PyResult<PyErr> {
        let error_module = if matches!(error, EngineError::Overflow(_)) {
            "builtins"
        } else {
            "seqblaze._errors"
        };
        let instance = py
            .import(error_module)?
            .getattr(name)?
            .call1((error.to_string(),))?;
        if let EngineError::Sequence {
            input,
            index,
            position,
            byte,
            n_bad,
            bad_indices,
            ..
        } = &error
        {
            instance.setattr("input", input)?;
            instance.setattr("index", index)?;
            instance.setattr("position", position)?;
            let character = byte.map(|byte| {
                if byte.is_ascii() {
                    (byte as char).to_string()
                } else {
                    format!("\\x{byte:02x}")
                }
            });
            instance.setattr("char", character)?;
            instance.setattr("n_bad", n_bad)?;
            instance.setattr("bad_indices", bad_indices.clone())?;
        }
        Ok(PyErr::from_value(instance))
    };
    make().unwrap_or_else(|error| error)
}

#[pyclass(frozen, module = "seqblaze._core", name = "_SeqTable")]
pub struct PySeqTable {
    table: TableStorage,
}

enum TableStorage {
    Host(Arc<SeqTable>),
    #[cfg(feature = "cuda")]
    Device {
        owner: Py<device::PyDeviceSeqTable>,
        rows: usize,
        text: bool,
    },
}
impl TableStorage {
    fn len(&self) -> usize {
        match self {
            Self::Host(table) => table.len(),
            #[cfg(feature = "cuda")]
            Self::Device { rows, .. } => *rows,
        }
    }
    fn is_text(&self) -> bool {
        match self {
            Self::Host(table) => table.is_text(),
            #[cfg(feature = "cuda")]
            Self::Device { text, .. } => *text,
        }
    }
    fn storage_kind(&self) -> &'static str {
        match self {
            Self::Host(table) => table.storage_kind(),
            #[cfg(feature = "cuda")]
            Self::Device { .. } => "device",
        }
    }
    fn host(&self) -> Result<Arc<SeqTable>, EngineError> {
        match self {
        Self::Host(table) => Ok(Arc::clone(table)),
        #[cfg(feature = "cuda")] Self::Device { .. } => Err(EngineError::Device("device sequence inputs require the CUDA backend; use device='auto' or their CUDA ordinal".into())),
    }
    }
}

#[pymethods]
impl PySeqTable {
    #[getter]
    fn _source_kind(&self) -> &'static str {
        self.table.storage_kind()
    }
    #[getter]
    fn storage_info<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        #[cfg(feature = "cuda")]
        if let TableStorage::Device { owner, .. } = &self.table {
            return owner.borrow(py).storage_info(py);
        }
        let host = self.table.host().map_err(|e| error(py, e))?;
        let info = host.storage_info();
        if let Some(arrow) = info.arrow {
            return arrow::metadata(py, arrow);
        }
        let result = PyDict::new(py);
        result.set_item("kind", info.kind)?;
        if let Some(chunked) = info.arrow_chunks {
            result.set_item("rows", info.rows)?;
            result.set_item("format", chunked.format)?;
            let mut chunks = Vec::with_capacity(chunked.chunks.len());
            for descriptor in chunked.chunks {
                chunks.push(arrow::metadata(py, descriptor)?);
            }
            result.set_item("chunks", PyTuple::new(py, chunks)?)?;
            return Ok(result);
        }
        result.set_item("address", info.address)?;
        result.set_item("rows", info.rows)?;
        if let Some(matrix) = info.matrix {
            result.set_item("columns", matrix.columns)?;
            result.set_item("row_stride", matrix.row_stride)?;
            result.set_item("column_stride", matrix.column_stride)?;
            result.set_item("lengths_address", matrix.lengths.address)?;
            result.set_item("lengths_itemsize", matrix.lengths.itemsize)?;
            result.set_item("lengths_stride", matrix.lengths.stride)?;
        } else {
            result.set_item("itemsize", info.itemsize)?;
            result.set_item("stride", info.stride)?;
        }
        Ok(result)
    }
    fn __len__(&self) -> usize {
        self.table.len()
    }
    fn __getattr__<'py>(&self, py: Python<'py>, name: &str) -> PyResult<Bound<'py, PyAny>> {
        #[cfg(not(feature = "cuda"))]
        let _ = py;
        #[cfg(feature = "cuda")]
        if let TableStorage::Device { owner, .. } = &self.table {
            if matches!(
                name,
                "_device_digest" | "_device_snapshot" | "_device_ordinal"
            ) {
                return owner.bind(py).getattr(name);
            }
        }
        Err(pyo3::exceptions::PyAttributeError::new_err(format!(
            "SeqTable has no attribute {name:?}"
        )))
    }
    /// Whether at least one row came from text: the manifest's
    /// `encoding: "utf-8"` (text-nulls-fp-environment.md §5).
    #[getter]
    fn _text(&self) -> bool {
        self.table.is_text()
    }
}

/// Append one non-null item (row `row`) to `data`: `bytes` as given, a `str`
/// as its strict UTF-8 encoding (D86 (3)). Returns whether the item was text.
/// No method of the item is called. A `str` with a lone surrogate has no UTF-8
/// encoding: ParameterError naming the row and code-point position (E4).
fn append_sequence(
    py: Python<'_>,
    object: &Bound<'_, PyAny>,
    row: usize,
    data: &mut Vec<u8>,
) -> PyResult<bool> {
    // An exact `str` or `bytes` (the common rows) is recognised by its type
    // alone, before the subtype checks; any other item takes them as before.
    let exact = object.cast_exact::<PyString>().ok();
    if exact.is_none() {
        if let Ok(bytes) = object
            .cast_exact::<PyBytes>()
            .or_else(|_| object.cast::<PyBytes>())
        {
            reserve_data(py, data, bytes.as_bytes().len())?;
            data.extend_from_slice(bytes.as_bytes());
            return Ok(false);
        }
    }
    if let Some(text) = exact.or_else(|| object.cast::<PyString>().ok()) {
        // CPython's cached UTF-8 form: it fails for a lone surrogate (E4) or
        // when CPython cannot allocate the cache (DeviceError, as a refused
        // reservation; engine-layout.md, Allocation).
        return match text.to_str() {
            Ok(text) => {
                reserve_data(py, data, text.len())?;
                data.extend_from_slice(text.as_bytes());
                Ok(true)
            }
            Err(failure) if failure.is_instance_of::<PyUnicodeEncodeError>(py) => {
                Err(not_utf8(py, text, row, failure))
            }
            Err(failure) if failure.is_instance_of::<PyMemoryError>(py) => Err(error(
                py,
                EngineError::Device(format!(
                    "unable to reserve the UTF-8 form of sequence {row}; no table was built"
                )),
            )),
            Err(failure) => Err(failure),
        };
    }
    Err(error(
        py,
        EngineError::Parameter(format!(
            "sequence {row} is a {}; sequences must be bytes, str or None (a null row)",
            object
                .get_type()
                .name()
                .map(|name| name.to_string())
                .unwrap_or_else(|_| "foreign object".into())
        )),
    ))
}

/// Grow the converted sequence bytes fallibly (amortized): a refused
/// reservation is the §3 DeviceError with the bytes the row needs
/// (host-allocation-failures.md §3; engine-layout.md, Allocation).
fn reserve_data(py: Python<'_>, data: &mut Vec<u8>, additional: usize) -> PyResult<()> {
    seqblaze_core::detail::try_reserve(data, additional, "the converted sequences")
        .map_err(|e| error(py, e))
}

/// E4 for a `str` whose encoding failed: the first code point without a UTF-8
/// encoding (a lone surrogate), from the encoder's reported position.
fn not_utf8(py: Python<'_>, text: &Bound<'_, PyString>, row: usize, failure: PyErr) -> PyErr {
    let position: Option<usize> = failure
        .value(py)
        .getattr("start")
        .and_then(|start| start.extract())
        .ok();
    let value: Option<u32> = position.and_then(|position| {
        text.get_item(position)
            .and_then(|item| py.import("builtins")?.getattr("ord")?.call1((item,)))
            .and_then(|code| code.extract())
            .ok()
    });
    let at = match (position, value) {
        (Some(position), Some(value)) => {
            format!("code point position {position} is U+{value:04X}, a surrogate code point")
        }
        _ => "a code point is a surrogate".into(),
    };
    error(
        py,
        EngineError::Parameter(format!(
            "text at sequence {row}, {at}, which has no UTF-8 encoding; pass valid text or the \
             intended bytes"
        )),
    )
}

/// One owned table from the items of a `str`/`bytes` Seq, a list, a tuple or
/// a 1-D object array: `None` items are null rows (recorded, never raised
/// here; D86 (1)), `str` items their UTF-8 bytes, `bytes` as given.
fn owned_sequence_table<'py>(
    py: Python<'py>,
    objects: impl IntoIterator<Item = PyResult<Bound<'py, PyAny>>>,
) -> PyResult<Bound<'py, PyAny>> {
    let mut data = Vec::new();
    let mut offsets = vec![0i64];
    let mut nulls = Vec::new();
    let mut text = false;
    for (row, object) in objects.into_iter().enumerate() {
        let object = object?;
        if object.is_none() {
            seqblaze_core::detail::try_reserve(&mut nulls, 1, "the null-row record")
                .map_err(|e| error(py, e))?;
            nulls.push(row);
        } else {
            text |= append_sequence(py, &object, row, &mut data)?;
        }
        seqblaze_core::detail::try_reserve(&mut offsets, 1, "the sequence offsets")
            .map_err(|e| error(py, e))?;
        offsets.push(i64::try_from(data.len()).map_err(|_| {
            error(
                py,
                EngineError::Parameter("sequence buffer exceeds int64 offset limit".into()),
            )
        })?);
    }
    // Copy each source byte directly into the final contiguous owned table once.
    let batch = SeqBatch::try_from_parts(data, offsets)
        .map_err(|message| error(py, EngineError::Parameter(message.into())))?;
    let table = SeqTable::with_nulls(batch, nulls, text).map_err(|e| error(py, e))?;
    Ok(Bound::new(
        py,
        PySeqTable {
            table: TableStorage::Host(Arc::new(table)),
        },
    )?
    .into_any())
}

#[pyfunction]
pub fn _native_seqs<'py>(
    py: Python<'py>,
    object: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    if object.is_instance_of::<PySeqTable>() {
        return Ok(object.clone());
    }
    if object.is_instance_of::<PyBytes>() || object.is_instance_of::<PyString>() {
        owned_sequence_table(py, std::iter::once(Ok(object.clone())))
    } else if object.is_instance_of::<PyList>() || object.is_instance_of::<PyTuple>() {
        owned_sequence_table(py, object.try_iter()?)
    } else {
        Err(error(
            py,
            EngineError::Parameter(
                "seqs expects bytes, str, a list/tuple of sequences (bytes, str or None), or a \
                 SeqTable"
                    .into(),
            ),
        ))
    }
}

enum Numeric {
    I32(Vec<i32>),
    I64(Vec<i64>),
    F64(Vec<f64>),
    Bool(Vec<u8>),
    U8(Vec<u8>),
}

#[pyclass(frozen, module = "seqblaze._core")]
struct OwnedColumn {
    values: Numeric,
}

#[pymethods]
impl OwnedColumn {
    /// NumPy retains this immutable owner as `array.base`; the Vec never resizes.
    #[getter]
    fn __array_interface__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let endian = if cfg!(target_endian = "little") {
            '<'
        } else {
            '>'
        };
        let (pointer, length, typestr) = match &self.values {
            Numeric::I32(values) => (
                values.as_ptr() as usize,
                values.len(),
                format!("{endian}i4"),
            ),
            Numeric::I64(values) => (
                values.as_ptr() as usize,
                values.len(),
                format!("{endian}i8"),
            ),
            Numeric::F64(values) => (
                values.as_ptr() as usize,
                values.len(),
                format!("{endian}f8"),
            ),
            Numeric::Bool(values) => (values.as_ptr() as usize, values.len(), "|b1".into()),
            Numeric::U8(values) => (values.as_ptr() as usize, values.len(), "|u1".into()),
        };
        let result = PyDict::new(py);
        result.set_item("version", 3)?;
        result.set_item("shape", (length,))?;
        result.set_item("typestr", typestr)?;
        result.set_item("data", (pointer, true))?;
        Ok(result)
    }
}

fn column(py: Python<'_>, values: Numeric) -> PyResult<Py<OwnedColumn>> {
    Py::new(py, OwnedColumn { values })
}

/// Add detail columns: numeric fields as owned arrays, list fields as
/// ("list", offsets, values, validity) and string (byte-string) fields as
/// ("string" | "binary", offsets, data, validity), every part an owned array. The int32
/// columns named in `bools` are exposed as bool: 1 is true, every other value (0 and the
/// INT32_MAX mask) false (hairpin-primer3.md §6, dimer-primer3.md §7 `found`).
fn set_details_as(
    py: Python<'_>,
    columns: &Bound<'_, PyDict>,
    details: Option<DetailColumns>,
    bools: &[&str],
) -> PyResult<()> {
    let Some(details) = details else {
        return Ok(());
    };
    for detail in details.into_columns() {
        match detail.data {
            DetailData::Int32(values) if bools.contains(&detail.name) => {
                // Fallible like every detail column (engine-layout §6).
                let mut flags = Vec::new();
                seqblaze_core::detail::try_reserve_exact(
                    &mut flags,
                    values.len(),
                    &format!("{} column", detail.name),
                )
                .map_err(|e| error(py, e))?;
                flags.extend(values.into_iter().map(|v| u8::from(v == 1)));
                columns.set_item(detail.name, column(py, Numeric::Bool(flags))?)?
            }
            DetailData::Int32(values) => {
                columns.set_item(detail.name, column(py, Numeric::I32(values))?)?
            }
            DetailData::Int64(values) => {
                columns.set_item(detail.name, column(py, Numeric::I64(values))?)?
            }
            DetailData::Float64(values) => {
                columns.set_item(detail.name, column(py, Numeric::F64(values))?)?
            }
            DetailData::ListInt64 {
                offsets,
                values,
                valid,
            } => columns.set_item(
                detail.name,
                (
                    "list",
                    column(py, Numeric::I64(offsets))?,
                    column(py, Numeric::I64(values))?,
                    column(py, Numeric::Bool(valid))?,
                ),
            )?,
            DetailData::Utf8 {
                offsets,
                data,
                valid,
            } => columns.set_item(
                detail.name,
                (
                    if detail.kind == FieldKind::Binary {
                        "binary"
                    } else {
                        "string"
                    },
                    column(py, Numeric::I64(offsets))?,
                    column(py, Numeric::U8(data))?,
                    column(py, Numeric::Bool(valid))?,
                ),
            )?,
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn work_source(
    py: Python<'_>,
    kind: &str,
    i: Option<PyRef<'_, PyIndexColumn>>,
    j: Option<PyRef<'_, PyIndexColumn>>,
    self_pairs: bool,
    n_a: usize,
    n_b: usize,
) -> PyResult<WorkSource> {
    oriented_work_source(py, kind, i, j, self_pairs, None, n_a, n_b)
}

/// `orientation` (both/as_given) turns within into the directed source
/// WithinOriented; None keeps the symmetric upper-triangle Within.
#[allow(clippy::too_many_arguments)]
fn oriented_work_source(
    py: Python<'_>,
    kind: &str,
    i: Option<PyRef<'_, PyIndexColumn>>,
    j: Option<PyRef<'_, PyIndexColumn>>,
    self_pairs: bool,
    orientation: Option<&str>,
    n_a: usize,
    n_b: usize,
) -> PyResult<WorkSource> {
    let orientation = match orientation {
        None => None,
        Some("both") => Some(Orientation::Both),
        Some("as_given") => Some(Orientation::AsGiven),
        Some(_) => {
            return Err(error(
                py,
                EngineError::Parameter("orientation must be both or as_given".into()),
            ))
        }
    };
    let kind = match kind {
        "zip" => PairKind::Zip,
        "cross" => PairKind::Cross,
        "one_to_many" => PairKind::OneToMany,
        "within" => match orientation {
            None => PairKind::Within { self_pairs },
            Some(orientation) => PairKind::WithinOriented {
                self_pairs,
                orientation,
            },
        },
        "index" => {
            let i = i.ok_or_else(|| error(py, EngineError::Pairing("missing i".into())))?;
            let j = j.ok_or_else(|| error(py, EngineError::Pairing("missing j".into())))?;
            PairKind::Index {
                i: i.column.clone(),
                j: j.column.clone(),
            }
        }
        _ => {
            return Err(error(
                py,
                EngineError::Parameter("unsupported pair kind".into()),
            ))
        }
    };
    WorkSource::new(kind, n_a, n_b).map_err(|e| error(py, e))
}

/// Validate a lazy pair descriptor without scoring or allocating pair grids.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn _native_pairs(
    py: Python<'_>,
    a: PyRef<'_, PySeqTable>,
    b: PyRef<'_, PySeqTable>,
    kind: &str,
    i: Option<PyRef<'_, PyIndexColumn>>,
    j: Option<PyRef<'_, PyIndexColumn>>,
    self_pairs: bool,
) -> PyResult<usize> {
    Ok(work_source(py, kind, i, j, self_pairs, a.table.len(), b.table.len())?.n_pairs)
}

#[derive(Clone)]
enum RuntimeKernel {
    Hamming,
    Levenshtein(Levenshtein),
    /// `score_fx`: whether the call returns the fixed-point score column.
    Align {
        kernel: Arc<Align>,
        score_fx: bool,
    },
    /// `dg_dcal`: whether the call returns the int32 key column beside `dg_kcal`.
    Duplex {
        kernel: Arc<seqblaze_thermo::Duplex>,
        dg_dcal: bool,
    },
    /// K3 `duplex(model="pf")`: an int64 key, the image of `dg_ens_kcal`.
    DuplexPf(Arc<seqblaze_thermo::DuplexPf>),
    /// K2 `duplex(compat="primer3")`: an int64 key, the image of `dg_kcal`
    /// (dimer-primer3.md §7, the K3 key).
    DuplexPrimer3(Arc<seqblaze_thermo::DimerPrimer3>),
    /// `tm` on Pairs (the register pairing, tm.md §13): an int64 key, the image of
    /// −`tm_c`; every published column is a detail field, except RowStats' `best_tm_c`.
    TmPairs(Arc<seqblaze_thermo::TmPairs>),
    #[cfg(feature = "testing")]
    LengthDiff,
}

/// Run `$body` with `$k` bound to the concrete kernel of a `&RuntimeKernel`. The body
/// must have one type for every kernel (K3, K2 and `tm` on Pairs have an int64 key).
macro_rules! dispatch {
    ($kernel:expr, |$k:ident| $body:expr) => {
        match $kernel {
            RuntimeKernel::DuplexPf(kernel) => {
                let $k: &seqblaze_thermo::DuplexPf = kernel;
                $body
            }
            RuntimeKernel::DuplexPrimer3(kernel) => {
                let $k: &seqblaze_thermo::DimerPrimer3 = kernel;
                $body
            }
            RuntimeKernel::TmPairs(kernel) => {
                let $k: &seqblaze_thermo::TmPairs = kernel;
                $body
            }
            other => $crate::runtime::dispatch_i32!(other, |$k| $body, unreachable!()),
        }
    };
}
pub(crate) use dispatch;

/// As `dispatch!` for the int32-key kernels (the CUDA backend's int32 path); `$other` for
/// the int64-key kernels (K3, K2, `tm` on Pairs; K3 and `tm` on Pairs take the CUDA
/// backend's int64-key entries through their own arms, duplex-pf-cuda.md §7 and
/// tm-pairs-cuda.md §7; K2 has no device path).
macro_rules! dispatch_i32 {
    ($kernel:expr, |$k:ident| $body:expr, $other:expr) => {
        match $kernel {
            RuntimeKernel::DuplexPf(_)
            | RuntimeKernel::DuplexPrimer3(_)
            | RuntimeKernel::TmPairs(_) => $other,
            RuntimeKernel::Hamming => {
                let $k = &seqblaze_edit::Hamming;
                $body
            }
            RuntimeKernel::Levenshtein(kernel) => {
                let $k = kernel;
                $body
            }
            RuntimeKernel::Align { kernel, .. } => {
                let $k: &seqblaze_align::Align = kernel;
                $body
            }
            RuntimeKernel::Duplex { kernel, .. } => {
                let $k: &seqblaze_thermo::Duplex = kernel;
                $body
            }
            #[cfg(feature = "testing")]
            RuntimeKernel::LengthDiff => {
                let $k = &seqblaze_probe::LengthDiff;
                $body
            }
        }
    };
}
pub(crate) use dispatch_i32;

/// A primary key column at the Python boundary: int32 keys, or K3's int64 keys.
pub(crate) enum Keys {
    I32(Vec<i32>),
    I64(Vec<i64>),
}

/// Key types the runtime can export.
pub(crate) trait Erase: Score {
    fn keys(values: Vec<Self>) -> Keys;
}

impl Erase for i32 {
    fn keys(values: Vec<Self>) -> Keys {
        Keys::I32(values)
    }
}

impl Erase for i64 {
    fn keys(values: Vec<Self>) -> Keys {
        Keys::I64(values)
    }
}

/// An optional entry of the align configuration dict.
fn align_item<'py, T: FromPyObjectOwned<'py>>(
    config: &Bound<'py, PyDict>,
    key: &str,
) -> Result<Option<T>, EngineError> {
    let invalid =
        |detail: String| EngineError::Parameter(format!("align configuration {key}: {detail}"));
    match config.get_item(key).map_err(|e| invalid(e.to_string()))? {
        None => Ok(None),
        Some(value) if value.is_none() => Ok(None),
        Some(value) => value
            .extract::<T>()
            .map(Some)
            .map_err(|e| invalid(Into::<PyErr>::into(e).to_string())),
    }
}

/// The facade's resolved align options (fixed-point values, spec §2).
fn align_config(
    mode: &str,
    details: &[String],
    check_bases: bool,
    config: Option<&Bound<'_, PyDict>>,
) -> Result<RuntimeKernel, EngineError> {
    let config =
        config.ok_or_else(|| EngineError::Parameter("align requires its configuration".into()))?;
    let mode = seqblaze_align::Mode::parse(mode).ok_or_else(|| {
        EngineError::Parameter(format!("mode={mode:?}; use mode=\"local\" or \"global\""))
    })?;
    let compat = align_item::<String>(config, "compat")?
        .map(|alias| {
            Compat::parse(&alias)
                .ok_or_else(|| EngineError::Parameter(format!("unknown align compat {alias:?}")))
        })
        .transpose()?;
    let required = |key: &str| -> Result<i64, EngineError> {
        align_item::<i64>(config, key)?
            .ok_or_else(|| EngineError::Parameter(format!("align configuration lacks {key}")))
    };
    let substitution = if let Some(name) = align_item::<String>(config, "matrix")? {
        if name != "EDNAFULL" {
            return Err(EngineError::Parameter(format!(
                "unknown align matrix {name:?}"
            )));
        }
        Substitution::Ednafull
    } else if let Some(table) = config
        .get_item("table")
        .ok()
        .flatten()
        .filter(|value| !value.is_none())
    {
        let table = table.cast::<PyBytes>().map_err(|_| {
            EngineError::Parameter("align table must be bytes of 65,536 int32 values".into())
        })?;
        let table = table.as_bytes();
        if table.len() != 4 << 16 {
            return Err(EngineError::Parameter(
                "align table must hold 65,536 little-endian int32 values".into(),
            ));
        }
        Substitution::Table(
            table
                .chunks_exact(4)
                .map(|bytes| i32::from_le_bytes(bytes.try_into().expect("four bytes")))
                .collect(),
        )
    } else {
        let narrow = |key: &str| -> Result<i32, EngineError> {
            i32::try_from(required(key)?).map_err(|_| {
                EngineError::Parameter(format!("align {key} exceeds the fixed-point range"))
            })
        };
        Substitution::MatchMismatch {
            matched: narrow("match")?,
            mismatched: narrow("mismatch")?,
        }
    };
    let fields: Vec<&str> = details.iter().map(String::as_str).collect();
    let kernel = Align::new(AlignConfig {
        mode,
        compat,
        gap_open: required("gap_open")?,
        gap_extend: required("gap_extend")?,
        end_gaps: align_item::<bool>(config, "end_gaps")?,
        substitution,
        check_bases,
        fields: &fields,
    })?;
    Ok(RuntimeKernel::Align {
        kernel: Arc::new(kernel),
        score_fx: align_item::<bool>(config, "score_fx")?.unwrap_or(false),
    })
}

/// The primary column of `keys`: `name` (int32) for distance kernels; align
/// keys become `score` (float64) and, if requested, `score_fx` (int32), masks
/// first (spec §8). `best` names RowStats' `best_<primary>`.
fn set_primary(
    py: Python<'_>,
    columns: &Bound<'_, PyDict>,
    kernel: &RuntimeKernel,
    best: bool,
    keys: Keys,
) -> PyResult<()> {
    let keys = match keys {
        Keys::I32(keys) => keys,
        // tm on Pairs: under Each, Hits, TopK, Matrix and Tiles, tm_c and the other fields
        // are the rows' detail columns (F10's bits, tm.md §13.11) and the selecting key is
        // not published. RowStats has no detail rows: `best_tm_c` is the key's inverse, in
        // the keys' buffer, which equals F10's bits of the row's best partner (bit
        // operations only, so no FP environment is involved).
        Keys::I64(keys) if matches!(kernel, RuntimeKernel::TmPairs(_)) => {
            if !best {
                return Ok(());
            }
            let values = seqblaze_thermo::duplex::register::values_of(keys);
            return columns.set_item("best_tm_c", column(py, Numeric::F64(values))?);
        }
        // K3 and K2: the K3 bit key of a float64 primary (duplex-pf.md §9.2,
        // dimer-primer3.md §7).
        Keys::I64(keys) => {
            let name = match (kernel, best) {
                (RuntimeKernel::DuplexPrimer3(_), false) => "dg_kcal",
                (RuntimeKernel::DuplexPrimer3(_), true) => "best_dg_kcal",
                (_, false) => "dg_ens_kcal",
                (_, true) => "best_dg_ens_kcal",
            };
            return set_key_primary(py, columns, name, keys);
        }
    };
    if let RuntimeKernel::Duplex { dg_dcal, .. } = kernel {
        return set_duplex_primary(py, columns, *dg_dcal && !best, best, keys);
    }
    let RuntimeKernel::Align { score_fx, .. } = kernel else {
        let name = if best { "best_dist" } else { "dist" };
        return columns.set_item(name, column(py, Numeric::I32(keys))?);
    };
    let mut scores = Vec::new();
    seqblaze_core::detail::try_reserve_exact(&mut scores, keys.len(), "score column")
        .map_err(|e| error(py, e))?;
    // fx / 1000 in E0 whatever the caller's FP state (§7.3), with the GIL held.
    fpenv::guarded(|| {
        scores.extend(keys.iter().map(|&key| {
            seqblaze_align::score_fx_of_key(key).map_or(f64::NAN, seqblaze_align::score_of_fx)
        }))
    });
    columns.set_item(
        if best { "best_score" } else { "score" },
        column(py, Numeric::F64(scores))?,
    )?;
    if *score_fx && !best {
        let mut fx = keys;
        for value in &mut fx {
            *value = seqblaze_align::score_fx_of_key(*value).unwrap_or(i32::MAX);
        }
        columns.set_item("score_fx", column(py, Numeric::I32(fx))?)?;
    }
    Ok(())
}

/// duplex keys become `dg_kcal` (float64) and, if requested, `dg_dcal` (int32), masks
/// first (duplex-mfe.md §7, §12.4): INT32_MAX (a mask or a hole) gives INT32_MAX and NaN;
/// the empty-space key INT32_MAX - 1 gives INT32_MAX and RN(INT32_MAX/100); any other
/// key d gives d and RN(d/100). RowStats emits `best_dg_kcal` only (A9). The sentinel
/// values are provisional: QK9.
fn set_duplex_primary(
    py: Python<'_>,
    columns: &Bound<'_, PyDict>,
    dg_dcal: bool,
    best: bool,
    keys: Vec<i32>,
) -> PyResult<()> {
    let mut kcal = Vec::new();
    seqblaze_core::detail::try_reserve_exact(&mut kcal, keys.len(), "dg_kcal column")
        .map_err(|e| error(py, e))?;
    // d / 100 and the sentinel in E0 whatever the caller's FP state (§7.3 item 4), with
    // the GIL held: no Python code runs inside the guarded body.
    fpenv::guarded(|| {
        kcal.extend(keys.iter().map(|&key| match key {
            i32::MAX => seqblaze_core::detail::CANONICAL_NAN,
            seqblaze_thermo::duplex::SENTINEL_KEY => f64::from(i32::MAX) / 100.0,
            d => f64::from(d) / 100.0,
        }))
    });
    columns.set_item(
        if best { "best_dg_kcal" } else { "dg_kcal" },
        column(py, Numeric::F64(kcal))?,
    )?;
    if dg_dcal {
        let mut dcal = keys;
        for value in &mut dcal {
            if *value == seqblaze_thermo::duplex::SENTINEL_KEY {
                *value = i32::MAX;
            }
        }
        columns.set_item("dg_dcal", column(py, Numeric::I32(dcal))?)?;
    }
    Ok(())
}

/// int64 keys become their float64 primary `name` (duplex-pf.md §9.2): the inverse of
/// the order-preserving image; the mask INT64_MAX gives NaN. K3's `dg_ens_kcal` (RowStats
/// `best_dg_ens_kcal`) and K2's `dg_kcal` (`best_dg_kcal`, dimer-primer3.md §7). The
/// values take the keys' buffer (a bit map between same-size types), so a matrix or tile
/// export needs no second N-length column.
fn set_key_primary(
    py: Python<'_>,
    columns: &Bound<'_, PyDict>,
    name: &str,
    keys: Vec<i64>,
) -> PyResult<()> {
    let values = seqblaze_thermo::duplex::pf::values_of(keys);
    columns.set_item(name, column(py, Numeric::F64(values))?)
}

/// The primary and detail columns of one output, in the kernel's field order: K2 emits
/// `found` before its primary `dg_kcal` (dimer-primer3.md §7), so a requested `found`
/// takes its dict slot first and is filled by the detail columns (exposed as bool).
fn set_primary_and_details(
    py: Python<'_>,
    columns: &Bound<'_, PyDict>,
    kernel: &RuntimeKernel,
    keys: Keys,
    details: Option<DetailColumns>,
) -> PyResult<()> {
    let bools = kernel.bool_fields();
    if let Some(details) = &details {
        for name in bools {
            if details.columns().iter().any(|c| c.name == *name) {
                columns.set_item(*name, py.None())?;
            }
        }
    }
    set_primary(py, columns, kernel, false, keys)?;
    set_details_as(py, columns, details, bools)
}

impl RuntimeKernel {
    fn bound_availability(&self) -> BoundAvailability {
        dispatch!(self, |kernel| kernel.bound_availability())
    }

    fn name(&self) -> &'static str {
        match self {
            Self::Hamming => "hamming",
            Self::Levenshtein(_) => "levenshtein",
            Self::Align { .. } => "align",
            Self::Duplex { .. } | Self::DuplexPf(_) | Self::DuplexPrimer3(_) => "duplex",
            Self::TmPairs(_) => "tm",
            #[cfg(feature = "testing")]
            Self::LengthDiff => "_probe.lendiff",
        }
    }

    /// Int32 detail fields exposed as bool (1 true, 0 and the mask false).
    fn bool_fields(&self) -> &'static [&'static str] {
        match self {
            Self::DuplexPrimer3(_) => &["found"],
            _ => &[],
        }
    }

    /// What the CUDA backend computes for this call (levenshtein-cuda.md §8,
    /// align-cuda.md §8, duplex-cuda.md §4): Hamming; Levenshtein's mode plus
    /// the indices of its two numeric fields; align's scoring pack and duplex's
    /// device pack, built from the call's kernel. `cuda_unavailable` for a
    /// kernel the backend lacks (the testing probe).
    fn device_call(&self) -> Result<DeviceCall, EngineError> {
        match self {
            Self::Hamming => Ok(DeviceCall::Hamming),
            Self::Levenshtein(kernel) => Ok(DeviceCall::Levenshtein {
                mode: match kernel.mode() {
                    Mode::Global => EditMode::Global,
                    Mode::Prefix => EditMode::Prefix,
                    Mode::Infix => EditMode::Infix,
                },
                b_end: levenshtein::B_END,
                n_ends: levenshtein::N_ENDS,
                // Set by the resident entry from the call's options (`device_run::prepare`).
                acgt: false,
            }),
            Self::Align { kernel, .. } => {
                let pack = kernel.device_pack();
                Ok(DeviceCall::Align(Arc::new(AlignCall {
                    kind: match pack.kind {
                        seqblaze_align::DeviceKind::Local => AlignKind::Local,
                        seqblaze_align::DeviceKind::Global => AlignKind::Global,
                        seqblaze_align::DeviceKind::Water => AlignKind::Water,
                        seqblaze_align::DeviceKind::Needle => AlignKind::Needle,
                    },
                    end_gaps: pack.end_gaps,
                    go: pack.go,
                    ge: pack.ge,
                    matched: pack.matched,
                    class: pack.class,
                    n_classes: pack.n_classes,
                    table: pack.table,
                })))
            }
            Self::Duplex { kernel, .. } => {
                let pack = kernel.call().device_pack()?;
                Ok(DeviceCall::Duplex(Arc::new(DuplexCall {
                    loop_max: pack.loop_max,
                    wide: pack.wide,
                    n_p: pack.n_p,
                    n_c: pack.n_c,
                    cls: pack.cls,
                    offsets: pack.offsets,
                    score: pack.score,
                    full: pack.full,
                    s_bits: pack.s_bits,
                    kappa4_bits: pack.kappa4_bits,
                    kappa1_bits: pack.kappa1_bits,
                    fields: [
                        seqblaze_thermo::duplex::DH_KCAL,
                        seqblaze_thermo::duplex::DS_CAL_PER_K,
                        seqblaze_thermo::duplex::TM_C,
                    ],
                })))
            }
            // K3 (duplex-pf-cuda.md §4): the weights as F53 values, built lazily.
            Self::DuplexPf(kernel) => {
                let pack = kernel.call().device_pack()?;
                Ok(DeviceCall::DuplexPf(Arc::new(PfDeviceCall {
                    loop_max: pack.loop_max,
                    n_p: pack.n_p,
                    n_c: pack.n_c,
                    cls: pack.cls,
                    offsets: pack.offsets,
                    m: pack.m,
                    e: pack.e,
                })))
            }
            // tm on Pairs (tm-pairs-cuda.md §4): K1's device pack of the call, full keys
            // only (QTP4), and T1's five fields in slot order.
            Self::TmPairs(kernel) => {
                let pack = kernel.call().device_pack()?;
                Ok(DeviceCall::TmPairs(Arc::new(register_call(pack))))
            }
            // K2: the handle's tables, the end-term tables and the pack bits
            // (primer3-cuda.md §4), built only for a resolved device.
            Self::DuplexPrimer3(kernel) => {
                let end = kernel.end();
                Ok(DeviceCall::DuplexPrimer3(Arc::new(hairpin::primer3_call(
                    kernel.device_pack()?,
                    end == seqblaze_thermo::dimer::End::End2,
                    end != seqblaze_thermo::dimer::End::Any,
                    [
                        seqblaze_thermo::dimer::FOUND,
                        seqblaze_thermo::dimer::DH_KCAL,
                        seqblaze_thermo::dimer::DS_CAL_PER_K,
                        seqblaze_thermo::dimer::TM_C,
                    ],
                ))))
            }
            #[cfg(feature = "testing")]
            Self::LengthDiff => Err(self.cuda_unavailable()),
        }
    }

    /// An int64 cutoff key must fit the kernel's key type (int32 except K3, K2 and `tm`
    /// on Pairs).
    fn check_cutoff(&self, cutoff: Option<i64>) -> Result<(), EngineError> {
        match (self, cutoff) {
            (Self::DuplexPf(_) | Self::DuplexPrimer3(_) | Self::TmPairs(_), _) | (_, None) => {
                Ok(())
            }
            (_, Some(value)) => i32::try_from(value).map(|_| ()).map_err(|_| {
                EngineError::Parameter(format!(
                    "cutoff={value} is outside the int32 range of {}",
                    self.name()
                ))
            }),
        }
    }

    fn on_cuda(&self) -> bool {
        match self {
            Self::Hamming
            | Self::Levenshtein(_)
            | Self::Align { .. }
            | Self::Duplex { .. }
            | Self::DuplexPf(_)
            | Self::TmPairs(_)
            | Self::DuplexPrimer3(_) => true,
            #[cfg(feature = "testing")]
            Self::LengthDiff => false,
        }
    }

    fn cuda_unavailable(&self) -> EngineError {
        // Every pair kernel has a device path; only the testing kernel reaches this.
        let call = self.name();
        EngineError::Device(format!(
            "{call} is not yet available on the CUDA backend; there is no CPU fallback for an \
             explicit device; use device='cpu' or device='auto'"
        ))
    }

    /// K1 with RowStats `lse_dg_kcal` row terms (duplex-mfe.md §8.6.5).
    fn mfe_lse(&self) -> bool {
        matches!(self, Self::Duplex { kernel, .. } if kernel.row_terms())
    }

    /// Reject an explicit device for a kernel the CUDA backend lacks, before
    /// the device is resolved (so the answer does not depend on the machine).
    fn check_backend(&self, device: Option<u64>) -> Result<(), EngineError> {
        if device.is_some() && !self.on_cuda() {
            Err(self.cuda_unavailable())
        } else {
            Ok(())
        }
    }

    /// The planner's bound capability of a kernel name; align declares its
    /// certified bound for every configuration (spec §6).
    fn plan_bound(name: &str) -> Result<BoundAvailability, EngineError> {
        match name {
            "align" => return Ok(BoundAvailability::Available),
            "duplex" => return Ok(seqblaze_thermo::DUPLEX_BOUND),
            _ => {}
        }
        Ok(Self::configure(name, "global", &[], true, None, None)?.bound_availability())
    }

    /// A kernel with its call configuration: levenshtein's mode and detail
    /// fields; align's mode, fields and resolved options; duplex's per-call pack and
    /// fields (`dg_dcal` is the key column, the others are detail fields); `tm` on
    /// Pairs: its K1 per-call pack (a `model="mfe"` call) and T1's fields.
    fn configure(
        name: &str,
        mode: &str,
        details: &[String],
        check_bases: bool,
        align: Option<&Bound<'_, PyDict>>,
        duplex: Option<&Bound<'_, PyAny>>,
    ) -> Result<Self, EngineError> {
        if name == "align" {
            return align_config(mode, details, check_bases, align);
        }
        if name == "tm" {
            // tm on Pairs (tm.md §13): the register pairing on K1's per-call pack. Any
            // other object (a K3 call, a K2 call) raises.
            let call = duplex
                .and_then(|call| call.cast::<duplex::PyDuplexCall>().ok())
                .filter(|call| call.get().pf.is_none())
                .ok_or_else(|| {
                    EngineError::Parameter(
                        "tm on Pairs requires its K1 per-call pack (model='mfe')".into(),
                    )
                })?;
            let fields: Vec<&str> = details.iter().map(String::as_str).collect();
            let kernel = seqblaze_thermo::TmPairs::new(Arc::clone(&call.get().call), &fields)?;
            return Ok(Self::TmPairs(Arc::new(kernel)));
        }
        if name == "duplex" {
            let missing = || EngineError::Parameter("duplex requires its per-call pack".into());
            let call = duplex.ok_or_else(missing)?;
            if let Ok(p3) = call.cast::<duplex::PyDimerCall>() {
                // K2: `dg_kcal` is the key column, the others are detail fields.
                let p3 = p3.get();
                let fields: Vec<&str> = details
                    .iter()
                    .map(String::as_str)
                    .filter(|name| *name != "dg_kcal")
                    .collect();
                let kernel = seqblaze_thermo::DimerPrimer3::new(
                    p3.pack,
                    Arc::clone(&p3.tables),
                    p3.end,
                    &fields,
                )?;
                return Ok(Self::DuplexPrimer3(Arc::new(kernel)));
            }
            let call = call.cast::<duplex::PyDuplexCall>().map_err(|_| missing())?;
            let call = call.get();
            if let Some(pf) = &call.pf {
                // K3: `lse_dg_kcal` asks RowStats for the exact row sums (§9.4).
                let fields: Vec<&str> = details
                    .iter()
                    .map(String::as_str)
                    .filter(|name| *name != "lse_dg_kcal")
                    .collect();
                let lse = details.iter().any(|name| name == "lse_dg_kcal");
                if lse && call.pf_bound.is_some() {
                    // duplex-pf-bound.md §6.1 (QZ7): lse_dg_kcal sums every partner, so its
                    // call object is `_duplex_pf_lse`'s, which carries no C2 certificate cell
                    // and resolves `bounded` to full with `no_bound` (as K1's `_duplex_lse`).
                    return Err(EngineError::Parameter(
                        "internal error: duplex model='pf' lse_dg_kcal requires the call \
                         object of _duplex_pf_lse"
                            .into(),
                    ));
                }
                let mut kernel = seqblaze_thermo::DuplexPf::new(Arc::clone(pf), &fields, lse)?;
                if let Some(cert) = call.pf_built_certificate() {
                    // The C2 bound (duplex-pf-bound.md §6): kept only when it passes. The
                    // certificate exists only if the plan could use it (§5 *When*); this
                    // never builds it.
                    kernel = kernel.with_bound(Arc::clone(cert));
                }
                return Ok(Self::DuplexPf(Arc::new(kernel)));
            }
            let fields: Vec<&str> = details
                .iter()
                .map(String::as_str)
                .filter(|name| *name != "dg_dcal" && *name != "lse_dg_kcal")
                .collect();
            let mut kernel = seqblaze_thermo::Duplex::new(Arc::clone(&call.call), &fields)?;
            if let Some(cert) = call.built_certificate() {
                // The C1 bound (duplex-mfe-bound.md §6): kept only when it passes. The
                // certificate exists only if the plan could use it (§5 *When*); this
                // never builds it.
                kernel = kernel.with_bound(Arc::clone(cert));
            }
            if details.iter().any(|name| name == "lse_dg_kcal") {
                // K1 RowStats lse_dg_kcal (duplex-mfe.md §8.6.5): the constants come from
                // `_duplex_lse`, which the facade calls during option validation.
                let constants = call.lse.as_ref().ok_or_else(|| {
                    EngineError::Parameter(
                        "internal error: duplex model='mfe' lse_dg_kcal requires the call \
                         object of _duplex_lse"
                            .into(),
                    )
                })?;
                kernel = kernel.with_lse(Arc::clone(constants));
            }
            return Ok(Self::Duplex {
                kernel: Arc::new(kernel),
                dg_dcal: details.iter().any(|name| name == "dg_dcal"),
            });
        }
        if name != "levenshtein" && !details.is_empty() {
            return Err(EngineError::Parameter(format!(
                "{name} has no detail fields"
            )));
        }
        match name {
            "hamming" => Ok(Self::Hamming),
            "levenshtein" => {
                let mode = Mode::parse(mode).ok_or_else(|| {
                    EngineError::Parameter(format!(
                        "mode={mode:?}; use mode=\"global\", \"prefix\" or \"infix\""
                    ))
                })?;
                let fields: Vec<&str> = details.iter().map(String::as_str).collect();
                Ok(Self::Levenshtein(Levenshtein::new(mode, &fields)?))
            }
            #[cfg(feature = "testing")]
            "_probe.lendiff" => Ok(Self::LengthDiff),
            _ => Err(EngineError::Parameter("unsupported testing kernel".into())),
        }
    }
}

/// The device call of `tm` on Pairs (tm-pairs-cuda.md §4, §11 item 3) from K1's device
/// pack of the call: Λ, N_P, N_C, CLS, the table offsets and full keys as they are (STATES
/// is never read; nndb2024-dna.md §5.4.6), the s/κ₄/κ₁ bits,
/// and the indices of T1's fields in slot order (`tm_c`, `dg_kcal`, `dg_dcal`, `dh_kcal`,
/// `ds_cal_per_k`), which are `TmPairs`' detail-field indices. The score table and the
/// width flag are not used (QTP4).
fn register_call(pack: seqblaze_thermo::duplex::DevicePack) -> RegisterCall {
    RegisterCall {
        loop_max: pack.loop_max,
        n_p: pack.n_p,
        n_c: pack.n_c,
        cls: pack.cls,
        offsets: pack.offsets,
        full: pack.full,
        s_bits: pack.s_bits,
        kappa4_bits: pack.kappa4_bits,
        kappa1_bits: pack.kappa1_bits,
        fields: seqblaze_thermo::duplex::register::DEVICE_FIELDS,
    }
}

/// Public-facade planning, separate from the PF-RUN direct-engine prototype.
/// `backend="cuda"` reports the CUDA backend's bound capability: K1's C1 bound only
/// (duplex-mfe-bound.md §8); no other kernel has one on the device (G1).
/// `duplex` is the configured duplex call: K1's bound is per call (duplex-mfe-bound.md
/// §6.1), resolved in two steps (revision 3): step 1 with the provisional availability
/// (a `model="mfe"` call without `lse_dg_kcal`), step 2 only for an effective `bounded`
/// plan: the certificate is taken (built or cached), and a failing certificate or a
/// refused reservation resolves again with Unavailable (`no_bound`). Nothing is raised
/// because of the certificate, and no other plan builds one.
#[pyfunction(signature=(kernel, keep, strategy, has_cutoff, backend="cpu", duplex=None))]
fn _native_plan<'py>(
    py: Python<'py>,
    kernel: &str,
    keep: &str,
    strategy: &str,
    has_cutoff: bool,
    backend: &str,
    duplex: Option<&Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyDict>> {
    // Unary tm has no bound on any backend (tm.md §6); the pair kernels report theirs.
    // Unary hairpin has no bound either (hairpin-primer3.md §6).
    let intrinsic = if kernel == "tm" || kernel == "hairpin" {
        seqblaze_thermo::TM_BOUND
    } else if kernel == "duplex" {
        // Step 1 (provisional): no certificate is built here.
        let eligible = duplex
            .and_then(|call| call.cast::<duplex::PyDuplexCall>().ok())
            .is_some_and(|call| call.get().bound_eligible());
        if eligible {
            BoundAvailability::Available
        } else {
            BoundAvailability::Unavailable
        }
    } else {
        RuntimeKernel::plan_bound(kernel).map_err(|e| error(py, e))?
    };
    let bound = match backend {
        "cpu" => intrinsic,
        // K1's C1 bound runs in the resident K1 stage (duplex-mfe-bound.md §8); every
        // other kernel has no bounded device path.
        "cuda" if kernel == "duplex" => intrinsic,
        "cuda" => seqblaze_cuda::bound_availability(intrinsic),
        _ => {
            return Err(error(
                py,
                EngineError::Parameter("backend must be cpu or cuda".into()),
            ))
        }
    };
    let requested = match strategy {
        "bounded" => Strategy::Bounded,
        "full" => Strategy::Full,
        _ => {
            return Err(error(
                py,
                EngineError::Parameter("strategy must be bounded or full".into()),
            ))
        }
    };
    let Some(demand) = output_demand(keep, has_cutoff) else {
        return Err(error(
            py,
            EngineError::Parameter("unsupported output sink for planning".into()),
        ));
    };
    let mut plan = ExecutionPlan::resolve(requested, bound, demand);
    if kernel == "duplex" && plan.effective == Strategy::Bounded {
        // Step 2 (duplex-mfe-bound.md §6.1, §5): only a plan that uses the bound takes
        // the certificate; a failing or refused one makes the bound Unavailable.
        let available = duplex
            .and_then(|call| call.cast::<duplex::PyDuplexCall>().ok())
            .is_some_and(|call| call.get().bound_available(py));
        if !available {
            plan = ExecutionPlan::resolve(requested, BoundAvailability::Unavailable, demand);
        }
    }
    let result = PyDict::new(py);
    result.set_item("requested", plan.requested.as_str())?;
    result.set_item("effective", plan.effective.as_str())?;
    result.set_item(
        "fallback_reason",
        plan.fallback_reason.map(|reason| reason.as_str()),
    )?;
    Ok(result)
}

/// The planner's output demand of a sink name (`_native_plan`, and the private entries'
/// certificate step, duplex-mfe-bound.md §6.1 *Private entries*); None for an unknown sink.
pub(crate) fn output_demand(keep: &str, has_cutoff: bool) -> Option<OutputDemand> {
    match keep {
        "matrix" | "tiles" => Some(OutputDemand::Dense),
        "each" => Some(OutputDemand::Each { has_cutoff }),
        "hits" | "topk" | "rowstats" => Some(OutputDemand::Selective),
        _ => None,
    }
}

/// Resolve `cuda:<ordinal>` (driver, device, context, module); no data moves.
pub(crate) fn resolve_device(
    py: Python<'_>,
    ordinal: Option<u64>,
) -> PyResult<Option<seqblaze_cuda::Device>> {
    let Some(ordinal) = ordinal else {
        return Ok(None);
    };
    let ordinal = usize::try_from(ordinal).unwrap_or(usize::MAX);
    py.detach(|| {
        let _guard = FpGuard::new();
        seqblaze_cuda::device(ordinal)
    })
    .map(Some)
    .map_err(|e| error(py, e))
}

/// Facade device resolution: name and compute capability, or DeviceError.
#[pyfunction]
fn _cuda_device(py: Python<'_>, ordinal: u64) -> PyResult<Bound<'_, PyDict>> {
    let device = resolve_device(py, Some(ordinal))?.expect("an ordinal was given");
    let info = device.info();
    let result = PyDict::new(py);
    result.set_item("ordinal", info.ordinal)?;
    result.set_item("name", info.name.as_str())?;
    result.set_item("arch", info.arch.as_str())?;
    Ok(result)
}

/// Testing builds: set the process-wide device poison byte (None: off, the
/// default) and return the previous one. Every device allocation made after
/// the call is filled with the byte before use (align-cuda.md §13).
#[cfg(feature = "testing")]
#[pyfunction(signature=(byte=None))]
fn _native_cuda_poison(byte: Option<u8>) -> Option<u8> {
    seqblaze_cuda::set_cuda_poison(byte)
}

/// Testing builds: replace the resident K1/K3 depth cap (None: the default 2^21) and
/// return the previous effective cap (nndb2024-dna.md §5.4.7). The Python wrapper
/// checks the type; a value outside [1, 2^21] raises ParameterError here too.
#[cfg(feature = "testing")]
#[pyfunction(signature=(cells=None))]
fn _native_cuda_resident_depth_cap(py: Python<'_>, cells: Option<u64>) -> PyResult<u64> {
    seqblaze_cuda::set_cuda_resident_depth_cap(cells).map_err(|e| error(py, e))
}

/// Testing builds: S_test and U_test of the resident K1/K3 fast paths (duplex-cuda.md
/// §15.4 step 1); returns the previous pair. A value outside its range raises
/// ParameterError and changes nothing.
#[cfg(feature = "testing")]
#[pyfunction(signature=(slots=None, unit=None))]
#[allow(clippy::type_complexity)]
fn _native_cuda_resident_launch(
    py: Python<'_>,
    slots: Option<u64>,
    unit: Option<u64>,
) -> PyResult<(Option<u64>, Option<u64>)> {
    seqblaze_cuda::set_cuda_resident_launch(slots, unit).map_err(|e| error(py, e))
}

/// Testing builds: turn the capture of the resident K3 stage's Ẑ terms on or off and
/// return what was captured, as (first work ID, interleaved (M, e) bytes) per chunk
/// (duplex-pf-cuda.md §19.4 step 2).
#[cfg(feature = "testing")]
#[pyfunction]
fn _native_cuda_pf_capture<'py>(
    py: Python<'py>,
    on: bool,
) -> PyResult<Vec<(u64, Bound<'py, pyo3::types::PyBytes>)>> {
    Ok(seqblaze_cuda::set_cuda_pf_capture(on)
        .into_iter()
        .map(|(k0, bytes)| (k0, pyo3::types::PyBytes::new(py, &bytes)))
        .collect())
}

/// Testing builds: G_res of a resident K1/K3 fast kernel (`kernel` 0 narrow, 1 wide,
/// 2 detail, 3 pf) at q = `staged`, or with `shared_band` > 0 K3's N_sh, on device
/// `ordinal` (duplex-cuda.md §15.4 step 1). Resolving the device may raise DeviceError.
#[cfg(feature = "testing")]
#[pyfunction]
fn _native_cuda_resident_wave(
    py: Python<'_>,
    kernel: usize,
    staged: u64,
    shared_band: u64,
    ordinal: u64,
) -> PyResult<u64> {
    let device = resolve_device(py, Some(ordinal))?.expect("an ordinal was given");
    device
        .resident_wave(kernel, staged, shared_band)
        .map_err(|e| error(py, e))
}

/// Testing builds: W₃₂ of the primer3 resident fast kernels on device `ordinal`
/// (primer3-cuda.md §22.4 step 3): SMs × resident warps per SM of the g = 32 kernel,
/// as (hairpin, dimer). Resolving the device may raise DeviceError.
#[cfg(feature = "testing")]
#[pyfunction]
fn _native_cuda_p3_wave(py: Python<'_>, ordinal: u64) -> PyResult<(u64, u64)> {
    let device = resolve_device(py, Some(ordinal))?.expect("an ordinal was given");
    let [hairpin, dimer] = device.p3_wave();
    Ok((hairpin, dimer))
}

/// Testing builds: force the lane width of align's inter-pair score pass
/// (gotoh-align.md §7): 0 (scalar fills only), 8, or 16 where AVX-512 is
/// available; None restores detection. Returns the previous forced width.
/// Outputs never depend on it.
#[cfg(feature = "testing")]
#[pyfunction(signature=(width=None))]
fn _native_align_lanes(py: Python<'_>, width: Option<usize>) -> PyResult<Option<usize>> {
    seqblaze_align::set_lane_width(width)
        .map_err(|message| error(py, EngineError::Parameter(message)))
}

/// Testing builds: force the instruction set of levenshtein's inter-pair
/// score pass (levenshtein.md §6, §12): "scalar" (every pair through the
/// scalar evaluation, one by one), "portable", or "avx2"/"avx512" where this
/// build and CPU support them; None restores detection. Returns the previous
/// forced value (None when detecting). Any other value raises ParameterError
/// naming the available values and changes nothing. Outputs never depend on it.
#[cfg(feature = "testing")]
#[pyfunction(signature=(isa=None))]
fn _native_edit_lanes(
    py: Python<'_>,
    isa: Option<Bound<'_, PyAny>>,
) -> PyResult<Option<&'static str>> {
    let Some(value) = isa else {
        return levenshtein::set_lane_isa(None)
            .map_err(|message| error(py, EngineError::Parameter(message)));
    };
    let name = value
        .cast::<PyString>()
        .ok()
        .map(|text| text.to_string_lossy());
    // A refused value is shown by its repr; a lossy text never names an isa
    // it does not spell (the names are ASCII).
    match name
        .as_deref()
        .map(|name| levenshtein::set_lane_isa(Some(name)))
    {
        Some(Ok(previous)) => Ok(previous),
        _ => {
            let shown = value.repr()?.to_string_lossy().into_owned();
            Err(error(
                py,
                EngineError::Parameter(levenshtein::lane_isa_unavailable(&shown)),
            ))
        }
    }
}

/// Testing builds: where the primer3 device planner puts one item
/// (primer3-cuda.md §6.6): `hairpin` (T2, length `l1`) or a K2 pair of operand
/// lengths (`l1`, `l2`) after the `end` mapping. Returns (g, depth, routed); pure
/// host arithmetic, no device.
#[cfg(feature = "testing")]
#[pyfunction(signature=(hairpin, l1, l2, group_lanes, pair_cells=None))]
fn _native_primer3_device_plan(
    hairpin: bool,
    l1: u64,
    l2: u64,
    group_lanes: u64,
    pair_cells: Option<u64>,
) -> (u64, u64, bool) {
    seqblaze_cuda::primer3_device_plan(hairpin, l1, l2, group_lanes, pair_cells)
}

/// Testing builds: bytes of device memory filled with the poison byte so far.
#[cfg(all(feature = "testing", feature = "cuda"))]
#[pyfunction]
fn _native_resident_launch_streams(py: Python<'_>) -> PyResult<Bound<'_, PyDict>> {
    let trace = seqblaze_cuda::interop::execution_stream::resident_launch_streams();
    let result = PyDict::new(py);
    for (name, handles) in [
        "kernel",
        "copy",
        "validate",
        "digest",
        "detail_plan",
        "detail_continue",
        "detail_progress",
        "detail_copy",
    ]
    .into_iter()
    .zip(trace)
    {
        result.set_item(name, handles)?;
    }
    Ok(result)
}
#[cfg(all(feature = "testing", feature = "cuda"))]
#[pyfunction]
fn _native_resident_launch_streams_reset() {
    seqblaze_cuda::interop::execution_stream::resident_launch_streams_reset();
}

/// Testing builds: the owned pool of `ordinal` (cuda-backend.md, per-call cost
/// S2, "PC6 testing hooks"): `pool`, `reserved`, `used`, `threshold`,
/// `cache_bytes` and `cache_entries`.
#[cfg(all(feature = "testing", feature = "cuda"))]
#[pyfunction]
fn _native_cuda_pool_stats(py: Python<'_>, ordinal: usize) -> PyResult<Bound<'_, PyDict>> {
    let stats = py
        .detach(|| seqblaze_cuda::testing_hooks::pool_stats(ordinal))
        .map_err(|e| error(py, e))?;
    let result = PyDict::new(py);
    result.set_item("pool", stats.pool)?;
    result.set_item("reserved", stats.reserved)?;
    result.set_item("used", stats.used)?;
    result.set_item("threshold", stats.threshold)?;
    result.set_item("cache_bytes", stats.cache_bytes)?;
    result.set_item("cache_entries", stats.cache_entries)?;
    Ok(result)
}
/// Testing builds: `cuMemPoolTrimTo(pool, 0)` on the ordinal's owned pool.
#[cfg(all(feature = "testing", feature = "cuda"))]
#[pyfunction]
fn _native_cuda_pool_trim(py: Python<'_>, ordinal: usize) -> PyResult<()> {
    py.detach(|| seqblaze_cuda::testing_hooks::pool_trim(ordinal))
        .map_err(|e| error(py, e))
}
#[cfg(all(feature = "testing", feature = "cuda"))]
#[pyfunction]
fn _native_cuda_register_defect(count: u64) {
    seqblaze_cuda::testing_hooks::set_register_defect(count);
}
/// Testing builds: refuse pool allocations (`count` None disarms).
#[cfg(all(feature = "testing", feature = "cuda"))]
#[pyfunction(signature=(count=None, skip=0))]
fn _native_cuda_pool_fail(count: Option<u64>, skip: u64) {
    seqblaze_cuda::testing_hooks::set_pool_fail(count, skip);
}
#[cfg(all(feature = "testing", feature = "cuda"))]
#[pyfunction]
fn _native_cuda_pool_fails_fired() -> u64 {
    seqblaze_cuda::testing_hooks::pool_fails_fired()
}
/// Testing builds: force the pool fallback mode before the first device open.
#[cfg(all(feature = "testing", feature = "cuda"))]
#[pyfunction]
fn _native_cuda_pool_fallback(py: Python<'_>) -> PyResult<()> {
    seqblaze_cuda::testing_hooks::force_pool_fallback().map_err(|e| error(py, e))
}
/// Testing builds: `cuCtxSynchronize` on the ordinal's context, GIL released.
#[cfg(all(feature = "testing", feature = "cuda"))]
#[pyfunction]
fn _native_cuda_device_synchronize(py: Python<'_>, ordinal: usize) -> PyResult<()> {
    py.detach(|| seqblaze_cuda::testing_hooks::device_synchronize(ordinal))
        .map_err(|e| error(py, e))
}
/// Testing builds: Drop-path host waits made and those that found the stream busy.
#[cfg(all(feature = "testing", feature = "cuda"))]
#[pyfunction]
fn _native_cuda_wait_stats(py: Python<'_>) -> PyResult<Bound<'_, PyDict>> {
    let stats = seqblaze_cuda::testing_hooks::wait_stats();
    let result = PyDict::new(py);
    result.set_item("drop_waits", stats.drop_waits)?;
    result.set_item("drop_busy", stats.drop_busy)?;
    Ok(result)
}
#[cfg(all(feature = "testing", feature = "cuda"))]
#[pyfunction]
fn _native_cuda_wait_stats_reset() {
    seqblaze_cuda::testing_hooks::wait_stats_reset();
}
/// Testing builds: device slices accessed through a stream other than their
/// allocation stream (S1 (b)); must stay 0.
#[cfg(all(feature = "testing", feature = "cuda"))]
#[pyfunction]
fn _native_cuda_stream_misuse_count() -> u64 {
    seqblaze_cuda::testing_hooks::stream_misuse_count()
}

#[cfg(feature = "testing")]
#[pyfunction]
fn _native_cuda_poisoned_bytes() -> u64 {
    seqblaze_cuda::cuda_poisoned_bytes()
}

#[cfg(feature = "testing")]
#[pyfunction]
fn _native_cuda_resident_poison_stats(py: Python<'_>) -> PyResult<Bound<'_, PyDict>> {
    let result = PyDict::new(py);
    for (route, counts) in ["k1_fast", "k1_continuation", "k3_fast", "k3_continuation"]
        .into_iter()
        .zip(seqblaze_cuda::cuda_resident_poison_stats())
    {
        let entry = PyDict::new(py);
        for (metric, value) in ["pairs", "fills", "bytes"].into_iter().zip(counts) {
            entry.set_item(metric, value)?;
        }
        result.set_item(route, entry)?;
    }
    Ok(result)
}

/// Testing builds: cumulative T2 output-refill counts, excluding allocation and scratch.
#[cfg(feature = "testing")]
#[pyfunction]
fn _native_cuda_unary_refill_stats(py: Python<'_>) -> PyResult<Bound<'_, PyDict>> {
    let stats = seqblaze_cuda::cuda_unary_refill_stats();
    let result = PyDict::new(py);
    result.set_item("chunks", stats.chunks)?;
    result.set_item("rows", stats.rows)?;
    result.set_item("field_bytes", stats.field_bytes)?;
    result.set_item("passed_bytes", stats.passed_bytes)?;
    Ok(result)
}

/// Testing builds: replace the process-wide CUDA launch override and return
/// the previous one (None means the default for that field). `bound_retain_bytes`
/// (R_b of duplex-mfe-bound.md §8 *Sweeps*, revision 3) is replaced too, but it is not
/// one of the fourteen keys: the returned dict carries it only when it was set (QB18).
#[cfg(feature = "testing")]
#[pyfunction(signature=(threads_per_block=None, piece_bytes=None, pairs_per_chunk=None, bytes_per_launch=None, thread_words=None, device_words=None, pair_steps=None, steps_per_launch=None, thread_rows=None, lane_rows=None, pair_cells=None, cells_per_launch=None, group_lanes=None, band_bytes=None, bound_retain_bytes=None))]
#[allow(clippy::too_many_arguments)]
fn _native_cuda_launch(
    py: Python<'_>,
    threads_per_block: Option<u64>,
    piece_bytes: Option<u64>,
    pairs_per_chunk: Option<u64>,
    bytes_per_launch: Option<u64>,
    thread_words: Option<u64>,
    device_words: Option<u64>,
    pair_steps: Option<u64>,
    steps_per_launch: Option<u64>,
    thread_rows: Option<u64>,
    lane_rows: Option<u64>,
    pair_cells: Option<u64>,
    cells_per_launch: Option<u64>,
    group_lanes: Option<u64>,
    band_bytes: Option<u64>,
    bound_retain_bytes: Option<u64>,
) -> PyResult<Bound<'_, PyDict>> {
    if bound_retain_bytes.is_some_and(|value| value > seqblaze_cuda::MAX_BOUND_RETAIN_BYTES) {
        return Err(error(
            py,
            EngineError::Parameter(format!(
                "bound_retain_bytes={}; use None or an int in [0, 2**40]",
                bound_retain_bytes.unwrap_or(0)
            )),
        ));
    }
    let threads_per_block = threads_per_block
        .map(|value| {
            u32::try_from(value).map_err(|_| {
                EngineError::Parameter(format!(
                    "threads_per_block={value}; use a multiple of 32 in [32, 1024] or None"
                ))
            })
        })
        .transpose()
        .map_err(|e| error(py, e))?;
    let previous = seqblaze_cuda::set_launch_override(seqblaze_cuda::LaunchOverride {
        threads_per_block,
        piece_bytes,
        pairs_per_chunk,
        bytes_per_launch,
        thread_words,
        device_words,
        pair_steps,
        steps_per_launch,
        thread_rows,
        lane_rows,
        pair_cells,
        cells_per_launch,
        group_lanes,
        band_bytes,
    })
    .map_err(|e| error(py, e))?;
    // Checked above, so this cannot fail after the launch override changed.
    let previous_retain =
        seqblaze_cuda::set_bound_retain_bytes(bound_retain_bytes).map_err(|e| error(py, e))?;
    let result = PyDict::new(py);
    result.set_item("threads_per_block", previous.threads_per_block)?;
    result.set_item("piece_bytes", previous.piece_bytes)?;
    result.set_item("pairs_per_chunk", previous.pairs_per_chunk)?;
    result.set_item("bytes_per_launch", previous.bytes_per_launch)?;
    result.set_item("thread_words", previous.thread_words)?;
    result.set_item("device_words", previous.device_words)?;
    result.set_item("pair_steps", previous.pair_steps)?;
    result.set_item("steps_per_launch", previous.steps_per_launch)?;
    result.set_item("thread_rows", previous.thread_rows)?;
    result.set_item("lane_rows", previous.lane_rows)?;
    result.set_item("pair_cells", previous.pair_cells)?;
    result.set_item("cells_per_launch", previous.cells_per_launch)?;
    result.set_item("group_lanes", previous.group_lanes)?;
    result.set_item("band_bytes", previous.band_bytes)?;
    if let Some(bytes) = previous_retain {
        result.set_item("bound_retain_bytes", bytes)?;
    }
    Ok(result)
}

/// Public calls use the minimum-work grain; the PF-RUN bridge passes "exact" so
/// tiny fixtures still exercise every worker partition and merge.
fn parse_grain(py: Python<'_>, grain: &str) -> PyResult<Grain> {
    match grain {
        "auto" => Ok(Grain::Auto),
        "exact" => Ok(Grain::Exact),
        _ => Err(error(
            py,
            EngineError::Parameter("grain must be auto or exact".into()),
        )),
    }
}

/// Trailing arguments: `kernel`, `grain`, `device` (CUDA ordinal, None for the
/// CPU), then levenshtein's `mode`, `details` and `orientation`, align's
/// resolved configuration dict `align` and duplex's per-call pack `duplex`.
/// `input_digests` (CPU only): the response also carries the logical input
/// facts of the validated inputs, as a resident run's does.
#[pyfunction(signature=(a,b,kind,i,j,self_pairs,keep,cutoff,check_bases,strategy,threads,k=None,kernel="hamming",grain="auto",device=None,mode="global",details=Vec::new(),orientation=None,align=None,duplex=None,progress=None,input_digests=false))]
#[allow(clippy::too_many_arguments)]
pub fn _native_run<'py>(
    py: Python<'py>,
    a: PyRef<'py, PySeqTable>,
    b: PyRef<'py, PySeqTable>,
    kind: &str,
    i: Option<PyRef<'_, PyIndexColumn>>,
    j: Option<PyRef<'_, PyIndexColumn>>,
    self_pairs: bool,
    keep: &str,
    cutoff: Option<i64>,
    check_bases: bool,
    strategy: &str,
    threads: usize,
    k: Option<usize>,
    kernel: &str,
    grain: &str,
    device: Option<u64>,
    mode: &str,
    details: Vec<String>,
    orientation: Option<&str>,
    align: Option<Bound<'py, PyDict>>,
    duplex: Option<Bound<'py, PyAny>>,
    progress: Option<&str>,
    input_digests: bool,
) -> PyResult<Bound<'py, PyDict>> {
    duplex::prepare_bounded(
        py,
        duplex.as_ref(),
        strategy,
        keep,
        cutoff.is_some(),
        &details,
    );
    let kernel = RuntimeKernel::configure(
        kernel,
        mode,
        &details,
        check_bases,
        align.as_ref(),
        duplex.as_ref(),
    )
    .map_err(|e| error(py, e))?;
    kernel.check_cutoff(cutoff).map_err(|e| error(py, e))?;
    if device.is_some() && kernel.mfe_lse() {
        // LK6 (duplex-mfe.md §8.6.5): the private Session sweeps carry no row terms.
        return Err(error(
            py,
            EngineError::Parameter(
                "duplex model='mfe' lse_dg_kcal has no private CUDA session path; use the \
                 public sb.duplex call"
                    .into(),
            ),
        ));
    }
    let grain = parse_grain(py, grain)?;
    // One cancel scope for the call's compute region (cancellation-progress.md §3.7).
    let scope = cancel::Call::new(progress);
    kernel.check_backend(device).map_err(|e| error(py, e))?;
    // Device resolution precedes pair and sequence validation (cuda-backend.md).
    let device = resolve_device(py, device)?;
    let work = oriented_work_source(
        py,
        kind,
        i,
        j,
        self_pairs,
        orientation,
        a.table.len(),
        b.table.len(),
    )?;
    let options = ExecOptions {
        strategy: if strategy == "bounded" {
            Strategy::Bounded
        } else {
            Strategy::Full
        },
        threads,
        grain,
        sink: match keep {
            "hits" => Sink::Hits,
            "each" => Sink::Each,
            "rowstats" => Sink::RowStats,
            "topk" => Sink::TopK,
            "matrix" => Sink::Matrix,
            _ => return Err(error(py, EngineError::Parameter("unsupported sink".into()))),
        },
        cutoff,
        check_bases,
        k,
    };
    let (a, b) = (
        a.table.host().map_err(|e| error(py, e))?,
        b.table.host().map_err(|e| error(py, e))?,
    );
    match &device {
        // tm on Pairs: the CUDA backend's int64-key path (tm-pairs-cuda.md §6, §7).
        Some(device) if matches!(kernel, RuntimeKernel::TmPairs(_)) => {
            let RuntimeKernel::TmPairs(tm_pairs) = &kernel else {
                unreachable!("matched above")
            };
            let k: &seqblaze_thermo::TmPairs = tm_pairs;
            let options = options.with_key::<i64>().map_err(|e| error(py, e))?;
            let computed = scope.detach(py, || {
                // Built only for a resolved device, after the options (§8 step 3).
                let call = kernel.device_call()?;
                exec::validated(k, &work, &a, &b, &options, |a, b, report| {
                    if work.n_pairs == 0 {
                        // No launch and no allocation: the shared zero-work output.
                        exec::compute(k, &work, a, b, &options, report)
                            .map(|output| (output, Some(0)))
                    } else {
                        seqblaze_cuda::run_i64(device, k, call, &work, (a, b), &options, report)
                            .map(|run| (run.output, Some(run.host_pairs)))
                    }
                })
            });
            let (result, host_pairs) = scope.finish(py, computed)?;
            respond(py, &kernel, strategy, result, host_pairs)
        }
        // K3 (duplex-pf-cuda.md §7): the 64-bit-key entry; Ẑ on the device, keys,
        // terms and sinks on the host.
        Some(device) if matches!(kernel, RuntimeKernel::DuplexPf(_)) => {
            let RuntimeKernel::DuplexPf(k) = &kernel else {
                unreachable!("matched above")
            };
            let k: &seqblaze_thermo::DuplexPf = k;
            let options = options.with_key::<i64>().map_err(|e| error(py, e))?;
            let computed = scope.detach(py, || {
                // The device pack is built only for a resolved device (§4).
                let call = kernel.device_call()?;
                exec::validated(k, &work, &a, &b, &options, |a, b, report| {
                    if work.n_pairs == 0 {
                        // No launch and no allocation: the shared zero-work output.
                        exec::compute(k, &work, a, b, &options, report)
                            .map(|output| (output, Some(0)))
                    } else {
                        seqblaze_cuda::run_pf(device, k, call, &work, (a, b), &options, report)
                            .map(|run| (run.output, Some(run.host_pairs)))
                    }
                })
            });
            let (result, host_pairs) = scope.finish(py, computed)?;
            respond(py, &kernel, strategy, result, host_pairs)
        }
        // K2's int64 keys (primer3-cuda.md §7), through the int64-key entry it shares
        // with `tm` on Pairs.
        Some(device) if matches!(kernel, RuntimeKernel::DuplexPrimer3(_)) => {
            let RuntimeKernel::DuplexPrimer3(k) = &kernel else {
                unreachable!("matched above")
            };
            let k: &seqblaze_thermo::DimerPrimer3 = k;
            let options = options.with_key::<i64>().map_err(|e| error(py, e))?;
            let computed = scope.detach(py, || {
                // The device pack is built only for a resolved device, before the
                // sequence checks (primer3-cuda.md §4, §9).
                let call = kernel.device_call()?;
                exec::validated(k, &work, &a, &b, &options, |a, b, report| {
                    if work.n_pairs == 0 {
                        // No launch and no allocation: the shared zero-work output.
                        exec::compute(k, &work, a, b, &options, report)
                            .map(|output| (output, Some(0)))
                    } else {
                        seqblaze_cuda::run_i64(device, k, call, &work, (a, b), &options, report)
                            .map(|run| (run.output, Some(run.host_pairs)))
                    }
                })
            });
            let (result, host_pairs) = scope.finish(py, computed)?;
            respond(py, &kernel, strategy, result, host_pairs)
        }
        // `host_pairs` (levenshtein-cuda.md §5.5) is reported for device calls only.
        Some(device) => {
            let options = options.with_key::<i32>().map_err(|e| error(py, e))?;
            let computed = scope.detach(py, || {
                // The device pack is built only for a resolved device, never for CPU
                // calls; `check_backend` rejected a device for a kernel the backend
                // lacks above.
                let call = kernel.device_call()?;
                dispatch_i32!(
                    &kernel,
                    |k| exec::validated(k, &work, &a, &b, &options, |a, b, report| {
                        if work.n_pairs == 0 {
                            // No launch and no allocation: the shared zero-work output.
                            exec::compute(k, &work, a, b, &options, report)
                                .map(|output| (output, Some(0)))
                        } else {
                            // The kernel object serves routed pairs and host detail fields.
                            seqblaze_cuda::run(device, k, call, &work, (a, b), &options, report)
                                .map(|run| (run.output, Some(run.host_pairs)))
                        }
                    },),
                    Err(kernel.cuda_unavailable())
                )
            });
            let (result, host_pairs) = scope.finish(py, computed)?;
            respond(py, &kernel, strategy, result, host_pairs)
        }
        // Hash the CPU call's own validated views after computation. Even a
        // successful speculative digest reservation can consume resources (or
        // a refusal schedule) needed by validation, changing its first error.
        None if input_digests => dispatch!(&kernel, |k| {
            let options = options.with_key().map_err(|e| error(py, e))?;
            let computed = scope.detach(py, || {
                exec::validated(k, &work, &a, &b, &options, |a, b, report| {
                    let output = exec::compute(k, &work, a, b, &options, report)?;
                    let digests = provenance::call_digests((a, b), &work, options.threads)?;
                    Ok((output, digests))
                })
            });
            let (result, digests) = scope.finish(py, computed)?;
            let response = respond(py, &kernel, strategy, result, None)?;
            response.set_item("input_digests", provenance::digests_dict(py, &digests)?)?;
            Ok(response)
        }),
        None => dispatch!(&kernel, |k| {
            let options = options.with_key().map_err(|e| error(py, e))?;
            let computed = scope.detach(py, || exec::run(k, &work, &a, &b, &options));
            let result = scope.finish(py, computed)?;
            respond(py, &kernel, strategy, result, None)
        }),
    }
}

/// The Python response of a run: primary and detail columns, row statistics or TopK
/// entries, the execution report.
fn respond<'py, S: Erase>(
    py: Python<'py>,
    kernel: &RuntimeKernel,
    strategy: &str,
    result: RunOutput<S>,
    host_pairs: Option<usize>,
) -> PyResult<Bound<'py, PyDict>> {
    let columns = PyDict::new(py);
    let details = result.columns.details;
    let mut tie_rows = None;
    if let Some(topk) = result.columns.topk {
        tie_rows = Some(topk.tie_at_k.iter().filter(|&&flag| flag != 0).count());
        set_primary_and_details(py, &columns, kernel, S::keys(topk.dist), details)?;
        columns.set_item("i", column(py, Numeric::I64(topk.i))?)?;
        columns.set_item("j", column(py, Numeric::I64(topk.j))?)?;
        columns.set_item("rank", column(py, Numeric::I32(topk.rank))?)?;
        columns.set_item("offsets", column(py, Numeric::I64(topk.offsets))?)?;
        columns.set_item("tie_at_k", column(py, Numeric::Bool(topk.tie_at_k))?)?;
    } else if let Some(stats) = result.columns.rowstats {
        set_primary(py, &columns, kernel, true, S::keys(stats.best_dist))?;
        columns.set_item("argbest", column(py, Numeric::I64(stats.argbest))?)?;
        columns.set_item("n_best", column(py, Numeric::I32(stats.n_best))?)?;
        if let Some(count) = stats.count {
            columns.set_item("count", column(py, Numeric::I64(count))?)?;
        }
        if let Some(values) = stats.row_value {
            // lse_dg_kcal of the row-term kernels: K3 (duplex-pf.md §9.4) and K1
            // (duplex-mfe.md §8.6).
            columns.set_item("lse_dg_kcal", column(py, Numeric::F64(values))?)?;
        }
    } else {
        set_primary_and_details(py, &columns, kernel, S::keys(result.columns.dist), details)?;
    }
    if let Some(values) = result.columns.passed {
        columns.set_item("passed", column(py, Numeric::Bool(values))?)?;
    }
    if let Some(values) = result.columns.i {
        columns.set_item("i", column(py, Numeric::I64(values))?)?;
    }
    if let Some(values) = result.columns.j {
        columns.set_item("j", column(py, Numeric::I64(values))?)?;
    }
    let report = PyDict::new(py);
    let strategy_info = PyDict::new(py);
    strategy_info.set_item("requested", strategy)?;
    strategy_info.set_item("effective", strategy)?;
    strategy_info.set_item("fallback_reason", py.None())?;
    report.set_item("strategy", strategy_info)?;
    let counts = PyDict::new(py);
    counts.set_item("certified", result.counts.certified)?;
    counts.set_item("pruned", result.counts.pruned)?;
    counts.set_item("full", result.counts.full)?;
    report.set_item("counts", counts)?;
    report.set_item("threads", result.threads)?;
    let flags = PyDict::new(py);
    flags.set_item("n_illegal_seqs", result.n_illegal_seqs)?;
    if let Some(tie_rows) = tie_rows {
        flags.set_item("tie_at_k", tie_rows)?;
    }
    report.set_item("flags", flags)?;
    let scans = PyDict::new(py);
    scans.set_item("a", result.validation_scans[0])?;
    scans.set_item("b", result.validation_scans[1])?;
    report.set_item("validation_scans", scans)?;
    if let Some(host_pairs) = host_pairs {
        report.set_item("host_pairs", host_pairs)?;
    }
    if let Some(replay) = result.topk_replay {
        let info = PyDict::new(py);
        let counts = PyDict::new(py);
        counts.set_item("full", replay.full)?;
        counts.set_item("pruned", replay.pruned)?;
        counts.set_item("skipped", replay.skipped)?;
        // Additive (duplex-mfe-bound.md §6.5, QB4): certified + full + pruned + skipped.
        counts.set_item("certified", replay.certified)?;
        info.set_item("passes", 2)?;
        info.set_item("replay", counts)?;
        report.set_item("topk", info)?;
    }
    let response = PyDict::new(py);
    response.set_item("columns", columns)?;
    response.set_item("shape", result.shape)?;
    response.set_item("report", report)?;
    Ok(response)
}

pub fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PySeqTable>()?;
    #[cfg(feature = "cuda")]
    device::register(module)?;
    #[cfg(feature = "cuda")]
    device_dlpack::register(module)?;
    #[cfg(feature = "cuda")]
    device_indices::register(module)?;
    #[cfg(feature = "cuda")]
    device_selection::register(module)?;
    #[cfg(feature = "cuda")]
    device_run::register(module)?;
    #[cfg(feature = "cuda")]
    device_tiles::register(module)?;
    module.add_function(wrap_pyfunction!(_native_pairs, module)?)?;
    module.add_function(wrap_pyfunction!(_native_plan, module)?)?;
    module.add_function(wrap_pyfunction!(_native_seqs, module)?)?;
    module.add_function(wrap_pyfunction!(_native_run, module)?)?;
    module.add_function(wrap_pyfunction!(_cuda_device, module)?)?;
    #[cfg(feature = "testing")]
    {
        module.add_function(wrap_pyfunction!(_native_cuda_launch, module)?)?;
        module.add_function(wrap_pyfunction!(_native_cuda_poison, module)?)?;
        module.add_function(wrap_pyfunction!(_native_cuda_resident_depth_cap, module)?)?;
        module.add_function(wrap_pyfunction!(_native_cuda_p3_wave, module)?)?;
        module.add_function(wrap_pyfunction!(_native_cuda_resident_launch, module)?)?;
        module.add_function(wrap_pyfunction!(_native_cuda_resident_wave, module)?)?;
        module.add_function(wrap_pyfunction!(_native_cuda_pf_capture, module)?)?;
        #[cfg(feature = "cuda")]
        {
            module.add_function(wrap_pyfunction!(_native_resident_launch_streams, module)?)?;
            module.add_function(wrap_pyfunction!(
                _native_resident_launch_streams_reset,
                module
            )?)?;
            module.add_function(wrap_pyfunction!(_native_cuda_pool_stats, module)?)?;
            module.add_function(wrap_pyfunction!(_native_cuda_register_defect, module)?)?;
            module.add_function(wrap_pyfunction!(_native_cuda_pool_trim, module)?)?;
            module.add_function(wrap_pyfunction!(_native_cuda_pool_fail, module)?)?;
            module.add_function(wrap_pyfunction!(_native_cuda_pool_fails_fired, module)?)?;
            module.add_function(wrap_pyfunction!(_native_cuda_pool_fallback, module)?)?;
            module.add_function(wrap_pyfunction!(_native_cuda_device_synchronize, module)?)?;
            module.add_function(wrap_pyfunction!(_native_cuda_wait_stats, module)?)?;
            module.add_function(wrap_pyfunction!(_native_cuda_wait_stats_reset, module)?)?;
            module.add_function(wrap_pyfunction!(_native_cuda_stream_misuse_count, module)?)?;
        }
        module.add_function(wrap_pyfunction!(_native_cuda_poisoned_bytes, module)?)?;
        module.add_function(wrap_pyfunction!(
            _native_cuda_resident_poison_stats,
            module
        )?)?;
        module.add_function(wrap_pyfunction!(_native_cuda_unary_refill_stats, module)?)?;
        module.add_function(wrap_pyfunction!(_native_primer3_device_plan, module)?)?;
        module.add_function(wrap_pyfunction!(_native_align_lanes, module)?)?;
        module.add_function(wrap_pyfunction!(_native_edit_lanes, module)?)?;
        module.add_function(wrap_pyfunction!(
            crate::alloc_fault::_native_alloc_fault,
            module
        )?)?;
        module.add_function(wrap_pyfunction!(
            crate::alloc_fault::_native_alloc_faults_fired,
            module
        )?)?;
    }
    arrow::register(module)?;
    cancel::register(module)?;
    fastx::register(module)?;
    fpbracket::register(module)?;
    indices::register(module)?;
    manifest::register(module)?;
    numpy::register(module)?;
    provenance::register(module)?;
    selection::register(module)?;
    tiles::register(module)?;
    tm::register(module)?;
    duplex::register(module)?;
    hairpin::register(module)?;
    Ok(())
}
