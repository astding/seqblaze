//! FASTA/FASTQ readers (D92, docs/wiki/io/fasta-readers.md): sources, the
//! GIL-free parse loop with signal checks, `SeqBatch` buffers and Arrow export.
use super::{arrow::export, error, PySeqTable};
use pyo3::{
    prelude::*,
    sync::PyOnceLock,
    types::{PyByteArray, PyByteArrayMethods, PyBytes, PyDict, PyString},
};
use seqblaze_core::{seq::SeqTable, validate::EngineError, SeqBatch};
use seqblaze_io::{Batch, FastxReader, Format, FormatError, Input, Limits, ReadError};
use std::{
    ffi::CStr,
    io::{self, Read},
    path::PathBuf,
    sync::{Arc, Condvar, Mutex},
    time::{Duration, Instant},
};

/// Bytes requested per file-object call (spec §16: about 1 MiB).
const FILE_READ: usize = 1 << 20;
/// Longest stretch of parsing between signal checks (D86: at least every 100 ms).
const SIGNAL_INTERVAL: Duration = Duration::from_millis(50);

fn parameter(py: Python<'_>, message: impl Into<String>) -> PyErr {
    error(py, EngineError::Parameter(message.into()))
}

fn parse_format(py: Python<'_>, format: &str) -> PyResult<Format> {
    match format {
        "fasta" => Ok(Format::Fasta),
        "fastq" => Ok(Format::Fastq),
        _ => Err(parameter(py, "format must be 'fasta' or 'fastq'")),
    }
}

/// Exceptions raised by a Python file object, kept to be re-raised unchanged.
type Slot = Arc<Mutex<Option<PyErr>>>;

/// What an error message and an `OSError` need to know about the source. The
/// name and path stay Python objects: a path's str may hold surrogate escapes
/// (a file name that is not UTF-8), and `.source` and `.filename` return it as
/// given.
struct Context {
    name: Py<PyAny>,
    path: Option<Py<PyAny>>,
    raised: Slot,
}

/// A binary file object read through `readinto` (preferred) or `read`. Each
/// call re-acquires the GIL (§9.1); exceptions are stored and re-raised as is.
struct FileReader {
    file: Py<PyAny>,
    readinto: bool,
    scratch: Option<Py<PyByteArray>>,
    pending: Vec<u8>,
    pending_at: usize,
    raised: Slot,
}

impl FileReader {
    /// Copy the current read straight into the Rust caller's buffer. Only bytes
    /// it cannot accept need an owned pending copy for a later call.
    fn receive(pending: &mut Vec<u8>, bytes: &[u8], buf: &mut [u8]) -> usize {
        let n = bytes.len().min(buf.len());
        pending.extend_from_slice(&bytes[n..]);
        buf[..n].copy_from_slice(&bytes[..n]);
        n
    }

    fn call(&mut self, py: Python<'_>, buf: &mut [u8]) -> PyResult<usize> {
        let file = self.file.bind(py);
        self.pending.clear();
        self.pending_at = 0;
        if self.readinto {
            let scratch = match &self.scratch {
                Some(scratch) => scratch.bind(py).clone(),
                None => {
                    let scratch = PyByteArray::new_with(py, FILE_READ, |_| Ok(()))?;
                    self.scratch = Some(scratch.clone().unbind());
                    scratch
                }
            };
            let result = file.call_method1("readinto", (&scratch,))?;
            if result.is_none() {
                return Err(parameter(
                    py,
                    "the file object's readinto returned None (a non-blocking stream with no data); pass a blocking binary file object",
                ));
            }
            let n: usize = result.extract()?;
            if n > scratch.len() {
                return Err(parameter(
                    py,
                    "the file object's readinto reported more bytes than the buffer holds",
                ));
            }
            // SAFETY: no Python code runs while the bytearray's bytes are copied;
            // this shared view ends before the GIL is released.
            let bytes = unsafe { &scratch.as_bytes()[..n] };
            return Ok(Self::receive(&mut self.pending, bytes, buf));
        }
        let result = file.call_method1("read", (FILE_READ,))?;
        if result.is_none() {
            return Err(parameter(
                py,
                "the file object's read returned None (a non-blocking stream with no data); pass a blocking binary file object",
            ));
        }
        if result.is_instance_of::<PyString>() {
            return Err(parameter(
                py,
                "the file object's read returned str (a text-mode file); open the file with \"rb\"",
            ));
        }
        if let Ok(bytes) = result.cast::<PyBytes>() {
            Ok(Self::receive(&mut self.pending, bytes.as_bytes(), buf))
        } else {
            let bytes = py.import("builtins")?.getattr("bytes")?.call1((result,))?;
            Ok(Self::receive(
                &mut self.pending,
                bytes.cast::<PyBytes>()?.as_bytes(),
                buf,
            ))
        }
    }
}

impl Read for FileReader {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        if self.pending_at == self.pending.len() {
            return Python::attach(|py| self.call(py, buf)).map_err(|raised| {
                *self.raised.lock().expect("reader error slot") = Some(raised);
                io::Error::other("the Python file object raised")
            });
        }
        let rest = &self.pending[self.pending_at..];
        let n = rest.len().min(buf.len());
        buf[..n].copy_from_slice(&rest[..n]);
        self.pending_at += n;
        Ok(n)
    }
}

fn os_error(py: Python<'_>, failure: &io::Error, filename: Option<&Py<PyAny>>) -> PyErr {
    if let (Some(code), Some(name)) = (failure.raw_os_error(), filename) {
        let built = (|| -> PyResult<PyErr> {
            let text: String = py
                .import("os")?
                .getattr("strerror")?
                .call1((code,))?
                .extract()?;
            let value = py
                .import("builtins")?
                .getattr("OSError")?
                .call1((code, text, name))?;
            Ok(PyErr::from_value(value))
        })();
        return built.unwrap_or_else(|e| e);
    }
    PyErr::from(io::Error::new(failure.kind(), failure.to_string()))
}

/// Open the source (§2.1): `kind` is "path", "buffer" (a C-contiguous
/// memoryview of format B) or "file", as resolved by the Python wrapper.
fn open_source(
    py: Python<'_>,
    kind: &str,
    source: &Bound<'_, PyAny>,
    name: Py<PyAny>,
) -> PyResult<(Input, Context)> {
    let raised: Slot = Arc::new(Mutex::new(None));
    match kind {
        "path" => {
            // PathBuf extraction encodes with the file-system encoding and
            // surrogateescape, so a non-UTF-8 file name reaches the OS unchanged.
            let path: PathBuf = source.extract()?;
            let filename = source.clone().unbind();
            let file = std::fs::File::open(&path).map_err(|e| os_error(py, &e, Some(&filename)))?;
            Ok((
                Input::reader(Box::new(file)),
                Context {
                    name,
                    path: Some(filename),
                    raised,
                },
            ))
        }
        "buffer" => {
            // A 1-D C-contiguous NumPy uint8 view of the caller's buffer, made by
            // the Python wrapper (np.frombuffer over a memoryview, no copy); the
            // view retains the buffer export.
            let (address, _readonly): (usize, bool) = source
                .getattr("__array_interface__")?
                .get_item("data")?
                .extract()?;
            let len: usize = source.getattr("nbytes")?.extract()?;
            let owner: Py<PyAny> = source.clone().unbind();
            // SAFETY: the retained view keeps the bytes alive; the caller must
            // not modify them until the read ends or the iterator closes (§2.1).
            let input = unsafe { Input::borrowed(address as *const u8, len, Box::new(owner)) };
            Ok((
                input,
                Context {
                    name,
                    path: None,
                    raised,
                },
            ))
        }
        "file" => {
            let readinto = source.hasattr("readinto")?;
            let reader = FileReader {
                file: source.clone().unbind(),
                readinto,
                scratch: None,
                pending: Vec::new(),
                pending_at: 0,
                raised: Arc::clone(&raised),
            };
            Ok((
                Input::reader(Box::new(reader)),
                Context {
                    name,
                    path: None,
                    raised,
                },
            ))
        }
        _ => Err(parameter(py, "unknown source kind")),
    }
}

fn format_error(py: Python<'_>, failure: &FormatError, context: &Context, iter: bool) -> PyErr {
    let mut message = format!(
        ": record {} (0-based), line {}: {}",
        failure.record, failure.line, failure.detail
    );
    if !message.ends_with('.') {
        message.push('.');
    }
    for hint in &failure.hints {
        message.push('\n');
        message.push_str(hint);
    }
    message.push('\n');
    message.push_str(if iter {
        "Batches already returned stay valid; this iterator is closed."
    } else {
        "Nothing was returned; no record was skipped."
    });
    let built = (|| -> PyResult<PyErr> {
        // The name is joined in Python: it may hold surrogate escapes.
        let name = context.name.bind(py);
        let message = name.str()?.add(PyString::new(py, &message))?;
        let value = py
            .import("seqblaze._errors")?
            .getattr("FormatError")?
            .call1((message,))?;
        value.setattr("source", name)?;
        value.setattr("record", failure.record)?;
        value.setattr("line", failure.line)?;
        value.setattr("reason", failure.reason.code())?;
        Ok(PyErr::from_value(value))
    })();
    built.unwrap_or_else(|e| e)
}

fn read_error(py: Python<'_>, failure: ReadError, context: &Context, iter: bool) -> PyErr {
    match failure {
        ReadError::Format(failure) => format_error(py, &failure, context, iter),
        ReadError::Io(failure) => {
            if let Some(raised) = context.raised.lock().expect("reader error slot").take() {
                return raised;
            }
            os_error(py, &failure, context.path.as_ref())
        }
        ReadError::Alloc { bytes } => error(
            py,
            EngineError::Device(format!(
                "unable to reserve {bytes} bytes of host memory for reader columns"
            )),
        ),
    }
}

enum Pause {
    Done,
    Ready,
    Tick,
}

/// Parse with the GIL released, re-acquiring it only to check signals.
fn drive(
    py: Python<'_>,
    reader: &mut FastxReader,
    until_ready: bool,
) -> PyResult<Result<(), ReadError>> {
    loop {
        let pause = py.detach(|| {
            let start = Instant::now();
            loop {
                if until_ready && reader.has_ready() {
                    return Ok(Pause::Ready);
                }
                if !reader.step()? {
                    return Ok(Pause::Done);
                }
                if start.elapsed() >= SIGNAL_INTERVAL {
                    return Ok(Pause::Tick);
                }
            }
        });
        match pause {
            Ok(Pause::Tick) => py.check_signals()?,
            Ok(Pause::Done | Pause::Ready) => return Ok(Ok(())),
            Err(failure) => return Ok(Err(failure)),
        }
    }
}

/// One immutable byte buffer of a batch column, viewed by NumPy without a copy.
#[pyclass(frozen, module = "seqblaze._core", name = "_BatchBuffer")]
struct PyBatchBuffer {
    batch: Arc<SeqBatch>,
    offsets: bool,
}

#[pymethods]
impl PyBatchBuffer {
    #[getter]
    fn __array_interface__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let endian = if cfg!(target_endian = "little") {
            '<'
        } else {
            '>'
        };
        let (pointer, length, typestr) = if self.offsets {
            let offsets = self.batch.offsets();
            (
                offsets.as_ptr() as usize,
                offsets.len(),
                format!("{endian}i8"),
            )
        } else {
            let data = self.batch.data();
            (data.as_ptr() as usize, data.len(), "|u1".to_string())
        };
        let result = PyDict::new(py);
        result.set_item("version", 3)?;
        result.set_item("shape", (length,))?;
        result.set_item("typestr", typestr)?;
        result.set_item("data", (pointer, true))?;
        Ok(result)
    }
}

/// Rust-only selection proof for our actual frozen reader buffer class.
/// Return initialized length, never Vec capacity; the caller must retain the
/// actual object and validate its ndarray view lies inside this allocation.
pub(super) fn selection_immutable_span(object: &Bound<'_, PyAny>) -> Option<(*const u8, usize)> {
    let buffer = object.cast::<PyBatchBuffer>().ok()?.borrow();
    if buffer.offsets {
        let values = buffer.batch.offsets();
        Some((
            values.as_ptr().cast::<u8>(),
            values.len().checked_mul(std::mem::size_of::<i64>())?,
        ))
    } else {
        let values = buffer.batch.data();
        Some((values.as_ptr(), values.len()))
    }
}

const ID: &CStr = c"id";
const DESCRIPTION: &CStr = c"description";
const SEQ: &CStr = c"seq";
const QUAL: &CStr = c"qual";

/// The Rust side of `sb.SeqBatch`: four immutable columns and the offset.
#[pyclass(frozen, module = "seqblaze._core", name = "_FastxBatch")]
pub struct PyFastxBatch {
    format: Format,
    offset: u64,
    id: Arc<SeqBatch>,
    description: Arc<SeqBatch>,
    seq: Arc<SeqBatch>,
    qual: Option<Arc<SeqBatch>>,
    table: PyOnceLock<Py<PySeqTable>>,
}

impl PyFastxBatch {
    fn new(format: Format, batch: Batch) -> Self {
        Self {
            format,
            offset: batch.offset,
            id: Arc::new(batch.id),
            description: Arc::new(batch.description),
            seq: Arc::new(batch.seq),
            qual: batch.qual.map(Arc::new),
            table: PyOnceLock::new(),
        }
    }

    fn columns(&self) -> Vec<(&'static CStr, Arc<SeqBatch>)> {
        let mut columns = vec![
            (ID, Arc::clone(&self.id)),
            (DESCRIPTION, Arc::clone(&self.description)),
            (SEQ, Arc::clone(&self.seq)),
        ];
        if let Some(qual) = &self.qual {
            columns.push((QUAL, Arc::clone(qual)));
        }
        columns
    }

    fn column(&self, py: Python<'_>, name: &str) -> PyResult<(&'static CStr, Arc<SeqBatch>)> {
        self.columns()
            .into_iter()
            .find(|(label, _)| label.to_bytes() == name.as_bytes())
            .ok_or_else(|| parameter(py, format!("no column {name:?} in this batch")))
    }
}

#[pymethods]
impl PyFastxBatch {
    #[getter]
    fn format(&self) -> &'static str {
        self.format.name()
    }
    #[getter]
    fn offset(&self) -> u64 {
        self.offset
    }
    fn __len__(&self) -> usize {
        self.seq.len()
    }
    /// (offsets owner, data owner) of one column, for zero-copy NumPy views.
    fn buffers(&self, py: Python<'_>, name: &str) -> PyResult<(Py<PyAny>, Py<PyAny>)> {
        let (_, batch) = self.column(py, name)?;
        let offsets = Py::new(
            py,
            PyBatchBuffer {
                batch: Arc::clone(&batch),
                offsets: true,
            },
        )?;
        let data = Py::new(
            py,
            PyBatchBuffer {
                batch,
                offsets: false,
            },
        )?;
        Ok((offsets.into_any(), data.into_any()))
    }
    /// Arrow export of one column (non-nullable large_binary).
    fn column_export(&self, py: Python<'_>, name: &str) -> PyResult<Py<export::ArrowExport>> {
        let column = self.column(py, name)?;
        export::binary_export(py, vec![column])
    }
    /// Arrow export of the batch: one struct of its columns (§6.3).
    fn export(&self, py: Python<'_>) -> PyResult<Py<export::ArrowExport>> {
        export::binary_export(py, self.columns())
    }
    /// The cached zero-copy `SeqTable` over the `seq` column (§6.4).
    fn table(&self, py: Python<'_>) -> PyResult<Py<PySeqTable>> {
        let table = self.table.get_or_try_init(py, || {
            Py::new(
                py,
                PySeqTable {
                    table: super::TableStorage::Host(Arc::new(SeqTable::from_shared(Arc::clone(
                        &self.seq,
                    )))),
                },
            )
        })?;
        Ok(table.clone_ref(py))
    }
}

fn limits(records: Option<u64>, bytes: Option<u64>, whole: bool) -> Limits {
    Limits {
        records,
        bytes,
        whole,
    }
}

/// `read_*`: the whole input as one batch.
#[pyfunction]
fn _fastx_read(
    py: Python<'_>,
    format: &str,
    kind: &str,
    source: &Bound<'_, PyAny>,
    name: Py<PyAny>,
) -> PyResult<PyFastxBatch> {
    let format = parse_format(py, format)?;
    let (input, context) = open_source(py, kind, source, name)?;
    let mut reader = FastxReader::new(input, format, limits(None, None, true));
    if let Err(failure) = drive(py, &mut reader, false)? {
        return Err(read_error(py, failure, &context, false));
    }
    let batch = reader.pop().expect("a whole read emits exactly one batch");
    Ok(PyFastxBatch::new(format, batch))
}

struct StreamState {
    reader: FastxReader,
    format: Format,
    context: Context,
    pending: Option<ReadError>,
}

enum StreamSlot {
    Idle(Box<StreamState>),
    Busy,
    Closed,
}

/// `iter_*`: consecutive batches; calls to `next` are serialized (§9.1).
#[pyclass(frozen, module = "seqblaze._core", name = "_FastxStream")]
pub struct PyFastxStream {
    slot: Mutex<StreamSlot>,
    idle: Condvar,
}

impl PyFastxStream {
    /// Take the state for exclusive use, waiting (GIL released) while another
    /// thread works on it. `None` once closed.
    fn acquire(&self, py: Python<'_>) -> Option<Box<StreamState>> {
        loop {
            let mut slot = self.slot.lock().expect("stream lock");
            match std::mem::replace(&mut *slot, StreamSlot::Busy) {
                StreamSlot::Idle(state) => return Some(state),
                StreamSlot::Closed => {
                    *slot = StreamSlot::Closed;
                    return None;
                }
                StreamSlot::Busy => {
                    drop(slot);
                    py.detach(|| {
                        let slot = self.slot.lock().expect("stream lock");
                        let _slot = self
                            .idle
                            .wait_while(slot, |s| matches!(s, StreamSlot::Busy))
                            .expect("stream lock");
                    });
                }
            }
        }
    }

    fn release(&self, state: Option<Box<StreamState>>) {
        let mut slot = self.slot.lock().expect("stream lock");
        *slot = match state {
            Some(state) => StreamSlot::Idle(state),
            None => StreamSlot::Closed,
        };
        self.idle.notify_all();
    }
}

fn advance(py: Python<'_>, state: &mut StreamState) -> PyResult<Option<PyFastxBatch>> {
    loop {
        if let Some(batch) = state.reader.pop() {
            return Ok(Some(PyFastxBatch::new(state.format, batch)));
        }
        if let Some(failure) = state.pending.take() {
            return Err(read_error(py, failure, &state.context, true));
        }
        if state.reader.is_done() {
            return Ok(None);
        }
        if let Err(failure) = drive(py, &mut state.reader, true)? {
            // Batches closed before the failure are returned first.
            state.pending = Some(failure);
        }
    }
}

#[pymethods]
impl PyFastxStream {
    /// The next batch, or None at the end. An error or an interrupt closes
    /// the stream.
    fn next(&self, py: Python<'_>) -> PyResult<Option<PyFastxBatch>> {
        let Some(mut state) = self.acquire(py) else {
            return Ok(None);
        };
        let result = advance(py, &mut state);
        let keep = matches!(result, Ok(Some(_)));
        if keep {
            self.release(Some(state));
        } else {
            self.release(None);
            drop(state);
        }
        result
    }

    /// Close the stream and release its source (idempotent).
    fn close(&self, py: Python<'_>) {
        let state = self.acquire(py);
        self.release(None);
        drop(state);
    }
}

#[pyfunction]
#[pyo3(signature = (format, kind, source, name, chunk_records, chunk_bytes))]
fn _fastx_stream(
    py: Python<'_>,
    format: &str,
    kind: &str,
    source: &Bound<'_, PyAny>,
    name: Py<PyAny>,
    chunk_records: Option<u64>,
    chunk_bytes: Option<u64>,
) -> PyResult<PyFastxStream> {
    let format = parse_format(py, format)?;
    let (input, context) = open_source(py, kind, source, name)?;
    let reader = FastxReader::new(input, format, limits(chunk_records, chunk_bytes, false));
    Ok(PyFastxStream {
        slot: Mutex::new(StreamSlot::Idle(Box::new(StreamState {
            reader,
            format,
            context,
            pending: None,
        }))),
        idle: Condvar::new(),
    })
}

pub fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyBatchBuffer>()?;
    module.add_class::<PyFastxBatch>()?;
    module.add_class::<PyFastxStream>()?;
    module.add_function(wrap_pyfunction!(_fastx_read, module)?)?;
    module.add_function(wrap_pyfunction!(_fastx_stream, module)?)?;
    Ok(())
}
