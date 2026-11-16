//! Serial CPU C Stream import, retaining independently owned chunks.
//! C ABI adapted from Apache Arrow (Apache-2.0):
//! https://arrow.apache.org/docs/format/CStreamInterface.html
use super::{
    all_null_table, array_descriptor, error, export_error, null_array_rows, parameter,
    schema_format, table_object, ArrowArrayStream, ArrowOwner,
};
use pyo3::{
    prelude::*,
    types::{PyCapsule, PyCapsuleMethods},
};
use seqblaze_core::{
    seq::{ArrowChunk, SeqTable},
    validate::EngineError,
};
use std::{
    ffi::{c_int, CStr},
    mem::align_of,
};

struct StreamOwner {
    stream: ArrowArrayStream,
}
impl Drop for StreamOwner {
    fn drop(&mut self) {
        Python::attach(|_| {
            if let Some(release) = self.stream.release {
                // SAFETY: only this moved base owns the live native stream.
                unsafe { release(&mut self.stream) };
            }
        });
    }
}
impl StreamOwner {
    fn error(&mut self, py: Python<'_>, stage: &str, status: c_int) -> PyErr {
        // Copy the description now: another callback/release can invalidate it.
        let description = self.stream.get_last_error.and_then(|get_error| {
            // SAFETY: called only immediately after a failed native operation.
            let pointer = unsafe { get_error(&mut self.stream) };
            if pointer.is_null() {
                None
            } else {
                // SAFETY: the last-error callback returns a live NUL-terminated string.
                Some(
                    unsafe { CStr::from_ptr(pointer) }
                        .to_string_lossy()
                        .into_owned(),
                )
            }
        });
        parameter(
            py,
            format!(
                "Arrow stream {stage} failed (status {status}): {}",
                description.as_deref().unwrap_or("no error description")
            ),
        )
    }
}

#[pyfunction]
fn _native_seqs_arrow_stream<'py>(
    py: Python<'py>,
    object: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let export = object
        .call_method0("__arrow_c_stream__")
        .map_err(|e| export_error(py, "stream", e))?;
    let capsule = export
        .cast::<PyCapsule>()
        .map_err(|_| parameter(py, "Arrow stream export must be a PyCapsule"))?;
    let pointer = capsule
        .pointer_checked(Some(c"arrow_array_stream"))
        .map_err(|_| parameter(py, "expected an arrow_array_stream capsule"))?
        .cast::<ArrowArrayStream>()
        .as_ptr();
    if (pointer as usize) % align_of::<ArrowArrayStream>() != 0 {
        return Err(parameter(py, "Arrow stream base must be aligned"));
    }
    // SAFETY: named aligned capsules supply a valid C Stream base. No arbitrary
    // Python code runs before moving it and nulling the source release callback.
    let mut owner = unsafe {
        if (*pointer).release.is_none() {
            return Err(parameter(
                py,
                "Arrow stream capsule was already consumed or released",
            ));
        }
        let stream = pointer.read();
        (*pointer).release = None;
        StreamOwner { stream }
    };
    let (Some(get_schema), Some(get_next), Some(_)) = (
        owner.stream.get_schema,
        owner.stream.get_next,
        owner.stream.get_last_error,
    ) else {
        return Err(parameter(
            py,
            "Arrow stream requires schema/next/error callbacks",
        ));
    };
    let mut schema = ArrowOwner::empty();
    // SAFETY: the stream is live, and the caller owns this initially empty output.
    let status = unsafe { get_schema(&mut owner.stream, &mut schema.schema) };
    if status != 0 {
        return Err(owner.error(py, "get_schema", status));
    }
    let (format, width) = schema_format(py, &schema.schema)?;
    let mut chunks = Vec::new();
    let mut null_rows = 0usize;
    loop {
        let mut chunk = ArrowOwner::empty();
        // SAFETY: one thread serializes pulls on this live CPU stream, and owns
        // the empty output. A partial live output is also cleaned up on errors.
        let status = unsafe { get_next(&mut owner.stream, &mut chunk.array) };
        if status != 0 {
            return Err(owner.error(py, "get_next", status));
        }
        if chunk.array.release.is_none() {
            break;
        }
        if format == "n" {
            null_rows = null_rows
                .checked_add(null_array_rows(py, &chunk.array)?)
                .ok_or_else(|| {
                    error(
                        py,
                        EngineError::Overflow("Arrow stream length exceeds platform limit".into()),
                    )
                })?;
            continue;
        }
        let descriptor = array_descriptor(py, &chunk.array, format, width)?;
        seqblaze_core::detail::try_reserve(&mut chunks, 1, "Arrow stream chunk owners")
            .map_err(|e| error(py, e))?;
        chunks.push(ArrowChunk {
            descriptor,
            owner: Box::new(chunk),
        });
    }
    if format == "n" {
        return table_object(py, all_null_table(py, null_rows)?);
    }
    // SAFETY: all pulled CPU exports are independently owned. Core checks every
    // chunk's metadata and records its null rows before publishing the table.
    let table = unsafe { SeqTable::from_arrow_chunks(chunks, format) }.map_err(|e| error(py, e))?;
    // Schema/stream owners drop on return; retained array lifetimes are independent.
    table_object(py, table)
}

pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(_native_seqs_arrow_stream, module)?)?;
    Ok(())
}
