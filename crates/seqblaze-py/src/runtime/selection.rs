//! Existing host Result selection with integer normalization and bit-copy gather.
//! Immutable output owners; no FP arithmetic, caller-state bracket or GIL detach.
use super::{column, error, fastx, Numeric, OwnedColumn};
use pyo3::{
    exceptions::PyIndexError,
    prelude::*,
    types::{PyBytes, PyDict, PyTuple},
};
use seqblaze_core::validate::EngineError;
use std::{
    mem::size_of,
    sync::atomic::{AtomicU8, Ordering},
};

fn parameter(py: Python<'_>, message: impl Into<String>) -> PyErr {
    error(py, EngineError::Parameter(message.into()))
}

/// Generalized one-dimensional geometry from core seq/view.rs::check_span.
/// IndexArray itself only accepts signed 32/64-bit descriptors; selection
/// also needs bool and signed/unsigned 8/16-bit values, without reinterpretation.
struct Array {
    address: usize,
    rows: usize,
    stride: isize,
    width: usize,
    kind: u8,
    little: bool,
    _owner: Py<PyAny>,
    // Present only after real built-in owner identification and span validation.
    // Retain that terminal owner independently of the ndarray/memoryview chain.
    immutable: Option<ImmutableSpan>,
}

impl Array {
    fn new(py: Python<'_>, object: &Bound<'_, PyAny>) -> PyResult<Self> {
        let numpy = py.import("numpy")?;
        let array_type = numpy.getattr("ndarray")?;
        if !object.is_instance(&array_type)?
            || numpy
                .getattr("ma")?
                .getattr("isMaskedArray")?
                .call1((object,))?
                .is_truthy()?
        {
            return Err(parameter(
                py,
                "selection buffers must be unmasked NumPy arrays",
            ));
        }
        let kwargs = PyDict::new(py);
        kwargs.set_item("type", &array_type)?;
        // Base ndarray metadata bypasses subclass overrides and row callbacks.
        let view = array_type.getattr("view")?.call((object,), Some(&kwargs))?;
        if view.getattr("ndim")?.extract::<usize>()? != 1 {
            return Err(parameter(
                py,
                "selection requires a 1-D integer array or bool mask",
            ));
        }
        let dtype = view.getattr("dtype")?;
        let kind = dtype.getattr("kind")?.extract::<String>()?;
        let width = dtype.getattr("itemsize")?.extract::<usize>()?;
        let kind = *kind.as_bytes().first().unwrap_or(&0);
        if !matches!(
            (kind, width),
            (b'i' | b'u', 1 | 2 | 4 | 8) | (b'f', 8) | (b'b', 1)
        ) {
            return Err(parameter(
                py,
                "selection buffers require bool, integer or float64 dtype",
            ));
        }
        let order = dtype.getattr("byteorder")?.extract::<String>()?;
        let little = match order.as_str() {
            "<" => true,
            ">" => false,
            "=" | "|" => cfg!(target_endian = "little"),
            _ => return Err(parameter(py, "unsupported selection buffer byte order")),
        };
        let (rows,): (usize,) = view.getattr("shape")?.extract()?;
        let (stride,): (isize,) = view.getattr("strides")?.extract()?;
        let (address, _readonly): (usize, bool) = view
            .getattr("__array_interface__")?
            .get_item("data")?
            .extract()?;
        if rows > isize::MAX as usize {
            return Err(parameter(
                py,
                "selection buffer rows exceed platform limits",
            ));
        }
        if rows != 0 {
            let extent = (rows - 1) as i128 * stride as i128;
            let low = address as i128 + extent.min(0);
            let high = address as i128 + extent.max(0) + width as i128;
            if extent < isize::MIN as i128
                || extent > isize::MAX as i128
                || address == 0
                || low < 0
                || high > usize::MAX as i128
                || high - low > isize::MAX as i128
            {
                return Err(parameter(
                    py,
                    "selection buffer stride/span exceeds platform limits",
                ));
            }
        }
        let owner = py
            .import("builtins")?
            .getattr("memoryview")?
            .call1((&view,))?
            .unbind();
        let immutable = immutable_owner(py, &view, &array_type, address, rows, stride, width)?;
        Ok(Self {
            address,
            rows,
            stride,
            width,
            kind,
            little,
            _owner: owner,
            immutable,
        })
    }

    /// Raw relaxed byte reads: no Rust shared slice spans borrowed storage.
    /// The native ndarray descriptor is trusted to describe a valid allocation,
    /// exactly as in the existing adapters. Geometry and row are checked first.
    fn bits(&self, row: usize) -> u64 {
        debug_assert!(row < self.rows);
        let origin = self
            .immutable
            .as_ref()
            .map_or(self.address as *const u8, |span| span.origin);
        let base = origin.wrapping_offset(row as isize * self.stride);
        if self.immutable.is_some() {
            // SAFETY: the complete strided byte span was checked against a
            // confirmed immutable OwnedColumn/_BatchBuffer/bytes allocation, retained
            // below. Integer unaligned loads preserve f64 payload bits and do
            // not manufacture shared slices over a foreign mutable buffer.
            let bits = unsafe {
                match self.width {
                    1 => u64::from(base.read()),
                    2 => u64::from(base.cast::<u16>().read_unaligned()),
                    4 => u64::from(base.cast::<u32>().read_unaligned()),
                    8 => base.cast::<u64>().read_unaligned(),
                    _ => unreachable!("validated selection element width"),
                }
            };
            return if self.width == 1 || self.little == cfg!(target_endian = "little") {
                bits
            } else {
                match self.width {
                    2 => u64::from((bits as u16).swap_bytes()),
                    4 => u64::from((bits as u32).swap_bytes()),
                    8 => bits.swap_bytes(),
                    _ => unreachable!("validated selection element width"),
                }
            };
        }
        let mut result = 0u64;
        for at in 0..self.width {
            // SAFETY: validated ndarray geometry, retained memoryview, held GIL,
            // and the existing no-mutation/no-reallocation-during-call contract.
            let byte =
                unsafe { (&*base.wrapping_add(at).cast::<AtomicU8>()).load(Ordering::Relaxed) };
            let shift = if self.little { at } else { self.width - 1 - at };
            result |= u64::from(byte) << (8 * shift);
        }
        result
    }

    /// Storage encodings for gathered fields: retain the original byte order.
    /// Python wraps the immutable result using a metadata view of source dtype.
    fn storage_bits(&self, row: usize) -> u64 {
        let bits = self.bits(row);
        if self.width == 1 || self.little == cfg!(target_endian = "little") {
            bits
        } else {
            match self.width {
                2 => u64::from((bits as u16).swap_bytes()),
                4 => u64::from((bits as u32).swap_bytes()),
                8 => bits.swap_bytes(),
                _ => unreachable!("validated selection element width"),
            }
        }
    }

    fn signed(&self, row: usize) -> i64 {
        let shift = 64 - 8 * self.width;
        ((self.bits(row) << shift) as i64) >> shift
    }

    fn integer(&self, row: usize) -> i128 {
        if self.kind == b'i' {
            self.signed(row) as i128
        } else {
            self.bits(row) as i128
        }
    }
}

/// Obtain backing from actual immutable engine owners or built-in bytes. A read-only
/// ndarray flag, a Python type name, __array_interface__ or an arbitrary owner
/// attribute is not proof: foreign storage keeps its AtomicU8 read path.
struct ImmutableSpan {
    // Derived from the real owner's allocation pointer, never reconstructed
    // solely from __array_interface__'s integer address.
    origin: *const u8,
    _root: Py<PyAny>,
}

fn own_span(object: &Bound<'_, PyAny>) -> Option<(*const u8, usize)> {
    if let Ok(owner) = object.cast::<OwnedColumn>() {
        let owner = owner.borrow();
        let span = match &owner.values {
            Numeric::I32(v) => (
                v.as_ptr().cast::<u8>(),
                v.len().checked_mul(size_of::<i32>())?,
            ),
            Numeric::I64(v) => (
                v.as_ptr().cast::<u8>(),
                v.len().checked_mul(size_of::<i64>())?,
            ),
            Numeric::F64(v) => (
                v.as_ptr().cast::<u8>(),
                v.len().checked_mul(size_of::<f64>())?,
            ),
            Numeric::Bool(v) | Numeric::U8(v) => (v.as_ptr(), v.len()),
        };
        Some(span)
    } else if let Ok(bytes) = object.cast::<PyBytes>() {
        // _locked user-built columns retain a real immutable Python bytes
        // object. Its actual pointer/len is proof; an arbitrary readonly
        // ndarray or a foreign __array_interface__ owner is not.
        let bytes = bytes.as_bytes();
        Some((bytes.as_ptr(), bytes.len()))
    } else {
        // Narrow Rust-only accessor lives inside fastx, whose buffer class and
        // Arc<SeqBatch> fields stay private. It also checks a real PyO3 type.
        fastx::selection_immutable_span(object)
    }
}

fn immutable_owner(
    py: Python<'_>,
    view: &Bound<'_, PyAny>,
    ndarray: &Bound<'_, PyAny>,
    address: usize,
    rows: usize,
    stride: isize,
    width: usize,
) -> PyResult<Option<ImmutableSpan>> {
    let mut owner = view.clone();
    let base_descriptor = ndarray.getattr("base")?;
    // Normal NumPy base chains are acyclic and short. A conservative depth
    // limit falls back to the existing safe foreign path, without allocation.
    for _ in 0..64 {
        if let Some((allocation, bytes)) = own_span(&owner) {
            let start = allocation as usize;
            let end = start.checked_add(bytes).ok_or_else(|| {
                parameter(py, "immutable selection owner span exceeds platform limits")
            })?;
            let extent = rows.saturating_sub(1) as i128 * stride as i128;
            let low = address as i128 + extent.min(0);
            let high = address as i128 + extent.max(0) + if rows == 0 { 0 } else { width as i128 };
            if low < start as i128 || high > end as i128 {
                // Known owner + impossible view is invalid, not a reason to
                // fall back and read outside that allocation through atomics.
                return Err(parameter(
                    py,
                    "selection view lies outside its immutable owner",
                ));
            }
            return Ok(Some(ImmutableSpan {
                origin: allocation.wrapping_add(address - start),
                _root: owner.unbind(),
            }));
        }
        if !owner.is_instance(ndarray)? {
            return Ok(None);
        }
        // Use the genuine ndarray descriptor, bypassing subclass .base
        // overrides. A foreign object cannot nominate an immutable owner.
        owner = base_descriptor.call_method1("__get__", (&owner,))?;
        if owner.is_none() {
            return Ok(None);
        }
    }
    Ok(None)
}

// Only these payload encodings admit every bit pattern. Keep the copy helper
// private and constrained so Rust bool/char cannot be instantiated accidentally.
trait CopyBits: Copy {}
impl CopyBits for i32 {}
impl CopyBits for i64 {}
impl CopyBits for f64 {}
impl CopyBits for u8 {}

impl Array {
    fn can_copy_range<T: CopyBits>(&self) -> bool {
        self.immutable.is_some()
            && self.width == size_of::<T>()
            && self.stride == self.width as isize
            && (self.width == 1 || self.little == cfg!(target_endian = "little"))
    }

    /// Copy a contiguous selected payload/run, preserving every input bit.
    /// This private generic is used ONLY with i32, i64, f64 and u8: all bit
    /// patterns are valid for these types, and output capacity is reserved by
    /// the caller. It is never instantiated for Rust bool or other types.
    fn copy_range<T: CopyBits>(&self, start: usize, end: usize, out: &mut Vec<T>) -> bool {
        if !self.can_copy_range::<T>() || start > end || end > self.rows {
            return false;
        }
        let count = end - start;
        let Some(new_len) = out.len().checked_add(count) else {
            return false;
        };
        if new_len > out.capacity() {
            return false;
        }
        let Some(bytes) = count.checked_mul(self.width) else {
            return false;
        };
        let Some(offset) = start.checked_mul(self.width) else {
            return false;
        };
        if bytes != 0 {
            // SAFETY: actual immutable built-in owner, retained for the call;
            // validated contiguous range inside its allocation; fresh Vec
            // cannot alias a live input; spare capacity covers all bytes.
            // Copy as u8 so an unaligned reinterpreted ndarray is supported.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    self.immutable
                        .as_ref()
                        .expect("confirmed immutable source")
                        .origin
                        .add(offset),
                    out.as_mut_ptr().add(out.len()).cast::<u8>(),
                    bytes,
                );
            }
        }
        // SAFETY: exactly count complete T encodings were initialized above;
        // zero-byte copies add zero elements since these T are nonzero-sized.
        unsafe {
            out.set_len(new_len);
        }
        true
    }
}

/// Consecutive source indices form copy ranges; requested order and repeats
/// remain intact. Foreign/noncontiguous/non-native-endian sources retain the
/// original per-element raw-reader path. No selected-position vector is added.
fn gather_values<T: CopyBits>(
    py: Python<'_>,
    source: &Array,
    idx: &Array,
    what: &str,
    decode: impl Fn(&Array, usize) -> T,
) -> PyResult<Vec<T>> {
    let mut out = reserve(py, idx.rows, what)?;
    let bulk = source.can_copy_range::<T>();
    let mut slot = 0;
    while slot < idx.rows {
        let row = at(idx, slot, source.rows)?;
        if bulk {
            let mut next_slot = slot + 1;
            let mut end = row + 1;
            while next_slot < idx.rows {
                if at(idx, next_slot, source.rows)? != end {
                    break;
                }
                next_slot += 1;
                end += 1;
            }
            if source.copy_range(row, end, &mut out) {
                slot = next_slot;
                continue;
            }
        }
        out.push(decode(source, row));
        slot += 1;
    }
    Ok(out)
}

fn reserve<T>(py: Python<'_>, rows: usize, what: &str) -> PyResult<Vec<T>> {
    let bytes = rows
        .checked_mul(size_of::<T>())
        .filter(|&bytes| bytes <= isize::MAX as usize)
        .ok_or_else(|| parameter(py, "selection allocation bytes exceed platform limits"))?;
    let mut result = Vec::new();
    result.try_reserve_exact(rows).map_err(|reason| {
        error(
            py,
            seqblaze_core::detail::device_error(format_args!(
                "unable to reserve {bytes} bytes for {what}: {reason}; no result was published"
            )),
        )
    })?;
    Ok(result)
}

fn normalized_value(value: i128, n: usize) -> PyResult<i64> {
    if value < -(n as i128) || value >= n as i128 {
        return Err(PyIndexError::new_err("selection index is out of range"));
    }
    let normalized = if value < 0 { value + n as i128 } else { value };
    Ok(normalized as i64)
}

/// The Python array-like parser supplies an ndarray. Each retains requested
/// order/repeats; canonical Hits/TopK positions are sorted and deduplicated
/// after negative-index normalization. Bool masks emit increasing positions.
#[pyfunction(signature=(key, n, *, allow_mask=false, canonical=false))]
fn _native_selection_indices<'py>(
    py: Python<'py>,
    key: &Bound<'py, PyAny>,
    n: usize,
    allow_mask: bool,
    canonical: bool,
) -> PyResult<Py<PyAny>> {
    if n > i64::MAX as usize {
        return Err(parameter(py, "selection row count exceeds int64 range"));
    }
    let source = Array::new(py, key)?;
    let mut values;
    if source.kind == b'b' && allow_mask {
        if source.rows != n {
            return Err(parameter(
                py,
                format!("selection mask has {} rows; expected {n}", source.rows),
            ));
        }
        let count = (0..source.rows)
            .filter(|&row| source.bits(row) != 0)
            .count();
        values = reserve(py, count, "selection indices")?;
        for row in 0..source.rows {
            if source.bits(row) != 0 {
                values.push(row as i64);
            }
        }
    } else if matches!(source.kind, b'i' | b'u') {
        // Preserve current _normalized priority: any uint64 above int64 max
        // raises the range-specific IndexError before signed-range checking.
        if source.kind == b'u'
            && source.width == 8
            && (0..source.rows).any(|row| source.bits(row) > i64::MAX as u64)
        {
            return Err(PyIndexError::new_err("selection index exceeds int64 range"));
        }
        values = reserve(py, source.rows, "selection indices")?;
        for row in 0..source.rows {
            values.push(normalized_value(source.integer(row), n)?);
        }
        if canonical {
            values.sort_unstable();
            values.dedup();
        }
    } else {
        return Err(parameter(
            py,
            "take requires 1-D integer indices; use r[mask] for bool masks",
        ));
    }
    Ok(column(py, Numeric::I64(values))?.into_any())
}

fn indices(py: Python<'_>, object: &Bound<'_, PyAny>) -> PyResult<Array> {
    let result = Array::new(py, object)?;
    if result.kind != b'i' || result.width != 8 {
        return Err(parameter(
            py,
            "internal selection requires normalized int64 indices",
        ));
    }
    Ok(result)
}

fn at(source: &Array, slot: usize, rows: usize) -> PyResult<usize> {
    let value = source.signed(slot);
    if value < 0 || value as u128 >= rows as u128 {
        return Err(PyIndexError::new_err("selection index is out of range"));
    }
    Ok(value as usize)
}

fn gather_numeric(py: Python<'_>, object: &Bound<'_, PyAny>, idx: &Array) -> PyResult<Py<PyAny>> {
    let source = Array::new(py, object)?;
    let numeric = match (source.kind, source.width) {
        (b'i', 4) => Numeric::I32(gather_values(
            py,
            &source,
            idx,
            "selected int32 column",
            |s, row| s.storage_bits(row) as u32 as i32,
        )?),
        (b'i', 8) => Numeric::I64(gather_values(
            py,
            &source,
            idx,
            "selected int64 column",
            |s, row| s.storage_bits(row) as i64,
        )?),
        // Bit construction/moves only: retain sNaNs, payloads and -0.
        (b'f', 8) => Numeric::F64(gather_values(
            py,
            &source,
            idx,
            "selected float64 column",
            |s, row| f64::from_bits(s.storage_bits(row)),
        )?),
        (b'b', 1) => Numeric::Bool(gather_values(
            py,
            &source,
            idx,
            "selected bool column",
            |s, row| s.bits(row) as u8,
        )?),
        _ => {
            return Err(parameter(
                py,
                "Result fields require int32/int64/float64/bool",
            ))
        }
    };
    Ok(column(py, numeric)?.into_any())
}

fn span(
    py: Python<'_>,
    offsets: &Array,
    row: usize,
    value_rows: usize,
) -> PyResult<(usize, usize)> {
    let start = offsets.signed(row);
    let end = offsets.signed(row + 1);
    if start < 0 || end < start || end as u128 > value_rows as u128 {
        return Err(parameter(
            py,
            "ragged offsets lie outside their values buffer",
        ));
    }
    Ok((start as usize, end as usize))
}

fn gather_ragged(py: Python<'_>, parts: &Bound<'_, PyTuple>, idx: &Array) -> PyResult<Py<PyAny>> {
    if parts.len() != 4 {
        return Err(parameter(
            py,
            "ragged selection requires kind/offsets/values/validity",
        ));
    }
    let kind: String = parts.get_item(0)?.extract()?;
    let offsets = Array::new(py, &parts.get_item(1)?)?;
    let values = Array::new(py, &parts.get_item(2)?)?;
    let validity = parts.get_item(3)?;
    // BinaryColumn reader input can omit its all-valid source bitmap. This
    // avoids allocating N flags just to select M << N rows. Other _parts()
    // callers already provide a real bool bitmap.
    let valid = if validity.is_none() {
        None
    } else {
        Some(Array::new(py, &validity)?)
    };
    let list = kind == "list";
    if !matches!(kind.as_str(), "string" | "binary" | "list")
        || offsets.kind != b'i'
        || offsets.width != 8
        || offsets.rows == 0
        || valid
            .as_ref()
            .is_some_and(|v| v.kind != b'b' || v.width != 1 || offsets.rows - 1 != v.rows)
        || (list && (values.kind != b'i' || values.width != 8))
        || (!list && (values.kind != b'u' || values.width != 1))
    {
        return Err(parameter(
            py,
            "invalid ragged selection buffer dtypes or lengths",
        ));
    }
    // Descriptor geometry/dtypes/lengths are checked above. Only selected
    // offset pairs need validation: no payload from unselected rows is read,
    // and a small take must not scan N source offsets. Nonzero origins work.
    let rows = offsets.rows - 1;
    let mut total = 0usize;
    // Selected pass 1: exact payload element count, not an upper-bound reserve.
    for k in 0..idx.rows {
        let row = at(idx, k, rows)?;
        let (start, end) = span(py, &offsets, row, values.rows)?;
        total = total
            .checked_add(end - start)
            .filter(|&n| n <= i64::MAX as usize)
            .ok_or_else(|| parameter(py, "selected ragged offsets exceed int64 range"))?;
    }
    let offset_rows = idx
        .rows
        .checked_add(1)
        .ok_or_else(|| parameter(py, "selected ragged rows exceed platform limits"))?;
    let mut out_offsets = reserve(py, offset_rows, "selected ragged offsets")?;
    // A BinaryColumn with no source bitmap remains all-valid without an
    // output bitmap. Explicit source validity still gathers its M flags.
    let mut out_valid = if kind == "binary" && valid.is_none() {
        None
    } else {
        Some(reserve(py, idx.rows, "selected ragged validity")?)
    };
    out_offsets.push(0i64);
    // Selected pass 2: copy each selected payload range directly. There is no
    // int64 positions array per byte/value, and no Python callback in the loop.
    let payload = if list {
        let mut out_values = reserve(py, total, "selected list values")?;
        for k in 0..idx.rows {
            let row = at(idx, k, rows)?;
            let (start, end) = span(py, &offsets, row, values.rows)?;
            if !values.copy_range(start, end, &mut out_values) {
                for value in start..end {
                    out_values.push(values.storage_bits(value) as i64);
                }
            }
            if let Some(out_valid) = &mut out_valid {
                out_valid.push(valid.as_ref().map_or(1, |valid| valid.bits(row) as u8));
            }
            out_offsets.push(out_values.len() as i64);
        }
        Numeric::I64(out_values)
    } else {
        let mut out_values = reserve(py, total, "selected string/binary data")?;
        for k in 0..idx.rows {
            let row = at(idx, k, rows)?;
            let (start, end) = span(py, &offsets, row, values.rows)?;
            if !values.copy_range(start, end, &mut out_values) {
                for value in start..end {
                    out_values.push(values.bits(value) as u8);
                }
            }
            if let Some(out_valid) = &mut out_valid {
                out_valid.push(valid.as_ref().map_or(1, |valid| valid.bits(row) as u8));
            }
            out_offsets.push(out_values.len() as i64);
        }
        Numeric::U8(out_values)
    };
    let validity = match out_valid {
        Some(values) => column(py, Numeric::Bool(values))?.into_any(),
        None => py.None(),
    };
    let result = (
        kind.as_str(),
        column(py, Numeric::I64(out_offsets))?,
        column(py, payload)?,
        validity,
    )
        .into_pyobject(py)?
        .into_any()
        .unbind();
    Ok(result)
}

fn gather_one(py: Python<'_>, value: &Bound<'_, PyAny>, idx: &Array) -> PyResult<Py<PyAny>> {
    match value.cast::<PyTuple>() {
        Ok(parts) => gather_ragged(py, parts, idx),
        Err(_) => gather_numeric(py, value, idx),
    }
}

#[pyfunction]
fn _native_selection_gather<'py>(
    py: Python<'py>,
    columns: &Bound<'py, PyDict>,
    normalized: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyDict>> {
    let idx = indices(py, normalized)?;
    let output = PyDict::new(py);
    for (name, value) in columns.iter() {
        output.set_item(name, gather_one(py, &value, &idx)?)?;
    }
    // The local dictionary is published only if every selected column succeeds.
    Ok(output)
}

#[pyfunction]
fn _native_selection_gather_column<'py>(
    py: Python<'py>,
    value: &Bound<'py, PyAny>,
    normalized: &Bound<'py, PyAny>,
) -> PyResult<Py<PyAny>> {
    let idx = indices(py, normalized)?;
    gather_one(py, value, &idx)
}

/// Rebuild only the row structure of an entry-selected TopK. The caller
/// gathers original ranks and all payload columns separately; canonical entry
/// positions preserve their (i, rank, j) order and distinct partners. This
/// helper validates its own i descriptor/range/grouping and never reranks.
#[pyfunction]
fn _native_topk_structure<'py>(
    py: Python<'py>,
    selected_i: &Bound<'py, PyAny>,
    n_rows: usize,
    k: usize,
) -> PyResult<Bound<'py, PyDict>> {
    if k == 0 {
        return Err(parameter(py, "TopK k must be positive"));
    }
    if n_rows > i64::MAX as usize {
        return Err(parameter(py, "TopK row count exceeds int64 range"));
    }
    let rows = indices(py, selected_i)?;
    let offset_count = n_rows
        .checked_add(1)
        .ok_or_else(|| parameter(py, "TopK offsets exceed platform limits"))?;
    let mut offsets = reserve(py, offset_count, "selected TopK offsets")?;
    offsets.resize(offset_count, 0i64);
    let mut flags = reserve(py, n_rows, "selected TopK tie flags")?;
    let mut previous = None;
    for slot in 0..rows.rows {
        let row = at(&rows, slot, n_rows)?;
        if previous.is_some_and(|previous| row < previous) {
            return Err(parameter(
                py,
                "selected TopK entries must be grouped in row order",
            ));
        }
        previous = Some(row);
        // Total entries fit int64 because Array bounds rows by isize::MAX.
        offsets[row + 1] += 1;
    }
    let mut tied_rows = 0usize;
    for row in 0..n_rows {
        let tied = offsets[row + 1] as u128 > k as u128;
        flags.push(u8::from(tied));
        tied_rows += usize::from(tied);
        offsets[row + 1] += offsets[row];
    }
    let output = PyDict::new(py);
    output.set_item("offsets", column(py, Numeric::I64(offsets))?)?;
    output.set_item("tie_at_k", column(py, Numeric::Bool(flags))?)?;
    output.set_item("tie_rows", tied_rows)?;
    Ok(output)
}

pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(_native_selection_indices, module)?)?;
    module.add_function(wrap_pyfunction!(_native_selection_gather, module)?)?;
    module.add_function(wrap_pyfunction!(_native_selection_gather_column, module)?)?;
    module.add_function(wrap_pyfunction!(_native_topk_structure, module)?)?;
    Ok(())
}
