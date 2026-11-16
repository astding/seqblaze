//! Result C Stream and column C Data export, without an Arrow runtime
//! dependency: int32/int64/float64/bool columns, large_utf8 / large_binary /
//! large_list<int64> detail columns (int64 offsets, values, validity bitmap),
//! and the non-nullable large_binary columns of reader batches (`sb.SeqBatch`,
//! D92), whose buffers a Rust `Arc` owns so that release never needs the GIL.
//! ABI and ownership follow Apache Arrow (Apache-2.0):
//! https://arrow.apache.org/docs/format/CDataInterface.html
//! https://arrow.apache.org/docs/format/CStreamInterface.html
use super::{error, parameter, ArrowArray, ArrowArrayStream, ArrowSchema};
use pyo3::{
    exceptions::PyMemoryError,
    prelude::*,
    types::{PyCapsule, PyDict, PyTuple},
};
use seqblaze_core::{detail::try_reserve_exact, SeqBatch};
use std::{
    any::Any,
    ffi::{c_char, c_int, c_void, CStr, CString},
    panic::{catch_unwind, AssertUnwindSafe},
    ptr,
    sync::Arc,
};

const INVALID: &CStr = c"invalid Arrow export callback arguments";
const ALLOCATION: &CStr = c"unable to allocate Arrow export headers";
const PANIC: &CStr = c"Arrow export callback failed";
/// ARROW_FLAG_NULLABLE: set on detail columns, whose masked rows are null.
const NULLABLE: i64 = 2;

/// A large_utf8 ("U"), large_binary ("Z") or large_list<int64> ("+L")
/// column: `rows + 1` int64 offsets into `n_values` values (bytes or int64)
/// and a packed validity bitmap, empty when no row is null.
struct Ragged {
    list: bool,
    binary: bool,
    offsets: usize,
    values: usize,
    n_values: i64,
    validity: Vec<u8>,
    null_count: i64,
}
enum Values {
    /// A fixed-width column: int32 ("i"), int64 ("l") or float64 ("g").
    Primitive {
        address: usize,
        format: &'static CStr,
    },
    Boolean(Vec<u8>),
    /// Small export-owned integer columns (shape and tile coordinates).
    Integers(Vec<i64>),
    /// A non-null struct or large-list node in a one-row complex Result.
    Struct(Vec<Column>),
    List {
        offsets: Vec<i64>,
        item: Box<Column>,
    },
    Ragged(Ragged),
    /// A non-nullable large_binary ("Z") column over a Rust-owned batch: no
    /// validity buffer, null_count 0.
    Binary {
        offsets: usize,
        values: usize,
    },
}
struct Column {
    name: CString,
    rows: i64,
    values: Values,
}
impl Column {
    fn format(&self) -> &'static CStr {
        match &self.values {
            Values::Primitive { format, .. } => format,
            Values::Boolean(_) => c"b",
            Values::Integers(_) => c"l",
            Values::Struct(_) => c"+s",
            Values::List { .. } => c"+L",
            Values::Ragged(Ragged { list: true, .. }) => c"+L",
            Values::Ragged(Ragged { binary: true, .. }) | Values::Binary { .. } => c"Z",
            Values::Ragged(Ragged { .. }) => c"U",
        }
    }
    fn ragged(&self) -> Option<&Ragged> {
        match &self.values {
            Values::Ragged(ragged) => Some(ragged),
            _ => None,
        }
    }
}

/// One retained node of the exported tree. Nested nodes reference the same
/// immutable BatchData as the flat export; every released child owns its Arc.
#[derive(Clone, Copy)]
enum Node<'a> {
    Batch,
    Column(&'a Column),
    Items(&'a Column),
}
impl<'a> Node<'a> {
    fn n_children(self, data: &BatchData) -> usize {
        match self {
            Node::Batch => data.columns.len(),
            Node::Column(column) => match &column.values {
                Values::Struct(columns) => columns.len(),
                Values::List { .. } => 1,
                _ => usize::from(column.ragged().is_some_and(|ragged| ragged.list)),
            },
            Node::Items(_) => 0,
        }
    }
    fn child(self, k: usize, data: &'a BatchData) -> Self {
        match self {
            Node::Batch => Node::Column(&data.columns[k]),
            Node::Column(column) => match &column.values {
                Values::Struct(columns) => Node::Column(&columns[k]),
                Values::List { item, .. } => Node::Column(item),
                _ => Node::Items(column),
            },
            Node::Items(column) => Node::Items(column),
        }
    }
}
struct BatchData {
    rows: i64,
    columns: Vec<Column>,
    owners: Vec<Py<PyAny>>,
    /// Rust owners of exported buffers; dropping them needs no GIL.
    _keep: Vec<Arc<dyn Any + Send + Sync>>,
}
impl Drop for BatchData {
    fn drop(&mut self) {
        // Only the final Arc drops Python owners. Native consumers may release
        // that Arc on a foreign thread; intermediate Arc drops need no GIL.
        let owners = std::mem::take(&mut self.owners);
        // A consumer may release its last reference after interpreter
        // finalization (e.g. a module-level pyarrow Table at exit); the owners
        // are then leaked, since Python can no longer be attached.
        // SAFETY: Py_IsInitialized may be called at any time.
        if unsafe { pyo3::ffi::Py_IsInitialized() } == 0 {
            std::mem::forget(owners);
            return;
        }
        if !owners.is_empty() {
            Python::attach(move |py| {
                for owner in owners {
                    owner.drop_ref(py);
                }
            });
        }
    }
}

#[pyclass(frozen, name = "_ArrowExport")]
pub(crate) struct ArrowExport {
    data: Arc<BatchData>,
}
#[pymethods]
impl ArrowExport {
    /// `(schema, array)` capsules of one column (the Arrow PyCapsule
    /// `__arrow_c_array__` protocol); unconsumed capsules release themselves.
    fn column<'py>(
        &self,
        py: Python<'py>,
        index: usize,
    ) -> PyResult<(Bound<'py, PyCapsule>, Bound<'py, PyCapsule>)> {
        if index >= self.data.columns.len() {
            return Err(parameter(py, "Arrow column index out of range"));
        }
        let failed = |_| PyMemoryError::new_err("unable to allocate Arrow export headers");
        let schema = schema(&self.data, Node::Column(&self.data.columns[index])).map_err(failed)?;
        let schema = PyCapsule::new_with_value(py, SchemaCapsule(schema), c"arrow_schema")?;
        let array = array(&self.data, Node::Column(&self.data.columns[index])).map_err(failed)?;
        let array = PyCapsule::new_with_value(py, ArrayCapsule(array), c"arrow_array")?;
        Ok((schema, array))
    }

    /// `(schema, array)` capsules of the whole batch as one struct array (the
    /// `__arrow_c_array__` protocol of a record-batch-like producer).
    fn batch<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<(Bound<'py, PyCapsule>, Bound<'py, PyCapsule>)> {
        let failed = |_| PyMemoryError::new_err("unable to allocate Arrow export headers");
        let schema = schema(&self.data, Node::Batch).map_err(failed)?;
        let schema = PyCapsule::new_with_value(py, SchemaCapsule(schema), c"arrow_schema")?;
        let array = array(&self.data, Node::Batch).map_err(failed)?;
        let array = PyCapsule::new_with_value(py, ArrayCapsule(array), c"arrow_array")?;
        Ok((schema, array))
    }

    fn stream<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyCapsule>> {
        let state = Box::new(StreamState {
            data: Arc::clone(&self.data),
            emitted: false,
            error: None,
        });
        PyCapsule::new_with_value(
            py,
            StreamCapsule(ArrowArrayStream {
                get_schema: Some(get_schema),
                get_next: Some(get_next),
                get_last_error: Some(get_last_error),
                release: Some(release_stream),
                private_data: Box::into_raw(state).cast(),
            }),
            c"arrow_array_stream",
        )
    }
}

// The boxed capsule value has precisely the public stream layout. The capsule
// owns only this original base; its private state may move to a consumer base.
#[repr(transparent)]
struct StreamCapsule(ArrowArrayStream);
// SAFETY: the immutable data is Send/Sync; consumers serialize stream callbacks
// and exclusively own a moved stream. Python-owner destruction attaches the GIL.
unsafe impl Send for StreamCapsule {}
impl Drop for StreamCapsule {
    fn drop(&mut self) {
        if let Some(release) = self.0.release {
            // SAFETY: a live original base is unconsumed and exclusively owned.
            unsafe { release(&mut self.0) };
        }
    }
}
// Column capsules follow the same rule: the capsule releases an unconsumed base.
#[repr(transparent)]
struct SchemaCapsule(ArrowSchema);
// SAFETY: immutable exported data; a consumer moves the base out exclusively.
unsafe impl Send for SchemaCapsule {}
impl Drop for SchemaCapsule {
    fn drop(&mut self) {
        if let Some(release) = self.0.release {
            // SAFETY: a live base still owned by the capsule.
            unsafe { release(&mut self.0) };
        }
    }
}
#[repr(transparent)]
struct ArrayCapsule(ArrowArray);
// SAFETY: as for SchemaCapsule.
unsafe impl Send for ArrayCapsule {}
impl Drop for ArrayCapsule {
    fn drop(&mut self) {
        if let Some(release) = self.0.release {
            // SAFETY: a live base still owned by the capsule.
            unsafe { release(&mut self.0) };
        }
    }
}

struct StreamState {
    data: Arc<BatchData>,
    emitted: bool,
    error: Option<&'static CStr>,
}

// Header addresses never identify their allocations. Each private allocation
// owns its children, pointer slots and independent data reference, so consumers
// may shallow-move a child and null its old release before releasing its parent.
// Boxes keep each child base at a stable address independently of its siblings.
#[allow(clippy::vec_box)]
struct SchemaPrivate {
    _data: Arc<BatchData>,
    children: Vec<Box<ArrowSchema>>,
    pointers: Vec<*mut ArrowSchema>,
}
impl Drop for SchemaPrivate {
    fn drop(&mut self) {
        for child in &mut self.children {
            if let Some(release) = child.release {
                // SAFETY: the live child remains owned by this parent; a moved
                // child's source has release=None and is deliberately skipped.
                unsafe { release(&mut **child) };
            }
        }
    }
}
#[allow(clippy::vec_box)]
struct ArrayPrivate {
    _data: Arc<BatchData>,
    children: Vec<Box<ArrowArray>>,
    pointers: Vec<*mut ArrowArray>,
    buffers: Vec<*const c_void>,
}
impl Drop for ArrayPrivate {
    fn drop(&mut self) {
        for child in &mut self.children {
            if let Some(release) = child.release {
                // SAFETY: identical move/ownership rule to SchemaPrivate.
                unsafe { release(&mut **child) };
            }
        }
    }
}

fn reserve<T>(values: &mut Vec<T>, size: usize) -> Result<(), &'static CStr> {
    values.try_reserve_exact(size).map_err(|_| ALLOCATION)
}
fn schema(data: &Arc<BatchData>, node: Node<'_>) -> Result<ArrowSchema, &'static CStr> {
    let mut private = Box::new(SchemaPrivate {
        _data: Arc::clone(data),
        children: Vec::new(),
        pointers: Vec::new(),
    });
    let n_children = node.n_children(data);
    reserve(&mut private.children, n_children)?;
    reserve(&mut private.pointers, n_children)?;
    for k in 0..n_children {
        let mut child = Box::new(schema(data, node.child(k, data))?);
        private.pointers.push(&mut *child);
        private.children.push(child);
    }
    let (format, name, flags) = match node {
        Node::Batch => (c"+s", c"", 0),
        Node::Column(column) => {
            let flags = if column.ragged().is_some() {
                NULLABLE
            } else {
                0
            };
            (column.format(), column.name.as_c_str(), flags)
        }
        Node::Items(_) => (c"l", c"item", 0),
    };
    let base = ArrowSchema {
        format: format.as_ptr(),
        name: name.as_ptr(),
        metadata: ptr::null(),
        flags,
        n_children: private.children.len() as i64,
        children: if private.children.is_empty() {
            ptr::null_mut()
        } else {
            private.pointers.as_mut_ptr()
        },
        dictionary: ptr::null_mut(),
        release: Some(release_schema),
        private_data: Box::into_raw(private).cast(),
    };
    Ok(base)
}
fn array(data: &Arc<BatchData>, node: Node<'_>) -> Result<ArrowArray, &'static CStr> {
    let mut private = Box::new(ArrayPrivate {
        _data: Arc::clone(data),
        children: Vec::new(),
        pointers: Vec::new(),
        buffers: Vec::new(),
    });
    let null = ptr::null::<c_void>();
    let (length, null_count) = match node {
        Node::Batch => {
            reserve(&mut private.buffers, 1)?;
            private.buffers.push(null);
            (data.rows, 0)
        }
        Node::Column(column) => match &column.values {
            Values::Primitive { address, .. } => {
                reserve(&mut private.buffers, 2)?;
                private.buffers.extend([null, *address as *const c_void]);
                (column.rows, 0)
            }
            Values::Boolean(bytes) => {
                reserve(&mut private.buffers, 2)?;
                private.buffers.extend([null, bytes.as_ptr().cast()]);
                (column.rows, 0)
            }
            Values::Integers(values) => {
                reserve(&mut private.buffers, 2)?;
                private.buffers.extend([null, values.as_ptr().cast()]);
                (column.rows, 0)
            }
            Values::Struct(_) => {
                reserve(&mut private.buffers, 1)?;
                private.buffers.push(null);
                (column.rows, 0)
            }
            Values::List { offsets, .. } => {
                reserve(&mut private.buffers, 2)?;
                private.buffers.extend([null, offsets.as_ptr().cast()]);
                (column.rows, 0)
            }
            Values::Binary { offsets, values } => {
                reserve(&mut private.buffers, 3)?;
                private
                    .buffers
                    .extend([null, *offsets as *const c_void, *values as *const c_void]);
                (column.rows, 0)
            }
            Values::Ragged(ragged) => {
                let validity = if ragged.null_count == 0 {
                    null
                } else {
                    ragged.validity.as_ptr().cast()
                };
                reserve(&mut private.buffers, 3)?;
                private
                    .buffers
                    .extend([validity, ragged.offsets as *const c_void]);
                if !ragged.list {
                    private.buffers.push(ragged.values as *const c_void);
                }
                (column.rows, ragged.null_count)
            }
        },
        Node::Items(column) => {
            let ragged = column.ragged().ok_or(INVALID)?;
            reserve(&mut private.buffers, 2)?;
            private
                .buffers
                .extend([null, ragged.values as *const c_void]);
            (ragged.n_values, 0)
        }
    };
    let n_children = node.n_children(data);
    reserve(&mut private.children, n_children)?;
    reserve(&mut private.pointers, n_children)?;
    for k in 0..n_children {
        let mut child = Box::new(array(data, node.child(k, data))?);
        private.pointers.push(&mut *child);
        private.children.push(child);
    }
    Ok(ArrowArray {
        length,
        null_count,
        offset: 0,
        n_buffers: private.buffers.len() as i64,
        n_children: private.children.len() as i64,
        buffers: private.buffers.as_ptr(),
        children: if private.children.is_empty() {
            ptr::null_mut()
        } else {
            private.pointers.as_mut_ptr()
        },
        dictionary: ptr::null_mut(),
        release: Some(release_array),
        private_data: Box::into_raw(private).cast(),
    })
}

unsafe extern "C" fn release_schema(base: *mut ArrowSchema) {
    let _ = catch_unwind(AssertUnwindSafe(|| {
        // SAFETY: the consumer supplies its exclusively owned live/moved base.
        let Some(base) = (unsafe { base.as_mut() }) else {
            return;
        };
        if base.release.take().is_none() {
            return;
        }
        let private = std::mem::replace(&mut base.private_data, ptr::null_mut());
        if !private.is_null() {
            // SAFETY: release is cleared before destructors can reenter. Only
            // this private allocation is reclaimed, never the caller's header.
            drop(unsafe { Box::from_raw(private.cast::<SchemaPrivate>()) });
        }
    }));
}
unsafe extern "C" fn release_array(base: *mut ArrowArray) {
    let _ = catch_unwind(AssertUnwindSafe(|| {
        // SAFETY: the consumer supplies its exclusively owned live/moved base.
        let Some(base) = (unsafe { base.as_mut() }) else {
            return;
        };
        if base.release.take().is_none() {
            return;
        }
        let private = std::mem::replace(&mut base.private_data, ptr::null_mut());
        if !private.is_null() {
            // SAFETY: this is the one private owner, detached before callbacks.
            drop(unsafe { Box::from_raw(private.cast::<ArrayPrivate>()) });
        }
    }));
}
unsafe extern "C" fn release_stream(base: *mut ArrowArrayStream) {
    let _ = catch_unwind(AssertUnwindSafe(|| {
        // SAFETY: the consumer/capsule exclusively owns this live stream base.
        let Some(base) = (unsafe { base.as_mut() }) else {
            return;
        };
        if base.release.take().is_none() {
            return;
        }
        let private = std::mem::replace(&mut base.private_data, ptr::null_mut());
        if !private.is_null() {
            // SAFETY: the private state survives shallow moves and is detached
            // before dropping any Python references can invoke user code.
            drop(unsafe { Box::from_raw(private.cast::<StreamState>()) });
        }
    }));
}

unsafe fn state<'a>(base: *mut ArrowArrayStream) -> Option<&'a mut StreamState> {
    // SAFETY: callbacks are serialized and the C caller owns a valid base.
    let base = unsafe { base.as_mut() }?;
    base.release?;
    // SAFETY: a live producer stream retains this separately boxed state.
    unsafe { base.private_data.cast::<StreamState>().as_mut() }
}
fn callback_status<T>(
    state: &mut StreamState,
    result: std::thread::Result<Result<T, &'static CStr>>,
) -> Result<T, c_int> {
    match result {
        Ok(Ok(value)) => Ok(value),
        Ok(Err(message)) => {
            state.error = Some(message);
            Err(12)
        }
        Err(_) => {
            state.error = Some(PANIC);
            Err(5)
        }
    }
}
unsafe extern "C" fn get_schema(base: *mut ArrowArrayStream, out: *mut ArrowSchema) -> c_int {
    // SAFETY: valid live bases are the C Stream caller's responsibility.
    let Some(state) = (unsafe { state(base) }) else {
        return 22;
    };
    state.error = None;
    if out.is_null() {
        state.error = Some(INVALID);
        return 22;
    }
    // SAFETY: the caller supplies writable storage; zero is a non-live header.
    unsafe { out.write(std::mem::zeroed()) };
    let result = catch_unwind(AssertUnwindSafe(|| schema(&state.data, Node::Batch)));
    match callback_status(state, result) {
        Ok(value) => {
            // SAFETY: publish only a complete independently owned schema tree.
            unsafe { out.write(value) };
            0
        }
        Err(code) => code,
    }
}
unsafe extern "C" fn get_next(base: *mut ArrowArrayStream, out: *mut ArrowArray) -> c_int {
    // SAFETY: valid live bases are the C Stream caller's responsibility.
    let Some(state) = (unsafe { state(base) }) else {
        return 22;
    };
    state.error = None;
    if out.is_null() {
        state.error = Some(INVALID);
        return 22;
    }
    // SAFETY: zero release also represents EOS, including after a live N=0 batch.
    unsafe { out.write(std::mem::zeroed()) };
    if state.emitted {
        return 0;
    }
    let result = catch_unwind(AssertUnwindSafe(|| array(&state.data, Node::Batch)));
    match callback_status(state, result) {
        Ok(value) => {
            // SAFETY: publish the complete batch before advancing this cursor.
            unsafe { out.write(value) };
            state.emitted = true;
            0
        }
        Err(code) => code,
    }
}
unsafe extern "C" fn get_last_error(base: *mut ArrowArrayStream) -> *const c_char {
    // SAFETY: callback serialization also covers last-error reads.
    unsafe { state(base) }
        .and_then(|state| state.error)
        .map_or(ptr::null(), CStr::as_ptr)
}

/// A readonly C-contiguous 0-D/1-D native NumPy buffer: its native view (the
/// retained owner), element count, dtype kind, item size and data address.
struct Buffer<'py> {
    view: Bound<'py, PyAny>,
    size: usize,
    kind: String,
    itemsize: usize,
    address: usize,
}
fn buffer<'py>(py: Python<'py>, object: &Bound<'py, PyAny>) -> PyResult<Buffer<'py>> {
    let numpy = py.import("numpy")?;
    let ndarray = numpy.getattr("ndarray")?;
    if !object.is_instance(&ndarray)?
        || numpy
            .getattr("ma")?
            .getattr("isMaskedArray")?
            .call1((object,))?
            .is_truthy()?
    {
        return Err(parameter(py, "Arrow columns must be unmasked NumPy arrays"));
    }
    // Inspect native ndarray metadata rather than subclass overrides.
    let kwargs = PyDict::new(py);
    kwargs.set_item("type", &ndarray)?;
    let view = ndarray.getattr("view")?.call((object,), Some(&kwargs))?;
    let ndim: usize = view.getattr("ndim")?.extract()?;
    let flags = view.getattr("flags")?;
    if ndim > 1
        || !flags.getattr("c_contiguous")?.is_truthy()?
        || flags.getattr("writeable")?.is_truthy()?
    {
        return Err(parameter(
            py,
            "Arrow columns require readonly C-contiguous 0-D/1-D NumPy arrays",
        ));
    }
    let size: usize = view.getattr("size")?.extract()?;
    i64::try_from(size).map_err(|_| parameter(py, "Arrow row count exceeds int64"))?;
    let dtype = view.getattr("dtype")?;
    let kind: String = dtype.getattr("kind")?.extract()?;
    let itemsize: usize = dtype.getattr("itemsize")?.extract()?;
    if !dtype.getattr("isnative")?.is_truthy()? {
        return Err(parameter(py, "Arrow columns require native byte order"));
    }
    let (address, readonly): (usize, bool) = view
        .getattr("__array_interface__")?
        .get_item("data")?
        .extract()?;
    if !readonly || (size != 0 && address == 0) {
        return Err(parameter(
            py,
            "Arrow columns require a valid readonly data buffer",
        ));
    }
    size.checked_mul(itemsize)
        .and_then(|bytes| address.checked_add(bytes))
        .ok_or_else(|| parameter(py, "Arrow column byte range overflows"))?;
    Ok(Buffer {
        view,
        size,
        kind,
        itemsize,
        address,
    })
}

/// Arrow's LSB-first bit layout of a native bool buffer, and its set-bit count. A
/// refused reservation raises the §3 DeviceError with `label` and the bytes
/// (host-allocation-failures.md §13.1, X2 and X3).
fn pack_bits(py: Python<'_>, buffer: &Buffer<'_>, label: &str) -> PyResult<(Vec<u8>, usize)> {
    let length = buffer.size.div_ceil(8);
    let mut packed = Vec::new();
    try_reserve_exact(&mut packed, length, label).map_err(|e| error(py, e))?;
    packed.resize(length, 0);
    let mut set = 0;
    for index in 0..buffer.size {
        // SAFETY: the native readonly C-contiguous bool ndarray retains size
        // byte-addressed values; zero rows perform no byte access.
        if unsafe { (buffer.address as *const u8).add(index).read() } != 0 {
            packed[index / 8] |= 1 << (index % 8);
            set += 1;
        }
    }
    Ok((packed, set))
}

/// A ragged detail column `(kind, offsets, values, validity)` with kind
/// "string" or "binary" (uint8 values) or "list" (int64 values). The offsets are checked
/// (non-negative, non-decreasing, within the values) so no consumer can read
/// outside the retained buffers.
fn ragged<'py>(
    py: Python<'py>,
    parts: &Bound<'py, PyTuple>,
    owners: &mut Vec<Py<PyAny>>,
) -> PyResult<(Values, usize)> {
    let invalid = || {
        parameter(
            py,
            "Arrow detail columns are (\"string\" | \"binary\" | \"list\", int64 offsets, values, bool validity)",
        )
    };
    if parts.len() != 4 {
        return Err(invalid());
    }
    let kind: String = parts.get_item(0)?.extract().map_err(|_| invalid())?;
    let (list, binary) = match kind.as_str() {
        "list" => (true, false),
        "string" => (false, false),
        "binary" => (false, true),
        _ => return Err(invalid()),
    };
    let offsets = buffer(py, &parts.get_item(1)?)?;
    let values = buffer(py, &parts.get_item(2)?)?;
    let validity = buffer(py, &parts.get_item(3)?)?;
    let value_type = if list { ("i", 8) } else { ("u", 1) };
    if (offsets.kind.as_str(), offsets.itemsize) != ("i", 8)
        || (values.kind.as_str(), values.itemsize) != value_type
        || (validity.kind.as_str(), validity.itemsize) != ("b", 1)
        || offsets.size != validity.size + 1
    {
        return Err(invalid());
    }
    let mut previous = 0i64;
    for index in 0..offsets.size {
        // SAFETY: the readonly C-contiguous int64 ndarray retains `size` values;
        // NumPy aligns int64 data, and an unaligned read is used regardless.
        let offset = unsafe { (offsets.address as *const i64).add(index).read_unaligned() };
        if offset < previous {
            return Err(parameter(
                py,
                "Arrow detail column offsets must be non-negative and non-decreasing",
            ));
        }
        previous = offset;
    }
    let n_values = i64::try_from(values.size).map_err(|_| invalid())?;
    if previous > n_values {
        return Err(parameter(
            py,
            "Arrow detail column offsets exceed its values buffer",
        ));
    }
    let rows = validity.size;
    let (packed, set) = pack_bits(py, &validity, "Arrow validity bitmap")?;
    let null_count = (rows - set) as i64;
    let column = Values::Ragged(Ragged {
        list,
        binary,
        offsets: offsets.address,
        values: values.address,
        n_values,
        validity: if null_count == 0 { Vec::new() } else { packed },
        null_count,
    });
    owners.push(offsets.view.unbind());
    owners.push(values.view.unbind());
    Ok((column, rows))
}

/// Parse a retained leaf using the same buffer checks for flat and nested exports.
fn parsed_column<'py>(
    py: Python<'py>,
    name: CString,
    object: &Bound<'py, PyAny>,
    owners: &mut Vec<Py<PyAny>>,
) -> PyResult<Column> {
    let (values, size) = if let Ok(parts) = object.cast::<PyTuple>() {
        ragged(py, parts, owners)?
    } else {
        let column = buffer(py, object)?;
        let format = match (column.kind.as_str(), column.itemsize) {
            ("i", 4) => c"i",
            ("i", 8) => c"l",
            ("f", 8) => c"g",
            ("b", 1) => c"b",
            _ => {
                return Err(parameter(
                    py,
                    "Arrow columns require native NumPy int32, int64, float64 or bool dtype",
                ))
            }
        };
        let values = if column.kind == "b" {
            Values::Boolean(pack_bits(py, &column, "Arrow boolean column")?.0)
        } else {
            Values::Primitive {
                address: column.address,
                format,
            }
        };
        let size = column.size;
        owners.push(column.view.unbind());
        (values, size)
    };
    let rows = i64::try_from(size).map_err(|_| parameter(py, "Arrow row count exceeds int64"))?;
    Ok(Column { name, rows, values })
}

fn column_name(py: Python<'_>, key: &Bound<'_, PyAny>) -> PyResult<CString> {
    let name: String = key.extract()?;
    CString::new(name).map_err(|_| parameter(py, "Arrow column names must not contain NUL"))
}

#[pyfunction]
fn _native_arrow_export<'py>(
    py: Python<'py>,
    columns: &Bound<'py, PyDict>,
) -> PyResult<Py<ArrowExport>> {
    if columns.is_empty() {
        return Err(parameter(py, "Arrow export requires at least one column"));
    }
    i64::try_from(columns.len()).map_err(|_| parameter(py, "too many Arrow columns"))?;
    let mut fields = Vec::new();
    let mut owners = Vec::new();
    // Refused reservations raise the §3 DeviceError with the bytes
    // (host-allocation-failures.md §13.1, X4 and X5); the owner-count overflow is
    // not a refusal and keeps its MemoryError.
    try_reserve_exact(&mut fields, columns.len(), "Arrow columns").map_err(|e| error(py, e))?;
    let n_owners = columns
        .len()
        .checked_mul(2)
        .ok_or_else(|| PyMemoryError::new_err("too many Arrow owners"))?;
    try_reserve_exact(&mut owners, n_owners, "Arrow owners").map_err(|e| error(py, e))?;
    let mut rows = None;
    for (key, object) in columns.iter() {
        let column = parsed_column(py, column_name(py, &key)?, &object, &mut owners)?;
        if rows.is_some_and(|previous| previous != column.rows) {
            return Err(parameter(py, "Arrow columns must have equal lengths"));
        }
        rows = Some(column.rows);
        fields.push(column);
    }
    Py::new(
        py,
        ArrowExport {
            data: Arc::new(BatchData {
                rows: rows.unwrap_or(0),
                columns: fields,
                owners,
                _keep: Vec::new(),
            }),
        },
    )
}

/// A one-row large-list holding a complete existing column, including zero rows.
fn whole_column(mut column: Column) -> Column {
    let name = std::mem::replace(&mut column.name, c"item".to_owned());
    Column {
        name,
        rows: 1,
        values: Values::List {
            offsets: vec![0, column.rows],
            item: Box::new(column),
        },
    }
}

fn coordinates(name: &CStr, values: (i64, i64)) -> Column {
    whole_column(Column {
        name: name.to_owned(),
        rows: 2,
        values: Values::Integers(vec![values.0, values.1]),
    })
}

/// Matrix rows are structural list offsets only: numeric leaves keep their
/// original buffers. Empty (0,n) and (n,0) matrices keep both dimensions.
fn matrix_column(py: Python<'_>, column: Column, shape: (i64, i64)) -> PyResult<Column> {
    let count = usize::try_from(shape.0)
        .ok()
        .and_then(|n| n.checked_add(1))
        .ok_or_else(|| parameter(py, "Arrow matrix row offsets exceed platform limits"))?;
    let mut offsets = Vec::new();
    // §13.1 X1: the §3 DeviceError with the bytes.
    try_reserve_exact(&mut offsets, count, "Arrow matrix row offsets").map_err(|e| error(py, e))?;
    for row in 0..=shape.0 {
        offsets.push(
            row.checked_mul(shape.1)
                .ok_or_else(|| parameter(py, "Arrow matrix offset exceeds int64"))?,
        );
    }
    let name = column.name.clone();
    let mut item = column;
    item.name = c"item".to_owned();
    Ok(whole_column(Column {
        name,
        rows: shape.0,
        values: Values::List {
            offsets,
            item: Box::new(item),
        },
    }))
}

fn topk_geometry<'py>(
    py: Python<'py>,
    columns: &Bound<'py, PyDict>,
    shape: (i64, i64),
) -> PyResult<usize> {
    let get = |name: &str, kind: &str, width: usize| -> PyResult<Buffer<'py>> {
        let object = columns
            .get_item(name)?
            .ok_or_else(|| parameter(py, format!("TopK export requires {name}")))?;
        let source = buffer(py, &object)?;
        if source.kind != kind || source.itemsize != width {
            return Err(parameter(
                py,
                format!("TopK export has invalid {name} dtype"),
            ));
        }
        Ok(source)
    };
    let i = get("i", "i", 8)?;
    let j = get("j", "i", 8)?;
    let rank = get("rank", "i", 4)?;
    let offsets = get("offsets", "i", 8)?;
    let flags = get("tie_at_k", "b", 1)?;
    let rows = usize::try_from(shape.0)
        .map_err(|_| parameter(py, "TopK row count exceeds platform limits"))?;
    let offset_count = rows
        .checked_add(1)
        .ok_or_else(|| parameter(py, "TopK row offsets exceed platform limits"))?;
    if i.size != j.size || i.size != rank.size || offsets.size != offset_count || flags.size != rows
    {
        return Err(parameter(
            py,
            "TopK entry columns, offsets and row flags have inconsistent lengths",
        ));
    }
    let mut previous = 0i64;
    for index in 0..offsets.size {
        // SAFETY: buffer() checked an immutable contiguous int64 view and
        // retained its complete allocation; this is the same read as ragged().
        let offset = unsafe { (offsets.address as *const i64).add(index).read_unaligned() };
        if (index == 0 && offset != 0) || offset < previous || offset as u128 > i.size as u128 {
            return Err(parameter(
                py,
                "TopK offsets must span the complete entry columns",
            ));
        }
        previous = offset;
    }
    if previous as u128 != i.size as u128 {
        return Err(parameter(py, "TopK offsets must end at the entry count"));
    }
    Ok(i.size)
}

/// Actual nested C Stream mapping: exactly one row per TopK/Matrix/Tile.
/// The Python facade passes only flattened readonly numeric matrix views;
/// its shape and global tile coordinates are ordinary exported list columns.
#[pyfunction(signature=(layout, columns, shape, rows=None, cols=None))]
fn _native_arrow_export_complex<'py>(
    py: Python<'py>,
    layout: &str,
    columns: &Bound<'py, PyDict>,
    shape: (i64, i64),
    rows: Option<(i64, i64)>,
    cols: Option<(i64, i64)>,
) -> PyResult<Py<ArrowExport>> {
    if !matches!(layout, "topk" | "matrix" | "tile")
        || shape.0 < 0
        || shape.1 < 0
        || columns.is_empty()
    {
        return Err(parameter(
            py,
            "complex Arrow export requires a TopK, Matrix or Tile with nonnegative shape",
        ));
    }
    let (entries, cells) = if layout == "topk" {
        (Some(topk_geometry(py, columns, shape)?), None)
    } else {
        (
            None,
            Some(
                shape
                    .0
                    .checked_mul(shape.1)
                    .ok_or_else(|| parameter(py, "Arrow matrix shape exceeds int64"))?,
            ),
        )
    };
    let topk_offsets = if entries.is_some() {
        Some(
            shape
                .0
                .checked_add(1)
                .ok_or_else(|| parameter(py, "TopK row offsets exceed int64"))?,
        )
    } else {
        None
    };
    if layout == "tile" {
        let valid = |range: Option<(i64, i64)>, n: i64| {
            range.is_some_and(|(start, stop)| start >= 0 && stop >= start && stop - start == n)
        };
        if !valid(rows, shape.0) || !valid(cols, shape.1) {
            return Err(parameter(
                py,
                "Tile export requires matching half-open global rows and cols",
            ));
        }
    } else if rows.is_some() || cols.is_some() {
        return Err(parameter(
            py,
            "global rows and cols apply only to a Tile export",
        ));
    }
    let mut fields = Vec::new();
    let mut owners = Vec::new();
    // §13.1 X6 and X7, as X4 and X5 of the flat export.
    try_reserve_exact(&mut fields, columns.len(), "Arrow complex columns")
        .map_err(|e| error(py, e))?;
    let n_owners = columns
        .len()
        .checked_mul(2)
        .ok_or_else(|| PyMemoryError::new_err("too many Arrow owners"))?;
    try_reserve_exact(&mut owners, n_owners, "Arrow complex owners").map_err(|e| error(py, e))?;
    for (key, object) in columns.iter() {
        let column = parsed_column(py, column_name(py, &key)?, &object, &mut owners)?;
        let field = if let Some(entries) = entries {
            let expected = match column.name.to_bytes() {
                b"offsets" => topk_offsets.unwrap_or(0),
                b"tie_at_k" => shape.0,
                _ => entries as i64,
            };
            if column.rows != expected {
                return Err(parameter(
                    py,
                    "TopK entry and auxiliary columns have inconsistent lengths",
                ));
            }
            whole_column(column)
        } else {
            if Some(column.rows) != cells || column.ragged().is_some() {
                return Err(parameter(
                    py,
                    "Matrix and Tile exports require numeric fields matching shape",
                ));
            }
            matrix_column(py, column, shape)?
        };
        fields.push(field);
    }
    let mut columns = vec![coordinates(c"shape", shape)];
    if let (Some(rows), Some(cols)) = (rows, cols) {
        columns.push(coordinates(c"rows", rows));
        columns.push(coordinates(c"cols", cols));
    }
    columns.push(Column {
        name: c"data".to_owned(),
        rows: 1,
        values: Values::Struct(fields),
    });
    Py::new(
        py,
        ArrowExport {
            data: Arc::new(BatchData {
                rows: 1,
                columns,
                owners,
                _keep: Vec::new(),
            }),
        },
    )
}

/// Export reader columns (name, batch) as non-nullable large_binary columns,
/// zero-copy: the export holds the batches' `Arc`s (spec §6.3).
pub(crate) fn binary_export(
    py: Python<'_>,
    columns: Vec<(&CStr, Arc<SeqBatch>)>,
) -> PyResult<Py<ArrowExport>> {
    let rows = columns.first().map_or(0, |(_, batch)| batch.len());
    let rows = i64::try_from(rows).map_err(|_| parameter(py, "Arrow row count exceeds int64"))?;
    let mut fields = Vec::with_capacity(columns.len());
    let mut keep: Vec<Arc<dyn Any + Send + Sync>> = Vec::with_capacity(columns.len());
    for (name, batch) in columns {
        debug_assert_eq!(batch.len() as i64, rows);
        fields.push(Column {
            name: name.to_owned(),
            rows,
            values: Values::Binary {
                offsets: batch.offsets().as_ptr() as usize,
                values: batch.data().as_ptr() as usize,
            },
        });
        keep.push(batch);
    }
    Py::new(
        py,
        ArrowExport {
            data: Arc::new(BatchData {
                rows,
                columns: fields,
                owners: Vec::new(),
                _keep: keep,
            }),
        },
    )
}

pub fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<ArrowExport>()?;
    module.add_function(wrap_pyfunction!(_native_arrow_export, module)?)?;
    module.add_function(wrap_pyfunction!(_native_arrow_export_complex, module)?)?;
    Ok(())
}
