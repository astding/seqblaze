//! Detail (non-selecting) result fields: typed columns filled only for kept pairs.
//! Spec: docs/wiki/engine/host-sinks.md ("Detail fields") and engine-layout.md §4.
//!
//! The selection score (`dist`) alone drives cutoffs, Hits, TopK ranks and
//! RowStats. A kernel may also declare detail fields; the engine computes them
//! only for pairs a sink keeps, in output order, and they never affect selection.
//! Unary kernels (`crate::unary`) emit every field, the primary included, through
//! the same columns.
use crate::validate::EngineError;

/// The one NaN encoding of every float64 field and mask (tm.md §5): the quiet NaN
/// 0x7FF8000000000000, so the bits do not depend on the platform's default NaN.
pub const CANONICAL_NAN: f64 = f64::from_bits(0x7FF8_0000_0000_0000);

/// `value`, with any NaN replaced by [`CANONICAL_NAN`].
#[inline]
pub fn canonical(value: f64) -> f64 {
    if value.is_nan() {
        CANONICAL_NAN
    } else {
        value
    }
}

/// The DeviceError for a buffer that cannot be allocated: the call stops and
/// publishes nothing (engine-layout §6, as for the other allocation sites).
/// The message itself is built without an infallible allocation (see
/// [`device_error`]), so a worker thread under memory pressure reports the
/// failure instead of aborting the process.
pub fn allocation_error(what: &str, reason: impl std::fmt::Display) -> EngineError {
    device_error(format_args!(
        "unable to reserve {what}: {reason}; no result was published"
    ))
}

/// A DeviceError whose message is formatted into a fallibly reserved String.
///
/// Allocation failures are reported from worker threads, where the allocator
/// may refuse even a small block right after a refused large one (a fresh
/// glibc thread arena cannot be created under an address-space limit, and the
/// per-allocation mmap fallback fails too). An infallible `format!` there
/// aborts the process. If the message cannot be reserved, the error carries an
/// empty message and displays [`crate::validate::OUT_OF_MEMORY`] instead
/// (formatted later, on the calling thread).
pub fn device_error(args: std::fmt::Arguments<'_>) -> EngineError {
    use std::fmt::Write;
    /// Counts the bytes a formatting would write, without allocating.
    struct Count(usize);
    impl Write for Count {
        fn write_str(&mut self, s: &str) -> std::fmt::Result {
            self.0 = self.0.saturating_add(s.len());
            Ok(())
        }
    }
    let mut count = Count(0);
    let mut message = String::new();
    // Display impls of the reasons (TryReserveError, &str, integers) do not
    // allocate; the second pass writes into the exact reservation.
    if count.write_fmt(args).is_ok()
        && message.try_reserve_exact(count.0).is_ok()
        && message.write_fmt(args).is_ok()
        && message.len() == count.0
    {
        EngineError::Device(message)
    } else {
        EngineError::Device(String::new())
    }
}

/// The DeviceError of a refused reservation of `elements` values of `T`
/// (host-allocation-failures.md §3): "unable to reserve {what} ({bytes} bytes):
/// {reason}; no result was published", with bytes = elements × size_of::<T>()
/// computed in u128, the minimum capacity the call needs. The message is built
/// as in [`device_error`], so it never aborts.
pub fn reservation_error<T>(
    what: &str,
    elements: u128,
    reason: impl std::fmt::Display,
) -> EngineError {
    let bytes = elements.saturating_mul(std::mem::size_of::<T>() as u128);
    device_error(format_args!(
        "unable to reserve {what} ({bytes} bytes): {reason}; no result was published"
    ))
}

/// Room for `additional` more elements, amortized (growing output columns).
pub fn try_reserve<T>(
    buffer: &mut Vec<T>,
    additional: usize,
    what: &str,
) -> Result<(), EngineError> {
    let len = buffer.len();
    buffer
        .try_reserve(additional)
        .map_err(|error| reservation_error::<T>(what, len as u128 + additional as u128, error))
}

/// Room for exactly `additional` more elements (fixed-size scratch).
pub fn try_reserve_exact<T>(
    buffer: &mut Vec<T>,
    additional: usize,
    what: &str,
) -> Result<(), EngineError> {
    let len = buffer.len();
    buffer
        .try_reserve_exact(additional)
        .map_err(|error| reservation_error::<T>(what, len as u128 + additional as u128, error))
}

/// Value type of one detail field.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FieldKind {
    /// int32; masked rows hold INT32_MAX.
    Int32,
    /// int64; masked rows hold INT64_MAX.
    Int64,
    /// float64; masked rows hold [`CANONICAL_NAN`].
    Float64,
    /// Arrow list<int64>; masked rows are null.
    ListInt64,
    /// Arrow (large) string; masked rows are null, distinct from "".
    Utf8,
    /// Arrow (large) binary: raw bytes in the Utf8 layout (offsets, bytes,
    /// validity); masked rows are null.
    Binary,
}

impl FieldKind {
    /// Fixed-width numeric fields are the only ones a dense sink can hold.
    pub fn is_numeric(self) -> bool {
        matches!(self, Self::Int32 | Self::Int64 | Self::Float64)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DetailField {
    pub name: &'static str,
    pub kind: FieldKind,
}

/// Requested detail fields: a bit set over a kernel's `detail_fields()` indices.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct DetailRequest(u32);

impl DetailRequest {
    pub const NONE: Self = Self(0);
    pub fn with(self, field: usize) -> Self {
        assert!(field < 32, "at most 32 detail fields");
        Self(self.0 | 1 << field)
    }
    pub fn contains(self, field: usize) -> bool {
        field < 32 && self.0 >> field & 1 == 1
    }
    pub fn is_empty(self) -> bool {
        self.0 == 0
    }
}

/// Largest number of numeric detail fields a dense sink carries per cell
/// (align native local: four coordinates, four alignment counts and n_ends).
pub const MAX_DENSE_DETAILS: usize = 9;
/// Numeric detail values of one pair, in requested-column order; a float64
/// value is carried as its bit pattern.
pub type DenseRow = [i64; MAX_DENSE_DETAILS];

/// Values of one detail column. Validity bytes are 1 for present, 0 for null.
/// Equality compares float64 values by bit pattern, so NaN masks compare equal.
#[derive(Debug, Clone)]
pub enum DetailData {
    Int32(Vec<i32>),
    Int64(Vec<i64>),
    Float64(Vec<f64>),
    ListInt64 {
        offsets: Vec<i64>,
        values: Vec<i64>,
        valid: Vec<u8>,
    },
    Utf8 {
        offsets: Vec<i64>,
        data: Vec<u8>,
        valid: Vec<u8>,
    },
}

impl PartialEq for DetailData {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Int32(x), Self::Int32(y)) => x == y,
            (Self::Int64(x), Self::Int64(y)) => x == y,
            (Self::Float64(x), Self::Float64(y)) => {
                x.len() == y.len() && x.iter().zip(y).all(|(a, b)| a.to_bits() == b.to_bits())
            }
            (
                Self::ListInt64 {
                    offsets,
                    values,
                    valid,
                },
                Self::ListInt64 {
                    offsets: o,
                    values: v,
                    valid: m,
                },
            ) => offsets == o && values == v && valid == m,
            (
                Self::Utf8 {
                    offsets,
                    data,
                    valid,
                },
                Self::Utf8 {
                    offsets: o,
                    data: d,
                    valid: m,
                },
            ) => offsets == o && data == d && valid == m,
            _ => false,
        }
    }
}
impl Eq for DetailData {}

impl DetailData {
    fn new(kind: FieldKind) -> Self {
        match kind {
            FieldKind::Int32 => Self::Int32(Vec::new()),
            FieldKind::Int64 => Self::Int64(Vec::new()),
            FieldKind::Float64 => Self::Float64(Vec::new()),
            FieldKind::ListInt64 => Self::ListInt64 {
                offsets: vec![0],
                values: Vec::new(),
                valid: Vec::new(),
            },
            FieldKind::Utf8 | FieldKind::Binary => Self::Utf8 {
                offsets: vec![0],
                data: Vec::new(),
                valid: Vec::new(),
            },
        }
    }
    /// [`Self::new`] with the ragged kinds' first offset reserved fallibly.
    fn try_new(kind: FieldKind) -> Result<Self, EngineError> {
        let offsets = || -> Result<Vec<i64>, EngineError> {
            let mut offsets = Vec::new();
            try_reserve_exact(&mut offsets, 1, "detail column offsets")?;
            offsets.push(0);
            Ok(offsets)
        };
        Ok(match kind {
            FieldKind::ListInt64 => Self::ListInt64 {
                offsets: offsets()?,
                values: Vec::new(),
                valid: Vec::new(),
            },
            FieldKind::Utf8 | FieldKind::Binary => Self::Utf8 {
                offsets: offsets()?,
                data: Vec::new(),
                valid: Vec::new(),
            },
            // Fixed-width kinds allocate nothing until rows are reserved.
            _ => Self::new(kind),
        })
    }
    fn rows(&self) -> usize {
        match self {
            Self::Int32(values) => values.len(),
            Self::Int64(values) => values.len(),
            Self::Float64(values) => values.len(),
            Self::ListInt64 { valid, .. } | Self::Utf8 { valid, .. } => valid.len(),
        }
    }
    /// Room for `rows` more rows (fixed-size fields exactly; ragged fields
    /// reserve offsets and validity only).
    fn reserve(&mut self, rows: usize) -> Result<(), EngineError> {
        const WHAT: &str = "detail column";
        match self {
            Self::Int32(values) => try_reserve_exact(values, rows, WHAT),
            Self::Int64(values) => try_reserve_exact(values, rows, WHAT),
            Self::Float64(values) => try_reserve_exact(values, rows, WHAT),
            Self::ListInt64 { offsets, valid, .. } | Self::Utf8 { offsets, valid, .. } => {
                try_reserve_exact(offsets, rows, WHAT)?;
                try_reserve_exact(valid, rows, WHAT)
            }
        }
    }
    fn push_masked(&mut self) -> Result<(), EngineError> {
        const WHAT: &str = "detail column";
        match self {
            Self::Int32(values) => {
                try_reserve(values, 1, WHAT)?;
                values.push(i32::MAX);
            }
            Self::Int64(values) => {
                try_reserve(values, 1, WHAT)?;
                values.push(i64::MAX);
            }
            Self::Float64(values) => {
                try_reserve(values, 1, WHAT)?;
                values.push(CANONICAL_NAN);
            }
            Self::ListInt64 { offsets, valid, .. } | Self::Utf8 { offsets, valid, .. } => {
                try_reserve(offsets, 1, WHAT)?;
                try_reserve(valid, 1, WHAT)?;
                offsets.push(*offsets.last().unwrap());
                valid.push(0);
            }
        }
        Ok(())
    }
    fn clear(&mut self) {
        match self {
            Self::Int32(values) => values.clear(),
            Self::Int64(values) => values.clear(),
            Self::Float64(values) => values.clear(),
            Self::ListInt64 {
                offsets,
                values,
                valid,
            } => {
                offsets.truncate(1);
                values.clear();
                valid.clear();
            }
            Self::Utf8 {
                offsets,
                data,
                valid,
            } => {
                offsets.truncate(1);
                data.clear();
                valid.clear();
            }
        }
    }
}

fn rebase(offsets: &mut Vec<i64>, other: &[i64]) -> Result<(), EngineError> {
    try_reserve(offsets, other.len() - 1, "detail column offsets")?;
    let base = *offsets.last().unwrap();
    offsets.extend(other[1..].iter().map(|&offset| base + offset));
    Ok(())
}

/// Move all of `other` after `target`, reserving fallibly first.
fn move_all<T>(target: &mut Vec<T>, other: &mut Vec<T>) -> Result<(), EngineError> {
    try_reserve(target, other.len(), "detail column")?;
    target.append(other);
    Ok(())
}

/// `rows` picked from `values`, into a fallibly reserved vector.
fn pick<T: Copy>(values: &[T], rows: &[usize]) -> Result<Vec<T>, EngineError> {
    let mut out = Vec::new();
    try_reserve_exact(&mut out, rows.len(), "detail column")?;
    out.extend(rows.iter().map(|&r| values[r]));
    Ok(out)
}

/// Rows `rows` of an offsets/values layout, into fallibly reserved vectors.
fn pick_ragged<T: Copy>(
    offsets: &[i64],
    values: &[T],
    rows: &[usize],
) -> Result<(Vec<i64>, Vec<T>), EngineError> {
    let total = rows.iter().try_fold(0usize, |sum, &r| {
        sum.checked_add((offsets[r + 1] - offsets[r]) as usize)
    });
    let total = total.ok_or_else(|| allocation_error("detail column", "size overflows usize"))?;
    let mut out_offsets = Vec::new();
    try_reserve_exact(&mut out_offsets, rows.len() + 1, "detail column offsets")?;
    let mut out_values = Vec::new();
    try_reserve_exact(&mut out_values, total, "detail column")?;
    out_offsets.push(0);
    for &r in rows {
        out_values.extend_from_slice(&values[offsets[r] as usize..offsets[r + 1] as usize]);
        out_offsets.push(out_values.len() as i64);
    }
    Ok((out_offsets, out_values))
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DetailColumn {
    /// Index of the field in the kernel's `detail_fields()`.
    pub field: usize,
    pub name: &'static str,
    /// Declared kind; Utf8 and Binary share the `DetailData::Utf8` layout.
    pub kind: FieldKind,
    pub data: DetailData,
}

/// Requested detail columns, all with the same number of rows.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct DetailColumns {
    columns: Vec<DetailColumn>,
    /// Column position of each requested field index (u8::MAX if absent).
    slot: [u8; 32],
}

impl DetailColumns {
    /// Empty columns for the requested fields, in the kernel's field order.
    pub fn new(fields: &[DetailField], request: DetailRequest) -> Self {
        let mut result = Self {
            columns: Vec::new(),
            slot: [u8::MAX; 32],
        };
        for (index, field) in fields.iter().enumerate() {
            if request.contains(index) {
                result.slot[index] = result.columns.len() as u8;
                result.columns.push(DetailColumn {
                    field: index,
                    name: field.name,
                    kind: field.kind,
                    data: DetailData::new(field.kind),
                });
            }
        }
        result
    }

    /// [`Self::new`] with every allocation fallible: a refused reservation is a
    /// DeviceError, not an abort. Worker threads build their columns this way
    /// (engine-layout §6).
    pub fn try_new(fields: &[DetailField], request: DetailRequest) -> Result<Self, EngineError> {
        let mut result = Self {
            columns: Vec::new(),
            slot: [u8::MAX; 32],
        };
        let wanted = (0..fields.len())
            .filter(|&index| request.contains(index))
            .count();
        try_reserve_exact(&mut result.columns, wanted, "detail columns")?;
        for (index, field) in fields.iter().enumerate() {
            if request.contains(index) {
                result.slot[index] = result.columns.len() as u8;
                result.columns.push(DetailColumn {
                    field: index,
                    name: field.name,
                    kind: field.kind,
                    data: DetailData::try_new(field.kind)?,
                });
            }
        }
        Ok(result)
    }

    /// `rows` masked rows of numeric columns, reserved fallibly (dense sinks).
    pub fn masked_numeric(
        fields: &[DetailField],
        request: DetailRequest,
        rows: usize,
        what: &str,
    ) -> Result<Self, EngineError> {
        let mut result = Self::new(fields, request);
        for column in &mut result.columns {
            match &mut column.data {
                DetailData::Int32(values) => {
                    *values = crate::sink::try_filled(rows, i32::MAX, what)?;
                }
                DetailData::Int64(values) => {
                    *values = crate::sink::try_filled(rows, i64::MAX, what)?;
                }
                DetailData::Float64(values) => {
                    *values = crate::sink::try_filled(rows, CANONICAL_NAN, what)?;
                }
                _ => {
                    return Err(EngineError::Parameter(format!(
                        "dense output holds numeric fields only; {} is not numeric",
                        column.name
                    )))
                }
            }
        }
        Ok(result)
    }

    pub fn is_empty(&self) -> bool {
        self.columns.is_empty()
    }
    pub fn columns(&self) -> &[DetailColumn] {
        &self.columns
    }
    pub fn into_columns(self) -> Vec<DetailColumn> {
        self.columns
    }
    /// Number of rows; every column has the same count.
    pub fn rows(&self) -> usize {
        self.columns.first().map_or(0, |column| column.data.rows())
    }
    /// Room for `rows` more rows in every column, reserved fallibly up front.
    pub fn reserve(&mut self, rows: usize) -> Result<(), EngineError> {
        for column in &mut self.columns {
            column.data.reserve(rows)?;
        }
        Ok(())
    }
    /// Whether the kernel field `field` is requested.
    #[inline]
    pub fn wants(&self, field: usize) -> bool {
        field < 32 && self.slot[field] != u8::MAX
    }
    #[inline]
    fn column(&mut self, field: usize) -> &mut DetailData {
        let slot = self.slot[field];
        assert!(slot != u8::MAX, "detail field {field} was not requested");
        &mut self.columns[slot as usize].data
    }
    // Each push reserves before it writes, so a failed push leaves every
    // column unchanged.
    #[inline]
    pub fn push_i32(&mut self, field: usize, value: i32) -> Result<(), EngineError> {
        match self.column(field) {
            DetailData::Int32(values) => {
                try_reserve(values, 1, "detail column")?;
                values.push(value);
            }
            _ => panic!("detail field {field} is not int32"),
        }
        Ok(())
    }
    #[inline]
    pub fn push_i64(&mut self, field: usize, value: i64) -> Result<(), EngineError> {
        match self.column(field) {
            DetailData::Int64(values) => {
                try_reserve(values, 1, "detail column")?;
                values.push(value);
            }
            _ => panic!("detail field {field} is not int64"),
        }
        Ok(())
    }
    /// Push a float64; a NaN is stored as [`CANONICAL_NAN`].
    #[inline]
    pub fn push_f64(&mut self, field: usize, value: f64) -> Result<(), EngineError> {
        match self.column(field) {
            DetailData::Float64(values) => {
                try_reserve(values, 1, "detail column")?;
                values.push(canonical(value));
            }
            _ => panic!("detail field {field} is not float64"),
        }
        Ok(())
    }
    pub fn push_list(&mut self, field: usize, items: &[i64]) -> Result<(), EngineError> {
        match self.column(field) {
            DetailData::ListInt64 {
                offsets,
                values,
                valid,
            } => {
                try_reserve(values, items.len(), "detail list values")?;
                try_reserve(offsets, 1, "detail column offsets")?;
                try_reserve(valid, 1, "detail column validity")?;
                values.extend_from_slice(items);
                offsets.push(values.len() as i64);
                valid.push(1);
            }
            _ => panic!("detail field {field} is not list<int64>"),
        }
        Ok(())
    }
    pub fn push_str(&mut self, field: usize, text: &[u8]) -> Result<(), EngineError> {
        match self.column(field) {
            DetailData::Utf8 {
                offsets,
                data,
                valid,
            } => {
                try_reserve(data, text.len(), "detail string data")?;
                try_reserve(offsets, 1, "detail column offsets")?;
                try_reserve(valid, 1, "detail column validity")?;
                data.extend_from_slice(text);
                offsets.push(data.len() as i64);
                valid.push(1);
            }
            _ => panic!("detail field {field} is not a string or binary field"),
        }
        Ok(())
    }
    /// Replace the whole column of requested field `field` with `data` of the
    /// same kind (a backend's bulk result). The caller keeps every column at
    /// the same row count.
    pub fn set_column(&mut self, field: usize, data: DetailData) {
        let column = self.column(field);
        assert!(
            std::mem::discriminant(column) == std::mem::discriminant(&data),
            "detail field {field} has another kind"
        );
        *column = data;
    }
    /// One row of mask values in every column (INT32_MAX, INT64_MAX or null).
    pub fn push_masked(&mut self) -> Result<(), EngineError> {
        for column in &mut self.columns {
            column.data.push_masked()?;
        }
        Ok(())
    }
    pub fn clear(&mut self) {
        for column in &mut self.columns {
            column.data.clear();
        }
    }
    /// Move every row of `other` (same request) after this one's rows.
    pub fn append(&mut self, other: &mut Self) -> Result<(), EngineError> {
        for (left, right) in self.columns.iter_mut().zip(&mut other.columns) {
            match (&mut left.data, &mut right.data) {
                (DetailData::Int32(x), DetailData::Int32(y)) => move_all(x, y)?,
                (DetailData::Int64(x), DetailData::Int64(y)) => move_all(x, y)?,
                (DetailData::Float64(x), DetailData::Float64(y)) => move_all(x, y)?,
                (
                    DetailData::ListInt64 {
                        offsets,
                        values,
                        valid,
                    },
                    DetailData::ListInt64 {
                        offsets: o,
                        values: v,
                        valid: m,
                    },
                ) => {
                    rebase(offsets, o)?;
                    move_all(values, v)?;
                    move_all(valid, m)?;
                }
                (
                    DetailData::Utf8 {
                        offsets,
                        data,
                        valid,
                    },
                    DetailData::Utf8 {
                        offsets: o,
                        data: d,
                        valid: m,
                    },
                ) => {
                    rebase(offsets, o)?;
                    move_all(data, d)?;
                    move_all(valid, m)?;
                }
                _ => unreachable!("appended detail columns share one request"),
            }
        }
        other.clear();
        Ok(())
    }
    /// Rows `rows` (any order, repeats allowed) as new columns.
    pub fn gather(&self, rows: &[usize]) -> Result<Self, EngineError> {
        let mut result = Self {
            columns: Vec::with_capacity(self.columns.len()),
            slot: self.slot,
        };
        for column in &self.columns {
            let data = match &column.data {
                DetailData::Int32(x) => DetailData::Int32(pick(x, rows)?),
                DetailData::Int64(x) => DetailData::Int64(pick(x, rows)?),
                DetailData::Float64(x) => DetailData::Float64(pick(x, rows)?),
                DetailData::ListInt64 {
                    offsets,
                    values,
                    valid,
                } => {
                    let (offsets, values) = pick_ragged(offsets, values, rows)?;
                    DetailData::ListInt64 {
                        offsets,
                        values,
                        valid: pick(valid, rows)?,
                    }
                }
                DetailData::Utf8 {
                    offsets,
                    data,
                    valid,
                } => {
                    let (offsets, data) = pick_ragged(offsets, data, rows)?;
                    DetailData::Utf8 {
                        offsets,
                        data,
                        valid: pick(valid, rows)?,
                    }
                }
            };
            result.columns.push(DetailColumn {
                field: column.field,
                name: column.name,
                kind: column.kind,
                data,
            });
        }
        Ok(result)
    }
    /// Number of columns; for a dense request, its numeric field count (the
    /// stride of flat numeric rows).
    pub fn width(&self) -> usize {
        self.columns.len()
    }
    /// Numeric values of row `row`, in column order (dense sinks only).
    pub fn numeric_row(&self, row: usize) -> DenseRow {
        let mut result = [0i64; MAX_DENSE_DETAILS];
        for (slot, column) in result.iter_mut().zip(&self.columns) {
            *slot = numeric_value(column, row);
        }
        result
    }
    /// Append the `width()` numeric values of row `row`, in column order, to
    /// `out` (dense sinks only).
    pub fn push_numeric_row(&self, row: usize, out: &mut Vec<i64>) {
        out.extend(self.columns.iter().map(|column| numeric_value(column, row)));
    }
    /// Store a numeric row at `cell` (dense sinks only).
    pub fn set_numeric_row(&mut self, cell: usize, values: &DenseRow) {
        for (&value, column) in values.iter().zip(&mut self.columns) {
            match &mut column.data {
                // Values came from an int32 column, so they fit.
                DetailData::Int32(x) => x[cell] = value as i32,
                DetailData::Int64(x) => x[cell] = value,
                DetailData::Float64(x) => x[cell] = f64::from_bits(value as u64),
                _ => unreachable!("dense detail columns are numeric"),
            }
        }
    }
}

/// One numeric value; a float64 value is carried as its bit pattern.
fn numeric_value(column: &DetailColumn, row: usize) -> i64 {
    match &column.data {
        DetailData::Int32(x) => i64::from(x[row]),
        DetailData::Int64(x) => x[row],
        DetailData::Float64(x) => x[row].to_bits() as i64,
        _ => unreachable!("dense detail columns are numeric"),
    }
}

/// The DenseRow of a flat numeric row (at most MAX_DENSE_DETAILS values),
/// zero-padded.
pub fn dense_row(values: &[i64]) -> DenseRow {
    let mut row = [0i64; MAX_DENSE_DETAILS];
    row[..values.len()].copy_from_slice(values);
    row
}

/// Check that a dense sink can hold every requested field; returns the
/// per-cell byte width of the requested numeric fields.
pub fn dense_request_bytes(
    fields: &[DetailField],
    request: DetailRequest,
) -> Result<usize, EngineError> {
    let mut bytes = 0;
    let mut count = 0;
    for (index, field) in fields.iter().enumerate() {
        if !request.contains(index) {
            continue;
        }
        bytes += match field.kind {
            FieldKind::Int32 => 4,
            FieldKind::Int64 | FieldKind::Float64 => 8,
            FieldKind::ListInt64 | FieldKind::Utf8 | FieldKind::Binary => {
                return Err(EngineError::Parameter(format!(
                    "field {:?} is a list/string/binary field; Matrix and Tiles hold numeric fields only. \
                     Use keep=sb.keep.each(), hits() or topk() for it",
                    field.name
                )))
            }
        };
        count += 1;
    }
    if count > MAX_DENSE_DETAILS {
        return Err(EngineError::Parameter(format!(
            "dense output carries at most {MAX_DENSE_DETAILS} numeric detail fields"
        )));
    }
    Ok(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;

    const FIELDS: [DetailField; 4] = [
        DetailField {
            name: "x",
            kind: FieldKind::Int64,
        },
        DetailField {
            name: "n",
            kind: FieldKind::Int32,
        },
        DetailField {
            name: "l",
            kind: FieldKind::ListInt64,
        },
        DetailField {
            name: "s",
            kind: FieldKind::Utf8,
        },
    ];

    fn all() -> DetailRequest {
        (0..4).fold(DetailRequest::NONE, DetailRequest::with)
    }

    /// The fallible constructor builds exactly what `new` builds, for every
    /// subset of a field list with each kind.
    #[test]
    fn try_new_matches_new() {
        for bits in 0..16u32 {
            let request = (0..4)
                .filter(|index| bits >> index & 1 == 1)
                .fold(DetailRequest::NONE, DetailRequest::with);
            let fallible = DetailColumns::try_new(&FIELDS, request).unwrap();
            assert_eq!(fallible, DetailColumns::new(&FIELDS, request));
            assert_eq!(fallible.columns.capacity(), bits.count_ones() as usize);
        }
    }

    /// Allocation errors keep their message; an error whose message could not
    /// be reserved (empty) displays the fixed out-of-memory text.
    #[test]
    fn allocation_error_messages() {
        let error = allocation_error("hit indices", "capacity overflow");
        assert!(matches!(error, EngineError::Device(_)));
        assert_eq!(
            error.to_string(),
            "unable to reserve hit indices: capacity overflow; no result was published"
        );
        let fallback = EngineError::Device(String::new());
        assert_eq!(fallback.to_string(), crate::validate::OUT_OF_MEMORY);
        // Row-reduction scratch, reserved on worker threads, keeps its message too,
        // with the bytes (n × size_of::<T>(), computed in u128) and the suffix.
        let error = crate::sink::try_filled(usize::MAX, 0u64, "rowstats worker state").unwrap_err();
        let reason = Vec::<u64>::new().try_reserve_exact(usize::MAX).unwrap_err();
        assert!(matches!(error, EngineError::Device(_)));
        let bytes = u128::from(u64::MAX) * 8;
        assert_eq!(
            error.to_string(),
            format!(
                "unable to reserve rowstats worker state ({bytes} bytes): {reason}; \
                 no result was published"
            )
        );
        // The detail helpers report (len + additional) × size_of::<T>().
        let mut buffer = vec![0i32; 3];
        let error = try_reserve(&mut buffer, usize::MAX, "the key column").unwrap_err();
        let reason = vec![0i32; 3].try_reserve(usize::MAX).unwrap_err();
        let bytes = (u128::from(u64::MAX) + 3) * 4;
        assert_eq!(
            error.to_string(),
            format!(
                "unable to reserve the key column ({bytes} bytes): {reason}; \
                 no result was published"
            )
        );
        let error = try_reserve_exact(&mut buffer, usize::MAX - 1, "a mask").unwrap_err();
        let bytes = (u128::from(u64::MAX) + 2) * 4;
        assert!(error
            .to_string()
            .starts_with(&format!("unable to reserve a mask ({bytes} bytes): ")));
        assert!(error.to_string().ends_with("; no result was published"));
        assert_eq!(buffer, [0, 0, 0]);
    }

    fn row(out: &mut DetailColumns, k: i64) {
        out.push_i64(0, k).unwrap();
        out.push_i32(1, k as i32).unwrap();
        out.push_list(2, &vec![k; k as usize]).unwrap();
        out.push_str(3, format!("{k}=").as_bytes()).unwrap();
    }

    #[test]
    fn append_gather_and_masks_keep_rows_aligned() {
        let mut left = DetailColumns::new(&FIELDS, all());
        row(&mut left, 1);
        left.push_masked().unwrap();
        let mut right = DetailColumns::new(&FIELDS, all());
        row(&mut right, 2);
        row(&mut right, 0);
        left.append(&mut right).unwrap();
        assert_eq!(right.rows(), 0);
        assert_eq!(left.rows(), 4);
        let mut expected = DetailColumns::new(&FIELDS, all());
        row(&mut expected, 1);
        expected.push_masked().unwrap();
        row(&mut expected, 2);
        row(&mut expected, 0);
        assert_eq!(left, expected);
        let picked = left.gather(&[2, 1, 2]).unwrap();
        let mut want = DetailColumns::new(&FIELDS, all());
        row(&mut want, 2);
        want.push_masked().unwrap();
        row(&mut want, 2);
        assert_eq!(picked, want);
        match &picked.columns()[3].data {
            DetailData::Utf8 {
                offsets,
                data,
                valid,
            } => {
                assert_eq!(offsets, &[0, 2, 2, 4]);
                assert_eq!(data, b"2=2=");
                assert_eq!(valid, &[1, 0, 1]);
            }
            _ => unreachable!(),
        }
    }

    #[test]
    fn failed_reservations_are_device_errors_and_change_nothing() {
        // Larger than isize::MAX bytes: refused without attempting an allocation.
        let mut buffer = vec![1u64, 2];
        let huge = usize::MAX / 8;
        match try_reserve_exact(&mut buffer, huge, "test scratch") {
            Err(EngineError::Device(message)) => {
                assert!(message.contains("test scratch"), "{message}");
                assert!(message.contains("no result was published"), "{message}");
            }
            other => panic!("expected DeviceError, got {other:?}"),
        }
        assert!(matches!(
            try_reserve(&mut buffer, huge, "test column"),
            Err(EngineError::Device(_))
        ));
        assert_eq!(buffer, [1, 2]);
    }

    #[test]
    fn dense_requests_reject_text_fields() {
        let numeric = DetailRequest::NONE.with(0).with(1);
        assert_eq!(dense_request_bytes(&FIELDS, numeric).unwrap(), 12);
        assert!(matches!(
            dense_request_bytes(&FIELDS, numeric.with(3)),
            Err(EngineError::Parameter(_))
        ));
        let mut dense = DetailColumns::masked_numeric(&FIELDS, numeric, 3, "test").unwrap();
        dense.set_numeric_row(1, &dense_row(&[7, 8]));
        assert_eq!(dense.numeric_row(1), dense_row(&[7, 8]));
        assert_eq!(
            dense.numeric_row(0),
            dense_row(&[i64::MAX, i64::from(i32::MAX)])
        );
    }
}
