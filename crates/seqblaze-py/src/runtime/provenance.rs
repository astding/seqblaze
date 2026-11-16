//! Versioned logical input hashes and independent, owned replay snapshots.
//!
//! The hash describes values, not a producer's layout: fixed-width padding,
//! strides, Arrow offsets/chunks and null payload bytes are never encoded.
//! Sequence tables use merkle-blake3-v1 (BLAKE3, D128); pair and selection
//! indices keep merkle-sha256-v1 (SHA-256). Spec: result-provenance.md.
use super::{error, indices::PyIndexColumn, PySeqTable};
use pyo3::{
    prelude::*,
    types::{PyBytes, PyDict},
};
use seqblaze_core::{
    detail,
    sched::{self, Grain, Workers},
    seq::{SeqTable, STREAM_HEAD, STREAM_LEAF},
    validate::EngineError,
    work::{IndexArray, IndexInfo, PairKind, WorkSource},
};
use sha2::{Digest, Sha256};
use std::sync::atomic::{AtomicU8, Ordering};

const PREFIX: &[u8] = b"SEQBLAZE\0";
const VERSION: u16 = 1;
const CHUNK_BYTES: usize = 65_536;
const INDEX_CHUNK_VALUES: usize = CHUNK_BYTES / 8;
const LENGTH_CHUNK_VALUES: usize = CHUNK_BYTES / 8;
/// A table leaf is framed in place: its frame head (at most 40 bytes for
/// `row-lengths`, 38 for `row-bytes`, both including the u64 count) ends at
/// LEAF_DATA, where the stream slice starts, so B hashes one contiguous frame.
/// The engine's stream pass leaves the same room before each leaf it hands
/// over, and its leaves are the byte-stream leaves.
const LEAF_DATA: usize = STREAM_HEAD;
const _: () = assert!(STREAM_LEAF == CHUNK_BYTES);
type LeafBuffer = [u8; LEAF_DATA + CHUNK_BYTES];

#[derive(Clone, Copy)]
struct Stats {
    n: u64,
    total_nt: u64,
    min_len: u64,
    max_len: u64,
    valid_rows: u64,
    text: bool,
}

struct Summary {
    stats: Stats,
    digest: [u8; 32],
}

/// Per-row logical lengths (-1 for a null row).
struct Geometry {
    stats: Stats,
    lengths: Vec<i64>,
}

#[derive(Clone, Copy)]
struct Node {
    count: u64,
    digest: [u8; 32],
}

const ZERO_NODE: Node = Node {
    count: 0,
    digest: [0; 32],
};

struct Snapshot {
    summary: Summary,
    data: Vec<u8>,
    offsets: Vec<i64>,
    validity: Vec<u8>,
    python_rows: Vec<u8>,
}

fn overflow() -> EngineError {
    EngineError::Overflow("logical input encoding exceeds platform or uint64 limits".into())
}

/// Read mutable length metadata afresh before accessing any row. This table is
/// the operation's `for_call` view; the retained owner must not mutate until the
/// operation completes, just as during a kernel call.
///
/// One pass over the prepared row lengths (`TableRows::for_each_len`): a null
/// row has an empty span in storage, so it adds nothing to the total or the
/// maximum, and its length is then replaced by -1 and left out of the minimum.
fn geometry(table: &SeqTable) -> Result<Geometry, EngineError> {
    table.validate_lengths()?;
    let n = table.len();
    let nulls = table.nulls();
    let mut stats = Stats {
        n: u64::try_from(n).map_err(|_| overflow())?,
        total_nt: 0,
        min_len: u64::MAX,
        max_len: 0,
        valid_rows: (n - nulls.len()) as u64,
        text: table.is_text(),
    };
    let mut lengths = Vec::new();
    reserve(&mut lengths, n, "logical row lengths")?;
    // A prepared full-width span already certifies every row's length. Fill
    // the canonical stream without reducing those same lengths again. Keep
    // the reservation above in its original order, after validation/computation.
    if let Some((_, bytes)) = table.contiguous_fixed_span().filter(|_| nulls.is_empty()) {
        let len = table.rows().seq_len(0);
        i64::try_from(len).map_err(|_| overflow())?;
        i64::try_from(bytes).map_err(|_| overflow())?;
        lengths.resize(n, len as i64);
        stats.total_nt = bytes as u64;
        stats.min_len = len as u64;
        stats.max_len = len as u64;
        return Ok(Geometry { stats, lengths });
    }
    // usize lengths fit u64; an i64 or total overflow raises the same error
    // as a row-by-row check would.
    let mut wrapped = false;
    table.rows().for_each_len(|len| {
        let len = len as u64;
        let (total, carry) = stats.total_nt.overflowing_add(len);
        stats.total_nt = total;
        wrapped |= carry;
        stats.min_len = stats.min_len.min(len);
        stats.max_len = stats.max_len.max(len);
        lengths.push(len as i64);
    });
    if wrapped || stats.max_len > i64::MAX as u64 {
        return Err(overflow());
    }
    if !nulls.is_empty() {
        for &row in nulls {
            lengths[row] = -1;
        }
        stats.min_len = lengths
            .iter()
            .filter(|&&len| len >= 0)
            .fold(u64::MAX, |low, &len| low.min(len as u64));
    }
    i64::try_from(stats.total_nt).map_err(|_| overflow())?;
    if stats.valid_rows == 0 {
        stats.min_len = 0;
    }
    Ok(Geometry { stats, lengths })
}

/// The exclusive prefix of the valid rows' lengths: row r owns byte-stream
/// positions offsets[r]..offsets[r + 1]. Only the per-row gather needs it.
fn offsets(geometry: &Geometry) -> Result<Vec<u64>, EngineError> {
    let mut offsets = Vec::new();
    reserve(
        &mut offsets,
        geometry.lengths.len().checked_add(1).ok_or_else(overflow)?,
        "logical row offsets",
    )?;
    let mut total = 0u64;
    offsets.push(0);
    for &len in &geometry.lengths {
        // geometry checked that the sum of the valid lengths fits u64.
        total += len.max(0) as u64;
        offsets.push(total);
    }
    Ok(offsets)
}

/// H: every merkle-sha256-v1 leaf, parent and root has the same versioned
/// SHA256 frame.
fn hasher(domain: &[u8], payload_len: u64) -> Sha256 {
    let mut digest = Sha256::new();
    digest.update(PREFIX);
    digest.update(VERSION.to_le_bytes());
    digest.update((domain.len() as u16).to_le_bytes());
    digest.update(domain);
    digest.update(payload_len.to_le_bytes());
    digest
}

/// Write v1's frame head (magic, version, domain length, domain, payload
/// length) so that it ends at `end`; returns where it starts.
fn frame_head(domain: &[u8], payload_len: u64, frame: &mut [u8], end: usize) -> usize {
    let start = end - (PREFIX.len() + 4 + domain.len() + 8);
    let mut at = start;
    for part in [
        PREFIX,
        &VERSION.to_le_bytes(),
        &(domain.len() as u16).to_le_bytes(),
        domain,
        &payload_len.to_le_bytes(),
    ] {
        frame[at..at + part.len()].copy_from_slice(part);
        at += part.len();
    }
    start
}

/// B (merkle-blake3-v1, D128): BLAKE3 in plain hash mode, 32-byte output, over
/// v1's frame of `domain` and the concatenated `payload` parts (at most 80
/// bytes: the empty tree, a node and the identity).
fn blake3_small(domain: &[u8], payload: &[&[u8]]) -> [u8; 32] {
    let mut frame = [0u8; 128];
    let len: usize = payload.iter().map(|part| part.len()).sum();
    let head = 48;
    let start = frame_head(domain, len as u64, &mut frame, head);
    let mut at = head;
    for part in payload {
        frame[at..at + part.len()].copy_from_slice(part);
        at += part.len();
    }
    *blake3::hash(&frame[start..at]).as_bytes()
}

/// The hash of an ordered leaf tree: H for indices (merkle-sha256-v1), B for
/// sequence tables (merkle-blake3-v1). Both use v1's tree rules and domains.
#[derive(Clone, Copy)]
enum TreeHash {
    Sha256,
    Blake3,
}

impl TreeHash {
    fn empty(self) -> [u8; 32] {
        match self {
            TreeHash::Sha256 => hasher(b"logical-empty", 0).finalize().into(),
            TreeHash::Blake3 => blake3_small(b"logical-empty", &[]),
        }
    }

    fn combine(self, left: Node, right: Node) -> Result<Node, EngineError> {
        let count = left.count.checked_add(right.count).ok_or_else(overflow)?;
        let (left_count, right_count) = (left.count.to_le_bytes(), right.count.to_le_bytes());
        let digest = match self {
            TreeHash::Sha256 => {
                let mut digest = hasher(b"logical-node", 80);
                digest.update(left_count);
                digest.update(left.digest);
                digest.update(right_count);
                digest.update(right.digest);
                digest.finalize().into()
            }
            TreeHash::Blake3 => blake3_small(
                b"logical-node",
                &[&left_count, &left.digest, &right_count, &right.digest],
            ),
        };
        Ok(Node { count, digest })
    }
}

fn worker_count(threads: usize, items: usize, bytes_per_item: usize) -> usize {
    Workers {
        limit: threads,
        grain: Grain::Auto,
    }
    .count(items, bytes_per_item)
}

fn nodes(count: usize, label: &str) -> Result<Vec<Node>, EngineError> {
    let mut output = Vec::new();
    reserve(&mut output, count, label)?;
    output.resize(count, ZERO_NODE);
    Ok(output)
}

fn tree(mut level: Vec<Node>, threads: usize, hash: TreeHash) -> Result<[u8; 32], EngineError> {
    if level.is_empty() {
        return Ok(hash.empty());
    }
    if level.len() == 1 {
        return Ok(level[0].digest);
    }
    let mut next = nodes(level.len().div_ceil(2), "Merkle reduction")?;
    while level.len() > 1 {
        next.resize(level.len().div_ceil(2), ZERO_NODE);
        let workers = worker_count(threads, next.len(), 80);
        sched::fill_chunks(&mut next, workers, &|start, output| {
            for (offset, node) in output.iter_mut().enumerate() {
                let index = 2 * (start + offset);
                *node = if index + 1 < level.len() {
                    hash.combine(level[index], level[index + 1])?
                } else {
                    level[index]
                };
            }
            Ok(())
        })?;
        std::mem::swap(&mut level, &mut next);
    }
    Ok(level[0].digest)
}

/// B("row-bytes", u64(c) || bytes) of the c bytes at buffer[LEAF_DATA..].
fn byte_leaf(buffer: &mut [u8], len: usize) -> [u8; 32] {
    let count = LEAF_DATA - 8;
    buffer[count..LEAF_DATA].copy_from_slice(&(len as u64).to_le_bytes());
    let start = frame_head(b"row-bytes", 8 + len as u64, buffer, count);
    *blake3::hash(&buffer[start..LEAF_DATA + len]).as_bytes()
}

/// B("row-lengths", u64(k) || k i64LE values).
fn length_leaf(values: &[i64], buffer: &mut LeafBuffer) -> [u8; 32] {
    let len = values.len() * 8;
    for (word, value) in buffer[LEAF_DATA..LEAF_DATA + len]
        .chunks_exact_mut(8)
        .zip(values)
    {
        word.copy_from_slice(&value.to_le_bytes());
    }
    let count = LEAF_DATA - 8;
    buffer[count..LEAF_DATA].copy_from_slice(&(values.len() as u64).to_le_bytes());
    let start = frame_head(b"row-lengths", 8 + len as u64, buffer, count);
    *blake3::hash(&buffer[start..LEAF_DATA + len]).as_bytes()
}

/// merkle-blake3-v1 (result-provenance.md, D128): ordered trees over the
/// length stream's 8192-value leaves and the byte stream's 65536-byte leaves,
/// all under B. `fill(first, out)` writes byte-stream positions
/// first..first + out.len() into `out`. Leaves of both streams are hashed in
/// parallel; neither tree depends on the scheduling geometry.
fn summarize<F>(geometry: &Geometry, threads: usize, fill: &F) -> Result<Summary, EngineError>
where
    F: Fn(u64, &mut [u8]) + Sync,
{
    let lengths = &geometry.lengths;
    let mut length_leaves = nodes(
        lengths.len().div_ceil(LENGTH_CHUNK_VALUES),
        "row-length digests",
    )?;
    // Rows of one length and no nulls: every full length leaf holds the same
    // values, so it is hashed once.
    let stats = &geometry.stats;
    let full = (stats.valid_rows == stats.n
        && stats.min_len == stats.max_len
        && lengths.len() >= LENGTH_CHUNK_VALUES)
        .then(|| {
            let mut buffer: LeafBuffer = [0u8; LEAF_DATA + CHUNK_BYTES];
            length_leaf(&lengths[..LENGTH_CHUNK_VALUES], &mut buffer)
        });
    let workers = worker_count(threads, length_leaves.len(), CHUNK_BYTES);
    sched::fill_chunks(&mut length_leaves, workers, &|start, output| {
        let mut buffer: LeafBuffer = [0u8; LEAF_DATA + CHUNK_BYTES];
        for (offset, node) in output.iter_mut().enumerate() {
            let first = (start + offset) * LENGTH_CHUNK_VALUES;
            let last = (first + LENGTH_CHUNK_VALUES).min(lengths.len());
            let digest = match full {
                Some(digest) if last - first == LENGTH_CHUNK_VALUES => digest,
                _ => length_leaf(&lengths[first..last], &mut buffer),
            };
            *node = Node { count: 1, digest };
        }
        Ok(())
    })?;
    let total = geometry.stats.total_nt;
    let byte_count = usize::try_from(total).map_err(|_| overflow())?;
    let mut byte_leaves = nodes(byte_count.div_ceil(CHUNK_BYTES), "row-byte digests")?;
    let workers = worker_count(threads, byte_leaves.len(), CHUNK_BYTES);
    sched::fill_chunks(&mut byte_leaves, workers, &|start, output| {
        let mut buffer: LeafBuffer = [0u8; LEAF_DATA + CHUNK_BYTES];
        for (offset, node) in output.iter_mut().enumerate() {
            let first = ((start + offset) * CHUNK_BYTES) as u64;
            let len = (total - first).min(CHUNK_BYTES as u64) as usize;
            fill(first, &mut buffer[LEAF_DATA..LEAF_DATA + len]);
            *node = Node {
                count: 1,
                digest: byte_leaf(&mut buffer, len),
            };
        }
        Ok(())
    })?;
    let length_root = tree(length_leaves, threads, TreeHash::Blake3)?;
    let byte_root = tree(byte_leaves, threads, TreeHash::Blake3)?;
    let digest = blake3_small(
        b"sequence-table-blake3",
        &[
            &geometry.stats.n.to_le_bytes(),
            &total.to_le_bytes(),
            &length_root,
            &byte_root,
        ],
    );
    Ok(Summary {
        stats: geometry.stats,
        digest,
    })
}

/// Gather byte-stream positions first..first + buffer.len() from the rows that
/// hold them. Null and empty rows own no positions and are never read.
fn gather(table: &SeqTable, offsets: &[u64], first: u64, buffer: &mut [u8]) {
    // The last row starting at or before `first` (offsets[0] = 0 <= first); it
    // holds position `first`, since first < total.
    let row = offsets.partition_point(|&offset| offset <= first) - 1;
    table
        .rows()
        .copy_concatenated(row, (first - offsets[row]) as usize, buffer);
}

/// S13b: the CPU encoding for a host table whose two logical streams each
/// fit one leaf. The caller has completed whole-input preflight on this view.
/// Reuse the CPU frame/leaf routines, with bounded stack scratch (no new
/// input-scaled reservations in the upload path).
#[cfg(feature = "cuda")]
pub(super) fn single_leaf_digest(table: &SeqTable, total: u64) -> Option<[u8; 32]> {
    if table.len() > LENGTH_CHUNK_VALUES || total > CHUNK_BYTES as u64 {
        return None;
    }
    let mut buffer: LeafBuffer = [0u8; LEAF_DATA + CHUNK_BYTES];
    let mut lengths = [0i64; LENGTH_CHUNK_VALUES];
    for (row, length) in lengths[..table.len()].iter_mut().enumerate() {
        *length = table.seq_len(row) as i64;
    }
    for &row in table.nulls() {
        lengths[row] = -1;
    }
    let length_root = if table.is_empty() {
        TreeHash::Blake3.empty()
    } else {
        length_leaf(&lengths[..table.len()], &mut buffer)
    };
    let byte_root = if total == 0 {
        TreeHash::Blake3.empty()
    } else {
        let out = &mut buffer[LEAF_DATA..LEAF_DATA + total as usize];
        if !table.copy_fixed_span(0, out) {
            table.rows().copy_concatenated(0, 0, out);
        }
        byte_leaf(&mut buffer, total as usize)
    };
    Some(blake3_small(
        b"sequence-table-blake3",
        &[
            &(table.len() as u64).to_le_bytes(),
            &total.to_le_bytes(),
            &length_root,
            &byte_root,
        ],
    ))
}

fn summarize_table(table: &SeqTable, threads: usize) -> Result<Summary, EngineError> {
    let geometry = geometry(table)?;
    let total = geometry.stats.total_nt;
    // Layouts that already hold the byte stream contiguously skip the per-row
    // walk; the bytes hashed are the same. An owned batch keeps its null rows
    // as empty rows (SeqTable::with_nulls), so its data is the stream itself.
    if let Some(data) = table.owned_batch().map(|batch| batch.data()) {
        if data.len() as u64 == total {
            return summarize(&geometry, threads, &|first, out| {
                let first = first as usize;
                out.copy_from_slice(&data[first..first + out.len()]);
            });
        }
    }
    // A full-width contiguous NumPy view: its span is the stream, read with
    // the same relaxed loads as a borrowed row.
    if table.nulls().is_empty()
        && table
            .contiguous_fixed_span()
            .is_some_and(|(_, bytes)| bytes as u64 == total)
    {
        return summarize(&geometry, threads, &|first, out| {
            // The span proved above holds every position (the stream is the
            // span), so the copy cannot fall short.
            assert!(table.copy_fixed_span(first as usize, out), "a proved span");
        });
    }
    let offsets = offsets(&geometry)?;
    summarize(&geometry, threads, &|first, out| {
        gather(table, &offsets, first, out)
    })
}

/// The logical input identities of a CPU call (result-provenance.md), from
/// its validated per-call tables: what the facade's capture would compute
/// for `a`, `b` and, for index pairs, `i` and `j`, in that order, with the
/// prepared lengths of the call's views. No digest storage is reserved until
/// validation and computation have both succeeded.
pub(super) struct CallDigests {
    a: Summary,
    b: Option<Summary>,
    indices: Option<[(usize, [u8; 32]); 2]>,
}

pub(super) fn call_digests(
    (a, b): (&SeqTable, &SeqTable),
    work: &WorkSource,
    threads: usize,
) -> Result<CallDigests, EngineError> {
    let shared = std::ptr::eq(a, b);
    let summary_a = summarize_table(a, threads)?;
    let summary_b = if shared {
        None
    } else {
        Some(summarize_table(b, threads)?)
    };
    let indices = match &work.kind {
        PairKind::Index { i, j } => Some([
            summarize_indices(i, threads)?,
            summarize_indices(j, threads)?,
        ]),
        _ => None,
    };
    Ok(CallDigests {
        a: summary_a,
        b: summary_b,
        indices,
    })
}

/// The `input_digests` dict of a native response, in the resident schema:
/// `a` and `b` (one dict object when the tables are shared), then `i` and `j`.
pub(super) fn digests_dict<'py>(
    py: Python<'py>,
    digests: &CallDigests,
) -> PyResult<Bound<'py, PyDict>> {
    let result = PyDict::new(py);
    let a = metadata(py, &digests.a)?;
    result.set_item("a", &a)?;
    match &digests.b {
        Some(b) => result.set_item("b", metadata(py, b)?)?,
        None => result.set_item("b", a)?,
    }
    if let Some([(n_i, i), (n_j, j)]) = &digests.indices {
        result.set_item("i", index_metadata(py, *n_i, i)?)?;
        result.set_item("j", index_metadata(py, *n_j, j)?)?;
    }
    Ok(result)
}

/// The host allocation failures §3 form, so a refusal reads like every other
/// input-scaled reservation of the call.
fn reserve<T>(values: &mut Vec<T>, count: usize, label: &str) -> Result<(), EngineError> {
    detail::try_reserve_exact(values, count, &format!("input provenance {label}"))
}

/// An ASCII Python list literal for the public `seqs` constructor. Every byte
/// uses a hex escape, so even quotes, newlines and non-UTF8 data stay literal.
/// Only immutable owned snapshot buffers are read here.
fn python_rows(data: &[u8], offsets: &[i64], validity: &[u8]) -> Result<Vec<u8>, EngineError> {
    let capacity = data
        .len()
        .checked_mul(4)
        .and_then(|bytes| {
            validity
                .len()
                .checked_mul(5)
                .and_then(|rows| bytes.checked_add(rows))
        })
        .and_then(|bytes| bytes.checked_add(2))
        .ok_or_else(overflow)?;
    let mut output = Vec::new();
    reserve(&mut output, capacity, "Python sequence literal")?;
    output.push(b'[');
    const HEX: &[u8; 16] = b"0123456789abcdef";
    for (row, &valid) in validity.iter().enumerate() {
        if row != 0 {
            output.push(b',');
        }
        if valid == 0 {
            output.extend_from_slice(b"None");
        } else {
            output.extend_from_slice(b"b'");
            for &byte in &data[offsets[row] as usize..offsets[row + 1] as usize] {
                output.extend_from_slice(&[
                    b'\\',
                    b'x',
                    HEX[(byte >> 4) as usize],
                    HEX[(byte & 15) as usize],
                ]);
            }
            output.push(b'\'');
        }
    }
    output.push(b']');
    Ok(output)
}

fn snapshot(table: &SeqTable, threads: usize) -> Result<Snapshot, EngineError> {
    let geometry = geometry(table)?;
    let stats = geometry.stats;
    // The packed representation uses signed int64 offsets, regardless of host
    // endian or the original producer's offset width.
    let size = usize::try_from(stats.total_nt).map_err(|_| overflow())?;
    i64::try_from(stats.total_nt).map_err(|_| overflow())?;
    let offset_count = table.len().checked_add(1).ok_or_else(overflow)?;
    let mut data = Vec::new();
    let mut offsets = Vec::new();
    let mut validity = Vec::new();
    reserve(&mut data, size, "sequence data")?;
    reserve(&mut offsets, offset_count, "offsets")?;
    reserve(&mut validity, table.len(), "validity")?;
    data.resize(size, 0);
    offsets.push(0);
    let mut end = 0usize;
    for (row, &length) in geometry.lengths.iter().enumerate() {
        if length < 0 {
            validity.push(0);
        } else {
            validity.push(1);
            let view = table.get(row);
            let next = end.checked_add(length as usize).ok_or_else(overflow)?;
            let destination = data.get_mut(end..next).ok_or_else(overflow)?;
            view.copy_to(0, destination);
            end = next;
        }
        offsets.push(i64::try_from(end).map_err(|_| overflow())?);
    }
    if end != data.len() {
        return Err(overflow());
    }
    // Hash only the owned copy. The caller compares this digest with the one
    // captured at compute time before using these same bytes for replay.
    let summary = summarize(&geometry, threads, &|first, out| {
        out.copy_from_slice(&data[first as usize..first as usize + out.len()]);
    })?;
    let python_rows = python_rows(&data, &offsets, &validity)?;
    Ok(Snapshot {
        summary,
        data,
        offsets,
        validity,
        python_rows,
    })
}

fn hex_digest(digest: &[u8; 32]) -> [u8; 64] {
    let mut hex = [0u8; 64];
    const DIGITS: &[u8; 16] = b"0123456789abcdef";
    for (position, &byte) in digest.iter().enumerate() {
        hex[2 * position] = DIGITS[(byte >> 4) as usize];
        hex[2 * position + 1] = DIGITS[(byte & 15) as usize];
    }
    hex
}

fn metadata<'py>(py: Python<'py>, summary: &Summary) -> PyResult<Bound<'py, PyDict>> {
    let hex = hex_digest(&summary.digest);
    let result = PyDict::new(py);
    result.set_item(
        "sha256",
        std::str::from_utf8(&hex).expect("ASCII hex digits"),
    )?;
    result.set_item("n", summary.stats.n)?;
    result.set_item("total_nt", summary.stats.total_nt)?;
    result.set_item("min_len", summary.stats.min_len)?;
    result.set_item("max_len", summary.stats.max_len)?;
    result.set_item("text", summary.stats.text)?;
    Ok(result)
}

/// Read one validated, retained native signed index without requiring it to be
/// non-negative or in the sequence-table range. Hashing never dereferences an
/// index as a sequence address. Foreign storage is read by relaxed byte loads.
fn index_value(info: &IndexInfo, row: usize) -> i64 {
    debug_assert!(row < info.len && matches!(info.itemsize, 4 | 8));
    let pointer = (info.address as *const u8).wrapping_offset(row as isize * info.stride);
    let mut bytes = [0u8; 8];
    for (offset, byte) in bytes[..info.itemsize].iter_mut().enumerate() {
        // SAFETY: PyIndexColumn's retained IndexArray validated the complete
        // descriptor and span at import. This row lies within that descriptor.
        // AtomicU8 has byte alignment and accepts every possible bit pattern;
        // no shared Rust slice is formed over the foreign allocation.
        *byte =
            unsafe { (&*pointer.wrapping_add(offset).cast::<AtomicU8>()).load(Ordering::Relaxed) };
    }
    if info.itemsize == 4 {
        i32::from_ne_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as i64
    } else {
        i64::from_ne_bytes(bytes)
    }
}

fn index_chunk_digest(values: usize, bytes: &[u8]) -> [u8; 32] {
    debug_assert!(values <= INDEX_CHUNK_VALUES && bytes.len() == values * 8);
    let mut digest = hasher(b"index-chunk", 8 + bytes.len() as u64);
    digest.update((values as u64).to_le_bytes());
    digest.update(bytes);
    digest.finalize().into()
}

/// Ordered index Merkle tree: every leaf contains at most 8192 canonical i64LE
/// values; its count is one chunk, independent of the signed index values.
fn index_digest<F>(n: usize, threads: usize, chunk_hash: &F) -> Result<[u8; 32], EngineError>
where
    F: Fn(usize, usize, &mut [u8; CHUNK_BYTES]) -> [u8; 32] + Sync,
{
    let mut leaves = nodes(n.div_ceil(INDEX_CHUNK_VALUES), "index chunk digests")?;
    let workers = worker_count(threads, leaves.len(), CHUNK_BYTES);
    sched::fill_chunks(&mut leaves, workers, &|start, output| {
        let mut buffer = [0u8; CHUNK_BYTES];
        for (offset, node) in output.iter_mut().enumerate() {
            let first = (start + offset) * INDEX_CHUNK_VALUES;
            let values = (n - first).min(INDEX_CHUNK_VALUES);
            *node = Node {
                count: 1,
                digest: chunk_hash(first, values, &mut buffer),
            };
        }
        Ok(())
    })?;
    let root = tree(leaves, threads, TreeHash::Sha256)?;
    let mut digest = hasher(b"pair-index", 40);
    digest.update((n as u64).to_le_bytes());
    digest.update(root);
    Ok(digest.finalize().into())
}

fn index_info(column: &IndexArray) -> Result<IndexInfo, EngineError> {
    let info = column.storage_info();
    if !matches!(info.itemsize, 4 | 8) {
        return Err(EngineError::Parameter(
            "logical index encoding requires native int32 or int64 items".into(),
        ));
    }
    u64::try_from(info.len).map_err(|_| overflow())?;
    Ok(info)
}

fn summarize_indices(
    column: &IndexArray,
    threads: usize,
) -> Result<(usize, [u8; 32]), EngineError> {
    let info = index_info(column)?;
    let digest = index_digest(info.len, threads, &|first, values, buffer| {
        for (offset, word) in buffer[..values * 8].chunks_exact_mut(8).enumerate() {
            word.copy_from_slice(&index_value(&info, first + offset).to_le_bytes());
        }
        index_chunk_digest(values, &buffer[..values * 8])
    })?;
    Ok((info.len, digest))
}

fn snapshot_indices(
    column: &IndexArray,
    threads: usize,
) -> Result<(usize, [u8; 32], Vec<u8>), EngineError> {
    let info = index_info(column)?;
    let size = info.len.checked_mul(8).ok_or_else(overflow)?;
    let mut data = Vec::new();
    reserve(&mut data, size, "index snapshot")?;
    data.resize(size, 0);
    // Disjoint byte chunks keep each whole signed value on one worker, without
    // a second full native-value array or a shared slice over foreign memory.
    let mut destinations = Vec::new();
    reserve(
        &mut destinations,
        info.len.div_ceil(INDEX_CHUNK_VALUES),
        "index snapshot ranges",
    )?;
    destinations.extend(data.chunks_mut(CHUNK_BYTES));
    let workers = worker_count(threads, destinations.len(), CHUNK_BYTES);
    sched::fill_chunks(&mut destinations, workers, &|start, output| {
        for (offset, destination) in output.iter_mut().enumerate() {
            let first = (start + offset) * INDEX_CHUNK_VALUES;
            for (index, word) in destination.chunks_exact_mut(8).enumerate() {
                word.copy_from_slice(&index_value(&info, first + index).to_le_bytes());
            }
        }
        Ok(())
    })?;
    drop(destinations);
    // The hash and Python bytes returned below use this same owned snapshot.
    let digest = index_digest(info.len, threads, &|first, values, _| {
        let start = first * 8;
        index_chunk_digest(values, &data[start..start + values * 8])
    })?;
    Ok((info.len, digest, data))
}

fn index_metadata<'py>(
    py: Python<'py>,
    n: usize,
    digest: &[u8; 32],
) -> PyResult<Bound<'py, PyDict>> {
    let result = PyDict::new(py);
    let hex = hex_digest(digest);
    result.set_item("n", n)?;
    result.set_item(
        "sha256",
        std::str::from_utf8(&hex).expect("ASCII hex digits"),
    )?;
    Ok(result)
}

fn bytes<'py>(py: Python<'py>, source: &[u8]) -> PyResult<Bound<'py, PyBytes>> {
    PyBytes::new_with(py, source.len(), |destination| {
        destination.copy_from_slice(source);
        Ok(())
    })
}

/// Bulk logical fingerprint. No floating-point operations, Python row callbacks
/// or full payload staging occur while the retained table is scanned.
#[pyfunction(signature = (table, threads=2))]
pub fn _native_input_digest<'py>(
    py: Python<'py>,
    table: &PySeqTable,
    threads: usize,
) -> PyResult<Bound<'py, PyDict>> {
    if threads == 0 {
        return Err(error(
            py,
            EngineError::Parameter("threads must be positive".into()),
        ));
    }
    let table = table
        .table
        .host()
        .map_err(|cause| error(py, cause))?
        .for_call();
    let summary = py
        .detach(move || summarize_table(&table, threads))
        .map_err(|cause| error(py, cause))?;
    metadata(py, &summary)
}

/// Own and fingerprint the same logical values in one native bulk operation.
/// Returned immutable bytes are independent of the retained input allocations.
#[pyfunction(signature = (table, threads=2))]
pub fn _native_input_snapshot<'py>(
    py: Python<'py>,
    table: &PySeqTable,
    threads: usize,
) -> PyResult<Bound<'py, PyDict>> {
    if threads == 0 {
        return Err(error(
            py,
            EngineError::Parameter("threads must be positive".into()),
        ));
    }
    let table = table
        .table
        .host()
        .map_err(|cause| error(py, cause))?
        .for_call();
    let snapshot = py
        .detach(move || snapshot(&table, threads))
        .map_err(|cause| error(py, cause))?;
    let result = metadata(py, &snapshot.summary)?;
    result.set_item("data", bytes(py, &snapshot.data)?)?;
    let byte_count = snapshot
        .offsets
        .len()
        .checked_mul(std::mem::size_of::<i64>())
        .filter(|&bytes| bytes <= isize::MAX as usize)
        .ok_or_else(|| error(py, overflow()))?;
    let offsets = PyBytes::new_with(py, byte_count, |destination| {
        for (offset, bytes) in snapshot.offsets.iter().zip(destination.chunks_exact_mut(8)) {
            bytes.copy_from_slice(&offset.to_le_bytes());
        }
        Ok(())
    })?;
    result.set_item("offsets", offsets)?;
    result.set_item("validity", bytes(py, &snapshot.validity)?)?;
    result.set_item("python_rows", bytes(py, &snapshot.python_rows)?)?;
    Ok(result)
}

#[pyfunction(signature = (column, threads=2))]
pub fn _native_index_digest<'py>(
    py: Python<'py>,
    column: &PyIndexColumn,
    threads: usize,
) -> PyResult<Bound<'py, PyDict>> {
    if threads == 0 {
        return Err(error(
            py,
            EngineError::Parameter("threads must be positive".into()),
        ));
    }
    let column = column.column.clone();
    let (n, digest) = py
        .detach(move || summarize_indices(&column, threads))
        .map_err(|cause| error(py, cause))?;
    index_metadata(py, n, &digest)
}

#[pyfunction(signature = (column, threads=2))]
pub fn _native_index_snapshot<'py>(
    py: Python<'py>,
    column: &PyIndexColumn,
    threads: usize,
) -> PyResult<Bound<'py, PyDict>> {
    if threads == 0 {
        return Err(error(
            py,
            EngineError::Parameter("threads must be positive".into()),
        ));
    }
    let column = column.column.clone();
    let (n, digest, data) = py
        .detach(move || snapshot_indices(&column, threads))
        .map_err(|cause| error(py, cause))?;
    let result = index_metadata(py, n, &digest)?;
    result.set_item("data", bytes(py, &data)?)?;
    Ok(result)
}

pub fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(_native_input_digest, module)?)?;
    module.add_function(wrap_pyfunction!(_native_input_snapshot, module)?)?;
    module.add_function(wrap_pyfunction!(_native_index_digest, module)?)?;
    module.add_function(wrap_pyfunction!(_native_index_snapshot, module)?)?;
    Ok(())
}
