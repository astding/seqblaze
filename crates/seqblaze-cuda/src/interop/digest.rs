//! The same logical codecs as CPU provenance: merkle-blake3-v1 for sequence
//! tables (D128) and merkle-sha256-v1 for indices. Device data is read in
//! parallel <=64KiB leaves (one thread per BLAKE3 chunk of a table leaf's
//! frame); only the final 32-byte digest leaves the GPU.
use cudarc::driver::{CudaFunction, CudaSlice, DeviceRepr, LaunchConfig, PushKernelArg};
use seqblaze_core::validate::EngineError;

use super::validate::{ControlBlock, DigestSlot, Retained as Pending};
use super::{DeviceColumn, DeviceDType, DeviceLease, DeviceRows, ValidatedDeviceRows};
use crate::{device::guard, session::failure_ordinal};

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub(crate) struct Digest {
    byte: [u8; 32],
}
// SAFETY: exactly the 32 initialized bytes of kernels/device_digest.cuh's SbDigest.
// The same 32-byte slots also hold its SbCv chunk chaining values, which only
// device kernels read.
unsafe impl DeviceRepr for Digest {}
const _: () = assert!(std::mem::size_of::<Digest>() == 32);

/// Payload bytes of a merkle-blake3-v1 leaf (8192 lengths or 65536 bytes).
const LEAF_BYTES: u64 = 65_536;
/// BLAKE3 chunk slots per table leaf: a leaf frame (head of 40 or 38 bytes plus
/// at most 65536 payload bytes) has at most 65 chunks (SB_B3_LEAF_CHUNKS).
const LEAF_CHUNKS: u64 = 65;
const LENGTHS_HEAD: u64 = 40;
const BYTES_HEAD: u64 = 38;

fn failed(lease: &DeviceLease, what: &str, error: cudarc::driver::DriverError) -> EngineError {
    failure_ordinal(lease.device(), what, error)
}
// Raw launches must retain scratch and source owners even when a stream error
// prevents proving completion: the shared retained guard (a failed retirement
// deliberately retains storage). In a control-block batch the scratch stays in
// the block until its read (per-call cost S6).
pub(super) fn allocate<T: DeviceRepr>(
    lease: &DeviceLease,
    len: usize,
) -> Result<Pending<crate::arena::Scratch<T>>, EngineError> {
    if len
        .checked_mul(std::mem::size_of::<T>())
        .is_none_or(|bytes| bytes > isize::MAX as usize)
    {
        return Err(EngineError::Device(
            "device input digest allocation exceeds platform limits".into(),
        ));
    }
    // SAFETY: all read elements are written by a prior kernel on this stream.
    unsafe { lease.execution().scratch(len.max(1)) }
        .map(|value| Pending::new(lease, value))
        .map_err(|e| failed(lease, "allocating logical device digests", e))
}
fn grid(items: u64) -> LaunchConfig {
    LaunchConfig {
        grid_dim: (items.div_ceil(128).clamp(1, 65535) as u32, 1, 1),
        block_dim: (128, 1, 1),
        shared_mem_bytes: 0,
    }
}

/// Reduce an ordered leaf tree by levels; `level` is the table (B) or index
/// (SHA-256) node kernel.
fn finish_tree(
    lease: &DeviceLease,
    level: &CudaFunction,
    values: &mut CudaSlice<Digest>,
    count: u64,
) -> Result<(), EngineError> {
    let mut width = 1u64;
    while width < count {
        let mut launch = lease.execution().digest_builder(level);
        launch.slice_arg_mut(&mut *values).arg(&count).arg(&width);
        // SAFETY: nodes at each level read disjoint already-initialized children;
        // the next level begins only after this stream-ordered launch.
        unsafe {
            launch.launch(LaunchConfig {
                grid_dim: (count.div_ceil(128 * width).clamp(1, 65535) as u32, 1, 1),
                block_dim: (128, 1, 1),
                shared_mem_bytes: 0,
            })
        }
        .map_err(|e| failed(lease, "reducing an ordered device digest tree", e))?;
        width = width.saturating_mul(128);
    }
    Ok(())
}

/// Fold each leaf's BLAKE3 chunk chaining values into its leaf digest.
fn leaf_roots(
    lease: &DeviceLease,
    chunks: &CudaSlice<Digest>,
    stream: u64,
    head: u64,
    leaves: &mut CudaSlice<Digest>,
    count: u64,
) -> Result<(), EngineError> {
    let mut launch = lease
        .execution()
        .digest_builder(&lease.device().0.kernels.digest_leaf_roots);
    launch
        .slice_arg(chunks)
        .arg(&stream)
        .arg(&head)
        .slice_arg_mut(leaves);
    // SAFETY: the chunk pass initialized every chunk slot that a leaf's frame
    // uses, on this stream; each leaf folds only its own slots.
    unsafe {
        launch.launch(LaunchConfig {
            grid_dim: (count.clamp(1, 65535) as u32, 1, 1),
            block_dim: (128, 1, 1),
            shared_mem_bytes: 0,
        })
    }
    .map_err(|e| failed(lease, "folding logical device leaf digests", e))
}

pub fn device_sequence_digest(
    lease: &DeviceLease,
    rows: &ValidatedDeviceRows,
) -> Result<[u8; 32], EngineError> {
    device_sequence_digest_of(lease, rows, rows.bound.rows)
}

/// `table` can also address a completed owned D2D snapshot. Its logical row
/// lengths/null facts must be exactly those of `rows` (no historical host copy).
pub(crate) fn device_sequence_digest_of(
    lease: &DeviceLease,
    rows: &ValidatedDeviceRows,
    table: DeviceRows,
) -> Result<[u8; 32], EngineError> {
    guard(|| {
        rows.bound.ensure_lease(lease)?;
        if rows.summary.span_row != u64::MAX {
            return Err(EngineError::Parameter(format!(
                "device sequence row {} has invalid offsets or length",
                rows.summary.span_row
            )));
        }
        let block = ControlBlock::new(lease)?;
        // Valid rows' bytes; the validation summary already holds this scalar.
        enqueue_sequence_digest_of(
            lease,
            rows,
            table,
            rows.summary.total,
            &block,
            DigestSlot::A,
        )?;
        let words = block.read_as(lease, "reading a logical device input digest")?;
        Ok(words.digest(DigestSlot::A))
    })
}

/// Enqueue a table's logical digest into `slot` of the control block (S6).
/// `total` is the byte total of its valid rows: the validation summary's, or
/// for a host-origin table the host's, so a batch needs no read first. The
/// rows' facts must come from validation enqueued earlier on this stream.
pub fn enqueue_sequence_digest(
    lease: &DeviceLease,
    rows: &ValidatedDeviceRows,
    total: u64,
    block: &ControlBlock,
    slot: DigestSlot,
) -> Result<(), EngineError> {
    enqueue_sequence_digest_of(lease, rows, rows.bound.rows, total, block, slot)
}

/// [`enqueue_sequence_digest`] of `table`, which can also address a completed
/// owned D2D snapshot with the same logical row facts.
fn enqueue_sequence_digest_of(
    lease: &DeviceLease,
    rows: &ValidatedDeviceRows,
    table: DeviceRows,
    total: u64,
    block: &ControlBlock,
    slot: DigestSlot,
) -> Result<(), EngineError> {
    guard(|| {
        rows.bound.ensure_lease(lease)?;
        let n = usize::try_from(table.rows).map_err(|_| {
            EngineError::Device("device digest row count exceeds the platform".into())
        })?;
        let limits = || EngineError::Device("device digest leaf count exceeds the platform".into());
        let length_stream = table.rows.checked_mul(8).ok_or_else(limits)?;
        let length_chunks = length_stream.div_ceil(LEAF_BYTES);
        let byte_chunks = total.div_ceil(LEAF_BYTES);
        let leaves = |count: u64| usize::try_from(count).map_err(|_| limits());
        let chunk_slots = |count: u64| {
            count
                .checked_mul(LEAF_CHUNKS)
                .ok_or_else(limits)
                .and_then(leaves)
        };
        if n <= 8192 && total <= LEAF_BYTES {
            let offsets = Pending::new(
                lease,
                super::validate::scan_with::<3>(lease, &rows.lengths, n, Some(block))?,
            );
            let length_hashes_address = rows
                .length_hashes
                .as_ref()
                .map_or(0, |hashes| crate::session::ptr(hashes, lease.stream()));
            let output = block.digest_at(slot);
            let mut launch = lease
                .execution()
                .digest_builder(&lease.device().0.kernels.digest_sequence_small);
            launch
                .arg(&table)
                .slice_arg(&*rows.facts)
                .slice_arg(&*offsets)
                .arg(&total)
                .arg(&length_hashes_address)
                .arg(&output);
            // SAFETY: validated facts/lengths and their exact prefix precede
            // this one-block, one-leaf-per-stream digest on the same stream.
            unsafe { launch.launch(grid(1)) }
                .map_err(|e| failed(lease, "finalizing a logical device sequence digest", e))?;
            block.keep(offsets);
            return Ok(());
        }
        let mut length_leaves = allocate::<Digest>(lease, leaves(length_chunks)?)?;
        let mut byte_leaves = allocate::<Digest>(lease, leaves(byte_chunks)?)?;
        let output = block.digest_at(slot);
        let kernels = &lease.device().0.kernels;
        if n != 0 {
            // Large possible digests prepared their length hashes in init;
            // a descriptor-proved single-leaf table takes the fused small path.
            let chunks = rows
                .length_hashes
                .as_ref()
                .expect("multi-leaf length hashes from validation init");
            leaf_roots(
                lease,
                chunks,
                length_stream,
                LENGTHS_HEAD,
                &mut length_leaves,
                length_chunks,
            )?;
            finish_tree(
                lease,
                &kernels.digest_table_tree_level,
                &mut length_leaves,
                length_chunks,
            )?;
        }
        if total != 0 {
            let offsets = Pending::new(
                lease,
                super::validate::scan_with::<3>(lease, &rows.lengths, n, Some(block))?,
            );
            let mut chunks = allocate::<Digest>(lease, chunk_slots(byte_chunks)?)?;
            let mut launch = lease.execution().digest_builder(&kernels.digest_row_bytes);
            launch
                .arg(&table)
                .slice_arg(&*offsets)
                .arg(&total)
                .slice_arg_mut(&mut *chunks);
            // SAFETY: validation established every logical source subspan, and
            // offsets is the exact prefix of those lengths, ending at `total`.
            unsafe { launch.launch(grid(byte_chunks * LEAF_CHUNKS)) }
                .map_err(|e| failed(lease, "hashing logical device sequence bytes", e))?;
            leaf_roots(
                lease,
                &chunks,
                total,
                BYTES_HEAD,
                &mut byte_leaves,
                byte_chunks,
            )?;
            finish_tree(
                lease,
                &kernels.digest_table_tree_level,
                &mut byte_leaves,
                byte_chunks,
            )?;
            block.keep(offsets);
            block.keep(chunks);
        }
        let mut launch = lease
            .execution()
            .digest_builder(&kernels.digest_sequence_table);
        launch
            .slice_arg(&*length_leaves)
            .arg(&length_chunks)
            .slice_arg(&*byte_leaves)
            .arg(&byte_chunks)
            .arg(&table.rows)
            .arg(&total)
            .arg(&output);
        // SAFETY: an empty stream's root is the domain-separated empty tree and
        // reads no leaf; other roots were initialized by the trees above. The
        // root goes to the block's digest slot.
        unsafe { launch.launch(grid(1)) }
            .map_err(|e| failed(lease, "finalizing a logical device sequence digest", e))?;
        block.keep(length_leaves);
        block.keep(byte_leaves);
        Ok(())
    })
}

/// Hash an immutable normalized index column without exposing or downloading it.
/// The input remains strongly retained until this consumer stream has completed.
pub fn device_column_index_digest(
    lease: &DeviceLease,
    values: &DeviceColumn,
) -> Result<[u8; 32], EngineError> {
    guard(|| {
        let block = ControlBlock::new(lease)?;
        enqueue_index_digest(lease, values, &block, DigestSlot::I)?;
        let words = block.read_as(lease, "reading a logical device index digest")?;
        Ok(words.digest(DigestSlot::I))
    })
}

/// Enqueue an index column's digest into `slot` of the control block (S6).
/// It first waits for the column's own stream, as before; a batch calls it
/// before enqueuing other work, so that wait finds the lease's work done.
pub fn enqueue_index_digest(
    lease: &DeviceLease,
    values: &DeviceColumn,
    block: &ControlBlock,
    slot: DigestSlot,
) -> Result<(), EngineError> {
    guard(|| {
        if values.dtype() != DeviceDType::I64 || values.shape().len() != 1 {
            return Err(EngineError::Parameter(
                "device index digest requires a one-dimensional int64 column".into(),
            ));
        }
        if values.ordinal() != lease.device().info().ordinal {
            return Err(EngineError::Device(
                "device index digest belongs to a different CUDA device".into(),
            ));
        }
        // Ordering the immutable source before reading on the hashing stream is
        // sufficient; no payload or per-index metadata crosses to the host.
        values.synchronize()?;
        let retained = Pending::new(lease, values.clone());
        let n = retained.shape()[0];
        let address = retained.address();
        let count = n as u64;
        let chunks = n.div_ceil(8192);
        let mut leaves = allocate::<Digest>(lease, chunks)?;
        let output = block.digest_at(slot);
        if chunks != 0 {
            let mut launch = lease
                .execution()
                .digest_builder(&lease.device().0.kernels.digest_index_chunks);
            launch.arg(&address).arg(&count).slice_arg_mut(&mut *leaves);
            // SAFETY: the retained normalized source has n initialized I64 words.
            unsafe { launch.launch(grid(chunks as u64)) }
                .map_err(|e| failed(lease, "hashing logical device index chunks", e))?;
            finish_tree(
                lease,
                &lease.device().0.kernels.digest_tree_level,
                &mut leaves,
                chunks as u64,
            )?;
        }
        let mut launch = lease
            .execution()
            .digest_builder(&lease.device().0.kernels.digest_pair_index);
        launch.slice_arg(&*leaves).arg(&count).arg(&output);
        // SAFETY: an empty root reads no leaf; other roots were initialized
        // above. The root goes to the block's digest slot.
        unsafe { launch.launch(grid(1)) }
            .map_err(|e| failed(lease, "finalizing a logical device index digest", e))?;
        block.keep(leaves);
        block.keep(retained);
        Ok(())
    })
}
