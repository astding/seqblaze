//! Explicit repro serialization. Copy the current logical input to independently
//! owned device buffers first; only that completed snapshot is copied to host.
use cudarc::driver::{LaunchConfig, PushKernelArg};
use seqblaze_core::validate::EngineError;

use super::{
    digest::device_sequence_digest_of,
    validate::{allocate, scan, Retained},
};
use super::{DeviceLease, DeviceRows, ValidatedDeviceRows};
use crate::{device::guard, session::failure_ordinal};

pub struct DeviceInputSnapshot {
    pub data: Vec<u8>,
    pub offsets: Vec<u64>,
    pub validity: Vec<u8>,
    pub digest: [u8; 32],
}
fn host<T: Copy + Default>(len: usize) -> Result<Vec<T>, EngineError> {
    let mut values = Vec::new();
    values.try_reserve_exact(len).map_err(|_| {
        EngineError::Device("unable to reserve explicit device input snapshot".into())
    })?;
    values.resize(len, T::default());
    Ok(values)
}

pub fn device_input_snapshot(
    lease: &DeviceLease,
    rows: &ValidatedDeviceRows,
) -> Result<DeviceInputSnapshot, EngineError> {
    guard(|| {
        rows.bound.ensure_lease(lease)?;
        let fail = |what: &str, error| failure_ordinal(lease.device(), what, error);
        if rows.summary.span_row != u64::MAX {
            return Err(EngineError::Parameter(
                "cannot snapshot an invalid device row span".into(),
            ));
        }
        let n = usize::try_from(rows.bound.rows.rows).map_err(|_| {
            EngineError::Device("device snapshot row count exceeds platform limits".into())
        })?;
        let total = usize::try_from(rows.summary.total).map_err(|_| {
            EngineError::Device("device snapshot byte count exceeds platform limits".into())
        })?;
        let mut lengths = Retained::new(lease, allocate::<u64>(lease, n)?);
        let mut validity = Retained::new(lease, allocate::<u8>(lease, n)?);
        let mut data = Retained::new(lease, allocate::<u8>(lease, total)?);
        let count = n as u64;
        let kernels = &lease.device().0.kernels;
        let grid = |items: u64| LaunchConfig {
            grid_dim: (items.div_ceil(128).clamp(1, 65535) as u32, 1, 1),
            block_dim: (128, 1, 1),
            shared_mem_bytes: 0,
        };
        if n != 0 {
            let mut launch = lease
                .execution()
                .launch_builder(&kernels.input_snapshot_lengths);
            launch
                .slice_arg(&*rows.facts)
                .arg(&count)
                .slice_arg_mut(&mut *lengths)
                .slice_arg_mut(&mut *validity);
            // SAFETY: every source fact is initialized and n outputs are reserved.
            unsafe { launch.launch(grid(count)) }
                .map_err(|e| fail("packing device snapshot row metadata", e))?;
        }
        let offsets = Retained::new(lease, scan(lease, &lengths, n)?);
        let snapshot_chunks = if !rows.chunks_scanned {
            Some(Retained::new(lease, scan(lease, &rows.chunks, n)?))
        } else {
            None
        };
        if rows.summary.total != 0 {
            let mut launch = lease
                .execution()
                .launch_builder(&kernels.input_snapshot_bytes);
            let table = rows.bound.rows;
            launch
                .arg(&table)
                .slice_arg(&*rows.facts)
                .slice_arg(&**snapshot_chunks.as_ref().unwrap_or(&rows.chunks))
                .slice_arg(&*offsets)
                .slice_arg_mut(&mut *data);
            // SAFETY: the disjoint validated logical chunks exactly cover total
            // bytes of this fresh allocation; null rows have no chunk. The
            // chunk count is the prefix's device-resident end (per-call cost S6).
            let config = LaunchConfig {
                grid_dim: (super::validate::alphabet_grid(rows) as u32, 1, 1),
                block_dim: (128, 1, 1),
                shared_mem_bytes: 0,
            };
            unsafe { launch.launch(config) }
                .map_err(|e| fail("copying the logical input device snapshot", e))?;
        }
        let table = DeviceRows {
            data: crate::session::ptr(&data, lease.stream()),
            rows: count,
            layout: 0,
            offsets: crate::session::ptr(&offsets, lease.stream()),
            offset_width: 8,
            offset_stride: 8,
            column_stride: 1,
            columns: rows.summary.total,
            ..DeviceRows::default()
        };
        // The digest reads the just-created owned snapshot, with the immutable
        // lengths/null facts that created it. It cannot accidentally hash a
        // later mutation of the caller's borrowed device allocation.
        let digest = device_sequence_digest_of(lease, rows, table)?;
        let mut host_data = host::<u8>(total)?;
        let mut host_offsets = host::<u64>(n.checked_add(1).ok_or_else(|| {
            EngineError::Device("device snapshot offset count overflows".into())
        })?)?;
        let mut host_validity = host::<u8>(n)?;
        lease
            .execution()
            .memcpy_dtoh(&data.slice(0..total), &mut host_data[..])
            .map_err(|e| fail("explicitly serializing device input bytes", e))?;
        lease
            .execution()
            .memcpy_dtoh(&**offsets, &mut host_offsets[..])
            .map_err(|e| fail("explicitly serializing device input offsets", e))?;
        lease
            .execution()
            .memcpy_dtoh(&validity.slice(0..n), &mut host_validity[..])
            .map_err(|e| fail("explicitly serializing device input validity", e))?;
        lease.synchronize()?;
        Ok(DeviceInputSnapshot {
            data: host_data,
            offsets: host_offsets,
            validity: host_validity,
            digest,
        })
    })
}
