//! Resident K3's finite-map finalize and exact dense-integer row accumulator.
//! No pair term/value is downloaded. Capacity retries copy one control scalar,
//! discard the old arena, and replay the producer with larger row capacities.
#[cfg(feature = "cuda")]
use seqblaze_core::validate::EngineError;

pub const MAX_ROW_LIMBS: u64 = 1 << 18;
#[cfg(feature = "cuda")]
const INITIAL_ROW_LIMBS: u64 = 64;

/// Host-built constants for the specified G1-G8 map, not an arbitrary ln oracle.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct PfFinalizeConstants {
    pub table: [u64; 128],
    pub l2: u64,
    pub c_t: u64,
    pub kappa: u64,
    pub b3: u64,
    pub b5: u64,
    pub b7: u64,
    pub loop_max: u64,
}
#[repr(C)]
#[derive(Clone, Copy)]
pub struct PfTermView {
    pub mantissa: u64,
    pub exponent: u64,
    pub stride_bytes: u64,
    pub count: u64,
}
#[repr(C)]
#[derive(Clone, Copy)]
struct ExactRow {
    beta: i64,
    used: u64,
    capacity: u64,
    offset: u64,
    partners: u64,
    needed: u64,
    status: u64,
    head: u64,
}
#[repr(C)]
#[derive(Clone, Copy)]
pub struct PfTables(pub [u64; 6]);
#[repr(C)]
#[derive(Clone, Copy)]
pub struct PfSpace(pub [u64; 13]);
const _: () = {
    assert!(std::mem::size_of::<PfFinalizeConstants>() == 1080);
    assert!(std::mem::offset_of!(PfFinalizeConstants, l2) == 1024);
    assert!(std::mem::offset_of!(PfFinalizeConstants, loop_max) == 1072);
    assert!(std::mem::size_of::<PfTermView>() == 32);
    assert!(std::mem::size_of::<ExactRow>() == 64);
    assert!(std::mem::offset_of!(PfTermView, mantissa) == 0);
    assert!(std::mem::offset_of!(PfTermView, exponent) == 8);
    assert!(std::mem::offset_of!(PfTermView, stride_bytes) == 16);
    assert!(std::mem::offset_of!(PfTermView, count) == 24);
    assert!(std::mem::offset_of!(ExactRow, beta) == 0);
    assert!(std::mem::offset_of!(ExactRow, used) == 8);
    assert!(std::mem::offset_of!(ExactRow, capacity) == 16);
    assert!(std::mem::offset_of!(ExactRow, offset) == 24);
    assert!(std::mem::offset_of!(ExactRow, partners) == 32);
    assert!(std::mem::offset_of!(ExactRow, needed) == 40);
    assert!(std::mem::offset_of!(ExactRow, status) == 48);
    assert!(std::mem::offset_of!(ExactRow, head) == 56);
    assert!(std::mem::size_of::<PfTables>() == 48);
    assert!(std::mem::size_of::<PfSpace>() == 104);
};
#[cfg(feature = "cuda")]
fn internal(message: &str) -> EngineError {
    EngineError::Device(format!(
        "internal error: resident duplex pf {message}; no result was published"
    ))
}
#[cfg(feature = "cuda")]
fn refused() -> EngineError {
    EngineError::Device(
        "unable to reserve the exact row sum (width); no result was published".into(),
    )
}
#[cfg(feature = "cuda")]
fn status(code: u32) -> Result<RowsProgress, EngineError> {
    match code {
        0 => Ok(RowsProgress::Ready),
        2 => Ok(RowsProgress::Replay),
        3 => Err(EngineError::Device("unable to reserve the exact row sum of lse_dg_kcal (width above 262144 limbs); no result was published".into())),
        _ => Err(internal("received an invalid partition-function value or row descriptor")),
    }
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RowsProgress {
    Ready,
    /// Stop consuming, call grow_reset(), reset the producer, and replay it.
    /// This is not a public failure and must never be published as one.
    Replay,
}

#[cfg(feature = "cuda")]
mod driver {
    use super::*;
    use crate::{
        device::{guard, Device},
        interop::{DeviceColumn, DeviceDType, DeviceLease},
        session::{failure_ordinal, ptr},
    };
    use cudarc::driver::{
        CudaFunction, CudaSlice, CudaStream, DeviceRepr, LaunchConfig, PushKernelArg,
    };
    use seqblaze_core::cancel::{self, Site};
    use std::sync::Arc;

    // SAFETY: plain repr(C) integer records, mirrored and size-checked in pf_resident.cuh.
    unsafe impl DeviceRepr for PfFinalizeConstants {}
    // SAFETY: four u64 fields in the CUDA SbPfTerms order.
    unsafe impl DeviceRepr for PfTermView {}
    // SAFETY: eight 64-bit integer fields in the CUDA SbPfExactRow order.
    unsafe impl DeviceRepr for ExactRow {}
    // SAFETY: same six u64 words as SbTables, including the two optional views.
    unsafe impl DeviceRepr for PfTables {}
    // SAFETY: same thirteen u64 words as SbSpace, unchanged by resident K3.
    unsafe impl DeviceRepr for PfSpace {}

    fn grid(
        lease: &DeviceLease,
        function: &CudaFunction,
        n: u64,
    ) -> Result<LaunchConfig, EngineError> {
        let limit = function
            .max_threads_per_block()
            .map_err(|e| fail(lease, "reading the PF kernel block limit", e))?;
        let block = u32::try_from(limit)
            .ok()
            .filter(|&n| n != 0)
            .ok_or_else(|| internal("kernel has an invalid block limit"))?
            .min(128);
        Ok(LaunchConfig {
            grid_dim: (n.div_ceil(u64::from(block)).clamp(1, 65535) as u32, 1, 1),
            block_dim: (block, 1, 1),
            shared_mem_bytes: 0,
        })
    }
    fn fail(lease: &DeviceLease, operation: &str, e: cudarc::driver::DriverError) -> EngineError {
        failure_ordinal(lease.device(), operation, e)
    }
    fn column(column: &DeviceColumn, dtype: DeviceDType, n: usize) -> Result<(), EngineError> {
        if column.dtype() != dtype || column.shape() != [n] {
            return Err(internal("received inconsistent resident column geometry"));
        }
        Ok(())
    }
    fn ready(value: &DeviceColumn) -> Result<(), EngineError> {
        if let Err(error) = value.synchronize() {
            // A failed producer stream must not release storage still used by raw work.
            std::mem::forget(value.clone());
            return Err(error);
        }
        Ok(())
    }
    /// The DP's native interleaved records or two independently owned columns.
    pub enum PfTerms<'a> {
        Interleaved(&'a DeviceColumn),
        Split {
            mantissa: &'a DeviceColumn,
            exponent: &'a DeviceColumn,
        },
    }
    impl PfTerms<'_> {
        fn view(&self, lease: &DeviceLease) -> Result<PfTermView, EngineError> {
            let ordinal = lease.device().info().ordinal;
            match self {
                Self::Interleaved(value) => {
                    if value.ordinal() != ordinal
                        || value.dtype() != DeviceDType::U64
                        || value.shape().len() != 2
                        || value.shape()[1] != 2
                    {
                        return Err(internal("requires interleaved U64[n,2] F53 records"));
                    }
                    ready(value)?;
                    Ok(PfTermView {
                        mantissa: value.address(),
                        exponent: value.address() + 8,
                        stride_bytes: 16,
                        count: value.shape()[0] as u64,
                    })
                }
                Self::Split { mantissa, exponent } => {
                    let n = mantissa.shape().first().copied().unwrap_or(0);
                    column(mantissa, DeviceDType::U64, n)?;
                    column(exponent, DeviceDType::I64, n)?;
                    if mantissa.ordinal() != ordinal || exponent.ordinal() != ordinal {
                        return Err(internal("received F53 columns on a different device"));
                    }
                    ready(mantissa)?;
                    ready(exponent)?;
                    Ok(PfTermView {
                        mantissa: mantissa.address(),
                        exponent: exponent.address(),
                        stride_bytes: 8,
                        count: n as u64,
                    })
                }
            }
        }
        fn keep(&self) -> Vec<DeviceColumn> {
            match self {
                Self::Interleaved(value) => vec![(*value).clone()],
                Self::Split { mantissa, exponent } => {
                    vec![(*mantissa).clone(), (*exponent).clone()]
                }
            }
        }
    }
    /// A failed synchronization must not release source owners for raw-pointer work.
    fn retire<T>(
        lease: &DeviceLease,
        keep: Vec<DeviceColumn>,
        result: Result<T, EngineError>,
    ) -> Result<T, EngineError> {
        if let Err(error) = lease.synchronize() {
            std::mem::forget(keep);
            return Err(error);
        }
        result
    }
    /// Raw-pointer kernels cannot register cudarc borrow events. Even a temporary
    /// control/list buffer must survive failed synchronization on an early exit.
    struct PendingSlice<T> {
        slice: Option<CudaSlice<T>>,
        stream: Arc<CudaStream>,
        device: Device,
    }
    impl<T> PendingSlice<T> {
        fn new(lease: &DeviceLease, slice: CudaSlice<T>) -> Self {
            Self {
                slice: Some(slice),
                stream: Arc::clone(lease.stream()),
                device: lease.device().clone(),
            }
        }
        fn get(&self) -> &CudaSlice<T> {
            self.slice.as_ref().unwrap()
        }
    }
    impl<T> Drop for PendingSlice<T> {
        fn drop(&mut self) {
            if guard(|| {
                crate::interop::execution_stream::retire_in_drop(&self.stream)
                    .map_err(|e| failure_ordinal(&self.device, "retiring PF scratch", e))
            })
            .is_err()
            {
                if let Some(slice) = self.slice.take() {
                    std::mem::forget(slice);
                }
            }
        }
    }
    fn zero_status(lease: &DeviceLease) -> Result<PendingSlice<u32>, EngineError> {
        // SAFETY: every control byte is initialized by the following same-stream copy.
        let mut out = unsafe { lease.execution().alloc::<u32>(1) }
            .map_err(|e| fail(lease, "allocating PF control", e))?;
        lease
            .execution()
            .fill(&mut out, 1, 0)
            .map_err(|e| fail(lease, "initializing PF control", e))?;
        Ok(PendingSlice::new(lease, out))
    }
    fn read_status(
        lease: &DeviceLease,
        control: &PendingSlice<u32>,
    ) -> Result<RowsProgress, EngineError> {
        let mut code = [0u32];
        lease
            .execution()
            .memcpy_dtoh(control.get(), &mut code)
            .map_err(|e| fail(lease, "reading PF control", e))?;
        lease.synchronize()?;
        status(code[0])
    }
    pub struct PfFinalized {
        pub keys: DeviceColumn,
        pub dg_ens_kcal: DeviceColumn,
        pub dg_ens_err_kcal: DeviceColumn,
    }
    /// Complete on-device G1-G8/E1-E4. Only the four-byte validity control is downloaded.
    pub fn finalize(
        lease: &DeviceLease,
        spec: PfFinalizeConstants,
        terms: PfTerms<'_>,
        tables: PfTables,
        space: PfSpace,
    ) -> Result<PfFinalized, EngineError> {
        let keep = terms.keep();
        let result = guard(|| {
            let view = terms.view(lease)?;
            let n = usize::try_from(view.count).map_err(|_| internal("term count overflows"))?;
            let keys = lease.output(DeviceDType::I64, vec![n])?;
            let dg = lease.output(DeviceDType::F64, vec![n])?;
            let error = lease.output(DeviceDType::F64, vec![n])?;
            let control = zero_status(lease)?;
            if n != 0 {
                let keys_ptr = keys.address();
                let dg_ptr = dg.address();
                let error_ptr = error.address();
                let control_ptr = ptr(control.get(), lease.stream());
                let mut call = lease
                    .execution()
                    .launch_builder(&lease.device().0.kernels.pf_resident_finalize);
                call.arg(&spec)
                    .arg(&view)
                    .arg(&tables)
                    .arg(&space)
                    .arg(&keys_ptr)
                    .arg(&dg_ptr)
                    .arg(&error_ptr)
                    .arg(&control_ptr);
                // SAFETY: caller's bound tables/space and owned term columns cover this chunk;
                // outputs are distinct full-sized allocations and sources survive synchronization.
                unsafe {
                    call.launch(grid(
                        lease,
                        &lease.device().0.kernels.pf_resident_finalize,
                        view.count,
                    )?)
                }
                .map_err(|e| fail(lease, "finalizing resident PF", e))?;
            }
            if read_status(lease, &control)? != RowsProgress::Ready {
                return Err(internal("unexpected finalize capacity request"));
            }
            // SAFETY: successful kernel wrote every key; empty columns need no write.
            let keys = unsafe { keys.finish()? };
            // SAFETY: the same kernel wrote every dg bit pattern.
            let dg_ens_kcal = unsafe { dg.finish()? };
            // SAFETY: the same kernel wrote every error-bound bit pattern.
            let dg_ens_err_kcal = unsafe { error.finish()? };
            Ok(PfFinalized {
                keys,
                dg_ens_kcal,
                dg_ens_err_kcal,
            })
        });
        retire(lease, keep, result)
    }

    /// Mutable private state. Outputs are published only by finish(), after the
    /// last successful replay. Row limbs never leave the device.
    pub struct PfExactRows {
        rows: Option<PendingSlice<ExactRow>>,
        arena: Option<PendingSlice<u64>>,
        control: Option<PendingSlice<u32>>,
        count: usize,
        stream: Arc<CudaStream>,
        device: Device,
    }
    impl PfExactRows {
        pub fn new(lease: &DeviceLease, count: usize) -> Result<Self, EngineError> {
            guard(|| {
                let mut host = Vec::new();
                seqblaze_core::detail::try_reserve_exact(&mut host, count, "exact PF row headers")?;
                for row in 0..count {
                    let offset = (row as u64)
                        .checked_mul(INITIAL_ROW_LIMBS)
                        .ok_or_else(refused)?;
                    host.push(ExactRow {
                        beta: 0,
                        used: 0,
                        capacity: INITIAL_ROW_LIMBS,
                        offset,
                        partners: 0,
                        needed: 0,
                        status: 0,
                        head: u64::MAX,
                    });
                }
                let words = count
                    .checked_mul(INITIAL_ROW_LIMBS as usize)
                    .filter(|&n| n <= isize::MAX as usize / 8)
                    .ok_or_else(refused)?;
                // SAFETY: row headers are initialized below; arena reads only limbs already written.
                let mut rows = unsafe { lease.execution().alloc::<ExactRow>(count.max(1)) }
                    .map_err(|_| refused())?;
                if count != 0 {
                    // The staging copy is reserved before the copy is enqueued
                    // (host-allocation-failures §12.6 rule 4).
                    let staging =
                        crate::interop::execution_stream::upload_staging::<ExactRow>(host.len())?;
                    lease
                        .execution()
                        .memcpy_htod_staged(&host, &mut rows, staging)
                        .map_err(|e| fail(lease, "initializing exact PF rows", e))?;
                }
                let rows = PendingSlice::new(lease, rows);
                // SAFETY: an empty row has used=0; add/lower writes every newly live limb.
                let arena = unsafe { lease.execution().alloc::<u64>(words.max(1)) }
                    .map_err(|_| refused())?;
                let arena = PendingSlice::new(lease, arena);
                let control = zero_status(lease)?;
                lease.synchronize()?;
                Ok(Self {
                    rows: Some(rows),
                    arena: Some(arena),
                    control: Some(control),
                    count,
                    stream: Arc::clone(lease.stream()),
                    device: lease.device().clone(),
                })
            })
        }
        fn check_lease(&self, lease: &DeviceLease) -> Result<(), EngineError> {
            if !Arc::ptr_eq(&self.stream, lease.stream()) {
                return Err(internal("row workspace changed its consumer stream"));
            }
            if self.arena.is_none() {
                return Err(internal("row arena is unavailable after a refused growth"));
            }
            Ok(())
        }
        /// keys must be unmasked pair keys. first=None means canonical unique work;
        /// symmetric within sets mirror=true (diagonal is added exactly once).
        #[allow(clippy::too_many_arguments)]
        pub fn consume(
            &mut self,
            lease: &DeviceLease,
            terms: PfTerms<'_>,
            keys: &DeviceColumn,
            space: PfSpace,
            first: Option<&DeviceColumn>,
            mirror: bool,
            cutoff: Option<i64>,
        ) -> Result<RowsProgress, EngineError> {
            let mut keep = terms.keep();
            keep.push(keys.clone());
            if let Some(first) = first {
                keep.push(first.clone());
            }
            let result = guard(|| {
                self.check_lease(lease)?;
                let view = terms.view(lease)?;
                column(keys, DeviceDType::I64, view.count as usize)?;
                if keys.ordinal() != self.device.info().ordinal {
                    return Err(internal("row keys changed CUDA device"));
                }
                ready(keys)?;
                if let Some(first) = first {
                    column(first, DeviceDType::Bool, view.count as usize)?;
                    if first.ordinal() != self.device.info().ordinal {
                        return Err(internal("first-occurrence mask changed CUDA device"));
                    }
                    ready(first)?;
                }
                let rows = ptr(self.rows.as_ref().unwrap().get(), lease.stream());
                let arena = ptr(self.arena.as_ref().unwrap().get(), lease.stream());
                let control = ptr(self.control.as_ref().unwrap().get(), lease.stream());
                let count = self.count as u64;
                let link_words = (view.count as usize).checked_mul(2).ok_or_else(refused)?;
                // SAFETY: rows_link initializes every node later reachable from a head.
                let next = unsafe { lease.execution().alloc::<u64>(link_words.max(1)) }
                    .map_err(|_| refused())?;
                let next = PendingSlice::new(lease, next);
                if self.count != 0 {
                    let mut call = lease
                        .execution()
                        .launch_builder(&lease.device().0.kernels.pf_rows_clear);
                    call.arg(&rows).arg(&count);
                    // SAFETY: the private row-header array covers count rows.
                    unsafe {
                        call.launch(grid(lease, &lease.device().0.kernels.pf_rows_clear, count)?)
                    }
                    .map_err(|e| fail(lease, "clearing exact PF row heads", e))?;
                }
                if view.count != 0 {
                    let keys_ptr = keys.address();
                    let mirror_arg = i32::from(mirror);
                    let has_cutoff = i32::from(cutoff.is_some());
                    let cutoff_arg = cutoff.unwrap_or(0);
                    let next_ptr = ptr(next.get(), lease.stream());
                    let first_ptr = first.map_or(0, DeviceColumn::address);
                    let mut call = lease
                        .execution()
                        .launch_builder(&lease.device().0.kernels.pf_rows_link);
                    call.arg(&space)
                        .arg(&view)
                        .arg(&keys_ptr)
                        .arg(&first_ptr)
                        .arg(&mirror_arg)
                        .arg(&has_cutoff)
                        .arg(&cutoff_arg)
                        .arg(&rows)
                        .arg(&count)
                        .arg(&next_ptr)
                        .arg(&control);
                    // SAFETY: terms/keys/mask cover the chunk; producer provides unique validated work.
                    unsafe {
                        call.launch(grid(
                            lease,
                            &lease.device().0.kernels.pf_rows_link,
                            view.count,
                        )?)
                    }
                    .map_err(|e| fail(lease, "grouping exact PF row terms", e))?;
                    if self.count != 0 {
                        let mut call = lease
                            .execution()
                            .launch_builder(&lease.device().0.kernels.pf_rows_add);
                        call.arg(&view)
                            .arg(&rows)
                            .arg(&count)
                            .arg(&next_ptr)
                            .arg(&arena)
                            .arg(&control);
                        // SAFETY: one worker mutates each private row; capacity is checked before writes.
                        unsafe {
                            call.launch(grid(lease, &lease.device().0.kernels.pf_rows_add, count)?)
                        }
                        .map_err(|e| fail(lease, "adding exact PF row terms", e))?;
                    }
                }
                read_status(lease, self.control.as_ref().unwrap())
            });
            retire(lease, keep, result)
        }
        /// Discard the old arena before allocating the exact requested capacities;
        /// caller must reset and replay the producer after success.
        pub fn grow_reset(&mut self, lease: &DeviceLease) -> Result<(), EngineError> {
            guard(|| {
                self.check_lease(lease)?;
                // SAFETY: one scalar is written by rows_grow before the copy below.
                let total = unsafe { lease.execution().alloc::<u64>(1) }.map_err(|_| refused())?;
                let total = PendingSlice::new(lease, total);
                let rows = ptr(self.rows.as_ref().unwrap().get(), lease.stream());
                let count = self.count as u64;
                let total_ptr = ptr(total.get(), lease.stream());
                let control_ptr = ptr(self.control.as_ref().unwrap().get(), lease.stream());
                let mut call = lease
                    .execution()
                    .launch_builder(&lease.device().0.kernels.pf_rows_grow);
                call.arg(&rows)
                    .arg(&count)
                    .arg(&total_ptr)
                    .arg(&control_ptr);
                // SAFETY: serial kernel only updates private row metadata and the total scalar.
                unsafe {
                    call.launch(LaunchConfig {
                        grid_dim: (1, 1, 1),
                        block_dim: (1, 1, 1),
                        shared_mem_bytes: 0,
                    })
                }
                .map_err(|e| fail(lease, "growing exact PF row metadata", e))?;
                if read_status(lease, self.control.as_ref().unwrap())? != RowsProgress::Ready {
                    return Err(internal("unresolved exact row growth"));
                }
                let mut words = [0u64];
                lease
                    .execution()
                    .memcpy_dtoh(total.get(), &mut words)
                    .map_err(|e| fail(lease, "reading exact PF arena size", e))?;
                lease.synchronize()?;
                let n = usize::try_from(words[0])
                    .ok()
                    .filter(|&n| n <= isize::MAX as usize / 8)
                    .ok_or_else(refused)?;
                drop(self.arena.take());
                // SAFETY: reset used=0; replay initializes every live word before it is read.
                let arena =
                    unsafe { lease.execution().alloc::<u64>(n.max(1)) }.map_err(|_| refused())?;
                self.arena = Some(PendingSlice::new(lease, arena));
                Ok(())
            })
        }
        pub fn finish(
            &self,
            lease: &DeviceLease,
            spec: PfFinalizeConstants,
        ) -> Result<DeviceColumn, EngineError> {
            guard(|| {
                self.check_lease(lease)?;
                if read_status(lease, self.control.as_ref().unwrap())? != RowsProgress::Ready {
                    return Err(internal("requires producer replay before row finalize"));
                }
                let output = lease.output(DeviceDType::F64, vec![self.count])?;
                if self.count != 0 {
                    let rows_ptr = ptr(self.rows.as_ref().unwrap().get(), lease.stream());
                    let count = self.count as u64;
                    let arena_ptr = ptr(self.arena.as_ref().unwrap().get(), lease.stream());
                    let output_ptr = output.address();
                    let control_ptr = ptr(self.control.as_ref().unwrap().get(), lease.stream());
                    let mut call = lease
                        .execution()
                        .launch_builder(&lease.device().0.kernels.pf_rows_finish);
                    call.arg(&spec)
                        .arg(&rows_ptr)
                        .arg(&count)
                        .arg(&arena_ptr)
                        .arg(&output_ptr)
                        .arg(&control_ptr);
                    // SAFETY: private rows/arena are stable; every output is written before publication.
                    unsafe {
                        call.launch(grid(
                            lease,
                            &lease.device().0.kernels.pf_rows_finish,
                            count,
                        )?)
                    }
                    .map_err(|e| fail(lease, "finalizing exact PF row sums", e))?;
                }
                if read_status(lease, self.control.as_ref().unwrap())? != RowsProgress::Ready {
                    return Err(internal("row finalize needs replay"));
                }
                // SAFETY: successful finish wrote each row, including empty NaN/all-zero infinity rows.
                unsafe { output.finish() }
            })
        }
    }
    /// The host weight routine of K1's `lse_dg_kcal` (duplex-mfe.md §8.6.1): W(D) as the
    /// table record (M, e) with value M · 2^(e − 52), the CPU path's function and bits.
    pub type LseWeight<'w> = &'w (dyn Fn(i64) -> Result<[u64; 2], EngineError> + Sync);

    /// The label of the table mirror's reservations (duplex-cuda.md §6.1).
    const LSE_WEIGHT_TABLE: &str = "lse weight table";
    /// A `continuation` boundary follows this many newly computed weights.
    const LSE_BOUNDARY_ENTRIES: usize = 4096;

    /// K1's per-call table of W(D) over the integer window [lo, lo + len) (duplex-cuda.md
    /// §6.1): a host mirror of 16-byte records and a device copy of the whole window.
    /// Empty at the start of a call; it only grows, to the hull of the chunks' looked-up
    /// keys, and is kept across producer replays, so no weight is computed twice.
    pub struct K1LseTable {
        lo: i64,
        host: Vec<u64>,
        device: Option<PendingSlice<u64>>,
        stream: Arc<CudaStream>,
    }
    impl K1LseTable {
        pub fn new(lease: &DeviceLease) -> Self {
            Self {
                lo: 0,
                host: Vec::new(),
                device: None,
                stream: Arc::clone(lease.stream()),
            }
        }
        fn len(&self) -> usize {
            self.host.len() / 2
        }
        /// The smallest and largest looked-up key of `keys` (int32, one per work item),
        /// or None: a device reduction into two int32 control words, then one control
        /// copy to the host.
        fn window(
            &self,
            lease: &DeviceLease,
            keys: &DeviceColumn,
            n: usize,
            cutoff: Option<i64>,
        ) -> Result<Option<(i64, i64)>, EngineError> {
            // SAFETY: both words are initialized by the following same-stream copy.
            let mut bounds = unsafe { lease.execution().alloc::<i32>(2) }
                .map_err(|e| fail(lease, "allocating the K1 lse window control", e))?;
            lease
                .execution()
                .memcpy_htod(&[i32::MAX, i32::MIN], &mut bounds)
                .map_err(|e| fail(lease, "initializing the K1 lse window control", e))?;
            let bounds = PendingSlice::new(lease, bounds);
            if n != 0 {
                let keys_ptr = keys.address();
                let count = n as u64;
                let has_cutoff = i32::from(cutoff.is_some());
                let cutoff_arg = cutoff.unwrap_or(0);
                let bounds_ptr = ptr(bounds.get(), lease.stream());
                let function = &lease.device().0.kernels.k1_lse_window;
                let mut call = lease.execution().launch_builder(function);
                call.arg(&keys_ptr)
                    .arg(&count)
                    .arg(&has_cutoff)
                    .arg(&cutoff_arg)
                    .arg(&bounds_ptr);
                // SAFETY: the int32 key column covers n items; the control holds two words.
                unsafe { call.launch(grid(lease, function, count)?) }
                    .map_err(|e| fail(lease, "reducing the K1 lse key window", e))?;
            }
            let mut words = [0i32; 2];
            lease
                .execution()
                .memcpy_dtoh(bounds.get(), &mut words)
                .map_err(|e| fail(lease, "reading the K1 lse key window", e))?;
            lease.synchronize()?;
            Ok((words[0] <= words[1]).then(|| (i64::from(words[0]), i64::from(words[1]))))
        }
        /// Grow the window to the hull of itself and [k_min, k_max]: reserve the new
        /// mirror exactly (16 bytes per entry), copy the known entries, compute each
        /// missing W(D) once in increasing D, upload the whole window through the staged
        /// upload and retire the old device copy. A `continuation` boundary of 0 items
        /// follows every 4,096 new weights and the upload (cancellation-progress.md §3.5).
        fn extend(
            &mut self,
            lease: &DeviceLease,
            (k_min, k_max): (i64, i64),
            weight: LseWeight<'_>,
        ) -> Result<(), EngineError> {
            let (old_lo, old_len) = (self.lo, self.len());
            let (lo, hi) = if old_len == 0 {
                (k_min, k_max)
            } else {
                let old_hi = old_lo + old_len as i64 - 1;
                if k_min >= old_lo && k_max <= old_hi {
                    return Ok(());
                }
                (old_lo.min(k_min), old_hi.max(k_max))
            };
            let len = usize::try_from(hi - lo + 1)
                .ok()
                .filter(|&n| n <= isize::MAX as usize / 16)
                .ok_or_else(|| {
                    EngineError::Device(format!(
                        "unable to reserve {LSE_WEIGHT_TABLE}: the window [{lo}, {hi}] exceeds \
                         the address space; no result was published"
                    ))
                })?;
            let mut host = Vec::new();
            seqblaze_core::detail::try_reserve_exact(&mut host, 2 * len, LSE_WEIGHT_TABLE)?;
            let mut fresh = 0usize;
            for d in lo..=hi {
                let known = d
                    .checked_sub(old_lo)
                    .and_then(|at| usize::try_from(at).ok())
                    .filter(|&at| at < old_len);
                match known {
                    Some(at) => host.extend_from_slice(&self.host[2 * at..2 * at + 2]),
                    None => {
                        host.extend_from_slice(&weight(d)?);
                        fresh += 1;
                        if fresh % LSE_BOUNDARY_ENTRIES == 0 {
                            cancel::boundary(Site::Continuation, 0)?;
                        }
                    }
                }
            }
            // SAFETY: every word is initialized by the following staged copy.
            let mut device = unsafe { lease.execution().alloc::<u64>(host.len()) }
                .map_err(|e| fail(lease, "allocating the K1 lse weight table", e))?;
            // The staging copy is reserved before the copy is enqueued
            // (host-allocation-failures §12.6 rule 4).
            let staging = crate::interop::execution_stream::upload_staging::<u64>(host.len())?;
            lease
                .execution()
                .memcpy_htod_staged(&host, &mut device, staging)
                .map_err(|e| fail(lease, "uploading the K1 lse weight table", e))?;
            // The old copy is retired by PendingSlice's rule (after the stream's work).
            self.device = Some(PendingSlice::new(lease, device));
            self.host = host;
            self.lo = lo;
            cancel::boundary(Site::Continuation, 0)
        }
        /// The int64 keys and interleaved term records (U64[n, 2]) of one chunk of K1
        /// engine keys (int32), for [`PfExactRows::consume`]: the window step, the table
        /// extension, then the device gather (duplex-cuda.md §6.1 steps 1-3).
        pub fn chunk_terms(
            &mut self,
            lease: &DeviceLease,
            keys: &DeviceColumn,
            cutoff: Option<i64>,
            weight: LseWeight<'_>,
        ) -> Result<(DeviceColumn, DeviceColumn), EngineError> {
            if !Arc::ptr_eq(&self.stream, lease.stream()) {
                return Err(internal("lse weight table changed its consumer stream"));
            }
            let n = keys.shape().first().copied().unwrap_or(0);
            column(keys, DeviceDType::I32, n)?;
            if keys.ordinal() != lease.device().info().ordinal {
                return Err(internal("lse keys changed CUDA device"));
            }
            let keep = vec![keys.clone()];
            let result = guard(|| {
                ready(keys)?;
                if let Some(window) = self.window(lease, keys, n, cutoff)? {
                    self.extend(lease, window, weight)?;
                }
                let keys64 = lease.output(DeviceDType::I64, vec![n])?;
                let terms = lease.output(DeviceDType::U64, vec![n, 2])?;
                let control = zero_status(lease)?;
                if n != 0 {
                    let keys_ptr = keys.address();
                    let count = n as u64;
                    let has_cutoff = i32::from(cutoff.is_some());
                    let cutoff_arg = cutoff.unwrap_or(0);
                    let table_ptr = self
                        .device
                        .as_ref()
                        .map_or(0, |table| ptr(table.get(), lease.stream()));
                    let lo = self.lo;
                    let len = self.len() as u64;
                    let keys64_ptr = keys64.address();
                    let terms_ptr = terms.address();
                    let control_ptr = ptr(control.get(), lease.stream());
                    let function = &lease.device().0.kernels.k1_lse_gather;
                    let mut call = lease.execution().launch_builder(function);
                    call.arg(&keys_ptr)
                        .arg(&count)
                        .arg(&has_cutoff)
                        .arg(&cutoff_arg)
                        .arg(&table_ptr)
                        .arg(&lo)
                        .arg(&len)
                        .arg(&keys64_ptr)
                        .arg(&terms_ptr)
                        .arg(&control_ptr);
                    // SAFETY: keys cover n items, the table holds len records, both outputs
                    // are distinct full-sized allocations written for every item.
                    unsafe { call.launch(grid(lease, function, count)?) }
                        .map_err(|e| fail(lease, "gathering K1 lse terms", e))?;
                }
                let mut code = [0u32];
                lease
                    .execution()
                    .memcpy_dtoh(control.get(), &mut code)
                    .map_err(|e| fail(lease, "reading the K1 lse gather control", e))?;
                lease.synchronize()?;
                if code[0] != 0 {
                    return Err(EngineError::Device(
                        "internal error: resident duplex lse_dg_kcal weight outside its table; \
                         no result was published"
                            .into(),
                    ));
                }
                // SAFETY: the gather wrote every key and record; empty columns need no write.
                let keys64 = unsafe { keys64.finish()? };
                // SAFETY: as above.
                let terms = unsafe { terms.finish()? };
                Ok((keys64, terms))
            });
            retire(lease, keep, result)
        }
    }

    impl Drop for PfExactRows {
        fn drop(&mut self) {
            if guard(|| {
                crate::interop::execution_stream::retire_in_drop(&self.stream)
                    .map_err(|e| failure_ordinal(&self.device, "retiring exact PF rows", e))
            })
            .is_err()
            {
                if let Some(rows) = self.rows.take() {
                    std::mem::forget(rows);
                }
                if let Some(arena) = self.arena.take() {
                    std::mem::forget(arena);
                }
                if let Some(control) = self.control.take() {
                    std::mem::forget(control);
                }
            }
        }
    }
}
#[cfg(feature = "cuda")]
pub use driver::{finalize, K1LseTable, LseWeight, PfExactRows, PfFinalized, PfTerms};
