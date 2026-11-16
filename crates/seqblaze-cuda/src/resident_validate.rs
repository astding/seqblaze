//! Device metadata errors are decided before scoring. Only fixed-size summaries
//! and the one failing pair's integer coordinates/lengths cross to the host.
use crate::{
    device::guard,
    interop::{
        ControlBlock, DeviceInputFact, DeviceInputSummary, DeviceLease, ValidatedDeviceRows,
        PAIR_BAD, PAIR_FIRST, PAIR_OUT,
    },
    resident::ResidentSpace,
    resident_native::grid,
    session::failure_ordinal,
};
use cudarc::driver::{LaunchConfig, PushKernelArg};
use seqblaze_core::validate::EngineError;

pub fn check_spans(lease: &DeviceLease, rows: &ValidatedDeviceRows) -> Result<(), EngineError> {
    guard(|| {
        let row = rows.summary.span_row;
        if row == u64::MAX {
            return Ok(());
        }
        let index = usize::try_from(row).map_err(|_| {
            EngineError::Parameter("invalid device row index exceeds the platform".into())
        })?;
        let mut fact = [DeviceInputFact::default()];
        lease
            .execution()
            .memcpy_dtoh(&rows.facts.slice(index..index + 1), &mut fact[..])
            .map_err(|e| {
                failure_ordinal(lease.device(), "reading invalid device span controls", e)
            })?;
        lease.synchronize()?;
        let table = rows.bound.rows;
        if table.layout == 1 {
            return Err(EngineError::Parameter(format!(
                "lengths[{row}]={} must be between 0 and matrix column capacity {}",
                fact[0].length as i64, table.columns
            )));
        }
        Err(EngineError::Parameter(format!("device string offsets at row {row} ({}, {}) must be ascending nonnegative offsets within byte capacity {}",fact[0].length as i64,fact[0].byte as i64,table.columns)))
    })
}

pub fn sequence_error(
    summary: DeviceInputSummary,
    input: &'static str,
    kernel: &str,
    accepts: &str,
    hint: &str,
) -> Option<EngineError> {
    if summary.bad_count == 0 {
        return None;
    }
    let index = summary.bad_row as usize;
    let n_bad = summary.bad_count as usize;
    let bad_indices: Vec<usize> = summary
        .bad_rows
        .into_iter()
        .filter(|&r| r != u64::MAX)
        .map(|r| r as usize)
        .collect();
    if summary.bad_position == u64::MAX {
        Some(EngineError::Sequence{message:format!("null sequence in input {input}: sequence {index} in input {input} is null; {kernel} cannot compute a null. {n_bad} of {} sequences are invalid (indices {bad_indices:?}). Nothing was computed.",summary.rows),input,index,position:None,byte:None,n_bad,bad_indices})
    } else {
        let position = summary.bad_position as usize;
        let byte = summary.bad_byte as u8;
        let extra = if hint.is_empty() {
            String::new()
        } else {
            format!(" {hint}")
        };
        Some(EngineError::Sequence{message:format!("illegal base in input {input}: sequence {index}, position {position} is byte {byte}; {kernel} accepts only {accepts}. {n_bad} of {} sequences are invalid (indices {bad_indices:?}). Nothing was computed.{extra}",summary.rows),input,index,position:Some(position),byte:Some(byte),n_bad,bad_indices})
    }
}

/// mode=0 checks equal lengths; mode=1 checks the primer3 joint rule that
/// at least one input strand of each requested pair has length <=60.
/// One control read of its own (device-only callers).
pub fn pair_violation(
    lease: &DeviceLease,
    a: &ValidatedDeviceRows,
    b: &ValidatedDeviceRows,
    space: ResidentSpace,
    self_pairs: bool,
    mode: u32,
) -> Result<Option<[u64; 5]>, EngineError> {
    guard(|| {
        let block = ControlBlock::new(lease)?;
        if !enqueue_pair_check(lease, a, b, space, self_pairs, mode, &block)? {
            return Ok(None);
        }
        Ok(block
            .read_as(lease, "reading device pair validation controls")?
            .pair())
    })
}

/// Enqueue the pair check of [`pair_violation`] into the control block (per-call
/// cost S6); the violating pair is `ControlWords::pair` after the block's read.
/// It reads only row facts and pair coordinates, never sequence bytes. Returns
/// false when there is no pair to check (nothing enqueued, no violation).
pub fn enqueue_pair_check(
    lease: &DeviceLease,
    a: &ValidatedDeviceRows,
    b: &ValidatedDeviceRows,
    space: ResidentSpace,
    self_pairs: bool,
    mode: u32,
    block: &ControlBlock,
) -> Result<bool, EngineError> {
    guard(|| {
        if space.n == 0 {
            return Ok(false);
        }
        let fail = |what: &str, e| failure_ordinal(lease.device(), what, e);
        let first = block.at(PAIR_FIRST);
        let bad = block.at(PAIR_BAD);
        let output = block.at(PAIR_OUT);
        // Constant fills of the block's scratch words (S5 (a)).
        // SAFETY: fixed words of the live control block on this lease's stream.
        unsafe { lease.execution().memset(first, 0xFF, 16) }
            .map_err(|e| fail("initializing device reference rows", e))?;
        unsafe { lease.execution().memset(bad, 0xFF, 8) }
            .map_err(|e| fail("initializing device pair validation", e))?;
        let kernels = &lease.device().0.kernels;
        let na = a.bound.rows.rows;
        let nb = b.bound.rows.rows;
        let mut launch = lease
            .execution()
            .validation_builder(&kernels.resident_first_rows);
        launch
            .slice_arg(&*a.facts)
            .arg(&na)
            .slice_arg(&*b.facts)
            .arg(&nb)
            .arg(&mode)
            .arg(&first);
        // SAFETY: validated fact arrays cover both original complete tables.
        unsafe { launch.launch(grid(lease, &kernels.resident_first_rows, na.max(nb))?) }
            .map_err(|e| fail("finding device reference rows", e))?;
        let abi = space.abi();
        let self_pairs = u32::from(self_pairs);
        let candidates = if space.kind <= 1 {
            space.n
        } else {
            na.saturating_add(nb)
        };
        let mut launch = lease
            .execution()
            .validation_builder(&kernels.resident_pair_check);
        launch
            .slice_arg(&*a.facts)
            .arg(&na)
            .slice_arg(&*b.facts)
            .arg(&nb)
            .arg(&abi)
            .arg(&mode)
            .arg(&self_pairs)
            .arg(&first)
            .arg(&bad);
        // SAFETY: every implicit shape has O(na+nb) candidates, explicit lists
        // remain original-order validated coordinates; only integer facts read.
        unsafe { launch.launch(grid(lease, &kernels.resident_pair_check, candidates)?) }
            .map_err(|e| fail("checking device pair metadata", e))?;
        let mut launch = lease
            .execution()
            .validation_builder(&kernels.resident_pair_error);
        launch
            .slice_arg(&*a.facts)
            .slice_arg(&*b.facts)
            .arg(&abi)
            .arg(&bad)
            .arg(&output);
        // SAFETY: one thread extracts at most one known violating pair control
        // into the block's five pair words.
        unsafe {
            launch.launch(LaunchConfig {
                grid_dim: (1, 1, 1),
                block_dim: (1, 1, 1),
                shared_mem_bytes: 0,
            })
        }
        .map_err(|e| fail("extracting device pair error controls", e))?;
        Ok(true)
    })
}
