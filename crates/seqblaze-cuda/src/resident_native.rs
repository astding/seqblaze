//! Native resident metadata and primary stages. Existing arithmetic kernels
//! consume the same validated borrowed row descriptors as every resident family.
use crate::{
    device::guard,
    interop::{allocate, scan, BoundDeviceRows, DeviceColumn, DeviceDType, DeviceLease, Retained},
    resident::{ResidentChunk, ResidentField, ResidentSpace},
    session::{failure_ordinal, SbTables},
    RegisterCall,
};
use cudarc::driver::{CudaFunction, DeviceRepr, LaunchConfig, PushKernelArg};
use seqblaze_core::validate::EngineError;
use std::sync::Arc;

pub(crate) fn grid(
    lease: &DeviceLease,
    function: &CudaFunction,
    n: u64,
) -> Result<LaunchConfig, EngineError> {
    let maximum = function
        .max_threads_per_block()
        .map_err(|e| failure_ordinal(lease.device(), "reading resident kernel block limit", e))?;
    let requested = crate::launch_override().threads_per_block.unwrap_or(128);
    let block = u32::try_from(maximum)
        .ok()
        .filter(|&n| n != 0)
        .ok_or_else(|| {
            EngineError::Device("resident kernel reports an invalid block limit".into())
        })?
        .min(requested)
        .clamp(1, 256);
    Ok(LaunchConfig {
        grid_dim: (n.div_ceil(u64::from(block)).clamp(1, 65535) as u32, 1, 1),
        block_dim: (block, 1, 1),
        shared_mem_bytes: 0,
    })
}
pub(crate) fn tables(a: &BoundDeviceRows, b: &BoundDeviceRows) -> SbTables {
    SbTables::from_views(a.address(), b.address())
}
#[repr(C)]
#[derive(Clone, Copy)]
struct Pieces {
    piece_bytes: u64,
    per_pair: u64,
    prefix: u64,
    total: u64,
}
// SAFETY: same four plain-u64 fields as SbPieces.
unsafe impl DeviceRepr for Pieces {}

/// Lanes per piece (levenshtein-cuda.md §15): one thread for rows of at most
/// 16 bytes, else the power of two nearest above ⌈longest / 16⌉, at most a
/// warp. Scheduling only: every group size gives every count.
fn group_lanes(longest: u64) -> u32 {
    if longest <= 16 {
        1
    } else {
        longest.div_ceil(16).next_power_of_two().min(32) as u32
    }
}

/// Launch the resident Hamming pieces `pieces` into `output` with lane groups
/// when the block is whole warps, else one thread per piece; `threads` is the
/// one-thread grid size (an upper bound of the pieces).
fn launch_pieces(
    lease: &DeviceLease,
    table: &SbTables,
    abi: &crate::session::SbSpace,
    pieces: &Pieces,
    address: u64,
    threads: u64,
    longest: u64,
) -> Result<(), EngineError> {
    let fail = |what: &str, e| failure_ordinal(lease.device(), what, e);
    let functions = &lease.device().0.kernels;
    let group = group_lanes(longest);
    if group > 1 {
        let function = &functions.resident_hamming_groups;
        let config = grid(lease, function, threads.saturating_mul(u64::from(group)))?;
        if config.block_dim.0 % 32 == 0 {
            let mut launch = lease.execution().launch_builder(function);
            launch
                .arg(table)
                .arg(abi)
                .arg(pieces)
                .arg(&address)
                .arg(&group);
            // SAFETY: matches sb_resident_hamming_groups; whole warps (checked),
            // each piece counted by one group and stored or added once.
            unsafe { launch.launch(config) }.map_err(|e| fail("launching resident Hamming", e))?;
            return Ok(());
        }
    }
    let function = if pieces.prefix == 0 {
        &functions.pieces
    } else {
        &functions.resident_hamming_pieces
    };
    let mut launch = lease.execution().launch_builder(function);
    launch.arg(table).arg(abi).arg(pieces).arg(&address);
    // SAFETY: matches sb_hamming_pieces / sb_resident_hamming_pieces; each
    // piece is counted by one thread and stored or added once.
    unsafe { launch.launch(grid(lease, function, threads)?) }
        .map_err(|e| fail("launching resident Hamming", e))?;
    Ok(())
}

/// `longest` is the validated longest row of table a; with equal pair lengths
/// it bounds every pair's piece count, so the piece launch needs no read of
/// the device total (per-call cost S6).
pub fn hamming(
    lease: &DeviceLease,
    a: &BoundDeviceRows,
    b: &BoundDeviceRows,
    space: ResidentSpace,
    longest: u64,
) -> Result<ResidentChunk, EngineError> {
    guard(|| {
        a.ensure_lease(lease)?;
        b.ensure_lease(lease)?;
        let fail = |what: &str, e| failure_ordinal(lease.device(), what, e);
        let n = usize::try_from(space.n).map_err(|_| {
            EngineError::Device("resident Hamming chunk exceeds the platform".into())
        })?;
        let piece = crate::launch_override()
            .piece_bytes
            .unwrap_or(crate::DEFAULT_PIECE_BYTES);
        // Every pair is at most one piece when the longest row of a fits one
        // (pairs have equal lengths): one piece per item, no count or scan.
        let single = longest <= piece;
        let mut counts = if single {
            None
        } else {
            Some(Retained::new(lease, allocate::<u64>(lease, n)?))
        };
        let output = lease.output(DeviceDType::I32, vec![n])?;
        if n != 0 {
            lease
                .device()
                .0
                .ctx
                .bind_to_thread()
                .map_err(|e| fail("binding resident Hamming", e))?;
            // SAFETY: output owns n*4 bytes on this stream. Empty pairs emit no
            // pieces and retain the exact integer zero written here.
            unsafe {
                lease
                    .execution()
                    .memset(output.address(), 0, output.nbytes())
            }
            .map_err(|e| fail("initializing resident Hamming keys", e))?;
            let functions = &lease.device().0.kernels;
            let table = tables(a, b);
            let abi = space.abi();
            let Some(counts) = counts.as_mut() else {
                // levenshtein-cuda.md §14: an empty pair's one piece counts and
                // stores zero, so every key is the exact sum, as with the prefix.
                let pieces = Pieces {
                    piece_bytes: piece,
                    per_pair: 1,
                    prefix: 0,
                    total: space.n,
                };
                // Item t owns exactly piece t of at most `piece` bytes and
                // stores its count once.
                launch_pieces(
                    lease,
                    &table,
                    &abi,
                    &pieces,
                    output.address(),
                    space.n,
                    longest,
                )?;
                // SAFETY: memset and the single-piece stores on this stream
                // initialized every key.
                let keys = unsafe { output.finish() }?;
                return Ok(ResidentChunk {
                    space,
                    fields: vec![ResidentField {
                        name: "dist".into(),
                        values: keys.clone(),
                        mask_bits: i32::MAX as u64,
                    }],
                    keys,
                });
            };
            let mut launch = lease
                .execution()
                .launch_builder(&functions.resident_hamming_counts);
            launch
                .arg(&table)
                .arg(&abi)
                .arg(&piece)
                .slice_arg_mut(&mut **counts);
            // SAFETY: one initialized piece count is produced for each validated pair.
            unsafe { launch.launch(grid(lease, &functions.resident_hamming_counts, space.n)?) }
                .map_err(|e| fail("counting resident Hamming pieces", e))?;
            let prefix = Retained::new(lease, scan(lease, counts, n)?);
            // Every pair has at most ceil(longest / piece) pieces: an upper
            // bound of the device total, which the kernel reads in place.
            let bound = space.n.saturating_mul(longest.div_ceil(piece.max(1)));
            if bound != 0 {
                let pieces = Pieces {
                    piece_bytes: piece,
                    per_pair: 0,
                    prefix: crate::session::ptr(&prefix, lease.stream()),
                    total: 0,
                };
                // The exact device prefix assigns every nonempty piece to its
                // pair, and integer atomic sums fit the validated i32 cap.
                launch_pieces(
                    lease,
                    &table,
                    &abi,
                    &pieces,
                    output.address(),
                    bound,
                    longest,
                )?;
            }
        }
        // SAFETY: memset initialized every key, and all pieces update that same stream.
        let keys = unsafe { output.finish() }?;
        Ok(ResidentChunk {
            space,
            fields: vec![ResidentField {
                name: "dist".into(),
                values: keys.clone(),
                mask_bits: i32::MAX as u64,
            }],
            keys,
        })
    })
}

pub struct ResidentRegisterStage {
    state: Retained<crate::reg::Reg>,
}
impl ResidentRegisterStage {
    pub fn new(lease: &DeviceLease, call: &Arc<RegisterCall>) -> Result<Self, EngineError> {
        Ok(Self {
            state: Retained::new(
                lease,
                crate::reg::Reg::new_on(
                    (lease.stream(), lease.device()),
                    call,
                    Some(lease.execution()),
                )?,
            ),
        })
    }
    pub fn stage(
        &mut self,
        lease: &DeviceLease,
        a: &BoundDeviceRows,
        b: &BoundDeviceRows,
        space: ResidentSpace,
    ) -> Result<ResidentChunk, EngineError> {
        self.state.resident_stage(lease, a, b, space)
    }
}

pub enum PrimaryFamily {
    Align,
    Duplex,
    Register,
    Pf,
}
/// Convert engine keys on-device; RowStats uses the same mapping with a best_
/// output name. This function never imports a kernel's inputs or recomputes it.
pub fn primary(
    lease: &DeviceLease,
    keys: &DeviceColumn,
    family: PrimaryFamily,
    name: &str,
    integer: Option<&str>,
) -> Result<Vec<ResidentField>, EngineError> {
    guard(|| {
        if keys.ordinal() != lease.device().info().ordinal {
            return Err(EngineError::Device(
                "resident keys belong to a different CUDA device".into(),
            ));
        }
        keys.synchronize()?;
        let _source = Retained::new(lease, keys.clone());
        let code = match family {
            PrimaryFamily::Align => 1u32,
            PrimaryFamily::Duplex => 2,
            PrimaryFamily::Register => 3,
            PrimaryFamily::Pf => 4,
        };
        let expected = if code <= 2 {
            DeviceDType::I32
        } else {
            DeviceDType::I64
        };
        if keys.dtype() != expected || keys.ordinal() != lease.device().info().ordinal {
            return Err(EngineError::Device(
                "resident primary received inconsistent keys".into(),
            ));
        }
        let count = keys
            .shape()
            .iter()
            .try_fold(1usize, |n, &d| n.checked_mul(d))
            .ok_or_else(|| EngineError::Device("resident primary shape overflows".into()))?;
        let floats = lease.output(DeviceDType::F64, keys.shape().to_vec())?;
        let ints = integer
            .map(|_| lease.output(DeviceDType::I32, keys.shape().to_vec()))
            .transpose()?;
        if count != 0 {
            let function = &lease.device().0.kernels.resident_primary;
            let input = keys.address();
            let n = count as u64;
            let out = floats.address();
            let int = ints.as_ref().map_or(0, |column| column.address());
            let mut launch = lease.execution().launch_builder(function);
            launch.arg(&input).arg(&n).arg(&code).arg(&out).arg(&int);
            // SAFETY: complete typed key input and fresh complete output columns.
            unsafe { launch.launch(grid(lease, function, n)?) }.map_err(|e| {
                failure_ordinal(lease.device(), "mapping resident primary values", e)
            })?;
        }
        let mut fields = Vec::new();
        fields
            .try_reserve_exact(1 + usize::from(ints.is_some()))
            .map_err(|_| EngineError::Device("unable to reserve resident primary fields".into()))?;
        // SAFETY: every nonempty output byte was written by the launch above.
        fields.push(ResidentField {
            name: name.into(),
            values: unsafe { floats.finish() }?,
            mask_bits: 0x7ff8000000000000,
        });
        if let Some(output) = ints {
            fields.push(ResidentField {
                name: integer.expect("integer output requested").into(),
                values: unsafe { output.finish() }?,
                mask_bits: i32::MAX as u64,
            });
        }
        Ok(fields)
    })
}

/// Oriented dense numeric details over bounded coordinate chunks. Matrix and
/// Tiles explicitly request dense numeric payloads; no global index product is
/// built. Holes and cutoff failures carry passed=false into the detail kernel.
// Keep the dense launch's independent layout and field controls explicit.
#[allow(clippy::too_many_arguments)]
pub(crate) fn dense_details(
    lease: &DeviceLease,
    a: &BoundDeviceRows,
    b: &BoundDeviceRows,
    shape: [usize; 2],
    origin: [usize; 2],
    keys: &DeviceColumn,
    passed: &DeviceColumn,
    call: &crate::detail::DetailCall<'_>,
    fields: &[&str],
    aligned_binary: bool,
) -> Result<Vec<ResidentField>, EngineError> {
    use crate::{
        interop::DeviceOutput,
        resident::{ResidentDetails, ResidentEntries},
    };
    let total = shape[0]
        .checked_mul(shape[1])
        .ok_or_else(|| EngineError::Device("dense detail shape overflows".into()))?;
    if keys.shape() != shape || passed.shape() != shape || passed.dtype() != DeviceDType::Bool {
        return Err(EngineError::Device(
            "dense detail geometry is inconsistent".into(),
        ));
    }
    let mut outputs: Vec<(String, u64, DeviceOutput)> = Vec::new();
    let mut position = 0usize;
    seqblaze_core::cancel::stage(seqblaze_core::cancel::Stage::Details, total);
    // Even an empty matrix passes one zero-sized detail chunk to obtain its
    // exact requested numeric schema without touching any sequence payload.
    loop {
        let n = (total - position).min(4096);
        let space = ResidentSpace {
            kind: 2,
            k0: position as u64,
            n: n as u64,
            r0: origin[0] as u64,
            c0: origin[1] as u64,
            width: shape[1] as u64,
            ..ResidentSpace::default()
        };
        let ids = lease.output(DeviceDType::U64, vec![n])?;
        let left = lease.output(DeviceDType::I64, vec![n])?;
        let right = lease.output(DeviceDType::I64, vec![n])?;
        let chunk_keys = lease.output(keys.dtype(), vec![n])?;
        let chunk_passed = lease.output(DeviceDType::Bool, vec![n])?;
        if n != 0 {
            let function = &lease.device().0.kernels.resident_dense_entries;
            let abi = space.abi();
            let input = keys.address();
            let mask = passed.address();
            let width = keys.dtype().width() as u64;
            let base = position as u64;
            let id = ids.address();
            let i = left.address();
            let j = right.address();
            let key = chunk_keys.address();
            let pass = chunk_passed.address();
            let mut launch = lease.execution().launch_builder(function);
            launch
                .arg(&abi)
                .arg(&input)
                .arg(&mask)
                .arg(&width)
                .arg(&base)
                .arg(&id)
                .arg(&i)
                .arg(&j)
                .arg(&key)
                .arg(&pass);
            // SAFETY: base..base+n is a checked contiguous output subspan;
            // rectangle coordinates refer to this tile's original input rows.
            unsafe { launch.launch(grid(lease, function, n as u64)?) }
                .map_err(|e| failure_ordinal(lease.device(), "forming dense detail chunk", e))?;
        }
        // SAFETY: every nonempty coordinate/key/pass buffer was filled above.
        let entries = ResidentEntries {
            work_ids: unsafe { ids.finish() }?,
            i: unsafe { left.finish() }?,
            j: unsafe { right.finish() }?,
            keys: unsafe { chunk_keys.finish() }?,
            passed: Some(unsafe { chunk_passed.finish() }?),
            rank: None,
        };
        let ResidentDetails { numeric, ragged } = crate::detail::resident_details(
            lease,
            a,
            b,
            &entries,
            call,
            fields,
            aligned_binary,
            false,
        )?;
        if !ragged.is_empty() {
            return Err(EngineError::Parameter(
                "dense results hold numeric detail fields only".into(),
            ));
        }
        if position == 0 {
            outputs.try_reserve_exact(numeric.len()).map_err(|_| {
                EngineError::Device("unable to retain dense device detail columns".into())
            })?;
            for field in &numeric {
                outputs.push((
                    field.name.clone(),
                    field.mask_bits,
                    lease.output(field.values.dtype(), shape.to_vec())?,
                ));
            }
        }
        if outputs.len() != numeric.len() {
            return Err(EngineError::Device(
                "dense detail schema changed across chunks".into(),
            ));
        }
        for ((name, _, output), field) in outputs.iter().zip(&numeric) {
            if *name != field.name || field.values.shape() != [n] {
                return Err(EngineError::Device(
                    "dense detail chunk has an inconsistent schema".into(),
                ));
            }
            if n != 0 {
                let width = field.values.dtype().width();
                let bytes = n.checked_mul(width).ok_or_else(|| {
                    EngineError::Device("dense detail byte count overflows".into())
                })?;
                let destination = output.address() + position as u64 * width as u64;
                // SAFETY: disjoint complete chunk buffers are copied into the
                // checked corresponding contiguous dense destination subspan.
                unsafe {
                    lease
                        .execution()
                        .copy_device(destination, field.values.address(), bytes)
                }
                .map_err(|e| failure_ordinal(lease.device(), "joining dense device details", e))?;
            }
        }
        position += n;
        // The dense detail chunk loop's host boundary (cancellation-progress.md §3.5).
        seqblaze_core::cancel::boundary(seqblaze_core::cancel::Site::Details, n)?;
        if position == total {
            break;
        }
    }
    let mut result = Vec::new();
    result
        .try_reserve_exact(outputs.len())
        .map_err(|_| EngineError::Device("unable to publish dense device details".into()))?;
    for (name, mask_bits, output) in outputs {
        // SAFETY: all dense positions, including masked holes, were copied from
        // fully initialized detail chunks; an empty matrix has no logical read.
        result.push(ResidentField {
            name,
            mask_bits,
            values: unsafe { output.finish() }?,
        });
    }
    Ok(result)
}
