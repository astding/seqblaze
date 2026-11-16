//! The C1 bound of K1 native `duplex` on the device (kernels/duplex_bound.cuh,
//! docs/wiki/thermo/duplex-mfe-bound.md §8): one thread per item, integer only. Used by
//! the testing hook `duplex_bound(..., device=...)`; its (LB, χ, outcome) bytes equal
//! the CPU's.
use cudarc::driver::{CudaSlice, DeviceRepr, DriverError, LaunchConfig, PushKernelArg};
use seqblaze_core::validate::EngineError;

use crate::{
    device::guard,
    plan::MAX_GRID,
    session::{dev_alloc, failure, ptr},
    BoundColumns, Device, DupBoundCall,
};

/// Mirrors `SbDupBound` in kernels/duplex_bound.cuh: 8-byte fields in its order.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct SbDupBound {
    pub(crate) keys: u64,
    pub(crate) cls: u64,
    pub(crate) relaxed: u64,
    pub(crate) n_relaxed: u64,
    pub(crate) lam: u64,
    pub(crate) n_p: u64,
    pub(crate) n_c: u64,
    pub(crate) has_states: u64,
    pub(crate) off: [u64; 14],
    pub(crate) codes: u64,
    pub(crate) a_start: u64,
    pub(crate) a_len: u64,
    pub(crate) b_start: u64,
    pub(crate) b_len: u64,
    pub(crate) n_items: u64,
    pub(crate) scratch: u64,
    pub(crate) stride: u64,
    pub(crate) has_theta: u64,
    pub(crate) theta: i64,
    pub(crate) lb_plus_one: u64,
    pub(crate) out_lb: u64,
    pub(crate) out_chi: u64,
    pub(crate) out_code: u64,
}
const _: () = assert!(std::mem::size_of::<SbDupBound>() == 36 * 8);

/// Words of one thread's bound scratch for strands of at most `longest` letters: the
/// RA band, N_C OUT bands and the F and G bands of every relaxed class, R·m words each
/// with R = min(longest, Λ + 2) (kernels/duplex_bound.cuh `sb_bnd_fill`).
pub(crate) fn scratch_words(call: &DupBoundCall, longest: u64) -> u64 {
    let rows = longest.min(call.loop_max as u64 + 2);
    let bands = 1 + call.n_c as u64 + 2 * call.n_relaxed as u64;
    rows * longest.max(1) * bands
}

// SAFETY: plain #[repr(C)] struct of 8-byte fields in the CUDA struct's order; passed by
// value as a kernel parameter.
unsafe impl DeviceRepr for SbDupBound {}

/// Scratch budget of one launch (the 256 MiB slot rule of duplex-cuda.md §5.4).
const SCRATCH_BYTES: u64 = 256 << 20;
const THREADS: u64 = 64;

/// (LB, χ, outcome) of every item (`a[t]`, `b[t]` as letter codes 0..4).
pub(crate) fn eval(
    device: &Device,
    call: &DupBoundCall,
    a: &[&[u8]],
    b: &[&[u8]],
    threshold: Option<i32>,
    lb_plus_one: bool,
) -> Result<BoundColumns, EngineError> {
    let count = a.len();
    if count == 0 {
        return Ok((Vec::new(), Vec::new(), Vec::new()));
    }
    guard(|| {
        let label = format!("cuda:{}", device.info().ordinal);
        let fail = |what: &str, error: DriverError| failure(&label, what, error);
        let stream = device
            .0
            .ctx
            .new_stream()
            .map_err(|error| fail("creating a stream", error))?;
        let mut codes = Vec::new();
        let (mut a_start, mut a_len, mut b_start, mut b_len) =
            (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        for t in 0..count {
            a_start.push(codes.len() as u64);
            a_len.push(a[t].len() as u64);
            codes.extend_from_slice(a[t]);
            b_start.push(codes.len() as u64);
            b_len.push(b[t].len() as u64);
            codes.extend_from_slice(b[t]);
        }
        codes.push(0);
        let max_m = b.iter().map(|s| s.len()).max().unwrap_or(0).max(1) as u64;
        let rows = call.loop_max as u64 + 2;
        let bands = 1 + call.n_c as u64 + 2 * call.n_relaxed as u64;
        let stride = rows * max_m * bands;
        let per_thread = stride * 8;
        let threads_total = (SCRATCH_BYTES / per_thread.max(1)).clamp(1, count as u64);
        let blocks = threads_total.div_ceil(THREADS).clamp(1, MAX_GRID);
        let launched = blocks * THREADS;
        fn up<T: DeviceRepr + Copy>(
            stream: &std::sync::Arc<cudarc::driver::CudaStream>,
            host: &[T],
        ) -> Result<CudaSlice<T>, DriverError> {
            // SAFETY: fully written by the copy below.
            let mut slice = unsafe { dev_alloc::<T>(stream, host.len().max(1)) }?;
            if !host.is_empty() {
                stream.memcpy_htod(host, &mut slice)?;
            }
            Ok(slice)
        }
        let keys = up(&stream, &call.keys).map_err(|e| fail("uploading the bound pack", e))?;
        let cls = up(&stream, &call.cls).map_err(|e| fail("uploading the bound pack", e))?;
        let relaxed =
            up(&stream, &call.relaxed).map_err(|e| fail("uploading the bound pack", e))?;
        let d_codes = up(&stream, &codes).map_err(|e| fail("uploading sequences", e))?;
        let d_as = up(&stream, &a_start).map_err(|e| fail("uploading sequences", e))?;
        let d_al = up(&stream, &a_len).map_err(|e| fail("uploading sequences", e))?;
        let d_bs = up(&stream, &b_start).map_err(|e| fail("uploading sequences", e))?;
        let d_bl = up(&stream, &b_len).map_err(|e| fail("uploading sequences", e))?;
        // SAFETY: each thread initialises its own scratch before reading it; the outputs
        // are written for every item before the copies read them.
        let scratch = unsafe { dev_alloc::<i64>(&stream, (launched * stride) as usize) }
            .map_err(|e| fail("allocating the bound scratch", e))?;
        let out_lb = unsafe { dev_alloc::<i64>(&stream, count) }
            .map_err(|e| fail("allocating bound outputs", e))?;
        let out_chi = unsafe { dev_alloc::<u8>(&stream, count) }
            .map_err(|e| fail("allocating bound outputs", e))?;
        let out_code = unsafe { dev_alloc::<u8>(&stream, count) }
            .map_err(|e| fail("allocating bound outputs", e))?;
        let mut off = [0u64; 14];
        for (k, slot) in off.iter_mut().enumerate() {
            *slot = call.offsets[k] as u64;
        }
        let params = SbDupBound {
            keys: ptr(&keys, &stream),
            cls: ptr(&cls, &stream),
            relaxed: ptr(&relaxed, &stream),
            n_relaxed: call.n_relaxed as u64,
            lam: call.loop_max as u64,
            n_p: call.n_p as u64,
            n_c: call.n_c as u64,
            has_states: u64::from(call.has_states),
            off,
            codes: ptr(&d_codes, &stream),
            a_start: ptr(&d_as, &stream),
            a_len: ptr(&d_al, &stream),
            b_start: ptr(&d_bs, &stream),
            b_len: ptr(&d_bl, &stream),
            n_items: count as u64,
            scratch: ptr(&scratch, &stream),
            stride,
            has_theta: u64::from(threshold.is_some()),
            theta: i64::from(threshold.unwrap_or(0)),
            lb_plus_one: u64::from(lb_plus_one),
            out_lb: ptr(&out_lb, &stream),
            out_chi: ptr(&out_chi, &stream),
            out_code: ptr(&out_code, &stream),
        };
        let config = LaunchConfig {
            grid_dim: (blocks as u32, 1, 1),
            block_dim: (THREADS as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut launch = stream.launch_builder(&device.0.kernels.dup_bound);
        launch.arg(&params);
        // SAFETY: matches sb_dup_bound; every array holds the sizes set above, and
        // `launched` scratch slots of `stride` words exist.
        unsafe { launch.launch(config) }.map_err(|e| fail("launching sb_dup_bound", e))?;
        let mut lb = vec![0i64; count];
        let mut chi = vec![0u8; count];
        let mut code = vec![0u8; count];
        stream
            .memcpy_dtoh(&out_lb, &mut lb[..])
            .and_then(|()| stream.memcpy_dtoh(&out_chi, &mut chi[..]))
            .and_then(|()| stream.memcpy_dtoh(&out_code, &mut code[..]))
            .and_then(|()| stream.synchronize())
            .map_err(|e| fail("device-to-host copy", e))?;
        Ok((lb, chi, code))
    })
}
