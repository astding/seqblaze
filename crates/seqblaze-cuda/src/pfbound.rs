//! The C2 bound of K3 `duplex(model="pf")` on the device (kernels/duplex_pf_bound.cuh,
//! docs/wiki/thermo/duplex-pf-bound.md §8): one thread per item, integer only. Used by
//! the testing hook `duplex_pf_bound(..., device=...)` and by the resident bound stage;
//! its (Ẑ_b, L, outcome) bytes equal the CPU's.
use cudarc::driver::{CudaSlice, DeviceRepr, DriverError, LaunchConfig, PushKernelArg};
use seqblaze_core::validate::EngineError;

use crate::{
    device::guard,
    interop::pf_finalize::PfFinalizeConstants,
    plan::MAX_GRID,
    session::{dev_alloc, failure, ptr},
    Device, PfBoundCall, PfBoundColumns,
};

/// U of the relaxed recurrence (duplex-pf-bound.md §3.1).
const U: u64 = 4;
/// Rows of each accumulator ring (`SB_PFB_ACC_ROWS`).
const ACC_ROWS: u64 = U + 3;
/// Words of one relaxed class (`SB_PFB_CLASS_WORDS`).
const CLASS_WORDS: usize = 5 + 3 * (2 * U as usize + 2);
/// Staged letter codes ahead of each thread's bands (`SB_PFB_CODE_VALUES`, F53 values).
pub(crate) const CODE_VALUES: u64 = 25;
/// The relaxed classes of the kernel (`SB_PFB_MAX_CLASSES`).
const MAX_CLASSES: usize = 3;

/// Mirrors `SbPfBound` in kernels/duplex_pf_bound.cuh: 8-byte fields in its order.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct SbPfBound {
    pub(crate) w: u64,
    pub(crate) lam: u64,
    pub(crate) n_p: u64,
    pub(crate) n_c: u64,
    pub(crate) cls: u64,
    pub(crate) start: u64,
    pub(crate) end: u64,
    pub(crate) stack: u64,
    pub(crate) bulge: u64,
    pub(crate) i11: u64,
    pub(crate) i12: u64,
    pub(crate) i21: u64,
    pub(crate) i22: u64,
    pub(crate) gen: u64,
    pub(crate) loop_out: u64,
    pub(crate) loop_in: u64,
    pub(crate) relaxed: u64,
    pub(crate) n_relaxed: u64,
    pub(crate) scale_down: u64,
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
    pub(crate) out_m: u64,
    pub(crate) out_e: u64,
    pub(crate) out_l: u64,
    pub(crate) out_code: u64,
}
const _: () = assert!(std::mem::size_of::<SbPfBound>() == 33 * 8);

// SAFETY: plain #[repr(C)] struct of 8-byte fields in the CUDA struct's order; passed by
// value as a kernel parameter.
unsafe impl DeviceRepr for SbPfBound {}

/// The call's constants checked against the kernel's layout: the relaxed words, at most
/// three classes, the weight pack and the 135 finalize words.
pub(crate) fn check(call: &PfBoundCall) -> Result<PfFinalizeConstants, EngineError> {
    let bad = |what: &str| {
        EngineError::Device(format!(
            "internal error: duplex model='pf' bound constants are inconsistent ({what}); no \
             result was published"
        ))
    };
    if call.n_relaxed > MAX_CLASSES
        || call.n_c > MAX_CLASSES
        || call.relaxed.len() != call.n_relaxed * CLASS_WORDS
    {
        return Err(bad("relaxed classes"));
    }
    if call.m.len() != call.offsets[12] || call.e.len() != call.m.len() || call.m.is_empty() {
        return Err(bad("weight pack"));
    }
    if call.cls.len() != (call.loop_max + 1) * (call.loop_max + 1) {
        return Err(bad("CLS"));
    }
    let words = &call.finalize;
    if words.len() != 135 {
        return Err(bad("finalize constants"));
    }
    let mut table = [0u64; 128];
    table.copy_from_slice(&words[..128]);
    Ok(PfFinalizeConstants {
        table,
        l2: words[128],
        c_t: words[129],
        kappa: words[130],
        b3: words[131],
        b5: words[132],
        b7: words[133],
        loop_max: words[134],
    })
}

/// F53 values of one thread's scratch for strands of at most `longest` letters: the
/// staged codes, then RA, RO, the accumulator rings and the column sums
/// (`sb_pfb_values`).
pub(crate) fn scratch_values(call: &PfBoundCall, longest: u64) -> u64 {
    let longest = longest.max(1);
    let lam = call.loop_max as u64;
    let rows = longest.min(lam + 2);
    let ro_rows = longest.min(U + 2);
    let nc = call.n_c as u64;
    let rel = call.n_relaxed as u64;
    CODE_VALUES + longest * (rows + ro_rows * nc + rel * 4 * ACC_ROWS + 1)
}

/// The kernel's constants (weight pack, CLS and relaxed words at the given device
/// addresses).
pub(crate) fn params(call: &PfBoundCall, w: u64, cls: u64, relaxed: u64) -> SbPfBound {
    let o = &call.offsets;
    SbPfBound {
        w,
        lam: call.loop_max as u64,
        n_p: call.n_p as u64,
        n_c: call.n_c as u64,
        cls,
        start: o[0] as u64,
        end: o[2] as u64,
        stack: o[3] as u64,
        bulge: o[4] as u64,
        i11: o[5] as u64,
        i12: o[6] as u64,
        i21: o[7] as u64,
        i22: o[8] as u64,
        gen: o[9] as u64,
        loop_out: o[10] as u64,
        loop_in: o[11] as u64,
        relaxed,
        n_relaxed: call.n_relaxed as u64,
        scale_down: u64::from(call.scale_down),
        ..SbPfBound::default()
    }
}

/// The weight pack as interleaved (M, e) words.
pub(crate) fn weight_words(call: &PfBoundCall) -> Vec<u64> {
    let mut flat = Vec::with_capacity(2 * call.m.len());
    for (&m, &e) in call.m.iter().zip(&call.e) {
        flat.push(m);
        flat.push(e as u64);
    }
    flat
}

/// Scratch budget of one launch (the 256 MiB slot rule of duplex-cuda.md §5.4).
const SCRATCH_BYTES: u64 = 256 << 20;
const THREADS: u64 = 64;

/// (Ẑ_b, L, outcome) of every item (`a[t]`, `b[t]` as letter codes 0..4).
pub(crate) fn eval(
    device: &Device,
    call: &PfBoundCall,
    a: &[&[u8]],
    b: &[&[u8]],
    threshold: Option<i64>,
) -> Result<PfBoundColumns, EngineError> {
    let count = a.len();
    if count == 0 {
        return Ok((Vec::new(), Vec::new(), Vec::new(), Vec::new()));
    }
    let finalize = check(call)?;
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
        let longest = a
            .iter()
            .chain(b.iter())
            .map(|s| s.len())
            .max()
            .unwrap_or(0)
            .max(1) as u64;
        let stride = scratch_values(call, longest);
        let per_thread = stride * 16;
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
        let pack = |e| fail("uploading the bound pack", e);
        let weights = up(&stream, &weight_words(call)).map_err(pack)?;
        let cls = up(&stream, &call.cls).map_err(pack)?;
        let relaxed = up(&stream, &call.relaxed).map_err(pack)?;
        let seqs = |e| fail("uploading sequences", e);
        let d_codes = up(&stream, &codes).map_err(seqs)?;
        let d_as = up(&stream, &a_start).map_err(seqs)?;
        let d_al = up(&stream, &a_len).map_err(seqs)?;
        let d_bs = up(&stream, &b_start).map_err(seqs)?;
        let d_bl = up(&stream, &b_len).map_err(seqs)?;
        // SAFETY: each thread initialises its own scratch before reading it (the column
        // sums are zeroed, every other band value is written before it is read); the
        // outputs are written for every item before the copies read them.
        let scratch = unsafe { dev_alloc::<u64>(&stream, (launched * stride * 2) as usize) }
            .map_err(|e| fail("allocating the bound scratch", e))?;
        let outs = |e| fail("allocating bound outputs", e);
        let out_m = unsafe { dev_alloc::<u64>(&stream, count) }.map_err(outs)?;
        let out_e = unsafe { dev_alloc::<i64>(&stream, count) }.map_err(outs)?;
        let out_l = unsafe { dev_alloc::<u64>(&stream, count) }.map_err(outs)?;
        let out_code = unsafe { dev_alloc::<u8>(&stream, count) }.map_err(outs)?;
        let mut params = params(
            call,
            ptr(&weights, &stream),
            ptr(&cls, &stream),
            ptr(&relaxed, &stream),
        );
        params.codes = ptr(&d_codes, &stream);
        params.a_start = ptr(&d_as, &stream);
        params.a_len = ptr(&d_al, &stream);
        params.b_start = ptr(&d_bs, &stream);
        params.b_len = ptr(&d_bl, &stream);
        params.n_items = count as u64;
        params.scratch = ptr(&scratch, &stream);
        params.stride = stride;
        params.has_theta = u64::from(threshold.is_some());
        params.theta = threshold.unwrap_or(0);
        params.out_m = ptr(&out_m, &stream);
        params.out_e = ptr(&out_e, &stream);
        params.out_l = ptr(&out_l, &stream);
        params.out_code = ptr(&out_code, &stream);
        let config = LaunchConfig {
            grid_dim: (blocks as u32, 1, 1),
            block_dim: (THREADS as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut launch = stream.launch_builder(&device.0.kernels.pf_bound);
        launch.arg(&params).arg(&finalize);
        // SAFETY: matches sb_pf_bound_eval; every array holds the sizes set above, and
        // `launched` scratch slots of `stride` F53 values exist.
        unsafe { launch.launch(config) }.map_err(|e| fail("launching sb_pf_bound_eval", e))?;
        let mut m = vec![0u64; count];
        let mut e = vec![0i64; count];
        let mut l = vec![0u64; count];
        let mut code = vec![0u8; count];
        stream
            .memcpy_dtoh(&out_m, &mut m[..])
            .and_then(|()| stream.memcpy_dtoh(&out_e, &mut e[..]))
            .and_then(|()| stream.memcpy_dtoh(&out_l, &mut l[..]))
            .and_then(|()| stream.memcpy_dtoh(&out_code, &mut code[..]))
            .and_then(|()| stream.synchronize())
            .map_err(|e| fail("device-to-host copy", e))?;
        Ok((m, e, l, code))
    })
}
