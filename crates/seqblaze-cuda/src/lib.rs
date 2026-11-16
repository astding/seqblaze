//! CUDA backend for seqblaze: device `hamming` (milestone G1), device
//! `levenshtein` (global, prefix, infix), device `align` (native local and
//! global, compat emboss-water and emboss-needle; keys on the device, detail
//! fields on the host), device `duplex` (K1 native MFE: keys and the opt-in
//! fields on the device; K3 native `model="pf"`: Ẑ on the device in the integer
//! software float F53, keys, terms and sinks on the host), the unary kernels
//! device `tm` (T1 native) with the shared soft-binary64 finalize (D55) and device
//! `hairpin` with the primer3 preset (T2), device `tm` on a Pairs input (the
//! register pairing on K1's device pack, int64 keys) and device `duplex` with the
//! primer3 preset (K2, int64 keys).
//!
//! Spec: docs/wiki/gpu/cuda-backend.md, levenshtein-cuda.md, align-cuda.md, duplex-cuda.md,
//! duplex-pf-cuda.md, tm-cuda.md, tm-pairs-cuda.md and primer3-cuda.md
//! (also hardware-and-toolchain.md and kernel-design.md for the later plan).
//! - Feature `cuda` (off by default) compiles `kernels/seqblaze.cu` with nvcc
//!   12.8/12.9 in `build.rs` into one fatbin of the architecture set named by
//!   `SEQBLAZE_CUDA_ARCHS` (`src/archsets.rs`; cuda-release-checks.md §2): `dev`
//!   (the default: SASS for sm_120 plus PTX for compute_120), `release` (D10:
//!   SASS for sm_75, sm_80, sm_86, sm_89, sm_90, sm_100 and sm_120 plus PTX for
//!   compute_120) or `ptx` (PTX for compute_120 only); [`nvcc_flags`] names the
//!   set that was built. The fatbin is embedded with `include_bytes!` and
//!   loaded at run time through cudarc's driver API with dynamic loading, so
//!   the extension has no libcuda `NEEDED` entry and
//!   libnvrtc is never opened.
//! - Determinism flags: `--fmad=false --prec-div=true --prec-sqrt=true
//!   --ftz=false`, never `--use_fast_math`. The kernels use integers and
//!   integer atomics only; float64 values are computed by the integer routines
//!   of `kernels/softf64.cuh` on bit patterns (K3's F53 by `kernels/f53.cuh`),
//!   and `build.rs` fails on any floating-point type in the module's PTX. Outputs are byte-identical to the CPU backend
//!   for every launch configuration.
//! - Without `cuda` this crate is a stub with the same API and no third-party
//!   dependency: `device()` returns a DeviceError ("built without CUDA").
//! - No CPU fallback: an explicit CUDA request with no build, driver or device
//!   is a DeviceError (D48 API §6). Only the caller's `device="auto"` placement
//!   stays on the CPU, and it never reaches this crate.
use seqblaze_core::{
    detail::DetailColumns,
    exec::ExecOptions,
    kernel::{PairDistanceKernel, Score},
    plan::BoundAvailability,
    sched::Workers,
    seq::SeqTable,
    sink::RunOutput,
    unary::{UnaryOptions, UnaryOutput},
    validate::{EngineError, ValidationReport},
    work::WorkSource,
};

#[cfg(feature = "cuda")]
pub mod detail;
pub mod interop;
mod launch;
#[cfg(feature = "cuda")]
pub mod resident;
#[cfg(feature = "cuda")]
pub mod resident_bound;
#[cfg(feature = "cuda")]
pub mod resident_edit_stage;
#[cfg(feature = "cuda")]
pub mod resident_indices;
#[cfg(feature = "cuda")]
pub mod resident_native;
#[cfg(feature = "cuda")]
pub mod resident_p3;
#[cfg(feature = "cuda")]
pub mod resident_producer;
#[cfg(feature = "cuda")]
pub mod resident_routes;
#[cfg(feature = "cuda")]
pub mod resident_run;
#[cfg(feature = "cuda")]
pub mod resident_selection;
#[cfg(feature = "cuda")]
pub mod resident_sinks;
#[cfg(feature = "cuda")]
pub mod resident_tiles;
#[cfg(feature = "cuda")]
pub mod resident_unary;
#[cfg(feature = "cuda")]
pub mod resident_validate;
#[cfg(feature = "cuda")]
pub mod resident_work;
#[cfg(feature = "testing")]
pub use launch::{
    cuda_poisoned_bytes, cuda_resident_poison_stats, cuda_unary_refill_stats,
    set_bound_retain_bytes, set_cuda_pf_capture, set_cuda_poison, set_cuda_resident_depth_cap,
    set_cuda_resident_launch, UnaryRefillStats,
};
pub use launch::{
    launch_override, set_launch_override, LaunchOverride, DEFAULT_BAND_BYTES,
    DEFAULT_BYTES_PER_LAUNCH, DEFAULT_CELLS_PER_LAUNCH, DEFAULT_DEVICE_WORDS, DEFAULT_GROUP_LANES,
    DEFAULT_LANE_ROWS, DEFAULT_PAIRS_PER_CHUNK, DEFAULT_PAIR_CELLS, DEFAULT_PAIR_STEPS,
    DEFAULT_PIECE_BYTES, DEFAULT_STEPS_PER_LAUNCH, DEFAULT_THREADS_PER_BLOCK, DEFAULT_THREAD_ROWS,
    DEFAULT_THREAD_WORDS, GROUP_LANES, LANE_ROWS, MAX_BAND_BYTES, MAX_BOUND_RETAIN_BYTES,
    MAX_DEVICE_WORDS, MAX_THREAD_ROWS, MAX_THREAD_WORDS, RESIDENT_DEPTH_CAP, RESIDENT_SLOTS_MAX,
};

#[cfg(feature = "cuda")]
mod aln;
#[cfg(feature = "cuda")]
mod arena;
#[cfg(feature = "cuda")]
mod device;
#[cfg(feature = "cuda")]
mod dup;
#[cfg(feature = "cuda")]
mod dupbound;
#[cfg(feature = "cuda")]
mod lev;
#[cfg(feature = "cuda")]
mod p3;
#[cfg(feature = "cuda")]
mod pack_cache;
#[cfg(feature = "cuda")]
mod pf;
#[cfg(feature = "cuda")]
mod pfbound;
#[cfg(feature = "cuda")]
mod pool;
#[cfg(feature = "cuda")]
mod reg;
// The PTX floating-point scan of build.rs (D55), here for its unit tests only.
#[cfg(test)]
mod ptx_fp;
// Host planning of the levenshtein and align stages; compiled in every build
// so that their unit tests run without a GPU.
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
mod plan;
#[cfg(feature = "cuda")]
mod session;
// Host staging of view tables; compiled in every build for its unit test.
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
mod staging;
#[cfg(feature = "cuda")]
mod unary;
/// Testing hooks of the per-call cost work (cuda-backend.md, "PC6 testing
/// hooks"): the owned pool, the host waits and the stream-use assertion.
#[cfg(all(feature = "cuda", feature = "testing"))]
pub mod testing_hooks {
    pub use crate::device::{
        device_synchronize, force_pool_fallback, pool_stats, pool_trim, PoolStats,
    };
    pub use crate::interop::execution_stream::{wait_stats, wait_stats_reset, WaitStats};
    pub use crate::pool::{pool_fails_fired, set_pool_fail};
    pub use crate::reg::set_register_defect;
    pub use crate::session::stream_misuse_count;
}

#[cfg(feature = "cuda")]
pub use {
    device::Device,
    reg::TileBatch64,
    session::{PfTileBatch, TileBatch, TileSession},
};

#[cfg(not(feature = "cuda"))]
mod stub;
#[cfg(not(feature = "cuda"))]
pub use stub::{Device, PfTileBatch, TileBatch, TileBatch64, TileSession};

/// Levenshtein mode (seqblaze-edit's `Mode`; this crate depends on core only).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EditMode {
    Global,
    Prefix,
    Infix,
}

/// The DP of a device align call (align-cuda.md §4, §5.1).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AlignKind {
    Local,
    Global,
    Water,
    Needle,
}

/// The scoring pack of a device align call (align-cuda.md §4), built on the
/// host once per call from the call's `Align` kernel: s(x, y) =
/// `table[class[x] * n_classes + class[y]]` for every byte pair, or, with
/// `matched = Some((match, mismatch))`, `match` if x = y else `mismatch`
/// (then `n_classes` is 0 and `table` empty). Fixed point, scale 1000; go and
/// ge in [0, 2^30].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AlignCall {
    pub kind: AlignKind,
    /// Global and needle end gaps (false for the local kinds).
    pub end_gaps: bool,
    pub go: i32,
    pub ge: i32,
    pub matched: Option<(i32, i32)>,
    pub class: [u8; 256],
    pub n_classes: usize,
    /// `n_classes` x `n_classes`, row-major.
    pub table: Vec<i32>,
}

/// The device pack of a K1 duplex call (duplex-cuda.md §4), built on the host once
/// per call from the call's `DuplexCall`, only when a device was resolved: Λ, the
/// score width, N_P, N_C and the dense CLS map (nndb2024-dna.md §5.4.2), every cell key
/// of the call in the CPU's flat layouts (tables concatenated in the order INIT,
/// INIT_SYM, END_L, END_R, STACK, BULGE, I11, I12, I21, I22, GEN, LOOP_OUT, LOOP_IN,
/// STATES; `offsets` holds each table's start, then the total; STATES has 200 keys or
/// none), the s/κ₄/κ₁ bit patterns and the kernel's field indices.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DuplexCall {
    pub loop_max: usize,
    /// int64 score width (B >= 2^26); else int32 with +∞ = 2^29.
    pub wide: bool,
    /// N_P (4 or 6) and N_C (1 or 3).
    pub n_p: usize,
    pub n_c: usize,
    /// CLS over (Λ + 1)²: the loop class at u·(Λ + 1) + v.
    pub cls: Vec<u8>,
    pub offsets: [usize; 15],
    /// D of every cell, ⊥ as +∞ of the width.
    pub score: Vec<i64>,
    /// (D, H, G, Q) of every cell, ⊥ as (2^60, 0, 0, 0).
    pub full: Vec<[i64; 4]>,
    pub s_bits: u64,
    pub kappa4_bits: u64,
    pub kappa1_bits: u64,
    /// Indices of `dh_kcal`, `ds_cal_per_k` and `tm_c` in the kernel's `detail_fields()`.
    pub fields: [usize; 3],
}

/// The device pack of a K3 `duplex(model="pf")` call (duplex-pf-cuda.md §4), built on
/// the host once per call from the call's `PfCall`, only when a device was resolved: Λ
/// and every weight entry as an F53 value (M, e), value M·2^(e − 52), zero as (0, 0).
/// Tables are concatenated in the order START (σ = 0), START (σ = 1), END, WSTACK,
/// WBULGE, WI11, WI12, WI21, WI22, WGEN, WOUT, WIN; `offsets` holds each table's start,
/// then the total.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PfDeviceCall {
    pub loop_max: usize,
    /// N_P, N_C and CLS as [`DuplexCall`]'s (nndb2024-dna.md §5.4.2).
    pub n_p: usize,
    pub n_c: usize,
    pub cls: Vec<u8>,
    pub offsets: [usize; 13],
    pub m: Vec<u64>,
    pub e: Vec<i64>,
}

/// The device pack of `tm` on a Pairs input (tm-pairs-cuda.md §4), built on the host once
/// per call from K1's `DuplexCall::device_pack()` of the call, only when a device was
/// resolved: Λ, the table offsets (in `DuplexCall`'s order, then the total), every full
/// cell key (D, H, G, Q) with ⊥ as (2^60, 0, 0, 0), the s/κ₄/κ₁ bit patterns and the
/// kernel's indices of T1's five fields. The score table is not needed (QTP4).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RegisterCall {
    pub loop_max: usize,
    /// N_P, N_C and CLS as [`DuplexCall`]'s (nndb2024-dna.md §5.4.2).
    pub n_p: usize,
    pub n_c: usize,
    pub cls: Vec<u8>,
    pub offsets: [usize; 15],
    pub full: Vec<[i64; 4]>,
    pub s_bits: u64,
    pub kappa4_bits: u64,
    pub kappa1_bits: u64,
    /// Indices of `tm_c`, `dg_kcal`, `dg_dcal`, `dh_kcal` and `ds_cal_per_k` (in this
    /// slot order, [`REG_SLOTS`]) in the kernel's `detail_fields()`.
    pub fields: [usize; REG_SLOTS],
}

/// The largest Λ of a K1 pack (the recipe maximum, duplex-mfe.md §12.2).
const PACK_MAX_LOOP: usize = 396;

/// N_P, N_C and CLS of a device pack (nndb2024-dna.md §5.4.2): N_P ∈ {4, 6}, N_C ∈ {1, 3},
/// Λ within the recipe maximum, CLS of (Λ + 1)² entries each below N_C.
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
fn pack_shape_ok(lam: usize, n_p: usize, n_c: usize, cls: &[u8]) -> bool {
    matches!(n_p, 4 | 6)
        && matches!(n_c, 1 | 3)
        && lam <= PACK_MAX_LOOP
        && cls.len() == (lam + 1) * (lam + 1)
        && cls.iter().all(|&c| usize::from(c) < n_c)
}

/// Whether `offsets` starts at 0 and steps by `sizes` (then holds the total).
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
fn offsets_step<const N: usize>(offsets: &[usize; N], sizes: &[usize]) -> bool {
    sizes.len() + 1 == N
        && offsets[0] == 0
        && sizes
            .iter()
            .enumerate()
            .all(|(k, &n)| offsets[k].checked_add(n) == Some(offsets[k + 1]))
}

/// The K1 table lengths in [`DuplexCall`]'s order (nndb2024-dna.md §5.3.1, §5.4.2) for Λ,
/// N_P, N_C and `states` STATES keys (0 or 200). Λ ≤ 396, so nothing overflows.
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
pub(crate) fn k1_table_sizes(lam: usize, n_p: usize, n_c: usize, states: usize) -> [usize; 14] {
    let pp = n_p * n_p;
    [
        1,
        1,
        25 * n_p,
        25 * n_p,
        pp,
        2 * lam * pp,
        16 * pp,
        64 * pp,
        64 * pp,
        256 * pp,
        (lam + 1) * (lam + 1),
        16 * n_c * n_p,
        16 * n_c * n_p,
        states,
    ]
}

/// The K3 table lengths in [`PfDeviceCall`]'s order (nndb2024-dna.md §5.4.2).
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
pub(crate) fn k3_table_sizes(lam: usize, n_p: usize, n_c: usize) -> [usize; 12] {
    let pp = n_p * n_p;
    [
        25 * n_p,
        25 * n_p,
        25 * n_p,
        pp,
        2 * lam * pp,
        16 * pp,
        64 * pp,
        64 * pp,
        256 * pp,
        (lam + 1) * (lam + 1),
        16 * n_c * n_p,
        16 * n_c * n_p,
    ]
}

/// The stage check of a K1 device pack (nndb2024-dna.md §5.4.2 "Stage checks"): the
/// shape, STATES of 0 or 200 keys and the offsets of every table.
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
fn k1_layout_ok(lam: usize, n_p: usize, n_c: usize, cls: &[u8], offsets: &[usize; 15]) -> bool {
    if !pack_shape_ok(lam, n_p, n_c, cls) {
        return false;
    }
    let states = offsets[14].wrapping_sub(offsets[13]);
    matches!(states, 0 | 200) && offsets_step(offsets, &k1_table_sizes(lam, n_p, n_c, states))
}

impl DuplexCall {
    /// The stage check of this pack (nndb2024-dna.md §5.4.2): shape and offsets.
    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    pub(crate) fn layout_ok(&self) -> bool {
        k1_layout_ok(self.loop_max, self.n_p, self.n_c, &self.cls, &self.offsets)
    }

    /// Whether the pack carries STATES keys (the K1 ρ prologue runs, q = 800).
    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    pub(crate) fn has_states(&self) -> bool {
        self.offsets[14] > self.offsets[13]
    }
}

impl PfDeviceCall {
    /// The stage check of this pack (nndb2024-dna.md §5.4.2): shape and offsets.
    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    pub(crate) fn layout_ok(&self) -> bool {
        pack_shape_ok(self.loop_max, self.n_p, self.n_c, &self.cls)
            && offsets_step(
                &self.offsets,
                &k3_table_sizes(self.loop_max, self.n_p, self.n_c),
            )
    }
}

impl RegisterCall {
    /// The stage check of this pack (nndb2024-dna.md §5.4.2): shape and offsets.
    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    pub(crate) fn layout_ok(&self) -> bool {
        k1_layout_ok(self.loop_max, self.n_p, self.n_c, &self.cls, &self.offsets)
    }
}

/// Output slots of device `tm` on Pairs, in T1's field order: `tm_c`, `dg_kcal`,
/// `dg_dcal`, `dh_kcal`, `ds_cal_per_k`; slot [`REG_INT32_SLOT`] is int32, the others float64.
pub const REG_SLOTS: usize = 5;
pub const REG_INT32_SLOT: usize = 2;

/// What the device computes for one call, beside the kernel object the host
/// uses for routed pairs and detail fields (levenshtein-cuda.md §8,
/// align-cuda.md §8).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DeviceCall {
    /// Byte Hamming distance (G1); no detail fields.
    Hamming,
    /// Unit-cost Levenshtein in `mode`. `b_end` and `n_ends` are the indices of
    /// those fields in the kernel's `detail_fields()`: the two the device
    /// computes in its forward pass when the request holds no other field.
    Levenshtein {
        mode: EditMode,
        b_end: usize,
        n_ends: usize,
        /// The call's legal byte set is a subset of {A, C, G, T} (`check_bases=True`,
        /// rejected otherwise before any numeric work): the resident runner's fast
        /// tiers then use the four-code kernels (levenshtein-cuda.md §13.4 step 2).
        /// Known from the options alone; the `Session` runner ignores it.
        acgt: bool,
    },
    /// Affine-gap alignment: the device computes the engine key of every
    /// pair; every detail field comes from the kernel's host `details` port.
    Align(std::sync::Arc<AlignCall>),
    /// K1 duplex MFE: the device computes the engine key of every pair and,
    /// in its detail pass, the three float64 fields of kept pairs.
    Duplex(std::sync::Arc<DuplexCall>),
    /// K3 duplex partition function: the device computes Ẑ of every pair; keys,
    /// terms and sinks run on the host (duplex-pf-cuda.md §7). Reached through the
    /// 64-bit-key entries ([`run_pf`], [`TileSession::bind_pf`]).
    DuplexPf(std::sync::Arc<PfDeviceCall>),
    /// `tm` on a Pairs input: the device computes the int64 key and every requested
    /// field of every pair (tm-pairs-cuda.md); run through the int64-key entries
    /// ([`run_i64`], `TileSession::bind_i64`), which put Each, Hits and RowStats on
    /// the device (K3's entries keep every sink on the host).
    TmPairs(std::sync::Arc<RegisterCall>),
    /// K2 duplex with the primer3 preset (primer3-cuda.md §7): the device computes the
    /// int64 key and the four detail fields of every pair in one pass. It shares the
    /// int64-key entries ([`run_i64`], `TileSession::bind_i64`) with `tm` on Pairs (the
    /// same sink placement: Each, Hits and RowStats on the device); its stage and sink
    /// kernels are its own (`p3.rs`, kernels/primer3.cuh).
    DuplexPrimer3(std::sync::Arc<Primer3Call>),
}

/// The bit patterns of every binary64 scalar the primer3 device replay reads
/// (primer3-cuda.md §4, QPC-6), mirrored by `SbP3Scalars` in kernels/primer3.cuh;
/// the per-call fields are σ, T_u and (K2) Q₄, Q₁, Θ_Ω at Q₄ and at Q₁.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct P3Scalars {
    pub t37: u64,
    pub phi: u64,
    pub fresh_h: u64,
    pub fresh_s: u64,
    pub s_cutoff: u64,
    pub tolerance: u64,
    pub margin: u64,
    pub alpha1: u64,
    pub at_h: u64,
    pub at_s: u64,
    pub omega_h: u64,
    pub omega_s: u64,
    pub k0: u64,
    pub kcal_div: u64,
    pub init_h: u64,
    pub init_s: u64,
    pub shift: u64,
    pub sigma: u64,
    pub t_u: u64,
    pub q4: u64,
    pub q1: u64,
    pub theta_omega4: u64,
    pub theta_omega1: u64,
}

/// The device pack of a T2 or K2 primer3 call (primer3-cuda.md §4), built on
/// the host once per call, only for a resolved device: the handle's tables as
/// (ΔH, ΔS) bit patterns in the device layout of seqblaze-thermo's
/// `p3files::device_layout`, the end-term tables (T2: one of 625 slots; K2: at
/// Q₄, then at Q₁) and the scalars. Never written to disk or logs.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Primer3Call {
    pub tables: Vec<u64>,
    pub ends: Vec<u64>,
    pub scalars: P3Scalars,
    /// K2 `end`: END2 swaps the operands; END1 and END2 select on the last row.
    pub swap: bool,
    pub end1: bool,
    /// K2: indices of `found`, `dh_kcal`, `ds_cal_per_k` and `tm_c` in the
    /// kernel's `detail_fields()` (T2: unused, its slots are its field order).
    pub fields: [usize; 4],
}

/// A device run: the shared output plus `execution.host_pairs`, the number of
/// distinct work items routed to the host CPU kernel (levenshtein-cuda.md §5.5).
pub struct DeviceRun<S: Score = i32> {
    pub output: RunOutput<S>,
    pub host_pairs: usize,
}

/// Whether this build contains the CUDA backend.
pub const COMPILED: bool = cfg!(feature = "cuda");

/// nvcc version that built the kernels, e.g. "12.8.61"; None without `cuda`.
pub fn nvcc_version() -> Option<&'static str> {
    option_env!("SEQBLAZE_NVCC_VERSION")
}

/// nvcc flags that built the kernels; None without `cuda`.
pub fn nvcc_flags() -> Option<&'static str> {
    option_env!("SEQBLAZE_NVCC_FLAGS")
}

/// Err("built without CUDA") when the backend is absent.
pub fn available() -> Result<(), &'static str> {
    if COMPILED {
        Ok(())
    } else {
        Err("built without CUDA")
    }
}

/// No bounded device path (G1, levenshtein-cuda.md §7): a kernel whose bound
/// exists reports `NotOnBackend` (fallback reason `device_no_bound`).
pub fn bound_availability(kernel: BoundAvailability) -> BoundAvailability {
    match kernel {
        BoundAvailability::Unavailable => BoundAvailability::Unavailable,
        _ => BoundAvailability::NotOnBackend,
    }
}

/// What a resolved device reports for the manifest.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DeviceInfo {
    pub ordinal: usize,
    pub name: String,
    /// The device's compute capability as `sm_<major><minor>`.
    pub arch: String,
}

/// Resolve `cuda:<ordinal>`: driver, device, primary context and kernel
/// module, shared process-wide. Transfers no data.
pub fn device(ordinal: usize) -> Result<Device, EngineError> {
    #[cfg(feature = "cuda")]
    {
        device::open(ordinal)
    }
    #[cfg(not(feature = "cuda"))]
    {
        Err(stub::unavailable(ordinal))
    }
}

/// Device compute for every non-Tiles sink, after `exec::validated` with the
/// same `kernel`. Each, Hits and RowStats run on the device; TopK and Matrix run
/// the shared host code on device distances. `kernel` computes routed pairs and
/// host detail fields. Always the full strategy (no device bound).
pub fn run(
    device: &Device,
    kernel: &dyn PairDistanceKernel,
    call: DeviceCall,
    work: &WorkSource,
    tables: (&SeqTable, &SeqTable),
    options: &ExecOptions,
    validation: ValidationReport,
) -> Result<DeviceRun, EngineError> {
    // Host work in E0 whatever the caller's FP state (§7.3 item 2).
    let _guard = seqblaze_core::fpenv::FpGuard::new();
    #[cfg(feature = "cuda")]
    {
        session::run(device, kernel, call, work, tables, options, validation)
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = (kernel, call, work, tables, options, validation);
        Err(stub::unavailable(device.info().ordinal))
    }
}

/// Device compute of K3 `duplex(model="pf")` for every sink but Tiles, after
/// `exec::validated` with the same `kernel` (duplex-pf-cuda.md §7): the device computes
/// Ẑ, the host the keys (`kernel.key_of_term`), the terms of `lse_dg_kcal` and every sink
/// through the engine core's batch drivers. `kernel` also computes routed pairs. Always
/// the full strategy (no bound). `call` must be [`DeviceCall::DuplexPf`].
pub fn run_pf(
    device: &Device,
    kernel: &dyn PairDistanceKernel<i64>,
    call: DeviceCall,
    work: &WorkSource,
    tables: (&SeqTable, &SeqTable),
    options: &ExecOptions<i64>,
    validation: ValidationReport,
) -> Result<DeviceRun<i64>, EngineError> {
    // Host work in E0 whatever the caller's FP state (§7.3 item 2).
    let _guard = seqblaze_core::fpenv::FpGuard::new();
    #[cfg(feature = "cuda")]
    {
        session::run_pf(device, kernel, call, work, tables, options, validation)
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = (kernel, call, work, tables, options, validation);
        Err(stub::unavailable(device.info().ordinal))
    }
}

/// Device compute of the int64-key pair kernels whose keys and fields come from the
/// device, for every non-Tiles sink, after `exec::validated` with the same `kernel`:
/// `tm` on Pairs (tm-pairs-cuda.md §6; `call` [`DeviceCall::TmPairs`]) and K2 `duplex`
/// with the primer3 preset (primer3-cuda.md §7; `call` [`DeviceCall::DuplexPrimer3`],
/// whose `kernel` also computes routed pairs). K3 has its own entry, [`run_pf`]. Each,
/// Hits and RowStats run on the device with each call's int64 sink kernels; TopK and
/// Matrix run the shared host code on device keys and fields. Always the full strategy
/// (no device bound).
pub fn run_i64(
    device: &Device,
    kernel: &dyn PairDistanceKernel<i64>,
    call: DeviceCall,
    work: &WorkSource,
    tables: (&SeqTable, &SeqTable),
    options: &ExecOptions<i64>,
    validation: ValidationReport,
) -> Result<DeviceRun<i64>, EngineError> {
    // Host work in E0 whatever the caller's FP state (§7.3 item 2).
    let _guard = seqblaze_core::fpenv::FpGuard::new();
    #[cfg(feature = "cuda")]
    {
        // One entry, two implementations: each call keeps its own stage and sinks.
        if matches!(call, DeviceCall::DuplexPrimer3(_)) {
            p3::run(device, kernel, call, work, tables, options, validation)
        } else {
            reg::run(device, kernel, call, work, tables, options, validation)
        }
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = (kernel, call, work, tables, options, validation);
        Err(stub::unavailable(device.info().ordinal))
    }
}

/// Where the primer3 device planner puts one item (primer3-cuda.md §6.6): T2
/// (`hairpin`) of length `l1`, or a K2 pair of operand lengths (`l1`, `l2`)
/// after the `end` mapping; `pair_cells` None is the launch default. Returns
/// (g, depth, routed). Testing aid; transfers nothing.
pub fn primer3_device_plan(
    hairpin: bool,
    l1: u64,
    l2: u64,
    group_lanes: u64,
    pair_cells: Option<u64>,
) -> (u64, u64, bool) {
    let settings = launch::Settings {
        group_lanes,
        pair_cells: pair_cells.unwrap_or(DEFAULT_PAIR_CELLS),
        ..launch::settings()
    };
    let item = if hairpin {
        plan::p3_plan_t2(&settings, l1)
    } else {
        plan::p3_plan_k2(&settings, l1, l2)
    };
    (item.g, item.depth, item.routed)
}

/// Ẑ of every work item of `work` on the device, (M, e) in item order, with the
/// session's launch settings and routing (duplex-pf-cuda.md §8 item 7; the testing hook
/// `duplex_pf_zhat`). Ẑ is in canonical orientation. `call` must be
/// [`DeviceCall::DuplexPf`]; the tables are validated by the caller. Testing only.
pub fn pf_zhat(
    device: &Device,
    kernel: &dyn PairDistanceKernel<i64>,
    call: DeviceCall,
    work: &WorkSource,
    tables: (&SeqTable, &SeqTable),
    workers: Workers,
) -> Result<(Vec<u64>, Vec<i64>), EngineError> {
    // Host work in E0 whatever the caller's FP state (§7.3 item 2).
    let _guard = seqblaze_core::fpenv::FpGuard::new();
    #[cfg(feature = "cuda")]
    {
        session::pf_zhat(device, kernel, call, work, tables, workers)
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = (kernel, call, work, tables, workers);
        Err(stub::unavailable(device.info().ordinal))
    }
}

/// The per-call constants of K1's C1 bound on the device (duplex-mfe-bound.md §8): the
/// D key of every cell of the device pack (+∞ = 2^60) in its table order with the table
/// starts, CLS, and the relaxed classes as words (32 per class: class, α + β, α − β,
/// the lower count and five (d0 + start, start, γ), the upper count and four
/// (start, d0 + start, γ)).
#[derive(Clone, Debug, Default)]
pub struct DupBoundCall {
    pub loop_max: usize,
    pub n_p: usize,
    pub n_c: usize,
    pub has_states: bool,
    pub cls: Vec<u8>,
    pub offsets: [usize; 15],
    pub keys: Vec<i64>,
    pub relaxed: Vec<i64>,
    pub n_relaxed: usize,
}

/// Per item: LB (INT64_MAX for +∞), χ (0/1) and the outcome code (0/1/2).
pub type BoundColumns = (Vec<i64>, Vec<u8>, Vec<u8>);

/// (LB, χ, outcome) of every item (`a[t]`, `b[t]` as letter codes) on the device
/// (duplex-mfe-bound.md §8): int64 LB (INT64_MAX for +∞), χ as 0/1 and the outcome code
/// against `threshold` (0 open, 1 certified, 2 pruned). `lb_plus_one`: the testing
/// fault. Byte-identical to the CPU's bound. Testing hook path.
pub fn duplex_bound_eval(
    device: &Device,
    call: &DupBoundCall,
    a: &[&[u8]],
    b: &[&[u8]],
    threshold: Option<i32>,
    lb_plus_one: bool,
) -> Result<BoundColumns, EngineError> {
    if a.len() != b.len() {
        return Err(EngineError::Parameter(
            "duplex bound: a and b differ in length".into(),
        ));
    }
    // Host work in E0 whatever the caller's FP state (§7.3 item 2).
    let _guard = seqblaze_core::fpenv::FpGuard::new();
    #[cfg(feature = "cuda")]
    {
        dupbound::eval(device, call, a, b, threshold, lb_plus_one)
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = (call, threshold, lb_plus_one);
        Err(stub::unavailable(device.info().ordinal))
    }
}

/// The per-call constants of K3's C2 bound on the device (duplex-pf-bound.md §8): K3's
/// device pack (the weight tables as [`PfDeviceCall`]'s) and the certificate's relaxed
/// classes as words (`seqblaze_thermo::duplex::pf_bound::DEVICE_CLASS_WORDS` per class:
/// class, x (M, e), z (M, e), then 2U + 2 region slots (used, M, e) in the order T_1 … T_U,
/// B_1 … B_U, LO, HI; each F53 as M · 2^(e − 52)), K3's finalize constants for L
/// (the 135 words of the resident finalize's constants: the 128 table entries, then ln 2,
/// c_T, κ, b3, b5, b7 and Λ, as binary64 bits) and the `zb_scale_down` testing fault.
#[derive(Clone, Debug, Default)]
pub struct PfBoundCall {
    pub loop_max: usize,
    pub n_p: usize,
    pub n_c: usize,
    pub cls: Vec<u8>,
    pub offsets: [usize; 13],
    pub m: Vec<u64>,
    pub e: Vec<i64>,
    pub relaxed: Vec<i64>,
    pub n_relaxed: usize,
    pub finalize: Vec<u64>,
    pub scale_down: bool,
}

/// Per item: Ẑ_b as (M, e) (value M · 2^(e − 52), (0, 0) for zero), the bits of L and the
/// outcome code (0 open, 1 certified, 2 pruned).
pub type PfBoundColumns = (Vec<u64>, Vec<i64>, Vec<u64>, Vec<u8>);

/// Ẑ_b, L and the outcome against `threshold` (an int64 engine key) of every item (`a[t]`,
/// `b[t]` as letter codes, in any orientation: the device takes the canonical one) on the
/// device (duplex-pf-bound.md §8), with the call's testing fault applied. Byte-identical
/// to the CPU's. Testing hook path.
pub fn duplex_pf_bound_eval(
    device: &Device,
    call: &PfBoundCall,
    a: &[&[u8]],
    b: &[&[u8]],
    threshold: Option<i64>,
) -> Result<PfBoundColumns, EngineError> {
    if a.len() != b.len() {
        return Err(EngineError::Parameter(
            "duplex pf bound: a and b differ in length".into(),
        ));
    }
    // Host work in E0 whatever the caller's FP state (§7.3 item 2).
    let _guard = seqblaze_core::fpenv::FpGuard::new();
    #[cfg(feature = "cuda")]
    {
        pfbound::eval(device, call, a, b, threshold)
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = (call, threshold);
        Err(stub::unavailable(device.info().ordinal))
    }
}

/// An F53 operation of the testing kernel `sb_f53_eval` (duplex-pf-cuda.md §5.5).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum F53Op {
    Add,
    Mul,
}

impl F53Op {
    pub fn parse(name: &str) -> Option<Self> {
        match name {
            "add" => Some(Self::Add),
            "mul" => Some(Self::Mul),
            _ => None,
        }
    }
}

/// Whether (M, e) is an F53 operand of the device's domain (duplex-pf-cuda.md §5.1,
/// §5.4): (0, 0), or 2^52 <= M < 2^53 and |e| < 2^60 + 403.
pub fn f53_canonical(m: u64, e: i64) -> bool {
    (m == 0 && e == 0)
        || ((1u64 << 52..1u64 << 53).contains(&m) && e.unsigned_abs() < (1u64 << 60) + 403)
}

/// Evaluate `op` elementwise on the device (duplex-pf-cuda.md §8 item 7): operands as
/// (M, e) arrays of one length, each canonical ([`f53_canonical`]; otherwise a
/// ParameterError naming the element). Testing only.
pub fn f53_eval(
    device: &Device,
    op: F53Op,
    x: (&[u64], &[i64]),
    y: (&[u64], &[i64]),
) -> Result<(Vec<u64>, Vec<i64>), EngineError> {
    let n = x.0.len();
    if x.1.len() != n || y.0.len() != n || y.1.len() != n {
        return Err(EngineError::Parameter(
            "cuda_f53 needs x_m, x_e, y_m and y_e of one length".into(),
        ));
    }
    for (name, (m, e)) in [("x", x), ("y", y)] {
        if let Some(k) = (0..n).find(|&k| !f53_canonical(m[k], e[k])) {
            return Err(EngineError::Parameter(format!(
                "cuda_f53 operand {name}[{k}] = (M = {}, e = {}) is not an F53 value: zero is \
                 (0, 0), every other value has 2^52 <= M < 2^53 and |e| < 2^60 + 403",
                m[k], e[k]
            )));
        }
    }
    // Host work in E0 whatever the caller's FP state (§7.3 item 2).
    let _guard = seqblaze_core::fpenv::FpGuard::new();
    #[cfg(feature = "cuda")]
    {
        pf::f53_eval(device, op == F53Op::Mul, x, y)
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = op;
        Err(stub::unavailable(device.info().ordinal))
    }
}

/// One (h, g, d) entry of the tm pack: ΔH, ΔG37 and G_e(T), dcal/mol.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TmStep {
    pub h: i64,
    pub g: i64,
    pub d: i64,
}

/// The device pack of T1 native `tm` (tm-cuda.md §3.1), mirrored by `SbTmPack`
/// in kernels/tm.cuh: 480 bytes of 8-byte fields, passed by value. Float64
/// constants travel as their bit patterns (`f64::to_bits`).
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TmPack {
    /// Index 4·code(X) + code(Y) of the top-strand step XY (A 0, C 1, G 2, T 3).
    pub step: [TmStep; 16],
    pub init: TmStep,
    pub terminal_at: TmStep,
    pub symmetry: TmStep,
    pub s: u64,
    pub kappa4: u64,
    pub kappa1: u64,
}

/// Output slots of device tm, in the order of tm's fields (tm.md §6): `tm_c`,
/// `dg_kcal`, `dg_dcal`, `dh_kcal`, `ds_cal_per_k`. A requested column's
/// kernel field index is its slot.
pub const TM_SLOTS: usize = 5;
/// Whether slot k of device tm is an int32 column (else float64).
pub const TM_INT32_SLOTS: [bool; TM_SLOTS] = [false, false, true, false, false];

/// The device pack of T1 `tm` `compat="biopython"` (tm-compat-cuda.md §3.3),
/// mirrored by `SbTmBiopythonPack` in kernels/tm_compat.cuh: 41 fields of 8 bytes,
/// every binary64 as its bit pattern, passed by value.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TmBiopythonPack {
    /// (H[k], S[k]) at k = 4·code(X) + code(Y) of the top-strand step XY.
    pub step: [[u64; 2]; 16],
    /// H_A, S_A, H_G, S_G (S4, S5).
    pub ends: [u64; 4],
    /// c_s, λ_m, ρ, κ, K₀ (F1-F6).
    pub salt: u64,
    pub ln_mon: u64,
    pub rho: u64,
    pub kcal_to_cal: u64,
    pub k0: u64,
}

/// The device pack of T1 `tm` `compat="primer3"` (tm-compat-cuda.md §3.4),
/// mirrored by `SbTmPrimer3Pack` in kernels/tm_compat.cuh: 54 fields of 8 bytes
/// (integer tallies as int64, every binary64 as its bit pattern), by value.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TmPrimer3Pack {
    /// (τ_H, τ_S) at k = 4·code(X) + code(Y) of the top-strand step XY.
    pub step: [[i64; 2]; 16],
    /// (τ_H, τ_S) of the A·T terminus, the G·C terminus and the symmetry entry.
    pub ends: [i64; 6],
    /// M: the long path for L > M.
    pub max_nn_length: i64,
    /// c_s, λ, γ₄, γ₁, δ_D, φ_F, B, Q.
    pub salt: u64,
    pub ln_mu: u64,
    pub gamma4: u64,
    pub gamma1: u64,
    pub dmso_t: u64,
    pub phi_formamide: u64,
    pub base: u64,
    pub sig10: u64,
    /// κ_H, κ_S, a_F, b_F, a_G, a_L, K₀.
    pub tally_h: u64,
    pub tally_s: u64,
    pub formamide_gc: u64,
    pub formamide_c: u64,
    pub long_gc: u64,
    pub long_len: u64,
    pub k0: u64,
}

const _: () = assert!(std::mem::size_of::<TmBiopythonPack>() == 328);
const _: () = assert!(std::mem::size_of::<TmPrimer3Pack>() == 432);

/// A unary kernel the device runs (tm-cuda.md §5.4): one stage kernel and its
/// pack.
// One value per call (built once, never stored in bulk), so the inline TmPack
// costs nothing and keeps T1's call sites unchanged.
#[allow(clippy::large_enum_variant)]
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum UnaryCall {
    Tm(TmPack),
    /// `tm` `compat="biopython"` (tm-compat-cuda.md §4.1, stage `sb_tm_biopython`).
    TmBiopython(TmBiopythonPack),
    /// `tm` `compat="primer3"` (tm-compat-cuda.md §4.2, stage `sb_tm_primer3`).
    TmPrimer3(TmPrimer3Pack),
    /// T2 `hairpin` with the primer3 preset (primer3-cuda.md §6.2): output slots
    /// in the order of its fields (`found` int32, then four float64).
    Hairpin(std::sync::Arc<Primer3Call>),
}

/// Device compute of a unary kernel over a validated table (after
/// `unary::validated` with the CPU kernel of the call, which supplies the
/// errors and `template`, its empty requested columns in output order). Each
/// and Hits (with a cutoff, compared `primary >= cutoff`) run on the device;
/// n = 0 must not reach here (the caller returns the CPU zero-work output).
pub fn run_unary(
    device: &Device,
    call: &UnaryCall,
    table: &SeqTable,
    template: DetailColumns,
    options: &UnaryOptions<f64>,
    validation: ValidationReport,
) -> Result<UnaryOutput, EngineError> {
    // Host work in E0 whatever the caller's FP state (§7.3 item 2).
    let _guard = seqblaze_core::fpenv::FpGuard::new();
    #[cfg(feature = "cuda")]
    {
        unary::run(device, call, table, template, options, validation)
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = (call, table, template, options, validation);
        Err(stub::unavailable(device.info().ordinal))
    }
}

/// A soft-binary64 operation of the testing kernel `sb_f64_eval` (P5).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum F64Op {
    Add,
    Sub,
    Mul,
    Div,
    Ge,
    FromI64,
    Lt,
    Le,
    Gt,
    Eq,
    IsFinite,
    Abs,
}

impl F64Op {
    pub fn parse(name: &str) -> Option<Self> {
        Some(match name {
            "add" => Self::Add,
            "sub" => Self::Sub,
            "mul" => Self::Mul,
            "div" => Self::Div,
            "ge" => Self::Ge,
            "from_i64" => Self::FromI64,
            "lt" => Self::Lt,
            "le" => Self::Le,
            "gt" => Self::Gt,
            "eq" => Self::Eq,
            "is_finite" => Self::IsFinite,
            "abs" => Self::Abs,
            _ => return None,
        })
    }

    /// Whether the operation reads only `a` (`b` must be None).
    pub fn unary(self) -> bool {
        matches!(self, Self::FromI64 | Self::IsFinite | Self::Abs)
    }

    /// Whether the result is a predicate (0 or 1).
    pub fn predicate(self) -> bool {
        matches!(
            self,
            Self::Ge | Self::Lt | Self::Le | Self::Gt | Self::Eq | Self::IsFinite
        )
    }
}

/// Evaluate `op` elementwise on the device (tm-cuda.md §10 item 11,
/// primer3-cuda.md §5): u64 bit patterns in and out; the predicates return 0 or
/// 1; `FromI64` reads `a` as int64 bits; the unary ops take no `b`. Testing only.
pub fn f64_eval(
    device: &Device,
    op: F64Op,
    a: &[u64],
    b: Option<&[u64]>,
) -> Result<Vec<u64>, EngineError> {
    // Host work in E0 whatever the caller's FP state (§7.3 item 2).
    let _guard = seqblaze_core::fpenv::FpGuard::new();
    #[cfg(feature = "cuda")]
    {
        unary::f64_eval(device, op, a, b)
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = (op, a, b);
        Err(stub::unavailable(device.info().ordinal))
    }
}
