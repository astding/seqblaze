//! Driver, device and module resolution (docs/wiki/gpu/cuda-backend.md,
//! "Runtime: driver, device, errors"). One primary context and one module per
//! ordinal and process, shared by all threads; failures are not cached.
use std::{
    panic::{catch_unwind, AssertUnwindSafe},
    sync::{Arc, Mutex},
};

use cudarc::{
    driver::{result, sys, CudaContext, CudaFunction, CudaModule, CudaStream, DriverError},
    nvrtc::Ptx,
};
use seqblaze_core::{fpenv::FpGuard, validate::EngineError};

use crate::DeviceInfo;

static FATBIN: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/seqblaze.fatbin"));

/// What the embedded fatbin holds, read from the `-gencode` pairs of the nvcc flags the
/// build recorded (`SEQBLAZE_NVCC_FLAGS`, which names the architecture set `dev`,
/// `release` or `ptx` of docs/wiki/gpu/cuda-release-checks.md §2.1), e.g. "SASS for
/// sm_120 and PTX for compute_120".
fn fatbin_contents() -> String {
    let (sass, ptx): (Vec<&str>, Vec<&str>) = crate::nvcc_flags()
        .unwrap_or_default()
        .split_whitespace()
        .filter_map(|token| token.split_once(",code=").map(|(_, code)| code))
        .partition(|code| code.starts_with("sm_"));
    let list = |codes: &[&str]| match codes {
        [] => "no architecture".to_owned(),
        _ => codes.join(", "),
    };
    format!("SASS for {} and PTX for {}", list(&sass), list(&ptx))
}

pub(crate) struct Kernels {
    pub resident_unary_keys: CudaFunction,
    pub selection_normalize: CudaFunction,
    pub selection_mask_counts: CudaFunction,
    pub selection_mask_positions: CudaFunction,
    pub selection_numeric: CudaFunction,
    pub selection_ragged_lengths: CudaFunction,
    pub selection_ragged_copy: CudaFunction,
    pub selection_topk_count: CudaFunction,
    pub selection_topk_flags: CudaFunction,
    pub selection_rowstats_counts: CudaFunction,
    pub selection_rowstats_pairs: CudaFunction,
    pub resident_edit_classify: CudaFunction,
    pub resident_edit_compact: CudaFunction,
    pub resident_edit_scatter: CudaFunction,

    pub resident_first_rows: CudaFunction,
    pub resident_pair_check: CudaFunction,
    pub resident_pair_error: CudaFunction,

    pub resident_dense_entries: CudaFunction,
    pub detail_plan: CudaFunction,
    pub detail_continue: CudaFunction,
    pub detail_progress: CudaFunction,

    pub resident_hamming_counts: CudaFunction,
    pub resident_hamming_pieces: CudaFunction,
    pub resident_hamming_groups: CudaFunction,
    pub resident_primary: CudaFunction,

    pub pf_resident_finalize: CudaFunction,
    pub pf_rows_clear: CudaFunction,
    pub pf_rows_link: CudaFunction,
    pub pf_rows_add: CudaFunction,
    pub pf_rows_grow: CudaFunction,
    pub pf_rows_finish: CudaFunction,
    pub k1_lse_window: CudaFunction,
    pub k1_lse_gather: CudaFunction,

    pub resident_p3_plan: CudaFunction,
    pub resident_p3_flags: CudaFunction,
    pub resident_p3_positions: CudaFunction,
    pub resident_p3_hairpin: [CudaFunction; 4],
    pub resident_p3_dimer: [CudaFunction; 4],
    pub resident_p3_init: CudaFunction,
    pub resident_p3_fill: CudaFunction,
    pub resident_p3_select: CudaFunction,
    pub resident_p3_finish: CudaFunction,
    pub resident_p3_columns: CudaFunction,

    pub resident_route_flags_dup: CudaFunction,
    pub resident_route_flags_pf: CudaFunction,
    pub resident_route_positions: CudaFunction,
    pub resident_dup_init: CudaFunction,
    pub resident_dup_step: CudaFunction,
    pub resident_dup_finish: CudaFunction,
    pub resident_pf_init: CudaFunction,
    pub resident_pf_step: CudaFunction,
    pub resident_pf_finish: CudaFunction,
    /// The C1 bounded K1 resident stage (kernels/resident_bound.cuh,
    /// duplex-mfe-bound.md §8): per-item outcomes, the decision probe, replay
    /// eligibility of probe items, multiplicities of canonical pairs, and a zero fill.
    pub resident_dup_bound: CudaFunction,
    pub resident_dup_probe: CudaFunction,
    /// The C2 bounded K3 resident stage (kernels/resident_pf_bound.cuh,
    /// duplex-pf-bound.md §8): per-item outcomes and the decision probe.
    pub resident_pf_bound: CudaFunction,
    pub resident_pf_probe: CudaFunction,
    pub resident_probe_lengths: CudaFunction,
    pub resident_probe_dups: CudaFunction,
    pub resident_pair_mult: CudaFunction,
    pub resident_bound_zero: CudaFunction,

    pub input_snapshot_lengths: CudaFunction,
    pub input_snapshot_bytes: CudaFunction,
    pub digest_row_bytes: CudaFunction,
    pub digest_leaf_roots: CudaFunction,
    pub digest_table_tree_level: CudaFunction,
    pub digest_tree_level: CudaFunction,
    pub digest_sequence_table: CudaFunction,
    pub digest_index_chunks: CudaFunction,
    pub digest_pair_index: CudaFunction,
    pub input_init: CudaFunction,
    pub input_small: CudaFunction,
    pub input_init_digest: CudaFunction,
    pub input_alphabet_rows: CudaFunction,
    pub digest_sequence_small: CudaFunction,
    pub input_scan_u64: CudaFunction,
    pub input_scan_add: CudaFunction,
    pub input_alphabet: CudaFunction,
    pub input_reduce_rows: CudaFunction,
    pub input_reduce_summaries: CudaFunction,
    pub pieces: CudaFunction,
    pub each: CudaFunction,
    pub hits_count: CudaFunction,
    pub hits_write: CudaFunction,
    pub rowstats_min: CudaFunction,
    pub rowstats_arg: CudaFunction,
    /// The int64-key instances of the Hits and RowStats sinks and the int64 fill
    /// (tm-pairs-cuda.md §7).
    pub hits_count_i64: CudaFunction,
    pub hits_write_i64: CudaFunction,
    pub rowstats_min_i64: CudaFunction,
    pub rowstats_arg_i64: CudaFunction,
    pub fill_i64: CudaFunction,
    /// `tm` on Pairs (kernels/register.cuh): the stage kernel.
    pub reg: CudaFunction,
    /// Levenshtein (kernels/levenshtein.cuh): thread tier, warp tier for
    /// Q = 1, 2, 4, 8 words per lane, each in a clean (four-code) and a raw
    /// (slot) form; routing count/write and the scatter of host results.
    pub lev_thread: [CudaFunction; 2],
    pub lev_warp: [[CudaFunction; 2]; 4],
    pub lev_route_count: CudaFunction,
    pub lev_route_write: CudaFunction,
    pub lev_scatter: CudaFunction,
    /// Align (kernels/align.cuh): the closed-form kernel, the thread tier and
    /// the warp tier for q = 1, 2, 4, 8 rows per lane, each per kind (local,
    /// global, water, needle), and the routing count/write. Routed results
    /// use `lev_scatter` with no field buffers.
    pub aln_closed: CudaFunction,
    pub aln_thread: [CudaFunction; 4],
    pub aln_warp: [[CudaFunction; 4]; 4],
    pub aln_route_count: CudaFunction,
    pub aln_route_write: CudaFunction,
    /// Unary kernels (tm-cuda.md §5.2): the tm stage, the generic Hits
    /// compaction, and the soft-binary64 testing kernel (P5).
    pub tm: CudaFunction,
    /// Duplex (kernels/duplex.cuh): the closed-form kernel, the DP per key
    /// width (narrow, wide, detail) and group size (4, 8, 16, 32), the
    /// routing count/write and the scatter of host results.
    pub dup_closed: CudaFunction,
    pub dup: [[CudaFunction; 4]; 3],
    pub dup_route_count: CudaFunction,
    pub dup_route_write: CudaFunction,
    pub dup_scatter: CudaFunction,
    /// The C1 bound of K1 (kernels/duplex_bound.cuh, duplex-mfe-bound.md §8).
    pub dup_bound: CudaFunction,
    /// The C2 bound of K3 (kernels/duplex_pf_bound.cuh, duplex-pf-bound.md §8).
    pub pf_bound: CudaFunction,
    pub unary_hits_count: CudaFunction,
    pub unary_hits_write: CudaFunction,
    pub f64_eval: CudaFunction,
    /// K3 duplex pf (kernels/duplex_pf.cuh): the closed-form kernel, the DP per group
    /// size (4, 8, 16, 32; the tier is a parameter), the routing count/write, and the
    /// F53 testing kernel (kernels/f53.cuh).
    pub pf_closed: CudaFunction,
    pub pf: [CudaFunction; 4],
    pub pf_route_count: CudaFunction,
    pub pf_route_write: CudaFunction,
    pub f53_eval: CudaFunction,
    /// The resident fast entries with sparse pair predecessors (kernels/duplex_sparse.cuh,
    /// duplex-cuda.md §15.4 step 3, duplex-pf-cuda.md §19.4 step 2): K1 narrow, wide and
    /// detail, and K3 (global and shared tier).
    pub dup_res: [CudaFunction; 3],
    pub pf_res: CudaFunction,
    /// The tm compatibility presets' unary stages (kernels/tm_compat.cuh,
    /// tm-compat-cuda.md §5.1).
    pub tm_biopython: CudaFunction,
    pub tm_primer3: CudaFunction,
    /// Primer3 presets (kernels/primer3.cuh): T2 and K2 per group size (4, 8,
    /// 16, 32), the int64-key sinks and the scatter of routed K2 results.
    pub p3_hairpin: [CudaFunction; 4],
    pub p3_dimer: [CudaFunction; 4],
    pub each64: CudaFunction,
    pub hits_count64: CudaFunction,
    pub hits_write64: CudaFunction,
    pub rowstats_min64: CudaFunction,
    pub rowstats_arg64: CudaFunction,
    pub p3_scatter: CudaFunction,
}

pub(crate) struct Inner {
    pub resident_sinks: crate::resident_sinks::Kernels,
    pub resident_indices: crate::resident_indices::Kernels,
    pub resident_tiles: crate::resident_tiles::Kernels,
    pub ctx: Arc<CudaContext>,
    pub kernels: Kernels,
    pub info: DeviceInfo,
    /// Threads resident at once: SMs × max threads per SM (the smallest
    /// levenshtein tier grid, levenshtein-cuda.md §5.4).
    pub resident_threads: u64,
    /// Largest block (a multiple of 32) every align thread-tier and every
    /// warp-tier kernel can launch with, from their register use; the align
    /// stage caps `threads_per_block` with it (launch parameters never
    /// change results).
    pub aln_block: [u32; 2],
    /// Duplex: the largest block (a multiple of 32) of each key width's DP
    /// kernels, the opt-in per-block shared memory S_max and the largest
    /// static shared memory s_block of those kernels (duplex-cuda.md §5.4).
    pub dup_block: [u32; 3],
    pub dup_s_max: u64,
    pub dup_s_block: u64,
    /// K3: the largest block (a multiple of 32) of the DP kernels and their largest
    /// static shared memory (duplex-pf-cuda.md §6.3); S_max is `dup_s_max`.
    pub pf_block: u32,
    pub pf_s_block: u64,
    /// s_block of K3's resident sparse entry (duplex-pf-cuda.md §19.4 step 3).
    pub pf_res_s_block: u64,
    /// Primer3: the largest block (a multiple of 32) of the T2 and the K2 kernels.
    pub p3_block: [u32; 2],
    /// Primer3 resident fast kernels (primer3-cuda.md §22.4 steps 3-4): the SM count
    /// and each kernel's resident 32-thread blocks per SM from the driver's occupancy
    /// query (static shared memory included), `[family][variant]`, family 0 T2 and 1
    /// K2, variants g = 4, 8, 16, 32. Launch sizing only; never a result.
    pub p3_sms: u64,
    pub p3_wave_blocks: [[u64; 4]; 2],
    /// Resident K1/K3 fast paths (duplex-cuda.md §15.4 step 1, duplex-pf-cuda.md §19.4
    /// step 1): G_res of each fast kernel, SMs × resident 128-thread blocks per SM × 4,
    /// for 4 × stride bytes of dynamic shared memory, `[kernel][q]` with the kernels of
    /// `resident_routes::FastKernel` and q = 400, 800. Launch sizing only; never a result.
    pub resident_wave: [[u64; 2]; 4],
    /// K3's shared tier (duplex-pf-cuda.md §19.4 step 3): N_sh per dynamic size d of a
    /// one-warp block, queried on first use and kept for the device's lifetime.
    pub pf_shared_wave: Mutex<Vec<(u64, u64)>>,
    /// Native tm, Biopython tm and primer3 tm: each stage's largest legal block,
    /// also bounded by both shared unary Hits kernels (a multiple of 32).
    pub unary_block: [u32; 3],
    /// Ordering events reused by every `order()` on this ordinal (S1 (c)).
    pub events: crate::pool::EventList,
    /// The seqblaze-owned allocation pool (S2); None in the fallback mode.
    pub pool: Option<crate::pool::DevicePool>,
    /// The pinned bounce buffer of batched host exports (S8).
    pub bounce: crate::pool::BounceSlot,
    pub pack_cache: crate::pack_cache::Cache,
    _module: Arc<CudaModule>,
    // Keep cudarc in multi-stream mode even between calls. Otherwise its next
    // new_stream() calls cuCtxSynchronize, including unrelated foreign streams.
    // With event tracking left off (S1) this is redundant but harmless.
    _stream_anchor: Arc<CudaStream>,
}

/// A resolved CUDA device: primary context plus the loaded seqblaze module.
#[derive(Clone)]
pub struct Device(pub(crate) Arc<Inner>);

impl Device {
    pub fn info(&self) -> &DeviceInfo {
        &self.0.info
    }

    /// W₃₂ of primer3-cuda.md §22.4 step 3: SMs × resident warps per SM of the
    /// g = 32 resident fast kernel, `[T2, K2]`. Launch planning only.
    pub fn p3_wave(&self) -> [u64; 2] {
        [0, 1].map(|family| self.0.p3_sms * self.0.p3_wave_blocks[family][3])
    }

    /// What the resident K1/K3 stages size their launches with (duplex-cuda.md §15.4
    /// step 1): with `shared_band` 0, G_res of fast kernel `kernel` (0 narrow, 1 wide,
    /// 2 detail, 3 pf) at q = `staged` (400 or 800); with `shared_band` = b > 0 (pf only),
    /// K3's N_sh for one-warp blocks of q + lists + b bytes (duplex-pf-cuda.md §19.4
    /// step 3), 0 when that size exceeds S_max. Launch planning only.
    pub fn resident_wave(
        &self,
        kernel: usize,
        staged: u64,
        shared_band: u64,
    ) -> Result<u64, EngineError> {
        let q = match staged {
            400 => 0,
            800 => 1,
            _ => {
                return Err(EngineError::Parameter(format!(
                    "staged={staged}; use 400 or 800"
                )))
            }
        };
        if kernel > 3 {
            return Err(EngineError::Parameter(
                "kernel must be dup_narrow, dup_wide, dup_detail or pf".into(),
            ));
        }
        if shared_band == 0 {
            return Ok(self.0.resident_wave[kernel][q]);
        }
        if kernel != 3 {
            return Err(EngineError::Parameter(
                "shared_band applies to kernel='pf' only".into(),
            ));
        }
        crate::resident_routes::pf_shared_wave(self, staged, shared_band)
    }
}

static DEVICES: Mutex<Vec<Option<Arc<Inner>>>> = Mutex::new(Vec::new());

/// The CUDA error name, e.g. `CUDA_ERROR_OUT_OF_MEMORY`.
pub(crate) fn error_name(error: DriverError) -> String {
    format!("{:?}", error.0)
}

/// Run cudarc calls with panics (missing library or symbol, broken
/// invariants) mapped to DeviceError: no partial result, no CPU fallback.
/// Every CUDA host entry point (`run`, `run_unary`, `f64_eval`, `TileSession`
/// creation and each bound tile call) runs through here, in the IEEE default
/// floating-point environment; the caller's state is restored on every exit
/// (text-nulls-fp-environment.md §7.3 item 2).
pub(crate) fn guard<T>(f: impl FnOnce() -> Result<T, EngineError>) -> Result<T, EngineError> {
    let _fp = FpGuard::new();
    catch_unwind(AssertUnwindSafe(f)).unwrap_or_else(|payload| {
        let message = payload
            .downcast_ref::<&str>()
            .map(|text| (*text).to_owned())
            .or_else(|| payload.downcast_ref::<String>().cloned())
            .unwrap_or_else(|| "unknown panic".into());
        Err(EngineError::Device(format!(
            "the CUDA backend failed: {message}; no result was produced; use device='cpu'"
        )))
    })
}

pub(crate) fn open(ordinal: usize) -> Result<Device, EngineError> {
    guard(|| {
        let mut devices = DEVICES.lock().unwrap_or_else(|poison| poison.into_inner());
        if let Some(Some(inner)) = devices.get(ordinal) {
            return Ok(Device(Arc::clone(inner)));
        }
        let label = format!("cuda:{ordinal}");
        // cudarc's `culib()` panics when no library is found, so check first.
        // SAFETY: only probes dlopen on cudarc's candidate names.
        if !unsafe { sys::is_culib_present() } {
            return Err(EngineError::Device(format!(
                "device='{label}': the CUDA driver library (libcuda) could not be loaded; under \
                 WSL put /usr/lib/wsl/lib first on LD_LIBRARY_PATH (docs/wiki/gpu/cuda-backend.md), \
                 or use device='cpu'"
            )));
        }
        result::init().map_err(|error| {
            EngineError::Device(format!(
                "device='{label}': cuInit failed with {}; under WSL put /usr/lib/wsl/lib first \
                 on LD_LIBRARY_PATH (docs/wiki/gpu/cuda-backend.md), or use device='cpu'",
                error_name(error)
            ))
        })?;
        let count = result::device::get_count().map_err(|error| {
            EngineError::Device(format!(
                "device='{label}': cuDeviceGetCount failed with {}; use device='cpu'",
                error_name(error)
            ))
        })?;
        let count = usize::try_from(count).unwrap_or(0);
        if ordinal >= count {
            let hint = if count > 0 {
                "device='cuda:0'"
            } else {
                "device='cpu'"
            };
            return Err(EngineError::Device(format!(
                "device='{label}': CUDA_ERROR_INVALID_DEVICE, this machine has {count} CUDA \
                 device(s); use {hint}"
            )));
        }
        let failed = |what: &str, error: DriverError| {
            EngineError::Device(format!(
                "device='{label}': {what} failed with {}; use device='cpu'",
                error_name(error)
            ))
        };
        let ctx = CudaContext::new(ordinal)
            .map_err(|error| failed("creating the primary context", error))?;
        // SAFETY: this context wrapper is not published yet and has no slices
        // or submitted work. Event tracking stays off for the process
        // (cuda-backend.md, per-call cost S1, Q7): every slice is allocated on,
        // and reached through, exactly one cudarc stream (the owned stream of a
        // resident lease or the single stream of a private session), so
        // tracking's waits and records would be same-stream no-ops; testing
        // builds count any other access (`stream_misuse_count`). Cross-stream
        // ordering is the explicit `order()`/`complete()` handoff. The empty
        // anchor keeps cudarc in multi-stream mode.
        unsafe { ctx.disable_event_tracking() };
        let stream_anchor = ctx.new_stream();
        let stream_anchor = stream_anchor
            .map_err(|error| failed("creating the allocation stream anchor", error))?;
        let name = ctx
            .name()
            .map_err(|error| failed("cuDeviceGetName", error))?;
        let (major, minor) = ctx
            .compute_capability()
            .map_err(|error| failed("reading the compute capability", error))?;
        let arch = format!("sm_{major}{minor}");
        let attribute = |attribute, what: &str| {
            ctx.attribute(attribute)
                .map(|value| u64::try_from(value).unwrap_or(0).max(1))
                .map_err(|error| failed(what, error))
        };
        let resident_threads = attribute(
            sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
            "reading the SM count",
        )? * attribute(
            sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR,
            "reading the threads per SM",
        )?;
        // `Ptx::from_binary` keeps this heap copy; its pointer reaches cuModuleLoadData.
        let image = FATBIN.to_vec();
        if image.as_ptr() as usize % 8 != 0 {
            return Err(EngineError::Device(
                "the embedded fatbin copy is not 8-byte aligned; use device='cpu'".into(),
            ));
        }
        let module = ctx.load_module(Ptx::from_binary(image)).map_err(|error| {
            EngineError::Device(format!(
                "device='{label}' ({name}, {arch}): loading the seqblaze kernels failed with {}; \
                 this build carries {} only; use device='cpu'",
                error_name(error),
                fatbin_contents()
            ))
        })?;
        let function = |symbol: &str| {
            module
                .load_function(symbol)
                .map_err(|error| failed(&format!("loading kernel {symbol}"), error))
        };
        let aln_warp = |q: u32| -> Result<[CudaFunction; 4], EngineError> {
            Ok([
                function(&format!("sb_aln_warp{q}_local"))?,
                function(&format!("sb_aln_warp{q}_global"))?,
                function(&format!("sb_aln_warp{q}_water"))?,
                function(&format!("sb_aln_warp{q}_needle"))?,
            ])
        };
        let dup = |width: &str| -> Result<[CudaFunction; 4], EngineError> {
            Ok([
                function(&format!("sb_dup_{width}_g4"))?,
                function(&format!("sb_dup_{width}_g8"))?,
                function(&format!("sb_dup_{width}_g16"))?,
                function(&format!("sb_dup_{width}_g32"))?,
            ])
        };
        let kernels = Kernels {
            resident_unary_keys: function("sb_resident_unary_keys")?,
            selection_normalize: function("sb_selection_normalize")?,
            selection_mask_counts: function("sb_selection_mask_counts")?,
            selection_mask_positions: function("sb_selection_mask_positions")?,
            selection_numeric: function("sb_selection_numeric")?,
            selection_ragged_lengths: function("sb_selection_ragged_lengths")?,
            selection_ragged_copy: function("sb_selection_ragged_copy")?,
            selection_topk_count: function("sb_selection_topk_count")?,
            selection_topk_flags: function("sb_selection_topk_flags")?,
            selection_rowstats_counts: function("sb_selection_rowstats_counts")?,
            selection_rowstats_pairs: function("sb_selection_rowstats_pairs")?,
            resident_edit_classify: function("sb_resident_edit_classify")?,
            resident_edit_compact: function("sb_resident_edit_compact")?,
            resident_edit_scatter: function("sb_resident_edit_scatter")?,

            resident_first_rows: function("sb_resident_first_rows")?,
            resident_pair_check: function("sb_resident_pair_check")?,
            resident_pair_error: function("sb_resident_pair_error")?,

            resident_dense_entries: function("sb_resident_dense_entries")?,
            detail_plan: function("sb_detail_plan")?,
            detail_continue: function("sb_detail_continue")?,
            detail_progress: function("sb_detail_progress")?,

            resident_hamming_counts: function("sb_resident_hamming_counts")?,
            resident_hamming_pieces: function("sb_resident_hamming_pieces")?,
            resident_hamming_groups: function("sb_resident_hamming_groups")?,
            resident_primary: function("sb_resident_primary")?,

            pf_resident_finalize: function("sb_pf_resident_finalize")?,
            pf_rows_clear: function("sb_pf_rows_clear")?,
            pf_rows_link: function("sb_pf_rows_link")?,
            pf_rows_add: function("sb_pf_rows_add")?,
            pf_rows_grow: function("sb_pf_rows_grow")?,
            pf_rows_finish: function("sb_pf_rows_finish")?,
            k1_lse_window: function("sb_k1_lse_window")?,
            k1_lse_gather: function("sb_k1_lse_gather")?,

            resident_p3_plan: function("sb_resident_p3_plan")?,
            resident_p3_flags: function("sb_resident_p3_flags")?,
            resident_p3_positions: function("sb_resident_p3_positions")?,
            resident_p3_hairpin: [
                function("sb_resident_p3_hairpin_g4")?,
                function("sb_resident_p3_hairpin_g8")?,
                function("sb_resident_p3_hairpin_g16")?,
                function("sb_resident_p3_hairpin_g32")?,
            ],
            resident_p3_dimer: [
                function("sb_resident_p3_dimer_g4")?,
                function("sb_resident_p3_dimer_g8")?,
                function("sb_resident_p3_dimer_g16")?,
                function("sb_resident_p3_dimer_g32")?,
            ],
            resident_p3_init: function("sb_resident_p3_init")?,
            resident_p3_fill: function("sb_resident_p3_fill")?,
            resident_p3_select: function("sb_resident_p3_select")?,
            resident_p3_finish: function("sb_resident_p3_finish")?,
            resident_p3_columns: function("sb_resident_p3_columns")?,

            resident_route_flags_dup: function("sb_resident_route_flags_dup")?,
            resident_route_flags_pf: function("sb_resident_route_flags_pf")?,
            resident_route_positions: function("sb_resident_route_positions")?,
            resident_dup_init: function("sb_resident_dup_init")?,
            resident_dup_step: function("sb_resident_dup_step")?,
            resident_dup_finish: function("sb_resident_dup_finish")?,
            resident_pf_init: function("sb_resident_pf_init")?,
            resident_pf_step: function("sb_resident_pf_step")?,
            resident_dup_bound: function("sb_resident_dup_bound")?,
            resident_dup_probe: function("sb_resident_dup_probe")?,
            resident_pf_bound: function("sb_resident_pf_bound")?,
            resident_pf_probe: function("sb_resident_pf_probe")?,
            resident_probe_lengths: function("sb_resident_probe_lengths")?,
            resident_probe_dups: function("sb_resident_probe_dups")?,
            resident_pair_mult: function("sb_resident_pair_mult")?,
            resident_bound_zero: function("sb_resident_bound_zero")?,
            resident_pf_finish: function("sb_resident_pf_finish")?,

            input_snapshot_lengths: function("sb_input_snapshot_lengths")?,
            input_snapshot_bytes: function("sb_input_snapshot_bytes")?,
            digest_row_bytes: function("sb_digest_row_bytes")?,
            digest_leaf_roots: function("sb_digest_leaf_roots")?,
            digest_table_tree_level: function("sb_digest_table_tree_level")?,
            digest_tree_level: function("sb_digest_tree_level")?,
            digest_sequence_table: function("sb_digest_sequence_table")?,
            digest_index_chunks: function("sb_digest_index_chunks")?,
            digest_pair_index: function("sb_digest_pair_index")?,
            input_init: function("sb_input_init")?,
            input_small: function("sb_input_small")?,
            input_init_digest: function("sb_input_init_digest")?,
            input_alphabet_rows: function("sb_input_alphabet_rows")?,
            digest_sequence_small: function("sb_digest_sequence_small")?,
            input_scan_u64: function("sb_input_scan_u64")?,
            input_scan_add: function("sb_input_scan_add")?,
            input_alphabet: function("sb_input_alphabet")?,
            input_reduce_rows: function("sb_input_reduce_rows")?,
            input_reduce_summaries: function("sb_input_reduce_summaries")?,
            pieces: function("sb_hamming_pieces")?,
            each: function("sb_each")?,
            hits_count: function("sb_hits_count")?,
            hits_write: function("sb_hits_write")?,
            rowstats_min: function("sb_rowstats_min")?,
            rowstats_arg: function("sb_rowstats_arg")?,
            hits_count_i64: function("sb_hits_count_i64")?,
            hits_write_i64: function("sb_hits_write_i64")?,
            rowstats_min_i64: function("sb_rowstats_min_i64")?,
            rowstats_arg_i64: function("sb_rowstats_arg_i64")?,
            fill_i64: function("sb_fill_i64")?,
            reg: function("sb_reg")?,
            lev_thread: [function("sb_lev_thread")?, function("sb_lev_thread_raw")?],
            lev_warp: [
                [function("sb_lev_warp1")?, function("sb_lev_warp1_raw")?],
                [function("sb_lev_warp2")?, function("sb_lev_warp2_raw")?],
                [function("sb_lev_warp4")?, function("sb_lev_warp4_raw")?],
                [function("sb_lev_warp8")?, function("sb_lev_warp8_raw")?],
            ],
            lev_route_count: function("sb_lev_route_count")?,
            lev_route_write: function("sb_lev_route_write")?,
            lev_scatter: function("sb_lev_scatter")?,
            aln_closed: function("sb_aln_closed")?,
            aln_thread: [
                function("sb_aln_thread_local")?,
                function("sb_aln_thread_global")?,
                function("sb_aln_thread_water")?,
                function("sb_aln_thread_needle")?,
            ],
            aln_warp: [aln_warp(1)?, aln_warp(2)?, aln_warp(4)?, aln_warp(8)?],
            aln_route_count: function("sb_aln_route_count")?,
            aln_route_write: function("sb_aln_route_write")?,
            tm: function("sb_tm")?,
            dup_closed: function("sb_dup_closed")?,
            dup: [dup("narrow")?, dup("wide")?, dup("detail")?],
            dup_route_count: function("sb_dup_route_count")?,
            dup_route_write: function("sb_dup_route_write")?,
            dup_scatter: function("sb_dup_scatter")?,
            dup_bound: function("sb_dup_bound")?,
            pf_bound: function("sb_pf_bound_eval")?,
            unary_hits_count: function("sb_unary_hits_count")?,
            unary_hits_write: function("sb_unary_hits_write")?,
            f64_eval: function("sb_f64_eval")?,
            pf_closed: function("sb_pf_closed")?,
            pf: [
                function("sb_pf_g4")?,
                function("sb_pf_g8")?,
                function("sb_pf_g16")?,
                function("sb_pf_g32")?,
            ],
            pf_route_count: function("sb_pf_route_count")?,
            pf_route_write: function("sb_pf_route_write")?,
            f53_eval: function("sb_f53_eval")?,
            dup_res: [
                function("sb_dup_res_narrow")?,
                function("sb_dup_res_wide")?,
                function("sb_dup_res_detail")?,
            ],
            pf_res: function("sb_pf_res")?,
            tm_biopython: function("sb_tm_biopython")?,
            tm_primer3: function("sb_tm_primer3")?,
            p3_hairpin: [
                function("sb_p3_hairpin_g4")?,
                function("sb_p3_hairpin_g8")?,
                function("sb_p3_hairpin_g16")?,
                function("sb_p3_hairpin_g32")?,
            ],
            p3_dimer: [
                function("sb_p3_dimer_g4")?,
                function("sb_p3_dimer_g8")?,
                function("sb_p3_dimer_g16")?,
                function("sb_p3_dimer_g32")?,
            ],
            each64: function("sb_each64")?,
            hits_count64: function("sb_hits_count64")?,
            hits_write64: function("sb_hits_write64")?,
            rowstats_min64: function("sb_rowstats_min64")?,
            rowstats_arg64: function("sb_rowstats_arg64")?,
            p3_scatter: function("sb_p3_scatter")?,
        };
        let block_cap = |functions: &[&CudaFunction]| -> Result<u32, EngineError> {
            let mut cap = 1024u32;
            for function in functions {
                let most = function
                    .max_threads_per_block()
                    .map_err(|error| failed("reading a kernel's block limit", error))?;
                cap = cap.min(u32::try_from(most).unwrap_or(0));
            }
            Ok((cap / 32 * 32).max(32))
        };
        let aln_block = [
            block_cap(&kernels.aln_thread.iter().collect::<Vec<_>>())?,
            block_cap(&kernels.aln_warp.iter().flatten().collect::<Vec<_>>())?,
        ];
        let dup_block = [
            block_cap(&kernels.dup[0].iter().collect::<Vec<_>>())?,
            block_cap(&kernels.dup[1].iter().collect::<Vec<_>>())?,
            block_cap(&kernels.dup[2].iter().collect::<Vec<_>>())?,
        ];
        let p3_block = [
            block_cap(&kernels.p3_hairpin.iter().collect::<Vec<_>>())?,
            block_cap(&kernels.p3_dimer.iter().collect::<Vec<_>>())?,
        ];
        let p3_sms = attribute(
            sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
            "reading the SM count",
        )?;
        let wave = |functions: &[CudaFunction; 4]| -> Result<[u64; 4], EngineError> {
            let mut blocks = [1u64; 4];
            for (slot, function) in blocks.iter_mut().zip(functions) {
                let most = function
                    .occupancy_max_active_blocks_per_multiprocessor(32, 0, None)
                    .map_err(|error| failed("reading a primer3 kernel's occupancy", error))?;
                *slot = u64::from(most).max(1);
            }
            Ok(blocks)
        };
        let p3_wave_blocks = [
            wave(&kernels.resident_p3_hairpin)?,
            wave(&kernels.resident_p3_dimer)?,
        ];
        let resident_wave = crate::resident_routes::FastKernel::ALL.map(|kernel| {
            [400usize, 800].map(|q| {
                crate::resident_routes::fast_wave(&kernels, kernel, q, p3_sms)
                    .map_err(|error| failed("reading a resident duplex kernel's occupancy", error))
            })
        });
        let resident_wave = resident_wave
            .into_iter()
            .map(|row| -> Result<[u64; 2], EngineError> {
                let [a, b] = row;
                Ok([a?, b?])
            })
            .collect::<Result<Vec<_>, _>>()?;
        let resident_wave: [[u64; 2]; 4] = [
            resident_wave[0],
            resident_wave[1],
            resident_wave[2],
            resident_wave[3],
        ];
        let unary_block = [
            block_cap(&[
                &kernels.tm,
                &kernels.unary_hits_count,
                &kernels.unary_hits_write,
            ])?,
            block_cap(&[
                &kernels.tm_biopython,
                &kernels.unary_hits_count,
                &kernels.unary_hits_write,
            ])?,
            block_cap(&[
                &kernels.tm_primer3,
                &kernels.unary_hits_count,
                &kernels.unary_hits_write,
            ])?,
        ];
        let dup_s_max = attribute(
            sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN,
            "reading the opt-in shared memory per block",
        )?;
        let static_shared = |functions: &[&CudaFunction]| -> Result<u64, EngineError> {
            let mut most = 0u64;
            for function in functions {
                let bytes = function
                    .get_attribute(sys::CUfunction_attribute::CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES)
                    .map_err(|error| failed("reading a kernel's static shared memory", error))?;
                most = most.max(u64::try_from(bytes).unwrap_or(0));
            }
            Ok(most)
        };
        // Dynamic shared memory above 48 KB needs the kernel's limit raised (§5.4).
        let raise = |functions: &[&CudaFunction], s_block: u64| -> Result<(), EngineError> {
            let dynamic = i32::try_from(dup_s_max.saturating_sub(s_block)).unwrap_or(i32::MAX);
            for function in functions {
                function
                    .set_attribute(
                        sys::CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                        dynamic,
                    )
                    .map_err(|error| failed("raising a kernel's dynamic shared memory", error))?;
            }
            Ok(())
        };
        let dup_kernels: Vec<&CudaFunction> = kernels.dup.iter().flatten().collect();
        let dup_s_block = static_shared(&dup_kernels)?;
        raise(&dup_kernels, dup_s_block)?;
        let pf_kernels: Vec<&CudaFunction> = kernels.pf.iter().collect();
        let pf_block = block_cap(&pf_kernels)?;
        let pf_s_block = static_shared(&pf_kernels)?;
        raise(&pf_kernels, pf_s_block)?;
        // K3's shared-tier resident entry (duplex-pf-cuda.md §19.4 step 3): S_max − s_block.
        let pf_res_s_block = static_shared(&[&kernels.pf_res])?;
        raise(&[&kernels.pf_res], pf_res_s_block)?;
        let resident_sinks = crate::resident_sinks::Kernels::load(&module)
            .map_err(|e| failed("loading resident sinks", e))?;
        let resident_indices = crate::resident_indices::Kernels::load(&module)
            .map_err(|e| failed("loading resident index adapters", e))?;
        let resident_tiles = crate::resident_tiles::Kernels::load(&module)
            .map_err(|e| failed("loading resident tile adapters", e))?;
        let inner = Arc::new(Inner {
            resident_sinks,
            resident_indices,
            resident_tiles,
            ctx,
            kernels,
            info: DeviceInfo {
                ordinal,
                name,
                arch,
            },
            resident_threads,
            aln_block,
            dup_block,
            dup_s_max,
            dup_s_block,
            pf_block,
            pf_s_block,
            pf_res_s_block,
            p3_block,
            p3_sms,
            p3_wave_blocks,
            resident_wave,
            pf_shared_wave: Mutex::new(Vec::new()),
            unary_block,
            events: crate::pool::EventList::new(),
            pool: crate::pool::DevicePool::create(ordinal, stream_anchor.cu_stream()),
            bounce: crate::pool::BounceSlot::new(),
            pack_cache: crate::pack_cache::Cache::default(),
            _module: module,
            _stream_anchor: stream_anchor,
        });
        if devices.len() <= ordinal {
            devices.resize(ordinal + 1, None);
        }
        devices[ordinal] = Some(Arc::clone(&inner));
        Ok(Device(inner))
    })
}

/// Testing builds: force the S2 fallback mode (`cuda_pool_fallback`). Only
/// before the first device open of the process; afterwards ParameterError.
#[cfg(feature = "testing")]
pub fn force_pool_fallback() -> Result<(), EngineError> {
    let devices = DEVICES.lock().unwrap_or_else(|poison| poison.into_inner());
    if devices.iter().any(Option::is_some) {
        return Err(EngineError::Parameter(
            "cuda_pool_fallback() must be called before the first CUDA device is opened in \
             this process"
                .into(),
        ));
    }
    crate::pool::force_fallback();
    Ok(())
}

/// Testing builds: the ordinal's pool statistics (`cuda_pool_stats`): whether
/// the owned pool is active, its reserved and used bytes and its release
/// threshold and immutable pack cache, including fallback-mode cache bytes.
#[cfg(feature = "testing")]
pub fn pool_stats(ordinal: usize) -> Result<PoolStats, EngineError> {
    let device = open(ordinal)?;
    guard(|| {
        let (cache_bytes, cache_entries) = device.0.pack_cache.stats();
        let Some(pool) = device.0.pool.as_ref() else {
            return Ok(PoolStats {
                cache_bytes,
                cache_entries,
                ..PoolStats::default()
            });
        };
        let failed =
            |error| crate::session::failure_ordinal(&device, "reading pool statistics", error);
        let read = |attribute| pool.attribute(attribute).map_err(failed);
        Ok(PoolStats {
            pool: true,
            reserved: read(sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_RESERVED_MEM_CURRENT)?,
            used: read(sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_USED_MEM_CURRENT)?,
            threshold: read(sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_RELEASE_THRESHOLD)?,
            cache_bytes,
            cache_entries,
        })
    })
}

/// `cuda_pool_stats` values (testing builds).
#[cfg(feature = "testing")]
#[derive(Clone, Copy, Debug, Default)]
pub struct PoolStats {
    pub pool: bool,
    pub reserved: u64,
    pub used: u64,
    pub threshold: u64,
    pub cache_bytes: u64,
    pub cache_entries: u64,
}

/// Testing builds: `cuMemPoolTrimTo(pool, 0)` on the ordinal's owned pool
/// (`cuda_pool_trim`); nothing in the fallback mode.
#[cfg(feature = "testing")]
pub fn pool_trim(ordinal: usize) -> Result<(), EngineError> {
    let device = open(ordinal)?;
    guard(|| {
        let Some(pool) = device.0.pool.as_ref() else {
            return Ok(());
        };
        device
            .0
            .ctx
            .bind_to_thread()
            .and_then(|()| pool.trim())
            .map_err(|error| crate::session::failure_ordinal(&device, "trimming the pool", error))
    })
}

/// Testing builds: `cuCtxSynchronize` on the ordinal's context
/// (`cuda_device_synchronize`), so that pending frees have run before the
/// pool statistics are read. No public call reaches it.
#[cfg(feature = "testing")]
pub fn device_synchronize(ordinal: usize) -> Result<(), EngineError> {
    let device = open(ordinal)?;
    guard(|| {
        device
            .0
            .ctx
            .bind_to_thread()
            .and_then(|()| unsafe { sys::cuCtxSynchronize() }.result())
            .map_err(|error| {
                crate::session::failure_ordinal(&device, "synchronizing the device", error)
            })
    })
}
