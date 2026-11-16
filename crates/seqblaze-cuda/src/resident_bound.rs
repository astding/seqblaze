//! The C1 bound of K1 native `duplex(model="mfe")` in the resident K1 stage
//! (kernels/resident_bound.cuh; docs/wiki/thermo/duplex-mfe-bound.md §6.2, §6.5, §6.6.2,
//! §8).
//!
//! Per pass (every sink's single pass, or TopK pass 1 and its replay) the device's
//! weak-bound decision of §6.6.2 is made once, before the pass's first chunk, by the core
//! rule (`seqblaze_core::plan::decide_outcome_bound`) with the DEVICE cost tables of
//! §6.6.3 (the host's cost closure): first a gather of the probe items' lengths and replay
//! skip status, which evaluates no bound; then the length and probe-cost gates; and only
//! if both pass, an outcome probe of at most 64 items on the device. The probe IDs refer
//! to the original-order work (CPU work IDs, duplicates of explicit pairs included), and
//! only lengths, skip flags and outcome codes are downloaded. Per chunk of a pass that
//! uses the bound, or of the TopK replay, one kernel then writes each item's skip flag,
//! the key λ and NaN fields of a resolved item, and integer outcome counts; the DP skips
//! resolved items. A certified item whose detail fields are requested keeps the DP for
//! them (its key equals λ). Counts weigh each canonical explicit pair by its occurrences.
//! Any other pass that does not use the bound launches nothing per chunk: the DP runs
//! without skip flags, and the host records its full count (the original item count).
//!
//! RowStats' second sweep (§8 *Sweeps*, revision 3) is a reset without new replay limits:
//! it keeps the pass, the decision and the counts, and restores each item's outcome from
//! the outcome words sweep 1 recorded (within R_b and the bound stage's launch budget,
//! §8 *Budget*), or evaluates the bound again without counting (QB14's provisional
//! choice, §8 *Above the budget*).
//!
//! The same stage carries the C2 bound of K3 `duplex(model="pf")`
//! (kernels/resident_pf_bound.cuh; docs/wiki/thermo/duplex-pf-bound.md §8) with K3's
//! evaluator: int64 outcome words (λ, or INT64_MAX for open), and every resolved item
//! skipped by K3's DP (its closed-form kernel writes Ẑ = 0, key(+∞)).
use std::sync::Arc;

use cudarc::driver::{CudaSlice, DeviceRepr, DriverError, LaunchConfig, PushKernelArg};
use seqblaze_core::{
    plan::{count_bound_evals, decide_outcome_bound, decision_fault, probe_ids, BoundFault},
    validate::EngineError,
};

use crate::{
    arena::Scratch,
    dupbound::{scratch_words, SbDupBound},
    interop::{allocate, pf_finalize::PfFinalizeConstants, DeviceLease},
    pfbound::{scratch_values, SbPfBound},
    resident::ResidentSpace,
    session::{failure_ordinal, ptr},
    DupBoundCall, PfBoundCall,
};

/// L_b: the bound stage's per-launch device budget (the 256 MiB slot rule of
/// duplex-cuda.md §5.4). While RowStats outcome words are held, they and one launch's
/// scratch together stay within it (duplex-mfe-bound.md §8 *Budget*).
const SCRATCH_BYTES: u64 = 256 << 20;
const THREADS: u32 = 64;
/// Staged letter codes ahead of each thread's bands (`SB_RB_CODE_WORDS`).
const CODE_WORDS: u64 = 50;
/// Explicit index space (`SB_LIST` of kernels/hamming.cuh).
const LIST: u64 = 1;

/// The per-call constants of the bound: K1's C1 (the certificate's relaxed classes and
/// the device pack's D keys) or K3's C2 (the certificate's relaxed classes, K3's weight
/// pack, the finalize constants and its testing fault).
#[derive(Clone)]
pub enum BoundCall {
    Mfe(Arc<DupBoundCall>),
    Pf(Arc<PfBoundCall>),
}

/// What the host passes for a bounded K1 or K3 call on CUDA (duplex-mfe-bound.md §8,
/// duplex-pf-bound.md §8).
#[derive(Clone)]
pub struct ResidentBound {
    pub call: BoundCall,
    /// The `lb_plus_one` testing fault (K1), read when the call was scheduled.
    pub lb_plus_one: bool,
    /// The device costs (e, ĉ_dev, q̂_dev) of an item's strand lengths (§6.6.2 step 3:
    /// the tables of §6.6.3, never the CPU's).
    pub cost: Arc<dyn Fn(usize, usize) -> (u64, u32, u32) + Send + Sync>,
    /// The original-order work space (CPU work IDs) and its item count; its control
    /// columns are owned by the caller for the whole call.
    pub original: ResidentSpace,
    pub original_items: usize,
    /// The sink runs a second sweep over the same work (RowStats, §8 *Sweeps*): sweep 1
    /// may keep per-item outcome words for it.
    pub sweeps: bool,
}

/// Mirrors `SbResidentBound` in kernels/resident_bound.cuh.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
struct SbResidentBound {
    mode: u64,
    theta: i64,
    thresholds: u64,
    active: u64,
    mirror: u64,
    used: u64,
    fields: u64,
    keys: u64,
    dh: u64,
    ds: u64,
    tm: u64,
    skip: u64,
    mult: u64,
    counts: u64,
    scratch: u64,
    stride: u64,
    count: u64,
    words: u64,
    words_mode: u64,
}
// SAFETY: plain #[repr(C)] struct of 8-byte fields in the CUDA struct's order; passed by
// value as a kernel parameter.
unsafe impl DeviceRepr for SbResidentBound {}
const _: () = assert!(std::mem::size_of::<SbResidentBound>() == 19 * 8);

/// `SbResidentBound::words_mode` (§8 *Sweeps*): no outcome words, record them (sweep 1),
/// or restore the outcomes from them (sweep 2).
const WORDS_NONE: u64 = 0;
const WORDS_RECORD: u64 = 1;
const WORDS_RESTORE: u64 = 2;

/// The θ rule of the current pass (§6.2).
#[derive(Clone, Copy)]
enum Theta {
    /// Certify only.
    None,
    Scalar(i64),
    /// TopK replay: per-row limits (`values`, `active` device addresses); a mirrored
    /// within item's limit is the larger of its two rows'.
    Replay {
        values: u64,
        active: u64,
        mirror: bool,
    },
}

/// The bound evaluator of a stage and its device constants.
enum Evaluator {
    Mfe {
        params: SbDupBound,
        _keys: CudaSlice<i64>,
        _cls: CudaSlice<u8>,
        _relaxed: CudaSlice<i64>,
    },
    Pf {
        params: SbPfBound,
        finalize: Box<PfFinalizeConstants>,
        _weights: CudaSlice<u64>,
        _cls: CudaSlice<u8>,
        _relaxed: CudaSlice<i64>,
    },
}

/// The device side of one bounded K1 or K3 call: the bound's constants, the per-pass
/// decision and the outcome counts of pass 0 and the TopK replay.
pub(crate) struct BoundStage {
    host: ResidentBound,
    eval: Evaluator,
    cutoff: Option<i64>,
    fields: bool,
    /// The stage's whole work space (deduplicated for explicit non-Each sinks).
    stage_space: ResidentSpace,
    replay: Option<(u64, u64, bool)>,
    /// Occurrences of each canonical explicit pair (when the stage's work is the
    /// deduplicated list of an original explicit list).
    mult: Option<Scratch<u64>>,
    /// [certified, pruned, full, skipped] of pass 0, then of the replay.
    counts: CudaSlice<u64>,
    /// The full counts of pass 0 and the replay recorded on the host: a pass that does
    /// not use the bound and is not the replay launches no classify (§8 *Per chunk*).
    host_full: [u64; 2],
    pass: usize,
    decision: Option<bool>,
    /// The TopK replay limits were set since the last reset: the next reset starts the
    /// replay pass (§8 *Sweeps*).
    replay_pending: bool,
    /// The current pass is in a repeated sweep (RowStats sweep 2): no probe, no count.
    sweep: bool,
    /// Sweep 1 decided whether to keep outcome words (once per call).
    words_decided: bool,
    /// One outcome word per stage work item: K1's int32 (the clamped key λ, or INT32_MAX
    /// for open, which no clamped key equals; §8 *Per-item reuse*), K3's int64 (λ, or
    /// INT64_MAX for open).
    words: Option<Scratch<u8>>,
    /// W = (word bytes)·(stage items) while `words` is held, else 0: the bound scratch of
    /// every launch is sized from L_b − W (§8 *Budget*).
    words_bytes: u64,
}

fn bad(message: &str) -> EngineError {
    EngineError::Device(format!(
        "resident C1 bound stage: {message}; no result was published"
    ))
}

fn grid(n: u64) -> LaunchConfig {
    LaunchConfig {
        grid_dim: (n.div_ceil(128).clamp(1, 65535) as u32, 1, 1),
        block_dim: (128, 1, 1),
        shared_mem_bytes: 0,
    }
}

impl BoundStage {
    /// Upload the bound's constants once per call. `cutoff`: the call's engine-key
    /// cutoff (θ of every pass but the replay); `fields`: dh/ds/tm requested;
    /// `stage_space`: the producer's whole work space.
    pub(crate) fn new(
        lease: &DeviceLease,
        host: ResidentBound,
        cutoff: Option<i64>,
        fields: bool,
        stage_space: ResidentSpace,
    ) -> Result<Self, EngineError> {
        let fail = |what: &str, e: DriverError| failure_ordinal(lease.device(), what, e);
        let upload_i64 = |values: &[i64]| {
            let mut padded = values.to_vec();
            if padded.is_empty() {
                padded.push(0);
            }
            lease.execution().clone_htod(&padded)
        };
        let upload_cls = |values: &[u8]| {
            let mut padded = values.to_vec();
            if padded.is_empty() {
                padded.push(0);
            }
            lease.execution().clone_htod(&padded)
        };
        let pack = |e| fail("uploading the bound pack", e);
        let eval = match &host.call {
            BoundCall::Mfe(call) => {
                let keys = upload_i64(&call.keys).map_err(pack)?;
                let cls = upload_cls(&call.cls).map_err(pack)?;
                let relaxed = upload_i64(&call.relaxed).map_err(pack)?;
                let mut off = [0u64; 14];
                for (k, slot) in off.iter_mut().enumerate() {
                    *slot = call.offsets[k] as u64;
                }
                let params = SbDupBound {
                    keys: ptr(&keys, lease.stream()),
                    cls: ptr(&cls, lease.stream()),
                    relaxed: ptr(&relaxed, lease.stream()),
                    n_relaxed: call.n_relaxed as u64,
                    lam: call.loop_max as u64,
                    n_p: call.n_p as u64,
                    n_c: call.n_c as u64,
                    has_states: u64::from(call.has_states),
                    off,
                    lb_plus_one: u64::from(host.lb_plus_one),
                    ..SbDupBound::default()
                };
                Evaluator::Mfe {
                    params,
                    _keys: keys,
                    _cls: cls,
                    _relaxed: relaxed,
                }
            }
            BoundCall::Pf(call) => {
                let finalize = crate::pfbound::check(call)?;
                let weights = lease
                    .execution()
                    .clone_htod(&crate::pfbound::weight_words(call))
                    .map_err(pack)?;
                let cls = upload_cls(&call.cls).map_err(pack)?;
                let relaxed = upload_i64(&call.relaxed).map_err(pack)?;
                let params = crate::pfbound::params(
                    call,
                    ptr(&weights, lease.stream()),
                    ptr(&cls, lease.stream()),
                    ptr(&relaxed, lease.stream()),
                );
                Evaluator::Pf {
                    params,
                    finalize: Box::new(finalize),
                    _weights: weights,
                    _cls: cls,
                    _relaxed: relaxed,
                }
            }
        };
        let counts = lease
            .execution()
            .clone_htod(&[0u64; 8])
            .map_err(|e| fail("initializing bound counts", e))?;
        Ok(Self {
            host,
            eval,
            cutoff,
            fields,
            stage_space,
            replay: None,
            mult: None,
            counts,
            host_full: [0; 2],
            pass: 0,
            decision: None,
            replay_pending: false,
            sweep: false,
            words_decided: false,
            words: None,
            words_bytes: 0,
        })
    }

    /// The TopK replay's per-row limits, set between the passes.
    pub(crate) fn replay_limits(&mut self, values: u64, active: u64, mirror: bool) {
        self.replay = Some((values, active, mirror));
        self.replay_pending = true;
    }

    /// The producer starts over its work (§8 *Sweeps*, revision 3). A replay reset (the
    /// replay limits were set since the last reset) begins the TopK replay pass, whose
    /// next chunk decides again. Any other reset (RowStats sweep 2) keeps the pass, its
    /// decision and its counts: no probe, no count, the same outcomes.
    pub(crate) fn reset(&mut self) {
        if std::mem::take(&mut self.replay_pending) {
            self.pass = (self.pass + 1).min(1);
            self.decision = None;
            self.sweep = false;
        } else {
            self.sweep = true;
        }
    }

    fn theta(&self) -> Theta {
        if self.pass == 1 {
            if let Some((values, active, mirror)) = self.replay {
                return Theta::Replay {
                    values,
                    active,
                    mirror,
                };
            }
        }
        self.cutoff.map_or(Theta::None, Theta::Scalar)
    }

    fn control(&self, theta: Theta) -> SbResidentBound {
        let mut r = SbResidentBound::default();
        match theta {
            Theta::None => {}
            Theta::Scalar(value) => {
                r.mode = 1;
                r.theta = value;
            }
            Theta::Replay {
                values,
                active,
                mirror,
            } => {
                r.mode = 2;
                r.thresholds = values;
                r.active = active;
                r.mirror = u64::from(mirror);
            }
        }
        r.fields = u64::from(self.fields);
        r
    }

    /// One bound thread's scratch for strands of at most `longest` letters: the kernel's
    /// stride (K1: int64 words; K3: F53 values) and its bytes.
    fn stride(&self, longest: u64) -> (u64, u64) {
        match &self.host.call {
            BoundCall::Mfe(call) => {
                let words = CODE_WORDS + scratch_words(call, longest);
                (words, words * 8)
            }
            BoundCall::Pf(call) => {
                let values = scratch_values(call, longest);
                (values, values * 16)
            }
        }
    }

    /// Bytes of one outcome word.
    fn word_bytes(&self) -> u64 {
        match self.eval {
            Evaluator::Mfe { .. } => 4,
            Evaluator::Pf { .. } => 8,
        }
    }

    /// Per-thread scratch for strands of at most `longest` letters, and the number of
    /// threads one launch over `n` items can hold. While outcome words are held (W bytes),
    /// the launch runs T = 64·min(⌈n/64⌉, ⌊(L_b − W)/S₆₄⌋) threads, so that W plus the
    /// scratch stays within L_b (§8 *Budget*; the keep rule makes T ≥ 64). Without words
    /// the sizing is unchanged: it may round up by less than one block.
    fn scratch(
        &self,
        lease: &DeviceLease,
        longest: u64,
        n: u64,
    ) -> Result<(Scratch<i64>, u64, u64), EngineError> {
        let (stride, bytes) = self.stride(longest);
        let block = u64::from(THREADS);
        let launched = if self.words_bytes == 0 {
            let threads = (SCRATCH_BYTES / bytes).clamp(1, n.max(1));
            threads.div_ceil(block) * block
        } else {
            let fit = SCRATCH_BYTES.saturating_sub(self.words_bytes) / (bytes * block);
            fit.clamp(1, n.max(1).div_ceil(block)) * block
        };
        let words = usize::try_from(launched * bytes / 8).map_err(|_| bad("scratch size"))?;
        Ok((allocate::<i64>(lease, words)?, stride, launched))
    }

    /// The §6.6.2 decision of the current pass, made once before its first chunk: the
    /// probe items' lengths and skip status first (no bound evaluated), then the length
    /// and probe-cost gates with the device tables, then the outcome probe only if both
    /// pass. The decision faults force it without any device work.
    fn decide(
        &mut self,
        lease: &DeviceLease,
        tables: crate::resident_routes::Tables,
        longest: u64,
    ) -> Result<bool, EngineError> {
        if let Some(used) = self.decision {
            return Ok(used);
        }
        let n = self.host.original_items;
        let ids = probe_ids(n);
        let forced = decision_fault() != BoundFault::None;
        let gathered = if !ids.is_empty() && !forced {
            Some(self.gather(lease, tables, &ids)?)
        } else {
            None
        };
        let at = |k: usize| ids.binary_search(&k).ok();
        let info = |t: usize| -> Result<(u64, u64, bool), EngineError> {
            let g = gathered
                .as_ref()
                .ok_or_else(|| bad("probe item without its gather"))?;
            Ok((g.info[3 * t], g.info[3 * t + 1], g.info[3 * t + 2] != 0))
        };
        let cost = &self.host.cost;
        let decision = decide_outcome_bound(
            n,
            |k| -> Result<Option<(u64, u32, u32)>, EngineError> {
                let t = at(k).ok_or_else(|| bad("probe ID outside the probe"))?;
                let (len_a, len_b, skipped) = info(t)?;
                Ok((!skipped).then(|| cost(len_a as usize, len_b as usize)))
            },
            |remaining, resolved| {
                // One launch evaluates every probe item the pass does not skip.
                let g = gathered
                    .as_ref()
                    .ok_or_else(|| bad("probe without its gather"))?;
                let codes = self.probe(lease, tables, longest, g)?;
                for (flag, &k) in resolved.iter_mut().zip(remaining) {
                    let t = at(k).ok_or_else(|| bad("probe ID outside the probe"))?;
                    *flag = matches!(codes[3 * t + 2], 1 | 2);
                }
                Ok(())
            },
        )?;
        self.decision = Some(decision.used);
        Ok(decision.used)
    }

    /// §6.6.2 step 2: per probe item (original work IDs `ids`), its strand lengths and
    /// whether the pass skips it (the TopK replay: no row limit, or a later occurrence of
    /// an explicit pair), 3 words each, read on the device and downloaded. No bound is
    /// evaluated.
    fn gather(
        &self,
        lease: &DeviceLease,
        tables: crate::resident_routes::Tables,
        ids: &[usize],
    ) -> Result<Gathered, EngineError> {
        let fail = |what: &str, e: DriverError| failure_ordinal(lease.device(), what, e);
        let kernels = &lease.device().0.kernels;
        let count = ids.len() as u64;
        let host_ids: Vec<u64> = ids.iter().map(|&k| k as u64).collect();
        let d_ids = lease
            .execution()
            .clone_htod(&host_ids)
            .map_err(|e| fail("uploading bound probe IDs", e))?;
        let theta = self.theta();
        let space = self.host.original;
        // Replay eligibility: a later occurrence of an explicit pair is skipped.
        let dup = if matches!(theta, Theta::Replay { .. }) && space.kind == LIST {
            let mut flags = lease
                .execution()
                .clone_htod(&vec![0u32; ids.len()])
                .map_err(|e| fail("initializing bound probe flags", e))?;
            let n = self.host.original_items as u64;
            let mut launch = lease
                .execution()
                .launch_builder(&kernels.resident_probe_dups);
            launch
                .arg(&space)
                .arg(&n)
                .slice_arg(&d_ids)
                .arg(&count)
                .slice_arg_mut(&mut flags);
            // SAFETY: the original explicit space covers n validated items; ids holds
            // `count` <= 64 of them and flags one word per probe item.
            unsafe { launch.launch(grid(n)) }
                .map_err(|e| fail("checking bound probe occurrences", e))?;
            Some(flags)
        } else {
            None
        };
        let r = self.control(theta);
        let mut out = allocate::<u64>(lease, ids.len() * 3)?;
        let dup_ptr = dup.as_ref().map_or(0, |flags| ptr(flags, lease.stream()));
        let mut launch = lease
            .execution()
            .launch_builder(&kernels.resident_probe_lengths);
        launch
            .arg(&tables)
            .arg(&space)
            .arg(&r)
            .slice_arg(&d_ids)
            .arg(&count)
            .arg(&dup_ptr)
            .slice_arg_mut(&mut out);
        // SAFETY: ids holds `count` <= 64 validated work IDs of the original space; out
        // holds three words per probe item, each written by exactly one thread.
        unsafe { launch.launch(grid(count)) }
            .map_err(|e| fail("reading the C1 bound probe lengths", e))?;
        let mut info = vec![0u64; ids.len() * 3];
        lease
            .execution()
            .memcpy_dtoh(&*out, &mut info[..])
            .map_err(|e| fail("reading the C1 bound probe lengths", e))?;
        lease.synchronize()?;
        Ok(Gathered {
            info,
            ids: d_ids,
            dup,
        })
    }

    /// §6.6.2 step 6: the outcome codes of the probe items the pass does not skip,
    /// evaluated on the device (3 words per item: lengths and code; a skipped item gets
    /// code 3 and no evaluation). Adds p to the testing evaluation counter.
    fn probe(
        &self,
        lease: &DeviceLease,
        tables: crate::resident_routes::Tables,
        longest: u64,
        gathered: &Gathered,
    ) -> Result<Vec<u64>, EngineError> {
        let fail = |what: &str, e: DriverError| failure_ordinal(lease.device(), what, e);
        let kernels = &lease.device().0.kernels;
        let count = (gathered.info.len() / 3) as u64;
        let evaluated = gathered
            .info
            .chunks_exact(3)
            .filter(|word| word[2] == 0)
            .count() as u64;
        let theta = self.theta();
        let space = self.host.original;
        let (scratch, stride, launched) = self.scratch(lease, longest, count)?;
        let mut r = self.control(theta);
        r.scratch = ptr(&scratch, lease.stream());
        r.stride = stride;
        let mut out = allocate::<u64>(lease, count as usize * 3)?;
        let dup_ptr = gathered
            .dup
            .as_ref()
            .map_or(0, |flags| ptr(flags, lease.stream()));
        let mut launch;
        match &self.eval {
            Evaluator::Mfe { params, .. } => {
                launch = lease
                    .execution()
                    .launch_builder(&kernels.resident_dup_probe);
                launch.arg(&tables).arg(&space).arg(params).arg(&r);
            }
            Evaluator::Pf {
                params, finalize, ..
            } => {
                launch = lease.execution().launch_builder(&kernels.resident_pf_probe);
                launch
                    .arg(&tables)
                    .arg(&space)
                    .arg(params)
                    .arg(finalize.as_ref())
                    .arg(&r);
            }
        }
        launch
            .slice_arg(&gathered.ids)
            .arg(&count)
            .arg(&dup_ptr)
            .slice_arg_mut(&mut out);
        let config = LaunchConfig {
            grid_dim: ((launched / u64::from(THREADS)).max(1) as u32, 1, 1),
            block_dim: (THREADS, 1, 1),
            shared_mem_bytes: 0,
        };
        // SAFETY: `launched` scratch slots of `stride` words exist; out holds three
        // words per probe item, each written by exactly one thread.
        unsafe { launch.launch(config) }.map_err(|e| fail("probing the bound", e))?;
        count_bound_evals(evaluated);
        let mut host = vec![0u64; count as usize * 3];
        lease
            .execution()
            .memcpy_dtoh(&*out, &mut host[..])
            .map_err(|e| fail("reading the C1 bound probe", e))?;
        lease.synchronize()?;
        drop(scratch);
        Ok(host)
    }

    /// Occurrences of each canonical pair of the stage's deduplicated explicit `space`
    /// in the original list, built once.
    fn multiplicities(&mut self, lease: &DeviceLease) -> Result<u64, EngineError> {
        let (stage_space, stage_items) = (self.stage_space, self.stage_space.n);
        let original = self.host.original;
        let dedup = stage_space.kind == LIST
            && original.kind == LIST
            && (stage_space.li != original.li || stage_space.lj != original.lj);
        if !dedup {
            return Ok(0);
        }
        if self.mult.is_none() {
            let fail = |what: &str, e: DriverError| failure_ordinal(lease.device(), what, e);
            let kernels = &lease.device().0.kernels;
            let nu = usize::try_from(stage_items).map_err(|_| bad("pair count"))?;
            let mut mult = allocate::<u64>(lease, nu)?;
            let mut launch = lease
                .execution()
                .launch_builder(&kernels.resident_bound_zero);
            let n64 = nu as u64;
            launch.slice_arg_mut(&mut mult).arg(&n64);
            // SAFETY: mult holds max(nu, 1) words; the kernel writes the first nu.
            unsafe { launch.launch(grid(n64)) }
                .map_err(|e| fail("initializing pair multiplicities", e))?;
            let n = self.host.original_items as u64;
            let (ci, cj) = (stage_space.li, stage_space.lj);
            let mut launch = lease
                .execution()
                .launch_builder(&kernels.resident_pair_mult);
            launch
                .arg(&original)
                .arg(&n)
                .arg(&ci)
                .arg(&cj)
                .arg(&n64)
                .slice_arg_mut(&mut mult);
            // SAFETY: the original list has n validated items; (ci, cj) are the nu
            // canonical pairs in ascending order, each found by binary search.
            unsafe { launch.launch(grid(n)) }
                .map_err(|e| fail("counting pair multiplicities", e))?;
            self.mult = Some(mult);
        }
        Ok(self.mult.as_ref().map_or(0, |m| ptr(m, lease.stream())))
    }

    /// Classify the items of one chunk (`space`, a range of the stage's work): skip
    /// flags, λ keys and NaN fields of resolved items, outcome counts. `outputs`: the
    /// chunk's key, dh, ds and tm addresses. Returns the skip flags for the DP launches,
    /// or `None` when no item is skipped.
    pub(crate) fn classify(
        &mut self,
        lease: &DeviceLease,
        tables: crate::resident_routes::Tables,
        space: ResidentSpace,
        longest: u64,
        outputs: [u64; 4],
    ) -> Result<Option<Scratch<u8>>, EngineError> {
        let n = space.n;
        let fresh = self.decision.is_none();
        let used = self.decide(lease, tables, longest)?;
        // §8 *Per chunk*: a pass that does not use the bound and is not the TopK replay
        // launches nothing and allocates no skip flags. Every item is open, which the DP
        // without skip flags already gives, and an open item's key and fields are the
        // DP's. Its counts go to the host, equal to what the classify launch counted:
        // full += each stage item's weight, its pair multiplicity for a deduplicated
        // explicit list and 1 otherwise. Over one sweep of the stage these weights sum to
        // the original item count (every original work item is one occurrence of exactly
        // one stage item), so the pass adds `original_items` once, when its decision is
        // made (sweep 1's first chunk; a repeated sweep adds nothing, as before).
        if !used && !matches!(self.theta(), Theta::Replay { .. }) {
            if fresh && !self.sweep {
                self.host_full[self.pass] += self.host.original_items as u64;
            }
            return Ok(None);
        }
        let mult = self.multiplicities(lease)?;
        let fail = |what: &str, e: DriverError| failure_ordinal(lease.device(), what, e);
        let kernels = &lease.device().0.kernels;
        // §8 *Sweeps* and *Budget*: sweep 1 of a multi-chunk RowStats pass that uses the
        // bound keeps one outcome word per stage item iff W = 4·items ≤ R_b and
        // W + S₆₄ ≤ L_b (one 64-thread block of bound scratch at the stage's longest
        // strand); the scratch of every later launch is then sized from L_b − W. A refused
        // allocation keeps none (sweep 2 then evaluates again), and raises nothing.
        let restoring = self.sweep;
        if !restoring && !self.words_decided && self.pass == 0 {
            self.words_decided = true;
            let items = self.stage_space.n;
            let block_scratch = self.stride(longest).1.checked_mul(u64::from(THREADS));
            let fits = items.checked_mul(self.word_bytes()).filter(|&bytes| {
                bytes <= crate::launch::bound_retain_bytes()
                    && block_scratch
                        .and_then(|s64| s64.checked_add(bytes))
                        .is_some_and(|total| total <= SCRATCH_BYTES)
            });
            if let (true, Some(bytes)) = (used && self.host.sweeps && n < items, fits) {
                self.words = usize::try_from(bytes)
                    .ok()
                    .and_then(|bytes| allocate::<u8>(lease, bytes).ok());
                self.words_bytes = if self.words.is_some() { bytes } else { 0 };
            }
        }
        let words_mode = match (&self.words, restoring) {
            (Some(_), _) if self.pass != 0 => WORDS_NONE,
            (Some(_), false) => WORDS_RECORD,
            (Some(_), true) => WORDS_RESTORE,
            (None, _) => WORDS_NONE,
        };
        let evaluate = used && words_mode != WORDS_RESTORE;
        let skip = allocate::<u8>(lease, n as usize)?;
        let mut r = self.control(self.theta());
        r.used = u64::from(evaluate);
        [r.keys, r.dh, r.ds, r.tm] = outputs;
        r.skip = ptr(&skip, lease.stream());
        r.mult = mult;
        r.counts = ptr(&self.counts, lease.stream()) + 32 * self.pass as u64;
        r.count = u64::from(!restoring);
        r.words_mode = words_mode;
        if let Some(words) = self.words.as_ref().filter(|_| words_mode != WORDS_NONE) {
            let offset = space
                .k0
                .checked_sub(self.stage_space.k0)
                .ok_or_else(|| bad("chunk outside the stage's work"))?;
            r.words = ptr(words, lease.stream()) + self.word_bytes() * offset;
        }
        let (scratch, launched) = if evaluate {
            let (scratch, stride, launched) = self.scratch(lease, longest, n)?;
            r.scratch = ptr(&scratch, lease.stream());
            r.stride = stride;
            (Some(scratch), launched)
        } else {
            (
                None,
                n.max(1).div_ceil(u64::from(THREADS)) * u64::from(THREADS),
            )
        };
        let config = LaunchConfig {
            grid_dim: ((launched / u64::from(THREADS)).clamp(1, 65535) as u32, 1, 1),
            block_dim: (THREADS, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut launch;
        match &self.eval {
            Evaluator::Mfe { params, .. } => {
                launch = lease
                    .execution()
                    .launch_builder(&kernels.resident_dup_bound);
                launch.arg(&tables).arg(&space).arg(params).arg(&r);
            }
            Evaluator::Pf {
                params, finalize, ..
            } => {
                launch = lease.execution().launch_builder(&kernels.resident_pf_bound);
                launch
                    .arg(&tables)
                    .arg(&space)
                    .arg(params)
                    .arg(finalize.as_ref())
                    .arg(&r);
            }
        }
        // SAFETY: the chunk's validated items, its key/field outputs and skip flags hold
        // `n` entries each; with the bound in use `launched` scratch slots exist and the
        // grid never exceeds them; outcome words, when used, cover the chunk's items.
        unsafe { launch.launch(config) }.map_err(|e| fail("evaluating the bound", e))?;
        if evaluate {
            // §10.2 (revision 3): the device's evaluations, counted by launch items.
            count_bound_evals(n);
        }
        drop(scratch);
        Ok(Some(skip))
    }

    /// [certified, pruned, full, skipped] of pass 0 and of the replay, downloaded once;
    /// a pass that launched no classify has its full count on the host (`host_full`).
    pub(crate) fn counts(&self, lease: &DeviceLease) -> Result<[[u64; 4]; 2], EngineError> {
        let mut host = [0u64; 8];
        lease
            .execution()
            .memcpy_dtoh(&self.counts, &mut host[..])
            .map_err(|e| failure_ordinal(lease.device(), "reading the C1 bound counts", e))?;
        lease.synchronize()?;
        Ok([
            [host[0], host[1], host[2] + self.host_full[0], host[3]],
            [host[4], host[5], host[6] + self.host_full[1], host[7]],
        ])
    }
}

/// The gather of §6.6.2 step 2: 3 words per probe item (lengths, skipped), the probe IDs
/// and the replay's later-occurrence flags on the device, for the outcome probe.
struct Gathered {
    info: Vec<u64>,
    ids: CudaSlice<u64>,
    dup: Option<CudaSlice<u32>>,
}
