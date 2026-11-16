//! One public CUDA compute stream for host, device and mixed input tables.
//! Every arithmetic stage returns device-owned columns; sinks never download
//! per-pair keys or numeric payloads. TopK replays this same validated state.
use crate::{
    interop::pf_finalize::{self, PfFinalizeConstants, PfSpace, PfTables, PfTerms},
    interop::{BoundDeviceRows, DeviceColumn, DeviceDType, DeviceLease},
    resident::{ResidentChunk, ResidentField, ResidentFieldSpec, ResidentProducer, ResidentSpace},
    resident_edit_stage::LevStage,
    resident_native::{self, PrimaryFamily, ResidentRegisterStage},
    resident_routes::{ResidentDuplexStage, ResidentPfStage},
    AlignCall, DeviceCall, EditMode,
};
use seqblaze_core::validate::EngineError;
use std::{
    mem,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc, Mutex,
    },
};

pub type SharedPfTerms = Arc<Mutex<Option<DeviceColumn>>>;
/// The resident runner's chunk cap for every family (cancellation-progress.md §3.5).
const RESIDENT_CHUNK: u64 = 65_536;
/// The largest chunk of the cost-bounded Hamming and Levenshtein stages: the
/// default `pairs_per_chunk` (levenshtein-cuda.md §14).
const CHEAP_CHUNK_MAX: u64 = 1 << 22;
// Inline stage state avoids an extra allocation on the producer's hot path.
#[allow(clippy::large_enum_variant)]
enum Stage {
    Hamming,
    Lev {
        mode: EditMode,
        options: LevStage,
        /// Items the device continuation (bucket 6) served so far; their in-pass
        /// fields are never written (levenshtein-cuda.md §13.4 step 1).
        continued: u64,
    },
    Align(Arc<AlignCall>),
    Duplex(ResidentDuplexStage),
    Pf {
        stage: ResidentPfStage,
        constants: PfFinalizeConstants,
        last: SharedPfTerms,
    },
    Register(ResidentRegisterStage),
    Primer3(crate::resident_p3::ResidentPrimer3Stage),
}
pub struct NativeProducer<'a> {
    a: &'a BoundDeviceRows,
    b: &'a BoundDeviceRows,
    space: ResidentSpace,
    next: u64,
    maximum: [u64; 2],
    stage: Stage,
    specs: Vec<ResidentFieldSpec>,
    stream: u64,
    halt: Arc<AtomicBool>,
    settings: crate::launch::Settings,
}
fn spec(name: &str, dtype: DeviceDType) -> ResidentFieldSpec {
    ResidentFieldSpec {
        name: name.into(),
        dtype,
        mask_bits: match dtype {
            DeviceDType::F64 => 0x7ff8000000000000,
            DeviceDType::I32 => i32::MAX as u64,
            DeviceDType::I64 => i64::MAX as u64,
            _ => 0,
        },
    }
}
impl<'a> NativeProducer<'a> {
    // These independent validated launch controls mirror the resident entry.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        lease: &DeviceLease,
        a: &'a BoundDeviceRows,
        b: &'a BoundDeviceRows,
        space: ResidentSpace,
        maximum: [u64; 2],
        call: DeviceCall,
        requested: &[&str],
        pf: Option<PfFinalizeConstants>,
    ) -> Result<Self, EngineError> {
        a.ensure_lease(lease)?;
        b.ensure_lease(lease)?;
        let (stage, all) = match call {
            DeviceCall::Hamming => (Stage::Hamming, vec![spec("dist", DeviceDType::I32)]),
            DeviceCall::Levenshtein { mode, acgt, .. } => {
                // The fast tiers' forward pass yields `b_end` and `n_ends` when the
                // request holds no heavier detail field (levenshtein-cuda.md §13.4
                // step 1); `global` offers neither.
                let heavy = requested
                    .iter()
                    .any(|name| matches!(*name, "b_start" | "n_starts" | "ends" | "cigar"));
                let in_pass = mode != EditMode::Global && !heavy;
                let options = LevStage {
                    acgt,
                    b_end: in_pass && requested.contains(&"b_end"),
                    n_ends: in_pass && requested.contains(&"n_ends"),
                    longest: Some(maximum),
                };
                let mut all = vec![spec("dist", DeviceDType::I32)];
                if options.b_end {
                    all.push(spec("b_end", DeviceDType::I64));
                }
                if options.n_ends {
                    all.push(spec("n_ends", DeviceDType::I32));
                }
                (
                    Stage::Lev {
                        mode,
                        options,
                        continued: 0,
                    },
                    all,
                )
            }
            DeviceCall::Align(call) => (
                Stage::Align(call),
                vec![
                    spec("score", DeviceDType::F64),
                    spec("score_fx", DeviceDType::I32),
                ],
            ),
            DeviceCall::Duplex(call) => (
                // The detail pass only when a detail field is requested (duplex-cuda.md
                // §15.4 step 2); otherwise the narrow or wide score pass.
                Stage::Duplex(ResidentDuplexStage::new(
                    lease,
                    &call,
                    requested
                        .iter()
                        .any(|name| matches!(*name, "dh_kcal" | "ds_cal_per_k" | "tm_c")),
                )?),
                vec![
                    spec("dg_kcal", DeviceDType::F64),
                    spec("dg_dcal", DeviceDType::I32),
                    spec("dh_kcal", DeviceDType::F64),
                    spec("ds_cal_per_k", DeviceDType::F64),
                    spec("tm_c", DeviceDType::F64),
                ],
            ),
            DeviceCall::DuplexPf(call) => (
                Stage::Pf {
                    stage: ResidentPfStage::new(lease, &call)?,
                    constants: pf.ok_or_else(|| {
                        EngineError::Device(
                            "resident K3 needs its exact finite-map constants".into(),
                        )
                    })?,
                    last: Arc::new(Mutex::new(None)),
                },
                vec![
                    spec("dg_ens_kcal", DeviceDType::F64),
                    spec("dg_ens_err_kcal", DeviceDType::F64),
                ],
            ),
            DeviceCall::TmPairs(call) => (
                Stage::Register(ResidentRegisterStage::new(lease, &call)?),
                vec![
                    spec("tm_c", DeviceDType::F64),
                    spec("dg_kcal", DeviceDType::F64),
                    spec("dg_dcal", DeviceDType::I32),
                    spec("dh_kcal", DeviceDType::F64),
                    spec("ds_cal_per_k", DeviceDType::F64),
                ],
            ),
            DeviceCall::DuplexPrimer3(call) => (
                Stage::Primer3(crate::resident_p3::ResidentPrimer3Stage::new(lease, &call)?),
                vec![
                    spec("found", DeviceDType::Bool),
                    spec("dg_kcal", DeviceDType::F64),
                    spec("dh_kcal", DeviceDType::F64),
                    spec("ds_cal_per_k", DeviceDType::F64),
                    spec("tm_c", DeviceDType::F64),
                ],
            ),
        };
        let mut specs = Vec::new();
        specs
            .try_reserve_exact(all.len())
            .map_err(|_| EngineError::Device("unable to retain resident stage schema".into()))?;
        specs.extend(
            all.into_iter()
                .filter(|field| requested.contains(&field.name.as_str())),
        );
        Ok(Self {
            a,
            b,
            space,
            next: 0,
            maximum,
            stage,
            specs,
            stream: lease.stream_handle(),
            halt: Arc::new(AtomicBool::new(false)),
            settings: crate::launch::settings(),
        })
    }
    /// Levenshtein's in-pass `b_end`/`n_ends` (levenshtein-cuda.md §13.4 step 1):
    /// `Some(continued)` when the chunks carry them, with the number of items the
    /// device continuation served, whose entries were never written.
    pub fn lev_in_pass(&self) -> Option<u64> {
        match &self.stage {
            Stage::Lev {
                options, continued, ..
            } if options.b_end || options.n_ends => Some(*continued),
            _ => None,
        }
    }
    /// RowStats publishes no per-item field: the chunks then carry no in-pass
    /// columns (the call's request is unchanged; finalize replays nothing there).
    pub fn without_in_pass(&mut self) {
        if let Stage::Lev { options, .. } = &mut self.stage {
            options.b_end = false;
            options.n_ends = false;
            self.specs
                .retain(|spec| !matches!(spec.name.as_str(), "b_end" | "n_ends"));
        }
    }
    /// Work items per chunk (levenshtein-cuda.md §14). Every family keeps the
    /// resident cap of 65,536 items (or a lower `pairs_per_chunk`). Hamming and
    /// Levenshtein, whose per-pair cost the host bounds from the table maxima,
    /// close a chunk at the G1 and levenshtein-cuda.md §4 budgets instead:
    /// `pairs_per_chunk` items or `bytes_per_launch` compared bytes (Hamming) or
    /// `steps_per_launch` word steps (Levenshtein, W·n per pair), whichever
    /// comes first, never below the resident cap and never above 2²² items.
    /// Chunking never changes a result byte.
    fn chunk_cap(&self) -> u64 {
        let requested = self.settings.pairs_per_chunk.max(1);
        if requested <= RESIDENT_CHUNK {
            return requested;
        }
        let [longest_a, longest_b] = self.maximum;
        let (budget, cost) = match self.stage {
            Stage::Hamming => (self.settings.bytes_per_launch, longest_a.max(1)),
            Stage::Lev { .. } => (
                self.settings.steps_per_launch,
                longest_a
                    .div_ceil(64)
                    .max(1)
                    .saturating_mul(longest_b.max(1)),
            ),
            _ => return RESIDENT_CHUNK,
        };
        (budget / cost).clamp(RESIDENT_CHUNK, requested.min(CHEAP_CHUNK_MAX))
    }
    /// Attach the C1 bound of a bounded K1 call (duplex-mfe-bound.md §8) or the C2 bound
    /// of a bounded K3 call (duplex-pf-bound.md §8): `cutoff` is the call's engine-key
    /// cutoff. Every other stage rejects a bound.
    pub fn with_bound(
        mut self,
        lease: &DeviceLease,
        bound: crate::resident_bound::ResidentBound,
        cutoff: Option<i64>,
    ) -> Result<Self, EngineError> {
        let fields = self
            .specs
            .iter()
            .any(|spec| matches!(spec.name.as_str(), "dh_kcal" | "ds_cal_per_k" | "tm_c"));
        let space = self.space;
        use crate::resident_bound::{BoundCall, BoundStage};
        match (&mut self.stage, &bound.call) {
            (Stage::Duplex(stage), BoundCall::Mfe(_)) => {
                stage.set_bound(BoundStage::new(lease, bound, cutoff, fields, space)?);
                Ok(self)
            }
            (Stage::Pf { stage, .. }, BoundCall::Pf(_)) => {
                stage.set_bound(BoundStage::new(lease, bound, cutoff, false, space)?);
                Ok(self)
            }
            _ => Err(EngineError::Device(
                "a bound applies to its own duplex stage only (C1: K1, C2: K3)".into(),
            )),
        }
    }
    /// The C1 bound's outcome counts of pass 0 and of the TopK replay, [certified,
    /// pruned, full, skipped] each (None without a bound).
    pub fn bound_counts(&self, lease: &DeviceLease) -> Result<Option<[[u64; 4]; 2]>, EngineError> {
        match &self.stage {
            Stage::Duplex(stage) => stage.bound_counts(lease),
            Stage::Pf { stage, .. } => stage.bound_counts(lease),
            _ => Ok(None),
        }
    }
    pub fn replay_control(&self) -> Arc<AtomicBool> {
        Arc::clone(&self.halt)
    }
    pub fn pf_terms(&self) -> Option<SharedPfTerms> {
        match &self.stage {
            Stage::Pf { last, .. } => Some(Arc::clone(last)),
            _ => None,
        }
    }
    /// Evaluate only the supplied bounded work geometry, used by streamed tiles.
    pub(crate) fn emit_space(
        &mut self,
        lease: &DeviceLease,
        space: ResidentSpace,
    ) -> Result<ResidentChunk, EngineError> {
        if lease.stream_handle() != self.stream {
            return Err(EngineError::Device(
                "resident producer used with a different consumer stream".into(),
            ));
        }
        self.emit(lease, space)
    }
    fn emit(
        &mut self,
        lease: &DeviceLease,
        space: ResidentSpace,
    ) -> Result<ResidentChunk, EngineError> {
        let mut chunk = match &mut self.stage {
            Stage::Hamming => {
                resident_native::hamming(lease, self.a, self.b, space, self.maximum[0])?
            }
            Stage::Lev {
                mode,
                options,
                continued,
            } => {
                let out = crate::resident_edit_stage::resident_keys_fast(
                    lease,
                    self.a,
                    self.b,
                    space,
                    &crate::detail::DetailCall::Levenshtein(*mode),
                    &self.settings,
                    *options,
                )?;
                *continued = continued.saturating_add(out.continued);
                let mut fields = vec![ResidentField {
                    name: "dist".into(),
                    values: out.keys.clone(),
                    mask_bits: i32::MAX as u64,
                }];
                if let Some(values) = out.b_end {
                    fields.push(ResidentField {
                        name: "b_end".into(),
                        values,
                        mask_bits: i64::MAX as u64,
                    });
                }
                if let Some(values) = out.n_ends {
                    fields.push(ResidentField {
                        name: "n_ends".into(),
                        values,
                        mask_bits: i32::MAX as u64,
                    });
                }
                ResidentChunk {
                    space,
                    fields,
                    keys: out.keys,
                }
            }
            Stage::Align(call) => {
                let keys = crate::resident_edit_stage::resident_keys_fast(
                    lease,
                    self.a,
                    self.b,
                    space,
                    &crate::detail::DetailCall::Align(call),
                    &self.settings,
                    LevStage::default(),
                )?
                .keys;
                let fields = resident_native::primary(
                    lease,
                    &keys,
                    PrimaryFamily::Align,
                    "score",
                    Some("score_fx"),
                )?;
                ResidentChunk {
                    space,
                    keys,
                    fields,
                }
            }
            Stage::Duplex(stage) => {
                let output = stage.stage(lease, self.a, self.b, space, self.maximum)?;
                let mut fields = resident_native::primary(
                    lease,
                    &output.keys,
                    PrimaryFamily::Duplex,
                    "dg_kcal",
                    Some("dg_dcal"),
                )?;
                // Present only in the detail pass, which every request of one runs.
                for (name, values) in [
                    ("dh_kcal", output.dh_kcal),
                    ("ds_cal_per_k", output.ds_cal_per_k),
                    ("tm_c", output.tm_c),
                ] {
                    if let Some(values) = values {
                        fields.push(ResidentField {
                            name: name.into(),
                            values,
                            mask_bits: 0x7ff8000000000000,
                        });
                    }
                }
                ResidentChunk {
                    space,
                    keys: output.keys,
                    fields,
                }
            }
            Stage::Pf {
                stage,
                constants,
                last,
            } => {
                let terms = stage.stage(lease, self.a, self.b, space, self.maximum)?;
                // Testing builds: the stage's Ẑ terms for `duplex_pf_zhat(runner="resident")`.
                #[cfg(feature = "testing")]
                crate::launch::pf_capture_push(space.k0, || terms.to_host_bytes())?;
                let p = space;
                let finalized = pf_finalize::finalize(
                    lease,
                    *constants,
                    PfTerms::Interleaved(&terms),
                    PfTables([0, 0, 0, 0, self.a.address(), self.b.address()]),
                    PfSpace([
                        p.kind, p.k0, p.n, p.r0, p.c0, p.width, p.li, p.lj, p.seg_k, p.seg_i,
                        p.seg_j, p.nseg, p.skip,
                    ]),
                )?;
                *last.lock().map_err(|_| {
                    EngineError::Device("resident PF term handoff was poisoned".into())
                })? = Some(terms);
                ResidentChunk {
                    space,
                    keys: finalized.keys,
                    fields: vec![
                        ResidentField {
                            name: "dg_ens_kcal".into(),
                            values: finalized.dg_ens_kcal,
                            mask_bits: 0x7ff8000000000000,
                        },
                        ResidentField {
                            name: "dg_ens_err_kcal".into(),
                            values: finalized.dg_ens_err_kcal,
                            mask_bits: 0x7ff8000000000000,
                        },
                    ],
                }
            }
            Stage::Register(stage) => stage.stage(lease, self.a, self.b, space)?,
            Stage::Primer3(stage) => {
                let output = stage.stage(lease, self.a, self.b, space, self.maximum)?;
                ResidentChunk {
                    space,
                    keys: output.keys,
                    fields: vec![
                        ResidentField {
                            name: "found".into(),
                            values: output.found,
                            mask_bits: 0,
                        },
                        ResidentField {
                            name: "dg_kcal".into(),
                            values: output.dg_kcal,
                            mask_bits: 0x7ff8000000000000,
                        },
                        ResidentField {
                            name: "dh_kcal".into(),
                            values: output.dh_kcal,
                            mask_bits: 0x7ff8000000000000,
                        },
                        ResidentField {
                            name: "ds_cal_per_k".into(),
                            values: output.ds_cal_per_k,
                            mask_bits: 0x7ff8000000000000,
                        },
                        ResidentField {
                            name: "tm_c".into(),
                            values: output.tm_c,
                            mask_bits: 0x7ff8000000000000,
                        },
                    ],
                }
            }
        };
        let all = mem::take(&mut chunk.fields);
        chunk
            .fields
            .try_reserve_exact(self.specs.len())
            .map_err(|_| EngineError::Device("unable to retain resident fields".into()))?;
        // Family schemas choose the final public order; the producer keeps one
        // stable order across every chunk and both TopK passes.
        for spec in &self.specs {
            if let Some(field) = all.iter().find(|field| field.name == spec.name) {
                chunk.fields.push(field.clone());
            } else {
                return Err(EngineError::Device(
                    "resident stage omitted a declared field".into(),
                ));
            }
        }
        Ok(chunk)
    }
}
impl ResidentProducer for NativeProducer<'_> {
    fn key_dtype(&self) -> DeviceDType {
        match self.stage {
            Stage::Primer3(_) | Stage::Pf { .. } | Stage::Register(_) => DeviceDType::I64,
            Stage::Hamming | Stage::Lev { .. } | Stage::Align(_) | Stage::Duplex(_) => {
                DeviceDType::I32
            }
        }
    }
    fn fields(&self) -> &[ResidentFieldSpec] {
        &self.specs
    }
    fn next_chunk(&mut self, lease: &DeviceLease) -> Result<Option<ResidentChunk>, EngineError> {
        if self.stream != lease.stream_handle() {
            return Err(EngineError::Device(
                "resident producer changed its consumer stream".into(),
            ));
        }
        if self.halt.load(Ordering::Relaxed) || self.next >= self.space.n {
            return Ok(None);
        }
        let n = (self.space.n - self.next).min(self.chunk_cap());
        let space = ResidentSpace {
            k0: self
                .space
                .k0
                .checked_add(self.next)
                .ok_or_else(|| EngineError::Device("resident work ID overflow".into()))?,
            n,
            ..self.space
        };
        let chunk = self.emit(lease, space)?;
        self.next += n;
        Ok(Some(chunk))
    }
    fn reset(&mut self) -> Result<(), EngineError> {
        self.next = 0;
        match &mut self.stage {
            Stage::Duplex(stage) => stage.reset_pass(),
            Stage::Pf { stage, .. } => stage.reset_pass(),
            _ => {}
        }
        Ok(())
    }
    fn replay_limits(&mut self, values: u64, active: u64, mirror: bool) {
        match &mut self.stage {
            Stage::Duplex(stage) => stage.replay_limits(values, active, mirror),
            Stage::Pf { stage, .. } => stage.replay_limits(values, active, mirror),
            _ => {}
        }
    }
}
