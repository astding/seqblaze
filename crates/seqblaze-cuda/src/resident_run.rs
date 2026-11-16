//! Public resident execution composition (registered after the first ABI build
//! checkpoint). Validation, input digests and this execution share one lease.
use crate::{
    interop::pf_finalize::{
        K1LseTable, LseWeight, PfExactRows, PfFinalizeConstants, PfSpace, PfTerms, RowsProgress,
    },
    interop::{DeviceLease, ValidatedDeviceRows},
    resident::{
        ResidentField, ResidentProducer, ResidentRaggedField, ResidentSinkConfig, ResidentWork,
    },
    resident_native::{self, PrimaryFamily},
    resident_producer::NativeProducer,
    resident_sinks, DeviceCall,
};
use seqblaze_core::{
    cancel::{self, Site, Stage},
    sink::Sink,
    validate::EngineError,
};
use std::sync::atomic::Ordering;

pub struct ResidentRun {
    pub columns: Vec<ResidentField>,
    pub ragged: Vec<ResidentRaggedField>,
    pub shape: [usize; 2],
    pub tie_rows: u64,
    /// A bounded K1 call's outcome counts (duplex-mfe-bound.md §6.5): [certified,
    /// pruned, full, skipped] of pass 0 and of the TopK replay.
    pub bound_counts: Option<[[u64; 4]; 2]>,
}
fn failure(message: &str) -> EngineError {
    EngineError::Device(message.into())
}

/// `lse` is K1's host weight routine for RowStats `lse_dg_kcal` (duplex-cuda.md §6.1);
/// K3's terms come from its own producer. `bound`: the C1 bound of a bounded K1 call
/// (duplex-mfe-bound.md §8), whose outcome counts the result reports.
#[allow(clippy::too_many_arguments)]
pub fn run(
    lease: &DeviceLease,
    a: &ValidatedDeviceRows,
    b: &ValidatedDeviceRows,
    work: &ResidentWork,
    call: DeviceCall,
    config: &ResidentSinkConfig,
    fields: &[&str],
    pf: Option<PfFinalizeConstants>,
    lse: Option<LseWeight<'_>>,
    aligned_binary: bool,
    bound: Option<crate::resident_bound::ResidentBound>,
) -> Result<ResidentRun, EngineError> {
    let mut producer = NativeProducer::new(
        lease,
        &a.bound,
        &b.bound,
        work.space,
        [a.summary.max_length, b.summary.max_length],
        call.clone(),
        fields,
        pf,
    )?;
    if config.sink == Sink::RowStats {
        producer.without_in_pass();
    }
    if let Some(bound) = bound {
        producer = producer.with_bound(lease, bound, config.cutoff)?;
    }
    let kernels = &lease.device().0.resident_sinks;
    let output = if config.sink == Sink::RowStats && fields.contains(&"lse_dg_kcal") {
        let spec = pf.ok_or_else(|| {
            failure("row partition-function output requires its exact finalize constants")
        })?;
        // K3: the producer's own F53 terms. K1 (duplex-cuda.md §6.1): terms gathered on
        // the device from the call's host-built table of W(D), kept across replays.
        let k3_terms = producer.pf_terms();
        let mut k1 = match (&call, &k3_terms, lse) {
            (_, Some(_), _) => None,
            (DeviceCall::Duplex(_), None, Some(weight)) => Some((K1LseTable::new(lease), weight)),
            _ => return Err(failure("row partition-function output requires F53 terms")),
        };
        let halt = producer.replay_control();
        let mut rows = PfExactRows::new(lease, work.n_a)?;
        loop {
            let mut observer = |lease: &DeviceLease,
                                chunk: &crate::resident::ResidentChunk,
                                pass: u32|
             -> Result<(), EngineError> {
                if pass != 0 || halt.load(Ordering::Relaxed) {
                    return Ok(());
                }
                let p = chunk.space;
                let space = PfSpace([
                    p.kind, p.k0, p.n, p.r0, p.c0, p.width, p.li, p.lj, p.seg_k, p.seg_i, p.seg_j,
                    p.nseg, p.skip,
                ]);
                let mirror = work.mirror && config.mirror_symmetric;
                let progress = if let Some((table, weight)) = k1.as_mut() {
                    let (keys, terms) =
                        table.chunk_terms(lease, &chunk.keys, config.cutoff, *weight)?;
                    rows.consume(
                        lease,
                        PfTerms::Interleaved(&terms),
                        &keys,
                        space,
                        None,
                        mirror,
                        config.cutoff,
                    )?
                } else {
                    let terms = k3_terms
                        .as_ref()
                        .ok_or_else(|| failure("row partition-function output requires F53 terms"))?
                        .lock()
                        .map_err(|_| failure("resident PF term handoff was poisoned"))?;
                    let terms = terms.as_ref().ok_or_else(|| {
                        failure("resident PF producer did not retain its current terms")
                    })?;
                    rows.consume(
                        lease,
                        PfTerms::Interleaved(terms),
                        &chunk.keys,
                        space,
                        None,
                        mirror,
                        config.cutoff,
                    )?
                };
                if progress == RowsProgress::Replay {
                    halt.store(true, Ordering::Relaxed);
                }
                Ok(())
            };
            let mut result = resident_sinks::run_observed(
                lease,
                kernels,
                work,
                &mut producer,
                config,
                Some(&mut observer),
            )?;
            drop(observer);
            if halt.load(Ordering::Relaxed) {
                drop(result);
                rows.grow_reset(lease)?;
                halt.store(false, Ordering::Relaxed);
                producer.reset()?;
                // A restart round is a host boundary of 0 items
                // (cancellation-progress.md §3.5, key `continuation`).
                cancel::boundary(Site::Continuation, 0)?;
                continue;
            }
            let values = rows.finish(lease, spec)?;
            result.columns.push(ResidentField {
                name: "lse_dg_kcal".into(),
                values,
                mask_bits: 0x7ff8000000000000,
            });
            break result;
        }
    } else if config.sink == Sink::Each
        && config.cutoff.is_none()
        && work.space.kind == 0
        && fields
            .iter()
            .all(|name| producer.fields().iter().any(|spec| spec.name == *name))
        && producer.lev_in_pass().is_none()
    {
        // No detail replay, index entries or selection lineage reads these
        // discarded sink records for zip. Index/cutoff calls keep the old path,
        // and so do levenshtein's in-pass b_end/n_ends: a device-continued item
        // makes finalize drop them and replay both from the sink's keys.
        resident_sinks::each_fields(lease, work, &mut producer, Stage::Pairs)?
    } else {
        resident_sinks::run(lease, kernels, work, &mut producer, config)?
    };
    let in_pass = in_pass(&producer, work);
    let bound_counts = producer.bound_counts(lease)?;
    drop(producer);
    let mut run = finalize(
        lease,
        a,
        b,
        call,
        config,
        fields,
        aligned_binary,
        in_pass,
        output,
    )?;
    run.bound_counts = bound_counts;
    Ok(run)
}

/// Whether levenshtein's in-pass `b_end`/`n_ends` columns are published
/// (levenshtein-cuda.md §13.4 step 1): `None` when the chunks carry none;
/// `Some(false)` when any chunk this finalize consumed had device-continued
/// items (whose entries were never written) or the space mirrors, and the
/// replay then computes both fields for every kept entry as before.
fn in_pass(producer: &NativeProducer<'_>, work: &ResidentWork) -> Option<bool> {
    producer
        .lev_in_pass()
        .map(|continued| continued == 0 && !work.mirror)
}

#[allow(clippy::too_many_arguments)]
fn finalize(
    lease: &DeviceLease,
    a: &ValidatedDeviceRows,
    b: &ValidatedDeviceRows,
    call: DeviceCall,
    config: &ResidentSinkConfig,
    fields: &[&str],
    aligned_binary: bool,
    in_pass: Option<bool>,
    mut output: crate::resident::ResidentSinkOutput,
) -> Result<ResidentRun, EngineError> {
    let detail_names = [
        "b_end",
        "n_ends",
        "b_start",
        "n_starts",
        "ends",
        "cigar",
        "a_start",
        "a_end",
        "aln_len",
        "n_identity",
        "n_similarity",
        "n_gaps",
        "aligned_a",
        "aligned_b",
    ];
    // Published in-pass columns replace the replay of exactly these two fields;
    // otherwise they are dropped and the replay computes them as before.
    let published = in_pass == Some(true);
    if in_pass == Some(false) {
        output
            .columns
            .retain(|field| !matches!(field.name.as_str(), "b_end" | "n_ends"));
    }
    let needs_detail = fields.iter().any(|name| {
        detail_names.contains(name) && !(published && matches!(*name, "b_end" | "n_ends"))
    });
    let detail = match &call {
        DeviceCall::Levenshtein { mode, .. } if needs_detail => {
            Some(crate::detail::DetailCall::Levenshtein(*mode))
        }
        DeviceCall::Align(scoring) if needs_detail => {
            Some(crate::detail::DetailCall::Align(scoring))
        }
        _ => None,
    };
    let mut ragged = Vec::new();
    if let Some(detail) = detail {
        if config.sink != Sink::RowStats {
            if let Some(entries) = &output.entries {
                let detail = crate::detail::resident_details(
                    lease,
                    &a.bound,
                    &b.bound,
                    entries,
                    &detail,
                    fields,
                    aligned_binary,
                    true,
                )?;
                output.columns.extend(detail.numeric);
                ragged = detail.ragged;
            } else {
                let keys = output
                    .columns
                    .iter()
                    .find(|f| f.name == "__keys")
                    .ok_or_else(|| failure("resident dense sink omitted internal keys"))?;
                let passed = output
                    .columns
                    .iter()
                    .find(|f| f.name == "passed")
                    .ok_or_else(|| failure("resident dense sink omitted its validity mask"))?;
                let origin = config.tile.map_or([0, 0], |tile| [tile[0], tile[1]]);
                let detail = resident_native::dense_details(
                    lease,
                    &a.bound,
                    &b.bound,
                    output.shape,
                    origin,
                    &keys.values,
                    &passed.values,
                    &detail,
                    fields,
                    aligned_binary,
                )?;
                output.columns.extend(detail);
            }
        }
    }
    // Internal keys remain private. The same exact device map constructs the
    // public primary for sparse/dense and RowStats; no host numeric finalizer.
    let best = config.sink == Sink::RowStats;
    let internal = if best { "__best_key" } else { "__keys" };
    if let Some(at) = output
        .columns
        .iter()
        .position(|field| field.name == internal)
    {
        let keys = output.columns.remove(at).values;
        let primary = match &call {
            DeviceCall::Hamming | DeviceCall::Levenshtein { .. } => {
                if best {
                    "best_dist"
                } else {
                    "dist"
                }
            }
            DeviceCall::Align(_) => {
                if best {
                    "best_score"
                } else {
                    "score"
                }
            }
            DeviceCall::Duplex(_) | DeviceCall::DuplexPrimer3(_) => {
                if best {
                    "best_dg_kcal"
                } else {
                    "dg_kcal"
                }
            }
            DeviceCall::DuplexPf(_) => {
                if best {
                    "best_dg_ens_kcal"
                } else {
                    "dg_ens_kcal"
                }
            }
            DeviceCall::TmPairs(_) => {
                if best {
                    "best_tm_c"
                } else {
                    "tm_c"
                }
            }
        };
        if !output.columns.iter().any(|field| field.name == primary) {
            let mapped = match &call {
                DeviceCall::Hamming | DeviceCall::Levenshtein { .. } => vec![ResidentField {
                    name: primary.into(),
                    values: keys,
                    mask_bits: i32::MAX as u64,
                }],
                DeviceCall::Align(_) => resident_native::primary(
                    lease,
                    &keys,
                    PrimaryFamily::Align,
                    primary,
                    (!best && fields.contains(&"score_fx")).then_some("score_fx"),
                )?,
                DeviceCall::Duplex(_) => resident_native::primary(
                    lease,
                    &keys,
                    PrimaryFamily::Duplex,
                    primary,
                    (!best && fields.contains(&"dg_dcal")).then_some("dg_dcal"),
                )?,
                DeviceCall::TmPairs(_) => {
                    resident_native::primary(lease, &keys, PrimaryFamily::Register, primary, None)?
                }
                DeviceCall::DuplexPf(_) | DeviceCall::DuplexPrimer3(_) => {
                    resident_native::primary(lease, &keys, PrimaryFamily::Pf, primary, None)?
                }
            };
            output.columns.extend(mapped);
        }
    }
    output.columns.retain(|field| !field.name.starts_with("__"));
    lease.synchronize()?;
    Ok(ResidentRun {
        columns: output.columns,
        ragged,
        shape: output.shape,
        tie_rows: output.tie_rows,
        bound_counts: None,
    })
}

/// Evaluate exactly the current tile, retaining the same common stage/sink and
/// oriented detail finalization as a non-streamed resident call.
#[allow(clippy::too_many_arguments)]
pub fn run_tile(
    lease: &DeviceLease,
    a: &ValidatedDeviceRows,
    b: &ValidatedDeviceRows,
    work: &ResidentWork,
    plan: &crate::resident_tiles::TilePlan,
    region: [usize; 4],
    call: DeviceCall,
    fields: &[&str],
    pf: Option<PfFinalizeConstants>,
    aligned_binary: bool,
    stage: Stage,
) -> Result<(ResidentRun, crate::resident_tiles::TileCounts), EngineError> {
    let mut base = NativeProducer::new(
        lease,
        &a.bound,
        &b.bound,
        work.space,
        [a.summary.max_length, b.summary.max_length],
        call.clone(),
        fields,
        pf,
    )?;
    let config = ResidentSinkConfig {
        sink: Sink::Matrix,
        cutoff: None,
        k: 0,
        mirror_symmetric: true,
        tile: Some(region),
    };
    let mut producer = plan.producer(lease, region, &mut base)?;
    // The tile's stage counts the candidate work items the tile producer hands
    // to the device (cancellation-progress.md §4.2); it reports them itself.
    cancel::stage(stage, producer.candidates());
    let output = resident_sinks::run(
        lease,
        &lease.device().0.resident_sinks,
        work,
        &mut producer,
        &config,
    )?;
    let counts = producer.counts();
    drop(producer);
    let in_pass = in_pass(&base, work);
    let output = finalize(
        lease,
        a,
        b,
        call,
        &config,
        fields,
        aligned_binary,
        in_pass,
        output,
    )?;
    Ok((output, counts))
}
