//! Public CUDA bridge: real host/device storage, one validate/hash/compute lease.
//! This file stays unregistered during the initial native ABI compile checkpoint.
use super::device_lifetime::Detached;
use super::*;
use seqblaze_core::validate::Alphabet;
use seqblaze_cuda::{
    interop::{
        BoundDeviceRows, ControlBlock, ControlWords, DeviceInputSummary, DeviceLease, DigestSlot,
        NullSummary, SummarySlot, ValidatedDeviceRows,
    },
    resident::{ResidentRaggedKind, ResidentSinkConfig},
    resident_validate,
};

fn host_summary(table: &seqblaze_core::seq::SeqTable) -> DeviceInputSummary {
    let mut summary = DeviceInputSummary {
        rows: table.len() as u64,
        valid: (table.len() - table.nulls().len()) as u64,
        min_length: u64::MAX,
        min_row: u64::MAX,
        max_row: u64::MAX,
        bad_row: u64::MAX,
        bad_position: u64::MAX,
        bad_rows: [u64::MAX; 5],
        span_row: u64::MAX,
        length_row: u64::MAX,
        ..DeviceInputSummary::default()
    };
    let mut row = 0u64;
    table.rows().for_each_len(|len| {
        let len = len as u64;
        summary.total += len;
        if len < summary.min_length {
            summary.min_length = len;
            summary.min_row = row;
        }
        if summary.max_row == u64::MAX || len > summary.max_length {
            summary.max_length = len;
            summary.max_row = row;
        }
        row += 1;
    });
    // Preflight rejects nulls; empty tables still have canonical extrema.
    if summary.valid == 0 {
        summary.min_length = 0;
    }
    summary
}

fn bind(
    py: Python<'_>,
    lease: &mut DeviceLease,
    table: &PySeqTable,
    workers: seqblaze_core::sched::Workers,
    scope: &super::cancel::Call,
) -> PyResult<Detached<BoundDeviceRows>> {
    let bound = match &table.table {
        TableStorage::Host(owner) => {
            // Mixed/device-index calls have no host preflight; prepare one
            // current view here. All-host calls bind inside validated below.
            let fresh = owner.fresh_view();
            let host = fresh.as_ref().unwrap_or(owner);
            // Staging units and the upload wait run inside the call's cancel
            // scope (cancellation-progress.md §3.7). The host step after the
            // upload is a `routed` boundary of 0 items (§3.5): an owned table is
            // uploaded from its own buffer, with no staging unit to check.
            scope.region(py, || {
                let bound = lease.bind_host_table(host, owner, workers)?;
                seqblaze_core::cancel::boundary(seqblaze_core::cancel::Site::Routed, 0)?;
                Ok(bound)
            })
        }
        TableStorage::Device { owner, .. } => owner.borrow(py).bind(py, lease),
    }?;
    Ok(Detached::new(bound))
}
// Host indices have already passed WorkSource's range checks. Preserve the
// original B6 reservations and upload order while running inside validated.
fn host_original(
    lease: &mut DeviceLease,
    work: &WorkSource,
) -> Result<seqblaze_cuda::resident::ResidentWork, EngineError> {
    if let seqblaze_core::work::PairKind::Index { i, j } = &work.kind {
        let upload = |column: &seqblaze_core::work::IndexArray| {
            let mut bytes = Vec::new();
            let capacity = column
                .len()
                .checked_mul(8)
                .filter(|n| *n <= isize::MAX as usize)
                .ok_or_else(|| {
                    EngineError::Overflow("resident index bytes exceed platform limits".into())
                })?;
            seqblaze_core::detail::try_reserve_exact(&mut bytes, capacity, "host index upload")?;
            for row in 0..column.len() {
                bytes.extend_from_slice(&(column.get(row) as i64).to_le_bytes());
            }
            lease.upload_bytes(
                seqblaze_cuda::interop::DeviceDType::I64,
                vec![column.len()],
                &bytes,
            )
        };
        let i = upload(i)?;
        let j = upload(j)?;
        seqblaze_cuda::resident_work::from_device_indices(
            lease,
            &i,
            &j,
            [work.shape.0, work.shape.1],
            Sink::Each,
        )
    } else {
        seqblaze_cuda::resident_work::from_host_work(lease, work, Sink::Each)
    }
}

struct Rules {
    legal: [u64; 4],
    strict: bool,
    acgt: bool,
    accepts: &'static str,
    hint: &'static str,
    caps: [Option<usize>; 2],
    ranges: [(usize, usize); 2],
}
fn rules(kernel: &RuntimeKernel, check: bool) -> Rules {
    let (alphabet, caps) = dispatch!(kernel, |k| (k.alphabet(), k.length_caps()));
    let (legal, strict, acgt, accepts, hint) = match alphabet {
        Alphabet::Acgt => {
            let mut legal = [0u64; 4];
            for byte in b"ACGT" {
                legal[*byte as usize / 64] |= 1u64 << (*byte as usize % 64);
            }
            (legal, check, true, "A, C, G, T (upper case)", "")
        }
        Alphabet::Exact {
            legal,
            describe,
            hint,
        } => (
            seqblaze_core::seq::legal_set(legal),
            true,
            false,
            describe,
            hint,
        ),
        Alphabet::Bytes => ([u64::MAX; 4], false, false, "", ""),
    };
    let ranges = match kernel {
        RuntimeKernel::Align { kernel, .. } => [kernel.metadata_length_range(); 2],
        RuntimeKernel::DuplexPrimer3(_) => [(1, 10_000); 2],
        _ => caps.map(|cap| (0, cap.unwrap_or(isize::MAX as usize))),
    };
    Rules {
        legal,
        strict,
        acgt,
        accepts,
        hint,
        caps,
        ranges,
    }
}
fn options(kernel: &RuntimeKernel, options: &ExecOptions<i64>) -> Result<(), EngineError> {
    if options.threads == 0 {
        return Err(EngineError::Parameter("threads must be positive".into()));
    }
    if options.sink == Sink::TopK && options.k.is_none_or(|k| k == 0) {
        return Err(EngineError::Parameter("TopK requires positive k".into()));
    }
    if options.sink != Sink::TopK && options.k.is_some() {
        return Err(EngineError::Parameter(
            "k is only supported with TopK".into(),
        ));
    }
    if options.sink == Sink::Matrix {
        if options.cutoff.is_some() {
            return Err(EngineError::Parameter(
                "Matrix does not support cutoff".into(),
            ));
        }
        dispatch!(kernel, |k| seqblaze_core::detail::dense_request_bytes(
            k.detail_fields(),
            k.detail_request()
        ))?;
    }
    Ok(())
}
fn first(summary: DeviceInputSummary) -> Option<(usize, usize)> {
    (summary.length_row != u64::MAX)
        .then_some((summary.length_row as usize, summary.length_value as usize))
}
fn longest(summary: DeviceInputSummary) -> (usize, usize) {
    (
        summary.max_length as usize,
        if summary.max_row == u64::MAX {
            0
        } else {
            summary.max_row as usize
        },
    )
}
fn length_error(
    kernel: &RuntimeKernel,
    input: &'static str,
    summary: DeviceInputSummary,
    cap: usize,
) -> EngineError {
    dispatch!(kernel, |k| k.length_error(
        input,
        summary.length_row as usize,
        summary.length_value as usize,
        cap
    ))
}

/// The device-side checks of one call, enqueued back to back into the lease's
/// control block and read once (per-call cost S6): both tables' validation,
/// their null-row summaries (check_bases=False), the pair check and, for
/// host-origin tables, the digests.
struct Checks {
    nulls: [Option<NullSummary>; 2],
    pair: bool,
    words: Option<ControlWords>,
}

// Preserve the independent row, pairing and kernel validation controls. The
// host applies today's precedence to the control block's values: table a's
// rows, then table b's, then the pair rules, with today's messages.
#[allow(clippy::too_many_arguments)]
fn validate(
    lease: &DeviceLease,
    kernel: &RuntimeKernel,
    a: &ValidatedDeviceRows,
    b: &ValidatedDeviceRows,
    shared: bool,
    rules: &Rules,
    checks: &Checks,
) -> Result<usize, EngineError> {
    let pair = || {
        checks
            .words
            .as_ref()
            .filter(|_| checks.pair)
            .and_then(ControlWords::pair)
    };
    // Step 2: structural row metadata, then equal-length pair rules, before
    // any public sequence/alphabet error. Original explicit order is retained.
    resident_validate::check_spans(lease, a)?;
    if !shared {
        resident_validate::check_spans(lease, b)?;
    }
    if matches!(kernel, RuntimeKernel::Hamming | RuntimeKernel::TmPairs(_)) {
        if let Some([k, i, j, la, lb]) = pair() {
            let requirement = if matches!(kernel, RuntimeKernel::TmPairs(_)) {
                "tm on Pairs requires equal lengths (register pairing)"
            } else {
                "hamming requires equal lengths"
            };
            return Err(EngineError::Pairing(format!("{requirement}: pair {k} ({i}, {j}) has lengths {la} and {lb}; nothing was computed")));
        }
    }
    let mut n_illegal = 0usize;
    // Both input alphabets precede every per-input length cap. Shared inputs
    // are scanned and counted once, exactly as core validation.
    for (side, (input, table)) in [("a", a), ("b", b)].into_iter().enumerate() {
        if side == 1 && shared {
            continue;
        }
        let summary = if rules.strict {
            table.summary
        } else {
            match (&checks.nulls[side], &checks.words) {
                (Some(null), Some(words)) => null.summary(words),
                _ => table.summary,
            }
        };
        if let Some(error) = resident_validate::sequence_error(
            summary,
            input,
            kernel.name(),
            rules.accepts,
            rules.hint,
        ) {
            return Err(error);
        }
        if rules.acgt {
            n_illegal = n_illegal
                .checked_add((table.summary.bad_count - summary.bad_count) as usize)
                .ok_or_else(|| {
                    EngineError::Overflow("illegal sequence count exceeds the platform".into())
                })?;
        }
    }
    for (side, (input, table)) in [("a", a), ("b", b)].into_iter().enumerate() {
        if let Some(cap) = rules.caps[side] {
            if table.summary.length_row != u64::MAX {
                return Err(length_error(kernel, input, table.summary, cap));
            }
        }
    }
    match kernel {
        RuntimeKernel::Align { kernel, .. } => kernel.metadata_preflight(
            [first(a.summary), first(b.summary)],
            [longest(a.summary), longest(b.summary)],
            shared,
        )?,
        RuntimeKernel::DuplexPrimer3(_) => {
            for (input, table) in [("a", a), ("b", b)] {
                if table.summary.length_row != u64::MAX {
                    return Err(length_error(kernel, input, table.summary, 10_000));
                }
            }
            if let Some([k, i, j, la, lb]) = pair() {
                // Provided only by the separately reviewed metadata-only hook
                // patch; no copied diagnostics or source payload is required.
                return Err(seqblaze_thermo::dimer::metadata_pair_error(
                    k as usize,
                    i as usize,
                    j as usize,
                    la as usize,
                    lb as usize,
                ));
            }
        }
        _ => {}
    }
    Ok(n_illegal)
}
/// The digest of a host-origin table batched with its validation: its host
/// byte total sized the digest, so the device's validated total must agree.
fn batched_digest(
    rows: &ValidatedDeviceRows,
    words: Option<&ControlWords>,
    slot: DigestSlot,
) -> Result<[u8; 32], EngineError> {
    match (rows.host_total(), words) {
        (Some(total), Some(words))
            if rows.summary.total == total && rows.summary.span_row == u64::MAX =>
        {
            Ok(words.digest(slot))
        }
        _ => Err(EngineError::Device(
            "device validation disagrees with the uploaded host table; no result was published"
                .into(),
        )),
    }
}
fn digest_record<'py>(
    py: Python<'py>,
    hash: [u8; 32],
    rows: &ValidatedDeviceRows,
    text: bool,
) -> PyResult<Bound<'py, PyDict>> {
    let record = PyDict::new(py);
    record.set_item(
        "sha256",
        hash.iter().map(|b| format!("{b:02x}")).collect::<String>(),
    )?;
    record.set_item("n", rows.summary.rows)?;
    record.set_item("total_nt", rows.summary.total)?;
    record.set_item("min_len", rows.summary.min_length)?;
    record.set_item("max_len", rows.summary.max_length)?;
    record.set_item("text", text)?;
    Ok(record)
}

pub(super) struct Prepared {
    pub lease: Detached<DeviceLease>,
    pub rows_a: Detached<ValidatedDeviceRows>,
    pub rows_b: Option<Detached<ValidatedDeviceRows>>,
    pub original: Detached<seqblaze_cuda::resident::ResidentWork>,
    pub host_work: Option<WorkSource>,
    pub call: DeviceCall,
    pub pf: Option<seqblaze_cuda::interop::pf_finalize::PfFinalizeConstants>,
    pub shared: bool,
    pub n_illegal: usize,
    pub input_digests: Py<PyDict>,
    pub pair_indices: Py<PyDict>,
}
#[allow(clippy::too_many_arguments)]
pub(super) fn prepare<'py>(
    py: Python<'py>,
    kernel: &RuntimeKernel,
    a: &PySeqTable,
    b: &PySeqTable,
    kind: &str,
    i: Option<&Bound<'py, PyAny>>,
    j: Option<&Bound<'py, PyAny>>,
    self_pairs: bool,
    orientation: Option<&str>,
    options_value: &ExecOptions<i64>,
    device: &seqblaze_cuda::Device,
    stream: u64,
    scope: &super::cancel::Call,
    arena: bool,
) -> PyResult<Prepared> {
    options(kernel, options_value).map_err(|e| error(py, e))?;
    py.detach(|| {
        seqblaze_cuda::interop::execution_stream::BorrowedExecutionStream::validate(device, stream)
    })
    .map_err(|e| error(py, e))?;
    // An all-host call must finish the same preflight as CPU before even
    // uploading triangular work controls or explicit indices. Device/mixed
    // inputs retain the device validation path below, without payload D2H.
    let host_indices = if kind == "index" {
        i.zip(j).and_then(|(i, j)| {
            Some((
                i.extract::<PyRef<'_, PyIndexColumn>>().ok()?,
                j.extract::<PyRef<'_, PyIndexColumn>>().ok()?,
            ))
        })
    } else {
        None
    };
    let host_indices_given = host_indices.is_some();
    let host_work = if kind != "index" || host_indices.is_some() {
        let (i, j) = host_indices.map_or((None, None), |(i, j)| (Some(i), Some(j)));
        Some(super::oriented_work_source(
            py,
            kind,
            i,
            j,
            self_pairs,
            orientation,
            a.table.len(),
            b.table.len(),
        )?)
    } else {
        None
    };
    let rules = rules(kernel, options_value.check_bases);
    // Build scalar/table call constants after option/device resolution and
    // before sequence checks, preserving the existing E4 public order.
    let call = py
        .detach(|| {
            let _fp = FpGuard::new();
            kernel.device_call()
        })
        .map_err(|e| error(py, e))?;
    // A strict ACGT alphabet (check_bases=True) admits only A, C, G and T, so the
    // resident levenshtein fast tiers take the four-code kernels (levenshtein-cuda.md
    // §13.4 step 2); every other legal set keeps the raw ones.
    let call = match call {
        DeviceCall::Levenshtein {
            mode,
            b_end,
            n_ends,
            ..
        } => DeviceCall::Levenshtein {
            mode,
            b_end,
            n_ends,
            acgt: rules.strict && rules.acgt,
        },
        call => call,
    };
    let pf = match kernel {
        RuntimeKernel::DuplexPf(kernel) => {
            Some(super::pf_device_finalize::constants(kernel.call()))
        }
        // K1 RowStats lse_dg_kcal (duplex-cuda.md §6.1): the finalize of its LSE constants.
        RuntimeKernel::Duplex { kernel, .. } => kernel.lse_constants().map(|constants| {
            super::pf_device_finalize::with(constants, kernel.call().pack.loop_max)
        }),
        _ => None,
    };
    let shared = std::ptr::eq(a, b);
    let (lease, original, bound_a, bound_b, preflight) =
        if let (TableStorage::Host(owner_a), TableStorage::Host(owner_b), Some(work)) =
            (&a.table, &b.table, host_work.as_ref())
        {
            // S10(c): the lease, work upload and bindings consume the validated
            // per-call views inside `then`. No second FixedBytes::lengths pass.
            let prepared = scope
                .region(py, || {
                    dispatch!(kernel, |k| {
                        let options = options_value.with_key()?;
                        exec::validated(k, work, owner_a, owner_b, &options, |a, b, report| {
                            let summaries = [host_summary(a), host_summary(b)];
                            let hashes = [
                                super::provenance::single_leaf_digest(a, summaries[0].total),
                                super::provenance::single_leaf_digest(b, summaries[1].total),
                            ];
                            let mut lease = DeviceLease::with_stream(device, stream)?;
                            if !arena {
                                lease.execution().disable_arena();
                            }
                            let original = host_original(&mut lease, work)?;
                            let bound_a = lease
                                .bind_host_table(a, owner_a, options_value.workers())?
                                .with_host_summary(summaries[0]);
                            seqblaze_core::cancel::boundary(
                                seqblaze_core::cancel::Site::Routed,
                                0,
                            )?;
                            let bound_b = if shared {
                                None
                            } else {
                                let bound = lease
                                    .bind_host_table(b, owner_b, options_value.workers())?
                                    .with_host_summary(summaries[1]);
                                seqblaze_core::cancel::boundary(
                                    seqblaze_core::cancel::Site::Routed,
                                    0,
                                )?;
                                Some(bound)
                            };
                            Ok((
                                lease,
                                original,
                                bound_a,
                                bound_b,
                                (summaries, report.n_illegal_seqs, hashes),
                            ))
                        })
                    })
                })
                .map(Detached::new)?;
            let (lease, original, a, b, preflight) = prepared.take();
            (
                Detached::new(lease),
                Detached::new(original),
                Detached::new(a),
                b.map(Detached::new),
                Some(preflight),
            )
        } else {
            let mut lease = py
                .detach(|| DeviceLease::with_stream(device, stream))
                .map(Detached::new)
                .map_err(|e| error(py, e))?;
            if !arena {
                lease.execution().disable_arena();
            }
            let original = if kind == "index" {
                let i = i.ok_or_else(|| error(py, EngineError::Pairing("missing i".into())))?;
                let j = j.ok_or_else(|| error(py, EngineError::Pairing("missing j".into())))?;
                let i = Detached::new(super::device_indices::normalized(py, &mut lease, i, "i")?);
                let j = Detached::new(super::device_indices::normalized(py, &mut lease, j, "j")?);
                let shape = [a.table.len(), b.table.len()];
                scope.region(py, || {
                    seqblaze_cuda::resident_work::from_device_indices(
                        &mut lease,
                        &i,
                        &j,
                        shape,
                        Sink::Each,
                    )
                })?
            } else {
                scope.region(py, || {
                    seqblaze_cuda::resident_work::from_host_work(
                        &lease,
                        host_work.as_ref().expect("implicit work"),
                        Sink::Each,
                    )
                })?
            };
            let original = Detached::new(original);
            let bound_a = bind(py, &mut lease, a, options_value.workers(), scope)?;
            let bound_b = if shared {
                None
            } else {
                Some(bind(py, &mut lease, b, options_value.workers(), scope)?)
            };
            (lease, original, bound_a, bound_b, None)
        };
    // Per-call cost S6: one control block per lease. Both tables' validation,
    // the null-row summaries, the pair check and, when both tables originate
    // on the host, every digest are enqueued back to back and read once; the
    // host then applies today's precedence. Device and mixed inputs read their
    // digests separately, after the checks that guard their offsets and bytes.
    let pair_mode = match kernel {
        RuntimeKernel::Hamming | RuntimeKernel::TmPairs(_) => Some(0),
        RuntimeKernel::DuplexPrimer3(_) => Some(1),
        _ => None,
    };
    let host_index = kind == "index" && host_indices_given;
    let checked = scope
        .region(py, || {
            let block = ControlBlock::new(&lease)?;
            let host_tables = bound_a.host_total().is_some()
                && bound_b.as_ref().is_none_or(|b| b.host_total().is_some());
            let batch_digests = host_tables && (kind != "index" || host_index);
            if batch_digests && kind == "index" {
                // First, so that each index column's own-stream wait finds
                // this lease's earlier work done.
                for (column, slot) in original.owners.iter().zip([DigestSlot::I, DigestSlot::J]) {
                    seqblaze_cuda::interop::enqueue_index_digest(&lease, column, &block, slot)?;
                }
            }
            let mut rows_a = seqblaze_cuda::interop::enqueue_device_rows(
                &lease,
                bound_a.take(),
                rules.legal,
                rules.ranges[0].0,
                rules.ranges[0].1,
                &block,
                SummarySlot::A,
            )?;
            let mut rows_b = bound_b
                .map(|bound| {
                    seqblaze_cuda::interop::enqueue_device_rows(
                        &lease,
                        bound.take(),
                        rules.legal,
                        rules.ranges[1].0,
                        rules.ranges[1].1,
                        &block,
                        SummarySlot::B,
                    )
                })
                .transpose()?;
            let table_b = rows_b.as_ref().unwrap_or(&rows_a);
            let mut nulls = [None, None];
            if !rules.strict && preflight.is_none() {
                nulls[0] = seqblaze_cuda::interop::enqueue_null_summary(
                    &lease,
                    &rows_a,
                    &block,
                    SummarySlot::NullA,
                )?;
                if !shared {
                    nulls[1] = seqblaze_cuda::interop::enqueue_null_summary(
                        &lease,
                        table_b,
                        &block,
                        SummarySlot::NullB,
                    )?;
                }
            }
            let pair = match pair_mode {
                Some(mode) if preflight.is_none() => resident_validate::enqueue_pair_check(
                    &lease,
                    &rows_a,
                    table_b,
                    original.space,
                    self_pairs,
                    mode,
                    &block,
                )?,
                _ => false,
            };
            if batch_digests {
                for (rows, slot) in [(&rows_a, DigestSlot::A), (table_b, DigestSlot::B)] {
                    if slot == DigestSlot::B && shared {
                        continue;
                    }
                    let side = usize::from(slot == DigestSlot::B);
                    if preflight.as_ref().is_some_and(|p| p.2[side].is_some()) {
                        continue;
                    }
                    let total = rows.host_total().expect("host-origin table");
                    seqblaze_cuda::interop::enqueue_sequence_digest(
                        &lease, rows, total, &block, slot,
                    )?;
                }
            }
            let device_digests = batch_digests
                && (kind == "index"
                    || preflight
                        .as_ref()
                        .is_none_or(|p| p.2[0].is_none() || (!shared && p.2[1].is_none())));
            let enqueued = (preflight.is_none()
                && (rows_a.row_count() != 0 || table_b.row_count() != 0))
                || device_digests;
            let words = if enqueued {
                let words = block.read(&lease)?;
                rows_a.settle(&words);
                if let Some(rows_b) = rows_b.as_mut() {
                    rows_b.settle(&words);
                }
                Some(words)
            } else {
                None
            };
            let checks = Checks { nulls, pair, words };
            let n_illegal = validate(
                &lease,
                kernel,
                &rows_a,
                rows_b.as_ref().unwrap_or(&rows_a),
                shared,
                &rules,
                &checks,
            )?;
            let n_illegal = preflight.as_ref().map_or(n_illegal, |p| p.1);
            // The host step after device validation is a boundary of 0 items: the
            // device scanned the rows, the host counted none (cancellation-progress.md §3.5).
            seqblaze_core::cancel::boundary(seqblaze_core::cancel::Site::Validate, 0)?;
            Ok((
                block,
                rows_a,
                rows_b,
                checks.words,
                batch_digests,
                n_illegal,
            ))
        })
        .map(Detached::new)?;
    let (block, rows_a, rows_b_owned, words, batch_digests, n_illegal) = checked.take();
    let rows_a = Detached::new(rows_a);
    let rows_b_owned = rows_b_owned.map(Detached::new);
    let block = Detached::new(block);
    let rows_b = rows_b_owned.as_ref().map_or(&*rows_a, |rows| &**rows);
    // The digests: from the control block when batched, otherwise enqueued now
    // and read once (device and mixed inputs).
    let digest_words = if batch_digests {
        None
    } else {
        Some(
            py.detach(|| {
                for (rows, slot) in [(&*rows_a, DigestSlot::A), (rows_b, DigestSlot::B)] {
                    if slot == DigestSlot::B && shared {
                        continue;
                    }
                    if rows.summary.span_row != u64::MAX {
                        return Err(EngineError::Parameter(format!(
                            "device sequence row {} has invalid offsets or length",
                            rows.summary.span_row
                        )));
                    }
                    seqblaze_cuda::interop::enqueue_sequence_digest(
                        &lease,
                        rows,
                        rows.summary.total,
                        &block,
                        slot,
                    )?;
                }
                if kind == "index" {
                    for (column, slot) in original.owners.iter().zip([DigestSlot::I, DigestSlot::J])
                    {
                        seqblaze_cuda::interop::enqueue_index_digest(&lease, column, &block, slot)?;
                    }
                }
                block.read_as(&lease, "reading a logical device input digest")
            })
            .map_err(|e| error(py, e))?,
        )
    };
    let table_digest = |rows: &ValidatedDeviceRows, slot| -> Result<[u8; 32], EngineError> {
        let side = usize::from(slot == DigestSlot::B);
        if let Some(hash) = preflight.as_ref().and_then(|p| p.2[side]) {
            return Ok(hash);
        }
        match &digest_words {
            Some(words) => Ok(words.digest(slot)),
            None => batched_digest(rows, words.as_ref(), slot),
        }
    };
    let digests = PyDict::new(py);
    let hash = table_digest(&rows_a, DigestSlot::A).map_err(|e| error(py, e))?;
    digests.set_item("a", digest_record(py, hash, &rows_a, a.table.is_text())?)?;
    if !shared {
        let hash = table_digest(rows_b, DigestSlot::B).map_err(|e| error(py, e))?;
        digests.set_item("b", digest_record(py, hash, rows_b, b.table.is_text())?)?;
    } else {
        digests.set_item("b", digests.get_item("a")?.expect("inserted"))?;
    }
    let pair_indices = PyDict::new(py);
    if kind == "index" {
        let words = digest_words.as_ref().or(words.as_ref()).ok_or_else(|| {
            error(
                py,
                EngineError::Device("device index digests were not computed".into()),
            )
        })?;
        for ((name, column), slot) in ["i", "j"]
            .into_iter()
            .zip(&original.owners)
            .zip([DigestSlot::I, DigestSlot::J])
        {
            let hash = words.digest(slot);
            let record = PyDict::new(py);
            record.set_item(
                "sha256",
                hash.iter().map(|b| format!("{b:02x}")).collect::<String>(),
            )?;
            record.set_item("n", column.shape()[0])?;
            digests.set_item(name, record)?;
            pair_indices.set_item(
                name,
                Py::new(
                    py,
                    super::device::PyDeviceColumn {
                        column: Detached::new(column.clone()),
                    },
                )?,
            )?;
        }
    }
    Ok(Prepared {
        lease,
        rows_a,
        rows_b: rows_b_owned,
        original,
        host_work,
        call,
        pf,
        shared,
        n_illegal,
        input_digests: digests.unbind(),
        pair_indices: pair_indices.unbind(),
    })
}

pub(super) fn requested(kernel: &RuntimeKernel) -> (&'static str, Vec<&'static str>, bool) {
    let mut requested = Vec::new();
    let primary = match kernel {
        RuntimeKernel::Hamming | RuntimeKernel::Levenshtein(_) => "dist",
        RuntimeKernel::Align { .. } => "score",
        RuntimeKernel::Duplex { .. } | RuntimeKernel::DuplexPrimer3(_) => "dg_kcal",
        RuntimeKernel::DuplexPf(_) => "dg_ens_kcal",
        RuntimeKernel::TmPairs(_) => "tm_c",
        #[cfg(feature = "testing")]
        RuntimeKernel::LengthDiff => "dist",
    };
    requested.push(primary);
    let fields = dispatch!(kernel, |k| k.detail_fields());
    let request = dispatch!(kernel, |k| k.detail_request());
    requested.extend(
        fields
            .iter()
            .enumerate()
            .filter(|(at, _)| request.contains(*at))
            .map(|(_, field)| field.name),
    );
    // PF row sums are requested independently of per-pair detail fields.
    // RuntimeKernel::configure stores this opt-in in Kernel::row_terms().
    if matches!(kernel, RuntimeKernel::DuplexPf(pf) if pf.row_terms())
        || matches!(kernel, RuntimeKernel::Duplex { kernel, .. } if kernel.row_terms())
    {
        requested.push("lse_dg_kcal");
    }
    if let RuntimeKernel::Align { score_fx: true, .. } = kernel {
        requested.push("score_fx");
    }
    if let RuntimeKernel::Duplex { dg_dcal: true, .. } = kernel {
        requested.push("dg_dcal");
    }
    requested.dedup();
    let aligned_binary = fields.iter().any(|field| {
        matches!(field.name, "aligned_a" | "aligned_b") && field.kind == FieldKind::Binary
    });
    (primary, requested, aligned_binary)
}
pub(super) fn publish_columns<'py>(
    py: Python<'py>,
    kernel: &RuntimeKernel,
    sink: Sink,
    primary: &str,
    requested: &[&str],
    numeric: Vec<seqblaze_cuda::resident::ResidentField>,
    ragged: Vec<seqblaze_cuda::resident::ResidentRaggedField>,
) -> PyResult<Bound<'py, PyDict>> {
    let mut numeric = Detached::new(numeric);
    let mut ragged = Detached::new(ragged);
    let columns = PyDict::new(py);

    let mut names = Vec::new();
    if sink == Sink::RowStats {
        names.push(format!("best_{primary}"));
        names.extend(["argbest", "n_best", "count", "lse_dg_kcal"].map(str::to_owned));
    } else {
        for &name in kernel.bool_fields() {
            if requested.contains(&name) {
                names.push(name.to_owned());
            }
        }
        names.push(primary.into());
        // Match set_primary: fixed-point companions precede detail fields.
        match kernel {
            RuntimeKernel::Align { score_fx: true, .. } => names.push("score_fx".into()),
            RuntimeKernel::Duplex { dg_dcal: true, .. } => names.push("dg_dcal".into()),
            _ => {}
        }
        names.extend(requested.iter().map(|name| (*name).to_owned()));
        names.extend(["passed", "i", "j", "rank", "offsets", "tie_at_k"].map(str::to_owned));
    }
    for name in names {
        if columns.contains(&name)? {
            continue;
        }
        if let Some(at) = numeric.iter().position(|field| field.name == name) {
            let field = numeric.remove(at);
            columns.set_item(
                name,
                Py::new(
                    py,
                    super::device::PyDeviceColumn {
                        column: Detached::new(field.values),
                    },
                )?,
            )?;
        } else if let Some(at) = ragged.iter().position(|field| field.name == name) {
            let field = ragged.remove(at);
            let offsets = Detached::new(field.offsets);
            let values = Detached::new(field.values);
            let validity = field.validity.map(Detached::new);
            let kind = match field.kind {
                ResidentRaggedKind::Utf8 => "string",
                ResidentRaggedKind::Binary => "binary",
                ResidentRaggedKind::ListI64 => "list",
            };
            let offsets = Py::new(py, super::device::PyDeviceColumn { column: offsets })?;
            let values = Py::new(py, super::device::PyDeviceColumn { column: values })?;
            let valid = validity
                .map(|column| Py::new(py, super::device::PyDeviceColumn { column }))
                .transpose()?;
            columns.set_item(name, (kind, offsets, values, valid))?;
        }
    }
    if !numeric.is_empty() || !ragged.is_empty() {
        return Err(error(
            py,
            EngineError::Device("resident result contains fields outside its public schema".into()),
        ));
    }
    Ok(columns)
}

#[allow(clippy::too_many_arguments)]
pub(super) fn run<'py>(
    py: Python<'py>,
    kernel: &RuntimeKernel,
    a: &PySeqTable,
    b: &PySeqTable,
    kind: &str,
    i: Option<&Bound<'py, PyAny>>,
    j: Option<&Bound<'py, PyAny>>,
    self_pairs: bool,
    orientation: Option<&str>,
    options_value: &ExecOptions<i64>,
    strategy: &str,
    device: &seqblaze_cuda::Device,
    stream: u64,
    progress: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
    // One cancel scope from validation to the sink run (cancellation-progress.md §3.7).
    let scope = super::cancel::Call::new(progress);
    let Prepared {
        mut lease,
        rows_a,
        rows_b,
        original,
        host_work,
        call,
        pf,
        shared,
        n_illegal,
        input_digests,
        pair_indices,
    } = prepare(
        py,
        kernel,
        a,
        b,
        kind,
        i,
        j,
        self_pairs,
        orientation,
        options_value,
        device,
        stream,
        &scope,
        true,
    )?;
    let rows_b = rows_b.as_ref().unwrap_or(&rows_a);
    let n_pairs = original.original_items;
    // The original-order work geometry (CPU work IDs); its control columns live until
    // the run ends, owned by `original` or `actual`.
    let original_space = original.space;
    let digests = input_digests.bind(py);
    let pair_indices = pair_indices.bind(py);
    let actual = if options_value.sink == Sink::Each {
        original
    } else if kind == "index" {
        let i = &original.owners[0];
        let j = &original.owners[1];
        scope
            .region(py, || {
                seqblaze_cuda::resident_work::from_device_indices(
                    &mut lease,
                    i,
                    j,
                    [original.n_a, original.n_b],
                    options_value.sink,
                )
            })
            .map(Detached::new)?
    } else {
        scope
            .region(py, || {
                seqblaze_cuda::resident_work::from_host_work(
                    &lease,
                    host_work.as_ref().expect("implicit work"),
                    options_value.sink,
                )
            })
            .map(Detached::new)?
    };
    let (primary, requested, aligned_binary) = requested(kernel);
    let config = ResidentSinkConfig {
        sink: options_value.sink,
        cutoff: options_value.cutoff,
        k: options_value.k.unwrap_or(0),
        mirror_symmetric: true,
        tile: None,
    };
    // K1 RowStats lse_dg_kcal (duplex-cuda.md §6.1): the CPU path's weight routine,
    // W(D) as the table record (M, e) with value M · 2^(e − 52).
    let lse_constants = match kernel {
        RuntimeKernel::Duplex { kernel, .. } => kernel.lse_constants().cloned(),
        _ => None,
    };
    let weight = |d: i64| -> Result<[u64; 2], EngineError> {
        let constants = lse_constants
            .as_ref()
            .ok_or_else(|| EngineError::Parameter("internal error: no lse constants".into()))?;
        let (m, e) = seqblaze_core::fpenv::guarded(|| constants.weight(d))?.parts();
        Ok([m, e.wrapping_add(52) as u64])
    };
    let lse = lse_constants
        .is_some()
        .then_some(&weight as seqblaze_cuda::interop::pf_finalize::LseWeight<'_>);
    // The C1 bound of a bounded K1 call (duplex-mfe-bound.md §8): the plan resolved
    // `bounded` only for a K1 call whose certificate passes and whose RowStats asks no
    // lse_dg_kcal. The weak-bound decision probes CPU work IDs, so the original-order
    // work (duplicates of explicit pairs included) goes with it.
    let bound = match kernel {
        RuntimeKernel::Duplex { kernel, .. } if strategy == "bounded" => kernel
            .certificate()
            .map(|cert| -> Result<_, EngineError> {
                let call = kernel.call();
                let bound_call = super::duplex::device_bound_call(call, cert)?;
                let (format, lam) = (call.pack.format, call.pack.loop_max);
                // The device decision (duplex-mfe-bound.md §6.6.2) uses the device tables
                // of §6.6.3, never the CPU's; the testing override is read once per call.
                let override_costs = seqblaze_core::plan::device_costs_override();
                Ok(seqblaze_cuda::resident_bound::ResidentBound {
                    call: seqblaze_cuda::resident_bound::BoundCall::Mfe(Arc::new(bound_call)),
                    lb_plus_one: seqblaze_core::plan::bound_fault()
                        == seqblaze_core::plan::BoundFault::LbPlusOne,
                    cost: Arc::new(move |n, m| {
                        seqblaze_thermo::duplex::bound::device_cost(
                            format,
                            lam,
                            n,
                            m,
                            override_costs,
                        )
                    }),
                    original: original_space,
                    original_items: n_pairs,
                    // RowStats runs two sweeps over the same work (§8 *Sweeps*).
                    sweeps: options_value.sink == Sink::RowStats,
                })
            })
            .transpose()
            .map_err(|e| error(py, e))?,
        // The C2 bound of a bounded K3 call (duplex-pf-bound.md §8): the plan resolved
        // `bounded` only for a K3 call whose certificate passes and that asks no
        // lse_dg_kcal; the device decision uses K3's device tables (§6.6).
        RuntimeKernel::DuplexPf(kernel) if strategy == "bounded" => kernel
            .certificate()
            .map(|cert| -> Result<_, EngineError> {
                let call = kernel.call();
                let bound_call = super::duplex::device_pf_bound_call(call, cert)?;
                let (format, lam) = (call.call.pack.format, call.call.pack.loop_max);
                let override_costs = seqblaze_core::plan::device_costs_override();
                Ok(seqblaze_cuda::resident_bound::ResidentBound {
                    call: seqblaze_cuda::resident_bound::BoundCall::Pf(Arc::new(bound_call)),
                    lb_plus_one: false,
                    cost: Arc::new(move |n, m| {
                        seqblaze_thermo::duplex::pf_bound::device_cost(
                            format,
                            lam,
                            n,
                            m,
                            override_costs,
                        )
                    }),
                    original: original_space,
                    original_items: n_pairs,
                    sweeps: options_value.sink == Sink::RowStats,
                })
            })
            .transpose()
            .map_err(|e| error(py, e))?,
        _ => None,
    };
    let output = scope.region(py, || {
        seqblaze_cuda::resident_run::run(
            &lease,
            &rows_a,
            rows_b,
            &actual,
            call,
            &config,
            &requested,
            pf,
            lse,
            aligned_binary,
            bound,
        )
    })?;
    if let Some(raised) = scope.exception() {
        return Err(raised);
    }
    let columns = publish_columns(
        py,
        kernel,
        options_value.sink,
        primary,
        &requested,
        output.columns,
        output.ragged,
    )?;
    let report = PyDict::new(py);
    let counts = PyDict::new(py);
    // A bounded K1 call reports its device outcome counts (duplex-mfe-bound.md §6.5);
    // every other call computed each of its n_pairs work items.
    let [first, replay_counts] = output
        .bound_counts
        .unwrap_or([[0, 0, n_pairs as u64, 0], [0, 0, n_pairs as u64, 0]]);
    counts.set_item("certified", first[0])?;
    counts.set_item("pruned", first[1])?;
    counts.set_item("full", first[2])?;
    report.set_item("counts", counts)?;
    report.set_item("threads", options_value.threads)?;
    report.set_item("host_pairs", 0)?;
    let flags = PyDict::new(py);
    flags.set_item("n_illegal_seqs", n_illegal)?;
    if options_value.sink == Sink::TopK {
        flags.set_item("tie_at_k", output.tie_rows)?;
    }
    report.set_item("flags", flags)?;
    let scans = PyDict::new(py);
    scans.set_item("a", 1)?;
    scans.set_item("b", usize::from(!shared))?;
    report.set_item("validation_scans", scans)?;
    let strategy_info = PyDict::new(py);
    strategy_info.set_item("requested", strategy)?;
    strategy_info.set_item("effective", strategy)?;
    strategy_info.set_item("fallback_reason", py.None())?;
    report.set_item("strategy", strategy_info)?;
    if options_value.sink == Sink::TopK {
        let topk = PyDict::new(py);
        let replay = PyDict::new(py);
        replay.set_item("full", replay_counts[2])?;
        replay.set_item("pruned", replay_counts[1])?;
        replay.set_item("skipped", replay_counts[3])?;
        replay.set_item("certified", replay_counts[0])?;
        topk.set_item("passes", 2)?;
        topk.set_item("replay", replay)?;
        report.set_item("topk", topk)?;
    }
    let result = PyDict::new(py);
    result.set_item("columns", columns)?;
    result.set_item("shape", (output.shape[0], output.shape[1]))?;
    result.set_item("report", report)?;
    result.set_item("input_digests", digests)?;
    result.set_item("pair_indices", pair_indices)?;
    trace_check(py)?;
    py.detach(|| lease.execution().publish_packs());
    Ok(result)
}

/// Testing builds: a refused reservation of the execution-stream trace during
/// this call raises its §3 DeviceError before the call publishes (F4 round 3;
/// always `Ok` in other builds).
pub(super) fn trace_check(py: Python<'_>) -> PyResult<()> {
    seqblaze_cuda::interop::execution_stream::resident_launch_streams_check()
        .map_err(|e| error(py, e))
}

#[pyfunction(signature=(a,b,kind,i,j,self_pairs,keep,cutoff,check_bases,strategy,threads,k=None,kernel="hamming",grain="auto",device=None,mode="global",details=Vec::new(),orientation=None,align=None,duplex=None,stream=1,stream_owner=None,progress=None))]
#[allow(clippy::too_many_arguments)]
fn _native_resident_run<'py>(
    py: Python<'py>,
    a: PyRef<'py, PySeqTable>,
    b: PyRef<'py, PySeqTable>,
    kind: &str,
    i: Option<Bound<'py, PyAny>>,
    j: Option<Bound<'py, PyAny>>,
    self_pairs: bool,
    keep: &str,
    cutoff: Option<i64>,
    check_bases: bool,
    strategy: &str,
    threads: usize,
    k: Option<usize>,
    kernel: &str,
    grain: &str,
    device: Option<u64>,
    mode: &str,
    details: Vec<String>,
    orientation: Option<&str>,
    align: Option<Bound<'py, PyDict>>,
    duplex: Option<Bound<'py, PyAny>>,
    stream: u64,
    stream_owner: Option<Py<PyAny>>,
    progress: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
    #[cfg(feature = "testing")]
    seqblaze_cuda::interop::execution_stream::resident_launch_streams_reset();
    let _stream_owner = stream_owner;
    super::duplex::prepare_bounded(
        py,
        duplex.as_ref(),
        strategy,
        keep,
        cutoff.is_some(),
        &details,
    );
    let kernel = RuntimeKernel::configure(
        kernel,
        mode,
        &details,
        check_bases,
        align.as_ref(),
        duplex.as_ref(),
    )
    .map_err(|e| error(py, e))?;
    kernel.check_cutoff(cutoff).map_err(|e| error(py, e))?;
    let grain = parse_grain(py, grain)?;
    kernel.check_backend(device).map_err(|e| error(py, e))?;
    let device = resolve_device(py, device)?.ok_or_else(|| {
        error(
            py,
            EngineError::Device("resident compute requires a CUDA device".into()),
        )
    })?;
    let options = ExecOptions {
        strategy: if strategy == "bounded" {
            Strategy::Bounded
        } else {
            Strategy::Full
        },
        threads,
        grain,
        sink: match keep {
            "each" => Sink::Each,
            "hits" => Sink::Hits,
            "topk" => Sink::TopK,
            "rowstats" => Sink::RowStats,
            "matrix" => Sink::Matrix,
            _ => return Err(error(py, EngineError::Parameter("unsupported sink".into()))),
        },
        cutoff,
        check_bases,
        k,
    };
    run(
        py,
        &kernel,
        &a,
        &b,
        kind,
        i.as_ref(),
        j.as_ref(),
        self_pairs,
        orientation,
        &options,
        strategy,
        &device,
        stream,
        progress,
    )
}
pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(_native_resident_run, module)?)?;
    Ok(())
}

/// The unary family shares the same validated device rows, digest and stream;
/// host input is uploaded once, and every result field is device-owned.
#[allow(clippy::too_many_arguments)]
pub(super) fn run_unary<'py, K: seqblaze_core::unary::UnaryKernel<Cutoff = f64>>(
    py: Python<'py>,
    kernel: &K,
    table: &PySeqTable,
    call: seqblaze_cuda::UnaryCall,
    options: &seqblaze_core::unary::UnaryOptions<f64>,
    device: &seqblaze_cuda::Device,
    stream: u64,
    progress: Option<&str>,
) -> PyResult<Bound<'py, PyDict>> {
    // One cancel scope from validation to the unary run (cancellation-progress.md §3.7).
    let scope = super::cancel::Call::new(progress);
    if options.threads == 0 {
        return Err(error(
            py,
            EngineError::Parameter("threads must be positive".into()),
        ));
    }
    if options.sink == Sink::Hits && options.cutoff.is_none() {
        return Err(error(
            py,
            EngineError::Parameter("Hits requires a cutoff".into()),
        ));
    }
    let fields = kernel.columns().map_err(|e| error(py, e))?.into_columns();
    let names = fields.iter().map(|field| field.name).collect::<Vec<_>>();
    py.detach(|| {
        seqblaze_cuda::interop::execution_stream::BorrowedExecutionStream::validate(device, stream)
    })
    .map_err(|e| error(py, e))?;
    let (lease, bound, host_hash) = if let TableStorage::Host(owner) = &table.table {
        let prepared = scope
            .region(py, || {
                seqblaze_core::unary::validated(kernel, owner, options, |host, _| {
                    let summary = host_summary(host);
                    let hash = super::provenance::single_leaf_digest(host, summary.total);
                    let mut lease = DeviceLease::with_stream(device, stream)?;
                    let bound = lease
                        .bind_host_table(host, owner, options.workers())?
                        .with_host_summary(summary);
                    seqblaze_core::cancel::boundary(seqblaze_core::cancel::Site::Routed, 0)?;
                    Ok((lease, bound, hash))
                })
            })
            .map(Detached::new)?;
        let (lease, bound, hash) = prepared.take();
        (Detached::new(lease), Detached::new(bound), hash)
    } else {
        let mut lease = py
            .detach(|| DeviceLease::with_stream(device, stream))
            .map(Detached::new)
            .map_err(|e| error(py, e))?;
        let bound = bind(py, &mut lease, table, options.workers(), &scope)?;
        (lease, bound, None)
    };
    let mut legal = [0u64; 4];
    for &byte in b"ACGT" {
        legal[byte as usize / 64] |= 1u64 << (byte as usize % 64);
    }
    let (min, max) = kernel.length_limits();
    // Per-call cost S6: validation and, for a host-origin table, its digest
    // enqueued back to back into the lease's control block and read once.
    let checked = scope
        .region(py, || {
            let block = ControlBlock::new(&lease)?;
            let mut rows = seqblaze_cuda::interop::enqueue_device_rows(
                &lease,
                bound.take(),
                legal,
                min,
                max,
                &block,
                SummarySlot::A,
            )?;
            let host_total = rows.host_total();
            if let Some(total) = host_total.filter(|_| host_hash.is_none()) {
                seqblaze_cuda::interop::enqueue_sequence_digest(
                    &lease,
                    &rows,
                    total,
                    &block,
                    DigestSlot::A,
                )?;
            }
            let words = if host_hash.is_none() && (rows.row_count() != 0 || host_total.is_some()) {
                let words = block.read(&lease)?;
                rows.settle(&words);
                Some(words)
            } else {
                None
            };
            resident_validate::check_spans(&lease, &rows)?;
            if let Some(error) = resident_validate::sequence_error(
                rows.summary,
                "a",
                kernel.name(),
                "A, C, G, T (upper case)",
                "",
            ) {
                return Err(error);
            }
            if rows.summary.length_row != u64::MAX {
                let length = rows.summary.length_value as usize;
                return Err(kernel.length_error(
                    rows.summary.length_row as usize,
                    length,
                    if length < min { min } else { max },
                ));
            }
            Ok((block, rows, words))
        })
        .map(Detached::new)?;
    let (block, rows, words) = checked.take();
    let block = Detached::new(block);
    let rows = Detached::new(rows);
    let hash = py
        .detach(|| {
            if let Some(hash) = host_hash {
                return Ok(hash);
            }
            if rows.host_total().is_some() {
                return batched_digest(&rows, words.as_ref(), DigestSlot::A);
            }
            if rows.summary.span_row != u64::MAX {
                return Err(EngineError::Parameter(format!(
                    "device sequence row {} has invalid offsets or length",
                    rows.summary.span_row
                )));
            }
            seqblaze_cuda::interop::enqueue_sequence_digest(
                &lease,
                &rows,
                rows.summary.total,
                &block,
                DigestSlot::A,
            )?;
            let words = block.read_as(&lease, "reading a logical device input digest")?;
            Ok(words.digest(DigestSlot::A))
        })
        .map_err(|e| error(py, e))?;
    drop(block);
    let inputs = PyDict::new(py);
    inputs.set_item("a", digest_record(py, hash, &rows, table.table.is_text())?)?;
    let decreasing = !matches!(call, seqblaze_cuda::UnaryCall::Hairpin(_));
    let cutoff = options.cutoff.map(|value| {
        let bits = value.to_bits();
        let magnitude = (bits & 0x7fff_ffff_ffff_ffff) as i64;
        if decreasing {
            if bits >> 63 != 0 {
                magnitude
            } else {
                -magnitude
            }
        } else if bits >> 63 != 0 {
            -magnitude
        } else {
            magnitude
        }
    });
    let columns_out = scope.region(py, || {
        seqblaze_cuda::resident_unary::run(
            &lease,
            &rows.bound,
            call,
            &names,
            options.sink,
            cutoff,
            rows.summary.max_length,
        )
    })?;
    if let Some(raised) = scope.exception() {
        return Err(raised);
    }
    let mut columns_out = Detached::new(columns_out);
    let columns = PyDict::new(py);
    // CPU publishes requested fields first, then Each's passed or Hits' i.
    // Reorder metadata only; every column keeps its original device owner.
    for name in names.iter().copied().chain(["passed", "i"]) {
        let Some(at) = columns_out.iter().position(|field| field.name == name) else {
            continue;
        };
        let field = columns_out.remove(at);
        columns.set_item(
            field.name,
            Py::new(
                py,
                super::device::PyDeviceColumn {
                    column: Detached::new(field.values),
                },
            )?,
        )?;
    }
    if !columns_out.is_empty() {
        return Err(error(
            py,
            EngineError::Device(
                "unary resident result contains fields outside its public schema".into(),
            ),
        ));
    }
    let report = PyDict::new(py);
    let counts = PyDict::new(py);
    counts.set_item("certified", 0)?;
    counts.set_item("pruned", 0)?;
    counts.set_item("full", rows.summary.rows)?;
    report.set_item("counts", counts)?;
    report.set_item("threads", options.threads)?;
    report.set_item("host_pairs", 0)?;
    let flags = PyDict::new(py);
    flags.set_item("n_illegal_seqs", 0)?;
    report.set_item("flags", flags)?;
    let scans = PyDict::new(py);
    scans.set_item("a", 1)?;
    report.set_item("validation_scans", scans)?;
    let result = PyDict::new(py);
    result.set_item("columns", columns)?;
    result.set_item("shape", (rows.summary.rows, 1usize))?;
    result.set_item("report", report)?;
    result.set_item("input_digests", inputs)?;
    trace_check(py)?;
    py.detach(|| lease.execution().publish_packs());
    Ok(result)
}

#[allow(clippy::too_many_arguments)]
pub(super) fn unary_dispatch<'py, K, F>(
    py: Python<'py>,
    kernel: &K,
    table: &PySeqTable,
    keep: &str,
    cutoff: Option<f64>,
    threads: usize,
    grain: &str,
    ordinal: Option<u64>,
    stream: u64,
    build: F,
    progress: Option<&str>,
) -> PyResult<Bound<'py, PyDict>>
where
    K: seqblaze_core::unary::UnaryKernel<Cutoff = f64>,
    F: FnOnce() -> Result<seqblaze_cuda::UnaryCall, EngineError> + Send,
{
    #[cfg(feature = "testing")]
    seqblaze_cuda::interop::execution_stream::resident_launch_streams_reset();
    let grain = parse_grain(py, grain)?;
    let sink = match keep {
        "each" => Sink::Each,
        "hits" => Sink::Hits,
        _ => {
            return Err(error(
                py,
                EngineError::Parameter(format!(
                "{} returns one row per sequence; use keep=sb.keep.each() or sb.keep.hits(cutoff)",
                kernel.name()
            )),
            ))
        }
    };
    if cutoff.is_some_and(|value| (value.to_bits() & 0x7FFF_FFFF_FFFF_FFFF) > 0x7FF0_0000_0000_0000)
    {
        return Err(error(
            py,
            EngineError::Parameter("cutoff must not be NaN".into()),
        ));
    }
    let options = seqblaze_core::unary::UnaryOptions {
        threads,
        grain,
        sink,
        cutoff,
    };
    let device = resolve_device(py, ordinal)?.ok_or_else(|| {
        error(
            py,
            EngineError::Device("resident compute requires a CUDA device".into()),
        )
    })?;
    let call = py
        .detach(|| {
            let _fp = FpGuard::new();
            build()
        })
        .map_err(|e| error(py, e))?;
    run_unary(py, kernel, table, call, &options, &device, stream, progress)
}
