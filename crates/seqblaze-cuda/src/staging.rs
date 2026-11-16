//! Host staging of a view table for upload (docs/wiki/gpu/cuda-backend.md,
//! "Data path"; tm-cuda.md §5.1): the concatenated row bytes and int64
//! offsets of a table that does not own a contiguous batch (NumPy and Arrow
//! views). Rows are copied in contiguous ranges by [`staging_parts`] threads;
//! the bytes never depend on the split. Compiled in every build so that its
//! property is unit-tested without a GPU.
use seqblaze_core::{
    cancel::{self, Site},
    detail::try_reserve_exact,
    fpenv::FpGuard,
    sched::Workers,
    seq::SeqTable,
    validate::EngineError,
};

/// Each staging worker copies at least this many bytes: tables below it are
/// staged by one thread, and a table of B bytes by at most B / this workers.
pub(crate) const PARALLEL_STAGING_BYTES: usize = 1 << 20;

/// Number of staging workers for `n` rows of `total` bytes: the scheduler's
/// count (`threads` and its grain), further capped at one worker per
/// `part_bytes` of rows, so a huge `threads` (or `grain="exact"`) never starts
/// one OS thread per row for a memcpy. At least 1.
pub(crate) fn staging_parts(n: usize, total: usize, workers: Workers, part_bytes: usize) -> usize {
    let by_bytes = total / part_bytes.max(1);
    workers
        .count(n, total / n.max(1))
        .min(by_bytes)
        .min(n)
        .max(1)
}

/// The concatenated bytes and n + 1 int64 offsets (from 0) of `table`.
pub(crate) fn stage_view(
    table: &SeqTable,
    workers: Workers,
    parallel_bytes: usize,
) -> Result<(Vec<u8>, Vec<i64>), EngineError> {
    let n = table.len();
    let mut offsets = Vec::new();
    try_reserve_exact(&mut offsets, n + 1, "staging offsets")?;
    offsets.push(0i64);
    let mut total = 0usize;
    for i in 0..n {
        total = total
            .checked_add(table.seq_len(i))
            .filter(|&total| i64::try_from(total).is_ok())
            .ok_or_else(|| EngineError::Overflow("sequence bytes exceed platform limit".into()))?;
        offsets.push(total as i64);
    }
    let mut bytes = Vec::new();
    try_reserve_exact(&mut bytes, total, "staging buffer")?;
    bytes.resize(total, 0);
    let parts = staging_parts(n, total, workers, parallel_bytes);
    // Staging units are CPU units of key `routed` (cancellation-progress.md §3.5).
    let fill = |rows: std::ops::Range<usize>, out: &mut [u8]| -> Result<(), EngineError> {
        let base = offsets[rows.start] as usize;
        let mut ticker = cancel::Ticker::new(Site::Routed);
        for i in rows {
            let (lo, hi) = (offsets[i] as usize - base, offsets[i + 1] as usize - base);
            table.get(i).copy_to(0, &mut out[lo..hi]);
            ticker.tick()?;
        }
        ticker.finish()
    };
    // Every staging unit, inline or spawned, holds its own FP guard
    // (text-nulls-fp-environment.md §7.3 item 1); the copy itself has no FP.
    if parts == 1 {
        let _guard = FpGuard::new();
        let _unit = cancel::inline_unit();
        fill(0..n, &mut bytes)?;
    } else {
        let captured = cancel::current();
        std::thread::scope(|scope| -> Result<(), EngineError> {
            let (done, finished) = std::sync::mpsc::channel();
            let mut pending = Vec::with_capacity(parts);
            let mut failure = None;
            let mut rest: &mut [u8] = &mut bytes;
            let mut start = 0;
            for part in 0..parts {
                let end = n * (part + 1) / parts;
                let len = (offsets[end] - offsets[start]) as usize;
                let (head, tail) = rest.split_at_mut(len);
                rest = tail;
                let fill = &fill;
                // Built inside the closure: a failed spawn signals nothing (§3.4).
                let (signal, captured) = (done.clone(), captured.clone());
                match std::thread::Builder::new().spawn_scoped(scope, move || {
                    let _signal = cancel::Done(signal);
                    let _guard = FpGuard::new();
                    let _unit = cancel::unit(captured);
                    fill(start..end, head)
                }) {
                    Ok(handle) => pending.push(handle),
                    Err(e) => {
                        failure =
                            Some(EngineError::Device(format!("cannot start CPU worker: {e}")));
                        break;
                    }
                }
                start = end;
            }
            drop(done);
            // A timed wait on the calling thread (§3.4), then every unit is joined.
            cancel::wait_done(&finished, pending.len());
            for handle in pending {
                let result = handle
                    .join()
                    .map_err(|_| EngineError::Device("CPU worker panicked".into()))
                    .and_then(|result| result);
                if let Err(error) = result {
                    failure.get_or_insert(error);
                }
            }
            match failure {
                Some(error) => Err(error),
                None => cancel::check(),
            }
        })?;
    }
    Ok((bytes, offsets))
}

#[cfg(test)]
mod tests {
    use super::*;
    use seqblaze_core::batch::SeqBatch;

    #[test]
    fn every_worker_split_stages_the_same_bytes() {
        let rows: Vec<Vec<u8>> = (0..37usize)
            .map(|i| (0..(i * 7) % 23).map(|k| b"ACGTN"[(i + k) % 5]).collect())
            .collect();
        let table = SeqTable::new(SeqBatch::from_seqs(&rows));
        let want: Vec<u8> = rows.concat();
        let mut want_off = vec![0i64];
        for row in &rows {
            want_off.push(want_off.last().unwrap() + row.len() as i64);
        }
        for limit in [1, 2, 3, 16, 64, 1 << 20] {
            for parallel_bytes in [0, 1, 5, usize::MAX] {
                let (bytes, off) =
                    stage_view(&table, Workers::exact(limit), parallel_bytes).unwrap();
                assert_eq!(bytes, want, "limit={limit} parallel_bytes={parallel_bytes}");
                assert_eq!(off, want_off);
            }
        }
        // 1 << 20 threads requested over 37 rows: still one worker per row at most.
        assert_eq!(
            staging_parts(37, want.len(), Workers::exact(1 << 20), 1),
            37
        );
        let empty = SeqTable::new(SeqBatch::from_seqs(Vec::<Vec<u8>>::new()));
        assert_eq!(
            stage_view(&empty, Workers::exact(4), 0).unwrap(),
            (vec![], vec![0])
        );
    }

    #[test]
    fn staging_workers_are_capped_by_bytes() {
        let mib = PARALLEL_STAGING_BYTES;
        // threads=10**6 with an exact grain over 60,000 20-mers (1.2 MB): one worker.
        assert_eq!(
            staging_parts(60_000, 1_200_000, Workers::exact(1_000_000), mib),
            1
        );
        // Below the threshold: one worker whatever the limit.
        assert_eq!(staging_parts(5_000, mib - 1, Workers::exact(64), mib), 1);
        // 10 MiB: at most 10 workers, and never more than the limit.
        assert_eq!(
            staging_parts(50_000, 10 * mib, Workers::exact(1_000_000), mib),
            10
        );
        assert_eq!(staging_parts(50_000, 10 * mib, Workers::exact(3), mib), 3);
        // Zero rows or a zero limit still give one (inline) part.
        assert_eq!(staging_parts(0, 0, Workers::exact(8), mib), 1);
        assert_eq!(staging_parts(10, 10 * mib, Workers::exact(0), mib), 1);
    }
}
