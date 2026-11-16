//! Disjoint canonical work-ID ranges shared by CPU execution stages.
//! Spec: docs/wiki/engine/planner-scheduler.md ("Tiles laziness and scheduler boundary").
use crate::{cancel, fpenv::FpGuard, validate::EngineError};

/// Fixed per-item overhead, in byte-comparison units, added to each item's byte estimate.
pub const ITEM_OVERHEAD: u64 = 32;
/// Under [`Grain::Auto`], each worker gets at least this much estimated work
/// (byte-comparison units, roughly 0.1-0.3 ms of scalar work), so that thread
/// start-up never dominates small stages such as a small tile.
pub const MIN_WORKER_COST: u64 = 1 << 18;

/// How many of the permitted workers a stage may start.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Grain {
    /// Start another worker only per [`MIN_WORKER_COST`] of estimated work.
    Auto,
    /// Start `min(limit, items)` workers, however small the stage. The testing
    /// bridge uses this so tiny fixtures still exercise partitions and merges.
    Exact,
}

/// Resolved CPU worker limit plus its scheduling grain. Neither changes outputs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Workers {
    pub limit: usize,
    pub grain: Grain,
}

impl Workers {
    pub fn exact(limit: usize) -> Self {
        Self {
            limit,
            grain: Grain::Exact,
        }
    }

    /// Worker count for `n_items` items of about `item_bytes` bytes each.
    /// Returns 0 only for a zero limit, which callers reject before scheduling.
    pub fn count(self, n_items: usize, item_bytes: usize) -> usize {
        let cap = self.limit.min(n_items.max(1));
        match self.grain {
            Grain::Exact => cap,
            Grain::Auto => {
                let cost = n_items as u128 * (item_bytes as u128 + u128::from(ITEM_OVERHEAD));
                let by_cost = cost.div_ceil(u128::from(MIN_WORKER_COST)).max(1);
                cap.min(usize::try_from(by_cost).unwrap_or(usize::MAX))
            }
        }
    }
}

/// Canonical half-open range of worker `worker` out of `workers` over `n` items.
pub(crate) fn bounds(n: usize, worker: usize, workers: usize) -> (usize, usize) {
    let start = (n as u128 * worker as u128 / workers as u128) as usize;
    let end = (n as u128 * (worker + 1) as u128 / workers as u128) as usize;
    (start, end)
}

/// Run `compute` over `min(worker_limit, n_items)` disjoint canonical ranges and
/// return the parts in range order. One worker runs inline on the caller.
///
/// Every unit, spawned or inline, runs under its own [`FpGuard`] (E0), installed
/// before any floating-point operation; a worker never relies on inheriting the
/// creating thread's environment (text-nulls-fp-environment.md §7.3 item 1).
/// Spawned units enter the caller's cancel scope; the caller waits for them
/// with timed polls and joins every one of them before it returns
/// (cancellation-progress.md §3.4).
pub fn map_ranges<T: Send, F>(
    n_items: usize,
    worker_limit: usize,
    compute: &F,
) -> Result<Vec<T>, EngineError>
where
    F: Fn(usize, usize) -> Result<T, EngineError> + Sync,
{
    // A required stage reports a refused start, so it is never skipped (None).
    ranges(n_items, worker_limit, compute, Refusal::Fail).map(Option::unwrap_or_default)
}

/// [`map_ranges`] for an optional pass: Ok(None) when one of its workers
/// cannot start. Every started unit is joined first and its result, an error
/// included, is discarded, so the refused start reports nothing; the caller
/// then runs as without the pass. The passes before a call's validation steps
/// use it, so that a refused thread start never comes before a validation
/// error (engine-layout.md §5).
pub fn try_map_ranges<T: Send, F>(
    n_items: usize,
    worker_limit: usize,
    compute: &F,
) -> Result<Option<Vec<T>>, EngineError>
where
    F: Fn(usize, usize) -> Result<T, EngineError> + Sync,
{
    ranges(n_items, worker_limit, compute, Refusal::Skip)
}

/// What a stage does when one of its workers cannot start.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Refusal {
    /// Fail with DeviceError ("cannot start CPU worker").
    Fail,
    /// Skip the stage (an optional pass), after joining every started unit.
    Skip,
}

/// [`map_ranges`] and [`try_map_ranges`]; None only under [`Refusal::Skip`].
fn ranges<T: Send, F>(
    n_items: usize,
    worker_limit: usize,
    compute: &F,
    refusal: Refusal,
) -> Result<Option<Vec<T>>, EngineError>
where
    F: Fn(usize, usize) -> Result<T, EngineError> + Sync,
{
    let workers = worker_limit.min(n_items.max(1));
    if workers == 1 {
        let _guard = FpGuard::new();
        let _unit = cancel::inline_unit();
        return Ok(Some(vec![compute(0, n_items)?]));
    }
    let scope_handle = cancel::current();
    std::thread::scope(|scope| {
        let (done, finished) = std::sync::mpsc::channel();
        let mut pending = Vec::with_capacity(workers);
        let mut refused = None;
        for worker in 0..workers {
            let (start, end) = bounds(n_items, worker, workers);
            // The unit's completion signal is built inside the closure, so a
            // failed spawn (which drops the closure unrun) signals nothing (§3.4).
            let (signal, captured) = (done.clone(), scope_handle.clone());
            match std::thread::Builder::new().spawn_scoped(scope, move || {
                let _signal = cancel::Done(signal);
                let _guard = FpGuard::new();
                let _unit = cancel::unit(captured);
                compute(start, end)
            }) {
                Ok(handle) => pending.push(handle),
                Err(e) => {
                    refused = Some(EngineError::Device(format!("cannot start CPU worker: {e}")));
                    break;
                }
            }
        }
        drop(done);
        cancel::wait_done(&finished, pending.len());
        let mut parts = Vec::with_capacity(pending.len());
        let mut failure = None;
        for worker in pending {
            let result = worker
                .join()
                .map_err(|_| EngineError::Device("CPU worker panicked".into()))
                .and_then(|result| result);
            match result {
                Ok(part) => parts.push(part),
                Err(error) => {
                    failure.get_or_insert(error);
                }
            }
        }
        if refused.is_some() && refusal == Refusal::Skip {
            return Ok(None);
        }
        // A refused start first, then the first unit failure in canonical order.
        match refused.or(failure) {
            Some(error) => Err(error),
            None => cancel::check().map(|()| Some(parts)),
        }
    })
}

/// Fill `out` in place from `min(worker_limit, out.len())` disjoint canonical
/// chunks; `fill(start, chunk)` writes items `start..start + chunk.len()`.
/// Each unit, spawned or inline, holds its own [`FpGuard`]; spawned units enter
/// the caller's cancel scope and are all joined before it returns.
pub fn fill_chunks<T: Send, F>(
    out: &mut [T],
    worker_limit: usize,
    fill: &F,
) -> Result<(), EngineError>
where
    F: Fn(usize, &mut [T]) -> Result<(), EngineError> + Sync,
{
    // A required stage reports a refused start, so it is never skipped.
    chunks(out, worker_limit, fill, Refusal::Fail).map(|_| ())
}

/// [`fill_chunks`] for an optional pass, as [`try_map_ranges`]: Ok(false)
/// when one of its workers cannot start, after every started unit is joined
/// and its result discarded. `out` is then partly filled and not to be used.
pub fn try_fill_chunks<T: Send, F>(
    out: &mut [T],
    worker_limit: usize,
    fill: &F,
) -> Result<bool, EngineError>
where
    F: Fn(usize, &mut [T]) -> Result<(), EngineError> + Sync,
{
    chunks(out, worker_limit, fill, Refusal::Skip)
}

/// [`fill_chunks`] and [`try_fill_chunks`]; false only under [`Refusal::Skip`].
fn chunks<T: Send, F>(
    out: &mut [T],
    worker_limit: usize,
    fill: &F,
    refusal: Refusal,
) -> Result<bool, EngineError>
where
    F: Fn(usize, &mut [T]) -> Result<(), EngineError> + Sync,
{
    let n = out.len();
    let workers = worker_limit.min(n.max(1));
    if workers <= 1 {
        let _guard = FpGuard::new();
        let _unit = cancel::inline_unit();
        return fill(0, out).map(|()| true);
    }
    let scope_handle = cancel::current();
    std::thread::scope(|scope| {
        let (done, finished) = std::sync::mpsc::channel();
        let mut pending = Vec::with_capacity(workers);
        let mut refused = None;
        let mut rest = out;
        for worker in 0..workers {
            let (start, end) = bounds(n, worker, workers);
            let (chunk, tail) = rest.split_at_mut(end - start);
            rest = tail;
            // The unit's completion signal is built inside the closure, so a
            // failed spawn (which drops the closure unrun) signals nothing (§3.4).
            let (signal, captured) = (done.clone(), scope_handle.clone());
            match std::thread::Builder::new().spawn_scoped(scope, move || {
                let _signal = cancel::Done(signal);
                let _guard = FpGuard::new();
                let _unit = cancel::unit(captured);
                fill(start, chunk)
            }) {
                Ok(handle) => pending.push(handle),
                Err(e) => {
                    refused = Some(EngineError::Device(format!("cannot start CPU worker: {e}")));
                    break;
                }
            }
        }
        drop(done);
        cancel::wait_done(&finished, pending.len());
        // Join every worker; report the first failure in canonical order.
        let mut first = None;
        for worker in pending {
            let result = worker
                .join()
                .map_err(|_| EngineError::Device("CPU worker panicked".into()))
                .and_then(|result| result);
            if let Err(error) = result {
                first.get_or_insert(error);
            }
        }
        if refused.is_some() && refusal == Refusal::Skip {
            return Ok(false);
        }
        match first.or(refused) {
            Some(error) => Err(error),
            None => cancel::check().map(|()| true),
        }
    })
}

/// Deliver bounded result chunks to one collector; no complete result list is retained.
/// Each unit, spawned or inline, holds its own [`FpGuard`] while it computes. Every
/// task record is one item of the unit's `tile` checkpoints (cancellation-progress.md
/// §3.2); the collector receives with timed polls on the calling thread.
pub fn stream_ranges<T: Send, I, F, C>(
    n_items: usize,
    worker_limit: usize,
    compute: &F,
    consume: &mut C,
) -> Result<(), EngineError>
where
    I: Iterator<Item = Result<T, EngineError>>,
    F: Fn(usize, usize) -> I + Sync,
    C: FnMut(T) -> Result<(), EngineError>,
{
    const CHUNK: usize = 4096;
    if worker_limit == 0 {
        return Err(EngineError::Parameter("threads must be positive".into()));
    }
    let workers = worker_limit.min(n_items.max(1));
    if workers == 1 {
        let _guard = FpGuard::new();
        let _unit = cancel::inline_unit();
        let mut ticker = cancel::Ticker::new(cancel::Site::Tile);
        for value in compute(0, n_items) {
            consume(value?)?;
            ticker.tick()?;
        }
        return ticker.finish();
    }
    let scope_handle = cancel::current();
    std::thread::scope(|scope| -> Result<(), EngineError> {
        let (sender, receiver) = std::sync::mpsc::sync_channel(workers);
        let mut pending = Vec::with_capacity(workers);
        let mut failure = None;
        for worker in 0..workers {
            let (start, end) = bounds(n_items, worker, workers);
            let sender = sender.clone();
            let captured = scope_handle.clone();
            let spawned = std::thread::Builder::new().spawn_scoped(scope, move || {
                let _guard = FpGuard::new();
                let _unit = cancel::unit(captured);
                let capacity = CHUNK.min(end - start);
                let mut records = Vec::with_capacity(capacity);
                let mut ticker = cancel::Ticker::new(cancel::Site::Tile);
                for value in compute(start, end) {
                    match value.and_then(|value| ticker.tick().map(|()| value)) {
                        Ok(value) => records.push(value),
                        Err(error) => {
                            let _ = sender.send(Err(error));
                            return;
                        }
                    }
                    if records.len() == CHUNK {
                        let ready = std::mem::replace(&mut records, Vec::with_capacity(capacity));
                        if sender.send(Ok(ready)).is_err() {
                            return;
                        }
                    }
                }
                if let Err(error) = ticker.finish() {
                    let _ = sender.send(Err(error));
                    return;
                }
                if !records.is_empty() {
                    let _ = sender.send(Ok(records));
                }
            });
            match spawned {
                Ok(handle) => pending.push(handle),
                Err(error) => {
                    failure = Some(EngineError::Device(format!(
                        "cannot start tile worker: {error}"
                    )));
                    break;
                }
            }
        }
        drop(sender);
        // Drain until every worker has gone (a bounded channel would block a
        // worker that still sends after the first failure).
        while let Ok(packet) = cancel::recv(&receiver) {
            match packet {
                Err(error) => {
                    failure.get_or_insert(error);
                }
                Ok(records) => {
                    if failure.is_none() {
                        for record in records {
                            if let Err(error) = consume(record) {
                                failure = Some(error);
                                break;
                            }
                        }
                    }
                }
            }
        }
        for worker in pending {
            if worker.join().is_err() {
                failure.get_or_insert_with(|| EngineError::Device("tile worker panicked".into()));
            }
        }
        match failure {
            Some(error) => Err(error),
            None => cancel::check(),
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn auto_grain_runs_small_stages_inline_and_caps_by_limit() {
        let auto = Workers {
            limit: 16,
            grain: Grain::Auto,
        };
        // 64x64 tile of 20-mers: about 213k units, below one worker's minimum.
        assert_eq!(auto.count(64 * 64, 20), 1);
        assert_eq!(auto.count(0, 20), 1);
        // Large stages use the whole limit.
        assert_eq!(auto.count(1 << 20, 20), 16);
        // Long rows reach the minimum with few items.
        assert_eq!(auto.count(3, 1 << 20), 3);
        assert_eq!(Workers::exact(16).count(5, 0), 5);
        assert_eq!(Workers::exact(0).count(5, 0), 0);
    }

    #[test]
    fn fill_chunks_covers_every_item_once_in_canonical_chunks() {
        for workers in [1, 2, 3, 7, 16] {
            let mut out = vec![usize::MAX; 10];
            fill_chunks(&mut out, workers, &|start, chunk: &mut [usize]| {
                for (offset, value) in chunk.iter_mut().enumerate() {
                    *value = start + offset;
                }
                Ok(())
            })
            .unwrap();
            assert_eq!(out, (0..10).collect::<Vec<_>>());
        }
        let mut empty: Vec<u8> = Vec::new();
        fill_chunks(&mut empty, 4, &|_, _: &mut [u8]| Ok(())).unwrap();
    }

    /// The optional passes equal the required stages when every worker
    /// starts, unit errors included (a refused start is tested end to end in
    /// tests/test_pre_validation_thread_refusal.py).
    #[test]
    fn optional_passes_equal_required_stages_when_every_worker_starts() {
        for workers in [1, 2, 3, 7, 16] {
            let range = |start, end| Ok((start, end));
            assert_eq!(
                try_map_ranges(10, workers, &range).unwrap(),
                Some(map_ranges(10, workers, &range).unwrap())
            );
            let mut out = vec![usize::MAX; 10];
            let fill = |start, chunk: &mut [usize]| {
                for (offset, value) in chunk.iter_mut().enumerate() {
                    *value = start + offset;
                }
                Ok(())
            };
            assert!(try_fill_chunks(&mut out, workers, &fill).unwrap());
            assert_eq!(out, (0..10).collect::<Vec<_>>());
            let failing = |start, _end| -> Result<(), EngineError> {
                Err(EngineError::Parameter(format!("unit {start}")))
            };
            let want = map_ranges(10, workers, &failing).unwrap_err().to_string();
            assert_eq!(
                try_map_ranges(10, workers, &failing)
                    .unwrap_err()
                    .to_string(),
                want
            );
        }
    }

    /// §11 item 13: every unit of every scheduler entry point computes in E0
    /// from a hostile caller that holds no guard, inline (one worker) and
    /// spawned (two or more), and the caller's state is intact afterwards.
    #[test]
    fn every_unit_computes_in_e0_from_an_unguarded_hostile_caller() {
        use crate::fpenv::{
            read_state,
            testing::{computes_in_e0, install, Restore, HOSTILE},
        };
        let _restore = Restore(read_state());
        for hostile in HOSTILE {
            let Some(hostile) = install(hostile) else {
                continue;
            };
            for workers in [1, 2, 3] {
                let parts = map_ranges(6, workers, &|start, end| {
                    Ok((start..end).map(|_| computes_in_e0()).collect::<Vec<_>>())
                })
                .unwrap();
                assert!(parts.concat().iter().all(|&e0| e0), "map_ranges {workers}");
                assert_eq!(read_state(), hostile, "map_ranges {workers}");

                let mut out = vec![false; 6];
                fill_chunks(&mut out, workers, &|_, chunk: &mut [bool]| {
                    chunk.iter_mut().for_each(|value| *value = computes_in_e0());
                    Ok(())
                })
                .unwrap();
                assert!(out.iter().all(|&e0| e0), "fill_chunks {workers}");
                assert_eq!(read_state(), hostile, "fill_chunks {workers}");

                let mut seen = Vec::new();
                stream_ranges(
                    6,
                    workers,
                    &|start, end| (start..end).map(|_| Ok(computes_in_e0())),
                    &mut |e0| {
                        seen.push(e0);
                        Ok(())
                    },
                )
                .unwrap();
                assert_eq!(seen.len(), 6);
                assert!(seen.iter().all(|&e0| e0), "stream_ranges {workers}");
                assert_eq!(read_state(), hostile, "stream_ranges {workers}");
            }
        }
    }
}
