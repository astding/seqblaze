//! Launch parameters. They never change results (docs/wiki/gpu/cuda-backend.md,
//! "Launch parameters"); testing builds override them process-wide.
#[cfg(feature = "testing")]
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::Mutex;

use seqblaze_core::validate::EngineError;

pub const DEFAULT_THREADS_PER_BLOCK: u32 = 256;
pub const DEFAULT_PIECE_BYTES: u64 = 4096;
pub const DEFAULT_PAIRS_PER_CHUNK: u64 = 1 << 22;
pub const DEFAULT_BYTES_PER_LAUNCH: u64 = 1 << 30;
/// Levenshtein (docs/wiki/gpu/levenshtein-cuda.md §8): largest W = ⌈m/64⌉ in
/// the thread tier, largest W on the device, largest serial depth of a device
/// pair, and word steps per chunk.
pub const DEFAULT_THREAD_WORDS: u64 = 4;
pub const DEFAULT_DEVICE_WORDS: u64 = 256;
pub const DEFAULT_PAIR_STEPS: u64 = 1 << 22;
pub const DEFAULT_STEPS_PER_LAUNCH: u64 = 1 << 34;
/// Compiled limits: 4 words per thread, 8 words per warp lane.
pub const MAX_THREAD_WORDS: u64 = 4;
pub const MAX_DEVICE_WORDS: u64 = 256;
/// Align (docs/wiki/gpu/align-cuda.md §8, P, QAG6): largest m in the thread
/// tier, largest rows per warp lane (strip height 32·lane_rows), largest serial
/// depth of a device pair in cell steps, and DP cells per chunk.
pub const DEFAULT_THREAD_ROWS: u64 = 32;
pub const DEFAULT_LANE_ROWS: u64 = 8;
pub const DEFAULT_PAIR_CELLS: u64 = 1 << 22;
pub const DEFAULT_CELLS_PER_LAUNCH: u64 = 1 << 34;
/// Compiled limits: 32 rows per thread; lane_rows takes the compiled warp
/// variants 1, 2, 4 and 8 only.
pub const MAX_THREAD_ROWS: u64 = 32;
pub const LANE_ROWS: [u64; 4] = [1, 2, 4, 8];
/// Duplex (docs/wiki/gpu/duplex-cuda.md §8, P, QKG3, QKG5): the group size of every
/// pair (0: by m) and the largest band of a shared-tier pair (0: every pair global).
pub const DEFAULT_GROUP_LANES: u64 = 0;
pub const GROUP_LANES: [u64; 5] = [0, 4, 8, 16, 32];
pub const DEFAULT_BAND_BYTES: u64 = 32_768;
pub const MAX_BAND_BYTES: u64 = 98_304;
/// Upper limits keep every grid and count in range; they are not tuning advice.
const MAX_PIECE_BYTES: u64 = i32::MAX as u64;
const MAX_PAIRS_PER_CHUNK: u64 = i32::MAX as u64;
const MAX_BYTES_PER_LAUNCH: u64 = 1 << 40;
const MAX_STEPS: u64 = 1 << 40;

/// A process-wide override; `None` means the default.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct LaunchOverride {
    pub threads_per_block: Option<u32>,
    pub piece_bytes: Option<u64>,
    pub pairs_per_chunk: Option<u64>,
    pub bytes_per_launch: Option<u64>,
    pub thread_words: Option<u64>,
    pub device_words: Option<u64>,
    pub pair_steps: Option<u64>,
    pub steps_per_launch: Option<u64>,
    pub thread_rows: Option<u64>,
    pub lane_rows: Option<u64>,
    pub pair_cells: Option<u64>,
    pub cells_per_launch: Option<u64>,
    pub group_lanes: Option<u64>,
    pub band_bytes: Option<u64>,
}

/// Resolved settings, snapshotted once per call.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Settings {
    pub threads_per_block: u32,
    pub piece_bytes: u64,
    pub pairs_per_chunk: u64,
    pub bytes_per_launch: u64,
    pub thread_words: u64,
    pub device_words: u64,
    pub pair_steps: u64,
    pub steps_per_launch: u64,
    pub thread_rows: u64,
    pub lane_rows: u64,
    pub pair_cells: u64,
    pub cells_per_launch: u64,
    pub group_lanes: u64,
    pub band_bytes: u64,
}

static OVERRIDE: Mutex<LaunchOverride> = Mutex::new(LaunchOverride {
    threads_per_block: None,
    piece_bytes: None,
    pairs_per_chunk: None,
    bytes_per_launch: None,
    thread_words: None,
    device_words: None,
    pair_steps: None,
    steps_per_launch: None,
    thread_rows: None,
    lane_rows: None,
    pair_cells: None,
    cells_per_launch: None,
    group_lanes: None,
    band_bytes: None,
});

fn check(name: &str, value: Option<u64>, low: u64, high: u64) -> Result<(), EngineError> {
    match value {
        Some(value) if !(low..=high).contains(&value) => Err(EngineError::Parameter(format!(
            "{name}={value} is out of range [{low}, {high}]; use None for the default"
        ))),
        _ => Ok(()),
    }
}

/// Replace the whole override and return the previous one. Each field is
/// checked: threads_per_block a multiple of 32 in [32, 1024]; piece_bytes in
/// [1, 2^31 - 1]; pairs_per_chunk in [1, 2^31 - 1]; bytes_per_launch in [1, 2^40];
/// thread_words in [0, 4]; device_words in [0, 256]; pair_steps in [0, 2^40];
/// steps_per_launch in [1, 2^40]; thread_rows in [0, 32]; lane_rows in
/// {1, 2, 4, 8}; pair_cells in [0, 2^40]; cells_per_launch in [1, 2^40];
/// group_lanes in {0, 4, 8, 16, 32}; band_bytes in [0, 98,304].
pub fn set_launch_override(new: LaunchOverride) -> Result<LaunchOverride, EngineError> {
    if let Some(threads) = new.threads_per_block {
        if !(32..=1024).contains(&threads) || threads % 32 != 0 {
            return Err(EngineError::Parameter(format!(
                "threads_per_block={threads}; use a multiple of 32 in [32, 1024] or None"
            )));
        }
    }
    check("piece_bytes", new.piece_bytes, 1, MAX_PIECE_BYTES)?;
    check(
        "pairs_per_chunk",
        new.pairs_per_chunk,
        1,
        MAX_PAIRS_PER_CHUNK,
    )?;
    check(
        "bytes_per_launch",
        new.bytes_per_launch,
        1,
        MAX_BYTES_PER_LAUNCH,
    )?;
    check("thread_words", new.thread_words, 0, MAX_THREAD_WORDS)?;
    check("device_words", new.device_words, 0, MAX_DEVICE_WORDS)?;
    check("pair_steps", new.pair_steps, 0, MAX_STEPS)?;
    check("steps_per_launch", new.steps_per_launch, 1, MAX_STEPS)?;
    check("thread_rows", new.thread_rows, 0, MAX_THREAD_ROWS)?;
    if let Some(rows) = new.lane_rows {
        if !LANE_ROWS.contains(&rows) {
            return Err(EngineError::Parameter(format!(
                "lane_rows={rows}; use 1, 2, 4 or 8 (the compiled warp variants) or None"
            )));
        }
    }
    check("pair_cells", new.pair_cells, 0, MAX_STEPS)?;
    check("cells_per_launch", new.cells_per_launch, 1, MAX_STEPS)?;
    if let Some(lanes) = new.group_lanes {
        if !GROUP_LANES.contains(&lanes) {
            return Err(EngineError::Parameter(format!(
                "group_lanes={lanes}; use 0 (by length), 4, 8, 16 or 32 or None"
            )));
        }
    }
    check("band_bytes", new.band_bytes, 0, MAX_BAND_BYTES)?;
    let mut current = OVERRIDE.lock().unwrap_or_else(|poison| poison.into_inner());
    Ok(std::mem::replace(&mut current, new))
}

pub fn launch_override() -> LaunchOverride {
    *OVERRIDE.lock().unwrap_or_else(|poison| poison.into_inner())
}

/// Testing builds only (align-cuda.md §13, cuda-backend.md): the byte every
/// new device allocation is filled with before use; `POISON_OFF` (the
/// default) leaves allocations as the driver returns them.
#[cfg(feature = "testing")]
const POISON_OFF: u32 = u32::MAX;
#[cfg(feature = "testing")]
static POISON: AtomicU32 = AtomicU32::new(POISON_OFF);
/// Bytes filled with the poison byte so far (process-wide, never reset).
#[cfg(feature = "testing")]
static POISONED_BYTES: AtomicU64 = AtomicU64::new(0);

/// Device-observed resident K1/K3 fast/continuation pairs, fills and bytes.
/// Allocation fills never contribute to these process-wide cumulative values.
#[cfg(feature = "testing")]
static RESIDENT_POISON_STATS: [AtomicU64; 12] = [const { AtomicU64::new(0) }; 12];

#[cfg(feature = "testing")]
pub fn cuda_resident_poison_stats() -> [[u64; 3]; 4] {
    std::array::from_fn(|route| {
        std::array::from_fn(|metric| {
            RESIDENT_POISON_STATS[route * 3 + metric].load(Ordering::SeqCst)
        })
    })
}

#[cfg(all(feature = "testing", feature = "cuda"))]
pub(crate) fn count_resident_poison(family: usize, counts: [u64; 6]) {
    for (counter, value) in RESIDENT_POISON_STATS[family * 6..][..6].iter().zip(counts) {
        counter.fetch_add(value, Ordering::SeqCst);
    }
}

/// Testing builds: T2 output refills only, excluding allocations and DP scratch.
#[cfg(feature = "testing")]
#[derive(Clone, Copy, Debug, Default)]
pub struct UnaryRefillStats {
    pub chunks: u64,
    pub rows: u64,
    pub field_bytes: u64,
    pub passed_bytes: u64,
}

#[cfg(feature = "testing")]
static UNARY_REFILL_STATS: Mutex<UnaryRefillStats> = Mutex::new(UnaryRefillStats {
    chunks: 0,
    rows: 0,
    field_bytes: 0,
    passed_bytes: 0,
});

/// Testing builds: cumulative T2 output-refill counts (process-wide, never reset).
#[cfg(feature = "testing")]
pub fn cuda_unary_refill_stats() -> UnaryRefillStats {
    *UNARY_REFILL_STATS
        .lock()
        .unwrap_or_else(|poison| poison.into_inner())
}

/// Record bytes only after their fill was queued successfully; record rows once
/// all output fills for the chunk have been queued successfully.
#[cfg(feature = "testing")]
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
pub(crate) fn count_unary_refill(rows: u64, field_bytes: u64, passed_bytes: u64) {
    let mut stats = UNARY_REFILL_STATS
        .lock()
        .unwrap_or_else(|poison| poison.into_inner());
    stats.chunks += u64::from(rows != 0);
    stats.rows += rows;
    stats.field_bytes += field_bytes;
    stats.passed_bytes += passed_bytes;
}

/// The resident K1/K3 fast-path depth cap: candidate evaluations of one lane, 2^21
/// (duplex-cuda.md §4, duplex-pf-cuda.md §3; nndb2024-dna.md §5.4.7).
pub const RESIDENT_DEPTH_CAP: u64 = 1 << 21;

/// Testing builds: the process-wide override of [`RESIDENT_DEPTH_CAP`].
#[cfg(feature = "testing")]
static RESIDENT_DEPTH: AtomicU64 = AtomicU64::new(RESIDENT_DEPTH_CAP);

/// Testing builds (nndb2024-dna.md §5.4.7): replace the resident K1/K3 depth cap with
/// `cells` (None restores 2^21) and return the previous effective cap. `cells` must lie
/// in [1, 2^21]; otherwise ParameterError and the stored value is unchanged. Launch-only:
/// pairs above the cap run through the continuation kernels; no output byte changes.
#[cfg(feature = "testing")]
pub fn set_cuda_resident_depth_cap(cells: Option<u64>) -> Result<u64, EngineError> {
    let value = cells.unwrap_or(RESIDENT_DEPTH_CAP);
    if !(1..=RESIDENT_DEPTH_CAP).contains(&value) {
        return Err(EngineError::Parameter(format!(
            "cells={value}; use None or an int in [1, {RESIDENT_DEPTH_CAP}]"
        )));
    }
    Ok(RESIDENT_DEPTH.swap(value, Ordering::SeqCst))
}

/// The resident depth cap a stage schedules with: [`RESIDENT_DEPTH_CAP`], or in testing
/// builds the value of [`set_cuda_resident_depth_cap`].
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
pub(crate) fn resident_depth_cap() -> u64 {
    #[cfg(feature = "testing")]
    {
        RESIDENT_DEPTH.load(Ordering::SeqCst)
    }
    #[cfg(not(feature = "testing"))]
    {
        RESIDENT_DEPTH_CAP
    }
}

/// Largest S_test of [`set_cuda_resident_launch`] (duplex-cuda.md §15.4 step 1).
pub const RESIDENT_SLOTS_MAX: u64 = 1 << 16;

/// Testing builds: the process-wide S_test and U_test of the resident K1/K3 fast paths
/// (u64::MAX: unset).
#[cfg(feature = "testing")]
static RESIDENT_LAUNCH: [AtomicU64; 2] = [AtomicU64::new(u64::MAX), AtomicU64::new(u64::MAX)];

/// Testing builds (duplex-cuda.md §15.4 step 1): set S_test (`slots`, None or [1, 2^16])
/// and U_test (`unit`, None or [1, 2^21]) of the resident K1 and K3 fast paths and return
/// the previous pair. A value outside its range raises ParameterError and changes
/// nothing. Launch-only: the values can only lower the slot count and the depth per
/// launch, never a result byte.
#[cfg(feature = "testing")]
#[allow(clippy::type_complexity)]
pub fn set_cuda_resident_launch(
    slots: Option<u64>,
    unit: Option<u64>,
) -> Result<(Option<u64>, Option<u64>), EngineError> {
    if let Some(value) = slots.filter(|v| !(1..=RESIDENT_SLOTS_MAX).contains(v)) {
        return Err(EngineError::Parameter(format!(
            "slots={value}; use None or an int in [1, {RESIDENT_SLOTS_MAX}]"
        )));
    }
    if let Some(value) = unit.filter(|v| !(1..=RESIDENT_DEPTH_CAP).contains(v)) {
        return Err(EngineError::Parameter(format!(
            "unit={value}; use None or an int in [1, {RESIDENT_DEPTH_CAP}]"
        )));
    }
    let old = |v: u64| (v != u64::MAX).then_some(v);
    let previous_slots = RESIDENT_LAUNCH[0].swap(slots.unwrap_or(u64::MAX), Ordering::SeqCst);
    let previous_unit = RESIDENT_LAUNCH[1].swap(unit.unwrap_or(u64::MAX), Ordering::SeqCst);
    Ok((old(previous_slots), old(previous_unit)))
}

/// (S_test, U_test) of the resident K1/K3 fast paths; u64::MAX (+∞) unless a testing
/// build set them through [`set_cuda_resident_launch`].
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
pub(crate) fn resident_launch() -> (u64, u64) {
    #[cfg(feature = "testing")]
    {
        (
            RESIDENT_LAUNCH[0].load(Ordering::SeqCst),
            RESIDENT_LAUNCH[1].load(Ordering::SeqCst),
        )
    }
    #[cfg(not(feature = "testing"))]
    {
        (u64::MAX, u64::MAX)
    }
}

/// Testing builds: the resident K3 terms captured while [`set_cuda_pf_capture`] is on,
/// as (first work ID of the chunk, the chunk's interleaved (M, e) bytes).
#[cfg(feature = "testing")]
#[allow(clippy::type_complexity)]
static PF_CAPTURE: Mutex<Option<Vec<(u64, Vec<u8>)>>> = Mutex::new(None);

/// Testing builds (duplex-pf-cuda.md §19.4 step 2, `duplex_pf_zhat(runner="resident")`):
/// turn the capture of the resident K3 stage's Ẑ terms on (clearing it) or off, and
/// return what was captured since it was turned on.
#[cfg(feature = "testing")]
pub fn set_cuda_pf_capture(on: bool) -> Vec<(u64, Vec<u8>)> {
    let mut capture = PF_CAPTURE
        .lock()
        .unwrap_or_else(|poison| poison.into_inner());
    let previous = capture.take().unwrap_or_default();
    if on {
        *capture = Some(Vec::new());
    }
    previous
}

/// Testing builds: record one chunk of the resident K3 stage's terms while the capture is on
/// (`bytes` is read only then).
#[cfg(feature = "testing")]
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
pub(crate) fn pf_capture_push(
    k0: u64,
    bytes: impl FnOnce() -> Result<Vec<u8>, EngineError>,
) -> Result<(), EngineError> {
    let mut capture = PF_CAPTURE
        .lock()
        .unwrap_or_else(|poison| poison.into_inner());
    if let Some(chunks) = capture.as_mut() {
        chunks.push((k0, bytes()?));
    }
    Ok(())
}

/// R_b of duplex-mfe-bound.md §8 *Sweeps* (QB14): the device bytes a bounded K1
/// RowStats call may keep for the per-item outcome words of its second sweep, 2^28.
pub const DEFAULT_BOUND_RETAIN_BYTES: u64 = 1 << 28;
/// The largest testing value of R_b (§10.2: an int in [0, 2^40]).
pub const MAX_BOUND_RETAIN_BYTES: u64 = 1 << 40;

/// Testing builds: the process-wide override of R_b (`u64::MAX`: none). It is a keyword
/// of `cuda_launch` but not a key of [`LaunchOverride`] (QB18): each `cuda_launch` call
/// replaces it with its argument, as it replaces the whole launch override.
#[cfg(feature = "testing")]
static BOUND_RETAIN: AtomicU64 = AtomicU64::new(u64::MAX);

/// Testing builds: replace the R_b override (None: the default) and return the previous
/// one. `bytes` must lie in [0, 2^40]; otherwise ParameterError and nothing changes.
#[cfg(feature = "testing")]
pub fn set_bound_retain_bytes(bytes: Option<u64>) -> Result<Option<u64>, EngineError> {
    if let Some(value) = bytes {
        if value > MAX_BOUND_RETAIN_BYTES {
            return Err(EngineError::Parameter(format!(
                "bound_retain_bytes={value} is out of range [0, {MAX_BOUND_RETAIN_BYTES}]; use \
                 None for the default"
            )));
        }
    }
    let previous = BOUND_RETAIN.swap(bytes.unwrap_or(u64::MAX), Ordering::SeqCst);
    Ok((previous != u64::MAX).then_some(previous))
}

/// R_b as a call reads it: [`DEFAULT_BOUND_RETAIN_BYTES`], or the testing override.
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
pub(crate) fn bound_retain_bytes() -> u64 {
    #[cfg(feature = "testing")]
    {
        match BOUND_RETAIN.load(Ordering::SeqCst) {
            u64::MAX => DEFAULT_BOUND_RETAIN_BYTES,
            value => value,
        }
    }
    #[cfg(not(feature = "testing"))]
    {
        DEFAULT_BOUND_RETAIN_BYTES
    }
}

/// Testing builds: set the process-wide poison byte (None: off) and return the
/// previous one. Only allocations made after the call are filled.
#[cfg(feature = "testing")]
pub fn set_cuda_poison(byte: Option<u8>) -> Option<u8> {
    let previous = POISON.swap(byte.map_or(POISON_OFF, u32::from), Ordering::SeqCst);
    u8::try_from(previous).ok()
}

/// Testing builds: bytes of device memory filled with the poison byte so far,
/// so a test can confirm the fill happened.
#[cfg(feature = "testing")]
pub fn cuda_poisoned_bytes() -> u64 {
    POISONED_BYTES.load(Ordering::SeqCst)
}

#[cfg(feature = "testing")]
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
pub(crate) fn poison() -> Option<u8> {
    u8::try_from(POISON.load(Ordering::SeqCst)).ok()
}

#[cfg(feature = "testing")]
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
pub(crate) fn count_poisoned(bytes: u64) {
    POISONED_BYTES.fetch_add(bytes, Ordering::SeqCst);
}

#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
pub(crate) fn settings() -> Settings {
    let current = launch_override();
    Settings {
        threads_per_block: current
            .threads_per_block
            .unwrap_or(DEFAULT_THREADS_PER_BLOCK),
        piece_bytes: current.piece_bytes.unwrap_or(DEFAULT_PIECE_BYTES),
        pairs_per_chunk: current.pairs_per_chunk.unwrap_or(DEFAULT_PAIRS_PER_CHUNK),
        bytes_per_launch: current.bytes_per_launch.unwrap_or(DEFAULT_BYTES_PER_LAUNCH),
        thread_words: current.thread_words.unwrap_or(DEFAULT_THREAD_WORDS),
        device_words: current.device_words.unwrap_or(DEFAULT_DEVICE_WORDS),
        pair_steps: current.pair_steps.unwrap_or(DEFAULT_PAIR_STEPS),
        steps_per_launch: current.steps_per_launch.unwrap_or(DEFAULT_STEPS_PER_LAUNCH),
        thread_rows: current.thread_rows.unwrap_or(DEFAULT_THREAD_ROWS),
        lane_rows: current.lane_rows.unwrap_or(DEFAULT_LANE_ROWS),
        pair_cells: current.pair_cells.unwrap_or(DEFAULT_PAIR_CELLS),
        cells_per_launch: current.cells_per_launch.unwrap_or(DEFAULT_CELLS_PER_LAUNCH),
        group_lanes: current.group_lanes.unwrap_or(DEFAULT_GROUP_LANES),
        band_bytes: current.band_bytes.unwrap_or(DEFAULT_BAND_BYTES),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn override_round_trip_and_ranges() {
        let custom = LaunchOverride {
            threads_per_block: Some(32),
            piece_bytes: Some(3),
            pairs_per_chunk: Some(7),
            thread_words: Some(0),
            device_words: Some(0),
            pair_steps: Some(0),
            steps_per_launch: Some(1),
            thread_rows: Some(0),
            lane_rows: Some(2),
            pair_cells: Some(0),
            cells_per_launch: Some(1),
            group_lanes: Some(4),
            band_bytes: Some(0),
            ..LaunchOverride::default()
        };
        let previous = set_launch_override(custom).unwrap();
        assert_eq!(settings().piece_bytes, 3);
        assert_eq!(settings().bytes_per_launch, DEFAULT_BYTES_PER_LAUNCH);
        assert_eq!(
            (
                settings().thread_words,
                settings().device_words,
                settings().pair_steps,
                settings().steps_per_launch
            ),
            (0, 0, 0, 1)
        );
        assert_eq!(
            (
                settings().thread_rows,
                settings().lane_rows,
                settings().pair_cells,
                settings().cells_per_launch
            ),
            (0, 2, 0, 1)
        );
        assert_eq!((settings().group_lanes, settings().band_bytes), (4, 0));
        assert_eq!(set_launch_override(previous).unwrap(), custom);
        for bad in [
            LaunchOverride {
                threads_per_block: Some(48),
                ..LaunchOverride::default()
            },
            LaunchOverride {
                threads_per_block: Some(2048),
                ..LaunchOverride::default()
            },
            LaunchOverride {
                piece_bytes: Some(0),
                ..LaunchOverride::default()
            },
            LaunchOverride {
                pairs_per_chunk: Some(0),
                ..LaunchOverride::default()
            },
            LaunchOverride {
                thread_words: Some(5),
                ..LaunchOverride::default()
            },
            LaunchOverride {
                device_words: Some(257),
                ..LaunchOverride::default()
            },
            LaunchOverride {
                pair_steps: Some((1 << 40) + 1),
                ..LaunchOverride::default()
            },
            LaunchOverride {
                steps_per_launch: Some(0),
                ..LaunchOverride::default()
            },
            LaunchOverride {
                thread_rows: Some(33),
                ..LaunchOverride::default()
            },
            LaunchOverride {
                lane_rows: Some(3),
                ..LaunchOverride::default()
            },
            LaunchOverride {
                lane_rows: Some(0),
                ..LaunchOverride::default()
            },
            LaunchOverride {
                pair_cells: Some((1 << 40) + 1),
                ..LaunchOverride::default()
            },
            LaunchOverride {
                cells_per_launch: Some(0),
                ..LaunchOverride::default()
            },
            LaunchOverride {
                group_lanes: Some(2),
                ..LaunchOverride::default()
            },
            LaunchOverride {
                group_lanes: Some(64),
                ..LaunchOverride::default()
            },
            LaunchOverride {
                band_bytes: Some(98_305),
                ..LaunchOverride::default()
            },
        ] {
            assert!(set_launch_override(bad).is_err());
        }
        assert_eq!(launch_override(), previous);
    }
}
