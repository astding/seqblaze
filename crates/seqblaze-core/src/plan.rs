//! Typed CPU strategy resolution; see engine/planner-scheduler.md.
//! Work scheduling/certificate/staged-plan extensions retain their I2/I6/I7 ports.
use crate::work::PairKind;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Strategy {
    Bounded,
    Full,
}
impl Strategy {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Bounded => "bounded",
            Self::Full => "full",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BoundAvailability {
    Available,
    Unavailable,
    /// The kernel has a bound, but the selected backend does not use it (CUDA G1).
    NotOnBackend,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutputDemand {
    Selective,
    Each { has_cutoff: bool },
    Dense,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FallbackReason {
    NoBound,
    DeviceNoBound,
    DenseOutput,
    EachWithoutCutoff,
}
impl FallbackReason {
    /// Draft report spellings; this is not an adopted I7 metadata codec.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::NoBound => "no_bound",
            Self::DeviceNoBound => "device_no_bound",
            Self::DenseOutput => "dense_output_requires_full",
            Self::EachWithoutCutoff => "each_without_cutoff",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExecutionPlan {
    pub requested: Strategy,
    pub effective: Strategy,
    pub fallback_reason: Option<FallbackReason>,
}
impl ExecutionPlan {
    /// Resolve without examining inputs, evaluating bounds or allocating buffers.
    pub fn resolve(requested: Strategy, bound: BoundAvailability, demand: OutputDemand) -> Self {
        let fallback_reason = if requested == Strategy::Full {
            None
        } else {
            match (bound, demand) {
                (BoundAvailability::Unavailable, _) => Some(FallbackReason::NoBound),
                (BoundAvailability::NotOnBackend, _) => Some(FallbackReason::DeviceNoBound),
                (_, OutputDemand::Dense) => Some(FallbackReason::DenseOutput),
                (_, OutputDemand::Each { has_cutoff: false }) => {
                    Some(FallbackReason::EachWithoutCutoff)
                }
                _ => None,
            }
        };
        Self {
            requested,
            effective: if fallback_reason.is_some() {
                Strategy::Full
            } else {
                requested
            },
            fallback_reason,
        }
    }
}

/// Whether bounded execution evaluates the kernel's per-pair lower bound before
/// its early-exit score. Either choice yields identical outputs and counts: a
/// pair the bound rejects (bound > cutoff) is also rejected by the early-exit
/// score, and both count it as pruned. Only the cost differs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BoundPrefilter {
    PerPair,
    EarlyExitOnly(PrefilterSkip),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PrefilterSkip {
    /// Zip and one-to-many read each row of a table about once, and an explicit
    /// index list carries no reuse guarantee, so per-row bound preparation would
    /// cost about as much as the early-exit scans it could save.
    UnamortizedPreparation,
}
impl PrefilterSkip {
    /// Draft spelling, recorded alongside the fallback reasons above.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::UnamortizedPreparation => "unamortized_bound_preparation",
        }
    }
}

impl BoundPrefilter {
    /// Pure function of the pair shape: within and cross reuse every row across
    /// many pairs, which amortizes the per-row preparation of the bound.
    pub fn for_work(kind: &PairKind) -> Self {
        match kind {
            PairKind::Within { .. } | PairKind::WithinOriented { .. } | PairKind::Cross => {
                Self::PerPair
            }
            PairKind::Zip | PairKind::Index { .. } | PairKind::OneToMany => {
                Self::EarlyExitOnly(PrefilterSkip::UnamortizedPreparation)
            }
        }
    }
    pub fn per_pair(self) -> bool {
        self == Self::PerPair
    }
}

/// Testing faults of a kernel's outcome bound (duplex-mfe-bound.md §10.2): process-wide
/// switches in two slots that compose when nested. The certificate slot holds
/// `GammaPlusOne` or `LbPlusOne` (read by bound builders and evaluators), the decision
/// slot `ProbeUse` or `ProbeSkip` (read by the per-pass weak-bound decision). Without the
/// `testing` feature both are always [`BoundFault::None`] and cannot be set, so no release
/// build carries a fault.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BoundFault {
    None,
    /// Every region constant γ + 1 before the certificate check (the check must fail).
    GammaPlusOne,
    /// Every finite bound value + 1 after the recurrence (bypasses the check).
    LbPlusOne,
    /// The weak-bound decision of every pass forced to "use".
    ProbeUse,
    /// The weak-bound decision of every pass forced to "skip".
    ProbeSkip,
    /// K3's C2 bound: every nonzero Ẑ_b times 2^−64 after the recurrence (bypasses the
    /// certificate; duplex-pf-bound.md §10.2).
    ZbScaleDown,
}

/// The slot of a [`BoundFault`]: certificate/LB faults and decision faults are independent.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FaultSlot {
    Certificate,
    Decision,
}

impl BoundFault {
    /// The slot this fault occupies (`None` has no slot of its own).
    pub fn slot(self) -> Option<FaultSlot> {
        match self {
            Self::None => None,
            Self::GammaPlusOne | Self::LbPlusOne | Self::ZbScaleDown => {
                Some(FaultSlot::Certificate)
            }
            Self::ProbeUse | Self::ProbeSkip => Some(FaultSlot::Decision),
        }
    }
    #[cfg(feature = "testing")]
    fn code(self) -> u8 {
        match self {
            Self::None => 0,
            Self::GammaPlusOne => 1,
            Self::LbPlusOne => 2,
            Self::ProbeUse => 3,
            Self::ProbeSkip => 4,
            Self::ZbScaleDown => 5,
        }
    }
    #[cfg(feature = "testing")]
    fn from_code(code: u8) -> Self {
        match code {
            1 => Self::GammaPlusOne,
            2 => Self::LbPlusOne,
            3 => Self::ProbeUse,
            4 => Self::ProbeSkip,
            5 => Self::ZbScaleDown,
            _ => Self::None,
        }
    }
}

#[cfg(feature = "testing")]
static CERTIFICATE_FAULT: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);
#[cfg(feature = "testing")]
static DECISION_FAULT: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);

/// The active certificate/LB fault: `None`, `GammaPlusOne` or `LbPlusOne` (always `None`
/// without the `testing` feature).
#[inline]
pub fn bound_fault() -> BoundFault {
    #[cfg(feature = "testing")]
    {
        BoundFault::from_code(CERTIFICATE_FAULT.load(std::sync::atomic::Ordering::SeqCst))
    }
    #[cfg(not(feature = "testing"))]
    {
        BoundFault::None
    }
}

/// The active decision fault: `None`, `ProbeUse` or `ProbeSkip` (always `None` without
/// the `testing` feature).
#[inline]
pub fn decision_fault() -> BoundFault {
    #[cfg(feature = "testing")]
    {
        BoundFault::from_code(DECISION_FAULT.load(std::sync::atomic::Ordering::SeqCst))
    }
    #[cfg(not(feature = "testing"))]
    {
        BoundFault::None
    }
}

/// Whether any bound fault is active in either slot.
#[inline]
pub fn any_bound_fault() -> bool {
    bound_fault() != BoundFault::None || decision_fault() != BoundFault::None
}

/// Testing builds: set `slot` to `fault` (which must be `None` or of that slot),
/// returning the slot's previous fault. The other slot is untouched, so nested faults of
/// different slots compose.
#[cfg(feature = "testing")]
pub fn set_bound_fault(slot: FaultSlot, fault: BoundFault) -> BoundFault {
    debug_assert!(fault.slot().is_none_or(|s| s == slot));
    let cell = match slot {
        FaultSlot::Certificate => &CERTIFICATE_FAULT,
        FaultSlot::Decision => &DECISION_FAULT,
    };
    BoundFault::from_code(cell.swap(fault.code(), std::sync::atomic::Ordering::SeqCst))
}

#[cfg(feature = "testing")]
static BOUND_EVALS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Count one outcome-bound evaluation (testing builds; a no-op otherwise).
#[inline]
pub fn count_bound_eval() {
    #[cfg(feature = "testing")]
    BOUND_EVALS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
}

/// Count `n` outcome-bound evaluations at once (testing builds; a no-op otherwise): the
/// CUDA backend adds the item counts of its probe and bound launches from the host
/// (duplex-mfe-bound.md §10.2, revision 3).
#[inline]
pub fn count_bound_evals(n: u64) {
    #[cfg(feature = "testing")]
    BOUND_EVALS.fetch_add(n, std::sync::atomic::Ordering::Relaxed);
    #[cfg(not(feature = "testing"))]
    let _ = n;
}

/// Testing builds: the outcome-bound evaluations counted since the last reset
/// (duplex-mfe-bound.md §10.2); `reset` returns the count and sets it to 0.
#[cfg(feature = "testing")]
pub fn bound_evals(reset: bool) -> u64 {
    if reset {
        BOUND_EVALS.swap(0, std::sync::atomic::Ordering::SeqCst)
    } else {
        BOUND_EVALS.load(std::sync::atomic::Ordering::SeqCst)
    }
}

#[cfg(feature = "testing")]
static CERTIFICATE_BUILDS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Count one started certificate build of an outcome bound (testing builds; a no-op
/// otherwise; duplex-mfe-bound.md §10.2, revision 3).
#[inline]
pub fn count_certificate_build() {
    #[cfg(feature = "testing")]
    CERTIFICATE_BUILDS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
}

/// Testing builds: the certificate builds started since the last reset (cache misses
/// and uncached builds, refused ones included; cache hits do not count); `reset`
/// returns the count and sets it to 0.
#[cfg(feature = "testing")]
pub fn certificate_builds(reset: bool) -> u64 {
    if reset {
        CERTIFICATE_BUILDS.swap(0, std::sync::atomic::Ordering::SeqCst)
    } else {
        CERTIFICATE_BUILDS.load(std::sync::atomic::Ordering::SeqCst)
    }
}

/// Testing builds: the process-wide override of an outcome bound's DEVICE cost tables
/// (duplex-mfe-bound.md §10.2 `duplex_bound_device_costs`, revision 3), encoded as
/// `1 << 63 | q̂ << 8 | ĉ`, or 0 for none. The CPU's table never reads it.
#[cfg(feature = "testing")]
static DEVICE_COSTS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// The largest ĉ and q̂ of a device cost table entry (§6.6.2 *Integer arithmetic*).
pub const DEVICE_C_HAT_MAX: u32 = 255;
pub const DEVICE_Q_HAT_MAX: u32 = (1 << 20) - 1;

/// The active device cost override (ĉ, q̂): every entry of the device tables reads as
/// these constants. Always `None` without the `testing` feature.
#[inline]
pub fn device_costs_override() -> Option<(u32, u32)> {
    #[cfg(feature = "testing")]
    {
        decode_device_costs(DEVICE_COSTS.load(std::sync::atomic::Ordering::SeqCst))
    }
    #[cfg(not(feature = "testing"))]
    {
        None
    }
}

/// Testing builds: set (or clear, with `None`) the device cost override, returning the
/// previous one. ĉ must lie in [0, 255] and q̂ in [0, 2^20 − 1].
#[cfg(feature = "testing")]
pub fn set_device_costs_override(
    costs: Option<(u32, u32)>,
) -> Result<Option<(u32, u32)>, crate::validate::EngineError> {
    let code = match costs {
        None => 0,
        Some((c, q)) if c <= DEVICE_C_HAT_MAX && q <= DEVICE_Q_HAT_MAX => {
            1u64 << 63 | u64::from(q) << 8 | u64::from(c)
        }
        Some((c, q)) => {
            return Err(crate::validate::EngineError::Parameter(format!(
                "duplex_bound_device_costs(c_hat={c}, q_hat={q}); use c_hat in [0, 255] and \
                 q_hat in [0, 2**20 - 1]"
            )))
        }
    };
    Ok(decode_device_costs(
        DEVICE_COSTS.swap(code, std::sync::atomic::Ordering::SeqCst),
    ))
}

#[cfg(feature = "testing")]
fn decode_device_costs(code: u64) -> Option<(u32, u32)> {
    (code >> 63 == 1).then_some(((code & 0xff) as u32, ((code >> 8) & 0xf_ffff) as u32))
}

/// The weak-bound rule of an outcome bound (duplex-mfe-bound.md §6.6, revision 2, QB3):
/// the largest probe P, the work items per probe item M and the smallest probe P_min.
pub const PROBE_P: usize = 64;
pub const PROBE_M: usize = 32;
pub const PROBE_P_MIN: usize = 16;
/// The unit of the cost-table entries ĉ (1/64).
pub const COST_UNIT: u128 = 64;
/// The device-only probe-cost gate (§6.6.2 step 5, QB16): the probe may cost at most
/// 1/32 of the pass's `full` time.
pub const PROBE_COST_FACTOR: u128 = 32;

/// The probe IDs of a pass of `n` work items (§6.6 step 1): k_t = ⌊t·n/p₀⌋ for
/// t < p₀ = min(P, ⌊n/M⌋), or none when p₀ < P_min (n < 512). Distinct and increasing.
pub fn probe_ids(n: usize) -> Vec<usize> {
    let p0 = PROBE_P.min(n / PROBE_M);
    if p0 < PROBE_P_MIN {
        return Vec::new();
    }
    // t·n < 64·2^64 fits u128, and the quotient is below n.
    (0..p0)
        .map(|t| (t as u128 * n as u128 / p0 as u128) as usize)
        .collect()
}

/// Why a pass does or does not use the outcome bound (§6.6; recorded in the typed plan
/// only, never in the manifest).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecisionReason {
    /// A testing fault forced the decision.
    Forced,
    /// p₀ < P_min: fewer than 512 work items.
    Small,
    /// Fewer than P_min probe items remain after the TopK replay's skipped IDs.
    ProbeSmall,
    /// E = 0 or C > 64·E: the bound cannot pay; no bound evaluated.
    Gate,
    /// Device only (§6.6.2 step 5): 32·p·max_S(q̂·e) > n·E, the probe would cost more
    /// than 1/32 of the pass; no bound evaluated.
    ProbeCost,
    /// The probe decided: used iff 64·R ≥ C.
    Probe,
}

/// One pass's weak-bound decision.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BoundDecision {
    pub used: bool,
    pub reason: DecisionReason,
}

/// The weak-bound decision of one pass of `n` work items (§6.6 on the CPU, §6.6.2 on
/// CUDA), a pure function of the inputs and the backend's cost tables, so every thread
/// count and run decides alike; no timing is read.
/// - `cost(k)`: for probe ID `k`, `None` when the pass skips the item (TopK replay),
///   else (e_k, ĉ_k, q̂_k) of its strand lengths: the kernel's `bound_cost` with q̂ = 0
///   on the CPU (which disables the probe-cost gate), the device tables on CUDA.
/// - `resolves(ids, out)`: the probe as one batch. For the remaining probe items `ids`
///   (increasing work IDs), set `out[t]` to whether item `ids[t]`'s bound outcome against
///   the pass threshold is pruned or certified (`out` starts all false and has one slot
///   per ID). It is called at most once, only after both gates pass. The caller may
///   evaluate the items in any order and on any number of workers (the CPU uses the
///   call's worker pool, §6.6): each flag is a pure function of its item, and R is an
///   exact integer sum over the flags, so the decision does not depend on how they were
///   computed.
pub fn decide_outcome_bound<E>(
    n: usize,
    mut cost: impl FnMut(usize) -> Result<Option<(u64, u32, u32)>, E>,
    resolves: impl FnOnce(&[usize], &mut [bool]) -> Result<(), E>,
) -> Result<BoundDecision, E> {
    let forced = |used| {
        Ok(BoundDecision {
            used,
            reason: DecisionReason::Forced,
        })
    };
    match decision_fault() {
        BoundFault::ProbeUse => return forced(true),
        BoundFault::ProbeSkip => return forced(false),
        _ => {}
    }
    let not_used = |reason| {
        Ok(BoundDecision {
            used: false,
            reason,
        })
    };
    let ids = probe_ids(n);
    if ids.is_empty() {
        return not_used(DecisionReason::Small);
    }
    let mut set = Vec::with_capacity(ids.len());
    for k in ids {
        if let Some((e, c, q)) = cost(k)? {
            set.push((k, u128::from(e), u128::from(c), u128::from(q)));
        }
    }
    if set.len() < PROBE_P_MIN {
        return not_used(DecisionReason::ProbeSmall);
    }
    let big_e: u128 = set.iter().map(|&(_, e, _, _)| e).sum();
    let big_c: u128 = set.iter().map(|&(_, e, c, _)| c * e).sum();
    if big_e == 0 || big_c > COST_UNIT * big_e {
        return not_used(DecisionReason::Gate);
    }
    // §6.6.2 step 5 (q̂ = 0 everywhere on the CPU disables it): exact in u128, since
    // q̂·e < 2^51, p ≤ 64 and n·E < 2^128.
    let big_q: u128 = set.iter().map(|&(_, e, _, q)| q * e).max().unwrap_or(0);
    if PROBE_COST_FACTOR * set.len() as u128 * big_q > n as u128 * big_e {
        return not_used(DecisionReason::ProbeCost);
    }
    let ids: Vec<usize> = set.iter().map(|&(k, _, _, _)| k).collect();
    let mut resolved = vec![false; ids.len()];
    resolves(&ids, &mut resolved)?;
    // Exact: e_k < 2^32 and p ≤ 64.
    let big_r: u128 = set
        .iter()
        .zip(&resolved)
        .filter(|&(_, &r)| r)
        .map(|(&(_, e, _, _), _)| e)
        .sum();
    Ok(BoundDecision {
        used: COST_UNIT * big_r >= big_c,
        reason: DecisionReason::Probe,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::work::Orientation;

    #[test]
    fn per_pair_bound_only_for_row_reusing_shapes() {
        let skip = BoundPrefilter::EarlyExitOnly(PrefilterSkip::UnamortizedPreparation);
        for kind in [PairKind::Zip, PairKind::OneToMany] {
            assert_eq!(BoundPrefilter::for_work(&kind), skip);
        }
        for kind in [
            PairKind::Cross,
            PairKind::Within { self_pairs: false },
            PairKind::WithinOriented {
                self_pairs: true,
                orientation: Orientation::Both,
            },
        ] {
            assert!(BoundPrefilter::for_work(&kind).per_pair());
        }
        assert_eq!(
            skip,
            BoundPrefilter::EarlyExitOnly(PrefilterSkip::UnamortizedPreparation)
        );
        assert_eq!(
            PrefilterSkip::UnamortizedPreparation.as_str(),
            "unamortized_bound_preparation"
        );
    }

    #[test]
    fn probe_ids_follow_the_weak_bound_rule() {
        assert!(probe_ids(511).is_empty());
        assert_eq!(probe_ids(512), (0..16).map(|t| t * 32).collect::<Vec<_>>());
        assert_eq!(probe_ids(2047).len(), 63);
        assert_eq!(probe_ids(2048).len(), 64);
        let ids = probe_ids(100_000);
        assert_eq!(ids.len(), 64);
        assert!(ids.windows(2).all(|w| w[0] < w[1]));
        assert_eq!(ids[1], 100_000 / 64);
    }

    fn never(_: &[usize], _: &mut [bool]) -> Result<(), ()> {
        panic!("no bound evaluated")
    }

    /// A batch probe that resolves exactly the IDs for which `f` holds.
    fn flags(f: impl Fn(usize) -> bool) -> impl FnOnce(&[usize], &mut [bool]) -> Result<(), ()> {
        move |ids, out| {
            assert_eq!(ids.len(), out.len());
            assert!(ids.windows(2).all(|w| w[0] < w[1]));
            for (slot, &k) in out.iter_mut().zip(ids) {
                *slot = f(k);
            }
            Ok(())
        }
    }

    #[test]
    fn decision_gate_and_equality() {
        // Small pass, short replay probe and the length gate evaluate nothing.
        let small = decide_outcome_bound(511, |_| Ok(Some((1, 1, 0))), never).unwrap();
        assert_eq!(small.reason, DecisionReason::Small);
        let short =
            decide_outcome_bound(512, |k| Ok((k < 15 * 32).then_some((1, 1, 0))), never).unwrap();
        assert_eq!(short.reason, DecisionReason::ProbeSmall);
        let gate = decide_outcome_bound(512, |_| Ok(Some((10, 65, 0))), never).unwrap();
        assert_eq!((gate.used, gate.reason), (false, DecisionReason::Gate));
        let empty = decide_outcome_bound(512, |_| Ok(Some((0, 1, 0))), never).unwrap();
        assert_eq!(empty.reason, DecisionReason::Gate);
        // 64·R = C uses the bound; one weight less does not (uniform ĉ = 48: r = 12/16).
        for (resolved, used) in [(12, true), (11, false)] {
            let d = decide_outcome_bound(
                512,
                |_| Ok::<_, ()>(Some((1, 48, 0))),
                flags(move |k| k / 32 < resolved),
            )
            .unwrap();
            assert_eq!((d.used, d.reason), (used, DecisionReason::Probe));
        }
        // The probe batch holds the remaining IDs only (a skipped item is not probed), and
        // R weighs each resolved item by its own e.
        let d = decide_outcome_bound(
            1024,
            |k| Ok::<_, ()>((k != 0).then_some((u64::from(k == 32) * 30 + 1, 32, 0))),
            |ids, out| {
                assert_eq!(ids.len(), 31);
                assert_eq!(ids[0], 32);
                out[0] = true; // e = 31 of E = 61: 64·31 ≥ 32·61
                Ok(())
            },
        )
        .unwrap();
        assert_eq!((d.used, d.reason), (true, DecisionReason::Probe));
    }

    #[test]
    fn probe_cost_gate() {
        // Uniform weights: the gate reads 32·q̂ > n. q̂ = 32: n = 1023 (p = 31) declines,
        // n = 1024 (p = 32) admits (equality).
        let d = decide_outcome_bound(1023, |_| Ok(Some((7, 0, 32))), never).unwrap();
        assert_eq!((d.used, d.reason), (false, DecisionReason::ProbeCost));
        let d = decide_outcome_bound(1024, |_| Ok::<_, ()>(Some((7, 0, 32))), flags(|_| false))
            .unwrap();
        assert_eq!((d.used, d.reason), (true, DecisionReason::Probe));
        // Q is a maximum over the probe: one heavy item decides.
        let d = decide_outcome_bound(
            2048,
            |k| Ok(Some(if k == 0 { (1000, 0, 40) } else { (1, 0, 40) })),
            never,
        )
        .unwrap();
        assert_eq!(d.reason, DecisionReason::ProbeCost);
        // q̂ = 0 disables the gate; the length gate comes first.
        let d =
            decide_outcome_bound(512, |_| Ok::<_, ()>(Some((7, 0, 0))), flags(|_| true)).unwrap();
        assert_eq!(d.reason, DecisionReason::Probe);
        let d = decide_outcome_bound(512, |_| Ok(Some((7, 65, 1 << 19))), never).unwrap();
        assert_eq!(d.reason, DecisionReason::Gate);
    }
}
