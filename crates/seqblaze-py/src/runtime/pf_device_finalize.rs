//! Host preparation of K3's finite log table and per-call constants only.
//! Pair terms, keys and row sums never enter this module. The existing CPU map
//! exposes the exact anchor-table entries when evaluated at c_i (residual zero).
use seqblaze_cuda::interop::pf_finalize::PfFinalizeConstants;
use seqblaze_thermo::duplex::pf::{log_map, PfCall, PfConstants};

pub(super) fn constants(call: &PfCall) -> PfFinalizeConstants {
    with(&call.constants, call.loop_max())
}

/// The finalize constants of `constants` (K3's, or K1's `lse_dg_kcal` constants,
/// duplex-cuda.md §6.1, where Λ is carried but unused by the row finalize).
pub(super) fn with(constants: &PfConstants, loop_max: usize) -> PfFinalizeConstants {
    seqblaze_core::fpenv::guarded(|| {
        let mut table = [0u64; 128];
        for (i, entry) in table.iter_mut().enumerate() {
            // c_i = 1 + i/128 exactly, formed without a rounded host operation.
            let anchor = f64::from_bits(0x3FF0_0000_0000_0000 | ((i as u64) << 45));
            *entry = log_map(anchor, 0).to_bits();
        }
        PfFinalizeConstants {
            table,
            l2: log_map(1.0, 1).to_bits(),
            c_t: constants.c_t.to_bits(),
            kappa: constants.kappa.to_bits(),
            // Same compile-time RN constants as duplex/pf.rs, not fitted coefficients.
            b3: (2.0f64 / 3.0).to_bits(),
            b5: (2.0f64 / 5.0).to_bits(),
            b7: (2.0f64 / 7.0).to_bits(),
            loop_max: loop_max as u64,
        }
    })
}
