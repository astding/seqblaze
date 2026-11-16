//! Host floating-point environment guard (D99; spec
//! docs/wiki/engine/text-nulls-fp-environment.md §7).
//!
//! Every host floating-point operation whose result reaches a returned value,
//! a kept row, a manifest value or an error runs in the IEEE default
//! environment E0, whatever the calling thread's environment is, and the
//! caller's environment is restored afterwards:
//!
//! - x86-64: MXCSR = 0x1F80 (round to nearest, FZ = DAZ = 0, every exception
//!   masked, status flags clear); the whole saved MXCSR is restored.
//! - aarch64: FPCR = 0 (round to nearest, FZ = DN = AHP = FZ16 = 0, no trap
//!   enabled, FEAT_AFP bits clear); FPCR, then FPSR, are restored.
//!
//! Other targets do not compile, so no build silently runs unguarded.
//!
//! [`FpGuard`] is a per-thread RAII value. Engine entry points, every
//! scheduler unit (spawned or inline) and every GIL-released binding region
//! hold one; guards are never held while Python code runs. LLVM assumes E0 and
//! may move floating-point operations on register values across the guard's
//! `asm!`, so scalar computations go through [`guarded`], which passes the
//! body and its result through `black_box` while the guard is held.
//!
//! [`FpBracket`] (caller-state bracket, §7.2) saves the thread's whole state
//! and writes it back when dropped, but installs nothing: code inside it runs
//! in the caller's environment, and only what it leaves in the state (status
//! flags, control changes) is undone. It encloses the few Python steps whose
//! status flags come from CPython's own arithmetic (§7.3 item 5). Guards stay
//! the only way to install E0.
use std::marker::PhantomData;

#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
compile_error!(
    "seqblaze's floating-point environment guard covers x86-64 and aarch64 only \
     (docs/wiki/engine/text-nulls-fp-environment.md §7.1)"
);

#[cfg(target_arch = "x86_64")]
mod arch {
    /// The whole MXCSR: control bits and status flags.
    pub type State = u32;
    /// E0: round to nearest, FZ = DAZ = 0, all six exceptions masked, flags clear.
    pub const E0: State = 0x1F80;

    #[inline(always)]
    pub fn read() -> State {
        let mut word: u32 = 0;
        // SAFETY: stmxcsr stores the 32-bit MXCSR into a valid, aligned local.
        // No `nomem`/`pure`: the compiler keeps memory accesses ordered with it.
        unsafe {
            core::arch::asm!(
                "stmxcsr [{}]",
                in(reg) &mut word as *mut u32,
                options(nostack, preserves_flags)
            );
        }
        word
    }

    /// # Safety
    /// `word` must be a valid MXCSR value (reserved bits 16-31 zero).
    #[inline(always)]
    pub unsafe fn write(word: State) {
        // SAFETY: the caller supplies a valid MXCSR value.
        unsafe {
            core::arch::asm!(
                "ldmxcsr [{}]",
                in(reg) &word as *const u32,
                options(nostack, preserves_flags)
            );
        }
    }

    #[inline(always)]
    pub fn install() -> State {
        let saved = read();
        // SAFETY: E0 is the architectural power-up value.
        unsafe { write(E0) };
        saved
    }

    #[inline(always)]
    pub fn restore(saved: State) {
        // SAFETY: `saved` was read from the register on this thread.
        unsafe { write(saved) };
    }

    /// The control bits only: the six status flags (bits 0-5) are ignored.
    pub fn is_e0(state: State) -> bool {
        state & !0x3F == E0
    }
}

#[cfg(target_arch = "aarch64")]
mod arch {
    /// (FPCR, FPSR).
    pub type State = (u64, u64);
    /// E0: FPCR = 0 (RES0 bits written as 0); FPSR cleared on entry.
    pub const E0: State = (0, 0);

    #[inline(always)]
    pub fn read() -> State {
        let fpcr: u64;
        let fpsr: u64;
        // SAFETY: reading FPCR and FPSR has no side effects. No `nomem`/`pure`.
        unsafe {
            core::arch::asm!("mrs {}, fpcr", out(reg) fpcr, options(nostack, preserves_flags));
            core::arch::asm!("mrs {}, fpsr", out(reg) fpsr, options(nostack, preserves_flags));
        }
        (fpcr, fpsr)
    }

    /// # Safety
    /// The values must be architecturally valid FPCR and FPSR contents.
    #[inline(always)]
    pub unsafe fn write(state: State) {
        // SAFETY: the caller supplies valid register contents; FPCR first.
        unsafe {
            core::arch::asm!("msr fpcr, {}", in(reg) state.0, options(nostack, preserves_flags));
            core::arch::asm!("msr fpsr, {}", in(reg) state.1, options(nostack, preserves_flags));
        }
    }

    #[inline(always)]
    pub fn install() -> State {
        let saved = read();
        // SAFETY: FPCR = 0 and FPSR = 0 are valid on every implementation.
        unsafe { write(E0) };
        saved
    }

    #[inline(always)]
    pub fn restore(saved: State) {
        // SAFETY: `saved` was read from the registers on this thread.
        unsafe { write(saved) };
    }

    pub fn is_e0(state: State) -> bool {
        state.0 == 0
    }
}

/// The thread's whole floating-point state (x86-64: MXCSR; aarch64: (FPCR, FPSR)).
pub type State = arch::State;

/// E0 as a [`State`] (x86-64: 0x1F80; aarch64: (0, 0)).
pub const E0: State = arch::E0;

/// The calling thread's floating-point state.
pub fn read_state() -> State {
    arch::read()
}

/// Write the calling thread's floating-point state (tests use it to set a
/// hostile caller state).
///
/// # Safety
/// The value must be a valid register state for the target. The optimizer
/// assumes E0, so floating-point code that runs in another state outside an
/// [`FpGuard`] may give environment-dependent results.
pub unsafe fn write_state(state: State) {
    // SAFETY: forwarded caller contract.
    unsafe { arch::write(state) }
}

/// Whether `state` holds E0's control bits.
pub fn is_e0(state: State) -> bool {
    arch::is_e0(state)
}

/// Saves the thread's floating-point state, installs E0, and restores the
/// saved state when dropped (normal return, `Err` return or unwinding).
/// Nested guards are allowed. Neither `Send` nor `Sync`: a guard belongs to
/// the thread that created it. Bind it to a named variable (`let _guard`),
/// never to `_`, which would drop it at once.
#[must_use = "a guard dropped at once restores the caller's state immediately"]
pub struct FpGuard {
    saved: State,
    _thread: PhantomData<*const ()>,
}

impl FpGuard {
    /// Save the current state and install E0.
    #[inline]
    pub fn new() -> Self {
        let saved = arch::install();
        debug_assert!(
            arch::read() == arch::E0,
            "floating-point environment E0 was not installed: {:?}",
            arch::read()
        );
        Self {
            saved,
            _thread: PhantomData,
        }
    }

    /// The state this guard restores on drop.
    pub fn saved(&self) -> State {
        self.saved
    }
}

impl Default for FpGuard {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for FpGuard {
    #[inline]
    fn drop(&mut self) {
        arch::restore(self.saved);
    }
}

/// Caller-state bracket: saves the thread's whole floating-point state
/// (x86-64: MXCSR; aarch64: FPCR and FPSR) and writes it back when dropped
/// (normal return, `Err` return or unwinding), without installing E0. Code
/// inside runs in the caller's environment; the flags it raises are discarded,
/// flags the caller had set are kept, and a control change it made is
/// reverted. Neither `Send` nor `Sync`, so only a state read on this thread is
/// ever written back. No floating-point result that reaches a value may be
/// computed under a bracket alone (that needs an [`FpGuard`]); a guard nested
/// inside restores what it saved.
#[must_use = "a bracket dropped at once restores the caller's state immediately"]
pub struct FpBracket {
    saved: State,
    _thread: PhantomData<*const ()>,
}

impl FpBracket {
    /// Save the current state; install nothing.
    #[inline]
    pub fn new() -> Self {
        Self {
            saved: arch::read(),
            _thread: PhantomData,
        }
    }

    /// The state this bracket writes back on drop.
    pub fn saved(&self) -> State {
        self.saved
    }
}

impl Default for FpBracket {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for FpBracket {
    #[inline]
    fn drop(&mut self) {
        arch::restore(self.saved);
    }
}

/// Run `body` under an [`FpGuard`], behind optimization barriers: the body
/// (and the values it captures) enters through `black_box` after E0 is
/// installed, and its result leaves through `black_box` before the caller's
/// state is restored, so no floating-point operation moves across the guard.
#[inline(never)]
pub fn guarded<R>(body: impl FnOnce() -> R) -> R {
    let _guard = FpGuard::new();
    let body = std::hint::black_box(body);
    std::hint::black_box(body())
}

/// Hostile caller states and an E0 probe for unit tests of guarded units
/// (§11 items 12-14).
#[cfg(test)]
pub(crate) mod testing {
    use super::*;

    #[cfg(target_arch = "x86_64")]
    pub const HOSTILE: [State; 5] = [0x9FFF, 0x3FBF, 0x5FBF, 0x7FBF, 0xFFFF];
    #[cfg(target_arch = "aarch64")]
    pub const HOSTILE: [State; 5] = [
        ((1 << 24) | (1 << 25), 0x9F),
        (0b01 << 22, 0x9F),
        (0b10 << 22, 0x9F),
        (0b11 << 22, 0x9F),
        ((1 << 24) | (1 << 25) | (0b11 << 22), 0x9F),
    ];

    /// Restores the test thread's state on every exit.
    pub struct Restore(pub State);
    impl Drop for Restore {
        fn drop(&mut self) {
            // SAFETY: the state was read on this thread.
            unsafe { write_state(self.0) };
        }
    }

    /// Install `state` and return what the register now holds (an emulator
    /// may not keep every bit), or None when the control bits did not stick.
    pub fn install(state: State) -> Option<State> {
        // SAFETY: valid register contents for the target.
        unsafe { write_state(state) };
        let got = read_state();
        #[cfg(target_arch = "x86_64")]
        let same = got & 0xFFC0 == state & 0xFFC0;
        #[cfg(target_arch = "aarch64")]
        let same = got.0 == state.0;
        same.then_some(got)
    }

    /// Whether the calling thread computes in E0: its control state is E0, and
    /// a subnormal quotient and a rounded sum have their round-to-nearest bits.
    pub fn computes_in_e0() -> bool {
        use std::hint::black_box;
        // FTZ: a subnormal result; DAZ: a subnormal operand.
        let tiny = black_box(f64::MIN_POSITIVE) / black_box(4.0);
        let back = black_box(f64::from_bits(0x0004_0000_0000_0000)) * black_box(4.0);
        // Directed rounding: one sum below and one above half an ulp of 1.0.
        let low = black_box(1.0f64) + black_box(f64::EPSILON / 4.0);
        let high = black_box(1.0f64) + black_box(f64::EPSILON / 4.0 * 3.0);
        is_e0(read_state())
            && tiny.to_bits() == 0x0004_0000_0000_0000
            && back == f64::MIN_POSITIVE
            && low.to_bits() == 1.0f64.to_bits()
            && high.to_bits() == (1.0f64 + f64::EPSILON).to_bits()
    }
}

#[cfg(test)]
mod tests {
    //! Guard behaviour from hostile states (§11 item 12). The public
    //! `tests/rust/fp_environment.rs` suite covers the same contract
    //! independently; these also run under `make test-aarch64`.
    use super::testing::{install, Restore, HOSTILE};
    use super::*;

    #[test]
    fn guard_installs_e0_and_restores_on_drop_err_and_panic() {
        let _restore = Restore(read_state());
        for hostile in HOSTILE {
            let Some(hostile) = install(hostile) else {
                continue;
            };
            {
                let _guard = FpGuard::new();
                assert!(is_e0(read_state()));
                std::hint::black_box(std::hint::black_box(1.0f64) / 3.0);
            }
            assert_eq!(read_state(), hostile);
            let err = || -> Result<(), ()> {
                let _guard = FpGuard::new();
                Err(())
            };
            assert!(err().is_err());
            assert_eq!(read_state(), hostile);
            let caught = std::panic::catch_unwind(|| {
                let _guard = FpGuard::new();
                panic!("unwinding through the guard");
            });
            assert!(caught.is_err());
            assert_eq!(read_state(), hostile);
            let value = guarded(|| {
                let _inner = FpGuard::new();
                assert!(is_e0(read_state()));
                std::hint::black_box(0.1f64) + std::hint::black_box(0.2f64)
            });
            assert_eq!(value.to_bits(), (0.1f64 + 0.2f64).to_bits());
            assert_eq!(read_state(), hostile);
        }
    }

    /// The same state with the six status flags clear.
    #[cfg(target_arch = "x86_64")]
    fn flags_clear(state: State) -> State {
        state & !0x3F
    }

    /// The same state with FPSR clear.
    #[cfg(target_arch = "aarch64")]
    fn flags_clear(state: State) -> State {
        (state.0, 0)
    }

    /// Whether the inexact flag PE (bit 5) is set.
    #[cfg(target_arch = "x86_64")]
    fn inexact(state: State) -> bool {
        state & 0x20 != 0
    }

    /// Whether the cumulative inexact flag FPSR.IXC (bit 4) is set.
    #[cfg(target_arch = "aarch64")]
    fn inexact(state: State) -> bool {
        state.1 & 0x10 != 0
    }

    fn third() -> f64 {
        std::hint::black_box(std::hint::black_box(1.0f64) / std::hint::black_box(3.0f64))
    }

    #[test]
    fn bracket_keeps_the_caller_environment_and_writes_its_whole_state_back() {
        let _restore = Restore(read_state());
        for hostile in HOSTILE {
            for start in [hostile, flags_clear(hostile)] {
                let Some(start) = install(start) else {
                    continue;
                };
                // Normal drop: inside, the caller's state (not E0); a flag raised and a
                // control change made inside are both undone at the exit.
                {
                    let bracket = FpBracket::new();
                    assert_eq!(bracket.saved(), start);
                    assert_eq!(read_state(), start);
                    third();
                    assert!(inexact(read_state()));
                    // SAFETY: E0 is valid on every target.
                    unsafe { write_state(E0) };
                    assert!(is_e0(read_state()));
                }
                assert_eq!(read_state(), start);
                // `Err` return.
                let err = || -> Result<(), ()> {
                    let _bracket = FpBracket::new();
                    third();
                    // SAFETY: E0 is valid on every target.
                    unsafe { write_state(E0) };
                    Err(())
                };
                assert!(err().is_err());
                assert_eq!(read_state(), start);
                // Panic caught with catch_unwind.
                let caught = std::panic::catch_unwind(|| {
                    let _bracket = FpBracket::new();
                    third();
                    // SAFETY: E0 is valid on every target.
                    unsafe { write_state(E0) };
                    panic!("unwinding through the bracket");
                });
                assert!(caught.is_err());
                assert_eq!(read_state(), start);
                // A guard nested inside a bracket restores the bracket's (the caller's) state.
                {
                    let _bracket = FpBracket::new();
                    {
                        let _guard = FpGuard::new();
                        assert!(is_e0(read_state()));
                        third();
                    }
                    assert_eq!(read_state(), start);
                    third();
                }
                assert_eq!(read_state(), start);
                // A bracket nested inside a guard restores E0 for the guard.
                {
                    let _guard = FpGuard::new();
                    {
                        let _bracket = FpBracket::new();
                        assert_eq!(read_state(), E0);
                        // SAFETY: `start` was read on this thread.
                        unsafe { write_state(start) };
                        third();
                    }
                    assert_eq!(read_state(), E0);
                }
                assert_eq!(read_state(), start);
            }
        }
    }
}
