//! Runtime description of the build: version, detected SIMD level, compiled
//! backends. Benchmarks and bug reports record this so results can be traced
//! to the code path that produced them.

/// What this build can run on the current machine.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BuildInfo {
    pub version: &'static str,
    /// Source commit captured when built, or "unknown" for a source archive.
    pub git: &'static str,
    /// Tracked or nonignored untracked source changes at build time; None when
    /// source Git state could not be established. Never queried at runtime.
    pub git_dirty: Option<bool>,
    pub target_arch: &'static str,
    /// Best SIMD level detected at runtime: "avx512", "avx2", "sse4.1", "neon" or "scalar".
    pub simd: &'static str,
    /// Always false here: core never depends on the CUDA crate. The
    /// composition root (`seqblaze-py`) reports `seqblaze_cuda::COMPILED`.
    pub cuda_compiled: bool,
    pub debug_assertions: bool,
}

pub fn build_info() -> BuildInfo {
    BuildInfo {
        version: crate::VERSION,
        git: env!("SEQBLAZE_BUILD_GIT"),
        git_dirty: match env!("SEQBLAZE_BUILD_GIT_DIRTY") {
            "true" => Some(true),
            "false" => Some(false),
            _ => None,
        },
        target_arch: std::env::consts::ARCH,
        simd: detect_simd(),
        cuda_compiled: false,
        debug_assertions: cfg!(debug_assertions),
    }
}

#[cfg(target_arch = "x86_64")]
fn detect_simd() -> &'static str {
    if std::is_x86_feature_detected!("avx512bw") {
        "avx512"
    } else if std::is_x86_feature_detected!("avx2") {
        "avx2"
    } else if std::is_x86_feature_detected!("sse4.1") {
        "sse4.1"
    } else {
        "scalar"
    }
}

#[cfg(target_arch = "aarch64")]
fn detect_simd() -> &'static str {
    "neon"
}

#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
fn detect_simd() -> &'static str {
    "scalar"
}

#[cfg(test)]
mod tests {
    #[test]
    fn reports_version() {
        let info = super::build_info();
        assert_eq!(info.version, crate::VERSION);
        assert!(!info.cuda_compiled);
    }
}
