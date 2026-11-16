//! Named `-gencode` sets of the CUDA build (docs/wiki/gpu/cuda-release-checks.md §2.1).
//!
//! `build.rs` includes this file by path (like `ptx_fp.rs`) and selects a set by the
//! environment variable `SEQBLAZE_CUDA_ARCHS`; `tests/rust/cuda_archsets.rs` includes it the
//! same way, so it is self-contained (no `crate::` paths) and needs neither nvcc nor the
//! `cuda` feature.
//!
//! - unset, empty or `dev`: `dev`, today's sm_120 SASS plus compute_120 PTX (the default; the
//!   flag string is byte-identical to the one used before the sets existed);
//! - `release`: D10 and D85 (9)(a), SASS for sm_75, sm_80, sm_86, sm_89, sm_90, sm_100 and
//!   sm_120 (plain targets only) plus compute_120 PTX;
//! - `ptx`: compute_120 PTX only, so the driver must JIT every kernel (§7.2);
//! - any other value (case-sensitive, untrimmed): an error naming the value.
//!
//! Every set shares [`COMMON`]: the AGENTS.md §3 determinism flags. Never fast-math.

/// The leading nvcc flags of every set: fatbin output, C++17 and the four determinism flags.
pub const COMMON: &[&str] = &[
    "--fatbin",
    "-std=c++17",
    "--fmad=false",
    "--prec-div=true",
    "--prec-sqrt=true",
    "--ftz=false",
];

/// `dev`: sm_120 SASS plus compute_120 PTX.
const DEV: &[&str] = &[
    "-gencode",
    "arch=compute_120,code=sm_120",
    "-gencode",
    "arch=compute_120,code=compute_120",
];

/// `release`: the D10 architecture list, in order, plus compute_120 PTX.
const RELEASE: &[&str] = &[
    "-gencode",
    "arch=compute_75,code=sm_75",
    "-gencode",
    "arch=compute_80,code=sm_80",
    "-gencode",
    "arch=compute_86,code=sm_86",
    "-gencode",
    "arch=compute_89,code=sm_89",
    "-gencode",
    "arch=compute_90,code=sm_90",
    "-gencode",
    "arch=compute_100,code=sm_100",
    "-gencode",
    "arch=compute_120,code=sm_120",
    "-gencode",
    "arch=compute_120,code=compute_120",
];

/// `ptx`: compute_120 PTX only.
const PTX: &[&str] = &["-gencode", "arch=compute_120,code=compute_120"];

/// The set selected by `value` (the content of `SEQBLAZE_CUDA_ARCHS`, `None` when unset):
/// its name and its `-gencode` arguments (two elements per pair), or an error message that
/// names the value and the accepted values.
pub fn gencodes(value: Option<&str>) -> Result<(&'static str, &'static [&'static str]), String> {
    match value {
        None | Some("") | Some("dev") => Ok(("dev", DEV)),
        Some("release") => Ok(("release", RELEASE)),
        Some("ptx") => Ok(("ptx", PTX)),
        Some(other) => Err(format!(
            "SEQBLAZE_CUDA_ARCHS={other:?} is not an architecture set; accepted values are \
             \"dev\" (the default: sm_120 SASS + compute_120 PTX), \"release\" (SASS for sm_75, \
             sm_80, sm_86, sm_89, sm_90, sm_100, sm_120 + compute_120 PTX) and \"ptx\" \
             (compute_120 PTX only). The value is case-sensitive and not trimmed; unset \
             SEQBLAZE_CUDA_ARCHS for the dev set."
        )),
    }
}
