//! Compiles kernels/seqblaze.cu into one fatbin when feature `cuda` is on, and
//! fails on any floating-point type in the module's PTX (tm-cuda.md §4.7,
//! duplex-pf-cuda.md §8 item 6); does nothing
//! otherwise. Spec: docs/wiki/gpu/cuda-backend.md (Build).
// The scan lives in src/ptx_fp.rs so that the crate's unit tests check it too.
#[path = "src/ptx_fp.rs"]
mod ptx_fp;
// The named -gencode sets (cuda-release-checks.md §2.1), unit-tested by
// tests/rust/cuda_archsets.rs.
#[path = "src/archsets.rs"]
mod archsets;

use std::{
    env,
    path::{Path, PathBuf},
    process::Command,
};

/// Selects the `-gencode` set ([`archsets::gencodes`]): unset, empty or `dev` (the
/// default), `release` or `ptx`; any other value fails the build.
const ARCHS_ENV: &str = "SEQBLAZE_CUDA_ARCHS";

/// The PTX of the same module for the floating-point scan (tm-cuda.md §4.7): the four
/// determinism flags, the virtual architecture only.
const PTX_FLAGS: &[&str] = &[
    "--ptx",
    "-std=c++17",
    "--fmad=false",
    "--prec-div=true",
    "--prec-sqrt=true",
    "--ftz=false",
    "-arch=compute_120",
];

/// Testing builds (feature `testing`) also compile the device-side poison fill of
/// the duplex band slots (duplex-cuda.md §8 item 8, QKG11); release builds
/// compile no fill. A preprocessor define only: no numeric flag changes.
const TESTING_DEFINE: &str = "-DSB_TESTING=1";

/// nvcc adds these to every invocation, after or before the command line, so
/// they could override the fatbin flags (for example `--fmad=true` or `--use_fast_math`)
/// while SEQBLAZE_NVCC_FLAGS still reported the flags passed.
const NVCC_ENV_FLAGS: &[&str] = &["NVCC_APPEND_FLAGS", "NVCC_PREPEND_FLAGS"];

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=kernels/");
    println!("cargo:rerun-if-changed=src/ptx_fp.rs");
    println!("cargo:rerun-if-changed=src/archsets.rs");
    println!("cargo:rerun-if-env-changed=SEQBLAZE_NVCC");
    println!("cargo:rerun-if-env-changed={ARCHS_ENV}");
    println!("cargo:rerun-if-env-changed=CUDA_HOME");
    for name in NVCC_ENV_FLAGS {
        println!("cargo:rerun-if-env-changed={name}");
    }
    if env::var_os("CARGO_FEATURE_CUDA").is_none() {
        return;
    }
    let flags = fatbin_flags();
    let nvcc = find_nvcc();
    // Fail rather than ignore: a set value means the caller expects nvcc to use it.
    for name in NVCC_ENV_FLAGS {
        if let Some(value) = env::var_os(name).filter(|value| !value.is_empty()) {
            panic!(
                "seqblaze-cuda feature `cuda`: {name}={:?} is set. nvcc would add it to every \
                 compile and could override the AGENTS.md §3 determinism flags ({}), while \
                 build_info()['nvcc_flags'] still reported them. Unset {name} and rebuild.",
                value,
                flags.join(" ")
            );
        }
    }
    let output = nvcc_command(&nvcc)
        .arg("--version")
        .output()
        .unwrap_or_else(|error| fail(&nvcc, &format!("cannot run it: {error}")));
    let text = String::from_utf8_lossy(&output.stdout).into_owned();
    let release = text
        .split("release ")
        .nth(1)
        .and_then(|rest| rest.split(',').next())
        .map(str::trim)
        .unwrap_or("")
        .to_owned();
    let full = text
        .split(", V")
        .nth(1)
        .and_then(|rest| rest.split_whitespace().next())
        .unwrap_or(&release)
        .to_owned();
    if !output.status.success() || !matches!(release.as_str(), "12.8" | "12.9") {
        fail(
            &nvcc,
            &format!(
                "it reports release {:?}; sm_120 needs nvcc 12.8 or 12.9 and D10 excludes 13.x",
                if release.is_empty() {
                    "unknown"
                } else {
                    &release
                }
            ),
        );
    }
    let out = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR")).join("seqblaze.fatbin");
    let status = nvcc_command(&nvcc)
        .args(&flags)
        .args(testing_define())
        .arg("-o")
        .arg(&out)
        .arg("kernels/seqblaze.cu")
        .status()
        .unwrap_or_else(|error| fail(&nvcc, &format!("cannot run it: {error}")));
    if !status.success() {
        fail(
            &nvcc,
            &format!("compiling kernels/seqblaze.cu failed ({status})"),
        );
    }
    no_fp(&nvcc, &out.with_file_name("seqblaze.ptx"));
    println!("cargo:rustc-env=SEQBLAZE_NVCC_VERSION={full}");
    println!("cargo:rustc-env=SEQBLAZE_NVCC_FLAGS={}", flags.join(" "));
}

/// The fatbin flags: [`archsets::COMMON`], then the `-gencode` pairs of the set named by
/// `SEQBLAZE_CUDA_ARCHS`. An unknown value or a non-UTF-8 one fails the build.
fn fatbin_flags() -> Vec<&'static str> {
    let value = env::var_os(ARCHS_ENV).map(|value| {
        value.into_string().unwrap_or_else(|value| {
            panic!(
                "seqblaze-cuda: {ARCHS_ENV}={value:?} is not UTF-8; accepted values are \
                 \"dev\", \"release\" and \"ptx\"; unset {ARCHS_ENV} for the dev set."
            )
        })
    });
    let (_, gencodes) =
        archsets::gencodes(value.as_deref()).unwrap_or_else(|why| panic!("seqblaze-cuda: {why}"));
    archsets::COMMON.iter().chain(gencodes).copied().collect()
}

/// D55 and the phase-1 "no device floating point" rule (tm-cuda.md §4.7, P6;
/// duplex-pf-cuda.md §8 item 6): compile the module to PTX and fail the build on any
/// floating-point PTX type ([`ptx_fp::is_fp_type`], by the type grammar, packed forms
/// included) outside comments and strings, wherever it stands: an instruction's type
/// suffix, a register declaration or a parameter, in any kernel.
fn no_fp(nvcc: &Path, ptx: &Path) {
    let status = nvcc_command(nvcc)
        .args(PTX_FLAGS)
        .args(testing_define())
        .arg("-o")
        .arg(ptx)
        .arg("kernels/seqblaze.cu")
        .status()
        .unwrap_or_else(|error| fail(nvcc, &format!("cannot run it: {error}")));
    if !status.success() {
        fail(
            nvcc,
            &format!("compiling kernels/seqblaze.cu to PTX failed ({status})"),
        );
    }
    let text = std::fs::read_to_string(ptx)
        .unwrap_or_else(|error| panic!("reading {}: {error}", ptx.display()));
    if let Some((number, line, token)) = ptx_fp::first_fp(&text) {
        panic!(
            "seqblaze-cuda: the device module uses floating point: PTX line {number} \
             of {} names the floating-point type `{token}`: {}. Device code must not use \
             any floating-point type (D55; docs/wiki/gpu/tm-cuda.md §4.7, \
             docs/wiki/gpu/duplex-pf-cuda.md §8 item 6): use the integer routines of \
             kernels/softf64.cuh or kernels/f53.cuh instead of `float`, `double`, a packed \
             form such as `float2`, or a half, bfloat16 or narrow float type.",
            ptx.display(),
            line.trim()
        );
    }
}

/// [`TESTING_DEFINE`] in a testing build, else nothing.
fn testing_define() -> Option<&'static str> {
    env::var_os("CARGO_FEATURE_TESTING").map(|_| TESTING_DEFINE)
}

/// nvcc with exactly the flags given on its command line (empty NVCC_*_FLAGS
/// values are removed too; non-empty ones were rejected above).
fn nvcc_command(nvcc: &Path) -> Command {
    let mut command = Command::new(nvcc);
    for name in NVCC_ENV_FLAGS {
        command.env_remove(name);
    }
    command
}

/// SEQBLAZE_NVCC (absolute path), then $CUDA_HOME/bin/nvcc, then nvcc on PATH.
fn find_nvcc() -> PathBuf {
    if let Some(path) = env::var_os("SEQBLAZE_NVCC").filter(|path| !path.is_empty()) {
        let path = PathBuf::from(path);
        if !path.is_absolute() {
            fail(&path, "SEQBLAZE_NVCC must be an absolute path");
        }
        return path;
    }
    if let Some(home) = env::var_os("CUDA_HOME").filter(|home| !home.is_empty()) {
        let path = Path::new(&home).join("bin").join("nvcc");
        if path.is_file() {
            return path;
        }
    }
    PathBuf::from("nvcc")
}

fn fail(nvcc: &Path, why: &str) -> ! {
    let hint = if env::var_os("SEQBLAZE_NVCC").is_some_and(|path| !path.is_empty()) {
        "Fix or unset SEQBLAZE_NVCC (it takes precedence over CUDA_HOME and PATH)"
    } else {
        "Set SEQBLAZE_NVCC to an absolute nvcc 12.8/12.9 path, or set CUDA_HOME, or put nvcc \
         on PATH"
    };
    panic!(
        "seqblaze-cuda feature `cuda` needs nvcc 12.8 or 12.9: {} failed: {why}. {hint}; \
         a cuda build never ships without kernels.",
        nvcc.display()
    )
}
