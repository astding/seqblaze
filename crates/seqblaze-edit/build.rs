//! Sets `seqblaze_avx512` when the compiler has the AVX-512 intrinsics and
//! target features (stable since Rust 1.89): the levenshtein lane pass and the
//! Hamming slice comparison then include their AVX-512 forms. Older compilers
//! within the workspace's rust-version build the levenshtein AVX2 and portable
//! forms and the portable Hamming comparison only; outputs never depend on the
//! form. Same probe as `crates/seqblaze-align/build.rs`.
use std::{env, process::Command};

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=RUSTC");
    println!("cargo:rustc-check-cfg=cfg(seqblaze_avx512)");
    let rustc = env::var_os("RUSTC").unwrap_or_else(|| "rustc".into());
    let version = Command::new(rustc)
        .arg("--version")
        .output()
        .ok()
        .and_then(|out| String::from_utf8(out.stdout).ok());
    // "rustc 1.99.0 (...)": the major and minor numbers.
    let release = version.as_deref().and_then(|text| {
        let mut parts = text.split_whitespace().nth(1)?.split('.');
        let major: u32 = parts.next()?.parse().ok()?;
        let minor: u32 = parts.next()?.parse().ok()?;
        Some((major, minor))
    });
    if release.is_some_and(|release| release >= (1, 89)) {
        println!("cargo:rustc-cfg=seqblaze_avx512");
    }
}
