mod build_support;
use std::{env, path::Path};

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=build_support.rs");
    // Only an invalidation key from the maintained harness. Git identity is
    // independently read below, never trusted from a caller-supplied env value.
    println!("cargo:rerun-if-env-changed=SEQBLAZE_BUILD_SOURCE_STATE");
    let manifest = env::var_os("CARGO_MANIFEST_DIR").expect("Cargo manifest directory");
    let root = Path::new(&manifest)
        .ancestors()
        .nth(2)
        .expect("workspace source root");
    let identity = build_support::source_identity(root);
    for path in identity.watched {
        if let Some(path) = path
            .to_str()
            .filter(|path| !path.contains('\n') && !path.contains('\r'))
        {
            println!("cargo:rerun-if-changed={path}");
        }
    }
    println!("cargo:rustc-env=SEQBLAZE_BUILD_GIT={}", identity.git);
    println!(
        "cargo:rustc-env=SEQBLAZE_BUILD_GIT_DIRTY={}",
        match identity.dirty {
            Some(true) => "true",
            Some(false) => "false",
            None => "unknown",
        }
    );
}
