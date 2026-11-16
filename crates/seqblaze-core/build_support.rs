//! Build-time Git discovery. No Git command or source path reaches the library's
//! runtime. The same std-only routine is exercised by tiny real-repository tests.
use std::{
    path::{Path, PathBuf},
    process::{Command, Stdio},
};

#[derive(Debug, PartialEq, Eq)]
pub struct SourceIdentity {
    pub git: String,
    pub dirty: Option<bool>,
    pub watched: Vec<PathBuf>,
}

fn git(root: &Path, args: &[&str]) -> Option<Vec<u8>> {
    let output = Command::new("git")
        .arg("--no-optional-locks")
        .arg("-C")
        .arg(root)
        .args(args)
        // A caller's unrelated Git context must not override this source root.
        .env_remove("GIT_DIR")
        .env_remove("GIT_WORK_TREE")
        .env_remove("GIT_INDEX_FILE")
        .env_remove("GIT_COMMON_DIR")
        .env_remove("GIT_NAMESPACE")
        .stderr(Stdio::null())
        .output()
        .ok()?;
    output.status.success().then_some(output.stdout)
}
fn text(root: &Path, args: &[&str]) -> Option<String> {
    String::from_utf8(git(root, args)?)
        .ok()
        .map(|value| value.trim().to_owned())
}
fn git_path(root: &Path, name: &str) -> Option<PathBuf> {
    let path = PathBuf::from(text(root, &["rev-parse", "--git-path", name])?);
    Some(if path.is_absolute() {
        path
    } else {
        root.join(path)
    })
}

pub fn source_identity(root: &Path) -> SourceIdentity {
    let unknown = || SourceIdentity {
        git: "unknown".into(),
        dirty: None,
        watched: Vec::new(),
    };
    let Ok(root) = root.canonicalize() else {
        return unknown();
    };
    let Some(top) = text(&root, &["rev-parse", "--show-toplevel"]) else {
        return unknown();
    };
    // A source archive nested in somebody else's repository has no source Git
    // identity. Never attribute it to that enclosing repository's commit.
    if Path::new(&top).canonicalize().ok().as_ref() != Some(&root) {
        return unknown();
    }
    let Some(commit) = text(&root, &["rev-parse", "--verify", "HEAD^{commit}"]) else {
        return unknown();
    };
    if !matches!(commit.len(), 40 | 64) || !commit.bytes().all(|b| b.is_ascii_hexdigit()) {
        return unknown();
    }
    // `make build-release` swaps the generated THIRD-PARTY-NOTICES for the build's
    // feature-specific notices while it compiles (harness/release/with_notices.py) and
    // the wheel check verifies that content, so that one file never marks a build dirty.
    let dirty = git(
        &root,
        &[
            "status",
            "--porcelain=v1",
            "--untracked-files=normal",
            "--",
            ":(top)",
            ":(top,exclude)THIRD-PARTY-NOTICES",
        ],
    )
    .map(|status| !status.is_empty());
    let mut watched = Vec::new();
    for name in ["HEAD", "index", "packed-refs"] {
        if let Some(path) = git_path(&root, name).filter(|path| path.exists()) {
            watched.push(path);
        }
    }
    if let Some(reference) = text(&root, &["symbolic-ref", "-q", "HEAD"]) {
        if let Some(path) = git_path(&root, &reference).filter(|path| path.exists()) {
            watched.push(path);
        }
    }
    // Git metadata covers commits/staging; source-file watches also refresh
    // direct Cargo builds on known tracked/untracked edits. Maintained Make
    // builds additionally refresh the existing lane-state env fingerprint,
    // covering newly created nonignored source files without watching target/.
    if let Some(files) = git(
        &root,
        &[
            "ls-files",
            "-z",
            "--cached",
            "--others",
            "--exclude-standard",
        ],
    ) {
        for name in files
            .split(|&byte| byte == 0)
            .filter(|name| !name.is_empty())
        {
            if let Ok(name) = std::str::from_utf8(name) {
                if !name.contains('\n') && !name.contains('\r') {
                    watched.push(root.join(name));
                }
            }
        }
    }
    watched.sort();
    watched.dedup();
    SourceIdentity {
        git: commit.to_ascii_lowercase(),
        dirty,
        watched,
    }
}
