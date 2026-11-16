//! seqblaze core: data containers, pair work-lists and CPU kernels.
//!
//! Phase 1 scope (see `docs/wiki/project/phase1-scope.md`): all-pairs and
//! pair-list thermodynamic interaction kernels for DNA sequence lists, with a
//! multithreaded CPU backend and a CUDA backend that agree bit for bit.

pub mod batch;
pub mod build_info;
pub mod cancel;
pub mod detail;
pub mod exec;
pub mod fpenv;
pub mod kernel;
pub mod plan;
pub mod sched;
pub mod seq;
pub mod sink;
pub mod unary;
pub mod validate;
pub mod work;

// aarch64-only run smoke test for `make test-aarch64` (dev-harness H-R7).

pub use batch::SeqBatch;
pub use build_info::{build_info, BuildInfo};

/// Crate version, shared with the Python package.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
