//! Thermodynamic kernels. T1 native `tm` (docs/wiki/thermo/tm.md): per-call host
//! constants in exact and correctly rounded arithmetic (`nat`, `cr`), parameters
//! generated from the accepted ledger (`params`), the unary kernel (`tm`); T1
//! `compat="biopython"` (docs/wiki/thermo/tm-biopython.md, `biopython`); T1
//! `compat="primer3"` (docs/wiki/thermo/tm-primer3.md, `primer3`). K1 native
//! `duplex` MFE (docs/wiki/thermo/duplex-mfe.md, `duplex`): the compiled pack loader,
//! the per-call cell keys and the integer recurrence; `tm` on `Pairs` (the register
//! pairing, docs/wiki/thermo/tm.md §13, `duplex::register`) on the same per-call pack.
//! T2 `hairpin` `compat="primer3"` (docs/wiki/thermo/hairpin-primer3.md, `hairpin`)
//! over the primer3-py parameter files
//! read at run time (docs/wiki/thermo/primer3-parameter-loader.md, `p3files`). K2
//! `duplex` `compat="primer3"` (docs/wiki/thermo/dimer-primer3.md, `dimer`): primer3-py's
//! heterodimer, homodimer and end stability over the same run-time tables.
//! No transcendental function runs per sequence; device code would only read the
//! per-call pack (numerics-determinism rule 2).
pub mod biopython;
pub mod cr;
pub mod dimer;
pub mod duplex;
pub mod hairpin;
pub mod nat;
pub mod p3files;
pub mod params;
pub mod primer3;
pub mod tm;

pub use biopython::{BiopythonPack, TmBiopython};
pub use dimer::{DimerPack, DimerPrimer3};
pub use duplex::{pf::DuplexPf, Caller, Duplex, DuplexCall, TmPairs, DUPLEX_BOUND};
pub use hairpin::{HairpinPack, HairpinPrimer3};
pub use p3files::P3Tables;
pub use params::{native_params, BIOPYTHON_NN3, PRIMER3_SANTALUCIA, SANTALUCIA2004};
pub use primer3::{Primer3Pack, TmPrimer3};
pub use tm::{CallPack, Conditions, Oligo, Tm};

/// T1 has no bound (tm.md §6): `strategy="bounded"` resolves to full.
pub const TM_BOUND: seqblaze_core::plan::BoundAvailability =
    seqblaze_core::plan::BoundAvailability::Unavailable;
