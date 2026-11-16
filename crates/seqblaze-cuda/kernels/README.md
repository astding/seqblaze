# CUDA kernels

Built (milestone G1, `docs/wiki/gpu/cuda-backend.md`): `seqblaze.cu` includes `hamming.cuh`,
the device `hamming` distance stage (`sb_hamming_pieces`) and its device sinks (`sb_each`,
`sb_hits_count`/`sb_hits_write`, `sb_rowstats_min`/`sb_rowstats_arg`). `build.rs` compiles it
with nvcc 12.8/12.9 (`SEQBLAZE_NVCC`) and `--fmad=false --prec-div=true --prec-sqrt=true
--ftz=false` into one fatbin (sm_120 SASS + compute_120 PTX) only when feature `cuda` is on.
Integer arithmetic and integer atomics only.

`levenshtein.cuh` holds device `levenshtein` (`docs/wiki/gpu/levenshtein-cuda.md`). Device `tm`
(`docs/wiki/gpu/tm-cuda.md`), the first unary kernel, adds three headers:
- `softf64.cuh`: the shared soft-binary64 routine (D55). It provides `from_i64`, `add`, `sub`, `mul`,
  `div`, `ge` and `canon` on u64 bit patterns (roundTiesToEven, gradual underflow, one canonical
  NaN), plus the testing kernel `sb_f64_eval`.
- `unary.cuh`: the generic unary Hits compaction, `sb_unary_hits_count`/`sb_unary_hits_write` with
  column descriptors.
- `tm.cuh`: the stage kernel `sb_tm`, which computes the integer sums, F1–F10, the cutoff and the
  Each masks.

`align.cuh` holds device `align`, the engine key of native local/global Gotoh and compat
`emboss-water`/`emboss-needle` (`docs/wiki/gpu/align-cuda.md`). Two duplex kernels are built:
- `duplex.cuh`: K1, native `duplex(model="mfe")`, the integer DP of one pair per sub-warp group, its
  tier predicate and routing, and the detail pass (`docs/wiki/gpu/duplex-cuda.md`). `finalize.cuh`
  holds the soft-binary64 finalize that `tm.cuh` and `duplex.cuh` share.
- `f53.cuh` and `duplex_pf.cuh`: K3, native `duplex(model="pf")`. `f53.cuh` is the integer software
  float F53 (u64 significand, i64 exponent, round to nearest even, the CPU's bits). `duplex_pf.cuh`
  computes Ẑ of one pair per sub-warp group in the CPU's fixed fold orders, with its tier predicate
  and routing; keys, row terms and sinks stay on the host (`docs/wiki/gpu/duplex-pf-cuda.md`).
- `k1_lse.cuh`: K1's RowStats `lse_dg_kcal` on the resident path (`docs/wiki/gpu/duplex-cuda.md`
  §6.1): the integer window reduction of a chunk's looked-up keys (`sb_k1_lse_window`) and the gather
  of each item's term from the host-built table of W(D) (`sb_k1_lse_gather`); the exact row sums are
  `pf_resident.cuh`'s.

`register.cuh` holds device `tm` on a `Pairs` input (`docs/wiki/gpu/tm-pairs-cuda.md`): the stage kernel
`sb_reg` (one thread per pair: the register key from K1's full cell keys, the shared finalize and the
int64 selecting key). The G1 Hits and RowStats sink kernels also have int64-key instances (`_i64`), and
`sb_fill_i64` sets RowStats' int64 best to `Score::MASK`.

`primer3.cuh` holds T2 `hairpin` and K2 `duplex` with the primer3 preset (`docs/wiki/gpu/primer3-cuda.md`):
the soft-binary64 replay of the CPU kernels, one sub-warp group per item (`sb_p3_hairpin_g*`,
`sb_p3_dimer_g*`) with the item's whole cell table in a global slot, K2's own int64-key Each, Hits and
RowStats sinks (`sb_each64`, `sb_hits_count64`/`sb_hits_write64`, `sb_rowstats_min64`/
`sb_rowstats_arg64`) and the scatter of routed K2 results (`sb_p3_scatter`). For it `softf64.cuh` also
provides the quiet comparisons `lt`, `le`, `gt` and `eq`, and `is_finite` and `abs`.

`build.rs` also compiles the module to PTX and fails the build on any floating-point PTX type outside
comments and strings (`src/ptx_fp.rs`: the type grammar, so `f16`, `bf16`, `tf32`, `f32`, `f64`, the
8-, 6- and 4-bit float types and their packed forms such as `f16x2`, `f32x2` and `e4m3x4`), in
instruction suffixes as well as declarations, so no device code uses floating point (D55).

Phase-1 kernels and their mapping, from the kernel design notes in `docs/wiki/gpu/kernel-design.md`:

| Kernel | What it computes | Mapping |
|---|---|---|
| K1 (built) | Intermolecular-only hybridization MFE (integer energies) | sub-warp per pair (g = 4–32 lanes by length bin), rows sequential, rolling band in shared memory or a global slot |
| K2 (built) | primer3-thal-compatible ANY/END1/END2 (int64 keys, FP64 replay; `docs/wiki/gpu/primer3-cuda.md`) | sub-warp per pair, the whole cell table in a global slot |
| K3 (built) | Intermolecular hybridization partition function in the integer software float F53 (`docs/wiki/gpu/duplex-pf-cuda.md`) | same as K1 |
| K4 | Cofold MFE / partition function | sub-warp diagonal wavefront (n+m ≤ ~80), block per pair beyond |
| sinks | thresholded COO, per-row min/argmin/count/LSE, top-k | warp-aggregated staging, deterministic ordering |

K1, K2 and K3 are built; K4 has no device kernel yet. Kernels are written
only from the specs in `docs/wiki/` and validated against the CPU reference bit for bit.
