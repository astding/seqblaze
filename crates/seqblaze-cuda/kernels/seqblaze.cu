// The single translation unit of the seqblaze fatbin (docs/wiki/gpu/cuda-backend.md).
// build.rs compiles it with nvcc 12.8/12.9 and the AGENTS.md §3 flags
// (--fmad=false --prec-div=true --prec-sqrt=true --ftz=false) into SASS for
// sm_120 plus PTX for compute_120, and scans the module's PTX for every floating-point
// type (no device floating point, D55; tm-cuda.md §4.7, duplex-pf-cuda.md §8 item 6). Each kernel family is one header; several .cu files in
// one fatbin would need -rdc and a device link.
#include "hamming.cuh"
#include "levenshtein.cuh"
#include "align.cuh"
#include "softf64.cuh"
#include "unary.cuh"
#include "tm.cuh"
#include "duplex.cuh"
#include "duplex_bound.cuh"
#include "f53.cuh"
#include "duplex_pf.cuh"
#include "pf_resident.cuh"
#include "k1_lse.cuh"
#include "resident_routes.cuh"
#include "duplex_sparse.cuh"
#include "resident_bound.cuh"
#include "duplex_pf_bound.cuh"
#include "resident_pf_bound.cuh"
#include "register.cuh"
#include "tm_compat.cuh"
#include "primer3.cuh"
#include "resident_p3.cuh"

#include "device_interop.cuh"
#include "device_digest.cuh"

#include "resident_native.cuh"

#include "detail.cuh"
#include "resident_sinks.cuh"

#include "resident_validate.cuh"

#include "resident_edit_stage.cuh"

#include "resident_indices.cuh"
#include "resident_tiles.cuh"

#include "resident_selection.cuh"

#include "resident_unary.cuh"
