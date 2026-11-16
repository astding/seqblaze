// Generic unary selecting key; arithmetic stays in the family's existing stage.
#pragma once
#include "register.cuh"
#include "primer3.cuh"
extern "C" __global__ void sb_resident_unary_keys(const sb_u64*tm,sb_u64 n,sb_i64*keys,sb_u64 decreasing) {
    const sb_u64 step=(sb_u64)gridDim.x*blockDim.x;
    for(sb_u64 t=(sb_u64)blockIdx.x*blockDim.x+threadIdx.x;t<n;t+=step)keys[t]=decreasing?sb_reg_key(tm[t]):sb_p3_key(tm[t]);
}
