// Device-native Result selection. No score kernel is invoked and no numeric,
// ragged or normalized-index column is downloaded. Integer controls only.
#pragma once
#include "resident_indices.cuh"

__device__ __forceinline__ sb_u64 sb_selection_bits(const SbIndexView& v, sb_u64 at) {
    const sb_u8* bytes=(const sb_u8*)(v.address+at*(sb_u64)v.stride);
    sb_u64 bits=0;
    for(sb_u64 k=0;k<v.width;++k) bits|=(sb_u64)bytes[k]<<(8*k);
    if(v.is_signed&&v.width<8&&(bits&(1ull<<(8*v.width-1))))
        bits|=(~0ull)<<(8*v.width);
    return bits;
}
extern "C" __global__ void sb_selection_normalize(
    SbIndexView input,sb_u64 rows,sb_u64 output,sb_u64 first_bad) {
    const sb_u64 step=(sb_u64)gridDim.x*blockDim.x;
    for(sb_u64 t=(sb_u64)blockIdx.x*blockDim.x+threadIdx.x;t<input.n;t+=step){
        const sb_u64 bits=sb_selection_bits(input,t);
        sb_u64 at=bits; bool bad=false;
        if(input.is_signed&&(bits>>63)){
            const sb_u64 magnitude=(~bits)+1;
            bad=magnitude>rows; at=rows-magnitude;
        }else bad=bits>=rows;
        if(bad){atomicMin((sb_u64*)first_bad,t);at=0;}
        ((sb_i64*)output)[t]=(sb_i64)at;
    }
}
extern "C" __global__ void sb_selection_mask_counts(SbIndexView input,sb_u64 counts){
    const sb_u64 step=(sb_u64)gridDim.x*blockDim.x;
    for(sb_u64 t=(sb_u64)blockIdx.x*blockDim.x+threadIdx.x;t<input.n;t+=step)
        ((sb_u64*)counts)[t]=sb_selection_bits(input,t)!=0;
}
extern "C" __global__ void sb_selection_mask_positions(
    sb_u64 counts,sb_u64 n,sb_u64 prefix,sb_u64 output){
    const sb_u64 step=(sb_u64)gridDim.x*blockDim.x;
    for(sb_u64 t=(sb_u64)blockIdx.x*blockDim.x+threadIdx.x;t<n;t+=step)
        if(((const sb_u64*)counts)[t]!=0)
            ((sb_i64*)output)[((const sb_u64*)prefix)[t]]=(sb_i64)t;
}
extern "C" __global__ void sb_selection_numeric(
    sb_u64 source,sb_u64 indices,sb_u64 output,sb_u64 n,sb_u64 width){
    const sb_u64 step=(sb_u64)gridDim.x*blockDim.x;
    for(sb_u64 t=(sb_u64)blockIdx.x*blockDim.x+threadIdx.x;t<n;t+=step){
        const sb_u64 row=(sb_u64)((const sb_i64*)indices)[t];
        for(sb_u64 byte=0;byte<width;++byte)
            ((sb_u8*)output)[t*width+byte]=((const sb_u8*)source)[row*width+byte];
    }
}

// Saturating checked addition proves the positive-integer prefix cannot wrap.
// This counts selected ranges, not source rows, and allocates no per-byte index.
__device__ __forceinline__ void sb_selection_total(sb_u64* total,sb_u64* overflow,sb_u64 n){
    sb_u64 old=atomicAdd(total,0ull);
    for(;;){
        if(n>9223372036854775807ull-old){atomicExch(overflow,1ull);return;}
        const sb_u64 before=atomicCAS(total,old,old+n);
        if(before==old)return;
        old=before;
    }
}
extern "C" __global__ void sb_selection_ragged_lengths(
    sb_u64 offsets,sb_u64 values_len,sb_u64 indices,sb_u64 n,
    sb_u64 counts,sb_u64 control){
    const sb_u64 step=(sb_u64)gridDim.x*blockDim.x;
    const sb_i64* off=(const sb_i64*)offsets;
    sb_u64* status=(sb_u64*)control;
    for(sb_u64 t=(sb_u64)blockIdx.x*blockDim.x+threadIdx.x;t<n;t+=step){
        const sb_u64 row=(sb_u64)((const sb_i64*)indices)[t];
        const sb_i64 a=off[row],b=off[row+1];
        if(a<0||b<a||(sb_u64)b>values_len){
            ((sb_u64*)counts)[t]=0;atomicMin(status,t);continue;
        }
        const sb_u64 len=(sb_u64)(b-a);
        ((sb_u64*)counts)[t]=len;
        sb_selection_total(status+1,status+2,len);
    }
}
extern "C" __global__ void sb_selection_ragged_copy(
    sb_u64 source_offsets,sb_u64 source_values,sb_u64 source_valid,
    sb_u64 indices,sb_u64 offsets,sb_u64 values,sb_u64 valid,
    sb_u64 n,sb_u64 width,sb_u64 group_width){
    const sb_u64 groups=blockDim.x/group_width;
    const sb_u64 group=(sb_u64)blockIdx.x*groups+threadIdx.x/group_width;
    const sb_u64 lane=threadIdx.x%group_width;
    const sb_u64 step=(sb_u64)gridDim.x*groups;
    for(sb_u64 t=group;t<n;t+=step){
        const sb_u64 row=(sb_u64)((const sb_i64*)indices)[t];
        const sb_i64* src=(const sb_i64*)source_offsets;
        const sb_u64 first=(sb_u64)src[row],len=(sb_u64)(src[row+1]-src[row]);
        const sb_u64 destination=(sb_u64)((const sb_i64*)offsets)[t];
        if(width==8){
            for(sb_u64 at=lane;at<len;at+=group_width)
                ((sb_u64*)values)[destination+at]=((const sb_u64*)source_values)[first+at];
        }else{
            for(sb_u64 at=lane;at<len;at+=group_width)
                ((sb_u8*)values)[destination+at]=((const sb_u8*)source_values)[first+at];
        }
        if(lane==0&&valid)((sb_u8*)valid)[t]=source_valid?((const sb_u8*)source_valid)[row]:1;
    }
}
extern "C" __global__ void sb_selection_topk_count(
    sb_u64 indices_i,sb_u64 n,sb_u64 rows,sb_u64 counts,sb_u64 first_bad){
    const sb_i64* i=(const sb_i64*)indices_i;
    const sb_u64 step=(sb_u64)gridDim.x*blockDim.x;
    for(sb_u64 t=(sb_u64)blockIdx.x*blockDim.x+threadIdx.x;t<n;t+=step){
        const sb_i64 row=i[t];
        if(row<0||(sb_u64)row>=rows||(t&&row<i[t-1]))atomicMin((sb_u64*)first_bad,t);
        else atomicAdd((sb_u64*)counts+(sb_u64)row,1ull);
    }
}
extern "C" __global__ void sb_selection_topk_flags(
    sb_u64 counts,sb_u64 rows,sb_u64 k,sb_u64 flags,sb_u64 tied_rows){
    const sb_u64 step=(sb_u64)gridDim.x*blockDim.x;
    for(sb_u64 row=(sb_u64)blockIdx.x*blockDim.x+threadIdx.x;row<rows;row+=step){
        const bool tied=((const sb_u64*)counts)[row]>k;
        ((sb_u8*)flags)[row]=tied?1:0;
        if(tied)atomicAdd((sb_u64*)tied_rows,1ull);
    }
}

extern "C" __global__ void sb_selection_rowstats_counts(sb_u64 argbest,sb_u64 n,sb_u64 counts){
    const sb_u64 step=(sb_u64)gridDim.x*blockDim.x;
    for(sb_u64 row=(sb_u64)blockIdx.x*blockDim.x+threadIdx.x;row<n;row+=step)
        ((sb_u64*)counts)[row]=((const sb_i64*)argbest)[row]>=0;
}
extern "C" __global__ void sb_selection_rowstats_pairs(
    sb_u64 argbest,sb_u64 n,sb_u64 prefix,sb_u64 output_i,sb_u64 output_j){
    const sb_u64 step=(sb_u64)gridDim.x*blockDim.x;
    for(sb_u64 row=(sb_u64)blockIdx.x*blockDim.x+threadIdx.x;row<n;row+=step){
        const sb_i64 partner=((const sb_i64*)argbest)[row];
        if(partner>=0){const sb_u64 at=((const sb_u64*)prefix)[row];
            ((sb_i64*)output_i)[at]=(sb_i64)row;((sb_i64*)output_j)[at]=partner;}
    }
}
