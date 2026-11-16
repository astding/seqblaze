// Generic resident stages use the existing arithmetic kernels and bounded
// work chunks. Metadata scans and primary maps here are exact integer work.
#pragma once
#include "hamming.cuh"
#include "softf64.cuh"
extern "C" __global__ void sb_resident_hamming_counts(SbTables tables,SbSpace space,sb_u64 piece_bytes,sb_u64*counts) {
    const sb_u64 step=(sb_u64)gridDim.x*blockDim.x;
    for(sb_u64 t=(sb_u64)blockIdx.x*blockDim.x+threadIdx.x;t<space.n;t+=step) {
        sb_u64 i,j;sb_decode(space,t,i,j);
        const sb_u64 length=(sb_u64)sb_a_row(tables,i).length;
        counts[t]=(length+piece_bytes-1)/piece_bytes;
    }
}
// The resident Hamming pieces (per-call cost S6): the piece total is the
// device-resident end of the chunk's piece prefix (n + 1 entries), so the host
// never reads it and launches a grid from its own upper bound.
extern "C" __global__ void sb_resident_hamming_pieces(SbTables tables,SbSpace space,SbPieces pieces,int*dist) {
    pieces.total=((const sb_u64*)pieces.prefix)[space.n];
    sb_hamming_pieces_body(tables,space,pieces,dist);
}
// The same pieces counted by lane groups (sb_hamming_group_body): one piece
// per item (prefix 0, `total` given), or the device piece prefix with its
// total read in place, as sb_resident_hamming_pieces.
extern "C" __global__ void sb_resident_hamming_groups(SbTables tables,SbSpace space,SbPieces pieces,int*dist,unsigned group) {
    if(pieces.prefix!=0)pieces.total=((const sb_u64*)pieces.prefix)[space.n];
    sb_hamming_group_body(tables,space,pieces,dist,group);
}
// Device key-to-public-primary mapping. No pair numeric payload is downloaded.
// family0=distance,1=align,2=duplex-mfe,3=negativefloat tmkey,4=positivefloatpf/p3.
extern "C" __global__ void sb_resident_primary(const void*input,sb_u64 n,unsigned family,
                                                sb_u64*primary,int*integer) {
    const sb_u64 step=(sb_u64)gridDim.x*blockDim.x;
    for(sb_u64 t=(sb_u64)blockIdx.x*blockDim.x+threadIdx.x;t<n;t+=step) {
        if(family==1) {
            const int key=((const int*)input)[t];
            if(key==2147483647){if(primary)primary[t]=SB_F64_QNAN;if(integer)integer[t]=2147483647;}
            else {const int fx=key==2147483646?(-2147483647-1):-key;
                if(integer)integer[t]=fx;
                if(primary)primary[t]=key==2147483646?0xC7EFFFFFE0000000ull:sb_f64_div(sb_f64_from_i64(fx),0x408F400000000000ull);
            }
        } else if(family==2) {
            const int key=((const int*)input)[t];
            const int d=key==2147483646?2147483647:key;
            if(integer)integer[t]=d;
            if(primary)primary[t]=key==2147483647?SB_F64_QNAN:sb_f64_div(sb_f64_from_i64(d),0x4059000000000000ull);
        } else {
            const sb_i64 key=((const sb_i64*)input)[t];
            sb_u64 bits=SB_F64_QNAN;
            if(key!=0x7fffffffffffffffll && (family!=3 || key!=0x7ffffffffffffffell)) {
                const bool negative=family==3?key>0:key<0;
                const sb_u64 magnitude=key<0?0ull-(sb_u64)key:(sb_u64)key;
                bits=magnitude|(negative?SB_F64_SIGN:0);
            }
            if(primary)primary[t]=bits;
        }
    }
}

// A bounded view of a requested dense output, used only for oriented detail
// replay. At most the driver chunk's coordinates exist, never a whole matrix
// of indices beside the requested result payload.
extern "C" __global__ void sb_resident_dense_entries(SbSpace space,const void*keys,
    const sb_u8*passed,sb_u64 width,sb_u64 base,sb_u64*ids,sb_i64*left,sb_i64*right,
    void*outkeys,sb_u8*outpassed) {
    const sb_u64 step=(sb_u64)gridDim.x*blockDim.x;
    for(sb_u64 t=(sb_u64)blockIdx.x*blockDim.x+threadIdx.x;t<space.n;t+=step) {
        sb_u64 i,j;sb_decode(space,t,i,j);ids[t]=base+t;left[t]=(sb_i64)i;right[t]=(sb_i64)j;
        if(width==4)((int*)outkeys)[t]=((const int*)keys)[base+t];
        else ((sb_i64*)outkeys)[t]=((const sb_i64*)keys)[base+t];
        outpassed[t]=passed[base+t];
    }
}
