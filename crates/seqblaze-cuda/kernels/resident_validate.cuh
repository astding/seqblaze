// Shape validation over device row facts. Implicit pair domains require only
// O(n_a+n_b) candidates, never an expanded Cartesian list or a score kernel.
#pragma once
#include "device_interop.cuh"
#include "hamming.cuh"
extern "C" __global__ void sb_resident_first_rows(const SbInputFact*a,sb_u64 na,
    const SbInputFact*b,sb_u64 nb,unsigned mode,sb_u64*first) {
    const sb_u64 step=(sb_u64)gridDim.x*blockDim.x;
    for(sb_u64 r=(sb_u64)blockIdx.x*blockDim.x+threadIdx.x;r<na;r+=step)
        if(a[r].kind==0&&(mode==0||a[r].length>60))atomicMin(first,r);
    for(sb_u64 r=(sb_u64)blockIdx.x*blockDim.x+threadIdx.x;r<nb;r+=step)
        if(b[r].kind==0&&(mode==0||b[r].length>60))atomicMin(first+1,r);
}
__device__ __forceinline__ sb_u64 sb_resident_tri_prefix(sb_u64 row,sb_u64 n,bool self) {
    // A halved factor prevents overflow even near the platform pair limit.
    sb_u64 left=row,right=2*n-row-(self?0:1)+(self?1:0);
    if((left&1)==0)left/=2;else right/=2;
    return left*right;
}
extern "C" __global__ void sb_resident_pair_check(const SbInputFact*a,sb_u64 na,
    const SbInputFact*b,sb_u64 nb,SbSpace space,unsigned mode,unsigned self,
    const sb_u64*first,sb_u64*bad) {
    const sb_u64 step=(sb_u64)gridDim.x*blockDim.x;
    const sb_u64 fa=first[0],fb=first[1];
    const sb_u64 candidates=(space.kind==SB_ZIP||space.kind==SB_LIST)?space.n:
        (space.kind==SB_RECT?(mode==0?na+nb:1):na);
    for(sb_u64 t=(sb_u64)blockIdx.x*blockDim.x+threadIdx.x;t<candidates;t+=step) {
        sb_u64 i,j,k;
        if(space.kind==SB_ZIP||space.kind==SB_LIST) {sb_decode(space,t,i,j);k=space.k0+t;}
        else if(space.kind==SB_RECT) {
            if(fa==SB_INPUT_NONE||fb==SB_INPUT_NONE)continue;
            if(mode!=0){i=fa;j=fb;}
            else if(t<nb){i=fa;j=t;}else{i=t-nb;j=fb;}
            k=i*nb+j;
        } else {
            if(fa==SB_INPUT_NONE)continue;
            i=fa;j=t;
            if(j<i||(!self&&j==i))continue;
            if(space.kind==SB_BOTH)k=i*(na-(self?0:1))+j-((!self&&j>i)?1:0);
            else k=sb_resident_tri_prefix(i,na,self!=0)+j-i-(self?0:1);
        }
        if(a[i].kind!=0||b[j].kind!=0)continue;
        const bool fails=mode==0?a[i].length!=b[j].length:(a[i].length>60&&b[j].length>60);
        if(fails)atomicMin(bad,k);
    }
}
extern "C" __global__ void sb_resident_pair_error(const SbInputFact*a,const SbInputFact*b,
    SbSpace space,const sb_u64*bad,sb_u64*out) {
    const sb_u64 k=*bad;
    out[1]=out[2]=out[3]=out[4]=0;
    if(k==SB_INPUT_NONE){out[0]=k;return;}
    sb_u64 i,j;sb_decode(space,k-space.k0,i,j);
    out[0]=k;out[1]=i;out[2]=j;out[3]=a[i].length;out[4]=b[j].length;
}
