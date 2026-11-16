#ifndef SEQBLAZE_RESIDENT_TOPK_CUH
#define SEQBLAZE_RESIDENT_TOPK_CUH

// Pass-one integer TopK. The including header defines the 40-byte
// SbResidentEntry {u64 i,j; i64 key; u64 work_id,source;}.
// Input chunks are sorted by (i, signed key). Equal keys need no j order.
// Partners are already globally distinct, cutoff-filtered and mirror-expanded.
// Only one consumer stream accesses this state.
//
// Rows start empty. Before each chunk they grow to
// c=min(k,next_power_of_two(cumulative distinct passing partners)). With K
// the next power of two >=c, one workspace has c+(K-1)+2*K int64 elements:
// sorted base[c], sorted binary forest[K-1], carry[K], scratch[K]. c=0 uses 0.
// An append-only arena keeps old offsets stable during device-to-device pool
// growth. Geometric row growth bounds obsolete storage by a constant multiple
// of current storage. No implicit pair space allocates nrows*k in advance.
// Sparse input touches only its rows and occupied carry levels, never every
// row's k values. Large sorted row segments merge their first c values in
// parallel. Forest flushes amortize a base merge over at least c new keys.
//
// Each phase is prepare -> merge_jobs -> commit_jobs, in stream order.
// Zero job_count before prepare, job_cursor before EACH consumer. Launch the
// consumers with a fixed occupancy-sized grid, NOT queue-capacity-sized grids.
// Allocate tasks >=max(nrows,segment_capacity), and jobs >=
// max(nrows,segment_capacity)+floor(pool_elements/1024). No device allocation.

typedef unsigned long long SbRtU64;
typedef long long SbRtI64;
static_assert(sizeof(SbResidentEntry) == 40, "resident entry ABI");
static_assert(sizeof(SbRtU64) == 8 && sizeof(SbRtI64) == 8, "64-bit keys");

struct SbResidentTopKRow {
    SbRtU64 capacity, workspace, base_size, forest_mask, final_size;
};
struct SbResidentTopKSegment { SbRtU64 row, first, count, carry; };
// source: 0 workspace, 1 entry.key.
struct SbResidentTopKView { SbRtU64 source, offset, size; };
struct SbResidentTopKTask {
    SbResidentTopKView a, b;
    SbRtU64 size, target, scratch;
};
struct SbResidentTopKJob { SbRtU64 task, first; };
struct SbResidentTopKGrowth {
    SbResidentTopKRow old_row;
    SbRtU64 capacity, workspace;
};
static_assert(sizeof(SbResidentTopKRow) == 40, "resident TopK row ABI");
static_assert(sizeof(SbResidentTopKSegment) == 32, "resident TopK segment ABI");
static_assert(sizeof(SbResidentTopKView) == 24, "resident TopK view ABI");
static_assert(sizeof(SbResidentTopKTask) == 72, "resident TopK task ABI");
static_assert(sizeof(SbResidentTopKJob) == 16, "resident TopK job ABI");
static_assert(sizeof(SbResidentTopKGrowth) == 56, "resident TopK growth ABI");

enum { SB_RT_BAD_ROW=1, SB_RT_BAD_WORKSPACE=2, SB_RT_QUEUE_CAPACITY=3,
       SB_RT_BAD_STATE=4, SB_RT_TILE=1024, SB_RT_ITEMS=4 };

__device__ __forceinline__ void sb_rt_fail(SbRtU64* status, SbRtU64 error) {
    if (status) atomicCAS(status, 0ULL, error);
}
__device__ __forceinline__ bool sb_rt_failed(const SbRtU64* status) {
    return status && *status;
}
__device__ __forceinline__ SbRtU64 sb_rt_min(SbRtU64 a, SbRtU64 b) {
    return a < b ? a : b;
}
__device__ __forceinline__ bool sb_rt_size(
    SbRtU64 capacity, SbRtU64* power, SbRtU64* needed
) {
    if (!capacity) { *power=0; *needed=0; return true; }
    if (capacity > (1ULL<<62)) return false;
    SbRtU64 k=1;
    while (k<capacity) k<<=1;
    if (k>(~0ULL-(capacity-1))/3) return false;
    *power=k; *needed=(capacity-1)+3*k;
    return true;
}
__device__ __forceinline__ bool sb_rt_layout(
    const SbResidentTopKRow& row, SbRtU64 pool_elements, SbRtU64* power
) {
    if (!row.capacity) {
        *power=0;
        return !row.base_size && !row.forest_mask && !row.final_size;
    }
    SbRtU64 k,needed;
    if (!sb_rt_size(row.capacity,&k,&needed)) return false;
    if (row.workspace>pool_elements || needed>pool_elements-row.workspace) return false;
    if (row.base_size>row.capacity || row.final_size>=k) return false;
    if (row.forest_mask & ~(k-1)) return false;
    *power=k;
    return true;
}
__device__ __forceinline__ SbRtU64 sb_rt_carry(const SbResidentTopKRow& r, SbRtU64 k) {
    return r.workspace+r.capacity+k-1;
}
__device__ __forceinline__ SbRtU64 sb_rt_scratch(const SbResidentTopKRow& r, SbRtU64 k) {
    return sb_rt_carry(r,k)+k;
}
__device__ __forceinline__ SbResidentTopKView sb_rt_view(SbRtU64 s, SbRtU64 p, SbRtU64 n) {
    SbResidentTopKView v={s,p,n}; return v;
}
__device__ __forceinline__ SbRtI64 sb_rt_key(
    const SbResidentTopKView& v, SbRtU64 i,
    const SbRtI64* pool, const SbResidentEntry* entries
) {
    return v.source ? static_cast<SbRtI64>(entries[v.offset+i].key) : pool[v.offset+i];
}
__device__ __forceinline__ void sb_rt_enqueue(
    SbRtU64 id, SbResidentTopKView a, SbResidentTopKView b,
    SbRtU64 n, SbRtU64 target, SbRtU64 scratch,
    SbResidentTopKTask* tasks, SbRtU64 task_capacity,
    SbResidentTopKJob* jobs, SbRtU64 job_capacity, SbRtU64* job_count, SbRtU64* status
) {
    if (!n) return;
    if (id>=task_capacity || a.size>~0ULL-b.size || n>a.size+b.size) {
        sb_rt_fail(status,SB_RT_BAD_STATE); return;
    }
    const SbRtU64 count=1+(n-1)/SB_RT_TILE;
    const SbRtU64 first=atomicAdd(job_count,count);
    if (first>job_capacity || count>job_capacity-first) {
        sb_rt_fail(status,SB_RT_QUEUE_CAPACITY); return;
    }
    SbResidentTopKTask task={a,b,n,target,scratch}; tasks[id]=task;
    // Queue metadata only: never a serial row-key scan.
    for (SbRtU64 j=0;j<count;++j) {
        SbResidentTopKJob job={id,j*SB_RT_TILE}; jobs[first+j]=job;
    }
}

extern "C" __global__ void sb_resident_topk_init(
    const SbRtU64* capacities, const SbRtU64* workspace_offsets,
    SbRtU64 nrows, SbRtU64 pool_elements, SbResidentTopKRow* rows, SbRtU64* status
) {
    const SbRtU64 i=static_cast<SbRtU64>(blockIdx.x)*blockDim.x+threadIdx.x;
    if (i>=nrows || sb_rt_failed(status)) return;
    SbResidentTopKRow row={capacities[i],workspace_offsets[i],0,0,0}; SbRtU64 k;
    if (!sb_rt_layout(row,pool_elements,&k)) { sb_rt_fail(status,SB_RT_BAD_WORKSPACE); return; }
    rows[i]=row;
}

// Production initialization needs no per-row capacity or prefix arrays.
// The caller zeros the two arena/job counters and status separately.
extern "C" __global__ void sb_resident_topk_init_empty(
    SbResidentTopKRow* rows,SbRtU64 nrows,SbRtU64* seen
) {
    const SbRtU64 step=static_cast<SbRtU64>(gridDim.x)*blockDim.x;
    for (SbRtU64 i=static_cast<SbRtU64>(blockIdx.x)*blockDim.x+threadIdx.x;i<nrows;i+=step) {
        SbResidentTopKRow row={0,0,0,0,0}; rows[i]=row; seen[i]=0;
    }
}

extern "C" __global__ void sb_resident_topk_segments(
    const SbResidentEntry* entries, SbRtU64 count, SbRtU64 nrows,
    SbResidentTopKSegment* segments, SbRtU64 segment_capacity,
    SbRtU64* segment_count, SbRtU64* status
) {
    const SbRtU64 i=static_cast<SbRtU64>(blockIdx.x)*blockDim.x+threadIdx.x;
    if (i>=count || sb_rt_failed(status)) return;
    const SbRtU64 row=static_cast<SbRtU64>(entries[i].i);
    if (row>=nrows) { sb_rt_fail(status,SB_RT_BAD_ROW); return; }
    if (i && entries[i-1].i==entries[i].i) return;
    SbRtU64 lo=i+1,hi=count;
    while (lo<hi) {
        const SbRtU64 m=lo+(hi-lo)/2;
        if (entries[m].i==entries[i].i) lo=m+1; else hi=m;
    }
    const SbRtU64 out=atomicAdd(segment_count,1ULL);
    if (out>=segment_capacity) { sb_rt_fail(status,SB_RT_QUEUE_CAPACITY); return; }
    SbResidentTopKSegment segment={row,i,lo-i,0}; segments[out]=segment;
}

__device__ __forceinline__ bool sb_rt_arena_reserve(
    SbRtU64* arena_used,SbRtU64 needed,SbRtU64* workspace,SbRtU64* status
) {
    SbRtU64 old=atomicAdd(arena_used,0ULL);
    while (true) {
        if (needed>~0ULL-old) { sb_rt_fail(status,SB_RT_BAD_WORKSPACE); return false; }
        const SbRtU64 previous=atomicCAS(arena_used,old,old+needed);
        if (previous==old) { *workspace=old; return true; }
        old=previous;
    }
}

// Run after segments and before chunk_large. This reads geometry only, never
// pool keys. arena_used is append-only across chunks. After preparation, the
// caller reads only arena_used/status, geometrically grows the pool if needed,
// and preserves the old allocation by D2D before running growth_copy_jobs.
// Until growth_commit the rows still describe the OLD allocation.
extern "C" __global__ void sb_resident_topk_growth_prepare(
    const SbResidentTopKRow* rows,SbRtU64 nrows,SbRtU64 old_pool_elements,
    const SbResidentTopKSegment* segments,const SbRtU64* segment_count,SbRtU64 segment_capacity,
    SbRtU64 k,SbRtU64* seen,SbRtU64* arena_used,
    SbResidentTopKGrowth* growth,SbRtU64 growth_capacity,
    SbResidentTopKJob* jobs,SbRtU64 job_capacity,SbRtU64* job_count,SbRtU64* status
) {
    const SbRtU64 i=static_cast<SbRtU64>(blockIdx.x)*blockDim.x+threadIdx.x;
    if (i>=segment_capacity || i>=*segment_count || sb_rt_failed(status)) return;
    if (!k || i>=growth_capacity) { sb_rt_fail(status,SB_RT_BAD_STATE); return; }
    const SbResidentTopKSegment s=segments[i];
    if (s.row>=nrows) { sb_rt_fail(status,SB_RT_BAD_ROW); return; }
    const SbResidentTopKRow old=rows[s.row]; SbRtU64 old_power;
    if (!sb_rt_layout(old,old_pool_elements,&old_power) || old.final_size) {
        sb_rt_fail(status,SB_RT_BAD_STATE); return;
    }
    const SbRtU64 previous_seen=seen[s.row];
    if (s.count>~0ULL-previous_seen) { sb_rt_fail(status,SB_RT_BAD_STATE); return; }
    const SbRtU64 current_seen=previous_seen+s.count;
    SbRtU64 capacity=k;
    if (current_seen<k) {
        capacity=current_seen?1:0;
        while (capacity<current_seen) {
            if (capacity>(~0ULL>>1)) { sb_rt_fail(status,SB_RT_BAD_WORKSPACE); return; }
            capacity<<=1;
        }
        capacity=sb_rt_min(k,capacity);
    }
    if (capacity<old.capacity) { sb_rt_fail(status,SB_RT_BAD_STATE); return; }
    SbResidentTopKGrowth plan={old,0,0};
    if (capacity>old.capacity) {
        SbRtU64 power,needed;
        if (!sb_rt_size(capacity,&power,&needed)) {
            sb_rt_fail(status,SB_RT_BAD_WORKSPACE); return;
        }
        if (!sb_rt_arena_reserve(arena_used,needed,&plan.workspace,status)) return;
        plan.capacity=capacity;
        // Cover base+forest geometry, but consumers copy only initialized
        // base positions and occupied forest runs; holes are never read.
        const SbRtU64 extent=old.capacity?old.capacity+old_power-1:0;
        const SbRtU64 count=extent?1+(extent-1)/SB_RT_TILE:0;
        const SbRtU64 first=atomicAdd(job_count,count);
        if (first>job_capacity || count>job_capacity-first) {
            sb_rt_fail(status,SB_RT_QUEUE_CAPACITY); return;
        }
        for (SbRtU64 j=0;j<count;++j) {
            SbResidentTopKJob job={i,j*SB_RT_TILE}; jobs[first+j]=job;
        }
    }
    growth[i]=plan; seen[s.row]=current_seen;
}

// Same occupancy-sized persistent CTA queue as merge_jobs. pool points to the
// preserved/grown arena. New row regions are appended beyond every old row,
// so copies do not overlap and require no scratch/commit data pass.
extern "C" __global__ void sb_resident_topk_growth_copy_jobs(
    SbRtI64* pool,const SbResidentTopKGrowth* growth,const SbResidentTopKJob* jobs,
    const SbRtU64* job_count,SbRtU64* job_cursor,const SbRtU64* status
) {
    if (sb_rt_failed(status)) return;
    __shared__ SbRtU64 ticket;
    while (true) {
        if (!threadIdx.x) ticket=atomicAdd(job_cursor,1ULL);
        __syncthreads();
        if (ticket>=*job_count) break;
        const SbResidentTopKJob job=jobs[ticket];
        const SbResidentTopKGrowth plan=growth[job.task];
        const SbResidentTopKRow old=plan.old_row;
        SbRtU64 power=1;
        while (power<old.capacity) power<<=1;
        const SbRtU64 end=sb_rt_min(old.capacity+power-1,job.first+SB_RT_TILE);
        for (SbRtU64 at=job.first+threadIdx.x;at<end;at+=blockDim.x) {
            if (at<old.capacity) {
                if (at<old.base_size) pool[plan.workspace+at]=pool[old.workspace+at];
            } else {
                const SbRtU64 position=at-old.capacity;
                const SbRtU64 run=1ULL<<(63-__clzll(position+1));
                if (old.forest_mask&run)
                    pool[plan.workspace+plan.capacity+position]=pool[old.workspace+at];
            }
        }
        __syncthreads();
    }
}

extern "C" __global__ void sb_resident_topk_growth_commit(
    SbResidentTopKRow* rows,SbRtU64 nrows,SbRtU64 pool_elements,
    const SbResidentTopKSegment* segments,const SbRtU64* segment_count,SbRtU64 segment_capacity,
    const SbResidentTopKGrowth* growth,SbRtU64 growth_capacity,SbRtU64* status
) {
    const SbRtU64 i=static_cast<SbRtU64>(blockIdx.x)*blockDim.x+threadIdx.x;
    if (i>=segment_capacity || i>=*segment_count || sb_rt_failed(status)) return;
    if (i>=growth_capacity) { sb_rt_fail(status,SB_RT_BAD_STATE); return; }
    const SbResidentTopKGrowth plan=growth[i];
    if (!plan.capacity) return;
    const SbRtU64 row=segments[i].row;
    if (row>=nrows) { sb_rt_fail(status,SB_RT_BAD_ROW); return; }
    SbResidentTopKRow updated=plan.old_row;
    updated.capacity=plan.capacity; updated.workspace=plan.workspace; SbRtU64 power;
    if (!sb_rt_layout(updated,pool_elements,&power)) {
        sb_rt_fail(status,SB_RT_BAD_WORKSPACE); return;
    }
    rows[row]=updated;
}

// Chunk preparation launches over segment_capacity; segment_count is a device
// scalar. Exactly one segment owns each touched row in this chunk.
extern "C" __global__ void sb_resident_topk_chunk_large(
    SbResidentTopKRow* rows, SbRtU64 nrows, SbRtU64 pool_elements,
    SbResidentTopKSegment* segments, const SbRtU64* segment_count, SbRtU64 segment_capacity,
    SbResidentTopKTask* tasks, SbRtU64 task_capacity,
    SbResidentTopKJob* jobs, SbRtU64 job_capacity, SbRtU64* job_count, SbRtU64* status
) {
    const SbRtU64 i=static_cast<SbRtU64>(blockIdx.x)*blockDim.x+threadIdx.x;
    if (i>=segment_capacity || i>=*segment_count || sb_rt_failed(status)) return;
    SbResidentTopKSegment& s=segments[i];
    if (s.row>=nrows) { sb_rt_fail(status,SB_RT_BAD_ROW); return; }
    SbResidentTopKRow& r=rows[s.row]; SbRtU64 k;
    if (!sb_rt_layout(r,pool_elements,&k) || !r.capacity) {
        sb_rt_fail(status,SB_RT_BAD_WORKSPACE); return;
    }
    s.carry=0;
    if (s.count<r.capacity) return;
    sb_rt_enqueue(i,sb_rt_view(0,r.workspace,r.base_size),sb_rt_view(1,s.first,r.capacity),
        r.capacity,r.workspace,sb_rt_scratch(r,k),tasks,task_capacity,jobs,job_capacity,job_count,status);
    r.base_size=r.capacity;
}

extern "C" __global__ void sb_resident_topk_chunk_level(
    unsigned int level,
    SbResidentTopKRow* rows, SbRtU64 nrows, SbRtU64 pool_elements,
    SbResidentTopKSegment* segments, const SbRtU64* segment_count, SbRtU64 segment_capacity,
    SbResidentTopKTask* tasks, SbRtU64 task_capacity,
    SbResidentTopKJob* jobs, SbRtU64 job_capacity, SbRtU64* job_count, SbRtU64* status
) {
    const SbRtU64 i=static_cast<SbRtU64>(blockIdx.x)*blockDim.x+threadIdx.x;
    if (i>=segment_capacity || i>=*segment_count || sb_rt_failed(status)) return;
    SbResidentTopKSegment& s=segments[i];
    if (s.row>=nrows) { sb_rt_fail(status,SB_RT_BAD_ROW); return; }
    SbResidentTopKRow& r=rows[s.row]; SbRtU64 k;
    if (!sb_rt_layout(r,pool_elements,&k)) { sb_rt_fail(status,SB_RT_BAD_WORKSPACE); return; }
    if (s.count>=r.capacity || level>=63) return;
    const SbRtU64 w=1ULL<<level;
    if (w>=k) return;
    const bool old=(r.forest_mask&w)!=0, fresh=(s.count&w)!=0, carry=s.carry!=0;
    const unsigned int inputs=static_cast<unsigned int>(old)+static_cast<unsigned int>(fresh)
                              +static_cast<unsigned int>(carry);
    const SbRtU64 slot=r.workspace+r.capacity+w-1;
    const SbResidentTopKView previous=sb_rt_view(0,slot,w);
    const SbResidentTopKView incoming=sb_rt_view(1,s.first+(s.count&(w-1)),w);
    const SbResidentTopKView pending=sb_rt_view(0,sb_rt_carry(r,k),w);
    if (inputs<=1) {
        s.carry=0;
        if (old || !inputs) return;
        sb_rt_enqueue(i,sb_rt_view(0,0,0),fresh?incoming:pending,w,slot,sb_rt_scratch(r,k),
            tasks,task_capacity,jobs,job_capacity,job_count,status);
        r.forest_mask|=w;
        return;
    }
    SbResidentTopKView a,b;
    if (inputs==3) { a=incoming; b=pending; } // Keep the old run.
    else if (old) { a=previous; b=fresh?incoming:pending; r.forest_mask&=~w; }
    else { a=incoming; b=pending; }
    sb_rt_enqueue(i,a,b,2*w,sb_rt_carry(r,k),sb_rt_scratch(r,k),
        tasks,task_capacity,jobs,job_capacity,job_count,status);
    s.carry=1;
}

extern "C" __global__ void sb_resident_topk_chunk_flush(
    SbResidentTopKRow* rows, SbRtU64 nrows, SbRtU64 pool_elements,
    SbResidentTopKSegment* segments, const SbRtU64* segment_count, SbRtU64 segment_capacity,
    SbResidentTopKTask* tasks, SbRtU64 task_capacity,
    SbResidentTopKJob* jobs, SbRtU64 job_capacity, SbRtU64* job_count, SbRtU64* status
) {
    const SbRtU64 i=static_cast<SbRtU64>(blockIdx.x)*blockDim.x+threadIdx.x;
    if (i>=segment_capacity || i>=*segment_count || sb_rt_failed(status)) return;
    SbResidentTopKSegment& s=segments[i];
    if (!s.carry) return;
    if (s.row>=nrows) { sb_rt_fail(status,SB_RT_BAD_ROW); return; }
    SbResidentTopKRow& r=rows[s.row]; SbRtU64 k;
    if (!sb_rt_layout(r,pool_elements,&k) || !r.capacity) {
        sb_rt_fail(status,SB_RT_BAD_WORKSPACE); return;
    }
    sb_rt_enqueue(i,sb_rt_view(0,r.workspace,r.base_size),sb_rt_view(0,sb_rt_carry(r,k),k),
        r.capacity,r.workspace,sb_rt_scratch(r,k),tasks,task_capacity,jobs,job_capacity,job_count,status);
    r.base_size=r.capacity; s.carry=0;
}

extern "C" __global__ void sb_resident_topk_final_begin(SbResidentTopKRow* rows,SbRtU64 nrows) {
    const SbRtU64 i=static_cast<SbRtU64>(blockIdx.x)*blockDim.x+threadIdx.x;
    if (i<nrows) rows[i].final_size=0;
}

// Ascending final levels accumulate forest in carry, then one base merge.
// Geometric run sizes bound final data movement by O(capacity), not O(k log k).
extern "C" __global__ void sb_resident_topk_final_level(
    unsigned int level,SbResidentTopKRow* rows,SbRtU64 nrows,SbRtU64 pool_elements,
    SbResidentTopKTask* tasks,SbRtU64 task_capacity,
    SbResidentTopKJob* jobs,SbRtU64 job_capacity,SbRtU64* job_count,SbRtU64* status
) {
    const SbRtU64 i=static_cast<SbRtU64>(blockIdx.x)*blockDim.x+threadIdx.x;
    if (i>=nrows || level>=63 || sb_rt_failed(status)) return;
    SbResidentTopKRow& r=rows[i]; SbRtU64 k;
    if (!sb_rt_layout(r,pool_elements,&k)) { sb_rt_fail(status,SB_RT_BAD_WORKSPACE); return; }
    const SbRtU64 w=1ULL<<level;
    if (w>=k || !(r.forest_mask&w)) return;
    if (r.final_size>=k-w) { sb_rt_fail(status,SB_RT_BAD_STATE); return; }
    const SbRtU64 size=r.final_size+w;
    sb_rt_enqueue(i,sb_rt_view(0,sb_rt_carry(r,k),r.final_size),
        sb_rt_view(0,r.workspace+r.capacity+w-1,w),size,sb_rt_carry(r,k),sb_rt_scratch(r,k),
        tasks,task_capacity,jobs,job_capacity,job_count,status);
    r.final_size=size; r.forest_mask&=~w;
}

extern "C" __global__ void sb_resident_topk_final_flush(
    SbResidentTopKRow* rows,SbRtU64 nrows,SbRtU64 pool_elements,
    SbResidentTopKTask* tasks,SbRtU64 task_capacity,
    SbResidentTopKJob* jobs,SbRtU64 job_capacity,SbRtU64* job_count,SbRtU64* status
) {
    const SbRtU64 i=static_cast<SbRtU64>(blockIdx.x)*blockDim.x+threadIdx.x;
    if (i>=nrows || sb_rt_failed(status)) return;
    SbResidentTopKRow& r=rows[i]; SbRtU64 k;
    if (!sb_rt_layout(r,pool_elements,&k)) { sb_rt_fail(status,SB_RT_BAD_WORKSPACE); return; }
    if (!r.final_size) return;
    const SbRtU64 size=sb_rt_min(r.capacity,r.base_size+r.final_size);
    sb_rt_enqueue(i,sb_rt_view(0,r.workspace,r.base_size),sb_rt_view(0,sb_rt_carry(r,k),r.final_size),
        size,r.workspace,sb_rt_scratch(r,k),tasks,task_capacity,jobs,job_capacity,job_count,status);
    r.base_size=size; r.final_size=0;
}

// Merge-path diagonal. Per-thread runs of four amortize the binary search.
__device__ __forceinline__ SbRtU64 sb_rt_partition(
    const SbResidentTopKTask& task,SbRtU64 diagonal,
    const SbRtI64* pool,const SbResidentEntry* entries
) {
    SbRtU64 lo=diagonal>task.b.size?diagonal-task.b.size:0;
    SbRtU64 hi=sb_rt_min(diagonal,task.a.size);
    while (lo<hi) {
        const SbRtU64 a=lo+(hi-lo)/2,b=diagonal-a;
        if (a<task.a.size && b && sb_rt_key(task.a,a,pool,entries)<=sb_rt_key(task.b,b-1,pool,entries)) lo=a+1;
        else hi=a;
    }
    return lo;
}

extern "C" __global__ void sb_resident_topk_merge_jobs(
    const SbResidentEntry* entries,SbRtI64* pool,
    const SbResidentTopKTask* tasks,const SbResidentTopKJob* jobs,
    const SbRtU64* job_count,SbRtU64* job_cursor,const SbRtU64* status
) {
    if (sb_rt_failed(status)) return;
    __shared__ SbRtU64 ticket;
    while (true) {
        if (!threadIdx.x) ticket=atomicAdd(job_cursor,1ULL);
        __syncthreads();
        if (ticket>=*job_count) break;
        const SbResidentTopKJob job=jobs[ticket];
        const SbResidentTopKTask task=tasks[job.task];
        const SbRtU64 end=sb_rt_min(task.size,job.first+SB_RT_TILE);
        for (SbRtU64 first=job.first+static_cast<SbRtU64>(threadIdx.x)*SB_RT_ITEMS;first<end;
             first+=static_cast<SbRtU64>(blockDim.x)*SB_RT_ITEMS) {
            SbRtU64 a=sb_rt_partition(task,first,pool,entries),b=first-a;
            const SbRtU64 stop=sb_rt_min(end,first+SB_RT_ITEMS);
            for (SbRtU64 out=first;out<stop;++out) {
                const bool take_a=a<task.a.size && (b==task.b.size
                    || sb_rt_key(task.a,a,pool,entries)<=sb_rt_key(task.b,b,pool,entries));
                pool[task.scratch+out]=take_a?sb_rt_key(task.a,a++,pool,entries):sb_rt_key(task.b,b++,pool,entries);
            }
        }
        __syncthreads();
    }
}

extern "C" __global__ void sb_resident_topk_commit_jobs(
    SbRtI64* pool,const SbResidentTopKTask* tasks,const SbResidentTopKJob* jobs,
    const SbRtU64* job_count,SbRtU64* job_cursor,const SbRtU64* status
) {
    if (sb_rt_failed(status)) return;
    __shared__ SbRtU64 ticket;
    while (true) {
        if (!threadIdx.x) ticket=atomicAdd(job_cursor,1ULL);
        __syncthreads();
        if (ticket>=*job_count) break;
        const SbResidentTopKJob job=jobs[ticket];
        const SbResidentTopKTask task=tasks[job.task];
        const SbRtU64 end=sb_rt_min(task.size,job.first+SB_RT_TILE);
        for (SbRtU64 out=job.first+threadIdx.x;out<end;out+=blockDim.x) pool[task.target+out]=pool[task.scratch+out];
        __syncthreads();
    }
}

extern "C" __global__ void sb_resident_topk_thresholds(
    const SbResidentTopKRow* rows,SbRtU64 nrows,const SbRtI64* pool,SbRtU64 pool_elements,
    SbRtI64* thresholds,unsigned char* active,SbRtU64* status
) {
    const SbRtU64 i=static_cast<SbRtU64>(blockIdx.x)*blockDim.x+threadIdx.x;
    if (i>=nrows || sb_rt_failed(status)) return;
    const SbResidentTopKRow r=rows[i]; SbRtU64 k;
    if (!sb_rt_layout(r,pool_elements,&k) || r.forest_mask || r.final_size) {
        sb_rt_fail(status,SB_RT_BAD_STATE); return;
    }
    active[i]=static_cast<unsigned char>(r.base_size!=0);
    thresholds[i]=r.base_size?pool[r.workspace+r.base_size-1]:static_cast<SbRtI64>(0x7FFFFFFFFFFFFFFFULL);
}

#endif
