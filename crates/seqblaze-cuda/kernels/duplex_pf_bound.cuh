// The C2 bound of K3 `duplex(model="pf")` on the device: Z-hat_b, the relaxed F53
// recurrence of one pair per thread, its lower bound L on K3's computed dg_ens_kcal and
// the outcome against a threshold. Spec: docs/wiki/thermo/duplex-pf-bound.md §3.5 (the
// binding fold orders), §4.4 (L), §1 (the outcome) and §8. Written from that page only.
//
// Integers only: F53 values (f53.cuh) for Z-hat_b, integer soft-binary64 (softf64.cuh)
// for L with the resident finalize's G1-G8 map (pf_resident.cuh). One thread evaluates a
// whole pair, so every fold runs in one lane in the page's order and the grid only
// changes which thread evaluates which pair (QZ10).
#pragma once
#include "duplex_pf.cuh"
#include "pf_resident.cuh"

#define SB_PFB_U 4
#define SB_PFB_REGIONS (2 * SB_PFB_U + 2)
// Words of one relaxed class: class, x (M, e), z (M, e), then per region (used, M, e).
#define SB_PFB_CLASS_WORDS (5 + 3 * SB_PFB_REGIONS)
// Rows of each accumulator ring: the queries of row i read row i - U - 2.
#define SB_PFB_ACC_ROWS (SB_PFB_U + 3)
#define SB_PFB_MAX_CLASSES 3
// Staged letter codes ahead of each thread's bands: 400 bytes = 25 F53 values.
#define SB_PFB_CODE_VALUES 25ull

#define SB_PFB_OPEN 0u
#define SB_PFB_CERTIFIED 1u
#define SB_PFB_PRUNED 2u

// Mirrors `SbPfBound` in src/pfbound.rs; every field is 8 bytes wide. Table offsets index
// the concatenated weight entries (`w`) of K3's device pack, as `SbPf`'s.
struct SbPfBound {
    sb_u64 w;          // const SbF53*: every weight entry
    sb_u64 lam;
    sb_u64 n_p;
    sb_u64 n_c;
    sb_u64 cls;        // const sb_u8*: CLS over (Λ + 1)²
    sb_u64 start;      // START, sigma = 0 (sigma = 1 follows at +25·N_P)
    sb_u64 end;
    sb_u64 stack;
    sb_u64 bulge;
    sb_u64 i11;
    sb_u64 i12;
    sb_u64 i21;
    sb_u64 i22;
    sb_u64 gen;
    sb_u64 loop_out;
    sb_u64 loop_in;
    sb_u64 relaxed;    // const sb_i64*: SB_PFB_CLASS_WORDS per relaxed class
    sb_u64 n_relaxed;
    sb_u64 scale_down; // testing fault zb_scale_down: a nonzero Z-hat_b times 2^-64
    // The testing evaluation (sb_pf_bound_eval) only.
    sb_u64 codes;      // const sb_u8*: letter codes of every strand
    sb_u64 a_start;    // const sb_u64* per item
    sb_u64 a_len;
    sb_u64 b_start;
    sb_u64 b_len;
    sb_u64 n_items;
    sb_u64 scratch;    // SbF53*: `stride` values per thread
    sb_u64 stride;
    sb_u64 has_theta;
    sb_i64 theta;
    sb_u64 out_m;      // sb_u64* per item: Z-hat_b as (M, e)
    sb_u64 out_e;      // sb_i64*
    sb_u64 out_l;      // sb_u64*: the bits of L
    sb_u64 out_code;   // sb_u8*: the outcome code
};

// One relaxed class, unpacked from its words.
struct SbPfbClass {
    int cls;
    bool need_h;
    bool need_v;
    SbF53 x;
    SbF53 z;
    bool used[SB_PFB_REGIONS];
    SbF53 omega[SB_PFB_REGIONS];
};

__device__ __forceinline__ void sb_pfb_load(const SbPfBound& p, SbPfbClass* rel) {
    const sb_i64* words = (const sb_i64*)p.relaxed;
    for (sb_u64 t = 0; t < p.n_relaxed; ++t) {
        const sb_i64* c = words + t * SB_PFB_CLASS_WORDS;
        SbPfbClass& r = rel[t];
        r.cls = (int)c[0];
        r.x = sb_f53_make((sb_u64)c[1], c[2]);
        r.z = sb_f53_make((sb_u64)c[3], c[4]);
        r.need_h = false;
        r.need_v = false;
        for (int k = 0; k < SB_PFB_REGIONS; ++k) {
            r.used[k] = c[5 + 3 * k] != 0;
            r.omega[k] = sb_f53_make((sb_u64)c[6 + 3 * k], c[7 + 3 * k]);
            if (r.used[k]) {
                // T_1..T_U and LO read H; B_1..B_U and HI read V.
                const bool vertical = (k >= SB_PFB_U && k < 2 * SB_PFB_U) || k == 2 * SB_PFB_U + 1;
                if (vertical) {
                    r.need_v = true;
                } else {
                    r.need_h = true;
                }
            }
        }
    }
}

// x (+) (w (*) y), skipping a zero y (adding 0 is exact).
__device__ __forceinline__ SbF53 sb_pfb_fma(const SbF53& x, const SbF53& w, const SbF53& y) {
    return y.m == 0ull ? x : sb_f53_add(x, sb_f53_mul(w, y));
}

// Values of one thread's bands for strands of at most `longest` letters: RA of
// R = min(longest, Λ + 2) rows, RO of min(longest, U + 2) rows of N_C values, four
// accumulator rings of U + 3 rows per relaxed class, then the column sums.
__device__ __host__ __forceinline__ sb_u64 sb_pfb_values(sb_u64 longest, sb_u64 lam, sb_u64 nc,
                                                         sb_u64 n_relaxed) {
    const sb_u64 rows = longest < lam + 2 ? longest : lam + 2;
    const sb_u64 ro_rows = longest < SB_PFB_U + 2 ? longest : SB_PFB_U + 2;
    return longest * (rows + ro_rows * nc + n_relaxed * 4 * SB_PFB_ACC_ROWS + 1);
}

// Z-hat_b of §3.5 for the canonical top strand `a` (n letters) and bottom strand `b`
// (m letters), codes 0..3, with `sigma` K1's symmetry flag. `s`: the thread's bands.
__device__ SbF53 sb_pfb_zhat(const SbPfBound& p, const SbPfbClass* rel, const sb_u8* a, int n,
                             const sb_u8* b, int m, int sigma, SbF53* s) {
    if (n < 2 || m < 2) {
        return sb_f53_zero();
    }
    const SbF53* w = (const SbF53*)p.w;
    const sb_u8* cls = (const sb_u8*)p.cls;
    const int lam = (int)p.lam;
    const int np = (int)p.n_p;
    const int nc = (int)p.n_c;
    const int nrel = (int)p.n_relaxed;
    const int width = lam + 1;
    const int rows = n < lam + 2 ? n : lam + 2;
    const int ro_rows = n < SB_PFB_U + 2 ? n : SB_PFB_U + 2;
    const int plane = SB_PFB_ACC_ROWS * m;
    SbF53* ra = s;
    SbF53* ro = ra + rows * m;
    SbF53* acc = ro + ro_rows * m * nc;
    SbF53* col = acc + nrel * 4 * plane;
    for (int c = 0; c < m; ++c) {
        col[c] = sb_f53_zero();
    }
    const sb_u64 start = p.start + (sb_u64)sigma * 25 * (sb_u64)np;
    int cur = 0;
    int cur_ro = 0;
    for (int i = 0; i < n; ++i) {
        const int ai = a[i];
        SbF53* ra_here = ra + cur * m;
        SbF53* ro_here = ro + cur_ro * m * nc;
        // Accumulator slots: row r lives at r mod (U + 3); row i - U - 2 at (i + 1) mod.
        const int query_slot = (i + 1) % SB_PFB_ACC_ROWS;
        for (int j = 0; j < m; ++j) {
            const int pi = sb_dup_pt(ai, b[j], (sb_u64)np);
            if (pi < 0) {
                ra_here[j] = sb_f53_zero();
                for (int c = 0; c < nc; ++c) {
                    ro_here[j * nc + c] = sb_f53_zero();
                }
                continue;
            }
            SbF53 special = sb_f53_zero();
            if (i >= 1 && j + 1 < m) {
                SbF53 generic[SB_PFB_MAX_CLASSES];
                for (int c = 0; c < SB_PFB_MAX_CLASSES; ++c) {
                    generic[c] = sb_f53_zero();
                }
                const int umax = lam < i - 1 ? lam : i - 1;
                const int room = m - 2 - j;
                int slot = cur == 0 ? rows - 1 : cur - 1;
                int slot_ro = cur_ro == 0 ? ro_rows - 1 : cur_ro - 1;
                for (int u = 0; u <= umax; ++u) {
                    const int k = i - 1 - u;
                    const int ak = a[k];
                    const SbF53* row = ra + slot * m;
                    // v = 0: a stack or a top bulge (every size is exact).
                    const SbF53 first = row[j + 1];
                    if (first.m != 0ull) {
                        const int pk = sb_dup_pk(ak, b[j + 1], (sb_u64)np);
                        const sb_u64 at =
                            u == 0 ? p.stack + (sb_u64)(pk * np + pi)
                                   : p.bulge + (sb_u64)(((u - 1) * np + pk) * np + pi);
                        special = sb_f53_add(special, sb_f53_mul(first, sb_pf_w(w, at)));
                    }
                    int vmax;
                    if (u == 0) {
                        vmax = lam < room ? lam : room;
                    } else if (u <= SB_PFB_U) {
                        vmax = SB_PFB_U < lam - u ? SB_PFB_U : lam - u;
                        vmax = vmax < room ? vmax : room;
                    } else {
                        vmax = 0;
                    }
                    for (int v = 1; v <= vmax; ++v) {
                        const int l = j + 1 + v;
                        if (u == 0) {
                            const SbF53 prev = row[l];
                            if (prev.m != 0ull) {
                                const int pk = sb_dup_pk(ak, b[l], (sb_u64)np);
                                const sb_u64 at =
                                    p.bulge + (sb_u64)(((lam + v - 1) * np + pk) * np + pi);
                                special = sb_f53_add(special, sb_f53_mul(prev, sb_pf_w(w, at)));
                            }
                        } else if (u <= 2 && v <= 2) {
                            const SbF53 prev = row[l];
                            if (prev.m == 0ull) {
                                continue;
                            }
                            const int pk = sb_dup_pk(ak, b[l], (sb_u64)np);
                            const int p1 = a[k + 1];
                            const int q1 = b[l - 1];
                            sb_u64 at;
                            if (u == 1 && v == 1) {
                                at = p.i11 + (sb_u64)(((pk * 4 + p1) * 4 + q1) * np + pi);
                            } else if (u == 1) {
                                const int q2 = b[l - 2];
                                at = p.i12 + (sb_u64)((((pk * 4 + p1) * 4 + q1) * 4 + q2) * np + pi);
                            } else if (v == 1) {
                                const int p2 = a[k + 2];
                                at = p.i21 + (sb_u64)((((pk * 4 + p1) * 4 + p2) * 4 + q1) * np + pi);
                            } else {
                                const int p2 = a[k + 2];
                                const int q2 = b[l - 2];
                                at = p.i22 +
                                     (sb_u64)(((((pk * 4 + p1) * 4 + p2) * 4 + q1) * 4 + q2) * np +
                                              pi);
                            }
                            special = sb_f53_add(special, sb_f53_mul(prev, sb_pf_w(w, at)));
                        } else {
                            // A generic window shape (1 <= u, v <= U): K3's term with RO.
                            const int shape = u * width + v;
                            const int cl = (int)__ldg(cls + shape);
                            const SbF53 prev = ro[(slot_ro * m + l) * nc + cl];
                            if (prev.m != 0ull) {
                                generic[cl] = sb_f53_add(
                                    generic[cl], sb_f53_mul(prev, sb_pf_w(w, p.gen + (sb_u64)shape)));
                            }
                        }
                    }
                    slot = slot == 0 ? rows - 1 : slot - 1;
                    slot_ro = slot_ro == 0 ? ro_rows - 1 : slot_ro - 1;
                }
                // Region terms (§3.4 queries), appended to their class's generic fold.
                for (int t = 0; t < nrel; ++t) {
                    const SbPfbClass& r = rel[t];
                    const SbF53* h = acc + t * 4 * plane;
                    const SbF53* v_acc = h + plane;
                    const SbF53* hd = v_acc + plane;
                    const SbF53* vd = hd + plane;
                    SbF53 g = generic[r.cls];
                    if (i > SB_PFB_U + 1) {
                        const int qrow = query_slot * m;
                        for (int v = 0; v < SB_PFB_U; ++v) {
                            if (r.used[v]) {
                                const int c = j + 2 + v;
                                if (c < m) {
                                    g = sb_pfb_fma(g, r.omega[v], h[qrow + c]);
                                }
                            }
                        }
                    }
                    const int col_b = j + SB_PFB_U + 2;
                    if (col_b < m) {
                        for (int u = 0; u < SB_PFB_U; ++u) {
                            if (r.used[SB_PFB_U + u] && i >= u + 2) {
                                // Row i - 2 - u (u is the index - 1).
                                const int brow = ((i - 2 - u) % SB_PFB_ACC_ROWS) * m;
                                g = sb_pfb_fma(g, r.omega[SB_PFB_U + u], v_acc[brow + col_b]);
                            }
                        }
                    }
                    if (i > SB_PFB_U + 1) {
                        const int qrow = query_slot * m;
                        if (r.used[2 * SB_PFB_U] && col_b < m) {
                            g = sb_pfb_fma(g, r.omega[2 * SB_PFB_U], hd[qrow + col_b]);
                        }
                        const int c = j + SB_PFB_U + 3;
                        if (r.used[2 * SB_PFB_U + 1] && c < m) {
                            g = sb_pfb_fma(g, r.omega[2 * SB_PFB_U + 1], vd[qrow + c]);
                        }
                    }
                    generic[r.cls] = g;
                }
                // The WIN terms are the last terms of RC, classes in grammar order.
                const sb_u64 in_base = (sb_u64)((a[i - 1] * 4 + b[j + 1]) * np + pi);
                for (int c = 0; c < nc; ++c) {
                    const SbF53 inner = sb_pf_w(w, p.loop_in + (sb_u64)c * 16 * (sb_u64)np + in_base);
                    special = sb_f53_add(special, sb_f53_mul(generic[c], inner));
                }
            }
            const int alpha = i >= 1 ? a[i - 1] : SB_PF_NONE;
            const int beta = j + 1 < m ? b[j + 1] : SB_PF_NONE;
            const SbF53 value =
                sb_f53_add(sb_pf_w(w, start + (sb_u64)((alpha * np + pi) * 5 + beta)), special);
            ra_here[j] = value;
            if (i + 1 < n && j >= 1) {
                const int tail = a[i + 1] * 4 + b[j - 1];
                for (int c = 0; c < nc; ++c) {
                    ro_here[j * nc + c] = sb_f53_mul(
                        value, sb_pf_w(w, p.loop_out + (sb_u64)((c * np + pi) * 16 + tail)));
                }
            } else {
                for (int c = 0; c < nc; ++c) {
                    ro_here[j * nc + c] = sb_f53_zero();
                }
            }
            if (special.m != 0ull) {
                const int alpha2 = i + 1 < n ? a[i + 1] : SB_PF_NONE;
                const int beta2 = j >= 1 ? b[j - 1] : SB_PF_NONE;
                const SbF53 end = sb_pf_w(w, p.end + (sb_u64)((pi * 5 + alpha2) * 5 + beta2));
                col[j] = sb_f53_add(col[j], sb_f53_mul(special, end));
            }
        }
        // Row i is final: the accumulators of row i (§3.4).
        const int here = (i % SB_PFB_ACC_ROWS) * m;
        const int before = ((i + SB_PFB_ACC_ROWS - 1) % SB_PFB_ACC_ROWS) * m;
        for (int t = 0; t < nrel; ++t) {
            const SbPfbClass& r = rel[t];
            SbF53* h = acc + t * 4 * plane;
            SbF53* v_acc = h + plane;
            SbF53* hd = v_acc + plane;
            SbF53* vd = hd + plane;
            const bool lo = r.used[2 * SB_PFB_U];
            const bool hi = r.used[2 * SB_PFB_U + 1];
            if (r.need_h) {
                for (int l = 0; l < m; ++l) {
                    const SbF53 prev = i >= 1 ? h[before + l] : sb_f53_zero();
                    h[here + l] = sb_pfb_fma(ro_here[l * nc + r.cls], r.x, prev);
                }
                if (lo) {
                    for (int l = 0; l < m; ++l) {
                        const SbF53 prev = (i >= 1 && l + 1 < m) ? hd[before + l + 1] : sb_f53_zero();
                        hd[here + l] = sb_pfb_fma(h[here + l], r.z, prev);
                    }
                }
            }
            if (r.need_v) {
                SbF53 right = sb_f53_zero();
                for (int l = m - 1; l >= 0; --l) {
                    right = sb_pfb_fma(ro_here[l * nc + r.cls], r.x, right);
                    v_acc[here + l] = right;
                }
                if (hi) {
                    for (int l = 0; l < m; ++l) {
                        const SbF53 prev = (i >= 1 && l + 1 < m) ? vd[before + l + 1] : sb_f53_zero();
                        vd[here + l] = sb_pfb_fma(v_acc[here + l], r.z, prev);
                    }
                }
            }
        }
        cur = cur + 1 == rows ? 0 : cur + 1;
        cur_ro = cur_ro + 1 == ro_rows ? 0 : cur_ro + 1;
    }
    SbF53 total = sb_f53_zero();
    for (int c = 0; c < m; ++c) {
        total = sb_f53_add(total, col[c]);
    }
    return total;
}

// Canonical orientation and sigma of the staged codes (K3's rule, sb_pf_body), then
// Z-hat_b with the zb_scale_down fault applied.
__device__ __forceinline__ SbF53 sb_pfb_pair(const SbPfBound& p, const SbPfbClass* rel,
                                             const sb_u8* ca, int na, const sb_u8* cb, int nb,
                                             SbF53* bands) {
    if (na < 2 || nb < 2) {
        return sb_f53_zero();
    }
    const int shorter = na < nb ? na : nb;
    int diff = shorter;
    for (int q = 0; q < shorter; ++q) {
        if (ca[q] != cb[q]) {
            diff = q;
            break;
        }
    }
    const bool swap = diff == shorter ? na > nb : ca[diff] > cb[diff];
    const sb_u8* x = swap ? cb : ca;
    const sb_u8* y = swap ? ca : cb;
    const int n = swap ? nb : na;
    const int m = swap ? na : nb;
    int sigma = 0;
    if (diff == shorter && na == nb) {
        sigma = 1;
        for (int q = 0; q < n; ++q) {
            if (x[q] + x[n - 1 - q] != 3) {
                sigma = 0;
                break;
            }
        }
    }
    SbF53 z = sb_pfb_zhat(p, rel, x, n, y, m, sigma, bands);
    if (p.scale_down != 0 && z.m != 0ull) {
        z.e -= 64;
    }
    return z;
}

// next_up of a finite binary64 (+-0's is the least subnormal).
__device__ __forceinline__ sb_u64 sb_pfb_next_up(sb_u64 x) {
    if ((x & ~SB_F64_SIGN) == 0) {
        return 1ull;
    }
    return (x & SB_F64_SIGN) ? x - 1 : x + 1;
}

// f64-up(N) of an integer 2^25 <= N < 2^64: the least binary64 >= N.
__device__ __forceinline__ sb_u64 sb_pfb_ceil(sb_u64 n) {
    int bits = 64 - __clzll((long long)n);
    sb_u64 mant;
    if (bits <= 53) {
        mant = n << (53 - bits);
    } else {
        const int shift = bits - 53;
        mant = n >> shift;
        if ((n & ((1ull << shift) - 1ull)) != 0ull) {
            mant += 1ull;
            if (mant == (1ull << 53)) {
                mant >>= 1;
                bits += 1;
            }
        }
    }
    return ((sb_u64)(bits - 1 + 1023) << 52) | (mant & SB_F64_FRAC);
}

// L of §4.4 as binary64 bits: +inf for Z-hat_b = 0; else N = 6|e_b| + 2^25,
// M = f64-up(N)·2^-53 (exact), Gamma = next_up(RN(g_b + M)), L = RN(c_T·Gamma), -0 -> +0.
__device__ __forceinline__ sb_u64 sb_pfb_lower(const SbF53& z, const SbPfFinalize& f) {
    if (z.m == 0ull) {
        return SB_F64_INF;
    }
    const sb_u64 mag = z.e < 0 ? (sb_u64)(-z.e) : (sb_u64)z.e;
    const sb_u64 margin = sb_pfb_ceil(6ull * mag + (1ull << 25)) - (53ull << 52);
    const sb_u64 gamma = sb_pfb_next_up(sb_f64_add(sb_pf_resident_g(z, f), margin));
    const sb_u64 l = sb_f64_mul(f.c_t, gamma);
    return sb_f64_is_zero(l) ? 0ull : l;
}

// The outcome of §1 and λ = key(L) (key(+inf) for Z-hat_b = 0).
__device__ __forceinline__ unsigned sb_pfb_outcome(const SbF53& z, sb_u64 l, bool has,
                                                   sb_i64 theta, sb_i64& lambda) {
    lambda = sb_pf_resident_key(l);
    if (has && lambda > theta) {
        return SB_PFB_PRUNED;
    }
    return z.m == 0ull ? SB_PFB_CERTIFIED : SB_PFB_OPEN;
}

// Testing evaluation: Z-hat_b, L and the outcome of every item, one thread per item with
// its own scratch slot of `stride` values (25 for the staged codes, then the bands).
extern "C" __global__ void sb_pf_bound_eval(SbPfBound p, SbPfFinalize f) {
    SbPfbClass rel[SB_PFB_MAX_CLASSES];
    sb_pfb_load(p, rel);
    const sb_u64 threads = (sb_u64)gridDim.x * blockDim.x;
    const sb_u64 self = (sb_u64)blockIdx.x * blockDim.x + threadIdx.x;
    SbF53* scratch = (SbF53*)p.scratch + self * p.stride;
    const sb_u8* codes = (const sb_u8*)p.codes;
    for (sb_u64 t = self; t < p.n_items; t += threads) {
        const sb_u64 as = ((const sb_u64*)p.a_start)[t];
        const sb_u64 al = ((const sb_u64*)p.a_len)[t];
        const sb_u64 bs = ((const sb_u64*)p.b_start)[t];
        const sb_u64 bl = ((const sb_u64*)p.b_len)[t];
        const SbF53 z = sb_pfb_pair(p, rel, codes + as, (int)al, codes + bs, (int)bl,
                                    scratch + SB_PFB_CODE_VALUES);
        const sb_u64 l = sb_pfb_lower(z, f);
        sb_i64 lambda;
        const unsigned code = sb_pfb_outcome(z, l, p.has_theta != 0, p.theta, lambda);
        ((sb_u64*)p.out_m)[t] = z.m;
        ((sb_i64*)p.out_e)[t] = z.e;
        ((sb_u64*)p.out_l)[t] = l;
        ((sb_u8*)p.out_code)[t] = (sb_u8)code;
    }
}
