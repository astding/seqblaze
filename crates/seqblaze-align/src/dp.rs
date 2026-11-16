//! The dynamic programs: native Gotoh (spec §2) with rule E traceback (§4),
//! compat water (§5.5) and compat needle (§5.6).
//!
//! Every value is an exact fixed-point integer held in i64. Under the §9.1
//! preflight every finite value lies within +-2^30, so i64 leaves room for the
//! -infinity sentinel `NEG` to absorb any number of subtractions without
//! reaching a finite value or overflowing; no clamping is needed.
use seqblaze_core::validate::EngineError;

use crate::scoring::Tables;
use crate::scratch::{checked_len, emptied, filled, reserve_total, Extent, Work};

/// -infinity: below every finite value by more than any chain of penalties.
pub(crate) const NEG: i64 = i64::MIN / 4;

const ROWS: &str = "align DP rows";
const CODES: &str = "align traceback";
const COLUMNS: &str = "align alignment columns";

/// A pair's score: fixed point, or water's -FLT_MAX sentinel (spec §5.5).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Score {
    Fx(i64),
    Sentinel,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Gaps {
    pub go: i64,
    pub ge: i64,
}

impl Gaps {
    /// gamma(L) = go + (L - 1) ge for a run of L >= 1 columns; 0 for L = 0.
    #[inline]
    pub fn cost(self, len: usize) -> i64 {
        if len == 0 {
            0
        } else {
            self.go + (len as i64 - 1) * self.ge
        }
    }
}

// State ids of the native traceback codes. A cell's byte holds the
// predecessor of V in bits 0-1, of P in bits 2-3 and of Q in bits 4-5.
const START: u8 = 0;
const P: u8 = 1;
const Q: u8 = 2;
const V: u8 = 3;

/// The first of (x1, x2, x3) equal to `top`, as P, Q, V.
#[inline(always)]
fn first3(top: i64, x1: i64, x2: i64) -> u8 {
    if x1 == top {
        P
    } else if x2 == top {
        Q
    } else {
        V
    }
}

fn push_run(ops: &mut Vec<u8>, op: u8, count: usize) {
    ops.extend(std::iter::repeat_n(op, count));
}

// ------------------------------------------------------------------ native

/// Native local score, linear memory (spec §2).
pub(crate) fn local_score(
    a: &[u8],
    b: &[u8],
    t: &Tables,
    g: Gaps,
    w: &mut Work,
) -> Result<i64, EngineError> {
    local_fill::<false>(a, b, t, g, w).map(|(best, _)| best)
}

/// Native local score and `n_ends`, the number of cells with V(i, j) = score
/// when score > 0 (0 otherwise), linear memory (spec §2, §3.1).
pub(crate) fn local_score_counted(
    a: &[u8],
    b: &[u8],
    t: &Tables,
    g: Gaps,
    w: &mut Work,
) -> Result<(i64, u64), EngineError> {
    local_fill::<true>(a, b, t, g, w)
}

/// The local fill; with COUNT, a running maximum and its count (spec §3.1:
/// V > best resets the count to 1, V = best > 0 adds 1). Without COUNT the
/// count is 0 and the maximum is taken branch-free, as before.
#[inline(always)]
fn local_fill<const COUNT: bool>(
    a: &[u8],
    b: &[u8],
    t: &Tables,
    g: Gaps,
    w: &mut Work,
) -> Result<(i64, u64), EngineError> {
    let n = b.len();
    if a.is_empty() || n == 0 {
        return Ok((0, 0));
    }
    // max(V, Q) and P of the previous row at column j (index j - 1); column 0
    // is -infinity in local mode.
    filled(&mut w.r0, n, NEG, ROWS, w.budget)?;
    filled(&mut w.r1, n, NEG, ROWS, w.budget)?;
    let (vq, pp) = (&mut w.r0[..n], &mut w.r1[..n]);
    let (mut best, mut count) = (0i64, 0u64);
    for &x in a {
        let row = &t.sub[x as usize];
        let (mut diag, mut vp_left, mut q_left) = (NEG, NEG, NEG);
        for ((&y, vq_j), pp_j) in b.iter().zip(vq.iter_mut()).zip(pp.iter_mut()) {
            let (up_vq, up_p) = (*vq_j, *pp_j);
            let v = i64::from(row[y as usize]) + diag.max(0);
            let p = (up_vq - g.go).max(up_p - g.ge);
            let q = (vp_left - g.go).max(q_left - g.ge);
            diag = up_vq.max(up_p);
            *vq_j = v.max(q);
            *pp_j = p;
            vp_left = v.max(p);
            q_left = q;
            if COUNT {
                if v > best {
                    best = v;
                    count = 1;
                } else if v == best && best > 0 {
                    count += 1;
                }
            } else {
                best = best.max(v);
            }
        }
    }
    Ok((best, count))
}

/// Native global score, linear memory: max(V(m, n), T_I(L), T_D(L)) (spec §2).
pub(crate) fn global_score(
    a: &[u8],
    b: &[u8],
    t: &Tables,
    g: Gaps,
    end_gaps: bool,
    w: &mut Work,
) -> Result<i64, EngineError> {
    let (m, n) = (a.len(), b.len());
    let lam = |len: usize| if end_gaps { g.cost(len) } else { 0 };
    if m == 0 || n == 0 {
        return Ok(-lam(m.max(n)));
    }
    filled(&mut w.r0, n + 1, NEG, ROWS, w.budget)?;
    filled(&mut w.r1, n + 1, NEG, ROWS, w.budget)?;
    let (vq, pp) = (&mut w.r0[..=n], &mut w.r1[..=n]);
    // Row 0: Z(0, 0) = 0 and the leading D runs Q(0, j).
    vq[0] = 0;
    for (j, value) in vq.iter_mut().enumerate().skip(1) {
        *value = -lam(j);
    }
    let mut ti = NEG;
    let mut score = NEG;
    for (i0, &x) in a.iter().enumerate() {
        let i = i0 + 1;
        // T_I(m - i0) from max(Z, Q, V)(i0, n).
        ti = ti.max(vq[n] - lam(m - i0));
        let row = &t.sub[x as usize];
        let mut diag = vq[0].max(pp[0]);
        let p0 = -lam(i);
        vq[0] = NEG;
        pp[0] = p0;
        let (mut vp_left, mut q_left) = (p0, NEG);
        let cells = b.iter().zip(vq[1..].iter_mut()).zip(pp[1..].iter_mut());
        if i < m {
            for ((&y, vq_j), pp_j) in cells {
                let (up_vq, up_p) = (*vq_j, *pp_j);
                let v = i64::from(row[y as usize]) + diag;
                let p = (up_vq - g.go).max(up_p - g.ge);
                let q = (vp_left - g.go).max(q_left - g.ge);
                diag = up_vq.max(up_p);
                *vq_j = v.max(q);
                *pp_j = p;
                vp_left = v.max(p);
                q_left = q;
            }
        } else {
            // Last row: T_D(n - j) from max(Z, P, V)(m, j) for j < n.
            let mut td = vp_left - lam(n);
            let mut corner = NEG;
            for (j0, ((&y, vq_j), pp_j)) in cells.enumerate() {
                let (up_vq, up_p) = (*vq_j, *pp_j);
                let v = i64::from(row[y as usize]) + diag;
                let p = (up_vq - g.go).max(up_p - g.ge);
                let q = (vp_left - g.go).max(q_left - g.ge);
                diag = up_vq.max(up_p);
                *vq_j = v.max(q);
                *pp_j = p;
                vp_left = v.max(p);
                q_left = q;
                if j0 + 1 < n {
                    td = td.max(vp_left - lam(n - j0 - 1));
                } else {
                    corner = v;
                }
            }
            score = corner.max(ti).max(td);
        }
    }
    Ok(score)
}

/// Native score and the rule E alignment (spec §4) into `w.ops` and
/// `w.extent`.
pub(crate) fn native_trace(
    a: &[u8],
    b: &[u8],
    t: &Tables,
    g: Gaps,
    local: bool,
    end_gaps: bool,
    w: &mut Work,
) -> Result<i64, EngineError> {
    let (m, n) = (a.len(), b.len());
    let lam = |len: usize| if end_gaps { g.cost(len) } else { 0 };
    w.ops.clear();
    w.n_ends = 0;
    reserve_total(&mut w.ops, m.saturating_add(n), COLUMNS, w.budget)?;
    if m == 0 || n == 0 {
        if local {
            w.extent = Extent::default();
            return Ok(0);
        }
        push_run(&mut w.ops, b'I', m);
        push_run(&mut w.ops, b'D', n);
        w.extent = Extent {
            a_start: 0,
            a_end: m,
            b_start: 0,
            b_end: n,
        };
        return Ok(-lam(m.max(n)));
    }
    let cells = checked_len(&[m, n], CODES)?;
    // Every cell's code is pushed in row-major order below.
    emptied(&mut w.codes, cells, CODES, w.budget)?;
    filled(&mut w.r0, n + 1, NEG, ROWS, w.budget)?;
    filled(&mut w.r1, n + 1, NEG, ROWS, w.budget)?;
    filled(&mut w.r2, n + 1, NEG, ROWS, w.budget)?;
    filled(&mut w.col, m + 1, [NEG; 3], ROWS, w.budget)?;
    let (v, p, q) = (&mut w.r0[..=n], &mut w.r1[..=n], &mut w.r2[..=n]);
    if !local {
        for (j, value) in q.iter_mut().enumerate().skip(1) {
            *value = -lam(j);
        }
    }
    // (V, Q) of column n per row, for the trailing I runs.
    w.col[0] = [v[n], q[n], NEG];
    // The local maximum, its first row-major cell and its count (spec §3.1).
    let (mut best, mut end, mut count) = (0i64, (0usize, 0usize), 0u64);
    for (i0, &x) in a.iter().enumerate() {
        let i = i0 + 1;
        let row = &t.sub[x as usize];
        let (mut dv, mut dp, mut dq) = (v[0], p[0], q[0]);
        v[0] = NEG;
        p[0] = if local { NEG } else { -lam(i) };
        q[0] = NEG;
        let (mut lv, mut lp, mut lq) = (v[0], p[0], q[0]);
        for j in 1..=n {
            let (uv, up, uq) = (v[j], p[j], q[j]);
            let top = dp.max(dq).max(dv);
            let (pred, vc) = if local {
                if top <= 0 {
                    (0, START)
                } else {
                    (top, first3(top, dp, dq))
                }
            } else if i == 1 && j == 1 {
                (0, START)
            } else {
                (top, first3(top, dp, dq))
            };
            let vv = i64::from(row[b[j - 1] as usize]) + pred;
            let (pe, qo, vo) = (up - g.ge, uq - g.go, uv - g.go);
            let pv = pe.max(qo).max(vo);
            let pc = first3(pv, pe, qo);
            let (lpo, lqe, lvo) = (lp - g.go, lq - g.ge, lv - g.go);
            let qv = lpo.max(lqe).max(lvo);
            let qc = first3(qv, lpo, lqe);
            w.codes.push(vc | pc << 2 | qc << 4);
            (dv, dp, dq) = (uv, up, uq);
            v[j] = vv;
            p[j] = pv;
            q[j] = qv;
            (lv, lp, lq) = (vv, pv, qv);
            if local {
                if vv > best {
                    best = vv;
                    end = (i, j);
                    count = 1;
                } else if vv == best && best > 0 {
                    count += 1;
                }
            }
        }
        w.col[i] = [v[n], q[n], NEG];
    }
    debug_assert_eq!(w.codes.len(), cells, "every traceback code is written");
    if local {
        if best <= 0 {
            w.extent = Extent::default();
            return Ok(0);
        }
        w.n_ends = count;
        let (i, j) = walk(&w.codes, n, V, end.0, end.1, &mut w.ops);
        w.ops.reverse();
        w.extent = Extent {
            a_start: i,
            a_end: end.0,
            b_start: j,
            b_end: end.1,
        };
        return Ok(best);
    }
    // Global end (spec §4 step 2); row m is in v, p.
    let ti = |len: usize| w.col[m - len][0].max(w.col[m - len][1]) - lam(len);
    let td = |len: usize| v[n - len].max(p[n - len]) - lam(len);
    let score = (1..=m).map(ti).chain((1..=n).map(td)).fold(v[n], i64::max);
    let (state, i, j) = if let Some(len) = (1..=m).rev().find(|&len| ti(len) == score) {
        let target = score + lam(len);
        let [cv, cq, _] = w.col[m - len];
        push_run(&mut w.ops, b'I', len);
        if cq == target {
            (Q, m - len, n)
        } else {
            debug_assert_eq!(cv, target);
            (V, m - len, n)
        }
    } else if let Some(longest) = (1..=n).rev().find(|&len| td(len) == score) {
        let mut len = 1;
        let state = loop {
            let target = score + lam(len);
            if p[n - len] == target {
                break P;
            }
            if len == longest {
                debug_assert_eq!(v[n - len], target);
                break V;
            }
            len += 1;
        };
        push_run(&mut w.ops, b'D', len);
        (state, m, n - len)
    } else {
        (V, m, n)
    };
    walk(&w.codes, n, state, i, j, &mut w.ops);
    w.ops.reverse();
    w.extent = Extent {
        a_start: 0,
        a_end: m,
        b_start: 0,
        b_end: n,
    };
    Ok(score)
}

/// Greedy rule E traceback from `state` at (i, j); pushes columns backwards
/// and returns the cell where the alignment starts.
fn walk(
    codes: &[u8],
    n: usize,
    mut state: u8,
    mut i: usize,
    mut j: usize,
    ops: &mut Vec<u8>,
) -> (usize, usize) {
    loop {
        match state {
            V => {
                let code = codes[(i - 1) * n + j - 1] & 3;
                ops.push(b'M');
                i -= 1;
                j -= 1;
                if code == START {
                    return (i, j);
                }
                state = code;
            }
            P => {
                if j == 0 {
                    // Global leading I run a_1..a_i.
                    push_run(ops, b'I', i);
                    return (0, 0);
                }
                let code = codes[(i - 1) * n + j - 1] >> 2 & 3;
                ops.push(b'I');
                i -= 1;
                state = code;
            }
            _ => {
                if i == 0 {
                    // Global leading D run b_1..b_j.
                    push_run(ops, b'D', j);
                    return (0, 0);
                }
                let code = codes[(i - 1) * n + j - 1] >> 4 & 3;
                ops.push(b'D');
                j -= 1;
                state = code;
            }
        }
    }
}

// ------------------------------------------------------------------- water

/// Water fill (spec §5.5): the interior maximum of R over i, j >= 2 and, when
/// tracing, the codes and the first row-major maximum of H.
///
/// Score pass (`!TRACE`): the compat domain has ge <= go (spec §5.2), so
/// E(i, j) = max(H(i, j-1) - go, E(i, j-1) - ge) equals
/// max(G(i, j-1) - go, E(i, j-1) - ge) with G = max(M, F, 0): the dropped
/// term E(i, j-1) - go never exceeds E(i, j-1) - ge. G does not depend on E,
/// which shortens the dependency chain along a row; every value is unchanged.
/// The trace keeps H, because the open flags (eo == e) compare against it.
/// `needle_fill` does the same with G = max(M, F).
fn water_fill<const TRACE: bool>(
    a: &[u8],
    b: &[u8],
    t: &Tables,
    g: Gaps,
    w: &mut Work,
) -> Result<(i64, i64, (usize, usize)), EngineError> {
    debug_assert!(g.ge <= g.go, "compat domain: ge <= go");
    let n = b.len();
    if TRACE {
        let cells = checked_len(&[a.len(), n], CODES)?;
        // Every cell's code is pushed in row-major order below.
        emptied(&mut w.codes, cells, CODES, w.budget)?;
    }
    // H and F of the previous row at column j (index j - 1); H(i, 0) = 0.
    filled(&mut w.r0, n, 0, ROWS, w.budget)?;
    filled(&mut w.r1, n, NEG, ROWS, w.budget)?;
    let (h, f) = (&mut w.r0[..n], &mut w.r1[..n]);
    let (mut rmax, mut best, mut end) = (NEG, 0i64, (0usize, 0usize));
    for (i0, &x) in a.iter().enumerate() {
        let row = &t.sub[x as usize];
        let (mut diag, mut e, mut hl) = (0i64, NEG, 0i64);
        let interior = i0 >= 1;
        for j0 in 0..n {
            let uh = h[j0];
            let eo = hl - g.go;
            e = eo.max(e - g.ge);
            let fo = uh - g.go;
            let fv = fo.max(f[j0] - g.ge);
            let mm = diag + i64::from(row[b[j0] as usize]);
            // max(M, F) first: R and G share it, and in this order the
            // compiler emits conditional moves instead of data-dependent
            // branches (about 3x faster on random sequences, measured).
            let mf = mm.max(fv);
            let r = mf.max(e);
            let hh = r.max(0);
            if interior && j0 >= 1 {
                rmax = rmax.max(r);
            }
            if TRACE {
                let choice = if hh == 0 {
                    0
                } else if mm == hh {
                    1
                } else if e == hh {
                    2
                } else {
                    3
                };
                w.codes
                    .push(choice | u8::from(eo == e) << 2 | u8::from(fo == fv) << 3);
                if hh > best {
                    best = hh;
                    end = (i0 + 1, j0 + 1);
                }
            }
            diag = uh;
            h[j0] = hh;
            f[j0] = fv;
            hl = if TRACE { hh } else { mf.max(0) };
        }
    }
    debug_assert!(
        !TRACE || w.codes.len() == a.len() * n,
        "every traceback code is written"
    );
    Ok((rmax, best, end))
}

pub(crate) fn water_score(
    a: &[u8],
    b: &[u8],
    t: &Tables,
    g: Gaps,
    w: &mut Work,
) -> Result<Score, EngineError> {
    if a.len() < 2 || b.len() < 2 {
        return Ok(Score::Sentinel);
    }
    Ok(Score::Fx(water_fill::<false>(a, b, t, g, w)?.0))
}

pub(crate) fn water_trace(
    a: &[u8],
    b: &[u8],
    t: &Tables,
    g: Gaps,
    w: &mut Work,
) -> Result<Score, EngineError> {
    let (m, n) = (a.len(), b.len());
    w.ops.clear();
    if m == 0 || n == 0 {
        // Unreachable under the compat preflight (non-empty sequences).
        w.extent = Extent::default();
        return Ok(Score::Sentinel);
    }
    reserve_total(&mut w.ops, m + n, COLUMNS, w.budget)?;
    let (rmax, best, end) = water_fill::<true>(a, b, t, g, w)?;
    let score = if m < 2 || n < 2 {
        Score::Sentinel
    } else {
        Score::Fx(rmax)
    };
    if best == 0 {
        w.ops.push(b'M');
        w.extent = Extent {
            a_start: 0,
            a_end: 1,
            b_start: 0,
            b_end: 1,
        };
        return Ok(score);
    }
    const H: u8 = 0;
    const E: u8 = 1;
    const F: u8 = 2;
    let (mut i, mut j) = end;
    let mut state = H;
    loop {
        if state == H && (i == 0 || j == 0) {
            break;
        }
        let code = w.codes[(i - 1) * n + j - 1];
        match state {
            H => match code & 3 {
                0 => break,
                1 => {
                    w.ops.push(b'M');
                    i -= 1;
                    j -= 1;
                }
                2 => state = E,
                _ => state = F,
            },
            E => {
                w.ops.push(b'D');
                j -= 1;
                state = if code & 4 != 0 { H } else { E };
            }
            _ => {
                w.ops.push(b'I');
                i -= 1;
                state = if code & 8 != 0 { H } else { F };
            }
        }
    }
    w.ops.reverse();
    w.extent = Extent {
        a_start: i,
        a_end: end.0,
        b_start: j,
        b_end: end.1,
    };
    Ok(score)
}

// ------------------------------------------------------------------ needle

/// Rule R-m1 (spec §5.6): end_gaps=False, m = 1, n >= 2, ge = 0 and
/// s(a_1, b_n) <= 0.
fn rule_m1(a: &[u8], b: &[u8], t: &Tables, g: Gaps, end_gaps: bool) -> bool {
    !end_gaps && a.len() == 1 && b.len() >= 2 && g.ge == 0 && t.score(a[0], b[b.len() - 1]) <= 0
}

/// Needle fill (spec §5.6). Without TRACE it returns the score (the largest
/// candidate); with TRACE it fills the codes and the values of row m and
/// column n.
///
/// Each traceback code holds the H choice (bits 0-1) and the rule B1
/// continuation bits of E (bit 2) and F (bit 3): with p the predecessor
/// along the run, the run continues iff p is not in column 0 (E) or row 0
/// (F) and G(p) = H(p) or G(p) - x = G(c), where x = ge off the border and
/// x_border (0 without end gaps, ge with them) on row 1 (E) or column 1 (F).
fn needle_fill<const TRACE: bool>(
    a: &[u8],
    b: &[u8],
    t: &Tables,
    g: Gaps,
    end_gaps: bool,
    w: &mut Work,
) -> Result<i64, EngineError> {
    debug_assert!(g.ge <= g.go, "compat domain: ge <= go");
    let (m, n) = (a.len(), b.len());
    let lead = |len: usize| if end_gaps { g.cost(len) } else { 0 };
    // x of the border test (spec §5.6, rule B1): EMBOSS's -endextend, which
    // the compat mapping sets to ge, with end gaps; 0 without.
    let x_border = if end_gaps { g.ge } else { 0 };
    if TRACE {
        let cells = checked_len(&[m, n], CODES)?;
        // Every cell's code is pushed in row-major order below.
        emptied(&mut w.codes, cells, CODES, w.budget)?;
        filled(&mut w.row, n + 1, [NEG; 3], ROWS, w.budget)?;
        filled(&mut w.col, m + 1, [NEG; 3], ROWS, w.budget)?;
    }
    filled(&mut w.r0, n + 1, NEG, ROWS, w.budget)?;
    filled(&mut w.r1, n + 1, NEG, ROWS, w.budget)?;
    let (h, f) = (&mut w.r0[..=n], &mut w.r1[..=n]);
    h[0] = 0;
    for (j, value) in h.iter_mut().enumerate().skip(1) {
        *value = -lead(j);
    }
    // Candidates other than C0 (score-only path).
    let mut cand = NEG;
    if n >= 2 {
        cand = cand.max(h[n] - lead(m));
    }
    if m >= 2 {
        cand = cand.max(-lead(m) - lead(n));
    }
    if m == 1 && n == 1 {
        cand = cand.max(-lead(1) - lead(1));
    }
    let mut corner = NEG;
    for (i0, &x) in a.iter().enumerate() {
        let i = i0 + 1;
        let row = &t.sub[x as usize];
        let mut diag = h[0];
        h[0] = -lead(i);
        let (mut e, mut hl, mut mm_last) = (NEG, h[0], NEG);
        let last = i == m;
        let xe = if i >= 2 { g.ge } else { x_border };
        for j in 1..n + 1 {
            let (uh, fu) = (h[j], f[j]);
            let ee = e - g.ge;
            let en = (hl - g.go).max(ee);
            let fe = fu - g.ge;
            let fv = (uh - g.go).max(fe);
            let mm = diag + i64::from(row[b[j - 1] as usize]);
            let hh = mm.max(en).max(fv);
            if TRACE {
                let choice = if mm == hh {
                    0
                } else if en == hh {
                    1
                } else {
                    2
                };
                // Rule B1 continuation bits; e, hl are E, H at (i, j-1)
                // and fu, uh are F, H at (i-1, j).
                let xf = if j >= 2 { g.ge } else { x_border };
                let cont_e = u8::from(j >= 2 && (e == hl || e - xe == en));
                let cont_f = u8::from(i >= 2 && (fu == uh || fu - xf == fv));
                w.codes.push(choice | cont_e << 2 | cont_f << 3);
                if last {
                    w.row[j] = [mm, en, fv];
                }
            } else if last && j < n && m >= 2 {
                cand = cand.max(mm - lead(n - j));
            }
            diag = uh;
            h[j] = hh;
            f[j] = fv;
            e = en;
            hl = if TRACE { hh } else { mm.max(fv) };
            mm_last = mm;
        }
        if TRACE {
            w.col[i] = [mm_last, e, f[n]];
        } else if last {
            corner = mm_last;
        } else if n >= 2 {
            cand = cand.max(mm_last - lead(m - i));
        }
    }
    debug_assert!(
        !TRACE || w.codes.len() == m * n,
        "every traceback code is written"
    );
    Ok(corner.max(cand))
}

pub(crate) fn needle_score(
    a: &[u8],
    b: &[u8],
    t: &Tables,
    g: Gaps,
    end_gaps: bool,
    w: &mut Work,
) -> Result<i64, EngineError> {
    if a.is_empty() || b.is_empty() {
        // Unreachable under the compat preflight (non-empty sequences).
        return global_score(a, b, t, g, end_gaps, w);
    }
    if rule_m1(a, b, t, g, end_gaps) {
        return Ok(0);
    }
    needle_fill::<false>(a, b, t, g, end_gaps, w)
}

pub(crate) fn needle_trace(
    a: &[u8],
    b: &[u8],
    t: &Tables,
    g: Gaps,
    end_gaps: bool,
    w: &mut Work,
) -> Result<i64, EngineError> {
    let (m, n) = (a.len(), b.len());
    if m == 0 || n == 0 {
        // Unreachable under the compat preflight (non-empty sequences).
        return native_trace(a, b, t, g, false, end_gaps, w);
    }
    w.ops.clear();
    reserve_total(&mut w.ops, m + n, COLUMNS, w.budget)?;
    w.extent = Extent {
        a_start: 0,
        a_end: m,
        b_start: 0,
        b_end: n,
    };
    if rule_m1(a, b, t, g, end_gaps) {
        w.ops.push(b'I');
        push_run(&mut w.ops, b'D', n);
        return Ok(0);
    }
    needle_fill::<true>(a, b, t, g, end_gaps, w)?;
    let lead = |len: usize| if end_gaps { g.cost(len) } else { 0 };
    // Candidates in tie order: C0, (m, j) by ascending j, (i, n) by ascending i,
    // and (1, 0) when m = n = 1.
    let (row, col) = (&w.row, &w.col);
    let candidates = std::iter::once(((m, n), row[n][0]))
        .chain((0..n).filter(|_| m >= 2).map(|j| {
            let value = if j == 0 { -lead(m) } else { row[j][0] };
            ((m, j), value - lead(n - j))
        }))
        .chain((0..m).filter(|_| n >= 2).map(|i| {
            let value = if i == 0 { -lead(n) } else { col[i][0] };
            ((i, n), value - lead(m - i))
        }))
        .chain(std::iter::once(((1, 0), -lead(1) - lead(1))).filter(|_| m == 1 && n == 1));
    let mut chosen = ((m, n), NEG);
    for (cell, value) in candidates {
        if value > chosen.1 {
            chosen = (cell, value);
        }
    }
    let ((ie, je), score) = chosen;
    // Trailing columns, pushed first because columns are built backwards.
    if ie == m && je < n {
        push_run(&mut w.ops, b'D', n - je);
    } else if je == n && ie < m {
        push_run(&mut w.ops, b'I', m - ie);
    }
    let (mut i, mut j) = (ie, je);
    if i >= 1 && j >= 1 {
        const M: u8 = 0;
        const E: u8 = 1;
        const F: u8 = 2;
        const H: u8 = 3;
        let [vm, ve, vf] = if ie == m { w.row[je] } else { w.col[ie] };
        let mut state = M;
        let mut top = vm;
        if ie != m && ve > top {
            state = E;
            top = ve;
        }
        if je != n && vf > top {
            state = F;
        }
        loop {
            match state {
                M => {
                    w.ops.push(b'M');
                    i -= 1;
                    j -= 1;
                    if i == 0 || j == 0 {
                        break;
                    }
                    state = H;
                }
                H => {
                    state = match w.codes[(i - 1) * n + j - 1] & 3 {
                        0 => M,
                        1 => E,
                        _ => F,
                    };
                }
                // Rule B1 (spec §5.6): a closed run steps once, stops on
                // column (row) 0 and otherwise goes to state H; no walk.
                E => {
                    w.ops.push(b'D');
                    let cont = w.codes[(i - 1) * n + j - 1] & 4 != 0;
                    j -= 1;
                    if !cont {
                        if j == 0 {
                            break;
                        }
                        state = H;
                    }
                }
                _ => {
                    w.ops.push(b'I');
                    let cont = w.codes[(i - 1) * n + j - 1] & 8 != 0;
                    i -= 1;
                    if !cont {
                        if i == 0 {
                            break;
                        }
                        state = H;
                    }
                }
            }
        }
    }
    // Leading columns (-, b_1..b_j) then (a_1..a_i, -), pushed reversed.
    push_run(&mut w.ops, b'I', i);
    push_run(&mut w.ops, b'D', j);
    w.ops.reverse();
    Ok(score)
}
