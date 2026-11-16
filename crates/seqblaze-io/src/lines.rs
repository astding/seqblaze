//! Newline search. SSE2 (baseline on x86_64) or a portable word-at-a-time scan;
//! both return the first `\n`, so the choice changes no output.

/// Index of the first `\n` byte in `hay`.
#[inline]
pub fn find_newline(hay: &[u8]) -> Option<usize> {
    #[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
    {
        sse2(hay)
    }
    #[cfg(not(all(target_arch = "x86_64", target_feature = "sse2")))]
    {
        words(hay)
    }
}

#[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
#[inline]
fn sse2(hay: &[u8]) -> Option<usize> {
    use std::arch::x86_64::{
        __m128i, _mm_cmpeq_epi8, _mm_loadu_si128, _mm_movemask_epi8, _mm_or_si128, _mm_set1_epi8,
    };
    let n = hay.len();
    let mut i = 0;
    let ptr = hay.as_ptr();
    // SAFETY: SSE2 is enabled for this target (cfg above). Every load reads 16
    // bytes at ptr + i with i + 16 <= n (32 for the paired loads), inside `hay`;
    // _mm_loadu_si128 has no alignment requirement.
    unsafe {
        let nl = _mm_set1_epi8(b'\n' as i8);
        while i + 32 <= n {
            let a = _mm_cmpeq_epi8(_mm_loadu_si128(ptr.add(i).cast::<__m128i>()), nl);
            let b = _mm_cmpeq_epi8(_mm_loadu_si128(ptr.add(i + 16).cast::<__m128i>()), nl);
            if _mm_movemask_epi8(_mm_or_si128(a, b)) != 0 {
                let low = _mm_movemask_epi8(a) as u32;
                if low != 0 {
                    return Some(i + low.trailing_zeros() as usize);
                }
                let high = _mm_movemask_epi8(b) as u32;
                return Some(i + 16 + high.trailing_zeros() as usize);
            }
            i += 32;
        }
        while i + 16 <= n {
            let mask = _mm_movemask_epi8(_mm_cmpeq_epi8(
                _mm_loadu_si128(ptr.add(i).cast::<__m128i>()),
                nl,
            )) as u32;
            if mask != 0 {
                return Some(i + mask.trailing_zeros() as usize);
            }
            i += 16;
        }
    }
    hay[i..].iter().position(|&b| b == b'\n').map(|k| i + k)
}

#[cfg_attr(all(target_arch = "x86_64", target_feature = "sse2"), allow(dead_code))]
#[inline]
fn words(hay: &[u8]) -> Option<usize> {
    const LO: u64 = 0x0101_0101_0101_0101;
    const HI: u64 = 0x8080_8080_8080_8080;
    const NL: u64 = LO * b'\n' as u64;
    let mut chunks = hay.chunks_exact(8);
    let mut base = 0;
    for chunk in &mut chunks {
        let word = u64::from_le_bytes(chunk.try_into().expect("8-byte chunk")) ^ NL;
        // Exact zero-byte test: the lowest set bit marks the first zero byte.
        let zero = (word.wrapping_sub(LO) & !word) & HI;
        if zero != 0 {
            return Some(base + (zero.trailing_zeros() / 8) as usize);
        }
        base += 8;
    }
    chunks
        .remainder()
        .iter()
        .position(|&b| b == b'\n')
        .map(|k| base + k)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn agrees_with_a_scalar_scan() {
        let mut data = [b'A'; 200];
        for len in 0..80 {
            for pos in 0..=len {
                data.iter_mut().for_each(|b| *b = b'A');
                if pos < len {
                    data[pos] = b'\n';
                    data[(pos + 7).min(len.max(1) - 1)] = b'\n';
                }
                for start in 0..3.min(len + 1) {
                    let hay = &data[start..len];
                    let want = hay.iter().position(|&b| b == b'\n');
                    assert_eq!(find_newline(hay), want, "len {len} pos {pos}");
                    assert_eq!(words(hay), want, "len {len} pos {pos}");
                }
            }
        }
        // High bytes next to the target must not produce false positives.
        let hay = [0x8a, 0x0b, 0x09, 0xff, 0x0a, 0x00, 0x0a, 0x8a, 0x8a];
        assert_eq!(words(&hay), Some(4));
        assert_eq!(find_newline(&hay), Some(4));
    }
}
