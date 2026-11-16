//! Fast clean-block certificates for the strict uppercase DNA alphabet.
//! Reads only immutable rows or caller-owned copies of borrowed NumPy bytes.

/// Certify the whole block; callers retain the ordered scalar error search.
#[inline]
pub(super) fn all_acgt(bytes: &[u8]) -> bool {
    #[cfg(target_arch = "x86_64")]
    if bytes.len() >= 64 && std::is_x86_feature_detected!("avx2") {
        // SAFETY: AVX2 was detected on this CPU.
        return unsafe { avx2::all_acgt(bytes) };
    }
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    {
        let mut blocks = bytes.chunks_exact(16);
        if !blocks.all(all_acgt_16) {
            return false;
        }
        scalar(blocks.remainder())
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        scalar(bytes)
    }
}

#[inline]
fn scalar(bytes: &[u8]) -> bool {
    bytes
        .iter()
        .all(|&byte| matches!(byte, b'A' | b'C' | b'G' | b'T'))
}

#[cfg(target_arch = "x86_64")]
#[inline]
fn all_acgt_16(bytes: &[u8]) -> bool {
    use core::arch::x86_64::{
        _mm_cmpeq_epi8, _mm_loadu_si128, _mm_movemask_epi8, _mm_or_si128, _mm_set1_epi8,
    };
    debug_assert_eq!(bytes.len(), 16);
    // SAFETY: SSE2 is an x86-64 baseline feature. The complete 16-byte chunk
    // belongs to a Rust slice; loadu permits its arbitrary byte alignment.
    unsafe {
        let block = _mm_loadu_si128(bytes.as_ptr().cast());
        let ac = _mm_or_si128(
            _mm_cmpeq_epi8(block, _mm_set1_epi8(b'A' as i8)),
            _mm_cmpeq_epi8(block, _mm_set1_epi8(b'C' as i8)),
        );
        let gt = _mm_or_si128(
            _mm_cmpeq_epi8(block, _mm_set1_epi8(b'G' as i8)),
            _mm_cmpeq_epi8(block, _mm_set1_epi8(b'T' as i8)),
        );
        _mm_movemask_epi8(_mm_or_si128(ac, gt)) == 0xffff
    }
}

#[cfg(target_arch = "aarch64")]
#[inline]
fn all_acgt_16(bytes: &[u8]) -> bool {
    use core::arch::aarch64::{vceqq_u8, vdupq_n_u8, vld1q_u8, vminvq_u8, vorrq_u8};
    debug_assert_eq!(bytes.len(), 16);
    // SAFETY: NEON is an aarch64 baseline feature. The complete 16-byte chunk
    // belongs to a Rust slice; vld1q_u8 permits arbitrary byte alignment.
    unsafe {
        let block = vld1q_u8(bytes.as_ptr());
        let ac = vorrq_u8(
            vceqq_u8(block, vdupq_n_u8(b'A')),
            vceqq_u8(block, vdupq_n_u8(b'C')),
        );
        let gt = vorrq_u8(
            vceqq_u8(block, vdupq_n_u8(b'G')),
            vceqq_u8(block, vdupq_n_u8(b'T')),
        );
        vminvq_u8(vorrq_u8(ac, gt)) == u8::MAX
    }
}

/// AVX2 form for spans of at least 64 bytes (blocks of the scan, leaves of
/// the stream pass): a byte x is A, C, G or T exactly when it equals the
/// entry of its low nibble in a 16-entry table (A 0x41, C 0x43, G 0x47, T
/// 0x54 at nibbles 1, 3, 7 and 4; every other entry's low nibble differs from
/// its index, so no byte equals it).
#[cfg(target_arch = "x86_64")]
mod avx2 {
    use core::arch::x86_64::{
        __m256i, _mm256_and_si256, _mm256_loadu_si256, _mm256_or_si256, _mm256_set1_epi8,
        _mm256_setr_epi8, _mm256_shuffle_epi8, _mm256_testz_si256, _mm256_xor_si256,
    };

    /// # Safety
    /// The CPU must have AVX2.
    #[target_feature(enable = "avx2")]
    pub(super) fn all_acgt(bytes: &[u8]) -> bool {
        // Entry n: the ACGT byte with low nibble n, else 0x0F ^ n.
        const T: [i8; 16] = [
            0x0F, 0x41, 0x0D, 0x43, 0x54, 0x0A, 0x09, 0x47, 0x07, 0x06, 0x05, 0x04, 0x03, 0x02,
            0x01, 0x00,
        ];
        let table = _mm256_setr_epi8(
            T[0], T[1], T[2], T[3], T[4], T[5], T[6], T[7], T[8], T[9], T[10], T[11], T[12], T[13],
            T[14], T[15], T[0], T[1], T[2], T[3], T[4], T[5], T[6], T[7], T[8], T[9], T[10], T[11],
            T[12], T[13], T[14], T[15],
        );
        let miss = |at: &[u8]| miss(table, at);
        let mut blocks = bytes.chunks_exact(128);
        for block in &mut blocks {
            let any = _mm256_or_si256(
                _mm256_or_si256(miss(&block[..32]), miss(&block[32..64])),
                _mm256_or_si256(miss(&block[64..96]), miss(&block[96..])),
            );
            if _mm256_testz_si256(any, any) == 0 {
                return false;
            }
        }
        let rest = blocks.remainder();
        let mut words = rest.chunks_exact(32);
        for word in &mut words {
            let any = miss(word);
            if _mm256_testz_si256(any, any) == 0 {
                return false;
            }
        }
        let tail = words.remainder();
        // The last 32 bytes of the span (overlapping the checked ones).
        tail.is_empty() || {
            let any = miss(&bytes[bytes.len() - 32..]);
            _mm256_testz_si256(any, any) != 0
        }
    }

    /// Nonzero bytes mark the bytes of `at[..32]` outside the alphabet.
    #[inline]
    #[target_feature(enable = "avx2")]
    fn miss(table: __m256i, at: &[u8]) -> __m256i {
        assert!(at.len() >= 32);
        // SAFETY: `at` holds at least 32 bytes; loadu permits any alignment.
        let x = unsafe { _mm256_loadu_si256(at.as_ptr().cast()) };
        _mm256_xor_si256(
            _mm256_shuffle_epi8(table, _mm256_and_si256(x, _mm256_set1_epi8(0x0F))),
            x,
        )
    }

    #[cfg(test)]
    mod tests {
        /// Every byte value, at every position of spans across block boundaries.
        #[test]
        fn avx2_certificate_matches_the_scalar_check() {
            if !std::is_x86_feature_detected!("avx2") {
                return;
            }
            for len in [64usize, 65, 95, 96, 127, 128, 129, 200, 255, 256, 300] {
                let clean: Vec<u8> = (0..len).map(|t| b"ACGT"[(t * 7 + t / 5) % 4]).collect();
                // SAFETY: AVX2 was detected above.
                assert!(unsafe { super::all_acgt(&clean) }, "len {len}");
                for byte in 0..=u8::MAX {
                    for at in [0, len / 3, len - 33, len - 1] {
                        let mut span = clean.clone();
                        span[at] = byte;
                        let expected = matches!(byte, b'A' | b'C' | b'G' | b'T');
                        // SAFETY: as above.
                        assert_eq!(
                            unsafe { super::all_acgt(&span) },
                            expected,
                            "{len} {at} {byte}"
                        );
                    }
                }
            }
        }
    }
}
