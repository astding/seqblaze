//! `SeqBatch`: a list of sequences stored as one contiguous byte buffer plus
//! offsets (the Arrow `LargeBinary` layout), so a batch can be handed to SIMD
//! kernels or copied to a GPU in one transfer.

/// Contiguous storage for many byte sequences.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct SeqBatch {
    data: Vec<u8>,
    /// `offsets.len() == len() + 1`; sequence `i` is `data[offsets[i]..offsets[i + 1]]`.
    offsets: Vec<i64>,
}

impl SeqBatch {
    /// Build a batch by copying each sequence once.
    pub fn from_seqs<I, S>(seqs: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: AsRef<[u8]>,
    {
        let mut data = Vec::new();
        let mut offsets = vec![0i64];
        for s in seqs {
            data.extend_from_slice(s.as_ref());
            offsets.push(data.len() as i64);
        }
        Self { data, offsets }
    }

    /// Take ownership of checked offset storage without copying sequence bytes.
    pub fn try_from_parts(data: Vec<u8>, offsets: Vec<i64>) -> Result<Self, &'static str> {
        if offsets.first() != Some(&0)
            || offsets.last().copied() != i64::try_from(data.len()).ok()
            || offsets.windows(2).any(|pair| pair[0] > pair[1])
        {
            return Err("invalid sequence offsets: expected monotone 0..data.len() int64 offsets");
        }
        Ok(Self { data, offsets })
    }

    /// Number of sequences.
    pub fn len(&self) -> usize {
        self.offsets.len() - 1
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Sequence `i` as a byte slice.
    #[inline]
    pub fn get(&self, i: usize) -> &[u8] {
        let (a, b) = (self.offsets[i] as usize, self.offsets[i + 1] as usize);
        &self.data[a..b]
    }

    /// Length of sequence `i`.
    #[inline]
    pub fn seq_len(&self, i: usize) -> usize {
        (self.offsets[i + 1] - self.offsets[i]) as usize
    }

    /// The raw byte buffer.
    pub fn data(&self) -> &[u8] {
        &self.data
    }

    /// The offsets array (`len() + 1` entries, starting at 0).
    pub fn offsets(&self) -> &[i64] {
        &self.offsets
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip() {
        let seqs = ["ACGT", "", "GGGCCC"];
        let b = SeqBatch::from_seqs(seqs);
        assert_eq!(b.len(), 3);
        assert_eq!(b.get(0), b"ACGT");
        assert_eq!(b.get(1), b"");
        assert_eq!(b.get(2), b"GGGCCC");
        assert_eq!(b.offsets(), &[0, 4, 4, 10]);
        assert_eq!(b.seq_len(2), 6);
    }

    #[test]
    fn empty_batch() {
        let b = SeqBatch::from_seqs(Vec::<&[u8]>::new());
        assert!(b.is_empty());
        assert_eq!(b.offsets(), &[0]);
    }
}
