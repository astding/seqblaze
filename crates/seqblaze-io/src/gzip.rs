//! gzip members (RFC 1952) and BGZF (SAMv1), member by member (spec §2.2).
//!
//! Headers and trailers are parsed here; the deflate data goes through
//! `flate2::Decompress` (raw deflate, miniz_oxide backend) so that member
//! boundaries, trailing bytes and the BGZF subfield stay visible.

/// gzip ID1, ID2 (RFC 1952 §2.3.1).
pub const MAGIC: [u8; 2] = [0x1f, 0x8b];
/// BGZF extra subfield identifiers SI1, SI2 (`BC`), SAMv1.
const BGZF_SI: [u8; 2] = [66, 67];

const FHCRC: u8 = 0x02;
const FEXTRA: u8 = 0x04;
const FNAME: u8 = 0x08;
const FCOMMENT: u8 = 0x10;
const RESERVED: u8 = 0xe0;

/// A parsed member header.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Header {
    /// Header bytes, including the optional fields.
    pub len: usize,
    /// The FEXTRA field carries the BGZF subfield (SI1 66, SI2 67, SLEN 2).
    pub bgzf: bool,
}

/// Why a header was rejected (`gzip_corrupt`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HeaderError {
    Method(u8),
    ReservedFlags(u8),
    HeaderCrc,
}

impl std::fmt::Display for HeaderError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            HeaderError::Method(cm) => write!(
                f,
                "a gzip member header has compression method {cm}; RFC 1952 defines only 8 (deflate)"
            ),
            HeaderError::ReservedFlags(flg) => write!(
                f,
                "a gzip member header sets reserved flag bits (FLG = {flg:#04x}; bits 5-7 must be zero)"
            ),
            HeaderError::HeaderCrc => write!(f, "a gzip member header fails its FHCRC check"),
        }
    }
}

fn le16(bytes: &[u8]) -> usize {
    usize::from(u16::from_le_bytes([bytes[0], bytes[1]]))
}

/// Parse the member header at the start of `b` (which begins with [`MAGIC`]).
/// `Ok(None)` when more bytes are needed.
pub fn parse_header(b: &[u8]) -> Result<Option<Header>, HeaderError> {
    if b.len() < 10 {
        return Ok(None);
    }
    debug_assert_eq!(b[..2], MAGIC);
    if b[2] != 8 {
        return Err(HeaderError::Method(b[2]));
    }
    let flags = b[3];
    if flags & RESERVED != 0 {
        return Err(HeaderError::ReservedFlags(flags));
    }
    let mut p = 10;
    let mut bgzf = false;
    if flags & FEXTRA != 0 {
        if b.len() < p + 2 {
            return Ok(None);
        }
        let xlen = le16(&b[p..]);
        let start = p + 2;
        if b.len() < start + xlen {
            return Ok(None);
        }
        let extra = &b[start..start + xlen];
        let mut q = 0;
        while q + 4 <= extra.len() {
            let len = le16(&extra[q + 2..]);
            if extra[q..q + 2] == BGZF_SI && len == 2 && q + 4 + len <= extra.len() {
                bgzf = true;
            }
            q += 4 + len;
        }
        p = start + xlen;
    }
    for flag in [FNAME, FCOMMENT] {
        if flags & flag != 0 {
            match b[p..].iter().position(|&x| x == 0) {
                Some(k) => p += k + 1,
                None => return Ok(None),
            }
        }
    }
    if flags & FHCRC != 0 {
        if b.len() < p + 2 {
            return Ok(None);
        }
        let mut crc = flate2::Crc::new();
        crc.update(&b[..p]);
        if (crc.sum() & 0xffff) as usize != le16(&b[p..]) {
            return Err(HeaderError::HeaderCrc);
        }
        p += 2;
    }
    Ok(Some(Header { len: p, bgzf }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn headers() {
        let eof = [
            0x1f, 0x8b, 8, 4, 0, 0, 0, 0, 0, 0xff, 6, 0, 0x42, 0x43, 2, 0, 0x1b, 0, 3, 0, 0, 0, 0,
            0, 0, 0, 0, 0,
        ];
        assert_eq!(
            parse_header(&eof).unwrap(),
            Some(Header {
                len: 18,
                bgzf: true
            })
        );
        assert_eq!(parse_header(&eof[..12]).unwrap(), None);
        let mut plain = [0x1f, 0x8b, 8, 0, 0, 0, 0, 0, 0, 3];
        assert_eq!(
            parse_header(&plain).unwrap(),
            Some(Header {
                len: 10,
                bgzf: false
            })
        );
        plain[3] = 0x20;
        assert_eq!(parse_header(&plain), Err(HeaderError::ReservedFlags(0x20)));
        let named = [0x1f, 0x8b, 8, FNAME, 0, 0, 0, 0, 0, 3, b'x', b'y'];
        assert_eq!(parse_header(&named).unwrap(), None);
        let mut with_crc = vec![0x1f, 0x8b, 8, FNAME | FHCRC, 0, 0, 0, 0, 0, 3, b'x', 0];
        let mut crc = flate2::Crc::new();
        crc.update(&with_crc);
        with_crc.extend_from_slice(&((crc.sum() & 0xffff) as u16).to_le_bytes());
        assert_eq!(
            parse_header(&with_crc).unwrap(),
            Some(Header {
                len: 14,
                bgzf: false
            })
        );
        with_crc[12] ^= 1;
        assert_eq!(parse_header(&with_crc), Err(HeaderError::HeaderCrc));
    }
}
