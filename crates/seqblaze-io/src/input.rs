//! Raw input bytes: a borrowed buffer (zero-copy) or a [`Read`] source read
//! once, in order, into a reusable buffer.
use crate::error::ReadError;
use std::{any::Any, io::Read};

/// Initial read buffer for `Read` sources (spec §16: about 1 MiB). It grows only
/// when one gzip header does not fit, and fallibly (D86).
pub const READ_BUFFER: usize = 1 << 20;

/// Grow `buf` to `len` zeroed bytes, reserving fallibly.
fn grow(buf: &mut Vec<u8>, len: usize) -> Result<(), ReadError> {
    buf.try_reserve_exact(len - buf.len())
        .map_err(|_| ReadError::Alloc { bytes: len })?;
    buf.resize(len, 0);
    Ok(())
}

enum Kind {
    /// Borrowed bytes kept alive by `_owner`.
    Slice {
        ptr: *const u8,
        len: usize,
        _owner: Box<dyn Any + Send>,
    },
    Reader {
        reader: Box<dyn Read + Send>,
        buf: Vec<u8>,
        end: usize,
        eof: bool,
    },
}

/// Bytes of the source not yet consumed, refilled on demand.
pub struct Input {
    kind: Kind,
    pos: usize,
}

// SAFETY: the raw slice pointer is only read, and its owner (Send) keeps the
// allocation alive and unmodified for the life of this value (contract of
// `Input::borrowed`); the Reader variant holds only Send values.
unsafe impl Send for Input {}

impl Input {
    /// Borrow `len` bytes at `ptr` without copying.
    ///
    /// # Safety
    /// `owner` must keep the `len` bytes at `ptr` alive, and nobody may modify
    /// them, until this `Input` is dropped.
    pub unsafe fn borrowed(ptr: *const u8, len: usize, owner: Box<dyn Any + Send>) -> Self {
        Self {
            kind: Kind::Slice {
                ptr,
                len,
                _owner: owner,
            },
            pos: 0,
        }
    }

    /// An owned in-memory input (tests and Rust callers).
    pub fn from_vec(data: Vec<u8>) -> Self {
        let ptr = data.as_ptr();
        let len = data.len();
        // SAFETY: the Vec is moved into the owner box; its heap buffer does not
        // move or change while the box lives.
        unsafe { Self::borrowed(ptr, len, Box::new(data)) }
    }

    /// A `Read` source, read from its current position to its end.
    pub fn reader(reader: Box<dyn Read + Send>) -> Self {
        Self {
            kind: Kind::Reader {
                reader,
                buf: Vec::new(),
                end: 0,
                eof: false,
            },
            pos: 0,
        }
    }

    /// The bytes available now.
    #[inline]
    pub fn available(&self) -> &[u8] {
        match &self.kind {
            Kind::Slice { ptr, len, .. } => {
                // SAFETY: `borrowed`'s contract keeps the bytes alive and unchanged.
                let all = unsafe { std::slice::from_raw_parts(*ptr, *len) };
                &all[self.pos..]
            }
            Kind::Reader { buf, end, .. } => &buf[self.pos..*end],
        }
    }

    #[inline]
    pub fn consume(&mut self, n: usize) {
        debug_assert!(n <= self.available().len());
        self.pos += n;
    }

    /// Read more bytes, keeping the unconsumed ones. `Ok(false)` at the end of
    /// the source (always for a borrowed buffer). A refused buffer growth is
    /// [`ReadError::Alloc`].
    pub fn fill(&mut self) -> Result<bool, ReadError> {
        let Kind::Reader {
            reader,
            buf,
            end,
            eof,
        } = &mut self.kind
        else {
            return Ok(false);
        };
        if *eof {
            return Ok(false);
        }
        if buf.is_empty() {
            grow(buf, READ_BUFFER)?;
        }
        if self.pos > 0 {
            buf.copy_within(self.pos..*end, 0);
            *end -= self.pos;
            self.pos = 0;
        }
        if *end == buf.len() {
            let grown = buf
                .len()
                .checked_mul(2)
                .ok_or(ReadError::Alloc { bytes: usize::MAX })?;
            grow(buf, grown)?;
        }
        loop {
            match reader.read(&mut buf[*end..]) {
                Ok(0) => {
                    *eof = true;
                    return Ok(false);
                }
                Ok(n) => {
                    *end += n;
                    return Ok(true);
                }
                Err(error) if error.kind() == std::io::ErrorKind::Interrupted => continue,
                Err(error) => return Err(ReadError::Io(error)),
            }
        }
    }

    /// Fill until at least `n` bytes are available or the source ends.
    pub fn want(&mut self, n: usize) -> Result<(), ReadError> {
        while self.available().len() < n {
            if !self.fill()? {
                break;
            }
        }
        Ok(())
    }
}
