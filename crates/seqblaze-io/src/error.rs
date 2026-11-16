//! Reader errors: format errors with their §8.1 position, source I/O errors and
//! refused allocations.
use std::fmt;

/// The §8.1 reason codes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Reason {
    ContentBeforeHeader,
    ExpectedHeader,
    TruncatedRecord,
    ExpectedPlus,
    PlusMismatch,
    LengthMismatch,
    GzipCorrupt,
    GzipTruncated,
    GzipTrailingData,
    BgzfMissingEof,
    UnsupportedCompression,
}

impl Reason {
    /// The `.reason` string of `sb.FormatError`.
    pub fn code(self) -> &'static str {
        match self {
            Reason::ContentBeforeHeader => "content_before_header",
            Reason::ExpectedHeader => "expected_header",
            Reason::TruncatedRecord => "truncated_record",
            Reason::ExpectedPlus => "expected_plus",
            Reason::PlusMismatch => "plus_mismatch",
            Reason::LengthMismatch => "length_mismatch",
            Reason::GzipCorrupt => "gzip_corrupt",
            Reason::GzipTruncated => "gzip_truncated",
            Reason::GzipTrailingData => "gzip_trailing_data",
            Reason::BgzfMissingEof => "bgzf_missing_eof",
            Reason::UnsupportedCompression => "unsupported_compression",
        }
    }

    /// True for the §2.2 container reasons.
    pub fn is_container(self) -> bool {
        matches!(
            self,
            Reason::GzipCorrupt
                | Reason::GzipTruncated
                | Reason::GzipTrailingData
                | Reason::BgzfMissingEof
                | Reason::UnsupportedCompression
        )
    }
}

/// Malformed input: §8.1 reason, 0-based record, 1-based line, an explanation
/// and optional hints. The binding adds the source name and the closing sentence.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FormatError {
    pub reason: Reason,
    pub record: u64,
    pub line: u64,
    pub detail: String,
    pub hints: Vec<String>,
}

impl fmt::Display for FormatError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "record {} (0-based), line {}: {}",
            self.record, self.line, self.detail
        )?;
        for hint in &self.hints {
            write!(f, " {hint}")?;
        }
        Ok(())
    }
}

/// Every way a read can fail.
#[derive(Debug)]
pub enum ReadError {
    Format(FormatError),
    /// The source failed (path read error, or a file object's exception, which
    /// the binding stores and re-raises unchanged).
    Io(std::io::Error),
    /// A refused host reservation of `bytes` bytes (D86 allocation rule).
    Alloc {
        bytes: usize,
    },
}

impl From<FormatError> for ReadError {
    fn from(error: FormatError) -> Self {
        ReadError::Format(error)
    }
}

impl From<std::io::Error> for ReadError {
    fn from(error: std::io::Error) -> Self {
        ReadError::Io(error)
    }
}

impl fmt::Display for ReadError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ReadError::Format(error) => error.fmt(f),
            ReadError::Io(error) => error.fmt(f),
            ReadError::Alloc { bytes } => write!(f, "unable to reserve {bytes} bytes"),
        }
    }
}

impl std::error::Error for ReadError {}
