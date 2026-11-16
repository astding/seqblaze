//! FASTA/FASTQ readers (D92, docs/wiki/io/fasta-readers.md).
//!
//! The text model is the spec's: `\n` ends a line, one `\r` directly before it
//! belongs to the line end, every other byte is content (§2.3). Parsing decides
//! nothing and repairs nothing; malformed input is a [`FormatError`] naming the
//! 0-based record and the 1-based line. gzip is recognised by its magic bytes
//! and read member by member (multi-member and BGZF, §2.2).
//!
//! Layers: [`input::Input`] (a borrowed buffer or a `Read`), [`gzip`] (RFC 1952
//! members over `flate2::Decompress`), [`parse::Parser`] (streaming lines into
//! Arrow-layout columns) and [`reader::FastxReader`] (sniffing and driving).

pub mod error;
pub mod gzip;
pub mod input;
pub mod lines;
pub mod parse;
pub mod reader;

pub use error::{FormatError, ReadError, Reason};
pub use input::Input;
pub use parse::{Batch, Format, Limits, Parser};
pub use reader::FastxReader;
