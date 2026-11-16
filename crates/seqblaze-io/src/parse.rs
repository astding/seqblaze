//! Streaming FASTA/FASTQ parser over decompressed text (spec §2.3–§7).
//!
//! [`Parser::feed`] takes text in pieces of any size; complete lines are parsed
//! where they lie, and only a line cut by a piece boundary is copied. Records go
//! straight into Arrow-layout columns (one byte buffer plus int64 offsets each).
//! [`Parser::finish`] handles the end of the text. Batch boundaries depend only
//! on the record sequence and [`Limits`] (§7), never on the piece sizes.
use crate::{
    error::{FormatError, ReadError, Reason},
    lines::find_newline,
};
use seqblaze_core::SeqBatch;
use std::collections::VecDeque;

/// The format named by the function (§1): nothing is detected from content.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Format {
    Fasta,
    Fastq,
}

impl Format {
    pub fn name(self) -> &'static str {
        match self {
            Format::Fasta => "fasta",
            Format::Fastq => "fastq",
        }
    }
}

/// §7 chunk limits; `None` is unlimited. `whole` makes the end of the text emit
/// one batch even when it is empty (the `read_*` result).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Limits {
    pub records: Option<u64>,
    pub bytes: Option<u64>,
    pub whole: bool,
}

/// One emitted batch: `offset` is the file-wide ordinal of row 0.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Batch {
    pub offset: u64,
    pub id: SeqBatch,
    pub description: SeqBatch,
    pub seq: SeqBatch,
    pub qual: Option<SeqBatch>,
}

impl Batch {
    pub fn len(&self) -> usize {
        self.seq.len()
    }
    pub fn is_empty(&self) -> bool {
        self.seq.is_empty()
    }
}

/// §5 separator bytes: space, tab, `\v`, `\f`, `\r`.
#[inline]
fn is_separator(byte: u8) -> bool {
    matches!(byte, b' ' | b'\t' | 0x0b | 0x0c | b'\r')
}

/// §5: (id, description) of a header text.
pub fn split_header(text: &[u8]) -> (&[u8], &[u8]) {
    match text.iter().position(|&b| is_separator(b)) {
        None => (text, &[]),
        Some(start) => {
            let end = text[start..]
                .iter()
                .position(|&b| !is_separator(b))
                .map_or(text.len(), |k| start + k);
            (&text[..start], &text[end..])
        }
    }
}

fn reserve(data: &mut Vec<u8>, additional: usize) -> Result<(), ReadError> {
    if data.capacity() - data.len() < additional {
        data.try_reserve(additional).map_err(|_| ReadError::Alloc {
            bytes: data.len().saturating_add(additional),
        })?;
    }
    Ok(())
}

/// One Arrow large-layout column under construction.
#[derive(Debug)]
struct Column {
    data: Vec<u8>,
    offsets: Vec<i64>,
}

impl Column {
    fn new() -> Self {
        Self {
            data: Vec::new(),
            offsets: vec![0],
        }
    }

    #[inline]
    fn extend(&mut self, bytes: &[u8]) -> Result<(), ReadError> {
        reserve(&mut self.data, bytes.len())?;
        self.data.extend_from_slice(bytes);
        Ok(())
    }

    #[inline]
    fn close_row(&mut self) -> Result<(), ReadError> {
        if self.offsets.len() == self.offsets.capacity() {
            self.offsets.try_reserve(1).map_err(|_| ReadError::Alloc {
                bytes: (self.offsets.len() + 1).saturating_mul(8),
            })?;
        }
        // A Vec never holds more than isize::MAX bytes, so this fits in i64.
        self.offsets.push(self.data.len() as i64);
        Ok(())
    }

    /// Reserve, as a hint, about the lengths of the column of the batch that
    /// just closed (`data` capped at `cap` bytes), so that the next batch does
    /// not regrow from empty. A refused hint is ignored: the growth itself
    /// reserves fallibly.
    fn seed(&mut self, previous: &Column, cap: Option<u64>) {
        let len = previous.data.len();
        let mut want = len.saturating_add(len / 16);
        if let Some(cap) = cap {
            want = want.min(usize::try_from(cap).unwrap_or(usize::MAX));
        }
        let _ = self.data.try_reserve_exact(want);
        let _ = self.offsets.try_reserve_exact(previous.offsets.len());
    }

    /// `exact`: shrink to the length (the `read_*` result). Otherwise a
    /// buffer is shrunk only when its spare capacity exceeds 1/8 of its
    /// length, so an emitted `iter_*` batch keeps at most that slack (§9.3).
    fn into_batch(mut self, exact: bool) -> SeqBatch {
        if exact || self.data.capacity() - self.data.len() > self.data.len() / 8 {
            self.data.shrink_to_fit();
        }
        if exact || self.offsets.capacity() - self.offsets.len() > self.offsets.len() / 8 {
            self.offsets.shrink_to_fit();
        }
        SeqBatch::try_from_parts(self.data, self.offsets)
            .expect("parser columns keep monotone offsets from 0 to the data length")
    }
}

/// The batch under construction.
#[derive(Debug)]
struct Builder {
    offset: u64,
    rows: u64,
    id: Column,
    description: Column,
    seq: Column,
    qual: Option<Column>,
}

impl Builder {
    fn new(offset: u64, format: Format) -> Self {
        Self {
            offset,
            rows: 0,
            id: Column::new(),
            description: Column::new(),
            seq: Column::new(),
            qual: (format == Format::Fastq).then(Column::new),
        }
    }

    /// The builder after `previous`, with capacities seeded from it.
    fn seeded(offset: u64, format: Format, previous: &Builder, cap: Option<u64>) -> Self {
        let mut next = Self::new(offset, format);
        next.id.seed(&previous.id, cap);
        next.description.seed(&previous.description, cap);
        next.seq.seed(&previous.seq, cap);
        if let (Some(q), Some(p)) = (next.qual.as_mut(), previous.qual.as_ref()) {
            q.seed(p, cap);
        }
        next
    }

    /// §7 payload: id + description + seq + qual bytes of the closed records.
    fn payload(&self) -> u64 {
        (self.id.data.len()
            + self.description.data.len()
            + self.seq.data.len()
            + self.qual.as_ref().map_or(0, |q| q.data.len())) as u64
    }

    fn into_batch(self, shrink: bool) -> Batch {
        Batch {
            offset: self.offset,
            id: self.id.into_batch(shrink),
            description: self.description.into_batch(shrink),
            seq: self.seq.into_batch(shrink),
            qual: self.qual.map(|q| q.into_batch(shrink)),
        }
    }
}

const BOM: &[u8] = b"\xef\xbb\xbf";
const FASTQ_SHAPE: &str = "A FASTQ record is exactly four lines: @header, sequence, + (empty or the header again), quality of the same length.";
const LONE_CR_HINT: &str = "The text contains a carriage return (\\r) that is not followed by \\n; only \\n and \\r\\n end lines, so a file with lone \\r line ends (classic Mac OS) is one long line.";
const BOM_HINT: &str =
    "The text starts with a UTF-8 byte-order mark (BOM, ef bb bf); remove it before reading.";
const GZIP_TWICE_HINT: &str = "The decompressed text starts with 1f 8b: it is itself gzip data (compressed twice); seqblaze decompresses once.";

fn has_lone_cr(line: &[u8]) -> bool {
    line.contains(&b'\r')
}

/// Streaming parser state.
#[derive(Debug)]
pub struct Parser {
    format: Format,
    limits: Limits,
    /// Complete lines parsed so far (the number of the last one).
    line: u64,
    /// The start of a line cut by a piece boundary.
    carry: Vec<u8>,
    current: Builder,
    ready: VecDeque<Batch>,
    /// Closed records, file-wide.
    records: u64,
    /// FASTA: a header line has been seen (a record is open).
    open: bool,
    /// FASTQ: lines of the current record seen (0 = between records).
    phase: u8,
    /// FASTQ: header text of the current record (for the `+` comparison).
    header: Vec<u8>,
    header_line: u64,
    seq_len: usize,
    /// FASTQ: the first empty line seen between records, pending (P3).
    blank: Option<u64>,
    /// The decompressed text starts with 1f 8b (§8.1 hint).
    gzip_inside: bool,
    finished: bool,
}

impl Parser {
    pub fn new(format: Format, limits: Limits) -> Self {
        Self {
            format,
            limits,
            line: 0,
            carry: Vec::new(),
            current: Builder::new(0, format),
            ready: VecDeque::new(),
            records: 0,
            open: false,
            phase: 0,
            header: Vec::new(),
            header_line: 0,
            seq_len: 0,
            blank: None,
            gzip_inside: false,
            finished: false,
        }
    }

    pub fn format(&self) -> Format {
        self.format
    }

    /// Mark that the decompressed text starts with the gzip magic (G8 hint).
    pub fn set_gzip_inside(&mut self) {
        self.gzip_inside = true;
    }

    /// The next emitted batch, in order.
    pub fn pop(&mut self) -> Option<Batch> {
        self.ready.pop_front()
    }

    pub fn has_ready(&self) -> bool {
        !self.ready.is_empty()
    }

    #[cfg(test)]
    pub(crate) fn ready_len(&self) -> usize {
        self.ready.len()
    }

    /// Records closed so far (file-wide).
    pub fn records(&self) -> u64 {
        self.records
    }

    /// (record, line) of the failure point for a container error (§8.1): the
    /// record being read and the line that holds the next unparsed byte.
    pub fn position(&self) -> (u64, u64) {
        let in_record = match self.format {
            Format::Fasta => self.open,
            Format::Fastq => self.phase > 0 || !self.carry.is_empty(),
        };
        let record = if in_record {
            self.records
        } else {
            self.records.saturating_sub(1)
        };
        (record, self.line + 1)
    }

    /// Parse the complete lines of `text` (after any carried partial line) and
    /// return the number of bytes consumed. Parsing stops right after a line
    /// that emits a batch (§7 limits; never in `whole` mode), so the caller can
    /// hand that batch out before more is parsed (§9.3); the rest of `text`
    /// must be fed again. Otherwise every byte is consumed and an incomplete
    /// last line is carried.
    #[must_use = "bytes after a batch emission are not consumed"]
    pub fn feed(&mut self, text: &[u8]) -> Result<usize, ReadError> {
        debug_assert!(!self.finished);
        let ready = self.ready.len();
        let mut rest = text;
        if !self.carry.is_empty() {
            match find_newline(rest) {
                None => {
                    reserve(&mut self.carry, rest.len())?;
                    self.carry.extend_from_slice(rest);
                    return Ok(text.len());
                }
                Some(end) => {
                    reserve(&mut self.carry, end)?;
                    self.carry.extend_from_slice(&rest[..end]);
                    let mut line = std::mem::take(&mut self.carry);
                    let result = self.terminated(&line);
                    line.clear();
                    self.carry = line;
                    result?;
                    rest = &rest[end + 1..];
                    if self.ready.len() > ready {
                        return Ok(text.len() - rest.len());
                    }
                }
            }
        }
        while let Some(end) = find_newline(rest) {
            self.terminated(&rest[..end])?;
            rest = &rest[end + 1..];
            if self.ready.len() > ready {
                return Ok(text.len() - rest.len());
            }
        }
        if !rest.is_empty() {
            reserve(&mut self.carry, rest.len())?;
            self.carry.extend_from_slice(rest);
        }
        Ok(text.len())
    }

    /// The end of the text: the unterminated last line, end-of-input checks and
    /// the final batch.
    pub fn finish(&mut self) -> Result<(), ReadError> {
        if self.finished {
            return Ok(());
        }
        if !self.carry.is_empty() {
            // Unterminated: a trailing \r is content (§2.3).
            let line = std::mem::take(&mut self.carry);
            self.line_content(&line)?;
        }
        match self.format {
            Format::Fasta => {
                if self.open {
                    self.close_record()?;
                    self.open = false;
                }
            }
            Format::Fastq => {
                if self.phase > 0 {
                    let missing = self.header_line + u64::from(self.phase);
                    let mut hints = Vec::new();
                    if has_lone_cr(&self.header) {
                        hints.push(LONE_CR_HINT.to_string());
                    }
                    return Err(self.error(
                        Reason::TruncatedRecord,
                        self.records,
                        missing,
                        format!("the input ends inside the record; line {missing} is missing. {FASTQ_SHAPE}"),
                        hints,
                    ));
                }
            }
        }
        self.finished = true;
        if self.current.rows > 0 || self.limits.whole {
            let next = Builder::new(self.records, self.format);
            let batch = std::mem::replace(&mut self.current, next);
            self.ready.push_back(batch.into_batch(self.limits.whole));
        }
        Ok(())
    }

    fn error(
        &self,
        reason: Reason,
        record: u64,
        line: u64,
        detail: String,
        mut hints: Vec<String>,
    ) -> ReadError {
        if self.gzip_inside && line == 1 && !reason.is_container() {
            hints.push(GZIP_TWICE_HINT.to_string());
        }
        ReadError::Format(FormatError {
            reason,
            record,
            line,
            detail,
            hints,
        })
    }

    #[inline]
    fn terminated(&mut self, raw: &[u8]) -> Result<(), ReadError> {
        // One \r directly before the \n belongs to the line end (§2.3).
        let content = match raw.last() {
            Some(b'\r') => &raw[..raw.len() - 1],
            _ => raw,
        };
        self.line_content(content)
    }

    #[inline]
    fn line_content(&mut self, content: &[u8]) -> Result<(), ReadError> {
        self.line += 1;
        match self.format {
            Format::Fasta => self.fasta_line(content),
            Format::Fastq => self.fastq_line(content),
        }
    }

    fn fasta_line(&mut self, content: &[u8]) -> Result<(), ReadError> {
        if content.first() == Some(&b'>') {
            if self.open {
                self.close_record()?;
            }
            self.open_record(&content[1..])?;
            self.open = true;
        } else if self.open {
            self.current.seq.extend(content)?;
        } else if !content.is_empty() {
            return Err(self.content_before_header(content));
        }
        Ok(())
    }

    fn content_before_header(&self, content: &[u8]) -> ReadError {
        let mut hints = Vec::new();
        if content.starts_with(BOM) {
            hints.push(BOM_HINT.to_string());
        }
        if content.first() == Some(&b'@') {
            hints.push(
                "The text looks like FASTQ (its first line starts with @); use sb.read_fastq or sb.iter_fastq."
                    .to_string(),
            );
        }
        if has_lone_cr(content) {
            hints.push(LONE_CR_HINT.to_string());
        }
        self.error(
            Reason::ContentBeforeHeader,
            0,
            self.line,
            "this line comes before the first header line (one starting with >) and is not empty; FASTA text must start with a header line".to_string(),
            hints,
        )
    }

    fn fastq_line(&mut self, content: &[u8]) -> Result<(), ReadError> {
        let n = self.line;
        match self.phase {
            0 => {
                if content.is_empty() {
                    // Empty lines are accepted only after the last record (P3).
                    self.blank.get_or_insert(n);
                    return Ok(());
                }
                if let Some(blank) = self.blank {
                    return Err(self.error(
                        Reason::ExpectedHeader,
                        self.records,
                        blank,
                        format!("this line is empty, but a record follows at line {n}; empty lines are accepted only after the last record. {FASTQ_SHAPE}"),
                        Vec::new(),
                    ));
                }
                if content[0] != b'@' {
                    let mut hints = Vec::new();
                    if content.starts_with(BOM) {
                        hints.push(BOM_HINT.to_string());
                    }
                    if content[0] == b'>' && self.records == 0 {
                        hints.push(
                            "The text looks like FASTA (its first line starts with >); use sb.read_fasta or sb.iter_fasta."
                                .to_string(),
                        );
                    }
                    return Err(self.error(
                        Reason::ExpectedHeader,
                        self.records,
                        n,
                        format!(
                            "a FASTQ record must start with a line beginning with @. {FASTQ_SHAPE}"
                        ),
                        hints,
                    ));
                }
                let text = &content[1..];
                self.header.clear();
                reserve(&mut self.header, text.len())?;
                self.header.extend_from_slice(text);
                self.open_record(text)?;
                self.header_line = n;
                self.phase = 1;
            }
            1 => {
                self.current.seq.extend(content)?;
                self.seq_len = content.len();
                self.phase = 2;
            }
            2 => {
                if content.first() != Some(&b'+') {
                    return Err(self.error(
                        Reason::ExpectedPlus,
                        self.records,
                        n,
                        format!("the third line of a record must start with +. {FASTQ_SHAPE} Wrapped (multi-line) FASTQ is not read."),
                        Vec::new(),
                    ));
                }
                let plus = &content[1..];
                if !plus.is_empty() && plus != self.header.as_slice() {
                    return Err(self.error(
                        Reason::PlusMismatch,
                        self.records,
                        n,
                        format!("the text after + is neither empty nor identical to the header text (byte for byte). {FASTQ_SHAPE}"),
                        Vec::new(),
                    ));
                }
                self.phase = 3;
            }
            _ => {
                if content.len() != self.seq_len {
                    return Err(self.error(
                        Reason::LengthMismatch,
                        self.records,
                        n,
                        format!(
                            "the sequence has {} bytes but the quality has {}. {FASTQ_SHAPE}",
                            self.seq_len,
                            content.len()
                        ),
                        Vec::new(),
                    ));
                }
                let qual = self
                    .current
                    .qual
                    .as_mut()
                    .expect("FASTQ builders hold a qual column");
                qual.extend(content)?;
                qual.close_row()?;
                self.phase = 0;
                self.close_record()?;
            }
        }
        Ok(())
    }

    /// Start a record: its id and description rows are complete at once.
    fn open_record(&mut self, header: &[u8]) -> Result<(), ReadError> {
        let (id, description) = split_header(header);
        self.current.id.extend(id)?;
        self.current.id.close_row()?;
        self.current.description.extend(description)?;
        self.current.description.close_row()
    }

    /// Close the open record (its `seq`, and `qual` already for FASTQ) and emit
    /// the batch when a limit is reached (§7).
    fn close_record(&mut self) -> Result<(), ReadError> {
        self.current.seq.close_row()?;
        self.current.rows += 1;
        self.records += 1;
        let full = self.limits.records.is_some_and(|n| self.current.rows >= n)
            || self
                .limits
                .bytes
                .is_some_and(|n| self.current.payload() >= n);
        if full {
            let next = Builder::seeded(self.records, self.format, &self.current, self.limits.bytes);
            let batch = std::mem::replace(&mut self.current, next);
            self.ready.push_back(batch.into_batch(false));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rows(batch: &Batch) -> Vec<Vec<Vec<u8>>> {
        (0..batch.len())
            .map(|i| {
                let mut row = vec![
                    batch.id.get(i).to_vec(),
                    batch.description.get(i).to_vec(),
                    batch.seq.get(i).to_vec(),
                ];
                if let Some(q) = &batch.qual {
                    row.push(q.get(i).to_vec());
                }
                row
            })
            .collect()
    }

    fn parse(format: Format, text: &[u8], piece: usize) -> Result<Vec<Batch>, ReadError> {
        let mut parser = Parser::new(
            format,
            Limits {
                whole: true,
                ..Limits::default()
            },
        );
        for chunk in text.chunks(piece.max(1)) {
            assert_eq!(parser.feed(chunk)?, chunk.len());
        }
        parser.finish()?;
        Ok(std::iter::from_fn(|| parser.pop()).collect())
    }

    fn triple(error: ReadError) -> (&'static str, u64, u64) {
        match error {
            ReadError::Format(e) => (e.reason.code(), e.record, e.line),
            other => panic!("{other}"),
        }
    }

    #[test]
    fn fasta_lines_and_records_for_every_piece_size() {
        let text = b"\n\r\n>a desc\r\nAC\rGT\r\n\nGG\n>b\n>c  x \nTT\r";
        for piece in 1..text.len() + 1 {
            let batches = parse(Format::Fasta, text, piece).unwrap();
            assert_eq!(batches.len(), 1);
            assert_eq!(
                rows(&batches[0]),
                vec![
                    vec![b"a".to_vec(), b"desc".to_vec(), b"AC\rGTGG".to_vec()],
                    vec![b"b".to_vec(), vec![], vec![]],
                    vec![b"c".to_vec(), b"x ".to_vec(), b"TT\r".to_vec()],
                ],
                "piece {piece}"
            );
        }
    }

    #[test]
    fn fastq_errors_follow_the_step_order() {
        type Case = (&'static [u8], (&'static str, u64, u64));
        let cases: [Case; 6] = [
            (
                b"@r1\nAC\n+\nII\n\n@r2\nAC\n+\nII\n",
                ("expected_header", 1, 5),
            ),
            (b"@r1\nACGT\n+\nIII\n", ("length_mismatch", 0, 4)),
            (b"@r1\nAC\n+", ("truncated_record", 0, 4)),
            (b"@r1\rAC\r+\rII\r", ("truncated_record", 0, 2)),
            (b"@r1\nAC\nGT\n+\nII\nII\n", ("expected_plus", 0, 3)),
            (b"@r1 d \nAC\n+r1 d\nII\n", ("plus_mismatch", 0, 3)),
        ];
        for (text, want) in cases {
            for piece in [1, 2, 3, 1000] {
                assert_eq!(triple(parse(Format::Fastq, text, piece).unwrap_err()), want);
            }
        }
        // \r\n on the sequence line and \n on the quality line: equal lengths.
        let ok = parse(Format::Fastq, b"@r\nAC\r\n+\nII\n\n\n\r\n", 2).unwrap();
        assert_eq!(
            rows(&ok[0]),
            vec![vec![b"r".to_vec(), vec![], b"AC".to_vec(), b"II".to_vec()]]
        );
    }

    #[test]
    fn chunk_limits_emit_after_the_closing_record() {
        let mut parser = Parser::new(
            Format::Fasta,
            Limits {
                records: None,
                bytes: Some(1),
                whole: false,
            },
        );
        let text = b">\n>\n>a\nAC\n>b\n>c\n";
        // The batch closes only once the record after it begins, and parsing
        // stops right after that header line.
        let used = parser.feed(text).unwrap();
        assert_eq!(used, b">\n>\n>a\nAC\n>b\n".len());
        let first = parser.pop().unwrap();
        assert_eq!((first.offset, first.len()), (0, 3));
        assert!(parser.pop().is_none());
        // `>c` closes `b` (payload 1 byte), which reaches the limit again.
        assert_eq!(parser.feed(&text[used..]).unwrap(), 3);
        let second = parser.pop().unwrap();
        assert_eq!((second.offset, second.len()), (3, 1));
        parser.finish().unwrap();
        let last = parser.pop().unwrap();
        assert_eq!((last.offset, last.len()), (4, 1));
    }

    #[test]
    fn feed_stops_after_each_emitted_batch() {
        let text = b">a\nA\n".repeat(50);
        let mut parser = Parser::new(
            Format::Fasta,
            Limits {
                records: Some(1),
                bytes: None,
                whole: false,
            },
        );
        let mut rest = &text[..];
        let mut emitted = Vec::new();
        while !rest.is_empty() {
            let used = parser.feed(rest).unwrap();
            assert!(used > 0);
            rest = &rest[used..];
            // At most one batch waits after each feed.
            assert!(parser.ready.len() <= 1);
            emitted.extend(std::iter::from_fn(|| parser.pop()));
        }
        parser.finish().unwrap();
        emitted.extend(std::iter::from_fn(|| parser.pop()));
        assert_eq!(emitted.len(), 50);
        for (k, batch) in emitted.iter().enumerate() {
            assert_eq!((batch.offset, batch.len()), (k as u64, 1));
            assert_eq!(batch.seq.get(0), b"A");
        }
    }

    #[test]
    fn header_split() {
        assert_eq!(
            split_header(b"a\x0bdesc\x0cmore"),
            (&b"a"[..], &b"desc\x0cmore"[..])
        );
        assert_eq!(split_header(b"  a desc"), (&b""[..], &b"a desc"[..]));
        assert_eq!(split_header(b"a   "), (&b"a"[..], &b""[..]));
        assert_eq!(
            split_header(b"a\xc2\xa0desc"),
            (&b"a\xc2\xa0desc"[..], &b""[..])
        );
    }
}
