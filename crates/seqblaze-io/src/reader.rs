//! Driver: sniff the first bytes, decompress gzip members, feed the parser.
//!
//! [`FastxReader::step`] does a bounded amount of work so that the binding can
//! check for signals between steps and hand out batches as they close.
use crate::{
    error::{FormatError, ReadError, Reason},
    gzip::{self, Header},
    input::Input,
    parse::{Batch, Format, Limits, Parser},
};
use flate2::{Crc, Decompress, FlushDecompress, Status};

/// Plain text handed to the parser per step (borrowed inputs need no copy).
const PLAIN_STEP: usize = 4 << 20;
/// Decompressed bytes per inflate call.
const INFLATE_OUT: usize = 256 << 10;

/// Signatures of formats that are not read in v1 (P7).
const OTHER_SIGNATURES: [(&[u8], &str); 3] = [
    (b"BZh", "bzip2"),
    (b"\xfd7zXZ\x00", "xz"),
    (b"\x28\xb5\x2f\xfd", "zstd"),
];

#[derive(Clone, Copy)]
enum Member {
    /// Expecting a member header at the input position.
    Header,
    Body,
    Trailer,
    /// After a trailer: end of input, another member, or trailing data.
    Between,
}

/// What follows once the pending inflate output has been parsed.
enum After {
    /// The member's deflate stream ended: read the trailer.
    StreamEnd,
    /// Invalid deflate data (reported after the text before it, §8.1).
    Corrupt(String),
    /// The last call neither consumed nor produced: more input is needed.
    Stalled,
}

struct Gzip {
    inflate: Decompress,
    crc: Crc,
    out: Vec<u8>,
    /// Inflate output not yet parsed: `out[out_at..out_end]`. The parser stops
    /// after a line that emits a batch, so output can outlive one step.
    out_at: usize,
    out_end: usize,
    after: Option<After>,
    state: Member,
    /// The current member carries the BGZF subfield.
    member_bgzf: bool,
    /// The last finished member carries the BGZF subfield and decompressed to
    /// 0 bytes, or does not carry it (P6: no end check then).
    last_ok: bool,
    /// Decompressed bytes so far, up to two (the G8 hint).
    head: Vec<u8>,
}

enum Mode {
    Sniff,
    Plain,
    Gzip(Box<Gzip>),
    Done,
}

/// One read of one source.
pub struct FastxReader {
    input: Input,
    mode: Mode,
    parser: Parser,
}

impl FastxReader {
    pub fn new(input: Input, format: Format, limits: Limits) -> Self {
        Self {
            input,
            mode: Mode::Sniff,
            parser: Parser::new(format, limits),
        }
    }

    /// The next closed batch, in order.
    pub fn pop(&mut self) -> Option<Batch> {
        self.parser.pop()
    }

    pub fn has_ready(&self) -> bool {
        self.parser.has_ready()
    }

    pub fn is_done(&self) -> bool {
        matches!(self.mode, Mode::Done)
    }

    /// Run to the end of the input (no signal checks; Rust callers and tests).
    pub fn run(&mut self) -> Result<(), ReadError> {
        while self.step()? {}
        Ok(())
    }

    fn container(&self, reason: Reason, detail: String) -> ReadError {
        let (record, line) = match reason {
            Reason::UnsupportedCompression => (0, 1),
            _ => self.parser.position(),
        };
        ReadError::Format(FormatError {
            reason,
            record,
            line,
            detail,
            hints: Vec::new(),
        })
    }

    /// Do a bounded amount of work. `Ok(false)` once the whole input has been
    /// parsed (the last batch, if any, is then ready).
    pub fn step(&mut self) -> Result<bool, ReadError> {
        match &mut self.mode {
            Mode::Done => Ok(false),
            Mode::Sniff => {
                self.input.want(6)?;
                let head = self.input.available();
                if head.starts_with(&gzip::MAGIC) {
                    self.mode = Mode::Gzip(Box::new(Gzip {
                        inflate: Decompress::new(false),
                        crc: Crc::new(),
                        out: vec![0; INFLATE_OUT],
                        out_at: 0,
                        out_end: 0,
                        after: None,
                        state: Member::Header,
                        member_bgzf: false,
                        last_ok: true,
                        head: Vec::with_capacity(2),
                    }));
                } else if let Some((_, name)) = OTHER_SIGNATURES
                    .iter()
                    .find(|(sig, _)| head.starts_with(sig))
                {
                    return Err(self.container(
                        Reason::UnsupportedCompression,
                        format!("the input starts with the {name} signature; {name} input is not read in v1 (gzip and BGZF are). Decompress it first, for example with the {name} command-line tool"),
                    ));
                } else {
                    self.mode = Mode::Plain;
                }
                Ok(true)
            }
            Mode::Plain => {
                let available = self.input.available();
                if available.is_empty() {
                    if self.input.fill()? {
                        return Ok(true);
                    }
                    self.parser.finish()?;
                    self.mode = Mode::Done;
                    return Ok(false);
                }
                let n = available.len().min(PLAIN_STEP);
                let used = self.parser.feed(&available[..n])?;
                self.input.consume(used);
                Ok(true)
            }
            Mode::Gzip(_) => self.gzip_step(),
        }
    }

    fn gz(&mut self) -> &mut Gzip {
        match &mut self.mode {
            Mode::Gzip(gz) => gz,
            _ => unreachable!("gzip state outside gzip mode"),
        }
    }

    fn gzip_step(&mut self) -> Result<bool, ReadError> {
        let state = self.gz().state;
        match state {
            Member::Header => {
                let header = loop {
                    match gzip::parse_header(self.input.available()) {
                        Ok(Some(header)) => break header,
                        Ok(None) => {
                            if !self.input.fill()? {
                                return Err(self.container(
                                    Reason::GzipTruncated,
                                    "the input ends inside a gzip member header".into(),
                                ));
                            }
                        }
                        Err(error) => {
                            return Err(self.container(Reason::GzipCorrupt, error.to_string()))
                        }
                    }
                };
                let Header { len, bgzf } = header;
                self.input.consume(len);
                let gz = self.gz();
                gz.inflate.reset(false);
                gz.crc.reset();
                gz.member_bgzf = bgzf;
                gz.state = Member::Body;
                Ok(true)
            }
            Member::Body => self.inflate(),
            Member::Trailer => {
                self.input.want(8)?;
                let trailer = self.input.available();
                if trailer.len() < 8 {
                    return Err(self.truncated_member());
                }
                let crc = u32::from_le_bytes(trailer[..4].try_into().expect("4 bytes"));
                let isize = u32::from_le_bytes(trailer[4..8].try_into().expect("4 bytes"));
                let gz = self.gz();
                let (want_crc, want_size) = (gz.crc.sum(), gz.crc.amount());
                gz.last_ok = !gz.member_bgzf || gz.inflate.total_out() == 0;
                gz.state = Member::Between;
                if crc != want_crc || isize != want_size {
                    let what = if crc != want_crc { "CRC32" } else { "ISIZE" };
                    return Err(self.container(
                        Reason::GzipCorrupt,
                        format!("a gzip member's {what} does not match its decompressed data"),
                    ));
                }
                self.input.consume(8);
                Ok(true)
            }
            Member::Between => {
                self.input.want(2)?;
                let rest = self.input.available();
                if rest.is_empty() {
                    // P6: a stream whose last member is a BGZF block must end
                    // with the empty one; other members end no BGZF run.
                    if !self.gz().last_ok {
                        return Err(self.container(
                            Reason::BgzfMissingEof,
                            "the BGZF input does not end with an empty BGZF block (the SAMv1 end-of-file marker); the file may have been cut at a block boundary".into(),
                        ));
                    }
                    self.parser.finish()?;
                    self.mode = Mode::Done;
                    return Ok(false);
                }
                if rest.starts_with(&gzip::MAGIC) {
                    self.gz().state = Member::Header;
                    return Ok(true);
                }
                Err(self.container(
                    Reason::GzipTrailingData,
                    "bytes after a gzip member do not start another member (1f 8b); trailing data, NUL padding included, is not accepted".into(),
                ))
            }
        }
    }

    /// Parse pending inflate output, apply the outcome of the inflate call that
    /// produced it, or make one inflate call on the available input.
    fn inflate(&mut self) -> Result<bool, ReadError> {
        let Mode::Gzip(gz) = &mut self.mode else {
            unreachable!("inflate runs in gzip mode")
        };
        if gz.out_at < gz.out_end {
            let used = self.parser.feed(&gz.out[gz.out_at..gz.out_end])?;
            gz.out_at += used;
            return Ok(true);
        }
        // The text before a failure is parsed first (§8.1 order).
        match gz.after.take() {
            Some(After::StreamEnd) => {
                gz.state = Member::Trailer;
                return Ok(true);
            }
            Some(After::Corrupt(detail)) => {
                return Err(self.container(Reason::GzipCorrupt, detail));
            }
            Some(After::Stalled) => {
                if !self.input.fill()? {
                    return Err(self.truncated_member());
                }
                return Ok(true);
            }
            None => {}
        }
        if self.input.available().is_empty() && !self.input.fill()? {
            return Err(self.truncated_member());
        }
        let Mode::Gzip(gz) = &mut self.mode else {
            unreachable!("inflate runs in gzip mode")
        };
        let input = self.input.available();
        let (in0, out0) = (gz.inflate.total_in(), gz.inflate.total_out());
        let status = gz
            .inflate
            .decompress(input, &mut gz.out, FlushDecompress::None);
        let consumed = (gz.inflate.total_in() - in0) as usize;
        let produced = (gz.inflate.total_out() - out0) as usize;
        self.input.consume(consumed);
        if produced > 0 {
            let text = &gz.out[..produced];
            gz.crc.update(text);
            if gz.head.len() < 2 {
                let take = (2 - gz.head.len()).min(text.len());
                gz.head.extend_from_slice(&text[..take]);
                if gz.head == gzip::MAGIC {
                    self.parser.set_gzip_inside();
                }
            }
        }
        gz.out_at = 0;
        gz.out_end = produced;
        gz.after = match status {
            Err(error) => Some(After::Corrupt(format!(
                "invalid deflate data in a gzip member ({error})"
            ))),
            Ok(Status::StreamEnd) => Some(After::StreamEnd),
            Ok(Status::Ok | Status::BufError) if consumed == 0 && produced == 0 => {
                Some(After::Stalled)
            }
            Ok(Status::Ok | Status::BufError) => None,
        };
        Ok(true)
    }

    fn truncated_member(&self) -> ReadError {
        self.container(
            Reason::GzipTruncated,
            "the input ends inside a gzip member".into(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn read(data: &[u8], format: Format) -> Result<Vec<Batch>, ReadError> {
        let mut reader = FastxReader::new(
            Input::from_vec(data.to_vec()),
            format,
            Limits {
                whole: true,
                ..Limits::default()
            },
        );
        reader.run()?;
        Ok(std::iter::from_fn(|| reader.pop()).collect())
    }

    fn reason(error: ReadError) -> &'static str {
        match error {
            ReadError::Format(e) => e.reason.code(),
            other => panic!("{other}"),
        }
    }

    // A stored (uncompressed) deflate member of `text`, built by hand.
    fn stored_member(text: &[u8]) -> Vec<u8> {
        let mut out = vec![0x1f, 0x8b, 8, 0, 0, 0, 0, 0, 0, 3];
        out.push(1); // BFINAL, BTYPE 00
        let len = text.len() as u16;
        out.extend_from_slice(&len.to_le_bytes());
        out.extend_from_slice(&(!len).to_le_bytes());
        out.extend_from_slice(text);
        let mut crc = Crc::new();
        crc.update(text);
        out.extend_from_slice(&crc.sum().to_le_bytes());
        out.extend_from_slice(&(text.len() as u32).to_le_bytes());
        out
    }

    // A stored member carrying the BGZF subfield (BSIZE is not checked).
    fn bgzf_member(text: &[u8]) -> Vec<u8> {
        let plain = stored_member(text);
        let mut out = vec![
            0x1f, 0x8b, 8, 4, 0, 0, 0, 0, 0, 0xff, 6, 0, 66, 67, 2, 0, 0, 0,
        ];
        out.extend_from_slice(&plain[10..]);
        out
    }

    #[test]
    fn bgzf_end_rule_follows_the_last_member() {
        let ok = |data: &[u8]| {
            let batches = read(data, Format::Fasta).unwrap();
            (0..batches[0].len())
                .map(|i| batches[0].id.get(i).to_vec())
                .collect::<Vec<_>>()
        };
        let eof = bgzf_member(b"");
        // BGZF with its EOF block, then a plain gzip member, in both orders.
        let mut bgzf_then_gzip = bgzf_member(b">a\nAC\n");
        bgzf_then_gzip.extend(&eof);
        bgzf_then_gzip.extend(stored_member(b">b\nGT\n"));
        assert_eq!(ok(&bgzf_then_gzip), [b"a".to_vec(), b"b".to_vec()]);
        let mut gzip_then_bgzf = stored_member(b">b\nGT\n");
        gzip_then_bgzf.extend(bgzf_member(b">a\nAC\n"));
        gzip_then_bgzf.extend(&eof);
        assert_eq!(ok(&gzip_then_bgzf), [b"b".to_vec(), b"a".to_vec()]);
        // BGZF without its EOF block followed by plain gzip: no BGZF run ends
        // the input.
        let mut cut_then_gzip = bgzf_member(b">a\nAC\n");
        cut_then_gzip.extend(stored_member(b">b\nGT\n"));
        assert_eq!(ok(&cut_then_gzip), [b"a".to_vec(), b"b".to_vec()]);
        // A BGZF run that reaches the end must end with the empty block.
        let mut gzip_then_cut = stored_member(b">b\nGT\n");
        gzip_then_cut.extend(bgzf_member(b">a\nAC\n"));
        assert_eq!(
            reason(read(&gzip_then_cut, Format::Fasta).unwrap_err()),
            "bgzf_missing_eof"
        );
        assert_eq!(
            reason(read(&bgzf_member(b">a\nAC\n"), Format::Fasta).unwrap_err()),
            "bgzf_missing_eof"
        );
        assert_eq!(read(&eof, Format::Fasta).unwrap()[0].len(), 0);
    }

    #[test]
    fn one_step_leaves_at_most_one_batch() {
        let text = b">a\nA\n".repeat(200);
        for data in [text.clone(), stored_member(&text)] {
            let mut reader = FastxReader::new(
                Input::from_vec(data),
                Format::Fasta,
                Limits {
                    records: Some(1),
                    bytes: None,
                    whole: false,
                },
            );
            let mut batches = Vec::new();
            loop {
                let more = reader.step().unwrap();
                assert!(reader.parser.ready_len() <= 1);
                batches.extend(std::iter::from_fn(|| reader.pop()));
                if !more {
                    break;
                }
            }
            assert_eq!(batches.len(), 200);
            assert!(batches
                .iter()
                .enumerate()
                .all(|(k, b)| b.offset == k as u64 && b.len() == 1));
        }
    }

    #[test]
    fn corrupt_deflate_after_text_reports_after_parsing_it() {
        // Valid text in a stored block, then an invalid block type (BTYPE 11).
        let text = b">a\nAC\n>b\nGT\n";
        let mut data = vec![0x1f, 0x8b, 8, 0, 0, 0, 0, 0, 0, 3, 0];
        data.extend_from_slice(&(text.len() as u16).to_le_bytes());
        data.extend_from_slice(&(!(text.len() as u16)).to_le_bytes());
        data.extend_from_slice(text);
        data.push(0b111);
        data.extend_from_slice(&[0; 8]);
        for records in [None, Some(1)] {
            let mut reader = FastxReader::new(
                Input::from_vec(data.clone()),
                Format::Fasta,
                Limits {
                    records,
                    bytes: None,
                    whole: records.is_none(),
                },
            );
            match reader.run().unwrap_err() {
                ReadError::Format(e) => {
                    assert_eq!((e.reason.code(), e.record, e.line), ("gzip_corrupt", 1, 5))
                }
                other => panic!("{other}"),
            }
        }
    }

    #[test]
    fn members_and_trailing_data() {
        let text = b">a\nAC\n>b\nGT\n";
        let mut two = stored_member(&text[..5]);
        two.extend(stored_member(&text[5..]));
        let batches = read(&two, Format::Fasta).unwrap();
        assert_eq!(batches[0].len(), 2);
        assert_eq!(batches[0].seq.get(1), b"GT");
        let mut tail = stored_member(text);
        tail.push(0x1f);
        assert_eq!(
            reason(read(&tail, Format::Fasta).unwrap_err()),
            "gzip_trailing_data"
        );
        let cut = stored_member(text);
        assert_eq!(
            reason(read(&cut[..cut.len() - 4], Format::Fasta).unwrap_err()),
            "gzip_truncated"
        );
        assert_eq!(
            reason(read(b"\x1f\x8b", Format::Fasta).unwrap_err()),
            "gzip_truncated"
        );
        assert_eq!(
            reason(read(b"BZh91AY", Format::Fastq).unwrap_err()),
            "unsupported_compression"
        );
        assert_eq!(
            reason(read(b"\x1f", Format::Fasta).unwrap_err()),
            "content_before_header"
        );
    }
}
