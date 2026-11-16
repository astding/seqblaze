//! Canonical manifest JSON of plain call-time facts (lossless-json-v2 and the
//! scientific projection; result-provenance.md, "Manifest JSON codecs").
//!
//! `seqblaze._manifest_codec` defines the codec and stays the fallback. This
//! encoder takes only exact `dict` (str keys, no reserved tag key), `tuple`,
//! `list`, `str`, `int` within i64, `float`, `bool` and `None`, at most
//! MAX_DEPTH containers deep, and returns None for anything else, so that the
//! caller encodes the value in Python (raising there if the codec rejects it).
//! The text equals `json.dumps(_encode(value), ensure_ascii=True,
//! sort_keys=True, separators=(",", ":"))` byte for byte. With `frozen`, a
//! list is written as the tuple record that `Manifest` freezing makes of it.
//! A float is read as its bits: no floating-point operation runs. A refused
//! reservation also returns None; the Python codec then reports it.
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyDict, PyFloat, PyInt, PyList, PyString, PyTuple};
use sha2::{Digest, Sha256};

const TAG: &[u8] = b"__seqblaze_type__";
/// Deeper values are left to the Python codec, which owns recursion errors.
const MAX_DEPTH: usize = 64;
const HEX: &[u8; 16] = b"0123456789abcdef";
/// Room for a call's whole manifest, reserved once.
const INITIAL: usize = 4096;
const INT_OPEN: &[u8] = b"{\"__seqblaze_type__\":\"int\",\"value\":\"";
const F64_OPEN: &[u8] = b"{\"__seqblaze_type__\":\"f64\",\"bits\":\"";
const TUPLE_OPEN: &[u8] = b"{\"__seqblaze_type__\":\"tuple\",\"items\":";

/// Appends `part` after a fallible reservation.
fn push(out: &mut Vec<u8>, part: &[u8]) -> Option<()> {
    out.try_reserve(part.len()).ok()?;
    out.extend_from_slice(part);
    Some(())
}

fn hex4(out: &mut Vec<u8>, unit: u32) -> Option<()> {
    let mut part = *b"\\u0000";
    for (slot, shift) in part[2..].iter_mut().zip([12, 8, 4, 0]) {
        *slot = HEX[((unit >> shift) & 0xF) as usize];
    }
    push(out, &part)
}

/// `json.encoder.encode_basestring_ascii` of the UTF-8 text `text`: printable
/// ASCII as is, the six short escapes, every other code point as lowercase
/// \uXXXX (a UTF-16 surrogate pair above U+FFFF). None for invalid UTF-8.
fn string(out: &mut Vec<u8>, text: &[u8]) -> Option<()> {
    out.try_reserve(text.len().checked_add(2)?).ok()?;
    if text
        .iter()
        .all(|&byte| matches!(byte, b' '..=b'~') && byte != b'"' && byte != b'\\')
    {
        out.push(b'"');
        out.extend_from_slice(text);
        out.push(b'"');
        return Some(());
    }
    push(out, b"\"")?;
    let mut start = 0;
    let mut at = 0;
    while at < text.len() {
        let short: &[u8] = match text[at] {
            b'"' => b"\\\"",
            b'\\' => b"\\\\",
            0x08 => b"\\b",
            0x0C => b"\\f",
            b'\n' => b"\\n",
            b'\r' => b"\\r",
            b'\t' => b"\\t",
            b' '..=b'~' => {
                at += 1;
                continue;
            }
            _ => b"",
        };
        push(out, &text[start..at])?;
        if !short.is_empty() {
            push(out, short)?;
            at += 1;
            start = at;
            continue;
        }
        // A control character or the lead byte of a multi-byte character.
        let width = match text[at] {
            0x00..=0x7F => 1,
            0xC0..=0xDF => 2,
            0xE0..=0xEF => 3,
            0xF0..=0xF7 => 4,
            _ => return None,
        };
        let code = u32::from(
            std::str::from_utf8(text.get(at..at + width)?)
                .ok()?
                .chars()
                .next()?,
        );
        if code < 0x1_0000 {
            hex4(out, code)?;
        } else {
            let offset = code - 0x1_0000;
            hex4(out, 0xD800 | ((offset >> 10) & 0x3FF))?;
            hex4(out, 0xDC00 | (offset & 0x3FF))?;
        }
        at += width;
        start = at;
    }
    push(out, &text[start..])?;
    push(out, b"\"")
}

/// `_manifest_codec.decimal`: the integer's plain decimal digits.
fn decimal(out: &mut Vec<u8>, number: i64) -> Option<()> {
    let mut digits = [0u8; 20];
    let mut at = digits.len();
    let mut rest = number.unsigned_abs();
    loop {
        at -= 1;
        digits[at] = b'0' + (rest % 10) as u8;
        rest /= 10;
        if rest == 0 {
            break;
        }
    }
    if number < 0 {
        push(out, b"-")?;
    }
    push(out, &digits[at..])
}

/// The output text and the entries of the maps being written. Every map
/// stacks its keys and values on `keys` and `entries` and removes them when
/// done, so a whole manifest takes a few reservations, not some per map.
struct Text<'py> {
    bytes: Vec<u8>,
    keys: Vec<u8>,
    /// (key start in `keys`, key length, value) of each open map's entries.
    entries: Vec<(usize, usize, Bound<'py, PyAny>)>,
}

impl<'py> Text<'py> {
    /// One value; None when it lies outside the subset or a reservation fails.
    fn value(&mut self, value: &Bound<'py, PyAny>, options: Options, depth: usize) -> Option<()> {
        let out = &mut self.bytes;
        if value.is_none() {
            return push(out, b"null");
        }
        if let Ok(text) = value.cast_exact::<PyString>() {
            return string(out, text.to_str().ok()?.as_bytes());
        }
        if let Ok(flag) = value.cast_exact::<PyBool>() {
            return push(out, if flag.is_true() { b"true" } else { b"false" });
        }
        if value.is_exact_instance_of::<PyInt>() {
            let number: i64 = value.extract().ok()?;
            push(out, INT_OPEN)?;
            decimal(out, number)?;
            return push(out, b"\"}");
        }
        if let Ok(float) = value.cast_exact::<PyFloat>() {
            let mut bits = float.value().to_bits();
            // Scientific identities canonicalize every NaN (quiet, positive).
            if options.scientific && bits & 0x7FFF_FFFF_FFFF_FFFF > 0x7FF0_0000_0000_0000 {
                bits = 0x7FF8_0000_0000_0000;
            }
            let mut digits = *b"0000000000000000\"}";
            for (index, slot) in digits[..16].iter_mut().enumerate() {
                *slot = HEX[((bits >> (60 - 4 * index)) & 0xF) as usize];
            }
            push(out, F64_OPEN)?;
            return push(out, &digits);
        }
        if depth >= MAX_DEPTH {
            return None;
        }
        if let Ok(map) = value.cast_exact::<PyDict>() {
            return self.map(map, options, depth + 1);
        }
        if let Ok(items) = value.cast_exact::<PyTuple>() {
            push(&mut self.bytes, TUPLE_OPEN)?;
            self.sequence(items.iter(), options, depth + 1)?;
            return push(&mut self.bytes, b"}");
        }
        if let Ok(items) = value.cast_exact::<PyList>() {
            if options.frozen {
                push(&mut self.bytes, TUPLE_OPEN)?;
                self.sequence(items.iter(), options, depth + 1)?;
                return push(&mut self.bytes, b"}");
            }
            return self.sequence(items.iter(), options, depth + 1);
        }
        None
    }

    fn sequence(
        &mut self,
        items: impl Iterator<Item = Bound<'py, PyAny>>,
        options: Options,
        depth: usize,
    ) -> Option<()> {
        push(&mut self.bytes, b"[")?;
        for (index, item) in items.enumerate() {
            if index > 0 {
                push(&mut self.bytes, b",")?;
            }
            self.value(&item, options, depth)?;
        }
        push(&mut self.bytes, b"]")
    }

    fn map(&mut self, map: &Bound<'py, PyDict>, options: Options, depth: usize) -> Option<()> {
        let (base, key_base) = (self.entries.len(), self.keys.len());
        let written = self.map_entries(map, options, depth, base);
        self.entries.truncate(base);
        self.keys.truncate(key_base);
        written
    }

    /// Keys in code-point order: UTF-8 bytes order as Python orders str.
    fn map_entries(
        &mut self,
        map: &Bound<'py, PyDict>,
        options: Options,
        depth: usize,
        base: usize,
    ) -> Option<()> {
        self.entries.try_reserve(map.len()).ok()?;
        for (key, item) in map.iter() {
            let text = key.cast_exact::<PyString>().ok()?.to_str().ok()?.as_bytes();
            if text == TAG {
                return None; // the codec's "map" record
            }
            let start = self.keys.len();
            push(&mut self.keys, text)?;
            self.entries.push((start, text.len(), item));
        }
        let end = self.entries.len();
        let Text { keys, entries, .. } = self;
        entries[base..end].sort_unstable_by(|left, right| {
            keys[left.0..left.0 + left.1].cmp(&keys[right.0..right.0 + right.1])
        });
        push(&mut self.bytes, b"{")?;
        for position in base..end {
            if position > base {
                push(&mut self.bytes, b",")?;
            }
            let (start, len, item) = {
                let entry = &self.entries[position];
                (entry.0, entry.1, entry.2.clone())
            };
            string(&mut self.bytes, &self.keys[start..start + len])?;
            push(&mut self.bytes, b":")?;
            self.value(&item, options, depth)?;
        }
        push(&mut self.bytes, b"}")
    }
}

#[derive(Clone, Copy)]
struct Options {
    scientific: bool,
    frozen: bool,
}

fn encode(value: &Bound<'_, PyAny>, scientific: bool, frozen: bool) -> Option<Vec<u8>> {
    let mut text = Text {
        bytes: Vec::new(),
        keys: Vec::new(),
        entries: Vec::new(),
    };
    text.bytes.try_reserve(INITIAL).ok()?;
    text.value(value, Options { scientific, frozen }, 0)?;
    Some(text.bytes)
}

/// `_manifest_codec.dumps` of a plain value, or None (the Python codec decides).
#[pyfunction(signature = (value, scientific=false, frozen=false))]
fn _native_manifest_json(
    value: &Bound<'_, PyAny>,
    scientific: bool,
    frozen: bool,
) -> Option<String> {
    String::from_utf8(encode(value, scientific, frozen)?).ok()
}

/// `_manifest_codec.digest`: H(domain, dumps(value)), the framed SHA-256 of
/// result-provenance.md, as 64 lowercase hex digits; None as for the text.
#[pyfunction(signature = (domain, value, scientific=false, frozen=false))]
fn _native_manifest_digest(
    domain: &Bound<'_, PyAny>,
    value: &Bound<'_, PyAny>,
    scientific: bool,
    frozen: bool,
) -> Option<String> {
    let domain = domain.cast_exact::<PyString>().ok()?.to_str().ok()?;
    let domain_len = u16::try_from(domain.len()).ok()?;
    if !domain.is_ascii() {
        return None;
    }
    let payload = encode(value, scientific, frozen)?;
    let mut digest = Sha256::new();
    digest.update(b"SEQBLAZE\0");
    digest.update(1u16.to_le_bytes());
    digest.update(domain_len.to_le_bytes());
    digest.update(domain.as_bytes());
    digest.update((payload.len() as u64).to_le_bytes());
    digest.update(&payload);
    let mut hex = String::new();
    hex.try_reserve_exact(64).ok()?;
    for byte in digest.finalize() {
        hex.push(char::from(HEX[usize::from(byte >> 4)]));
        hex.push(char::from(HEX[usize::from(byte & 0xF)]));
    }
    Some(hex)
}

pub fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(_native_manifest_json, module)?)?;
    module.add_function(wrap_pyfunction!(_native_manifest_digest, module)?)?;
    Ok(())
}
