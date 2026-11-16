//! The primer3-py 2.3.1 thermodynamic parameter set, parsed from the sixteen parameter
//! files that the Python loader located and verified at run time (D63 clause 3).
//! Spec: docs/wiki/thermo/primer3-parameter-loader.md §4 (parsing), §5 (tables built at
//! load time and content facts), §7 (an immutable handle shared by every worker);
//! the shared constants of docs/wiki/thermo/primer3-alignment.md §3.2–§3.3.
//! Derivation route: a behaviour specification written from a reading of primer3-py
//! 2.3.1's source (no code copied; D94), confirmed against the unmodified tool; this
//! module was written from the wiki pages and those specifications only.
//!
//! No value read from a file ever appears in an error message: errors name the role
//! label (E1–E16), a one-based line number or a content fact, never file text.
//! Building a handle rounds (decimal text to binary64), so it runs under the host
//! floating-point guard (D99; loader §7).
use seqblaze_core::{fpenv, validate::EngineError};

/// Number of parameter files (roles E1–E16, loader §3.1).
pub const ROLES: usize = 16;
/// φ = fl(10⁻¹¹), "zero, but not forbidden" (shared page §3.3, class C):
/// `0x1.5fd7fe1796495p-37`.
pub const PHI: f64 = 1e-11;
/// Terminal A·T term of SantaLucia & Hicks 2004 Table 1 (ledger `sh2004:terminal_at`:
/// 2.2 kcal/mol, exactly 2200 cal/mol; 6.9 cal/(mol·K)), shared page §3.2.
pub const AT_H: f64 = 2200.0;
pub const AT_S: f64 = 6.9;
/// Longest tabulated loop (the loop-length files' 30 rows, class F).
pub const LOOP_ROWS: usize = 30;
/// Capacity of the three- and four-nucleotide hairpin-loop bonus files (loader §4.3).
const B3_CAPACITY: usize = 20;
const B4_CAPACITY: usize = 146;

/// One table entry (ΔH cal/mol, ΔS cal/(mol·K)).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Hs {
    pub h: f64,
    pub s: f64,
}

/// The forbidden pair Ω = (+∞, −1) (shared page §2.3, class C).
pub const OMEGA: Hs = Hs {
    h: f64::INFINITY,
    s: -1.0,
};

/// A hairpin-loop bonus: the ΔH and ΔS maps are separate (loader §5), so either
/// component may be absent.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Bonus {
    pub h: Option<f64>,
    pub s: Option<f64>,
}

/// Index of a doublet slot (a, b, c, d), codes 0–4 (A, C, G, T, N).
#[inline]
pub const fn idx4(a: u8, b: u8, c: u8, d: u8) -> usize {
    ((a as usize * 5 + b as usize) * 5 + c as usize) * 5 + d as usize
}

/// Index of a dangling-end slot (a, c, d), codes 0–4.
#[inline]
pub const fn idx3(a: u8, c: u8, d: u8) -> usize {
    (a as usize * 5 + c as usize) * 5 + d as usize
}

/// Whether two codes form a Watson–Crick pair (A·T, T·A, C·G, G·C).
#[inline]
pub const fn is_wc(a: u8, b: u8) -> bool {
    a < 4 && b < 4 && a + b == 3
}

/// The terminal A·T term A(p, q): (2200, fl(6.9)) for A·T and T·A, (0, φ) for every
/// other pair, code 4 included (shared page §3.2).
#[inline]
pub fn at(p: u8, q: u8) -> Hs {
    if (p == 0 && q == 3) || (p == 3 && q == 0) {
        Hs { h: AT_H, s: AT_S }
    } else {
        Hs { h: 0.0, s: PHI }
    }
}

/// The parsed, immutable table set of loader §5 plus its identity (§3.3, §7).
#[derive(Debug, Clone)]
pub struct P3Tables {
    /// `primer3-py@2.3.1:files`, or `synthetic` for a testing handle.
    pub name: String,
    /// The set digest of loader §3.3 (computed by the Python loader from the files).
    pub sha256: String,
    /// Built by the testing constructor from caller-supplied bytes (no pin check).
    pub synthetic: bool,
    /// Watson–Crick doublets (E1, E2), 5⁴ slots by [`idx4`].
    pub nn: Vec<Hs>,
    /// Single-mismatch doublets (E3, E4), 5⁴ slots.
    pub mm: Vec<Hs>,
    /// Terminal mismatches, set M1 (E13, E14) and set M2 (E15, E16), 5⁴ slots each.
    pub m1: Vec<Hs>,
    pub m2: Vec<Hs>,
    /// 3′ and 5′ dangling ends (E5, E6), 5³ slots each by [`idx3`].
    pub d3: Vec<Hs>,
    pub d5: Vec<Hs>,
    /// Loop-length terms (E7, E8) for k = 1…30 at index k (index 0 unused).
    pub loop_in: [Hs; LOOP_ROWS + 1],
    pub loop_bu: [Hs; LOOP_ROWS + 1],
    pub loop_hp: [Hs; LOOP_ROWS + 1],
    /// Three-nucleotide hairpin-loop bonuses (E11, E12) by the 5-code key in base 4.
    pub b3: Vec<Bonus>,
    /// Four-nucleotide hairpin-loop bonuses (E9, E10) by the 6-code key in base 4.
    pub b4: Vec<Bonus>,
}

/// Index of an all-base bonus key in base 4; None if a code is 4 (such a key can never
/// be looked up from an upper-case ACGT input, loader §4.3).
#[inline]
pub fn bonus_index(codes: &[u8]) -> Option<usize> {
    let mut index = 0usize;
    for &code in codes {
        if code > 3 {
            return None;
        }
        index = index * 4 + code as usize;
    }
    Some(index)
}

fn parameter(message: String) -> EngineError {
    EngineError::Parameter(message)
}

const REINSTALL: &str = "with the pinned primer3-py 2.3.1 files this means a corrupted \
                         installation; reinstall it with pip install \"seqblaze[primer3]\"";

fn line_error(role: usize, line: usize, what: &str) -> EngineError {
    parameter(format!(
        "primer3 parameter file E{} (line {line}): {what}; {REINSTALL}",
        role + 1
    ))
}

fn fact_error(number: usize, fact: &str, role: usize, record: usize) -> EngineError {
    parameter(format!(
        "primer3 parameter files: content fact {number} ({fact}) fails at file E{} line {}; \
         {REINSTALL}",
        role + 1,
        record + 1
    ))
}

/// The records of one file (loader §4.1 L1): LF-separated, one CR before an LF
/// removed; a final LF does not start another record.
fn records(data: &[u8]) -> Vec<&[u8]> {
    let mut out: Vec<&[u8]> = Vec::new();
    let mut rest = data;
    while !rest.is_empty() {
        match rest.iter().position(|&b| b == b'\n') {
            Some(end) => {
                let mut record = &rest[..end];
                if let Some((&b'\r', head)) = record.split_last() {
                    record = head;
                }
                out.push(record);
                rest = &rest[end + 1..];
            }
            None => {
                out.push(rest);
                rest = &[];
            }
        }
    }
    out
}

#[inline]
fn is_blank(b: u8) -> bool {
    b == b' ' || b == b'\t'
}

/// Whether `text` is decimal text of loader §4.1 L2: an optional minus sign, digits,
/// and an optional point followed by digits.
fn is_decimal(text: &[u8]) -> bool {
    let body = text.strip_prefix(b"-").unwrap_or(text);
    let (int, frac) = match body.iter().position(|&b| b == b'.') {
        Some(dot) => (&body[..dot], Some(&body[dot + 1..])),
        None => (body, None),
    };
    !int.is_empty()
        && int.iter().all(u8::is_ascii_digit)
        && frac.is_none_or(|f| !f.is_empty() && f.iter().all(u8::is_ascii_digit))
}

/// fl(decimal), correctly rounded (Rust's conversion; the caller holds the guard).
fn decimal(text: &[u8]) -> Option<f64> {
    if !is_decimal(text) {
        return None;
    }
    std::str::from_utf8(text).ok()?.parse::<f64>().ok()
}

/// One field of a multi-field file (L2, L4): exactly `inf`, or decimal text.
fn field(text: &[u8]) -> Option<f64> {
    if text == b"inf" {
        Some(f64::INFINITY)
    } else {
        decimal(text)
    }
}

/// A one-number file (E1–E6, E13–E16): its first `count` records (L1–L3, §4.3, §4.4).
fn one_number(role: usize, data: &[u8], count: usize) -> Result<Vec<f64>, EngineError> {
    let lines = records(data);
    if lines.len() < count {
        return Err(line_error(
            role,
            lines.len() + 1,
            &format!("too few records ({count} needed)"),
        ));
    }
    let mut out = Vec::with_capacity(count);
    for (index, &line) in lines[..count].iter().enumerate() {
        if line.starts_with(b"inf") {
            out.push(f64::INFINITY);
            continue;
        }
        if line.first().is_some_and(|&b| is_blank(b)) {
            return Err(line_error(role, index + 1, "leading whitespace"));
        }
        let end = line.len() - line.iter().rev().take_while(|&&b| is_blank(b)).count();
        let text = &line[..end];
        if text.is_empty() {
            return Err(line_error(role, index + 1, "a blank record"));
        }
        let value = decimal(text).ok_or_else(|| {
            line_error(
                role,
                index + 1,
                "not a plain decimal number or inf (exponents, signs other than a leading \
                 minus, hexadecimal forms and nan are not accepted)",
            )
        })?;
        out.push(value);
    }
    Ok(out)
}

fn fields_of(line: &[u8]) -> Vec<&[u8]> {
    line.split(|&b| is_blank(b))
        .filter(|f| !f.is_empty())
        .collect()
}

/// A loop-length file (E7, E8): 30 rows of label, internal, bulge and hairpin values.
fn loop_file(role: usize, data: &[u8]) -> Result<Vec<[f64; 3]>, EngineError> {
    let lines = records(data);
    if lines.len() < LOOP_ROWS {
        return Err(line_error(
            role,
            lines.len() + 1,
            &format!("too few lines ({LOOP_ROWS} needed)"),
        ));
    }
    let mut out = Vec::with_capacity(LOOP_ROWS);
    for (index, &line) in lines[..LOOP_ROWS].iter().enumerate() {
        let parts = fields_of(line);
        if parts.len() < 4 {
            return Err(line_error(
                role,
                index + 1,
                "too few fields (a label and three values needed)",
            ));
        }
        let mut row = [0.0; 3];
        for (slot, text) in row.iter_mut().zip(&parts[1..4]) {
            *slot = field(text).ok_or_else(|| {
                line_error(
                    role,
                    index + 1,
                    "a field is neither a plain decimal nor inf",
                )
            })?;
        }
        out.push(row);
    }
    Ok(out)
}

/// A hairpin-bonus file (E9–E12): (key codes padded with 4, value), keys strictly
/// ascending, at most `capacity` records (loader §4.2–§4.3).
fn bonus_file(
    role: usize,
    data: &[u8],
    width: usize,
    capacity: usize,
) -> Result<Vec<(Vec<u8>, f64)>, EngineError> {
    let mut out: Vec<(Vec<u8>, f64)> = Vec::new();
    for (index, &line) in records(data).iter().enumerate() {
        let parts = fields_of(line);
        if parts.is_empty() {
            return Err(line_error(role, index + 1, "a blank record"));
        }
        if parts.len() != 2 {
            return Err(line_error(
                role,
                index + 1,
                "a record holds a key and one number",
            ));
        }
        if out.len() == capacity {
            return Err(line_error(
                role,
                index + 1,
                &format!("more than {capacity} records"),
            ));
        }
        let mut key = vec![4u8; width];
        for (slot, &byte) in key.iter_mut().zip(parts[0]) {
            *slot = match byte {
                b'A' | b'0' => 0,
                b'C' | b'1' => 1,
                b'G' | b'2' => 2,
                b'T' | b'3' => 3,
                _ => 4,
            };
        }
        let value = field(parts[1]).ok_or_else(|| {
            line_error(
                role,
                index + 1,
                "the number is neither a plain decimal nor inf",
            )
        })?;
        if out.last().is_some_and(|(previous, _)| *previous >= key) {
            return Err(line_error(
                role,
                index + 1,
                "keys do not ascend strictly in base-code order",
            ));
        }
        out.push((key, value));
    }
    Ok(out)
}

#[inline]
fn finite(x: f64) -> bool {
    x.is_finite()
}

/// The doublet codes (a, b, c, d) of record p = 64a + 16b + 4c + d (loader §4.2).
#[inline]
fn doublet(p: usize) -> (u8, u8, u8, u8) {
    (
        (p >> 6) as u8,
        (p >> 4 & 3) as u8,
        (p >> 2 & 3) as u8,
        (p & 3) as u8,
    )
}

/// Tandem G·T doublets GG/TT, GT/TG, TG/GT and TT/GG (part C §11.7 item 1).
#[inline]
fn tandem_gt(a: u8, b: u8, c: u8, d: u8) -> bool {
    matches!(
        (a, b, c, d),
        (2, 2, 3, 3) | (2, 3, 3, 2) | (3, 2, 2, 3) | (3, 3, 2, 2)
    )
}

/// The record of E3 that is infinite although its doublet has one Watson–Crick position
/// (top 5′-TA-3′, bottom 3′-AA-5′; zero-based record 192), part C §11.7 item 2.
const E3_ASYMMETRIC: usize = 192;

/// The content facts of loader §5 (part C §11.7), on the decoded one-number files.
/// Each failure names the fact, the role and the line, never a value.
fn content_facts(v: &[Vec<f64>; ROLES]) -> Result<(), EngineError> {
    let check = |number: usize, fact: &str, role: usize, want: &dyn Fn(usize) -> bool| match (0..v
        [role]
        .len())
        .find(|&p| finite(v[role][p]) != want(p))
    {
        Some(p) => Err(fact_error(number, fact, role, p)),
        None => Ok(()),
    };
    let wcwc = |p: usize| {
        let (a, b, c, d) = doublet(p);
        is_wc(a, c) && is_wc(b, d)
    };
    let fact1 = "Watson–Crick doublets finite exactly where both positions pair";
    check(1, fact1, 0, &wcwc)?;
    check(1, fact1, 1, &wcwc)?;
    let mismatch = |p: usize| {
        let (a, b, c, d) = doublet(p);
        (is_wc(a, c) != is_wc(b, d)) || tandem_gt(a, b, c, d)
    };
    let fact2 = "mismatch doublets finite exactly where one position pairs, or tandem G·T, \
                 with the one asymmetric enthalpy record";
    check(2, fact2, 2, &|p| mismatch(p) && p != E3_ASYMMETRIC)?;
    check(2, fact2, 3, &mismatch)?;
    let fact3 = "dangling ends: enthalpies all finite, entropies finite exactly where the \
                 pair is Watson–Crick";
    check(3, fact3, 4, &|_| true)?;
    check(3, fact3, 5, &|p| {
        let q = p & 63;
        is_wc((q >> 4) as u8, (q >> 2 & 3) as u8)
    })?;
    // Facts 4 and 5 are asserted where the two descriptions agree: part C states them for
    // records whose closing pair a·c is Watson–Crick (the only records a kernel reads);
    // where only b·d pairs, the installed 2.3.1 files are finite and part C's wording says
    // infinite, so those 48 records are not asserted (finding of the real-file run).
    let only_bd = |p: usize| {
        let (a, b, c, d) = doublet(p);
        !is_wc(a, c) && is_wc(b, d)
    };
    let check_some = |number: usize,
                      fact: &str,
                      role: usize,
                      want: &dyn Fn(usize) -> bool|
     -> Result<(), EngineError> {
        match (0..256).find(|&p| !only_bd(p) && finite(v[role][p]) != want(p)) {
            Some(p) => Err(fact_error(number, fact, role, p)),
            None => Ok(()),
        }
    };
    check_some(
        4,
        "M1 entropies finite where the closing pair is Watson–Crick and the mismatch \
         position is not, infinite where both or neither pair",
        13,
        &|p| {
            let (a, b, c, d) = doublet(p);
            is_wc(a, c) && !is_wc(b, d)
        },
    )?;
    check_some(
        5,
        "M2 entropies finite where the closing pair is Watson–Crick, infinite where neither \
         facing position pairs",
        15,
        &|p| {
            let (a, _, c, _) = doublet(p);
            is_wc(a, c)
        },
    )?;
    Ok(())
}

/// Doublet entry: Ω if either decoded value is not finite (loader §5).
#[inline]
fn entry(h: f64, s: f64) -> Hs {
    if finite(h) && finite(s) {
        Hs { h, s }
    } else {
        OMEGA
    }
}

/// A 5⁴ doublet table (NN, MM): Ω at every code-4 slot.
fn doublet_table(h: &[f64], s: &[f64]) -> Vec<Hs> {
    let mut table = vec![OMEGA; 625];
    for (p, (&xh, &xs)) in h.iter().zip(s).enumerate() {
        let (a, b, c, d) = doublet(p);
        table[idx4(a, b, c, d)] = entry(xh, xs);
    }
    table
}

/// A 5⁴ terminal-mismatch table (M1, M2): Ω if a or c is code 4, (0, φ) if a and c are
/// bases and b or d is code 4, else the file entry.
fn mismatch_table(h: &[f64], s: &[f64]) -> Vec<Hs> {
    let mut table = vec![OMEGA; 625];
    for a in 0..4u8 {
        for c in 0..4u8 {
            for b in 0..5u8 {
                for d in 0..5u8 {
                    table[idx4(a, b, c, d)] = if b == 4 || d == 4 {
                        Hs { h: 0.0, s: PHI }
                    } else {
                        let p =
                            (a as usize) << 6 | (b as usize) << 4 | (c as usize) << 2 | d as usize;
                        entry(h[p], s[p])
                    };
                }
            }
        }
    }
    table
}

impl P3Tables {
    /// Parse the sixteen files (role order E1…E16) into the table set of loader §5 and
    /// assert the content facts. Runs in the IEEE default floating-point environment
    /// whatever the caller's state, which is restored on return (D99; loader §7).
    pub fn parse(
        files: &[&[u8]],
        name: &str,
        sha256: &str,
        synthetic: bool,
    ) -> Result<Self, EngineError> {
        fpenv::guarded(|| Self::build(files, name, sha256, synthetic))
    }

    fn build(
        files: &[&[u8]],
        name: &str,
        sha256: &str,
        synthetic: bool,
    ) -> Result<Self, EngineError> {
        if files.len() != ROLES {
            return Err(parameter(format!(
                "primer3 parameter set: {ROLES} files (E1–E16) are needed, got {}",
                files.len()
            )));
        }
        // One-number files: decoded values (loader §4.1, §4.2).
        let mut decoded: [Vec<f64>; ROLES] = Default::default();
        for role in (0..6).chain(12..16) {
            let count = if role == 4 || role == 5 { 128 } else { 256 };
            decoded[role] = one_number(role, files[role], count)?;
        }
        let loop_h = loop_file(6, files[6])?;
        let loop_s = loop_file(7, files[7])?;
        let b4h = bonus_file(8, files[8], 6, B4_CAPACITY)?;
        let b4s = bonus_file(9, files[9], 6, B4_CAPACITY)?;
        let b3h = bonus_file(10, files[10], 5, B3_CAPACITY)?;
        let b3s = bonus_file(11, files[11], 5, B3_CAPACITY)?;
        for (first, a, b) in [(8usize, &b4h, &b4s), (10, &b3h, &b3s)] {
            if a.len() != b.len() {
                return Err(parameter(format!(
                    "primer3 parameter files E{} and E{} hold different numbers of records; \
                     {REINSTALL}",
                    first + 1,
                    first + 2
                )));
            }
        }
        content_facts(&decoded)?;
        let nn = doublet_table(&decoded[0], &decoded[1]);
        let mm = doublet_table(&decoded[2], &decoded[3]);
        let m1 = mismatch_table(&decoded[12], &decoded[13]);
        let m2 = mismatch_table(&decoded[14], &decoded[15]);
        // Fact 6: M1 and M2 differ only where both facing positions pair.
        for p in 0..256 {
            let (a, b, c, d) = doublet(p);
            let k = idx4(a, b, c, d);
            let same =
                m1[k].h.to_bits() == m2[k].h.to_bits() && m1[k].s.to_bits() == m2[k].s.to_bits();
            if !(is_wc(a, c) && is_wc(b, d)) && !same {
                return Err(fact_error(
                    6,
                    "M1/M2 equivalence: the two terminal-mismatch sets differ only where both \
                     facing positions pair",
                    14,
                    p,
                ));
            }
        }
        let mut d3 = vec![OMEGA; 125];
        let mut d5 = vec![OMEGA; 125];
        for p in 0..64 {
            let (a, c, d) = ((p >> 4) as u8, (p >> 2 & 3) as u8, (p & 3) as u8);
            d3[idx3(a, c, d)] = entry(decoded[4][p], decoded[5][p]);
            d5[idx3(a, c, d)] = entry(decoded[4][64 + p], decoded[5][64 + p]);
        }
        let mut loop_in = [OMEGA; LOOP_ROWS + 1];
        let mut loop_bu = [OMEGA; LOOP_ROWS + 1];
        let mut loop_hp = [OMEGA; LOOP_ROWS + 1];
        for k in 1..=LOOP_ROWS {
            let (h, s) = (loop_h[k - 1], loop_s[k - 1]);
            loop_in[k] = Hs { h: h[0], s: s[0] };
            loop_bu[k] = Hs { h: h[1], s: s[1] };
            loop_hp[k] = Hs { h: h[2], s: s[2] };
        }
        let bonus = |h: &[(Vec<u8>, f64)], s: &[(Vec<u8>, f64)], width: u32| {
            let mut table = vec![Bonus::default(); 4usize.pow(width)];
            for (key, value) in h {
                if let Some(index) = bonus_index(key) {
                    table[index].h = Some(*value);
                }
            }
            for (key, value) in s {
                if let Some(index) = bonus_index(key) {
                    table[index].s = Some(*value);
                }
            }
            table
        };
        Ok(Self {
            name: name.to_owned(),
            sha256: sha256.to_owned(),
            synthetic,
            nn,
            mm,
            m1,
            m2,
            d3,
            d5,
            loop_in,
            loop_bu,
            loop_hp,
            b3: bonus(&b3h, &b3s, 5),
            b4: bonus(&b4h, &b4s, 6),
        })
    }
}

/// Word offsets of the device copy of a handle (docs/wiki/gpu/primer3-cuda.md §4): the
/// tables of [`P3Tables`] as binary64 bit patterns in one flat array, each (ΔH, ΔS)
/// entry two words at `table + 2·index`, each bonus three words at `table + 3·index`
/// (ΔH, ΔS, presence bits: 1 for ΔH, 2 for ΔS; an absent component is +0 and adds
/// nothing). Mirrored by the SB_P3_* offsets of kernels/primer3.cuh.
pub mod device_layout {
    pub const NN: usize = 0;
    pub const MM: usize = NN + 2 * 625;
    pub const M1: usize = MM + 2 * 625;
    pub const M2: usize = M1 + 2 * 625;
    pub const D3: usize = M2 + 2 * 625;
    pub const D5: usize = D3 + 2 * 125;
    pub const LOOP_IN: usize = D5 + 2 * 125;
    pub const LOOP_BU: usize = LOOP_IN + 2 * 31;
    pub const LOOP_HP: usize = LOOP_BU + 2 * 31;
    pub const B3: usize = LOOP_HP + 2 * 31;
    pub const B4: usize = B3 + 3 * 1024;
    pub const WORDS: usize = B4 + 3 * 4096;
}

/// The bit patterns of every binary64 scalar the device replay of T2 or K2 reads
/// (primer3-cuda.md §4, QPC-6): the class-C constants of the Rust sources and the
/// per-call values. K2-only fields are +0 in a T2 pack.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct P3DeviceScalars {
    pub t37: u64,
    pub phi: u64,
    pub fresh_h: u64,
    pub fresh_s: u64,
    pub s_cutoff: u64,
    pub tolerance: u64,
    pub margin: u64,
    pub alpha1: u64,
    pub at_h: u64,
    pub at_s: u64,
    pub omega_h: u64,
    pub omega_s: u64,
    pub k0: u64,
    pub kcal_div: u64,
    pub init_h: u64,
    pub init_s: u64,
    pub shift: u64,
    pub sigma: u64,
    pub t_u: u64,
    pub q4: u64,
    pub q1: u64,
    pub theta_omega4: u64,
    pub theta_omega1: u64,
}

/// The device pack of one T2 or K2 call (primer3-cuda.md §4): the handle's tables
/// ([`P3Tables::device_words`]), the end-term table(s) and the scalars, as bit
/// patterns. `ends` holds (ΔH, ΔS) of slot k at `2k`: T2 one table of 625 slots by
/// `idx4(p, p′, q, q′)` of the closing pair and its neighbours; K2 two tables of 625
/// slots (at Q₄, then at Q₁) by `idx4(φ, ψ, φ′, ψ′)`.
#[derive(Clone, Debug)]
pub struct P3DevicePack {
    pub tables: Vec<u64>,
    pub ends: Vec<u64>,
    pub scalars: P3DeviceScalars,
}

/// The bit patterns of a list of entries, two words each; a refused reservation is a
/// DeviceError.
pub(crate) fn entry_words(entries: &[Hs]) -> Result<Vec<u64>, EngineError> {
    let mut words = Vec::new();
    seqblaze_core::detail::try_reserve_exact(&mut words, 2 * entries.len(), "primer3 end terms")?;
    for entry in entries {
        words.push(entry.h.to_bits());
        words.push(entry.s.to_bits());
    }
    Ok(words)
}

impl P3Tables {
    /// The device copy of the handle (primer3-cuda.md §4): every table the T2 and K2
    /// recurrences read, as bit patterns in the layout of [`device_layout`]. Copies bits
    /// and performs no floating-point operation. A refused reservation is a DeviceError
    /// (D85 (2)).
    pub fn device_words(&self) -> Result<Vec<u64>, EngineError> {
        use device_layout as l;
        let mut words = Vec::new();
        seqblaze_core::detail::try_reserve_exact(&mut words, l::WORDS, "primer3 device tables")?;
        let pairs = |words: &mut Vec<u64>, table: &[Hs]| {
            for entry in table {
                words.push(entry.h.to_bits());
                words.push(entry.s.to_bits());
            }
        };
        let bonuses = |words: &mut Vec<u64>, table: &[Bonus]| {
            for entry in table {
                words.push(entry.h.map_or(0, f64::to_bits));
                words.push(entry.s.map_or(0, f64::to_bits));
                words.push(u64::from(entry.h.is_some()) | (u64::from(entry.s.is_some()) << 1));
            }
        };
        let checks = [
            (self.nn.len(), 625),
            (self.mm.len(), 625),
            (self.m1.len(), 625),
            (self.m2.len(), 625),
            (self.d3.len(), 125),
            (self.d5.len(), 125),
            (self.b3.len(), 1024),
            (self.b4.len(), 4096),
        ];
        if checks.iter().any(|&(have, want)| have != want) {
            return Err(EngineError::Device(
                "the primer3 parameter handle has an unexpected table size; use device='cpu'"
                    .into(),
            ));
        }
        for table in [&self.nn, &self.mm, &self.m1, &self.m2, &self.d3, &self.d5] {
            pairs(&mut words, table);
        }
        for table in [&self.loop_in, &self.loop_bu, &self.loop_hp] {
            pairs(&mut words, table);
        }
        bonuses(&mut words, &self.b3);
        bonuses(&mut words, &self.b4);
        debug_assert_eq!(words.len(), l::WORDS);
        Ok(words)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constants_have_the_documented_bits() {
        assert_eq!(PHI.to_bits(), 0x3DA5_FD7F_E179_6495);
        assert_eq!(AT_S, 6.9);
    }

    #[test]
    fn decimal_grammar() {
        for good in ["0", "-0", "12", "-12.5", "0.0", "3.25"] {
            assert!(is_decimal(good.as_bytes()), "{good}");
        }
        for bad in [
            "", "-", "1e3", "+1", ".5", "5.", "0x1p3", "nan", "-inf", "1.2.3", "1 2",
        ] {
            assert!(!is_decimal(bad.as_bytes()), "{bad}");
        }
    }

    #[test]
    fn records_strip_one_cr_and_the_final_lf() {
        assert_eq!(records(b"1\r\n2\n3"), vec![&b"1"[..], b"2", b"3"]);
        assert_eq!(records(b"1\n2\n"), vec![&b"1"[..], b"2"]);
        assert_eq!(records(b"1\n\n2\n"), vec![&b"1"[..], b"", b"2"]);
    }

    #[test]
    fn one_number_rules() {
        let v = one_number(0, b"inf_x\n-1.5 \t\n2\nextra\n", 3).unwrap();
        assert_eq!(v, vec![f64::INFINITY, -1.5, 2.0]);
        assert!(one_number(0, b" 1\n2\n", 2).is_err());
        assert!(one_number(0, b"1\n\n2\n", 3).is_err());
        assert!(one_number(0, b"1\n1e3\n", 2).is_err());
        assert!(one_number(0, b"1\n", 2).is_err());
    }
}
