//! The K1 compiled pack (docs/wiki/thermo/duplex-mfe.md §4.5, §12.2): the loader of
//! the canonical text that `harness/tools/gen_params.py` (`compile_k1`) writes.
//!
//! Grammar (UTF-8, LF line ends, one record per line, fields separated by TAB):
//! `seqblaze-k1-pack 1`; `lines N` (every line of the text, these two included);
//! `name`, `definition_sha256`, `loop_max`, `salt_b`, `gas_r`, `naeq_mg` (rationals as
//! numerator and denominator); `entries E` followed by E lines
//! `entry id h g q role key` with ids 0..E-1 in order; then the tables INIT, SYM, END_L,
//! END_R, STACK, BULGE, I11, I12, I21, I22, GEN, LOOP_OUT, LOOP_IN, each a line
//! `table NAME count` followed by exactly its cells `cell NAME index content` in
//! canonical index order (the last index varies fastest; GEN lists (u, v) with
//! u, v >= 1, u + v <= Λ, outside the four small shapes); finally `end`. A cell's
//! content is `bot` (⊥), `empty` or `id x multiplicity` items joined by `,` with
//! strictly increasing ids. The loader reads nothing else and checks everything:
//! a malformed, incomplete or out-of-range text is a ParameterError naming the first
//! bad line.
//!
//! Format 2 (docs/wiki/thermo/nndb2024-dna.md §4.4, §5.3.2) adds the lines `pairs`
//! (`wc` or `wc+gt`) and `classes` (`1 all` or `3 1n,23,other`) after `naeq_mg`, the
//! entry role `states`, tables indexed by the pair type P (N_P = 4 or 6) instead of the
//! top letter, the class index c in front of LOOP_OUT and LOOP_IN, the class map CLS
//! (`cls CLS u,v c` per GEN shape, after GEN) and the 200 STATES cells (σ = 1 … 200,
//! after LOOP_IN). A format-1 pack is expanded in memory to the same layout: N_P = 4
//! (a Watson–Crick pair's P is its top letter), N_C = 1, CLS ≡ 0 and empty STATES
//! cells, so the kernels have one code path.
//!
//! Messages name the line and what failed (tag, field, table, index, check); none quotes
//! the text of a field of an entry, salt_b, gas_r or naeq_mg line or prints a computed
//! bound (nndb2024-dna.md §5.3.8).
use std::sync::Arc;

use seqblaze_core::{detail::try_reserve_exact, validate::EngineError};

use crate::params::Ratio;

pub const MAGIC: &str = "seqblaze-k1-pack\t1";
/// The first line of a format-2 pack (nndb2024-dna.md §4.4).
pub const MAGIC_2: &str = "seqblaze-k1-pack\t2";
/// STATES[σ] exists for σ = 1 … 200 (nndb2024-dna.md §3.6).
pub const STATES_MAX: usize = 200;
/// The pair-type map's "no pair" (nndb2024-dna.md §5.3.1).
pub const NO_PAIR: u8 = u8::MAX;
/// The three loop classes of a format-2 pack, in grammar order (nndb2024-dna.md §3.5).
pub const CLASS_NAMES: &str = "1n,23,other";
/// The largest loop two 200-nt strands can hold (§12.2 `loop_max`).
pub const LOOP_MAX: usize = 396;
/// The number of letters, and the ∅ index of the end tables.
pub const NONE: usize = 4;
const TWO53: i128 = 1 << 53;
/// Base-entry roles of §12.2 (the salt-weight roles).
/// Format 2 adds the role `states` (the last item, nndb2024-dna.md §4.4).
const ROLES: [&str; 13] = [
    "stack",
    "imm",
    "tmm_loop",
    "tmm_end",
    "int",
    "de",
    "loop",
    "asym",
    "closing_at",
    "terminal_at",
    "init",
    "symmetry",
    "states",
];
/// Every line tag of the grammar: a wrong line is reported by its tag only when the tag
/// is one of these (never by another field's text, §5.3.8).
const TAGS: [&str; 16] = [
    "seqblaze-k1-pack",
    "lines",
    "name",
    "definition_sha256",
    "loop_max",
    "salt_b",
    "gas_r",
    "naeq_mg",
    "pairs",
    "classes",
    "entries",
    "entry",
    "table",
    "cell",
    "cls",
    "end",
];

/// A base entry (§4.5): h and ĝ in dcal/mol, q = 2p.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Entry {
    pub h: i64,
    pub g: i64,
    pub q: i64,
}

/// A motif cell: `None` is ⊥, otherwise a multiset of (entry id, multiplicity).
pub type Cell = Option<Box<[(u32, u32)]>>;

/// The motif tables of §4.5, each in its canonical index order, indexed by the pair type
/// P (nndb2024-dna.md §5.3.1). `gen` is dense over (u, v) in (Λ+1) × (Λ+1), ⊥ outside
/// the GEN shapes; `loop_out` and `loop_in` carry the class index first; `states` holds
/// STATES[1] … STATES[200] (all empty for a format-1 pack).
#[derive(Debug, Clone)]
pub struct Tables<T> {
    pub init: T,
    pub sym: T,
    pub end_l: Vec<T>,
    pub end_r: Vec<T>,
    pub stack: Vec<T>,
    pub bulge: Vec<T>,
    pub i11: Vec<T>,
    pub i12: Vec<T>,
    pub i21: Vec<T>,
    pub i22: Vec<T>,
    pub gen: Vec<T>,
    pub loop_out: Vec<T>,
    pub loop_in: Vec<T>,
    pub states: Vec<T>,
}

impl<T> Tables<T> {
    /// Apply `f` to every cell, keeping the layout. Every table is reserved fallibly
    /// (§12.4): a failed reservation is a `DeviceError` naming `what`, never an abort.
    pub fn try_map<U>(
        &self,
        mut f: impl FnMut(&T) -> U,
        what: &str,
    ) -> Result<Tables<U>, EngineError> {
        let mut all = |cells: &Vec<T>| -> Result<Vec<U>, EngineError> {
            let mut out = Vec::new();
            try_reserve_exact(&mut out, cells.len(), what)?;
            out.extend(cells.iter().map(&mut f));
            Ok(out)
        };
        let end_l = all(&self.end_l)?;
        let end_r = all(&self.end_r)?;
        let stack = all(&self.stack)?;
        let bulge = all(&self.bulge)?;
        let i11 = all(&self.i11)?;
        let i12 = all(&self.i12)?;
        let i21 = all(&self.i21)?;
        let i22 = all(&self.i22)?;
        let gen = all(&self.gen)?;
        let loop_out = all(&self.loop_out)?;
        let loop_in = all(&self.loop_in)?;
        let states = all(&self.states)?;
        Ok(Tables {
            init: f(&self.init),
            sym: f(&self.sym),
            end_l,
            end_r,
            stack,
            bulge,
            i11,
            i12,
            i21,
            i22,
            gen,
            loop_out,
            loop_in,
            states,
        })
    }
}

/// A loaded pack: constants, base entries and motif tables.
#[derive(Debug, Clone)]
pub struct Pack {
    pub name: String,
    /// SHA-256 of the build-definition text (`model.params.sha256`).
    pub sha256: String,
    /// Λ.
    pub loop_max: usize,
    pub salt_b: Ratio,
    pub gas_r: Ratio,
    pub naeq_mg: Ratio,
    pub entries: Vec<Entry>,
    pub tables: Tables<Cell>,
    /// The step shapes (u, v) of GEN, in canonical order.
    pub gen_shapes: Vec<(usize, usize)>,
    /// The text's format, 1 or 2.
    pub format: u8,
    /// N_P: 4 (`pairs wc`, every format-1 pack) or 6 (`pairs wc+gt`).
    pub n_p: usize,
    /// N_C: 1 (`classes 1 all`, every format-1 pack) or 3 (`1n`, `23`, `other`).
    pub n_c: usize,
    /// PT[x·4 + y]: the pair type of top letter x against bottom letter y, or
    /// [`NO_PAIR`] (nndb2024-dna.md §5.3.1).
    pub pt: [u8; 16],
    /// CLS dense over (u, v) in (Λ+1) × (Λ+1); 0 outside the GEN shapes (never read).
    /// Shared, so a per-call table needs no copy.
    pub cls: Arc<[u8]>,
}

impl Pack {
    /// True iff the pack has the shape of format 1: N_P = 4, N_C = 1 and every STATES
    /// cell empty (nndb2024-dna.md §5.3.7).
    pub fn format1_shaped(&self) -> bool {
        self.n_p == 4
            && self.n_c == 1
            && self
                .tables
                .states
                .iter()
                .all(|cell| matches!(cell.as_deref(), Some([])))
    }
}

/// The pair-type map of a `pairs` token (nndb2024-dna.md §5.3.1): A–T 0, C–G 1, G–C 2,
/// T–A 3, and with G·T pairs G–T 4, T–G 5.
pub fn pair_map(gt: bool) -> [u8; 16] {
    let mut pt = [NO_PAIR; 16];
    for x in 0..4 {
        pt[x * 4 + (3 - x)] = x as u8;
    }
    if gt {
        pt[2 * 4 + 3] = 4;
        pt[3 * 4 + 2] = 5;
    }
    pt
}

/// The class of a GEN shape (nndb2024-dna.md §3.5): 0 (`1n`) iff min(u, v) = 1, 1 (`23`)
/// iff (u, v) is (2, 3) or (3, 2), 2 (`other`) otherwise.
pub fn class_of(u: usize, v: usize) -> u8 {
    if u.min(v) == 1 {
        0
    } else if (u, v) == (2, 3) || (u, v) == (3, 2) {
        1
    } else {
        2
    }
}

fn bad(line: usize, what: impl std::fmt::Display) -> EngineError {
    EngineError::Parameter(format!(
        "duplex pack line {line}: {what}; the pack was not loaded"
    ))
}

struct Reader<'t> {
    lines: Vec<&'t str>,
    pos: usize,
}

impl<'t> Reader<'t> {
    /// The fields of the next line, which must start with `tag` and have `n` fields.
    fn take(&mut self, tag: &str, n: usize) -> Result<(usize, Vec<&'t str>), EngineError> {
        let number = self.pos + 1;
        let Some(line) = self.lines.get(self.pos) else {
            return Err(bad(number, format!("missing; expected a {tag:?} line")));
        };
        self.pos += 1;
        let fields: Vec<&str> = line.split('\t').collect();
        if fields[0] != tag || fields.len() != n {
            // The found line by its tag (a grammar tag only) and field count (§5.3.8).
            let found = if TAGS.contains(&fields[0]) {
                format!("a {:?} line", fields[0])
            } else {
                "a line with another tag".to_string()
            };
            return Err(bad(
                number,
                format!(
                    "expected a {tag:?} line with {n} fields, found {found} with {} fields",
                    fields.len()
                ),
            ));
        }
        Ok((number, fields))
    }
}

fn int<T: std::str::FromStr>(line: usize, text: &str, what: &str) -> Result<T, EngineError> {
    let canonical = text == "0"
        || text
            .strip_prefix('-')
            .unwrap_or(text)
            .bytes()
            .enumerate()
            .all(|(k, c)| c.is_ascii_digit() && (k > 0 || c != b'0'))
            && !text.is_empty()
            && text != "-";
    // The field by its name only, never its text (§5.3.8).
    if !canonical {
        return Err(bad(line, format!("{what} is not a canonical integer")));
    }
    text.parse::<T>()
        .map_err(|_| bad(line, format!("{what} is out of range")))
}

fn ratio(reader: &mut Reader<'_>, tag: &str) -> Result<Ratio, EngineError> {
    let (line, f) = reader.take(tag, 3)?;
    let num: i64 = int(line, f[1], &format!("the {tag} numerator"))?;
    let den: u64 = int(line, f[2], &format!("the {tag} denominator"))?;
    if num <= 0 || den == 0 {
        return Err(bad(line, format!("{tag} must be a positive rational")));
    }
    Ok(Ratio { num, den })
}

/// Canonical indices of a table (the last index varies fastest).
fn product(dims: &[usize]) -> Vec<Vec<usize>> {
    let mut out = vec![Vec::new()];
    for &d in dims {
        out = out
            .into_iter()
            .flat_map(|prefix| {
                (0..d).map(move |k| {
                    let mut next = prefix.clone();
                    next.push(k);
                    next
                })
            })
            .collect();
    }
    out
}

fn gen_shapes(lam: usize) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    for u in 1..=lam {
        for v in 1..=lam - u {
            if !(u <= 2 && v <= 2) {
                out.push((u, v));
            }
        }
    }
    out
}

fn cell(line: usize, text: &str, entries: usize) -> Result<Cell, EngineError> {
    match text {
        "bot" => return Ok(None),
        "empty" => return Ok(Some(Box::new([]))),
        _ => {}
    }
    let mut items = Vec::new();
    for item in text.split(',') {
        let (id, mult) = item
            .split_once('x')
            .ok_or_else(|| bad(line, "a cell item is not <id>x<multiplicity>"))?;
        let id: u32 = int(line, id, "entry id")?;
        let mult: u32 = int(line, mult, "multiplicity")?;
        if id as usize >= entries {
            return Err(bad(line, format!("entry id {id} is not below {entries}")));
        }
        if mult == 0 || mult as usize > 2 * LOOP_MAX {
            return Err(bad(line, format!("multiplicity {mult} out of range")));
        }
        if items.last().is_some_and(|&(last, _)| last >= id) {
            return Err(bad(
                line,
                format!("entry ids must increase strictly within a cell ({id})"),
            ));
        }
        items.push((id, mult));
    }
    Ok(Some(items.into_boxed_slice()))
}

/// A table: its header and exactly its cells in `indices` order. `never_bot` rejects a ⊥
/// cell on its own line (GEN and STATES test no opposition, so they are never ⊥).
fn table(
    reader: &mut Reader<'_>,
    name: &str,
    indices: &[Vec<usize>],
    entries: usize,
    never_bot: bool,
) -> Result<Vec<Cell>, EngineError> {
    let (line, f) = reader.take("table", 3)?;
    if f[1] != name {
        return Err(bad(line, format!("expected table {name}, found {}", f[1])));
    }
    let count: usize = int(line, f[2], "cell count")?;
    if count != indices.len() {
        return Err(bad(
            line,
            format!(
                "table {name} must have {} cells, not {count}",
                indices.len()
            ),
        ));
    }
    let mut out = Vec::with_capacity(count);
    for index in indices {
        let (line, f) = reader.take("cell", 4)?;
        let want = index
            .iter()
            .map(usize::to_string)
            .collect::<Vec<_>>()
            .join(",");
        if f[1] != name || f[2] != want {
            return Err(bad(
                line,
                format!("expected cell {name}[{want}], found {}[{}]", f[1], f[2]),
            ));
        }
        let content = cell(line, f[3], entries)?;
        if never_bot && content.is_none() {
            // ⊥ marks rule F only (§4.5), and these cells test no opposition. Every
            // generic step adds GEN to the two closing-pair cells, so a ⊥ there would
            // also let a finite LOOP_OUT or LOOP_IN cancel +∞.
            return Err(bad(
                line,
                format!("{name}[{want}] is bot; a {name} cell is never bot (duplex-mfe.md §4.5)"),
            ));
        }
        out.push(content);
    }
    Ok(out)
}

/// The §4.4 bound M(q) in pack terms for a per-entry quantity `value(id)`, in the form of
/// nndb2024-dna.md §4.6 (the compiler's `_k1_bound` and the reference's `bound`).
pub fn pack_bound(pack: &Pack, value: impl Fn(usize) -> i128) -> i128 {
    pack_bound_parts(pack, value).0
}

/// B of [`pack_bound`] with its parts max_step and max_end (§4.4; the C1 width analysis,
/// duplex-mfe-bound.md §4.6): `(B, max_step, max_end)`.
pub fn pack_bound_parts(pack: &Pack, value: impl Fn(usize) -> i128) -> (i128, i128, i128) {
    let total = |cell: &Cell| -> Option<i128> {
        cell.as_ref().map(|items| {
            items
                .iter()
                .map(|&(id, m)| value(id as usize).abs() * i128::from(m))
                .sum()
        })
    };
    // Iterators only: this runs on every call, and must not allocate (§12.4).
    let largest = |cells: &mut dyn Iterator<Item = &Cell>| cells.filter_map(&total).max();
    let t = &pack.tables;
    let mut steps = [&t.stack, &t.bulge, &t.i11, &t.i12, &t.i21, &t.i22]
        .into_iter()
        .flat_map(|cells| cells.iter());
    let step = largest(&mut steps).unwrap_or(0);
    // LOOP_OUT and LOOP_IN count even when Λ leaves GEN without shapes (Λ ≤ 3): the
    // bound covers every cell of the pack, read or not (§4.4, §4.5), so no cell key
    // can leave the int64 range the recurrence relies on.
    let gen = largest(&mut t.gen.iter()).unwrap_or(0);
    let out = largest(&mut t.loop_out.iter()).unwrap_or(0);
    let inn = largest(&mut t.loop_in.iter()).unwrap_or(0);
    // nndb2024-dna.md §4.6: LOOP_OUT and LOOP_IN maxima over all classes (the flat
    // tables hold every class), and the largest STATES sum added to every step (0 when
    // every STATES cell is empty, so format-1 packs keep today's bound).
    let states = largest(&mut t.states.iter()).unwrap_or(0);
    let step = step.max(gen + out + inn) + states;
    let end = largest(&mut t.end_l.iter().chain(t.end_r.iter())).unwrap_or(0);
    let bound = total(&t.init).unwrap_or(0) + total(&t.sym).unwrap_or(0) + 2 * end + 199 * step;
    (bound, step, end)
}

impl Pack {
    /// Parse and check a compiled pack text of format 1 or 2.
    pub fn parse(text: &str) -> Result<Self, EngineError> {
        if text.is_empty() {
            return Err(bad(1, "the pack text is empty"));
        }
        if text.contains('\r') || !text.ends_with('\n') {
            return Err(bad(
                1,
                "the pack text must use LF line ends and end with LF",
            ));
        }
        let lines: Vec<&str> = text[..text.len() - 1].split('\n').collect();
        let mut reader = Reader { lines, pos: 0 };
        let (line, f) = reader.take("seqblaze-k1-pack", 2)?;
        let joined = f.join("\t");
        let format: u8 = if joined == MAGIC {
            1
        } else if joined == MAGIC_2 {
            2
        } else {
            return Err(bad(line, "not a seqblaze K1 pack of format 1 or 2"));
        };
        let (line, f) = reader.take("lines", 2)?;
        let total: usize = int(line, f[1], "line count")?;
        if total != reader.lines.len() {
            return Err(bad(
                line,
                format!(
                    "the pack declares {total} lines but has {}",
                    reader.lines.len()
                ),
            ));
        }
        let (line, f) = reader.take("name", 2)?;
        let name = f[1].to_string();
        if name.is_empty() || name.chars().any(|c| c.is_whitespace() || c.is_control()) {
            return Err(bad(line, format!("invalid pack name {name:?}")));
        }
        let (line, f) = reader.take("definition_sha256", 2)?;
        let sha256 = f[1].to_string();
        if sha256.len() != 64
            || !sha256
                .bytes()
                .all(|c| c.is_ascii_digit() || (b'a'..=b'f').contains(&c))
        {
            return Err(bad(
                line,
                "definition_sha256 must be 64 lower-case hex digits",
            ));
        }
        let (line, f) = reader.take("loop_max", 2)?;
        let loop_max: usize = int(line, f[1], "loop_max")?;
        if loop_max > LOOP_MAX {
            return Err(bad(line, format!("loop_max {loop_max} exceeds {LOOP_MAX}")));
        }
        let salt_b = ratio(&mut reader, "salt_b")?;
        let gas_r = ratio(&mut reader, "gas_r")?;
        let naeq_mg = ratio(&mut reader, "naeq_mg")?;
        // Format 2: the pair set and the loop classes (nndb2024-dna.md §5.3.2 items 2-3).
        let (gt, n_c) = if format == 2 {
            let (line, f) = reader.take("pairs", 2)?;
            let gt = match f[1] {
                "wc" => false,
                "wc+gt" => true,
                _ => return Err(bad(line, "the pairs token must be wc or wc+gt")),
            };
            let (line, f) = reader.take("classes", 3)?;
            let n_c = match (f[1], f[2]) {
                ("1", "all") => 1,
                ("3", CLASS_NAMES) => 3,
                _ => {
                    return Err(bad(
                        line,
                        format!("the classes line must be 1 all or 3 {CLASS_NAMES}"),
                    ))
                }
            };
            (gt, n_c)
        } else {
            (false, 1)
        };
        let n_p = if gt { 6 } else { 4 };
        let (line, f) = reader.take("entries", 2)?;
        let count: usize = int(line, f[1], "entry count")?;
        if count > 1 << 20 {
            return Err(bad(line, format!("{count} entries is out of range")));
        }
        // The role `states` (the last of ROLES) exists in format 2 only.
        let roles = if format == 2 {
            &ROLES[..]
        } else {
            &ROLES[..ROLES.len() - 1]
        };
        let mut entries = Vec::with_capacity(count);
        for id in 0..count {
            let (line, f) = reader.take("entry", 7)?;
            let k: usize = int(line, f[1], "entry id")?;
            if k != id {
                return Err(bad(line, format!("expected entry {id}, found another id")));
            }
            let h: i64 = int(line, f[2], "the entry's h")?;
            let g: i64 = int(line, f[3], "the entry's g")?;
            let q: i64 = int(line, f[4], "the entry's q")?;
            if i128::from(h).abs() >= TWO53 || i128::from(g).abs() >= TWO53 {
                return Err(bad(
                    line,
                    "an entry energy is out of range (|value| >= 2^53)",
                ));
            }
            if ![0, 1, 2, 4].contains(&q) {
                return Err(bad(line, "the entry's q = 2p must be 0, 1, 2 or 4"));
            }
            if !roles.contains(&f[5]) || f[6].is_empty() {
                return Err(bad(
                    line,
                    format!("unknown entry role for pack format {format}, or an empty key"),
                ));
            }
            entries.push(Entry { h, g, q });
        }
        let e = entries.len();
        let one = [vec![0usize]];
        let init = table(&mut reader, "INIT", &one, e, false)?.pop().flatten();
        let sym = table(&mut reader, "SYM", &one, e, false)?.pop().flatten();
        let end_l = table(&mut reader, "END_L", &product(&[5, n_p, 5]), e, false)?;
        let end_r = table(&mut reader, "END_R", &product(&[n_p, 5, 5]), e, false)?;
        let stack = table(&mut reader, "STACK", &product(&[n_p, n_p]), e, false)?;
        let bulge_index: Vec<Vec<usize>> = product(&[2, loop_max, n_p, n_p])
            .into_iter()
            .map(|mut index| {
                index[1] += 1;
                index
            })
            .collect();
        let bulge = table(&mut reader, "BULGE", &bulge_index, e, false)?;
        let i11 = table(&mut reader, "I11", &product(&[n_p, 4, 4, n_p]), e, false)?;
        let i12 = table(&mut reader, "I12", &product(&[n_p, 4, 4, 4, n_p]), e, false)?;
        let i21 = table(&mut reader, "I21", &product(&[n_p, 4, 4, 4, n_p]), e, false)?;
        let i22 = table(
            &mut reader,
            "I22",
            &product(&[n_p, 4, 4, 4, 4, n_p]),
            e,
            false,
        )?;
        let shapes = gen_shapes(loop_max);
        let gen_index: Vec<Vec<usize>> = shapes.iter().map(|&(u, v)| vec![u, v]).collect();
        let listed = table(&mut reader, "GEN", &gen_index, e, true)?;
        let width = loop_max + 1;
        let mut gen: Vec<Cell> = vec![None; width * width];
        for (&(u, v), cell) in shapes.iter().zip(listed) {
            gen[u * width + v] = cell;
        }
        let mut cls = vec![0u8; width * width];
        if format == 2 {
            // The class map, one `cls` line per GEN shape in GEN's order (§5.3.2 item 6),
            // checked against the classes the header names (§3.5; N_C = 1: all 0).
            let (line, f) = reader.take("table", 3)?;
            if f[1] != "CLS" {
                return Err(bad(line, "expected table CLS"));
            }
            let count: usize = int(line, f[2], "cell count")?;
            if count != shapes.len() {
                return Err(bad(
                    line,
                    format!("table CLS must have {} lines, not {count}", shapes.len()),
                ));
            }
            for &(u, v) in &shapes {
                let (line, f) = reader.take("cls", 4)?;
                let want = format!("{u},{v}");
                if f[1] != "CLS" || f[2] != want {
                    return Err(bad(line, format!("expected the CLS line of ({want})")));
                }
                let c: u8 = int(line, f[3], "the class index")?;
                let expected = if n_c == 1 { 0 } else { class_of(u, v) };
                if c != expected {
                    return Err(bad(
                        line,
                        format!(
                            "CLS[{u}][{v}] must be {expected}, the class of ({u}, {v}) under \
                             the classes line (nndb2024-dna.md §3.5)"
                        ),
                    ));
                }
                cls[u * width + v] = c;
            }
        }
        let (out_dims, in_dims): (&[usize], &[usize]) = if format == 2 {
            (&[n_c, n_p, 4, 4], &[n_c, 4, 4, n_p])
        } else {
            (&[4, 4, 4], &[4, 4, 4])
        };
        let loop_out = table(&mut reader, "LOOP_OUT", &product(out_dims), e, false)?;
        let loop_in = table(&mut reader, "LOOP_IN", &product(in_dims), e, false)?;
        let states = if format == 2 {
            let index: Vec<Vec<usize>> = (1..=STATES_MAX).map(|sigma| vec![sigma]).collect();
            table(&mut reader, "STATES", &index, e, true)?
        } else {
            vec![Some(Box::<[(u32, u32)]>::from([])); STATES_MAX]
        };
        let (line, _) = reader.take("end", 1)?;
        if reader.pos != reader.lines.len() {
            return Err(bad(line + 1, "text after the end line"));
        }
        // INIT and SYM are one base entry each (§4.5).
        for (what, cell) in [("INIT", &init), ("SYM", &sym)] {
            if !matches!(cell.as_deref(), Some([(_, 1)])) {
                return Err(bad(line, format!("{what} must hold one entry once")));
            }
        }
        let pack = Self {
            name,
            sha256,
            loop_max,
            salt_b,
            gas_r,
            naeq_mg,
            entries,
            tables: Tables {
                init,
                sym,
                end_l,
                end_r,
                stack,
                bulge,
                i11,
                i12,
                i21,
                i22,
                gen,
                loop_out,
                loop_in,
                states,
            },
            gen_shapes: shapes,
            format,
            n_p,
            n_c,
            pt: pair_map(gt),
            cls: cls.into(),
        };
        // §4.4 build checks with the bound of nndb2024-dna.md §4.6, re-checked on load:
        // 200(H_max + G_max), 10·H_max, Q_max < 2^53. The message names the failed
        // inequality, never a maximum (§5.3.8).
        let entry = |id: usize| pack.entries[id];
        let h_max = pack_bound(&pack, |id| i128::from(entry(id).h));
        let g_max = pack_bound(&pack, |id| i128::from(entry(id).g));
        let q_max = pack_bound(&pack, |id| i128::from(entry(id).q));
        let failed: Vec<&str> = [
            ("200·(H + G) < 2^53", 200 * (h_max + g_max) < TWO53),
            ("10·H < 2^53", 10 * h_max < TWO53),
            ("Q < 2^53", q_max < TWO53),
        ]
        .into_iter()
        .filter_map(|(what, ok)| (!ok).then_some(what))
        .collect();
        if !failed.is_empty() {
            return Err(bad(
                line,
                format!(
                    "the pack is out of range: its largest structure sums H, G and Q of h, ĝ \
                     and q fail {} and break the 2^53 bounds of duplex-mfe.md §4.4",
                    failed.join(", ")
                ),
            ));
        }
        Ok(pack)
    }
}
