//! Substitution tables in fixed point (spec §2, §5.3): the 256 x 256 score
//! table the DP reads, symbol classes for the score bound (§6) and the compat
//! identity and display maps.
use std::sync::{Arc, Mutex, OnceLock};

use seqblaze_core::validate::EngineError;

use crate::nuc44;

/// Fixed-point scale K (spec §2, Q8).
pub const SCALE: i64 = 1000;
/// Value range of the native preflight (spec §9.1).
pub const VALUE_RANGE: i64 = 1 << 30;

/// Substitution score source of one call, in fixed point (k = 1000 x).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Substitution {
    /// NCBI NUC.4.4 (spec §5.3).
    Ednafull,
    /// `match` if the bytes are equal, else `mismatch`.
    MatchMismatch { matched: i32, mismatched: i32 },
    /// A user 256 x 256 table, row-major (`table[x * 256 + y]`).
    Table(Vec<i32>),
}

/// The 15 EDNAFULL letters, upper case (native `matrix="EDNAFULL"`).
pub static EDNAFULL_LEGAL: [bool; 256] = ednafull_legal();
/// The compat alphabet: the 15 letters, `U` and `X`, either case (spec §5.3, Q3).
pub static COMPAT_LEGAL: [bool; 256] = compat_legal();

const fn ednafull_legal() -> [bool; 256] {
    let mut table = [false; 256];
    let mut k = 0;
    while k < nuc44::LETTERS.len() {
        table[nuc44::LETTERS[k] as usize] = true;
        k += 1;
    }
    table
}

const fn compat_legal() -> [bool; 256] {
    let mut table = ednafull_legal();
    let mut k = 0;
    while k < nuc44::LETTERS.len() {
        table[nuc44::LETTERS[k].to_ascii_lowercase() as usize] = true;
        k += 1;
    }
    table[b'U' as usize] = true;
    table[b'u' as usize] = true;
    table[b'X' as usize] = true;
    table[b'x' as usize] = true;
    table
}

/// NUC.4.4 row of an upper-case letter.
fn letter_index(letter: u8) -> Option<usize> {
    nuc44::LETTERS.iter().position(|&l| l == letter)
}

/// Compat scoring key (spec §5.3): upper case, then `U` -> `T`, then `X` -> `N`.
fn compat_key(byte: u8) -> Option<usize> {
    let upper = match byte.to_ascii_uppercase() {
        b'U' => b'T',
        b'X' => b'N',
        other => other,
    };
    letter_index(upper)
}

/// Tables of the fixed substitutions, built once per process: they depend on
/// the substitution alone, and a build fills and scans 256 x 256 entries, which
/// is most of a short call's time.
static EDNAFULL_TABLES: OnceLock<Arc<Tables>> = OnceLock::new();
static COMPAT_TABLES: OnceLock<Arc<Tables>> = OnceLock::new();
/// (match, mismatch) and the tables built for them.
type MatchTables = ((i32, i32), Arc<Tables>);
/// The last `match`/`mismatch` tables built.
static MATCH_TABLES: Mutex<Option<MatchTables>> = Mutex::new(None);

/// Fixed-point tables of one call; read-only once built, so calls share them.
pub(crate) struct Tables {
    /// `sub[x][y]`: fixed-point score of byte x (in `a`) against byte y (in `b`).
    pub sub: Box<[[i32; 256]; 256]>,
    /// Symbol class of each byte for the score bound (spec §6); compat classes
    /// are scoring keys.
    pub class: [u8; 256],
    pub n_classes: usize,
    /// `class_score[x * n_classes + y]`.
    pub class_score: Vec<i32>,
    /// sigma = max |s| over the legal alphabet (spec §9.1).
    pub sigma: i64,
    /// Identity key: two `M` bytes are an identity iff their keys are equal.
    pub identity: [u8; 256],
    /// Display map of the aligned rows.
    pub display: [u8; 256],
}

/// Row-equivalence classes of a 256 x 256 table (align-cuda.md §4, QAG5):
/// (class map, class count c, the c x c class table). Two bytes share a class
/// iff their rows are equal; classes follow the smallest byte of each.
pub(crate) fn row_classes(sub: &[[i32; 256]; 256]) -> ([u8; 256], usize, Vec<i32>) {
    let mut class = [0u8; 256];
    let mut first: Vec<usize> = Vec::new();
    let mut seen: std::collections::HashMap<&[i32; 256], u8> = std::collections::HashMap::new();
    for (x, row) in sub.iter().enumerate() {
        let next = first.len();
        let id = *seen.entry(row).or_insert_with(|| {
            first.push(x);
            next as u8
        });
        class[x] = id;
    }
    let c = first.len();
    let mut table = Vec::with_capacity(c * c);
    for &x in &first {
        for &y in &first {
            table.push(sub[x][y]);
        }
    }
    (class, c, table)
}

fn boxed_table() -> Box<[[i32; 256]; 256]> {
    vec![[0i32; 256]; 256]
        .into_boxed_slice()
        .try_into()
        .expect("256 rows")
}

fn identity_map() -> [u8; 256] {
    std::array::from_fn(|k| k as u8)
}

impl Tables {
    /// The compat tables, shared by every call.
    pub fn shared_compat() -> Arc<Self> {
        Arc::clone(COMPAT_TABLES.get_or_init(|| Arc::new(Self::compat())))
    }

    /// Native tables of `substitution`: EDNAFULL's are shared by every call,
    /// the last `match`/`mismatch` pair's by the calls that repeat it, and a
    /// user table is built for its call.
    pub fn shared_native(substitution: &Substitution) -> Result<Arc<Self>, EngineError> {
        match substitution {
            Substitution::Ednafull => {
                if let Some(tables) = EDNAFULL_TABLES.get() {
                    return Ok(Arc::clone(tables));
                }
                let built = Arc::new(Self::native(substitution)?);
                Ok(Arc::clone(EDNAFULL_TABLES.get_or_init(|| built)))
            }
            Substitution::MatchMismatch {
                matched,
                mismatched,
            } => {
                let key = (*matched, *mismatched);
                // A poisoned lock only skips the cache.
                if let Ok(last) = MATCH_TABLES.lock() {
                    if let Some((_, tables)) = last.as_ref().filter(|(seen, _)| *seen == key) {
                        return Ok(Arc::clone(tables));
                    }
                }
                let built = Arc::new(Self::native(substitution)?);
                if let Ok(mut last) = MATCH_TABLES.lock() {
                    *last = Some((key, Arc::clone(&built)));
                }
                Ok(built)
            }
            Substitution::Table(_) => Ok(Arc::new(Self::native(substitution)?)),
        }
    }

    /// Tables for native scoring; `table` entries are already fixed point.
    pub fn native(substitution: &Substitution) -> Result<Self, EngineError> {
        let mut sub = boxed_table();
        let (class, n_classes, legal): ([u8; 256], usize, Vec<u8>) = match substitution {
            Substitution::Ednafull => {
                let mut class = [0u8; 256];
                for (x, &lx) in nuc44::LETTERS.iter().enumerate() {
                    class[lx as usize] = x as u8;
                    for (y, &ly) in nuc44::LETTERS.iter().enumerate() {
                        sub[lx as usize][ly as usize] =
                            i32::from(nuc44::SCORES[x][y]) * SCALE as i32;
                    }
                }
                (class, nuc44::LETTERS.len(), nuc44::LETTERS.to_vec())
            }
            Substitution::MatchMismatch {
                matched,
                mismatched,
            } => {
                for (x, row) in sub.iter_mut().enumerate() {
                    for (y, value) in row.iter_mut().enumerate() {
                        *value = if x == y { *matched } else { *mismatched };
                    }
                }
                (identity_map(), 256, (0..=255).collect())
            }
            Substitution::Table(values) => {
                if values.len() != 1 << 16 {
                    return Err(EngineError::Parameter(format!(
                        "align matrix must have 65,536 entries (256 x 256), got {}",
                        values.len()
                    )));
                }
                for (x, row) in sub.iter_mut().enumerate() {
                    row.copy_from_slice(&values[x << 8..(x + 1) << 8]);
                }
                for x in 0..256 {
                    for y in 0..x {
                        if sub[x][y] != sub[y][x] {
                            return Err(EngineError::Parameter(format!(
                                "align matrix must be symmetric: entry [{x}, {y}] differs from [{y}, {x}]"
                            )));
                        }
                    }
                }
                (identity_map(), 256, (0..=255).collect())
            }
        };
        let mut tables = Self {
            sub,
            class,
            n_classes,
            class_score: Vec::new(),
            sigma: 0,
            identity: identity_map(),
            display: identity_map(),
        };
        tables.finish(&legal);
        Ok(tables)
    }

    /// Compat tables (spec §5.3): EDNAFULL through the scoring key over the
    /// 34 compat symbols, with the identity and display maps.
    pub fn compat() -> Self {
        let mut sub = boxed_table();
        let mut class = [0u8; 256];
        let mut identity = identity_map();
        let mut display = identity_map();
        let legal: Vec<u8> = (0..=255u8).filter(|&x| COMPAT_LEGAL[x as usize]).collect();
        for &x in &legal {
            let kx = compat_key(x).expect("compat symbol");
            class[x as usize] = kx as u8;
            for &y in &legal {
                let ky = compat_key(y).expect("compat symbol");
                sub[x as usize][y as usize] = i32::from(nuc44::SCORES[kx][ky]) * SCALE as i32;
            }
            identity[x as usize] = match x.to_ascii_uppercase() {
                b'X' => b'N',
                other => other,
            };
            display[x as usize] = match x {
                b'X' => b'N',
                b'x' => b'n',
                other => other,
            };
        }
        let mut tables = Self {
            sub,
            class,
            n_classes: nuc44::LETTERS.len(),
            class_score: Vec::new(),
            sigma: 0,
            identity,
            display,
        };
        tables.finish(&legal);
        tables
    }

    /// Class scores and sigma over the legal bytes.
    fn finish(&mut self, legal: &[u8]) {
        let k = self.n_classes;
        self.class_score = vec![i32::MIN; k * k];
        let mut sigma = 0i64;
        for &x in legal {
            for &y in legal {
                let value = self.sub[x as usize][y as usize];
                sigma = sigma.max(i64::from(value).abs());
                let slot = self.class[x as usize] as usize * k + self.class[y as usize] as usize;
                debug_assert!(
                    self.class_score[slot] == i32::MIN || self.class_score[slot] == value
                );
                self.class_score[slot] = value;
            }
        }
        self.sigma = sigma;
    }

    #[inline]
    pub fn score(&self, x: u8, y: u8) -> i64 {
        i64::from(self.sub[x as usize][y as usize])
    }

    #[inline]
    pub fn is_identity(&self, x: u8, y: u8) -> bool {
        self.identity[x as usize] == self.identity[y as usize]
    }
}
