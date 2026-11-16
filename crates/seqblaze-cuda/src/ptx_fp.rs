//! The floating-point scan of the device module's PTX (D55, the phase-1 "no device
//! floating point" rule; tm-cuda.md §4.7, duplex-pf-cuda.md §8 item 6). `build.rs`
//! includes this file and runs [`first_fp`] on the module's PTX in every `cuda` build;
//! `lib.rs` includes it under `cfg(test)` only, so the default `make test` checks the
//! scan itself without the `cuda` feature.

/// Whether `word` names a floating-point PTX type (PTX ISA 8.x and 9.x). The type
/// grammar, not a list, so packed forms added later are covered too: one or more float
/// formats, then an optional packing suffix `x<N>` (`f16x2`, `f32x2`, `e4m3x4`,
/// `ue8m0x2`). A format is `f`, `bf` or `tf` with a bit width (`f16`, `bf16`, `tf32`,
/// `f32`, `f64`), or `e<E>m<M>` with an optional `u` (`e4m3`, `e5m2`, `e2m1`, `e3m2`,
/// `e2m3`, `ue8m0`, `ue4m3`), optionally prefixed by `mx` or `nv` (the `.kind::` operand
/// classes `f8f6f4`, `mxf8f6f4`, `mxf4nvf4`). Every format needs its digits, so `ftz`,
/// `full`, `func`, `entry`, `extern` and `eq` are not types.
pub(crate) fn is_fp_type(word: &str) -> bool {
    let mut rest = word.as_bytes();
    let mut formats = 0;
    while format(&mut rest) {
        formats += 1;
    }
    formats > 0
        && (rest.is_empty() || (eat(&mut rest, b"x") && eat_digits(&mut rest) && rest.is_empty()))
}

/// One float format at the start of `rest`; advances past it only when there is one.
fn format(rest: &mut &[u8]) -> bool {
    let mut tail = *rest;
    if !eat(&mut tail, b"mx") {
        eat(&mut tail, b"nv");
    }
    let found = if eat(&mut tail, b"bf") || eat(&mut tail, b"tf") || eat(&mut tail, b"f") {
        eat_digits(&mut tail)
    } else {
        eat(&mut tail, b"u");
        eat(&mut tail, b"e")
            && eat_digits(&mut tail)
            && eat(&mut tail, b"m")
            && eat_digits(&mut tail)
    };
    if found {
        *rest = tail;
    }
    found
}

/// Advances past `prefix` when `rest` starts with it.
fn eat(rest: &mut &[u8], prefix: &[u8]) -> bool {
    match rest.strip_prefix(prefix) {
        Some(tail) => {
            *rest = tail;
            true
        }
        None => false,
    }
}

/// Advances past the leading decimal digits; whether there was at least one.
fn eat_digits(rest: &mut &[u8]) -> bool {
    let count = rest.iter().take_while(|c| c.is_ascii_digit()).count();
    *rest = &rest[count..];
    count > 0
}

/// The first identifier of `code` that follows a `.` or a `::` and names a
/// floating-point type ([`is_fp_type`]). Every `.`-component counts, wherever it
/// stands: the type suffixes of an instruction (`add.rn.f32x2`, `cvt.rn.f16x2.e4m3x2`)
/// as well as declarations (`.reg .f32`) and `.kind::f16`. PTX identifiers contain no
/// `.`, so no name is split.
pub(crate) fn fp_token(code: &str) -> Option<&str> {
    let bytes = code.as_bytes();
    let ident = |c: &u8| c.is_ascii_alphanumeric() || *c == b'_' || *c == b'$';
    let mut at = 0;
    while at < bytes.len() {
        let start = if bytes[at] == b'.' {
            at + 1
        } else if bytes[at..].starts_with(b"::") {
            at + 2
        } else {
            at += 1;
            continue;
        };
        let end = start + bytes[start..].iter().take_while(|c| ident(c)).count();
        let word = &code[start..end];
        if is_fp_type(word) {
            return Some(word);
        }
        at = end;
    }
    None
}

/// The first line (1-based) whose code, without `//` and `/* */` comments and string
/// literals, names a floating-point PTX type, with that line and the type.
pub(crate) fn first_fp(text: &str) -> Option<(usize, &str, String)> {
    let mut in_block = false;
    for (index, line) in text.lines().enumerate() {
        let code = code_of(line, &mut in_block);
        if let Some(token) = fp_token(&code) {
            return Some((index + 1, line, token.to_owned()));
        }
    }
    None
}

/// `line` without its comments and string literals (each replaced by a space);
/// `in_block` carries an open `/* */` comment from line to line.
fn code_of(line: &str, in_block: &mut bool) -> String {
    let bytes = line.as_bytes();
    let mut code = String::new();
    // `run`: start of the code not yet copied into `code`.
    let (mut at, mut run) = (0, 0);
    while at < bytes.len() {
        let rest = &bytes[at..];
        if *in_block {
            if rest.starts_with(b"*/") {
                *in_block = false;
                at += 2;
                run = at;
            } else {
                at += 1;
            }
        } else if rest.starts_with(b"//") {
            break;
        } else if rest.starts_with(b"/*") || rest[0] == b'"' {
            code.push_str(&line[run..at]);
            code.push(' ');
            if rest[0] == b'"' {
                at += 1;
                while at < bytes.len() && bytes[at] != b'"' {
                    at += if bytes[at] == b'\\' { 2 } else { 1 };
                }
                at = (at + 1).min(bytes.len());
                run = at;
            } else {
                *in_block = true;
                at += 2;
            }
        } else {
            at += 1;
        }
    }
    if !*in_block {
        code.push_str(&line[run..at]);
    }
    code
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every floating-point PTX type of PTX ISA 8.x/9.x, the packed forms included.
    const TYPES: &[&str] = &[
        "f16", "f16x2", "bf16", "bf16x2", "tf32", "f32", "f32x2", "f64", "e4m3", "e5m2", "e4m3x2",
        "e5m2x2", "e4m3x4", "e5m2x4", "e2m1", "e2m1x2", "e2m1x4", "e3m2", "e3m2x2", "e3m2x4",
        "e2m3", "e2m3x2", "e2m3x4", "ue8m0", "ue8m0x2", "ue4m3", "f8f6f4", "mxf8f6f4", "mxf4",
        "mxf4nvf4",
    ];

    /// Integer and bit types, modifiers and directive names that must not match.
    const NOT_TYPES: &[&str] = &[
        "b8", "b16", "b32", "b64", "b128", "u8", "u16", "u32", "u64", "s8", "s16", "s32", "s64",
        "u16x2", "s16x2", "b8x16", "pred", "v2", "v4", "x4", "ftz", "full", "func", "entry",
        "extern", "eq", "rn", "rz", "approx", "sat", "f", "tf", "bf", "e", "e4", "e4m", "u", "x2",
        "f16x", "e4m3y", "m16n8k16", "L2", "nc", "global", "shared", "param", "reg", "loc", "file",
        "version", "target", "kind", "i8", "cta", "",
    ];

    #[test]
    fn the_type_grammar_covers_every_floating_point_type() {
        for word in TYPES {
            assert!(is_fp_type(word), "{word} is a floating-point type");
        }
        for word in NOT_TYPES {
            assert!(!is_fp_type(word), "{word:?} is not a floating-point type");
        }
    }

    #[test]
    fn instruction_suffixes_and_packed_forms_are_found() {
        // The first two lines are nvcc 12.8 PTX of `__fadd2_rn` and `__ffma2_rn` on b64
        // registers (no `.f32` anywhere), and the next two are the inline asm of `__hadd`
        // and `__hadd2` on b16 and b32 registers (four determinism flags,
        // -arch=compute_120; measured 2026-10-18). The scan before this change passed
        // all four lines.
        for (line, token) in [
            ("\tadd.rn.f32x2 \t%rd5, %rd3, %rd4;", "f32x2"),
            ("\tfma.rn.f32x2 \t%rd6, %rd5, %rd3, %rd4;", "f32x2"),
            ("\t{add.f16 %rs1,%rs2,%rs3;", "f16"),
            ("\t{add.f16x2 %r1,%r2,%r3;", "f16x2"),
            ("\tadd.rn.f32 \t%f3, %f1, %f2;", "f32"),
            ("\t.reg .f32 \t%f<4>;", "f32"),
            ("\tcvt.rn.f64.s64 \t%fd1, %rd1;", "f64"),
            ("\tcvt.rn.satfinite.e4m3x2.f32 \t%rs1, %f1, %f2;", "e4m3x2"),
            ("\tcvt.rn.f16x2.e2m1x2 \t%r1, %rs1;", "f16x2"),
            ("\tcvt.rn.satfinite.ue8m0x2.bf16x2 %rs1, %r1;", "ue8m0x2"),
            (
                "\ttcgen05.mma.cta.kind::f16 [%r1], %rd1, %rd2, %r2, %p1;",
                "f16",
            ),
            (
                "\tmma.sync.aligned.kind::mxf8f6f4.block_scale.m16n8k32.row.col",
                "mxf8f6f4",
            ),
        ] {
            assert_eq!(fp_token(line), Some(token), "{line}");
            assert_eq!(first_fp(line), Some((1, line, token.to_owned())), "{line}");
        }
    }

    #[test]
    fn integer_code_and_directives_pass() {
        for line in [
            "\tadd.s64 \t%rd3, %rd1, %rd2;",
            "\t.reg .b64 \t%rd<7>;",
            "\tsetp.eq.s32 \t%p1, %r1, 0;",
            "\tmul.hi.u64 \t%rd3, %rd1, %rd2;",
            "\tmad.lo.s64 \t%rd4, %rd1, %rd2, %rd3;",
            "\tmov.u32 \t%r1, %tid.x;",
            "\tmov.b32 \t%r1, 0f3F800000;",
            "\tld.global.nc.L1::no_allocate.b32 \t%r1, [%rd1];",
            "\tst.shared.v2.u64 \t[%rd1], {%rd2, %rd3};",
            "\t@%p1 bra \t$L__BB0_2;",
            ".version 8.7",
            ".target sm_120",
            ".address_size 64",
            ".visible .entry sb_duplex_pf(",
            "\t.param .u64 .ptr .global .align 8 sb_duplex_pf_param_0,",
            "\t.local .align 8 .b8 \t__local_depot0[16];",
        ] {
            assert_eq!(fp_token(line), None, "{line}");
        }
    }

    #[test]
    fn comments_and_strings_are_not_code() {
        let text = "// add.rn.f32 in a line comment\n\
                    /* .f64 opens a block\n still the block .f16x2 */ add.s64 %rd1, %rd2, %rd3;\n\
                    \t.file 1 \"/x/kernels.f16/a.f32x2.cu\"\n\
                    \tmov.u32 %r1, %tid.x; // .f32\n\
                    \t.pragma \"a \\\" .f64\";\n";
        assert_eq!(first_fp(text), None);
        let text = "\tadd.s64 %rd1, %rd2, %rd3;\n/* .s32 */ add.rn.f32x2 %rd5, %rd3, %rd4;\n";
        assert_eq!(
            first_fp(text),
            Some((
                2,
                "/* .s32 */ add.rn.f32x2 %rd5, %rd3, %rd4;",
                "f32x2".to_owned()
            ))
        );
        let text = "/* a\n b */ cvt.rn.f64.s64 %fd1, %rd1;\n";
        assert_eq!(first_fp(text).map(|hit| hit.0), Some(2));
    }
}
