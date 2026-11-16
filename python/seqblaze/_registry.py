"""Internal implemented-kernel registry; reconcile its format with I11 before freeze."""
from copy import deepcopy

_EXECUTION = {"strategy": "bounded", "device": "auto", "threads": None,
              "stream": None, "progress": False}
_SHAPES = ("zip", "index", "within", "cross", "one_to_many")
_SINKS = ("each", "hits", "topk", "rowstats", "matrix", "tiles")


def _signature(info):
    """The signed signature; a parameter marked "required" has no default (hairpin's
    compat, D48)."""
    return f"({info.get('input', 'pairs')}, /, *, " + ", ".join(
        name if definition.get("required") else f"{name}={definition['default']!r}"
        for name, definition in info["parameters"].items()
    ) + ", **exec)"


HAMMING = {
    "name": "hamming",
    "parameters": {"check_bases": {"default": True}, "fields": {"default": "default"},
                   "keep": {"default": None}},
    "execution": dict(_EXECUTION),
    "fields": {"dist": {"dtype": "int32", "unit": "edit operations", "exactness": "exact"}},
    "default_fields": ("dist",),
    "primary": "dist", "direction": "<=",
    "shapes": _SHAPES,
    "sinks": _SINKS,
    "backends": ("cpu", "cuda"), "presets": {},
    "citations": {"hamming1950": {
        "author": "Hamming, R. W.",
        "title": "Error Detecting and Error Correcting Codes",
        "journal": "Bell System Technical Journal",
        "year": 1950, "volume": 29, "number": 2, "pages": "147--160",
        "month": "April", "doi": "10.1002/j.1538-7305.1950.tb00463.x",
        "url": "https://doi.org/10.1002/j.1538-7305.1950.tb00463.x",
    }},
    "research_use_only": True,
}
HAMMING["signature"] = _signature(HAMMING)

_PREFIX_INFIX = ("dist", "b_end", "n_ends", "b_start", "n_starts", "ends", "cigar")

# docs/wiki/edit/levenshtein.md: §1 modes, §3 fields and masks, §4 cigar rule E
# (provisional, Q1), §7 within orientation, Sources line for the citations.
LEVENSHTEIN = {
    "name": "levenshtein",
    "parameters": {"mode": {"default": "global", "choices": ("global", "prefix", "infix")},
                   "orientation": {"default": "both", "choices": ("both", "as_given")},
                   "check_bases": {"default": True}, "fields": {"default": "default"},
                   "keep": {"default": None}},
    "execution": dict(_EXECUTION),
    "fields": {
        "dist": {"dtype": "int32", "unit": "edit operations", "exactness": "exact"},
        "b_end": {"dtype": "int64", "unit": "position", "exactness": "exact"},
        "n_ends": {"dtype": "int32", "unit": "count", "exactness": "exact"},
        "b_start": {"dtype": "int64", "unit": "position", "exactness": "exact"},
        # Optimal starts for the reported b_end, every tie counted (D88 (5); spec §3.1).
        "n_starts": {"dtype": "int32", "unit": "count", "exactness": "exact"},
        "ends": {"dtype": "list<int64>", "unit": "position", "exactness": "exact"},
        "cigar": {"dtype": "string", "unit": None, "exactness": "exact"},
    },
    "mode_fields": {"global": ("dist", "cigar"), "prefix": _PREFIX_INFIX, "infix": _PREFIX_INFIX},
    "default_fields": {"global": ("dist",), "prefix": ("dist", "b_end", "n_ends"),
                       "infix": ("dist", "b_end", "n_ends")},
    "primary": "dist", "direction": "<=",
    "shapes": _SHAPES,
    "sinks": _SINKS,
    "backends": ("cpu", "cuda"), "presets": {},
    "cigar_rule": "E: greedy from the window end, I then D then diagonal (provisional, Q1)",
    "citations": {
        "levenshtein1966": {
            "author": "Levenshtein, V. I.",
            "title": "Binary codes capable of correcting deletions, insertions, and reversals",
            "journal": "Soviet Physics Doklady", "year": 1966, "volume": 10, "number": 8,
            "pages": "707--710",
        },
        "wagner1974": {
            "author": "Wagner, R. A. and Fischer, M. J.",
            "title": "The string-to-string correction problem",
            "journal": "Journal of the ACM", "year": 1974, "volume": 21, "number": 1,
            "pages": "168--173", "doi": "10.1145/321796.321811",
            "url": "https://doi.org/10.1145/321796.321811",
        },
        "sellers1980": {
            "author": "Sellers, P. H.",
            "title": "The theory and computation of evolutionary distances: pattern recognition",
            "journal": "Journal of Algorithms", "year": 1980, "volume": 1, "number": 4,
            "pages": "359--373", "doi": "10.1016/0196-6774(80)90016-4",
            "url": "https://doi.org/10.1016/0196-6774(80)90016-4",
        },
        "ukkonen1985algorithms": {
            "author": "Ukkonen, E.",
            "title": "Algorithms for approximate string matching",
            "journal": "Information and Control", "year": 1985, "volume": 64, "number": "1--3",
            "pages": "100--118", "doi": "10.1016/S0019-9958(85)80046-2",
            "url": "https://doi.org/10.1016/S0019-9958(85)80046-2",
        },
        "ukkonen1985patterns": {
            "author": "Ukkonen, E.",
            "title": "Finding approximate patterns in strings",
            "journal": "Journal of Algorithms", "year": 1985, "volume": 6, "number": 1,
            "pages": "132--137", "doi": "10.1016/0196-6774(85)90023-9",
            "url": "https://doi.org/10.1016/0196-6774(85)90023-9",
        },
        "myers1999": {
            "author": "Myers, G.",
            "title": "A fast bit-vector algorithm for approximate string matching based on "
                     "dynamic programming",
            "journal": "Journal of the ACM", "year": 1999, "volume": 46, "number": 3,
            "pages": "395--415", "doi": "10.1145/316542.316550",
            "url": "https://doi.org/10.1145/316542.316550",
        },
        "hyyro2003": {
            "author": "Hyyrö, H.",
            "title": "A bit-vector algorithm for computing Levenshtein and Damerau edit distances",
            "journal": "Nordic Journal of Computing", "year": 2003, "volume": 10, "number": 1,
            "pages": "29--39",
        },
        # Composition (frequency-distance) lower bound of the bounded strategy (spec §5).
        "kahveci2001": {
            "author": "Kahveci, T. and Singh, A. K.",
            "title": "An efficient index structure for string databases",
            "booktitle": "Proceedings of the 27th International Conference on Very Large Data "
                         "Bases (VLDB 2001)",
            "address": "Roma, Italy", "year": 2001, "pages": "351--360",
        },
        # Bit-parallel alignment traceback behind the cigar field (spec §6).
        "hyyro2004": {
            "author": "Hyyrö, H.",
            "title": "A note on bit-parallel alignment computation",
            "booktitle": "Proceedings of the Prague Stringology Conference 2004",
            "year": 2004, "pages": "79--87",
        },
    },
    "research_use_only": True,
}
LEVENSHTEIN["signature"] = _signature(LEVENSHTEIN)

_COMPAT_IDS = {"emboss-water": "emboss-6.6.0.0-water", "emboss-needle": "emboss-6.6.0.0-needle"}
_ALIGN_FIELDS = ("score", "score_fx", "a_start", "a_end", "b_start", "b_end", "aln_len",
                 "n_identity", "n_similarity", "n_gaps", "cigar", "aligned_a", "aligned_b")
_ALIGNED_DTYPE = ("binary (Arrow large_binary) under check_bases=False or with a 256x256 matrix, "
                  "string (Arrow large_utf8) otherwise (QA12, QA14)")

# docs/wiki/align/gotoh-align.md: §1 signature and Scoring, §3 fields and masks,
# §4 rule E (QA1), §5 compat presets, Sources line for the citations.
ALIGN = {
    "name": "align",
    "parameters": {"mode": {"default": None, "choices": ("local", "global")},
                   "check_bases": {"default": True},
                   "scoring": {"default": None},
                   "compat": {"default": None, "choices": ("emboss-water", "emboss-needle")},
                   "fields": {"default": "default"}, "keep": {"default": None}},
    "execution": dict(_EXECUTION),
    "fields": {
        "score": {"dtype": "float64", "unit": "score", "exactness": "exact"},
        "score_fx": {"dtype": "int32", "unit": "score x 1000", "exactness": "exact"},
        "a_start": {"dtype": "int64", "unit": "position", "exactness": "exact"},
        "a_end": {"dtype": "int64", "unit": "position", "exactness": "exact"},
        "b_start": {"dtype": "int64", "unit": "position", "exactness": "exact"},
        "b_end": {"dtype": "int64", "unit": "position", "exactness": "exact"},
        "aln_len": {"dtype": "int32", "unit": "count", "exactness": "exact"},
        "n_identity": {"dtype": "int32", "unit": "count", "exactness": "exact"},
        "n_similarity": {"dtype": "int32", "unit": "count", "exactness": "exact"},
        "n_gaps": {"dtype": "int32", "unit": "count", "exactness": "exact"},
        "cigar": {"dtype": "string", "unit": None, "exactness": "exact"},
        # The type follows the parameters, never the data (QA12, QA14).
        "aligned_a": {"dtype": "string | binary", "unit": None, "exactness": "exact",
                      "note": _ALIGNED_DTYPE},
        "aligned_b": {"dtype": "string | binary", "unit": None, "exactness": "exact",
                      "note": _ALIGNED_DTYPE},
        # Optimal local end cells, every position tie counted (D89 QA5; spec §3.1).
        "n_ends": {"dtype": "int64", "unit": "count", "exactness": "exact",
                   "note": "native mode='local' only (QA5): the number of cells with "
                           "V(i, j) = score > 0; 0 when the score is 0"},
    },
    "field_groups": {"coords": ("a_start", "a_end", "b_start", "b_end"),
                     "stats": ("aln_len", "n_identity", "n_similarity", "n_gaps"),
                     "aligned": ("aligned_a", "aligned_b")},
    # Fields per selector (spec §3.1): compat when a preset is given, else mode.
    "mode_fields": {"local": _ALIGN_FIELDS + ("n_ends",), "global": _ALIGN_FIELDS,
                    "emboss-water": _ALIGN_FIELDS, "emboss-needle": _ALIGN_FIELDS},
    "default_fields": {"local": ("score",), "global": ("score",),
                       "emboss-water": ("score",), "emboss-needle": ("score",)},
    "primary": "score", "direction": ">=",
    "shapes": _SHAPES,
    "sinks": _SINKS,
    # Device path: docs/wiki/gpu/align-cuda.md (keys on the device, details on the host).
    "backends": ("cpu", "cuda"),
    "presets": {"compat": dict(_COMPAT_IDS), "scoring": dict(_COMPAT_IDS)},
    "scale": 1000,
    "tie_rule": "E: smallest row-major local end cell; backward I < D < M, stopping first "
                "(QA1, approved 2026-10-16)",
    "citations": {
        "gotoh1982": {
            "author": "Gotoh, O.",
            "title": "An improved algorithm for matching biological sequences",
            "journal": "Journal of Molecular Biology", "year": 1982, "volume": 162,
            "number": 3, "pages": "705--708", "doi": "10.1016/0022-2836(82)90398-9",
            "url": "https://doi.org/10.1016/0022-2836(82)90398-9",
        },
        "smith1981": {
            "author": "Smith, T. F. and Waterman, M. S.",
            "title": "Identification of common molecular subsequences",
            "journal": "Journal of Molecular Biology", "year": 1981, "volume": 147,
            "number": 1, "pages": "195--197", "doi": "10.1016/0022-2836(81)90087-5",
            "url": "https://doi.org/10.1016/0022-2836(81)90087-5",
        },
        "needleman1970": {
            "author": "Needleman, S. B. and Wunsch, C. D.",
            "title": "A general method applicable to the search for similarities in the amino "
                     "acid sequence of two proteins",
            "journal": "Journal of Molecular Biology", "year": 1970, "volume": 48,
            "number": 3, "pages": "443--453", "doi": "10.1016/0022-2836(70)90057-4",
            "url": "https://doi.org/10.1016/0022-2836(70)90057-4",
        },
        # EDNAFULL scores (spec §5.3); a data file, cited as such.
        "lowe1992nuc44": {
            "author": "Lowe, T.",
            "title": "NUC.4.4 nucleotide substitution matrix",
            "howpublished": "NCBI BLAST matrices, https://ftp.ncbi.nlm.nih.gov/blast/matrices/NUC.4.4",
            "year": 1992,
            "note": "file sha256 c603e46f9a159fb451632cdf9f9c687bae7daadb30fb2645f5fd09cace707490",
        },
        "rice2000": {
            "author": "Rice, P., Longden, I. and Bleasby, A.",
            "title": "EMBOSS: The European Molecular Biology Open Software Suite",
            "journal": "Trends in Genetics", "year": 2000, "volume": 16, "number": 6,
            "pages": "276--277", "doi": "10.1016/S0168-9525(00)02024-2",
            "url": "https://doi.org/10.1016/S0168-9525(00)02024-2",
        },
    },
    "research_use_only": True,
}
ALIGN["signature"] = _signature(ALIGN)

# docs/wiki/thermo/tm.md: §1 definition, §6 fields, masks, sinks and cutoff, §11
# answers (D90, 2026-10-16), Sources line for the citations.
# Unary: input x is a Seq (Scalar result under each, a 0- or 1-row Hits under hits)
# or Seqs. params: santalucia2004 and nndb2024-dna (the latter opened at stage S9,
# nndb2024-dna.md §5.5; a later closed preset would sit in choices only, and its gate error,
# naming the units not yet accepted and the pending items, comes from the T1 and K1 loaders,
# §4.7). A Pairs input is the register pairing
# of each requested pair (§13, K1 QK25 (a), decided 2026-10-17): the five pair kinds,
# every pairwise sink, the preset's K1 pack, CPU or CUDA (docs/wiki/gpu/tm-pairs-cuda.md);
# the compat presets raise for it.
# compat="biopython": docs/wiki/thermo/tm-biopython.md §1, §6 (fields tm_c only; the
# reference id records the correctly rounded log, P9), user answers QB1-QB3 (2026-10-16).
# compat="primer3": docs/wiki/thermo/tm-primer3.md §1, §6 (fields tm_c only; the reference
# id records the correctly rounded log and log10), provisional choices QPT1-QPT6.
# Backends: native tm on CPU and CUDA (docs/wiki/gpu/tm-cuda.md) and both compat presets
# on CPU and CUDA (docs/wiki/gpu/tm-compat-cuda.md), byte-identical.
_F64 = "binary64 in a fixed operation order (tm.md §5), bit-exact across backends"
_DCAL = "exact integer sum of per-entry rounded energies (tm.md §3.4, §9)"
TM = {
    "name": "tm",
    "input": "x",
    "parameters": {"cond": {"default": None},
                   "compat": {"default": None, "choices": ("primer3", "biopython"),
                              "available": ("biopython", "primer3"),
                              "fields": {"biopython": ("tm_c",), "primer3": ("tm_c",)},
                              "id": {"biopython": "biopython@1.88/cr-log",
                                     "primer3": "primer3-py@2.3.1/cr-log"}},
                   "params": {"default": "santalucia2004",
                              "choices": ("santalucia2004", "nndb2024-dna"),
                              "available": ("santalucia2004", "nndb2024-dna")},
                   "fields": {"default": "default"}, "keep": {"default": None}},
    "execution": dict(_EXECUTION),
    "fields": {
        "tm_c": {"dtype": "float64", "unit": "°C", "exactness": _F64},
        "dg_kcal": {"dtype": "float64", "unit": "kcal/mol", "exactness": _F64},
        "dg_dcal": {"dtype": "int32", "unit": "dcal/mol", "exactness": _DCAL},
        "dh_kcal": {"dtype": "float64", "unit": "kcal/mol", "exactness": _F64},
        "ds_cal_per_k": {"dtype": "float64", "unit": "cal/(mol*K)", "exactness": _F64},
    },
    "default_fields": ("tm_c", "dg_kcal", "dg_dcal", "dh_kcal", "ds_cal_per_k"),
    "primary": "tm_c", "direction": ">=",
    "inputs": ("Seq", "Seqs", "Pairs"),
    # Pairs only (tm.md §13.5, PR5); symmetric for within (upper triangle, §13.4).
    "shapes": _SHAPES,
    "symmetric": True,
    # Sinks of a Seq or Seqs input (tm.md §6; D47: no row LSE for tm).
    "sinks": ("each", "hits"),
    # Sinks of a Pairs input: every pairwise output mode (AGENTS §2 principle 4), as for
    # duplex. RowStats publishes best_tm_c; within and cross still need an explicit keep.
    "pair_sinks": _SINKS,
    # The unary path and a Pairs input both run on the CPU or CUDA (tm-pairs-cuda.md §8).
    "backends": ("cpu", "cuda"),
    "presets": {"compat": ("biopython", "primer3"), "conditions": ("primer3", "biopython"),
                "params": ("santalucia2004", "nndb2024-dna")},
    "citations": {
        # compat="biopython" only: the tool paper and Biopython's attribution of DNA_NN3.
        "biopython2009": {
            "author": "Cock, P. J. A. and Antao, T. and Chang, J. T. and Chapman, B. A. and "
                      "Cox, C. J. and Dalke, A. and Friedberg, I. and Hamelryck, T. and "
                      "Kauff, F. and Wilczynski, B. and de Hoon, M. J. L.",
            "title": "Biopython: freely available Python tools for computational molecular "
                     "biology and bioinformatics",
            "journal": "Bioinformatics", "year": 2009, "volume": 25, "number": 11,
            "pages": "1422--1423", "doi": "10.1093/bioinformatics/btp163",
            "url": "https://doi.org/10.1093/bioinformatics/btp163",
        },
        "allawi1997": {
            "author": "Allawi, H. T. and SantaLucia, Jr., J.",
            "title": "Thermodynamics and NMR of internal G·T mismatches in DNA",
            "journal": "Biochemistry", "year": 1997, "volume": 36, "number": 34,
            "pages": "10581--10594", "doi": "10.1021/bi962590c",
            "url": "https://doi.org/10.1021/bi962590c",
        },
        "santalucia1998": {
            "author": "SantaLucia, Jr., J.",
            "title": "A unified view of polymer, dumbbell, and oligonucleotide DNA "
                     "nearest-neighbor thermodynamics",
            "journal": "Proceedings of the National Academy of Sciences of the USA",
            "year": 1998, "volume": 95, "number": 4, "pages": "1460--1465",
            "doi": "10.1073/pnas.95.4.1460", "url": "https://doi.org/10.1073/pnas.95.4.1460",
        },
        "santalucia2004": {
            "author": "SantaLucia, Jr., J. and Hicks, D.",
            "title": "The thermodynamics of DNA structural motifs",
            "journal": "Annual Review of Biophysics and Biomolecular Structure",
            "year": 2004, "volume": 33, "pages": "415--440",
            "doi": "10.1146/annurev.biophys.32.110601.141800",
            "url": "https://doi.org/10.1146/annurev.biophys.32.110601.141800",
        },
        "vonahsen2001": {
            "author": "von Ahsen, N. and Wittwer, C. T. and Schütz, E.",
            "title": "Oligonucleotide melting temperatures under PCR conditions: nearest-neighbor "
                     "corrections for Mg2+, deoxynucleotide triphosphate, and dimethyl sulfoxide "
                     "concentrations with comparison to alternative empirical formulas",
            "journal": "Clinical Chemistry", "year": 2001, "volume": 47, "number": 11,
            "pages": "1956--1961", "doi": "10.1093/clinchem/47.11.1956",
            "url": "https://doi.org/10.1093/clinchem/47.11.1956",
        },
        # compat="primer3" only (tm-primer3.md §6): the primer3 tool papers and the
        # published sources of the long-sequence, formamide and DMSO terms. Bibliographic
        # data from Crossref (2026-10-16; Blake & Delcourt as the reader spec and the page
        # cite it). primer3-py itself has no paper (libnano; B. Pruitt, N. Conway).
        "untergasser2012": {
            "author": "Untergasser, A. and Cutcutache, I. and Koressaar, T. and Ye, J. and "
                      "Faircloth, B. C. and Remm, M. and Rozen, S. G.",
            "title": "Primer3 — new capabilities and interfaces",
            "journal": "Nucleic Acids Research", "year": 2012, "volume": 40, "number": 15,
            "pages": "e115", "doi": "10.1093/nar/gks596",
            "url": "https://doi.org/10.1093/nar/gks596",
        },
        "koressaar2007": {
            "author": "Koressaar, T. and Remm, M.",
            "title": "Enhancements and modifications of primer design program Primer3",
            "journal": "Bioinformatics", "year": 2007, "volume": 23, "number": 10,
            "pages": "1289--1291", "doi": "10.1093/bioinformatics/btm091",
            "url": "https://doi.org/10.1093/bioinformatics/btm091",
        },
        "bolton1962": {
            "author": "Bolton, E. T. and McCarthy, B. J.",
            "title": "A general method for the isolation of RNA complementary to DNA",
            "journal": "Proceedings of the National Academy of Sciences of the USA",
            "year": 1962, "volume": 48, "number": 8, "pages": "1390--1397",
            "doi": "10.1073/pnas.48.8.1390", "url": "https://doi.org/10.1073/pnas.48.8.1390",
        },
        "sambrook1989": {
            "author": "Sambrook, J. and Fritsch, E. F. and Maniatis, T.",
            "title": "Molecular Cloning: A Laboratory Manual, 2nd edition",
            "year": 1989,
            "howpublished": "Cold Spring Harbor Laboratory Press, Cold Spring Harbor, NY",
            "note": "p. 11.46, the long-sequence Tm formula, which the Primer3 manual attributes to Bolton and McCarthy (1962)",
        },
        "blake1996": {
            "author": "Blake, R. D. and Delcourt, S. G.",
            "title": "Thermodynamic effects of formamide on DNA stability",
            "journal": "Nucleic Acids Research", "year": 1996, "volume": 24, "number": 11,
            "pages": "2095--2103", "doi": "10.1093/nar/24.11.2095",
            "url": "https://doi.org/10.1093/nar/24.11.2095",
        },
        "musielski1981": {
            "author": "Musielski, H. and Mann, W. and Laue, R. and Michel, S.",
            "title": "Influence of dimethylsulfoxide on transcription by bacteriophage "
                     "T3-induced RNA polymerase",
            "journal": "Zeitschrift für allgemeine Mikrobiologie", "year": 1981, "volume": 21,
            "number": 6, "pages": "447--456", "doi": "10.1002/jobm.19810210606",
            "url": "https://doi.org/10.1002/jobm.19810210606",
        },
        # Correctly rounded host logarithm of s, kappa4, kappa1 (spec §3.3).
        "ziv1991": {
            "author": "Ziv, A.",
            "title": "Fast evaluation of elementary mathematical functions with correctly "
                     "rounded last bit",
            "journal": "ACM Transactions on Mathematical Software", "year": 1991,
            "volume": 17, "number": 3, "pages": "410--423", "doi": "10.1145/114697.116813",
            "url": "https://doi.org/10.1145/114697.116813",
        },
    },
    "research_use_only": True,
}
TM["signature"] = _signature(TM)

# docs/wiki/thermo/duplex-mfe.md: §1 signature and options, §7 fields and masks, §8 shapes,
# sinks, cutoff and strategy, §12.4 registry entry; QK1-QK24 decided 2026-10-17 (§13).
# params="santalucia2004" and params="nndb2024-dna" are available for model="mfe" and
# model="pf" (every build unit of §3.5 accepted; nndb2024-dna opened at stage S9,
# nndb2024-dna.md §5.5; a later closed preset would sit in choices only, and the K1 loader's
# gate error names its missing units and pending items, §4.7); testing builds also take a
# pack handle from seqblaze._testing.duplex_pack (§12.3).
_DUPLEX_F64 = "binary64 finalize F1-F10 of tm.md §5 on the reported key (duplex-mfe.md §7)"
_DUPLEX_DCAL = ("exact integer MFE: lexicographic (D, H, G, Q) minimum of per-entry rounded "
                "energies (duplex-mfe.md §4-§6)")
DUPLEX = {
    "name": "duplex",
    "parameters": {"cond": {"default": None},
                   "model": {"default": "mfe", "choices": ("mfe", "pf"), "available": ("mfe", "pf")},
                   "compat": {"default": None, "choices": ("primer3",), "available": ("primer3",),
                              "id": {"primer3": "primer3-py@2.3.1/cr-log"}},
                   "end": {"default": "any", "choices": ("any", "end1", "end2")},
                   "orientation": {"default": "both", "choices": ("both", "as_given")},
                   "params": {"default": "santalucia2004",
                              "choices": ("santalucia2004", "nndb2024-dna"),
                              "available": ("santalucia2004", "nndb2024-dna")},
                   "fields": {"default": "default"}, "keep": {"default": None}},
    "execution": dict(_EXECUTION),
    "fields": {
        # compat="primer3" (K2, docs/wiki/thermo/dimer-primer3.md §7).
        "found": {"dtype": "bool", "unit": None,
                  "exactness": "exact (primer3's structure_found; compat='primer3' only)"},
        "dg_kcal": {"dtype": "float64", "unit": "kcal/mol", "exactness": "RN(dg_dcal/100)"},
        "dg_dcal": {"dtype": "int32", "unit": "dcal/mol", "exactness": _DUPLEX_DCAL},
        "dh_kcal": {"dtype": "float64", "unit": "kcal/mol", "exactness": _DUPLEX_F64},
        "ds_cal_per_k": {"dtype": "float64", "unit": "cal/(mol*K)", "exactness": _DUPLEX_F64},
        "tm_c": {"dtype": "float64", "unit": "°C", "exactness": _DUPLEX_F64},
        # model="pf" (K3, docs/wiki/thermo/duplex-pf.md §9.1).
        "dg_ens_kcal": {"dtype": "float64", "unit": "kcal/mol",
                        "exactness": "F53 software float, fixed fold order, canonical "
                                     "orientation (duplex-pf.md §3–§7)"},
        "dg_ens_err_kcal": {"dtype": "float64", "unit": "kcal/mol",
                            "exactness": "upward-rounded a priori bound (duplex-pf.md §7.3, §8)"},
        "lse_dg_kcal": {"dtype": "float64", "unit": "kcal/mol",
                        "exactness": "exact row sum of the pairs' F53 values, one RN53, the "
                                     "duplex-pf.md §7 finalize (§9.4); RowStats only"},
    },
    # Reserved field names per model (D47): ParameterError, never computed. QK14's
    # "not yet available" for lse_dg_kcal is withdrawn for model="mfe" (F7, duplex-mfe.md §8.6).
    "reserved_fields": {"mfe": {"structure": "reserved, not in v1 (D47)"},
                        "pf": {"structure": "reserved, not in v1 (D47)"}},
    # Fields per model (the `mode_fields` mechanism, keyed by `model`); lse_dg_kcal is a
    # RowStats column of both models (duplex-pf.md §9.1, QP9; duplex-mfe.md §8.6.4, LK4).
    # Its exactness string is K3's, unchanged, so K3 manifests keep their B1 scientific
    # identity (duplex-mfe.md §8.6.4, LK7): for model="mfe" the F53 values are W(dg_dcal).
    "mode_fields": {"mfe": ("dg_kcal", "dg_dcal", "dh_kcal", "ds_cal_per_k", "tm_c", "lse_dg_kcal"),
                    "pf": ("dg_ens_kcal", "dg_ens_err_kcal", "lse_dg_kcal")},
    "default_fields": {"mfe": ("dg_kcal", "dg_dcal"), "pf": ("dg_ens_kcal", "dg_ens_err_kcal")},
    # Columns derived from the engine key, not kernel detail fields (§12.4).
    "key_fields": ("dg_kcal", "dg_dcal"),
    "primary": "dg_kcal", "direction": "<=",
    "model_primary": {"mfe": "dg_kcal", "pf": "dg_ens_kcal"},
    # compat="primer3" (K2): docs/wiki/thermo/dimer-primer3.md §1 (definition, call
    # mapping QD2), §7 (fields, QD7-QD9, QD12) and §12 (provisional choices QD1-QD12).
    # Asymmetric: `orientation` reaches the engine under `within` (§2). CPU and CUDA
    # (docs/wiki/gpu/primer3-cuda.md).
    # Derived from a behaviour specification written from a reading of primer3-py 2.3.1's
    # source (no code copied; D94), confirmed against the unmodified tool.
    "compat": {"primer3": {
        "id": "primer3-py@2.3.1/cr-log",
        "fields": ("found", "dg_kcal", "dh_kcal", "ds_cal_per_k", "tm_c"),
        "default_fields": ("found", "dg_kcal", "dh_kcal", "ds_cal_per_k", "tm_c"),
        "exactness": "binary64 replay of primer3-py 2.3.1 calc_heterodimer, calc_homodimer "
                     "and calc_end_stability (dimer-primer3.md §5-§6), bit-exact to the "
                     "named reference primer3-py@2.3.1/cr-log",
        "primary": "dg_kcal", "direction": "<=",
        # Reserved field names (D47, QD8): ParameterError, never computed.
        "reserved_fields": {"structure": "reserved, not in v1 (D47)",
                            "near_tie": "not provided (QD8)"},
        "model": "mfe",
        "end": {"any": "calc_heterodimer(a, b)", "end1": "calc_end_stability(a, b)",
                "end2": "calc_end_stability(b, a)"},
        "symmetric": False,
        "backends": ("cpu", "cuda"),
        "max_length": {"short": 60, "each": 10_000},
        "loop_max": 30,
        "params": "primer3-py@2.3.1:files (read at run time, extra seqblaze[primer3])",
    }},
    "shapes": _SHAPES,
    "sinks": _SINKS,
    "symmetric": True,
    # Native model="mfe", model="pf" (duplex-pf-cuda.md) and compat="primer3" (its own
    # "backends", docs/wiki/gpu/primer3-cuda.md) all run on both.
    "backends": ("cpu", "cuda"),
    "presets": {"params": ("santalucia2004", "nndb2024-dna"), "compat": ("primer3",)},
    # model="mfe" only; model="pf" has no ties to break (duplex-pf.md §14.3).
    "tie_rule": "lexmin (D, H, G, Q)",
    # Arithmetic of model="pf" in the manifest's model.numerics (duplex-pf.md §14.3, QP4).
    "numerics": {"pf": "F53/v1"},
    "loop_max": {"santalucia2004": 30, "nndb2024-dna": 30},
    "citations": {
        "santalucia2004": TM["citations"]["santalucia2004"],
        "santalucia1998": TM["citations"]["santalucia1998"],
        # compat="primer3" only: the tool papers of primer3 (primer3-py has no paper:
        # libnano; B. Pruitt, N. Conway), dimer-primer3.md §7.
        "untergasser2012": TM["citations"]["untergasser2012"],
        "koressaar2007": TM["citations"]["koressaar2007"],
        # model="pf" only (duplex-pf.md §1, §8).
        "mccaskill1990": {
            "author": "McCaskill, J. S.",
            "title": "The equilibrium partition function and base pair binding probabilities "
                     "for RNA secondary structure",
            "journal": "Biopolymers", "year": 1990, "volume": 29, "number": "6-7",
            "pages": "1105--1119", "doi": "10.1002/bip.360290621",
            "url": "https://doi.org/10.1002/bip.360290621",
        },
        "higham2002": {
            "author": "Higham, N. J.",
            "title": "Accuracy and Stability of Numerical Algorithms", "edition": "2nd",
            "publisher": "SIAM", "year": 2002, "doi": "10.1137/1.9780898718027",
            "url": "https://doi.org/10.1137/1.9780898718027",
        },
        "zuker1981": {
            "author": "Zuker, M. and Stiegler, P.",
            "title": "Optimal computer folding of large RNA sequences using thermodynamics and "
                     "auxiliary information",
            "journal": "Nucleic Acids Research", "year": 1981, "volume": 9, "number": 1,
            "pages": "133--148", "doi": "10.1093/nar/9.1.133",
            "url": "https://doi.org/10.1093/nar/9.1.133",
        },
        "dimitrov2004": {
            "author": "Dimitrov, R. A. and Zuker, M.",
            "title": "Prediction of hybridization and melting for double-stranded nucleic acids",
            "journal": "Biophysical Journal", "year": 2004, "volume": 87, "number": 1,
            "pages": "215--226", "doi": "10.1529/biophysj.103.020743",
            "url": "https://doi.org/10.1529/biophysj.103.020743",
        },
        "rehmsmeier2004": {
            "author": "Rehmsmeier, M. and Steffen, P. and Höchsmann, M. and Giegerich, R.",
            "title": "Fast and effective prediction of microRNA/target duplexes",
            "journal": "RNA", "year": 2004, "volume": 10, "number": 10,
            "pages": "1507--1517", "doi": "10.1261/rna.5248604",
            "url": "https://doi.org/10.1261/rna.5248604",
        },
        "allawi1997": TM["citations"]["allawi1997"],
        "allawi1998ga": {
            "author": "Allawi, H. T. and SantaLucia, Jr., J.",
            "title": "Nearest neighbor thermodynamic parameters for internal G·A mismatches in DNA",
            "journal": "Biochemistry", "year": 1998, "volume": 37, "number": 8,
            "pages": "2170--2179", "doi": "10.1021/bi9724873",
            "url": "https://doi.org/10.1021/bi9724873",
        },
        "allawi1998ct": {
            "author": "Allawi, H. T. and SantaLucia, Jr., J.",
            "title": "Thermodynamics of internal C·T mismatches in DNA",
            "journal": "Nucleic Acids Research", "year": 1998, "volume": 26, "number": 11,
            "pages": "2694--2701", "doi": "10.1093/nar/26.11.2694",
            "url": "https://doi.org/10.1093/nar/26.11.2694",
        },
        "allawi1998ac": {
            "author": "Allawi, H. T. and SantaLucia, Jr., J.",
            "title": "Nearest-neighbor thermodynamics of internal A·C mismatches in DNA: "
                     "sequence dependence and pH effects",
            "journal": "Biochemistry", "year": 1998, "volume": 37, "number": 26,
            "pages": "9435--9444", "doi": "10.1021/bi9803729",
            "url": "https://doi.org/10.1021/bi9803729",
        },
        "peyret1999": {
            "author": "Peyret, N. and Seneviratne, P. A. and Allawi, H. T. and SantaLucia, Jr., J.",
            "title": "Nearest-neighbor thermodynamics and NMR of DNA sequences with internal "
                     "A·A, C·C, G·G, and T·T mismatches",
            "journal": "Biochemistry", "year": 1999, "volume": 38, "number": 12,
            "pages": "3468--3477", "doi": "10.1021/bi9825091",
            "url": "https://doi.org/10.1021/bi9825091",
        },
        "bommarito2000": {
            "author": "Bommarito, S. and Peyret, N. and SantaLucia, Jr., J.",
            "title": "Thermodynamic parameters for DNA sequences with dangling ends",
            "journal": "Nucleic Acids Research", "year": 2000, "volume": 28, "number": 9,
            "pages": "1929--1934", "doi": "10.1093/nar/28.9.1929",
            "url": "https://doi.org/10.1093/nar/28.9.1929",
        },
        "wo0194611": {
            "author": "SantaLucia, Jr., J. and Peyret, N.",
            "title": "Method and system for predicting nucleic acid hybridization thermodynamics "
                     "and computer-readable storage medium for use therein",
            "howpublished": "International Publication No. WO 01/94611 A2",
            "year": 2001,
        },
        "mittal2024": {
            "author": "Mittal, A. and Turner, D. H. and Mathews, D. H.",
            "title": "NNDB: An expanded database of nearest neighbor parameters for predicting "
                     "stability of nucleic acid secondary structures",
            "journal": "Journal of Molecular Biology", "year": 2024, "volume": 436,
            "number": 17, "pages": "168549", "doi": "10.1016/j.jmb.2024.168549",
            "url": "https://doi.org/10.1016/j.jmb.2024.168549",
        },
        # The NNDB asks to be cited by both of its works (NOTICE); its own reference list
        # prints 2009, the issue year 2010 is used.
        "turner2010": {
            "author": "Turner, D. H. and Mathews, D. H.",
            "title": "NNDB: the nearest neighbor parameter database for predicting stability "
                     "of nucleic acid secondary structure",
            "journal": "Nucleic Acids Research", "year": 2010, "volume": 38,
            "number": "Database issue", "pages": "D280--D282", "doi": "10.1093/nar/gkp892",
            "url": "https://doi.org/10.1093/nar/gkp892",
        },
        "reuter2010": {
            "author": "Reuter, J. S. and Mathews, D. H.",
            "title": "RNAstructure: software for RNA secondary structure prediction and analysis",
            "journal": "BMC Bioinformatics", "year": 2010, "volume": 11, "pages": "129",
            "doi": "10.1186/1471-2105-11-129", "url": "https://doi.org/10.1186/1471-2105-11-129",
        },
        "vonahsen2001": TM["citations"]["vonahsen2001"],
        "ziv1991": TM["citations"]["ziv1991"],
    },
    "research_use_only": True,
}
DUPLEX["signature"] = _signature(DUPLEX)

# tm on Pairs (tm.md §13.3, §13.5 PR9) reads the mismatch, terminal-mismatch and 2x2-loop
# entries of K1's pack: their records, shared with duplex.
_TM_PAIRS_ONLY = ("allawi1998ga", "allawi1998ct", "allawi1998ac", "peyret1999", "wo0194611",
                  "mittal2024", "turner2010", "reuter2010")
TM["citations"].update({key: DUPLEX["citations"][key] for key in _TM_PAIRS_ONLY})
# PR9: the records of the parameters a register structure reads, and the host logarithm.
_TM_PAIRS_CITATIONS = frozenset(("santalucia2004", "santalucia1998", "allawi1997", *_TM_PAIRS_ONLY,
                                 "vonahsen2001", "ziv1991"))
# params="nndb2024-dna" (nndb2024-dna.md §5.5.5): duplex and tm on Pairs cite santalucia2004's
# records minus wo0194611 (the source of santalucia2004's terminal mismatches, not read by this
# preset; the six SantaLucia-lab records stay as primary papers NNDB's ch. 31 lists); unary tm
# adds the three NNDB records, whose stacks, initiation and A.T end entries it reads.
_NNDB_RECORDS = ("mittal2024", "turner2010", "reuter2010")
_PARAMS_PRESETS = ("santalucia2004", "nndb2024-dna")
_NOT_READ = {"santalucia2004": frozenset(), "nndb2024-dna": frozenset({"wo0194611"})}

# docs/wiki/thermo/hairpin-primer3.md: §1 definition, §6 fields, masks, sinks, cutoff and
# manifest, §7 order, §11 choices QH1-QH8 and QH-C (the class-C constants of the shared
# page §3.3), settled on 2026-10-17. The reported values follow §5.7, tm = +0 at an
# exact-zero entropy included (oracle X8, §10.1). Unary: input x is a Seq (Scalar
# under each, a 0- or 1-row Hits under hits) or Seqs. compat is required in v1 (the
# signed D48 signature); native single-strand folding is planned for v1.1. CPU and CUDA
# (docs/wiki/gpu/primer3-cuda.md).
_HP_F64 = ("binary64 replay of primer3-py 2.3.1 calc_hairpin (hairpin-primer3.md §5), "
           "bit-exact to the named reference primer3-py@2.3.1/cr-log")
HAIRPIN = {
    "name": "hairpin",
    "input": "x",
    "parameters": {"compat": {"required": True, "choices": ("primer3",),
                              "available": ("primer3",),
                              "id": {"primer3": "primer3-py@2.3.1/cr-log"}},
                   "cond": {"default": None},
                   "fields": {"default": "default"}, "keep": {"default": None}},
    "execution": dict(_EXECUTION),
    "fields": {
        "found": {"dtype": "bool", "unit": None, "exactness": "exact (primer3's structure_found)"},
        "dg_kcal": {"dtype": "float64", "unit": "kcal/mol", "exactness": _HP_F64},
        "dh_kcal": {"dtype": "float64", "unit": "kcal/mol", "exactness": _HP_F64},
        "ds_cal_per_k": {"dtype": "float64", "unit": "cal/(mol*K)", "exactness": _HP_F64},
        "tm_c": {"dtype": "float64", "unit": "°C", "exactness": _HP_F64},
    },
    # Reserved field names (D47): ParameterError, never computed.
    "reserved_fields": {"structure": "reserved, not in v1 (D47)"},
    "default_fields": ("found", "dg_kcal", "dh_kcal", "ds_cal_per_k", "tm_c"),
    "primary": "dg_kcal", "direction": "<=",
    "inputs": ("Seq", "Seqs"),
    "shapes": (),
    "sinks": ("each", "hits"),
    "backends": ("cpu", "cuda"),
    "presets": {"compat": ("primer3",), "conditions": ("primer3",)},
    "max_length": {"primer3": 60},
    "loop_max": {"primer3": 30},
    "citations": {
        # Tool papers of primer3 (primer3-py has no paper: libnano; B. Pruitt, N. Conway).
        "untergasser2012": TM["citations"]["untergasser2012"],
        "koressaar2007": TM["citations"]["koressaar2007"],
        # Model, salt and host arithmetic.
        "santalucia1998": TM["citations"]["santalucia1998"],
        "santalucia2004": TM["citations"]["santalucia2004"],
        "vonahsen2001": TM["citations"]["vonahsen2001"],
        "ziv1991": TM["citations"]["ziv1991"],
        # The parameter-set papers the files follow according to reader part C (§6).
        "allawi1997": TM["citations"]["allawi1997"],
        "allawi1998ga": DUPLEX["citations"]["allawi1998ga"],
        "allawi1998ct": DUPLEX["citations"]["allawi1998ct"],
        "allawi1998ac": DUPLEX["citations"]["allawi1998ac"],
        "peyret1999": DUPLEX["citations"]["peyret1999"],
        "bommarito2000": DUPLEX["citations"]["bommarito2000"],
    },
    "research_use_only": True,
}
HAIRPIN["signature"] = _signature(HAIRPIN)

_REGISTRY = {spec["name"]: spec for spec in (HAMMING, LEVENSHTEIN, ALIGN, TM, HAIRPIN, DUPLEX)}


def citation_keys(spec, model, fields=(), strategy=None, pairs=False):
    """Method records a call actually uses: Sellers (1980) only for infix,
    Kahveci and Singh (2001) only for an effective bounded strategy, and
    Hyyrö (2004) only when cigar is requested. ``pairs`` marks a call on the
    pairwise path (tm on Pairs cites the records of tm.md §13.5 PR9)."""
    keys = tuple(spec["citations"])
    if spec["name"] == "align":
        unused = {"needleman1970"} if model.get("mode") == "local" else {"smith1981"}
        if (model.get("scoring") or {}).get("matrix") != "EDNAFULL":
            unused.add("lowe1992nuc44")
        if model.get("compat") is None:
            unused.add("rice2000")
        return tuple(key for key in keys if key not in unused)
    if spec["name"] == "levenshtein":
        unused = set()
        if model.get("mode") != "infix":
            unused.add("sellers1980")
        if strategy != "bounded":
            unused.add("kahveci2001")
        if "cigar" not in fields:
            unused.add("hyyro2004")
        keys = tuple(key for key in keys if key not in unused)
    elif spec["name"] == "duplex" and (model.get("compat") or {}).get("alias") == "primer3":
        # compat="primer3" (dimer-primer3.md §7): the tool papers, the model, salt and host
        # arithmetic, and the parameter-set papers the files follow (not for a synthetic
        # testing handle).
        used = {"untergasser2012", "koressaar2007", "santalucia1998", "santalucia2004",
                "vonahsen2001", "ziv1991"}
        if (model.get("params") or {}).get("name") != "synthetic":
            used |= {"allawi1997", "allawi1998ga", "allawi1998ct", "allawi1998ac", "peyret1999",
                     "bommarito2000"}
        keys = tuple(key for key in keys if key in used)
    elif spec["name"] == "duplex":
        keys = tuple(key for key in keys if key not in ("untergasser2012", "koressaar2007"))
        # A synthetic (testing) pack cites the model form, the algorithm (it runs the same
        # recurrence, §12.4) and the host arithmetic only; the parameter sources belong to
        # a preset (santalucia2004, duplex-mfe.md §3.5; nndb2024-dna, nndb2024-dna.md §5.5.5,
        # also for a handle of that name).
        name = (model.get("params") or {}).get("name")
        if name not in _PARAMS_PRESETS:
            method = {"santalucia2004", "santalucia1998", "zuker1981", "dimitrov2004",
                      "rehmsmeier2004", "vonahsen2001", "ziv1991", "mccaskill1990",
                      "higham2002"}
            keys = tuple(key for key in keys if key in method)
        else:
            keys = tuple(key for key in keys if key not in _NOT_READ[name])
        if model.get("model") != "pf":
            keys = tuple(key for key in keys if key not in ("mccaskill1990", "higham2002"))
    elif spec["name"] == "hairpin":
        # A synthetic (testing) parameter set cites the tools, the model and the host
        # arithmetic only; the parameter-set papers belong to primer3-py's files.
        if (model.get("params") or {}).get("name") == "synthetic":
            sets = {"allawi1997", "allawi1998ga", "allawi1998ct", "allawi1998ac", "peyret1999",
                    "bommarito2000"}
            keys = tuple(key for key in keys if key not in sets)
    elif spec["name"] == "tm" and pairs:
        # tm on Pairs (tm.md §13.5 PR9): the parameter records of the register structures
        # and the host logarithm; no dangle and no DP paper.
        unused = _NOT_READ.get((model.get("params") or {}).get("name"), frozenset())
        keys = tuple(key for key in keys if key in _TM_PAIRS_CITATIONS and key not in unused)
    elif spec["name"] == "tm":
        # The compat presets use sl1998, not santalucia2004; each cites its own records
        # (tm-biopython.md §6, tm-primer3.md §6); native uses no compat record, and no
        # unary call uses the records that only tm on Pairs reads.
        only = {"biopython": {"biopython2009", "allawi1997"},
                "primer3": {"untergasser2012", "koressaar2007", "bolton1962", "sambrook1989",
                            "blake1996", "musielski1981"}}
        alias = (model.get("compat") or {}).get("alias")
        unused = set().union(*(records for name, records in only.items() if name != alias))
        unused.update(_TM_PAIRS_ONLY)
        if alias is None and (model.get("params") or {}).get("name") == "nndb2024-dna":
            unused.difference_update(_NNDB_RECORDS)
        if alias is not None:
            unused.add("santalucia2004")
        keys = tuple(key for key in keys if key not in unused)
    return keys


def kernels():
    """Return the names of currently callable public kernels."""
    return list(_REGISTRY)


def describe(kernel):
    """Describe an implemented kernel. The returned metadata is an independent copy."""
    from ._errors import ParameterError
    from ._fp import close_matches
    if type(kernel) is not str or kernel not in _REGISTRY:
        similar = close_matches(kernel, _REGISTRY, n=1) if isinstance(kernel, str) else []
        hint = f"; did you mean {similar[0]!r}?" if similar else ""
        names = ", ".join(f"sb.describe({name!r})" for name in _REGISTRY)
        raise ParameterError(f"unknown kernel {kernel!r}{hint}; use one of {names}")
    return deepcopy(_REGISTRY[kernel])


_PRESETS = {
    # alias -> full id: align's compat presets and their Scoring presets (gotoh-align.md §5),
    # and the tm compat presets (tm.md §1, tm-biopython.md §6, tm-primer3.md §6).
    "compat": {
        **_COMPAT_IDS,
        "biopython": "biopython@1.88/cr-log: compatible with Biopython 1.88 "
                     "Bio.SeqUtils.MeltingTemp.Tm_NN(seq) (DNA_NN3, saltcorr=5) evaluated "
                     "with a correctly rounded log; values from ledger units sl1998 and "
                     "va2001; tm_c only (sb.tm(x, compat='biopython'))",
        "primer3": "primer3-py@2.3.1/cr-log: compatible with primer3-py 2.3.1 (libnano; "
                   "B. Pruitt, N. Conway). sb.tm(x, compat='primer3'): primer3.calc_tm with its "
                   "default method pair (santalucia table and salt correction, max_nn_length "
                   "60, dmso_fact 0.6) evaluated with a correctly rounded log and log10; values "
                   "from ledger units sl1998 and va2001; tm_c only. sb.hairpin(x, "
                   "compat='primer3'): primer3.calc_hairpin with max_loop 30 evaluated with a "
                   "correctly rounded log, over primer3-py 2.3.1's own parameter files read at "
                   "run time (extra seqblaze[primer3]); found, dg_kcal, dh_kcal, ds_cal_per_k, "
                   "tm_c. sb.duplex(pairs, compat='primer3'): primer3.calc_heterodimer "
                   "(end='any'), calc_end_stability (end='end1'; end='end2' on the swapped pair) "
                   "and, for a pair of identical sequences, calc_homodimer, with max_loop 30, "
                   "evaluated with a correctly rounded log over the same run-time parameter "
                   "files; found, dg_kcal, dh_kcal, ds_cal_per_k, tm_c. All three derived from "
                   "a behaviour specification written from a reading of primer3-py 2.3.1's "
                   "source (no code copied), confirmed against the unmodified tool by "
                   "black-box runs",
    },
    "scoring": dict(_COMPAT_IDS),
    "conditions": {
        "primer3": "primer3 documented defaults: 37 °C, monovalent 50 mM, Mg 1.5 mM, "
                   "dNTP 0.6 mM, DNA 50 nM (sb.Conditions.preset('primer3'))",
        "biopython": "Biopython Tm_NN documented defaults: Na 50 mM, 25 nM + 25 nM, "
                     "temp_c 37 (sb.Conditions.preset('biopython'))",
    },
    "params": {
        "santalucia2004": (
            "santalucia2004: SantaLucia & Hicks 2004 nearest-neighbour model with the von Ahsen "
            "2001 sodium equivalent. Unary tm (a Seq or Seqs input: each sequence with its "
            "perfect complement) uses ledger units sh2004 and va2001 only. duplex (K1 MFE and "
            "K3 partition function) uses the build units sh2004, va2001, sh2004-mm (mismatch "
            "dG), al1997-mm, al1998-ga-mm, al1998-ct-mm, al1998-ac-mm and pe1999-mm (mismatch "
            "dH), sh2004-de (dangles), sh2004-loop (loop increments), sp2001-tmm (terminal "
            "mismatches) and nndb2024-dna-int21 and nndb2024-dna-int22 (1x2, 2x1 and 2x2 "
            "loops). tm on a Pairs input (register pairing) reads the same compiled pack as "
            "duplex, and its manifest records the same build-definition sha256; of the build "
            "units it reads all but sh2004-de (no register structure has a dangle) and "
            "nndb2024-dna-int21 (no register structure has a 1x2 or 2x1 loop), so the notes on "
            "terminal mismatches, 2x2 loops and the erratum below apply to it as well, and "
            "they apply to no unary tm call. The terminal mismatches come from WO 01/94611 A2 "
            "(SantaLucia & Peyret), a patent application published 2001-12-13 and later "
            "abandoned. The 1x2, 2x1 and 2x2 "
            "loops come from NNDB 2024 DNA (Mittal, Turner & Mathews 2024; the NNDB itself: Turner "
            "& Mathews 2010); their dH lineage "
            "differs from the SantaLucia & Hicks loops, whose dH is zero. One build-time "
            "erratum: the dH of the terminal mismatch tmm:GG/CG is derived from its printed "
            "dG37 and dS (the printed dH is inconsistent with them; the ledger keeps the "
            "print). SantaLucia & Hicks 2004 (p. 418) state that their parameters are \"not "
            "appropriate for partition function computations\"; duplex(model='pf') uses them "
            "with that caveat and issues no warning at run time. Sources: "
            "docs/wiki/thermo/duplex-mfe.md §3.5, §3.6; docs/wiki/thermo/tm.md §13.3"
        ),
        # nndb2024-dna.md §5.5.4 (binding text; opened at stage S9). No parameter value.
        "nndb2024-dna": (
            'nndb2024-dna: the DNA parameter set of the Nearest Neighbor Database, section "DNA '
            '(RNAstructure)" (NNDB 2024: Mittal, Turner & Mathews 2024; the NNDB itself: Turner '
            "& Mathews 2010), which NNDB credits to RNAstructure (Reuter & Mathews 2010) and to "
            "the primary papers listed in its DNA references chapter (ch. 31, "
            "https://rna.urmc.rochester.edu/NNDB/dna/dna_references.html). The values were "
            "transcribed from an NNDB snapshot of 2026-10-16 into the ledger units "
            "nndb2024-dna-stack, nndb2024-dna-helix, nndb2024-dna-example, nndb2024-dna-de, "
            "nndb2024-dna-tmx, nndb2024-dna-loop, nndb2024-dna-int11, nndb2024-dna-int21, "
            "nndb2024-dna-int21-gt, nndb2024-dna-int22 and nndb2024-dna-int22-gt, plus sh2004 "
            "and va2001. They ship as published; seqblaze fits nothing. G.T pairs are base "
            "pairs in this preset. Unary tm (a Seq or Seqs input) reads the Watson-Crick "
            "stacks, the intermolecular initiation and the terminal A.T term of these units, "
            "the symmetry term of sh2004 and the salt model. duplex (K1 MFE and K3 partition "
            "function) and tm on a Pairs input read the compiled pack of all the units. "
            "Lineage: these are the RNAstructure-lineage DNA parameters, not the SantaLucia & "
            "Hicks 2004 set, and their loop dH is not zero. Example-only constants: NNDB prints "
            "the intermolecular initiation and the internal-loop asymmetry only inside worked "
            "examples; they are transcribed from the ch. 22 G.T helix example and the ch. 27 "
            "1x5 loop example, and every other NNDB example that uses them prints the same "
            "numbers. Cross-lineage term: the dG37 of the symmetry correction of a "
            "self-complementary duplex is SantaLucia & Hicks 2004's, because NNDB's only "
            "example of it is internally inconsistent; its dH is zero, as NNDB states. "
            "Substitute: NNDB publishes no first-mismatch table for 2x3 and other generic "
            "internal loops; its terminal-mismatch table is used for both, and 1x(n-1) loops "
            "take zero, as NNDB states. End rule: an end with both flanking nucleotides takes "
            "the lowest-dG37 of terminal mismatch, 5' dangle and 3' dangle (ties: terminal "
            "mismatch, then 3', then 5'), and its dH and salt weight follow that choice; the "
            "choice is made once, at 37 °C and 1 M, so at other conditions it need not be the "
            "lowest. RNAstructure DuplexFold and ViennaRNA duplexfold take the terminal "
            "mismatch at such ends. Erratum: NNDB prints a different dH for the two "
            "orientations of one G.T stack (AG/TT and TT/GA); the build keeps the print at "
            "TT/GA, the nearer to Allawi & SantaLucia 1997, and the ledger keeps both. "
            "Partition function: the single-nucleotide bulge term -RT ln(number of states) "
            "enters duplex(model='mfe') only (tm on Pairs evaluates register structures, which "
            "have no bulge); duplex(model='pf') omits it, because "
            "the ensemble already sums the equivalent bulge positions, so dg_ens_kcal <= "
            "dg_dcal/100 + dg_ens_err_kcal holds only up to a rounding slack per such bulge, "
            "and can fail where a bulged run of identical bases reaches a helix end. Fill-like "
            "cells: in NNDB's terminal-mismatch table the Watson-Crick-pairable flank cells of "
            "the Watson-Crick-closed blocks share one value, the Watson-Crick-pairable cells of "
            "the 1x1 table take two values, and the dangle rows on G.T end pairs are flat; they "
            "look like fill values, are used as printed, and are read because this preset has "
            "no rule F (structures with pairable flanks are admitted). Salt and temperature: "
            "NNDB's values are for 37 °C and 1 M NaCl, and NNDB prints no salt model; the "
            "SantaLucia & Hicks 2004 per-phosphate entropy correction with the von Ahsen 2001 "
            "sodium equivalent applies, as for santalucia2004. Sources: "
            "docs/wiki/thermo/nndb2024-dna.md §2, §3, §6, §10"
        ),
    },
}


def presets(kind):
    from ._errors import ParameterError
    if type(kind) is not str or kind not in _PRESETS:
        raise ParameterError(f"kind={kind!r}; use 'compat', 'conditions', 'scoring', or 'params'")
    return dict(_PRESETS[kind])


def _format_citations(kernel, keys, format):
    """Render the actual method records referenced by a Result's manifest."""
    from ._errors import ParameterError
    if type(format) is not str or format not in ("text", "bibtex"):
        raise ParameterError(f"format={format!r}; valid formats are 'text' and 'bibtex'; use r.cite(format='text')")
    records = _REGISTRY[kernel]["citations"]
    rendered = []
    for key in keys:
        record = records[key]
        proceedings = "booktitle" in record
        dataset = "howpublished" in record
        book = "publisher" in record and "journal" not in record
        if format == "bibtex":
            fields = ",\n".join(f"  {name} = {{{value}}}" for name, value in record.items())
            entry = ("misc" if dataset else "inproceedings" if proceedings else "book" if book
                     else "article")
            rendered.append(f"@{entry}{{{key},\n{fields}\n}}")
        elif book:
            edition = f", {record['edition']} ed." if "edition" in record else ""
            link = f" {record['url']}" if "url" in record else ""
            rendered.append(f"{record['author']} ({record['year']}). {record['title']}{edition} "
                            f"{record['publisher']}.{link}")
        elif dataset:
            note = f" ({record['note']})" if "note" in record else ""
            rendered.append(f"{record['author']} ({record['year']}). {record['title']}. "
                            f"{record['howpublished']}.{note}")
        elif proceedings:
            pages = record["pages"].replace("--", "–")
            rendered.append(f"{record['author']} ({record['year']}). {record['title']}. "
                            f"In {record['booktitle']}, pp. {pages}.")
        else:
            pages = record["pages"].replace("--", "–")
            number = f"({str(record['number']).replace('--', '–')})" if "number" in record else ""
            when = f"{record['year']}, {record['month']}" if "month" in record else f"{record['year']}"
            link = f" {record['url']}" if "url" in record else ""
            rendered.append(
                f"{record['author']} ({when}). {record['title']}. {record['journal']} "
                f"{record['volume']}{number}, {pages}.{link}"
            )
    return "\n\n".join(rendered)
