"""seqblaze: bit-exact batch nucleic-acid computation. For research use only.

Callable kernels: hamming (CPU or CUDA), levenshtein (global, prefix, infix; CPU
or CUDA), align (native local/global; compat emboss-water, emboss-needle; CPU or
CUDA) and native tm (two-state melting temperature; CPU or CUDA, on Pairs too) with
its Conditions, hairpin (compat='primer3': primer3-py 2.3.1 calc_hairpin over its own
parameter files read at run time, extra seqblaze[primer3]; CPU or CUDA), and duplex
(intermolecular hybridization: native MFE, model="mfe", CPU or CUDA; native
partition function, model="pf", CPU or CUDA; santalucia2004 and nndb2024-dna
presets, sources and caveats in presets("params"); compat='primer3': primer3-py 2.3.1
heterodimer, homodimer and end stability over the same run-time parameter files; CPU or
CUDA). Readers: read_fasta, read_fastq, iter_fasta and iter_fastq return SeqBatch.
describe(name) lists a kernel's fields, presets and papers; Result.cite() names the
papers behind one call. Installation, examples and known limitations: README.md.
"""
from ._core import __version__, build_info
from . import pairs, keep
from ._errors import (SeqblazeError, PairingError, ParameterError, SequenceError,
                      UnitError, DeviceError, FormatError, SeqblazeWarning, IllegalBaseWarning)
from ._inputs import SeqTable, seqs
from ._conditions import Conditions
from ._execution import execution
from ._registry import kernels, describe, presets
from ._result import (Result, FieldInfo, Manifest, Tile, TileIterator, StringColumn, ListColumn,
                      BytesColumn, BinaryColumn)
from ._scoring import Scoring
from ._device import DeviceRaggedCopy, ReadOnlyDeviceColumn
from ._fastx import SeqBatch, read_fasta, read_fastq, iter_fasta, iter_fastq
from ._generated.api import hamming, levenshtein, align, tm, hairpin, duplex

__all__ = ["__version__", "build_info", "seqs", "SeqTable", "pairs", "keep", "hamming",
           "levenshtein", "align", "Scoring", "tm", "hairpin", "duplex", "Conditions", "execution", "kernels",
           "describe", "presets", "Result", "FieldInfo", "Manifest", "Tile", "TileIterator",
           "StringColumn", "ListColumn", "BytesColumn", "BinaryColumn", "ReadOnlyDeviceColumn", "DeviceRaggedCopy", "SeqBatch",
           "read_fasta", "read_fastq", "iter_fasta", "iter_fastq", "SeqblazeError",
           "PairingError", "ParameterError", "SequenceError", "UnitError", "DeviceError",
           "FormatError",
           "SeqblazeWarning", "IllegalBaseWarning"]
