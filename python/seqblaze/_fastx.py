"""FASTA/FASTQ readers returning ``sb.SeqBatch`` (D92; docs/wiki/io/fasta-readers.md).

Parsing and decompression run in Rust with the GIL released. Parsing decides
nothing and repairs nothing: only ``\\n`` and ``\\r\\n`` line ends are removed, every
other byte is kept, and malformed input raises ``sb.FormatError`` naming the source,
the 0-based record and the 1-based line. gzip (multi-member, BGZF) is recognised by
its magic bytes 1f 8b; nothing is detected from file names.
"""
import io
import operator
import os

import numpy as np

from . import _core
from ._errors import ParameterError
from ._result import BinaryColumn

DEFAULT_CHUNK_BYTES = 64 << 20
_U64_MAX = (1 << 64) - 1
_COLUMNS = ("id", "description", "seq", "qual")
_ACCEPTED = ("a path (str or os.PathLike), a bytes-like object (bytes, bytearray, "
             "memoryview, a C-contiguous NumPy array, ...) or a binary file object "
             "(with readinto or read)")


def _source(source):
    """(kind, native source, name) for the Rust reader (§2.1, first rule wins)."""
    if isinstance(source, (str, os.PathLike)):
        path = os.fspath(source)
        if isinstance(path, bytes):
            path = os.fsdecode(path)
        if not isinstance(path, str):
            raise ParameterError(f"os.fspath(source) returned {type(path).__name__}; expected str or bytes")
        return "path", path, path
    try:
        view = memoryview(source)
    except TypeError:
        view = None
    if view is not None:
        if not view.c_contiguous:
            raise ParameterError(f"a bytes-like source must be C-contiguous; {type(source).__name__} is not "
                                 "(copy it first, for example with np.ascontiguousarray)")
        if view.format != "B" or view.ndim != 1:
            view = view.cast("B")
        return "buffer", np.frombuffer(view, dtype=np.uint8), f"<{type(source).__name__}>"
    if isinstance(source, io.TextIOBase):
        raise ParameterError(f"source is a text-mode file ({type(source).__name__}); open the file with \"rb\"")
    if hasattr(source, "readinto") or hasattr(source, "read"):
        name = getattr(source, "name", None)
        return "file", source, name if isinstance(name, str) else f"<{type(source).__name__}>"
    raise ParameterError(f"source must be {_ACCEPTED}; got {type(source).__name__}")


def _note_path(error, source):
    """Add the §2.1 note when a str that cannot be opened looks like content."""
    if (isinstance(source, str) and ("\n" in source or source.startswith((">", "@")))
            and hasattr(error, "add_note")):
        error.add_note("a str is a path; pass bytes for in-memory text")


def _limit(value, name):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)):
        raise ParameterError(f"{name} must be None or an integer >= 1, not bool")
    try:
        number = operator.index(value)
    except TypeError:
        raise ParameterError(f"{name} must be None or an integer >= 1; got {type(value).__name__}") from None
    if number < 1:
        raise ParameterError(f"{name} must be None or an integer >= 1; got {number}")
    return min(number, _U64_MAX)


def _read(fmt, source):
    kind, native, name = _source(source)
    try:
        core = _core._fastx_read(fmt, kind, native, name)
    except OSError as error:
        _note_path(error, source)
        raise
    return SeqBatch._wrap(core)


def _iter(fmt, source, chunk_records, chunk_bytes):
    records = _limit(chunk_records, "chunk_records")
    nbytes = _limit(chunk_bytes, "chunk_bytes")
    kind, native, name = _source(source)
    try:
        stream = _core._fastx_stream(fmt, kind, native, name, records, nbytes)
    except OSError as error:
        _note_path(error, source)
        raise
    return _BatchIterator(stream)


def read_fasta(source):
    """Read every FASTA record of ``source`` into one ``SeqBatch`` (offset 0).

    ``source`` is a path, a bytes-like object or a binary file object; gzip and
    BGZF input are recognised by their magic bytes. Malformed input raises
    ``FormatError`` (docs/wiki/io/fasta-readers.md §3, §8).
    """
    return _read("fasta", source)


def read_fastq(source):
    """Read every FASTQ record (strictly four lines each) into one ``SeqBatch``;
    ``qual`` keeps the raw quality bytes (nothing is subtracted)."""
    return _read("fastq", source)


def iter_fasta(source, *, chunk_records=None, chunk_bytes=DEFAULT_CHUNK_BYTES):
    """Iterate over consecutive FASTA batches; their concatenation equals
    ``read_fasta(source)``. A batch is emitted after the record that reaches
    ``chunk_records`` records or ``chunk_bytes`` payload bytes (§7)."""
    return _iter("fasta", source, chunk_records, chunk_bytes)


def iter_fastq(source, *, chunk_records=None, chunk_bytes=DEFAULT_CHUNK_BYTES):
    """Iterate over consecutive FASTQ batches (see ``iter_fasta``)."""
    return _iter("fastq", source, chunk_records, chunk_bytes)


class _BatchIterator:
    """Iterator over ``SeqBatch`` objects; ``close()`` and context manager (§9.2)."""
    __slots__ = ("_stream",)

    def __init__(self, stream):
        self._stream = stream

    def __iter__(self):
        return self

    def __next__(self):
        core = self._stream.next()
        if core is None:
            raise StopIteration
        return SeqBatch._wrap(core)

    def close(self):
        self._stream.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


def _human(n):
    for unit in ("B", "KiB", "MiB", "GiB"):
        if n < 1024 or unit == "GiB":
            return f"{n} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n} B"  # unreachable


class SeqBatch:
    """Records read from FASTA/FASTQ (D92): Arrow-layout ``large_binary`` columns
    ``id``, ``description``, ``seq`` and, for FASTQ, ``qual`` (``None`` for FASTA),
    plus ``offset``, the file-wide ordinal of row 0.

    Immutable; produced by the readers only. Every kernel accepts it as ``Seqs``
    through a zero-copy view of ``seq``. ``pa.table(batch)`` and
    ``pl.DataFrame(batch)`` import it without a copy (Arrow PyCapsule interface).
    """
    __slots__ = ("_core", "_columns")

    def __init__(self, *args, **kwargs):
        raise TypeError("SeqBatch has no public constructor; use sb.read_fasta, sb.read_fastq, "
                        "sb.iter_fasta or sb.iter_fastq")

    @classmethod
    def _wrap(cls, core):
        batch = object.__new__(cls)
        object.__setattr__(batch, "_core", core)
        object.__setattr__(batch, "_columns", {})
        return batch

    def __setattr__(self, name, value):
        raise AttributeError("SeqBatch is immutable")

    def __reduce__(self):
        raise TypeError("SeqBatch cannot be pickled in v1; export it with pa.table(batch) instead")

    def _column(self, name):
        column = self._columns.get(name)
        if column is None:
            offsets_owner, data_owner = self._core.buffers(name)
            column = BinaryColumn._reader(np.asarray(offsets_owner), np.asarray(data_owner),
                                          self._core.column_export(name))
            self._columns[name] = column
        return column

    @property
    def id(self):
        return self._column("id")

    @property
    def description(self):
        return self._column("description")

    @property
    def seq(self):
        return self._column("seq")

    @property
    def qual(self):
        return self._column("qual") if self._core.format == "fastq" else None

    @property
    def offset(self):
        return self._core.offset

    def __len__(self):
        return len(self._core)

    def _table(self):
        """The cached zero-copy SeqTable over ``seq`` (``sb.seqs(batch)``)."""
        return self._core.table()

    def __arrow_c_array__(self, requested_schema=None):
        """One struct array of the columns, zero-copy (Arrow C Data interface)."""
        if requested_schema is not None:
            raise NotImplementedError("Arrow requested_schema conversion is not implemented yet")
        return self._core.export().batch()

    def __arrow_c_stream__(self, requested_schema=None):
        """A stream that yields this one batch (Arrow C Stream interface)."""
        if requested_schema is not None:
            raise NotImplementedError("Arrow requested_schema conversion is not implemented yet")
        return self._core.export().stream()

    def __repr__(self):
        names = _COLUMNS if self._core.format == "fastq" else _COLUMNS[:3]
        sizes = ", ".join(f"{name}={_human(int(self._column(name).data.nbytes))}" for name in names)
        return f"SeqBatch({self._core.format}, {len(self)} records, offset={self.offset}, {sizes})"
