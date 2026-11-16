"""sb.Scoring: immutable alignment scoring in fixed point (docs/wiki/align/gotoh-align.md §1, §2).

Every value is held as the integer k = 1000 x (scale 1000). A value is on the
grid iff x == fl(k / 1000) for k the round-to-nearest of 1000 x in exact
rational arithmetic; off the grid is a UnitError (D48 §6).

fl(k / 1000) is the round-to-nearest-even binary64 (E0) value in every caller
floating-point environment: it is computed exactly (``_fp``), and no NumPy ufunc
runs on these paths, so the caller's status flags are left as they were
(docs/wiki/engine/text-nulls-fp-environment.md §7.4).
"""
from fractions import Fraction
import functools
import hashlib
import math
import numbers
import struct

import numpy as np

from . import _fp
from ._errors import ParameterError, UnitError

SCALE = 1000
#: |k| <= 2^30 for every value (the native value range, spec §9.1).
LIMIT = 1 << 30
MATRICES = ("EDNAFULL",)
PRESETS = {
    "emboss-water": {"gap_open": 10, "gap_extend": 0.5, "matrix": "EDNAFULL"},
    "emboss-needle": {"gap_open": 10, "gap_extend": 0.5, "matrix": "EDNAFULL", "end_gaps": False},
}


def _number(name, value):
    """A real, finite scalar as an exact int or a float; ParameterError otherwise."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real):
        raise ParameterError(f"{name}={value!r}; use a finite number, e.g. {name}=10 or {name}=0.5")
    if isinstance(value, numbers.Integral):
        return int(value)
    if isinstance(value, numbers.Rational):
        # float() of a Fraction rounds in the caller's environment; fl_RN exactly.
        number = _fp.fl_rn(value.numerator, value.denominator)
        if math.isinf(number):
            # As float(value) does for a Rational beyond the binary64 range.
            raise OverflowError("integer division result too large for a float")
        if _fp.exact(number) != value:
            raise ParameterError(f"{name}={value!r}; use a finite float64 or int value")
        return number
    if isinstance(value, (float, np.floating)):
        # From the bits: NumPy's conversion and comparison of a binary32 follow
        # the caller's DAZ/FTZ state (text-nulls-fp-environment.md §7.4).
        number = _fp.numpy_float(value)
        if number is None:  # np.longdouble: exact only when binary64 holds it
            number, ratio = _fp.longdouble(value)  # from its bits (§7.4)
            if ratio is None:
                number = math.inf
            if math.isfinite(number) and _fp.exact(number) != Fraction(*ratio):
                number = math.inf
        if not math.isfinite(number):
            raise ParameterError(f"{name}={_fp.repr_value(value)}; use a finite float64 or int value")
        return number
    number = float(value)  # another numbers.Real: its own __float__ (the caller's code)
    if not math.isfinite(number) or number != value:
        raise ParameterError(f"{name}={value!r}; use a finite float64 or int value")
    return number


@functools.lru_cache(maxsize=1024, typed=True)
def _value(k):
    """fl_RN(k / 1000), exactly, in every caller environment (§7.4). A pure function
    of the integer k, so its value is kept: every call reads its scoring record."""
    return _fp.fl_rn(k, SCALE)


def _fixed(name, value):
    """k = 1000 x, exactly; UnitError off the grid, ParameterError above 2^30."""
    if isinstance(value, int):
        k = value * SCALE
    else:
        exact = _fp.exact(value) if isinstance(value, float) else Fraction(value)
        k = round(exact * SCALE)
        if not _fp.same_value(_value(k), value):
            raise UnitError(f"{name}={_fp.repr_float(value)} is not on the fixed-point grid of "
                            f"1/{SCALE} (scale {SCALE}); use a multiple of 0.001 such as "
                            f"{_value(k)!r}")
    if abs(k) > LIMIT:
        raise ParameterError(f"{name}={_fp.repr_float(value)} exceeds the fixed-point range "
                             f"|{name}| <= {_value(LIMIT):,.3f}")
    return k


def _order_key(value):
    """Exact ascending order of a matrix entry (int or float) as an integer key; NaN last.

    A float's key is read from its binary64 bits (sign-magnitude order), so no
    float comparison runs and +0.0 and -0.0 share a key."""
    if isinstance(value, int):
        return (0, value)
    if _fp.is_nan(value):
        return (1, 0)
    u = _fp.bits(value)
    magnitude = u & ~(1 << 63)
    return (0, -magnitude if u >> 63 else magnitude)


def _distinct(array):
    """``(flat, patterns, values)`` of a 2-D numeric array: its row-major entries
    as integers (bit patterns for floats), their sorted distinct patterns, and
    each pattern's value as a Python int or float.

    binary16/32 values are decoded from their bits: NumPy's float32 to float
    conversion reads a subnormal as zero under DAZ (§7.4). np.longdouble entries
    must be exact binary64 values, as for a scalar, and are repacked as binary64."""
    if array.dtype.kind == "f" and array.dtype.itemsize not in (2, 4, 8):  # np.longdouble
        entries = [_number("matrix entry", value) for value in array.ravel()]
        array = np.frombuffer(struct.pack(f"={len(entries)}d", *entries), dtype=np.float64)
    if array.dtype.kind != "f":
        flat = array.ravel()
        patterns = sorted(set(flat.tolist()))
        return flat, patterns, patterns
    flat = array.view(array.dtype.str.replace("f", "u")).ravel()
    patterns = sorted(set(flat.tolist()))
    shape = _fp.BINARY_FORMATS.get(array.dtype.itemsize)
    if shape is None:  # binary64: the pattern is the value
        return flat, patterns, [_fp.from_bits(u) for u in patterns]
    return flat, patterns, [_fp.decode(u, *shape) for u in patterns]


def _table(matrix):
    """A 256 x 256 numeric array as read-only int32 fixed point (k = 1000 x).

    Entries are checked in ascending order of their distinct values (NaN last),
    in Python on exact values. NumPy only sorts, indexes and copies integers
    here; no ufunc or cast runs, so the caller's floating-point status flags
    stay as they were (§7.4)."""
    try:
        array = np.asarray(matrix)
    except Exception as error:  # noqa: BLE001 - any conversion failure is a parameter error
        raise ParameterError(f"matrix could not be read as an array: {error}") from None
    if array.shape != (256, 256):
        raise ParameterError(f"matrix has shape {array.shape}; use matrix='EDNAFULL' or a 256 x 256 "
                             "array indexed by byte value")
    if array.dtype.kind not in "iuf":
        raise ParameterError(f"matrix has dtype {array.dtype}; use an integer or float array")
    flat, patterns, values = _distinct(array)
    keys = [_order_key(value) for value in values]
    fixed = {}
    for key, value in sorted(zip(keys, values), key=lambda item: item[0]):
        if key in fixed:  # +0.0 and -0.0, or another NaN
            continue
        if isinstance(value, float) and not math.isfinite(value):
            raise ParameterError(f"matrix entry {value!r} is not finite")
        fixed[key] = _fixed("matrix entry", value)
    index = np.searchsorted(np.array(patterns, dtype=flat.dtype), flat)
    table = np.array([fixed[key] for key in keys], dtype=np.int32).take(index).reshape(256, 256)
    if np.ascontiguousarray(table.T).tobytes() != table.tobytes():
        rows = table.tolist()
        x, y = next((x, y) for x in range(256) for y in range(x + 1, 256) if rows[x][y] != rows[y][x])
        raise ParameterError(f"matrix is not symmetric: entry [{x}, {y}] differs from "
                             f"[{y}, {x}]; native align requires a symmetric matrix (D48)")
    # Built from bytes: read-only.
    return np.frombuffer(table.tobytes(), dtype=np.int32).reshape(256, 256)


class Scoring:
    """Immutable, hashable alignment scoring (D48 §1.4).

    Penalties ``gap_open`` and ``gap_extend`` are >= 0; a gap run of L columns costs
    gap_open + (L - 1) gap_extend. Give either ``matrix`` ("EDNAFULL" or a symmetric
    256 x 256 array) or ``match`` and ``mismatch``. ``end_gaps`` (required for
    global alignment) says whether end gaps are penalized.
    """
    __slots__ = ("_go", "_ge", "_matrix", "_table", "_sha256", "_match", "_mismatch", "_end_gaps")

    def __init__(self, *, gap_open, gap_extend, matrix=None, match=None, mismatch=None,
                 end_gaps=None):
        values = {}
        for name, value in (("gap_open", gap_open), ("gap_extend", gap_extend)):
            number = _number(name, value)
            # Sign by the bits: DAZ would read a negative subnormal as -0 (§7.4).
            if number < 0 if isinstance(number, int) else _fp.is_negative(number):
                raise ParameterError(f"{name}={_fp.repr_value(value)}; penalties are non-negative "
                                     f"(a gap run of L costs gap_open + (L - 1) * gap_extend)")
            values[name] = _fixed(name, number)
        has_scores = match is not None or mismatch is not None
        if matrix is None and not has_scores:
            raise ParameterError("Scoring needs matrix='EDNAFULL' (or a 256 x 256 array) or "
                                 "match= and mismatch=, e.g. sb.Scoring(gap_open=10, "
                                 "gap_extend=0.5, matrix='EDNAFULL')")
        if matrix is not None and has_scores:
            raise ParameterError("Scoring takes either matrix or match/mismatch, not both")
        table = sha256 = None
        if has_scores:
            if match is None or mismatch is None:
                raise ParameterError("match and mismatch go together, e.g. match=5, mismatch=-4")
            match_k = _fixed("match", _number("match", match))
            mismatch_k = _fixed("mismatch", _number("mismatch", mismatch))
        else:
            match_k = mismatch_k = None
            if isinstance(matrix, str):
                if matrix not in MATRICES:
                    similar = _fp.close_matches(matrix, MATRICES, n=1, cutoff=0.5)
                    hint = f"; did you mean {similar[0]!r}?" if similar else ""
                    raise ParameterError(f"matrix={matrix!r} is not a named matrix{hint}; named "
                                         f"matrices: {', '.join(MATRICES)}, or pass a 256 x 256 array")
            else:
                table = _table(matrix)
                sha256 = hashlib.sha256(table.astype("<i4").tobytes()).hexdigest()
                matrix = None
        if end_gaps is not None and not isinstance(end_gaps, (bool, np.bool_)):
            raise ParameterError(f"end_gaps={end_gaps!r}; use end_gaps=True, False or None")
        for name, value in (("_go", values["gap_open"]), ("_ge", values["gap_extend"]),
                            ("_matrix", matrix), ("_table", table), ("_sha256", sha256),
                            ("_match", match_k), ("_mismatch", mismatch_k),
                            ("_end_gaps", None if end_gaps is None else bool(end_gaps))):
            object.__setattr__(self, name, value)

    @staticmethod
    def preset(name):
        """The scoring of a compat preset: 'emboss-water' or 'emboss-needle'."""
        if type(name) is not str or name not in PRESETS:
            similar = _fp.close_matches(name, PRESETS, n=1) if isinstance(name, str) else []
            hint = f"; did you mean {similar[0]!r}?" if similar else ""
            raise ParameterError(f"unknown scoring preset {name!r}{hint}; presets: "
                                 f"{', '.join(map(repr, PRESETS))}")
        return Scoring(**PRESETS[name])

    def __setattr__(self, name, value):
        raise AttributeError("Scoring is immutable")

    def __delattr__(self, name):
        raise AttributeError("Scoring is immutable")

    @property
    def gap_open(self):
        return _value(self._go)

    @property
    def gap_extend(self):
        return _value(self._ge)

    @property
    def matrix(self):
        """'EDNAFULL', a read-only float64 256 x 256 array, or None."""
        if self._table is not None:
            ks = self._table.ravel().tolist()
            values = {k: _value(k) for k in set(ks)}
            data = struct.pack(f"={len(ks)}d", *(values[k] for k in ks))
            return np.frombuffer(data, dtype=np.float64).reshape(256, 256)
        return self._matrix

    @property
    def match(self):
        return None if self._match is None else _value(self._match)

    @property
    def mismatch(self):
        return None if self._mismatch is None else _value(self._mismatch)

    @property
    def end_gaps(self):
        return self._end_gaps

    def _key(self):
        return (self._go, self._ge, self._matrix, self._sha256, self._match, self._mismatch,
                self._end_gaps)

    def __eq__(self, other):
        return isinstance(other, Scoring) and self._key() == other._key()

    def __hash__(self):
        return hash(self._key())

    def __repr__(self):
        if self._table is not None:
            source = f"matrix=<256x256 sha256 {self._sha256[:12]}>"
        elif self._matrix is not None:
            source = f"matrix={self._matrix!r}"
        else:
            source = f"match={self.match!r}, mismatch={self.mismatch!r}"
        return (f"sb.Scoring(gap_open={self.gap_open!r}, gap_extend={self.gap_extend!r}, "
                f"{source}, end_gaps={self._end_gaps!r})")

    def __reduce__(self):
        kwargs = dict(gap_open=self.gap_open, gap_extend=self.gap_extend, end_gaps=self._end_gaps)
        if self._table is not None:
            kwargs["matrix"] = self.matrix
        elif self._matrix is not None:
            kwargs["matrix"] = self._matrix
        else:
            kwargs.update(match=self.match, mismatch=self.mismatch)
        return (_rebuild, (kwargs,))

    def _manifest(self):
        """The manifest's scoring record (spec §8)."""
        return {"gap_open": self.gap_open, "gap_extend": self.gap_extend,
                "matrix": self._sha256 if self._table is not None else self._matrix,
                "match": self.match, "mismatch": self.mismatch, "end_gaps": self._end_gaps,
                "scale": SCALE}

    def _native(self):
        """The native configuration: fixed-point integers, the table as int32 LE bytes."""
        config = {"gap_open": self._go, "gap_extend": self._ge, "matrix": self._matrix,
                  "match": self._match, "mismatch": self._mismatch, "end_gaps": self._end_gaps}
        if self._table is not None:
            config["table"] = self._table.astype("<i4").tobytes()
        return config


def _rebuild(kwargs):
    return Scoring(**kwargs)
