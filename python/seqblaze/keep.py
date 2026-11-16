"""Immutable output requests. Cutoffs use the primary field's exposed unit."""
from dataclasses import dataclass
from fractions import Fraction
import math
import operator
import sys

import numpy as np

from . import _fp
from ._errors import ParameterError


@dataclass(frozen=True, slots=True)
class Sink:
    kind: str
    cutoff: int | float | None = None
    k: int | None = None
    tile: tuple[int, int] | None = None


class EachSink(Sink):
    __slots__ = ()


class HitsSink(Sink):
    __slots__ = ()


class TopKSink(Sink):
    __slots__ = ()


class RowStatsSink(Sink):
    __slots__ = ()


class MatrixSink(Sink):
    __slots__ = ()


class TilesSink(Sink):
    __slots__ = ()


def _integer(value):
    """Return a built-in int via the index protocol (np.integer included), or None.

    bool and np.bool_ are not integers here, although bool implements __index__.
    """
    if isinstance(value, (bool, np.bool_)):
        return None
    try:
        return operator.index(value)
    except TypeError:
        return None


def _cutoff(value):
    """Normalize a real cutoff once to a built-in int or float of the same value.

    Integers (int, np.integer, any __index__ type) stay exact ints and never pass
    through float. A float is kept, and NumPy binary16/32 values are decoded from
    their bits (always exact); NumPy's own conversion and comparison follow the
    caller's DAZ/FTZ state (text-nulls-fp-environment.md §7.4). A wider float
    that float64 cannot hold exactly is rejected rather than rounded, because
    rounding could change the inclusive predicate; that check is exact.
    """
    if value is None:
        return None
    number = _integer(value)
    if number is not None:
        return number
    if isinstance(value, (float, np.floating)):
        number = _fp.numpy_float(value)
        if number is None:  # np.longdouble, decoded from its bits (§7.4)
            number, ratio = _fp.longdouble(value)
            if ratio is not None:
                if not math.isfinite(number):  # beyond the binary64 range
                    ratio = None
                elif _fp.exact(number) == Fraction(*ratio):
                    return number
            if ratio is not None:
                raise ParameterError(f"cutoff={_fp.repr_value(value)} is not exactly representable "
                                     "as a float64; use a float64 or int cutoff")
        elif math.isfinite(number):
            return number
    raise ParameterError(f"cutoff={_fp.repr_value(value)}; use a finite int, float, or None, "
                         "e.g. sb.keep.hits(cutoff=1)")


def _positive(name, value, hint):
    number = _integer(value)
    if number is None or not 0 < number <= sys.maxsize:
        raise ParameterError(f"{name}={value!r}; {hint}")
    return number


def each(*, cutoff=None):
    return EachSink("each", _cutoff(cutoff))


def hits(cutoff):
    if cutoff is None:
        raise ParameterError("cutoff=None is invalid for hits; use sb.keep.hits(cutoff=1)")
    return HitsSink("hits", _cutoff(cutoff))


def topk(k, *, cutoff=None):
    k = _positive("k", k, "use sb.keep.topk(k=1) with a positive platform-sized int")
    return TopKSink("topk", _cutoff(cutoff), k=k)


def rowstats(*, cutoff=None):
    return RowStatsSink("rowstats", _cutoff(cutoff))


def matrix():
    return MatrixSink("matrix")


def tiles(*, tile=4096):
    if type(tile) is tuple:
        dims = tile if len(tile) == 2 else ()
    else:
        dims = (tile, tile)
    shape = tuple(_integer(n) for n in dims)
    if len(shape) != 2 or any(n is None or not 0 < n <= sys.maxsize for n in shape):
        raise ParameterError(f"tile={tile!r}; use sb.keep.tiles(tile=4096) or tile=(rows, cols) with positive ints")
    return TilesSink("tiles", tile=shape)
