"""Exact binary64 helpers for the Python facade (docs/wiki/engine/text-nulls-fp-environment.md §7.4).

The facade runs in the caller's floating-point environment, which may round
toward a direction, flush subnormal results to zero (FTZ) or read subnormal
operands as zero (DAZ). Every value the facade derives must nevertheless be the
round-to-nearest-even (E0) value, so this module never performs a rounded float
operation and never compares or decomposes a float with float arithmetic:
values are taken apart and built from their bit patterns, and rounding is done
in integer arithmetic. ``fl_rn(n, d)`` is the binary64 nearest to the rational
n/d, ties to the even significand, overflowing to +-inf (IEEE 754-2019
roundTiesToEven).

The one exception is ``bracket`` and its user ``close_matches``: a caller-state
bracket (§7.2) encloses the few Python steps whose status flags come from
CPython's own floating-point arithmetic and cannot be made exact (§7.3 item 5:
the primer3 parameter-loader step and the did-you-mean suggestions, extended
in §7.4 to metadata codecs and optional device-library imports/stream metadata). No float
computed inside a bracket reaches a value; the bracket only writes the caller's
whole state back at its exit.
"""
from difflib import get_close_matches as _get_close_matches
from fractions import Fraction
import struct
import sys

import numpy as np

from . import _core

_SIGN = 1 << 63
_MAGNITUDE = _SIGN - 1
_INF_BITS = 0x7FF0000000000000
_HIDDEN = 1 << 52


def bits(x):
    """The binary64 encoding of ``x`` as an unsigned integer."""
    return struct.unpack("<Q", struct.pack("<d", x))[0]


def from_bits(u):
    """The float whose binary64 encoding is ``u``."""
    return struct.unpack("<d", struct.pack("<Q", u))[0]


def ratio(x):
    """Exact (numerator, denominator) of a finite float, read from its bits."""
    u = bits(x)
    exponent = (u >> 52) & 0x7FF
    if exponent == 0x7FF:
        raise ValueError("ratio of a non-finite float")
    mantissa = u & (_HIDDEN - 1)
    if exponent:
        mantissa |= _HIDDEN
        shift = exponent - 1075
    else:
        shift = -1074
    if u & _SIGN:
        mantissa = -mantissa
    if shift >= 0:
        return mantissa << shift, 1
    # Reduce by the common power of two only (enough for an exact Fraction).
    trailing = (mantissa & -mantissa).bit_length() - 1 if mantissa else -shift
    drop = min(trailing, -shift)
    return mantissa >> drop, 1 << (-shift - drop)


def exact(x):
    """The exact value of a finite float, as a Fraction."""
    return Fraction(*ratio(x))


def is_zero(x):
    return bits(x) & _MAGNITUDE == 0


def is_nan(x):
    return bits(x) & _MAGNITUDE > _INF_BITS


def is_negative(x):
    """x < 0 by the bits (a negative subnormal counts; -0.0 and NaN do not)."""
    u = bits(x)
    return bool(u & _SIGN) and 0 < u & _MAGNITUDE <= _INF_BITS


def is_positive(x):
    """x > 0 by the bits (a positive subnormal counts; +0.0 and NaN do not)."""
    u = bits(x)
    return not u & _SIGN and 0 < u <= _INF_BITS


def is_subnormal(x):
    u = bits(x) & _MAGNITUDE
    return 0 < u < _HIDDEN


#: (exponent bits, fraction bits) of the NumPy binary formats decoded from their bits.
BINARY_FORMATS = {2: (5, 10), 4: (8, 23)}


def numpy_float(value):
    """The binary64 of a NumPy binary16/binary32/binary64 scalar, from its bits.

    NumPy's own conversion (``float()``, ``tolist()``, ``==`` against a float)
    of a binary32 value is a hardware cvtss2sd, which reads a subnormal as zero
    under DAZ and whose comparisons follow FTZ (text-nulls-fp-environment.md
    §7.4). Every binary16/32 value is a binary64 value, so decoding the
    encoding in integer arithmetic gives it exactly in every environment.
    Returns None for other NumPy floats (np.longdouble).
    """
    if isinstance(value, float):  # np.float64 is a float: no conversion happens
        return from_bits(bits(value))
    shape = BINARY_FORMATS.get(value.dtype.itemsize) if isinstance(value, np.floating) else None
    if shape is None:
        return None
    return decode(int.from_bytes(value.tobytes(), sys.byteorder), *shape)


def decode(u, exponent_bits, fraction_bits):
    """The binary64 of the IEEE binary encoding ``u`` with the given field widths."""
    sign = _SIGN if u >> (exponent_bits + fraction_bits) & 1 else 0
    exponent = (u >> fraction_bits) & ((1 << exponent_bits) - 1)
    fraction = u & ((1 << fraction_bits) - 1)
    if exponent == (1 << exponent_bits) - 1:  # inf, or a NaN (quieted, as converted)
        payload = (fraction << (52 - fraction_bits)) | (1 << 51) if fraction else 0
        return from_bits(sign | _INF_BITS | payload)
    bias = (1 << (exponent_bits - 1)) - 1
    if exponent:
        fraction |= 1 << fraction_bits
    shift = max(exponent, 1) - bias - fraction_bits
    magnitude = fl_rn(fraction << shift) if shift >= 0 else fl_rn(fraction, 1 << -shift)
    return from_bits(sign | bits(magnitude))


def _longdouble_layout():
    """The np.longdouble storage format, from the bytes of np.longdouble(1).

    Converting the integer 1 is exact in every environment, and no float
    arithmetic runs here (np.finfo(np.longdouble) would: its format discovery
    divides long doubles, which follows the x87 precision control)."""
    one = int.from_bytes(np.longdouble(1).tobytes(), sys.byteorder)
    size = np.dtype(np.longdouble).itemsize
    if size == 8 and one == 0x3FF0000000000000:
        return "binary64"
    if size >= 10 and one & ((1 << 80) - 1) == (0x3FFF << 64) | (1 << 63):
        return "x87"  # 80-bit extended in the low 10 bytes; the rest is padding
    if size == 16 and one == 0x3FFF << 112:
        return "binary128"
    if size == 16 and np.longdouble(1).tobytes() == struct.pack("=dd", 1.0, 0.0):
        return "double-double"
    return None


_LONGDOUBLE = _longdouble_layout()


def _quiet_nan(sign, fraction, fraction_bits):
    """A quiet binary64 NaN carrying the top fraction bits, as a narrowing conversion keeps them."""
    payload = fraction >> (fraction_bits - 52) if fraction_bits >= 52 else fraction << (52 - fraction_bits)
    return from_bits(sign | _INF_BITS | (payload & (_HIDDEN - 1)) | (1 << 51))


def longdouble(value):
    """``(binary64, ratio)`` of an np.longdouble scalar, read from its bits.

    ``binary64`` is fl_RN of the value with the sign of a zero kept (+-inf
    beyond the binary64 range; inf and NaN map to inf and a quiet NaN), and
    ``ratio`` is the exact ``(numerator, denominator)`` of a finite value or None
    for inf and NaN. NumPy's ``float()`` and ``as_integer_ratio()`` of a long
    double run x87 instructions on x86-64, which follow the caller's x87
    rounding and precision control (text-nulls-fp-environment.md §7.1, §7.4);
    decoding the encoding in integer arithmetic gives the same answer in every
    environment.
    """
    u = int.from_bytes(value.tobytes(), sys.byteorder)
    if _LONGDOUBLE == "binary64":
        x = from_bits(u)
        return x, (None if bits(x) & _INF_BITS == _INF_BITS else ratio(x))
    if _LONGDOUBLE == "double-double":
        hi, lo = struct.unpack("=dd", value.tobytes())
        if bits(hi) & _INF_BITS == _INF_BITS:
            return hi, None
        n_hi, d_hi = ratio(hi)
        n_lo, d_lo = ratio(lo)
        n, d = n_hi * d_lo + n_lo * d_hi, d_hi * d_lo
        x = fl_rn(n, d)
        return from_bits((bits(hi) & _SIGN) | (bits(x) & _MAGNITUDE)), (n, d)
    if _LONGDOUBLE == "x87":
        sign = _SIGN if u >> 79 & 1 else 0
        exponent = (u >> 64) & 0x7FFF
        significand = u & ((1 << 64) - 1)
        if exponent == 0x7FFF or (exponent and not significand >> 63):
            # inf (integer bit set, fraction 0); otherwise a NaN, or an
            # encoding the x87 unit rejects as invalid (a NaN when converted)
            fraction = significand & ((1 << 63) - 1)
            if exponent == 0x7FFF and significand == 1 << 63:
                return from_bits(sign | _INF_BITS), None
            return _quiet_nan(sign, fraction, 63), None
        shift = max(exponent, 1) - 16383 - 63
    elif _LONGDOUBLE == "binary128":
        sign = _SIGN if u >> 127 & 1 else 0
        exponent = (u >> 112) & 0x7FFF
        significand = u & ((1 << 112) - 1)
        if exponent == 0x7FFF:
            if not significand:
                return from_bits(sign | _INF_BITS), None
            return _quiet_nan(sign, significand, 112), None
        if exponent:
            significand |= 1 << 112
        shift = max(exponent, 1) - 16383 - 112
    else:  # an unknown layout: NumPy's own exact ratio (never reached on x86-64 or aarch64)
        try:
            n, d = value.as_integer_ratio()
        except (OverflowError, ValueError):
            return float(value), None
        x = fl_rn(n, d)
        return from_bits((_SIGN if np.signbit(value) else 0) | (bits(x) & _MAGNITUDE)), (n, d)
    n, d = (significand << shift, 1) if shift >= 0 else (significand, 1 << -shift)
    magnitude = fl_rn(n, d)
    return from_bits(sign | bits(magnitude)), (-n if sign else n, d)


def integer(n):
    """fl_RN of an int; OverflowError beyond the binary64 range, as float(int) raises."""
    result = fl_rn(n)
    if bits(result) & _MAGNITUDE == _INF_BITS:
        raise OverflowError("int too large to convert to float")
    return result


def decimal(digits):
    """The non-negative int spelt by the ASCII decimal digits ``digits``, folded in integer
    arithmetic; ValueError for an empty string or any other character, as ``int()`` raises.

    ``int()`` of a decimal string raises the inexact flag PE in the caller's
    floating-point environment (measured on CPython 3.12: every decimal string,
    ``"0"`` included; base 2 and base 16 do not), so no call path parses a decimal
    string with it (docs/wiki/engine/text-nulls-fp-environment.md §7.4, status flags).
    """
    if not digits:
        raise ValueError("decimal() of an empty string")
    value = 0
    for character in digits:
        digit = ord(character) - 48
        if not 0 <= digit <= 9:
            raise ValueError(f"decimal() of a non-digit character {character!r}")
        value = value * 10 + digit
    return value


def _round_half_even(n, d):
    q, r = divmod(n, d)
    if 2 * r > d or (2 * r == d and q & 1):
        q += 1
    return q


def fl_rn(n, d=1):
    """The binary64 nearest to the rational n/d (d > 0), ties to even; +-inf on overflow."""
    if d <= 0:
        raise ValueError("fl_rn needs a positive denominator")
    sign = _SIGN if n < 0 else 0
    n = abs(n)
    if n == 0:
        return from_bits(sign)
    # e such that 2**52 <= n / (d * 2**e) < 2**53, then the subnormal floor.
    e = n.bit_length() - d.bit_length() - 53
    while True:
        num, den = (n, d << e) if e >= 0 else (n << -e, d)
        if num >= den << 53:
            e += 1
        elif num < den << 52:
            e -= 1
        else:
            break
    if e < -1074:
        e = -1074
        num, den = (n, d << e) if e >= 0 else (n << -e, d)
    q = _round_half_even(num, den)
    if q == 1 << 53:
        q, e = _HIDDEN, e + 1
    if q < _HIDDEN:  # subnormal (e == -1074)
        return from_bits(sign | q)
    biased = e + 52 + 1023
    if biased >= 0x7FF:
        return from_bits(sign | _INF_BITS)
    return from_bits(sign | (biased << 52) | (q - _HIDDEN))


def fl_rn_fraction(q):
    """fl_rn of a Fraction (or any numbers.Rational)."""
    return fl_rn(q.numerator, q.denominator)


def same_value(x, y):
    """x == y for floats, by the bits: +0.0 equals -0.0; NaN equals nothing."""
    ux, uy = bits(x), bits(y)
    if is_nan(x) or is_nan(y):
        return False
    return ux == uy or (ux & _MAGNITUDE == 0 and uy & _MAGNITUDE == 0)


def _decimal_exponent(num, den):
    """p with 10**p <= num/den < 10**(p + 1), for num, den > 0."""
    p = len(str(num // den)) - 1 if num >= den else -len(str(den // num))
    while (num * 10 ** -p if p < 0 else num) < (den if p < 0 else den * 10 ** p):
        p -= 1
    while (num * 10 ** (-p - 1) if p + 1 <= 0 else num) >= \
            (den if p + 1 <= 0 else den * 10 ** (p + 1)):
        p += 1
    return p


def _shortest_subnormal(x):
    """CPython's repr of a subnormal float, computed exactly.

    The neighbours of a subnormal are equally spaced (2**-1074), so the shortest
    decimal that rounds back to x is the n-digit decimal nearest to x for the
    smallest n at which that one rounds back; an exact tie between two n-digit
    decimals cannot occur for n <= 17. repr then uses exponent notation.
    """
    u = bits(x)
    num, den = ratio(from_bits(u & _MAGNITUDE))
    p = _decimal_exponent(num, den)
    for digits in range(1, 18):
        scale = digits - 1 - p  # candidate = m * 10**-scale
        if scale >= 0:
            m = _round_half_even(num * 10 ** scale, den)
            back = fl_rn(m, 10 ** scale)
        else:
            m = _round_half_even(num, den * 10 ** -scale)
            back = fl_rn(m * 10 ** -scale)
        if bits(back) == u & _MAGNITUDE:
            text = str(m)
            exponent = len(text) - 1 - scale
            rest = text[1:].rstrip("0")
            mantissa = text[0] + ("." + rest if rest else "")
            sign = "-" if u & _SIGN else ""
            return f"{sign}{mantissa}e{'-' if exponent < 0 else '+'}{abs(exponent):02d}"
    raise AssertionError("no shortest decimal found")  # unreachable for binary64


def _numpy_subnormal(x):
    """Whether ``x`` is a NumPy binary16/32/64 scalar that is subnormal in its own format."""
    if not isinstance(x, np.floating):
        return False
    shape = BINARY_FORMATS.get(x.dtype.itemsize) or ((11, 52) if x.dtype.itemsize == 8 else None)
    if shape is None:  # np.longdouble
        return False
    exponent_bits, fraction_bits = shape
    u = int.from_bytes(x.tobytes(), sys.byteorder)
    return (u >> fraction_bits) & ((1 << exponent_bits) - 1) == 0 and u & ((1 << fraction_bits) - 1) != 0


def repr_float(x):
    """repr(x) as CPython or NumPy prints it in the default environment.

    CPython's float repr is unchanged under directed rounding but differs for
    subnormal values under DAZ (text-nulls-fp-environment.md §8.2), and so does
    NumPy's scalar repr, which picks positional or exponent notation by a float
    comparison. A subnormal Python float is formatted here exactly. A subnormal
    NumPy binary16/32/64 scalar is printed in exponent notation, as its E0 repr
    is, with NumPy's shortest digits (Dragon4 on the bits, which no environment
    changes) inside the scalar's own repr wrapper. Everything else uses repr."""
    if type(x) is float and is_subnormal(x):
        return _shortest_subnormal(x)
    if _numpy_subnormal(x):
        digits = np.format_float_scientific(x, unique=True, trim="-")
        return repr(type(x)(0)).replace("0.0", digits, 1)  # e.g. 'np.float32(0.0)'
    return repr(x)


def repr_value(value):
    """repr(value) as in the default environment, for a float or a tuple/list of
    values (an ``oligo_nM`` pair); other objects use their own repr."""
    if type(value) in (tuple, list):
        inner = ", ".join(repr_value(item) for item in value)
        if type(value) is list:
            return f"[{inner}]"
        return f"({inner},)" if len(value) == 1 else f"({inner})"
    return repr_float(value)


def bracket():
    """A caller-state bracket (text-nulls-fp-environment.md §7.2): ``with bracket(): ...``
    saves the calling thread's whole floating-point state at entry and writes it back on
    every exit (normal end, exception, KeyboardInterrupt, a panic of native code inside),
    without installing the default environment, so the code inside runs in the caller's
    environment and only what it leaves in the state (status flags, a control change) is
    undone. Single use and bound to the calling thread. Used only at the sites of §7.3
    item 5; never around facade arithmetic whose result reaches a value, an input adapter,
    a producer callback or a user ``__float__``/``__index__``."""
    return _core._FpBracket()


def close_matches(word, possibilities, n=3, cutoff=0.6):
    """``difflib.get_close_matches(word, possibilities, n, cutoff)`` inside a caller-state
    bracket (§7.3 item 5): the same list, but the inexact flag that difflib's similarity
    ratios raise does not reach the caller. Every did-you-mean suggestion of the package
    goes through here, with its own ``n`` and ``cutoff``."""
    with bracket():
        return _get_close_matches(word, possibilities, n=n, cutoff=cutoff)
