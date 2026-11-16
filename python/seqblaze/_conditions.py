"""Thermodynamic conditions (signed API §1.4; docs/wiki/thermo/tm.md §3.1)."""
from fractions import Fraction
import math
import numbers

import numpy as np

from . import _fp
from ._errors import ParameterError

_ABSOLUTE_ZERO = Fraction(-5463, 20)  # -273.15 exactly (tm.md §2.2)
_OPTIONAL = ("k_mM", "tris_mM", "mg_mM", "dntp_mM", "dmso_pct", "formamide_M")
_FIELDS = ("temp_c", "na_mM", "oligo_nM") + _OPTIONAL
_EXAMPLE = "sb.Conditions(temp_c=37, na_mM=50, oligo_nM=250)"


class _Zero:
    """Default marker of an optional argument: shown as 0, recorded as defaulted."""
    __slots__ = ()

    def __repr__(self):
        return "0"


_DEFAULT = _Zero()


def _to_float(value):
    """The binary64 of a real value, as float() gives it in the default environment.

    The facade runs in the caller's floating-point environment
    (text-nulls-fp-environment.md §7.4), so nothing here uses a hardware
    conversion: a float is taken as is, an int (np.integer included, whose
    float() is a hardware conversion that follows the rounding mode) is rounded
    in integer arithmetic, NumPy binary16/32 values are decoded from their bits
    (float() of a binary32 reads a subnormal as zero under DAZ), and other
    rationals (Fraction, np.longdouble decoded from its bits) get fl_RN exactly.
    Any other numbers.Real converts by its own __float__, which is the caller's
    code.
    """
    if isinstance(value, float):
        return float(value)
    if isinstance(value, (int, np.integer)):
        return _fp.integer(int(value))
    decoded = _fp.numpy_float(value) if isinstance(value, np.floating) else None
    if decoded is not None:
        return decoded
    if isinstance(value, numbers.Rational):
        result = _fp.fl_rn(value.numerator, value.denominator)
        if math.isinf(result):
            # As float(value) does for a Rational beyond the binary64 range.
            raise OverflowError("integer division result too large for a float")
        return result
    if isinstance(value, np.floating):  # np.longdouble: wider than binary64
        # fl_RN of the exact value from its bits, the sign of zero kept; +-inf
        # beyond the range and inf/NaN kept, as float() gives in E0.
        return _fp.longdouble(value)[0]
    return float(value)


def _number(name, value):
    """``float(value)`` of a real, finite, non-bool number; else ParameterError."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real):
        raise ParameterError(f"{name}={_fp.repr_value(value)} is not a number; e.g. {_EXAMPLE}")
    try:
        result = _to_float(value)
    except OverflowError:
        raise ParameterError(f"{name}={_fp.repr_value(value)} is too large for a float64; e.g. {_EXAMPLE}") from None
    if not math.isfinite(result):
        raise ParameterError(f"{name}={_fp.repr_value(value)} is not finite; e.g. {_EXAMPLE}")
    return result


def _oligo(value):
    if isinstance(value, (tuple, list)):
        if len(value) != 2:
            raise ParameterError(f"oligo_nM={_fp.repr_value(value)}: give one total concentration or a pair "
                                 "(strand a, strand b), e.g. oligo_nM=(25, 25)")
        a, b = (_number("oligo_nM", x) for x in value)
        # Sign tests by the bits: DAZ would read a subnormal as zero (§7.4).
        if _fp.is_negative(a) or _fp.is_negative(b) or not (_fp.is_positive(a) or _fp.is_positive(b)):
            raise ParameterError(f"oligo_nM={_fp.repr_value(value)}: both strands must be >= 0 and one > 0, "
                                 "e.g. oligo_nM=(25, 25)")
        return (a, b)
    total = _number("oligo_nM", value)
    if not _fp.is_positive(total):
        raise ParameterError(f"oligo_nM={_fp.repr_value(value)} must be > 0 (total strand concentration, nM)")
    return total


class Conditions:
    """Immutable, hashable thermodynamic conditions of a call (API §1.4).

    ``temp_c`` (°C), ``na_mM`` and ``oligo_nM`` are required: one total strand
    concentration in nM, or a pair (strand a = the input sequence, strand b = its
    complement). The other concentrations default to 0 and are recorded as
    defaulted. Every value is stored as ``float(v)``; equality and hashing use the
    values only.
    """
    __slots__ = ("_values", "_defaulted")

    def __init__(self, *, temp_c, na_mM, oligo_nM, k_mM=_DEFAULT, tris_mM=_DEFAULT,
                 mg_mM=_DEFAULT, dntp_mM=_DEFAULT, dmso_pct=_DEFAULT, formamide_M=_DEFAULT):
        given = dict(temp_c=temp_c, na_mM=na_mM, oligo_nM=oligo_nM, k_mM=k_mM, tris_mM=tris_mM,
                     mg_mM=mg_mM, dntp_mM=dntp_mM, dmso_pct=dmso_pct, formamide_M=formamide_M)
        defaulted = tuple(name for name in _OPTIONAL if given[name] is _DEFAULT)
        values = {}
        for name in _FIELDS:
            value = given[name]
            if value is _DEFAULT:
                values[name] = 0.0
            elif name == "oligo_nM":
                values[name] = _oligo(value)
            else:
                number = _number(name, value)
                if name == "temp_c":
                    if _fp.exact(number) <= _ABSOLUTE_ZERO:
                        raise ParameterError(f"temp_c={_fp.repr_value(value)} is at or below absolute zero (-273.15 °C)")
                elif _fp.is_negative(number):
                    raise ParameterError(f"{name}={_fp.repr_value(value)} is negative; concentrations are >= 0")
                values[name] = number
        object.__setattr__(self, "_values", tuple(values[name] for name in _FIELDS))
        object.__setattr__(self, "_defaulted", defaulted)

    @staticmethod
    def preset(name):
        """Create a Conditions value from a reference's documented defaults.

        The supported names are ``primer3`` and ``biopython``. Each preset
        sets the solution composition, temperature and oligonucleotide amounts
        for that reference. For custom experimental conditions, call the
        Conditions constructor with explicit values.

        ``oligo_nM`` can describe one total amount or the two strands separately.
        The selected thermodynamic model defines how those amounts are used.

        References
        ----------
        Untergasser et al. (2012).
        Primer3—new capabilities and interfaces.
        Nucleic Acids Research, 40(15), e115. DOI: 10.1093/nar/gks596.

        Koressaar and Remm (2007).
        Enhancements and modifications of primer design program Primer3.
        Bioinformatics, 23(10), 1289–1291. DOI: 10.1093/bioinformatics/btm091.

        Cock et al. (2009).
        Biopython: freely available Python tools for computational molecular biology and bioinformatics.
        Bioinformatics, 25(11), 1422–1423. DOI: 10.1093/bioinformatics/btp163."""
        if type(name) is not str or name not in _PRESETS:
            similar = _fp.close_matches(name, _PRESETS, n=1) if isinstance(name, str) else []
            hint = f"; did you mean {similar[0]!r}?" if similar else ""
            raise ParameterError(f"unknown Conditions preset {name!r}{hint}; use one of "
                                 f"{', '.join(repr(k) for k in _PRESETS)}")
        return Conditions(**_PRESETS[name])

    @property
    def defaulted(self):
        """Names of the optional arguments that were not passed."""
        return self._defaulted

    def _asdict(self):
        return dict(zip(_FIELDS, self._values))

    def __setattr__(self, name, value):
        raise AttributeError("Conditions is immutable; create a new one")

    def __delattr__(self, name):
        raise AttributeError("Conditions is immutable")

    def _key(self):
        """The stored values as exact rationals: equality and hashing never compare
        floats in the caller's environment, where DAZ reads a subnormal as zero
        (§7.4). hash(Fraction) is the default-environment float hash."""
        return tuple(tuple(_fp.exact(x) for x in value) if isinstance(value, tuple)
                     else _fp.exact(value) for value in self._values)

    def __eq__(self, other):
        if not isinstance(other, Conditions):
            return NotImplemented
        return self._key() == other._key()

    def __hash__(self):
        return hash(("Conditions", self._key()))

    def __reduce__(self):
        kwargs = {name: value for name, value in zip(_FIELDS, self._values) if name not in self._defaulted}
        return (_rebuild, (kwargs,))

    def __repr__(self):
        shown = ", ".join(f"{name}={_fp.repr_value(value)}" for name, value in zip(_FIELDS, self._values)
                          if name not in self._defaulted)
        return f"Conditions({shown})"


def _field(index, name):
    return property(lambda self: self._values[index], doc=f"{name} as a float (oligo_nM: float or pair)")


for _index, _name in enumerate(_FIELDS):
    setattr(Conditions, _name, _field(_index, _name))
del _index, _name


def _rebuild(kwargs):
    return Conditions(**kwargs)


# API §1.4: the reference tools' documented defaults (confirmed by the golden sets later).
_PRESETS = {
    # primer3: 37 °C, monovalent 50 mM, Mg 1.5 mM, dNTP 0.6 mM, DNA 50 nM.
    "primer3": dict(temp_c=37, na_mM=50, mg_mM=1.5, dntp_mM=0.6, oligo_nM=50),
    # Biopython Tm_NN: Na 50 mM, K/Tris/Mg/dNTP 0, 25 nM + 25 nM; temp_c 37 for the ΔG fields.
    "biopython": dict(temp_c=37, na_mM=50, k_mM=0, tris_mM=0, mg_mM=0, dntp_mM=0, oligo_nM=(25, 25)),
}
