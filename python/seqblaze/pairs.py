"""Lazy pair spaces; implicit shapes never allocate coordinate arrays."""
from dataclasses import dataclass

from . import _core
from ._errors import PairingError, ParameterError
from ._frames import releasing
from ._inputs import SeqTable, index_array, native_call, seqs, two_tables


@dataclass(frozen=True, slots=True, eq=False)
class Pairs:
    kind: str
    a: SeqTable
    b: SeqTable
    _n_pairs: int
    _i: object = None
    _j: object = None
    _ii: object = None
    _jj: object = None
    _self_pairs: bool = False
    _scalar: bool = False

    @property
    def shape(self):
        return len(self.a), len(self.b)

    @property
    def n_pairs(self):
        return self._n_pairs

    @property
    def i(self):
        if self.kind != "index":
            raise AttributeError("i is available on index pairs only")
        return self._i

    @property
    def j(self):
        if self.kind != "index":
            raise AttributeError("j is available on index pairs only")
        return self._j


def _make(kind, a, b, *, i=None, j=None, self_pairs=False, scalar=False):
    ii = jj = None
    if kind == "index":
        i, j = index_array(i, "i"), index_array(j, "j")
        from ._device_inputs import _DeviceIndex
        ii = i._native if isinstance(i, _DeviceIndex) else native_call(_core._native_indices, i, "i")
        jj = j._native if isinstance(j, _DeviceIndex) else native_call(_core._native_indices, j, "j")
        devices = [x._device_ordinal for x in (i, j) if isinstance(x, _DeviceIndex)]
        if devices:
            if any(device != devices[0] for device in devices):
                from ._errors import DeviceError
                raise DeviceError("device index inputs belong to different CUDA devices")
            n = native_call(_core._native_pairs_device, a, b, ii, jj, devices[0])
        else:
            n = native_call(_core._native_pairs, a, b, kind, ii, jj, self_pairs)
    else:
        n = native_call(_core._native_pairs, a, b, kind, ii, jj, self_pairs)
    if kind == "index":
        i = i._public if isinstance(i, _DeviceIndex) and i._public is not None else i
        j = j._public if isinstance(j, _DeviceIndex) and j._public is not None else j
    return Pairs(kind, a, b, n, i, j, ii, jj, self_pairs, scalar)


@releasing
def zip(a, b):
    """Pair corresponding rows. Two strings/bytes give a Scalar with Each."""
    scalar_a, scalar_b = isinstance(a, (str, bytes)), isinstance(b, (str, bytes))
    if scalar_a != scalar_b:
        raise PairingError("zip received one Seq and one Seqs; use sb.pairs.one_to_many(query, sequences)")
    a, b = two_tables(a, b)
    return _make("zip", a, b, scalar=scalar_a and scalar_b)


@releasing
def index(table, i, j, *, table_b=None):
    """Explicit indices in caller order, including duplicates."""
    a, b = two_tables(table, table if table_b is None else table_b)
    return _make("index", a, b, i=i, j=j)


@releasing
def within(a, *, self_pairs=False):
    """Upper triangle of a symmetric pair space in row-major order."""
    if type(self_pairs) is not bool:
        raise ParameterError(f"self_pairs={self_pairs!r}; use self_pairs=True or False")
    a = seqs(a)
    return _make("within", a, a, self_pairs=self_pairs)


@releasing
def cross(a, b):
    """Lazy rectangular Cartesian pair space."""
    a, b = two_tables(a, b)
    return _make("cross", a, b)


@releasing
def one_to_many(q, b):
    """Pair one scalar string/bytes query with every row of b."""
    if not isinstance(q, (str, bytes)):
        raise PairingError(f"q must be one str or bytes, received {type(q).__name__}; use sb.pairs.one_to_many('AC', b)")
    a, b = two_tables(q, b)
    return _make("one_to_many", a, b)
