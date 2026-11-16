"""Immutable Result columns backed by retained host or device buffers."""
from collections.abc import Mapping
from dataclasses import dataclass, field
from html import escape
from threading import RLock
from types import MappingProxyType

import numpy as np

from . import _device_exports, _device_selection, _fp, _manifest_codec
from ._device import ReadOnlyDeviceColumn
from ._errors import ParameterError
# The module object, not its names: _provenance reads this module's names.
from . import _provenance
from ._registry import _REGISTRY
from . import _result_export


def _unpickle_indices(data, dtype):
    # Reconstruct only on explicit pickle load, retaining immutable bytes.
    return np.frombuffer(data, dtype=dtype)


class _PickledIndices:
    __slots__ = ("_array",)

    def __init__(self, array):
        self._array = array

    def __reduce__(self):
        return (_unpickle_indices, (self._array.tobytes(), self._array.dtype.str))


def _unpickle_frozen_array(data, dtype, shape):
    return _FrozenArray._owned(data, dtype, shape)


class _FrozenArray:
    """Private immutable metadata payload with no shared ndarray descriptor.

    External arrays are copied to bytes. The selection factory alone retains
    native immutable storage; its byte memoryview fixes the exported span even
    if a caller changes an ndarray header reachable through a public .base.
    """
    __slots__ = ("_buffer", "_dtype", "_shape")

    def __init__(self, array):
        array = np.asarray(array)
        if array.dtype.kind not in "biufc" or array.dtype.fields is not None:
            raise ParameterError("manifest arrays require numeric, non-structured dtypes")
        self._set(memoryview(array.tobytes(order="C")), array.dtype.str, array.shape)

    def _set(self, buffer, dtype, shape):
        self._buffer = buffer
        self._dtype = np.dtype(dtype)
        self._shape = tuple(shape)

    @classmethod
    def _owned(cls, data, dtype, shape):
        # Pickle's byte value is already immutable; no second payload copy.
        value = cls.__new__(cls)
        value._set(memoryview(data), dtype, shape)
        return value

    @classmethod
    def _native_indices(cls, owner):
        # Only _select calls this, with the native normalizer's OwnedColumn.
        # Construct a private header directly from that immutable Rust owner,
        # not from idx: no .base path may reach a header used by selected Pairs.
        array = np.asarray(owner)
        if (type(array) is not np.ndarray or array.base is not owner or array.ndim != 1
                or array.dtype != np.dtype("int64") or not array.flags.c_contiguous
                or array.flags.writeable):
            raise ParameterError("internal selection indices require readonly native int64 storage")
        value = cls.__new__(cls)
        buffer = memoryview(array).cast("B") if array.size else memoryview(b"")
        value._set(buffer, array.dtype.str, array.shape)
        return value

    def array(self):
        # Both ndarray descriptors and the public memoryview are fresh. Even
        # changes to a reachable owner header cannot alter the saved byte span,
        # dtype or shape, or a descriptor used by the Result's selected Pairs.
        return np.frombuffer(memoryview(self._buffer), dtype=self._dtype).reshape(self._shape)

    def __reduce__(self):
        return (_unpickle_frozen_array,
                (self._buffer.tobytes(), self._dtype.str, self._shape))


def _expose(value):
    if isinstance(value, _FrozenArray):
        return value.array()
    if isinstance(value, tuple):
        return tuple(_expose(item) for item in value)
    return value


class _FrozenMap(Mapping):
    __slots__ = ("_data",)

    def __init__(self, data):
        object.__setattr__(self, "_data", MappingProxyType({key: _freeze(value) for key, value in data.items()}))

    def __setattr__(self, name, value):
        raise AttributeError("manifest metadata is immutable")

    def __getitem__(self, key):
        return _expose(self._data[key])

    def __iter__(self):
        return iter(self._data)

    def __len__(self):
        return len(self._data)

    def __getattr__(self, name):
        # Private and special names never resolve to keys. This also stops copy
        # and pickle (which build instances without __init__) from recursing.
        if name.startswith("_"):
            raise AttributeError(name)
        try:
            return _expose(self._data[name])
        except (AttributeError, KeyError):
            raise AttributeError(name) from None

    def __reduce__(self):
        # Preserve private holders on shallow copies; their own reducer writes
        # immutable bytes only for explicit deepcopy/pickle serialization.
        return (type(self), (dict(self._data),))

    def __repr__(self):
        return f"{type(self).__name__}({_plain(self)!r})"


class _SelectionNode(_FrozenMap):
    """One immutable step, sharing its parent and caching fixed-size hashes."""
    __slots__ = ("root_sha256", "indices_sha256", "step_sha256", "lineage_sha256", "depth")

    def __init__(self, facts, indices, parent, root_sha256):
        if parent is not None and (not isinstance(parent, _SelectionNode)
                                   or parent.root_sha256 != root_sha256):
            raise ParameterError("selection parent must belong to the same v2 root")
        if isinstance(indices, _device_selection._DeviceSelectionIndices):
            # The device owner already validates shape/dtype and carries its
            # native digest. Never pass it through np.asarray or the CPU hash.
            if (facts.get("index_hash_algorithm") != _manifest_codec.INPUT_HASH
                    or facts.get("indices_sha256") != indices._selection_sha256):
                raise ParameterError("device selection facts must match the owned index digest")
        else:
            if not isinstance(indices, _FrozenArray):
                indices = _FrozenArray(indices)
            if indices._dtype.kind != "i" or indices._dtype.itemsize != 8 or len(indices._shape) != 1:
                raise ParameterError("selection indices require a one-dimensional int64 array")
        facts = _FrozenMap(facts)
        super().__init__(dict(facts, indices=indices, parent=parent))
        step = _manifest_codec.digest("selection-step-v2", facts)
        lineage = _manifest_codec.digest(_manifest_codec.SELECTION_HASH, {
            "parent_sha256": _provenance.selection_lineage_digest(parent, root_sha256),
            "step_sha256": step})
        for name, value in (("root_sha256", root_sha256),
                            ("indices_sha256", facts["indices_sha256"]),
                            ("step_sha256", step), ("lineage_sha256", lineage),
                            ("depth", 1 if parent is None else parent.depth + 1)):
            object.__setattr__(self, name, value)

    def _steps(self):
        steps, node = [], self
        while node is not None:
            steps.append({key: value for key, value in node._data.items() if key != "parent"})
            node = node._data["parent"]
        steps.reverse()
        return steps

    def __reduce__(self):
        # No parent object appears in the reducer arguments. copy/deepcopy and
        # pickle can therefore handle any selection depth with their defaults.
        return (_restore_selection, (self.root_sha256, self._steps()))

    def __repr__(self):
        return f"_SelectionNode(depth={self.depth}, lineage_sha256={self.lineage_sha256!r})"


def _restore_selection(root_sha256, steps):
    if root_sha256 is not None and not isinstance(root_sha256, str):
        raise ParameterError("invalid selection-chain-v2 root")
    if not isinstance(steps, (list, tuple)) or not steps:
        raise ParameterError("selection-chain-v2 requires nonempty flat steps")
    required = {"kind", "policy", "parent_rows", "rows", "indices",
                "indices_sha256", "index_hash_algorithm"}
    parent = None
    for step in steps:
        if not isinstance(step, Mapping) or not required <= step.keys() or "parent" in step:
            raise ParameterError("invalid selection-chain-v2 step")
        facts = dict(step)
        indices = facts.pop("indices")
        parent = _SelectionNode(facts, indices, parent, root_sha256)
    return parent


def _plain(value):
    if isinstance(value, _SelectionNode):
        return value  # Its repr reports depth/digest without reading indices.
    if isinstance(value, Mapping):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_plain(item) for item in value]
    if isinstance(value, np.ndarray):
        # Selection retains immutable native index owners. Python row objects
        # are made only for an explicit metadata projection/export.
        return value.tolist()
    return value


def _freeze(value):
    kind = type(value)
    if kind is str or kind is int or kind is float or kind is bool or value is None:
        return value
    if kind is dict:
        return _FrozenMap(value)
    if kind is tuple or kind is list:
        return tuple([_freeze(x) for x in value])
    if isinstance(value, (_FrozenArray, _SelectionNode)):
        return value
    if isinstance(value, _PickledIndices):
        # Compatibility with the earlier int64-only copy/pickle wrapper.
        return _FrozenArray(value._array)
    if isinstance(value, _FrozenMap):
        return value
    if isinstance(value, Mapping):
        return _FrozenMap(value)
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(x) for x in value)
    if isinstance(value, np.ndarray):
        # A readonly flag does not prove immutable backing or descriptor state.
        # Only the private native selection factory may skip this byte snapshot.
        return _FrozenArray(value)
    return value


# Freezes a deferred Manifest once, whichever thread reads it first.
_FREEZING = RLock()


class Manifest(_FrozenMap):
    """Immutable call metadata with a versioned, lossless JSON codec."""
    __slots__ = ("_tree",)

    @classmethod
    def _deferred(cls, tree):
        """A computed call's metadata from its plain facts: dicts, tuples, lists and
        immutable scalars that this Manifest alone owns and nobody changes. Freezing
        them cannot fail and gives the same mapping whenever it runs, so it waits for
        the first read; the JSON text needs no freezing at all."""
        manifest = cls.__new__(cls)
        object.__setattr__(manifest, "_tree", tree)
        return manifest

    def __getattr__(self, name):
        if name == "_data":
            return self._freeze_tree()
        return super().__getattr__(name)

    def _freeze_tree(self):
        with _FREEZING:
            try:
                return object.__getattribute__(self, "_data")
            except AttributeError:
                pass
            tree = self._plain_tree()
            if tree is None:  # built without __init__ (copy and pickle protocols)
                raise AttributeError("_data")
            data = MappingProxyType({key: _freeze(value) for key, value in tree.items()})
            object.__setattr__(self, "_data", data)
            return data

    def _plain_tree(self):
        """The plain facts of a deferred Manifest, else None."""
        try:
            return object.__getattribute__(self, "_tree")
        except AttributeError:
            return None

    def to_json(self):
        tree = self._plain_tree()
        if tree is not None:
            return _manifest_codec.dumps(tree, frozen=True)
        return _manifest_codec.dumps(self)

    @classmethod
    def from_json(cls, source):
        decoded = _manifest_codec.loads(source)
        if not isinstance(decoded, Mapping):
            raise ParameterError("manifest JSON must contain a mapping")
        info = decoded.get("seqblaze")
        if (isinstance(info, Mapping) and info.get("manifest_codec", _manifest_codec.CODEC)
                not in _manifest_codec.SUPPORTED_CODECS):
            raise ParameterError("unsupported manifest codec version")
        return cls(decoded)


@dataclass(frozen=True, slots=True)
class FieldInfo:
    name: str
    dtype: str
    unit: str | None
    exactness: str = "exact"


# Unit or role of each numeric column (facade-result.md, "Result, Scalar and
# Tiles"). Boolean flags (passed, tie_at_k) and cigar strings have none.
_UNITS = {
    "dist": "edit operations", "best_dist": "edit operations",
    "i": "index", "j": "index", "argbest": "index",
    "n_best": "count", "count": "count", "n_ends": "count", "n_starts": "count",
    "rank": "competition rank", "offsets": "row offset",
    "b_end": "position", "b_start": "position", "ends": "position",
    # align (docs/wiki/align/gotoh-align.md §3)
    "score": "score", "best_score": "score", "score_fx": "score x 1000",
    "a_start": "position", "a_end": "position",
    "aln_len": "count", "n_identity": "count", "n_similarity": "count", "n_gaps": "count",
    # tm (thermo/tm.md §6); "i" there is the sequence index.
    "tm_c": "°C", "dg_kcal": "kcal/mol", "dg_dcal": "dcal/mol", "dh_kcal": "kcal/mol",
    "ds_cal_per_k": "cal/(mol*K)",
    # tm on Pairs (thermo/tm.md §13.5); RowStats carries best_tm_c only.
    "best_tm_c": "°C",
    # duplex (thermo/duplex-mfe.md §7); RowStats carries best_dg_kcal only.
    "best_dg_kcal": "kcal/mol",
    # duplex model="pf" (thermo/duplex-pf.md §9.1); lse_dg_kcal is a RowStats column.
    "dg_ens_kcal": "kcal/mol", "dg_ens_err_kcal": "kcal/mol", "best_dg_ens_kcal": "kcal/mol",
    "lse_dg_kcal": "kcal/mol",
}


def _locked(array):
    """A read-only array over memory no caller can reopen for writing.

    Selections and user-built columns move their data behind an immutable
    bytes owner, as numeric selections do, so setflags(write=True) cannot
    change offsets after an Arrow export has checked them.
    """
    array = np.asarray(array)
    if isinstance(array.base, bytes) and not array.flags.writeable:
        return array
    return np.frombuffer(np.ascontiguousarray(array).tobytes(), dtype=array.dtype)


class _RaggedColumn:
    """Arrow-layout variable-length column: int64 offsets, values, validity.

    Masked (non-passing) rows are null; validity is a read-only bool array.
    Element access returns one row; to_pylist() converts the whole column.
    """
    __slots__ = ("_offsets", "_values", "_valid", "_export")
    _kind = None

    def __init__(self, offsets, values, valid):
        self._set(_locked(offsets), _locked(values), _locked(valid))

    @classmethod
    def _native(cls, offsets, values, valid):
        """Wrap engine-owned read-only buffers without a copy."""
        column = cls.__new__(cls)
        column._set(np.asarray(offsets), np.asarray(values), np.asarray(valid))
        return column

    def _set(self, offsets, values, valid):
        object.__setattr__(self, "_offsets", offsets)
        object.__setattr__(self, "_values", values)
        object.__setattr__(self, "_valid", valid)
        object.__setattr__(self, "_export", None)

    def __setattr__(self, name, value):
        raise AttributeError(f"{type(self).__name__} is immutable")

    def __len__(self):
        return len(self._valid)

    @property
    def offsets(self):
        return self._offsets

    @property
    def validity(self):
        return self._valid

    @property
    def null_count(self):
        return int(len(self._valid) - np.count_nonzero(self._valid))

    def _row(self, k):
        k = _integer_index(k, len(self))
        if not self._valid[k]:
            return None
        return self._values[self._offsets[k]:self._offsets[k + 1]]

    def take(self, idx):
        """Rows idx as a new column: 1-D integers in [-n, n-1], negative values
        counting from the end, requested order and duplicates kept."""
        try:
            array = np.asarray(idx)
        except (TypeError, ValueError, OverflowError):
            raise ParameterError("take requires 1-D integer indices") from None
        if array.ndim != 1:
            raise ParameterError("take requires 1-D integer indices")
        if array.dtype.kind not in "iu":
            if array.size == 0 and isinstance(idx, (list, tuple)):
                array = np.empty(0, dtype=np.int64)
            else:
                raise ParameterError("take requires 1-D integer indices; use np.flatnonzero(mask) for a bool mask")
        from . import _core
        normalized = _column(_core._native_selection_indices(array, len(self)))
        _, offsets, values, valid = _core._native_selection_gather_column(self._parts(), normalized)
        values = np.asarray(values).view(self._values.dtype)
        return type(self)._native(offsets, values, valid)

    def _parts(self):
        return (self._kind, self._offsets, self._values, self._valid)

    def __arrow_c_array__(self, requested_schema=None):
        """Export this column through the Arrow C Data interface, zero-copy."""
        if requested_schema is not None:
            raise NotImplementedError("Arrow requested_schema conversion is not implemented yet")
        if self._export is None:
            from . import _core
            object.__setattr__(self, "_export", _core._native_arrow_export({"values": self._parts()}))
        return self._export.column(0)

    def __repr__(self):
        # Formats the first rows only; never converts the whole column.
        n = len(self)
        rows = (self[k] for k in range(min(_REPR_ROWS, n)))
        shown = ", ".join(repr(v.tolist() if isinstance(v, np.ndarray) else v) for v in rows)
        shown += ", ..." if n > _REPR_ROWS else ""
        return f"{type(self).__name__}([{shown}], len={n}, nulls={self.null_count})"


_REPR_ROWS = 6


def _integer_index(k, n):
    from .keep import _integer
    index = _integer(k)
    if index is None:
        raise TypeError(f"column indices must be integers, not {type(k).__name__}")
    if not -n <= index < n:
        raise IndexError("column index out of range")
    return index + n if index < 0 else index


class StringColumn(_RaggedColumn):
    """String field (D48 §5): int64 offsets plus a bytes buffer, Arrow layout."""
    __slots__ = ()
    _kind = "string"

    @property
    def data(self):
        return self._values

    def __getitem__(self, k):
        row = self._row(k)
        return None if row is None else row.tobytes().decode("ascii")

    def to_pylist(self):
        text = self._values.tobytes().decode("ascii")
        bounds = self._offsets.tolist()
        return [text[bounds[k]:bounds[k + 1]] if ok else None
                for k, ok in enumerate(self._valid.tolist())]


class BinaryColumn(_RaggedColumn):
    """Arrow large_binary column (format Z): int64 offsets plus one byte buffer,
    ``bytes`` items (docs/wiki/io/fasta-readers.md §6.2, P10).

    Reader columns are zero-copy views of the batch's Rust buffers and have no
    nulls; their validity array is created only when first read.
    """
    __slots__ = ()
    _kind = "binary"

    def __init__(self, offsets, values, valid=None):
        self._set(_locked(offsets), _locked(values), None if valid is None else _locked(valid))

    @classmethod
    def _native(cls, offsets, values, valid):
        """Wrap native buffers while retaining an absent all-valid bitmap."""
        column = cls.__new__(cls)
        column._set(np.asarray(offsets), np.asarray(values),
                    None if valid is None else np.asarray(valid))
        return column

    @classmethod
    def _reader(cls, offsets, values, export):
        """Wrap a reader batch's read-only buffers and its native Arrow export."""
        column = cls.__new__(cls)
        column._set(offsets, values, None)
        object.__setattr__(column, "_export", export)
        return column

    def __len__(self):
        return len(self._offsets) - 1

    @property
    def data(self):
        return self._values

    @property
    def validity(self):
        if self._valid is None:
            # Keep the existing explicit-read cache behind an immutable owner;
            # selection passes raw _valid and never calls this property.
            valid = np.frombuffer(b"\x01" * len(self), dtype=bool)
            object.__setattr__(self, "_valid", valid)
        return self._valid

    @property
    def null_count(self):
        if self._valid is None:
            return 0
        return int(len(self._valid) - np.count_nonzero(self._valid))

    def _row(self, k):
        k = _integer_index(k, len(self))
        if self._valid is not None and not self._valid[k]:
            return None
        return self._values[self._offsets[k]:self._offsets[k + 1]]

    def __getitem__(self, k):
        row = self._row(k)
        return None if row is None else row.tobytes()

    def to_pylist(self):
        data = self._values.tobytes()
        bounds = self._offsets.tolist()
        if self._valid is None:
            return [data[bounds[k]:bounds[k + 1]] for k in range(len(bounds) - 1)]
        return [data[bounds[k]:bounds[k + 1]] if ok else None
                for k, ok in enumerate(self._valid.tolist())]

    def take(self, idx):
        """Rows idx as a new column: 1-D integers in [-n, n-1], negative values
        counting from the end, requested order and duplicates kept."""
        try:
            array = np.asarray(idx)
        except (TypeError, ValueError, OverflowError):
            raise ParameterError("take requires 1-D integer indices") from None
        if array.ndim != 1:
            raise ParameterError("take requires 1-D integer indices")
        if array.dtype.kind not in "iu":
            if array.size == 0 and isinstance(idx, (list, tuple)):
                array = np.empty(0, dtype=np.int64)
            else:
                raise ParameterError("take requires 1-D integer indices; use np.flatnonzero(mask) for a bool mask")
        from . import _core
        normalized = _column(_core._native_selection_indices(array, len(self)))
        # Preserve an absent all-valid source bitmap: no N flags for M rows.
        parts = (self._kind, self._offsets, self._values, self._valid)
        _, offsets, values, valid = _core._native_selection_gather_column(parts, normalized)
        values = np.asarray(values).view(self._values.dtype)
        return type(self)._native(offsets, values, valid)

    def _parts(self):
        return (self._kind, self._offsets, self._values, self.validity)


class ListColumn(_RaggedColumn):
    """Arrow list<int64> field: int64 offsets plus an int64 values buffer."""
    __slots__ = ()
    _kind = "list"

    @property
    def values(self):
        return self._values

    def __getitem__(self, k):
        return self._row(k)

    def to_pylist(self):
        values = self._values.tolist()
        bounds = self._offsets.tolist()
        return [values[bounds[k]:bounds[k + 1]] if ok else None
                for k, ok in enumerate(self._valid.tolist())]


class BytesColumn(_RaggedColumn):
    """Byte-string field (Arrow large_binary): int64 offsets plus a bytes buffer.

    align's aligned rows under check_bases=False hold raw bytes (spec Q12).
    """
    __slots__ = ()
    _kind = "binary"

    @property
    def data(self):
        return self._values

    def __getitem__(self, k):
        row = self._row(k)
        return None if row is None else row.tobytes()

    def to_pylist(self):
        data = self._values.tobytes()
        bounds = self._offsets.tolist()
        return [data[bounds[k]:bounds[k + 1]] if ok else None
                for k, ok in enumerate(self._valid.tolist())]


# Kernel Results wrap native "binary" detail columns as BytesColumn (align); reader
# batches build BinaryColumn directly (fasta-readers.md §6.2). Both are large_binary.
_RAGGED = {"string": StringColumn, "list": ListColumn, "binary": BytesColumn}
_RAGGED_DTYPES = {StringColumn: "string", ListColumn: "list<int64>", BytesColumn: "binary",
                  BinaryColumn: "binary"}


# Native owner types already seen to hold host columns. The wrapping below
# depends on the owner's type alone, so a seen type takes the host path again.
_HOST_OWNERS = set()


def _column(owner):
    """Wrap one retained native column, on its original host or device."""
    if type(owner) in _HOST_OWNERS:
        return np.asarray(owner)
    device_column = _device_exports.wrap_column(owner)
    if device_column is not None:
        return device_column
    if isinstance(owner, tuple):
        kind, offsets, values, valid = owner
        return _RAGGED[kind]._native(offsets, values, valid)
    if type(owner).__module__ == "seqblaze._core":
        _HOST_OWNERS.add(type(owner))
    return np.asarray(owner)


# str(dtype) runs NumPy Python code; native columns use these dtype singletons.
_DTYPE_NAMES = {id(dtype): (dtype, str(dtype))
                for dtype in map(np.dtype, ("int32", "int64", "float64", "bool"))}


def _dtype_name(dtype):
    known = _DTYPE_NAMES.get(id(dtype))
    return known[1] if known is not None and known[0] is dtype else str(dtype)


# FieldInfo of computed columns by (kernel, name, dtype or column class): immutable,
# from the fixed registry, over the finite field set of the registered kernels.
_FIELD_INFOS = {}


def _computed_field_info(name, array, kernel):
    key = (kernel, name, type(array) if isinstance(array, _RaggedColumn) else _dtype_name(array.dtype))
    info = _FIELD_INFOS.get(key)
    if info is None:
        info = _FIELD_INFOS.setdefault(key, _field_info(name, array, kernel))
    return info


def _field_info(name, array, kernel):
    fields = _REGISTRY.get(kernel, {}).get("fields", {})
    definition = fields.get(name)
    if definition is None and name.startswith("best_"):
        # RowStats' best_<primary> holds the primary's values: same exactness
        # (duplex-pf.md §9.1). Not a registry field, so not selectable by name.
        definition = fields.get(name[len("best_"):])
    exactness = (definition or {}).get("exactness", "exact")
    if isinstance(array, _RaggedColumn):
        return FieldInfo(name, _RAGGED_DTYPES[type(array)], _UNITS.get(name), exactness)
    return FieldInfo(name, _dtype_name(array.dtype), _UNITS.get(name), exactness)


def _unknown(name, columns, kernel="hamming"):
    similar = _fp.close_matches(name, columns, n=1)
    hint = f"; did you mean {similar[0]!r}?" if similar else ""
    return f"unknown {kernel.capitalize()} column {name!r}{hint}; available: {', '.join(columns)}"


@dataclass(frozen=True, slots=True, init=False, eq=False)
class Result:
    kernel: str
    layout: str
    shape: tuple[int, int]
    fields: tuple[FieldInfo, ...]
    manifest: Manifest
    _columns: Mapping
    _input_pairs: object
    _pairs_cache: object
    _arrow_export: object
    _replay: object
    _pair_indices: object

    def __init__(self, layout, columns, shape, pairs, manifest, kernel="hamming", *, replay=None,
                 pair_indices=None):
        self._assemble(layout, columns, shape, pairs, manifest, kernel, replay, pair_indices)

    @classmethod
    def _computed(cls, layout, columns, shape, pairs, facts, kernel, replay, pair_indices=None):
        """A facade call's Result. ``facts`` is the call's plain manifest tree, which
        the new Result alone owns (``Manifest._deferred``)."""
        result = cls.__new__(cls)
        result._assemble(layout, columns, shape, pairs, facts, kernel, replay, pair_indices,
                         owned=True)
        return result

    def _assemble(self, layout, columns, shape, pairs, manifest, kernel, replay, pair_indices,
                  owned=False):
        info = _computed_field_info if owned else _field_info
        fields = tuple([info(n, a, kernel) for n, a in columns.items()])
        if pair_indices is not None:
            if not isinstance(pair_indices, Mapping) or set(pair_indices) != {"i", "j"}:
                raise ParameterError("call-time pair indices require i and j owners")
            pair_indices = MappingProxyType(dict(pair_indices))
        # User-built Results may carry arbitrary metadata. Computed Results
        # receive the completed layout and lineage facts without output scans.
        if replay is not None:
            manifest = _provenance.complete_manifest(manifest, layout=layout, shape=shape,
                                                     fields=fields, columns=columns,
                                                     root=replay, owned=owned)
        for name, value in (("kernel", kernel), ("layout", layout), ("shape", shape),
                            ("fields", fields),
                            ("manifest", manifest), ("_columns", MappingProxyType(dict(columns))),
                            ("_input_pairs", pairs), ("_pairs_cache", None), ("_arrow_export", None),
                            ("_replay", replay), ("_pair_indices", pair_indices)):
            object.__setattr__(self, name, value)

    def __len__(self):
        if self.layout == "scalar":
            return 1
        if self.layout == "topk":
            return len(self._columns["i"])
        return len(next(iter(self._columns.values())))

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        columns = object.__getattribute__(self, "_columns")
        if name not in columns:
            raise AttributeError(_unknown(name, columns, self.kernel))
        return self._value(columns[name])

    def _value(self, value):
        if self.layout != "scalar" or isinstance(value, ReadOnlyDeviceColumn):
            return value
        return value[0] if isinstance(value, _RaggedColumn) else value[()]

    def __getitem__(self, key):
        if isinstance(key, str):
            if key not in self._columns:
                raise KeyError(_unknown(key, self._columns, self.kernel))
            return self._value(self._columns[key])
        return self._select(key, allow_mask=True)

    @property
    def pairs(self):
        """The pair space of the rows; None for a unary kernel (tm), whose rows are sequences."""
        if self._input_pairs is None:
            return None
        if self.layout not in ("hits", "topk", "rowstats"):
            return self._input_pairs
        if self._pairs_cache is None:
            from .pairs import index
            if self.layout == "rowstats":
                from . import _core
                # Native integer filtering performs no floating-point work.
                # Both index columns retain immutable native owners, as sink
                # columns do, without NumPy arithmetic or a Python row loop.
                if _device_exports.has_device_columns(self):
                    value = _device_selection.rowstats_pairs(self)
                    object.__setattr__(self, "_pairs_cache", value)
                    return value
                columns = _core._native_rowstats_pairs(self._columns["argbest"])
                i, j = _column(columns["i"]), _column(columns["j"])
            else:
                i, j = self._columns["i"], self._columns["j"]
            value = index(self._input_pairs.a, i, j, table_b=self._input_pairs.b)
            object.__setattr__(self, "_pairs_cache", value)
        return self._pairs_cache

    def to_numpy(self):
        """Return host columns; device columns are explicitly copied to host.

        Host fields keep their retained readonly views. Variable-length fields
        use the existing StringColumn/ListColumn/BytesColumn host layout.
        """
        return _device_exports.to_numpy(self)

    def _check_tabular_layout(self):
        return _result_export.check(self)

    def __arrow_c_stream__(self, requested_schema=None):
        """Export a fresh native stream retaining immutable flat or nested buffers."""
        return _result_export.arrow_stream(self, requested_schema)

    def to_arrow(self):
        """Return an Arrow Table through the installed optional PyArrow consumer."""
        return _result_export.to_arrow(self)

    def to_pandas(self):
        """Return an independently mutable pandas frame, with an explicit copy."""
        return _result_export.to_pandas(self)

    def to_polars(self):
        """Import a fresh native Arrow stream directly into installed Polars."""
        return _result_export.to_polars(self)

    def to_cupy(self):
        """Return independent device copies, preserving each field's shape."""
        return _device_exports.to_cupy(self)

    def to_cudf(self):
        """Return an independent device table with the Result's table layout."""
        return _device_exports.to_cudf(self)

    def to_parquet(self, path):
        _manifest_codec.write_parquet(self, path)

    def take(self, idx):
        return self._select(idx, allow_mask=False)

    def _select(self, key, *, allow_mask):
        if _device_exports.has_device_columns(self):
            return _device_selection.select(self, key, allow_mask=allow_mask)
        if self.layout not in ("each", "hits", "topk"):
            raise NotImplementedError(f"selection for {self.layout} results is not implemented yet")
        try:
            idx = np.asarray(key)
        except (TypeError, ValueError, OverflowError):
            raise ParameterError("selection requires a 1-D integer array or bool mask") from None
        if idx.ndim != 1:
            raise ParameterError("selection requires a 1-D integer array or bool mask")
        masked = idx.dtype.kind == "b" and allow_mask
        if masked:
            if len(idx) != len(self):
                raise ParameterError(f"selection mask has {len(idx)} rows; expected {len(self)}")
        elif idx.dtype.kind not in "iu":
            if idx.size == 0 and isinstance(key, (list, tuple)):
                idx = np.empty(0, dtype=np.int64)
            else:
                raise ParameterError("take requires 1-D integer indices; use r[mask] for bool masks")
        from . import _core
        # Each is an ordered gather. Hits and TopK are sets of original entry
        # positions: normalize negatives first, then sort and deduplicate.
        # Masks already emit increasing original positions on every layout.
        canonical = self.layout in ("hits", "topk")
        normalized_owner = _core._native_selection_indices(
            idx, len(self), allow_mask=allow_mask, canonical=canonical)
        idx = _column(normalized_owner)
        k = None
        if self.layout == "topk":
            from .keep import _integer
            try:
                k = _integer(self.manifest["model"]["keep"]["k"])
            except (KeyError, TypeError):
                pass
            if k is None or k < 1:
                raise ParameterError("TopK selection requires the original positive model.keep.k")
        # BinaryColumn._parts() expands absent validity to N source flags.
        # The native gather accepts None for an all-valid source instead.
        # TopK offsets/flags are row metadata, never entry columns to gather.
        packed = {name: ((array._kind, array._offsets, array._values, array._valid)
                         if isinstance(array, _RaggedColumn) else array)
                  for name, array in self._columns.items()
                  if self.layout != "topk" or name not in ("offsets", "tie_at_k")}
        columns = {}
        for name, owner in _core._native_selection_gather(packed, idx).items():
            source = self._columns[name]
            if isinstance(source, _RaggedColumn):
                _, offsets, values, valid = owner
                values = np.asarray(values).view(source._values.dtype)
                columns[name] = type(source)._native(offsets, values, valid)
            else:
                columns[name] = _column(owner).view(source.dtype)
        tied_rows = None
        if self.layout == "topk":
            structure = _core._native_topk_structure(columns["i"], self.shape[0], k)
            for name in ("offsets", "tie_at_k"):
                columns[name] = _column(structure[name])
            tied_rows = structure["tie_rows"]
            # Retain the parent's public field order, including row metadata.
            columns = {name: columns[name] for name in self._columns}
        from .pairs import index
        original = self._input_pairs
        if original is None:
            selected_pairs = None  # unary rows (tm): no pair space to select
        elif self.layout == "each":
            if original.kind == "zip":
                i = j = idx
            elif original.kind == "one_to_many":
                i, j = np.zeros(len(idx), dtype=np.int64), idx
            else:
                pair_columns = _core._native_selection_gather(
                    {"i": original.i, "j": original.j}, idx)
                i, j = _column(pair_columns["i"]), _column(pair_columns["j"])
            selected_pairs = index(original.a, i, j, table_b=original.b)
        else:
            selected_pairs = original
        metadata = dict(self.manifest)
        if tied_rows is not None:
            metadata["flags"] = dict(self.manifest.get("flags", {}), tie_at_k=tied_rows)
        # Counts retain the original computation. Owned normalized positions
        # capture every selection step without borrowing the mutable selector
        # or constructing per-entry Python objects on the selection path.
        selection = {"parent_rows": len(self), "rows": len(idx),
                     "kind": "mask" if masked else "take",
                     "policy": "stable" if masked else "canonical" if canonical else "requested"}
        if self.layout == "topk":
            selection.update(scope="entry", rank_basis="original")
        root_id = self.manifest.get("call", {}).get("repro", {}).get("root_sha256")
        metadata["selection"] = _provenance.selection_node(
            selection, normalized_owner, idx, self.manifest.get("selection"), root_id,
            threads=self.manifest.get("execution", {}).get("threads", 1))
        return Result(self.layout, columns, self.shape, selected_pairs, Manifest(metadata),
                      kernel=self.kernel, replay=self._replay)

    def cite(self, format="text"):
        """Return method citations as text or BibTeX, without recomputing results."""
        from ._registry import _format_citations
        return _format_citations(self.kernel, self.manifest.citations, format)

    def repro(self):
        return _provenance.render(self)

    def __repr__(self):
        units = ", ".join(f"{f.name}:{f.dtype}" + (f" [{f.unit}]" if f.unit else "") for f in self.fields)
        execution = self.manifest.execution
        counts = dict(execution.counts)
        model = self.manifest["model"].get("mode") if "model" in self.manifest else None
        if self.kernel == "levenshtein":
            model = f"Levenshtein {model}"
        elif self.kernel == "align":
            compat = self.manifest["model"].get("compat")
            model = compat["alias"] if compat else f"Align {model}"
        elif self.kernel in ("tm", "duplex", "hairpin"):
            model = f"{self.kernel} {self.manifest['model']['params']['name']}"
        else:
            model = self.kernel.capitalize()
        return (f"seqblaze.Result(kernel={self.kernel!r}, model={model!r}, layout={self.layout!r}, "
                f"rows={len(self)}, fields=({units}), strategy={execution.strategy['effective']!r}, "
                f"counts={counts}, device={execution['device']!r}, research use only)")

    def _repr_html_(self):
        return f"<pre>{escape(repr(self))}</pre>"


@dataclass(frozen=True, slots=True, eq=False)
class Tile:
    rows: slice
    cols: slice
    shape: tuple[int, int]
    _columns: Mapping
    manifest: Manifest | None = None
    _arrow_export: object = field(default=None, init=False, repr=False, compare=False)

    def __post_init__(self):
        if self.manifest is None:
            object.__setattr__(self, "manifest", _provenance.tile_manifest(
                None, self.rows, self.cols, self.shape, self._columns))

    def __getattr__(self, name):
        columns = object.__getattribute__(self, "_columns")
        if name in columns:
            return columns[name]
        raise AttributeError(_unknown(name, columns, "tile"))

    def __getitem__(self, name):
        if name not in self._columns:
            raise KeyError(_unknown(name, self._columns, "tile"))
        return self._columns[name]

    def to_numpy(self):
        return _device_exports.to_numpy(self)

    def to_cupy(self):
        return _device_exports.to_cupy(self)

    def to_cudf(self):
        return _device_exports.to_cudf(self)


    def to_parquet(self, path):
        _manifest_codec.write_parquet(self, path)

    def __arrow_c_stream__(self, requested_schema=None):
        return _result_export.arrow_stream(self, requested_schema)

    def to_arrow(self):
        return _result_export.to_arrow(self)


    def to_pandas(self):
        return _result_export.to_pandas(self)

    def to_polars(self):
        return _result_export.to_polars(self)


class TileIterator:
    """One native tile per next(); closing releases input owners immediately."""
    __slots__ = ("_source", "_manifest_factory")

    def __init__(self, source, manifest_factory):
        self._source = source
        self._manifest_factory = manifest_factory

    def __iter__(self):
        return self

    def __next__(self):
        try:
            tile = next(self._source)
            columns = {name: _column(owner).reshape(tile["shape"])
                       for name, owner in tile["columns"].items()}
            rows, cols = slice(*tile["rows"]), slice(*tile["cols"])
            metadata = _provenance.tile_manifest(self.manifest, rows, cols, tile["shape"],
                                                 columns, self.shape)
            return Tile(rows, cols, tile["shape"], MappingProxyType(columns), manifest=metadata)
        except StopIteration:
            raise
        except BaseException:
            self.close()
            raise

    @property
    def shape(self):
        return self._source.shape

    @property
    def manifest(self):
        return _provenance.stream_manifest(self._manifest_factory(self._source.report),
                                           self.shape)

    def close(self):
        self._source.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
