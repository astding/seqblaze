"""Selection over immutable device Result buffers, with no implicit host read.

The native bridge normalizes/canonicalizes selectors, gathers numeric/ragged
columns and rebuilds TopK rows. Complete normalized positions remain resident;
ordinary manifest/scientific facts use their backend-neutral Merkle digest.
"""
from collections.abc import Mapping

import numpy as np

from . import _core, _manifest_codec
from ._device import ReadOnlyDeviceColumn, _DeviceRaggedColumn
from ._device_exports import wrap_column
from ._device_inputs import _DeviceIndex, index_device, is_device_input
from ._errors import ParameterError


class _DeviceSelectionIndices(ReadOnlyDeviceColumn):
    """Owned normalized positions; full materialization is an explicit export."""
    __slots__ = ("__sha256",)

    def __init__(self, owner, digest):
        super().__init__(owner)
        if self.ndim != 1 or self.dtype != np.dtype("int64"):
            raise ParameterError("normalized device selection indices require 1-D int64 storage")
        if (type(digest) is not str or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)):
            raise ParameterError("normalized device selection indices require their logical digest")
        object.__setattr__(self, "_DeviceSelectionIndices__sha256", digest)

    @property
    def _selection_sha256(self):
        return self.__sha256

    def _manifest_selection_array(self):
        # The shared codec/repro caller invokes this ONLY for explicit JSON or
        # repro. Scientific/lineage hashes project this owner to its digest.
        return self.to_numpy()

    def __copy__(self):
        return self

    def __deepcopy__(self, memo):
        memo[id(self)] = self
        return self

    def __reduce__(self):
        raise TypeError("device selection metadata does not implicitly copy indices to host; "
                        "use manifest.to_json() or Result.repro() for an explicit export")


def _owner(column):
    return object.__getattribute__(column, "_ReadOnlyDeviceColumn__owner")


def _packed(column):
    if isinstance(column, _DeviceRaggedColumn):
        valid = column.validity
        values = column.values if column._kind == "list" else column.data
        return (column._kind, _owner(column.offsets), _owner(values),
                None if valid is None else _owner(valid))
    if isinstance(column, ReadOnlyDeviceColumn):
        return _owner(column)
    raise ParameterError("device selection requires every result field to have an immutable device owner")


def _selector(key, rows, allow_mask):
    if isinstance(key, _DeviceRaggedColumn):
        raise ParameterError("selection requires a 1-D integer array or bool mask")
    if isinstance(key, ReadOnlyDeviceColumn):
        if key.ndim != 1:
            raise ParameterError("selection requires a 1-D integer array or bool mask")
        # Preserve the public reshape metadata. Native verifies that this
        # contiguous view covers the actual immutable owner's complete span.
        return _owner(key), False, False, key.size
    if isinstance(key, getattr(_core, "_DeviceColumn", ())):
        return key, False, False, None
    if isinstance(key, _DeviceIndex):
        return key._native, False, False, None
    if is_device_input(key):
        return key, False, False, None
    try:
        array = np.asarray(key)
    except (TypeError, ValueError, OverflowError):
        raise ParameterError("selection requires a 1-D integer array or bool mask") from None
    if array.ndim != 1:
        raise ParameterError("selection requires a 1-D integer array or bool mask")
    masked = array.dtype.kind == "b" and allow_mask
    if masked:
        if len(array) != rows:
            raise ParameterError(f"selection mask has {len(array)} rows; expected {rows}")
    elif array.dtype.kind not in "iu":
        if array.size == 0 and isinstance(key, (list, tuple)):
            array = np.empty(0, dtype=np.int64)
        else:
            raise ParameterError("take requires 1-D integer indices; use r[mask] for bool masks")
    # Reuse the existing host integer adapter for host-origin selectors only.
    # Normalized positions are copied H2D; result buffers never visit NumPy.
    from ._result import _column
    normalized = _column(_core._native_selection_indices(array, rows, allow_mask=allow_mask))
    return normalized.tobytes(order="C"), True, masked, None


def select(value, key, *, allow_mask):
    """Return the same Result layout with fresh immutable device field owners."""
    if value.layout not in ("each", "hits", "topk"):
        raise NotImplementedError(f"selection for {value.layout} results is not implemented yet")
    canonical = value.layout in ("hits", "topk")
    selector, host_normalized, host_masked, selector_rows = _selector(key, len(value), allow_mask)
    k = None
    if value.layout == "topk":
        from .keep import _integer
        try:
            k = _integer(value.manifest["model"]["keep"]["k"])
        except (KeyError, TypeError):
            pass
        if k is None or k < 1:
            raise ParameterError("TopK selection requires the original positive model.keep.k")
    packed = {name: _packed(column) for name, column in value._columns.items()
              if value.layout != "topk" or name not in ("offsets", "tie_at_k")}
    original = value._input_pairs
    pair_kind = None
    pair_columns = None
    if value.layout == "each" and original is not None:
        pair_kind = original.kind
        if pair_kind == "index":
            source = value._pair_indices
            if not isinstance(source, Mapping) or set(source) != {"i", "j"}:
                raise ParameterError("Each index selection requires retained call-time device pair indices")
            pair_columns = source["i"], source["j"]
        elif pair_kind not in ("zip", "one_to_many"):
            # The compute entry rejects Each on implicit cross/within shapes.
            # Selection does not expand that public call-shape vocabulary.
            raise ParameterError("Each device selection requires zip, index or one-to-many pairs")
    gathered = _core._native_device_selection(
        packed, selector, len(value), allow_mask=allow_mask, canonical=canonical,
        topk_rows=value.shape[0] if value.layout == "topk" else None, k=k,
        host_normalized=host_normalized, host_masked=host_masked,
        pair_kind=pair_kind, pair_columns=pair_columns, selector_rows=selector_rows)
    if gathered.get("index_hash_algorithm") != _manifest_codec.INPUT_HASH:
        raise ParameterError("device selection returned an unsupported index hash algorithm")
    native_columns = gathered["columns"]
    columns = {name: wrap_column(native_columns[name]) for name in value._columns}
    if any(column is None for column in columns.values()):
        raise ParameterError("device selection returned an unknown column owner")
    indices = _DeviceSelectionIndices(gathered["indices"], gathered["indices_sha256"])
    native_pairs = gathered["pair_indices"]
    pair_indices = None
    if original is None:
        selected_pairs = None
    elif value.layout == "each":
        if native_pairs is None:
            raise ParameterError("device selection did not return its selected pair mapping")
        i, j = native_pairs
        from .pairs import index
        selected_pairs = index(original.a, index_device(i, owned=True, label="i"),
                               index_device(j, owned=True, label="j"), table_b=original.b)
        pair_indices = {"i": i, "j": j}
    else:
        selected_pairs = original
    metadata = dict(value.manifest)
    tied_rows = gathered["tie_rows"]
    if tied_rows is not None:
        metadata["flags"] = dict(value.manifest.get("flags", {}), tie_at_k=tied_rows)
    masked = gathered["masked"]
    selection = {
        "parent_rows": len(value), "rows": len(indices),
        "kind": "mask" if masked else "take",
        "policy": "stable" if masked else "canonical" if canonical else "requested",
        "indices_sha256": gathered["indices_sha256"],
        "index_hash_algorithm": gathered["index_hash_algorithm"],
    }
    if value.layout == "topk":
        selection.update(scope="entry", rank_basis="original")
    from ._result import Manifest, Result, _SelectionNode
    root_id = value.manifest.get("call", {}).get("repro", {}).get("root_sha256")
    metadata["selection"] = _SelectionNode(
        selection, indices, value.manifest.get("selection"), root_id)
    return Result(value.layout, columns, value.shape, selected_pairs, Manifest(metadata),
                  kernel=value.kernel, replay=value._replay, pair_indices=pair_indices)


def rowstats_pairs(value):
    """Build the lazy D87 pair view from device argbest without a host scan."""
    original = value._input_pairs
    if original is None:
        return None
    source = value._columns["argbest"]
    if not isinstance(source, ReadOnlyDeviceColumn) or isinstance(source, _DeviceRaggedColumn):
        raise ParameterError("device RowStats pairs require an immutable numeric argbest owner")
    columns = _core._native_device_rowstats_pairs(_owner(source))
    from .pairs import index
    return index(original.a, index_device(columns["i"], owned=True, label="i"),
                 index_device(columns["j"], owned=True, label="j"), table_b=original.b)
