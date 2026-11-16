"""Optional device consumers over explicit independent Result buffer copies.

Host fields use one bulk H2D per buffer; device fields use D2D only.
CuPy retains each numeric field's original dtype and shape. Table consumers
use the same one-row TopK/Matrix/Tile mapping as the host adapters. No payload
or offsets pass through NumPy/Python rows. pylibcudf's native bools_to_mask
returns only the exact null-count control scalar while packing flags on GPU.

Consumer APIs: pylibcudf Column/from_cuda_array_interface and gpumemoryview;
cuDF DataFrame.from_pylibcudf and TableWithMetadata. Imports remain optional.
"""
from types import MappingProxyType

import numpy as np

from . import _core, _fp
# The module object, not its names: _result imports this module first.
from . import _result

from ._device import DeviceRaggedCopy, ReadOnlyDeviceColumn, _DeviceRaggedColumn
from ._errors import ParameterError

_FLAT = ("each", "hits", "rowstats", "scalar")
_I32_MAX = 2**31 - 1


def wrap_column(owner):
    """Return a retained device proxy, or None for an ordinary host column."""
    native = getattr(_core, "_DeviceColumn", ())
    if isinstance(owner, ReadOnlyDeviceColumn):
        return owner
    if isinstance(owner, native):
        return ReadOnlyDeviceColumn(owner)
    if (isinstance(owner, tuple) and len(owner) == 4
            and isinstance(owner[1], native)):
        return _DeviceRaggedColumn(*owner)
    return None


def to_numpy(value):
    """Host columns retain their old views; device reads are explicit copies.

    Every device buffer of the value goes through one native batch export
    (per-call cost S8): each column's host bytes are reserved first, then the
    copies share one wait. A single column's own to_numpy() keeps its path.
    """
    columns = value._columns
    device = [column for column in columns.values() if isinstance(column, ReadOnlyDeviceColumn)]
    if not device:
        return dict(columns)
    from . import _core
    parts = [part for column in device for part in column._export_parts()]
    blobs = iter(_core._device_host_bytes_batch([part._export_owner() for part in parts]))
    out = {}
    for name, column in columns.items():
        if isinstance(column, ReadOnlyDeviceColumn):
            out[name] = column._from_export([next(blobs) for _ in column._export_parts()])
        else:
            out[name] = column
    return out


def has_device_columns(value):
    """Inspect ownership only; never read device values or import a consumer."""
    return any(isinstance(column, ReadOnlyDeviceColumn) for column in value._columns.values())


class _HostExportView:
    """Actual readonly host buffers and retained metadata for table consumers.

    A Result/Tile caches this view in its existing private Arrow-owner slot;
    the view in turn owns the existing native Arrow exporter when needed.
    It does not retain the source Result, so this cache introduces no cycle.
    """
    __slots__ = ("layout", "shape", "rows", "cols", "manifest", "_columns", "_arrow_export")

    def __init__(self, value, columns):
        layout = _layout(value)
        facts = {"layout": layout, "shape": value.shape,
                 "rows": value.rows if layout == "tile" else None,
                 "cols": value.cols if layout == "tile" else None,
                 "manifest": getattr(value, "manifest", None),
                 "_columns": MappingProxyType(columns), "_arrow_export": None}
        for name, item in facts.items():
            object.__setattr__(self, name, item)

    def __setattr__(self, name, value):
        raise AttributeError("host table snapshots are immutable")

    def __arrow_c_stream__(self, requested_schema=None):
        from . import _result_export
        return _result_export.arrow_stream(self, requested_schema)

    def stream(self):
        from . import _result_export
        return _result_export.arrow_stream(self)


def host_view(value):
    """Explicit bulk D2H, cached for later host table exports of this Result.

    CPU inputs are returned unchanged. Each device numeric buffer is copied
    through its native synchronized byte export; ragged fields copy complete
    offset/payload/validity buffers, then use the ordinary host column class.
    No GPU row is indexed and no proxy goes through np.asarray. Missing
    optional consumers must be checked by the caller before invoking this.
    """
    if not has_device_columns(value):
        return value
    cached = value._arrow_export
    if isinstance(cached, _HostExportView):
        return cached
    if cached is not None:
        raise ParameterError("device host export has an incompatible cached owner")
    # Publish only a complete snapshot. An interrupted/failed D2H leaves the
    # Result's original metadata/cache intact and releases partial host copies.
    view = _HostExportView(value, to_numpy(value))
    # Native copies release the GIL. Concurrent first callers may each copy,
    # but all reuse the first complete published snapshot and its Arrow cache.
    cached = value._arrow_export
    if isinstance(cached, _HostExportView):
        return cached
    if cached is not None:
        raise ParameterError("device host export has an incompatible cached owner")
    object.__setattr__(value, "_arrow_export", view)
    return view


def _is_ragged(column):
    return isinstance(column, (_DeviceRaggedColumn, _result._RaggedColumn))


def _size(column):
    return len(column) if _is_ragged(column) else column.size


def _ndim(column):
    return 1 if _is_ragged(column) else column.ndim


def _export_columns(value):
    from ._result import _RaggedColumn
    columns = value._columns
    if not columns or any(not isinstance(column, (np.ndarray, ReadOnlyDeviceColumn, _RaggedColumn))
                          for column in columns.values()):
        raise ParameterError("Device export requires numeric or variable-length Result fields")
    ordinals = {column._ordinal for column in columns.values()
                if isinstance(column, ReadOnlyDeviceColumn)}
    if len(ordinals) > 1:
        raise ParameterError("Device result columns must share one CUDA device")
    # Host-only exports use the consumer's current device. A mixed user-built
    # Result uploads host fields beside its already-resident device fields.
    return columns, next(iter(ordinals)) if ordinals else None


def _cupy():
    try:
        with _fp.bracket():
            import cupy as cp
    except ImportError as error:
        raise ImportError("Device export requires CuPy; install a CuPy build matching your CUDA environment") from error
    return cp


def _copy_numeric(column, cp):
    if isinstance(column, ReadOnlyDeviceColumn):
        return column.to_cupy()
    # Upload directly into the allocation returned to the consumer. Holding
    # the host input until stream completion also covers pinned host buffers.
    copied = cp.array(column, copy=True, order="C")
    cp.cuda.get_current_stream().synchronize()
    return copied


def to_cupy(value):
    """Independent arrays or ragged buffer copies, retaining numeric shapes."""
    columns, ordinal = _export_columns(value)
    cp = _cupy()
    with cp.cuda.Device() if ordinal is None else cp.cuda.Device(ordinal):
        return {name: (DeviceRaggedCopy._from_copied_buffers(_copy_ragged_buffers(column, cp))
                       if _is_ragged(column) else _copy_numeric(column, cp))
                for name, column in columns.items()}


def _layout(value):
    from ._result import Tile
    return "tile" if isinstance(value, Tile) else value.layout


def _check(value, columns):
    layout = _layout(value)
    ragged = _is_ragged
    if layout in _FLAT:
        numeric = [column for column in columns.values() if not ragged(column)]
        sizes = {_size(column) for column in columns.values()}
        ndim = 0 if layout == "scalar" else 1
        if (len(sizes) != 1 or (layout == "scalar" and sizes != {1})
                or any(column.ndim != ndim for column in numeric)):
            raise ParameterError("Device table fields have inconsistent flat dimensions")
    elif layout in ("topk", "matrix", "tile"):
        if (len(value.shape) != 2 or any(type(n) is not int or not 0 <= n < 2**63
                                       for n in value.shape)):
            raise ParameterError("Device table export requires a two-dimensional shape")
        if layout == "topk":
            required = {"i": "int64", "j": "int64", "rank": "int32",
                        "offsets": "int64", "tie_at_k": "bool"}
            for name, dtype in required.items():
                column = columns.get(name)
                if column is None or column.ndim != 1 or str(column.dtype) != dtype:
                    raise ParameterError(f"TopK device export requires a 1-D {dtype} {name} field")
            entries = columns["i"].size
            for name, column in columns.items():
                expected = (value.shape[0] + 1 if name == "offsets" else
                            value.shape[0] if name == "tie_at_k" else entries)
                if _ndim(column) != 1 or _size(column) != expected:
                    raise ParameterError("TopK entries, offsets and row flags have inconsistent lengths")
        elif any(ragged(column) or column.shape != value.shape or column.ndim != 2
                 for column in columns.values()):
            raise ParameterError("Matrix and Tile device fields must match their two-dimensional shape")
        if layout == "tile":
            for name, size in zip(("rows", "cols"), value.shape):
                span = getattr(value, name)
                if (not isinstance(span, slice) or span.step not in (None, 1)
                        or type(span.start) is not int or type(span.stop) is not int
                        or not 0 <= span.start <= span.stop < 2**63
                        or span.stop - span.start != size):
                    raise ParameterError("Tile export requires matching half-open global coordinates")
    else:
        raise NotImplementedError(f"Device table mapping for {layout} is not implemented")
    # libcudf size_type and LIST offsets are int32. Reject metadata overflow
    # before copying; original trusted result offsets are bounded by payload.
    for column in columns.values():
        if _size(column) > _I32_MAX:
            raise ParameterError("cuDF column rows exceed its int32 size limit")
        if ragged(column):
            payload_size = (column._copy_value_size if isinstance(column, _DeviceRaggedColumn)
                            else column._values.size)
            if (column.offsets.size > _I32_MAX
                    or (column._kind != "string" and payload_size > _I32_MAX)):
                raise ParameterError("cuDF variable-length column exceeds its int32 child limit")
        elif str(column.dtype) not in ("int32", "int64", "float64", "bool"):
            raise ParameterError("Unsupported device result dtype for cuDF")
    if layout in ("matrix", "tile") and value.shape[0] >= _I32_MAX:
        raise ParameterError("cuDF list offsets exceed its int32 size limit")
    return layout


def _consumer():
    hint = "Device table export requires compatible CuPy, cuDF and pylibcudf installations"
    try:
        with _fp.bracket():
            import cupy as cp
            import cudf
            import pylibcudf as plc
    except ImportError as error:
        raise ImportError(hint) from error
    if not (hasattr(plc.Column, "from_cuda_array_interface")
            and callable(getattr(plc, "gpumemoryview", None))
            and hasattr(cudf.DataFrame, "from_pylibcudf")
            and callable(getattr(plc.io, "TableWithMetadata", None))):
        raise ImportError(hint + " providing Column.from_cuda_array_interface, "
                          "TableWithMetadata and DataFrame.from_pylibcudf")
    return cp, cudf, plc


def _primitive(array, cp, plc):
    # A consumer may use a different default stream (including PTDS). Finish
    # our copied/generated input first; no payload moves to the host.
    cp.cuda.get_current_stream().synchronize()
    return plc.Column.from_cuda_array_interface(array.reshape(-1))


def _list(offsets, child, size, cp, plc, mask=None, null_count=0):
    return plc.Column(plc.DataType(plc.TypeId.LIST), size, None, mask,
                      null_count, 0, [_primitive(offsets, cp, plc), child])


def _outer(child, size, cp, plc):
    return _list(cp.asarray((0, size), dtype=cp.int32), child, 1, cp, plc)


def _host_ragged_parts(column):
    # Host columns can be explicitly user-built. Check their bulk buffers
    # before handing offsets to a GPU consumer; native device columns have
    # already been validated by their producer and never download for this.
    kind, offsets, values, valid = column._kind, column._offsets, column._values, column._valid
    rows = len(column)
    if (kind not in ("string", "binary", "list") or offsets.dtype != np.dtype("int64")
            or offsets.ndim != 1 or offsets.size != rows + 1 or values.ndim != 1
            or values.dtype != np.dtype("int64" if kind == "list" else "uint8")
            or (valid is not None and (valid.dtype != np.dtype("bool") or valid.shape != (rows,)))):
        raise ParameterError("Inconsistent host variable-length result buffers")
    if (offsets[0] < 0 or offsets[-1] > values.size
            or np.any(offsets[1:] < offsets[:-1])):
        raise ParameterError("Variable-length offsets must be nonnegative, nondecreasing and in bounds")
    return kind, offsets, values, valid


def _copy_ragged_buffers(column, cp, *, list_offsets=False):
    """One bulk copy per buffer; wrapping these copies needs no new transfer."""
    if isinstance(column, _DeviceRaggedColumn):
        buffers = column._copy_buffers()
        if list_offsets and buffers["kind"] != "string":
            buffers["offsets"] = buffers["offsets"].astype(cp.int32)
        return buffers
    kind, offsets, values, valid = _host_ragged_parts(column)
    # libcudf LIST needs int32 offsets. Cast on host before the single H2D,
    # rather than uploading int64 and allocating a redundant device copy.
    if list_offsets and kind != "string":
        offsets = offsets.astype(np.int32, copy=False)
    return {"kind": kind, "offsets": _copy_numeric(offsets, cp),
            "values" if kind == "list" else "data": _copy_numeric(values, cp),
            "validity": None if valid is None else _copy_numeric(valid, cp)}


def _column(column, cp, plc):
    if not _is_ragged(column):
        return _primitive(_copy_numeric(column, cp), cp, plc)
    buffers = _copy_ragged_buffers(column, cp, list_offsets=True)
    rows = _size(column)
    mask, null_count = None, 0
    if buffers["validity"] is not None and rows:
        mask, null_count = plc.transform.bools_to_mask(_primitive(buffers["validity"], cp, plc))
    offsets = buffers["offsets"]
    if buffers["kind"] == "string":
        # STRING has raw UTF-8 chars plus an int32/int64 offsets child.
        return plc.Column(plc.DataType(plc.TypeId.STRING), rows,
                          plc.gpumemoryview(buffers["data"]), mask, null_count, 0,
                          [_primitive(offsets, cp, plc)])
    payload = buffers["values" if buffers["kind"] == "list" else "data"]
    # BINARY is a list<uint8>, not decoded text; embedded NUL and invalid UTF-8
    # are retained. LIST int64 follows the identical native child path.
    return _list(offsets, _primitive(payload, cp, plc), rows, cp, plc, mask, null_count)


def _child_names(column):
    # Only STRUCT field names are logical. Other physical child labels are
    # metadata needed to preserve the complete native List/STRING hierarchy.
    return [(str(index), _child_names(child)) for index, child in enumerate(column.children())]


def _frame(columns, metadata, cudf, plc):
    table = plc.io.TableWithMetadata(plc.Table(columns), metadata)
    # Public constructor retains the independent copied buffers and their
    # recursive STRUCT names without DataFrame.to_struct's extra payload copy.
    return cudf.DataFrame.from_pylibcudf(table)


def to_cudf(value):
    """Independent native table: CPU buffers use H2D, GPU buffers use D2D."""
    columns, ordinal = _export_columns(value)
    layout = _check(value, columns)
    cp, cudf, plc = _consumer()
    with cp.cuda.Device() if ordinal is None else cp.cuda.Device(ordinal):
        if layout in _FLAT:
            copied = {name: _column(column, cp, plc) for name, column in columns.items()}
            return _frame(list(copied.values()),
                          [(name, _child_names(column)) for name, column in copied.items()], cudf, plc)
        nested = {}
        for name, column in columns.items():
            child = _column(column, cp, plc)
            if layout in ("matrix", "tile"):
                rows, cols = value.shape
                offsets = (cp.arange(rows + 1, dtype=cp.int32) * cp.int32(cols)
                           if rows and cols else cp.zeros(rows + 1, dtype=cp.int32))
                child = _list(offsets, child, rows, cp, plc)
                child = _outer(child, rows, cp, plc)
            else:
                child = _outer(child, _size(column), cp, plc)
            nested[name] = child
        result = {"shape": _outer(
            _primitive(cp.asarray(value.shape, dtype=cp.int64), cp, plc), 2, cp, plc)}
        if layout == "tile":
            for name in ("rows", "cols"):
                span = getattr(value, name)
                result[name] = _outer(
                    _primitive(cp.asarray((span.start, span.stop), dtype=cp.int64), cp, plc),
                    2, cp, plc)
        result["data"] = plc.Column(plc.DataType(plc.TypeId.STRUCT), 1, None, None,
                                    0, 0, list(nested.values()))
        metadata = [(name, _child_names(column)) for name, column in result.items() if name != "data"]
        metadata.append(("data", [(name, _child_names(column)) for name, column in nested.items()]))
        return _frame(list(result.values()), metadata, cudf, plc)
