"""Host Result/Tile table adapters; optional consumers are imported on demand.

Flat layouts keep their entry rows. A TopK, Matrix or Tile contributes one
row, with ordinary shape/coordinate columns and its complete data in a struct.
The native C Stream owns all nested headers and retained numeric buffers.
"""
import re

import numpy as np

from . import _device_exports
from ._errors import ParameterError

_FLAT = ("each", "hits", "rowstats", "scalar")
_NUMERIC = (np.dtype("int32"), np.dtype("int64"), np.dtype("float64"), np.dtype("bool"))


def _layout(value):
    # This module loads with _result. Resolve its classes only when called,
    # after that module has finished defining them; no first-use file import.
    from ._result import Tile
    return "tile" if isinstance(value, Tile) else value.layout


def _ragged(array):
    from ._result import _RaggedColumn
    return isinstance(array, _RaggedColumn)


def check(value):
    """Validate actual host buffers; device snapshots require explicit D2H."""
    value = _device_exports.host_view(value)
    layout = _layout(value)
    columns = value._columns
    if layout in _FLAT:
        ndim = 0 if layout == "scalar" else 1
        numeric = [array for array in columns.values() if not _ragged(array)]
        sizes = {array.size for array in numeric}
        sizes |= {len(array) for array in columns.values() if _ragged(array)}
        if (len(sizes) != 1 or (layout == "scalar" and sizes != {1}) or
                any(array.ndim != ndim or array.dtype not in _NUMERIC for array in numeric)):
            raise ParameterError("Table export requires equal-length int32/int64/float64/bool, "
                                 "string, binary or list<int64> columns; use a flat Result")
        return layout
    if layout not in ("topk", "matrix", "tile"):
        raise NotImplementedError(f"Table mapping for {layout} results is not implemented")
    if (len(value.shape) != 2 or any(type(n) is not int or not 0 <= n < 2**63 for n in value.shape)
            or not columns):
        raise ParameterError("Complex table export requires a nonnegative two-dimensional shape")
    if layout == "topk":
        required = {"i": np.dtype("int64"), "j": np.dtype("int64"),
                    "rank": np.dtype("int32"), "offsets": np.dtype("int64"),
                    "tie_at_k": np.dtype("bool")}
        for name, dtype in required.items():
            array = columns.get(name)
            if not isinstance(array, np.ndarray) or array.ndim != 1 or array.dtype != dtype:
                raise ParameterError(f"TopK table export requires a 1-D {dtype} {name} column")
        entries = len(columns["i"])
        for name, array in columns.items():
            expected = value.shape[0] + 1 if name == "offsets" else value.shape[0] if name == "tie_at_k" else entries
            if ((not _ragged(array) and (array.ndim != 1 or array.dtype not in _NUMERIC))
                    or len(array) != expected):
                raise ParameterError("TopK entry columns, offsets and row flags have inconsistent lengths")
    else:
        if any(_ragged(array) or array.ndim != 2 or array.shape != value.shape
               or array.dtype not in _NUMERIC or not array.flags.c_contiguous
               or array.flags.writeable for array in columns.values()):
            raise ParameterError("Matrix and Tile table exports require readonly contiguous numeric fields matching shape")
        if layout == "tile":
            for name, size in zip(("rows", "cols"), value.shape):
                span = getattr(value, name)
                if (not isinstance(span, slice) or span.step not in (None, 1)
                        or type(span.start) is not int or type(span.stop) is not int
                        or not 0 <= span.start <= span.stop < 2**63
                        or span.stop - span.start != size):
                    raise ParameterError("Tile export requires matching half-open global rows and cols")
    return layout


def _exporter(value):
    value = _device_exports.host_view(value)
    layout = check(value)
    if value._arrow_export is None:
        from . import _core
        packed = {name: (array._parts() if _ragged(array) else
                         array.reshape(-1) if layout in ("matrix", "tile") else array)
                  for name, array in value._columns.items()}
        if layout in _FLAT:
            exporter = _core._native_arrow_export(packed)
        else:
            extra = ({"rows": (value.rows.start, value.rows.stop),
                      "cols": (value.cols.start, value.cols.stop)} if layout == "tile" else {})
            exporter = _core._native_arrow_export_complex(layout, packed, value.shape, **extra)
        object.__setattr__(value, "_arrow_export", exporter)
    return value._arrow_export


def arrow_stream(value, requested_schema=None):
    # Unsupported schema conversion must fail before downloading payload.
    if requested_schema is not None:
        raise NotImplementedError("Arrow requested_schema conversion is not implemented yet")
    return _exporter(value).stream()


def to_arrow(value):
    device = _device_exports.has_device_columns(value)
    if not device:
        check(value)
    try:
        import pyarrow as pa
    except ImportError as error:
        raise ImportError(f"{type(value).__name__}.to_arrow() requires PyArrow; "
                          "install it with: pip install pyarrow") from error
    if device:
        value = _device_exports.host_view(value)
        check(value)
    return pa.table(value)


def to_pandas(value):
    layout = None if _device_exports.has_device_columns(value) else check(value)
    try:
        import pandas as pd
    except ImportError as error:
        raise ImportError(f"{type(value).__name__}.to_pandas() requires pandas; "
                          "install it with: pip install pandas") from error
    value = _device_exports.host_view(value)
    if layout is None:
        layout = check(value)
    if layout in _FLAT:
        columns = {name: (pd.Series(array.to_pylist(), dtype=object) if _ragged(array)
                          else array.reshape(-1)) for name, array in value._columns.items()}
        return pd.DataFrame(columns, copy=True)
    # Python lists are an explicit copying convenience. Nested payloads must
    # not share NumPy arrays or buffers with the immutable Result/Tile.
    row = {"shape": [int(n) for n in value.shape]}
    if layout == "tile":
        row.update(rows=[value.rows.start, value.rows.stop], cols=[value.cols.start, value.cols.stop])
    row["data"] = {name: (array.to_pylist() if _ragged(array) else array.tolist())
                   for name, array in value._columns.items()}
    return pd.DataFrame({name: [item] for name, item in row.items()}, copy=True)


def to_polars(value):
    device = _device_exports.has_device_columns(value)
    if not device:
        check(value)
    hint = f"{type(value).__name__}.to_polars() requires Polars>=1.3; install it with: pip install 'polars>=1.3'"
    try:
        import polars as pl
    except ImportError as error:
        raise ImportError(hint) from error
    version = re.match(r"^([0-9]+)\.([0-9]+)", pl.__version__)
    if version is None or tuple(map(int, version.groups())) < (1, 3):
        raise ImportError(hint)
    if device:
        value = _device_exports.host_view(value)
        check(value)
    return pl.DataFrame(value)
