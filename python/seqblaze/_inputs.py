"""Host and device sequence adapters shared by the public facade."""
import numpy as np
# Native adapters also check masked arrays. Load this required module with the
# package: NumPy's first lazy ma import sets PE on an otherwise integer path.
from numpy.ma import isMaskedArray as _is_masked_array

from . import _core
from ._device_inputs import is_device_input, seqs_device
from ._errors import ParameterError, PairingError, _PairingError
from ._fastx import SeqBatch
from ._frames import releasing

SeqTable = _core._SeqTable


def native_call(function, *args, **kwargs):
    try:
        return function(*args, **kwargs)
    except _PairingError as error:
        raise PairingError(str(error)) from None


@releasing
def seqs(x, *, lengths=None):
    """Convert containers once, or borrow supported host sequence buffers.

    Text (``str``, NumPy ``U<n>``, Arrow string/large_string/string_view) is
    its UTF-8 encoding, byte for byte (no normalization); bytes are used as
    given. ``None`` items and Arrow nulls are null rows: they are recorded here
    and raise SequenceError at the call that uses the table (D86 (1), (3)).
    Unicode and object arrays convert once into an immutable Rust snapshot.
    Borrowed arrays must not be resized while retained, or modified during a
    computation or unfinished tile iteration. A SeqBatch gives its cached
    zero-copy table over the ``seq`` column.
    """
    # Device protocols precede host Arrow/NumPy fallbacks. No producer's
    # __array__, get, to_numpy or Arrow-host export is invoked on this path.
    # Exact built-in containers, NumPy arrays and tables take the branch the
    # checks below select for them.
    kind = type(x)
    if lengths is None and (kind is list or kind is str or kind is bytes or kind is tuple):
        return native_call(_core._native_seqs, x)
    if kind is np.ndarray:
        return native_call(_core._native_seqs_numpy, x, lengths)
    if kind is SeqTable and lengths is None and x._source_kind != "device":
        return x
    if is_device_input(x):
        return seqs_device(x, lengths)
    if isinstance(x, SeqBatch):
        if lengths is not None:
            raise ParameterError("lengths requires a uint8 matrix; a SeqBatch carries its own lengths")
        return x._table()
    if isinstance(x, np.ndarray):
        return native_call(_core._native_seqs_numpy, x, lengths)
    if lengths is not None:
        raise ParameterError("lengths requires a uint8 matrix; use seqs(matrix, lengths=L)")
    if isinstance(x, SeqTable):
        return x
    if hasattr(x, "__arrow_c_array__") or hasattr(x, "__arrow_c_stream__"):
        return _arrow_or_array(x, native_call)
    return native_call(_core._native_seqs, x)


def _arrow_or_array(x, call):
    """Import a CPU Arrow producer; fall back to ``__array__`` only when its export
    raises ImportError (an optional Arrow runtime is missing, engine-layout §9).

    Prefer ``__arrow_c_array__``; use ``__arrow_c_stream__`` only when it is absent.
    The fallback accepts a 1-D NumPy array of kind O, U or S and hands it to the
    NumPy adapter: O and U are copied once into owned bytes, S is borrowed.
    """
    native = (_core._native_seqs_arrow if hasattr(x, "__arrow_c_array__")
              else _core._native_seqs_arrow_stream)
    try:
        return call(native, x)
    except ImportError as error:
        missing = error
    array = None
    if hasattr(x, "__array__"):
        try:
            array = x.__array__()
        except Exception:  # an unusable fallback leaves the original failure
            array = None
    if isinstance(array, np.ndarray) and array.ndim == 1 and array.dtype.kind in "OUS":
        return call(_core._native_seqs_numpy, array, None)
    raise ParameterError(
        f"Arrow export of {type(x).__name__} failed: ImportError: {missing}; and it has no 1-D "
        "text or bytes __array__ fallback. Install the producer's Arrow runtime, or pass "
        "a NumPy array (for example np.asarray(x, dtype=object))") from None


def two_tables(a, b):
    shared = a is b
    a = seqs(a)
    return a, a if shared else seqs(b)


def index_array(x, label):
    """Normalize integer ArrayLike once; borrow native contiguous i32/i64."""
    from ._device_inputs import _DeviceIndex, index_device, is_device_input
    from ._device import ReadOnlyDeviceColumn
    if isinstance(x, (_DeviceIndex, ReadOnlyDeviceColumn)) or is_device_input(x):
        return index_device(x, label=label)
    if _is_masked_array(x):
        raise PairingError(f"{label} must be unmasked one-dimensional integers; use np.array([0], dtype='int64')")
    try:
        array = np.asarray(x)
    except (TypeError, ValueError, OverflowError) as error:
        raise PairingError(f"{label} must be one-dimensional integers: {error}") from None
    if array.ndim != 1 or (array.dtype.kind not in "iu" and array.size != 0):
        raise PairingError(f"{label} must be one-dimensional integers; use np.array([0], dtype='int64')")
    if array.dtype.kind not in "iu" and not isinstance(x, (list, tuple)):
        raise PairingError(f"{label} has dtype {array.dtype}; use native int32 or int64 indices")
    if array.dtype.kind == "u" and array.size and int(array.max()) > np.iinfo(np.int64).max:
        raise PairingError(f"{label} contains an index above int64 max; indices must fit int64")
    if array.dtype not in (np.dtype("int32"), np.dtype("int64")) or not array.dtype.isnative:
        array = array.astype(np.int64)
    elif not array.flags.c_contiguous:
        array = np.ascontiguousarray(array)
    return array
