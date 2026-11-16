"""Context-local execution defaults; contexts never change numerical results."""
from contextlib import contextmanager
from contextvars import ContextVar
import re
import importlib
import struct
from dataclasses import dataclass
import sys

import numpy as np

from . import _core
from . import _device_inputs
from . import _fp
from ._errors import DeviceError, ParameterError
from ._registry import HAMMING
from .keep import _integer

_DEFAULTS = dict(HAMMING["execution"])
_CONTEXT = ContextVar("seqblaze_execution", default=None)


@dataclass(frozen=True)
class StreamSelection:
    """Options-only selection; owner stays live through call/iterator completion."""
    handle: int | None
    owner: object = None


def _stream_selection(value):
    if isinstance(value, StreamSelection):
        return value
    if value is None:
        return StreamSelection(None)
    if type(value) is int:
        if 0 <= value < 1 << (8 * struct.calcsize("P")):
            return StreamSelection(value or 1)
    elif not isinstance(value, bool):
        try:
            with _fp.bracket():
                cp = importlib.import_module("cupy")
        except ImportError:
            cp = None
        with _fp.bracket():
            if cp is not None and isinstance(value, (cp.cuda.Stream, cp.cuda.ExternalStream)):
                handle = value.ptr
                if type(handle) is int and 0 <= handle < 1 << (8 * struct.calcsize("P")):
                    return StreamSelection(handle or 1, value)
    raise ParameterError("stream must be None, a CuPy stream, or an unsigned platform-sized CUDA handle")


def resolve_stream(selection, device):
    """Resolve only after placement. CPU auto never imports CuPy or selects CUDA."""
    if device is None:
        if selection.handle is not None:
            raise ParameterError("stream requires CUDA placement; host device='auto' stays on the CPU")
        return None
    owner = selection.owner
    handle = selection.handle
    if handle is None:
        try:
            with _fp.bracket():
                cp = importlib.import_module("cupy")
        except ImportError:
            cp = None
        if cp is None:
            handle = 1
        else:
            try:
                with _fp.bracket():
                    with cp.cuda.Device(device["ordinal"]):
                        owner = cp.cuda.get_current_stream()
                        handle = owner.ptr or 1
            except Exception as error:
                raise DeviceError(f"resolving current CUDA stream failed: {error}") from None
    if owner is not None:
        with _fp.bracket():
            ordinal = getattr(owner, "device_id", -1)
        if ordinal not in (-1, device["ordinal"]):
            raise DeviceError("stream belongs to a different CUDA device")
    # Metadata validation only: no input payload has been imported or scanned.
    _core._native_execution_stream_validate(device["ordinal"], handle)
    return StreamSelection(handle, owner)


def _validate(options):
    """Check execution options; return them with threads normalized to a built-in int."""
    options = dict(options)
    for name in options:
        if name not in _DEFAULTS:
            similar = _fp.close_matches(name, _DEFAULTS, n=1)
            hint = f"; did you mean {similar[0]!r}?" if similar else ""
            raise ParameterError(f"unknown execution option {name!r}{hint}; valid options: {', '.join(_DEFAULTS)}")
    if "strategy" in options and options["strategy"] not in ("bounded", "full"):
        raise ParameterError(f"strategy={options['strategy']!r}; use strategy='bounded' or 'full'")
    if "device" in options:
        device = options["device"]
        if type(device) is not str or (device not in ("auto", "cpu", "cuda") and not re.fullmatch(r"cuda:[0-9]+", device)):
            raise ParameterError(f"device={device!r}; use device='auto', 'cpu', 'cuda', or 'cuda:<n>'")
    if "threads" in options and options["threads"] is not None:
        threads = _integer(options["threads"])
        if threads is None or not 0 < threads <= sys.maxsize:
            raise ParameterError(f"threads={options['threads']!r}; use threads=None or a positive platform-sized int")
        options["threads"] = threads
    if "progress" in options and type(options["progress"]) is not bool:
        raise ParameterError(f"progress={options['progress']!r}; use progress=True or False")
    if "stream" in options:
        options["stream"] = _stream_selection(options["stream"])
    return options


@contextmanager
def execution(**options):
    """Set execution defaults within this thread/async context; calls override them."""
    options = _validate(options)
    defaults = dict(_CONTEXT.get() or {})
    defaults.update(options)
    token = _CONTEXT.set(defaults)
    try:
        yield
    finally:
        _CONTEXT.reset(token)


def resolve(options):
    """Merge and check execution options. Raises only ParameterError; the
    device is resolved separately, after every option check."""
    options = _validate(options)
    merged = dict(_DEFAULTS)
    merged.update(_CONTEXT.get() or {})
    merged.update(options)
    merged["stream"] = _stream_selection(merged["stream"])
    if merged["threads"] is None:
        merged["threads"] = _core._available_threads()
    return merged


def resolve_device(device, *, inputs=()):
    """Return None for the CPU, else the resolved CUDA device as a dict with
    ordinal, name and arch. device='auto' stays on the CPU for host inputs. An
    explicit CUDA request with no CUDA build, driver or device raises
    DeviceError; there is no CPU fallback. No data moves."""
    resident = []
    for value in inputs:
        if value is None:
            continue
        kind = type(value)
        if kind is _core._SeqTable and value._source_kind != "device":
            continue  # a host table carries no device protocol
        if kind is np.ndarray:
            continue  # NumPy declares the CPU through DLPack
        if getattr(value, "_source_kind", None) == "device":
            resident.append(value._device_ordinal)
        elif _device_inputs.is_device_input(value):
            if device == "cpu":
                raise DeviceError("device sequence/index inputs require the CUDA backend; use device='auto' or their CUDA ordinal")
            if device == "auto":
                resident.append(_device_inputs.ordinal(value))
    if device == "cpu":
        if resident:
            raise DeviceError("device sequence/index inputs require the CUDA backend; use device='auto' or their CUDA ordinal")
        return None
    if device == "auto":
        if not resident:
            return None
        if any(item != resident[0] for item in resident[1:]):
            raise DeviceError("device inputs belong to different CUDA devices; cross-device inputs are unsupported")
        device = f"cuda:{resident[0]}"
    if not _core.build_info()["cuda_compiled"]:
        raise DeviceError(f"device={device!r} is unavailable: this seqblaze build has no CUDA "
                          "backend (built without CUDA; build_info()['cuda_compiled'] is False); "
                          "use device='cpu'")
    limit = 2**63 - 1  # no device has a larger ordinal; the native check reports it invalid
    # ASCII digits only (checked by resolve). Leading zeros are dropped and more than 19
    # digits clamp to `limit` before the parse. The parse is an integer digit fold
    # (`_fp.decimal`): int() of a decimal string raises the inexact flag in the caller's
    # floating-point environment (text-nulls-fp-environment.md §7.4, status flags).
    digits = "0" if device == "cuda" else device[len("cuda:"):].lstrip("0") or "0"
    ordinal = limit if len(digits) > 19 else min(_fp.decimal(digits), limit)
    try:
        return dict(_core._cuda_device(ordinal))
    except DeviceError as error:
        if ordinal < limit:
            raise
        raise DeviceError(str(error).replace(f"'cuda:{limit}'", repr(device))) from None
