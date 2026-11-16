"""Explicit device result reads and independent exports.

Result storage stays private and immutable. Reading an item or calling to_numpy
synchronizes and copies to the host; to_cupy returns an independent device copy.
No implicit NumPy conversion or writable device borrow is provided.
"""
import operator
import sys

import numpy as np

from . import _fp


def _host_array(data, dtype, shape):
    """A read-only array over exported host bytes: ``bytes`` (empty exports) or
    the native owner of the export's own buffer, read without a further copy."""
    if isinstance(data, bytes):
        return np.frombuffer(data, dtype=dtype).reshape(shape)
    return np.asarray(data).view(dtype).reshape(shape)


class ReadOnlyDeviceColumn:
    __slots__ = ("__owner", "__dtype", "__shape")

    def __init__(self, owner):
        from . import _core
        if not isinstance(owner, _core._DeviceColumn):
            raise TypeError("ReadOnlyDeviceColumn requires an owned seqblaze device result")
        object.__setattr__(self, "_ReadOnlyDeviceColumn__owner", owner)
        object.__setattr__(self, "_ReadOnlyDeviceColumn__dtype", np.dtype(owner.dtype))
        object.__setattr__(self, "_ReadOnlyDeviceColumn__shape", tuple(owner.shape))

    def __setattr__(self, name, value):
        raise AttributeError("device result columns are immutable")

    @property
    def dtype(self):
        return self.__dtype

    @property
    def shape(self):
        return self.__shape

    @property
    def ndim(self):
        return len(self.__shape)

    @property
    def size(self):
        count = 1
        for dimension in self.__shape:
            count *= dimension
        return count

    @property
    def nbytes(self):
        return self.__owner.nbytes

    @property
    def _ordinal(self):
        return self.__owner.ordinal

    @property
    def device(self):
        return f"cuda:{self.__owner.ordinal}"

    def __len__(self):
        if not self.__shape:
            raise TypeError("len() of unsized scalar device column")
        return self.__shape[0]

    def __array__(self, dtype=None, copy=None):
        raise TypeError("device columns do not convert to NumPy implicitly; use column.to_numpy()")

    def reshape(self, *shape):
        """Return an immutable metadata view retaining this device allocation."""
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        shape = tuple(operator.index(dimension) for dimension in shape)
        inferred = None
        known = 1
        for index, dimension in enumerate(shape):
            if dimension == -1 and inferred is None:
                inferred = index
            elif dimension < 0 or dimension > sys.maxsize:
                raise ValueError("invalid device result reshape dimension")
            else:
                known *= dimension
        if inferred is not None:
            if known == 0 or self.size % known:
                raise ValueError("cannot infer the device result reshape dimension")
            shape = tuple(self.size // known if index == inferred else dimension
                          for index, dimension in enumerate(shape))
        elif known != self.size:
            raise ValueError("device result reshape must preserve the number of elements")
        view = object.__new__(ReadOnlyDeviceColumn)
        object.__setattr__(view, "_ReadOnlyDeviceColumn__owner", self.__owner)
        object.__setattr__(view, "_ReadOnlyDeviceColumn__dtype", self.__dtype)
        object.__setattr__(view, "_ReadOnlyDeviceColumn__shape", shape)
        return view

    def __getitem__(self, key):
        # This is an explicit host read, preserving NumPy's complete indexing
        # behavior and dtypes without exposing an internal writable device view.
        return self.to_numpy()[key]

    def to_numpy(self):
        """Copy to immutable host storage; synchronizes this result's stream."""
        return _host_array(self.__owner._host_bytes(), self.__dtype, self.__shape)

    def _export_parts(self):
        """The device buffers of a batched Result.to_numpy(), in export order."""
        return (self,)

    def _export_owner(self):
        return self.__owner

    def _from_export(self, arrays):
        """Rebuild this column's host value from its parts' host bytes."""
        (data,) = arrays
        return _host_array(data, self.__dtype, self.__shape)

    def to_cupy(self):
        """Return a fresh mutable CuPy allocation, copied device to device."""
        try:
            with _fp.bracket():
                import cupy as cp
        except ImportError as error:
            raise ImportError("Device export requires CuPy; install a CuPy build matching your CUDA environment") from error
        with _fp.bracket():
            with cp.cuda.Device(self.__owner.ordinal):
                output = cp.empty(tuple(self.__owner.shape), dtype=self.__dtype)
                self.__owner._copy_into(output)
            return output.reshape(self.__shape)

    def __dlpack_device__(self):
        return 2, self.__owner.ordinal

    def __dlpack__(self, *, stream=None, max_version=None, dl_device=None, copy=None):
        # Native export makes one independent D2D allocation and owns the
        # capsule/deleter/stream protocol, including versioned IS_COPIED.
        from . import _core
        return _core._device_dlpack(self.__owner, self.__shape, stream=stream,
                                    max_version=max_version, dl_device=dl_device, copy=copy)

    def __repr__(self):
        return f"ReadOnlyDeviceColumn(shape={self.shape}, dtype={self.dtype}, device={self.device!r})"


class DeviceRaggedCopy:
    """Independent writable CuPy buffers with immutable field structure.

    Returned by to_cupy(), never a borrow of Result storage. Construction and
    representation inspect metadata only. to_pylist() explicitly copies the
    buffers to the host and materializes Python rows.
    """
    __slots__ = ("__kind", "__offsets", "__payload", "__validity")

    def __new__(cls, *args, **kwargs):
        raise TypeError("DeviceRaggedCopy objects are returned by to_cupy(); direct construction is not supported")

    @classmethod
    def _from_copied_buffers(cls, buffers):
        """Adopt the existing export helper's copies without reading payload."""
        kind = buffers["kind"]
        if kind not in ("binary", "string", "list"):
            raise TypeError("unknown copied variable-length column kind")
        payload = "values" if kind == "list" else "data"
        if set(buffers) != {"kind", "offsets", payload, "validity"}:
            raise TypeError("inconsistent copied variable-length column buffers")
        copied = object.__new__(cls)
        for name, value in (("kind", kind), ("offsets", buffers["offsets"]),
                            ("payload", buffers[payload]), ("validity", buffers["validity"])):
            object.__setattr__(copied, "_DeviceRaggedCopy__" + name, value)
        return copied

    def __setattr__(self, name, value):
        raise AttributeError("copied device field structure is immutable; edit its CuPy buffers")

    def __delattr__(self, name):
        raise AttributeError("copied device field structure is immutable")

    @property
    def kind(self):
        return self.__kind

    @property
    def offsets(self):
        return self.__offsets

    @property
    def validity(self):
        """Writable row-byte flags, or None when the source has no bitmap."""
        return self.__validity

    @property
    def data(self):
        if self.__kind == "list":
            raise AttributeError("list<int64> copies expose values; use copy.values")
        return self.__payload

    @property
    def values(self):
        if self.__kind != "list":
            raise AttributeError("string and binary copies expose bytes; use copy.data")
        return self.__payload

    @property
    def device(self):
        return f"cuda:{self.__offsets.device.id}"

    def __len__(self):
        return self.__offsets.size - 1

    def to_pylist(self):
        """Explicit D2H and Python rows; binary bytes are never decoded."""
        from ._device_exports import _cupy
        cp = _cupy()
        with cp.cuda.Device(self.__offsets.device.id):
            offsets = cp.asnumpy(self.__offsets)
            payload = cp.asnumpy(self.__payload)
            validity = None if self.__validity is None else cp.asnumpy(self.__validity)
        expected = np.dtype("int64" if self.__kind == "list" else "uint8")
        if (offsets.dtype != np.dtype("int64") or offsets.ndim != 1 or offsets.size < 1
                or payload.dtype != expected or payload.ndim != 1
                or (validity is not None and (validity.dtype != np.dtype("bool")
                    or validity.shape != (offsets.size - 1,)))
                or offsets[0] < 0 or offsets[-1] > payload.size
                or np.any(offsets[1:] < offsets[:-1])):
            raise ValueError("copied variable-length buffers no longer form a valid column")
        rows = []
        for row in range(offsets.size - 1):
            if validity is not None and not validity[row]:
                rows.append(None)
                continue
            part = payload[offsets[row]:offsets[row + 1]]
            if self.__kind == "list":
                rows.append(part.tolist())
            else:
                value = part.tobytes()
                rows.append(value.decode("utf-8") if self.__kind == "string" else value)
        return rows

    def __repr__(self):
        return f"DeviceRaggedCopy(kind={self.kind!r}, rows={len(self)}, device={self.device!r})"


class _DeviceRaggedColumn(ReadOnlyDeviceColumn):
    """One immutable device field, held as Arrow-style device buffers.

    The private buffer copy is reversible and independently mutable. Binary
    values remain bytes (list<uint8> in cuDF); only a string
    field has UTF-8 semantics. No row or offset is read during construction.
    """
    __slots__ = ("__kind", "__offsets", "__values", "__valid")

    def __init__(self, kind, offsets, values, valid):
        if kind not in ("string", "binary", "list"):
            raise TypeError("unknown device variable-length column kind")
        offsets = ReadOnlyDeviceColumn(offsets)
        values = ReadOnlyDeviceColumn(values)
        valid = None if valid is None else ReadOnlyDeviceColumn(valid)
        value_dtype = np.dtype("int64" if kind == "list" else "uint8")
        if (offsets.dtype != np.dtype("int64") or offsets.ndim != 1 or offsets.size < 1
                or values.dtype != value_dtype or values.ndim != 1
                or offsets._ordinal != values._ordinal
                or (valid is not None and (valid.dtype != np.dtype("bool")
                    or valid.shape != (offsets.size - 1,) or valid._ordinal != offsets._ordinal))):
            raise TypeError("inconsistent device variable-length result buffers")
        object.__setattr__(self, "_DeviceRaggedColumn__kind", kind)
        object.__setattr__(self, "_DeviceRaggedColumn__offsets", offsets)
        object.__setattr__(self, "_DeviceRaggedColumn__values", values)
        object.__setattr__(self, "_DeviceRaggedColumn__valid", valid)

    @property
    def _kind(self):
        return self.__kind

    @property
    def dtype(self):
        return "list<int64>" if self.__kind == "list" else self.__kind

    @property
    def shape(self):
        return (self.__offsets.size - 1,)

    @property
    def ndim(self):
        return 1

    @property
    def size(self):
        return self.__offsets.size - 1

    @property
    def _copy_value_size(self):
        return self.__values.size

    @property
    def nbytes(self):
        return (self.__offsets.nbytes + self.__values.nbytes
                + (0 if self.__valid is None else self.__valid.nbytes))

    @property
    def _ordinal(self):
        return self.__offsets._ordinal

    @property
    def device(self):
        return self.__offsets.device

    @property
    def offsets(self):
        return self.__offsets

    @property
    def validity(self):
        """Read-only row-byte flags, or None when every row is valid."""
        return self.__valid

    @property
    def data(self):
        if self.__kind == "list":
            raise AttributeError("list<int64> columns expose values; use column.values")
        return self.__values

    @property
    def values(self):
        if self.__kind != "list":
            raise AttributeError("string and binary columns expose bytes; use column.data")
        return self.__values

    def __len__(self):
        return self.size

    def reshape(self, *shape):
        raise TypeError("variable-length device fields have one row axis and cannot be reshaped")

    def to_numpy(self):
        """Explicitly copy each buffer to an immutable host ragged column."""
        from ._result import _RAGGED
        valid = (np.frombuffer(bytes([1]) * self.size, dtype=np.bool_)
                 if self.__valid is None else self.__valid.to_numpy())
        return _RAGGED[self.__kind]._native(
            self.__offsets.to_numpy(), self.__values.to_numpy(), valid)

    def _export_parts(self):
        """The device buffers of a batched Result.to_numpy(), in the order of
        to_numpy(): validity (when present), offsets, values."""
        if self.__valid is None:
            return (self.__offsets, self.__values)
        return (self.__valid, self.__offsets, self.__values)

    def _from_export(self, arrays):
        from ._result import _RAGGED
        parts = self._export_parts()
        hosts = [part._from_export((data,)) for part, data in zip(parts, arrays)]
        if self.__valid is None:
            valid = np.frombuffer(bytes([1]) * self.size, dtype=np.bool_)
            offsets, values = hosts
        else:
            valid, offsets, values = hosts
        return _RAGGED[self.__kind]._native(offsets, values, valid)

    def to_pylist(self):
        """Explicit host materialization, including null and empty rows."""
        return self.to_numpy().to_pylist()

    def _copy_buffers(self):
        """Private columnar preparation; every returned buffer owns a D2D copy."""
        return {"kind": self.__kind, "offsets": self.__offsets.to_cupy(),
                "values" if self.__kind == "list" else "data": self.__values.to_cupy(),
                "validity": None if self.__valid is None else self.__valid.to_cupy()}

    def to_cupy(self):
        """Wrap fresh D2D buffer copies without another copy or host read."""
        return DeviceRaggedCopy._from_copied_buffers(self._copy_buffers())

    def __dlpack_device__(self):
        return self.__offsets.__dlpack_device__()

    def __dlpack__(self, **options):
        raise TypeError("a variable-length column is not a DLPack tensor; export its individual buffers")
