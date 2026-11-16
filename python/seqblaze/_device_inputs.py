"""Device input metadata adapters. These functions never request host payloads.

cuDF strings use libcudf's Arrow-style character data, offset child and bitmask.
The wrappers retain both the original producer and its column/buffer owners.
"""
from . import _core, _fp
from ._errors import DeviceError, PairingError, ParameterError


def is_device_input(value):
    module = type(value).__module__.split('.', 1)[0]
    if module in ('cudf', 'pylibcudf'):
        return True
    if hasattr(value, '__cuda_array_interface__'):
        return True
    return (hasattr(value, '__dlpack_device__') and
            value.__dlpack_device__()[0] in (2, 13))


def _cudf_column(value):
    """Use the public zero-copy column view; labels do not enter sequences."""
    with _fp.bracket():
        import pylibcudf as plc
    exported = value.to_pylibcudf()
    # cuDF 26.10 returns (Column, metadata). Accept a direct public Column
    # view too; retain the actual producer and Column as storage owners.
    column = exported[0] if isinstance(exported, tuple) and len(exported) == 2 else exported
    if not isinstance(column, plc.Column):
        raise ParameterError('cuDF sequence inputs require a string column')
    return column


def ordinal(value):
    """Use DLPack's declaration or the driver's actual CAI allocation ordinal."""
    if not _core.build_info()["cuda_compiled"]:
        raise DeviceError('this seqblaze build has no CUDA device input support')
    if type(value).__module__.split('.', 1)[0] in ('cudf', 'pylibcudf'):
        column = (_cudf_column(value)
                  if type(value).__module__.split('.', 1)[0] == 'cudf' else value)
        buffer = column.data()
        if buffer is not None:
            address = buffer.__cuda_array_interface__['data'][0]
            if address:
                return _core._native_device_pointer_ordinal(address)
        # Nonempty string columns with no characters still have an offsets
        # allocation. Its actual ordinal is stronger than the current device.
        with _fp.bracket():
            import pylibcudf as plc
        if column.size() and column.type().id() == plc.types.TypeId.STRING:
            offsets = column.child(0).data()
            if offsets is not None:
                address = offsets.__cuda_array_interface__['data'][0]
                if address:
                    return _core._native_device_pointer_ordinal(address)
        with _fp.bracket():
            import cupy as cp
            return cp.cuda.Device().id
    if hasattr(value, '__dlpack_device__'):
        kind, device = value.__dlpack_device__()
        if kind not in (2, 13):
            raise DeviceError('sequence device inputs require CUDA device memory')
        return device
    descriptor = value.__cuda_array_interface__
    address = descriptor['data'][0]
    if address:
        return _core._native_device_pointer_ordinal(address)
    # Empty CAI views have no allocation to query. A producer may provide its
    # device descriptor; ordinary single-device protocol wrappers default to 0.
    device = getattr(value, 'device', None)
    return getattr(device, 'id', 0)


class _CudaView:
    __slots__ = ('_descriptor', '_owners')

    def __init__(self, buffer, owners, *, typestr='|u1', count=None, width=1, byte_offset=0):
        # libcudf may represent a zero-byte chars allocation with no data
        # buffer, even when the string column has nonzero (empty/null) rows.
        descriptor = ({'version': 3, 'shape': (0,), 'typestr': '|u1',
                       'data': (0, True), 'strides': None, 'stream': None}
                      if buffer is None else dict(buffer.__cuda_array_interface__))
        if descriptor.get('version') != 3:
            raise ParameterError('cuDF buffers require CUDA Array Interface version 3')
        if count is None:
            count = descriptor['shape'][0]
        descriptor.update(typestr=typestr, shape=(count,), strides=None,
                          data=(descriptor['data'][0] + byte_offset, True))
        self._descriptor = descriptor
        self._owners = (buffer, *owners)

    @property
    def __cuda_array_interface__(self):
        return self._descriptor


def _strings(value):
    with _fp.bracket():
        import pylibcudf as plc
    module = type(value).__module__.split('.', 1)[0]
    column = _cudf_column(value) if module == 'cudf' else value
    if not isinstance(column, plc.Column) or column.type().id() != plc.types.TypeId.STRING:
        raise ParameterError('cuDF sequence inputs require a string column')
    rows = column.size()
    if rows == 0:
        # libcudf empty strings may have no offsets child; accessing one is not
        # valid. Supply our own one-element zero control buffer, no input data.
        with _fp.bracket():
            import cupy as cp
            offsets = cp.zeros(1, dtype=cp.int32)
            data = cp.empty(0, dtype=cp.uint8)
            device = cp.cuda.Device().id
        return _core._native_seqs_device_strings(data, offsets, None, 0, 0,
                                                device, True)
    offsets_column = column.child(0)
    offset_type = offsets_column.type().id()
    if offset_type == plc.types.TypeId.INT32:
        typestr, width = '<i4', 4
    elif offset_type == plc.types.TypeId.INT64:
        typestr, width = '<i8', 8
    else:
        raise ParameterError('cuDF strings require int32 or int64 offsets')
    owners = (value, column)
    data = _CudaView(column.data(), owners)
    offsets = _CudaView(offsets_column.data(), (*owners, offsets_column),
                        typestr=typestr, count=offsets_column.size(), width=width,
                        byte_offset=offsets_column.offset() * width)
    mask_buffer = column.null_mask()
    validity = None if mask_buffer is None else _CudaView(mask_buffer, owners)
    device = ordinal(data if data.__cuda_array_interface__['data'][0] else offsets)
    return _core._native_seqs_device_strings(data, offsets, validity, rows,
                                            column.offset(), device, True)


def seqs_device(value, lengths):
    if not hasattr(_core, '_native_seqs_device_matrix'):
        raise DeviceError('this seqblaze build has no CUDA device input support')
    if type(value).__module__.split('.', 1)[0] in ('cudf', 'pylibcudf'):
        if lengths is not None:
            raise ParameterError('cuDF strings carry their own offsets; omit lengths')
        return _strings(value)
    if lengths is None:
        raise ParameterError('device uint8 matrices require explicit lengths; use seqs(matrix, lengths=L)')
    return _core._native_seqs_device_matrix(value, lengths, ordinal(value))


class _DeviceIndex:
    """Retained device indices; normalization happens on each compute call."""
    __slots__ = ('_native', '_public')

    def __init__(self, native, public=None):
        object.__setattr__(self, '_native', native)
        object.__setattr__(self, '_public', public)

    def __setattr__(self, name, value):
        raise AttributeError('device index ownership is read-only')

    def __len__(self):
        return len(self._native)

    @property
    def _source_kind(self):
        return 'device'

    @property
    def _device_ordinal(self):
        return self._native._device_ordinal

    @property
    def shape(self):
        return (len(self),)

    @property
    def dtype(self):
        import numpy as np
        return np.dtype('int64')

    @property
    def storage_info(self):
        return self._native.storage_info

    def to_numpy(self):
        """Explicitly snapshot and copy current logical indices to readonly host storage."""
        import numpy as np
        snapshot = self._native._device_index_snapshot()
        values = np.frombuffer(snapshot['data'], dtype='<i8', count=snapshot['n'])
        if not values.dtype.isnative:
            values = np.frombuffer(values.astype(np.int64).tobytes(), dtype=np.int64)
        return values

    def _device_index_digest(self):
        return self._native._device_index_digest()

    def _device_index_snapshot(self):
        return self._native._device_index_snapshot()

    def __array__(self, *args, **kwargs):
        raise TypeError('device indices require an explicit device export; implicit host conversion is disabled')


def index_device(value, *, owned=False, label='indices'):
    """Retain an external producer or an immutable native result column."""
    if isinstance(value, _DeviceIndex):
        return value
    from ._device import ReadOnlyDeviceColumn
    from ._inputs import native_call
    elements = None
    public = None
    if isinstance(value, ReadOnlyDeviceColumn):
        if value.ndim != 1:
            raise PairingError(f"{label} must be one-dimensional integers; use np.array([0], dtype='int64')")
        elements = value.size
        public = value
        native = object.__getattribute__(value, '_ReadOnlyDeviceColumn__owner')
        device = native.ordinal
    elif isinstance(value, getattr(_core, '_DeviceColumn', ())):
        native = value
        public = ReadOnlyDeviceColumn(value)
        device = value.ordinal
    else:
        if owned:
            raise TypeError('owned device indices require a native device column')
        native = value
        device = ordinal(value)
    return _DeviceIndex(native_call(_core._native_indices_device, native, label, device, elements), public)
