"""Versioned, lossless Result metadata. No optional consumer is imported here."""
from collections.abc import Mapping
import hashlib
import json
import struct

import numpy as np

from . import _core, _fp
from ._errors import ParameterError

CODEC = "lossless-json-v2"
SUPPORTED_CODECS = ("lossless-json-v1", CODEC)
# Pair and selection indices (SHA-256); sequence tables use TABLE_HASH, BLAKE3
# over the length and byte streams (D128, result-provenance.md). Older
# manifests labelled merkle-sha256-v1 for tables keep their labels and values.
# The manifest key inputs.<role>.sha256 keeps its name for schema stability and
# holds the identity under the recorded hash_algorithm.
INPUT_HASH = "merkle-sha256-v1"
TABLE_HASH = "merkle-blake3-v1"
SELECTION_HASH = "selection-chain-v2"
SCIENTIFIC_HASH = "b1-manifest-v2"
_TAG = "__seqblaze_type__"
PARQUET_KEY = b"seqblaze:manifest:v2"


def decimal(value):
    """Exact decimal text without CPython's process-wide digit-limit switch."""
    if value == 0:
        return "0"
    negative = value < 0
    value = -value if negative else value
    chunks = []
    while value:
        value, remainder = divmod(value, 1_000_000_000)
        chunks.append(remainder)
    body = str(chunks[-1]) + "".join(f"{part:09d}" for part in reversed(chunks[:-1]))
    return "-" + body if negative else body


def _integer(text):
    negative = text.startswith("-")
    digits = text[1:] if negative else text
    if not digits or any(char not in "0123456789" for char in digits) or (
            len(digits) > 1 and digits[0] == "0") or (negative and digits == "0"):
        raise ParameterError("invalid manifest integer")
    value = 0
    for start in range(0, len(digits), 9):
        chunk = digits[start:start + 9]
        value = value * 10 ** len(chunk) + int(chunk)
    return -value if negative else value


def framed_hash(domain, payload):
    domain = domain.encode("ascii")
    header = b"SEQBLAZE\0" + struct.pack("<HH", 1, len(domain)) + domain
    return hashlib.sha256(header + struct.pack("<Q", len(payload)) + payload).hexdigest()


def _encode(value, scientific=False):
    if value is None or type(value) in (str, bool):
        return value
    if isinstance(value, np.generic):
        if value.dtype.kind not in "biufc":
            raise ParameterError("manifest NumPy scalars require a numeric dtype")
        return {_TAG: "npscalar", "array": _encode(np.asarray(value), scientific)}
    if isinstance(value, (int, np.integer)) and not isinstance(value, np.bool_):
        return {_TAG: "int", "value": decimal(int(value))}
    if isinstance(value, (float, np.floating)):
        if isinstance(value, np.floating) and value.dtype.itemsize != 8:
            return _encode(np.asarray(value), scientific)
        bits = _fp.bits(value)
        if scientific and bits & 0x7FFFFFFFFFFFFFFF > 0x7FF0000000000000:
            bits = 0x7FF8000000000000
        return {_TAG: "f64", "bits": f"{bits:016x}"}
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, bytes):
        return {_TAG: "bytes", "hex": value.hex()}
    if isinstance(value, np.ndarray):
        if value.dtype.kind not in "biufc" or value.dtype.fields is not None:
            raise ParameterError("manifest arrays require a numeric, non-structured dtype")
        array = np.ascontiguousarray(value)
        # Only scientific identities canonicalize NaNs; the lossless wire keeps bits.
        if scientific and array.dtype.kind in "fc":
            size = array.dtype.itemsize // (2 if array.dtype.kind == "c" else 1)
            if size not in (2, 4, 8):
                raise ParameterError("scientific identity requires IEEE binary16/32/64 components")
            endian = array.dtype.byteorder
            raw = array.view(np.dtype(f"{endian}u{size}")).copy()
            mantissa = {2: 10, 4: 23, 8: 52}[size]
            exponent = ((1 << (size * 8 - mantissa - 1)) - 1) << mantissa
            magnitude = (1 << (size * 8 - 1)) - 1
            raw[(raw & magnitude) > exponent] = exponent | (1 << (mantissa - 1))
            array = raw.view(array.dtype)
        return {_TAG: "ndarray", "dtype": value.dtype.str, "shape": list(value.shape),
                "hex": array.tobytes(order="C").hex()}
    if isinstance(value, Mapping):
        if type(value) is not dict:
            from ._result import _SelectionNode, _expose
            if isinstance(value, _SelectionNode):
                return {_TAG: SELECTION_HASH, "root_sha256": value.root_sha256,
                        "steps": [_encode({key: _expose(item) for key, item in step.items()},
                                          scientific) for step in value._steps()]}
        if any(type(key) is not str for key in value):
            raise ParameterError("manifest mapping keys must be strings")
        encoded = {key: _encode(item, scientific) for key, item in value.items()}
        return {_TAG: "map", "items": encoded} if _TAG in encoded else encoded
    if isinstance(value, tuple):
        return {_TAG: "tuple", "items": [_encode(item, scientific) for item in value]}
    if isinstance(value, list):
        return [_encode(item, scientific) for item in value]
    export = getattr(value, "_manifest_selection_array", None)
    if export is not None and not scientific:
        return _encode(export(), scientific=False)  # Explicit JSON only.
    raise ParameterError(f"unsupported manifest value: {type(value).__name__}")


def dumps(value, *, scientific=False, frozen=False):
    """The canonical text. ``frozen`` encodes a plain call-time tree as the Manifest
    freezing it would read (a list as a tuple). Plain trees take the native encoder,
    which writes the same text with no floating-point operation; anything it does not
    take is encoded here (result-provenance.md, "Manifest JSON codecs")."""
    text = _core._native_manifest_json(value, scientific, frozen)
    if text is not None:
        return text
    if frozen:
        from ._result import _freeze
        value = _freeze(value)
    with _fp.bracket():
        return json.dumps(_encode(value, scientific), ensure_ascii=True, allow_nan=False,
                          sort_keys=True, separators=(",", ":"))


def digest(domain, value, *, scientific=False, frozen=False):
    hexdigest = _core._native_manifest_digest(domain, value, scientific, frozen)
    if hexdigest is not None:
        return hexdigest
    return framed_hash(domain, dumps(value, scientific=scientific, frozen=frozen).encode("ascii"))


def _decode(value):
    if isinstance(value, list):
        return [_decode(item) for item in value]
    if not isinstance(value, dict):
        # A strict producer never emits JSON numbers: these would lose raw bits.
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            raise ParameterError("untagged numeric value in manifest codec")
        return value
    tag = value.get(_TAG)
    if tag is None:
        return {key: _decode(item) for key, item in value.items()}
    if tag == SELECTION_HASH and set(value) == {_TAG, "root_sha256", "steps"}:
        from ._result import _restore_selection
        if not isinstance(value["steps"], list):
            raise ParameterError("selection-chain-v2 requires flat steps")
        return _restore_selection(value["root_sha256"], [_decode(step) for step in value["steps"]])
    if tag == "map" and set(value) == {_TAG, "items"} and isinstance(value["items"], dict):
        return {key: _decode(item) for key, item in value["items"].items()}
    if tag == "tuple" and set(value) == {_TAG, "items"} and isinstance(value["items"], list):
        return tuple(_decode(item) for item in value["items"])
    if tag == "int" and set(value) == {_TAG, "value"} and type(value["value"]) is str:
        return _integer(value["value"])
    if tag == "f64" and set(value) == {_TAG, "bits"} and type(value["bits"]) is str:
        text = value["bits"]
        if len(text) != 16 or any(char not in "0123456789abcdefABCDEF" for char in text):
            raise ParameterError("invalid manifest binary64 encoding")
        return _fp.from_bits(int.from_bytes(bytes.fromhex(text), "big"))
    if tag == "bytes" and set(value) == {_TAG, "hex"} and type(value["hex"]) is str:
        return bytes.fromhex(value["hex"])
    if tag == "ndarray" and set(value) == {_TAG, "dtype", "shape", "hex"}:
        dtype = np.dtype(value["dtype"])
        shape = value["shape"]
        if dtype.kind not in "biufc" or dtype.fields is not None or not isinstance(shape, list):
            raise ParameterError("invalid manifest ndarray schema")
        size = 1
        for length in shape:
            if type(length) is not int or length < 0:
                raise ParameterError("invalid manifest ndarray shape")
            size *= length
        data = bytes.fromhex(value["hex"])
        if len(data) != size * dtype.itemsize:
            raise ParameterError("manifest ndarray byte length differs from shape")
        return np.frombuffer(data, dtype=dtype).reshape(shape)
    if tag == "npscalar" and set(value) == {_TAG, "array"}:
        array = _decode(value["array"])
        if not isinstance(array, np.ndarray) or array.shape != ():
            raise ParameterError("invalid manifest NumPy scalar")
        return array[()]
    raise ParameterError("unknown or malformed manifest codec record")


def loads(source):
    def reject_constant(_):
        raise ParameterError("non-finite JSON literal in manifest")

    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ParameterError("duplicate manifest JSON key")
            result[key] = value
        return result

    with _fp.bracket():
        try:
            return _decode(json.loads(source, parse_constant=reject_constant, object_pairs_hook=unique))
        except (ValueError, TypeError, OverflowError, UnicodeError) as error:
            raise ParameterError(f"invalid {CODEC} manifest: {error}") from None


def selection_facts(selection):
    """Backend-neutral selection identity without materializing owned indices.

    Legacy mapping/bridge projection only: computed v2 scientific hashing uses
    the cached lineage commitment. Walk parents iteratively, without exporting
    device indices. Metadata without a digest retains explicit array identity.
    """
    if selection is None or not isinstance(selection, Mapping):
        return selection
    result, previous = None, None
    while isinstance(selection, Mapping):
        facts = {key: value for key, value in selection.items() if key != "parent"}
        if (facts.get("index_hash_algorithm") == INPUT_HASH
                and isinstance(facts.get("indices_sha256"), str)):
            facts.pop("indices", None)
        if previous is None:
            result = facts
        else:
            previous["parent"] = facts
        previous = facts
        if "parent" not in selection:
            return result
        selection = selection["parent"]
    previous["parent"] = selection
    return result


def scientific_facts(manifest):
    """The B1 identity excludes placement, counters and descriptive spelling."""
    model = dict(manifest.get("model", {}))
    if "cond" in model:
        model["cond"] = {key: value for key, value in model["cond"].items() if key != "defaulted"}
    if model.get("compat") is not None:
        model["compat"] = {"id": model["compat"]["id"]}
    if model.get("params", {}).get("sha256") is not None:
        model["params"] = {"sha256": model["params"]["sha256"]}
    model.pop("n_illegal_seqs", None)
    inputs = {}
    for name, record in manifest.get("inputs", {}).items():
        if name in ("a", "b"):
            inputs[name] = {key: record[key] for key in ("n", "sha256", "hash_algorithm") if key in record}
        elif name == "pairs":
            inputs[name] = {key: value for key, value in record.items() if key != "same_table"}
        else:
            inputs[name] = record
    selection = manifest.get("selection")
    lineage = manifest.get("lineage", {})
    if lineage.get("algorithm") == SELECTION_HASH:
        selection = {"algorithm": SELECTION_HASH, "sha256": lineage["sha256"]}
    else:
        selection = selection_facts(selection)
    return {"call": {key: value for key, value in manifest.get("call", {}).items() if key != "repro"},
            "model": model, "inputs": inputs, "output": manifest.get("output"),
            "selection": selection,
            "flags": {key: value for key, value in manifest.get("flags", {}).items()
                      if key not in ("research_use_only", "host_pairs")}}


def write_parquet(result, path):
    try:
        import pyarrow.parquet as pq
    except ImportError as error:
        raise ImportError("Result.to_parquet() requires PyArrow; install it with: pip install pyarrow") from error
    table = result.to_arrow()
    metadata = dict(table.schema.metadata or {})
    wire = result.manifest.to_json().encode("ascii")
    version = result.manifest.get("seqblaze", {}).get("manifest_codec", CODEC)
    if version not in SUPPORTED_CODECS:
        raise ParameterError("unsupported manifest codec version")
    key = b"seqblaze:manifest:v1" if version == "lossless-json-v1" else PARQUET_KEY
    if key in metadata and metadata[key] != wire:
        raise ParameterError("Arrow schema already contains conflicting seqblaze manifest metadata")
    metadata[key] = wire
    table = table.replace_schema_metadata(metadata)
    pq.write_table(table, path, use_dictionary=False, use_byte_stream_split=False,
                   write_statistics=False, store_schema=True)
