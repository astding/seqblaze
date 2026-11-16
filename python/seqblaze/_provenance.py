"""Call-time logical identity and explicit, mutation-checked public API replay."""
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from . import _core, _fp
from ._errors import ParameterError
from . import _manifest_codec as codec
# The module object, not its names: _result imports this module first.
from . import _result


def _table_digest(table, *, snapshot=False, threads=2):
    # Device adapters implement exactly the same logical Merkle codec
    # (merkle-blake3-v1). No host table conversion or whole-input D2H is
    # allowed in the digest hook.
    hook = getattr(table, "_device_snapshot" if snapshot else "_device_digest", None)
    if hook is not None:
        return dict(hook())
    entry = _core._native_input_snapshot if snapshot else _core._native_input_digest
    return dict(entry(table, threads=threads))


def _indices(array, native, *, snapshot=False, threads=2):
    hook_name = "_device_index_snapshot" if snapshot else "_device_index_digest"
    hook = getattr(array, hook_name, None) or getattr(native, hook_name, None)
    if hook is not None:
        facts = dict(hook())
    else:
        entry = _core._native_index_snapshot if snapshot else _core._native_index_digest
        facts = dict(entry(native, threads=threads))
    return facts.get("data"), facts["sha256"]


def _facts(table, previous, threads, captured=None):
    facts = _table_digest(table, threads=threads) if captured is None else captured
    return dict(previous, **{name: facts[name] for name in
                            ("n", "total_nt", "min_len", "max_len", "sha256")},
                hash_algorithm=codec.TABLE_HASH)


@dataclass(frozen=True, slots=True)
class ReplayContext:
    a: object
    b: object
    pairs: object
    metadata: Mapping
    scalar: bool
    scoring: object
    root_id: str

    def snapshots(self):
        saved = []
        threads = self.metadata["execution"]["threads"]
        for name, table in (("a", self.a), ("b", self.b)):
            if table is None:
                continue
            if name == "b" and table is self.a:
                # The context retains both table references: identity is stable.
                # One owned snapshot is also the source for replay's b = a.
                snapshot = saved[0]
            else:
                try:
                    snapshot = _table_digest(table, snapshot=True, threads=threads)
                except (ValueError, OverflowError) as error:
                    raise ParameterError(f"repro input {name} changed or is no longer valid: {error}") from None
            # Both role digests must match, including a reused shared snapshot.
            if snapshot["sha256"] != self.metadata["inputs"][name]["sha256"]:
                raise ParameterError(f"repro input {name} changed since computation; restore the original logical input")
            saved.append(snapshot)
        indices = None
        if self.pairs is not None and self.pairs.kind == "index":
            i, hi = _indices(self.pairs.i, self.pairs._ii, snapshot=True, threads=threads)
            j, hj = _indices(self.pairs.j, self.pairs._jj, snapshot=True, threads=threads)
            expected = self.metadata["inputs"]["pairs"]
            if hi != expected["i_sha256"] or hj != expected["j_sha256"]:
                raise ParameterError("repro pair indices changed since computation; restore the original logical input")
            indices = i, j
        return saved, indices


def capture(metadata, *, a, b=None, pairs=None, scalar=False, scoring=None, tile=None,
            captured_inputs=None):
    """Fill logical facts and retain references, never full input history.

    Resident calls supply facts already validated/hashed on their compute lease;
    adopting them must not reopen a stream or hash the mutable producer again.
    """
    inputs = dict(metadata["inputs"])
    threads = metadata["execution"]["threads"]
    captured = None
    if captured_inputs is not None:
        expected = {"a"} | ({"b"} if b is not None else set())
        if pairs is not None and pairs.kind == "index":
            expected |= {"i", "j"}
        if not isinstance(captured_inputs, Mapping) or set(captured_inputs) != expected:
            raise ParameterError("resident call digest facts do not match its logical inputs")
        captured = {}
        for name in expected:
            facts = captured_inputs[name]
            if not isinstance(facts, Mapping):
                raise ParameterError("resident call digest facts must be mappings")
            required = {"sha256", "n"} if name in ("i", "j") else {
                "sha256", "n", "total_nt", "min_len", "max_len", "text"}
            if set(facts) != required:
                raise ParameterError("resident call digest facts have an invalid schema")
            digest = facts["sha256"]
            if (type(digest) is not str or len(digest) != 64
                    or any(char not in "0123456789abcdef" for char in digest)):
                raise ParameterError("resident call digest must be a 64-digit lowercase hexadecimal identity")
            if any(type(facts[key]) is not int or facts[key] < 0
                   for key in required - {"sha256", "text"}):
                raise ParameterError("resident call digest counts must be nonnegative integers")
            if "text" in required and type(facts["text"]) is not bool:
                raise ParameterError("resident call text fact must be bool")
            captured[name] = dict(facts)
    inputs["a"] = _facts(a, inputs["a"], threads, None if captured is None else captured["a"])
    if b is not None:
        inputs["b"] = _facts(b, inputs["b"], threads, None if captured is None else captured["b"])
    if pairs is not None:
        recipe = dict(inputs["pairs"], shape=pairs.shape, self_pairs=pairs._self_pairs,
                      scalar=bool(scalar), same_table=a is b)
        if pairs.kind == "index":
            if captured is None:
                _, recipe["i_sha256"] = _indices(pairs.i, pairs._ii, threads=threads)
                _, recipe["j_sha256"] = _indices(pairs.j, pairs._jj, threads=threads)
            else:
                if captured["i"]["n"] != pairs.n_pairs or captured["j"]["n"] != pairs.n_pairs:
                    raise ParameterError("resident pair digest length differs from original occurrences")
                recipe["i_sha256"] = captured["i"]["sha256"]
                recipe["j_sha256"] = captured["j"]["sha256"]
            recipe["index_hash_algorithm"] = codec.INPUT_HASH
        inputs["pairs"] = recipe
    metadata["inputs"] = inputs
    metadata["seqblaze"] = dict(metadata["seqblaze"], manifest_codec=codec.CODEC,
                                input_hash=codec.TABLE_HASH, scientific_hash=codec.SCIENTIFIC_HASH)
    if tile is not None:
        metadata["model"]["keep"] = dict(metadata["model"]["keep"], tile=tile)
    if scalar and metadata["model"]["keep"]["sink"] == "each":
        metadata["call"]["layout"] = "scalar"
    # A structured replay recipe is available without eagerly serializing and
    # retaining full historical input. Result.repro() supplies executable code.
    root = {"call": dict(metadata["call"]), "model": metadata["model"], "inputs": inputs}
    root_id = codec.digest("root-call", codec.scientific_facts(root), scientific=True)
    metadata["call"]["repro"] = {"version": 1, "root_sha256": root_id,
                                    "input_policy": "call-digest-verify-owned-snapshot"}
    metadata["classification"] = {
        "b1": ("call.kernel", "call.layout", "model.resolved", "inputs.logical", "output.schema", "selection"),
        "information": ("seqblaze", "execution", "citations", "inputs.source_type", "inputs.encoding",
                        "model.cond.defaulted", "model.compat.alias"),
    }
    # The context owns these facts as they are now: the facade assembles the
    # final manifest in new dicts and changes none of these after capture.
    return ReplayContext(a, b, pairs, _result.Manifest._deferred(dict(metadata)), bool(scalar),
                         scoring, root_id)


def selection_lineage_digest(selection, root_id):
    """Use cached v2 lineage; adopt older mapping facts without index export."""
    if selection is None:
        return codec.digest("selection-root-v2", root_id)
    if isinstance(selection, _result._SelectionNode):
        return selection.lineage_sha256
    # Compatibility for manually assembled metadata/provenance bridges. New
    # CPU/device selections are nodes and never walk or rehash their parents.
    steps = []
    while selection is not None and not isinstance(selection, _result._SelectionNode):
        if not isinstance(selection, Mapping):
            raise ParameterError("selection history requires immutable step mappings")
        facts = {key: value for key, value in selection.items() if key != "parent"}
        steps.append(codec.selection_facts(facts))
        selection = selection.get("parent")
    lineage = (codec.digest("selection-root-v2", root_id) if selection is None
               else selection.lineage_sha256)
    for facts in reversed(steps):
        lineage = codec.digest(codec.SELECTION_HASH, {
            "parent_sha256": lineage, "step_sha256": codec.digest("selection-step-v2", facts)})
    return lineage


def selection_node(facts, owner, indices, parent, root_id, *, threads):
    native = _core._native_indices(indices, "selection.indices")
    indices_sha256 = _core._native_index_digest(native, threads=threads)["sha256"]
    facts = dict(facts, index_hash_algorithm=codec.INPUT_HASH, indices_sha256=indices_sha256)
    return _result._SelectionNode(facts, _result._FrozenArray._native_indices(owner), parent,
                                  root_id)


def complete_manifest(manifest, *, layout, shape, fields, columns, root=None, geometry=None,
                      owned=False):
    """Layout and lineage metadata only: never scan numeric output values.

    ``owned`` marks the plain facts of a facade call, owned by the new Result alone.
    Those, and a frozen Manifest's facts, are frozen on first read; other plain
    mappings are frozen now, as their caller may still change them.
    """
    deferred = owned or isinstance(manifest, _result._FrozenMap)
    tree = manifest._plain_tree() if type(manifest) is _result.Manifest else None
    data = dict(manifest if tree is None else tree)
    data["output"] = {"layout": layout, "shape": tuple(shape), "columns": tuple(
        {"name": field.name, "dtype": field.dtype, "unit": field.unit, "exactness": field.exactness,
         # Device columns expose validated shape metadata. Reading it cannot
         # trigger np.asarray, a host copy, or len() of a scalar array.
         "shape": tuple(columns[field.name].shape) if getattr(columns[field.name], "shape", None) is not None
         else (len(columns[field.name]),)} for field in fields)}
    if geometry is not None:
        data["output"].update(geometry)
    selection = data.get("selection")
    root_id = root.root_id if root is not None else data.get("call", {}).get("repro", {}).get("root_sha256")
    if root_id is not None:
        data["lineage"] = {"root_sha256": root_id,
                           "algorithm": codec.SELECTION_HASH,
                           "sha256": selection_lineage_digest(selection, root_id)}
    # The identity reads the frozen form, as when the facts were frozen first.
    data["scientific_sha256"] = codec.digest(codec.SCIENTIFIC_HASH, codec.scientific_facts(data),
                                             scientific=True, frozen=deferred)
    return _result.Manifest._deferred(data) if deferred else _result.Manifest(data)


def stream_manifest(manifest, shape):
    """Pure stream facts: retaining them cannot retain the stream's inputs.

    ``manifest`` is the facade's fresh plain facts of the stream (``_manifest``),
    owned by the returned Manifest alone."""
    data = dict(manifest)
    data["output"] = {"layout": "tiles", "shape": tuple(shape),
                      "requested_fields": tuple(data.get("model", {}).get("fields", ()))}
    root_id = data.get("call", {}).get("repro", {}).get("root_sha256")
    if root_id is not None:
        data["lineage"] = {"root_sha256": root_id,
                           "algorithm": codec.SELECTION_HASH,
                           "sha256": selection_lineage_digest(data.get("selection"), root_id)}
    data["scientific_sha256"] = codec.digest(codec.SCIENTIFIC_HASH, codec.scientific_facts(data),
                                             scientific=True, frozen=True)
    return _result.Manifest._deferred(data)


def tile_manifest(manifest, rows, cols, shape, columns, global_shape=None):
    if manifest is None:
        manifest = _result.Manifest({"call": {"layout": "tiles"}})
    kernel = manifest.get("call", {}).get("kernel", "tile")
    fields = tuple(_result._field_info(name, array, kernel) for name, array in columns.items())
    return complete_manifest(manifest, layout="tile", shape=shape, fields=fields, columns=columns,
                             geometry={"rows": (rows.start, rows.stop), "cols": (cols.start, cols.stop),
                                       "global_shape": tuple(global_shape) if global_shape is not None else None})


def selection_digest(indices, *, threads=2):
    """Bulk logical identity of already-owned normalized CPU positions."""
    native = _core._native_indices(indices, "selection")
    return _core._native_index_digest(native, threads=threads)["sha256"]


def _literal(value):
    """Safe, exact Python expressions; input bytes never become source code."""
    export = getattr(value, "_manifest_selection_array", None)
    if export is not None:
        return _literal(export())  # Explicit repro only, never scientific hashing.
    if type(value) is int:
        if value.bit_length() < 1024:
            return repr(value)
        magnitude = abs(value)
        raw = magnitude.to_bytes((magnitude.bit_length() + 7) // 8, "little")
        prefix = "-" if value < 0 else ""
        return f"{prefix}int.from_bytes({raw!r}, 'little')"
    if value is None or type(value) in (str, bytes, bool):
        return repr(value)
    if isinstance(value, np.integer):
        return _literal(int(value))
    if isinstance(value, (float, np.floating)):
        return f"_f('{_fp.bits(value):016x}')"
    if isinstance(value, np.ndarray):
        return (f"np.frombuffer({value.tobytes(order='C')!r}, dtype={value.dtype.str!r})"
                f".reshape({value.shape!r})")
    if isinstance(value, Mapping):
        return "{" + ", ".join(f"{key!r}: {_literal(item)}" for key, item in value.items()) + "}"
    if isinstance(value, (tuple, list)):
        inner = ", ".join(_literal(item) for item in value)
        return f"({inner},)" if len(value) == 1 else f"({inner})"
    raise ParameterError(f"cannot serialize replay value {type(value).__name__}")


def _keep(model):
    keep = model["keep"]
    kind, cutoff = keep["sink"], keep.get("cutoff")
    if kind == "matrix":
        return "sb.keep.matrix()"
    if kind == "tiles":
        return f"sb.keep.tiles(tile={_literal(keep['tile'])})"
    if kind == "topk":
        return f"sb.keep.topk({keep['k']!r}, cutoff={_literal(cutoff)})"
    if kind == "hits":
        return f"sb.keep.hits({_literal(cutoff)})"
    return f"sb.keep.{kind}(cutoff={_literal(cutoff)})"


def _scoring(scoring):
    kwargs = {"gap_open": scoring.gap_open, "gap_extend": scoring.gap_extend,
              "end_gaps": scoring.end_gaps}
    if scoring._table is not None:
        kwargs["matrix"] = scoring.matrix
    elif scoring._matrix is not None:
        kwargs["matrix"] = scoring._matrix
    else:
        kwargs.update(match=scoring.match, mismatch=scoring.mismatch)
    return "sb.Scoring(" + ", ".join(f"{name}={_literal(value)}" for name, value in kwargs.items()) + ")"


def render(result):
    context = result._replay
    if context is None:
        raise ParameterError("this Result has no captured original call; repro requires a computed Result")
    snapshots, indices = context.snapshots()
    metadata = context.metadata
    model = metadata["model"]
    lines = ["# seqblaze executable replay v2; full root call, then recorded selections.",
             "import struct", "import numpy as np", "import seqblaze as sb", "",
             "def _f(bits):", "    return struct.unpack('>d', bytes.fromhex(bits))[0]", "",
             "def _verify(actual, expected, role, recovery):",
             "    if actual != expected:",
             "        raise sb.ParameterError(f'{role} differs: expected={expected}, actual={actual}; {recovery}')", "",
             "def _verify_versions(result):",
             f"    _verify(result.manifest.get('seqblaze', {{}}).get('manifest_codec'), {codec.CODEC!r},",
             "            'manifest codec version', 'restore a matching runtime before regenerating or executing replay')",
             f"    _verify(result.manifest.get('seqblaze', {{}}).get('scientific_hash'), {codec.SCIENTIFIC_HASH!r},",
             "            'scientific identity algorithm version', 'restore a matching runtime before regenerating or executing replay')", ""]
    for name, snapshot in zip(("a", "b"), snapshots):
        if name == "b" and metadata["inputs"].get("pairs", {}).get("same_table"):
            lines.extend(("b_rows = a_rows", "b = a"))
            continue
        rows = snapshot["python_rows"].decode("ascii")
        lines.append(f"{name}_rows = {rows}")
        lines.append(f"{name} = sb.seqs({name}_rows)")
    pairs = context.pairs
    if pairs is None:
        arg = "a_rows[0]" if context.scalar else "a"
    elif pairs.kind == "zip":
        arg = "sb.pairs.zip(a_rows[0], b_rows[0])" if context.scalar else "sb.pairs.zip(a, b)"
    elif pairs.kind == "index":
        ii, jj = indices
        lines += [f"i = np.frombuffer({ii!r}, dtype='<i8')", f"j = np.frombuffer({jj!r}, dtype='<i8')"]
        arg = "sb.pairs.index(a, i, j, table_b=b)"
    elif pairs.kind == "within":
        arg = f"sb.pairs.within(a, self_pairs={pairs._self_pairs!r})"
    elif pairs.kind == "cross":
        arg = "sb.pairs.cross(a, b)"
    else:
        arg = "sb.pairs.one_to_many(a_rows[0], b)"
    kwargs = {"fields": tuple(model["fields"])}
    kernel = result.kernel
    if kernel in ("hamming", "levenshtein", "align"):
        kwargs["check_bases"] = model["check_bases"]
    for key in ("mode", "orientation", "end"):
        if key in model:
            kwargs[key] = model[key]
    compat = model.get("compat")
    if compat is not None:
        kwargs["compat"] = compat["alias"]
    if kernel == "duplex":
        kwargs["model"] = model["model"]
    if "params" in model and compat is None:
        kwargs["params"] = model["params"]["name"]
    expressions = [f"{key}={_literal(value)}" for key, value in kwargs.items()]
    if "cond" in model:
        cond = model["cond"]
        values = {key: value for key, value in cond.items()
                  if key != "defaulted" and key not in cond.get("defaulted", ())}
        expressions.append("cond=sb.Conditions(" + ", ".join(f"{key}={_literal(value)}" for key, value in values.items()) + ")")
    if kernel == "align":
        if context.scoring is None:
            raise ParameterError("alignment replay requires its captured immutable Scoring")
        expressions.append(f"scoring={_scoring(context.scoring)}")
    expressions.append(f"keep={_keep(model)}")
    execution = metadata["execution"]
    expressions += [f"device={execution['device']!r}", f"threads={execution['threads']!r}",
                    f"strategy={execution['strategy']['requested']!r}"]
    lines.append(f"result = sb.{kernel}({arg}, " + ", ".join(expressions) + ")")
    lines.append("_verify_versions(result)")
    for name in ("a", "b"):
        if name in metadata["inputs"]:
            expected = metadata["inputs"][name]["sha256"]
            lines.append(f"_verify(result.manifest['inputs'][{name!r}]['sha256'], {expected!r}, "
                         f"'logical input digest for {name}', 'restore matching inputs before regenerating or executing replay')")
    if "params" in model:
        lines.append(f"_verify(result.manifest['model']['params']['sha256'], {model['params']['sha256']!r}, "
                     "'installed parameter digest for model.params', 'restore matching parameters before regenerating or executing replay')")
    selection = result.manifest.get("selection")
    chain = []
    while selection is not None:
        chain.append(selection)
        selection = selection.get("parent")
    for number, item in enumerate(reversed(chain)):
        if "indices" not in item:
            raise ParameterError("selection lacks recorded normalized indices required for replay")
        indices = item["indices"]
        lines.append(f"_selection{number} = {_literal(indices)}")
        if item["kind"] == "mask":
            lines += [f"_mask{number} = np.zeros({item['parent_rows']!r}, dtype=bool)",
                      f"_mask{number}[_selection{number}] = True", f"result = result[_mask{number}]"]
        else:
            lines.append(f"result = result.take(_selection{number})")
    lines.append("_verify_versions(result)")
    lines.append(f"_verify(result.manifest['scientific_sha256'], {result.manifest['scientific_sha256']!r}, "
                 "'scientific call/selection identity', 'restore matching inputs, parameters and runtime before regenerating or executing replay')")
    return "\n".join(lines) + "\n"
