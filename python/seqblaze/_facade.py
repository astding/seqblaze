"""Batch facade implementation (CPU kernels and CUDA dispatch) behind the registry-generated callable."""
from fractions import Fraction
import math
from time import perf_counter_ns
import warnings

import numpy as np

from . import _core
from . import _fp
from . import keep as keep_module
from ._errors import DeviceError, IllegalBaseWarning, PairingError, ParameterError, UnitError
from ._conditions import Conditions
from ._device_exports import _is_ragged
from ._execution import resolve, resolve_device, resolve_stream
from ._frames import release_frames as _release_frames
from ._inputs import native_call, seqs
from ._registry import ALIGN, DUPLEX, HAIRPIN, HAMMING, LEVENSHTEIN, TM, citation_keys
from ._result import Result, TileIterator, _column
from ._scoring import Scoring
from ._provenance import capture as _capture_replay
from .pairs import Pairs


# The compiled artifact's facts (build_info()): fixed when the extension was built.
_BUILD = _core.build_info()
# Execution plans by (kernel, keep, strategy, cutoff given, backend): a plan depends on
# these and on the build alone, except duplex's, which also depends on its per-call pack
# and is never memoized. Shared by the calls' manifests, which only read them.
_PLANS = {}


# The Conditions of a compat preset's bare call (cond=None), built once: Conditions is
# immutable and a preset's values are fixed.
_PRESET_CONDITIONS = {}


def _preset_conditions(name):
    cond = _PRESET_CONDITIONS.get(name)
    if cond is None:
        cond = _PRESET_CONDITIONS.setdefault(name, Conditions.preset(name))
    return cond


def _plan(kernel, keep, strategy, has_cutoff, backend):
    key = (kernel, keep, strategy, has_cutoff, backend)
    plan = _PLANS.get(key)
    if plan is None:
        plan = _PLANS.setdefault(key, _core._native_plan(*key))
    return plan


def _mode_fields(spec, mode):
    """(fields the mode supports, its default fields) in canonical order."""
    if "mode_fields" in spec:
        return spec["mode_fields"][mode], spec["default_fields"][mode]
    return tuple(spec["fields"]), spec["default_fields"]


def _fields(spec, value, mode=None, *, option="mode", primary=None):
    """Resolve fields= to the emitted fields: canonical order, primary first.

    The primary field (dist) selects every sink, so it is always emitted;
    "default" and "all" are the mode's default and full field sets. ``option``
    names the option that selects the field set in messages (duplex: model) and
    ``primary`` overrides the spec's primary for that set.
    """
    allowed, defaults = _mode_fields(spec, mode)
    title = spec["name"].capitalize()
    if isinstance(value, str):
        names = defaults if value == "default" else allowed if value == "all" else (value,)
    elif isinstance(value, (list, tuple)):
        names = value
    else:
        raise ParameterError(f"fields={value!r}; use fields='default', 'all', or a list such as {list(defaults)!r}")
    if not names:
        raise ParameterError(f"fields cannot be empty; use fields={list(defaults)!r}")
    where = f" {option}={mode!r}" if mode is not None else ""
    primary = spec["primary"] if primary is None else primary
    # Field groups (align: coords, stats, aligned) are expanded before this
    # check; unknown names are matched against them too and they are listed.
    groups = list(spec.get("field_groups", {}))
    for name in names:
        if type(name) is not str or name not in spec["fields"]:
            candidates = [*spec["fields"], *groups]
            similar = _fp.close_matches(name, candidates, n=1) if isinstance(name, str) else []
            hint = f"; did you mean {similar[0]!r}?" if similar else ""
            supported = f"fields={list(allowed)!r}" + (f" or the groups {groups!r}" if groups else "")
            raise ParameterError(f"unknown field {name!r}{hint}; {title}{where} supports {supported}")
        if name not in allowed:
            raise ParameterError(f"field {name!r} does not exist in {spec['name']}{where}; it supports "
                                 f"fields={list(allowed)!r}")
    requested = set(names)
    return tuple(name for name in allowed if name == primary or name in requested)


def _choice(spec, name, value):
    choices = spec["parameters"][name]["choices"]
    if type(value) is not str or value not in choices:
        raise ParameterError(f"{name}={value!r}; use one of {', '.join(repr(c) for c in choices)}")
    return value


def _sink(value, spec=HAMMING, sinks=None):
    """The checked keep of a call; ``sinks`` overrides the spec's sink list (tm on Pairs
    takes ``TM["pair_sinks"]``, tm.md §13.5)."""
    if value is None:
        return keep_module.each()
    allowed = spec["sinks"] if sinks is None else sinks
    if not isinstance(value, keep_module.Sink) or value.kind not in allowed:
        raise ParameterError(f"keep={value!r}; use sb.keep.each(), hits(), topk(), rowstats(), matrix(), or tiles()")
    if value.kind in ("matrix", "tiles"):
        if value.cutoff is not None or value.k is not None:
            raise ParameterError(f"keep={value.kind!r} does not take cutoff or k; use sb.keep.{value.kind}()")
        checked = keep_module.tiles(tile=value.tile) if value.kind == "tiles" else keep_module.matrix()
    elif value.kind == "topk":
        checked = keep_module.topk(value.k, cutoff=value.cutoff)
    elif value.kind == "hits":
        checked = keep_module.hits(value.cutoff)
    else:
        checked = getattr(keep_module, value.kind)(cutoff=value.cutoff)
    if (value.k != checked.k or value.tile != checked.tile):
        raise ParameterError(f"contradictory keep={value!r}; construct it with sb.keep.{value.kind}()")
    return checked


def _positional_error(kernel, pairs, extra):
    """PairingError for extra positional arguments to a pairwise kernel; never guess pairing."""
    received = ", ".join(type(value).__name__ for value in (pairs, *extra))
    count = f"{1 + len(extra)} positional arguments ({received})"
    if isinstance(pairs, Pairs):
        keep = next((value for value in extra if isinstance(value, keep_module.Sink)), None)
        example = (f"sb.{kernel}(pairs, keep=sb.keep.{keep.kind}(...))" if keep is not None
                   else f"sb.{kernel}(pairs, keep=sb.keep.hits(cutoff))")
        return PairingError(f"{kernel}() takes one positional argument, the pairs, but got {count}; "
                            f"every other option is keyword-only, e.g. {example}")
    return PairingError(
        f"{kernel}() takes pairs as its first argument, but got {count}. Say how they pair:\n"
        f"    sb.{kernel}(sb.pairs.zip(a, b))                               # a[k] with b[k], one value per pair\n"
        f"    sb.{kernel}(sb.pairs.cross(a, b), keep=sb.keep.hits(cutoff))  # every a with every b")


def _bound(cutoff):
    """The integer distance bound of a cutoff (an int, or a float read exactly).

    A float cutoff is taken as its exact rational, so the sign test and floor
    hold in every caller environment (under DAZ a hardware comparison and floor
    read a subnormal as zero; text-nulls-fp-environment.md §7.4). No int is
    converted to a float, including very large ones."""
    if cutoff is None:
        return None
    exact = _fp.exact(cutoff) if isinstance(cutoff, float) else cutoff
    if exact < 0:
        return -1
    if exact >= 2**31 - 1:
        return 2**31 - 1
    return math.floor(exact)


_INT32_MAX = 2**31 - 1
_INT32_MIN = -2**31
#: FLT_MAX as a double: water's sentinel score (spec §3).
_FLT_MAX = float(np.finfo(np.float32).max)
#: Clamp range of an align cutoff's t (spec §6): key = -t then spans INT32_MIN
#: (no real row passes) to INT32_MAX - 2 (every real row passes). §9 keeps
#: |score_fx| <= 2^30, so a clamped t selects exactly the same real rows.
_T_MAX = -_INT32_MIN
_T_MIN = -(_INT32_MAX - 2)


def _align_cutoff(cutoff):
    """(key threshold, t) of an align cutoff c (spec §6, §8).

    t is the smallest integer k with fl(k / 1000) >= c, clamped to
    [_T_MIN, _T_MAX]; real rows pass iff score_fx >= t, i.e. key = -score_fx
    <= -t. Sentinel rows (key INT32_MAX - 1) pass only when c <= -FLT_MAX.
    k0 is exact rational arithmetic, and the float step runs only for k0 inside
    the clamp range, so an int cutoff of any size (e.g. -10**400) is taken
    exactly and never converted to a float.
    """
    if cutoff is None:
        return None, None
    # Exact rationals throughout: fl(k / 1000) is fl_RN in every caller FP
    # environment, and a float cutoff is read from its bits (§7.4).
    c = _fp.exact(cutoff) if isinstance(cutoff, float) else Fraction(cutoff)
    k0 = math.ceil(c * 1000)
    if k0 > _T_MAX:
        t = _T_MAX
    elif k0 <= _T_MIN:
        t = _T_MIN
    else:
        t = k0 - 1 if _fp.exact(_fp.fl_rn(k0 - 1, 1000)) >= c else k0
    if c <= -_fp.exact(_FLT_MAX):
        return _INT32_MAX - 1, t
    return -t, t


def _placement(build, device, options, plan):
    if device is None:
        return {"backend": "cpu", "device": "cpu", "arch": build["target_arch"],
                "threads": options["threads"], "strategy": plan}
    return {"backend": "cuda", "device": f"cuda:{device['ordinal']}",
            "device_name": device["name"], "arch": device["arch"],
            "threads": options["threads"], "strategy": plan}


def _build_metadata(build):
    """The compiled artifact's identity; no runtime filesystem/Git lookup."""
    return {"version": build["version"], "git": build["git"], "git_dirty": build["git_dirty"],
            "build": {"target": build["target_arch"], "simd": build["simd"],
                      "cuda_compiled": build["cuda_compiled"], "nvcc_version": build["nvcc_version"]}}


def _metadata(spec, model, pairs, n_pairs, sink, check_bases, fields, options, bound, plan, device,
              cutoff_int=None):
    build = _BUILD
    return {
        "seqblaze": _build_metadata(build),
        "call": {"kernel": spec["name"], "layout": sink.kind},
        "model": dict(model, check_bases=check_bases, fields=fields,
                      keep={"sink": sink.kind, "k": sink.k, "cutoff": sink.cutoff,
                            "cutoff_int": bound if cutoff_int is None else cutoff_int}),
        "inputs": {"a": _input_record(pairs.a), "b": _input_record(pairs.b),
                   "pairs": {"kind": pairs.kind, "n_pairs": n_pairs}},
        "execution": _placement(build, device, options, plan),
        "flags": {"research_use_only": True},
        "citations": citation_keys(spec, model, fields, plan["effective"], pairs=True),
    }


def _input_record(table):
    """inputs.a / inputs.b: n and source_type, plus encoding "utf-8" exactly when at
    least one row came from text (D86 (3); text-nulls-fp-environment.md §5)."""
    record = {"n": len(table), "source_type": table._source_kind}
    if table._text:
        record["encoding"] = "utf-8"
    return record


def _elapsed(start_ns):
    """execution.wall_s: the seconds since ``start_ns``, a ``perf_counter_ns()`` reading.

    fl_RN(dns / 10**9) of the integer nanosecond difference dns, rounded in integer
    arithmetic. ``time.perf_counter()`` converts the clock to seconds with a float
    division in the caller's floating-point environment, which raises the inexact
    flag there, and a float subtraction would round in that environment; the integer
    clock and ``_fp.fl_rn`` perform no rounded float operation and raise no flag
    (text-nulls-fp-environment.md §7.4, timing row).
    """
    return _fp.fl_rn(perf_counter_ns() - start_ns, 10**9)


def _manifest(metadata, report, elapsed):
    """The call's final plain facts. New dicts replace the groups that change; the
    captured groups are shared and never changed (``_provenance.capture``)."""
    result = dict(metadata)
    result["execution"] = dict(metadata["execution"], counts=report["counts"],
                               threads=report["threads"], wall_s=elapsed)
    # Both backends report the final sink's flag counts. In particular TopK's
    # tie_at_k has one authority, without a second facade count/override.
    result["flags"] = dict(metadata["flags"], **report["flags"])
    result["model"] = dict(metadata["model"], n_illegal_seqs=report["flags"]["n_illegal_seqs"])
    if "topk" in report:
        result["execution"]["topk"] = report["topk"]
    if "tiles" in report:
        result["execution"]["tiles"] = report["tiles"]
    if "host_pairs" in report:
        # Device calls only: distinct work items the CUDA backend routed to the
        # host CPU kernel (levenshtein-cuda.md §5.5); 0 for hamming.
        result["execution"]["host_pairs"] = report["host_pairs"]
    return result


def _hamming(pairs, *, check_bases, fields, sink, options):
    options = resolve(options)
    if type(check_bases) is not bool:
        raise ParameterError(f"check_bases={check_bases!r}; use check_bases=True or False")
    fields = _fields(HAMMING, fields)
    return _run(HAMMING, pairs, {}, check_bases=check_bases, fields=fields, sink=sink,
                options=options, orientation=None)


def _levenshtein(pairs, *, mode, orientation, check_bases, fields, sink, options):
    # Options first (D48 §1.5): nothing below inspects the inputs.
    options = resolve(options)
    mode = _choice(LEVENSHTEIN, "mode", mode)
    orientation = _choice(LEVENSHTEIN, "orientation", orientation)
    if type(check_bases) is not bool:
        raise ParameterError(f"check_bases={check_bases!r}; use check_bases=True or False")
    fields = _fields(LEVENSHTEIN, fields, mode)
    # orientation reaches the engine only for prefix/infix (spec §7); global is
    # symmetric and records it in the manifest without effect (Q4).
    oriented = orientation if mode != "global" else None
    return _run(LEVENSHTEIN, pairs, {"mode": mode, "orientation": orientation},
                check_bases=check_bases, fields=fields, sink=sink, options=options,
                orientation=oriented, mode=mode)


def _align(pairs, *, mode, check_bases, scoring, compat, fields, sink, options):
    """sb.align (docs/wiki/align/gotoh-align.md). Option checks in the spec §8 order:
    mode, scoring presence, the Scoring rules, the compat domain, check_bases with a
    matrix, then fields and the cutoff; pairs and sequences follow in the engine."""
    options = resolve(options)
    if compat is not None:
        compat = _choice(ALIGN, "compat", compat)
    if mode is not None:
        mode = _choice(ALIGN, "mode", mode)
    if compat is not None:
        implied = "local" if compat == "emboss-water" else "global"
        if mode is not None and mode != implied:
            raise ParameterError(f"mode={mode!r} contradicts compat={compat!r}, which is a "
                                 f"{implied} alignment; omit mode or use mode={implied!r}")
        mode = implied
    elif mode is None:
        raise ParameterError("align requires mode='local' or mode='global' (or a compat preset), "
                             "e.g. sb.align(pairs, mode='local', scoring=sb.Scoring(gap_open=10, "
                             "gap_extend=0.5, matrix='EDNAFULL'))")
    if scoring is None:
        if compat is None:
            raise ParameterError(f"align(mode={mode!r}) requires scoring= (D41), e.g. "
                                 "scoring=sb.Scoring(gap_open=10, gap_extend=0.5, matrix='EDNAFULL'"
                                 + (", end_gaps=False)" if mode == "global" else ")"))
        scoring = Scoring.preset(compat)
    elif not isinstance(scoring, Scoring):
        raise ParameterError(f"scoring={scoring!r}; use an sb.Scoring, e.g. "
                             "sb.Scoring(gap_open=10, gap_extend=0.5, matrix='EDNAFULL')")
    if mode == "global" and scoring.end_gaps is None:
        raise ParameterError("align(mode='global') requires scoring.end_gaps: True penalizes end "
                             "gaps, False leaves them free (EMBOSS needle's default); e.g. "
                             "sb.Scoring(gap_open=10, gap_extend=0.5, matrix='EDNAFULL', end_gaps=False)")
    if compat is not None:
        _check_compat_domain(compat, scoring)
    if type(check_bases) is not bool:
        raise ParameterError(f"check_bases={check_bases!r}; use check_bases=True or False")
    if not check_bases and scoring.match is None:
        raise ParameterError("check_bases=False applies to match/mismatch scoring only; a matrix "
                             "defines its own alphabet (D48), so use check_bases=True")
    # The field set follows the selector: compat when a preset is given, else mode
    # (gotoh-align.md §3.1; n_ends is native mode='local' only).
    selector, option = (compat, "compat") if compat is not None else (mode, "mode")
    fields = _fields(ALIGN, _expand_groups(fields), selector, option=option)
    model = {"mode": mode, "scoring": scoring._manifest(),
             "compat": None if compat is None else {"alias": compat,
                                                    "id": ALIGN["presets"]["compat"][compat]},
             "tie_rule": "native rule E" if compat is None else "compat"}
    config = dict(scoring._native(), compat=compat, score_fx="score_fx" in fields)
    return _run(ALIGN, pairs, model, check_bases=check_bases, fields=fields, sink=sink,
                options=options, orientation=None, mode=mode, align=config, replay_scoring=scoring)


def _expand_groups(value):
    """fields= with the coords/stats/aligned group names expanded in place."""
    groups = ALIGN["field_groups"]
    if isinstance(value, str):
        return list(groups[value]) if value in groups else value
    if isinstance(value, (list, tuple)):
        expanded = []
        for name in value:
            expanded.extend(groups[name] if isinstance(name, str) and name in groups else (name,))
        return expanded
    return value


# What switching a compat call to native mode needs (spec §5.2, §5.3; QA2, QA3, QA14);
# the same sentence as seqblaze_align::NATIVE_SWITCH in the kernel's messages.
_NATIVE_SWITCH = ("Switching to native mode needs the input upper-cased (U and X mapped to T and "
                  "N) or a 256x256 matrix, because native matrix=\"EDNAFULL\" accepts only the 15 "
                  "upper-case letters ATGCSWRYKMBVHDN.")


def _check_compat_domain(compat, scoring):
    """§5.2: the certified domain of the compat presets (D62), on the 1/1000 grid."""
    program = compat.removeprefix("emboss-")
    go, ge = scoring._go, scoring._ge
    low = "5 <= gap_open" if program == "water" else "5 < gap_open"
    domain = (f"{compat} is certified bit-exact (D62) for matrix='EDNAFULL', gap_open and "
              f"gap_extend multiples of 1/8, {low} <= 100 and 0 <= gap_extend <= min(gap_open, 10)")
    hint = f"; use native mode (mode=..., scoring=...) for other values. {_NATIVE_SWITCH}"
    if scoring._matrix != "EDNAFULL":
        raise ParameterError(f"{domain}; scoring={scoring!r} is outside the certified domain{hint}")
    go_ok = 5000 <= go <= 100_000 if program == "water" else 5000 < go <= 100_000
    if go % 125 or ge % 125 or not go_ok or ge > min(go, 10_000):
        raise ParameterError(f"{domain}; gap_open={scoring.gap_open!r}, gap_extend="
                             f"{scoring.gap_extend!r} lie outside the certified domain{hint}")


def _pair_count(pairs, orientation):
    if pairs.kind != "within" or orientation != "both":
        return pairs.n_pairs
    n = len(pairs.a)
    return n * (n - 1) + (n if pairs._self_pairs else 0)


def _run(spec, pairs, model, *, check_bases, fields, sink, options, orientation, mode="global",
         align=None, duplex=None, key_cutoff=None, primary=None, cutoff_int=True, sinks=None,
         replay_scoring=None):
    """The shared pairwise path. ``duplex`` is duplex's per-call pack and ``key_cutoff``
    its cutoff already converted to the integer key (duplex-mfe.md §8.3, duplex-pf.md
    §9.3); ``primary`` overrides the spec's primary (duplex model="pf"), and
    ``cutoff_int=False`` leaves keep.cutoff_int out of the manifest (duplex-pf.md §9.3).
    ``tm`` on Pairs (tm.md §13) passes its K1 per-call pack as ``duplex``, its key
    cutoff as ``key_cutoff`` and its sink list (``TM["pair_sinks"]``) as ``sinks``; its
    manifest keep is {sink, cutoff}, as unary tm's, plus k under topk."""
    kernel = spec["name"]
    explicit_keep = sink is not None
    sink = _sink(sink, spec, sinks)
    primary = spec["primary"] if primary is None else primary
    # score_fx is the primary's fixed-point form, not a kernel detail; duplex's dg_dcal
    # is the key column, which the engine emits beside dg_kcal when requested.
    details = [name for name in fields if name not in (primary, "score_fx")]
    if sink.kind == "rowstats":
        # RowStats carries best_dist/argbest/n_best/count only (spec §7), and duplex
        # model="pf"'s opt-in row column lse_dg_kcal (duplex-pf.md §9.4).
        details = [name for name in details if name == "lse_dg_kcal"]
    elif sink.kind in ("matrix", "tiles"):
        # Fixed-width fields only (D88); bool is K2's found (dimer-primer3.md §7, QD9).
        numeric = ("int32", "int64", "float64", "bool")
        ragged = [name for name in details if spec["fields"][name]["dtype"] not in numeric]
        if ragged:
            raise ParameterError(f"keep=sb.keep.{sink.kind}() holds numeric fields only; {ragged!r} "
                                 f"are list/string/binary fields; use sb.keep.each(), hits() or topk() for them")
    if not isinstance(pairs, Pairs):
        raise PairingError(f"{kernel} requires Pairs, received {type(pairs).__name__}; use "
                           f"sb.{kernel}(sb.pairs.zip(a, b)) or sb.pairs.one_to_many(q, b)")
    if pairs.kind in ("within", "cross") and sink.kind == "each":
        permitted = "use sb.keep.hits(cutoff), topk(k), rowstats(), tiles(), or matrix()"
        if explicit_keep:
            raise PairingError(f"keep=sb.keep.each() is not allowed for {pairs.kind} pairs: it would "
                               f"return one value per pair of the whole pair space; {permitted}")
        raise PairingError(f"{pairs.kind} requires keep; {permitted}")
    # Every option check precedes device resolution, which precedes the input scan.
    if options["device"].startswith("cuda") and "cuda" not in spec["backends"]:
        # No CPU fallback for an explicit device (D48 §6; cuda-backend.md).
        raise DeviceError(f"device={options['device']!r}: {kernel} is not yet available on the "
                          f"CUDA backend; use device='cpu' or device='auto'")
    device = resolve_device(options["device"], inputs=(pairs.a, pairs.b, pairs._i, pairs._j))
    stream = resolve_stream(options["stream"], device)
    ordinal = None if device is None else device["ordinal"]
    cutoff_int_value = None
    if spec is ALIGN:
        bound, cutoff_int_value = _align_cutoff(sink.cutoff)
    elif spec is DUPLEX or spec is TM:
        bound = cutoff_int_value = key_cutoff
    else:
        bound = _bound(sink.cutoff)
    backend = "cpu" if device is None else "cuda"
    if kernel == "duplex":
        # duplex's plan depends on its per-call pack (the C1 bound's eligibility and
        # certificate, duplex-mfe-bound.md §6.1), so it is resolved per call, not memoized.
        plan = _core._native_plan(kernel, sink.kind, options["strategy"], bound is not None,
                                  backend, duplex=duplex)
    else:
        plan = _plan(kernel, sink.kind, options["strategy"], bound is not None, backend)
    orientation = orientation if pairs.kind == "within" else None
    metadata = _metadata(spec, model, pairs, _pair_count(pairs, orientation), sink, check_bases,
                         fields, options, bound, plan, device, cutoff_int_value)
    if not cutoff_int:
        del metadata["model"]["keep"]["cutoff_int"]
    if spec is TM and sink.kind != "topk":
        # tm's keep is {sink, cutoff} (tm.md §6, §13.5); topk on Pairs also records its k.
        del metadata["model"]["keep"]["k"]
    start = perf_counter_ns()
    args = (pairs.a, pairs.b, pairs.kind, pairs._ii, pairs._jj, pairs._self_pairs)
    # Trailing native arguments by keyword: kernel, grain ("auto" for public
    # calls), device ordinal (None for the CPU), then mode, details, orientation.
    extra = {"kernel": kernel, "grain": "auto", "device": ordinal, "mode": mode,
             "details": details, "orientation": orientation}
    if options["progress"]:
        # Progress lines name the public function (cancellation-progress.md §4.1);
        # the setting is recorded nowhere. Passed only when on.
        extra["progress"] = kernel
    if ordinal is not None:
        extra["stream"] = stream.handle
        extra["stream_owner"] = stream.owner
    if align is not None:
        extra["align"] = align
    if duplex is not None:
        extra["duplex"] = duplex
    if sink.kind == "tiles":
        tiles_entry = _core._native_tiles if ordinal is None else _core._native_resident_tiles
        source = native_call(tiles_entry, *args, check_bases, plan["effective"],
                             options["threads"], sink.tile, **extra)
        _capture_replay(metadata, a=pairs.a, b=pairs.b, pairs=pairs,
                        scalar=pairs._scalar, scoring=replay_scoring, tile=sink.tile,
                        captured_inputs=None if ordinal is None else source.input_digests)
        # This closure retains facts only. It must not keep input Pairs/buffers
        # alive after the native stream completes or close() is called.
        result = TileIterator(source, lambda report: _manifest(metadata, report, _elapsed(start)))
        report = source.report
    else:
        native_function = _core._native_run if ordinal is None else _core._native_resident_run
        if ordinal is None:
            extra["input_digests"] = True
        native = native_call(native_function, *args, sink.kind, bound, check_bases,
                             plan["effective"], options["threads"], sink.k, **extra)
        # Successful calls fingerprint before delivery: both runs return the
        # facts of the inputs they validated. Inputs cannot change during
        # computation; failed calls retain native validation order.
        replay = _capture_replay(metadata, a=pairs.a, b=pairs.b, pairs=pairs,
                                 scalar=pairs._scalar, scoring=replay_scoring,
                                 captured_inputs=native["input_digests"])
        columns = {name: _column(owner) for name, owner in native["columns"].items()}
        layout = sink.kind
        if layout == "matrix":
            columns = {name: array.reshape(native["shape"]) for name, array in columns.items()}
        elif pairs._scalar and sink.kind == "each":
            # capture recorded this layout in call.layout already.
            layout = "scalar"
            columns = {name: (array if _is_ragged(array) else array.reshape(()))
                       for name, array in columns.items()}
        report = native["report"]
        result = Result._computed(layout, columns, native["shape"], pairs,
                                  _manifest(metadata, report, _elapsed(start)), kernel, replay,
                                  pair_indices=(native["pair_indices"] if ordinal is not None
                                                and pairs.kind == "index" else None))
    n_bad = report["flags"]["n_illegal_seqs"]
    if not check_bases and n_bad:
        warnings.warn(f"{kernel}(check_bases=False): {n_bad} sequences contain bytes other than "
                      "A, C, G, T; computing over raw bytes, case-sensitive",
                      IllegalBaseWarning, stacklevel=4)
    return result


_TM_EXAMPLE = "sb.tm(x, cond=sb.Conditions(temp_c=37, na_mM=50, oligo_nM=250))"


def _tm_cutoff(cutoff):
    """float(c) once (tm.md §6); keep.* already rejected bool, NaN and infinities."""
    if cutoff is None:
        return None
    try:
        value = float(cutoff)
    except OverflowError:
        raise ParameterError(f"cutoff={cutoff!r} is too large for a float64; tm compares "
                             "tm_c >= float(cutoff) in °C") from None
    if not math.isfinite(value):
        raise ParameterError(f"cutoff={cutoff!r}; use a finite int or float in °C")
    return value


def _tm_compat_pairs_error(alias):
    """tm.md §13.6 step 1.1 (PR1): the compat presets keep raising for a Pairs input."""
    return ParameterError(f"tm compat={alias!r} on Pairs is not yet available; for a given "
                          "pairing use compat=None (native register pairing), or pass one "
                          "sequence or a list of sequences")


def _tm_compat(compat):
    """§7 step 1.1 of tm-biopython.md: None (native), an available preset, a planned
    preset (not yet available) or an unknown value (did-you-mean)."""
    definition = TM["parameters"]["compat"]
    choices, available = definition["choices"], definition["available"]
    if compat is None or (type(compat) is str and compat in available):
        return compat
    if type(compat) is str and compat in choices:
        raise ParameterError(f"compat={compat!r} is not yet available for tm; available: "
                             f"{', '.join(repr(c) for c in available)}; use compat=None for "
                             "the native model")
    similar = _fp.close_matches(compat, choices, n=1) if isinstance(compat, str) else []
    hint = f"; did you mean {similar[0]!r}?" if similar else ""
    raise ParameterError(f"unknown compat={compat!r}{hint}; tm takes compat=None (native) or "
                         f"one of {', '.join(repr(c) for c in choices)}")


def _tm(x, *, cond, compat, params, fields, sink, options):
    """T1 tm: native (docs/wiki/thermo/tm.md), compat="biopython"
    (docs/wiki/thermo/tm-biopython.md) or compat="primer3" (docs/wiki/thermo/tm-primer3.md);
    checks in the §7 order, then one unary run. A native Pairs input takes the register
    pairing path (§13, ``_tm_pairs``)."""
    spec = TM
    # 1. compat, Pairs, params, cond (§7): every model option before any input.
    compat = _tm_compat(compat)
    if compat == "biopython":
        return _tm_biopython(x, cond=cond, params=params, fields=fields, sink=sink,
                             options=options)
    if compat == "primer3":
        return _tm_primer3(x, cond=cond, params=params, fields=fields, sink=sink,
                           options=options)
    if isinstance(x, Pairs):
        # 1.2: the register pairing of each requested pair (tm.md §13).
        return _tm_pairs(x, cond=cond, params=params, fields=fields, sink=sink,
                         options=options)
    available = spec["parameters"]["params"]["available"]
    if params in spec["parameters"]["params"]["choices"] and params not in available:
        # A planned preset (none since stage S9; kept for a later closed preset,
        # nndb2024-dna.md §5.5.3): the T1 loader's gate error names the build units not yet
        # accepted and the pending items (§4.7); this message only if its gate is open.
        _core._tm_params(params)
        raise ParameterError(f"params={params!r} is not yet available for tm; use "
                             f"params={available[0]!r}")
    if type(params) is not str or params not in available:
        similar = _fp.close_matches(params, available, n=1) if isinstance(params, str) else []
        hint = f"; did you mean {similar[0]!r}?" if similar else ""
        raise ParameterError(f"unknown params={params!r}{hint}; use one of "
                             f"{', '.join(repr(p) for p in available)}")
    if not isinstance(cond, Conditions):
        received = "missing" if cond is None else f"a {type(cond).__name__}"
        raise ParameterError(f"tm requires cond=sb.Conditions(...) (D38: no default conditions); "
                             f"cond is {received}. Corrected call: {_TM_EXAMPLE}")
    for name in ("dmso_pct", "formamide_M"):
        # Zero test by the bits: DAZ would read a subnormal as zero (§7.4).
        if not _fp.is_zero(getattr(cond, name)):
            raise ParameterError(f"cond.{name}={_fp.repr_float(getattr(cond, name))} is not supported by native "
                                 f"tm (D50: no {name} correction); use {name}=0")
    values = cond._asdict()
    oligo = values["oligo_nM"]
    pair = isinstance(oligo, tuple)
    # Na_eq (§3.2), host constants (§3.3) and the range check (§3.4) raise ParameterError.
    # The pack of the named preset; its cache is keyed on the preset too (nndb2024-dna.md
    # §5.5.2 item 3).
    pack = _core._tm_pack(values["temp_c"], values["na_mM"], values["k_mM"], values["tris_mM"],
                          values["mg_mM"], values["dntp_mM"], oligo[0] if pair else oligo,
                          oligo[1] if pair else None, params=params)
    # 2. fields, keep and cutoff (§6), then execution options.
    fields = _fields(spec, fields)
    model = {"params": {"name": params, "sha256": pack.params_sha256},
             "cond": dict(values, defaulted=cond.defaulted)}
    return _tm_execute(x, model, fields, sink, options, pack, _core._native_tm)


def _tm_execute(x, model, fields, sink, options, pack, native_entry):
    """The shared tail of native and compat tm (tm.md §6-§7, tm-biopython.md §6-§7):
    keep, cutoff and execution options, then the sequences and one unary run."""
    spec = TM
    if isinstance(sink, keep_module.Sink) and sink.kind not in spec["sinks"]:
        raise ParameterError(f"keep=sb.keep.{sink.kind}() is not available for tm on a Seq or "
                             "Seqs input: it returns one row per sequence and has no row LSE "
                             "(D47); use sb.keep.each(), sb.keep.each(cutoff=c) or "
                             "sb.keep.hits(c) (native tm on Pairs takes every pairwise keep)")
    if sink is not None and not isinstance(sink, keep_module.Sink):
        raise ParameterError(f"keep={sink!r}; use sb.keep.each(), sb.keep.each(cutoff=c) or "
                             "sb.keep.hits(c)")
    sink = _sink(sink, spec)
    # A Seq is Scalar under Each only. Under hits it is the ordinary Hits layout
    # with 0 or 1 rows, i = 0 for the hit, shape (1, 1) (QT18, QT20; tm.md §6).
    scalar = isinstance(x, (str, bytes)) and sink.kind == "each"
    cutoff = _tm_cutoff(sink.cutoff)
    options = resolve(options)
    # Device resolution after every option check and before the sequence scan; no data
    # moves and no CPU fallback for an explicit device (tm-cuda.md §7, D85 (4)).
    device = resolve_device(options["device"], inputs=(x,))
    stream = resolve_stream(options["stream"], device)
    ordinal = None if device is None else device["ordinal"]
    plan = _plan("tm", sink.kind, options["strategy"], cutoff is not None,
                 "cpu" if device is None else "cuda")
    # 3. Sequences: the engine scans the alphabet, then the length limits (§7).
    table = seqs(x)
    build = _BUILD
    layout = "scalar" if scalar else sink.kind
    model = dict(model, check_bases=True, fields=fields,
                 keep={"sink": sink.kind, "cutoff": cutoff})
    metadata = {
        "seqblaze": _build_metadata(build),
        "call": {"kernel": "tm", "layout": layout},
        "model": model,
        "inputs": {"a": _input_record(table)},
        "execution": _placement(build, device, options, plan),
        "flags": {"research_use_only": True},
        "citations": citation_keys(spec, model, fields, plan["effective"]),
    }
    start = perf_counter_ns()
    args = (table, pack, sink.kind, cutoff, options["threads"], list(fields), "auto")
    # Progress lines name the public function (cancellation-progress.md §4.1); passed only when on.
    progress = {"progress": "tm"} if options["progress"] else {}
    # Native tm and both compat presets have a device entry (tm-cuda.md, tm-compat-cuda.md).
    native = (native_call(native_entry, *args, **progress) if ordinal is None
              else native_call(getattr(_core, native_entry.__name__.replace("_native_", "_native_resident_", 1)), *args, ordinal, stream.handle, **progress))
    replay = _capture_replay(metadata, a=table, scalar=isinstance(x, (str, bytes)),
                             captured_inputs=None if ordinal is None else native["input_digests"])
    columns = {name: _column(owner) for name, owner in native["columns"].items()}
    if scalar:
        columns = {name: array.reshape(()) for name, array in columns.items()}
    # API §5 shape (n_a, n_b): (n, 1) for Each/Hits, n = the number of input sequences
    # (1 for a Seq under hits), and (1, 1) for Scalar (QT20); rows are sequences, so there
    # is no pair space and pairs is None.
    return Result._computed(layout, columns, tuple(native["shape"]), None,
                            _manifest(metadata, native["report"], _elapsed(start)), "tm", replay)


_TM_PAIRS_EXAMPLE = ("sb.tm(sb.pairs.zip(a, b), cond=sb.Conditions(temp_c=37, na_mM=50, "
                     "oligo_nM=250))")
_SIGN = 1 << 63


def _tm_pairs_key(cutoff):
    """The engine key of a tm-on-Pairs cutoff c, already float(c) and finite (tm.md §13.5,
    §13.11): the int64 order-preserving image of −c, −0.0 read as +0.0. The kernel maps
    tm_c the same way (every NaN above every key), so a row passes iff key(tm_c) <= key(c),
    that is iff tm_c >= c in IEEE comparison. Bit operations only, so the key does not
    depend on the caller's FP environment (text-nulls-fp-environment.md §7.4)."""
    if cutoff is None:
        return None
    negated = _fp.bits(cutoff) ^ _SIGN
    magnitude = negated & (_SIGN - 1)
    return magnitude if negated & _SIGN == 0 else -magnitude


def _tm_pairs_params(params):
    """§13.6 step 1.3 (PR1): an available preset name ("santalucia2004", "nndb2024-dna") is
    K1's compiled preset through the gated loader (nndb2024-dna.md §5.5.3); a planned preset
    (in ``choices``, not in ``available``; none since stage S9) is not yet available (the
    loader's gate error names the missing units and pending items, §4.7); anything else, a
    testing pack handle included, raises (did-you-mean for a near name)."""
    definition = TM["parameters"]["params"]
    available = definition["available"]
    if type(params) is str and params in available:
        # The gate (a K1 build unit not yet accepted) is decided by the loader alone.
        return _core._duplex_preset(params)
    if type(params) is str and params in definition["choices"]:
        _core._duplex_preset(params)  # the K1 gate's error while it is closed (nndb2024-dna.md §4.7)
        raise ParameterError(f"params={params!r} is not yet available for tm on Pairs; use "
                             "params='santalucia2004'")
    similar = _fp.close_matches(params, definition["choices"], n=1) if isinstance(params, str) else []
    hint = f"; did you mean {similar[0]!r}?" if similar else ""
    raise ParameterError(f"unknown params={params!r}{hint}; tm on Pairs takes a preset name: "
                         f"{', '.join(repr(p) for p in available)}")


def _tm_pairs(pairs, *, cond, params, fields, sink, options):
    """sb.tm on a Pairs input: the register pairing of each requested pair on the compiled
    K1 pack of the ``params`` preset, santalucia2004 or nndb2024-dna (docs/wiki/thermo/tm.md
    §13; docs/wiki/thermo/nndb2024-dna.md §3.8). Checks in the §13.6 order after
    step 1.1 (compat=None): params, cond, dmso/formamide, the per-call pack (K1's range
    check, naming tm), fields, keep (every pairwise sink, ``TM["pair_sinks"]``), the pair
    kind, the cutoff, the execution options; then the shared pairwise path (device
    resolution, equal lengths per pair, then the sequences), on the CPU
    or CUDA (docs/wiki/gpu/tm-pairs-cuda.md §8)."""
    spec = TM
    pack = _tm_pairs_params(params)
    if not isinstance(cond, Conditions):
        received = "missing" if cond is None else f"a {type(cond).__name__}"
        raise ParameterError(f"tm requires cond=sb.Conditions(...) (D38: no default conditions); "
                             f"cond is {received}. Corrected call: {_TM_PAIRS_EXAMPLE}")
    for name in ("dmso_pct", "formamide_M"):
        # Zero test by the bits: DAZ would read a subnormal as zero (§7.4).
        if not _fp.is_zero(getattr(cond, name)):
            raise ParameterError(f"cond.{name}={_fp.repr_float(getattr(cond, name))} is not supported by native "
                                 f"tm (D50: no {name} correction); use {name}=0")
    values = cond._asdict()
    oligo = values["oligo_nM"]
    pair = isinstance(oligo, tuple)
    # T1 §3.1 checks, Na_eq, s, κ₄, κ₁, G_e(T) and K1's range check (PR4) raise
    # ParameterError naming tm.
    call = _core._duplex_call(pack, values["temp_c"], values["na_mM"], values["k_mM"],
                              values["tris_mM"], values["mg_mM"], values["dntp_mM"],
                              oligo[0] if pair else oligo, oligo[1] if pair else None,
                              model="mfe", kernel="tm")
    # 1.7 fields, keep (every pairwise sink, AGENTS §2 principle 4), the pair kind.
    fields = _fields(spec, fields)
    sinks = spec["pair_sinks"]
    explicit_keep = sink is not None
    checked = _sink(sink, spec, sinks)
    if pairs.kind in ("within", "cross") and checked.kind == "each":
        # API §1.3, D87: one value per pair of the whole pair space is never implied.
        given = "keep=sb.keep.each() is not allowed" if explicit_keep else "keep is required"
        raise PairingError(f"tm on {pairs.kind} pairs: {given}; use sb.keep.hits(cutoff) (the pairs "
                           "with tm_c >= cutoff), topk(k), rowstats(), tiles(), or matrix()")
    # 1.8 the cutoff, float(c) once (§6), then the execution options. matrix() and tiles()
    # take no cutoff (_sink).
    cutoff = _tm_cutoff(checked.cutoff)
    if cutoff is not None:
        if checked.kind == "hits":
            checked = keep_module.hits(cutoff)
        elif checked.kind == "topk":
            checked = keep_module.topk(checked.k, cutoff=cutoff)
        elif checked.kind == "rowstats":
            checked = keep_module.rowstats(cutoff=cutoff)
        else:
            checked = keep_module.each(cutoff=cutoff)
    options = resolve(options)
    model = {"params": {"name": params, "sha256": call.params_sha256},
             "cond": dict(values, defaulted=cond.defaulted)}
    return _run(spec, pairs, model, check_bases=True, fields=fields, sink=checked,
                options=options, orientation=None, duplex=call,
                key_cutoff=_tm_pairs_key(cutoff), cutoff_int=False, sinks=sinks)


_BIOPYTHON = "compat='biopython'"
_PRIMER3 = "compat='primer3'"
# What each compat tool returns, for the fields message (tm-biopython.md §6, tm-primer3.md §6).
_COMPAT_RETURNS = {"biopython": "Biopython's Tm_NN returns Tm", "primer3": "primer3's calc_tm returns Tm"}


def _tm_compat_fields(value, alias):
    """tm-biopython.md §6, tm-primer3.md §6: tm compat has tm_c only; "default" and "all"
    give it."""
    allowed = TM["parameters"]["compat"]["fields"][alias]
    if isinstance(value, str):
        names = allowed if value in ("default", "all") else (value,)
    elif isinstance(value, (list, tuple)):
        names = value
    else:
        raise ParameterError(f"fields={value!r}; use fields='default', 'all', or {list(allowed)!r}")
    if not names:
        raise ParameterError(f"fields cannot be empty; use fields={list(allowed)!r}")
    for name in names:
        if type(name) is not str or name not in allowed:
            raise ParameterError(f"field {name!r} is not available under compat={alias!r}: it "
                                 f"has fields={list(allowed)!r} only ({_COMPAT_RETURNS[alias]}); "
                                 "use fields='default' or compat=None for the native fields")
    return tuple(allowed)


def _tm_biopython(x, *, cond, params, fields, sink, options):
    """tm compat="biopython" (docs/wiki/thermo/tm-biopython.md): Biopython 1.88 Tm_NN(seq)
    with a correctly rounded log; §7 order after step 1.1."""
    # 1.2 Pairs, 1.3 params (native-only, QB1), 1.4 cond, 1.5 dmso/formamide (P7).
    if isinstance(x, Pairs):
        raise _tm_compat_pairs_error("biopython")
    if type(params) is not str or params != TM["parameters"]["params"]["default"]:
        raise ParameterError(f"params={params!r}: params is a native-only option (D48 §2); "
                             f"{_BIOPYTHON} uses Biopython's default table DNA_NN3; omit params")
    if cond is None:
        cond = _preset_conditions("biopython")  # D38: Biopython's bare call
    elif not isinstance(cond, Conditions):
        raise ParameterError(f"cond={cond!r}; {_BIOPYTHON} takes cond=None (Biopython's "
                             "defaults) or sb.Conditions(...)")
    for name in ("dmso_pct", "formamide_M"):
        if not _fp.is_zero(getattr(cond, name)):  # by the bits (§7.4)
            raise ParameterError(f"cond.{name}={_fp.repr_float(getattr(cond, name))} is not yet available under "
                                 f"{_BIOPYTHON} (Biopython applies it through chem_correction, "
                                 f"not Tm_NN); use {name}=0")
    # 1.6 B1-B10 (§3 mapping: scalar C -> dnac1 = dnac2 = C*0.5 (P3), a pair verbatim (P4)).
    values = cond._asdict()
    oligo = values["oligo_nM"]
    # A scalar C is halved natively (C*0.5 in E0: a subnormal half is rounded, §7.4).
    d1, d2 = oligo if isinstance(oligo, tuple) else (oligo, None)
    pack = _core._tm_biopython_pack(values["na_mM"], values["k_mM"], values["tris_mM"],
                                    values["mg_mM"], values["dntp_mM"], d1, d2)
    # 1.7 fields, then keep, cutoff and execution options (shared tail).
    fields = _tm_compat_fields(fields, "biopython")
    compat = TM["parameters"]["compat"]
    model = {"compat": {"alias": "biopython", "id": compat["id"]["biopython"]},
             "params": {"name": pack.params, "sha256": pack.params_sha256},
             "cond": dict(values, defaulted=cond.defaulted)}
    return _tm_execute(x, model, fields, sink, options, pack, _core._native_tm_biopython)


_DUPLEX_EXAMPLE = ("sb.duplex(sb.pairs.zip(a, b), cond=sb.Conditions(temp_c=37, na_mM=50, "
                   "oligo_nM=250))")
#: The unit guard of a duplex cutoff, kcal/mol (duplex-mfe.md §8.3).
_DUPLEX_CUTOFF_LIMIT = 1000


def _duplex_cutoff(cutoff):
    """c_int = max{d : fl_RN(d/100) <= c} of a duplex cutoff c (duplex-mfe.md §8.3).

    Exact in every caller floating-point environment (text-nulls-fp-environment.md
    §7.4): c is the exact rational of the int, or of the float read from its bits;
    d0 = floor(100 c) in rational arithmetic; then d0 + 1 if
    exact(fl_RN((d0 + 1)/100)) <= c, with fl_RN computed in integer arithmetic
    (``_fp.fl_rn``), else d0. No float is divided, compared or decomposed with float
    arithmetic: CPython's int / int division follows the caller's rounding mode, and
    Fraction(x) of a subnormal float x reads it as zero under DAZ. |c| > 1000 kcal/mol
    (compared exactly) raises UnitError; keep.* has already rejected NaN, infinities
    and bool, and gives an int or a float.
    """
    if cutoff is None:
        return None
    c = _duplex_unit_guard(cutoff, "dg_kcal")
    d0 = math.floor(c * 100)
    return d0 + 1 if _fp.exact(_fp.fl_rn(d0 + 1, 100)) <= c else d0


def _duplex_unit_guard(cutoff, primary):
    """The exact value of a duplex cutoff (int or float, read from its bits); UnitError
    beyond ±1000 kcal/mol."""
    c = _fp.exact(cutoff) if isinstance(cutoff, float) else Fraction(cutoff)
    if abs(c) > _DUPLEX_CUTOFF_LIMIT:
        raise UnitError(f"cutoff={_fp.repr_value(cutoff)} kcal/mol is beyond the duplex unit "
                        f"guard of ±{_DUPLEX_CUTOFF_LIMIT} kcal/mol: the cutoff is in kcal/mol "
                        f"({primary} <= cutoff), not dcal/mol or cal/mol; e.g. "
                        "sb.keep.hits(-10.0)")
    return c


def _pf_key(value):
    """The int64 order-preserving image of a binary64 value (duplex-pf.md §9.2): its bits
    when the sign bit is clear, minus its magnitude bits otherwise; −0.0 reads as +0.0.
    Bit operations only, so the key does not depend on the caller's FP environment
    (text-nulls-fp-environment.md §7.4): an int (|c| ≤ 1000) converts exactly through
    ``_fp.integer``, a float is read from its encoding."""
    bits = _fp.bits(value) if isinstance(value, float) else _fp.bits(_fp.integer(value))
    if bits == 1 << 63:
        bits = 0  # −0.0 reads as +0.0
    return bits if bits >> 63 == 0 else -(bits & 0x7FFF_FFFF_FFFF_FFFF)


def _pf_cutoff(cutoff):
    """The engine key of a model="pf" cutoff c (duplex-pf.md §9.3): c as binary64 (exact for
    an int of magnitude ≤ 1000), −0.0 read as +0.0; rows pass iff key(dg) ≤ key(c).
    |c| > 1000 kcal/mol raises UnitError; keep.* has already rejected NaN, ±∞ and bool."""
    if cutoff is None:
        return None
    _duplex_unit_guard(cutoff, "dg_ens_kcal")
    return _pf_key(cutoff)


def _duplex_params(params):
    """§9 step 1 params: a testing pack handle, an available preset ("santalucia2004",
    "nndb2024-dna": its generated pack; the Rust loader raises "not yet available" only while
    its gate is closed, nndb2024-dna.md §5.5.3), a planned preset (in ``choices``, not in
    ``available``; none since stage S9: the loader's gate error names the missing units and
    pending items, §4.7), or an unknown value (did-you-mean)."""
    if isinstance(params, _core._DuplexPack):
        return params
    choices = DUPLEX["parameters"]["params"]["choices"]
    available = DUPLEX["parameters"]["params"]["available"]
    if type(params) is str and params in available:
        # The generated preset, through the guarded loader; the gate (a build unit not yet
        # accepted, or a pending item) is decided there alone and raises "not yet available".
        return _core._duplex_preset(params)
    if type(params) is str and params in choices:
        _core._duplex_preset(params)  # the K1 gate's error while it is closed (nndb2024-dna.md §4.7)
        raise ParameterError(f"params={params!r} is not yet available for duplex; use "
                             "params='santalucia2004'")
    similar = _fp.close_matches(params, choices, n=1) if isinstance(params, str) else []
    hint = f"; did you mean {similar[0]!r}?" if similar else ""
    raise ParameterError(f"unknown params={params!r}{hint}; duplex takes params="
                         f"{', '.join(repr(c) for c in choices)}")


def _duplex_fields(value, model):
    """fields= of duplex for a model: reserved and deferred names first (D47, QK14), then
    the shared resolution with did-you-mean (fields of the other model do not exist)."""
    names = (value,) if isinstance(value, str) else value if isinstance(value, (list, tuple)) else ()
    reserved = DUPLEX["reserved_fields"][model]
    for name in names:
        if type(name) is str and name in reserved:
            raise ParameterError(f"field {name!r} is {reserved[name]} for duplex model={model!r}; "
                                 f"use fields={list(DUPLEX['mode_fields'][model])!r} or a subset")
    return _fields(DUPLEX, value, model, option="model", primary=DUPLEX["model_primary"][model])


def _duplex(pairs, *, cond, model, compat, end, orientation, params, fields, sink, options):
    """K1 sb.duplex(model="mfe") (docs/wiki/thermo/duplex-mfe.md) and K3
    sb.duplex(model="pf") (docs/wiki/thermo/duplex-pf.md). Checks in the K1 §9 order with
    the K3 §10 steps: model, compat, end, params (the gate), orientation; cond and the
    per-call pack (T1 checks, Na_eq, host constants, range, K3's exponent check); fields,
    keep and the pair kind, then lse_dg_kcal outside rowstats (both models) and, for K1
    with lse_dg_kcal, the duplex-mfe.md §8.6.2 range checks; the cutoff; the
    execution options; then the shared pairwise path (device, pairs, sequences)."""
    spec = DUPLEX
    # 1. Model options.
    if type(model) is not str or model not in spec["parameters"]["model"]["choices"]:
        raise ParameterError(f"model={model!r}; use model='mfe' (minimum free energy) or "
                             "model='pf' (partition function)")
    if compat is not None and model == "pf":
        raise ParameterError(f"compat={compat!r} does not apply to model='pf': the partition "
                             "function is a native model only (D48 §2); use compat=None, or "
                             "model='mfe' for a compat preset")
    if compat is not None:
        choices = spec["parameters"]["compat"]["choices"]
        if type(compat) is str and compat in spec["parameters"]["compat"]["available"]:
            return _duplex_primer3(pairs, cond=cond, end=end, orientation=orientation,
                                   params=params, fields=fields, sink=sink, options=options)
        if type(compat) is str and compat in choices:
            raise ParameterError(f"compat={compat!r} is not yet available for duplex; use "
                                 "compat=None for the native model")
        similar = _fp.close_matches(compat, choices, n=1) if isinstance(compat, str) else []
        hint = f"; did you mean {similar[0]!r}?" if similar else ""
        raise ParameterError(f"unknown compat={compat!r}{hint}; duplex takes compat=None or "
                             f"{', '.join(repr(c) for c in choices)}")
    ends = spec["parameters"]["end"]["choices"]
    if type(end) is not str or end not in ends:
        raise ParameterError(f"end={end!r}; native duplex takes end='any' (end1/end2 belong to "
                             "compat='primer3')")
    if end != "any":  # QK16: only a non-default value raises
        raise ParameterError(f"end={end!r} applies to compat='primer3' only (D48 §2); native "
                             "duplex scores every structure, use end='any'")
    pack = _duplex_params(params)
    orientation = _choice(spec, "orientation", orientation)
    # cond (D38), dmso/formamide (D50), then the per-call pack.
    if not isinstance(cond, Conditions):
        received = "missing" if cond is None else f"a {type(cond).__name__}"
        raise ParameterError(f"duplex requires cond=sb.Conditions(...) (D38: no default "
                             f"conditions); cond is {received}. Corrected call: {_DUPLEX_EXAMPLE}")
    for name in ("dmso_pct", "formamide_M"):
        # Zero test by the bits: DAZ would read a subnormal as zero (§7.4).
        if not _fp.is_zero(getattr(cond, name)):
            raise ParameterError(f"cond.{name}={_fp.repr_float(getattr(cond, name))} is not supported "
                                 f"by native duplex (D50: no {name} correction); use {name}=0")
    values = cond._asdict()
    oligo = values["oligo_nM"]
    pair = isinstance(oligo, tuple)
    # T1 §3.1 checks, Na_eq, s, κ₄, κ₁, G_e(T) and the §4.4 range check raise ParameterError.
    call = _core._duplex_call(pack, values["temp_c"], values["na_mM"], values["k_mM"],
                              values["tris_mM"], values["mg_mM"], values["dntp_mM"],
                              oligo[0] if pair else oligo, oligo[1] if pair else None, model=model)
    # 2. fields, keep and the pair kind, the cutoff, then the execution options.
    requested = fields
    fields = _duplex_fields(fields, model)
    explicit_keep = sink is not None
    checked = _sink(sink, spec)
    if not isinstance(pairs, Pairs):
        raise PairingError(f"duplex requires Pairs, received {type(pairs).__name__}; use "
                           f"{_DUPLEX_EXAMPLE} or sb.pairs.one_to_many(q, b)")
    if pairs.kind in ("within", "cross") and checked.kind == "each":
        permitted = "use sb.keep.hits(cutoff), topk(k), rowstats(), tiles(), or matrix()"
        if explicit_keep:
            raise PairingError(f"keep=sb.keep.each() is not allowed for {pairs.kind} pairs: it would "
                               f"return one value per pair of the whole pair space; {permitted}")
        raise PairingError(f"{pairs.kind} requires keep; {permitted}")
    if checked.kind != "rowstats" and "lse_dg_kcal" in fields:
        # duplex-pf.md §9.1, §9.4 (QP9), duplex-mfe.md §8.6.4 (LK4): "all" outside RowStats
        # is the per-pair fields.
        if isinstance(requested, str) and requested == "all":
            fields = tuple(name for name in fields if name != "lse_dg_kcal")
        else:
            raise ParameterError(f"field 'lse_dg_kcal' is a per-row column of "
                                 f"keep=sb.keep.rowstats() only, not of keep=sb.keep."
                                 f"{checked.kind}(); use sb.keep.rowstats() or drop it from fields")
    if model == "mfe" and "lse_dg_kcal" in fields:
        # duplex-mfe.md §8.6.2 (LK3): K3's exponent-range and finalize-range checks with
        # K1's bound B, only when the field is requested, before the cutoff and device.
        call = _core._duplex_lse(call, values["temp_c"], values["na_mM"], values["k_mM"],
                                 values["tris_mM"], values["mg_mM"], values["dntp_mM"],
                                 oligo[0] if pair else oligo, oligo[1] if pair else None)
    elif model == "pf" and "lse_dg_kcal" in fields:
        # duplex-pf-bound.md §6.1 (QZ7): RowStats lse_dg_kcal sums every partner, so the
        # C2 bound is unavailable (no_bound) and its certificate is never built.
        call = _core._duplex_pf_lse(call)
    key_cutoff = _pf_cutoff(checked.cutoff) if model == "pf" else _duplex_cutoff(checked.cutoff)
    options = resolve(options)
    manifest_model = {"model": model, "params": {"name": call.params, "sha256": call.params_sha256},
                      "cond": dict(values, defaulted=cond.defaulted), "end": end,
                      "orientation": orientation}
    if model == "pf":
        manifest_model["numerics"] = spec["numerics"]["pf"]
    else:
        manifest_model["tie_rule"] = spec["tie_rule"]
    # duplex is symmetric (§6.4): orientation never reaches the engine.
    return _run(spec, pairs, manifest_model, check_bases=True, fields=fields, sink=sink,
                options=options, orientation=None, duplex=call, key_cutoff=key_cutoff,
                primary=spec["model_primary"][model], cutoff_int=model == "mfe")


_DIMER = "duplex compat='primer3'"
_DIMER_CALLS = "calc_heterodimer, calc_homodimer and calc_end_stability"


def _duplex_primer3_end(end):
    """§8 step 1.1: the three `end` values of D48 §2 (QD2)."""
    ends = DUPLEX["parameters"]["end"]["choices"]
    if type(end) is not str or end not in ends:
        raise ParameterError(f"end={end!r}; {_DIMER} takes end={', '.join(repr(e) for e in ends)} "
                             "(heterodimer, end stability of (a, b), end stability of (b, a))")
    return end


def _duplex_primer3_cond(cond):
    """§8 step 1.3-1.4: the conditions (None takes Conditions.preset("primer3")), a pair
    oligo_nM and the unsupported fields, then the per-call pack H1-H8 (its errors in step
    order). Returns (cond, values, pack)."""
    if cond is None:
        cond = _preset_conditions("primer3")  # D38: the dimer calls' documented defaults
    elif not isinstance(cond, Conditions):
        raise ParameterError(f"cond={cond!r}; {_DIMER} takes cond=None (primer3's defaults) "
                             "or sb.Conditions(...)")
    values = cond._asdict()
    oligo = values["oligo_nM"]
    if isinstance(oligo, tuple):
        # QD5. No computed total: a float + would round in the caller's environment
        # (text-nulls-fp-environment.md §7.4).
        raise ParameterError(f"cond.oligo_nM={_fp.repr_value(oligo)} is a pair, but primer3's "
                             f"{_DIMER_CALLS} take one DNA concentration (dna_conc, nM); give "
                             "a number, e.g. oligo_nM=50")
    for name in ("dmso_pct", "formamide_M"):
        # Zero test by the bits: DAZ would read a subnormal as zero (§7.4).
        if not _fp.is_zero(values[name]):
            raise ParameterError(f"cond.{name}={_fp.repr_float(values[name])} is not supported by "
                                 f"{_DIMER}: primer3's {_DIMER_CALLS} have no {name} "
                                 f"correction; use {name}=0")
    # H1-H8 (§5): mv = Na + K + Tris/2 is formed in the pack, never here (§13.1).
    pack = _core._duplex_primer3_pack(values["na_mM"], values["k_mM"], values["tris_mM"],
                                      values["mg_mM"], values["dntp_mM"], oligo,
                                      values["temp_c"])
    return cond, values, pack


def _primer3_handle():
    """Step 1.6 of the primer3 presets (hairpin-primer3.md §7, dimer-primer3.md §8): the
    parameter set from the D63 loader (primer3-parameter-loader.md §7, §8).

    The loader module's first import and every lookup that is not a cache hit read file
    status information, which sets the inexact flag in the caller's floating-point state
    (text-nulls-fp-environment.md §8.6). So the whole step, the import included, runs
    inside a caller-state bracket (§7.2, §7.3 item 5), which writes the caller's state back
    when the handle is returned or a loader ParameterError propagates. It installs nothing:
    the lookup runs in the caller's environment, its floats reach no value, and the parse
    holds its own guard. The preset entries import the loader module nowhere else, so a call
    that fails before step 1.6 never imports it."""
    with _fp.bracket():
        from . import _p3params
        return _p3params.handle()


def _duplex_primer3_fields(value):
    """§7: "default" and "all" give the five fields; a list names a non-empty subset, the
    primary dg_kcal is always emitted, repeated names count once, columns in table order.
    structure is reserved (D47), near_tie not provided (QD8); the native fields raise
    naming them as fields of the native models (D47)."""
    preset = DUPLEX["compat"]["primer3"]
    allowed = preset["fields"]
    if isinstance(value, str):
        names = allowed if value in ("default", "all") else (value,)
    elif isinstance(value, (list, tuple)):
        names = value
    else:
        raise ParameterError(f"fields={value!r}; use fields='default', 'all', or a subset of "
                             f"{list(allowed)!r}")
    if not names:
        raise ParameterError(f"fields cannot be empty; use fields={list(allowed)!r} or a subset")
    reserved = preset["reserved_fields"]
    native = [name for model in ("mfe", "pf") for name in DUPLEX["mode_fields"][model]
              if name not in allowed]
    for name in names:
        if type(name) is str and name in reserved:
            raise ParameterError(f"field {name!r} is {reserved[name]} for {_DIMER}; use fields="
                                 f"{list(allowed)!r} or a subset")
        if type(name) is str and name in native:
            raise ParameterError(f"field {name!r} is a field of the native duplex models "
                                 f"(compat=None), not of {_DIMER}; use fields="
                                 f"{list(allowed)!r} or a subset")
        if type(name) is not str or name not in allowed:
            similar = _fp.close_matches(name, allowed, n=1) if isinstance(name, str) else []
            hint = f"; did you mean {similar[0]!r}?" if similar else ""
            raise ParameterError(f"unknown field {name!r}{hint}; {_DIMER} supports fields="
                                 f"{list(allowed)!r}")
    requested = set(names)
    primary = preset["primary"]
    return tuple(name for name in allowed if name == primary or name in requested)


def _duplex_primer3_cutoff(cutoff):
    """The engine key of a cutoff c (§7, QD12): the K3 bit key of c as binary64 (exact for
    an int of magnitude ≤ 1000), −0.0 read as +0.0; rows pass iff dg_kcal <= c. |c| > 1000
    kcal/mol raises UnitError; keep.* has already rejected NaN, ±∞ and bool."""
    if cutoff is None:
        return None
    _duplex_unit_guard(cutoff, "dg_kcal")
    return _pf_key(cutoff)


def _duplex_primer3(pairs, *, cond, end, orientation, params, fields, sink, options):
    """K2 sb.duplex(compat="primer3") (docs/wiki/thermo/dimer-primer3.md): primer3-py
    2.3.1's calc_heterodimer (end="any"), calc_end_stability (end="end1"; "end2" on the
    swapped pair) and, for identical sequences, calc_homodimer, max_loop 30, with a
    correctly rounded log, over primer3-py's own parameter files read at run time (D63).
    Checks in the §8 order (model and compat were checked by the caller): the native
    parameter-set option, end, orientation; the call shape; cond and the per-call pack;
    fields, the cutoff and the execution options; the parameter loader (also for an
    empty input; inside the caller-state bracket of ``_primer3_handle``); device
    resolution in ``_run`` (primer3-cuda.md §9); then the engine's sequence checks."""
    spec = DUPLEX
    # 1.1 The native parameter-set option is only compared with its default, never
    # resolved, so the native gate (D98) never fires for a preset call.
    default = spec["parameters"]["params"]["default"]
    if not (type(params) is str and params == default):
        shown = "a testing pack handle" if isinstance(params, _core._DuplexPack) else repr(params)
        raise ParameterError(f"params={shown}: params is a native-only option (D48 §2); "
                             f"{_DIMER} uses primer3-py's own parameter files; omit params")
    end = _duplex_primer3_end(end)
    orientation = _choice(spec, "orientation", orientation)
    # 1.2 The call shape: independent of the data and of the installed primer3-py.
    if not isinstance(pairs, Pairs):
        raise PairingError(f"duplex requires Pairs, received {type(pairs).__name__}; use "
                           "sb.duplex(sb.pairs.zip(a, b), compat='primer3') or "
                           "sb.pairs.one_to_many(q, b)")
    kind = sink.kind if isinstance(sink, keep_module.Sink) else "each" if sink is None else None
    if pairs.kind in ("within", "cross") and kind == "each":
        permitted = "use sb.keep.hits(cutoff), topk(k), rowstats(), tiles(), or matrix()"
        if sink is not None:
            raise PairingError(f"keep=sb.keep.each() is not allowed for {pairs.kind} pairs: it would "
                               f"return one value per pair of the whole pair space; {permitted}")
        raise PairingError(f"{pairs.kind} requires keep; {permitted}")
    # 1.3-1.4 Conditions and the per-call constants H1-H8.
    cond, values, pack = _duplex_primer3_cond(cond)
    # 1.5 fields, keep and the cutoff, then the execution options.
    fields = _duplex_primer3_fields(fields)
    checked = _sink(sink, spec)
    key_cutoff = _duplex_primer3_cutoff(checked.cutoff)
    options = resolve(options)
    # 1.6 The parameter loader (loader §8): its install hint does not depend on the data.
    # An explicit CUDA device is resolved by _run after it (primer3-cuda.md §9, QPC-13).
    handle = _primer3_handle()
    call = _core._duplex_primer3_call(pack, handle, end)
    model = {"model": "mfe",
             "compat": {"alias": "primer3", "id": spec["compat"]["primer3"]["id"]},
             "params": {"name": handle.name, "sha256": handle.sha256},
             "cond": dict(values, defaulted=cond.defaulted), "end": end,
             "orientation": orientation}
    # Asymmetric (§2): orientation reaches the engine under within. No keep.cutoff_int,
    # tie_rule or numerics (QD12).
    return _run(spec, pairs, model, check_bases=True, fields=fields, sink=sink,
                options=options, orientation=orientation, duplex=call, key_cutoff=key_cutoff,
                primary=spec["compat"]["primer3"]["primary"], cutoff_int=False)


def _tm_primer3(x, *, cond, params, fields, sink, options):
    """tm compat="primer3" (docs/wiki/thermo/tm-primer3.md): primer3-py 2.3.1 calc_tm with
    its default method pair and a correctly rounded log and log10; §7 order after step 1.1.
    Needs no primer3-py at run time: calc_tm reads no data file (§13, D63 scope note)."""
    # 1.2 Pairs, 1.3 params (native-only), 1.4 cond, 1.5 a pair oligo_nM (QPT3).
    if isinstance(x, Pairs):
        raise _tm_compat_pairs_error("primer3")
    if type(params) is not str or params != TM["parameters"]["params"]["default"]:
        raise ParameterError(f"params={params!r}: params is a native-only option (D48 §2); "
                             f"{_PRIMER3} uses primer3's SantaLucia 1998 table; omit params")
    if cond is None:
        cond = _preset_conditions("primer3")  # D38: calc_tm's documented defaults
    elif not isinstance(cond, Conditions):
        raise ParameterError(f"cond={cond!r}; {_PRIMER3} takes cond=None (primer3's defaults) "
                             "or sb.Conditions(...)")
    values = cond._asdict()
    oligo = values["oligo_nM"]
    if isinstance(oligo, tuple):
        # The total hint is fl_RN of the exact sum, rounded in integer arithmetic: a float
        # + would round in the caller's environment (text-nulls-fp-environment.md §7.4).
        total = _fp.fl_rn_fraction(_fp.exact(oligo[0]) + _fp.exact(oligo[1]))
        raise ParameterError(f"cond.oligo_nM={_fp.repr_value(oligo)} is a pair, but primer3's calc_tm "
                             "takes one total strand concentration (dna_conc, nM); give a number, "
                             f"e.g. oligo_nM={_fp.repr_float(total)} for the total of both strands")
    # 1.6 P1-P13 (§3 mapping: mv = Na + K + Tris/2 in the pack, dv = Mg, dntp, dna_conc = C,
    # dmso_conc, formamide_conc); P6, P7, P9, P11 raise ParameterError in that order.
    pack = _core._tm_primer3_pack(values["na_mM"], values["k_mM"], values["tris_mM"],
                                  values["mg_mM"], values["dntp_mM"], oligo, values["dmso_pct"],
                                  values["formamide_M"])
    # 1.7 fields, then keep, cutoff and execution options (shared tail).
    fields = _tm_compat_fields(fields, "primer3")
    compat = TM["parameters"]["compat"]
    model = {"compat": {"alias": "primer3", "id": compat["id"]["primer3"]},
             "params": {"name": pack.params, "sha256": pack.params_sha256},
             "cond": dict(values, defaulted=cond.defaulted)}
    return _tm_execute(x, model, fields, sink, options, pack, _core._native_tm_primer3)


_HAIRPIN = "hairpin compat='primer3'"


def _hairpin_compat(compat):
    """§7 step 1.1 of hairpin-primer3.md: "primer3", else ParameterError (did-you-mean)."""
    choices = HAIRPIN["parameters"]["compat"]["choices"]
    if type(compat) is str and compat in choices:
        return compat
    if compat is None:
        raise ParameterError("hairpin requires compat='primer3' in v1: the only hairpin model is "
                             "primer3-py 2.3.1's calc_hairpin; native single-strand folding is "
                             "planned for v1.1. Corrected call: sb.hairpin(x, compat='primer3')")
    similar = _fp.close_matches(compat, choices, n=1) if isinstance(compat, str) else []
    hint = f"; did you mean {similar[0]!r}?" if similar else ""
    raise ParameterError(f"unknown compat={compat!r}{hint}; hairpin takes compat="
                         f"{', '.join(repr(c) for c in choices)}")


def _hairpin_fields(value):
    """§6: "default" and "all" give the five fields; a list names a subset, and the primary
    dg_kcal, which every cutoff compares, is always emitted (as tm_c is for tm and dist for
    the edit distances); "structure" is reserved (D47); another name raises listing the
    five. Canonical order."""
    allowed = tuple(HAIRPIN["fields"])
    if isinstance(value, str):
        names = allowed if value in ("default", "all") else (value,)
    elif isinstance(value, (list, tuple)):
        names = value
    else:
        raise ParameterError(f"fields={value!r}; use fields='default', 'all', or a subset of "
                             f"{list(allowed)!r}")
    if not names:
        raise ParameterError(f"fields cannot be empty; use fields={list(allowed)!r} or a subset")
    reserved = HAIRPIN["reserved_fields"]
    for name in names:
        if type(name) is str and name in reserved:
            raise ParameterError(f"field {name!r} is {reserved[name]} for hairpin; use fields="
                                 f"{list(allowed)!r} or a subset")
        if type(name) is not str or name not in allowed:
            raise ParameterError(f"field {name!r} is not available for {_HAIRPIN}; use fields="
                                 f"{list(allowed)!r} or a subset")
    requested = set(names)
    primary = HAIRPIN["primary"]
    return tuple(name for name in allowed if name == primary or name in requested)


def _hairpin_cutoff(cutoff):
    """float(c) once (§6): dg_kcal <= float(c) on the exposed column; keep.* already
    rejected bool, NaN and infinities."""
    if cutoff is None:
        return None
    try:
        value = float(cutoff)
    except OverflowError:
        raise ParameterError(f"cutoff={cutoff!r} is too large for a float64; hairpin compares "
                             "dg_kcal <= float(cutoff) in kcal/mol") from None
    if not math.isfinite(value):
        raise ParameterError(f"cutoff={cutoff!r}; use a finite int or float in kcal/mol")
    return value


def _hairpin_cond(cond):
    """§7 steps 1.2-1.4: the conditions (None takes Conditions.preset("primer3")), the
    unsupported fields, then the per-call pack H1-H6 (its errors in step order)."""
    if cond is None:
        cond = _preset_conditions("primer3")  # D38: calc_hairpin's documented defaults
    elif not isinstance(cond, Conditions):
        raise ParameterError(f"cond={cond!r}; {_HAIRPIN} takes cond=None (primer3's defaults) "
                             "or sb.Conditions(...)")
    values = cond._asdict()
    oligo = values["oligo_nM"]
    if isinstance(oligo, tuple):
        # QH3. No computed total in the message: a float + would round in the caller's
        # environment (text-nulls-fp-environment.md §7.4).
        raise ParameterError(f"cond.oligo_nM={_fp.repr_value(oligo)} is a pair, but primer3's "
                             "calc_hairpin takes one DNA concentration (dna_conc, nM), which a "
                             "hairpin does not use; give a number, e.g. oligo_nM=50")
    for name in ("dmso_pct", "formamide_M"):
        # Zero test by the bits: DAZ would read a subnormal as zero (§7.4).
        if not _fp.is_zero(values[name]):
            raise ParameterError(f"cond.{name}={_fp.repr_float(values[name])} is not supported by "
                                 f"{_HAIRPIN}: primer3's hairpin computation has no {name} "
                                 f"correction; use {name}=0")
    # H1-H6 (§4): mv = Na + K + Tris/2 is formed in the pack, never here (§12.1).
    pack = _core._hairpin_primer3_pack(values["na_mM"], values["k_mM"], values["tris_mM"],
                                       values["mg_mM"], values["dntp_mM"], values["temp_c"])
    return cond, values, pack


def _hairpin(x, *, compat, cond, fields, sink, options):
    """T2 sb.hairpin(compat="primer3") (docs/wiki/thermo/hairpin-primer3.md): primer3-py
    2.3.1 calc_hairpin with max_loop 30 and a correctly rounded log, over primer3-py's own
    parameter files read at run time (D63). Checks in the §7 order: compat, the input kind
    (Pairs raises, as for tm), cond and the per-call pack, fields, keep and cutoff,
    execution options, the parameter loader (also for an empty input; inside the
    caller-state bracket of ``_primer3_handle``), device resolution (primer3-cuda.md §9);
    then the sequences."""
    spec = HAIRPIN
    # 1. Options.
    _hairpin_compat(compat)
    # The input kind is a call-shape error, independent of the data and of the installed
    # primer3-py, so it precedes the conditions and the loader (as tm's step 1.2 does).
    if isinstance(x, Pairs):
        raise ParameterError("hairpin is unary: pass one sequence or a list of sequences, not "
                             "Pairs, e.g. sb.hairpin(['ACGT...'], compat='primer3')")
    cond, values, pack = _hairpin_cond(cond)
    fields = _hairpin_fields(fields)
    if isinstance(sink, keep_module.Sink) and sink.kind not in spec["sinks"]:
        raise ParameterError(f"keep=sb.keep.{sink.kind}() is not available for hairpin: it "
                             "returns one row per sequence; use sb.keep.each(), "
                             "sb.keep.each(cutoff=c) or sb.keep.hits(c)")
    if sink is not None and not isinstance(sink, keep_module.Sink):
        raise ParameterError(f"keep={sink!r}; use sb.keep.each(), sb.keep.each(cutoff=c) or "
                             "sb.keep.hits(c)")
    sink = _sink(sink, spec)
    scalar = isinstance(x, (str, bytes)) and sink.kind == "each"
    cutoff = _hairpin_cutoff(sink.cutoff)
    options = resolve(options)
    # 1.6 The parameter loader (loader §8): its install hint does not depend on the data.
    handle = _primer3_handle()
    # Device resolution after every option check and the loader, before the sequence
    # scan; no data moves and no CPU fallback for an explicit device (primer3-cuda.md §9,
    # D85 (4)). device="auto" runs on the CPU.
    device = resolve_device(options["device"], inputs=(x,))
    stream = resolve_stream(options["stream"], device)
    ordinal = None if device is None else device["ordinal"]
    plan = _plan("hairpin", sink.kind, options["strategy"], cutoff is not None,
                 "cpu" if device is None else "cuda")
    # 2. Sequences: the engine scans the alphabet, then the length limits (§7).
    table = seqs(x)
    build = _BUILD
    layout = "scalar" if scalar else sink.kind
    model = {"compat": {"alias": "primer3", "id": spec["parameters"]["compat"]["id"]["primer3"]},
             "params": {"name": handle.name, "sha256": handle.sha256},
             "cond": dict(values, defaulted=cond.defaulted),
             "check_bases": True, "fields": fields,
             "keep": {"sink": sink.kind, "cutoff": cutoff}}
    metadata = {
        "seqblaze": _build_metadata(build),
        "call": {"kernel": "hairpin", "layout": layout},
        "model": model,
        "inputs": {"a": _input_record(table)},
        "execution": _placement(build, device, options, plan),
        "flags": {"research_use_only": True},
        "citations": citation_keys(spec, model, fields, plan["effective"]),
    }
    start = perf_counter_ns()
    native = native_call(_core._native_hairpin_primer3 if ordinal is None else _core._native_resident_hairpin_primer3,
                         table, pack, handle, sink.kind, cutoff,
                         options["threads"], list(fields), "auto", ordinal,
                         **({"progress": "hairpin"} if options["progress"] else {}),
                         **({"stream": stream.handle} if ordinal is not None else {}))
    replay = _capture_replay(metadata, a=table, scalar=isinstance(x, (str, bytes)),
                             captured_inputs=None if ordinal is None else native["input_digests"])
    columns = {name: _column(owner) for name, owner in native["columns"].items()}
    if scalar:
        columns = {name: array.reshape(()) for name, array in columns.items()}
    # API §5 shape (n, 1) for Each/Hits and (1, 1) for Scalar; rows are sequences.
    return Result._computed(layout, columns, tuple(native["shape"]), None,
                            _manifest(metadata, native["report"], _elapsed(start)), "hairpin",
                            replay)
