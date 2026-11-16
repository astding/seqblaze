"""Run-time loader of primer3-py 2.3.1's thermodynamic parameter files (D63 clause 3).

Spec: docs/wiki/thermo/primer3-parameter-loader.md §1-§3 (contract, lookup without
import, pins and set digest), §7 (cache) and §8 (errors). Shared by the T2 hairpin and the
K2 dimer presets with ``compat="primer3"``.

The installed distribution ``primer3-py`` is found through the standard library's
installed-distribution lookup, which reads its ``dist-info`` directory and never imports
the package. Each of the sixteen files is resolved through the distribution's ``RECORD``
by its pinned sha256, read in full and hashed again; the verified bytes go to the Rust
parser (``_core._primer3_params``). Nothing here converts text to a float, and no
floating-point value of these steps reaches a result (D99; loader §7); no message ever
contains a value or file text (D70, D76). seqblaze never imports primer3 and never ships its
files.

The steps are not free of status flags: this module's first import and every lookup that is
not a cache hit read file status information, which sets the inexact flag in the calling
thread's floating-point state (text-nulls-fp-environment.md §8.6). The facade therefore
imports this module and calls ``handle`` only inside a caller-state bracket
(``_facade._primer3_handle``; §7.3 item 5), which writes the caller's state back.
"""
import base64
import contextvars
import csv
import hashlib
import importlib.metadata as _metadata
import io
import os
import posixpath
import threading

from . import _core
from ._errors import ParameterError

DIST_NAME = "primer3-py"
DIST_VERSION = "2.3.1"
#: ``model.params.name`` of the located set (hairpin-primer3.md §6).
NAME = "primer3-py@2.3.1:files"
ROLES = tuple(f"E{n}" for n in range(1, 17))
#: Loader §3.1: role -> sha256 of the primer3-py 2.3.1 file.
PINS = {
    "E1": "c7305ddec2c690990358e207f5a6beb118283a74ce711a4aa68f89b788fa7c27",
    "E2": "4def88bee217039784583159a69ea50cab4e9ea5b82132f7938dfd2866d0524e",
    "E3": "89e1069d20209f1a0c4c3ddf89bad19af167ddc343e45423f39d232026d064da",
    "E4": "7ba2b0669b5bf6f4c78ea2543f8d39f75f9d5a3429e1cf26eaf46f70b6bb2704",
    "E5": "c88f09dc5be829a130ccbb00f142e4ff89259411355262911799d0be0571d4aa",
    "E6": "167c423714d0d350cfb8df5439a7621c0e34033b1f6b87c1cb58760d75bdb529",
    "E7": "2d26b74caf8e80ece50b659acb4bd547812ead5143ffb35338428637368a3087",
    "E8": "68b67437eb75485231ab8792e58b1e7ca488b81d4640b61d40f0d1402291da90",
    "E9": "a4b9e1ca1015f89254fad5cb63d9ffcde6e996618ee1a7d417ee032a16613f4f",
    "E10": "36f1b2927f2fabbc7135a9bdfe9c415efcbf10951c680a92baa331e7cb408ebc",
    "E11": "71aa016d8c94371e715dbcc161f6884a6243d0c050932bfadc446168aecc3c2a",
    "E12": "117eed025fbd50f0543083f5bd2db4edf9b54df748f6a1b57caa271cc9ab409e",
    "E13": "12821532c4408bf12f81b4fa3deb62452926506902c925134b72fb3cdeb5c5cf",
    "E14": "181c3175ba570c433ce7a345043003259a510f5960b41beb1c96ba73658e60ba",
    "E15": "35fa4a70818c13bd4b3aeea40322368cba3af19192dcef92d28d17f881434ee8",
    "E16": "c3e1b6e80924c0d47e9d8f3ab928f665a88f993d44019e0cebea72612e7a6034",
}
#: Loader §3.3: the set digest of the pins.
SET_DIGEST = "ecf487dfbf58b1d08c9082ce7ff9c850b9cb4e10806b59eb07f7452da0429e45"

_INSTALL = 'pip install "seqblaze[primer3]"'
_NO_IMPORT = "seqblaze does not import primer3; it only reads these data files (D63)"


class _Overrides:
    """Testing overrides (loader §9.1, T2 §10.3), local to the calling thread and context.

    Each name holds, in its own ``contextvars.ContextVar``, a tuple of (key, value)
    entries; the current value is the newest entry's, or None. A ``seqblaze._testing``
    block adds one entry on entry and removes exactly that entry on exit, so blocks in
    other threads never see each other's values, and exits in any order leave nothing
    installed once every block has exited. A new thread starts with no override.
    """

    def __init__(self, names):
        self._vars = {name: contextvars.ContextVar(f"seqblaze_p3_override_{name}",
                                                   default=())
                      for name in names}

    def __getitem__(self, name):
        entries = self._vars[name].get()
        return entries[-1][1] if entries else None

    def __setitem__(self, name, value):
        """Replace every entry of ``name`` in the current context (None clears it)."""
        self._vars[name].set(() if value is None else ((object(), value),))

    def push(self, name, value):
        """Add one entry; returns the key that :meth:`pop` takes."""
        key = object()
        var = self._vars[name]
        var.set(var.get() + ((key, value),))
        return key

    def pop(self, name, key):
        """Remove the entry added under ``key`` (if still present)."""
        var = self._vars[name]
        var.set(tuple(entry for entry in var.get() if entry[0] is not key))


# Testing builds only: set through seqblaze._testing, honoured only when the extension is
# a testing build. Release builds have no override.
_OVERRIDE = _Overrides(("handle", "paths", "pins"))
_LOCK = threading.Lock()
_CACHE = None  # (pin items, handle) of the first successful load (loader §7)


def set_digest(digests):
    """Loader §3.3: sha256 of the sixteen lines ``E<n>\\t<hex>\\n``, n = 1..16."""
    text = "".join(f"{role}\t{digests[role]}\n" for role in ROLES)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _testing():
    return bool(getattr(_core, "_testing_enabled", False))


def _pins():
    if _testing() and _OVERRIDE["pins"] is not None:
        return _OVERRIDE["pins"]
    return PINS


def _error(what, fix):
    return ParameterError(f"compat='primer3' {what}; {fix}. {_NO_IMPORT}")


def _record_digest(hex_digest):
    """The RECORD digest field of a sha256 hex digest (PEP 376, PEP 627)."""
    raw = base64.urlsafe_b64encode(bytes.fromhex(hex_digest)).rstrip(b"=").decode("ascii")
    return f"sha256={raw}"


def _inside(path):
    """Whether a RECORD path stays inside the distribution root."""
    if not path or path.startswith(("/", "\\")) or os.path.isabs(path):
        return False
    normal = posixpath.normpath(path.replace("\\", "/"))
    return normal != ".." and not normal.startswith("../")


def load(paths=None, pins=None):
    """One uncached locate-verify-parse (loader §2-§5). ``paths`` None searches the
    interpreter's search path; a list searches exactly those directories."""
    pins = PINS if pins is None else pins
    context = {"name": DIST_NAME} if paths is None else {"name": DIST_NAME, "path": list(paths)}
    where = "" if paths is None else f" in {list(paths)!r}"
    dist = next(iter(_metadata.distributions(**context)), None)
    if dist is None:  # row 1
        raise _error(f"needs primer3-py {DIST_VERSION}'s thermodynamic parameter files, but no "
                     f"distribution primer3-py is installed{where}",
                     f"install them with {_INSTALL}")
    version = dist.version
    if version != DIST_VERSION:  # row 2
        raise _error(f"needs primer3-py {DIST_VERSION}, but primer3-py {version} is installed"
                     f"{where}", f"install {DIST_VERSION} with {_INSTALL}")
    record = dist.read_text("RECORD")
    if record is None:  # row 3
        raise _error(f"cannot verify the installed primer3-py {DIST_VERSION}: its distribution "
                     "has no RECORD file", f"reinstall it with pip install --force-reinstall "
                     f'"seqblaze[primer3]"')
    first = {}
    for row in csv.reader(io.StringIO(record)):
        if len(row) < 2 or not row[1] or not _inside(row[0]):
            continue
        first.setdefault(row[1], row[0])
    files = []
    for role in ROLES:
        pin = pins[role]
        path = first.get(_record_digest(pin))
        if path is None:  # row 4
            raise _error(f"cannot find the parameter file of role {role} (sha256 {pin[:16]}...) "
                         f"in the installed primer3-py {DIST_VERSION}: the installation is "
                         "incomplete or modified",
                         f'reinstall it with pip install --force-reinstall "seqblaze[primer3]"')
        try:
            data = dist.locate_file(path).read_bytes()
        except OSError:
            data = None
        if data is None or hashlib.sha256(data).hexdigest() != pin:  # row 5
            raise _error(f"found the parameter file of role {role} (sha256 {pin[:16]}...) at "
                         f"{path} in the installed primer3-py {DIST_VERSION}, but its bytes do "
                         "not match the pin: the installation is modified",
                         f'reinstall it with pip install --force-reinstall "seqblaze[primer3]"')
        files.append(data)
    # Parse rules and content facts (row 6) raise ParameterError from the Rust parser.
    return _core._primer3_params(files, NAME, set_digest(pins), False)


def handle():
    """The parameter set of a T2 or K2 primer3 call (loader §1, §7): the testing handle
    when one is active; else the cached set, loaded on first use under a lock. Failures
    are not cached. A testing search path bypasses the cache."""
    global _CACHE
    if _testing() and _OVERRIDE["handle"] is not None:
        return _OVERRIDE["handle"]
    pins = _pins()
    if _testing() and _OVERRIDE["paths"] is not None:
        return load(_OVERRIDE["paths"], pins)
    key = tuple(sorted(pins.items()))
    with _LOCK:
        if _CACHE is not None and _CACHE[0] == key:
            return _CACHE[1]
        loaded = load(None, pins)
        _CACHE = (key, loaded)
        return loaded
