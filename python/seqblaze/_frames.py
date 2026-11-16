"""Error-path release of caller inputs (text-nulls-fp-environment.md §12)."""
import functools


def release_frames(error):
    """Drop the input references that a raised error's traceback keeps alive.

    A traceback keeps every frame it passed through, so a SequenceError for a
    null row (raised after the input tables exist) would keep the Pairs, the
    tables and their Arrow producers alive for as long as the caller holds the
    exception. The public kernel entry points, ``sb.seqs`` and the ``sb.pairs``
    constructors call this before re-raising: it clears the locals of the
    finished seqblaze frames along the error's traceback and its cause/context
    chain. Frames of other modules (user producers) are left alone, and a
    frame still running is skipped.
    """
    pending, seen = [error], set()
    while pending:
        current = pending.pop()
        if current is None or id(current) in seen:
            continue
        seen.add(id(current))
        trace = current.__traceback__
        while trace is not None:
            frame = trace.tb_frame
            if str(frame.f_globals.get("__name__", "")).startswith("seqblaze"):
                try:
                    frame.clear()
                except RuntimeError:  # still executing (the entry point itself)
                    pass
            trace = trace.tb_next
        pending.extend((current.__cause__, current.__context__))


def releasing(function):
    """Wrap a public converter so a rejected call releases its inputs at once.

    Pairwise inputs are converted in the ``sb.pairs`` constructors, before any
    kernel entry point runs, so an adapter error (E4, a wrong container) raised
    there would otherwise keep the producer alive through the traceback.
    """
    @functools.wraps(function)
    def wrapper(*args, **kwargs):
        try:
            return function(*args, **kwargs)
        except BaseException as error:
            args = kwargs = None
            release_frames(error)
            raise
    return wrapper
