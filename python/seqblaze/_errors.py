"""Public exception hierarchy and private PF-RUN compatibility error."""


class SeqblazeError(Exception):
    pass


class ParameterError(SeqblazeError, ValueError):
    pass


class PairingError(SeqblazeError, TypeError):
    pass


class _PairingError(SeqblazeError, ValueError):
    """Private native error, translated at the public facade boundary."""


class SequenceError(SeqblazeError, ValueError):
    pass


class UnitError(SeqblazeError, ValueError):
    pass


class FormatError(SeqblazeError, ValueError):
    """Malformed FASTA/FASTQ input (D92; docs/wiki/io/fasta-readers.md §8.1).

    ``source`` names the input (the path, ``"<bytes>"``, ``"<BytesIO>"``, ...),
    ``record`` is the 0-based record ordinal, ``line`` the 1-based line number and
    ``reason`` one of the §8.1 codes (for example ``"length_mismatch"``). No record
    is ever skipped.
    """
    source = None
    record = None
    line = None
    reason = None


class DeviceError(SeqblazeError, RuntimeError):
    pass


class SeqblazeWarning(UserWarning):
    pass


class IllegalBaseWarning(SeqblazeWarning):
    pass
