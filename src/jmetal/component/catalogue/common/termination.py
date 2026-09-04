"""Component that checks whether an algorithm should stop."""

from collections.abc import Mapping
from typing import Protocol, runtime_checkable


@runtime_checkable
class Termination(Protocol):
    """Checks the termination condition of an algorithm.

    This is a structural (`typing.Protocol`) interface: any object exposing a
    single-argument `is_met()` method satisfies it, including a plain function.

    The status mapping uses the same string keys the rest of jMetalPy already relies
    on (`"EVALUATIONS"`, `"COMPUTING_TIME"`, ...), so existing conventions keep
    working; see `AlgorithmState`.
    """

    def is_met(self, status: Mapping[str, object]) -> bool:
        """Check whether the algorithm should stop.

        Args:
            status: The algorithm's current status, e.g. `{"EVALUATIONS": 500}`.

        Returns:
            True if the algorithm should stop, False otherwise.
        """
        ...


class TerminationByEvaluations:
    """Stops the algorithm once a maximum number of evaluations has been reached.

    Args:
        max_evaluations: The number of evaluations after which the algorithm stops.
    """

    def __init__(self, max_evaluations: int):
        self.max_evaluations = max_evaluations

    def is_met(self, status: Mapping[str, object]) -> bool:
        """Check whether at least `max_evaluations` evaluations have been computed.

        Args:
            status: The algorithm's current status; must contain `"EVALUATIONS"`.

        Returns:
            True once `status["EVALUATIONS"] >= max_evaluations`.
        """
        return status["EVALUATIONS"] >= self.max_evaluations
