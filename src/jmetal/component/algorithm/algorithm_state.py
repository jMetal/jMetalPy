"""Typed snapshot of an algorithm's progress."""

from dataclasses import dataclass
from typing import Generic, TypeVar

from jmetal.core.problem import Problem

S = TypeVar("S")


@dataclass
class AlgorithmState(Generic[S]):
    """A typed snapshot of an algorithm's progress at a point in time.

    Historically, jMetalPy has passed this same information around as a plain
    `dict` with string keys (`"PROBLEM"`, `"EVALUATIONS"`, `"SOLUTIONS"`,
    `"COMPUTING_TIME"`) -- a typo in one of those keys silently returns `None`
    instead of failing. `AlgorithmState` gives the `EvolutionaryAlgorithm` template a
    typed value to build and pass around internally, while `as_dict()` produces the
    exact same string-keyed mapping so every existing `Observer`
    (`jmetal.util.observer`) and `TerminationCriterion`
    (`jmetal.util.termination_criterion`) keeps working unchanged.

    Attributes:
        problem: The problem being solved.
        evaluations: The number of evaluations computed so far.
        solutions: The current population.
        computing_time: Elapsed time, in seconds, since the algorithm started.
    """

    problem: Problem[S]
    evaluations: int
    solutions: list[S]
    computing_time: float

    def as_dict(self) -> dict[str, object]:
        """Return this state as the string-keyed mapping jMetalPy already expects.

        Returns:
            A dict with `"PROBLEM"`, `"EVALUATIONS"`, `"SOLUTIONS"` and
            `"COMPUTING_TIME"` keys, suitable for `Observer.update(**kwargs)` and
            for `Termination.is_met(status)`.
        """
        return {
            "PROBLEM": self.problem,
            "EVALUATIONS": self.evaluations,
            "SOLUTIONS": self.solutions,
            "COMPUTING_TIME": self.computing_time,
        }
