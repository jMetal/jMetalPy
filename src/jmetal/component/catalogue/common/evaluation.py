"""Component that evaluates a list of solutions."""

import copy
from typing import Generic, Protocol, TypeVar, runtime_checkable

from jmetal.core.problem import Problem
from jmetal.util.archive import Archive
from jmetal.util.evaluator import Evaluator, SequentialEvaluator

S = TypeVar("S")


@runtime_checkable
class Evaluation(Protocol[S]):
    """Evaluates a list of solutions against a problem.

    This is a structural (`typing.Protocol`) interface: any object exposing
    `evaluate()`, `computed_evaluations()` and a `problem` attribute satisfies it.
    """

    problem: Problem

    def evaluate(self, solution_list: list[S]) -> list[S]:
        """Evaluate every solution in the list, in place.

        Args:
            solution_list: The solutions to evaluate.

        Returns:
            The same list, with each solution's objectives computed.
        """
        ...

    def computed_evaluations(self) -> int:
        """Return how many evaluations the last `evaluate()` call performed."""
        ...


class SequentialEvaluation(Generic[S]):
    """Evaluates a population by delegating to a `jmetal.util.evaluator.Evaluator`.

    Args:
        problem: The problem to evaluate solutions against.
        evaluator: The evaluation strategy. Defaults to `SequentialEvaluator`, but
            any `Evaluator` works -- e.g. `MultiprocessEvaluator` for parallel
            evaluation.
    """

    def __init__(self, problem: Problem[S], evaluator: Evaluator[S] | None = None):
        self.problem = problem
        self._evaluator = evaluator if evaluator is not None else SequentialEvaluator[S]()
        self._computed_evaluations = 0

    def evaluate(self, solution_list: list[S]) -> list[S]:
        """Evaluate every solution in the list using the wrapped `Evaluator`.

        Args:
            solution_list: The solutions to evaluate.

        Returns:
            The same list, with each solution's objectives computed.
        """
        evaluated = self._evaluator.evaluate(solution_list, self.problem)
        self._computed_evaluations = len(solution_list)

        return evaluated

    def computed_evaluations(self) -> int:
        """Return how many evaluations the last `evaluate()` call performed."""
        return self._computed_evaluations


class SequentialEvaluationWithArchive(SequentialEvaluation[S]):
    """`SequentialEvaluation` that also feeds every evaluated solution into an archive.

    A copy of each evaluated solution is added to the archive as a side effect --
    the archive accumulates independently of whatever the population/replacement
    strategy decides to keep, which is exactly what makes it "external". Pairing
    this with `EvolutionaryAlgorithm(..., archive=archive)` makes `result()` return
    the archive's contents instead of the final population.

    Args:
        problem: The problem to evaluate solutions against.
        archive: The archive every evaluated solution is copied into.
        evaluator: The evaluation strategy. Defaults to `SequentialEvaluator`.
    """

    def __init__(
        self, problem: Problem[S], archive: Archive[S], evaluator: Evaluator[S] | None = None
    ):
        super().__init__(problem, evaluator)
        self.archive = archive

    def evaluate(self, solution_list: list[S]) -> list[S]:
        """Evaluate every solution, then copy the whole batch into the archive.

        Uses `Archive.add_batch()` rather than calling `add()` once per solution --
        for `NonDominatedSolutionsArchive`, that turns k one-at-a-time O(n)
        insertions into a single batch filter, which matters once the archive
        grows into the thousands (see `docs/advanced-topics/component-architecture.md`).

        Args:
            solution_list: The solutions to evaluate.

        Returns:
            The same list, with each solution's objectives computed.
        """
        evaluated = super().evaluate(solution_list)
        self.archive.add_batch([copy.copy(solution) for solution in evaluated])

        return evaluated
