"""Component that evaluates a list of solutions."""

from typing import Generic, Protocol, TypeVar, runtime_checkable

from jmetal.core.problem import Problem
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
