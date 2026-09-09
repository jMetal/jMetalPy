"""Component that creates the initial population of an algorithm."""

from typing import Generic, Protocol, TypeVar, runtime_checkable

import numpy as np

from jmetal.core.problem import Problem

S = TypeVar("S")


@runtime_checkable
class SolutionsCreation(Protocol[S]):
    """Creates a list of solutions applying some strategy (e.g. random sampling).

    This is a structural (`typing.Protocol`) interface: any object exposing a
    zero-argument `create()` method satisfies it, including a plain function wrapped
    in a small adapter -- no subclassing is required.
    """

    def create(self) -> list[S]:
        """Create a new list of solutions.

        Returns:
            The newly created solutions.
        """
        ...


class RandomSolutionsCreation(Generic[S]):
    """Creates a population by repeatedly asking the problem for a new random solution.

    Args:
        problem: The problem whose `create_solution()` generates each solution.
        number_of_solutions_to_create: How many solutions to create.
        rng: Optional random generator, forwarded to `problem.create_solution()`. Left
            as a plain `self.rng` attribute (not resolved to a default here) so that
            `EvolutionaryAlgorithm._thread_rng_into_components()` can share the
            algorithm's own generator with it when none is given explicitly.
    """

    def __init__(
        self,
        problem: Problem[S],
        number_of_solutions_to_create: int,
        rng: np.random.Generator | None = None,
    ):
        self.problem = problem
        self.number_of_solutions_to_create = number_of_solutions_to_create
        self.rng = rng

    def create(self) -> list[S]:
        """Create the population.

        Returns:
            A list of `number_of_solutions_to_create` solutions, each produced by
            `problem.create_solution(rng=self.rng)`.
        """
        return [
            self.problem.create_solution(rng=self.rng)
            for _ in range(self.number_of_solutions_to_create)
        ]
