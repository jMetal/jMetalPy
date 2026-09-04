"""Component that turns a mating pool into an offspring population."""

import math
from typing import Generic, Protocol, TypeVar, runtime_checkable

from jmetal.core.operator import Crossover, Mutation

S = TypeVar("S")


@runtime_checkable
class Variation(Protocol[S]):
    """Turns a mating pool into an offspring population.

    This is a structural (`typing.Protocol`) interface: any object exposing
    `variate()`, `mating_pool_size()` and `offspring_population_size()` satisfies it.
    """

    def variate(self, solution_list: list[S], mating_pool: list[S]) -> list[S]:
        """Produce the offspring population from a mating pool.

        Args:
            solution_list: The current population (available for strategies that
                need it beyond the mating pool itself; unused by simple strategies
                such as `CrossoverAndMutationVariation`).
            mating_pool: The parents to combine, of size `mating_pool_size()`.

        Returns:
            The offspring population, of size `offspring_population_size()`.
        """
        ...

    def mating_pool_size(self) -> int:
        """Return how many parents `variate()` expects in the mating pool."""
        ...

    def offspring_population_size(self) -> int:
        """Return how many offspring `variate()` produces."""
        ...


class CrossoverAndMutationVariation(Generic[S]):
    """Produces offspring by applying crossover and then mutation.

    The mating pool size is derived from the crossover operator so that exactly
    `offspring_population_size` children can be produced:
    `mating_pool_size = crossover.get_number_of_parents() * ceil(offspring_population_size
    / crossover.get_number_of_children())`.

    Args:
        offspring_population_size: How many offspring to produce.
        crossover: The crossover operator.
        mutation: The mutation operator, applied to every offspring individually.
    """

    def __init__(
        self, offspring_population_size: int, crossover: Crossover, mutation: Mutation
    ):
        self.crossover = crossover
        self.mutation = mutation
        self._offspring_population_size = offspring_population_size
        self._mating_pool_size = crossover.get_number_of_parents() * math.ceil(
            offspring_population_size / crossover.get_number_of_children()
        )

    def variate(self, solution_list: list[S], mating_pool: list[S]) -> list[S]:
        """Apply crossover to consecutive groups of parents, then mutate each child.

        Args:
            solution_list: Unused; kept to satisfy the `Variation` protocol.
            mating_pool: The parents to combine, of size `mating_pool_size()`.

        Returns:
            The offspring population, of size `offspring_population_size()`.

        Raises:
            ValueError: If `mating_pool`'s size is not a multiple of the crossover
                operator's number of required parents.
        """
        number_of_parents = self.crossover.get_number_of_parents()
        if len(mating_pool) % number_of_parents != 0:
            raise ValueError(
                f"Wrong number of parents: the mating pool size ({len(mating_pool)}) is not "
                f"divisible by the crossover's number of required parents ({number_of_parents})"
            )

        offspring_population: list[S] = []
        for i in range(0, self._mating_pool_size, number_of_parents):
            parents = mating_pool[i : i + number_of_parents]
            offspring = self.crossover.execute(parents)

            for child in offspring:
                self.mutation.execute(child)
                offspring_population.append(child)
                if len(offspring_population) == self._offspring_population_size:
                    break

        return offspring_population

    def mating_pool_size(self) -> int:
        """Return the mating pool size required to produce the offspring population."""
        return self._mating_pool_size

    def offspring_population_size(self) -> int:
        """Return the configured offspring population size."""
        return self._offspring_population_size
