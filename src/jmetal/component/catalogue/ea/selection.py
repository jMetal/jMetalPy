"""Component that builds a mating pool from a population."""

from typing import Generic, Protocol, TypeVar, runtime_checkable

from jmetal.operator.selection import RandomSelection as RandomSelectionOperator
from jmetal.operator.selection import TournamentSelection as TournamentSelectionOperator

S = TypeVar("S")


@runtime_checkable
class Selection(Protocol[S]):
    """Builds a mating pool from a population.

    This is a structural (`typing.Protocol`) interface: any object exposing a
    single-argument `select()` method satisfies it.
    """

    def select(self, solution_list: list[S]) -> list[S]:
        """Build a mating pool from the given population.

        Args:
            solution_list: The population to select from.

        Returns:
            The mating pool. Its size is a property of the selection strategy, not
            of the input population -- it is not required to match `Variation`'s
            `mating_pool_size()`, but in practice the two are kept in sync so that
            `Variation.variate()` receives exactly what it expects.
        """
        ...


class TournamentSelection(Generic[S]):
    """Fills a mating pool by running a tournament once per slot.

    Wraps `jmetal.operator.selection.TournamentSelection`, which selects a single
    winner per call; this component repeats that call `mating_pool_size` times.

    Args:
        selection_operator: The tournament selection operator to repeat.
        mating_pool_size: How many solutions to select. Typically set to match the
            `Variation` component's `mating_pool_size()`.
    """

    def __init__(self, selection_operator: TournamentSelectionOperator, mating_pool_size: int):
        self.selection_operator = selection_operator
        self.mating_pool_size = mating_pool_size

    def select(self, solution_list: list[S]) -> list[S]:
        """Run the tournament `mating_pool_size` times.

        Args:
            solution_list: The population to select from.

        Returns:
            A mating pool of size `mating_pool_size`.
        """
        return [self.selection_operator.execute(solution_list) for _ in range(self.mating_pool_size)]


class RandomSelection(Generic[S]):
    """Fills a mating pool by picking a solution uniformly at random, once per slot.

    Wraps `jmetal.operator.selection.RandomSelection`, which selects a single solution
    per call with no regard for its quality; this component repeats that call
    `mating_pool_size` times. This is the default mating selection for SMS-EMOA
    (`jmetal.algorithm.multiobjective.smsemoa.SMSEMOA` also defaults to it), which relies
    on its replacement strategy alone -- not selection pressure -- to drive convergence.

    Args:
        selection_operator: The random selection operator to repeat.
        mating_pool_size: How many solutions to select. Typically set to match the
            `Variation` component's `mating_pool_size()`.
    """

    def __init__(self, selection_operator: RandomSelectionOperator, mating_pool_size: int):
        self.selection_operator = selection_operator
        self.mating_pool_size = mating_pool_size

    def select(self, solution_list: list[S]) -> list[S]:
        """Draw `mating_pool_size` solutions uniformly at random, with replacement.

        Args:
            solution_list: The population to select from.

        Returns:
            A mating pool of size `mating_pool_size`.
        """
        return [self.selection_operator.execute(solution_list) for _ in range(self.mating_pool_size)]
