"""The component-based evolutionary algorithm template."""

import time
from typing import Generic, TypeVar

from jmetal.component.algorithm.algorithm_state import AlgorithmState
from jmetal.component.catalogue.common.evaluation import Evaluation
from jmetal.component.catalogue.common.solutions_creation import SolutionsCreation
from jmetal.component.catalogue.common.termination import Termination
from jmetal.component.catalogue.ea.replacement import Replacement
from jmetal.component.catalogue.ea.selection import Selection
from jmetal.component.catalogue.ea.variation import Variation
from jmetal.util.observable import DefaultObservable

S = TypeVar("S")


class EvolutionaryAlgorithm(Generic[S]):
    """An evolutionary algorithm assembled from six independently swappable components.

    This is a direct translation of jMetal Java's `EvolutionaryAlgorithm`
    (`jmetal-component/.../algorithm/EvolutionaryAlgorithm.java`): `run()` reproduces
    its loop unchanged --

        population = solutions_creation.create()
        population = evaluation.evaluate(population)
        while not termination.is_met(state):
            mating_population = selection.select(population)
            offspring_population = variation.variate(population, mating_population)
            offspring_population = evaluation.evaluate(offspring_population)
            population = replacement.replace(population, offspring_population)

    Unlike `jmetal.core.algorithm.Algorithm`, this class does not inherit from
    `threading.Thread`: nothing in jMetalPy calls `start()`/`join()` on an algorithm,
    so the only feature that inheritance provided (calling `run()`) is provided here
    directly, without threading.Thread's pickling and API baggage.

    Args:
        name: The algorithm's name, e.g. `"NSGAII"`.
        solutions_creation: Creates the initial population.
        evaluation: Evaluates a list of solutions; also the source of `problem`.
        termination: Decides when to stop.
        selection: Builds a mating pool from the population.
        variation: Turns a mating pool into an offspring population.
        replacement: Selects survivors from the population and the offspring.
    """

    def __init__(
        self,
        name: str,
        solutions_creation: SolutionsCreation[S],
        evaluation: Evaluation[S],
        termination: Termination,
        selection: Selection[S],
        variation: Variation[S],
        replacement: Replacement[S],
    ):
        self.name = name
        self.solutions_creation = solutions_creation
        self.evaluation = evaluation
        self.termination = termination
        self.selection = selection
        self.variation = variation
        self.replacement = replacement

        self.solutions: list[S] = []
        self.evaluations = 0
        self.start_computing_time = 0.0
        self.total_computing_time = 0.0
        self.observable = DefaultObservable()

    def run(self) -> None:
        """Run the algorithm to completion.

        Creates and evaluates the initial population, then repeats
        selection/variation/evaluation/replacement until `termination` is met.
        """
        self.start_computing_time = time.time()

        self.solutions = self.solutions_creation.create()
        self.solutions = self.evaluation.evaluate(self.solutions)
        self._init_progress()

        while not self.termination.is_met(self._state().as_dict()):
            mating_population = self.selection.select(self.solutions)
            offspring_population = self.variation.variate(self.solutions, mating_population)
            offspring_population = self.evaluation.evaluate(offspring_population)
            self.solutions = self.replacement.replace(self.solutions, offspring_population)
            self._update_progress()

        self.total_computing_time = self._current_computing_time()

    def result(self) -> list[S]:
        """Return the current population.

        Returns:
            The population, valid at any point during or after `run()`.
        """
        return self.solutions

    def get_name(self) -> str:
        """Return the algorithm's name."""
        return self.name

    def _state(self) -> AlgorithmState[S]:
        return AlgorithmState(
            problem=self.evaluation.problem,
            evaluations=self.evaluations,
            solutions=self.solutions,
            computing_time=self._current_computing_time(),
        )

    def _current_computing_time(self) -> float:
        return time.time() - self.start_computing_time

    def _init_progress(self) -> None:
        self.evaluations = len(self.solutions)
        self.total_computing_time = self._current_computing_time()
        # Unlike the Java template, this also notifies observers here (not only from
        # _update_progress()), matching jmetal.core.algorithm.Algorithm.init_progress()
        # so that existing observers (jmetal.util.observer) see the same sequence of
        # notifications they already do with the classic algorithms.
        self.observable.notify_all(**self._state().as_dict())

    def _update_progress(self) -> None:
        self.evaluations += self.variation.offspring_population_size()
        self.total_computing_time = self._current_computing_time()
        self.observable.notify_all(**self._state().as_dict())
