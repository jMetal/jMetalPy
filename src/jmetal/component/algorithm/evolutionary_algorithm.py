"""The component-based evolutionary algorithm template."""

import time
from typing import Generic, TypeVar

import numpy as np

from jmetal.component.algorithm.algorithm_state import AlgorithmState
from jmetal.component.catalogue.common.evaluation import Evaluation
from jmetal.component.catalogue.common.solutions_creation import SolutionsCreation
from jmetal.component.catalogue.common.termination import Termination
from jmetal.component.catalogue.ea.replacement import Replacement
from jmetal.component.catalogue.ea.selection import Selection
from jmetal.component.catalogue.ea.variation import Variation
from jmetal.util.archive import Archive, distance_based_subset_selection_robust
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
        archive: An optional external archive. When given, `result()` returns the
            archive's contents instead of the final population -- pair it with
            `SequentialEvaluationWithArchive` (or any `Evaluation` that feeds the
            same archive) so it actually accumulates solutions during the run;
            the template itself never writes to it.
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
        rng: np.random.Generator | None = None,
        archive: Archive[S] | None = None,
    ):
        self.name = name
        self.solutions_creation = solutions_creation
        self.evaluation = evaluation
        self.termination = termination
        self.selection = selection
        self.variation = variation
        self.replacement = replacement
        self.archive = archive

        self.solutions: list[S] = []
        self.evaluations = 0
        self.start_computing_time = 0.0
        self.total_computing_time = 0.0
        self.observable = DefaultObservable()

        self.rng = rng if rng is not None else np.random.default_rng()
        self._thread_rng_into_components()

    def _thread_rng_into_components(self) -> None:
        """Share this algorithm's rng with every component that accepts one.

        Not every component is RNG-aware yet -- operators are being migrated to the
        injectable-rng pattern incrementally (see MODERNIZATION.md's L1 notes) -- so
        this only assigns `rng` on components that already expose it as a plain
        attribute, leaving the rest untouched. A component built with its own
        explicit `rng` before being handed to this template is not overridden here;
        wire it through `build_nsgaii()` (or an equivalent factory) instead if it
        should share this algorithm's generator.

        `solutions_creation` is deliberately excluded from this list. Every other
        RNG-aware component here falls back to a *fresh* `np.random.default_rng()`
        when its own `rng` is `None`, so handing it this algorithm's generator instead
        is a side-grade, not a behavior change. `RandomSolutionsCreation` (via
        `Problem.create_solution()`) is different: its `rng=None` deliberately keeps
        drawing from the global `random`/`numpy.random` state, matching
        `create_solution()`'s pre-existing, unparameterized behavior -- auto-threading
        this algorithm's `rng` into it here would silently switch every unseeded
        `build_nsgaii()`/`build_smsemoa()` call onto a different random source for
        population creation, breaking the equivalence tests that seed the global
        `random` module and expect `create_solution()` to still consume it. Population
        creation only becomes `rng`-reproducible when a factory explicitly forwards
        its own `rng` parameter into `RandomSolutionsCreation` at construction time
        (see `build_nsgaii()`/`build_smsemoa()`/`build_moead()`).
        """
        for component in (
            self.evaluation,
            self.termination,
            self.selection,
            self.variation,
            self.replacement,
        ):
            if getattr(component, "rng", None) is None and hasattr(component, "rng"):
                component.rng = self.rng

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
        """Return the algorithm's result.

        Returns:
            If an `archive` was given at construction time: its contents, reduced
            to the population size via distance-based subset selection if it grew
            larger (an unbounded archive can accumulate far more than that -- a
            bounded one, e.g. `CrowdingDistanceArchive`, never exceeds the
            population size to begin with, so this is a no-op for it). Otherwise
            the current population. Valid at any point during or after `run()`.
        """
        if self.archive is None:
            return self.solutions

        archive_solutions = self.archive.solution_list
        population_size = len(self.solutions)
        if len(archive_solutions) <= population_size:
            return archive_solutions

        return distance_based_subset_selection_robust(
            archive_solutions, population_size, rng=self.rng
        )

    def get_name(self) -> str:
        """Return the algorithm's name."""
        return self.name

    def observable_data(self) -> dict:
        """Return the data broadcast to observers on each progress update.

        Returns:
            The same `"PROBLEM"`/`"EVALUATIONS"`/`"SOLUTIONS"`/`"COMPUTING_TIME"`
            mapping `jmetal.core.algorithm.Algorithm` subclasses return, so this
            template satisfies `jmetal.core.algorithm.AlgorithmProtocol` too.
        """
        return self._state().as_dict()

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
