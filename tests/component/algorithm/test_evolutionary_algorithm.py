"""Tests for the EvolutionaryAlgorithm template, using test doubles for every component.

These tests exercise the template's run() loop in isolation from any real problem or
operator -- "solutions" are plain integers -- so that a failure here can only be a
template bug, not a bug in one of the catalogue components (which have their own
tests).
"""

import threading

import numpy as np

from jmetal.component.algorithm.evolutionary_algorithm import EvolutionaryAlgorithm


class StubSolutionsCreation:
    def __init__(self, population_size: int):
        self.population_size = population_size

    def create(self) -> list[int]:
        return list(range(self.population_size))


class StubEvaluation:
    def __init__(self, problem: object):
        self.problem = problem
        self.evaluate_call_count = 0

    def evaluate(self, solution_list: list[int]) -> list[int]:
        self.evaluate_call_count += 1
        return solution_list

    def computed_evaluations(self) -> int:
        return 0


class StubTermination:
    def __init__(self, max_evaluations: int):
        self.max_evaluations = max_evaluations

    def is_met(self, status: dict) -> bool:
        return status["EVALUATIONS"] >= self.max_evaluations


class StubSelection:
    def select(self, solution_list: list[int]) -> list[int]:
        return list(solution_list)


class StubVariation:
    def __init__(self, offspring_size: int):
        self._offspring_size = offspring_size
        self.variate_call_count = 0

    def variate(self, solution_list: list[int], mating_pool: list[int]) -> list[int]:
        self.variate_call_count += 1
        return [-1] * self._offspring_size

    def mating_pool_size(self) -> int:
        return self._offspring_size

    def offspring_population_size(self) -> int:
        return self._offspring_size


class StubReplacement:
    def replace(self, solution_list: list[int], offspring_list: list[int]) -> list[int]:
        # Prefers offspring over the current population, truncated to size.
        combined = offspring_list + solution_list
        return combined[: len(solution_list)]


def build_algorithm(population_size=4, offspring_size=4, generations=2) -> EvolutionaryAlgorithm:
    return EvolutionaryAlgorithm(
        name="StubEA",
        solutions_creation=StubSolutionsCreation(population_size),
        evaluation=StubEvaluation(problem=object()),
        termination=StubTermination(max_evaluations=population_size + generations * offspring_size),
        selection=StubSelection(),
        variation=StubVariation(offspring_size),
        replacement=StubReplacement(),
    )


class TestEvolutionaryAlgorithm:
    def test_run_produces_a_population_of_the_configured_size(self):
        algorithm = build_algorithm(population_size=4, offspring_size=4)

        algorithm.run()

        assert len(algorithm.result()) == 4

    def test_run_stops_at_the_configured_number_of_generations(self):
        algorithm = build_algorithm(population_size=4, offspring_size=4, generations=3)

        algorithm.run()

        assert algorithm.evaluations == 4 + 3 * 4

    def test_result_reflects_the_replacement_component(self):
        # StubReplacement always keeps the offspring (marked -1), so after at least
        # one generation the population should be entirely replaced.
        algorithm = build_algorithm(population_size=4, offspring_size=4, generations=1)

        algorithm.run()

        assert algorithm.result() == [-1, -1, -1, -1]

    def test_get_name_returns_the_configured_name(self):
        algorithm = build_algorithm()

        assert algorithm.get_name() == "StubEA"

    def test_run_notifies_observers_with_the_legacy_keys(self):
        received_updates = []

        class RecordingObserver:
            def update(self, *args, **kwargs):
                received_updates.append(kwargs)

        algorithm = build_algorithm(population_size=4, offspring_size=4, generations=2)
        algorithm.observable.register(RecordingObserver())

        algorithm.run()

        # One notification from _init_progress() plus one per generation.
        assert len(received_updates) == 3
        assert all(
            {"PROBLEM", "EVALUATIONS", "SOLUTIONS", "COMPUTING_TIME"} == set(update)
            for update in received_updates
        )
        assert [update["EVALUATIONS"] for update in received_updates] == [4, 8, 12]

    def test_does_not_inherit_from_threading_thread(self):
        # Nothing in jMetalPy calls start()/join() on an algorithm; the template
        # intentionally avoids that coupling from the start (see Phase 1b).
        algorithm = build_algorithm()

        assert not isinstance(algorithm, threading.Thread)


class RngAwareSelection(StubSelection):
    """A component that declares an rng slot but doesn't have one yet."""

    def __init__(self):
        self.rng: np.random.Generator | None = None


class TestRngThreading:
    def test_defaults_to_a_fresh_generator_when_none_is_given(self):
        algorithm = build_algorithm()

        assert isinstance(algorithm.rng, np.random.Generator)

    def test_uses_the_given_generator_instead_of_a_fresh_one(self):
        rng = np.random.default_rng(42)

        algorithm = EvolutionaryAlgorithm(
            name="StubEA",
            solutions_creation=StubSolutionsCreation(4),
            evaluation=StubEvaluation(problem=object()),
            termination=StubTermination(max_evaluations=4),
            selection=StubSelection(),
            variation=StubVariation(4),
            replacement=StubReplacement(),
            rng=rng,
        )

        assert algorithm.rng is rng

    def test_shares_its_rng_with_components_that_declare_an_rng_attribute(self):
        rng = np.random.default_rng(7)
        selection = RngAwareSelection()

        algorithm = EvolutionaryAlgorithm(
            name="StubEA",
            solutions_creation=StubSolutionsCreation(4),
            evaluation=StubEvaluation(problem=object()),
            termination=StubTermination(max_evaluations=4),
            selection=selection,
            variation=StubVariation(4),
            replacement=StubReplacement(),
            rng=rng,
        )

        assert selection.rng is rng
        assert algorithm.selection.rng is algorithm.rng

    def test_does_not_touch_a_component_that_already_has_its_own_rng(self):
        own_rng = np.random.default_rng(1)
        selection = RngAwareSelection()
        selection.rng = own_rng

        algorithm = EvolutionaryAlgorithm(
            name="StubEA",
            solutions_creation=StubSolutionsCreation(4),
            evaluation=StubEvaluation(problem=object()),
            termination=StubTermination(max_evaluations=4),
            selection=selection,
            variation=StubVariation(4),
            replacement=StubReplacement(),
            rng=np.random.default_rng(2),
        )

        assert selection.rng is own_rng

    def test_ignores_components_that_do_not_declare_an_rng_attribute(self):
        # StubSolutionsCreation, StubEvaluation, StubTermination, StubVariation and
        # StubReplacement have no rng attribute; construction must not raise.
        algorithm = build_algorithm()

        assert not hasattr(algorithm.solutions_creation, "rng")


class StubArchive:
    """Minimal stand-in for jmetal.util.archive.Archive: just accumulates."""

    def __init__(self):
        self.solution_list: list[int] = []

    def add(self, solution: int) -> bool:
        self.solution_list.append(solution)
        return True


class StubEvaluationWithArchive(StubEvaluation):
    """Mirrors SequentialEvaluationWithArchive: feeds every evaluated solution into
    an archive as a side effect, independent of what the template keeps as its
    population.
    """

    def __init__(self, problem: object, archive: StubArchive):
        super().__init__(problem)
        self.archive = archive

    def evaluate(self, solution_list: list[int]) -> list[int]:
        evaluated = super().evaluate(solution_list)
        for solution in evaluated:
            self.archive.add(solution)
        return evaluated


class StubSolutionWithObjectives:
    """A minimal stand-in with just enough shape (`.objectives`) for
    `distance_based_subset_selection_robust` -- the plain ints the other stubs use
    don't have that attribute.
    """

    def __init__(self, *objectives: float):
        self.objectives = list(objectives)


def _minimal_algorithm(archive=None) -> EvolutionaryAlgorithm:
    """An EvolutionaryAlgorithm with no-op components, for testing result() against
    manually-set .solutions/.archive without going through a real run().
    """
    return EvolutionaryAlgorithm(
        name="StubEA",
        solutions_creation=StubSolutionsCreation(0),
        evaluation=StubEvaluation(problem=object()),
        termination=StubTermination(max_evaluations=0),
        selection=StubSelection(),
        variation=StubVariation(0),
        replacement=StubReplacement(),
        archive=archive,
    )


class TestExternalArchive:
    def test_without_an_archive_result_returns_the_population(self):
        algorithm = build_algorithm(population_size=4, offspring_size=4, generations=1)

        algorithm.run()

        assert algorithm.result() == algorithm.solutions

    def test_with_an_archive_no_larger_than_the_population_returns_it_directly(self):
        # No subset selection needed here: a bounded archive (e.g.
        # CrowdingDistanceArchive) never exceeds the population size to begin
        # with, so this is the common case for it.
        archive = StubArchive()
        archive.solution_list.extend(
            [StubSolutionWithObjectives(0.0, 1.0), StubSolutionWithObjectives(0.5, 0.5)]
        )
        algorithm = _minimal_algorithm(archive=archive)
        algorithm.solutions = [StubSolutionWithObjectives(0.0, 1.0), StubSolutionWithObjectives(0.5, 0.5)]

        assert algorithm.result() is archive.solution_list

    def test_with_an_archive_larger_than_the_population_is_reduced_via_subset_selection(self):
        # This is the case an unbounded archive (e.g. NonDominatedSolutionsArchive)
        # actually hits: it can accumulate far more solutions than the population
        # size over a run, so result() must reduce it rather than return it whole.
        archive = StubArchive()
        archive.solution_list.extend(
            StubSolutionWithObjectives(float(i), float(10 - i)) for i in range(10)
        )
        algorithm = _minimal_algorithm(archive=archive)
        algorithm.solutions = [StubSolutionWithObjectives(0.0, 0.0) for _ in range(4)]

        result = algorithm.result()

        assert len(result) == 4
        assert result is not archive.solution_list
        # Every returned solution must still come from the archive, not be invented.
        assert all(solution in archive.solution_list for solution in result)

    def test_the_archive_accumulates_every_evaluated_solution_not_just_survivors(self):
        # Population size 4, one generation of 4 offspring: 8 evaluations total,
        # all of which the archive should have seen even though only 4 survive
        # into algorithm.solutions.
        archive = StubArchive()
        algorithm = EvolutionaryAlgorithm(
            name="StubEA",
            solutions_creation=StubSolutionsCreation(4),
            evaluation=StubEvaluationWithArchive(problem=object(), archive=archive),
            termination=StubTermination(max_evaluations=8),
            selection=StubSelection(),
            variation=StubVariation(4),
            replacement=StubReplacement(),
            archive=archive,
        )

        algorithm.run()

        assert len(archive.solution_list) == 8
        assert len(algorithm.solutions) == 4
