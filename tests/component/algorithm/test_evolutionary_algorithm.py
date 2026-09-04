"""Tests for the EvolutionaryAlgorithm template, using test doubles for every component.

These tests exercise the template's run() loop in isolation from any real problem or
operator -- "solutions" are plain integers -- so that a failure here can only be a
template bug, not a bug in one of the catalogue components (which have their own
tests).
"""

import threading

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
