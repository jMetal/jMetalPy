"""Tests for the EvolutionStrategy algorithm.

Covers unit tests for each public method and a lightweight integration test
that runs the algorithm on the Sphere problem.
"""

from copy import copy
from unittest.mock import MagicMock

import pytest

from jmetal.algorithm.singleobjective.evolution_strategy import EvolutionStrategy
from jmetal.core.solution import FloatSolution
from jmetal.operator.mutation import PolynomialMutation
from jmetal.problem import Sphere
from jmetal.util.termination_criterion import StoppingByEvaluations


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sphere_problem() -> Sphere:
    """A 3-variable Sphere problem used across multiple tests."""
    return Sphere(number_of_variables=3)


@pytest.fixture
def mutation(sphere_problem: Sphere) -> PolynomialMutation:
    """Default polynomial mutation operator for the Sphere problem."""
    return PolynomialMutation(
        probability=1.0 / sphere_problem.number_of_variables(),
        distribution_index=20,
    )


@pytest.fixture
def termination() -> StoppingByEvaluations:
    """A short termination criterion for unit tests."""
    return StoppingByEvaluations(max_evaluations=100)


def _build_algorithm(
    problem: Sphere,
    mutation: PolynomialMutation,
    termination: StoppingByEvaluations,
    *,
    mu: int = 5,
    lambda_: int = 5,
    elitist: bool = True,
) -> EvolutionStrategy:
    """Helper to create an EvolutionStrategy with common defaults."""
    return EvolutionStrategy(
        problem=problem,
        mu=mu,
        lambda_=lambda_,
        elitist=elitist,
        mutation=mutation,
        termination_criterion=termination,
    )


def _make_solution(
    objective: float,
    constraint_violation: float | None = None,
) -> FloatSolution:
    """Create a minimal FloatSolution with a given objective value and optional constraint."""
    n_constraints = 1 if constraint_violation is not None else 0
    solution = FloatSolution(
        lower_bound=[0.0],
        upper_bound=[1.0],
        number_of_objectives=1,
        number_of_constraints=n_constraints,
    )
    solution.objectives[0] = objective
    solution.variables = [0.5]
    if constraint_violation is not None:
        solution.constraints[0] = constraint_violation
    return solution


# ---------------------------------------------------------------------------
# Unit tests – construction
# ---------------------------------------------------------------------------

class TestEvolutionStrategyConstruction:
    """Tests for correct initialisation of the algorithm."""

    def test_given_valid_params_when_created_then_stores_mu_and_lambda(
        self, sphere_problem, mutation, termination
    ) -> None:
        algorithm = _build_algorithm(
            sphere_problem, mutation, termination, mu=10, lambda_=20
        )

        assert algorithm.mu == 10
        assert algorithm.lambda_ == 20

    def test_given_elitist_true_when_created_then_flag_is_true(
        self, sphere_problem, mutation, termination
    ) -> None:
        algorithm = _build_algorithm(
            sphere_problem, mutation, termination, elitist=True
        )

        assert algorithm.elitist is True

    def test_given_elitist_false_when_created_then_flag_is_false(
        self, sphere_problem, mutation, termination
    ) -> None:
        algorithm = _build_algorithm(
            sphere_problem, mutation, termination, elitist=False
        )

        assert algorithm.elitist is False


# ---------------------------------------------------------------------------
# Unit tests – get_name
# ---------------------------------------------------------------------------

class TestGetName:
    """Tests for the get_name method returning the correct variant label."""

    def test_given_elitist_when_get_name_then_returns_mu_plus_lambda(
        self, sphere_problem, mutation, termination
    ) -> None:
        algorithm = _build_algorithm(
            sphere_problem, mutation, termination, elitist=True
        )

        assert algorithm.get_name() == "(mu + lambda) Evolution Strategy"

    def test_given_non_elitist_when_get_name_then_returns_mu_comma_lambda(
        self, sphere_problem, mutation, termination
    ) -> None:
        algorithm = _build_algorithm(
            sphere_problem, mutation, termination, elitist=False
        )

        assert algorithm.get_name() == "(mu, lambda) Evolution Strategy"


# ---------------------------------------------------------------------------
# Unit tests – selection
# ---------------------------------------------------------------------------

class TestSelection:
    """Tests for the selection operator (identity in ES)."""

    def test_given_population_when_selection_then_returns_same_population(
        self, sphere_problem, mutation, termination
    ) -> None:
        algorithm = _build_algorithm(sphere_problem, mutation, termination)
        population = [_make_solution(i) for i in range(5)]

        result = algorithm.selection(population)

        assert result is population


# ---------------------------------------------------------------------------
# Unit tests – reproduction
# ---------------------------------------------------------------------------

class TestReproduction:
    """Tests for offspring generation via mutation."""

    def test_given_population_when_reproduction_then_offspring_size_equals_lambda(
        self, sphere_problem, mutation, termination
    ) -> None:
        mu, lambda_ = 5, 10
        algorithm = _build_algorithm(
            sphere_problem, mutation, termination, mu=mu, lambda_=lambda_
        )
        population = [sphere_problem.create_solution() for _ in range(mu)]
        for s in population:
            sphere_problem.evaluate(s)

        offspring = algorithm.reproduction(population)

        assert len(offspring) == lambda_

    def test_given_mu_equals_lambda_when_reproduction_then_one_child_per_parent(
        self, sphere_problem, mutation, termination
    ) -> None:
        mu = lambda_ = 4
        algorithm = _build_algorithm(
            sphere_problem, mutation, termination, mu=mu, lambda_=lambda_
        )
        population = [sphere_problem.create_solution() for _ in range(mu)]
        for s in population:
            sphere_problem.evaluate(s)

        offspring = algorithm.reproduction(population)

        assert len(offspring) == lambda_


# ---------------------------------------------------------------------------
# Unit tests – replacement
# ---------------------------------------------------------------------------

class TestReplacement:
    """Tests for the replacement step, including elitist vs non-elitist behaviour."""

    def test_given_elitist_when_replacement_then_pool_contains_parents_and_offspring(
        self, sphere_problem, mutation, termination
    ) -> None:
        algorithm = _build_algorithm(
            sphere_problem, mutation, termination, mu=2, lambda_=2, elitist=True
        )
        parents = [_make_solution(10.0), _make_solution(20.0)]
        offspring = [_make_solution(5.0), _make_solution(15.0)]

        result = algorithm.replacement(parents, offspring)

        # Best 2 from merged pool (5.0, 10.0, 15.0, 20.0) → [5.0, 10.0]
        assert len(result) == 2
        assert result[0].objectives[0] == 5.0
        assert result[1].objectives[0] == 10.0

    def test_given_non_elitist_when_replacement_then_pool_contains_only_offspring(
        self, sphere_problem, mutation, termination
    ) -> None:
        algorithm = _build_algorithm(
            sphere_problem, mutation, termination, mu=2, lambda_=3, elitist=False
        )
        parents = [_make_solution(1.0), _make_solution(2.0)]
        offspring = [_make_solution(10.0), _make_solution(5.0), _make_solution(8.0)]

        result = algorithm.replacement(parents, offspring)

        # Only offspring compete: best 2 from (5.0, 8.0, 10.0) → [5.0, 8.0]
        assert len(result) == 2
        assert result[0].objectives[0] == 5.0
        assert result[1].objectives[0] == 8.0

    def test_given_replacement_when_called_then_returns_mu_solutions(
        self, sphere_problem, mutation, termination
    ) -> None:
        mu = 3
        algorithm = _build_algorithm(
            sphere_problem, mutation, termination, mu=mu, lambda_=5, elitist=True
        )
        parents = [_make_solution(float(i)) for i in range(mu)]
        offspring = [_make_solution(float(i + 10)) for i in range(5)]

        result = algorithm.replacement(parents, offspring)

        assert len(result) == mu


# ---------------------------------------------------------------------------
# Unit tests – constraint handling in replacement
# ---------------------------------------------------------------------------

class TestConstraintHandling:
    """Tests verifying that feasible solutions are preferred over infeasible ones."""

    def test_given_feasible_and_infeasible_when_replacement_then_feasible_first(
        self, sphere_problem, mutation, termination
    ) -> None:
        """A feasible solution with a worse objective must be preferred over
        an infeasible solution with a better objective."""
        algorithm = _build_algorithm(
            sphere_problem, mutation, termination, mu=1, lambda_=1, elitist=True
        )
        feasible = _make_solution(objective=100.0, constraint_violation=0.0)
        infeasible = _make_solution(objective=1.0, constraint_violation=-5.0)

        result = algorithm.replacement([feasible], [infeasible])

        assert len(result) == 1
        assert result[0].objectives[0] == 100.0  # feasible wins

    def test_given_two_infeasible_when_replacement_then_less_violated_first(
        self, sphere_problem, mutation, termination
    ) -> None:
        """Between two infeasible solutions, the one with smaller violation
        (closer to 0) should be ranked first."""
        algorithm = _build_algorithm(
            sphere_problem, mutation, termination, mu=2, lambda_=2, elitist=True
        )
        slightly_violated = _make_solution(objective=50.0, constraint_violation=-1.0)
        heavily_violated = _make_solution(objective=10.0, constraint_violation=-10.0)

        result = algorithm.replacement(
            [slightly_violated], [heavily_violated]
        )

        # Slightly violated (violation degree -1.0) should rank before heavily violated (-10.0)
        assert result[0].constraints[0] == -1.0

    def test_given_two_feasible_when_replacement_then_best_objective_first(
        self, sphere_problem, mutation, termination
    ) -> None:
        """When both solutions are feasible the objective value decides the ranking."""
        algorithm = _build_algorithm(
            sphere_problem, mutation, termination, mu=2, lambda_=2, elitist=True
        )
        better = _make_solution(objective=3.0, constraint_violation=0.0)
        worse = _make_solution(objective=7.0, constraint_violation=0.0)

        result = algorithm.replacement([worse], [better])

        assert result[0].objectives[0] == 3.0
        assert result[1].objectives[0] == 7.0

    def test_given_no_constraints_when_replacement_then_sorted_by_objective(
        self, sphere_problem, mutation, termination
    ) -> None:
        """For unconstrained problems the ordering should be purely by objective."""
        algorithm = _build_algorithm(
            sphere_problem, mutation, termination, mu=3, lambda_=3, elitist=True
        )
        solutions = [_make_solution(5.0), _make_solution(1.0), _make_solution(3.0)]

        result = algorithm.replacement([], solutions)

        objectives = [s.objectives[0] for s in result]
        assert objectives == [1.0, 3.0, 5.0]


# ---------------------------------------------------------------------------
# Integration test – run on Sphere
# ---------------------------------------------------------------------------

class TestEvolutionStrategyIntegration:
    """Lightweight integration tests that run the full algorithm loop."""

    def test_given_sphere_when_elitist_es_runs_then_finds_near_optimal_solution(
        self,
    ) -> None:
        problem = Sphere(number_of_variables=3)
        algorithm = EvolutionStrategy(
            problem=problem,
            mu=10,
            lambda_=10,
            elitist=True,
            mutation=PolynomialMutation(
                probability=1.0 / problem.number_of_variables(),
                distribution_index=20,
            ),
            termination_criterion=StoppingByEvaluations(max_evaluations=5000),
        )

        algorithm.run()
        result = algorithm.result()

        # Sphere optimum is 0.0; after 5 000 evaluations we expect a value < 1.0
        assert result.objectives[0] < 1.0

    def test_given_sphere_when_non_elitist_es_runs_then_completes_without_error(
        self,
    ) -> None:
        problem = Sphere(number_of_variables=3)
        algorithm = EvolutionStrategy(
            problem=problem,
            mu=10,
            lambda_=10,
            elitist=False,
            mutation=PolynomialMutation(
                probability=1.0 / problem.number_of_variables(),
                distribution_index=20,
            ),
            termination_criterion=StoppingByEvaluations(max_evaluations=1000),
        )

        algorithm.run()
        result = algorithm.result()

        # Simply assert that the algorithm completed and returned a valid solution
        assert result is not None
        assert len(result.objectives) == 1
