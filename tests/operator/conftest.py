from collections.abc import Callable
from typing import TypeVar

import pytest

from jmetal.core.solution import (
    BinarySolution,
    CompositeSolution,
    FloatSolution,
    IntegerSolution,
    PermutationSolution,
    Solution,
)

# Type variable for solutions
S = TypeVar("S", bound=Solution)

# --- Common solution factories ---


@pytest.fixture
def float_solution_factory() -> Callable[[list[float], list[float]], FloatSolution]:
    """Factory fixture for creating float solutions with specified objectives and variables."""

    def _create_float_solution(
        objectives: list[float], variables: list[float] = None
    ) -> FloatSolution:
        if variables is None:
            variables = [0.0] * len(objectives)
        solution = FloatSolution(
            lower_bound=[0.0] * len(variables),
            upper_bound=[1.0] * len(variables),
            number_of_objectives=len(objectives),
        )
        solution.objectives = objectives
        solution.variables = variables
        return solution

    return _create_float_solution


@pytest.fixture
def binary_solution_factory() -> Callable[[list[bool], list[float]], BinarySolution]:
    """Factory fixture for creating binary solutions."""

    def _create_binary_solution(
        bits: list[bool] = None, objectives: list[float] = None, number_of_variables: int = 1
    ) -> BinarySolution:
        if objectives is None:
            objectives = [0.0]  # Default to single objective
        if bits is None:
            bits = [True, False, True, False]  # Default bits

        solution = BinarySolution(
            number_of_variables=number_of_variables, number_of_objectives=len(objectives)
        )
        if number_of_variables == 1:
            solution.variables = [bits]
        else:
            solution.variables = [bits] * number_of_variables
        solution.objectives = objectives
        return solution

    return _create_binary_solution


@pytest.fixture
def integer_solution_factory() -> Callable[[list[int], list[float]], IntegerSolution]:
    """Factory fixture for creating integer solutions."""

    def _create_integer_solution(
        variables: list[int], objectives: list[float] = None
    ) -> IntegerSolution:
        if objectives is None:
            objectives = [0.0] * 1  # Default to single objective

        solution = IntegerSolution(
            lower_bound=[0] * len(variables),
            upper_bound=[100] * len(variables),  # Arbitrary upper bound
            number_of_objectives=len(objectives),
        )
        solution.variables = variables
        solution.objectives = objectives
        return solution

    return _create_integer_solution


@pytest.fixture
def permutation_solution_factory() -> Callable[[list[int], list[float]], PermutationSolution]:
    """Factory fixture for creating permutation solutions."""

    def _create_permutation_solution(
        permutation: list[int] = None, objectives: list[float] = None
    ) -> PermutationSolution:
        if permutation is None:
            permutation = list(range(5))  # Default permutation
        if objectives is None:
            objectives = [0.0]  # Default to single objective

        solution = PermutationSolution(
            number_of_variables=1,
            permutation_length=len(permutation),
            number_of_objectives=len(objectives),
        )
        solution.variables = [permutation]
        solution.objectives = objectives
        return solution

    return _create_permutation_solution


@pytest.fixture
def composite_solution_factory(
    float_solution_factory,
    binary_solution_factory,
    integer_solution_factory,
    permutation_solution_factory,
) -> Callable[[list[type[Solution]]], CompositeSolution]:
    """Factory fixture for creating composite solutions."""

    def _create_composite_solution(types: list[type[Solution]]) -> CompositeSolution:
        solutions = []
        for sol_type in types:
            if sol_type == FloatSolution:
                solutions.append(float_solution_factory([0.5, 0.5]))
            elif sol_type == BinarySolution:
                solutions.append(binary_solution_factory([True, False, True]))
            elif sol_type == IntegerSolution:
                solutions.append(integer_solution_factory([1, 2, 3]))
            else:
                raise ValueError(f"Unsupported solution type: {sol_type}")
        return CompositeSolution(solutions)

    return _create_composite_solution


@pytest.fixture
def solution_list_factory(
    float_solution_factory,
) -> Callable[[list[list[float]]], list[FloatSolution]]:
    """Create a list of float solutions from a list of objective values."""

    def _create_solution_list(objectives_list: list[list[float]]) -> list[FloatSolution]:
        return [float_solution_factory(objectives) for objectives in objectives_list]

    return _create_solution_list


@pytest.fixture
def non_dominated_solutions(float_solution_factory) -> list[FloatSolution]:
    """Create a set of non-dominated solutions."""
    return [
        float_solution_factory([1.0, 0.0]),
        float_solution_factory([0.0, 1.0]),
        float_solution_factory([0.5, 0.5]),
    ]


@pytest.fixture
def dominated_solutions(float_solution_factory) -> list[FloatSolution]:
    """Create a set of dominated solutions."""
    return [
        float_solution_factory([0.9, 0.2]),
        float_solution_factory([0.8, 0.3]),
        float_solution_factory([0.2, 0.9]),
    ]


@pytest.fixture
def mixed_front_solutions(non_dominated_solutions, dominated_solutions) -> list[FloatSolution]:
    """Create a mixed front with both non-dominated and dominated solutions."""
    return non_dominated_solutions + dominated_solutions


# Common test parameters
def pytest_addoption(parser):
    """Add custom command line options for tests."""
    parser.addoption("--run-slow", action="store_true", default=False, help="run slow tests")


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "slow: mark test as slow to run")


def pytest_collection_modifyitems(config, items):
    """Skip slow tests unless --run-slow is given."""
    if config.getoption("--run-slow"):
        return
    skip_slow = pytest.mark.skip(reason="need --run-slow option to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)
