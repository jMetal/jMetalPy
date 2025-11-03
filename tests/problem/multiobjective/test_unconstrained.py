import numpy as np
import pytest

from jmetal.problem.multiobjective.unconstrained import (
    Fonseca,
    Kursawe,
    Schaffer,
    Viennet2,
    OneZeroMax,
)

# Test data
FONSECA_TEST_CASES = [
    # Test case 1: Random values
    ([-1.3, 1.5, 1.21], [0.9915701157761243, 0.9996751762268354]),
    # Test case 2: All zeros
    ([0.0, 0.0, 0.0], [0.6321205588285578, 0.6321205588285578]),
]

KURSAWE_TEST_CASES = [
    # Test case 1: Edge values
    ([-5.0, 5.0, 5.0], [-4.8623346886842835, 7.791492659222151]),
    # Test case 2: All zeros
    ([0.0, 0.0, 0.0], [-20.0, 0.0]),
]

SCHAFFER_TEST_CASES = [
    # Test case 1: Positive value
    ([3.0], [9.0, 1.0]),  # x² and (x-2)²
    # Test case 2: Negative value
    ([-2.6], [6.76, 21.16]),  # x² and (x-2)²
]

VIENNET2_TEST_CASES = [
    # Test case 1: Specific values
    ([-2.6, 1.5], [14.0607692307, -11.8818055555, -11.1532369747]),
    # Test case 2: All zeros
    ([0.0, 0.0], [5.07692307692, -16.25, -12.994285714285715]),
]

ONE_ZERO_MAX_TEST_CASES = [
    ([True] * 10 + [False] * 6, [-10.0, -6.0]),  # 10 ones, 6 zeros
    ([False] * 10, [0.0, -10.0]),  # 0 ones, 10 zeros
    ([True] * 10, [-10.0, 0.0]),  # 10 ones, 0 zeros
]

# Fixtures
@pytest.fixture
def fonseca():
    return Fonseca()

@pytest.fixture
def kursawe():
    return Kursawe()

@pytest.fixture
def schaffer():
    return Schaffer()

@pytest.fixture
def viennet2():
    return Viennet2()

@pytest.fixture
def one_zero_max():
    return OneZeroMax()

# Parameterized tests
@pytest.mark.parametrize("variables, expected_objectives", FONSECA_TEST_CASES)
def test_fonseca_evaluate(fonseca, variables, expected_objectives):
    """Test Fonseca problem evaluation."""
    solution = fonseca.create_solution()
    solution.variables = variables
    fonseca.evaluate(solution)
    
    assert len(solution.objectives) == 2
    for i, (obj, expected) in enumerate(zip(solution.objectives, expected_objectives)):
        assert obj == pytest.approx(expected, rel=1e-6), f"Objective {i} mismatch"

@pytest.mark.parametrize("variables, expected_objectives", KURSAWE_TEST_CASES)
def test_kursawe_evaluate(kursawe, variables, expected_objectives):
    """Test Kursawe problem evaluation."""
    solution = kursawe.create_solution()
    solution.variables = variables.copy()
    kursawe.evaluate(solution)
    
    assert len(solution.objectives) == 2
    for i, (obj, expected) in enumerate(zip(solution.objectives, expected_objectives)):
        if not np.isinf(expected):
            assert obj == pytest.approx(expected, rel=1e-6), f"Objective {i} mismatch"

@pytest.mark.parametrize("variables, expected_objectives", SCHAFFER_TEST_CASES)
def test_schaffer_evaluate(schaffer, variables, expected_objectives):
    """Test Schaffer problem evaluation."""
    solution = schaffer.create_solution()
    solution.variables = variables.copy()
    schaffer.evaluate(solution)
    
    assert len(solution.objectives) == 2
    for i, (obj, expected) in enumerate(zip(solution.objectives, expected_objectives)):
        assert obj == pytest.approx(expected, rel=1e-6), f"Objective {i} mismatch"

@pytest.mark.parametrize("variables, expected_objectives", VIENNET2_TEST_CASES)
def test_viennet2_evaluate(viennet2, variables, expected_objectives):
    """Test Viennet2 problem evaluation."""
    solution = viennet2.create_solution()
    solution.variables = variables.copy()
    viennet2.evaluate(solution)
    
    assert len(solution.objectives) == 3
    for i, (obj, expected) in enumerate(zip(solution.objectives, expected_objectives)):
        assert obj == pytest.approx(expected, rel=1e-6), f"Objective {i} mismatch"

# OneZeroMax specific tests
def test_one_zero_max_initialization(one_zero_max):
    """Test OneZeroMax problem initialization."""
    assert one_zero_max.number_of_variables() == 256
    assert one_zero_max.number_of_objectives() == 2
    assert one_zero_max.number_of_constraints() == 0
    assert one_zero_max.name() == "OneZeroMax"

@pytest.mark.parametrize("bits, expected_objectives", ONE_ZERO_MAX_TEST_CASES)
def test_one_zero_max_evaluate(one_zero_max, bits, expected_objectives):
    """Test OneZeroMax problem evaluation."""
    # Create a custom OneZeroMax with smaller bit length for testing
    problem = OneZeroMax(number_of_bits=len(bits))
    solution = problem.create_solution()
    solution.variables = bits
    
    problem.evaluate(solution)
    
    assert len(solution.objectives) == 2
    for i, (obj, expected) in enumerate(zip(solution.objectives, expected_objectives)):
        assert obj == pytest.approx(expected, rel=1e-6), f"Objective {i} mismatch"

def test_one_zero_max_create_solution(one_zero_max):
    """Test OneZeroMax solution creation."""
    solution = one_zero_max.create_solution()
    assert len(solution.variables) == 256
    assert all(isinstance(bit, bool) for bit in solution.variables)

# Test problem properties
def test_problem_properties():
    """Test common problem properties."""
    # Test Fonseca bounds
    fonseca = Fonseca()
    assert len(fonseca.lower_bound) == 3
    assert len(fonseca.upper_bound) == 3
    assert all(lb == -4 for lb in fonseca.lower_bound)
    assert all(ub == 4 for ub in fonseca.upper_bound)
    
    # Test Viennet2 bounds
    viennet2 = Viennet2()
    assert len(viennet2.lower_bound) == 2
    assert len(viennet2.upper_bound) == 2
    assert all(lb == -4 for lb in viennet2.lower_bound)
    assert all(ub == 4 for ub in viennet2.upper_bound)

# Test solution creation
def test_solution_creation():
    """Test that solutions are created with valid values within bounds."""
    problem = Fonseca()
    solution = problem.create_solution()
    
    for var, lb, ub in zip(solution.variables, problem.lower_bound, problem.upper_bound):
        assert lb <= var <= ub
