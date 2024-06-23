import unittest

from jmetal.core.problem import FloatProblem, IntegerProblem
from jmetal.core.solution import FloatSolution, IntegerSolution


class FakeIntegerProblem(IntegerProblem):
    """
    Fake class used only for testing purposes.
    """

    def __init__(self):
        super(FakeIntegerProblem, self).__init__()

    def number_of_objectives(self) -> int:
        return 2

    def number_of_constraints(self) -> int:
        return 0

    def evaluate(self, solution: IntegerSolution) -> IntegerSolution:
        return solution

    def name(self) -> str:
        return "Dummy integer problem"


class FakeFloatProblem(FloatProblem):
    """
    Fake class used only for testing purposes.
    """

    def __init__(self):
        super(FakeFloatProblem, self).__init__()

    def number_of_objectives(self) -> int:
        return 2

    def number_of_constraints(self) -> int:
        return 0

    def evaluate(self, solution: FloatSolution) -> FloatSolution:
        return solution

    def name(self) -> str:
        return "Dummy float problem"


class FloatProblemTestCases(unittest.TestCase):
    def test_should_default_constructor_create_a_valid_problem(self):
        lower_bound = [-1.0]
        upper_bound = [1.0]

        problem = FakeFloatProblem()
        problem.lower_bound = lower_bound
        problem.upper_bound = upper_bound

        self.assertEqual(1, problem.number_of_variables())
        self.assertEqual(2, problem.number_of_objectives())
        self.assertEqual(0, problem.number_of_constraints())
        self.assertEqual([-1], problem.lower_bound)
        self.assertEqual([1], problem.upper_bound)

    def test_should_create_solution_create_a_valid_solution(self):
        problem = FakeFloatProblem()

        problem.lower_bound = [-1.0, -2.0]
        problem.upper_bound = [1.0, 2.0]

        solution = problem.create_solution()
        self.assertIsNotNone(solution)
        self.assertTrue(-1.0 <= solution.variables[0] <= 1.0)
        self.assertTrue(-2.0 <= solution.variables[1] <= 2.0)


class IntegerProblemTestCases(unittest.TestCase):
    def test_should_default_constructor_create_a_valid_problem(self):
        problem = FakeIntegerProblem()

        problem.lower_bound = [-1]
        problem.upper_bound = [1]

        self.assertEqual(1, problem.number_of_variables())
        self.assertEqual(2, problem.number_of_objectives())
        self.assertEqual(0, problem.number_of_constraints())
        self.assertEqual([-1], problem.lower_bound)
        self.assertEqual([1], problem.upper_bound)

    def test_should_create_solution_create_a_valid_solution(self):
        problem = FakeIntegerProblem()

        problem.lower_bound = [-1, -2]
        problem.upper_bound = [1, 2]

        print(problem.number_of_variables())

        solution = problem.create_solution()
        self.assertIsNotNone(solution)
        self.assertTrue(-1 <= solution.variables[0] <= 1)
        self.assertTrue(-2 <= solution.variables[1] <= 2)


if __name__ == "__main__":
    unittest.main()
