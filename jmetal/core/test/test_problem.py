import unittest

from jmetal.core.problem import FloatProblem, IntegerProblem
from jmetal.core.solution import FloatSolution, IntegerSolution


class DummyIntegerProblem(IntegerProblem):

    def __init__(self):
        super(DummyIntegerProblem, self).__init__()

    def evaluate(self, solution: IntegerSolution) -> IntegerSolution:
        pass

    def get_name(self) -> str:
        pass


class DummyFloatProblem(FloatProblem):

    def __init__(self):
        super(DummyFloatProblem, self).__init__()

    def evaluate(self, solution: FloatSolution) -> FloatSolution:
        pass

    def get_name(self) -> str:
        pass


class FloatProblemTestCases(unittest.TestCase):

    def test_should_default_constructor_create_a_valid_problem(self) -> None:
        number_of_objectives = 2
        number_of_constraints = 0
        lower_bound = [-1.0]
        upper_bound = [1.0]

        problem = DummyFloatProblem()
        problem.lower_bound = lower_bound
        problem.upper_bound = upper_bound
        problem.number_of_constraints = number_of_constraints
        problem.number_of_objectives = number_of_objectives
        problem.number_of_variables = len(lower_bound)

        self.assertEqual(1, problem.number_of_variables)
        self.assertEqual(2, problem.number_of_objectives)
        self.assertEqual(0, problem.number_of_constraints)
        self.assertEqual([-1], problem.lower_bound)
        self.assertEqual([1], problem.upper_bound)

    def test_should_create_solution_create_a_valid_solution(self) -> None:
        problem = DummyFloatProblem()
        problem.number_of_variables = 2
        problem.number_of_objectives = 2
        problem.number_of_constraints = 0
        problem.lower_bound = [-1.0, -2.0]
        problem.upper_bound = [1.0, 2.0]

        solution = problem.create_solution()
        self.assertNotEqual(None, solution)
        self.assertTrue(-1.0 <= solution.variables[0] <= 1.0)
        self.assertTrue(-2.0 <= solution.variables[1] <= 2.0)


class IntegerProblemTestCases(unittest.TestCase):

    def test_should_default_constructor_create_a_valid_problem(self) -> None:
        problem = DummyIntegerProblem()
        problem.number_of_variables = 1
        problem.number_of_objectives = 2
        problem.number_of_constraints = 0
        problem.lower_bound = [-1]
        problem.upper_bound = [1]

        self.assertEqual(1, problem.number_of_variables)
        self.assertEqual(2, problem.number_of_objectives)
        self.assertEqual(0, problem.number_of_constraints)
        self.assertEqual([-1], problem.lower_bound)
        self.assertEqual([1], problem.upper_bound)

    def test_should_create_solution_create_a_valid_solution(self) -> None:
        problem = DummyIntegerProblem()
        problem.number_of_variables = 2
        problem.number_of_objectives = 2
        problem.number_of_constraints = 0
        problem.lower_bound = [-1, -2]
        problem.upper_bound = [1, 2]

        solution = problem.create_solution()
        self.assertNotEqual(None, solution)
        self.assertTrue(-1 <= solution.variables[0] <= 1)
        self.assertTrue(-2 <= solution.variables[1] <= 2)


if __name__ == '__main__':
    unittest.main()
