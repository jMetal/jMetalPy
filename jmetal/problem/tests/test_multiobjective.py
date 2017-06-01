import unittest

from jmetal.problem.multiobjectiveproblem import Kursawe

__author__ = "Antonio J. Nebro"


class KursaweTestCase(unittest.TestCase):

    def setUp(self):
        pass

    def test_should_constructor_create_a_non_null_object(self) -> None:
        problem = Kursawe(3)
        self.assertIsNotNone(problem)

    def test_should_constructor_create_a_valid_problem_with_default_settings(self) -> None:
        problem = Kursawe()
        self.assertEqual(3, problem.number_of_variables)
        self.assertEqual(2, problem.number_of_objectives)
        self.assertEqual(0, problem.number_of_constraints)
        self.assertEqual([-5.0, -5.0, -5.0], problem.lower_bound)
        self.assertEqual([5.0, 5.0, 5.0], problem.upper_bound)

    def test_should_constructor_create_a_valid_problem_with_5_variables(self) -> None:
        problem = Kursawe(5)
        self.assertEqual(5, problem.number_of_variables)
        self.assertEqual(2, problem.number_of_objectives)
        self.assertEqual(0, problem.number_of_constraints)

        self.assertEqual([-5.0, -5.0, -5.0, -5.0, -5.0], problem.lower_bound)
        self.assertEqual([5.0, 5.0, 5.0, 5.0, 5.0], problem.upper_bound)

    def test_should_create_solution_create_a_valid_float_solution(self) -> None:
        problem = Kursawe(3)
        solution = problem.create_solution()
        self.assertEqual(3, solution.number_of_variables)
        self.assertEqual(3, len(solution.variables))
        self.assertEqual(2, solution.number_of_objectives)
        self.assertEqual(2, len(solution.objectives))
        self.assertEqual(0, problem.number_of_constraints)
        self.assertEqual([-5.0, -5.0, -5.0], problem.lower_bound)
        self.assertEqual([5.0, 5.0, 5.0], problem.upper_bound)
        self.assertTrue(solution.variables[0] >= -5.0)
        self.assertTrue(solution.variables[0] <= 5.0)

<<<<<<< HEAD:jmetal/problem/test/kursaweTest.py
    def test_should_get_name_return_the_right_name(self):
        problem = Kursawe()
        self.assertEqual("Kursawe", problem.get_name())
=======
>>>>>>> 0c3a3b5ecb116c4ec22fd8540d233f554fdd700a:jmetal/problem/tests/test_multiobjective.py

if __name__ == '__main__':
    unittest.main()
