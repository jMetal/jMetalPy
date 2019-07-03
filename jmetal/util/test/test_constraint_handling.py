import unittest

from jmetal.core.solution import Solution
from jmetal.util.constraint_handling import is_feasible, number_of_violated_constraints, \
    overall_constraint_violation_degree, feasibility_ratio


class ConstraintHandlingTestCases(unittest.TestCase):

    def test_should_is_feasible_return_true_if_the_solution_has_no_constraints(self) -> None:
        solution = Solution(number_of_variables=2, number_of_objectives=2, number_of_constraints=0)

        self.assertEqual(True, is_feasible(solution))

    def test_should_is_feasible_return_true_if_the_solution_has_constraints_and_is_feasible(self) -> None:
        solution = Solution(number_of_variables=2, number_of_objectives=2, number_of_constraints=1)
        solution.constraints[0] = 0

        self.assertEqual(True, is_feasible(solution))

    def test_should_is_feasible_return_false_if_the_solution_has_is_not_feasible(self) -> None:
        solution = Solution(number_of_variables=2, number_of_objectives=2, number_of_constraints=1)
        solution.constraints[0] = -1

        self.assertEqual(False, is_feasible(solution))

    def test_should_number_of_violated_constraints_return_zero_if_the_solution_has_no_constraints(self) -> None:
        solution = Solution(number_of_variables=2, number_of_objectives=2, number_of_constraints=0)

        self.assertEqual(0, number_of_violated_constraints(solution))

    def test_should_number_of_violated_constraints_return_zero_if_the_solution_has_not_violated_constraints(self) -> None:
        solution = Solution(number_of_variables=2, number_of_objectives=2, number_of_constraints=2)

        self.assertEqual(0, number_of_violated_constraints(solution))

    def test_should_number_of_violated_constraints_return_the_right_number_of_violated_constraints(self) -> None:
        solution = Solution(number_of_variables=2, number_of_objectives=2, number_of_constraints=2)
        solution.constraints[0] = 0
        solution.constraints[1] = -2

        self.assertEqual(1, number_of_violated_constraints(solution))

    def test_should_constraint_violation_degree_return_zero_if_the_solution_has_no_constraints(self) -> None:
        solution = Solution(number_of_variables=2, number_of_objectives=2, number_of_constraints=0)

        self.assertEqual(0, overall_constraint_violation_degree(solution))

    def test_should_constraint_violation_degree_return_zero_if_the_solution_has_not_violated_constraints(self) -> None:
        solution = Solution(number_of_variables=2, number_of_objectives=2, number_of_constraints=2)

        self.assertEqual(0, overall_constraint_violation_degree(solution))

    def test_should_constraint_violation_degree_return_the_right_violation_degree(self) -> None:
        solution = Solution(number_of_variables=2, number_of_objectives=2, number_of_constraints=2)
        solution.constraints[0] = -1
        solution.constraints[1] = -2

        self.assertEqual(-3, overall_constraint_violation_degree(solution))

    def test_should_feasibility_ratio_raise_and_exception_if_the_solution_list_is_empty(self) -> None:
        with self.assertRaises(Exception):
            feasibility_ratio([])

    def test_should_feasibility_ratio_return_zero_if_all_the_solutions_in_a_list_are_unfeasible(self) -> None:
        solution1 = Solution(2, 2, 2)
        solution2 = Solution(2, 2, 2)
        solution1.constraints[0] = 0
        solution1.constraints[1] = -1
        solution2.constraints[0] = -2
        solution2.constraints[1] = 0

        self.assertEqual(0, feasibility_ratio([solution1, solution2]))

    def test_should_feasibility_ratio_return_one_if_all_the_solutions_in_a_list_are_feasible(self) -> None:
        solution1 = Solution(2, 2, 2)
        solution2 = Solution(2, 2, 2)
        solution1.constraints[0] = 0
        solution1.constraints[1] = 0
        solution2.constraints[0] = 0
        solution2.constraints[1] = 0

        self.assertEqual(1.0, feasibility_ratio([solution1, solution2]))

    def test_should_feasibility_ratio_return_the_right_percentage_of_feasible_solutions(self) -> None:
        solution1 = Solution(2, 2, 1)
        solution2 = Solution(2, 2, 1)
        solution3 = Solution(2, 2, 1)
        solution1.constraints[0] = -1
        solution2.constraints[0] = 0
        solution3.constraints[0] = -2

        self.assertEqual(1/3, feasibility_ratio([solution1, solution2, solution3]))


if __name__ == '__main__':
    unittest.main()
