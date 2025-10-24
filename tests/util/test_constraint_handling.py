import unittest

from jmetal.core.solution import FloatSolution
from jmetal.util.constraint_handling import (
    feasibility_ratio,
    is_feasible,
    number_of_violated_constraints,
    overall_constraint_violation_degree,
)


class ConstraintHandlingTestCases(unittest.TestCase):
    def test_should_is_feasible_return_true_if_the_solution_has_no_constraints(self) -> None:
        solution = FloatSolution([0.0, 0.0], [1.0, 1.0], number_of_objectives=2, number_of_constraints=0)

        self.assertEqual(True, is_feasible(solution))

    def test_should_is_feasible_return_true_if_the_solution_has_constraints_and_is_feasible(self) -> None:
        solution = FloatSolution([0.0, 0.0], [1.0, 1.0], number_of_objectives=2, number_of_constraints=1)
        # A constraint value >= 0 means no violation
        solution.constraints[0] = 0.0
        solution.variables = [0.0, 0.0]

        self.assertTrue(is_feasible(solution), 
                       f"Expected solution with constraint {solution.constraints} to be feasible")

    def test_should_is_feasible_return_false_if_the_solution_has_is_not_feasible(self) -> None:
        solution = FloatSolution([0.0, 0.0], [1.0, 1.0], number_of_objectives=2, number_of_constraints=1)
        # A negative constraint value means violation
        solution.constraints[0] = -0.5  # Violation
        solution.variables = [0.0, 0.0]

        # The solution has a constraint violation, so it should not be feasible
        self.assertFalse(is_feasible(solution), 
                        f"Expected is_feasible to return False for solution with constraint {solution.constraints}")

    def test_should_number_of_violated_constraints_return_zero_if_the_solution_has_no_constraints(self) -> None:
        solution = FloatSolution([0.0, 0.0], [1.0, 1.0], number_of_objectives=2, number_of_constraints=0)

        self.assertEqual(0, number_of_violated_constraints(solution))

    def test_should_number_of_violated_constraints_return_zero_if_the_solution_has_not_violated_constraints(
        self,
    ) -> None:
        solution = FloatSolution([0.0, 0.0], [1.0, 1.0], number_of_objectives=2, number_of_constraints=2)
        # Non-negative constraint values mean no violations
        solution.constraints[0] = 0.0
        solution.constraints[1] = 1.0  # Positive is also considered non-violating
        solution.variables = [0.0, 0.0]

        self.assertEqual(0, number_of_violated_constraints(solution), 
                         f"Expected 0 violated constraints, got {number_of_violated_constraints(solution)} for constraints {solution.constraints}")

    def test_should_number_of_violated_constraints_return_the_right_number_of_violated_constraints(self) -> None:
        solution = FloatSolution([0.0, 0.0], [1.0, 1.0], number_of_objectives=2, number_of_constraints=3)
        # Negative constraint values indicate violations
        solution.constraints[0] = -1.0  # Violation
        solution.constraints[1] = 0.0   # No violation
        solution.constraints[2] = -0.5  # Violation
        solution.variables = [0.0, 0.0]

        # We expect 2 violated constraints (the negative ones)
        violated = number_of_violated_constraints(solution)
        self.assertEqual(2, violated, 
                        f"Expected 2 violated constraints, got {violated} for constraints {solution.constraints}")

    def test_should_constraint_violation_degree_return_zero_if_the_solution_has_no_constraints(self) -> None:
        solution = FloatSolution([0.0, 0.0], [1.0, 1.0], number_of_objectives=2, number_of_constraints=0)

        self.assertEqual(0.0, overall_constraint_violation_degree(solution))

    def test_should_constraint_violation_degree_return_zero_if_the_solution_has_not_violated_constraints(
        self,
    ) -> None:
        solution = FloatSolution([0.0, 0.0], [1.0, 1.0], number_of_objectives=2, number_of_constraints=2)
        # Non-negative constraint values mean no violations
        solution.constraints[0] = 0.0
        solution.constraints[1] = 1.0  # Positive is also considered non-violating
        solution.variables = [0.0, 0.0]

        self.assertEqual(0.0, overall_constraint_violation_degree(solution),
                         f"Expected 0.0 violation degree, got {overall_constraint_violation_degree(solution)} for constraints {solution.constraints}")

    def test_should_constraint_violation_degree_return_the_right_violation_degree(self) -> None:
        solution = FloatSolution([0.0, 0.0], [1.0, 1.0], number_of_objectives=2, number_of_constraints=3)
        # Negative constraint values indicate violations, and their sum is the violation degree
        solution.constraints[0] = -1.0  # Violation
        solution.constraints[1] = 0.0   # No violation
        solution.constraints[2] = -0.5  # Violation
        solution.variables = [0.0, 0.0]
        
        # The function should return the sum of negative constraint values: -1.0 + (-0.5) = -1.5
        expected_violation = -1.5
        actual_violation = overall_constraint_violation_degree(solution)
        self.assertAlmostEqual(expected_violation, actual_violation, 
                             msg=f"Expected violation degree {expected_violation}, got {actual_violation} for constraints {solution.constraints}")

    def test_should_feasibility_ratio_raise_and_exception_if_the_solution_list_is_empty(self) -> None:
        with self.assertRaises(Exception):
            feasibility_ratio([])

    def test_should_feasibility_ratio_return_one_if_all_the_solutions_in_a_list_are_feasible(self) -> None:
        solution1 = FloatSolution([0.0, 0.0], [1.0, 1.0], number_of_objectives=2, number_of_constraints=0)
        solution2 = FloatSolution([0.0, 0.0], [1.0, 1.0], number_of_objectives=2, number_of_constraints=0)
        solution3 = FloatSolution([0.0, 0.0], [1.0, 1.0], number_of_objectives=2, number_of_constraints=0)

        self.assertEqual(1.0, feasibility_ratio([solution1, solution2, solution3]))

    def test_should_feasibility_ratio_return_zero_if_all_the_solutions_in_a_list_are_unfeasible(self) -> None:
        solution1 = FloatSolution([0.0, 0.0], [1.0, 1.0], number_of_objectives=2, number_of_constraints=1)
        solution1.constraints[0] = -0.1  # Violation (negative)
        solution2 = FloatSolution([0.0, 0.0], [1.0, 1.0], number_of_objectives=2, number_of_constraints=1)
        solution2.constraints[0] = -1.0  # Violation (negative)
        solution3 = FloatSolution([0.0, 0.0], [1.0, 1.0], number_of_objectives=2, number_of_constraints=1)
        solution3.constraints[0] = -0.5  # Violation (negative)

        # All solutions have violations (negative constraints), so ratio should be 0.0
        ratio = feasibility_ratio([solution1, solution2, solution3])
        self.assertEqual(0.0, ratio, msg=f"Expected ratio of 0.0, got {ratio}")

    def test_should_feasibility_ratio_return_the_right_percentage_of_feasible_solutions(self) -> None:
        solution1 = FloatSolution([0.0, 0.0], [1.0, 1.0], number_of_objectives=2, number_of_constraints=1)
        solution1.constraints[0] = 0.0   # Feasible (non-negative)
        solution2 = FloatSolution([0.0, 0.0], [1.0, 1.0], number_of_objectives=2, number_of_constraints=1)
        solution2.constraints[0] = -1.0  # Violation (negative)
        solution3 = FloatSolution([0.0, 0.0], [1.0, 1.0], number_of_objectives=2, number_of_constraints=1)
        solution3.constraints[0] = -0.5  # Violation (negative)

        # 1 out of 3 solutions is feasible (non-negative constraint)
        ratio = feasibility_ratio([solution1, solution2, solution3])
        self.assertAlmostEqual(1.0 / 3.0, ratio, msg=f"Expected ratio of 1/3, got {ratio}")


if __name__ == "__main__":
    unittest.main()
