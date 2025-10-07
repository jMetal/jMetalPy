import unittest
import random
import numpy as np
from unittest.mock import Mock

from jmetal.core.solution import FloatSolution
from jmetal.util.archive import BestSolutionsArchive, distance_based_subset_selection
from jmetal.util.distance import EuclideanDistance


class BestSolutionsArchiveTestCase(unittest.TestCase):

    def setUp(self):
        self.archive = BestSolutionsArchive(maximum_size=5)

    def test_should_create_archive_with_correct_maximum_size(self):
        self.assertEqual(5, self.archive.maximum_size)
        self.assertEqual(0, self.archive.size())

    def test_should_add_solution_when_archive_is_empty(self):
        solution = FloatSolution([], [], 2)
        solution.objectives = [1.0, 2.0]
        
        result = self.archive.add(solution)
        
        self.assertTrue(result)
        self.assertEqual(1, self.archive.size())

    def test_should_add_multiple_non_dominated_solutions(self):
        # Create non-dominated solutions
        solutions = []
        for i in range(3):
            solution = FloatSolution([], [], 2)
            solution.objectives = [float(i), 2.0 - float(i)]  # Trade-off solutions
            solutions.append(solution)
            
        for solution in solutions:
            self.archive.add(solution)
            
        self.assertEqual(3, self.archive.size())

    def test_should_not_exceed_maximum_size_with_crowding_distance_for_2_objectives(self):
        # Add 7 solutions to archive with max size 5
        for i in range(7):
            solution = FloatSolution([], [], 2)
            solution.objectives = [float(i), 6.0 - float(i)]  # Non-dominated front
            self.archive.add(solution)
            
        self.assertEqual(5, self.archive.size())

    def test_should_handle_dominated_solutions_correctly(self):
        # Add first solution
        solution1 = FloatSolution([], [], 2)
        solution1.objectives = [1.0, 1.0]
        self.archive.add(solution1)
        
        # Add dominated solution (should be rejected)
        solution2 = FloatSolution([], [], 2)
        solution2.objectives = [2.0, 2.0]  # Dominated by solution1
        result = self.archive.add(solution2)
        
        self.assertFalse(result)
        self.assertEqual(1, self.archive.size())

    def test_should_handle_dominating_solution_correctly(self):
        # Add first solution
        solution1 = FloatSolution([], [], 2)
        solution1.objectives = [2.0, 2.0]
        self.archive.add(solution1)
        
        # Add dominating solution (should replace first)
        solution2 = FloatSolution([], [], 2)
        solution2.objectives = [1.0, 1.0]  # Dominates solution1
        result = self.archive.add(solution2)
        
        self.assertTrue(result)
        self.assertEqual(1, self.archive.size())
        # Verify the dominating solution is in the archive
        self.assertEqual([1.0, 1.0], self.archive.get(0).objectives)

    def test_should_work_with_many_objectives(self):
        # Test with 5 objectives
        archive = BestSolutionsArchive(maximum_size=3)
        
        # Add non-dominated solutions
        for i in range(5):
            solution = FloatSolution([], [], 5)
            objectives = [1.0] * 5
            objectives[i] = 0.0  # Make each solution best in one objective
            solution.objectives = objectives
            archive.add(solution)
            
        self.assertEqual(3, archive.size())  # Should be limited by max size

    def test_should_reject_duplicate_solutions(self):
        solution1 = FloatSolution([], [], 2)
        solution1.objectives = [1.0, 2.0]
        
        solution2 = FloatSolution([], [], 2)
        solution2.objectives = [1.0, 2.0]  # Same objectives
        
        result1 = self.archive.add(solution1)
        result2 = self.archive.add(solution2)
        
        self.assertTrue(result1)
        self.assertFalse(result2)  # Duplicate should be rejected
        self.assertEqual(1, self.archive.size())

    def test_should_use_custom_distance_measure(self):
        # Test using a different distance measure
        archive = BestSolutionsArchive(maximum_size=3, distance_measure=EuclideanDistance())
        
        # Add non-dominated solutions (each best in one objective)
        for i in range(5):
            solution = FloatSolution([], [], 3)
            objectives = [1.0] * 3
            objectives[i % 3] = 0.0  # Make each solution best in one objective
            solution.objectives = objectives
            archive.add(solution)
            
        self.assertEqual(3, archive.size())


class DistanceBasedSubsetSelectionTestCase(unittest.TestCase):

    def test_should_return_empty_list_for_empty_input(self):
        result = distance_based_subset_selection([], 5)
        self.assertEqual([], result)

    def test_should_return_all_solutions_when_subset_size_equals_list_size(self):
        solutions = []
        for i in range(3):
            solution = FloatSolution([], [], 2)
            solution.objectives = [float(i), float(i)]
            solutions.append(solution)
            
        result = distance_based_subset_selection(solutions, 3)
        self.assertEqual(3, len(result))

    def test_should_return_all_solutions_when_subset_size_exceeds_list_size(self):
        solutions = []
        for i in range(3):
            solution = FloatSolution([], [], 2)
            solution.objectives = [float(i), float(i)]
            solutions.append(solution)
            
        result = distance_based_subset_selection(solutions, 5)
        self.assertEqual(3, len(result))

    def test_should_raise_error_for_invalid_subset_size(self):
        solutions = [Mock()]
        
        with self.assertRaises(ValueError):
            distance_based_subset_selection(solutions, 0)
            
        with self.assertRaises(ValueError):
            distance_based_subset_selection(solutions, -1)

    def test_should_use_crowding_distance_for_2_objectives(self):
        # Create solutions on a non-dominated front
        solutions = []
        for i in range(5):
            solution = FloatSolution([], [], 2)
            solution.objectives = [float(i), 4.0 - float(i)]
            solutions.append(solution)
            
        result = distance_based_subset_selection(solutions, 3)
        self.assertEqual(3, len(result))

    def test_should_use_distance_based_selection_for_many_objectives(self):
        # Test with 4 objectives
        solutions = []
        for i in range(6):
            solution = FloatSolution([], [], 4)
            solution.objectives = [float(i), float(i+1), float(i+2), float(i+3)]
            solutions.append(solution)
            
        result = distance_based_subset_selection(solutions, 3)
        self.assertEqual(3, len(result))

    def test_should_handle_single_solution_selection(self):
        solutions = []
        for i in range(5):
            solution = FloatSolution([], [], 3)
            solution.objectives = [float(i), float(i), float(i)]
            solutions.append(solution)
            
        result = distance_based_subset_selection(solutions, 1)
        self.assertEqual(1, len(result))

    def test_should_handle_constant_objectives(self):
        # Test with some constant objectives (zero range)
        solutions = []
        for i in range(4):
            solution = FloatSolution([], [], 3)
            solution.objectives = [1.0, float(i), 2.0]  # First and third objectives constant
            solutions.append(solution)
            
        result = distance_based_subset_selection(solutions, 2)
        self.assertEqual(2, len(result))

    def test_should_be_deterministic_with_fixed_seed(self):
        # Test that results are deterministic when random seed is fixed
        solutions = []
        for i in range(5):
            solution = FloatSolution([], [], 3)
            solution.objectives = [float(i), float(i+1), float(i+2)]
            solutions.append(solution)
        
        # Run twice with same seed
        random.seed(42)
        result1 = distance_based_subset_selection(solutions, 3)
        
        random.seed(42)
        result2 = distance_based_subset_selection(solutions, 3)
        
        # Results should be identical
        self.assertEqual(len(result1), len(result2))
        for sol1, sol2 in zip(result1, result2):
            self.assertEqual(sol1.objectives, sol2.objectives)


if __name__ == '__main__':
    unittest.main()