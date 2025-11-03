import unittest

from jmetal.core.problem import Problem
from jmetal.core.solution import FloatSolution
from jmetal.util.archive import NonDominatedSolutionsArchive, DistanceBasedArchive, Archive
from jmetal.util.evaluator import SequentialEvaluatorWithArchive


class SimpleArchive(Archive):
    """Simple archive that stores all solutions without filtering."""
    
    def add(self, solution) -> bool:
        self.solution_list.append(solution)
        return True


class MockProblem(Problem):
    """Mock problem for testing purposes."""
    
    def __init__(self):
        super().__init__()
        self.lower_bound = [-1.0, -1.0]
        self.upper_bound = [1.0, 1.0]
    
    def number_of_variables(self) -> int:
        return 2
    
    def number_of_objectives(self) -> int:
        return 2
    
    def number_of_constraints(self) -> int:
        return 0
    
    def name(self) -> str:
        return "MockProblem"
    
    def evaluate(self, solution: FloatSolution) -> FloatSolution:
        """Simple evaluation: objectives are just the variables squared."""
        solution.objectives[0] = solution.variables[0] ** 2
        solution.objectives[1] = solution.variables[1] ** 2
        return solution
    
    def create_solution(self) -> FloatSolution:
        solution = FloatSolution(
            lower_bound=self.lower_bound,
            upper_bound=self.upper_bound,
            number_of_objectives=self.number_of_objectives()
        )
        solution.variables = [0.0, 0.0]
        solution.objectives = [0.0, 0.0]
        return solution


class SequentialEvaluatorWithArchiveTestCase(unittest.TestCase):
    """Test cases for SequentialEvaluatorWithArchive class."""
    
    def setUp(self):
        self.problem = MockProblem()
        self.archive = SimpleArchive()  # Use simple archive that doesn't filter
        self.evaluator = SequentialEvaluatorWithArchive(self.archive)
    
    def test_should_create_evaluator_with_archive(self):
        """Test that evaluator is created with correct archive."""
        self.assertIsInstance(self.evaluator, SequentialEvaluatorWithArchive)
        self.assertIs(self.evaluator.get_archive(), self.archive)
        self.assertEqual(0, self.archive.size())
    
    def test_should_evaluate_single_solution_and_store_in_archive(self):
        """Test evaluation of single solution and storage in archive."""
        solution = self.problem.create_solution()
        solution.variables = [0.5, 0.3]
        
        # Evaluate solution
        evaluated_solutions = self.evaluator.evaluate([solution], self.problem)
        
        # Check evaluation result
        self.assertEqual(1, len(evaluated_solutions))
        self.assertAlmostEqual(0.25, evaluated_solutions[0].objectives[0], places=6)  # 0.5^2
        self.assertAlmostEqual(0.09, evaluated_solutions[0].objectives[1], places=6)  # 0.3^2
        
        # Check archive contains copy of solution
        self.assertEqual(1, self.archive.size())
        archived_solution = self.archive.get(0)
        self.assertAlmostEqual(0.25, archived_solution.objectives[0], places=6)
        self.assertAlmostEqual(0.09, archived_solution.objectives[1], places=6)
        
        # Verify it's a copy, not the same object
        self.assertIsNot(solution, archived_solution)
    
    def test_should_evaluate_multiple_solutions_and_store_all(self):
        """Test evaluation of multiple solutions and storage in archive."""
        solutions = []
        for i in range(3):
            solution = self.problem.create_solution()
            solution.variables = [i * 0.1, (i + 1) * 0.1]
            solutions.append(solution)
        
        # Evaluate solutions
        evaluated_solutions = self.evaluator.evaluate(solutions, self.problem)
        
        # Check all solutions were evaluated
        self.assertEqual(3, len(evaluated_solutions))
        
        # Check all solutions were stored in archive
        self.assertEqual(3, self.archive.size())
        
        # Verify evaluations
        for i in range(3):
            expected_obj1 = (i * 0.1) ** 2
            expected_obj2 = ((i + 1) * 0.1) ** 2
            
            self.assertAlmostEqual(expected_obj1, evaluated_solutions[i].objectives[0], places=6)
            self.assertAlmostEqual(expected_obj2, evaluated_solutions[i].objectives[1], places=6)
            
            # Check archived solution
            archived_solution = self.archive.get(i)
            self.assertAlmostEqual(expected_obj1, archived_solution.objectives[0], places=6)
            self.assertAlmostEqual(expected_obj2, archived_solution.objectives[1], places=6)
    
    def test_should_work_with_distance_based_archive(self):
        """Test that evaluator works with different archive types."""
        distance_archive = DistanceBasedArchive(maximum_size=2)
        evaluator = SequentialEvaluatorWithArchive(distance_archive)
        
        # Create non-dominated solutions
        solutions = []
        for i in range(4):
            solution = self.problem.create_solution()
            solution.variables = [i * 0.2, 1.0 - i * 0.2]  # Trade-off solutions
            solutions.append(solution)
        
        # Evaluate solutions
        evaluated_solutions = evaluator.evaluate(solutions, self.problem)
        
        # Check that solutions were evaluated
        self.assertEqual(4, len(evaluated_solutions))
        
        # Archive should contain at most 2 solutions (maximum_size)
        self.assertLessEqual(distance_archive.size(), 2)
        self.assertGreater(distance_archive.size(), 0)
    
    def test_should_not_modify_original_solutions_when_archiving(self):
        """Test that archiving doesn't modify original solutions."""
        solution = self.problem.create_solution()
        solution.variables = [0.7, 0.4]
        original_variables = solution.variables.copy()
        
        # Evaluate solution
        evaluated_solutions = self.evaluator.evaluate([solution], self.problem)
        
        # Original solution variables should be unchanged
        self.assertEqual(original_variables, solution.variables)
        
        # But objectives should be set
        self.assertAlmostEqual(0.49, solution.objectives[0], places=6)  # 0.7^2
        self.assertAlmostEqual(0.16, solution.objectives[1], places=6)  # 0.4^2
        
        # Archived solution should be independent
        archived_solution = self.archive.get(0)
        self.assertIsNot(solution, archived_solution)
        self.assertEqual(solution.variables, archived_solution.variables)
        self.assertEqual(solution.objectives, archived_solution.objectives)
    
    def test_should_handle_empty_solution_list(self):
        """Test that evaluator handles empty solution lists gracefully."""
        evaluated_solutions = self.evaluator.evaluate([], self.problem)
        
        self.assertEqual(0, len(evaluated_solutions))
        self.assertEqual(0, self.archive.size())
    
    def test_should_accumulate_solutions_across_multiple_evaluations(self):
        """Test that archive accumulates solutions across multiple evaluation calls."""
        # First evaluation
        solution1 = self.problem.create_solution()
        solution1.variables = [0.1, 0.2]
        self.evaluator.evaluate([solution1], self.problem)
        
        self.assertEqual(1, self.archive.size())
        
        # Second evaluation
        solution2 = self.problem.create_solution()
        solution2.variables = [0.3, 0.4]
        self.evaluator.evaluate([solution2], self.problem)
        
        # Archive should contain both solutions
        self.assertEqual(2, self.archive.size())
        
        # Verify both solutions are present and correct
        archived_solutions = [self.archive.get(i) for i in range(2)]
        variables_in_archive = [sol.variables for sol in archived_solutions]
        
        self.assertIn([0.1, 0.2], variables_in_archive)
        self.assertIn([0.3, 0.4], variables_in_archive)
    
    def test_should_work_with_non_dominated_archive(self):
        """Test that evaluator works correctly with NonDominatedSolutionsArchive filtering."""
        non_dom_archive = NonDominatedSolutionsArchive()
        evaluator = SequentialEvaluatorWithArchive(non_dom_archive)
        
        # Create truly non-dominated solutions and one dominated solution
        solutions = []
        
        # Solution 1: (0.1, 0.4) -> objectives: (0.01, 0.16) - good at obj1, bad at obj2
        solution1 = self.problem.create_solution()
        solution1.variables = [0.1, 0.4]
        solutions.append(solution1)
        
        # Solution 2: (0.4, 0.1) -> objectives: (0.16, 0.01) - bad at obj1, good at obj2
        solution2 = self.problem.create_solution()
        solution2.variables = [0.4, 0.1]
        solutions.append(solution2)
        
        # Solution 3: (0.2, 0.2) -> objectives: (0.04, 0.04) - middle ground, non-dominated
        solution3 = self.problem.create_solution()
        solution3.variables = [0.2, 0.2]
        solutions.append(solution3)
        
        # Solution 4: (0.5, 0.5) -> objectives: (0.25, 0.25) - dominated by solution 3
        solution4 = self.problem.create_solution() 
        solution4.variables = [0.5, 0.5]
        solutions.append(solution4)
        
        # Evaluate solutions
        evaluated_solutions = evaluator.evaluate(solutions, self.problem)
        
        # All solutions should be evaluated
        self.assertEqual(4, len(evaluated_solutions))
        
        # Only the first 3 non-dominated solutions should be in archive
        self.assertEqual(3, non_dom_archive.size())
        
        # Verify the dominated solution (0.5, 0.5) is not in archive
        archived_variables = [sol.variables for sol in non_dom_archive.solution_list]
        self.assertNotIn([0.5, 0.5], archived_variables)
        self.assertIn([0.1, 0.4], archived_variables)
        self.assertIn([0.4, 0.1], archived_variables)
        self.assertIn([0.2, 0.2], archived_variables)


if __name__ == '__main__':
    unittest.main()