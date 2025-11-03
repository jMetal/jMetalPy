import unittest

from jmetal.core.solution import FloatSolution
from jmetal.util.archive import (
    Archive,
    BoundedArchive,
    CrowdingDistanceArchive,
    NonDominatedSolutionsArchive,
)


class ArchiveTestCases(unittest.TestCase):
    class DummyArchive(Archive):
        def add(self, solution) -> bool:
            pass

    def setUp(self):
        self.archive = self.DummyArchive()

    def test_should_constructor_create_a_non_null_object(self):
        self.assertIsNotNone(self.archive)

    def test_should_constructor_create_an_empty_list(self):
        self.assertEqual(0, len(self.archive.solution_list))


class BoundedArchiveTestCases(unittest.TestCase):
    def setUp(self):
        self.archive = BoundedArchive(5)

    def test_should_constructor_create_a_non_null_object(self):
        self.assertIsNotNone(self.archive)

    def test_should_constructor_set_the_max_size(self):
        self.assertEqual(5, self.archive.maximum_size)


class NonDominatedSolutionListArchiveTestCases(unittest.TestCase):
    def setUp(self):
        self.archive = NonDominatedSolutionsArchive()

    def test_should_constructor_create_a_non_null_object(self):
        self.assertIsNotNone(self.archive)

    def test_should_adding_one_solution_work_properly(self):
        solution = FloatSolution([0.0], [1.0], 1)
        self.archive.add(solution)
        self.assertEqual(1, self.archive.size())
        self.assertEqual(solution, self.archive.solution_list[0])

    def test_should_adding_two_solutions_work_properly_if_one_is_dominated(self):
        dominated_solution = FloatSolution([0.0], [1.0], 2)
        dominated_solution.objectives = [2.0, 2.0]

        dominant_solution = FloatSolution([0.0], [1.0], 2)
        dominant_solution.objectives = [1.0, 1.0]

        self.archive.add(dominated_solution)
        self.archive.add(dominant_solution)

        self.assertEqual(1, self.archive.size())
        self.assertEqual(dominant_solution, self.archive.solution_list[0])

    def test_should_adding_two_solutions_work_properly_if_both_are_non_dominated(self):
        solution1 = FloatSolution([0.0], [1.0], 2)
        solution1.objectives = [1.0, 0.0]

        solution2 = FloatSolution([0.0], [1.0], 2)
        solution2.objectives = [0.0, 1.0]

        self.archive.add(solution1)
        self.archive.add(solution2)

        self.assertEqual(2, self.archive.size())
        self.assertTrue(solution1 in self.archive.solution_list and solution2 in self.archive.solution_list)

    def test_should_adding_four_solutions_work_properly_if_one_dominates_the_others(self):
        solution1 = FloatSolution([0.0], [1.0], 2)
        solution1.objectives = [1.0, 1.0]

        solution2 = FloatSolution([0.0], [1.0], 2)
        solution2.objectives = [0.0, 2.0]

        solution3 = FloatSolution([0.0], [1.0], 2)
        solution3.objectives = [0.5, 1.5]

        solution4 = FloatSolution([0.0], [1.0], 2)
        solution4.objectives = [0.0, 0.0]

        self.archive.add(solution1)
        self.archive.add(solution2)
        self.archive.add(solution3)
        self.archive.add(solution4)

        self.assertEqual(1, self.archive.size())
        self.assertEqual(solution4, self.archive.solution_list[0])

    def test_should_adding_three_solutions_work_properly_if_two_of_them_are_equal(self):
        solution1 = FloatSolution([0.0], [1.0], 2)
        solution1.objectives = [1.0, 1.0]

        solution2 = FloatSolution([0.0], [1.0], 2)
        solution2.objectives = [0.0, 2.0]

        solution3 = FloatSolution([0.0], [1.0], 2)
        solution3.objectives = [1.0, 1.0]

        self.archive.add(solution1)
        self.archive.add(solution2)
        result = self.archive.add(solution3)

        self.assertEqual(2, self.archive.size())
        self.assertFalse(result)
        self.assertTrue(solution1 in self.archive.solution_list or solution3 in self.archive.solution_list)

    def test_should_add_high_dimensional_solutions(self):
        """Test behavior with solutions having more than 2 objectives. Only one solution should remain due to dominance logic."""
        archive = NonDominatedSolutionsArchive()
        s1 = FloatSolution([0.0], [1.0], 5)
        s1.objectives = [0.0, 1.0, 2.0, 3.0, 4.0]
        s2 = FloatSolution([0.0], [1.0], 5)
        s2.objectives = [1.0, 2.0, 3.0, 4.0, 5.0]
        s3 = FloatSolution([0.0], [1.0], 5)
        s3.objectives = [0.5, 1.5, 2.5, 3.5, 4.5]
        archive.add(s1)
        archive.add(s2)
        archive.add(s3)
        # Only one solution should remain, as the dominance logic removes dominated solutions
        self.assertEqual(1, archive.size())
        self.assertTrue(s1 in archive.solution_list or s2 in archive.solution_list or s3 in archive.solution_list)

    def test_should_add_with_numerical_tolerance(self):
        """Test adding nearly identical solutions (numerical tolerance). Only one should be kept if they are equal within tolerance."""
        archive = NonDominatedSolutionsArchive(objective_tolerance=1e-5)
        s1 = FloatSolution([0.0], [1.0], 2)
        s1.objectives = [1.000000, 2.000000]
        s2 = FloatSolution([0.0], [1.0], 2)
        s2.objectives = [1.000001, 2.000001]
        archive.add(s1)
        archive.add(s2)
        # Only one solution should be kept, as they are equal within the tolerance
        self.assertEqual(1, archive.size())
        self.assertTrue(s1 in archive.solution_list or s2 in archive.solution_list)


class CrowdingDistanceArchiveTestCases(unittest.TestCase):
    def setUp(self):
        self.archive = CrowdingDistanceArchive[FloatSolution](5)

    def test_should_constructor_create_a_non_null_object(self):
        self.assertIsNotNone(self.archive)

    def test_should_constructor_set_the_max_size(self):
        self.assertEqual(5, self.archive.maximum_size)

    def test_should_constructor_create_an_empty_archive(self):
        self.assertEqual(0, self.archive.size())

    def test_should_add_a_solution_when_the_archive_is_empty_work_properly(self):
        solution = FloatSolution([0.0, 0.0], [1.0, 1.0], 3)
        self.archive.add(solution)

        self.assertEqual(1, self.archive.size())
        self.assertEqual(solution, self.archive.get(0))

    def test_should_add_work_properly_case1(self):
        """Case 1: add a dominated solution when the archive size is 1 must not include the solution."""
        solution1 = FloatSolution([0.0, 0.0], [1.0, 1.0], 2)
        solution1.objectives = [1, 2]
        solution2 = FloatSolution([0.0, 0.0], [1.0, 1.0], 2)
        solution2.objectives = [3, 4]

        self.archive.add(solution1)
        self.archive.add(solution2)

        self.assertEqual(1, self.archive.size())
        self.assertEqual(solution1, self.archive.get(0))

    def test_should_add_work_properly_case2(self):
        """Case 2: add a non-dominated solution when the archive size is 1 must include the solution."""
        solution1 = FloatSolution([0.0, 0.0], [1.0, 1.0], 2)
        solution1.objectives = [1, 2]
        solution2 = FloatSolution([0.0, 0.0], [1.0, 1.0], 2)
        solution2.objectives = [0, 4]

        self.archive.add(solution1)
        self.archive.add(solution2)

        self.assertEqual(2, self.archive.size())
        self.assertTrue(solution1 in self.archive.solution_list)
        self.assertTrue(solution2 in self.archive.solution_list)

    def test_should_add_work_properly_case3(self):
        """Case 3: add a non-dominated solution when the archive size is 3 must include the solution."""
        solution1 = FloatSolution([0.0, 0.0], [1.0, 1.0], 2)
        solution1.objectives = [1.0, 2.0]
        solution2 = FloatSolution([0.0, 0.0], [1.0, 1.0], 2)
        solution2.objectives = [0.0, 4.0]
        solution3 = FloatSolution([0.0, 0.0], [1.0, 1.0], 2)
        solution3.objectives = [1.5, 1.5]
        solution4 = FloatSolution([0.0, 0.0], [1.0, 1.0], 2)
        solution4.objectives = [1.6, 1.2]

        self.archive.add(solution1)
        self.archive.add(solution2)
        self.archive.add(solution3)
        self.archive.add(solution4)

        self.assertEqual(4, self.archive.size())
        self.assertTrue(solution1 in self.archive.solution_list)
        self.assertTrue(solution2 in self.archive.solution_list)
        self.assertTrue(solution3 in self.archive.solution_list)
        self.assertTrue(solution4 in self.archive.solution_list)

    def test_should_add_work_properly_case4(self):
        """Case 4: add a dominated solution when the archive size is 3 must not include the solution."""
        solution1 = FloatSolution([0.0, 0.0], [1.0, 1.0], 2)
        solution1.objectives = [1.0, 2.0]
        solution2 = FloatSolution([0.0, 0.0], [1.0, 1.0], 2)
        solution2.objectives = [0.0, 4.0]
        solution3 = FloatSolution([0.0, 0.0], [1.0, 1.0], 2)
        solution3.objectives = [1.5, 1.5]
        solution4 = FloatSolution([0.0, 0.0], [1.0, 1.0], 2)
        solution4.objectives = [5.0, 6.0]

        self.archive.add(solution1)
        self.archive.add(solution2)
        self.archive.add(solution3)
        self.archive.add(solution4)

        self.assertEqual(3, self.archive.size())
        self.assertTrue(solution1 in self.archive.solution_list)
        self.assertTrue(solution2 in self.archive.solution_list)
        self.assertTrue(solution3 in self.archive.solution_list)

    def test_should_add_work_properly_case5(self):
        """Case 5: add a dominated solution when the archive is full should not include the solution."""
        solution1 = FloatSolution([0.0, 0.0], [1.0, 1.0], 2)
        solution1.objectives = [1.0, 2.0]
        solution2 = FloatSolution([0.0, 0.0], [1.0, 1.0], 2)
        solution2.objectives = [0.0, 4.0]
        solution3 = FloatSolution([0.0, 0.0], [1.0, 1.0], 2)
        solution3.objectives = [1.5, 1.5]
        solution4 = FloatSolution([0.0, 0.0], [1.0, 1.0], 2)
        solution4.objectives = [5.0, 6.0]

        self.archive.add(solution1)
        self.archive.add(solution2)
        self.archive.add(solution3)
        self.archive.add(solution4)

        self.assertEqual(3, self.archive.size())
        self.assertTrue(solution1 in self.archive.solution_list)
        self.assertTrue(solution2 in self.archive.solution_list)
        self.assertTrue(solution3 in self.archive.solution_list)

    def test_should_add_work_properly_case6(self):
        """Case 6: add a non-dominated solution when the archive is full should not include
        the solution if it has the highest distance crowding value.
        """
        archive = CrowdingDistanceArchive(4)

        solution1 = FloatSolution([0.0], [1.0], 2)
        solution1.variables = [1.0]
        solution1.objectives = [0.0, 3.0]
        solution2 = FloatSolution([0.0], [1.0], 2)
        solution2.variables = [2.0]
        solution2.objectives = [1.0, 2.0]
        solution3 = FloatSolution([0.0], [1.0], 2)
        solution3.variables = [3.0]
        solution3.objectives = [2.0, 1.5]
        solution4 = FloatSolution([0.0], [1.0], 2)
        solution4.variables = [4.0]
        solution4.objectives = [3.0, 0.0]

        new_solution = FloatSolution([0.0], [1.0], 2)
        new_solution.variables = [5.0]
        new_solution.objectives = [1.1, 1.9]

        archive.add(solution1)
        archive.add(solution2)
        archive.add(solution3)
        archive.add(solution4)
        archive.add(new_solution)

        self.assertEqual(4, archive.size())
        self.assertTrue(new_solution not in archive.solution_list)

    def test_should_add_work_properly_case7(self):
        """Case 7: add a non-dominated solution when the archive is full should remove all the dominated solutions."""
        archive = CrowdingDistanceArchive(4)

        solution1 = FloatSolution([0.0, 0.0], [1.0, 1.0], 2)
        solution1.objectives = [0.0, 3.0]
        solution2 = FloatSolution([0.0, 0.0], [1.0, 1.0], 2)
        solution2.objectives = [1.0, 2.0]
        solution3 = FloatSolution([0.0, 0.0], [1.0, 1.0], 2)
        solution3.objectives = [2.0, 1.5]
        solution4 = FloatSolution([0.0, 0.0], [1.0, 1.0], 2)
        solution4.objectives = [3.0, 0.0]

        new_solution = FloatSolution([0.0, 0.0], [1.0, 1.0], 2)
        new_solution.objectives = [-1.0, -1.0]

        archive.add(solution1)
        archive.add(solution2)
        archive.add(solution3)
        archive.add(solution4)
        archive.add(new_solution)

        self.assertEqual(1, archive.size())
        self.assertTrue(new_solution in archive.solution_list)

    def test_should_compute_density_estimator_work_properly_case1(self):
        """Case 1: The archive contains one solution."""
        archive = CrowdingDistanceArchive[FloatSolution](4)

        solution1 = FloatSolution([0.0, 0.0], [1.0, 1.0], 2)
        solution1.objectives = [0.0, 3.0]
        archive.add(solution1)

        archive.compute_density_estimator()

        self.assertEqual(1, archive.size())
        self.assertEqual(float("inf"), solution1.attributes["crowding_distance"])


if __name__ == "__main__":
    unittest.main()
