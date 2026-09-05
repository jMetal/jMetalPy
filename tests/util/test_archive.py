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

    def test_add_batch_default_implementation_calls_add_once_per_solution(self):
        added = []

        class RecordingArchive(Archive):
            def add(self, solution) -> bool:
                added.append(solution)
                return True

        archive = RecordingArchive()
        solutions = [object(), object(), object()]

        archive.add_batch(solutions)

        self.assertEqual(solutions, added)

    def test_add_batch_default_implementation_does_nothing_for_an_empty_list(self):
        self.archive.add_batch([])

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
        self.assertTrue(
            solution1 in self.archive.solution_list and solution2 in self.archive.solution_list
        )

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
        self.assertTrue(
            solution1 in self.archive.solution_list or solution3 in self.archive.solution_list
        )

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
        self.assertTrue(
            s1 in archive.solution_list
            or s2 in archive.solution_list
            or s3 in archive.solution_list
        )

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

    def test_add_batch_keeps_every_mutually_non_dominated_solution(self):
        solutions = []
        for f1, f2 in [(0.0, 1.0), (0.5, 0.5), (1.0, 0.0)]:
            solution = FloatSolution([0.0], [1.0], 2)
            solution.objectives = [f1, f2]
            solutions.append(solution)

        self.archive.add_batch(solutions)

        self.assertEqual(3, self.archive.size())

    def test_add_batch_filters_out_dominated_solutions_within_the_same_batch(self):
        dominated = FloatSolution([0.0], [1.0], 2)
        dominated.objectives = [2.0, 2.0]
        dominant = FloatSolution([0.0], [1.0], 2)
        dominant.objectives = [1.0, 1.0]

        self.archive.add_batch([dominated, dominant])

        self.assertEqual(1, self.archive.size())
        self.assertEqual(dominant, self.archive.solution_list[0])

    def test_add_batch_removes_existing_solutions_dominated_by_the_new_batch(self):
        existing = FloatSolution([0.0], [1.0], 2)
        existing.objectives = [2.0, 2.0]
        self.archive.add(existing)

        dominant = FloatSolution([0.0], [1.0], 2)
        dominant.objectives = [1.0, 1.0]
        self.archive.add_batch([dominant])

        self.assertEqual(1, self.archive.size())
        self.assertEqual(dominant, self.archive.solution_list[0])

    def test_add_batch_does_nothing_for_an_empty_batch(self):
        existing = FloatSolution([0.0], [1.0], 2)
        existing.objectives = [1.0, 1.0]
        self.archive.add(existing)

        self.archive.add_batch([])

        self.assertEqual(1, self.archive.size())

    def test_add_batch_matches_adding_the_same_solutions_one_at_a_time(self):
        def make_solutions():
            data = [(0.1, 0.9), (0.9, 0.1), (0.5, 0.5), (0.5, 0.5), (0.9, 0.9)]
            result = []
            for f1, f2 in data:
                solution = FloatSolution([0.0], [1.0], 2)
                solution.objectives = [f1, f2]
                result.append(solution)
            return result

        one_at_a_time_archive = NonDominatedSolutionsArchive()
        for solution in make_solutions():
            one_at_a_time_archive.add(solution)

        self.archive.add_batch(make_solutions())

        expected = sorted(tuple(s.objectives) for s in one_at_a_time_archive.solution_list)
        actual = sorted(tuple(s.objectives) for s in self.archive.solution_list)
        self.assertEqual(expected, actual)

    def test_add_batch_falls_back_to_the_default_loop_for_a_custom_comparator(self):
        class AlwaysEqualComparator:
            """A comparator under which nothing ever dominates anything else."""

            def compare(self, solution1, solution2) -> int:
                return 0

        archive = NonDominatedSolutionsArchive(dominance_comparator=AlwaysEqualComparator())
        solution1 = FloatSolution([0.0], [1.0], 2)
        solution1.objectives = [1.0, 1.0]
        solution2 = FloatSolution([0.0], [1.0], 2)
        solution2.objectives = [2.0, 2.0]

        # Under plain Pareto dominance (what moocore.is_nondominated assumes) solution2
        # would be dropped as dominated; the custom comparator says otherwise, so both
        # must survive -- this only holds if the custom comparator was actually used
        # instead of silently falling through to moocore.
        archive.add_batch([solution1, solution2])

        self.assertEqual(2, archive.size())


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
