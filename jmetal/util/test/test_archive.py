import unittest

from jmetal.core.solution import Solution
from jmetal.util.archive import NonDominatedSolutionsArchive, BoundedArchive, CrowdingDistanceArchive, Archive


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
        solution = Solution(1, 1)
        self.archive.add(solution)
        self.assertEqual(1, self.archive.size())
        self.assertEqual(solution, self.archive.solution_list[0])

    def test_should_adding_two_solutions_work_properly_if_one_is_dominated(self):
        dominated_solution = Solution(1, 2)
        dominated_solution.objectives = [2.0, 2.0]

        dominant_solution = Solution(1, 2)
        dominant_solution.objectives = [1.0, 1.0]

        self.archive.add(dominated_solution)
        self.archive.add(dominant_solution)

        self.assertEqual(1, self.archive.size())
        self.assertEqual(dominant_solution, self.archive.solution_list[0])

    def test_should_adding_two_solutions_work_properly_if_both_are_non_dominated(self):
        solution1 = Solution(1, 2)
        solution1.objectives = [1.0, 0.0]

        solution2 = Solution(1, 2)
        solution2.objectives = [0.0, 1.0]

        self.archive.add(solution1)
        self.archive.add(solution2)

        self.assertEqual(2, self.archive.size())
        self.assertTrue(solution1 in self.archive.solution_list and
                        solution2 in self.archive.solution_list)

    def test_should_adding_four_solutions_work_properly_if_one_dominates_the_others(self):
        solution1 = Solution(1, 2)
        solution1.objectives = [1.0, 1.0]

        solution2 = Solution(1, 2)
        solution2.objectives = [0.0, 2.0]

        solution3 = Solution(1, 2)
        solution3.objectives = [0.5, 1.5]

        solution4 = Solution(1, 2)
        solution4.objectives = [0.0, 0.0]

        self.archive.add(solution1)
        self.archive.add(solution2)
        self.archive.add(solution3)
        self.archive.add(solution4)

        self.assertEqual(1, self.archive.size())
        self.assertEqual(solution4, self.archive.solution_list[0])

    def test_should_adding_three_solutions_work_properly_if_two_of_them_are_equal(self):
        solution1 = Solution(1, 2)
        solution1.objectives = [1.0, 1.0]

        solution2 = Solution(1, 2)
        solution2.objectives = [0.0, 2.0]

        solution3 = Solution(1, 2)
        solution3.objectives = [1.0, 1.0]

        self.archive.add(solution1)
        self.archive.add(solution2)
        result = self.archive.add(solution3)

        self.assertEqual(2, self.archive.size())
        self.assertFalse(result)
        self.assertTrue(solution1 in self.archive.solution_list
                        or solution3 in self.archive.solution_list)


class CrowdingDistanceArchiveTestCases(unittest.TestCase):

    def setUp(self):
        self.archive = CrowdingDistanceArchive[Solution](5)

    def test_should_constructor_create_a_non_null_object(self):
        self.assertIsNotNone(self.archive)

    def test_should_constructor_set_the_max_size(self):
        self.assertEqual(5, self.archive.maximum_size)

    def test_should_constructor_create_an_empty_archive(self):
        self.assertEqual(0, self.archive.size())

    def test_should_add_a_solution_when_the_archive_is_empty_work_properly(self):
        solution = Solution(2, 3)
        self.archive.add(solution)

        self.assertEqual(1, self.archive.size())
        self.assertEqual(solution, self.archive.get(0))

    def test_should_add_work_properly_case1(self):
        """ Case 1: add a dominated solution when the archive size is 1 must not include the solution.
        """
        solution1 = Solution(2, 2)
        solution1.objectives = [1, 2]
        solution2 = Solution(2, 2)
        solution2.objectives = [3, 4]

        self.archive.add(solution1)
        self.archive.add(solution2)

        self.assertEqual(1, self.archive.size())
        self.assertEqual(solution1, self.archive.get(0))

    def test_should_add_work_properly_case2(self):
        """ Case 2: add a non-dominated solution when the archive size is 1 must include the solution.
        """
        solution1 = Solution(2, 2)
        solution1.objectives = [1, 2]
        solution2 = Solution(2, 2)
        solution2.objectives = [0, 4]

        self.archive.add(solution1)
        self.archive.add(solution2)

        self.assertEqual(2, self.archive.size())
        self.assertTrue(solution1 in self.archive.solution_list)
        self.assertTrue(solution2 in self.archive.solution_list)

    def test_should_add_work_properly_case3(self):
        """ Case 3: add a non-dominated solution when the archive size is 3 must include the solution.
        """
        solution1 = Solution(2, 2)
        solution1.objectives = [1.0, 2.0]
        solution2 = Solution(2, 2)
        solution2.objectives = [0.0, 4.0]
        solution3 = Solution(2, 2)
        solution3.objectives = [1.5, 1.5]
        solution4 = Solution(2, 2)
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
        """ Case 4: add a dominated solution when the archive size is 3 must not include the solution.
        """
        solution1 = Solution(2, 2)
        solution1.objectives = [1.0, 2.0]
        solution2 = Solution(2, 2)
        solution2.objectives = [0.0, 4.0]
        solution3 = Solution(2, 2)
        solution3.objectives = [1.5, 1.5]
        solution4 = Solution(2, 2)
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
        """ Case 5: add a dominated solution when the archive is full should not include the solution.
        """
        solution1 = Solution(2, 2)
        solution1.objectives = [1.0, 2.0]
        solution2 = Solution(2, 2)
        solution2.objectives = [0.0, 4.0]
        solution3 = Solution(2, 2)
        solution3.objectives = [1.5, 1.5]
        solution4 = Solution(2, 2)
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
        """ Case 6: add a non-dominated solution when the archive is full should not include
        the solution if it has the highest distance crowding value.
        """
        archive = CrowdingDistanceArchive(4)

        solution1 = Solution(1, 2)
        solution1.variables = [1.0]
        solution1.objectives = [0.0, 3.0]
        solution2 = Solution(1, 2)
        solution2.variables = [2.0]
        solution2.objectives = [1.0, 2.0]
        solution3 = Solution(1, 2)
        solution3.variables = [3.0]
        solution3.objectives = [2.0, 1.5]
        solution4 = Solution(1, 2)
        solution4.variables = [4.0]
        solution4.objectives = [3.0, 0.0]

        new_solution = Solution(1, 2)
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
        """ Case 7: add a non-dominated solution when the archive is full should remove all the dominated solutions.
        """
        archive = CrowdingDistanceArchive(4)

        solution1 = Solution(2, 2)
        solution1.objectives = [0.0, 3.0]
        solution2 = Solution(2, 2)
        solution2.objectives = [1.0, 2.0]
        solution3 = Solution(2, 2)
        solution3.objectives = [2.0, 1.5]
        solution4 = Solution(2, 2)
        solution4.objectives = [3.0, 0.0]

        new_solution = Solution(2, 2)
        new_solution.objectives = [-1.0, -1.0]

        archive.add(solution1)
        archive.add(solution2)
        archive.add(solution3)
        archive.add(solution4)
        archive.add(new_solution)

        self.assertEqual(1, archive.size())
        self.assertTrue(new_solution in archive.solution_list)

    def test_should_compute_density_estimator_work_properly_case1(self):
        """ Case 1: The archive contains one solution.
        """
        archive = CrowdingDistanceArchive(4)

        solution1 = Solution(2, 2)
        solution1.objectives = [0.0, 3.0]
        archive.add(solution1)

        archive.compute_density_estimator()

        self.assertEqual(1, archive.size())
        self.assertEqual(float("inf"), solution1.attributes["crowding_distance"])

    def test_should_compute_density_estimator_work_properly_case2(self):
        """ Case 2: The archive contains two solutions.
        """
        archive = CrowdingDistanceArchive(4)

        solution1 = Solution(2, 2)
        solution1.objectives = [0.0, 3.0]
        solution2 = Solution(2, 2)
        solution2.objectives = [1.0, 2.0]

        archive.add(solution1)
        archive.add(solution2)

        archive.compute_density_estimator()

        self.assertEqual(2, archive.size())
        self.assertEqual(float("inf"), solution1.attributes["crowding_distance"])
        self.assertEqual(float("inf"), solution2.attributes["crowding_distance"])

    def test_should_compute_density_estimator_work_properly_case3(self):
        """ Case 3: The archive contains two solutions.
        """
        archive = CrowdingDistanceArchive(4)

        solution1 = Solution(2, 2)
        solution1.objectives = [0.0, 3.0]
        solution2 = Solution(2, 2)
        solution2.objectives = [1.0, 2.0]
        solution3 = Solution(2, 2)
        solution3.objectives = [2.0, 1.5]

        archive.add(solution1)
        archive.add(solution2)
        archive.add(solution3)

        archive.compute_density_estimator()

        self.assertEqual(3, archive.size())
        self.assertEqual(float("inf"), solution1.attributes["crowding_distance"])
        self.assertEqual(float("inf"), solution3.attributes["crowding_distance"])
        self.assertTrue(solution2.attributes["crowding_distance"] < float("inf"))


if __name__ == '__main__':
    unittest.main()
