import unittest

from jmetal.component.archive import NonDominatedSolutionListArchive, BoundedArchive, CrowdingDistanceArchive, Archive
from jmetal.core.solution import Solution


class ArchiveTestCases(unittest.TestCase):
    def setUp(self):
        self.archive = Archive[Solution]()

    def test_should_constructor_create_a_non_null_object(self):
        self.assertIsNotNone(self.archive)

    def test_should_constructor_create_an_empty_list(self):
        self.assertEqual(0, len(self.archive.get_solution_list()))


class BoundedArchiveTestCases(unittest.TestCase):
    def setUp(self):
        self.archive = BoundedArchive[Solution](5)

    def test_should_constructor_create_a_non_null_object(self):
        self.assertIsNotNone(self.archive)

    def test_should_constructor_set_the_max_size(self):
        self.assertEqual(5, self.archive.get_max_size())


class NonDominatedSolutionListArchiveTestCases(unittest.TestCase):

    def setUp(self):
        self.archive = NonDominatedSolutionListArchive[Solution]()

    def test_should_constructor_create_a_non_null_object(self):
        self.assertIsNotNone(self.archive)

    def test_should_adding_one_solution_work_properly(self):
        solution = Solution(1,1)
        self.archive.add(solution)
        self.assertEqual(1, self.archive.size())
        self.assertEqual(solution, self.archive.get_solution_list()[0])

    def test_should_adding_two_solutions_work_properly_if_one_is_dominated(self):
        dominated_solution = Solution(1,2)
        dominated_solution.objectives = [2.0,2.0]

        dominant_solution = Solution(1,2)
        dominant_solution.objectives = [1.0,1.0]

        self.archive.add(dominated_solution)
        self.archive.add(dominant_solution)

        self.assertEqual(1, self.archive.size())
        self.assertEqual(dominant_solution, self.archive.get_solution_list()[0])

    def test_should_adding_two_solutions_work_properly_if_both_are_non_dominated(self):
        solution1 = Solution(1, 2)
        solution1.objectives = [1.0, 0.0]

        solution2 = Solution(1, 2)
        solution2.objectives = [0.0, 1.0]

        self.archive.add(solution1)
        self.archive.add(solution2)

        self.assertEqual(2, self.archive.size())
        self.assertTrue(solution1 in self.archive.get_solution_list() and
                        solution2 in self.archive.get_solution_list())

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
        self.assertEqual(solution4, self.archive.get_solution_list()[0])

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
        self.assertTrue(solution1 in self.archive.get_solution_list()
                        or solution3 in self.archive.get_solution_list())


class CrowdingDistanceArchiveTestCases(unittest.TestCase):
    def setUp(self):
        pass

    def test_should_constructor_create_a_non_null_object(self):
        archive = CrowdingDistanceArchive[Solution](5)
        self.assertIsNotNone(archive)

    def test_should_constructor_set_the_max_size(self):
        archive = CrowdingDistanceArchive[Solution](5)
        self.assertEqual(5, archive.get_max_size())

    def test_should_constructor_create_an_empty_archive(self):
        archive = CrowdingDistanceArchive[Solution](5)
        self.assertEqual(0, archive.size())

    def test_should_add_a_solution_when_the_archive_is_empty_work_properly(self):
        archive = CrowdingDistanceArchive[Solution](5)

        solution = Solution(2, 3)
        archive.add(solution)

        self.assertEqual(1, archive.size())
        self.assertEqual(solution, archive.get(0))

    def test_should_add_work_properly_case1(self) :
        """
        Case 1: add a dominated solution when the archive size is 1 must not include the solution
        """
        archive = CrowdingDistanceArchive[Solution](5)

        solution1 = Solution(2, 2)
        solution1.objectives = [1, 2]
        solution2 = Solution(2, 2)
        solution2.objectives = [3, 4]

        archive.add(solution1)
        archive.add(solution2)

        self.assertEqual(1, archive.size())
        self.assertEqual(solution1, archive.get(0))

    def test_should_add_work_properly_case2(self) :
        """
        Case 2: add a non-dominated solution when the archive size is 1 must include the solution
        """
        archive = CrowdingDistanceArchive[Solution](5)

        solution1 = Solution(2, 2)
        solution1.objectives = [1, 2]
        solution2 = Solution(2, 2)
        solution2.objectives = [0, 4]

        archive.add(solution1)
        archive.add(solution2)

        self.assertEqual(2, archive.size())
        self.assertTrue(solution1 in archive.get_solution_list())
        self.assertTrue(solution2 in archive.get_solution_list())

    def test_should_add_work_properly_case3(self) :
        """
        Case 3: add a non-dominated solution when the archive size is 3 must include the solution
        """
        archive = CrowdingDistanceArchive[Solution](5)

        solution1 = Solution(2, 2)
        solution1.objectives = [1.0, 2.0]
        solution2 = Solution(2, 2)
        solution2.objectives = [0.0, 4.0]
        solution3 = Solution(2, 2)
        solution3.objectives = [1.5, 1.5 ]
        solution4 = Solution(2, 2)
        solution4.objectives = [1.6, 1.2]

        archive.add(solution1)
        archive.add(solution2)
        archive.add(solution3)
        archive.add(solution4)

        self.assertEqual(4, archive.size())
        self.assertTrue(solution1 in archive.get_solution_list())
        self.assertTrue(solution2 in archive.get_solution_list())
        self.assertTrue(solution3 in archive.get_solution_list())
        self.assertTrue(solution4 in archive.get_solution_list())

    def test_should_add_work_properly_case4(self) :
        """
        Case 4: add a dominated solution when the archive size is 3 must not include the solution
        """
        archive = CrowdingDistanceArchive[Solution](5)

        solution1 = Solution(2, 2)
        solution1.objectives = [1.0, 2.0]
        solution2 = Solution(2, 2)
        solution2.objectives = [0.0, 4.0]
        solution3 = Solution(2, 2)
        solution3.objectives = [1.5, 1.5 ]
        solution4 = Solution(2, 2)
        solution4.objectives = [5.0, 6.0]

        archive.add(solution1)
        archive.add(solution2)
        archive.add(solution3)
        archive.add(solution4)

        self.assertEqual(3, archive.size())
        self.assertTrue(solution1 in archive.get_solution_list())
        self.assertTrue(solution2 in archive.get_solution_list())
        self.assertTrue(solution3 in archive.get_solution_list())

    def test_should_add_work_properly_case5(self) :
        """
        Case 5: add a dominated solution when the archive is full should not include the solution
        """
        archive = CrowdingDistanceArchive[Solution](3)

        solution1 = Solution(2, 2)
        solution1.objectives = [1.0, 2.0]
        solution2 = Solution(2, 2)
        solution2.objectives = [0.0, 4.0]
        solution3 = Solution(2, 2)
        solution3.objectives = [1.5, 1.5 ]
        solution4 = Solution(2, 2)
        solution4.objectives = [5.0, 6.0]

        archive.add(solution1)
        archive.add(solution2)
        archive.add(solution3)
        archive.add(solution4)

        self.assertEqual(3, archive.size())
        self.assertTrue(solution1 in archive.get_solution_list())
        self.assertTrue(solution2 in archive.get_solution_list())
        self.assertTrue(solution3 in archive.get_solution_list())

    def test_should_add_work_properly_case6(self) :
        """
        Case 6: add a non-dominated solution when the archive is full should not include
                the solution if it has the highest distance crowding value
        """
        archive = CrowdingDistanceArchive[Solution](4)

        solution1 = Solution(2, 2)
        solution1.objectives = [0.0, 3.0]
        solution2 = Solution(2, 2)
        solution2.objectives = [1.0, 2.0]
        solution3 = Solution(2, 2)
        solution3.objectives = [2.0, 1.5 ]
        solution4 = Solution(2, 2)
        solution4.objectives = [3.0, 0.0]

        new_solution = Solution(2, 2)
        new_solution.objectives = [1.1, 1.9]

        archive.add(solution1)
        archive.add(solution2)
        archive.add(solution3)
        archive.add(solution4)
        archive.add(new_solution)

        self.assertEqual(4, archive.size())
        self.assertTrue(new_solution not in archive.get_solution_list())

    def test_should_add_work_properly_case7(self) :
        """
        Case 7: add a non-dominated solution when the archive is full should remove all the
                dominated solutions
        """
        archive = CrowdingDistanceArchive[Solution](4)

        solution1 = Solution(2, 2)
        solution1.objectives = [0.0, 3.0]
        solution2 = Solution(2, 2)
        solution2.objectives = [1.0, 2.0]
        solution3 = Solution(2, 2)
        solution3.objectives = [2.0, 1.5 ]
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
        self.assertTrue(new_solution in archive.get_solution_list())

    def test_should_compute_density_estimator_work_properly_case1(self) :
        """
        Case 1: The archive contains one solution
        """
        archive = CrowdingDistanceArchive[Solution](4)

        solution1 = Solution(2, 2)
        solution1.objectives = [0.0, 3.0]
        archive.add(solution1)

        archive.compute_density_estimator()

        self.assertEqual(1, archive.size())
        self.assertEqual(float("inf"), solution1.attributes["crowding_distance"])

    def test_should_compute_density_estimator_work_properly_case2(self) :
        """
        Case 2: The archive contains two solutions
        """
        archive = CrowdingDistanceArchive[Solution](4)

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

    def test_should_compute_density_estimator_work_properly_case3(self) :
        """
        Case 3: The archive contains two solutions
        """
        archive = CrowdingDistanceArchive[Solution](4)

        solution1 = Solution(2, 2)
        solution1.objectives = [0.0, 3.0]
        solution2 = Solution(2, 2)
        solution2.objectives = [1.0, 2.0]
        solution3 = Solution(2, 2)
        solution3.objectives = [2.0, 1.5 ]

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
