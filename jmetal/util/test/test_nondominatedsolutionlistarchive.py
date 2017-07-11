import unittest

from jmetal.component.archive import NonDominatedSolutionListArchive
from jmetal.core.solution import Solution


class NonDominatedTestCases(unittest.TestCase):

    def setUp(self):
        pass

    def test_should_constructor_create_a_non_null_object(self):
        archive = NonDominatedSolutionListArchive[Solution]()
        self.assertIsNotNone(archive)

    def test_should_adding_one_solution_work_properly(self):
        archive = NonDominatedSolutionListArchive[Solution]()
        solution = Solution(1,1)
        archive.add(solution)
        self.assertEqual(1, archive.size())
        self.assertEqual(solution, archive.get_solution_list()[0])

    def test_should_adding_two_solutions_work_properly_if_one_is_dominated(self):
        dominated_solution = Solution(1,2)
        dominated_solution.objectives = [2.0,2.0]

        dominant_solution = Solution(1,2)
        dominant_solution.objectives = [1.0,1.0]

        archive = NonDominatedSolutionListArchive[Solution]()
        archive.add(dominated_solution)
        archive.add(dominant_solution)

        self.assertEqual(1, archive.size())
        self.assertEqual(dominant_solution, archive.get_solution_list()[0])

    def test_should_adding_two_solutions_work_properly_if_both_are_non_dominated(self):
        solution1 = Solution(1, 2)
        solution1.objectives = [1.0, 0.0]

        solution2 = Solution(1, 2)
        solution2.objectives = [0.0, 1.0]

        archive = NonDominatedSolutionListArchive[Solution]()
        archive.add(solution1)
        archive.add(solution2)

        self.assertEqual(2, archive.size())
        self.assertTrue(solution1 in archive.get_solution_list() and solution2 in archive.get_solution_list())


    def test_should_adding_four_solutions_work_properly_if_one_dominates_the_others(self):
        solution1 = Solution(1, 2)
        solution1.objectives = [1.0, 1.0]

        solution2 = Solution(1, 2)
        solution2.objectives = [0.0, 2.0]

        solution3 = Solution(1, 2)
        solution3.objectives = [0.5, 1.5]

        solution4 = Solution(1, 2)
        solution4.objectives = [0.0, 0.0]

        archive = NonDominatedSolutionListArchive[Solution]()
        archive.add(solution1)
        archive.add(solution2)
        archive.add(solution3)
        archive.add(solution4)

        self.assertEqual(1, archive.size())
        self.assertEqual(solution4, archive.get_solution_list()[0])


    def test_should_adding_three_solutions_work_properly_if_two_of_them_are_equal(self):
        solution1 = Solution(1, 2)
        solution1.objectives = [1.0, 1.0]

        solution2 = Solution(1, 2)
        solution2.objectives = [0.0, 2.0]

        solution3 = Solution(1, 2)
        solution3.objectives = [1.0, 1.0]

        archive = NonDominatedSolutionListArchive[Solution]()
        archive.add(solution1)
        archive.add(solution2)
        result = archive.add(solution3)

        self.assertEqual(2, archive.size())
        self.assertFalse(result)
        self.assertTrue(solution1 in archive.get_solution_list() or solution3 in archive.get_solution_list())


if __name__ == '__main__':
    unittest.main()
