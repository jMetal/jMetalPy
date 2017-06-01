import unittest

from jmetal.core.solution import FloatSolution
from jmetal.util.nondominatedsolutionlistarchive import NonDominatedSolutionListArchive


class TestMethods(unittest.TestCase):

    def setUp(self):
        pass

    def test_should_constructor_create_a_non_null_object(self):
        archive = NonDominatedSolutionListArchive[FloatSolution]()
        self.assertIsNotNone(archive)

    def test_should_default_constructor_create_a_valid_solution(self):
        archive = NonDominatedSolutionListArchive[FloatSolution]()
        archive.add(FloatSolution(1,1))
        self.assertEqual(1, archive.size())

    def test_should_constructor_create_a_valid_solution1(self):
        solution1 = FloatSolution(1,2)
        solution1.objectives = [1.0,1.0]

        solution2 = FloatSolution(1,2)
        solution2.objectives = [2.0,2.0]

        archive = NonDominatedSolutionListArchive[FloatSolution]()
        archive.add(solution1)
        archive.add(solution2)

        self.assertEqual(1, archive.size())

    def test_should_constructor_create_a_valid_solution2(self):
        solution1 = FloatSolution(1, 2)
        solution1.objectives = [1.0, 1.0]

        solution2 = FloatSolution(1, 2)
        solution2.objectives = [0.0, 0.0]

        archive = NonDominatedSolutionListArchive[FloatSolution]()
        archive.add(solution1)
        archive.add(solution2)

        self.assertEqual(1, archive.size())

    def test_should_constructor_create_a_valid_solution3(self):
        solution1 = FloatSolution(1, 2)
        solution1.objectives = [1.0, 1.0]

        solution2 = FloatSolution(1, 2)
        solution2.objectives = [0.0, 2.0]

        solution3 = FloatSolution(1, 2)
        solution3.objectives = [0.5, 1.5]

        solution4 = FloatSolution(1, 2)
        solution4.objectives = [0.0, 0.0]

        archive = NonDominatedSolutionListArchive[FloatSolution]()
        archive.add(solution1)
        archive.add(solution2)
        archive.add(solution3)
        archive.add(solution4)

        self.assertEqual(1, archive.size())

    def test_should_constructor_create_a_valid_solution4(self):
        solution1 = FloatSolution(1, 2)
        solution1.objectives = [1.0, 1.0]

        solution2 = FloatSolution(1, 2)
        solution2.objectives = [0.0, 2.0]

        solution3 = FloatSolution(1, 2)
        solution3.objectives = [1.0, 1.0]

        archive = NonDominatedSolutionListArchive[FloatSolution]()
        archive.add(solution1)
        archive.add(solution2)
        result = archive.add(solution3)

        self.assertEqual(2, archive.size())
        self.assertFalse(result)
        self.assertTrue(solution1 in archive.get_solution_list() or solution3 in archive.get_solution_list())


if __name__ == '__main__':
    unittest.main()
