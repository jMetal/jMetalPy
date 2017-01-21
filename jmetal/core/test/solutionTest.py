import unittest
import jmetal.core.solution

__author__ = "Antonio J. Nebro"


class TestMethods(unittest.TestCase):

    def setUp(self):
        pass

    def test_should_constructor_create_a_non_null_object(self):
        solution = jmetal.core.solution.Solution(int)
        self.assertIsNotNone(solution)
        """
        self.assertEqual("ACCGGGTTTT", recv.complement_DNA(dna))
        """

if __name__ == '__main__':
    unittest.main()
