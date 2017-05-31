import unittest
from unittest import mock


class TestMethods(unittest.TestCase):

    def setUp(self):
        pass

    def test_should_constructor_create_a_non_null_object(self):
        solution = Null()
        self.assertIsNotNone(solution)

    def test_should_constructor_create_a_valid_operator(self):
        operator = Null()
        self.assertEqual(0, operator.probability)

    def test_should_the_solution_remain_unchanged(self):
        operator = Null()
        solution1 = BinarySolution(number_of_variables=1, number_of_objectives=1)
        solution1.variables[0] = [True, False, False, True, True, False]
        solution2 = BinarySolution(number_of_variables=1, number_of_objectives=1)
        solution2.variables[0] = [False, True, False, False, True, False]

        offspring = operator.execute([solution1, solution2])
        self.assertEqual([True, False, False, True, True, False], offspring[0].variables[0])
        self.assertEqual([False, True, False, False, True, False], offspring[1].variables[0])

if __name__ == '__main__':
    unittest.main()
