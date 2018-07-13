import unittest

from jmetal.core.solution import BinarySolution, FloatSolution, IntegerSolution
from jmetal.operator.mutation import BitFlip, Uniform, SimpleRandom, Polynomial, IntegerPolynomial


class PolynomialMutationTestMethods(unittest.TestCase):

    def test_should_constructor_create_a_non_null_object(self):
        mutation = Polynomial(1.0)
        self.assertIsNotNone(mutation)

    def test_should_constructor_create_a_valid_operator(self):
        operator = Polynomial(0.5, 20)
        self.assertEqual(0.5, operator.probability)
        self.assertEqual(20, operator.distribution_index)

    def test_should_constructor_raise_an_exception_if_the_probability_is_greater_than_one(self):
        with self.assertRaises(Exception):
            Polynomial(2)

    def test_should_constructor_raise_an_exception_if_the_probability_is_lower_than_zero(self):
        with self.assertRaises(Exception):
            Polynomial(-12)

    def test_should_the_solution_remain_unchanged_if_the_probability_is_zero(self):
        operator = Polynomial(0.0)
        solution = FloatSolution(2, 1, 0, [-5, -5, -5], [5, 5, 5])
        solution.variables = [1.0, 2.0, 3.0]

        mutated_solution = operator.execute(solution)
        self.assertEqual([1.0, 2.0, 3.0], mutated_solution.variables)

    def test_should_the_solution_change__if_the_probability_is_one(self):
        operator = Polynomial(1.0)
        solution = FloatSolution(2, 1, 0, [-5, -5, -5], [5, 5, 5])
        solution.variables = [1.0, 2.0, 3.0]
        FloatSolution.lower_bound = [-5, -5, -5]
        FloatSolution.upper_bound = [5, 5, 5]

        mutated_solution = operator.execute(solution)

        self.assertNotEqual([1.0, 2.0, 3.0], mutated_solution.variables)


class BitFlipTestCases(unittest.TestCase):

    def test_should_constructor_create_a_non_null_object(self):
        solution = BitFlip(1.0)
        self.assertIsNotNone(solution)

    def test_should_constructor_create_a_valid_operator(self):
        operator = BitFlip(0.5)
        self.assertEqual(0.5, operator.probability)

    def test_should_constructor_raise_an_exception_if_the_probability_is_greater_than_one(self):
        with self.assertRaises(Exception):
            BitFlip(2)

    def test_should_constructor_raise_an_exception_if_the_probability_is_lower_than_zero(self):
        with self.assertRaises(Exception):
            BitFlip(-12)

    def test_should_the_solution_remain_unchanged_if_the_probability_is_zero(self):
        operator = BitFlip(0.0)
        solution = BinarySolution(number_of_variables=1, number_of_objectives=1)
        solution.variables[0] = [True, True, False, False, True, False]

        mutated_solution = operator.execute(solution)
        self.assertEqual([True, True, False, False, True, False], mutated_solution.variables[0])

    def test_should_the_solution_change_all_the_bits_if_the_probability_is_one(self):
        operator = BitFlip(1.0)
        solution = BinarySolution(number_of_variables=2, number_of_objectives=1)
        solution.variables[0] = [True, True, False, False, True, False]
        solution.variables[1] = [False, True, True, False, False, True]

        mutated_solution = operator.execute(solution)
        self.assertEqual([False, False, True, True, False, True], mutated_solution.variables[0])
        self.assertEqual([True, False, False, True, True, False], mutated_solution.variables[1])


class UniformMutationTestCases(unittest.TestCase):

    def test_should_constructor_create_a_non_null_object(self):
        operator = Uniform(0.3)
        operator2 = Uniform(0.3, 0.7)
        self.assertIsNotNone(operator)
        self.assertIsNotNone(operator2)

    def test_should_constructor_create_a_valid_operator(self):
        operator = Uniform(0.5, 20)
        self.assertEqual(0.5, operator.probability)
        self.assertEqual(20, operator.perturbation)

    def test_should_constructor_raise_an_exception_if_the_probability_is_greater_than_one(self):
        with self.assertRaises(Exception):
            Uniform(2)

    def test_should_constructor_raise_an_exception_if_the_probability_is_lower_than_zero(self):
        with self.assertRaises(Exception):
            Uniform(-12)

    def test_should_the_solution_remain_unchanged_if_the_probability_is_zero(self):
        operator = Uniform(0.0, 3.0)
        solution = FloatSolution(3, 1, 0, [-5, -5, -5], [5, 5, 5])
        solution.variables = [1.0, 2.0, 3.0]

        mutated_solution = operator.execute(solution)
        self.assertEqual([1.0, 2.0, 3.0], mutated_solution.variables)

    def test_should_the_solution_change_if_the_probability_is_one(self):
        operator = Uniform(1.0, 3.0)
        solution = FloatSolution(3, 1, 0, [-5, -5, -5], [5, 5, 5])
        solution.variables = [1.0, 2.0, 3.0]

        mutated_solution = operator.execute(solution)
        self.assertNotEqual([1.0, 2.0, 3.0], mutated_solution.variables)

    def test_should_the_solution_change_between_max_and_min_value(self):
        operator = Uniform(1.0, 5)
        solution = FloatSolution(4, 1, 0, [-1, 12, -3, -5], [1, 17, 3, -2])
        solution.variables = [-7.0, 3.0, 12.0, 13.4]

        mutated_solution = operator.execute(solution)
        for i in range(solution.number_of_variables):
            self.assertGreaterEqual(mutated_solution.variables[i], solution.lower_bound[i])
            self.assertLessEqual(mutated_solution.variables[i], solution.upper_bound[i])


class RandomMutationTestCases(unittest.TestCase):

    def test_should_constructor_create_a_non_null_object(self):
        operator = SimpleRandom(1.0)
        self.assertIsNotNone(operator)

    def test_should_constructor_create_a_valid_operator(self):
        operator = SimpleRandom(0.5)
        self.assertEqual(0.5, operator.probability)

    def test_should_constructor_raise_an_exception_if_the_probability_is_greater_than_one(self):
        with self.assertRaises(Exception):
            SimpleRandom(2)

    def test_should_constructor_raise_an_exception_if_the_probability_is_lower_than_zero(self):
        with self.assertRaises(Exception):
            SimpleRandom(-12)

    def test_should_the_solution_remain_unchanged_if_the_probability_is_zero(self):
        operator = SimpleRandom(0.0)
        solution = FloatSolution(3, 1, 0, [-5, -5, -5], [5, 5, 5])
        solution.variables = [1.0, 2.0, 3.0]

        mutated_solution = operator.execute(solution)
        self.assertEqual([1.0, 2.0, 3.0], mutated_solution.variables)

    def test_should_the_solution_change_if_the_probability_is_one(self):
        operator = SimpleRandom(1.0)
        solution = FloatSolution(3, 1, 0, [-5, -5, -5], [5, 5, 5])
        solution.variables = [1.0, 2.0, 3.0]

        mutated_solution = operator.execute(solution)
        self.assertNotEqual([1.0, 2.0, 3.0], mutated_solution.variables)

    def test_should_the_solution_change_between_max_and_min_value(self):
        operator = SimpleRandom(1.0)
        solution = FloatSolution(4, 1, 0, [-1, 12, -3, -5], [1, 17, 3, -2])
        solution.variables = [-7.0, 3.0, 12.0, 13.4]

        mutated_solution = operator.execute(solution)
        for i in range(solution.number_of_variables):
            self.assertGreaterEqual(mutated_solution.variables[i], solution.lower_bound[i])
            self.assertLessEqual(mutated_solution.variables[i], solution.upper_bound[i])


class IntegerPolynomialMutationTestCases(unittest.TestCase):

    def test_should_constructor_create_a_non_null_object(self):
        operator = IntegerPolynomial(1.0)
        self.assertIsNotNone(operator)

    def test_should_constructor_create_a_valid_operator(self):
        operator = IntegerPolynomial(0.5, 20)
        self.assertEqual(0.5, operator.probability)
        self.assertEqual(20, operator.distribution_index)

    def test_should_constructor_raise_an_exception_if_the_probability_is_greater_than_one(self):
        with self.assertRaises(Exception):
            IntegerPolynomial(2)

    def test_should_constructor_raise_an_exception_if_the_probability_is_lower_than_zero(self):
        with self.assertRaises(Exception):
            IntegerPolynomial(-12)

    def test_should_the_solution_remain_unchanged_if_the_probability_is_zero(self):
        operator = IntegerPolynomial(0.0)
        solution = IntegerSolution(2, 1, 0, [-5, -5, -5], [5, 5, 5])
        solution.variables = [1, 2, 3]

        mutated_solution = operator.execute(solution)
        self.assertEqual([1, 2, 3], mutated_solution.variables)
        self.assertEqual([True, True, True], [isinstance(x, int) for x in mutated_solution.variables])

    def test_should_the_solution_change__if_the_probability_is_one(self):
        operator = IntegerPolynomial(1.0)
        solution = IntegerSolution(2, 1, 0, [-5, -5, -5], [5, 5, 5])
        solution.variables = [1, 2, 3]

        mutated_solution = operator.execute(solution)
        self.assertNotEqual([1, 2, 3], mutated_solution.variables)
        self.assertEqual([True, True, True], [isinstance(x, int) for x in mutated_solution.variables])


if __name__ == '__main__':
    unittest.main()
