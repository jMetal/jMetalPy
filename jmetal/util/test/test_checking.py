import unittest

from jmetal.util.ckecking import Check, NoneParameterException, InvalidProbabilityValueException, \
    ValueOutOfRangeException, InvalidConditionException


class CheckingTestCases(unittest.TestCase):

    def test_should_is_not_null_raise_an_exception(self) -> None:
        with self.assertRaises(NoneParameterException):
            Check.is_not_none(None)

    def test_should_is_valid_probability_raise_an_exception_if_the_value_is_negative(self) -> None:
        with self.assertRaises(InvalidProbabilityValueException):
            Check.probability_is_valid(-1.0)

    def test_should_is_valid_probability_raise_an_exception_if_the_value_is_higher_than_one(self) -> None:
        with self.assertRaises(InvalidProbabilityValueException):
            Check.probability_is_valid(1.1)

    def test_should_is_value_in_range_raise_an_exception_if_the_value_is_lower_than_the_lower_bound(self) -> None:
        with self.assertRaises(ValueOutOfRangeException):
            Check.value_is_in_range(2, 3, 5)

    def test_should_is_value_in_range_raise_an_exception_if_the_value_is_higher_than_the_upper_bound(self) -> None:
        with self.assertRaises(ValueOutOfRangeException):
            Check.value_is_in_range(7, 3, 5)

    def test_should_that_raise_an_exception_if_the_expression_is_false(self) -> None:
        with self.assertRaises(InvalidConditionException):
            Check.that(False, "The expression is false")


if __name__ == '__main__':
    unittest.main()
