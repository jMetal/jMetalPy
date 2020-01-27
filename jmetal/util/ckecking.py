class NoneParameterException(Exception):
    def __init__(self, message: str = ""):
        self.error_message = message


class InvalidConditionException(Exception):
    def __init__(self, message: str):
        self.error_message = message


class EmptyCollectionException(RuntimeError):
    def __init__(self):
        super(EmptyCollectionException, self).__init__("The collection is empty")


# class InvalidConditionException(RuntimeError):
#    def __init__(self, message):
#        super(InvalidConditionException, self).__init__(message)


class InvalidProbabilityValueException(RuntimeError):
    def __init__(self, value: float):
        super(InvalidProbabilityValueException, self).__init__(
            "The parameter " + str(value) + " is not a valid probability value")


class ValueOutOfRangeException(RuntimeError):
    def __init__(self, value: float, lowest_value: float, highest_value: float):
        super(ValueOutOfRangeException, self).__init__(
            "The parameter " + str(value) + " is not in the range (" + str(lowest_value) + ", " + str(
                highest_value) + ")")


class Check:
    @staticmethod
    def is_not_none(obj):
        if obj is None:
            raise NoneParameterException()

    @staticmethod
    def probability_is_valid(value: float):
        if value < 0.0 or value > 1.0:
            raise InvalidProbabilityValueException(value)

    @staticmethod
    def value_is_in_range(value: float, lowest_value: float, highest_value: float):
        if value < lowest_value or value > highest_value:
            raise ValueOutOfRangeException(value, lowest_value, highest_value)

    @staticmethod
    def collection_is_not_empty(collection):
        if len(collection) == 0:
            raise EmptyCollectionException

    @staticmethod
    def that(expression: bool, message: str):
        if not expression:
            raise InvalidConditionException(message)


"""
class Check:
    @staticmethod
    def is_not_null(o: object, message: str = ""):
        if o is None:
            raise NoneParameterException(message)

    @staticmethod
    def that(expression: bool, message: str = ""):
        if not expression:
            raise InvalidConditionException(message)
"""
