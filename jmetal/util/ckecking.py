class NoneParameterException(Exception):
    def __init__(self, message:str = ""):
        self.error_message = message


class InvalidConditionException(Exception):
    def __init__(self, message:str):
        self.error_message = message


class Check:
    @staticmethod
    def is_not_null(object, message:str = ""):
        if object is None:
            raise NoneParameterException(message)

    @staticmethod
    def that(expression:bool, message: str = ""):
        if not expression:
            raise InvalidConditionException(message)