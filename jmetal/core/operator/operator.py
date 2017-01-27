from typing import TypeVar, Generic

Source = TypeVar('S')
Result = TypeVar('R')


""" Class representing operators """
__author__ = "Antonio J. Nebro"


class Operator(Generic[Source, Result]):
    def __init__(self):
        pass

    def execute(self, source: Source) -> Result:
        pass
