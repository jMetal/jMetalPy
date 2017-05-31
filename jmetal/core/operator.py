from typing import TypeVar, Generic

__author__ = "Antonio J. Nebro"

Source = TypeVar('S')
Result = TypeVar('R')


class Operator(Generic[Source, Result]):
    """ Class representing operators """

    def execute(self, source: Source) -> Result:
        pass


class MutationOperator(Operator[Source, Source]):
    """ Class representing mutation operators """

    def execute(self, source: Source) -> Source:
        pass
