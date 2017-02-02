from typing import TypeVar, List

from jmetal.core.operator.operator import Operator

Source = TypeVar('S')
Result = TypeVar('S')

""" Class representing selection operators """
__author__ = "Antonio J. Nebro"


class SelectionOperator(Operator[Source, Result]):
    def __init__(self):
        super(SelectionOperator, self).__init__()

    def execute(self, source: Source) -> Result:
        pass
