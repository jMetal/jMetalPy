from typing import TypeVar

from jmetal.core.operator.operator import Operator

Source = TypeVar('S')

""" Class representing mutation operators """
__author__ = "Antonio J. Nebro"


class Operator(Operator[Source, Source]):
    def __init__(self):
        pass

    def execute(self, source: Source) -> Source:
        pass
