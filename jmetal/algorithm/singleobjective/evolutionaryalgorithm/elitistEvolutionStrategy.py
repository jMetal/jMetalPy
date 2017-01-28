from typing import TypeVar, Generic, List

Solution = TypeVar('S')
Result = TypeVar('R')

""" Class representing evolutionary algorithms """
__author__ = "Antonio J. Nebro"


class EvolutionStrategy((Generic[Solution], Result)):
    def __init__(self):
        pass