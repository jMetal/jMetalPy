from typing import TypeVar, Generic

S = TypeVar('S')

""" Class representing solutions """
__author__ = "Antonio J. Nebro"


class Problem(Generic[S]):
    def __init__(self):
        self.number_of_variables = 0
        self.number_of_objectives = 0
        self.number_of_constraints = 0

    def evaluate(self, solution: S):
        pass

    def create_solution(self)->S:
        pass

    def get_name(self) -> str:
        pass


