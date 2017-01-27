from typing import TypeVar, Generic

T = TypeVar('T')

""" Class representing solutions """
__author__ = "Antonio J. Nebro"


class Solution(Generic[T]):

    def __init__(self, number_of_variables: int, number_of_objectives: int):
        self.number_of_objectives = number_of_objectives
        self.number_of_variables = number_of_variables

        self.objectives = [0.0 for x in range(self.number_of_objectives)]
        self.variables = [[] for x in range(self.number_of_variables)]
        self.attributes = {}

    '''

    def set_objective(self, index: int, value: float) -> None:
        self.objectives[index] = value

    def get_objective(self, index: int) -> float:
        return self.objectives[index]

    def set_variable(self, index: int, value: T) -> None:
        self.variables[index] = value

    def get_objective(self, index: int) -> T:
        return self.variables[index]

    def get_number_of_objectives(self) -> int:
        return self.number_of_objectives

    def get_number_of_variables(self) -> int:
        return self.number_of_variables

    def set_attribute(self, key: Any, value: Any) -> None:
        self.attributes[key] = value

    def get_attribute(self, key: Any) -> Any:
        return self.attributes[key]

    '''