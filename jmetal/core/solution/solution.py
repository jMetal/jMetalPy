from typing import Any, TypeVar, Generic

T = TypeVar('T')

""" Class representing solutions """
__author__ = "Antonio J. Nebro"


class Solution(Generic[T]):

    def __init__(self, number_of_objectives: int, number_of_variables: int):
        self.objective = []
        self.attribute = {}
        self.variable = []
        self.number_of_objectives = number_of_objectives
        self.number_of_variables = number_of_variables

    def set_objective(self, index: int, value: float) -> None:
        self.objective[index] = value

    def get_objective(self, index: int) -> float:
        return self.objective[index]

    def set_variable(self, index: int, value: T) -> None:
        self.variable[index] = value

    def get_objective(self, index: int) -> T:
        return self.variable[index]

    def get_number_of_objectives(self) -> int:
        return len(self.number_of_objectives)

    def set_attribute(self, key: Any, value: Any) -> None:
        self.attributes[key] = value

    def get_attribute(self, key: Any) -> Any:
        return self.attributes[key]
