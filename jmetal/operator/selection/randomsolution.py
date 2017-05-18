import random
from typing import List, TypeVar
from jmetal.core.operator.selectionoperator import SelectionOperator


""" Class implementing a random solution selection operator """

S = TypeVar('S')


class RandomSolution(SelectionOperator[List[S], S]):
    def __init__(self):
        super(RandomSolution, self).__init__()

    def execute(self, solution_list: List[S]) -> S:
        if solution_list is None:
            raise Exception("The solution list is null")
        elif len(solution_list) == 0:
            raise Exception("The solution is empty")

        return random.choice(solution_list)
