from typing import List, TypeVar

from jmetal.core.operator.selectionoperator import SelectionOperator
from jmetal.core.util.comparator import dominance_comparator

""" Class implementing a best solution selection operator """

S = TypeVar('S')


class BestSolution(SelectionOperator[List[S], S]):
    def __init__(self):
        super(BestSolution, self).__init__()

    def execute(self, solution_list: List[S]) -> S:
        if solution_list is None:
            raise Exception("The solution list is null")
        elif len(solution_list) == 0:
            raise Exception("The solution is empty")

        result = solution_list[0]
        for solution in solution_list[1:]:
            if dominance_comparator(solution, result)<0:
                result = solution

        return result
