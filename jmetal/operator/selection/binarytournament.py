import random
from typing import List, TypeVar

from jmetal.core.operator.selectionoperator import SelectionOperator
from jmetal.core.solution.solution import Solution
from jmetal.core.util.comparator import dominance_comparator

""" Class implementing a binary tournament selection operator """
__author__ = "Antonio J. Nebro"

S = TypeVar('S')


class BinaryTournament(SelectionOperator[List[S], S]):
    def __init__(self):
        super(BinaryTournament, self).__init__()

    def execute(self, solution_list: List[S]) -> S:
        result = None
        if solution_list is None:
            raise Exception("The solution list is null")
        elif len(solution_list) == 0:
            raise Exception("The solution is empty")

        if len(solution_list) == 1:
            result = solution_list[0]
        else:
            solution1 = solution_list[random.randint(0, len(solution_list)-1)]
            solution2 = solution_list[random.randint(0, len(solution_list)-1)]
            while solution2 == solution1:
                solution2 = solution_list[random.randint(0, len(solution_list)-1)]

            flag = dominance_comparator(solution1, solution2)
            if flag == -1:
                result = solution1
            elif flag == 1:
                result = solution2
            else:
                if random.random() < 0.5:
                    result = solution1
                else:
                    result = solution2

        return result
