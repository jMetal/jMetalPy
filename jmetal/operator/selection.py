import random
from typing import List, TypeVar

from jmetal.core.operator import Selection
from jmetal.util.comparator import dominance_comparator

""" Class implementing a best solution selection operator """

S = TypeVar('S')


class BinaryTournament(Selection[List[S], S]):
    def __init__(self):
        super(BinaryTournament, self).__init__()

    def get_name(self):
        return "Bynary tournament selection"

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


class BestSolution(Selection[List[S], S]):
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


class NaryRandomSolution(Selection[List[S], S]):
    def __init__(self, number_of_solutions_to_be_returned:int = 1):
        super(NaryRandomSolution, self).__init__()
        if number_of_solutions_to_be_returned<0:
            raise Exception("The number of solutions to be returned must be positive integer")
        self.number_of_solutions_to_be_returned = number_of_solutions_to_be_returned

    def execute(self, solution_list: List[S]) -> S:
        if solution_list is None:
            raise Exception("The solution list is null")
        if len(solution_list) == 0:
            raise Exception("The solution is empty")
        if len(solution_list)<self.number_of_solutions_to_be_returned:
            raise Exception("The solution list contains less elements then requred")

        #random sampling without replacement
        return random.sample(solution_list, self.number_of_solutions_to_be_returned)


class RandomSolution(Selection[List[S], S]):
    def __init__(self):
        super(RandomSolution, self).__init__()

    def execute(self, solution_list: List[S]) -> S:
        if solution_list is None:
            raise Exception("The solution list is null")
        elif len(solution_list) == 0:
            raise Exception("The solution is empty")

        return random.choice(solution_list)
