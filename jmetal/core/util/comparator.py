from math import floor

from jmetal.core.solution.solution import Solution


def dominance_comparator(solution1: Solution, solution2: Solution, epsilon: float = 0.0) -> int:
    if solution1 is None:
        raise Exception("The solution1 is None")
    elif solution2 is None:
        raise Exception("The solution2 is None")
    elif len(solution1.objectives) != len(solution2.objectives):
        raise Exception("The solutions have different number of objectives")

    solution1_dominates = False
    solution2_dominates = False

    for i in range(solution1.number_of_objectives):
        value1 = solution1.objectives[i]
        value2 = solution2.objectives[i]
        if (value1 / (1.0 + epsilon)) < value2:
            flag = -1
        elif (value2 / (1.0 + epsilon)) < value1:
            flag = 1
        else:
            flag = 0

        if flag == -1:
            solution1_dominates = True

        if flag == 1:
            solution2_dominates = True

    if solution1_dominates == solution2_dominates:
        result = 0
    elif solution1_dominates:
        result = -1
    else:
        result = 1

    return result
