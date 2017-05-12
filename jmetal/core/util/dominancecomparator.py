from math import floor

from jmetal.core.solution.solution import Solution


def dominance_comparator(solution1, solution2, epsilon = 0.0):
    if solution1 is None:
        raise Exception("The solution1 is None")
    elif solution2 is None:
        raise Exception("The solution2 is None")
    elif len(solution1.objectives) != len(solution2.objectives):
        raise Exception("The solutions have different number of objectives")

    best_is_one = 0
    best_is_two = 0

    for i in range(solution1.number_of_objectives):
        value1 = solution1.objectives[i]
        value2 = solution2.objectives[i]
        if value1 != value2:
            if value1 < value2:
                best_is_one = 1
            if value2 < value1:
                best_is_two = 1

    if best_is_one > best_is_two:
        result = -1
    elif best_is_two > best_is_one:
        result = 1
    else:
        result = 0

    return result
