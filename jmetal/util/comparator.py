from jmetal.core.solution import Solution


def dominance_comparator(solution1: Solution, solution2: Solution) -> int:
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


def equal_solutions_comparator(solution1: Solution, solution2: Solution) -> int:
    if solution1 is None:
        return 1
    elif solution2 is None:
        return -1

    dominate1 = 0
    dominate2 = 0

    for i in range(len(solution1.objectives)):
        value1 = solution1.objectives[i]
        value2 = solution2.objectives[i]

        if value1<value2:
            flag = -1
        elif value1 > value2:
            flag = 1
        else:
            flag = 0

        if flag == -1:
            dominate1 = 1;
        if flag == 1:
            dominate2 = 1;

    if dominate1 == 0 and dominate2 == 0:
        return 0
    elif dominate1 == 1:
        return -1
    elif dominate2 == 1:
        return 1