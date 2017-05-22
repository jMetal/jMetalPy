

from jmetal.core.solution.solution import Solution


def EqualSolutionsComparator(solution1: Solution, solution2: Solution) -> int:
    if solution1 is None:
        return 1
    elif solution2 is None:
        return -1

    dominate1 = 0;
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

