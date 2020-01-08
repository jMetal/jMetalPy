import math
from abc import ABC, abstractmethod
from typing import TypeVar, Generic

from jmetal.core.solution import Solution
from jmetal.util.constraint_handling import overall_constraint_violation_degree

S = TypeVar('S')


class Comparator(Generic[S], ABC):

    @abstractmethod
    def compare(self, solution1: S, solution2: S) -> int:
        pass


class EqualSolutionsComparator(Comparator):

    def compare(self, solution1: Solution, solution2: Solution) -> int:
        if solution1 is None:
            return 1
        elif solution2 is None:
            return -1

        dominate1 = 0
        dominate2 = 0

        for i in range(len(solution1.objectives)):
            value1 = solution1.objectives[i]
            value2 = solution2.objectives[i]

            if value1 < value2:
                flag = -1
            elif value1 > value2:
                flag = 1
            else:
                flag = 0

            if flag == -1:
                dominate1 = 1
            if flag == 1:
                dominate2 = 1

        if dominate1 == 0 and dominate2 == 0:
            return 0
        elif dominate1 == 1:
            return -1
        elif dominate2 == 1:
            return 1


class SolutionAttributeComparator(Comparator):

    def __init__(self, key: str, lowest_is_best: bool = True):
        self.key = key
        self.lowest_is_best = lowest_is_best

    def compare(self, solution1: Solution, solution2: Solution) -> int:
        value1 = solution1.attributes.get(self.key)
        value2 = solution2.attributes.get(self.key)

        result = 0
        if value1 is not None and value2 is not None:
            if self.lowest_is_best:
                if value1 < value2:
                    result = -1
                elif value1 > value2:
                    result = 1
                else:
                    result = 0
            else:
                if value1 > value2:
                    result = -1
                elif value1 < value2:
                    result = 1
                else:
                    result = 0

        return result


class MultiComparator(Comparator):
    """
    This comparator takes a list of comparators and check all of them iteratively until a
    value != 0 is obtained or the list becomes empty
    """

    def __init__(self, comparator_list: [Comparator]):
        self.comparator_list: [Comparator] = comparator_list

    def compare(self, solution1: Solution, solution2: Solution) -> int:
        for comparator in self.comparator_list:
            flag = comparator.compare(solution1, solution2)
            if flag != 0:
                return flag

        return 0


class RankingAndCrowdingDistanceComparator(Comparator):

    def compare(self, solution1: Solution, solution2: Solution) -> int:
        result = \
            SolutionAttributeComparator("dominance_ranking").compare(solution1, solution2)

        if result is 0:
            result = \
                SolutionAttributeComparator("crowding_distance", lowest_is_best=False).compare(solution1, solution2)

        return result


class StrengthAndKNNDistanceComparator(Comparator):

    def compare(self, solution1: Solution, solution2: Solution) -> int:
        result = \
            SolutionAttributeComparator('dominance_ranking').compare(solution1, solution2)

        if result is 0:
            result = \
                SolutionAttributeComparator("knn_density", lowest_is_best=False).compare(solution1, solution2)

        return result


class OverallConstraintViolationComparator(Comparator):
    def compare(self, solution1: Solution, solution2: Solution) -> int:
        violation_degree_solution_1 = overall_constraint_violation_degree(solution1)
        violation_degree_solution_2 = overall_constraint_violation_degree(solution2)
        if violation_degree_solution_1 < 0 and violation_degree_solution_2 < 0:
            if violation_degree_solution_1 > violation_degree_solution_2:
                result = -1
            elif violation_degree_solution_2 > violation_degree_solution_1:
                result = 1
            else:
                result = 0
        elif violation_degree_solution_1 == 0 and violation_degree_solution_2 < 0:
            result = -1
        elif violation_degree_solution_2 == 0 and violation_degree_solution_1 < 0:
            result = 1
        else:
            result = 0

        return result


class DominanceComparator(Comparator):

    def __init__(self, constraint_comparator: Comparator = OverallConstraintViolationComparator()):
        self.constraint_comparator = constraint_comparator

    def compare(self, solution1: Solution, solution2: Solution) -> int:
        if solution1 is None:
            raise Exception("The solution1 is None")
        elif solution2 is None:
            raise Exception("The solution2 is None")

        result = self.constraint_comparator.compare(solution1, solution2)
        if result == 0:
            # result = self.__dominance_test(solution1, solution2)
            result = self.dominance_test(solution1.objectives, solution2.objectives)

        return result

    def __dominance_test(self, solution1: Solution, solution2: Solution) -> float:
        best_is_one = 0
        best_is_two = 0

        for i in range(solution1.number_of_objectives):
            value1 = solution1.objectives[i]
            value2 = solution2.objectives[i]
            if value1 != value2:
                if value1 < value2:
                    best_is_one = 1
                if value1 > value2:
                    best_is_two = 1

        if best_is_one > best_is_two:
            result = -1
        elif best_is_two > best_is_one:
            result = 1
        else:
            result = 0

        return result

    @staticmethod
    def dominance_test(vector1: [float], vector2: [float]) -> int:
        result = 0
        for i in range(len(vector1)):
            if vector1[i] > vector2[i]:
                if result == -1:
                    return 0
                result = 1
            elif vector2[i] > vector1[i]:
                if result == 1:
                    return 0
                result = -1

        return result


class GDominanceComparator(DominanceComparator):

    def __init__(self,
                 reference_point: (),
                 constraint_comparator: Comparator = SolutionAttributeComparator('overall_constraint_violation', False)):
        super(GDominanceComparator, self).__init__(constraint_comparator)
        self.reference_point = reference_point

    def compare(self, solution1: Solution, solution2: Solution):
        if self.__flag(solution1) > self.__flag(solution2):
            result = -1
        elif self.__flag(solution1) < self.__flag(solution2):
            result = 1
        else:
            result = super(GDominanceComparator, self).compare(solution1, solution2)

        return result

    def __flag(self, solution: Solution):
        result = 1
        for i in range(solution.number_of_objectives):
            if solution.objectives[i] > self.reference_point[i]:
                result = 0

        if result == 0:
            result = 1
            for i in range(solution.number_of_objectives):
                if solution.objectives[i] < self.reference_point[i]:
                    result = 0

        return result


class EpsilonDominanceComparator(DominanceComparator):

    def __init__(self,
                 epsilon: float,
                 constraint_comparator: Comparator = SolutionAttributeComparator('overall_constraint_violation', False)):
        super(EpsilonDominanceComparator, self).__init__(constraint_comparator)
        self.epsilon = epsilon

    def compare(self, solution1: Solution, solution2: Solution):
        result = self.constraint_comparator.compare(solution1, solution2)
        if result == 0:
            result = self.__dominance_test(solution1, solution2)

        return result

    def __dominance_test(self, solution1: Solution, solution2: Solution):
        best_is_one = False
        best_is_two = False
        for i in range(solution1.number_of_objectives):
            value1 = math.floor(solution1.objectives[i] / self.epsilon)
            value2 = math.floor(solution2.objectives[i] / self.epsilon)

            if value1 < value2:
                best_is_one = True
                if best_is_two:
                    return 0
            elif value2 < value1:
                best_is_two = True
                if best_is_one:
                    return 0

        if not best_is_one and not best_is_two:
            dist1 = 0.0
            dist2 = 0.0

            for i in range(solution1.number_of_objectives):
                index1 = math.floor(solution1.objectives[i] / self.epsilon)
                index2 = math.floor(solution2.objectives[i] / self.epsilon)

                dist1 += math.pow(solution1.objectives[i] - index1 * self.epsilon, 2.0)
                dist2 += math.pow(solution2.objectives[i] - index2 * self.epsilon, 2.0)

            if dist1 < dist2:
                return -1
            else:
                return 1
        else:
            if best_is_two:
                return 1
            else:
                return -1
