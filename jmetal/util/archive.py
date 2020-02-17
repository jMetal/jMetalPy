import copy
import random
from abc import ABC, abstractmethod
from threading import Lock
from typing import TypeVar, Generic, List

from jmetal.util.comparator import Comparator, DominanceComparator, SolutionAttributeComparator
from jmetal.util.density_estimator import DensityEstimator, CrowdingDistance

S = TypeVar('S')

"""
.. module:: archive
   :platform: Unix, Windows
   :synopsis: Archive implementation.

.. moduleauthor:: Antonio J. Nebro <antonio@lcc.uma.es>
"""


class Archive(Generic[S], ABC):

    def __init__(self):
        self.solution_list: List[S] = []

    @abstractmethod
    def add(self, solution: S) -> bool:
        pass

    def get(self, index: int) -> S:
        return self.solution_list[index]

    def size(self) -> int:
        return len(self.solution_list)

    def get_name(self) -> str:
        return self.__class__.__name__


class BoundedArchive(Archive[S]):

    def __init__(self,
                 maximum_size: int,
                 comparator: Comparator[S] = None,
                 density_estimator: DensityEstimator = None):
        super(BoundedArchive, self).__init__()
        self.maximum_size = maximum_size
        self.comparator = comparator
        self.density_estimator = density_estimator
        self.non_dominated_solution_archive = NonDominatedSolutionsArchive()
        self.solution_list = self.non_dominated_solution_archive.solution_list

    def compute_density_estimator(self):
        self.density_estimator.compute_density_estimator(self.solution_list)

    def add(self, solution: S) -> bool:
        success = self.non_dominated_solution_archive.add(solution)

        if success:
            if self.size() > self.maximum_size:
                self.compute_density_estimator()
                worst_solution, index_to_remove = self.__find_worst_solution(self.solution_list)
                self.solution_list.pop(index_to_remove)

        return success

    def __find_worst_solution(self, solution_list: List[S]) -> S:
        if solution_list is None:
            raise Exception("The solution list is None")
        elif len(solution_list) is 0:
            raise Exception("The solution list is empty")

        worst_solution = solution_list[0]
        index_to_remove = 0

        for solution_index, solution in enumerate(solution_list[1:]):
            if self.comparator.compare(worst_solution, solution) < 0:
                worst_solution = solution
                index_to_remove = solution_index + 1

        return worst_solution, index_to_remove


class NonDominatedSolutionsArchive(Archive[S]):

    def __init__(self, dominance_comparator: Comparator = DominanceComparator()):
        super(NonDominatedSolutionsArchive, self).__init__()
        self.comparator = dominance_comparator

    def add(self, solution: S) -> bool:
        is_dominated = False
        is_contained = False

        if len(self.solution_list) == 0:
            self.solution_list.append(solution)
            return True
        else:
            number_of_deleted_solutions = 0

            # New copy of list and enumerate
            for index, current_solution in enumerate(list(self.solution_list)):
                is_dominated_flag = self.comparator.compare(solution, current_solution)
                if is_dominated_flag == -1:
                    del self.solution_list[index - number_of_deleted_solutions]
                    number_of_deleted_solutions += 1
                elif is_dominated_flag == 1:
                    is_dominated = True
                    break
                elif is_dominated_flag == 0:
                    if solution.objectives == current_solution.objectives:
                        is_contained = True
                        break

        if not is_dominated and not is_contained:
            self.solution_list.append(solution)
            return True

        return False


class CrowdingDistanceArchive(BoundedArchive[S]):

    def __init__(self,
                 maximum_size: int):
        super(CrowdingDistanceArchive, self).__init__(
            maximum_size=maximum_size,
            comparator=SolutionAttributeComparator("crowding_distance", lowest_is_best=False),
            density_estimator=CrowdingDistance())


class ArchiveWithReferencePoint(BoundedArchive[S]):

    def __init__(self,
                 maximum_size: int,
                 reference_point: List[float],
                 comparator: Comparator[S],
                 density_estimator: DensityEstimator):
        super(ArchiveWithReferencePoint, self).__init__(maximum_size, comparator, density_estimator)
        self.__reference_point = reference_point
        self.__comparator = comparator
        self.__density_estimator = density_estimator
        self.lock = Lock()

    def add(self, solution: S) -> bool:
        with self.lock:
            dominated_solution = None

            if self.__dominance_test(solution.objectives, self.__reference_point) == 0:
                if len(self.solution_list) == 0:
                    result = True
                else:
                    if random.uniform(0.0, 1.0) < 0.05:
                        result = True
                        dominated_solution = solution
                    else:
                        result = False
            else:
                result = True

            if result:
                result = super(ArchiveWithReferencePoint, self).add(solution)

            if result and dominated_solution is not None and len(self.solution_list) > 1:
                if dominated_solution in self.solution_list:
                    self.solution_list.remove(dominated_solution)

            if result and len(self.solution_list) > self.maximum_size:
                self.compute_density_estimator()

        return result

    def filter(self):
        # In case of having at least a solution which is non-dominated with the reference point, filter it
        if len(self.solution_list) > 1:
            self.solution_list[:] = \
                [sol for sol in self.solution_list
                 if self.__dominance_test(sol.objectives, self.__reference_point) != 0]

    def update_reference_point(self, new_reference_point) -> None:
        with self.lock:
            self.__reference_point = new_reference_point

            first_solution = copy.deepcopy(self.solution_list[0])
            self.filter()

            if len(self.solution_list) == 0:
                self.solution_list.append(first_solution)

    def get_reference_point(self) -> List[float]:
        with self.lock:
            return self.__reference_point

    def __dominance_test(self, vector1: List[float], vector2: List[float]) -> int:
        best_is_one = 0
        best_is_two = 0

        for value1, value2 in zip(vector1, vector2):
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


class CrowdingDistanceArchiveWithReferencePoint(ArchiveWithReferencePoint[S]):

    def __init__(self,
                 maximum_size: int,
                 reference_point: List[float]):
        super(CrowdingDistanceArchiveWithReferencePoint, self).__init__(
            maximum_size=maximum_size,
            reference_point=reference_point,
            comparator=SolutionAttributeComparator("crowding_distance", lowest_is_best=False),
            density_estimator=CrowdingDistance())
