from typing import TypeVar, Generic, List

from jmetal.component.density_estimator import CrowdingDistance
from jmetal.util.comparator import DominanceComparator, EqualSolutionsComparator, SolutionAttributeComparator

S = TypeVar('S')

"""
.. module:: archive
   :platform: Unix, Windows
   :synopsis: Archive implementation.

.. moduleauthor:: Antonio J. Nebro <antonio@lcc.uma.es>
"""


class Archive(Generic[S]):
    def __init__(self):
        self.solution_list: List[S] = []

    def add(self, solution: S) -> bool:
        pass

    def get(self, index:int) -> S:
        return self.solution_list[index]

    def get_solution_list(self) -> List[S]:
        return self.solution_list

    def size(self) -> int:
        return len(self.solution_list)

    def get_comparator(self):
        pass


class BoundedArchive(Archive[S]):
    def __init__(self, maximum_size: int):
        super(BoundedArchive, self).__init__()
        self.maximum_size = maximum_size

    def get_max_size(self) -> int:
        return self.maximum_size

    def compute_density_estimator(self):
        pass

    def sort(self):
        pass


class NonDominatedSolutionListArchive(Archive[S]):
    def __init__(self):
        super(NonDominatedSolutionListArchive, self).__init__()
        self.comparator = DominanceComparator()

    def add(self, solution:S) -> bool:
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
                    del self.solution_list[index-number_of_deleted_solutions]
                    number_of_deleted_solutions += 1
                elif is_dominated_flag == 1:
                    is_dominated = True
                    break
                elif is_dominated_flag == 0:
                    if EqualSolutionsComparator().compare(solution, current_solution) == 0:
                        is_contained = True
                        break

        if not is_dominated and not is_contained:
            self.solution_list.append(solution)
            return True

        return False

    def get_comparator(self):
        return self.comparator


class CrowdingDistanceArchive(BoundedArchive[S]):
    def __init__(self, maximum_size: int):
        super(CrowdingDistanceArchive, self).__init__(maximum_size)

        self.__non_dominated_solution_archive = NonDominatedSolutionListArchive[S]()
        self.__comparator = SolutionAttributeComparator("crowding_distance", lowest_is_best=False)
        self.__crowding_distance = CrowdingDistance()
        self.solution_list = self.__non_dominated_solution_archive.get_solution_list()

    def add(self, solution: S) -> bool:
        success: bool = self.__non_dominated_solution_archive.add(solution)
        if success:
            if self.size() > self.get_max_size():
                self.compute_density_estimator()
                worst_solution = self.__find_worst_solution(self.get_solution_list())
                self.get_solution_list().remove(worst_solution)

        return success

    def compute_density_estimator(self):
        self.__crowding_distance.compute_density_estimator(self.get_solution_list())

    def __find_worst_solution(self, solution_list: List[S]) -> S:
        if solution_list is None:
            raise Exception("The solution list is None")
        elif len(solution_list) is 0:
            raise Exception("The solution list is empty")

        worst_solution = solution_list[0]
        for solution in solution_list[1:]:
            if self.__comparator.compare(worst_solution, solution) < 0:
                worst_solution = solution

        return worst_solution

    def get_comparator(self):
        return self.__comparator
