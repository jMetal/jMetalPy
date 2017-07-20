from typing import TypeVar, Generic, List

from jmetal.component.density_estimator import CrowdingDistance
from jmetal.util.comparator import DominanceComparator, EqualSolutionsComparator, Comparator, \
    SolutionAttributeComparator

S = TypeVar('S')


class Archive(Generic[S]):
    def add(self, solution: S) -> bool:
        pass

    def get(self, index:int) -> S:
        pass

    def get_solution_list(self) -> List[S]:
        pass

    def size(self) -> int:
        pass


class BoundedArchive(Archive[S]):
    def __init__(self, maximum_size: int):
        self.maximum_size = maximum_size

    def get_max_size(self) -> int:
        return self.maximum_size

    def get_comparator(self) -> Comparator[S]:
        pass

    def compute_density_estimator(self):
        pass

    def sort(self):
        pass


class NonDominatedSolutionListArchive(Archive[S]):
    def __init__(self):
        self.solution_list = []

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
                is_dominated_flag = DominanceComparator().compare(solution, current_solution)
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

    def get(self, index: int) -> S:
        return self.solution_list[index]

    def get_solution_list(self) -> List[S]:
        return self.solution_list

    def size(self) -> int:
        return len(self.solution_list)


class CrowdingDistanceArchive(BoundedArchive[S]):
    def __init__(self, maximum_size: int):
        super(CrowdingDistanceArchive, self).__init__(maximum_size)

        self.solution_list = []
        self.__non_dominated_solution_archive = NonDominatedSolutionListArchive[S]()
        self.__comparator[S] = SolutionAttributeComparator("crowding_distance", lowest_is_best=False)
        self.__crowding_distance = CrowdingDistance()

    def add(self, solution: S) -> bool:
        success : bool = self.__non_dominated_solution_archive.add(solution)
        if success:
            if self.size() > self.get_max_size():
                self.compute_density_estimator()
                worst_solution = self.__find_worst_solution(self.get_solution_list())
                self.get_solution_list().remove(worst_solution)

        return success

    def get_comparator(self):
        return self.get_comparator()

    def __find_worst_solution(self, solution_list: List[S]) -> S:
        if solution_list is None:
            raise Exception("The solution list is None")
        elif len(solution_list) is 0:
            raise Exception("The solution list is empty")

        worst_solution = solution_list[0]
        for solution in solution_list:
            if self.get_comparator().compare(worst_solution, solution_list) < 0:
                worst_solution = solution

        return worst_solution

