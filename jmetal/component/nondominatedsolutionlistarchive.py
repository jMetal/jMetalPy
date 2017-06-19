from typing import TypeVar, Generic, List

from jmetal.util.comparator import dominance_comparator, equal_solutions_comparator

S = TypeVar('S')

class NonDominatedSolutionListArchive(Generic[S]):
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
            #New copy of list and enumerate
            for index, current_solution in enumerate(list(self.solution_list)):
                is_dominated_flag = dominance_comparator(solution, current_solution)
                if is_dominated_flag == -1:
                    del self.solution_list[index-number_of_deleted_solutions]
                    number_of_deleted_solutions+=1
                elif is_dominated_flag == 1:
                    is_dominated = True
                    break;
                elif is_dominated_flag == 0:
                    if equal_solutions_comparator(solution, current_solution) == 0:
                        is_contained = True
                        break;

        if not is_dominated and not is_contained:
            self.solution_list.append(solution)
            return True
        return False

    def get(self, index:int) -> S:
        return self.solution_list[index]

    def get_solution_list(self) -> List[S]:
        return self.solution_list

    def size(self) -> int:
        return len(self.solution_list)