from typing import TypeVar, List, Generic

S = TypeVar('S')


class SolutionListOutput(Generic[S]):
    @staticmethod
    def print_variables_to_screen(solution_list:List[S]):
        for solution in solution_list:
            print(solution.variables[0])

    @staticmethod
    def print_function_values_to_screen(solution_list:List[S]):
        for solution in solution_list:
            print(str(solution_list.index(solution)) + ": ", sep='  ', end='', flush=True)
            print(solution.objectives, sep='  ', end='', flush=True)
            print()

    @staticmethod
    def print_function_values_to_file(file_name, solution_list:List[S]):
        print(file_name)
        with open(file_name, 'w') as of:
            for solution in solution_list:
                for function_value in solution.objectives:
                    print(function_value)
                    of.write(str(function_value) + " ")

