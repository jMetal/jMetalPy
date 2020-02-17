from jmetal.algorithm.multiobjective.gde3 import GDE3
from jmetal.problem import ZDT2
from jmetal.util.comparator import GDominanceComparator
from jmetal.util.solution import read_solutions, print_function_values_to_file, print_variables_to_file
from jmetal.util.termination_criterion import StoppingByEvaluations

if __name__ == '__main__':
    problem = ZDT2()
    problem.reference_front = read_solutions(filename='resources/reference_front/ZDT2.pf')

    max_evaluations = 25000
    reference_point = [0.2, 0.5]

    algorithm = GDE3(
        problem=problem,
        population_size=100,
        cr=0.5,
        f=0.5,
        termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations),
        dominance_comparator=GDominanceComparator(reference_point)
    )

    algorithm.run()
    front = algorithm.get_result()

    # Save results to file
    print_function_values_to_file(front, 'FUN.' + algorithm.label)
    print_variables_to_file(front, 'VAR.'+ algorithm.label)

    print(f'Algorithm: ${algorithm.get_name()}')
    print(f'Problem: ${problem.get_name()}')
    print(f'Computing time: ${algorithm.total_computing_time}')
