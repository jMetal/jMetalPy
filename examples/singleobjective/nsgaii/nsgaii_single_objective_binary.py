from jmetal.algorithm.multiobjective.nsgaii import NSGAII
from jmetal.operator import BitFlipMutation, SPXCrossover
from jmetal.problem import OneMax
from jmetal.util.comparator import DominanceComparator
from jmetal.util.solution import print_function_values_to_file, print_variables_to_file
from jmetal.util.termination_criterion import StoppingByEvaluations

if __name__ == '__main__':
    binary_string_length = 512
    problem = OneMax(binary_string_length)

    max_evaluations = 20000

    algorithm = NSGAII(
        problem=problem,
        population_size=100,
        offspring_population_size=1,
        mutation=BitFlipMutation(probability=1.0 / binary_string_length),
        crossover=SPXCrossover(probability=1.0),
        termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations),
        dominance_comparator=DominanceComparator()
    )

    algorithm.run()
    front = algorithm.get_result()

    # Save results to file
    print_function_values_to_file(front, 'FUN.'+ algorithm.get_name()+"-"+problem.get_name())
    print_variables_to_file(front, 'VAR.' + algorithm.get_name()+"-"+problem.get_name())

    print('Algorithm (continuous problem): ' + algorithm.get_name())
    print('Problem: ' + problem.get_name())
    print('Computing time: ' + str(algorithm.total_computing_time))
