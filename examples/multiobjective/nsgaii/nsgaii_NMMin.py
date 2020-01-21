from jmetal.algorithm.multiobjective.nsgaii import NSGAII
from jmetal.operator import SBXCrossover, PolynomialMutation, NullMutation
from jmetal.operator.crossover import IntegerFloatSBXCrossover, IntegerSBXCrossover
from jmetal.problem import ZDT1
from jmetal.problem.multiobjective.unconstrained import NMMin
from jmetal.util.solution import get_non_dominated_solutions, read_solutions, print_function_values_to_file, \
    print_variables_to_file
from jmetal.util.termination_criterion import StoppingByEvaluations

if __name__ == '__main__':
    problem = NMMin(10,10, 100, -100, -1000, 1000)

    max_evaluations = 25000
    algorithm = NSGAII(
        problem=problem,
        population_size=100,
        offspring_population_size=100,
        mutation=NullMutation(),
        crossover=IntegerFloatSBXCrossover(IntegerSBXCrossover(probability=1.0, distribution_index=20),
                                           SBXCrossover(probability=1.0, distribution_index=20)),
        termination_criterion=StoppingByEvaluations(max=max_evaluations)
    )

    algorithm.run()
    front = get_non_dominated_solutions(algorithm.get_result())

    # Save results to file
    print_function_values_to_file(front, 'FUN.' + algorithm.label)
    print_variables_to_file(front, 'VAR.' + algorithm.label)

    print('Algorithm (continuous problem): ' + algorithm.get_name())
    print('Problem: ' + problem.get_name())
    print('Computing time: ' + str(algorithm.total_computing_time))
