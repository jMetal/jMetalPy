from jmetal.algorithm.multiobjective.nsgaii import NSGAII
from jmetal.operator import BitFlipMutation, SPXCrossover
from jmetal.problem.multiobjective.unconstrained import OneZeroMax
from jmetal.util.solution import print_function_values_to_file, print_variables_to_file
from jmetal.util.termination_criterion import StoppingByEvaluations

"""  
Program to  configure and run the NSGA-II algorithm configured to solve a binary problem, OneZeroMax, which is 
multiobjective version of the ONE_MAX problem where the numbers of 1s and 0s have to be maximized at the same time.
"""

if __name__ == "__main__":
    binary_string_length = 512
    problem = OneZeroMax(binary_string_length)

    max_evaluations = 30000
    algorithm = NSGAII(
        problem=problem,
        population_size=100,
        offspring_population_size=100,
        mutation=BitFlipMutation(probability=1.0 / binary_string_length),
        crossover=SPXCrossover(probability=1.0),
        termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations),
    )

    algorithm.run()
    front = algorithm.result()

    # Save results to file
    print_function_values_to_file(front, "FUN." + algorithm.label)
    print_variables_to_file(front, "VAR." + algorithm.label)

    print(f"Algorithm: {algorithm.get_name()}")
    print(f"Problem: {problem.name()}")
    print(f"Computing time: {algorithm.total_computing_time}")
