import numpy as np

from jmetal.algorithm.multiobjective.nsgaii import NSGAII
from jmetal.core.quality_indicator import HyperVolume
from jmetal.operator.crossover import SBXCrossover
from jmetal.operator.mutation import PolynomialMutation
from jmetal.problem import ZDT1, ZDT4
from jmetal.util.solution import (
    get_non_dominated_solutions,
    read_solutions, print_function_values_to_file, print_variables_to_file,
)
from jmetal.util.termination_criterion import StoppingByEvaluations

"""  
 Program to  configure and run the NSGA-II algorithm configured with standard settings.
"""
if __name__ == "__main__":
    problem = ZDT4()

    max_evaluations = 25000
    algorithm = NSGAII(
        problem=problem,
        population_size=100,
        offspring_population_size=100,
        mutation=PolynomialMutation(probability=1.0 / problem.number_of_variables(), distribution_index=20),
        crossover=SBXCrossover(probability=1.0, distribution_index=20),
        termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations),
    )

    algorithm.run()

    front = get_non_dominated_solutions(algorithm.result())

    print(f"Algorithm: {algorithm.get_name()}")
    print(f"Problem: {problem.name()}")
    print(f"Computing time: {algorithm.total_computing_time}")

    # Save results to file
    print_function_values_to_file(front, "FUN." + algorithm.label)
    print_variables_to_file(front, "VAR." + algorithm.label)

    reference_front = read_solutions(filename="resources/reference_fronts/ZDT1.pf")
    reference_point = np.max([solutions.objectives for solutions in reference_front], axis=0)

    hv = HyperVolume(reference_point=reference_point)
    objective_values = [solutions.objectives for solutions in front]
    value = hv.compute(np.array(objective_values))

    print("Hypervolume: ", value)