import numpy as np

from jmetal.algorithm.multiobjective.nsgaii import NSGAII
from jmetal.core.quality_indicator import HyperVolume, NormalizedHyperVolume
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


    # Read the reference front and derive the reference point automatically
    reference_front_solutions = read_solutions(filename="resources/reference_fronts/ZDT1.pf")
    reference_front = np.array([s.objectives for s in reference_front_solutions])

    # Use the new API: pass the reference front and a small offset so extreme
    # points still contribute to the hypervolume
    hv = HyperVolume(reference_front=reference_front, reference_point_offset=0.1)
    objective_values = np.array([s.objectives for s in front])
    hv_value = hv.compute(objective_values)

    nhv = NormalizedHyperVolume(reference_front=reference_front, reference_point_offset=0.1)
    # must set/calculate the reference hypervolume used for normalization
    nhv.set_reference_front(reference_front)
    nhv_value = nhv.compute(objective_values)

    print("Hypervolume: ", hv_value)
    print("Normalized Hypervolume: ", nhv_value)