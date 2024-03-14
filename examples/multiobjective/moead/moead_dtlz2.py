from jmetal.algorithm.multiobjective.moead import MOEAD
from jmetal.core.quality_indicator import HyperVolume
from jmetal.operator import DifferentialEvolutionCrossover, PolynomialMutation
from jmetal.problem import DTLZ2
from jmetal.util.aggregation_function import PenaltyBoundaryIntersection

from jmetal.util.solution import (
    print_function_values_to_file,
    print_variables_to_file,
    read_solutions,
)
from jmetal.util.termination_criterion import StoppingByEvaluations

if __name__ == "__main__":
    problem = DTLZ2()
    problem.reference_front = read_solutions(filename="resources/reference_front/DTLZ2.3D.pf")

    max_evaluations = 50000

    algorithm = MOEAD(
        problem=problem,
        population_size=91,
        crossover=DifferentialEvolutionCrossover(CR=1.0, F=0.5),
        mutation=PolynomialMutation(probability=1.0 / problem.number_of_variables(), distribution_index=20),
        aggregation_function=PenaltyBoundaryIntersection(dimension=problem.number_of_objectives()),
        neighbor_size=20,
        neighbourhood_selection_probability=0.9,
        max_number_of_replaced_solutions=2,
        weight_files_path="resources/MOEAD_weights",
        termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations),
    )

    algorithm.run()
    front = algorithm.result()

    hypervolume = HyperVolume([1.0, 1.0, 1.0])
    print("Hypervolume: " + str(hypervolume.compute([front[i].objectives for i in range(len(front))])))

    # Save results to file
    print_function_values_to_file(front, "FUN." + algorithm.label)
    print_variables_to_file(front, "VAR." + algorithm.label)

    print(f"Algorithm: {algorithm.get_name()}")
    print(f"Problem: {problem.name()}")
    print(f"Computing time: {algorithm.total_computing_time}")
