from jmetal.algorithm.multiobjective.nsgaii import NSGAII
from jmetal.operator.crossover import SBXCrossover
from jmetal.operator.mutation import PolynomialMutation
from jmetal.problem import ZDT4
from jmetal.util.archive import DistanceBasedArchive
from jmetal.util.distance import DistanceMetric
from jmetal.util.evaluator import SequentialEvaluatorWithArchive
from jmetal.util.solution import (
    get_non_dominated_solutions,
    print_function_values_to_file,
    print_variables_to_file, read_solutions,
)
from jmetal.util.termination_criterion import StoppingByEvaluations

"""  
 Program to configure and run the NSGA-II algorithm with a DistanceBasedArchive using LINF metric.
 This example demonstrates the use of L-infinity (maximum) distance metric for maintaining diversity.
"""
if __name__ == "__main__":
    problem = ZDT4()

    problem.reference_front = read_solutions(filename="resources/reference_fronts/ZDT4.pf")

    # Create distance-based archive with size 100 using L-infinity distance metric
    archive = DistanceBasedArchive(maximum_size=100, metric=DistanceMetric.LINF)
    evaluator = SequentialEvaluatorWithArchive(archive)

    max_evaluations = 25000
    algorithm = NSGAII(
        problem=problem,
        population_size=100,
        offspring_population_size=100,
        mutation=PolynomialMutation(probability=1.0 / problem.number_of_variables(), distribution_index=20),
        crossover=SBXCrossover(probability=1.0, distribution_index=20),
        termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations),
        population_evaluator=evaluator,
    )

    algorithm.run()

    # Get solutions from the archive instead of the algorithm result
    front = evaluator.get_archive().solution_list

    # Save results to file with specific suffix
    print_function_values_to_file(front, "FUN." + algorithm.label + ".LINF")
    print_variables_to_file(front, "VAR." + algorithm.label + ".LINF")

    print(f"Algorithm: {algorithm.get_name()}")
    print(f"Problem: {problem.name()}")
    print(f"Archive size: {len(front)} solutions")
    print(f"Distance metric: {archive.metric.name}")
    print(f"Computing time: {algorithm.total_computing_time}")