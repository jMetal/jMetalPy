from jmetal.algorithm.multiobjective.nsgaii import NSGAII
from jmetal.operator.crossover import SBXCrossover
from jmetal.operator.mutation import PolynomialMutation
from jmetal.problem.multiobjective.dtlz import DTLZ2
from jmetal.util.archive import DistanceBasedArchive
from jmetal.util.distance import DistanceMetric
from jmetal.util.evaluator import SequentialEvaluatorWithArchive
from jmetal.util.plotting import save_plt_to_file
from jmetal.util.solution import (
    print_function_values_to_file,
    print_variables_to_file, read_solutions,
)
from jmetal.util.termination_criterion import StoppingByEvaluations

"""  
 Program to configure and run the NSGA-II algorithm with a DistanceBasedArchive.
 This example uses L2 squared distance metric for maintaining diversity in the archive.
 Testing with DTLZ2 problem (3 objectives) - convex Pareto front.
 
 The DistanceBasedArchive supports two implementations:
 - use_vectorized=True (default): Optimized vectorized implementation (~28% faster)
 - use_vectorized=False: Original iterative implementation
 Both implementations produce mathematically identical results.
"""
if __name__ == "__main__":
    problem = DTLZ2()

    problem.reference_front = read_solutions(filename="resources/reference_fronts/DTLZ2.3D.pf")

    # Create distance-based archive with size 100 using L2 squared distance metric
    # use_vectorized=True enables optimized vectorized implementation for better performance
    archive = DistanceBasedArchive(maximum_size=100, metric=DistanceMetric.L2_SQUARED, use_vectorized=True)
    evaluator = SequentialEvaluatorWithArchive(archive)

<<<<<<< HEAD
    max_evaluations = 50000
=======
    max_evaluations = 40000
>>>>>>> 6f4d831 (Add Optuna hyperparameter tuning package)
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

    # Save results to file
    print_function_values_to_file(front, "FUN." + algorithm.label)
    print_variables_to_file(front, "VAR." + algorithm.label)
    # Save a PNG visualization of the front (and optional HTML if Plotly available)
    try:
        png = save_plt_to_file(front, "FUN." + algorithm.label, out_dir='.', html_plotly=True)
        print(f"Saved front plot to: {png}")
    except Exception as e:
        print(f"Warning: could not generate front plot: {e}")

    # Save a PNG visualization of the front (and optional HTML if Plotly available)
    png = save_plt_to_file(front, "FUN." + algorithm.label, out_dir='.', html_plotly=True)
    print(f"Saved front plot to: {png}")


    print(f"Algorithm: {algorithm.get_name()}")
    print(f"Problem: {problem.name()}")
    print(f"Archive size: {len(front)} solutions")
    print(f"Distance metric: {archive.metric.name}")
    print(f"Computing time: {algorithm.total_computing_time}")
    print(f"Total evaluations performed: {algorithm.evaluations}")
    print(f"Configured max evaluations: {max_evaluations}")
    print(f"Evaluations/second: {algorithm.evaluations / algorithm.total_computing_time:.1f}")
    print(f"Archive utilization: {len(front)}/{archive.maximum_size} ({len(front)/archive.maximum_size*100:.1f}%)")