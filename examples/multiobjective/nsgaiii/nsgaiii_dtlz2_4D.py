from jmetal.algorithm.multiobjective.nsgaiii import (
    NSGAIII,
    UniformReferenceDirectionFactory,
)
from jmetal.operator.crossover import SBXCrossover
from jmetal.operator.mutation import PolynomialMutation
from jmetal.problem import DTLZ2
from jmetal.util.plotting import save_plt_to_file
from jmetal.util.solution import (
    get_non_dominated_solutions,
    print_function_values_to_file,
    print_variables_to_file,
)
from jmetal.util.termination_criterion import StoppingByEvaluations

if __name__ == "__main__":
    problem = DTLZ2(12, 4)

    max_evaluations = 30000

    algorithm = NSGAIII(
        problem=problem,
        population_size=100,
        reference_directions=UniformReferenceDirectionFactory(4, n_points=100),
        mutation=PolynomialMutation(probability=1.0 / problem.number_of_variables(), distribution_index=20),
        crossover=SBXCrossover(probability=1.0, distribution_index=20),
        termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations),
    )

    algorithm.run()
    front = get_non_dominated_solutions(algorithm.result())

    # Save results to file
    print_function_values_to_file(front, "FUN." + algorithm.label)
    print_variables_to_file(front, "VAR." + algorithm.label)

    # Save a PNG visualization of the front (and optional HTML if Plotly available)
    try:
        png = save_plt_to_file(front, "FUN." + algorithm.label, out_dir='.', html_plotly=True)
        print(f"Saved front plot to: {png}")
    except Exception as e:
        print(f"Warning: could not generate front plot: {e}")

    print(f"Algorithm: {algorithm.get_name()}")
    print(f"Problem: {problem.name()}")
    print(f"Computing time: {algorithm.total_computing_time}")
