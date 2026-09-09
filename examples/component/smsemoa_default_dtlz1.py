from jmetal.component.algorithm.multiobjective.smsemoa import build_smsemoa
from jmetal.component.catalogue.common.termination import TerminationByEvaluations
from jmetal.operator.crossover import SBXCrossover
from jmetal.operator.mutation import PolynomialMutation
from jmetal.problem import DTLZ1
from jmetal.util.plotting import save_plt_to_file
from jmetal.util.solution import (
    get_non_dominated_solutions,
    print_function_values_to_file,
    print_variables_to_file,
)

"""
Program to configure and run the component-based SMS-EMOA (build_smsemoa()) on DTLZ1
(3 objectives), using the same evaluation budget as the three-objective classic
SMS-EMOA example (examples/multiobjective/smsemoa/smsemoa_standard_settings_three_objective_problem.py).
DTLZ1's Pareto front is a flat hyperplane reachable through 11^k - 1 local optima,
exercising the replacement strategy's ranking on a very different front geometry than
ZDT4's or DTLZ2's/DTLZ3's curved fronts.
"""

if __name__ == "__main__":
    problem = DTLZ1()

    max_evaluations = 40000
    algorithm = build_smsemoa(
        problem,
        population_size=100,
        crossover=SBXCrossover(probability=1.0, distribution_index=20),
        mutation=PolynomialMutation(
            probability=1.0 / problem.number_of_variables(), distribution_index=20
        ),
        termination=TerminationByEvaluations(max_evaluations=max_evaluations),
    )

    algorithm.run()

    front = get_non_dominated_solutions(algorithm.result())
    label = f"{algorithm.get_name()}.{problem.name()}"

    # Save results to file
    print_function_values_to_file(front, "FUN." + label)
    print_variables_to_file(front, "VAR." + label)

    # Save a PNG visualization of the front (and optional HTML if Plotly available)
    try:
        png = save_plt_to_file(front, "FUN." + label, out_dir=".", html_plotly=True)
        print(f"Saved front plot to: {png}")
    except Exception as e:
        print(f"Warning: could not generate front plot: {e}")

    print(f"Algorithm: {algorithm.get_name()}")
    print(f"Problem: {problem.name()}")
    print(f"Computing time: {algorithm.total_computing_time}")
