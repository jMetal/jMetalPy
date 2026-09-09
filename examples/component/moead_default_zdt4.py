from jmetal.component.algorithm.multiobjective.moead import build_moead
from jmetal.component.catalogue.common.termination import TerminationByEvaluations
from jmetal.operator.crossover import SBXCrossover
from jmetal.operator.mutation import PolynomialMutation
from jmetal.problem import ZDT4
from jmetal.util.plotting import save_plt_to_file
from jmetal.util.solution import (
    get_non_dominated_solutions,
    print_function_values_to_file,
    print_variables_to_file,
)

"""
Program to configure and run the component-based, classic MOEA/D (build_moead()) --
SBX crossover and polynomial mutation, not differential evolution; see
moead_de_default_zdt4.py for the MOEA/D-DE variant on the same problem. ZDT4 is
multi-modal (many local Pareto fronts): a 2-objective problem, so its weight vectors
are generated analytically and no weight-vector file is needed.
"""

if __name__ == "__main__":
    problem = ZDT4()

    max_evaluations = 25000
    algorithm = build_moead(
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
