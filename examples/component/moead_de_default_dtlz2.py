from jmetal.component.algorithm.multiobjective.moead import build_moead_de
from jmetal.component.catalogue.common.termination import TerminationByEvaluations
from jmetal.operator.mutation import PolynomialMutation
from jmetal.problem import DTLZ2
from jmetal.util.plotting import save_plt_to_file
from jmetal.util.solution import (
    get_non_dominated_solutions,
    print_function_values_to_file,
    print_variables_to_file,
)

"""
Program to configure and run the component-based MOEA/D-DE (build_moead_de()) on
DTLZ2 (3 objectives). population_size=91 matches a weight-vector file already bundled
in resources/MOEAD_weights/ (W3D_91.dat). DTLZ2's Pareto front is a smooth,
convex/spherical octant, a comparatively easy 3-objective convergence test.
"""

if __name__ == "__main__":
    problem = DTLZ2()

    max_evaluations = 40000
    algorithm = build_moead_de(
        problem,
        population_size=91,
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
