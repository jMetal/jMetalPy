from jmetal.component.algorithm.multiobjective.moead import build_moead_de
from jmetal.component.catalogue.common.termination import TerminationByEvaluations
from jmetal.operator.mutation import PolynomialMutation
from jmetal.problem import ZDT4
from jmetal.util.plotting import save_plt_to_file
from jmetal.util.solution import (
    get_non_dominated_solutions,
    print_function_values_to_file,
    print_variables_to_file,
)

"""
Program to configure and run the component-based MOEA/D-DE (build_moead_de()) --
differential-evolution crossover, matching what jMetalPy's classic (non-component)
MOEAD class actually implements (see moead_default_zdt4.py's docstring for the
classic-vs-DE naming story). ZDT4 is multi-modal (many local Pareto fronts): a
2-objective problem, so no weight-vector file is needed.

Note on the evaluation budget: with the default CR=1.0/F=0.5, MOEA/D-DE needs
noticeably more evaluations than the SBX-based classic variant to escape ZDT4's local
fronts reliably -- 25000 evaluations (matching moead_default_zdt4.py's budget) landed
around HV=0.31-0.48 across a few seeds during development, well below the SBX
variant's ~0.64; doubling the budget to 50000 closed that gap almost entirely
(~0.66). This isn't a bug -- SBX's stronger exploration on this specific multi-modal
landscape is a known property of the operator, not of MOEA/D itself.
"""

if __name__ == "__main__":
    problem = ZDT4()

    max_evaluations = 50000
    algorithm = build_moead_de(
        problem,
        population_size=100,
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
