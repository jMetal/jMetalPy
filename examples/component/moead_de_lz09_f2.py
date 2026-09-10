from jmetal.component.algorithm.multiobjective.moead import build_moead_de
from jmetal.component.catalogue.common.termination import TerminationByEvaluations
from jmetal.operator.mutation import PolynomialMutation
from jmetal.problem import LZ09_F2
from jmetal.util.plotting import save_plt_to_file
from jmetal.util.solution import (
    get_non_dominated_solutions,
    print_function_values_to_file,
    print_variables_to_file,
    read_solutions,
)

"""
Program to configure and run the component-based MOEA/D-DE (build_moead_de()) on
LZ09_F2 -- the same problem examples/multiobjective/moead/moead_lz09.py configures
the classic algorithm with (population_size=300, 150000 evaluations there; this
example uses population_size=100, 175000 evaluations instead). For a look at how the
front evolves over the course of this run, see the "observer pattern" section of
notebooks/MOEADComponentBased.ipynb, which runs this exact configuration and plots a
snapshot grid at several evaluation checkpoints -- non-interactive, so it renders the
same way every time regardless of environment, unlike a live-updating plot window.
"""

if __name__ == "__main__":
    problem = LZ09_F2()
    problem.reference_front = read_solutions(filename="resources/reference_fronts/LZ09_F2.pf")

    max_evaluations = 175000
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
