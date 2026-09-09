from jmetal.component.algorithm.multiobjective.moead import build_moead_de
from jmetal.component.catalogue.common.termination import TerminationByEvaluations
from jmetal.operator.mutation import PolynomialMutation
from jmetal.problem import LZ09_F2
from jmetal.util.observer import VisualizerObserver
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
example uses population_size=100, 175000 evaluations instead).

Registers a VisualizerObserver to watch the front evolve in a live-updating
matplotlib window while the algorithm runs. This needs no special support on the
component side: algorithm.observable is the exact same DefaultObservable mechanism
the classic algorithms use, notified with the same
"PROBLEM"/"EVALUATIONS"/"SOLUTIONS"/"COMPUTING_TIME" keys, so every existing observer
in jmetal.util.observer (VisualizerObserver, ProgressBarObserver,
WriteFrontToFileObserver, ...) already works against a component-based algorithm
unchanged -- see examples/multiobjective/nsgaii/nsgaii_steady_state_with_real_time_plotting.py
for the same observer used with the classic algorithm hierarchy.

MOEA/D-DE is steady-state (one evaluation per generation), so display_frequency=1000
redraws roughly every 1000 generations -- about 175 updates over the full run.
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

    algorithm.observable.register(
        observer=VisualizerObserver(reference_front=problem.reference_front, display_frequency=1000)
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
