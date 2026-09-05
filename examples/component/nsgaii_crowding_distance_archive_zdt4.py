from jmetal.component.algorithm.multiobjective.nsgaii import build_nsgaii
from jmetal.component.catalogue.common.termination import TerminationByEvaluations
from jmetal.operator.crossover import SBXCrossover
from jmetal.operator.mutation import PolynomialMutation
from jmetal.problem import ZDT4
from jmetal.util.archive import CrowdingDistanceArchive
from jmetal.util.plotting import save_plt_to_file
from jmetal.util.solution import print_function_values_to_file, print_variables_to_file

"""
Program to configure and run the component-based NSGA-II with a bounded, external
crowding-distance archive on ZDT4. ZDT4 is multi-modal (many local Pareto fronts),
which is exactly where an external archive helps: every evaluated solution is kept
as a candidate regardless of what the population/replacement strategy discards,
guarding against the population converging on a local front. The archive result is
compared against the analogous classic-algorithm example,
examples/multiobjective/nsgaii/nsgaii_standard_settings_with_crowding_distance_archive.py.
"""

if __name__ == "__main__":
    problem = ZDT4()

    archive = CrowdingDistanceArchive(maximum_size=100)

    max_evaluations = 20000
    algorithm = build_nsgaii(
        problem,
        population_size=100,
        offspring_population_size=100,
        crossover=SBXCrossover(probability=1.0, distribution_index=20),
        mutation=PolynomialMutation(
            probability=1.0 / problem.number_of_variables(), distribution_index=20
        ),
        termination=TerminationByEvaluations(max_evaluations=max_evaluations),
        archive=archive,
    )

    algorithm.run()

    # algorithm.result() returns the archive's contents, since an archive was given.
    front = algorithm.result()
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
    print(f"Archive size: {len(front)}")
    print(f"Computing time: {algorithm.total_computing_time}")
