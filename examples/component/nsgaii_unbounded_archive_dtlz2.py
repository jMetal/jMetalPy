from jmetal.component.algorithm.multiobjective.nsgaii import build_nsgaii
from jmetal.component.catalogue.common.termination import TerminationByEvaluations
from jmetal.operator.crossover import SBXCrossover
from jmetal.operator.mutation import PolynomialMutation
from jmetal.problem import DTLZ2
from jmetal.util.archive import NonDominatedSolutionsArchive
from jmetal.util.plotting import save_plt_to_file
from jmetal.util.solution import print_function_values_to_file, print_variables_to_file

"""
Program to configure and run the component-based NSGA-II with an unbounded external
archive (every non-dominated solution ever evaluated is kept, so the archive can
grow much larger than the population) on DTLZ2 (3 objectives).

Note on runtime: NonDominatedSolutionsArchive.add() is O(n) per insertion, and the
archive can grow into the thousands over a full run, so this takes noticeably
longer than the bounded-archive example -- tens of seconds at 40,000 evaluations,
not a couple of seconds.
"""

if __name__ == "__main__":
    problem = DTLZ2()

    archive = NonDominatedSolutionsArchive()

    max_evaluations = 40000
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
