import argparse

from jmetal.algorithm.multiobjective.nsgaii import NSGAII
from jmetal.lab.visualization import Plot
from jmetal.operator.crossover import PMXCrossover
from jmetal.operator.mutation import PermutationSwapMutation
from jmetal.problem.multiobjective.multiobjective_tsp import MultiObjectiveTSP
from jmetal.util.observer import ProgressBarObserver, VisualizerObserver
from jmetal.util.solution import print_function_values_to_file, print_variables_to_file
from jmetal.util.termination_criterion import StoppingByEvaluations


def main(max_evaluations: int):
    # two-objective TSP using two distance matrices
    problem = MultiObjectiveTSP([
        "kroA100.tsp",
        "kroB100.tsp",
    ])

    print("Cities:", problem.number_of_variables())

    from jmetal.operator.mutation import ScrambleMutation

    algorithm = NSGAII(
        problem=problem,
        population_size=200,
        offspring_population_size=200,
        mutation=ScrambleMutation(0.2),
        crossover=PMXCrossover(0.9),
        termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations),
    )

    algorithm.observable.register(observer=ProgressBarObserver(max=max_evaluations))
    algorithm.observable.register(
        observer=VisualizerObserver(reference_front=None, display_frequency=max(1, max_evaluations // 50))
    )

    algorithm.run()
    front = algorithm.result()

    # Save results
    print_function_values_to_file(front, "FUN." + algorithm.label)
    print_variables_to_file(front, "VAR." + algorithm.label)

    # 2D plot
    plot_front = Plot(
        title="Pareto front approximation - Multiobjective TSP",
        reference_front=None,
        axis_labels=["kroA100", "kroB100"],
    )
    plot_front.plot(front, label=algorithm.label, filename=algorithm.get_name())

    print(f"Algorithm: {algorithm.get_name()}")
    print(f"Problem: MultiObjectiveTSP (kroA100, kroB100)")
    print(f"Computing time: {algorithm.total_computing_time}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-evals", type=int, default=25000, help="Maximum number of evaluations")
    args = parser.parse_args()

    main(args.max_evals)
