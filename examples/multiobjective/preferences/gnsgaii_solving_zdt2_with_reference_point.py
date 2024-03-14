from jmetal.algorithm.multiobjective.nsgaii import NSGAII
from jmetal.lab.visualization import InteractivePlot, Plot
from jmetal.operator import PolynomialMutation, SBXCrossover
from jmetal.problem import ZDT2
from jmetal.util.comparator import GDominanceComparator
from jmetal.util.observer import ProgressBarObserver, VisualizerObserver
from jmetal.util.solution import (
    print_function_values_to_file,
    print_variables_to_file,
    read_solutions,
)
from jmetal.util.termination_criterion import StoppingByEvaluations

"""  
Program to configure and run G-NSGA-II (NSGA-II with G-Dominance) to solve problem ZDT2 with 
reference point = [0.2, 0.5].
"""

if __name__ == "__main__":
    problem = ZDT2()
    reference_front = read_solutions(filename="resources/reference_front/ZDT2.pf")

    reference_point = [0.3, 0.5]

    max_evaluations = 25000
    algorithm = NSGAII(
        problem=problem,
        population_size=100,
        offspring_population_size=100,
        mutation=PolynomialMutation(probability=1.0 / problem.number_of_variables(), distribution_index=20),
        crossover=SBXCrossover(probability=1.0, distribution_index=20),
        dominance_comparator=GDominanceComparator(reference_point),
        termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations),
    )

    algorithm.observable.register(observer=ProgressBarObserver(max=max_evaluations))
    algorithm.observable.register(
        observer=VisualizerObserver(reference_front=reference_front, reference_point=reference_point)
    )

    algorithm.run()
    front = algorithm.result()

    # Plot front
    plot_front = Plot(
        title="Pareto front approximation. Problem: " + problem.name(),
        reference_front=reference_front,
        axis_labels=problem.obj_labels,
    )
    plot_front.plot(front, label=algorithm.label, filename=algorithm.get_name())

    # Plot interactive front
    plot_front = InteractivePlot(
        title="Pareto front approximation. Problem: " + problem.name(),
        reference_front=reference_front,
        axis_labels=problem.obj_labels,
    )
    plot_front.plot(front, label=algorithm.label, filename=algorithm.get_name())

    # Save results to file
    print_function_values_to_file(front, "FUN." + algorithm.label)
    print_variables_to_file(front, "VAR." + algorithm.label)

    print(f"Algorithm: {algorithm.get_name()}")
    print(f"Problem: {problem.name()}")
    print(f"Computing time: {algorithm.total_computing_time}")
