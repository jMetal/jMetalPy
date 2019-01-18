from jmetal.algorithm.multiobjective.nsgaiii import NSGAIII
from jmetal.operator import SBXCrossover, PolynomialMutation, BinaryTournamentSelection
from jmetal.problem import DTLZ1
from jmetal.util.comparator import RankingAndCrowdingDistanceComparator
from jmetal.util.observer import ProgressBarObserver, VisualizerObserver, PlotFrontToFileObserver
from jmetal.util.solution_list import read_solutions, print_function_values_to_file, print_variables_to_file
from jmetal.util.termination_criterion import StoppingByEvaluations
from jmetal.util.visualization import Plot

if __name__ == '__main__':
    problem = DTLZ1()
    problem.reference_front = read_solutions(filename='../../resources/reference_front/DTLZ1.3D.pf')

    max_evaluations = 10000
    algorithm = NSGAIII(
        problem=problem,
        mutation=PolynomialMutation(probability=1.0 / problem.number_of_variables, distribution_index=20),
        crossover=SBXCrossover(probability=1.0, distribution_index=20),
        selection=BinaryTournamentSelection(comparator=RankingAndCrowdingDistanceComparator()),
        termination_criterion=StoppingByEvaluations(max=max_evaluations)
    )

    algorithm.observable.register(observer=ProgressBarObserver(max=max_evaluations))

    algorithm.run()
    front = algorithm.get_result()

    # Plot front
    plot_front = Plot(plot_title='NSGAIII-DTLZ1', axis_labels=problem.obj_labels, reference_front=problem.reference_front)
    plot_front.plot([algorithm.get_result()], labels=['Pareto front aprox.'], filename='NSGAIII-DTLZ1')

    # Save results to file
    print_function_values_to_file(front, 'FUN.NSGAIII.DTLZ1')
    print_variables_to_file(front, 'VAR.NSGAIII.DTLZ1')
