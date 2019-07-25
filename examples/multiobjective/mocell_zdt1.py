from jmetal.algorithm.multiobjective.mocell import MOCell
from jmetal.lab.visualization import Plot
from jmetal.operator import SBXCrossover, PolynomialMutation
from jmetal.problem import ZDT4
from jmetal.util.archive import CrowdingDistanceArchive
from jmetal.util.neighborhood import C9
from jmetal.util.observer import ProgressBarObserver
from jmetal.util.solutions import read_solutions
from jmetal.util.termination_criterion import StoppingByEvaluations

if __name__ == '__main__':
    problem = ZDT4()
    problem.reference_front = read_solutions(filename='../../resources/reference_front/ZDT4.pf')

    max_evaluations = 25000
    algorithm = MOCell(
        problem=problem,
        population_size=100,
        neighborhood=C9(10, 10),
        archive=CrowdingDistanceArchive(100),
        mutation=PolynomialMutation(probability=1.0 / problem.number_of_variables, distribution_index=20),
        crossover=SBXCrossover(probability=1.0, distribution_index=20),
        termination_criterion=StoppingByEvaluations(max=max_evaluations)
    )

    algorithm.observable.register(observer=ProgressBarObserver(max=max_evaluations))

    algorithm.run()
    front = algorithm.get_result()

    # Plot front
    plot_front = Plot(plot_title='Pareto front approximation', reference_front=problem.reference_front, axis_labels=problem.obj_labels)
    plot_front.plot(front, label=algorithm.label )

    print('Algorithm (continuous problem): ' + algorithm.get_name())
    print('Problem: ' + problem.get_name())
    print('Computing time: ' + str(algorithm.total_computing_time))
