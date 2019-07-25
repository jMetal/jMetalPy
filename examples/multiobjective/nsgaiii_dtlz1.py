from jmetal.algorithm.multiobjective.nsgaiii import NSGAIII, UniformReferenceDirectionFactory
from jmetal.lab.visualization import Plot
from jmetal.operator import SBXCrossover, PolynomialMutation
from jmetal.problem import DTLZ1
from jmetal.util.observer import ProgressBarObserver
from jmetal.util.solutions import read_solutions
from jmetal.util.termination_criterion import StoppingByEvaluations

if __name__ == '__main__':
    problem = DTLZ1()
    problem.reference_front = read_solutions(filename='../../resources/reference_front/DTLZ1.3D.pf')

    max_evaluations = 25000

    algorithm = NSGAIII(
        problem=problem,
        population_size=92,
        reference_directions=UniformReferenceDirectionFactory(3, n_points=91),
        mutation=PolynomialMutation(probability=1.0 / problem.number_of_variables, distribution_index=20),
        crossover=SBXCrossover(probability=1.0, distribution_index=30),
        termination_criterion=StoppingByEvaluations(max=max_evaluations)
    )

    algorithm.observable.register(observer=ProgressBarObserver(max=max_evaluations))

    algorithm.run()
    front = algorithm.get_result()

    # Plot front
    plot_front = Plot(plot_title='Pareto front approximation', axis_labels=problem.obj_labels)
    plot_front.plot(front, label=algorithm.label)

    print('Algorithm (continuous problem): ' + algorithm.get_name())
    print('Problem: ' + problem.get_name())
    print('Computing time: ' + str(algorithm.total_computing_time))
