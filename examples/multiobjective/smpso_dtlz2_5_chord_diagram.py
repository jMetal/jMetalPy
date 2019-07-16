from jmetal.algorithm.multiobjective.smpso import SMPSO
from jmetal.lab.visualization import InteractivePlot, Plot
from jmetal.operator import PolynomialMutation
from jmetal.problem import DTLZ2
from jmetal.util.archive import CrowdingDistanceArchive
from jmetal.util.observer import ProgressBarObserver
from jmetal.util.termination_criterion import StoppingByEvaluations
from jmetal.lab.visualization.chord_plot import chord_diagram

if __name__ == '__main__':
    problem = DTLZ2(number_of_objectives=5)

    algorithm = SMPSO(
        problem=problem,
        swarm_size=100,
        mutation=PolynomialMutation(probability=1.0 / problem.number_of_variables, distribution_index=20),
        leaders=CrowdingDistanceArchive(100),
        termination_criterion=StoppingByEvaluations(max=25000)
    )

    algorithm.observable.register(observer=ProgressBarObserver(max=25000))

    algorithm.run()
    front = algorithm.get_result()

    # Plot front
    plot_front = Plot(plot_title='Pareto front approximation', reference_front=problem.reference_front, axis_labels=problem.obj_labels)
    plot_front.plot(front, label=algorithm.label, filename=algorithm.get_name())

    # Plot interactive front
    plot_front = InteractivePlot(plot_title='Pareto front approximation', reference_front=problem.reference_front, axis_labels=problem.obj_labels)
    plot_front.plot(front, label=algorithm.label, filename=algorithm.get_name())

    # Chord interactive plot
    chord_diagram(front, nbins='auto')
    print('Hover mouse over the white patches to depict samples as chords among objectives')

    print('Algorithm (continuous problem): ' + algorithm.get_name())
    print('Problem: ' + problem.get_name())
    print('Computing time: ' + str(algorithm.total_computing_time))
