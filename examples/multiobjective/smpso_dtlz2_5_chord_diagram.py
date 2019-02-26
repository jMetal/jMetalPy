from jmetal.algorithm.multiobjective.smpso import SMPSO
from jmetal.operator import PolynomialMutation
from jmetal.problem import DTLZ2
from jmetal.util.archive import CrowdingDistanceArchive
from jmetal.util.observer import ProgressBarObserver
from jmetal.util.termination_criterion import StoppingByEvaluations
from jmetal.util.visualization.chord_plot import chord_diagram

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

    # Chord interactive plot
    chord_diagram(front, nbins='auto')
    print('Hover mouse over the white patches to depict samples as chords among objectives')
