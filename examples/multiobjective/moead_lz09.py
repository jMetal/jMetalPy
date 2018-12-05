from jmetal.algorithm.multiobjective.moead import MOEAD
from jmetal.component import ProgressBarObserver, VisualizerObserver
from jmetal.operator import Polynomial, DifferentialEvolution
from jmetal.problem import LZ09_F2
from jmetal.util.aggregative_function import Chebyshev
from jmetal.util.graphic import FrontPlot
from jmetal.util.neighborhood import WeightVectorNeighborhood
from jmetal.util.solution_list import read_front
from jmetal.util.termination_criteria import StoppingByEvaluations

if __name__ == '__main__':
    problem = LZ09_F2()
    problem.reference_front = read_front(file_path='../../resources/reference_front/{}.pf'.format(problem.get_name()))

    population_size = 100

    algorithm = MOEAD(
        problem=problem,
        population_size=population_size,
        crossover=DifferentialEvolution(CR=1.0, F=0.5, K=0.5),
        mutation=Polynomial(probability=1.0 / problem.number_of_variables, distribution_index=20),
        aggregative_function=Chebyshev(dimension=problem.number_of_objectives),
        neighbourhood=WeightVectorNeighborhood(population_size, 20, weights_path='../../resources/MOEAD_weights/'),
        neighbourhood_selection_probability=0.9,
        max_number_of_replaced_solutions=2,
        termination_criteria=StoppingByEvaluations(max=175000)
    )

    progress_bar = ProgressBarObserver(max=175000)
    algorithm.observable.register(observer=progress_bar)
    algorithm.observable.register(observer=VisualizerObserver())
    algorithm.observable.register(observer=VisualizerObserver())

    algorithm.run()
    front = algorithm.get_result()

    # Plot frontier to file
    pareto_front = FrontPlot(plot_title='MOEAD-{}'.format(problem.get_name()), axis_labels=problem.obj_labels)
    pareto_front.plot(front, reference_front=problem.reference_front)
    pareto_front.to_html(filename='MOEAD-{}'.format(problem.get_name()))
