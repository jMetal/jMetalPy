from jmetal.core.problem import Problem

from jmetal.component import ProgressBarObserver

from jmetal.algorithm import MOEAD
from jmetal.problem import LZ09_F2
from jmetal.operator import Polynomial, DifferentialEvolution
from jmetal.util import FrontPlot

if __name__ == '__main__':
    reference_front = Problem.read_front_from_file_as_solutions('../../resources/reference_front/LZ09_F2.pf')
    problem = LZ09_F2(reference_front=reference_front)

    algorithm = MOEAD(
        problem=problem,
        population_size=300,
        output_population_size=300,
        max_evaluations=150000,
        neighbourhood_size=20,
        neighbourhood_selection_probability=0.9,
        max_number_of_replaced_solutions=2,
        ffunction_type=MOEAD.FitnessFunction.AGG,
        weights_path='../../resources/MOEAD_weights/',
        crossover=DifferentialEvolution(CR=1.0, F=0.5, K=0.5),
        mutation=Polynomial(probability=1.0 / problem.number_of_variables, distribution_index=20),
    )

    progress_bar = ProgressBarObserver(step=algorithm.population_size, maximum=algorithm.max_evaluations)
    algorithm.observable.register(observer=progress_bar)

    algorithm.run()
    front = algorithm.get_result()

    # Plot frontier to file
    pareto_front = FrontPlot(plot_title='MOEAD-LZ09_F2', axis_labels=problem.obj_labels)
    pareto_front.plot(front, reference_front=problem.reference_front)
    pareto_front.to_html(filename='MOEAD-LZ09_F2')
