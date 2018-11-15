from jmetal.algorithm import MOEAD
from jmetal.problem import ZDT1
from jmetal.operator import Polynomial, DifferentialEvolution
from jmetal.util import FrontPlot

if __name__ == '__main__':
    problem = ZDT1()

    algorithm = MOEAD(
        problem=problem,
        population_size=100,
        max_evaluations=50000,
        delta=0.9,
        nr=2,
        neighbourhood_size=20,
        function_type=MOEAD.AGG,
        crossover=DifferentialEvolution(probability=0.0, distribution_index=20, neighbour_index=2, CR=2, F=1, gamma=1),
        mutation=Polynomial(probability=1.0 / problem.number_of_variables, distribution_index=20),
    )

    algorithm.run()
    front = algorithm.get_result()

    # Plot frontier to file
    pareto_front = FrontPlot(plot_title='MOEAD-ZDT1', axis_labels=problem.obj_labels)
    pareto_front.plot(front, reference_front=problem.reference_front)
    pareto_front.to_html(filename='MOEAD-ZDT1')