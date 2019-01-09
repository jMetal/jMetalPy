from jmetal.algorithm.multiobjective.moead import MOEAD
from jmetal.operator import Polynomial, DifferentialEvolutionCrossover
from jmetal.problem import LZ09_F2
from jmetal.util.aggregative_function import Chebyshev
from jmetal.util.neighborhood import WeightVectorNeighborhood
from jmetal.util.observer import ProgressBarObserver, VisualizerObserver
from jmetal.util.solution_list import read_solutions
from jmetal.util.termination_criterion import StoppingByEvaluations

if __name__ == '__main__':
    problem = LZ09_F2()
    problem.reference_front = read_solutions(
        file_path='../../resources/reference_front/{}.pf'.format(problem.get_name()))

    population_size = 100
    max_evaluations = 175000

    algorithm = MOEAD(
        problem=problem,
        population_size=population_size,
        crossover=DifferentialEvolutionCrossover(CR=1.0, F=0.5, K=0.5),
        mutation=Polynomial(probability=1.0 / problem.number_of_variables, distribution_index=20),
        aggregative_function=Chebyshev(dimension=problem.number_of_objectives),
        neighbourhood=WeightVectorNeighborhood(population_size, 20, weights_path='../../resources/MOEAD_weights/'),
        neighbourhood_selection_probability=0.9,
        max_number_of_replaced_solutions=2,
        termination_criterion=StoppingByEvaluations(max=max_evaluations)
    )

    algorithm.observable.register(observer=ProgressBarObserver(max=max_evaluations))
    algorithm.observable.register(observer=VisualizerObserver())

    algorithm.run()
    front = algorithm.get_result()
