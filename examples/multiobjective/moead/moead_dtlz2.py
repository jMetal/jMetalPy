from jmetal.algorithm.multiobjective.moead import MOEAD
from jmetal.core.quality_indicator import HyperVolume, InvertedGenerationalDistance
from jmetal.operator import PolynomialMutation, DifferentialEvolutionCrossover
from jmetal.problem import DTLZ2
from jmetal.util.aggregative_function import Tschebycheff
from jmetal.util.solution import read_solutions, print_function_values_to_file, print_variables_to_file
from jmetal.util.termination_criterion import StoppingByEvaluations

if __name__ == '__main__':
    problem = DTLZ2()
    problem.reference_front = read_solutions(filename='resources/reference_front/DTLZ2.3D.pf')

    max_evaluations = 50000

    algorithm = MOEAD(
        problem=problem,
        population_size=300,
        crossover=DifferentialEvolutionCrossover(CR=1.0, F=0.5),
        mutation=PolynomialMutation(probability=1.0 / problem.number_of_variables, distribution_index=20),
        aggregative_function=Tschebycheff(dimension=problem.number_of_objectives),
        neighbor_size=20,
        neighbourhood_selection_probability=0.9,
        max_number_of_replaced_solutions=2,
        weight_files_path='resources/MOEAD_weights',
        termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations)
    )

    algorithm.run()
    front = algorithm.get_result()

    hypervolume = HyperVolume([1.0, 1.0, 1.0])
    print("Hypervolume: " + str(hypervolume.compute([front[i].objectives for i in range(len(front))])))

    # Save results to file
    print_function_values_to_file(front, 'FUN.' + algorithm.label)
    print_variables_to_file(front, 'VAR.'+ algorithm.label)

    print(f'Algorithm: ${algorithm.get_name()}')
    print(f'Problem: ${problem.get_name()}')
    print(f'Computing time: ${algorithm.total_computing_time}')
