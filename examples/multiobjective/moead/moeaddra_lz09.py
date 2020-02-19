from jmetal.algorithm.multiobjective.moead import MOEAD_DRA
from jmetal.operator import PolynomialMutation, DifferentialEvolutionCrossover
from jmetal.problem import LZ09_F2
from jmetal.problem.multiobjective.uf import UF1
from jmetal.util.aggregative_function import Tschebycheff
from jmetal.util.solution import read_solutions, print_function_values_to_file, print_variables_to_file
from jmetal.util.termination_criterion import StoppingByEvaluations
from jmetal.core.quality_indicator import HyperVolume, InvertedGenerationalDistance

if __name__ == '__main__':
    problem = UF1()
    problem.reference_front = read_solutions(filename='resources/reference_front/UF1.pf')

    max_evaluations = 300000

    algorithm = MOEAD_DRA(
        problem=problem,
        population_size=600,
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

    hypervolume = HyperVolume([2.0, 2.0])
    print("Hypervolume: " + str(hypervolume.compute([front[i].objectives for i in range(len(front))])))

    # Save results to file
    print_function_values_to_file(front, 'FUN.' + algorithm.label)
    print_variables_to_file(front, 'VAR.'+ algorithm.label)

    print(f'Algorithm: ${algorithm.get_name()}')
    print(f'Problem: ${problem.get_name()}')
    print(f'Computing time: ${algorithm.total_computing_time}')


