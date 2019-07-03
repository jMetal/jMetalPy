from examples.multiobjective.parallel.zdt1_modified import ZDT1Modified
from jmetal.algorithm.multiobjective.nsgaii import NSGAII
<<<<<<< HEAD
from jmetal.operator import SBXCrossover, PolynomialMutation, BinaryTournamentSelection
from jmetal.util.comparator import RankingAndCrowdingDistanceComparator
=======
from jmetal.operator import SBXCrossover, PolynomialMutation
>>>>>>> 52e0b172f0c6d651ba08b961a90a382f0a4b8e0f
from jmetal.util.solution_list import print_function_values_to_file, print_variables_to_file
from jmetal.util.termination_criterion import StoppingByEvaluations

if __name__ == '__main__':
    problem = ZDT1Modified()

    max_evaluations = 100

    algorithm = NSGAII(
        problem=problem,
        population_size=10,
        offspring_population_size=10,
        mutation=PolynomialMutation(probability=1.0 / problem.number_of_variables, distribution_index=20),
        crossover=SBXCrossover(probability=1.0, distribution_index=20),
<<<<<<< HEAD
        selection=BinaryTournamentSelection(comparator=RankingAndCrowdingDistanceComparator()),
=======
>>>>>>> 52e0b172f0c6d651ba08b961a90a382f0a4b8e0f
        termination_criterion=StoppingByEvaluations(max=max_evaluations)
    )

    algorithm.run()
    front = algorithm.get_result()

    # Save results to file
    print_function_values_to_file(front, 'FUN.NSGAII.ZDT1')
    print_variables_to_file(front, 'VAR.NSGAII.ZDT1')

    print('Algorithm (continuous problem): ' + algorithm.get_name())
    print('Problem: ' + problem.get_name())
    print('Computing time: ' + str(algorithm.total_computing_time))
