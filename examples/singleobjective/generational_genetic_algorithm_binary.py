from jmetal.algorithm.singleobjective.genetic_algorithm import GeneticAlgorithm
from jmetal.operator import SPXCrossover, BitFlipMutation, BinaryTournamentSelection
from jmetal.problem import OneMax
from jmetal.util.observer import ObjectivesObserver
from jmetal.util.termination_criterion import StoppingByEvaluations

if __name__ == '__main__':
    problem = OneMax(number_of_bits=512)

    algorithm = GeneticAlgorithm(
        problem=problem,
        population_size=100,
        offspring_population_size=100,
        mutation=BitFlipMutation(1.0 / problem.number_of_bits),
        crossover=SPXCrossover(1.0),
        selection=BinaryTournamentSelection(),
        termination_criterion=StoppingByEvaluations(max=20000)
    )

    algorithm.observable.register(observer=ObjectivesObserver())

    algorithm.run()
    result = algorithm.get_result()

    print('Algorithm: {}'.format(algorithm.get_name()))
    print('Problem: {}'.format(problem.get_name()))
    print('Solution: {}'.format(result.variables))
    print('Fitness: {}'.format(result.objectives[0]))
    print('Computing time: {}'.format(algorithm.total_computing_time))
