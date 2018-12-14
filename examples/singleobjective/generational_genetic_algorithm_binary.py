from jmetal.algorithm.singleobjective.genetic_algorithm import GeneticAlgorithm
from jmetal.operator import SPX, BitFlip, BinaryTournamentSelection
from jmetal.problem import OneMax
from jmetal.util.termination_criteria import StoppingByEvaluations

if __name__ == '__main__':
    problem = OneMax(number_of_bits=256)

    algorithm = GeneticAlgorithm(
        problem=problem,
        population_size=100,
        mating_pool_size=100,
        offspring_size=100,
        mutation=BitFlip(1.0 / problem.number_of_bits),
        crossover=SPX(0.9),
        selection=BinaryTournamentSelection(),
        termination_criteria=StoppingByEvaluations(max=25000)
    )

    algorithm.run()
    result = algorithm.get_result()

    print('Algorithm: {}'.format(algorithm.get_name()))
    print('Problem: {}'.format(problem.get_name()))
    print('Solution: {}'.format(result.variables))
    print('Fitness: {}'.format(result.objectives[0]))
    print('Computing time: {}'.format(algorithm.total_computing_time))
