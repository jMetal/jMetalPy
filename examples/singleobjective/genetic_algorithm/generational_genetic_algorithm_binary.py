from jmetal.algorithm.singleobjective.evolution_strategy import GenerationalGeneticAlgorithm
from jmetal.core.solution import BinarySolution
from jmetal.operator.crossover import SPX
from jmetal.operator.mutation import BitFlip
from jmetal.operator.selection import BinaryTournamentSelection
from jmetal.problem.singleobjective.unconstrained import OneMax


def main() -> None:
    bits = 256
    problem = OneMax(bits)

    algorithm = GenerationalGeneticAlgorithm[BinarySolution, BinarySolution](
        problem=problem,
        population_size=100,
        mating_pool_size=100,
        offspring_population_size=100,
        max_evaluations=150000,
        mutation=BitFlip(1.0/bits),
        crossover=SPX(0.9),
        selection=BinaryTournamentSelection()
    )

    algorithm.run()
    result = algorithm.get_result()

    print('Algorithm: ' + algorithm.get_name())
    print('Problem: ' + problem.get_name())
    print('Solution: ' + str(result.variables))
    print('Fitness: ' + str(result.objectives[0]))
    print('Computing time: ' + str(algorithm.total_computing_time))


if __name__ == '__main__':
    main()
