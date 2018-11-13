from jmetal.algorithm.singleobjective.evolutionaryalgorithm import GenerationalGeneticAlgorithm
from jmetal.core.solution import FloatSolution
from jmetal.operator.crossover import SBX
from jmetal.operator.mutation import Polynomial
from jmetal.operator.selection import BinaryTournamentSelection
from jmetal.problem.singleobjective.unconstrained import Sphere


def main() -> None:
    variables = 10
    problem = Sphere(variables)

    algorithm = GenerationalGeneticAlgorithm[FloatSolution, FloatSolution](
        problem=problem,
        population_size=100,
        max_evaluations=25000,
        mutation=Polynomial(probability=1.0/variables, distribution_index=20),
        crossover=SBX(probability=1.0, distribution_index=20),
        selection=BinaryTournamentSelection()
    )

    algorithm.start()
    print('Algorithm (running as a thread): ' + algorithm.get_name())
    print('Problem: ' + problem.get_name())
    algorithm.join()

    result = algorithm.get_result()

    print('Solution: ' + str(result.variables))
    print('Fitness:  ' + str(result.objectives[0]))
    print('Computing time: ' + str(algorithm.total_computing_time))


if __name__ == '__main__':
    main()
