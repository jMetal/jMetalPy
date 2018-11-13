from jmetal.algorithm.singleobjective.evolutionaryalgorithm import GenerationalGeneticAlgorithm
from jmetal.core.solution import FloatSolution
from jmetal.operator.crossover import SBX
from jmetal.operator.mutation import Polynomial
from jmetal.operator.selection import BinaryTournamentSelection
from jmetal.problem.singleobjective.unconstrained import Sphere


def main() -> None:
    class GGA2(GenerationalGeneticAlgorithm[FloatSolution, FloatSolution]):
        def is_stopping_condition_reached(self):
            # Re-define the stopping condition
            reached = [False, True][self.get_current_computing_time() > 4]

            if reached:
                print('Stopping condition reached!')

            return reached

    variables = 10
    problem = Sphere(variables)

    algorithm = GGA2(
        problem=problem,
        population_size=100,
        mating_pool_size=100,
        offspring_population_size=100,
        max_evaluations=0,
        mutation=Polynomial(1.0/variables, distribution_index=20),
        crossover=SBX(1.0, distribution_index=20),
        selection=BinaryTournamentSelection()
    )

    algorithm.run()
    result = algorithm.get_result()

    print('Algorithm (stop for timeout): ' + algorithm.get_name())
    print('Problem: ' + problem.get_name())
    print('Solution: ' + str(result.variables))
    print('Fitness:  ' + str(result.objectives[0]))
    print('Computing time: ' + str(algorithm.total_computing_time))


if __name__ == '__main__':
    main()
