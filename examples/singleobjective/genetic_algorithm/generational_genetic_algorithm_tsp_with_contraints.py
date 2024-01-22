from jmetal.algorithm.singleobjective.genetic_algorithm import GeneticAlgorithm
from jmetal.core.solution import PermutationSolution
from jmetal.operator import BinaryTournamentSelection
from jmetal.operator.crossover import PMXCrossover
from jmetal.operator.mutation import PermutationSwapMutation
from jmetal.problem.singleobjective.tsp import TSP
from jmetal.util.comparator import MultiComparator, OverallConstraintViolationComparator, ObjectiveComparator
from jmetal.util.observer import PrintObjectivesObserver
from jmetal.util.termination_criterion import StoppingByEvaluations

if __name__ == "__main__":
    class TSPWithConstraints(TSP):
        def __init__(self, instance: str):
            super(TSPWithConstraints, self).__init__(instance)

        def number_of_constraints(self) -> int:
            return 1

        def evaluate(self, solution: PermutationSolution) -> PermutationSolution:
            super().evaluate(solution)
            self.__evaluate_constraints__(solution)

            return solution

        def __evaluate_constraints__(self, solution: PermutationSolution):
            """ Constraint: city 17 must be in the first position of the tour
            """
            city = 17
            position = solution.variables.index(city)

            constraint = 0 - position

            solution.constraints = [constraint]

            return solution


    problem = TSPWithConstraints(instance="resources/TSP_instances/kroA100.tsp")

    solution_comparator = MultiComparator([OverallConstraintViolationComparator(), ObjectiveComparator(0)])
    algorithm = GeneticAlgorithm(
        problem=problem,
        population_size=100,
        offspring_population_size=100,
        mutation=PermutationSwapMutation(1.0 / problem.number_of_variables()),
        crossover=PMXCrossover(0.9),
        selection=BinaryTournamentSelection(solution_comparator),
        termination_criterion=StoppingByEvaluations(max_evaluations=10000),
        solution_comparator=solution_comparator
    )

    algorithm.observable.register(observer=PrintObjectivesObserver(1000))

    algorithm.run()
    result = algorithm.result()

    print("Algorithm: {}".format(algorithm.get_name()))
    print("Problem: {}".format(problem.name()))
    print("Solution: {}".format(result.variables))
    print("Fitness: {}".format(result.objectives[0]))
    print("Computing time: {}".format(algorithm.total_computing_time))
