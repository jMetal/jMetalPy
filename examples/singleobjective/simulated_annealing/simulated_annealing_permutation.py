from jmetal.algorithm.singleobjective.simulated_annealing import SimulatedAnnealing
from jmetal.operator import ScrambleMutation
from jmetal.problem import TSP
from jmetal.util.termination_criterion import StoppingByTime

if __name__ == "__main__":
    problem = TSP(instance="resources/TSP_instances/kroA100.tsp")

    print(f"Solving TSP problem with {problem.number_of_cities} cities.")

    algorithm = SimulatedAnnealing(
        problem=problem,
        mutation=ScrambleMutation(probability=1.0 / problem.number_of_cities),
        termination_criterion=StoppingByTime(max_seconds=10),
    )

    algorithm.run()
    result = algorithm.result()

    print(f"Algorithm: {algorithm.get_name()}")
    print(f"Solution: {result.variables}")
    print(f"The shortest path length:  {str(result.objectives[0])}")
    print(f"Computing time: {str(algorithm.total_computing_time)}")
    print(f"Problem evaluations: {str(algorithm.evaluations)}")
