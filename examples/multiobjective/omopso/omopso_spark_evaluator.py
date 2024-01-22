from jmetal.algorithm.multiobjective.omopso import OMOPSO
from jmetal.operator import UniformMutation
from jmetal.operator.mutation import NonUniformMutation
from jmetal.problem.multiobjective.zdt import ZDT1Modified
from jmetal.util.archive import CrowdingDistanceArchive
from jmetal.util.evaluator import SparkEvaluator
from jmetal.util.solution import (
    print_function_values_to_file,
    print_variables_to_file,
    read_solutions,
)
from jmetal.util.termination_criterion import StoppingByEvaluations

if __name__ == "__main__":
    problem = ZDT1Modified()
    problem.reference_front = read_solutions(filename="resources/reference_front/ZDT1.pf")
    mutation_probability = 1.0 / problem.number_of_variables()

    max_evaluations = 100
    swarm_size = 10
    algorithm = OMOPSO(
        problem=problem,
        swarm_size=swarm_size,
        epsilon=0.0075,
        uniform_mutation=UniformMutation(probability=mutation_probability, perturbation=0.5),
<<<<<<< HEAD
        non_uniform_mutation=NonUniformMutation(
            mutation_probability, perturbation=0.5, max_iterations=max_evaluations / swarm_size
        ),
=======
        non_uniform_mutation=NonUniformMutation(mutation_probability, perturbation=0.5,
                                                max_iterations=max_evaluations / swarm_size),
>>>>>>> 8c0a6cf (Feature/mixed solution (#73))
        leaders=CrowdingDistanceArchive(10),
        termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations),
        swarm_evaluator=SparkEvaluator(),
    )

    algorithm.run()
    front = algorithm.get_result()

    # Save results to file
<<<<<<< HEAD
    print_function_values_to_file(front, "FUN." + algorithm.get_name() + "." + problem.name())
    print_variables_to_file(front, "VAR." + algorithm.get_name() + "." + problem.name())
=======
    print_function_values_to_file(front, 'FUN.' + algorithm.get_name() + "." + problem.get_name())
    print_variables_to_file(front, 'VAR.' + algorithm.get_name() + "." + problem.get_name())
>>>>>>> 8c0a6cf (Feature/mixed solution (#73))

    print(f"Algorithm: {algorithm.get_name()}")
    print(f"Problem: {problem.name()}")
    print(f"Computing time: {algorithm.total_computing_time}")
