from jmetal.algorithm.multiobjective.smpso import SMPSO
from jmetal.core.problem import OnTheFlyFloatProblem
from jmetal.operator import PolynomialMutation
from jmetal.util.archive import CrowdingDistanceArchive
from jmetal.util.observer import ProgressBarObserver, VisualizerObserver
from jmetal.util.solutions import print_function_values_to_file, print_variables_to_file
from jmetal.util.termination_criterion import StoppingByEvaluations

if __name__ == '__main__':
    # Defining problem Srinivas on the fly
    def f1(x: [float]):
        return 2.0 + (x[0] - 2.0) * (x[0] - 2.0) + (x[1] - 1.0) * (x[1] - 1.0)


    def f2(x: [float]):
        return 9.0 * x[0] - (x[1] - 1.0) * (x[1] - 1.0)


    def c1(x: [float]):
        return 1.0 - (x[0] * x[0] + x[1] * x[1]) / 225.0


    def c2(x: [float]):
        return (3.0 * x[1] - x[0]) / 10.0 - 1.0


    problem = OnTheFlyFloatProblem() \
        .set_name('Srinivas') \
        .add_variable(-20.0, 20.0) \
        .add_variable(-20.0, 20.0) \
        .add_function(f1) \
        .add_function(f2) \
        .add_constraint(c1) \
        .add_constraint(c2)

    max_evaluations = 25000
    algorithm = SMPSO(
        problem=problem,
        swarm_size=100,
        mutation=PolynomialMutation(probability=1.0 / problem.number_of_variables, distribution_index=20),
        leaders=CrowdingDistanceArchive(100),
        termination_criterion=StoppingByEvaluations(max=max_evaluations)
    )

    algorithm.observable.register(observer=ProgressBarObserver(max=max_evaluations))
    algorithm.observable.register(observer=VisualizerObserver())

    algorithm.run()
    front = algorithm.get_result()

    # Save results to file
    print_function_values_to_file(front, 'FUN.' + algorithm.get_name() + "." + problem.get_name())
    print_variables_to_file(front, 'VAR.' + algorithm.get_name() + "." + problem.get_name())

    print('Algorithm (continuous problem): ' + algorithm.get_name())
    print('Problem: ' + problem.get_name())
    print('Computing time: ' + str(algorithm.total_computing_time))
