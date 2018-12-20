from jmetal.algorithm.multiobjective.smpso import DynamicSMPSO
from jmetal.component import ProgressBarObserver, CrowdingDistanceArchive, VisualizerObserver
from jmetal.component.observable import TimeCounter
from jmetal.core.problem import DynamicProblem
from jmetal.operator import Polynomial
from jmetal.problem.multiobjective.fda import FDA2
from jmetal.util.termination_criterion import StoppingByEvaluations

if __name__ == '__main__':
    problem: DynamicProblem = FDA2()
    time_counter = TimeCounter(delay=1)
    time_counter.observable.register(problem)
    time_counter.start()

    max_evaluations = 25000

    algorithm = DynamicSMPSO(
        problem=problem,
        swarm_size=100,
        mutation=Polynomial(probability=1.0 / problem.number_of_variables, distribution_index=20),
        leaders=CrowdingDistanceArchive(100),
        termination_criterion=StoppingByEvaluations(max=max_evaluations)
    )

    algorithm.run()
