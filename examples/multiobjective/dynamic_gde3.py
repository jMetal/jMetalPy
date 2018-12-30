from jmetal.algorithm.multiobjective.gde3 import DynamicGDE3
from jmetal.component import RankingAndCrowdingDistanceComparator, ProgressBarObserver

from jmetal.algorithm.multiobjective.nsgaii import DynamicNSGAII
from jmetal.component.observable import TimeCounter
from jmetal.core.problem import DynamicProblem
from jmetal.operator import Polynomial, SBXCrossover, BinaryTournamentSelection
from jmetal.problem.multiobjective.fda import FDA2
from jmetal.util.termination_criterion import StoppingByEvaluations

if __name__ == '__main__':
    problem: DynamicProblem = FDA2()
    time_counter = TimeCounter(delay=1)
    time_counter.observable.register(problem)
    time_counter.start()

    max_evaluations = 25000
    algorithm = DynamicGDE3(
        problem=problem,
        population_size=100,
        cr=0.5,
        f=0.5,
        termination_criterion=StoppingByEvaluations(max=max_evaluations)
    )

    algorithm.run()
