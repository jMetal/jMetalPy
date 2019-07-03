from jmetal.algorithm.multiobjective.nsgaii import DynamicNSGAII
<<<<<<< HEAD
from jmetal.operator import PolynomialMutation, SBXCrossover, BinaryTournamentSelection
from jmetal.problem.multiobjective.fda import FDA2
from jmetal.util.comparator import RankingAndCrowdingDistanceComparator
=======
from jmetal.operator import PolynomialMutation, SBXCrossover
from jmetal.problem.multiobjective.fda import FDA2
>>>>>>> 52e0b172f0c6d651ba08b961a90a382f0a4b8e0f
from jmetal.util.observable import TimeCounter
from jmetal.util.observer import PlotFrontToFileObserver, WriteFrontToFileObserver
from jmetal.util.termination_criterion import StoppingByEvaluations

if __name__ == '__main__':
    problem = FDA2()

    time_counter = TimeCounter(delay=1)
    time_counter.observable.register(problem)
    time_counter.start()

    max_evaluations = 25000
    algorithm = DynamicNSGAII(
        problem=problem,
        population_size=100,
        offspring_population_size=100,
        mutation=PolynomialMutation(probability=1.0 / problem.number_of_variables, distribution_index=20),
        crossover=SBXCrossover(probability=1.0, distribution_index=20),
<<<<<<< HEAD
        selection=BinaryTournamentSelection(comparator=RankingAndCrowdingDistanceComparator()),
        termination_criterion=StoppingByEvaluations(max=max_evaluations)
    )

    algorithm.observable.register(observer=PlotFrontToFileObserver('dynamic_front'))
=======
        termination_criterion=StoppingByEvaluations(max=max_evaluations)
    )

    algorithm.observable.register(observer=PlotFrontToFileObserver('dynamic_front_vis'))
>>>>>>> 52e0b172f0c6d651ba08b961a90a382f0a4b8e0f
    algorithm.observable.register(observer=WriteFrontToFileObserver('dynamic_front'))

    algorithm.run()
