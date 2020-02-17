from jmetal.algorithm.multiobjective.gde3 import DynamicGDE3
from jmetal.problem.multiobjective.fda import FDA2
from jmetal.util.observable import TimeCounter
from jmetal.util.observer import WriteFrontToFileObserver, PlotFrontToFileObserver
from jmetal.util.termination_criterion import StoppingByEvaluations

if __name__ == '__main__':
    problem = FDA2()

    time_counter = TimeCounter(delay=1)
    time_counter.observable.register(problem)
    time_counter.start()

    algorithm = DynamicGDE3(
        problem=problem,
        population_size=100,
        cr=0.5,
        f=0.5,
        termination_criterion=StoppingByEvaluations(max_evaluations=500)
    )

    algorithm.observable.register(observer=PlotFrontToFileObserver('dynamic_front_vis'))
    algorithm.observable.register(observer=WriteFrontToFileObserver('dynamic_front'))

    algorithm.run()
