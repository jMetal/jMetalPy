from jmetal.algorithm.multiobjective.gde3 import DynamicGDE3
from jmetal.component import VisualizerObserver
from jmetal.component.observable import TimeCounter
from jmetal.problem.multiobjective.fda import FDA2
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
        termination_criterion=StoppingByEvaluations(max=500)
    )

    algorithm.observable.register(observer=VisualizerObserver(reference_front=problem.reference_front))

    algorithm.run()
