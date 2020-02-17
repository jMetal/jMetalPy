from jmetal.algorithm.multiobjective.smpso import DynamicSMPSO
from jmetal.operator import PolynomialMutation
from jmetal.problem.multiobjective.fda import FDA2
from jmetal.util.archive import CrowdingDistanceArchive
from jmetal.util.observable import TimeCounter
<<<<<<< HEAD
from jmetal.util.observer import PlotFrontToFileObserver, WriteFrontToFileObserver
=======

>>>>>>> master
from jmetal.util.termination_criterion import StoppingByEvaluations

if __name__ == '__main__':
    problem = FDA2()

    time_counter = TimeCounter(delay=15)
    time_counter.observable.register(problem)
    time_counter.start()

    max_evaluations = 25000
    algorithm = DynamicSMPSO(
        problem=problem,
        swarm_size=100,
        mutation=PolynomialMutation(probability=1.0 / problem.number_of_variables, distribution_index=20),
        leaders=CrowdingDistanceArchive(100),
        termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations)
    )

    algorithm.observable.register(observer=PlotFrontToFileObserver('dynamic_front_vis'))
    algorithm.observable.register(observer=WriteFrontToFileObserver('dynamic_front'))

    algorithm.run()
