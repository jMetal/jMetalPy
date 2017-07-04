#from graphtiny.bs import DataStreamWindowBS, ChartBS
#from graphtiny.domain import DataStreamWindow, Chart

from jmetal.algorithm.singleobjective.evolutionaryalgorithm import GenerationalGeneticAlgorithm
from jmetal.component.consumer import BasicSingleObjectiveAlgorithmConsumer
from jmetal.core.solution import FloatSolution
from jmetal.operator.crossover import SBX
from jmetal.operator.mutation import Polynomial
from jmetal.operator.selection import BinaryTournament
from jmetal.problem.singleobjectiveproblem import Sphere
from jmetal.util.observable import Observer


"""
class ChartAlgorithmObserver(Observer):
    def __init__(self) -> None:
        self.window = DataStreamWindow()
        self.chart = Chart()
        self.chart.left_label = 'Objectives'
        self.chart.bottom_label = 'Evaluations'
        self.window.charts_list.append(self.chart)
        DataStreamWindowBS().launch_window(self.window)

    def update(self, *args, **kwargs):
        ChartBS().set_data_stream(self.chart, kwargs["computing time"], kwargs["best"].objectives[0])

        ChartBS().set_data_stream(self.chart, kwargs["evaluations"], kwargs["best"].objectives[0])
"""
def main() -> None:
    variables = 10
    problem = Sphere(variables)
    algorithm = GenerationalGeneticAlgorithm[FloatSolution, FloatSolution](
        problem,
        population_size=100,
        max_evaluations=25000,
        mutation=Polynomial(1.0 / variables, distribution_index=20),
        crossover=SBX(1.0, distribution_index=20),
        selection=BinaryTournament())

    observer = BasicSingleObjectiveAlgorithmConsumer()

    algorithm.observable.register(observer=observer)

    algorithm.start()
    algorithm.join()

    result = algorithm.get_result()
    print("Algorithm: " + algorithm.get_name())
    print("Problem: " + problem.get_name())
    print("Solution: " + str(result.variables))
    print("Fitness:  " + str(result.objectives[0]))


if __name__ == '__main__':
    main()
