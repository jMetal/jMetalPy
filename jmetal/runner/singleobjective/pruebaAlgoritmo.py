from jmetal.algorithm.singleobjective.evolutionaryalgorithm import GenerationalGeneticAlgorithm
from jmetal.core.solution import FloatSolution
from jmetal.operator.crossover import SBX
from jmetal.operator.mutation import Polynomial
from jmetal.operator.selection import BinaryTournament
from jmetal.problem.singleobjectiveproblem import Sphere
from jmetal.runner.singleobjective.libreriaVentanaGraficos import DataStreamWindow, Chart
from jmetal.util.observable import Observer


def main():
    float_example()


class AlgorithmObserver(Observer):

    def __init__(self) -> None:
        self.window = DataStreamWindow()
        self.window.columns_display = 2
        self.chart1 = Chart()
        self.chart2 = Chart()
        self.chart3 = Chart()
        self.chart4 = Chart()
        self.window.charts_list.append(self.chart1)
        self.window.charts_list.append(self.chart2)
        self.window.charts_list.append(self.chart3)
        self.window.charts_list.append(self.chart4)
        self.window.print_window()

    def update(self, *args, **kwargs):
        print("Evaluations: " + str(kwargs["evaluations"]) +
              ". Best fitness: " + str(kwargs["best"].objectives[0]) +
              ". Computing time: " + str(kwargs["computing time"]))
        self.chart1.set_data_stream(kwargs["evaluations"], kwargs["best"].objectives[0])
        self.chart2.set_data_stream(kwargs["evaluations"], kwargs["best"].objectives[0])
        self.chart3.set_data_stream(kwargs["evaluations"], kwargs["best"].objectives[0])
        self.chart4.set_data_stream(kwargs["evaluations"], kwargs["best"].objectives[0])


def float_example() -> None:
    variables = 10
    problem = Sphere(variables)
    algorithm = GenerationalGeneticAlgorithm[FloatSolution, FloatSolution](
        problem,
        population_size = 100,
        max_evaluations = 25000,
        mutation = Polynomial(1.0/variables, distribution_index=20),
        crossover = SBX(1.0, distribution_index=20),
        selection = BinaryTournament())

    observer = AlgorithmObserver()

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
