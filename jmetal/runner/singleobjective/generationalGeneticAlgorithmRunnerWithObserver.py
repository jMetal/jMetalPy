from jmetal.algorithm.singleobjective.evolutionaryalgorithm import GenerationalGeneticAlgorithm
from jmetal.core.solution import FloatSolution
from jmetal.operator.crossover import SBX
from jmetal.operator.mutation import Polynomial
from jmetal.operator.selection import BinaryTournament
from jmetal.problem.singleobjectiveproblem import Sphere
from jmetal.util.observable import Observer


import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
import numpy as np




def main():
    float_example()


class AlgorithmObserver(Observer):

    def __init__(self) -> None:
        # Configure graphic window
        win = pg.GraphicsWindow()
        win.setWindowTitle('Algorithm observer')

        # Load char
        p1 = win.addPlot()
        p2 = win.addPlot()

        # Load data
        self.data1 = np.random.normal(size=300)
        self.curve1 = p1.plot(self.data1)
        self.curve2 = p2.plot(self.data1)
        self.ptr1 = 0

    def update(self, *args, **kwargs):
        print("Evaluations: " + str(kwargs["evaluations"]) +
              ". Best fitness: " + str(kwargs["best"].objectives[0]) +
              ". Computing time: " + str(kwargs["computing time"]))
        self.data1[:-1] = self[1:]  # shift data in the array one sample left
        # (see also: np.roll)
        self.data1[-1] = np.random.normal()
        self.curve1.setData(self.data1)
        self.ptr1 += 1
        self.curve2.setData(self.data1)
        self.curve2.setPos(self.ptr1, 0)


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
