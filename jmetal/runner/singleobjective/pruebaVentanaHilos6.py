from time import sleep

import numpy as np
import pyqtgraph as pg
import threading

from PyQt5.QtCore import QThread

from jmetal.algorithm.singleobjective.evolutionaryalgorithm import GenerationalGeneticAlgorithm
from jmetal.core.solution import FloatSolution
from jmetal.operator.crossover import SBX
from jmetal.operator.mutation import Polynomial
from jmetal.operator.selection import BinaryTournament
from jmetal.problem.singleobjectiveproblem import Sphere
from jmetal.util.observable import Observer

# FUNCIONA!!!


class FuncThread(threading.Thread):
    def __init__(self, t, *a):
        self._t = t
        self._a = a
        threading.Thread.__init__(self)

    def run(self):
        self._t(*self._a)


class AlgorithmObserver(Observer):

    def __init__(self) -> None:
        super().__init__()
        self.number_of_data_steps = 100000
        self.qapp = None
        self.win = None
        self.x = np.zeros(9000)
        self.y = np.zeros(9000)
        self.ptr = 0

        print("Hilo AlgorithmObserver", threading.get_ident())

        # Hilo para pintar la ventana
        calculating_thread = FuncThread(self.printWindow)
        calculating_thread.start()

        # importante para que comience el hilo de mostrar ventana antes, de esta forma se inicializan las
        # los atributos necesarios antes de que se vaya lanzando update(), ya que en el se usa self.win y self.y
        sleep(1)

    def printWindow(self):
        print("Hilo printWindow", threading.get_ident())
        self.qapp = pg.mkQApp()
        self.win = pg.GraphicsWindow()  # raise window!
        plot1 = self.win.addPlot()
        plot1.setDownsampling(mode='peak')
        plot1.setClipToView(True)
        curve = plot1.plot()
        curve.setPen('r')  # color del grafico

        while self.win.isVisible():
            # refresh data
            # curve.setData(self.x, self.y)
            curve.setData(self.x[:self.ptr], self.y[:self.ptr])
            self.qapp.processEvents()

    def update(self, *args, **kwargs):
        print("Evaluations: " + str(kwargs["evaluations"]) +
              ". Best fitness: " + str(kwargs["best"].objectives[0]) +
              ". Computing time: " + str(kwargs["computing time"]))

        if self.win.isVisible():
            self.x[self.ptr] = kwargs["evaluations"]
            self.y[self.ptr] = kwargs["best"].objectives[0]
            self.ptr += 1


class MyApp():

    def __init__(self):
        print("Hilo init MyApp", threading.get_ident())
        # Algoritmo
        variables = 10
        problem = Sphere(variables)
        algorithm = GenerationalGeneticAlgorithm[FloatSolution, FloatSolution](
            problem,
            population_size=100,
            max_evaluations=25000,
            mutation=Polynomial(1.0 / variables, distribution_index=20),
            crossover=SBX(1.0, distribution_index=20),
            selection=BinaryTournament())
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
    m = MyApp()
