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
import sys
import threading





def main():
    float_example()


class AlgorithmObserver(Observer):

    def __init__(self) -> None:

        self.win = None
        self.curve3 = None
        self.curve4 = None
        self.data3 = None
        self.ptr3 = None
        t = threading.Thread(target=self.worker)

    def worker(self) -> None:
        # timer refresh
        # timer = pg.QtCore.QTimer()
        # timer.timeout.connect(self.update())

        # Configure graphic window
        self.win = pg.GraphicsWindow()
        self.win.setWindowTitle('Algorithm observer')

        # Load chart
        #p1 = self.win.addPlot()
        # p2 = self.win.addPlot()
        # p2.setDownsampling(mode='peak')
        # p2.setClipToView(True)
        # .........................


        # Load data
        # self.data1 = np.random.normal(size=300)
        # self.curve1 = p1.plot(self.data1)
        # self.curve2 = p2.plot(self.data1)
        # self.ptr1 = 0
        # .........................

        p3 = self.win.addPlot()
        p4 = self.win.addPlot()
        # Use automatic downsampling and clipping to reduce the drawing load
        p3.setDownsampling(mode='peak')
        p4.setDownsampling(mode='peak')
        p3.setClipToView(True)
        p4.setClipToView(True)
        p3.setRange(xRange=[-100, 0])
        p3.setLimits(xMax=0)
        self.curve3 = p3.plot()
        self.curve4 = p4.plot()
        self.data3 = np.empty(100)  # devuelve ndarray de numeros aletarios de tamano 100
        self.ptr3 = 0

        # timer = pg.QtCore.QTimer()
        # timer.timeout.connect(self.update)
        # timer.start(50)

        if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
            QtGui.QApplication.instance().exec_()

    def update(self, *args, **kwargs):

        print("Evaluations: " + str(kwargs["evaluations"]) +
              ". Best fitness: " + str(kwargs["best"].objectives[0]) +
              ". Computing time: " + str(kwargs["computing time"]))
        # self.data1[:-1] = self.data1[1:]  # shift data in the array one sample left
        # self.data1[-1] = np.random.normal()
        #
        # self.curve1.setData(self.data1)
        # self.ptr1 += 1
        # self.curve2.setData(self.data1)
        # self.curve2.setPos(self.ptr1, 0)
        self.data3[self.ptr3] = np.random.normal() # dibuja muestras aleatorias de una distribución normal (Gaussiana). Devuelve ndarray
        self.ptr3 += 1
        if self.ptr3 >= self.data3.shape[0]:
            tmp = self.data3
            self.data3 = np.empty(self.data3.shape[0] * 2)
            self.data3[:tmp.shape[0]] = tmp
        self.curve3.setData(self.data3[:self.ptr3])
        self.curve3.setPos(-self.ptr3, 0)
        self.curve4.setData(self.data3[:self.ptr3])



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
