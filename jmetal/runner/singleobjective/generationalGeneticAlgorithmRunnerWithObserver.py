# from queue import Queue
# from threading import Thread
#
# from jmetal.algorithm.singleobjective.evolutionaryalgorithm import GenerationalGeneticAlgorithm
# from jmetal.core.solution import FloatSolution
# from jmetal.operator.crossover import SBX
# from jmetal.operator.mutation import Polynomial
# from jmetal.operator.selection import BinaryTournament
# from jmetal.problem.singleobjectiveproblem import Sphere
# from jmetal.util.observable import Observer
# import pyqtgraph as pg
# from pyqtgraph.Qt import QtCore, QtGui
# import numpy as np
# import sys
#
#
# def main():
#     float_example()
#
#
# class WinowWorker(Thread):
#     def __init__(self, queue):
#         Thread.__init__(self)
#         self.queue = queue
#
#         self.win = None
#         self.data1 = None
#         self.curve1 = None
#         self.curve2 = None
#         self.ptr1 = None
#
#
#     def run(self):
#         # Configure graphic window
#         self.win = pg.GraphicsWindow()
#         self.win.setWindowTitle('Algorithm observer')
#
#         # Load chart
#         p1 = self.win.addPlot()
#         p2 = self.win.addPlot()
#
#         # Load data
#         self.data1 = np.random.normal(size=300)
#         self.curve1 = p1.plot(self.data1)
#         self.curve2 = p2.plot(self.data1)
#         self.ptr1 = 0
#         if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
#             QtGui.QApplication.instance().exec_()
#         while True:
#             # Get the work from the queue and expand the tuple
#             evaluations, best_fitness, computing_time = self.queue.get()
#             print("PASADOOOOOOOO")
#             self.data1[:-1] = self.data1[1:]  # shift data in the array one sample left
#             self.data1[-1] = np.random.normal()
#
#             self.curve1.setData(self.data1)
#             self.ptr1 += 1
#             self.curve2.setData(self.data1)
#             self.curve2.setPos(self.ptr1, 0)
#
#             self.queue.task_done()
#
#
# class AlgorithmObserver(Observer):
#
#     def __init__(self) -> None:
#         self.queue = Queue()
#         worker = WinowWorker(self.queue)
#         # worker.daemon = True
#         worker.start()
#
#     def update(self, *args, **kwargs):
#         print("Evaluations: " + str(kwargs["evaluations"]) +
#               ". Best fitness: " + str(kwargs["best"].objectives[0]) +
#               ". Computing time: " + str(kwargs["computing time"]))
#         self.queue.put((str(kwargs["evaluations"]), str(kwargs["best"].objectives[0]), str(kwargs["computing time"])))
#
#
# def float_example() -> None:
#     variables = 10
#     problem = Sphere(variables)
#     algorithm = GenerationalGeneticAlgorithm[FloatSolution, FloatSolution](
#         problem,
#         population_size=100,
#         max_evaluations=25000,
#         mutation=Polynomial(1.0 / variables, distribution_index=20),
#         crossover=SBX(1.0, distribution_index=20),
#         selection=BinaryTournament())
#
#     observer = AlgorithmObserver()
#
#     algorithm.observable.register(observer=observer)
#
#     algorithm.start()
#     algorithm.join()
#
#     result = algorithm.get_result()
#     print("Algorithm: " + algorithm.get_name())
#     print("Problem: " + problem.get_name())
#     print("Solution: " + str(result.variables))
#     print("Fitness:  " + str(result.objectives[0]))
#
#
# if __name__ == '__main__':
#     main()
