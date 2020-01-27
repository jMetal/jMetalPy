import logging
import os
from pathlib import Path
from typing import List, TypeVar

from tqdm import tqdm

from jmetal.core.observer import Observer
from jmetal.core.problem import DynamicProblem
from jmetal.core.quality_indicator import InvertedGenerationalDistance
from jmetal.lab.visualization import StreamingPlot, Plot
from jmetal.util.solution import print_function_values_to_file

S = TypeVar('S')

LOGGER = logging.getLogger('jmetal')

"""
.. module:: observer
   :platform: Unix, Windows
   :synopsis: Implementation of algorithm's observers.

.. moduleauthor:: Antonio J. Nebro <antonio@lcc.uma.es>
"""


class ProgressBarObserver(Observer):

    def __init__(self, max: int) -> None:
        """ Show a smart progress meter with the number of evaluations and computing time.

        :param max: Number of expected iterations.
        """
        self.progress_bar = None
        self.progress = 0
        self._max = max

    def update(self, *args, **kwargs):
        if not self.progress_bar:
            self.progress_bar = tqdm(total=self._max, ascii=True, desc='Progress')

        evaluations = kwargs['EVALUATIONS']

        self.progress_bar.update(evaluations - self.progress)
        self.progress = evaluations

        if self.progress >= self._max:
            self.progress_bar.close()


class BasicObserver(Observer):

    def __init__(self, frequency: float = 1.0) -> None:
        """ Show the number of evaluations, best fitness and computing time.

        :param frequency: Display frequency. """
        self.display_frequency = frequency

    def update(self, *args, **kwargs):
        computing_time = kwargs['COMPUTING_TIME']
        evaluations = kwargs['EVALUATIONS']
        solutions = kwargs['SOLUTIONS']

        if (evaluations % self.display_frequency) == 0 and solutions:
            if type(solutions) == list:
                fitness = solutions[0].objectives
            else:
                fitness = solutions.objectives

            LOGGER.info(
                'Evaluations: {} \n Best fitness: {} \n Computing time: {}'.format(
                    evaluations, fitness, computing_time
                )
            )


class PrintObjectivesObserver(Observer):

    def __init__(self, frequency: float = 1.0) -> None:
        """ Show the number of evaluations, best fitness and computing time.

        :param frequency: Display frequency. """
        self.display_frequency = frequency

    def update(self, *args, **kwargs):
        evaluations = kwargs['EVALUATIONS']
        solutions = kwargs['SOLUTIONS']

        if (evaluations % self.display_frequency) == 0 and solutions:
            if type(solutions) == list:
                fitness = solutions[0].objectives
            else:
                fitness = solutions.objectives

            LOGGER.info(
                'Evaluations: {}. fitness: {}'.format(
                    evaluations, fitness
                )
            )


class WriteFrontToFileObserver(Observer):

    def __init__(self, output_directory: str) -> None:
        """ Write function values of the front into files.

        :param output_directory: Output directory. Each front will be saved on a file `FUN.x`. """
        self.counter = 0
        self.directory = output_directory

        if Path(self.directory).is_dir():
            LOGGER.warning('Directory {} exists. Removing contents.'.format(self.directory))
            for file in os.listdir(self.directory):
                os.remove('{0}/{1}'.format(self.directory, file))
        else:
            LOGGER.warning('Directory {} does not exist. Creating it.'.format(self.directory))
            Path(self.directory).mkdir(parents=True)

    def update(self, *args, **kwargs):
        problem = kwargs['PROBLEM']
        solutions = kwargs['SOLUTIONS']

        if solutions:
            if isinstance(problem, DynamicProblem):
                termination_criterion_is_met = kwargs.get('TERMINATION_CRITERIA_IS_MET', None)

                if termination_criterion_is_met:
                    print_function_values_to_file(solutions, '{}/FUN.{}'.format(self.directory, self.counter))
                    self.counter += 1
            else:
                print_function_values_to_file(solutions, '{}/FUN.{}'.format(self.directory, self.counter))
                self.counter += 1


class PlotFrontToFileObserver(Observer):

    def __init__(self, output_directory: str, step: int = 100, **kwargs) -> None:
        """ Plot and save Pareto front approximations into files.

        :param output_directory: Output directory.
        """
        self.directory = output_directory
        self.plot_front = Plot(title='Pareto front approximation', **kwargs)
        self.last_front = []
        self.fronts = []
        self.counter = 0
        self.step = step

        if Path(self.directory).is_dir():
            LOGGER.warning('Directory {} exists. Removing contents.'.format(self.directory))
            for file in os.listdir(self.directory):
                os.remove('{0}/{1}'.format(self.directory, file))
        else:
            LOGGER.warning('Directory {} does not exist. Creating it.'.format(self.directory))
            Path(self.directory).mkdir(parents=True)

    def update(self, *args, **kwargs):
        problem = kwargs['PROBLEM']
        solutions = kwargs['SOLUTIONS']
        evaluations = kwargs['EVALUATIONS']

        if solutions:
            if (evaluations % self.step) == 0:
                if isinstance(problem, DynamicProblem):
                    termination_criterion_is_met = kwargs.get('TERMINATION_CRITERIA_IS_MET', None)

                    if termination_criterion_is_met:
                        if self.counter > 0:
                            igd = InvertedGenerationalDistance(self.last_front)
                            igd_value = igd.compute(solutions)
                        else:
                            igd_value = 1

                        if igd_value > 0.005:
                            self.fronts += solutions
                            self.plot_front.plot([self.fronts],
                                                 label=problem.get_name(),
                                                 filename=f'{self.directory}/front-{evaluations}')
                        self.counter += 1
                        self.last_front = solutions
                else:
                    self.plot_front.plot([solutions],
                                         label=f'{evaluations} evaluations',
                                         filename=f'{self.directory}/front-{evaluations}')
                    self.counter += 1


class VisualizerObserver(Observer):

    def __init__(self,
                 reference_front: List[S] = None,
                 reference_point: list = None,
                 display_frequency: int = 1) -> None:
        self.figure = None
        self.display_frequency = display_frequency

        self.reference_point = reference_point
        self.reference_front = reference_front

    def update(self, *args, **kwargs):
        evaluations = kwargs['EVALUATIONS']
        solutions = kwargs['SOLUTIONS']

        if solutions:
            if self.figure is None:
                self.figure = StreamingPlot(reference_point=self.reference_point,
                                            reference_front=self.reference_front)
                self.figure.plot(solutions)

            if (evaluations % self.display_frequency) == 0:
                # check if reference point has changed
                reference_point = kwargs.get('REFERENCE_POINT', None)

                if reference_point:
                    self.reference_point = reference_point
                    self.figure.update(solutions, reference_point)
                else:
                    self.figure.update(solutions)

                self.figure.ax.set_title('Eval: {}'.format(evaluations), fontsize=13)
