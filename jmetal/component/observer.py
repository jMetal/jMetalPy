import logging
import os
from pathlib import Path

from tqdm import tqdm

from jmetal.core.observable import Observer
from jmetal.util.graphic import StreamingPlot
from jmetal.util.solution_list import print_function_values_to_file

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
        :param desc: Prefix for the progressbar.
        """
        self.progress_bar = None
        self.progress = 0
        self.maxx = max

    def update(self, *args, **kwargs):
        if not self.progress_bar:
            self.progress_bar = tqdm(total=self.maxx, ascii=True, desc='Progress')

        evaluations = kwargs['EVALUATIONS']

        self.progress_bar.update(evaluations - self.progress)
        self.progress = evaluations

        if self.progress >= self.maxx:
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
            LOGGER.debug(
                'Evaluations: {} \n Best fitness: {} \n Computing time: {}'.format(
                    evaluations, solutions[0].objectives, computing_time
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
        solutions = kwargs['SOLUTIONS']

        if solutions:
            print_function_values_to_file(solutions, '{}/FUN.{}'.format(self.directory, self.counter))
            self.counter += 1


class VisualizerObserver(Observer):

    def __init__(self, display_frequency: float = 1.0) -> None:
        self.figure = StreamingPlot()
        self.display_frequency = display_frequency

    def update(self, *args, **kwargs):
        evaluations = kwargs['EVALUATIONS']
        solutions = kwargs['SOLUTIONS']
        problem = kwargs['PROBLEM']

        if (evaluations % self.display_frequency) == 0 and solutions:
            self.figure.update(solutions, problem.reference_front)
