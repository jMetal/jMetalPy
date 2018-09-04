import logging
import os

from tqdm import tqdm

from jmetal.util.graphic import ScatterStreaming
from jmetal.core.observable import Observer
from jmetal.util.solution_list_output import SolutionList

jMetalPyLogger = logging.getLogger('jMetalPy')

"""
.. module:: observer
   :platform: Unix, Windows
   :synopsis: Implementation of algorithm's observers.

.. moduleauthor:: Antonio J. Nebro <antonio@lcc.uma.es>
"""


class ProgressBarObserver(Observer):

    def __init__(self, step: int, maximum: int, desc: str= 'Progress') -> None:
        """ Show a smart progress meter with the number of evaluations and computing time.

        :param step: Initial counter value.
        :param maximum: Number of expected iterations.
        :param desc: Prefix for the progressbar. """
        self.progress_bar = tqdm(total=maximum, initial=step, ascii=True, desc=desc)
        self.progress = step
        self.step = step
        self.maxx = maximum

    def update(self, *args, **kwargs):
        self.progress_bar.update(self.step)
        self.progress += self.step

        if self.progress >= self.maxx:
            self.progress_bar.close()


class BasicAlgorithmObserver(Observer):

    def __init__(self, frequency: float = 1.0) -> None:
        """ Show the number of evaluations, best fitness and computing time.

        :param frequency: Display frequency. """
        self.display_frequency = frequency

    def update(self, *args, **kwargs):
        computing_time = kwargs['computing time']
        evaluations = kwargs['evaluations']
        front = kwargs['population']

        if (evaluations % self.display_frequency) == 0:
            jMetalPyLogger.debug(
                'Evaluations: {0} \n Best fitness: {1} \n Computing time: {2}'.format(
                    evaluations, front[0].objectives, computing_time
                )
            )


class WriteFrontToFileObserver(Observer):

    def __init__(self, output_directory) -> None:
        """ Write function values of the front into files.

        :param output_directory: Output directory. Each front will be saved on a file `FUN.x`. """
        self.counter = 0
        self.directory = output_directory

        if os.path.isdir(self.directory):
            jMetalPyLogger.warning('Directory {} exists. Removing contents.'.format(self.directory))
            for file in os.listdir(self.directory):
                os.remove('{0}/{1}'.format(self.directory, file))
        else:
            jMetalPyLogger.warning('Directory {} does not exist. Creating it.'.format(self.directory))
            os.mkdir(self.directory)

    def update(self, *args, **kwargs):
        front = kwargs['population']

        SolutionList.print_function_values_to_file(front, '{0}/FUN.{1}'.format(self.directory, self.counter))
        self.counter += 1


class VisualizerObserver(Observer):

    def __init__(self, replace: bool=True) -> None:
        self.display_frequency = 1.0
        self.replace = replace
        self.plot = ScatterStreaming(plot_title='jMetalPy')

    def update(self, *args, **kwargs):
        computing_time = kwargs['computing time']
        evaluations = kwargs['evaluations']

        front = kwargs['population']
        reference_front = kwargs['reference_front']

        title = '{0}, Eval: {1}, Time: {2}'.format('VisualizerObserver', evaluations, computing_time)

        if (evaluations % self.display_frequency) == 0:
            self.plot.update(front, reference_front, rename_title=title, persistence=self.replace)
