import logging
import os

from jmetal.core.problem import Problem
from jmetal.util.graphic import ScatterBokeh, ScatterMatplotlib
from jmetal.util.observable import Observer
from jmetal.util.solution_list_output import SolutionList

logger = logging.getLogger(__name__)


class BasicAlgorithmObserver(Observer):

    def __init__(self, frequency: float = 1.0) -> None:
        self.display_frequency = frequency

    def update(self, *args, **kwargs):
        evaluations = kwargs["evaluations"]

        if (evaluations % self.display_frequency) == 0:
            logger.debug("Evaluations: " + str(evaluations) +
                         ". Best fitness: " + str(kwargs["population"][0].objectives) +
                         ". Computing time: " + str(kwargs["computing time"]))


class WriteFrontToFileObserver(Observer):

    def __init__(self, output_directory) -> None:
        self.counter = 0
        self.directory = output_directory

        if os.path.isdir(self.directory):
            logger.warning("Directory " + self.directory + " exists. Removing contents.")
            for file in os.listdir(self.directory):
                os.remove(self.directory + "/" + file)
        else:
            logger.warning("Directory " + self.directory + " does not exist. Creating it.")
            os.mkdir(self.directory)

    def update(self, *args, **kwargs):
        SolutionList.print_function_values_to_file(
            kwargs["population"], self.directory + "/FUN." + str(self.counter))

        self.counter += 1


class VisualizerObserver(Observer):

    def __init__(self, problem: Problem, replace: bool=True) -> None:
        self.display_frequency = 1.0
        self.replace = replace
        self.reference = problem.get_reference_front()
        self.plot = ScatterMatplotlib(plot_title='VisualizerObserver', number_of_objectives=problem.number_of_objectives)

    def update(self, *args, **kwargs):
        evaluations = kwargs["evaluations"]
        computing_time = kwargs["computing time"]
        solution_list = kwargs["population"]

        title = '{0}, Eval: {1}, Time: {2}'.format("VisualizerObserver", evaluations, computing_time)

        if (evaluations % self.display_frequency) == 0:
            self.plot.update(solution_list=solution_list, reference=self.reference, new_title=title, persistence=self.replace)
