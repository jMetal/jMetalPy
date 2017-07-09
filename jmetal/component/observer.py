import logging
import os

from jmetal.util.observable import Observer
from jmetal.util.solution_list_output import SolutionListOutput

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BasicSingleObjectiveAlgorithmObserver(Observer):
    def __init__(self) -> None:
        pass

    def update(self, *args, **kwargs):
        logger.info("Evaluations: " + str(kwargs["evaluations"]) +
                    ". Best fitness: " + str(kwargs["population"][0].objectives[0]))


class WriteFrontToFileObserver(Observer):
    def __init__(self, output_directory) -> None:
        self.counter = 0
        self.directory = output_directory

        if os.path.isdir(self.directory):
            logger.info("Directory " + self.directory + " exists. Removing contents.")
            for file in os.listdir(self.directory):
                os.remove(self.directory + "/" + file)
        else:
            logger.info("Directory " + self.directory + " does not exist. Creating it.")
            os.mkdir(self.directory)

    def update(self, *args, **kwargs):
        SolutionListOutput.print_function_values_to_file(
            self.directory + "/FUN." + str(self.counter), kwargs["population"])

        self.counter += 1


class AlgorithmObserver(Observer):
    def __init__(self) -> None:
        pass

    def update(self, *args, **kwargs):
        SolutionListOutput.plot_scatter_real_time(kwargs["population"])
