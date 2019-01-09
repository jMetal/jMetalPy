import logging
from abc import ABC
from typing import TypeVar, List, Tuple

import matplotlib.pyplot as plt
import pandas as pd

LOGGER = logging.getLogger('jmetal')

S = TypeVar('S')


class Plot(ABC):

    def __init__(self, plot_title: str, reference_front: List[S], reference_point: list, axis_labels: list):
        """
        :param plot_title: Title of the graph.
        :param axis_labels: List of axis labels.
        :param reference_point:
        :param reference_front:
        """
        self.plot_title = plot_title
        self.axis_labels = axis_labels
        self.reference_point = reference_point
        self.reference_front = reference_front
        self.dimension = None

    @staticmethod
    def get_points(front: List[S]) -> Tuple[pd.DataFrame, int]:
        """ Get points for each solution of the front.

        :param front: List of solutions.
        :return: Pandas dataframe with one column for each objective and one row for each solution. """
        if front is None:
            raise Exception('Front is none!')

        points = pd.DataFrame(list(solution.objectives for solution in front))
        return points, points.shape[1]

    def two_dim(self, solutions: S):
        points, _ = self.get_points(solutions)
        points.plot(kind='scatter', x=0, y=1)

    def three_dim(self, solutions: S):
        fig = plt.figure()

        ax = fig.add_subplot(111, projection='3d')
        ax.scatter([s.objectives[0] for s in solutions],
                   [s.objectives[1] for s in solutions],
                   [s.objectives[2] for s in solutions])
        ax.set_xlim([0, 1.1])
        ax.set_ylim([0, 1.1])
        ax.set_zlim([0, 1.1])
        ax.view_init(elev=30.0, azim=15)
        ax.set_title(self.plot_title)

        plt.show()
