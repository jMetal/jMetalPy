import logging
from typing import TypeVar, List, Tuple

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from pandas import plotting

LOGGER = logging.getLogger('jmetal')

S = TypeVar('S')


class Plot:

    def __init__(self,
                 title: str = 'Pareto front approximation',
                 reference_front: List[S] = None,
                 reference_point: list = None,
                 axis_labels: list = None):
        """
        :param title: Title of the graph.
        :param axis_labels: List of axis labels.
        :param reference_point: Reference point (e.g., [0.4, 1.2]).
        :param reference_front: Reference Pareto front (if any) as solutions.
        """
        self.plot_title = title
        self.axis_labels = axis_labels

        if reference_point and not isinstance(reference_point[0], list):
            reference_point = [reference_point]

        self.reference_point = reference_point
        self.reference_front = reference_front
        self.dimension = None

    @staticmethod
    def get_points(solutions: List[S]) -> Tuple[pd.DataFrame, int]:
        """ Get points for each solution of the front.

        :param solutions: List of solutions.
        :return: Pandas dataframe with one column for each objective and one row for each solution.
        """
        if solutions is None:
            raise Exception('Front is none!')

        points = pd.DataFrame(list(solution.objectives for solution in solutions))
        return points, points.shape[1]

    def plot(self, front, label='', normalize: bool = False, filename: str = None, format: str = 'eps'):
        """ Plot any arbitrary number of fronts in 2D, 3D or p-coords.

        :param front: Pareto front or a list of them.
        :param label: Pareto front title or a list of them.
        :param normalize: If True, normalize data (for p-coords).
        :param filename: Output filename.
        :param format: Output file format.
        """
        if not isinstance(front[0], list):
            front = [front]

        if not isinstance(label, list):
            label = [label]

        if len(front) != len(label):
            raise Exception('Number of fronts and labels must be the same')

        dimension = front[0][0].number_of_objectives

        if dimension == 2:
            self.two_dim(front, label, filename, format)
        elif dimension == 3:
            self.three_dim(front, label, filename, format)
        else:
            self.pcoords(front, normalize, filename, format)

    def two_dim(self, fronts: List[list], labels: List[str] = None, filename: str = None, format: str = 'eps'):
        """ Plot any arbitrary number of fronts in 2D.

        :param fronts: List of fronts (containing solutions).
        :param labels: List of fronts title (if any).
        :param filename: Output filename.
        """
        n = int(np.ceil(np.sqrt(len(fronts))))
        fig = plt.figure()
        fig.suptitle(self.plot_title, fontsize=16)

        reference = None
        if self.reference_front:
            reference, _ = self.get_points(self.reference_front)

        for i, _ in enumerate(fronts):
            points, _ = self.get_points(fronts[i])

            ax = fig.add_subplot(n, n, i + 1)
            points.plot(kind='scatter', x=0, y=1, ax=ax, s=10, color='#236FA4', alpha=1.0)

            if labels:
                ax.set_title(labels[i])

            if self.reference_front:
                reference.plot(x=0, y=1, ax=ax, color='k', legend=False)

            if self.reference_point:
                for point in self.reference_point:
                    plt.plot([point[0]], [point[1]], marker='o', markersize=5, color='r')
                    plt.axvline(x=point[0], color='r', linestyle=':')
                    plt.axhline(y=point[1], color='r', linestyle=':')

            if self.axis_labels:
                plt.xlabel(self.axis_labels[0])
                plt.ylabel(self.axis_labels[1])

        if filename:
            plt.savefig(filename + '.' + format, format=format, dpi=200)
        else:
            plt.show()

        plt.close(fig=fig)

    def three_dim(self, fronts: List[list], labels: List[str] = None, filename: str = None, format: str = 'eps'):
        """ Plot any arbitrary number of fronts in 3D.

        :param fronts: List of fronts (containing solutions).
        :param labels: List of fronts title (if any).
        :param filename: Output filename.
        """
        n = int(np.ceil(np.sqrt(len(fronts))))
        fig = plt.figure()
        fig.suptitle(self.plot_title, fontsize=16)

        for i, _ in enumerate(fronts):
            ax = fig.add_subplot(n, n, i + 1, projection='3d')
            ax.scatter([s.objectives[0] for s in fronts[i]],
                       [s.objectives[1] for s in fronts[i]],
                       [s.objectives[2] for s in fronts[i]])

            if labels:
                ax.set_title(labels[i])

            if self.reference_front:
                ax.scatter([s.objectives[0] for s in self.reference_front],
                           [s.objectives[1] for s in self.reference_front],
                           [s.objectives[2] for s in self.reference_front])

            if self.reference_point:
                # todo
                pass

            ax.relim()
            ax.autoscale_view(True, True, True)
            ax.view_init(elev=30.0, azim=15.0)
            ax.locator_params(nbins=4)

        if filename:
            plt.savefig(filename + '.' + format, format=format, dpi=1000)
        else:
            plt.show()

        plt.close(fig=fig)

    def pcoords(self, fronts: List[list], normalize: bool = False, filename: str = None, format: str = 'eps'):
        """ Plot any arbitrary number of fronts in parallel coordinates.

        :param fronts: List of fronts (containing solutions).
        :param filename: Output filename.
        """
        n = int(np.ceil(np.sqrt(len(fronts))))
        fig = plt.figure()
        fig.suptitle(self.plot_title, fontsize=16)

        for i, _ in enumerate(fronts):
            points, _ = self.get_points(fronts[i])

            if normalize:
                points = (points - points.min()) / (points.max() - points.min())

            ax = fig.add_subplot(n, n, i + 1)

            min_, max_ = points.values.min(), points.values.max()
            points['scale'] = np.linspace(0, 1, len(points)) * (max_ - min_) + min_
            pd.plotting.parallel_coordinates(points, 'scale', ax=ax)

            ax.get_legend().remove()

            if self.axis_labels:
                ax.set_xticklabels(self.axis_labels)

        if filename:
            plt.savefig(filename + '.' + format, format=format, dpi=1000)
        else:
            plt.show()

        plt.close(fig=fig)
