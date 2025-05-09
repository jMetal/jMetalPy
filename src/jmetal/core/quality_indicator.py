from abc import ABC, abstractmethod
from typing import Iterable

import moocore
import numpy as np
from scipy import spatial


class QualityIndicator(ABC):
    def __init__(self, is_minimization: bool):
        self.is_minimization = is_minimization

    @abstractmethod
    def compute(self, solutions: np.array):
        """
        :param solutions: [m, n] bi-dimensional numpy array, being m the number of solutions and n the dimension of
        each solution
        :return: the value of the quality indicator
        """
        pass

    @abstractmethod
    def get_name(self) -> str:
        pass

    @abstractmethod
    def get_short_name(self) -> str:
        pass


class FitnessValue(QualityIndicator):
    def __init__(self, is_minimization: bool = True):
        super(FitnessValue, self).__init__(is_minimization=is_minimization)

    def compute(self, solutions: np.array):
        if self.is_minimization:
            mean = np.mean([s.objectives for s in solutions])
        else:
            mean = -np.mean([s.objectives for s in solutions])

        return mean

    def get_name(self) -> str:
        return "Fitness"

    def get_short_name(self) -> str:
        return "Fitness"


class GenerationalDistance(QualityIndicator):
    def __init__(self, reference_front: np.array = None):
        """
        * Van Veldhuizen, D.A., Lamont, G.B.: Multiobjective Evolutionary Algorithm Research: A History and Analysis.
          Technical Report TR-98-03, Dept. Elec. Comput. Eng., Air Force. Inst. Technol. (1998)
        """
        super(GenerationalDistance, self).__init__(is_minimization=True)
        self.reference_front = reference_front

    def compute(self, solutions: np.array):
        if self.reference_front is None:
            raise Exception("Reference front is none")

        distances = spatial.distance.cdist(solutions, self.reference_front)

        return np.mean(np.min(distances, axis=1))

    def get_short_name(self) -> str:
        return "GD"

    def get_name(self) -> str:
        return "Generational Distance"


class InvertedGenerationalDistance(QualityIndicator):
    def __init__(self, reference_front: np.array = None):
        super(InvertedGenerationalDistance, self).__init__(is_minimization=True)
        self.reference_front = reference_front

    def compute(self, solutions: np.array):
        if self.reference_front is None:
            raise Exception("Reference front is none")

        distances = spatial.distance.cdist(self.reference_front, solutions)

        return np.mean(np.min(distances, axis=1))

    def get_short_name(self) -> str:
        return "IGD"

    def get_name(self) -> str:
        return "Inverted Generational Distance"


class EpsilonIndicator(QualityIndicator):
    def __init__(self, reference_front: np.array = None):
        super(EpsilonIndicator, self).__init__(is_minimization=True)
        self.reference_front = reference_front

    def compute(self, front: np.array) -> float:
        return max([min([max([s2[k] - s1[k] for k in range(len(s2))]) for s2 in front]) for s1 in self.reference_front])

    def get_short_name(self) -> str:
        return "EP"

    def get_name(self) -> str:
        return "Additive Epsilon"


class HyperVolume(QualityIndicator):
    """Hypervolume computation offered by the moocore project (https://multi-objective.github.io/moocore/python/)
    """

    def __init__(self, reference_point: [float] = None):
        super(HyperVolume, self).__init__(is_minimization=False)
        self.referencePoint = reference_point
        self.hv = moocore.Hypervolume(ref=reference_point)

    def compute(self, solutions: np.array):
        """

        :return: The hypervolume that is dominated by a non-dominated front.
        """
        return self.hv(solutions)

    def get_name(self) -> str:
        return "HV"

    def get_short_name(self) -> str:
        return "Hypervolume quality indicator"

class NormalizedHyperVolume(QualityIndicator):
    """Implementation of the normalized hypervolume, which is calculated as follows:

    relative hypervolume = 1 - (HV of the front / HV of the reference front).

    Minimization is implicitly assumed here!
    """

    def __init__(self, reference_point: Iterable[float], reference_front: np.array):
        """Delegates the computation of the HyperVolume to `jMetal.core.quality_indicator.HyperVolume`.

        Fails if the HV of the reference front is zero."""
        self.reference_point = reference_point
        self._hv = HyperVolume(reference_point=reference_point)
        self._reference_hypervolume = self._hv.compute(reference_front)

        assert self._reference_hypervolume != 0, "Hypervolume of reference front is zero"

    def compute(self, solutions: np.array) -> float:
        hv = self._hv.compute(solutions=solutions)

        return 1 - (hv / self._reference_hypervolume)

    def get_short_name(self) -> str:
        return "NHV"

    def get_name(self) -> str:
        return "Normalized Hypervolume"
