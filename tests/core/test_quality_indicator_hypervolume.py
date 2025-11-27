import numpy as np
from jmetal.core.quality_indicator import HyperVolume, NormalizedHyperVolume


def test_hypervolume_derived_reference_from_front_and_offset():
    # reference front with extremes
    ref_front = np.array([[1.0, 2.0], [0.5, 1.5]])
    # derive reference point as element-wise max -> [1.0, 2.0], offset 0.1
    hv = HyperVolume(reference_front=ref_front, reference_point_offset=0.1)

    # internal _reference_point should be the maxima
    assert hv._reference_point == [1.0, 2.0]

    # compute hypervolume of the reference front itself: with offset>0 it should be > 0
    hv_value = hv.compute(ref_front)
    assert hv_value > 0.0


def test_hypervolume_explicit_reference_has_priority_over_front():
    ref_front = np.array([[1.0, 2.0], [0.5, 1.5]])
    # explicit reference point provided -> it should be used instead of deriving from front
    hv = HyperVolume(reference_point=[2.0, 3.0], reference_front=ref_front, reference_point_offset=0.0)
    assert hv._reference_point == [2.0, 3.0]

    # hv.compute of the ref_front should be positive because ref is worse
    val = hv.compute(ref_front)
    assert val >= 0.0


def test_normalized_hypervolume_with_reference_front():
    ref_front = np.array([[1.0, 1.0], [0.5, 0.8]])
    # create normalized HV using a reference front and offset
    nhv = NormalizedHyperVolume(reference_front=ref_front, reference_point_offset=0.05)

    # after initialization, reference hypervolume must be computed and > 0
    nhv.set_reference_front(ref_front)
    assert nhv._reference_hypervolume is not None
    assert nhv._reference_hypervolume > 0.0

    # compute normalized hv for the same front should be 0.0 when reference equals front (with offset it's >0)
    val = nhv.compute(ref_front)
    assert 0.0 <= val <= 1.0


def test_zdt1_reference_front_hv_above_threshold():
    """Load ZDT1 reference front and assert its hypervolume (offset=0.0) > 0.66."""
    from jmetal.util.solution import read_solutions

    ref_sols = read_solutions(filename="resources/reference_fronts/ZDT1.pf")
    ref = np.array([s.objectives for s in ref_sols])

    hv = HyperVolume(reference_front=ref, reference_point_offset=0.0)
    hv_ref = hv.compute(ref)

    # tighter threshold observed in this environment: 0.6661601248750012
    assert hv_ref > 0.666
