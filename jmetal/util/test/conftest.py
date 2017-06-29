import pytest
from jmetal.util.test.test_crowding_distance import CrowdingDistanceTestCases
from jmetal.util.test.test_dominancecomparator import DominanceComparatorTestCases
from jmetal.util.test.test_nondominatedsolutionlistarchive import NonDominatedTestCases
from jmetal.util.test.test_observable import ObservableTestCases
#from jmetal.util.test.test_ranking import DominanceRankingTestCases


def pytest_addoption(parser):
    parser.addoption('--self', default=1, help="run many tests")


def pytest_generate_tests(metafunc):
    self = metafunc.config.getoption('--self')
    if 'self' in metafunc.fixturenames:
        for x in range(1, self):
            metafunc.parametrize("self", self)

#@pytest.mark.parametrize("test_should_the_crowding_distance_of_an_empty_set_do_nothing", test_should_the_crowding_distance_of_an_empty_set_do_nothing())
#def test_funcion_crowding(test_should_the_crowding_distance_of_an_empty_set_do_nothing)