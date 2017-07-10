#from jmetal.util.test.test_ranking import DominanceRankingTestCases


def pytest_addoption(parser):
    parser.addoption('--self', default=1, help="run many tests")


def pytest_generate_tests(metafunc):
    self = metafunc.config.getoption('--self')
    if 'self' in metafunc.fixturenames:
        for x in range(1, self):
            metafunc.parametrize("self", self)