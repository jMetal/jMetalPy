import unittest

from jmetal.core.util.observer.impl.defaultobservable import DefaultObservable
from jmetal.core.util.observer.observer import Observer

"""
class MockObserver(Observer):
    def __init__(self):
        self.args = []
        self.kwargs = {}

    def update(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        return
"""

class TestMethods(unittest.TestCase):
    def setUp(self):
        self.observable = DefaultObservable()

    def test_should_register_add_one_observer(self):
        observer = Observer()
        self.observable.register(observer)

        self.assertEqual(1, len(self.observable.observers))

if __name__ == "__main__":
    unittest.main()