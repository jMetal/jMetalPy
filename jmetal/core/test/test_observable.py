import unittest

from jmetal.core.observer import Observer
from jmetal.util.observable import DefaultObservable


class ObservableTestCases(unittest.TestCase):

    class DummyObserver(Observer):

        def update(self, *args, **kwargs):
            pass

    def setUp(self):
        self.observable = DefaultObservable()
        self.observer = self.DummyObserver()

    def test_should_register_add_one_observer(self):
        self.observable.register(self.observer)

        self.assertEqual(1, len(self.observable.observers))

    def test_should_register_add_two_observers(self):
        observer_two = self.DummyObserver()

        self.observable.register(self.observer)
        self.observable.register(observer_two)

        self.assertEqual(2, len(self.observable.observers))

    def test_should_deregister_remove_the_observer_if_it_is_registered(self):
        observer_two = self.DummyObserver()

        self.observable.register(self.observer)
        self.observable.register(observer_two)
        self.observable.deregister(self.observer)

        self.assertEqual(1, len(self.observable.observers))

    def test_should_deregister_not_remove_the_observer_if_it_is_not_registered(self):
        observer_two = self.DummyObserver()

        self.observable.register(self.observer)
        self.observable.deregister(observer_two)

        self.assertEqual(1, len(self.observable.observers))
        self.assertTrue(self.observer in self.observable.observers)
        self.assertFalse(observer_two in self.observable.observers)

    def test_should_deregister_all_remove_all_the_observers(self):
        self.observable.register(self.observer)
        self.observable.register(self.observer)
        self.observable.register(self.observer)
        self.observable.register(self.observer)
        self.observable.deregister_all()

        self.assertEqual(0, len(self.observable.observers))


if __name__ == "__main__":
    unittest.main()
