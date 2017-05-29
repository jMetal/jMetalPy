import unittest

from jmetal.core.util.observer.impl.defaultobservable import DefaultObservable
from jmetal.core.util.observer.observer import Observer


class TestMethods(unittest.TestCase):
    def setUp(self):
        self.observable = DefaultObservable()

    def test_should_register_add_one_observer(self):
        observer = Observer()
        self.observable.register(observer)

        self.assertEqual(1, len(self.observable.observers))

    def test_should_register_add_two_observer(self):
        self.observable.register(Observer())
        self.observable.register(Observer())

        self.assertEqual(2, len(self.observable.observers))

    def test_should_register_add_two_observer(self):
        self.observable.register(Observer())
        self.observable.register(Observer())

        self.assertEqual(2, len(self.observable.observers))

    def test_should_unregister_remove_the_observer_if_it_is_registered(self):
        observer = Observer()
        self.observable.register(observer)
        self.observable.unregister(observer)

        self.assertEqual(0, len(self.observable.observers))

    def test_should_unregister_not_remove_the_observer_if_it_is_not_registered(self):
        observer = Observer()
        observer2 = Observer()
        self.observable.register(observer)
        self.observable.unregister(observer2)

        self.assertEqual(1, len(self.observable.observers))
        self.assertTrue(observer in self.observable.observers)
        self.assertFalse(observer2 in self.observable.observers)

if __name__ == "__main__":
    unittest.main()