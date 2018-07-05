import unittest

from jmetal.core.observable import DefaultObservable, Observer


class ObservableTestCases(unittest.TestCase):

    def setUp(self):
        self.observable = DefaultObservable()

    def test_should_register_add_one_observer(self):
        observer = Observer()
        self.observable.register(observer)

        self.assertEqual(1, len(self.observable.observers))

    def test_should_register_add_two_observers(self):
        self.observable.register(Observer())
        self.observable.register(Observer())

        self.assertEqual(2, len(self.observable.observers))

    def test_should_register_add_two_observer(self):
        self.observable.register(Observer())
        self.observable.register(Observer())

        self.assertEqual(2, len(self.observable.observers))

    def test_should_deregister_remove_the_observer_if_it_is_registered(self):
        observer = Observer()
        self.observable.register(observer)
        self.observable.register(Observer())
        self.observable.deregister(observer)

        self.assertEqual(1, len(self.observable.observers))

    def test_should_deregister_not_remove_the_observer_if_it_is_not_registered(self):
        observer = Observer()
        observer2 = Observer()
        self.observable.register(observer)
        self.observable.deregister(observer2)

        self.assertEqual(1, len(self.observable.observers))
        self.assertTrue(observer in self.observable.observers)
        self.assertFalse(observer2 in self.observable.observers)

    def test_should_deregister_all_remove_all_the_observers(self):
        self.observable.register(Observer())
        self.observable.register(Observer())
        self.observable.register(Observer())
        self.observable.register(Observer())
        self.observable.deregister_all()

        self.assertEqual(0, len(self.observable.observers))


if __name__ == "__main__":
    unittest.main()
