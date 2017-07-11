from abc import ABCMeta, abstractmethod


class Observable(object):
    def register(self, observer):
        pass

    def deregister(self, observer):
        pass

    def deregister_all(self):
        pass

    def notify_all(self, *args, **kwargs):
        pass


class Observer(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def update(self, *args, **kwargs):
        """
        Update method
        :param args:
        :param kwargs:
        :return:
        """
        pass


class DefaultObservable(Observable):
    def __init__(self):
        self.observers = []

    def register(self, observer):
        if observer not in self.observers:
            self.observers.append(observer)

    def deregister(self, observer):
        if observer in self.observers:
            self.observers.remove(observer)

    def deregister_all(self):
        if self.observers:
            del self.observers[:]

    def notify_all(self, *args, **kwargs):
        for observer in self.observers:
            observer.update(*args, **kwargs)
