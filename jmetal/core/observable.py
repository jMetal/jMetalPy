from abc import ABCMeta, abstractmethod

"""
.. module:: Observable
   :platform: Unix, Windows
   :synopsis: Implementation of the observer-observable pattern.

.. moduleauthor:: Antonio J. Nebro <antonio@lcc.uma.es>
"""


class Observer:

    __metaclass__ = ABCMeta

    @abstractmethod
    def update(self, *args, **kwargs):
        """ Update method

        :param args:
        :param kwargs:
        :return:
        """
        pass


class Observable:

    __metaclass__ = ABCMeta

    @abstractmethod
    def register(self, observer):
        pass

    @abstractmethod
    def deregister(self, observer):
        pass

    @abstractmethod
    def deregister_all(self):
        pass

    @abstractmethod
    def notify_all(self, *args, **kwargs):
        pass


class DefaultObservable(Observable):

    def __init__(self):
        self.observers = []

    def register(self, observer: Observer):
        if observer not in self.observers:
            self.observers.append(observer)

    def deregister(self, observer: Observer):
        if observer in self.observers:
            self.observers.remove(observer)

    def deregister_all(self):
        if self.observers:
            del self.observers[:]

    def notify_all(self, *args, **kwargs):
        for observer in self.observers:
            observer.update(*args, **kwargs)
