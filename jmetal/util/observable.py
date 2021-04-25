import logging
import threading
import time

from jmetal.core.observer import Observable, Observer
from jmetal.util.reader import Reader

LOGGER = logging.getLogger('jmetal')

"""
.. module:: observable
   :platform: Unix, Windows
   :synopsis: Implementation of observable entities (using delegation)
.. moduleauthor:: Antonio J. Nebro <antonio@lcc.uma.es>
"""


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


class TimeCounter(threading.Thread):
    def __init__(self, delay: int, observable: Observable = DefaultObservable()):
        super(TimeCounter, self).__init__()
        self.observable = observable
        self.delay = delay

    def run(self):
        counter = 0
        observable_data = {}

        while True:
            time.sleep(self.delay)

            observable_data["COUNTER"] = counter
            self.observable.notify_all(**observable_data)

            counter += 1


class FileObservable(threading.Thread):
    def __init__(self, cost_matrix, cost_to_warehouse, file, num_vehicles, delay = 2, observable: Observable = DefaultObservable()):
        super(FileObservable, self).__init__()
        self.observable = observable
        self.counter = 0   
        self.delay = delay
        self.cost_matrix = cost_matrix     
        self.cost_to_warehouse = cost_to_warehouse
        self.reader = Reader(file, num_vehicles)
                
    def run(self):
        observable_data = {}
        while True:      
            time.sleep(self.delay)           
            cost_matrix, cost_to_warehouse = self.reader.read_from_file()
            self.cost_matrix = cost_matrix     
            self.cost_to_warehouse = cost_to_warehouse  
            observable_data["COST_MATRIX"] = self.cost_matrix
            observable_data["COST_TO_WAREHOUSE"] = self.cost_to_warehouse
            self.observable.notify_all(**observable_data) 
        