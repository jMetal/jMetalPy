import logging
import threading
import time

from jmetal.core.observable import DefaultObservable
from jmetal.core.observable import Observable

LOGGER = logging.getLogger('jmetal')

"""
.. module:: observable
   :platform: Unix, Windows
   :synopsis: Implementation of observable entities (using delegation)
.. moduleauthor:: Antonio J. Nebro <antonio@lcc.uma.es>
"""


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
