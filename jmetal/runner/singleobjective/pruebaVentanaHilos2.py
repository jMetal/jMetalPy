from PyQt5.QtCore import pyqtSignal

class Main(QWidget):

    def __init__(self):
        super().__init__()

    def StartButtonEvent(self):
        self.test = ExecuteThread()
        self.test.start()
        self.test.finished.connect(thread_finished)
        self.test.my_signal.connect(my_event)

    def thread_finished(self):
        # gets executed if thread finished
        pass

    def my_event(self):
        # gets executed on my_signal
        pass


class ExecuteThread(QThread):
    my_signal = pyqtSignal()

    def run(self):
        # do something here
        self.my_signal.emit()
        pass