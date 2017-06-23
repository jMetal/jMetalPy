import threading
import sys
import time

from PyQt5 import QtWidgets, QtCore



class Application(QMainWindow):
    counter = pyqtSignal(str)
    counting = False

    def __init__(self):
        super(Application, self).__init__()

        self.button = QPushButton()
        self.button.setText('99')
        self.button.clicked.connect(self.startCounting)
        self.counter.connect(self.button.setText)

        self.layout = QVBoxLayout()
        self.layout.addWidget(self.button)
        self.frame = QFrame()
        self.frame.setLayout(self.layout)
        self.setCentralWidget(self.frame)

    def startCounting(self):
        if not self.counting:
            self.counting = True
            thread = threading.Thread(target=self.something)
            thread.start()

    def something(self):
        for x in range(100):
            self.counter.emit(str(x))
            time.sleep(.01)
        self.counting = False



if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = Application()
    window.show()
    sys.exit(app.exec_())