from time import sleep
import pyqtgraph as pg
import threading
import numpy as np


# ==========================
# v0.1
# ==========================


class FuncThread(threading.Thread):
    def __init__(self, t, *a):
        self._t = t
        self._a = a
        threading.Thread.__init__(self)

    def run(self):
        self._t(*self._a)


class Chart(object):
    def __init__(self) -> None:
        self.plot = None
        self.curve = None
        self.downsampling = 'peak'
        self.clipToView = True
        self.line_color = 'r'
        self.ptr = 0
        self.x = np.zeros(9000)
        self.y = np.zeros(9000)

    def set_data_stream(self, x, y) -> None:
        # if self.win.isVisible():
        self.x[self.ptr] = x
        self.y[self.ptr] = y
        self.ptr += 1


class DataStreamWindow(object):

    def __init__(self, background_color: str='w', coordinate_system_color: str='b') -> None:
        self.qapp = None
        self.win = None
        self.charts_list = list()
        self.columns_display = 1
        self.background_color = background_color
        self.coordinate_system_color = coordinate_system_color

    def print_window(self):

        calculating_thread = FuncThread(self.__raise_thread_with_window)
        calculating_thread.start()

        # importante para que comience el hilo de mostrar ventana antes, de esta forma se inicializan las
        # los atributos necesarios antes de que se vaya lanzando update(), ya que en el se usa self.win y self.y
        sleep(1)

    def __raise_thread_with_window(self):

        # Importante que este la condiguracion de la ventana y el bucle de abajo en el mismo hilo, ya que si no, no se
        # dibuja la grafia en la ventana
        self.qapp = pg.mkQApp()
        self.win = pg.GraphicsWindow()  # raise window!

        if self.background_color:
            self.win.setBackground(self.background_color)
        if self.coordinate_system_color:
            pg.setConfigOption('foreground', self.coordinate_system_color)

        i = 0
        for chart in self.charts_list:
            if i % self.columns_display == 0 and i >= self.columns_display:
                self.win.nextRow()
            chart.plot = self.win.addPlot()
            if chart.downsampling:
                chart.plot.setDownsampling(mode=chart.downsampling)
            if chart.clipToView:
                chart.plot.setClipToView(True)
            chart.curve = chart.plot.plot()
            if chart.line_color:
                chart.curve.setPen(chart.line_color)
            i += 1

        while self.win.isVisible():
            # refresh data
            for chart in self.charts_list:
                chart.curve.setData(chart.x[:chart.ptr], chart.y[:chart.ptr])
                self.qapp.processEvents()

