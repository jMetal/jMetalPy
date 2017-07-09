import logging

import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ScatterPlot():
    def __init__(self, plot_title: str, animation_speed: float = 0.00001):
        self.plot_title = plot_title
        self.fig = plt.figure()
        self.axis = self.fig.add_subplot(111)
        self.sc = None

        # Real-time plotting options
        self.animation_speed = animation_speed

    def _init_plot(self, is_auto_scalable: bool) -> None:
        if is_auto_scalable:
            self.axis.set_autoscale_on(True)
            self.axis.autoscale_view(True, True, True)

        logger.info("Generating plot...")

        # Style options
        self.axis.grid(color='#f0f0f5', linestyle='-', linewidth=2, alpha=0.5)
        self.fig.suptitle(self.plot_title, fontsize=14, fontweight='bold')

    def simple_plot(self, x_val: list, y_val: list, file_name: str = "output",
                    format: str = 'png', dpi: int = 200, save: bool = False) -> None:
        self._init_plot(is_auto_scalable=True)
        self.sc, = self.axis.plot(x_val, y_val, 'bx', markersize=5)

        if save:
            # Supported formats: eps, jpeg, jpg, pdf, pgf, png, ps, raw, rgba, svg, svgz, tif, tiff
            logger.info("Output file (function plot): " + file_name + '.' + format)
            self.fig.savefig(file_name + '.' + format, format=format, dpi=dpi)

    def live_plot(self, x_val: list, y_val: list) -> None:
        if not self.sc:
            # The first time, initialize plot if it doesn't exist
            self.simple_plot(x_val, y_val)

        # Update
        self.sc.set_data(x_val, y_val)

        self.axis.relim()
        self.axis.autoscale_view(True, True, True)

        plt.draw()
        plt.pause(self.animation_speed)