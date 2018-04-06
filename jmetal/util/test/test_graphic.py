import os
import unittest

from jmetal.core.solution import FloatSolution
from jmetal.util.graphic import ScatterPlot


class GraphicTestCases(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        if os.path.exists("test.png"):
            os.remove("test.png")

    def test_should_print_solution_points_correctly(self):
        solution1, solution2 = FloatSolution(1, 2, 0, [], []), FloatSolution(1, 2, 0, [], [])
        solution1.objectives[0], solution2.objectives[0] = 5.0, 1.0 # x values
        solution1.objectives[1], solution2.objectives[1] = 3.0, 6.0 # y values
        solution_list = [solution1, solution2]

        plot = ScatterPlot(plot_title="title", animation_speed=1*10e-10)
        plot.simple_plot(solution_list=solution_list, file_name="test", save=False)

        x_values, y_values = plot.sc.get_xdata(), plot.sc.get_ydata()

        self.assertEqual(solution1.objectives[0], x_values[0])
        self.assertEqual(solution2.objectives[0], x_values[1])
        self.assertEqual(solution1.objectives[1], y_values[0])
        self.assertEqual(solution2.objectives[1], y_values[1])

    def test_should_raise_an_exception_when_format_is_not_supported(self):
        solution1, solution2 = FloatSolution(1, 2, 0, [], []), FloatSolution(1, 2, 0, [], [])
        solution1.objectives[0], solution2.objectives[0] = 5.0, 1.0 # x values
        solution1.objectives[1], solution2.objectives[1] = 3.0, 6.0 # y values
        solution_list = [solution1, solution2]

        plot = ScatterPlot(plot_title="title", animation_speed=1*10e-10)

        with self.assertRaises(Exception):
            plot.simple_plot(solution_list=solution_list, file_name="file", fmt="null", save=True)

    def test_should_raise_an_exception_when_updating_a_plot_that_doesnt_exist(self):
        plot = ScatterPlot(plot_title="title", animation_speed=1*10e-10)

        with self.assertRaises(Exception):
            plot.update(solution_list=None)


if __name__ == "__main__":
    unittest.main()