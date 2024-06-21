import unittest

from jmetal.problem.multiobjective.rwa import Ahmad2017, Chen2015, Ganesan2013, Gao2020, Goel2007, Liao2008, Padhi2016, \
    Subasi2016, Vaidyanathan2004, Xu2020


class Ahmad2017TestCases(unittest.TestCase):
    def setUp(self):
        self.problem = Ahmad2017()

    def test_number_of_variables(self):
        self.assertEqual(3, self.problem.number_of_variables())

    def test_number_of_objectives(self):
        self.assertEqual(7, self.problem.number_of_objectives())

    def test_number_of_constraints(self):
        self.assertEqual(0, self.problem.number_of_constraints())

    def test_variable_bounds(self):
        self.assertEqual(10.0, self.problem.lower_bound[0])
        self.assertEqual(10.0, self.problem.lower_bound[1])
        self.assertEqual(150.0, self.problem.lower_bound[2])

        self.assertEqual(50.0, self.problem.upper_bound[0])
        self.assertEqual(50.0, self.problem.upper_bound[1])
        self.assertEqual(170.0, self.problem.upper_bound[2])

    def test_name(self):
        self.assertEqual("Ahmad2017", self.problem.name())


class Chen2015TestCases(unittest.TestCase):
    def setUp(self):
        self.problem = Chen2015()

    def test_number_of_variables(self):
        self.assertEqual(6, self.problem.number_of_variables())

    def test_number_of_objectives(self):
        self.assertEqual(5, self.problem.number_of_objectives())

    def test_number_of_constraints(self):
        self.assertEqual(0, self.problem.number_of_constraints())

    def test_variable_bounds(self):
        self.assertEqual(17.5, self.problem.lower_bound[0])
        self.assertEqual(17.5, self.problem.lower_bound[1])
        self.assertEqual(2.0, self.problem.lower_bound[2])
        self.assertEqual(2.0, self.problem.lower_bound[3])
        self.assertEqual(5.0, self.problem.lower_bound[4])
        self.assertEqual(5.0, self.problem.lower_bound[5])

        self.assertEqual(22.5, self.problem.upper_bound[0])
        self.assertEqual(22.5, self.problem.upper_bound[1])
        self.assertEqual(3.0, self.problem.upper_bound[2])
        self.assertEqual(3.0, self.problem.upper_bound[3])
        self.assertEqual(7.0, self.problem.upper_bound[4])
        self.assertEqual(6.0, self.problem.upper_bound[5])

    def test_name(self):
        self.assertEqual("Chen2015", self.problem.name())

class Ganesan2013TestCases(unittest.TestCase):
    def setUp(self):
        self.problem = Ganesan2013()

    def test_number_of_variables(self):
        self.assertEqual(3, self.problem.number_of_variables())

    def test_number_of_objectives(self):
        self.assertEqual(3, self.problem.number_of_objectives())

    def test_number_of_constraints(self):
        self.assertEqual(0, self.problem.number_of_constraints())

    def test_variable_bounds(self):
        self.assertEqual(0.25, self.problem.lower_bound[0])
        self.assertEqual(10000.0, self.problem.lower_bound[1])
        self.assertEqual(600.0, self.problem.lower_bound[2])

        self.assertEqual(0.55, self.problem.upper_bound[0])
        self.assertEqual(20000.0, self.problem.upper_bound[1])
        self.assertEqual(1100.0, self.problem.upper_bound[2])

    def test_name(self):
        self.assertEqual("Ganesan2013", self.problem.name())


class Gao2020TestCases(unittest.TestCase):
    def setUp(self):
        self.problem = Gao2020()

    def test_number_of_variables(self):
        self.assertEqual(9, self.problem.number_of_variables())

    def test_number_of_objectives(self):
        self.assertEqual(3, self.problem.number_of_objectives())

    def test_number_of_constraints(self):
        self.assertEqual(0, self.problem.number_of_constraints())

    def test_variable_bounds(self):
        self.assertEqual(40.0, self.problem.lower_bound[0])
        self.assertEqual(0.35, self.problem.lower_bound[1])
        self.assertEqual(333.0, self.problem.lower_bound[2])
        self.assertEqual(20.0, self.problem.lower_bound[3])
        self.assertEqual(3000., self.problem.lower_bound[4])
        self.assertEqual(0.1, self.problem.lower_bound[5])
        self.assertEqual(308.0, self.problem.lower_bound[6])
        self.assertEqual(150.0, self.problem.lower_bound[7])
        self.assertEqual(0.1, self.problem.lower_bound[8])

        self.assertEqual(100.0, self.problem.upper_bound[0])
        self.assertEqual(0.5, self.problem.upper_bound[1])
        self.assertEqual(363.0, self.problem.upper_bound[2])
        self.assertEqual(40.0, self.problem.upper_bound[3])
        self.assertEqual(4000.0, self.problem.upper_bound[4])
        self.assertEqual(3.0, self.problem.upper_bound[5])
        self.assertEqual(328.0, self.problem.upper_bound[6])
        self.assertEqual(200, self.problem.upper_bound[7])
        self.assertEqual(2.0, self.problem.upper_bound[8])

    def test_name(self):
        self.assertEqual("Gao2020", self.problem.name())


class Goel2007TestCases(unittest.TestCase):
    def setUp(self):
        self.problem = Goel2007()

    def test_number_of_variables(self):
        self.assertEqual(4, self.problem.number_of_variables())

    def test_number_of_objectives(self):
        self.assertEqual(3, self.problem.number_of_objectives())

    def test_number_of_constraints(self):
        self.assertEqual(0, self.problem.number_of_constraints())

    def test_variable_bounds(self):
        self.assertEqual([0.0]*4, self.problem.lower_bound)
        self.assertEqual([1.0]*4, self.problem.upper_bound)

    def test_name(self):
        self.assertEqual("Goel2007", self.problem.name())

class Liao2008TestCases(unittest.TestCase):
    def setUp(self):
        self.problem = Liao2008()

    def test_number_of_variables(self):
        self.assertEqual(5, self.problem.number_of_variables())

    def test_number_of_objectives(self):
        self.assertEqual(3, self.problem.number_of_objectives())

    def test_number_of_constraints(self):
        self.assertEqual(0, self.problem.number_of_constraints())

    def test_variable_bounds(self):
        self.assertEqual([1.0]*5, self.problem.lower_bound)
        self.assertEqual([3.0]*5, self.problem.upper_bound)

    def test_name(self):
        self.assertEqual("Liao2008", self.problem.name())


class Padhi2016TestCases(unittest.TestCase):
    def setUp(self):
        self.problem = Padhi2016()

    def test_number_of_variables(self):
        self.assertEqual(5, self.problem.number_of_variables())

    def test_number_of_objectives(self):
        self.assertEqual(3, self.problem.number_of_objectives())

    def test_number_of_constraints(self):
        self.assertEqual(0, self.problem.number_of_constraints())

    def test_variable_bounds(self):
        self.assertEqual(1.0, self.problem.lower_bound[0])
        self.assertEqual(10.0, self.problem.lower_bound[1])
        self.assertEqual(850.0, self.problem.lower_bound[2])
        self.assertEqual(20.0, self.problem.lower_bound[3])
        self.assertEqual(4.0, self.problem.lower_bound[4])

        self.assertEqual(1.4, self.problem.upper_bound[0])
        self.assertEqual(26.0, self.problem.upper_bound[1])
        self.assertEqual(1650.0, self.problem.upper_bound[2])
        self.assertEqual(40.0, self.problem.upper_bound[3])
        self.assertEqual(8.0, self.problem.upper_bound[4])

    def test_name(self):
        self.assertEqual("Padhi2016", self.problem.name())


class Subasi2016TestCases(unittest.TestCase):
    def setUp(self):
        self.problem = Subasi2016()

    def test_number_of_variables(self):
        self.assertEqual(5, self.problem.number_of_variables())

    def test_number_of_objectives(self):
        self.assertEqual(2, self.problem.number_of_objectives())

    def test_number_of_constraints(self):
        self.assertEqual(0, self.problem.number_of_constraints())

    def test_variable_bounds(self):
        self.assertEqual(20.0, self.problem.lower_bound[0])
        self.assertEqual(6.0, self.problem.lower_bound[1])
        self.assertEqual(20.0, self.problem.lower_bound[2])
        self.assertEqual(0.0, self.problem.lower_bound[3])
        self.assertEqual(8000.0, self.problem.lower_bound[4])

        self.assertEqual(60.0, self.problem.upper_bound[0])
        self.assertEqual(15.0, self.problem.upper_bound[1])
        self.assertEqual(40.0, self.problem.upper_bound[2])
        self.assertEqual(30.0, self.problem.upper_bound[3])
        self.assertEqual(25000.0, self.problem.upper_bound[4])

    def test_name(self):
        self.assertEqual("Subasi2016", self.problem.name())


class Vaidyanathan2004TestCases(unittest.TestCase):
    def setUp(self):
        self.problem = Vaidyanathan2004()

    def test_number_of_variables(self):
        self.assertEqual(4, self.problem.number_of_variables())

    def test_number_of_objectives(self):
        self.assertEqual(4, self.problem.number_of_objectives())

    def test_number_of_constraints(self):
        self.assertEqual(0, self.problem.number_of_constraints())

    def test_variable_bounds(self):
        self.assertEqual([0.0]*4, self.problem.lower_bound)
        self.assertEqual([1.0]*4, self.problem.upper_bound)

    def test_name(self):
        self.assertEqual("Vaidyanathan2004", self.problem.name())


class Xu2020TestCases(unittest.TestCase):
    def setUp(self):
        self.problem = Xu2020()

    def test_number_of_variables(self):
        self.assertEqual(4, self.problem.number_of_variables())

    def test_number_of_objectives(self):
        self.assertEqual(3, self.problem.number_of_objectives())

    def test_number_of_constraints(self):
        self.assertEqual(0, self.problem.number_of_constraints())

    def test_variable_bounds(self):
        self.assertEqual(12.56, self.problem.lower_bound[0])
        self.assertEqual(0.02, self.problem.lower_bound[1])
        self.assertEqual(1.0, self.problem.lower_bound[2])
        self.assertEqual(0.5, self.problem.lower_bound[3])

        self.assertEqual(25.12, self.problem.upper_bound[0])
        self.assertEqual(0.06, self.problem.upper_bound[1])
        self.assertEqual(5.0, self.problem.upper_bound[2])
        self.assertEqual(2.0, self.problem.upper_bound[3])

    def test_name(self):
        self.assertEqual("Xu2020", self.problem.name())


if __name__ == "__main__":
    unittest.main()
