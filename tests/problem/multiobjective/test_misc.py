#!/usr/bin/env python
# -*- coding: utf-8-"""

import unittest

from numpy.testing import assert_allclose

from jmetal.problem.multiobjective.misc import (
    CONV2, CONV3_4, CONV3, CONV4_2F, DENT, SYM_PART
)


class CONV2Test(unittest.TestCase):
    """Test class for the CONV2 problem."""

    def setUp(self):
        """Set up the test fixture."""
        self.problem = CONV2()
        self.solution = self.problem.create_solution()

    def test_initialization(self):
        """Test problem initialization."""
        self.assertEqual(self.problem.number_of_variables(), 2)
        self.assertEqual(self.problem.number_of_objectives(), 2)
        self.assertEqual(self.problem.number_of_constraints(), 0)
        self.assertEqual(self.problem.lower_bound, [0.0, 0.0])
        self.assertEqual(self.problem.upper_bound, [10.0, 10.0])
        self.assertEqual(self.problem.obj_directions, [self.problem.MINIMIZE] * 2)
        self.assertEqual(self.problem.obj_labels, ['f1', 'f2'])
    
    def test_name(self):
        self.assertEqual(self.problem.name(), 'CONV2')
    
    def test_evaluate_at_origin(self):
        self.solution.variables = [0.0, 0.0]
        self.problem.evaluate(self.solution)
        assert_allclose(self.solution.objectives, [0.0, 100.0])
    
    def test_evaluate_at_upper_bound(self):
        self.solution.variables = [10.0, 10.0]
        self.problem.evaluate(self.solution)
        assert_allclose(self.solution.objectives, [200.0, 100.0])
    
    def test_evaluate_at_midpoint(self):
        self.solution.variables = [5.0, 5.0]
        self.problem.evaluate(self.solution)
        assert_allclose(self.solution.objectives, [50.0, 50.0])
    
    def test_solution_independence(self):
        sol1 = self.problem.create_solution()
        sol2 = self.problem.create_solution()
        sol1.variables = [1.0, 2.0]
        sol2.variables = [3.0, 4.0]
        self.problem.evaluate(sol1)
        self.problem.evaluate(sol2)
        self.assertNotEqual(sol1.objectives, sol2.objectives)


class CONV3_4Test(unittest.TestCase):
    """Test class for the CONV3_4 problem."""

    def setUp(self):
        self.problem = CONV3_4()
        self.solution = self.problem.create_solution()

    def test_initialization(self):
        self.assertEqual(self.problem.number_of_variables(), 3)
        self.assertEqual(self.problem.number_of_objectives(), 3)
        self.assertEqual(self.problem.number_of_constraints(), 0)
        self.assertEqual(self.problem.lower_bound, [-3.0] * 3)
        self.assertEqual(self.problem.upper_bound, [3.0] * 3)
        self.assertEqual(self.problem.obj_directions, [self.problem.MINIMIZE] * 3)
        self.assertEqual(self.problem.obj_labels, ['f1', 'f2', 'f3'])
    
    def test_name(self):
        self.assertEqual(self.problem.name(), 'CONV3_4')
    
    def test_evaluate_at_reference_points(self):
        # Test at reference point a1 = [-1, -1, -1]
        self.solution.variables = self.problem.a1.copy()
        self.problem.evaluate(self.solution)
        # f1 = 0 (at a1) + 0 + 0 = 0
        # f2 = 4 + 16 + 4 = 24
        # f3 = 0 + 4 + 0 = 4
        assert_allclose(self.solution.objectives, [0.0, 24.0, 4.0])
    
        # Test at reference point a2 = [1, 1, 1]
        self.solution.variables = self.problem.a2.copy()
        self.problem.evaluate(self.solution)
        # f1 = 16 + 4 + 4 = 24
        # f2 = 0 (at a2) + 0 + 0 = 0
        # f3 = 4 + 0 + 16 = 20
        assert_allclose(self.solution.objectives, [24.0, 0.0, 20.0])
    
        # Test at reference point a3 = [-1, 1, -1]
        self.solution.variables = self.problem.a3.copy()
        self.problem.evaluate(self.solution)
        # f1 = 0 + 4 + 0 = 4
        # f2 = 4 + 0 + 4 = 8
        # f3 = 0 (at a3) + 0 + 0 = 0
        assert_allclose(self.solution.objectives, [4.0, 8.0, 0.0])

class CONV3Test(unittest.TestCase):
    def setUp(self):
        self.problem = CONV3()
        self.solution = self.problem.create_solution()

    def test_initialization(self):
        self.assertEqual(self.problem.number_of_variables(), 3)
        self.assertEqual(self.problem.number_of_objectives(), 3)
        self.assertEqual(self.problem.number_of_constraints(), 0)
        self.assertEqual(self.problem.lower_bound, [-3.0] * 3)
        self.assertEqual(self.problem.upper_bound, [3.0] * 3)
        self.assertEqual(self.problem.obj_directions, [self.problem.MINIMIZE] * 3)
        self.assertEqual(self.problem.obj_labels, ['f1', 'f2', 'f3'])
    
    def test_name(self):
        self.assertEqual(self.problem.name(), 'CONV3')
    
    def test_evaluate_at_origin(self):
        self.solution.variables = [0.0, 0.0, 0.0]
        self.problem.evaluate(self.solution)
        # f1 = (0+1)² + (0+1)² + (0+1)² = 3
        # f2 = (0-1)² + (0-1)² + (0-1)² = 3
        # f3 = (0+1)² + (0-1)² + (0+1)² = 3
        assert_allclose(self.solution.objectives, [3.0, 3.0, 3.0])


class CONV4_2FTest(unittest.TestCase):
    def setUp(self):
        self.problem = CONV4_2F()
        self.solution = self.problem.create_solution()

    def test_initialization(self):
        self.assertEqual(self.problem.number_of_variables(), 4)
        self.assertEqual(self.problem.number_of_objectives(), 4)
        self.assertEqual(self.problem.number_of_constraints(), 0)
        self.assertEqual(self.problem.lower_bound, [-3.0] * 4)
        self.assertEqual(self.problem.upper_bound, [3.0] * 4)
        self.assertEqual(self.problem.obj_directions, [self.problem.MINIMIZE] * 4)
        self.assertEqual(self.problem.obj_labels, ['f1', 'f2', 'f3', 'f4'])
    
    def test_name(self):
        self.assertEqual(self.problem.name(), 'CONV4-2F')
    
    def test_evaluate_all_negative(self):
        self.solution.variables = [-1.0, -1.0, -1.0, -1.0]
        self.problem.evaluate(self.solution)
        # With all x_i < 0, it uses the convex formulation
        # This is a basic check, exact values would require calculating the sigma terms
        self.assertEqual(len(self.solution.objectives), 4)
    
    def test_evaluate_some_positive(self):
        self.solution.variables = [1.0, -1.0, 1.0, -1.0]
        self.problem.evaluate(self.solution)
        # With some x_i >= 0, it uses the product-based formulation
        # f1 = (1*1)² + 0 + 0 + 0 = 1
        # f2 = 0 + (-1*1)² + 0 + 0 = 1
        # f3 = 0 + 0 + (1*1)² + 0 = 1
        # f4 = 0 + 0 + 0 + (-1*1)² = 1
        assert_allclose(self.solution.objectives, [1.0, 1.0, 1.0, 1.0])


class DENTTest(unittest.TestCase):
    """Test class for the DENT problem."""

    def setUp(self):
        self.problem = DENT()
        self.solution = self.problem.create_solution()

    def test_initialization(self):
        self.assertEqual(self.problem.number_of_variables(), 2)
        self.assertEqual(self.problem.number_of_objectives(), 2)
        self.assertEqual(self.problem.number_of_constraints(), 0)
        self.assertEqual(self.problem.lower_bound, [-2.0, -2.0])
        self.assertEqual(self.problem.upper_bound, [2.0, 2.0])
        self.assertEqual(self.problem.obj_directions, [self.problem.MINIMIZE] * 2)
        self.assertEqual(self.problem.obj_labels, ['f1', 'f2'])
    
    def test_name(self):
        self.assertEqual(self.problem.name(), 'DENT')
    
    def test_evaluate_at_origin(self):
        self.solution.variables = [0.0, 0.0]
        self.problem.evaluate(self.solution)
        # At (0,0), exp_term = 0.85 * exp(0) = 0.85
        # term1 = term2 = sqrt(1 + 0) = 1
        # f1 = 0.5 * (1 + 1 + 0 - 0) + 0.85 = 1.85
        # f2 = 0.5 * (1 + 1 - 0 + 0) + 0.85 = 1.85
        assert_allclose(self.solution.objectives, [1.85, 1.85])


class SYM_PARTTest(unittest.TestCase):
    """Test class for the SYM_PART problem."""

    def setUp(self):
        self.problem = SYM_PART()
        self.solution = self.problem.create_solution()

    def test_initialization(self):
        self.assertEqual(self.problem.number_of_variables(), 2)
        self.assertEqual(self.problem.number_of_objectives(), 2)
        self.assertEqual(self.problem.number_of_constraints(), 0)
        self.assertEqual(self.problem.lower_bound, [-0.5, -0.5])
        self.assertEqual(self.problem.upper_bound, [0.5, 0.5])
        self.assertEqual(self.problem.obj_directions, [self.problem.MINIMIZE] * 2)
        self.assertEqual(self.problem.obj_labels, ['f1', 'f2'])
    
    def test_name(self):
        self.assertEqual(self.problem.name(), 'SYM-PART')
    
    def test_evaluate_at_origin(self):
        self.solution.variables = [0.0, 0.0]
        self.problem.evaluate(self.solution)
        # At (0,0): t1 = t2 = 0
        # f1 = (0 - 0*(c+2a) + a)² + (0 - 0*b)² = a² + 0 = 0.25
        # f2 = (0 - 0*(c+2a) - a)² + (0 - 0*b)² = a² + 0 = 0.25
        assert_allclose(self.solution.objectives, [0.25, 0.25], atol=1e-6)


if __name__ == "__main__":
    unittest.main()