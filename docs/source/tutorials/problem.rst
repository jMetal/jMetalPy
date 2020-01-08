Defining new problems
========================

To include a problem in jMetalPy, it must implement the :code:`Problem` interface from the :py:mod:`jmetal.core.problem` module.

Use case: Subset Sum
------------------------

The goal is to find a subset S of W (list of non-negative integers) whose elements sum is closest to (without exceeding) C.
For example, for the input :math:`W=\{3, 34, 4, 12, 5, 2\}` and :math:`C=9`, one output could be :math:`S=\{4, 5\}` (as it is a subset with sum 9).

In jMetalPy, this problem can be encoded as a binary problem with one objective (to be maximized) and one variable
(a binary array representing whethever the ith element of W is selected or not):

.. code-block:: python

   class SubsetSum(BinaryProblem):

      def __init__(self, C: int, W: list):
         super(SubsetSum, self).__init__(reference_front=None)
         self.C = C
         self.W = W

         self.number_of_bits = len(self.W)
         self.number_of_objectives = 1
         self.number_of_variables = 1
         self.number_of_constraints = 0

         self.obj_directions = [self.MAXIMIZE]
         self.obj_labels = ['Sum']

      def evaluate(self, solution: BinarySolution) -> BinarySolution:
         pass

      def create_solution(self) -> BinarySolution:
         pass

      def get_name(self) -> str:
         return 'Subset Sum'

Now we have to define the abstract methods :code:`evaluate` and :code:`create_solution` from the :py:mod:`jmetal.core.problem.Problem` class.

Note that each solution consists of one objective function to be maximized to be as close as possible to :math:`C`:

.. math::
   \max{\sum_{i \in S}{s_i}}

Taking this into account, one solution could be created and evaluated as follows:

.. note::

   jMetalPy assumes minimization by default. Therefore, we will have to multiply the solution objective by :math:`-1.0`.

.. code-block:: python

   def evaluate(self, solution: BinarySolution) -> BinarySolution:
       total_sum = 0.0

       for index, bits in enumerate(solution.variables[0]):
           if bits:
               total_sum += self.W[index]

       if total_sum > self.C:
           total_sum = self.C - total_sum * 0.1

           if total_sum < 0.0:
               total_sum = 0.0

       solution.objectives[0] = -1.0 * total_sum

       return solution

   def create_solution(self) -> BinarySolution:
       new_solution = BinarySolution(number_of_variables=self.number_of_variables,
                                     number_of_objectives=self.number_of_objectives)
       new_solution.variables[0] = \
           [True if random.randint(0, 1) == 0 else False for _ in range(self.number_of_bits)]

       return new_solution

Use case: Multi-objective Subset Sum
------------------------------------

The former problem can be formulated as a multi-objective binary problem whose objectives are as follows:

1. Maximize the sum of subsets to be as close as possible to :math:`C` and
2. Minimize the number of elements selected from :math:`W`.

This can be done by incorporating the objective function and increasing the number of objectives in consecuence:

.. code-block:: diff

   class SubsetSum(BinaryProblem):

      def __init__(self, C: int, W: list):
         super(SubsetSum, self).__init__(reference_front=None)
         self.C = C
         self.W = W

         self.number_of_bits = len(self.W)
   +     self.number_of_objectives = 2
         self.number_of_variables = 1
         self.number_of_constraints = 0

   +     self.obj_directions = [self.MAXIMIZE, self.MINIMIZE]
   +     self.obj_labels = ['Sum', 'No. of Objects']

      def evaluate(self, solution: BinarySolution) -> BinarySolution:
         total_sum = 0.0
   +     number_of_objects = 0

   +     for index, bits in enumerate(solution.variables[0]):
   +        if bits:
   +           total_sum += self.W[index]
   +              number_of_objects += 1

         if total_sum > self.C:
            total_sum = self.C - total_sum * 0.1

            if total_sum < 0.0:
                total_sum = 0.0

         solution.objectives[0] = -1.0 * total_sum
   +     solution.objectives[1] = number_of_objects

         return solution

API
------------------------

.. automodule:: jmetal.core.problem
   :members:
   :undoc-members:
   :show-inheritance:
