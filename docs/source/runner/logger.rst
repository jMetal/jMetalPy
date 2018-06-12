Set up logger
========================

jMetalPy support the default Python's logging mechanism. This code will record logging events in a file and print them on console:

.. code-block:: python

   logging.basicConfig(
       level=logging.INFO,
       format="%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s",
       handlers=[
           logging.FileHandler('jmetalpy.log'),
           logging.StreamHandler()
       ]
   )