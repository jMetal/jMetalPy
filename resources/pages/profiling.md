# Profiling with Python using `cProfile`

Python allow us to collect and analize statistics about it's performance using the profiler module.
This can be done by running

`CProfile` is one of the three built-in modules of Python. [It is recommended for most users](https://docs.python.org/2/library/profile.html#introduction-to-the-profilers).

This way we can know which functions are faster than others. The module can be invoked as a script:

```bash
python -m cProfile -o [OUTPUT.FILE] [SCRIPT.PY]
```

When `-o` is not supplied, we can use `-s` to specify one of the `sort_stats()` sort values to sort the output by. 

## Reading profile results

The `pstats.Stats` class allow us to examine the profile data: 

```python
import pstats

p = pstats.Stats('restats')
p.strip_dirs().sort_stats(-1).print_stats()
```

# Memory usage

`memory_profiler` is a python module for monitoring memory consumption of a process as well as line-by-line analysis of memory consumption for python programs

To use it, simply install with

```bash
 pip install -U memory_profiler 
 pip install psutil
```

From its website:
> (Installing the psutil package here is recommended because it greatly improves the performance of the memory_profiler).

Then decorate a function with `@profile` to view line-by-line memory usage:

```python
from memory_profiler import profile

@profile
def function():
    ...
```

And execute:

```bash
python -m memory_profiler example.py
```