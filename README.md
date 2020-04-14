# storm-project-starter-python
Starter project for the Python API of Storm via Stormpy

## Getting Started
Before starting, make sure that Storm and stormpy are installed. If not, see the [documentation](https://moves-rwth.github.io/stormpy/installation.html) for details on how to install stormpy.

First, install the Python package. If you use a virtual environment, make sure to use it.
To install the starter package, execute
```
python setup.py develop
```

Then, run the script using 
```
python stormpy_starter/check.py --model examples/die.pm --property examples/die.pctl
```
The answer should be no.

Then, run the script using 
```
python stormpy_starter/check.py --model examples/die.pm --property examples/die2.pctl
```
The answer should be yes.

## What is next?
You are all set to implement your own tools and algorithms on top of stormpy.
Feel free to contribute your new algorithms to stormpy, such that others can enjoy them.