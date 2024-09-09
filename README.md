# aintelope

TODO: Intro for biological compatibility and cooperation themes.


## Project setup

### Installation

The project installation is managed via `make` and `pip`. Please see the
respective commanads in the `Makefile`. To setup the environment follow these
steps:

0. Install CPython from python.org. The code is tested with Python version 3.10.10
1. Create a virtual python environment: `make venv`
2. Activate the environment: `source venv_aintelope/bin/activate`
3. Install dependencies: `make install`

For development and testing follow (active environment):

1. Install development dependencies: `make install-dev`
2. Install project locally: `make build-local`
3. Run tests: `make tests-local`

### Code formatting and style

To automatically sort the imports you can run
[`isort aintelope tests`](https://github.com/PyCQA/isort) from the root level of the project.
To autoformat python files you can use [`black .`](https://github.com/psf/black) from the root level of the project.
Configurations of the formatters can be found in `pyproject.toml`.
For linting/code style use [`flake8`](https://flake8.pycqa.org/en/latest/).

These tools can be invoked via `make`:

```bash
make isort
make format
make flake8
```

## Executing `aintelope`

Try `make run-training`. Then look in `aintelope/outputs/memory_records`. (WIP)
There should be two new files named `Record_{current timestamp}.csv` and
`Record_{current timestamp}_plot.png`. The plot will be an image of the path the
agent took during the test episode, using the best agent that the training
produced. Green dots are food in the environment, blue dots are water.

TODO


## Logging

TODO


# Windows

Aintelope code base is compatible with Windows. No extra steps needed. GPU computation works fine as well. WSL is not needed.
