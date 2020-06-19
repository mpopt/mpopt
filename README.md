[![pypi pacakge](https://img.shields.io/pypi/v/mpopt.svg)](https://pypi.org/project/mpopt)
[![Build Status](https://travis-ci.org/mpopt/mpopt.svg?branch=master)](https://travis-ci.org/mpopt/mpopt.svg?branch=master)
[![Coverage Status](https://coveralls.io/repos/github/mpopt/mpopt/badge.svg)](https://coveralls.io/github/mpopt/mpopt)

### MPOPT

*MPOPT* is a collection of modules to solve multi-stage optimal control problems(OCPs) using pseudo-spectral collocation method. This module creates Nonlinear programming problem (NLP) from the given OCP description, which is then solved by CasADi nlpsolver using various available plugins such as *ipopt*, *snopt* etc.

Main features of the solver are :

* Customizable collocation approximation, compatable with Legendre-Gauss-Radau, Legendre-Gauss-Lobatto, Chebyshev-Gauss-Lobatto roots.
* Intuitive definition of OCP/multi-phase OCP
* Single-phase as well as multi-phase OCP solving capability using user defined collocation approximation
* Adaptive grid refinement schemes for robust solutions
* NLP solution using algorithmic differentiation capability offered by [CasADi](https://web.casadi.org/), multiple NLP solver compatibility 'ipopt', 'snopt', 'sqpmethod' etc.
* Sophisticated post-processing module for interactive data visualization

### Installation

Install the package using

```
$ pip install mpopt
```

If you want to downloaded it from source, you may do so either by:

- Downloading it from [GitHub](https://github.com/mpopt/mpopt) page
    - Unzip the folder and you are ready to go
- Or cloning it to a desired directory using git:
    - ```$ git clone https://github.com/mpopt/mpopt.git```

```
$ make init
$ make test
$ python examples/moon_lander.py
```

### Getting started

A brief overview of the package and capabilities are demonstrated with simple moon-lander OCP example in Jupyter notebook.

- Get started with [MPOPT](https://github.com/mpopt/mpopt/blob/master/getting_started.ipynb)

### Documentation

Work under progress.

### A sample code to solve moon-lander OCP (2D)
```python
# Moon lander OCP direct collocation/multi-segment collocation
from mpopt import mp

# Define OCP
ocp = mp.OCP(n_states=2, n_controls=1)
ocp.dynamics[0] = lambda x, u, t: [x[1], u[0] - 1.5]
ocp.running_costs[0] = lambda x, u, t: u[0]
ocp.terminal_constraints[0] = lambda xf, tf, x0, t0: [xf[0], xf[1]]
ocp.x00[0] = [10.0, -2.0]
ocp.lbu[0], ocp.ubu[0] = 0, 3

# Create optimizer(mpo), solve and post process(post) the solution
mpo, post = mp.solve(ocp, n_segments=20, poly_orders=3, scheme="LGR", plot=True)
```

## Authors

* **Devakumar THAMMISETTY**
* **Prof. Colin Jones** (Co-author)


## License

This project is licensed under the GNU LGPL v3 - see the [LICENSE](https://github.com/mpopt/mpopt/blob/master/LICENSE) file for details

## Acknowledgements

* **Petr Listov**
