[![pypi
pacakge](https://img.shields.io/pypi/v/mpopt.svg)](https://pypi.org/project/mpopt)
[![Build
Status](https://travis-ci.org/mpopt/mpopt.svg?branch=master)](https://travis-ci.org/mpopt/mpopt.svg?branch=master)
[![Coverage
Status](https://coveralls.io/repos/github/mpopt/mpopt/badge.svg)](https://coveralls.io/github/mpopt/mpopt)
[![Documentation
Status](https://readthedocs.org/projects/mpopt/badge/?version=latest)](https://mpopt.readthedocs.io/en/latest/?badge=latest)
[![python](https://img.shields.io/pypi/pyversions/mpopt)](https://pypi.org/project/mpopt/)
[![continuous-integration](https://github.com/mpopt/mpopt/actions/workflows/python-package.yml/badge.svg)](https://github.com/mpopt/mpopt/actions/workflows/python-package.yml)
[![Downloads](https://static.pepy.tech/badge/mpopt)](https://pepy.tech/project/mpopt)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![License: LGPL v3](https://img.shields.io/badge/License-LGPL_v3-blue.svg)](https://www.gnu.org/licenses/lgpl-3.0)

MPOPT
=====

*MPOPT* is an open-source, extensible, customizable and easy
to use python package that includes a collection of modules to solve
multi-stage non-linear optimal control problems(OCP) using
pseudo-spectral collocation methods.

The package uses collocation methods to construct a Nonlinear programming problem (NLP) representation of OCP. The resulting NLP is then solved by algorithmic differentiation based [CasADi nlpsolver](https://casadi.sourceforge.net/v3.3.0/api/html/d4/d89/group__nlpsol.html)
( NLP solver supports multiple solver plugins including
[IPOPT](https://casadi.sourceforge.net/v3.3.0/api/html/d4/d89/group__nlpsol.html#plugin_Nlpsol_ipopt),
[SNOPT](https://casadi.sourceforge.net/v3.3.0/api/html/d4/d89/group__nlpsol.html#plugin_Nlpsol_snopt),
[sqpmethod](https://casadi.sourceforge.net/v3.3.0/api/html/d4/d89/group__nlpsol.html#plugin_Nlpsol_sqpmethod),
[scpgen](https://casadi.sourceforge.net/v3.3.0/api/html/d4/d89/group__nlpsol.html#plugin_Nlpsol_scpgen)).

Main features of the package are :

-   Customizable collocation approximation, compatible with
    Legendre-Gauss-Radau (LGR), Legendre-Gauss-Lobatto (LGL),
    Chebyshev-Gauss-Lobatto (CGL) roots.
-   Intuitive definition of single/multi-phase OCP.
-   Supports Differential-Algebraic Equations (DAEs).
-   Customized adaptive grid refinement schemes (Extendable)
-   Gaussian quadrature and differentiation matrices are evaluated using algorithmic differentiation, thus, supporting arbitrarily high number of collocation points limited only by the computational resources.
-   Intuitive post-processing module to retrieve and visualize the solution
-   Good test coverage of the overall package
-   Active development

Quick start
----------------

-   Install from the [Python Package Index](https://pypi.org/project/mpopt/) repository using the following terminal command, then copy paste the code from example below in a file (test.py) and run (python3 test.py) to confirm the installation.

```bash
pip install mpopt
```

-   (OR) Build directly from source (Terminal). Finally, `make run` to solve the moon-lander example described below.

```bash
git clone https://github.com/mpopt/mpopt.git --branch master
cd mpopt
make install
make test
```

A sample code to solve moon-lander OCP (2D) under 10 lines
-------------------------------------------------------------

**OCP** :
> Find optimal path, i.e Height ( $x_0$ ), Velocity ( $x_1$ ) and Throttle ( $u$ ) to reach the surface: Height (0m), Velocity (0m/s) from: Height (10m) and velocity(-2m/s) with: minimum fuel (u).

$$\begin{aligned}
&\min_{x, u}        & \qquad & J = 0 + \int_{t_0}^{t_f}u\ dt\\
&\text{subject to} &      & \dot{x_0} = x_1; \dot{x_1} = u - 1.5\\
 &                 &     & x_0(t_f) = 0; \ x_1(t_f) = 0\\
&                  &      & x_0(t_0) = 10; \ x_1(t_0) = -2\\
&                  &       & x_0 \geq 0; 0 \leq u \leq 3\\
&                 &     & t_0 = 0.0; t_f = \text{free variable}
\end{aligned}$$

```python
# Moon lander OCP direct collocation/multi-segment collocation

# from context import mpopt # (Uncomment if running from source)
from mpopt import mp

# Define OCP
ocp = mp.OCP(n_states=2, n_controls=1)
ocp.dynamics[0] = lambda x, u, t: [x[1], u[0] - 1.5]
ocp.running_costs[0] = lambda x, u, t: u[0]
ocp.terminal_constraints[0] = lambda xf, tf, x0, t0: [xf[0], xf[1]]
ocp.x00[0] = [10.0, -2.0]
ocp.lbu[0], ocp.ubu[0] = 0, 3
ocp.lbx[0][0] = 0

# Create optimizer(mpo), solve and post process(post) the solution
mpo, post = mp.solve(ocp, n_segments=20, poly_orders=3, scheme="LGR", plot=True)
x, u, t, _ = post.get_data()
mp.plt.show()
```

-  Experiment with different collocation schemes by changing "LGR" to "CGL" or "LGL" in the above script.
-  Update the grid to recompute solution (Ex. n_segments=3, poly_orders=[3, 30, 3]).
-  For a detailed demo of the mpopt features, refer the notebook [getting_started.ipynb](https://github.com/mpopt/mpopt/blob/master/getting_started.ipynb)

Resources
------------
-  Detailed implementation aspects of MPOPT are part of the [master thesis](https://github.com/mpopt/mpopt/blob/01f4612ec84a5f6bec8f694c19b129d9fbc12527/docs/Devakumar-Master-Thesis-Report.pdf).
-  Documentation at [mpopt.readthedocs.io/](mpopt.readthedocs.io/)

Features and Limitations
---------------------------
While MPOPT is able to solve any Optimal control problem formulation in the Bolza form, the present limitations of MPOPT are,

- Only continuous functions and derivatives are supported
- Dynamics and constraints are to be written in CasADi variables (Familiarity with casadi variables and expressions is expected)
- The adaptive grid though successful in generating robust solutions for simple problems, doesn't have a concrete proof on convergence.


Authors
=======

-   **Devakumar THAMMISETTY**
-   **Prof. Colin Jones** (Co-author)

License
=======

This project is licensed under the GNU LGPL v3 - see the
[LICENSE](https://github.com/mpopt/mpopt/blob/master/LICENSE) file for
details

Acknowledgements
================

-   **Petr Listov**

Cite
=====

-  D. Thammisetty, “Development of a multi-phase optimal control software for aerospace applications (mpopt),” Master’s thesis, Lausanne, EPFL, 2020.

**BibTex entry**:

    @mastersthesis{thammisetty2020development,
          title={Development of a multi-phase optimal control software for aerospace applications (mpopt)},
          author={Thammisetty, Devakumar},
          year={2020},
          school={Master’s thesis, Lausanne, EPFL}}
