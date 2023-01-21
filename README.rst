|pypi pacakge| |Build Status| |Coverage Status| |Documentation Status|

MPOPT
=====

*MPOPT* is a open-source, extensible, customizable and easy to use
python package that includes a collection of modules to solve
multi-stage non-linear optimal control problems(OCP) using
pseudo-spectral collocation methods.

The package uses collocation methods to construct a Nonlinear
programming problem (NLP) representation of OCP. The resulting NLP is
then solved by algorithmic differentiation based `CasADi
nlpsolver <https://casadi.sourceforge.net/v3.3.0/api/html/d4/d89/group__nlpsol.html>`__
( NLP solver supports multiple solver plugins including
`IPOPT <https://casadi.sourceforge.net/v3.3.0/api/html/d4/d89/group__nlpsol.html#plugin_Nlpsol_ipopt>`__,
`SNOPT <https://casadi.sourceforge.net/v3.3.0/api/html/d4/d89/group__nlpsol.html#plugin_Nlpsol_snopt>`__,
`sqpmethod <https://casadi.sourceforge.net/v3.3.0/api/html/d4/d89/group__nlpsol.html#plugin_Nlpsol_sqpmethod>`__,
`scpgen <https://casadi.sourceforge.net/v3.3.0/api/html/d4/d89/group__nlpsol.html#plugin_Nlpsol_scpgen>`__).

Main features of the package are :

-  Customizable collocation approximation, compatible with
   Legendre-Gauss-Radau (LGR), Legendre-Gauss-Lobatto (LGL),
   Chebyshev-Gauss-Lobatto (CGL) roots.
-  Intuitive definition of single/multi-phase OCP.
-  Supports Differential-Algebraic Equations (DAEs).
-  Customized adaptive grid refinement schemes (Extendable)
-  Gaussian quadrature and differentiation matrices are evaluated using
   algorithmic differentiation, thus, supporting arbitrarily high number
   of collocation points limited only by the computational resources.
-  Intuitive post-processing module to retrieve and visualize the
   solution
-  Good test coverage of the overall package
-  Active development

Quick start
===========

-  Install from `PyPI <https://pypi.org/project/mpopt/>`__ using the
   following terminal command, then copy paste the code from example
   below in a file (test.py) and run (python3 test.py) to confirm the
   installation.

.. code:: bash

   pip install mpopt

-  (OR) Build directly from source (Terminal). Finally, ``make run`` to
   solve the moon-lander example described below.

.. code:: bash

   git clone https://github.com/mpopt/mpopt.git --branch master
   cd mpopt
   make install
   make test

A sample code to solve moon-lander OCP (2D) under 10 lines
----------------------------------------------------------

**OCP** : > Find optimal path, i.e Height ( :math:`x_0` ), Velocity (
:math:`x_1` ) and Throttle ( :math:`u` ) to reach the surface: Height
(0m), Velocity (0m/s) from Height (10m) and velocity(-2m/s) with minimum
fuel (u).

.. math::

   \begin{aligned}
   &\min_{x, u}        & \qquad & J = 0 + \int_{t_0}^{t_f}u\ dt\\
   &\text{subject to} &      & \dot{x_0} = x_1; \dot{x_1} = u - 1.5; x_0 \geq 0; 0 \leq u \leq 3\\
   &                  &      & x_0(t_0) = 10; \ x_1(t_0) = -2; t_0 = 0.0; x_0(t_f) = 0; \ x_1(t_f) = 0; t_f = \text{free variable}
   \end{aligned}

.. code:: python

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

   # Create optimizer(mpo), solve and post process(post) the solution
   mpo, post = mp.solve(ocp, n_segments=20, poly_orders=3, scheme="LGR", plot=True)
   x, u, t, _ = post.get_data()
   mp.plt.show()

-  Experiment with different collocation schemes by changing “LGR” to
   “CGL” or “LGL” in the above script.
-  Update the grid (n_segments, poly_orders) to recompute solution (Ex.
   n_segments=3, poly_orders=[3, 30, 3]).
-  For a detailed demo of the mpopt features, refer the notebook
   `getting_started.ipynb <https://github.com/mpopt/mpopt/blob/master/getting_started.ipynb>`__

Results
-------

`!Non-adaptive grid <docs/plots/moon_lander_gh.png>`__ `!Adaptive grid
(Equal residual segments) <docs/plots/ml_h_ad_eq_res.png>`__ `!Adaptive
grid <docs/plots/ml_ad.png>`__

Authors
=======

-  **Devakumar THAMMISETTY**
-  **Prof. Colin Jones** (Co-author)

License
=======

This project is licensed under the GNU LGPL v3 - see the
`LICENSE <https://github.com/mpopt/mpopt/blob/master/LICENSE>`__ file
for details

Acknowledgements
================

-  **Petr Listov**

.. |pypi pacakge| image:: https://img.shields.io/pypi/v/mpopt.svg
   :target: https://pypi.org/project/mpopt
.. |Build Status| image:: https://travis-ci.org/mpopt/mpopt.svg?branch=master
   :target: https://travis-ci.org/mpopt/mpopt.svg?branch=master
.. |Coverage Status| image:: https://coveralls.io/repos/github/mpopt/mpopt/badge.svg
   :target: https://coveralls.io/github/mpopt/mpopt
.. |Documentation Status| image:: https://readthedocs.org/projects/mpopt/badge/?version=latest
   :target: https://mpopt.readthedocs.io/en/latest/?badge=latest
