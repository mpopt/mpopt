|pypi pacakge| |Build Status| |Coverage Status| |Documentation Status|

MPOPT
~~~~~

*MPOPT* is a collection of modules to solve multi-stage optimal control
problems(OCPs) using pseudo-spectral collocation method. This module
creates Nonlinear programming problem (NLP) from the given OCP
description, which is then solved by CasADi nlpsolver using various
available plugins such as *ipopt*, *snopt* etc.

Main features of the solver are :

-  Customizable collocation approximation, compatable with
   Legendre-Gauss-Radau, Legendre-Gauss-Lobatto, Chebyshev-Gauss-Lobatto
   roots.
-  Intuitive definition of OCP/multi-phase OCP
-  Single-phase as well as multi-phase OCP solving capability using user
   defined collocation approximation
-  Adaptive grid refinement schemes for robust solutions
-  NLP solution using algorithmic differentiation capability offered by
   `CasADi <https://web.casadi.org/>`__, multiple NLP solver
   compatibility 'ipopt', 'snopt', 'sqpmethod' etc.
-  Sophisticated post-processing module for interactive data
   visualization

Getting started
~~~~~~~~~~~~~~~

A brief overview of the package and capabilities are demonstrated with
simple moon-lander OCP example in Jupyter notebook.

-  Get started with
   `MPOPT <https://github.com/mpopt/mpopt/blob/master/getting_started.ipynb>`__

Installation
~~~~~~~~~~~~

Install and try the package using

::

    $ pip install mpopt
    $ wget https://raw.githubusercontent.com/mpopt/mpopt/master/examples/moon_lander.py
    $ python3 moon_lander.py

If you want to downloaded it from source, you may do so either by:

-  Downloading it from `GitHub <https://github.com/mpopt/mpopt>`__ page

   -  Unzip the folder and you are ready to go

-  Or cloning it to a desired directory using git:

   -  ``$ git clone https://github.com/mpopt/mpopt.git --branch master``

-  Install package using

   -  ``$ make install``

-  Test installation using

   -  ``$ make test``

-  Try moon-lander example using

   -  ``$ make run``

Documentation
~~~~~~~~~~~~~

-  Refer `Documentation <https://mpopt.readthedocs.io/en/latest/>`__

A sample code to solve moon-lander OCP (2D)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

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

Authors
~~~~~~~

-  **Devakumar THAMMISETTY**
-  **Prof. Colin Jones** (Co-author)

License
~~~~~~~

This project is licensed under the GNU LGPL v3 - see the
`LICENSE <https://github.com/mpopt/mpopt/blob/master/LICENSE>`__ file
for details

Acknowledgements
~~~~~~~~~~~~~~~~

-  **Petr Listov**

.. |pypi pacakge| image:: https://img.shields.io/pypi/v/mpopt.svg
   :target: https://pypi.org/project/mpopt
.. |Build Status| image:: https://travis-ci.org/mpopt/mpopt.svg?branch=master
   :target: https://travis-ci.org/mpopt/mpopt.svg?branch=master
.. |Coverage Status| image:: https://coveralls.io/repos/github/mpopt/mpopt/badge.svg
   :target: https://coveralls.io/github/mpopt/mpopt
.. |Documentation Status| image:: https://readthedocs.org/projects/mpopt/badge/?version=latest
   :target: https://mpopt.readthedocs.io/en/latest/?badge=latest
