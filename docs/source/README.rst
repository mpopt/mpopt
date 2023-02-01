Getting started
--------------------

-  Install from `PyPI <https://pypi.org/project/mpopt/>`_ using the
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

Solve moon-lander OCP in under 10 lines
-----------------------------------------

**OCP** :

Find optimal path, i.e Height ( :math:`x_0` ), Velocity (
:math:`x_1` ) and Throttle ( :math:`u` ) to reach the surface: Height
(0m), Velocity (0m/s) from Height (10m) and velocity(-2m/s) with minimum
fuel (u).

.. math::

   \begin{aligned}
   &\min_{x, u}        & \qquad & J = 0 + \int_{t_0}^{t_f}u\ dt\\
   &\text{subject to} &      & \dot{x_0} = x_1; \dot{x_1} = u - 1.5\\
   &                  &       & x_0 \geq 0; 0 \leq u \leq 3\\
   &                  &      & x_0(t_0) = 10; \ x_1(t_0) = -2\\
    &                 &     & x_0(t_f) = 0; \ x_1(t_f) = 0\\
    &                 &     & t_0 = 0.0; t_f = \text{free variable}
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
-  Update the grid to recompute solution (Ex. n_segments=3,
   poly_orders=[3, 30, 3]).
-  For a detailed demo of the mpopt features, refer the notebook
   `getting_started.ipynb <https://github.com/mpopt/mpopt/blob/master/docs/notebooks/getting_started.ipynb>`_

Resources
------------
-  Detailed implementation aspects of MPOPT are part of the `master thesis <https://github.com/mpopt/mpopt/blob/01f4612ec84a5f6bec8f694c19b129d9fbc12527/docs/Devakumar-Master-Thesis-Report.pdf>`_.
-  Documentation at `mpopt.readthedocs.io <mpopt.readthedocs.io>`_

Features and Limitations
---------------------------
While MPOPT is able to solve any Optimal control formulation in the Bolza form, the present limitations of MPOPT are,

- Only continuous functions and derivatives are supported
- Dynamics and constraints are to be written in CasADi variables (Familiarity with casadi variables and expressions is expected)
- The adaptive grid though successful in generating robust solutions for simple problems, doesnt have a concrete proof on convergence.
