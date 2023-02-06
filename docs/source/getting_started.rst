:github_url: https://github.com/mpopt/mpopt/blob/docs/docs/source/getting_started.rst

.. title:: Getting started

.. _getting-started:

###################
Getting started
###################

Solve moon-lander OCP in under 10 lines
-----------------------------------------

**OCP** :

Find optimal path, i.e Height ( :math:`x_0` ), Velocity (
:math:`x_1` ) and Throttle ( :math:`u` ) to reach the surface: Height
(0m), Velocity (0m/s) from Height (10m) and velocity(-2m/s) with minimum
fuel (u).

.. math::

   \begin{aligned}
   & \min_{x, u}        & \qquad & J = 0 + \int_{t_0}^{t_f}u\ dt\\
   & \text{subject to} &      & \dot{x_0} = x_1; \dot{x_1} = u - 1.5\\
    &                 &     & x_0(t_f) = 0; \ x_1(t_f) = 0\\
   &                  &      & x_0(t_0) = 10; \ x_1(t_0) = -2\\
   &                  &       & x_0 \geq 0; 0 \leq u \leq 3\\
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
   ocp.lbx[0][0] = 0.0
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

For issues related to getting started examples, refer `issues <https://github.com/mpopt/mpopt/discussions/13>`_

Next steps: :doc:`./examples`_, :doc:`./notebooks`_

.. toctree::
   :maxdepth: 1

   notebooks/getting_started.ipynb
