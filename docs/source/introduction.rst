.. title::Introduction

#################
Introduction
#################

MPOPT package implements libraries to solve multi-stage or single-stage optimal control problems of the Bolza form given below,

.. math::

  \begin{aligned}
  &\!\min_{x, u, t_0, t_f, s}        & \qquad & J = M(x_0, t_0, x_f, t_f, s) + \int_{0}^{t_f}L(x, u, t, s)dt\\
  &\text{subject to} &      & \Dot{x} = f(x, u, t, s) \\
  &                  &      & g(x, u, t, s) \leq 0  \\
  &                  &      & h(x_0, t_0, x_f, t_f, s) = 0
  \end{aligned}

The package uses collocation methods to construct a Nonlinear
programming problem (NLP) representation of OCP. The resulting NLP is
then solved by algorithmic differentiation based `CasADi
nlpsolver <https://casadi.sourceforge.net/v3.3.0/api/html/d4/d89/group__nlpsol.html>`_
( NLP solver supports multiple solver plugins including
`IPOPT <https://casadi.sourceforge.net/v3.3.0/api/html/d4/d89/group__nlpsol.html#plugin_Nlpsol_ipopt>`_,
`SNOPT <https://casadi.sourceforge.net/v3.3.0/api/html/d4/d89/group__nlpsol.html#plugin_Nlpsol_snopt>`_,
`sqpmethod <https://casadi.sourceforge.net/v3.3.0/api/html/d4/d89/group__nlpsol.html#plugin_Nlpsol_sqpmethod>`_,
`scpgen <https://casadi.sourceforge.net/v3.3.0/api/html/d4/d89/group__nlpsol.html#plugin_Nlpsol_scpgen>`_).

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

Access quick introduction `presentation PDF <http://dx.doi.org/10.13140/RG.2.2.14486.63040>`_.

Similar solver packages: `GPOPS II <https://www.gpops2.com/>`_, `ICLOCS2 <http://www.ee.ic.ac.uk/ICLOCS/>`_, `PSOPT <https://www.psopt.net/>`_, `DIRCOL <https://www.sim.informatik.tu-darmstadt.de/en/res/sw/dircol/>`_, `DIDO <https://elissarglobal.com/get-dido/>`_, `SOS <https://www.astos.de/products/sos/details>`_, `ACADO <https://acado.github.io/>`_, `OPTY <https://opty.readthedocs.io/>`_

Next steps: `Installation <Installation>`, `Getting started <Getting started>`
