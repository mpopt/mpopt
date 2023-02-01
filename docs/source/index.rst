.. mpopt documentation master file, created by
   sphinx-quickstart on Fri Jun 19 11:46:01 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

MPOPT: An optimal control problem solver
=============================================

.. toctree::
   :maxdepth: 2
   :caption: Contents

   README
   MPOPTDOC

*MPOPT* is an open-source, extensible, customizable and easy to use
python package that includes a collection of modules to solve
multi-stage non-linear optimal control problems(OCP) using
pseudo-spectral collocation methods.

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


Authors
========

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

Cite
=====

-  D. Thammisetty, “Development of a multi-phase optimal control software for aerospace applications (mpopt),” Master’s thesis, Lausanne, EPFL, 2020.

**BibTex entry**:

   @mastersthesis{thammisetty2020development,
         title={Development of a multi-phase optimal control software for aerospace applications (mpopt)},
         author={Thammisetty, Devakumar},
         year={2020},
         school={Master’s thesis, Lausanne, EPFL}}

Indices and tables
=====================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
