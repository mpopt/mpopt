:github_url: https://github.com/mpopt/mpopt/blob/docs/docs/source/index.rst

.. title:: Table of contents

########################################
MPOPT : Optimal control problem solver
########################################

Simple to use, optimal control problem solver library in Python `GitHub <https://github.com/mpopt/mpopt/>`_.

*MPOPT* libray is an open-source, extensible, customizable and easy to use
python package that includes a collection of modules to solve
multi-stage non-linear optimal control problems(OCP) in the standard Bolza form using
pseudo-spectral collocation methods.

.. toctree::
    :maxdepth: 1

    introduction.rst
    installation.rst
    getting_started.rst
    examples.rst
    notebooks.rst
    documentation.rst
    developer_notes.rst

A pdf version of this documentation can be downloaded from `PDF document <https://mpopt.readthedocs.io/_/downloads/en/latest/pdf/>`_

Resources
===========
-  Detailed implementation aspects of MPOPT are part of the `master thesis <http://dx.doi.org/10.13140/RG.2.2.19519.79528>`_.
-  Quick introduction `presentation <http://dx.doi.org/10.13140/RG.2.2.14486.63040>`_.
-  Documentation at `mpopt.readthedocs.io <mpopt.readthedocs.io>`_
-  List of solved `examples <Examples>`
-  Features of MPOPT in `Jupyter Notebooks <Notebooks>`

Features and Limitations
===============================
While MPOPT is able to solve any Optimal control formulation in the Bolza form, the present limitations of MPOPT are,

- Only continuous functions and derivatives are supported
- Dynamics and constraints are to be written in CasADi variables (Familiarity with casadi variables and expressions is expected)
- The adaptive grid though successful in generating robust solutions for simple problems, doesnt have a concrete proof on convergence.

Authors
========

-  **Devakumar THAMMISETTY**
-  **Prof. Colin Jones** (Co-author)

License
=======

This project is licensed under the GNU LGPL v3 - see the
`LICENSE <https://github.com/mpopt/mpopt/blob/master/LICENSE>`__ file
for details

Acknowledgments
===================

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
