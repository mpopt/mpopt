:github_url: https://github.com/mpopt/mpopt/blob/docs/docs/source/installation.rst

.. title:: Installation

##########################
Installation
##########################

-  Install from `PyPI <https://pypi.org/project/mpopt/>`_ using the
   following terminal command. Refer getting started to solve OCPs.

.. code:: bash

   pip install mpopt

-  (OR) Build directly from source (Terminal). Finally, ``make run`` to
   solve the moon-lander example described below.

.. code:: bash

   git clone https://github.com/mpopt/mpopt.git --branch master
   cd mpopt
   make install
   make test

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

Next steps: `Getting started <Getting started>`, `Examples <Examples>`
