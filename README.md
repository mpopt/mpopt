[![pypi pacakge](https://img.shields.io/pypi/v/mpopt.svg)](https://pypi.org/project/mpopt)
[![Build Status](https://travis-ci.org/mpopt/mpopt.svg?branch=master)](https://travis-ci.org/mpopt/mpopt.svg?branch=master)
[![Coverage Status](https://coveralls.io/repos/github/mpopt/mpopt/badge.svg)](https://coveralls.io/github/mpopt/mpopt)

### MPOPT

*MPOPT* is a collection of modules to solve multi-stage optimal control problems(OCPs) using pseudo-spectral collocation method. This module creates Nonlinear programming problem (NLP) from the given OCP description, which is then solved by CasADi nlpsolver using various available plugins such as *ipopt*, *snopt* etc.

Main features of the solver are :

### Examples
* Single-phase OCPs
    - Moon lander (2-states, 1-control)
    - Van der pol oscillator ocp (2-states, 1-control)
    - Hyper-sensitive problem (1-state, 1-control)

* Multi-stage OCPs
    - Two-phase schwartz OCP (2 phases, 2-states, 1-control)
    - Multi-stage launch vehicle trajectory optimization (4-phases, 7-states, 3-controls)
