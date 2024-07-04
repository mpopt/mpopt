#  -*- coding: utf-8 -*-
#
# Copyright (c) 2020 LA EPFL.
#
# This file is part of MPOPT
# (see http://github.com/mpopt).
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.
#
from typing import List, Dict, Tuple, Set, TYPE_CHECKING, Optional, Callable
import numpy as np
import itertools
import casadi as ca  # type: ignore
import matplotlib.pyplot as plt
import copy
import math
import time


class mpopt:
    """Multiphase Optimal Control Problem Solver

    This is the base class, implementing the OCP discretization, transcription
    and calls to NLP solver

    Examples :
        >>> # Moon lander problem
        >>> from mpopt import mp
        >>> ocp = mp.OCP(n_states=2, n_controls=1, n_phases=1)
        >>> ocp.dynamics[0] = lambda x, u, t: [x[1], u[0] - 1.5]
        >>> ocp.running_costs[0] = lambda x, u, t: u[0]
        >>> ocp.terminal_constraints[0] = lambda xf, tf, x0, t0: [xf[0], xf[1]]
        >>> ocp.x00[0] = [10, -2]
        >>> ocp.lbu[0] = 0; ocp.ubu[0] = 3
        >>> ocp.lbtf[0] = 3; ocp.ubtf[0] = 5
        >>> opt = mp.mpopt(ocp, n_segments=20, poly_orders=[3]*20)
        >>> solution = opt.solve()
        >>> post = opt.process_results(solution, plot=True)
    """

    _GRID_TYPE = "fixed"  # mid-points, spectral, fixed
    _MAX_GRID_POINTS = 15  # Per phase
    _MUTE_ = False

    def __init__(
        self: "mpopt",
        problem: "OCP",
        n_segments: int = 1,
        poly_orders: List[int] = [9],
        scheme: str = "LGR",
        **kwargs,
    ):
        """Initialize the optimizer
        args:
            n_segments: number of segments in each phase
            poly_orders: degree of the polynomial in each segment
            problem: instance of the OCP class
        """
        self.n_segments = n_segments

        # if poly_orders is an integer, convert it to list with n_segments elements
        self.poly_orders = (
            [poly_orders] * n_segments if isinstance(poly_orders, int) else poly_orders
        )

        self._ocp = copy.deepcopy(problem)
        self.colloc_scheme = scheme  # available LGR, LGL, CGL
        self.reset_mpopt()

    def reset_mpopt(self):
        # Assert that poly_orders is defined for all the segments
        assert len(self.poly_orders) == self.n_segments
        self._Npoints = sum(self.poly_orders) + 1

        # Set the status of internal function evaluations
        self._collocation_approximation_computed = False
        self._variables_created = False
        self._nlpsolver_initialized = False
        self.grid_type = [self._GRID_TYPE for _ in range(self._ocp.n_phases)]
        self.max_grid_points = [
            self._MAX_GRID_POINTS for _ in range(self._ocp.n_phases)
        ]

    def compute_numerical_approximation(self, scheme: str = None) -> None:
        if scheme is None:
            scheme = self.colloc_scheme
        self.collocation = Collocation(self.poly_orders, scheme)
        self._compD = self.collocation.get_composite_differentiation_matrix()
        self._compW = self.collocation.get_composite_quadrature_weights()
        self._taus = self.collocation.roots
        self.tau0, self.tau1 = self.collocation.tau0, self.collocation.tau1
        self._collocation_approximation_computed = True

    def create_variables(self) -> None:
        """Create casadi variables for states, controls, time and segment widths
        which are used in NLP transcription

        Initialized casadi varaibles for optimization.

        args: None
        returns : None
        """
        self.X = ca.SX.sym("x", self._Npoints, self._ocp.nx, self._ocp.n_phases)
        self.U = ca.SX.sym("u", self._Npoints, self._ocp.nu, self._ocp.n_phases)
        self.A = ca.SX.sym("a", self._ocp.na, self._ocp.n_phases)
        self.t0 = ca.SX.sym("t0", self._ocp.n_phases)
        self.tf = ca.SX.sym("tf", self._ocp.n_phases)

        # Initialize scaling matrices
        self._scX = ca.diag(ca.vertcat(self._ocp.scale_x))
        self.__invScX = ca.solve(self._scX, np.eye(self._ocp.nx))

        self.__scU = ca.diag(ca.vertcat(self._ocp.scale_u))
        self.__invScU = ca.solve(self.__scU, np.eye(self._ocp.nu))

        self._scA = ca.diag(ca.vertcat(self._ocp.scale_a))
        self.__invScA = ca.solve(self._scA, np.eye(self._ocp.na))

        self._optimization_vars_per_phase = (
            self._Npoints * (self._ocp.nx + self._ocp.nu)
            + self._ocp.n_phases * self._ocp.na
            + 2
        )

        self.time_grid = [
            [None for i in range(self._Npoints)] for _ in range(self._ocp.n_phases)
        ]

        # Create segment width, varies depending on if its adaptive method or equal width scheme
        self.init_segment_width()
        self._variables_created = True

    def init_segment_width(self) -> None:
        """Initialize segment width in each phase

        Segment width is normalized so that sum of all the segment widths equal 1

        args: None
        returns: None
        """
        self.seg_widths = ca.SX.sym("h_seg", self.n_segments, self._ocp.n_phases)

    def get_discretized_dynamics_constraints_and_cost_matrices(
        self, phase: int = 0
    ) -> Tuple:
        """Get discretized dynamics, path constraints and running cost at each collocation node in a list

        args:
            :phase: index of phase

        returns:
            Tuple : (f, c, q)
                f - List of constraints for discretized dynamics
                c - List of constraints for discretized path constraints
                q - List of constraints for discretized running costs
        """
        # Initialize variables to store dynamics(f), contraints(c) and cost function(g) at each collocation point
        f, q = [None] * self._Npoints, [None] * self._Npoints
        # Check if OCP has path constraints
        has_constraints = self._ocp.has_path_constraints(phase)
        c = [None] * self._Npoints if has_constraints else []

        # Get unscaled starting and final times for given phase
        t0 = self.t0[phase] / self._ocp.scale_t
        tf = self.tf[phase] / self._ocp.scale_t
        a = ca.mtimes(self.__invScA, self.A[:, phase])

        # Get unscaled terminal states and controls to evaluate terminal cost and constraints
        t_seg0 = t0
        seg, point = 0, 0

        # Initialize starting segment width (Unscaled)
        h_seg = (tf - t0) / (self.tau1 - self.tau0) * self.seg_widths[seg, phase]
        # Populate the discretized dynamics, constraints and cost vectors
        dynamics = self._ocp.get_dynamics(phase)
        path_constraints = self._ocp.get_path_constraints(phase)
        running_costs = self._ocp.get_running_costs(phase)
        for index in range(self._Npoints):
            if point > self.poly_orders[seg]:
                seg, point = seg + 1, 1
                t_seg0 += h_seg * (self.tau1 - self.tau0)
                h_seg = (
                    (tf - t0) / (self.tau1 - self.tau0) * self.seg_widths[seg, phase]
                )
            x = ca.mtimes(self.__invScX, self.X[phase][index, :].T)
            u = ca.mtimes(self.__invScU, self.U[phase][index, :].T)
            t = t_seg0 + h_seg * (self._taus[self.poly_orders[seg]][point] - self.tau0)

            # discretize dynamics
            f[index] = h_seg * ca.mtimes(self._scX, ca.vertcat(*dynamics(x, u, t, a))).T
            # discretize path_constraints
            if has_constraints:
                c[index] = ca.vertcat(*path_constraints(x, u, t, a)).T
            # running cost
            q[index] = h_seg * ca.vertcat(running_costs(x, u, t, a)).T

            point += 1
            self.time_grid[phase][index] = t

        return (f, c, q)

    def get_nlp_constraints_for_dynamics(self, f: List = [], phase: int = 0) -> Tuple:
        """Get NLP constraints for discretized dynamics

        args:
           :f: Discretized vector of dynamics function evaluated at collocation nodes
           :phase: index of the phase corresponding to the given dynamics

         returns:
            Tuple : (F, Fmin, Fmax)
                F - CasADi vector of constraints for the dynamics
                Fmin - Respective lowever bound vector
                Fmax - Respective upper bound vector
        """
        # Estimate constraint vector and bounds for the collocated dynamics
        _compD = (
            ca.kron(ca.DM.eye(self._ocp.nx), self._compD)
            if self._ocp.nx > 1
            else self._compD
        )
        F = ca.mtimes(_compD, self.X[phase][:]) - ca.vertcat(*f)[:]
        nF = F.size1()
        Fmin = [self._ocp.LB_DYNAMICS] * (nF)
        Fmax = [self._ocp.UB_DYNAMICS] * (nF)

        return (F, Fmin, Fmax)

    def get_nlp_constraints_for_path_contraints(
        self, c: List = [], phase: int = 0
    ) -> Tuple:
        """Get NLP constraints for discretized path constraints

        args:
           :c: Discretized vector of path constraints evaluated at collocation nodes
           :phase: index of the corresponding phase

         returns:
            Tuple : (C, Cmin, Cmax)
                C - CasADi vector of constraints for the path constraints
                Cmin - Respective lowever bound vector
                Cmax - Respective upper bound vector
        """
        if c:
            C = ca.vertcat(*c)[:]
            nC = C.size1()
            Cmin = [self._ocp.LB_PATH_CONSTRAINTS] * (nC)
            Cmax = [self._ocp.UB_PATH_CONSTRAINTS] * (nC)
        else:
            C, Cmin, Cmax = [], [], []

        return (C, Cmin, Cmax)

    def get_nlp_constraints_for_terminal_contraints(self, phase: int = 0) -> Tuple:
        """Get NLP constraints for discretized terminal constraints

        args:
           :phase: index of the corresponding phase

         returns:
            Tuple : (TC, TCmin, TCmax, J)
                TC - CasADi vector of constraints for the terminal constraints
                TCmin - Respective lowever bound vector
                TCmax - Respective upper bound vector
                J - Terminal cost
        """
        x0 = ca.mtimes(self.__invScX, self.X[phase][0, :].T)
        xf = ca.mtimes(self.__invScX, self.X[phase][-1, :].T)
        a = ca.mtimes(self.__invScA, self.A[:, phase])
        t0 = self.t0[phase] / self._ocp.scale_t
        tf = self.tf[phase] / self._ocp.scale_t

        # Check if OCP has terminal constraints
        has_terminal_constraints = self._ocp.has_terminal_constraints(phase)
        # Estimate constraint vector and bounds for the collocated terminal constraints
        if has_terminal_constraints:
            terminal_constraints = self._ocp.get_terminal_constraints(phase)
            # Get unscaled starting and final times for given phase
            TC = ca.vertcat(*terminal_constraints(xf, tf, x0, t0, a))
            ntc = TC.size1()
            TCmin = [self._ocp.LB_TERMINAL_CONSTRAINTS] * (ntc)
            TCmax = [self._ocp.UB_TERMINAL_CONSTRAINTS] * (ntc)
        else:
            TC, TCmin, TCmax = [], [], []

        # Mayer term
        terminal_costs = self._ocp.get_terminal_costs(phase)
        J = ca.vertcat(terminal_costs(xf, tf, x0, t0, a))

        return (TC, TCmin, TCmax, J)

    def get_nlp_constraints_for_control_input_slope(self, phase: int = 0) -> Tuple:
        """Get NLP constraints slope on control input (U)

        args:
           :phase: index of the corresponding phase

         returns:
            Tuple : (DU, DUmin, DUmax)
                DU - CasADi vector of constraints for the slope of control input
                DUmin - Respective lowever bound vector
                DUmax - Respective upper bound vector
        """
        # Estimate constraint vector and bounds for the collocated constraint on slope of control input
        if self._ocp.diff_u[phase]:
            _compDU = (
                ca.kron(ca.DM.eye(self._ocp.nu), self._compD)
                if self._ocp.nu > 1
                else self._compD
            )
            DU = ca.mtimes(_compDU, self.U[phase][:])
            nDu = DU.size1()
            DUmin = [self._ocp.lbdu[phase]] * (nDu)
            DUmax = [self._ocp.ubdu[phase]] * (nDu)
        else:
            DU, DUmin, DUmax = [], [], []

        return (DU, DUmin, DUmax)

    def get_nlp_constrains_for_control_input_at_mid_colloc_points(
        self, phase: int = 0
    ) -> Tuple:
        """Get NLP constrains on control input at mid points of the collocation nodes

        Box constraints on control input

        args:
           :phase: index of the corresponding phase

         returns:
            Tuple : (DU, DUmin, DUmax)
                mU - CasADi vector of constraints for the control input at mid colloc points
                mUmin - Respective lowever bound vector
                mUmax - Respective upper bound vector
        """
        if not self._ocp.midu[phase]:
            return ([], [], [])
        mU, mUmin, mUmax = [], [], []
        # Mid point control input, dynamics constraints
        mid_points = lambda tau: [
            (tau[i] + tau[i + 1]) / 2.0 for i in range(len(tau) - 1)
        ]
        taus_mid = [
            mid_points(self.collocation._taus_fn(deg)) for deg in self.poly_orders
        ]

        comp_interpolation_I = self.collocation.get_composite_interpolation_matrix(
            taus_mid, self.poly_orders
        )
        ui = ca.mtimes(comp_interpolation_I, self.U[phase])

        # Control input constraints
        if (self._ocp.lbu[phase] > -np.inf).any() or (
            self._ocp.ubu[phase] < np.inf
        ).any():
            mU.append(ui[:])
            n_mid = ui.size1()
            mUmin.append(np.repeat(self._ocp.lbu[phase] * self._ocp.scale_u, n_mid))
            mUmax.append(np.repeat(self._ocp.ubu[phase] * self._ocp.scale_u, n_mid))

            (mU, mUmin, mUmax) = (
                ca.vertcat(*mU),
                np.concatenate(mUmin),
                np.concatenate(mUmax),
            )

        return (mU, mUmin, mUmax)

    def get_nlp_constrains_for_control_slope_continuity_across_segments(
        self, phase: int = 0
    ) -> Tuple:
        """Get NLP constrains to maintain control input slope continuity across segments

        args:
           :phase: index of the corresponding phase

        returns:
            Tuple : (DU, DUmin, DUmax)
                DU - CasADi vector of constraints for the slope of control input
                    across segments
                DUmin - Respective lowever bound vector
                DUmax - Respective upper bound vector
        """
        if (self.n_segments == 1) or (not self._ocp.du_continuity[phase]):
            return ([], [], [])
        dU, dUmin, dUmax = [], [], []
        # Mid point control input, dynamics constraints
        taus_end = [np.array([self.tau0, self.tau1]) for deg in self.poly_orders]

        comp_interpolation_D = self.collocation.get_composite_interpolation_Dmatrix_at(
            taus_end, self.poly_orders, order=1
        )
        _compD = comp_interpolation_D[1:-1][::2] - comp_interpolation_D[2:-1][::2]

        _compDU = (
            ca.kron(ca.DM.eye(self._ocp.nu), _compD) if self._ocp.nu > 1 else _compD
        )
        dU = ca.mtimes(_compDU, self.U[phase][:])
        nDu = dU.size1()
        dUmin = [0] * (nDu)
        dUmax = [0] * (nDu)

        return (dU, dUmin, dUmax)

    def discretize_phase(self, phase: int) -> Tuple:
        """Discretize single phase of the Optimal Control Problem

        args:
            phase: index of the phase (starting from 0)
        returns :
            Tuple : Constraint vector (G, Gmin, Gmax) and objective function (J)
        """
        if not self._collocation_approximation_computed:
            self.compute_numerical_approximation()
        if not self._variables_created:
            self.create_variables()

        # Discretize OCP
        (f, c, q) = self.get_discretized_dynamics_constraints_and_cost_matrices(phase)
        (F, Fmin, Fmax) = self.get_nlp_constraints_for_dynamics(f, phase)
        (C, Cmin, Cmax) = self.get_nlp_constraints_for_path_contraints(c, phase)

        (
            TC,
            TCmin,
            TCmax,
            mayer_term,
        ) = self.get_nlp_constraints_for_terminal_contraints(phase)

        (DU, DUmin, DUmax) = self.get_nlp_constraints_for_control_input_slope(phase)

        (
            mU,
            mUmin,
            mUmax,
        ) = self.get_nlp_constrains_for_control_input_at_mid_colloc_points(phase)

        (
            dU,
            dUmin,
            dUmax,
        ) = self.get_nlp_constrains_for_control_slope_continuity_across_segments(phase)

        # Add running cost to the mayer term
        J = mayer_term + ca.mtimes(self._compW, ca.vertcat(*q))

        # Merge constraint vectors into sigle constraint vector
        G = ca.vertcat(*[F, C, DU, mU, dU, TC])
        Gmin = np.concatenate([Fmin, Cmin, DUmin, mUmin, dUmin, TCmin])
        Gmax = np.concatenate([Fmax, Cmax, DUmax, mUmax, dUmax, TCmax])

        return (G, Gmin, Gmax, J)

    def get_event_constraints(self) -> Tuple:
        """Estimate the constraint vectors for linking the phases

        args:
            None

        returns:
            Tuple : Constraint vectors (E, Emin, Emax) containing
                    phase linking constraints, discontinuities across
                    states, controls and time variables.
        """
        if self._ocp.n_phases < 2:
            return ([], [], [])
        if not self._variables_created:
            self.create_variables()
        E, Emin, Emax = [None] * 3, [None] * 3, [None] * 3

        n = len(self._ocp.phase_links)

        # State continuity constraints across phases
        E[0] = ca.vertcat(
            *[
                (self.X[phase_j][0, :] - self.X[phase_i][-1, :]).T
                for phase_i, phase_j in self._ocp.phase_links
            ]
        )[:]
        # Lower bound for the state continuity constraint
        Emin[0] = np.concatenate(
            [self._ocp.lbe[phase] * self._ocp.scale_x for phase in range(n)]
        )
        # Upper bound for the state continuity constraint
        Emax[0] = np.concatenate(
            [self._ocp.ube[phase] * self._ocp.scale_x for phase in range(n)]
        )

        # Control continuity constraints across phases
        E[1] = ca.vertcat(
            *[
                (self.U[phase_j][0, :] - self.U[phase_i][-1, :]).T
                for phase_i, phase_j in self._ocp.phase_links
            ]
        )[:]
        # bounds for the control continuity constraints
        Emin[1] = np.concatenate([[0] * self._ocp.nu for phase in range(n)])
        Emax[1] = np.concatenate([[0] * self._ocp.nu for phase in range(n)])

        # Time continuity across phases
        E[2] = ca.vertcat(
            *[
                self.t0[phase_j] - self.tf[phase_i]
                for phase_i, phase_j in self._ocp.phase_links
            ]
        )
        # Time continuity across phases
        Emin[2] = [0] * n
        Emax[2] = [0] * n

        return (E, Emin, Emax)

    def get_nlp_variables(self, phase: int) -> Tuple:
        """Retrieve optimization variables and their bounds for a given phase

        args:
            :phase: index of the phase (starting from 0)

        returns:
            Tuple : (Z, Zmin, Zmax)
                Z - Casadi SX vector containing optimization variables for the given phase (X, U, t0, tf)
                Zmin - Lower bound for the variables in 'Z'
                Zmax - Upper bound for the variables in 'Z'
        """
        if not self._variables_created:
            self.create_variables()
        Z = ca.vertcat(
            self.X[phase][:],
            self.U[phase][:],
            self.t0[phase],
            self.tf[phase],
            self.A[:, phase],
        )

        # Lower and upper bounds for the states in OCP
        xmin_vec = [self._ocp.lbx[phase] * self._ocp.scale_x] * (self._Npoints)
        xmax_vec = [self._ocp.ubx[phase] * self._ocp.scale_x] * (self._Npoints)

        # Impose constraints on initial conditions (Only in phase 0)
        if phase == 0:
            xmin_vec[0] = xmax_vec[0] = self._ocp.x00[0] * self._ocp.scale_x

        Zmin = np.concatenate(
            [
                np.concatenate(np.array(xmin_vec).T),
                np.repeat(self._ocp.lbu[phase] * self._ocp.scale_u, self._Npoints),
                self._ocp.lbt0[phase] * self._ocp.scale_t,
                self._ocp.lbtf[phase] * self._ocp.scale_t,
                self._ocp.lba[phase] * self._ocp.scale_a,
            ]
        )
        Zmax = np.concatenate(
            [
                np.concatenate(np.array(xmax_vec).T),
                np.repeat(self._ocp.ubu[phase] * self._ocp.scale_u, self._Npoints),
                self._ocp.ubt0[phase] * self._ocp.scale_t,
                self._ocp.ubtf[phase] * self._ocp.scale_t,
                self._ocp.uba[phase] * self._ocp.scale_a,
            ]
        )

        return (Z, Zmin, Zmax)

    def create_nlp(self) -> Tuple:
        """Create Nonlinear Programming problem for the given OCP

        args:
            None

        returns:
            Tuple: (nlp_problem, nlp_bounds)
                :nlp_problem: Dictionary (f, x, g, p)
                    f - Objective function
                    x - Optimization variables vector
                    g - constraint vector
                    p - parameter vector
                :nlp_bounds: Dictionary (lbx, ubx, lbg, ubg)
                    lbx - Lower bound for the optimization variables (x)
                    ubx - Upper bound for the optimization variables (x)
                    lbu - Lower bound for the constraints vector (g)
                    ubu - Upper bound for the constraints vector (g)

        """
        # Clear NLP data
        self.J = 0
        G, Gmin, Gmax = (
            [None] * self._ocp.n_phases,
            [None] * self._ocp.n_phases,
            [None] * self._ocp.n_phases,
        )
        Z, Zmin, Zmax = (
            [None] * self._ocp.n_phases,
            [None] * self._ocp.n_phases,
            [None] * self._ocp.n_phases,
        )

        # initialize basis, variables
        self.compute_numerical_approximation()
        self.create_variables()

        # Populate NLP
        for phase in range(self._ocp.n_phases):
            Z[phase], Zmin[phase], Zmax[phase] = self.get_nlp_variables(phase)
            G[phase], Gmin[phase], Gmax[phase], J = self.discretize_phase(phase)
            self.J += J

        if self._ocp.n_phases > 1:
            E, Emin, Emax = self.get_event_constraints()
            G.extend(E)
            Gmin.extend(Emin)
            Gmax.extend(Emax)

        # Create nlp
        self.G = ca.vertcat(*G)
        self.Gmin = np.concatenate(Gmin)
        self.Gmax = np.concatenate(Gmax)
        self.Z = ca.vertcat(*Z)
        self.Zmin = np.concatenate(Zmin)
        self.Zmax = np.concatenate(Zmax)

        nlp_prob = {"f": self.J, "x": self.Z, "g": self.G, "p": self.seg_widths[:]}
        nlp_bounds = {
            "lbg": self.Gmin,
            "ubg": self.Gmax,
            "lbx": self.Zmin,
            "ubx": self.Zmax,
        }

        return (nlp_prob, nlp_bounds)

    def init_solution_per_phase(self, phase: int) -> np.ndarray:
        """Initialize solution vector at all collocation nodes of a given phase.

        The initial solution for a given phase is estimated from the initial and
        terminal conditions defined in the OCP. Simple linear interpolation between
        initial and terminal conditions is used to estimate solution at interior collocation nodes.

        args:
            :phase: index of phase

        returns:
            solution : initialized solution for given phase

        """
        z0 = [None] * 5
        x00 = self._ocp.x00[phase] * self._ocp.scale_x
        xf0 = self._ocp.xf0[phase] * self._ocp.scale_x
        u00 = self._ocp.u00[phase] * self._ocp.scale_u
        uf0 = self._ocp.uf0[phase] * self._ocp.scale_u
        t00 = self._ocp.t00[phase] * self._ocp.scale_t
        tf0 = self._ocp.tf0[phase] * self._ocp.scale_t
        a0 = self._ocp.a0[phase] * self._ocp.scale_a

        # Linear interpolation of states
        z0[0] = np.concatenate(
            np.array(
                [
                    x00 + (xf0 - x00) / (tf0 - t00) * (t - t00)
                    for t in np.linspace(
                        t00,
                        tf0,
                        self._Npoints,
                    )
                ]
            ).T
        )
        # Linear interpolation of controls
        z0[1] = np.concatenate(
            np.array(
                [
                    u00 + (uf0 - u00) / (tf0 - t00) * (t - t00)
                    for t in np.linspace(
                        t00,
                        tf0,
                        self._Npoints,
                    )
                ]
            )
        )
        z0[2], z0[3], z0[4] = t00, tf0, a0

        return np.concatenate(z0)

    def initialize_solution(self) -> np.ndarray:
        """Initialize solution for the NLP from given OCP description

        args:
            None

        returns:
            solution : Initialized solution for the NLP

        """
        Z0 = [None] * self._ocp.n_phases
        for phase in range(self._ocp.n_phases):
            Z0[phase] = self.init_solution_per_phase(phase)

        return np.concatenate(Z0)

    def get_segment_width_parameters(self, solution: Dict) -> List:
        """Get segment widths in all phases

        All segment widths are considered equal

        args:
            :solution: Solution to the nlp from wich the seg_width parameters are
                    computed (if Adaptive)

        returns:
            :seg_widths: numerical values for the fractions of the segment widths
                that equal 1 in each phase
        """
        return [1.0 / self.n_segments] * (self.n_segments * self._ocp.n_phases)

    def create_solver(self, solver: str = "ipopt", options: Dict = {}) -> None:
        """Create NLP solver

        args:
            :solver: Optimization method to be used in nlp_solver (List of plugins
                    avaiable at http://casadi.sourceforge.net/v2.0.0/api/html/d6/d07/classcasadi_1_1NlpSolver.html)
            :options: Dictionary
                List of options for the optimizer (Based on CasADi documentation)

        returns:
            None

        Updates the nlpsolver object in the present optimizer class
        """
        nlp_problem, self.nlp_bounds = self.create_nlp()

        # NLP solver options from casadi doc
        if solver == "ipopt":
            default_options = {
                "ipopt.max_iter": 2000,
                "ipopt.acceptable_tol": 1e-4,
                "ipopt.print_level": 0,
                "ipopt.sb": "yes",
                "print_time": 0,
            }
        else:
            default_options = dict()

        for key in options:
            default_options[key] = options[key]

        # Create NLP solver object -> costly operation
        self.nlp_solver = ca.nlpsol("solver", solver, nlp_problem, default_options)
        self._nlpsolver_initialized = True

    def solve(
        self,
        initial_solution: Dict = None,
        reinitialize_nlp: bool = False,
        solver: str = "ipopt",
        nlp_solver_options: Dict = {},
        mpopt_options: Dict = {},
        **kwargs,
    ) -> Dict:
        """Solve the Nonlinear Programming problem

        args:
            :init_solution: Dictionary containing initial solution with keys
                x or x0 - Initional solution for the nlp variables

            :reinitialize_nlp: (True, False)
                True - Reinitialize NLP solver object
                False - Use already created object if available else create new one

            :nlp_solver_options: Options to be passed to the nlp_solver while creating
                the solver object, not while solving (like initial conditions)
            :mpopt_options: Options dict for the optimizer

        returns:
            :solution: Solution as reported by the given nlp_solver object

        """
        if not self._MUTE_:
            print(f"\n *********** MPOPT Summary ********** \n")

        start_time = time.monotonic()
        if (not self._nlpsolver_initialized) or (reinitialize_nlp):
            self.create_solver(solver=solver, options=nlp_solver_options)

        # By default, these paramers are of equal segment width
        if "nlp_sw_params" in mpopt_options:
            self._nlp_sw_params = mpopt_options["nlp_sw_params"]
        else:
            self._nlp_sw_params = self.get_segment_width_parameters(initial_solution)

        solver_inputs = self.get_solver_warm_start_input_parameters(initial_solution)
        solver_inputs["p"] = self._nlp_sw_params
        end_time_ocp = time.monotonic()

        solution = self.nlp_solver(**solver_inputs, **self.nlp_bounds)

        end_time = time.monotonic()

        if not self._MUTE_:
            print(" Optimal cost (J): ", solution["f"], "\n")
            print(f" Solved in {round((end_time - start_time)*1e3, 3)} ms")

            print(
                f" \t OCP transcription time  : {round((end_time_ocp - start_time)*1e3, 3)} ms"
            )
            print(
                f" \t NLP solution time       : {round((end_time - end_time_ocp)*1e3, 3)} ms"
            )
            # print(f"\n *********** End ********** \n")

        return solution

    def get_solver_warm_start_input_parameters(self, solution: Dict = None):
        """Create dictionary of objects for warm starting the solver using results in 'solution'

        args:
            :solution: Solution of nlp_solver

        returns:
            :dict: (x0, lam_x0, lam_g0)

        """
        key_target_pair = {
            "x": "x0",
            "x0": "x0",
            "lam_x": "lam_x0",
            "lam_x0": "lam_x0",
            "lam_g": "lam_g0",
            "lam_g0": "lam_g0",
        }
        solver_inputs = dict()
        # Initialize the solution based on given options
        if solution is None:
            solver_inputs["x0"] = self.initialize_solution()
        else:
            # Update with warm start init options
            for key in solution:
                if key in key_target_pair:
                    solver_inputs[key_target_pair[key]] = solution[key]

        if "x0" not in solver_inputs:
            # Warning: solution doesnot contain 'x' or 'x0'
            # initialize with default initial solution
            solver_inputs["x0"] = self.initialize_solution()

        return solver_inputs

    def init_trajectories(self, phase: int = 0) -> ca.Function:
        """Initialize trajectories of states, constrols and time variables

        args:
            :phase: index of the phase

        returns:
            :trajectories: CasADi function which returns states, controls and time variable for the given phase when called with NLP solution vector of all phases
                t0, tf - unscaled AND
                x, u, t - scaled trajectories

        """
        x = self.X[phase]
        u = self.U[phase]
        a = self.A[:, phase]
        t0, tf = self.t0[phase] / self._ocp.scale_t, self.tf[phase] / self._ocp.scale_t
        t = ca.vertcat(*self.time_grid[phase])
        trajectories = ca.Function(
            "x_traj",
            [self.Z, self.seg_widths[:]],
            [x, u, t, t0, tf, a],
            ["z", "h"],
            ["x", "u", "t", "t0", "tf", "a"],
        )

        return trajectories

    def process_results(
        self,
        solution,
        plot: bool = True,
        scaling: bool = False,
        residual_x: bool = False,
        residual_dx: bool = True,
    ):
        """Post process the solution of the NLP

        args:
            :solution: NLP solution as reported by the solver
            :plot: bool
                True - Plot states and variables in a single plot with states in a subplot and controls in another.
                False - No plot
            :scaling: bool
                True - Plot the scaled variables
                False - Plot unscaled variables meaning, original solution to the problem
            :residuals: bool
                To plot norma of the residual in the dynamics evaluated at points different from collocation nodes

        returns:
            :post: Object of post_process class (Initialized)

        """
        start_time = time.monotonic()
        trajectories = [
            self.init_trajectories(phase) for phase in range(self._ocp.n_phases)
        ]
        traj_time = time.monotonic()

        resid_value = {}
        if residual_x:
            x_int, u_int, ti, res_x = self.get_states_residuals(solution)
            resid_value["t_x"] = [ti, res_x]
            rx_time = time.monotonic()
        if residual_dx:
            tdx, res_dx = self.get_dynamics_residuals(solution)
            resid_value["t_dx"] = [tdx, res_dx]
            rdx_time = time.monotonic()
        else:
            resid_value = None

        r_time = time.monotonic()

        options = {
            "nx": self._ocp.nx,
            "nu": self._ocp.nu,
            "na": self._ocp.na,
            "nPh": self._ocp.n_phases,
            "ns": self.n_segments,
            "poly_orders": self.poly_orders,
            "N": self._Npoints,
            "phases_to_plot": self._ocp.phases_to_plot,
            "scale_x": self._ocp.scale_x,
            "scale_u": self._ocp.scale_u,
            "scale_a": self._ocp.scale_a,
            "scale_t": self._ocp.scale_t,
            "scaling": scaling,
            "colloc_scheme": self.colloc_scheme,
            "tau0": self.tau0,
            "tau1": self.tau1,
            "interpolation_depth": 3,
            "seg_widths": self._nlp_sw_params,
            "residuals": resid_value,
        }

        post = post_process(solution, trajectories, options)

        if plot:
            for phases in self._ocp.phases_to_plot:
                residuals = residual_x or residual_dx
                post.plot_phases(phases, residuals=residuals)

        post_time = time.monotonic()

        if not self._MUTE_:
            print(f"\n Post processed in {round((post_time - start_time)*1e3, 3)} ms")
            print(
                f" \t Solution retrieval            : {round((traj_time - start_time)*1e3, 3)} ms"
            )
            if residual_x:
                print(
                    f" \t Residual in states             : {round((rx_time - traj_time)*1e3, 3)} ms"
                )
                if residual_dx:
                    print(
                        f" \t Residual in dynamics          : {round((rdx_time - rx_time)*1e3, 3)} ms"
                    )
            elif residual_dx:
                print(
                    f" \t Residual in dynamics           : {round((rdx_time - traj_time)*1e3, 3)} ms"
                )
            print(
                f" \t Process solution and plot      : {round((post_time - r_time)*1e3, 3)} ms"
            )

        return post

    def validate(self):
        """
        Validate initialization of the optimizer object
        """
        pass

    def compute_states_from_solution_dynamics(
        self, solution, phase: int = 0, nodes=None
    ):
        """
        solution : NLP solution
        """
        trajectories = self.init_trajectories(phase)
        x, u, t, t0, tf, a = trajectories(solution["x"], self._nlp_sw_params)
        t0, tf = np.concatenate(t0.full()), np.concatenate(tf.full())

        t_seg = [t[0]] + [
            t[sum(self.poly_orders[: (i + 1)])] for i in range(len(self.poly_orders))
        ]
        x_seg = [x[0, :]] + [
            x[sum(self.poly_orders[: (i + 1)]), :] for i in range(len(self.poly_orders))
        ]

        if nodes is None:
            target_nodes = self.get_residual_grid_taus(
                phase=phase, grid_type=self.grid_type[phase]
            )
        else:
            target_nodes = nodes

        xi, ui, ti, a, Dxi, Dui, taus_grid, t0, tf = self.interpolate_single_phase(
            solution, phase=phase, target_nodes=target_nodes, options={}
        )
        # print("taus", taus_grid)
        seg_widths = self._nlp_sw_params[
            self.n_segments * phase : self.n_segments * (phase + 1)
        ]
        index = 0
        n_taus = [len(taus) for taus in taus_grid]
        xint_phase = [None] * self.n_segments
        residual_phase = [None] * self.n_segments
        u_phase = [None] * self.n_segments
        dynamics = self._ocp.get_dynamics(phase)
        ti_phase = [None] * self.n_segments

        roots_dict = {}
        for seg in range(self.n_segments):
            roots_dict["c{}".format(seg)] = taus_grid[seg]

        self.collocation.init_polynomials_with_customized_roots(roots_dict)

        for seg in range(self.n_segments):
            taus = taus_grid[seg]
            f = [None] * n_taus[seg]
            t = [None] * n_taus[seg]
            xint_seg = [None] * n_taus[seg]
            residual_seg = [None] * n_taus[seg]
            u_seg = [None] * n_taus[seg]
            xstart = x_seg[seg]
            h_seg = (tf[0] - t0[0]) / (self.tau1 - self.tau0) * seg_widths[seg]
            for i, tau in enumerate(taus):
                f[i] = dynamics(
                    xi[index, :].T / self._ocp.scale_x,
                    ui[index, :].T / self._ocp.scale_u,
                    ti[index],
                    a / self._ocp.scale_a,
                )
                f[i] = ca.DM(f[i])
                t[i] = ti[index]
                u_seg[i] = ui[index, :]
                residual_seg[i] = np.concatenate(xi[index, :].full())

                index += 1
            if f:
                f = np.array(f).reshape(len(f), f[0].size()[0])
            for i, tau in enumerate(taus):
                quad_tau = self.collocation.quad_matrix_fn(
                    self.collocation, "c{}".format(seg), tau0=self.tau0, tau1=tau
                )
                xint_seg[i] = np.concatenate(xstart.full()) + np.concatenate(
                    h_seg * (np.dot(quad_tau.T, (f * self._ocp.scale_x)))
                )
                residual_seg[i] = residual_seg[i] - xint_seg[i]

            start, end = sum(n_taus[:seg]), sum(n_taus[: (seg + 1)])
            if start == end:
                continue

            ti_phase[seg] = np.concatenate(np.array(t).reshape(len(t), 1))
            xint_phase[seg] = np.array(xint_seg)
            residual_phase[seg] = residual_seg
            u_phase[seg] = np.array(u_seg)

        return xint_phase, u_phase, ti_phase, residual_phase

    def get_states_residuals(
        self,
        solution,
        phases=None,
        nodes=None,
        residual_type=None,
        plot=False,
        fig=None,
        axs=None,
    ):
        """Compute residual of the system dynamics at given taus (Normalized [0, 1]) by interpolating the
        given solution onto a fixed grid consisting of single segment per phase with
         roots at given target_nodes.

        args:
            :grid_type: target grid type (normalized between 0 and 1)
            :solution: solution of the NLP as reported by the solver
            :phases: As a list with indices ex. [0,1]
            :nodes: grid where the residual is computed (between tau0, tau1) in a list (nodes[0] -> Nodes for phase0)
            :options: Options for the target grid

        returns:
            :residuals: residual vector for the dynamics at the given taus
        """
        x_int, u_int, residuals, ti = (
            [None] * self._ocp.n_phases,
            [None] * self._ocp.n_phases,
            [None] * self._ocp.n_phases,
            [None] * self._ocp.n_phases,
        )
        if phases == None:
            phases = range(self._ocp.n_phases)
        for phase in phases:
            if nodes is None:
                target_nodes = self.get_residual_grid_taus(
                    phase, grid_type=self.grid_type[phase]
                )
            else:
                target_nodes = nodes[phase]

            (
                x_int[phase],
                u_int[phase],
                ti[phase],
                residuals[phase],
            ) = self.compute_states_from_solution_dynamics(
                solution, phase, nodes=target_nodes
            )

            # Compute relative residual in each segment (Global relative)
            if residual_type == "relative":
                # print(residuals[phase])  # = np.array(residuals[phase])
                max_val = np.zeros(self._ocp.nx)
                for seg, res_seg in enumerate(x_int[phase]):
                    if res_seg is not None:
                        seg_max = abs(np.array(res_seg)).max(axis=0)
                        for id, val in enumerate(max_val):
                            max_val[id] = (
                                max_val[id]
                                if max_val[id] > seg_max[id]
                                else seg_max[id]
                            )
                for seg, res_seg in enumerate(residuals[phase]):
                    if res_seg is not None:
                        residuals[phase][seg] = np.array(res_seg) / max_val
                        # assert abs(residuals[phase][seg]).max() <= 1

        if plot:
            fig, axs = post_process.plot_residuals(
                ti, residuals, phases=range(self._ocp.n_phases), fig=fig, axs=axs
            )

        return x_int, u_int, ti, residuals

    def get_residual_grid_taus(self, phase: int = 0, grid_type: str = None):
        """Select the non-collocation nodes in a given phase

        This is often useful in estimation of residual once the OCP is solved.
        Starting and end nodes are not included.

        args:
            :phase: Index of the phase
            :grid_type: Type of non-collocation nodes (fixed, mid-points, spectral)

        returns:
            :points: List of normalized collocation points in each segment of the phase

        """
        if grid_type == None:
            grid_type = self.grid_type[phase]
        if grid_type == "fixed":
            # Equally spaced nodes per phase
            n_nodes = max(sum(self.poly_orders) + 2, self._MAX_GRID_POINTS + 2)

            target_nodes = np.linspace(self.tau0, self.tau1, n_nodes)
            taus_on_original_grid = (
                self.compute_interpolation_taus_corresponding_to_original_grid(
                    target_nodes,
                    self._nlp_sw_params[
                        self.n_segments * phase : self.n_segments * (phase + 1)
                    ],
                    tau0=self.tau0,
                    tau1=self.tau1,
                )
            )
            taus_on_original_grid[0] = taus_on_original_grid[0][:-1]
        elif grid_type == "mid-points":
            mid_points = lambda x: np.array(
                [(x[i] + x[i + 1]) / 2.0 for i in range(len(x) - 1)]
            )
            taus_on_original_grid = np.array(
                [mid_points(self.collocation._taus_fn(deg)) for deg in self.poly_orders]
            )
        elif grid_type == "spectral":
            taus_on_original_grid = np.array(
                [
                    np.array(self.collocation._taus_fn(self._MAX_GRID_POINTS + 2)[1:-1])
                    for deg in self.poly_orders
                ]
            )
        else:
            # Automatically select mid points of collocation nodes to evaluate residuals
            taus_on_original_grid = None

        return taus_on_original_grid

    @staticmethod
    def compute_interpolation_taus_corresponding_to_original_grid(
        nodes_req, seg_widths, tau0=0, tau1=1
    ):
        """Compute the taus on original solution grid corresponding to the required interpolation nodes

        args:
            :nodes_req: target_nodes
            :seg_widths: width of the segments whose sum equal 1

        returns:
            :taus: List of taus in each segment corresponding to nodes_req on target_grid
        """
        cumulative_sw = np.append(0, np.cumsum(seg_widths))
        assert abs(cumulative_sw[-1] - 1) < 1e-6
        n_segments = len(seg_widths)

        # Scale nodes_req between 0 & 1
        nodes_req_scaled = 0 + (1 - 0) / (tau1 - tau0) * (nodes_req - tau0)

        interp_taus = [None] * n_segments
        for i, seg in enumerate(seg_widths):
            # Starting node (0) won't be part of interpolation
            interp_taus[i] = nodes_req_scaled[nodes_req_scaled > cumulative_sw[i]]
            # Terminal state is included
            interp_taus[i] = interp_taus[i][interp_taus[i] <= cumulative_sw[i + 1]]
            # Normalize the taus to 1 by dividing with segment width
            interp_taus[i] = (interp_taus[i] - cumulative_sw[i]) / seg

            # Rescale them back to original grid (between self.tau0 and self.tau1)
            interp_taus[i] = tau0 + (tau1 - tau0) / (1 - 0) * (interp_taus[i] - 0)

        return interp_taus

    def get_state_second_derivative(
        self,
        solution,
        grid_type="spectral",
        nodes=None,
        plot=False,
        fig=None,
        axs=None,
    ):
        """Compute residual of the system states at given taus (Normalized [0, 1]) by interpolating the
        given solution onto a fixed grid consisting of single segment per phase with
         roots at given target_nodes.

        args:
            :grid_type: target grid type (normalized between 0 and 1)
            :solution: solution of the NLP as reported by the solver
            :options: Options for the target grid

        returns:
            :residuals: residual vector for the states at the given taus
        """
        DDx, DDu, ti = (
            [None] * self._ocp.n_phases,
            [None] * self._ocp.n_phases,
            [None] * self._ocp.n_phases,
        )
        for phase in range(self._ocp.n_phases):
            if nodes is None:
                target_nodes = self.get_residual_grid_taus(phase, grid_type=grid_type)
            else:
                target_nodes = nodes[phase]

            (
                ti[phase],
                DDx[phase],
                DDu[phase],
            ) = self.get_state_second_derivative_single_phase(
                solution, phase, nodes=target_nodes
            )

        if plot:
            fig, axs = post_process.plot_residuals(
                ti, DDx, phases=range(self._ocp.n_phases), fig=fig, axs=axs
            )

        return ti, DDx, DDu

    def get_state_second_derivative_single_phase(
        self,
        solution,
        phase: int = 0,
        nodes: List = None,
        grid_type: str = None,
        residual_type: str = None,
    ):
        """Compute residual of the system dynamics at given taus (Normalized [0, 1]) by interpolating the
        given solution onto a fixed grid consisting of single segment per phase with
         roots at given target_nodes.

        args:
            :target_nodes: target grid nodes (normalized between 0 and 1)
            :solution: solution of the NLP as reported by the solver
            :residual_type: 'relative' if relative is req.

        returns:
            :residuals: residual vector for the dynamics at the given taus
        """
        trajectories = self.init_trajectories(phase)
        x, u, t, t0, tf, a = trajectories(solution["x"], self._nlp_sw_params)
        t0, tf = np.concatenate(t0.full()), np.concatenate(tf.full())

        if nodes is None:
            if grid_type is None:
                grid_type = self.grid_type[phase]
            target_nodes = self.get_residual_grid_taus(
                phase=phase, grid_type=self.grid_type[phase]
            )
        else:
            target_nodes = nodes
        ti = self.get_interpolated_time_grid(
            t, target_nodes, self.poly_orders, self.tau0, self.tau1
        )

        comp_interpolation_D = self.collocation.get_composite_interpolation_Dmatrix_at(
            target_nodes, self.poly_orders, order=2
        )

        DDXi = ca.mtimes(comp_interpolation_D, x)
        DDUi = ca.mtimes(comp_interpolation_D, u)

        seg_widths = self._nlp_sw_params[
            self.n_segments * phase : self.n_segments * (phase + 1)
        ]
        index = 0
        n_taus = [len(taus) for taus in target_nodes]
        ddx_phase = [None] * self.n_segments
        ddu_phase = [None] * self.n_segments
        ti_phase = [None] * self.n_segments
        for seg in range(self.n_segments):
            taus = target_nodes[seg]
            r = [None] * n_taus[seg]
            u = [None] * n_taus[seg]
            t = [None] * n_taus[seg]
            for i, tau in enumerate(taus):
                r[i] = DDXi[index, :].T.full()
                u[i] = DDUi[index, :].T.full()
                t[i] = ti[index].full()[0]
                index += 1
            start, end = sum(n_taus[:seg]), sum(n_taus[: (seg + 1)])
            if start == end:
                continue
            R = np.array(r)  # numpy multiplication
            U = np.array(u)  # numpy multiplication
            ddx_phase[seg] = R
            ddu_phase[seg] = U
            if residual_type == "relative":
                ddx_phase[seg] = ddx_phase[seg] / (ddx_phase[seg].max())
                ddu_phase[seg] = ddu_phase[seg] / (ddu_phase[seg].max())
            ti_phase[seg] = np.array(t)

        return ti_phase, ddx_phase, ddu_phase

    def get_dynamics_residuals(
        self,
        solution,
        nodes=None,
        grid_type=None,
        residual_type=None,
        plot=False,
        fig=None,
        axs=None,
    ):
        """Compute residual of the system dynamics at given taus (Normalized [0, 1]) by interpolating the
        given solution onto a fixed grid consisting of single segment per phase with
         roots at given target_nodes.

        args:
            :grid_type: target grid type (normalized between 0 and 1)
            :solution: solution of the NLP as reported by the solver
            :nodes: grid where the residual is computed (between tau0, tau1) in a list (nodes[0] -> Nodes for phase0)
            :options: Options for the target grid
            :residual_type:
                None - Actual residual values
                "relative" - Scaled residual between -1 and 1

        returns:
            :residuals: residual vector for the dynamics at the given taus
        """
        residuals, ti = [None] * self._ocp.n_phases, [None] * self._ocp.n_phases
        for phase in range(self._ocp.n_phases):
            if nodes is None:
                if grid_type is None:
                    grid_type = self.grid_type[phase]
                target_nodes = self.get_residual_grid_taus(phase, grid_type=grid_type)
            else:
                target_nodes = nodes[phase]

            (
                ti[phase],
                residuals[phase],
                dyn_phase,
            ) = self.get_dynamics_residuals_single_phase(
                solution, phase, target_nodes=target_nodes
            )

            # Compute relative residual in each segment (Global relative)
            if residual_type == "relative":
                # print(residuals[phase])  # = np.array(residuals[phase])
                max_val = np.zeros(self._ocp.nx)
                for seg, res_seg in enumerate(dyn_phase):
                    if res_seg is not None:
                        seg_max = abs(np.array(res_seg)).max(axis=0)
                        for id, val in enumerate(max_val):
                            max_val[id] = (
                                max_val[id]
                                if max_val[id] > seg_max[id]
                                else seg_max[id]
                            )
                for seg, res_seg in enumerate(residuals[phase]):
                    if res_seg is not None:
                        residuals[phase][seg] = np.array(res_seg) / max_val
                        assert abs(residuals[phase][seg]).max() <= 1

        if plot:
            fig, axs = post_process.plot_residuals(
                ti, residuals, phases=range(self._ocp.n_phases), fig=fig, axs=axs
            )

        return ti, residuals

    def get_dynamics_residuals_single_phase(
        self,
        solution,
        phase: int = 0,
        target_nodes: List = None,
    ):
        """Compute residual of the system dynamics at given taus (Normalized [0, 1]) by interpolating the
        given solution onto a fixed grid consisting of single segment per phase with
         roots at given target_nodes.

        args:
            :target_nodes: target grid nodes (normalized between 0 and 1)
            :solution: solution of the NLP as reported by the solver

        returns:
            :residuals: residual vector for the dynamics at the given taus
        """
        xi, ui, ti, a, Dxi, Dui, taus_grid, t0, tf = self.interpolate_single_phase(
            solution, phase=phase, target_nodes=target_nodes, options={}
        )

        seg_widths = self._nlp_sw_params[
            self.n_segments * phase : self.n_segments * (phase + 1)
        ]
        index = 0
        n_taus = [len(taus) for taus in taus_grid]
        residual_phase = [None] * self.n_segments
        dyn_phase = [None] * self.n_segments
        dynamics = self._ocp.get_dynamics(phase)
        ti_phase = [None] * self.n_segments
        for seg in range(self.n_segments):
            taus = taus_grid[seg]
            f = [None] * n_taus[seg]
            t = [None] * n_taus[seg]
            for i, tau in enumerate(taus):
                f[i] = dynamics(
                    xi[index, :].T / self._ocp.scale_x,
                    ui[index, :].T / self._ocp.scale_u,
                    ti[index],
                    a / self._ocp.scale_a,
                )
                f[i] = ca.DM(f[i])
                t[i] = ti[index]

                index += 1
            if f:
                f = np.array(f).reshape(len(f), f[0].size()[0])
            start, end = sum(n_taus[:seg]), sum(n_taus[: (seg + 1)])
            if start == end:
                continue
            h_seg = (tf[0] - t0[0]) / (self.tau1 - self.tau0) * seg_widths[seg]
            F = h_seg * (f * self._ocp.scale_x)  # numpy multiplication
            residual_phase[seg] = Dxi[start:end, :].full().reshape(F.shape) - F
            dyn_phase[seg] = F
            ti_phase[seg] = np.concatenate(np.array(t).reshape(len(t), 1))

        for i, it in enumerate(ti_phase):
            if it is None:
                ti_phase[i] = []
        return ti_phase, residual_phase, dyn_phase

    def interpolate_single_phase(
        self,
        solution,
        phase: int = 0,
        target_nodes: np.ndarray = None,
        grid_type=None,
        options: Set = {},
    ):
        """Interpolate the solution at given taus

        args:
            :solution: solution as reported by nlp solver
            :phase: index of the phase
            :target_nodes: List of nodes at which interpolation is performed

        returns:
            Tuple - (X, DX, DU)
                X - Interpolated states
                DX - Derivative of the interpolated states based on PS polynomials
                DU - Derivative of the interpolated controls based on PS polynomials
        """
        trajectories = self.init_trajectories(phase)
        x, u, t, t0, tf, a = trajectories(solution["x"], self._nlp_sw_params)
        t0, tf = np.concatenate(t0.full()), np.concatenate(tf.full())

        if target_nodes is None:
            if grid_type is None:
                grid_type = self.grid_type[phase]
            target_nodes = self.get_residual_grid_taus(phase=phase, grid_type=grid_type)
        ti = self.get_interpolated_time_grid(
            t, target_nodes, self.poly_orders, self.tau0, self.tau1
        )

        comp_interpolation_I = self.collocation.get_composite_interpolation_matrix(
            target_nodes, self.poly_orders
        )
        comp_interpolation_D = self.collocation.get_composite_interpolation_Dmatrix_at(
            target_nodes, self.poly_orders, order=1
        )

        Xi = ca.mtimes(comp_interpolation_I, x)
        Ui = ca.mtimes(comp_interpolation_I, u)
        DXi = ca.mtimes(comp_interpolation_D, x)
        DUi = ca.mtimes(comp_interpolation_D, u)

        # Plot to check if the interpolation is good
        if "plot" in options:
            fig, axs = post_process.plot_all(
                x.full(), u.full(), t.full(), tics=["."] * 15
            )
            fig, axs = post_process.plot_all(
                Xi.full(), Ui.full(), ti, fig=fig, axs=axs, legend=False
            )

        return (Xi, Ui, ti, a, DXi, DUi, target_nodes, t0, tf)

    @staticmethod
    def get_interpolated_time_grid(
        t_orig, taus: np.ndarray, poly_orders: np.ndarray, tau0: float, tau1: float
    ):
        """Update the time vector with the interpolated grid across each segment of
        the original optimization problem

        args:
            :t_orig: Time grid of the original optimization problem (unscaled/scaled)
            :taus: grid of the interpolation taus across each segment of the original OCP
            :poly_orders: Order of the polynomials across each segment used in solving OCP

        returns:
            :time: Interpolated time grid
        """
        # Get the time corresponding to the segment start and end in the original opt. problem
        t_seg = [t_orig[0]] + [
            t_orig[sum(poly_orders[: (i + 1)])] for i in range(len(poly_orders))
        ]

        # Interpolate the original time grid
        time_grid = [
            t_seg[i]
            + (t_seg[i + 1] - t_seg[i])
            * (0 + (1 - 0) / (tau1 - tau0) * (taus[i] - tau0))
            for i in range(len(t_seg) - 1)
        ]

        return ca.vertcat(*time_grid)


class post_process:
    """Process the results of mpopt optimizer for further processing and interactive
     visualization

    This class contains various methods for processing the solution of OCP

    Examples:
        >>> post = post_process(solution, trajectories, options)

    """

    __TICS = ["-"] * 20
    _INTERPOLATION_NODES_PER_SEG = 50

    def __init__(
        self, solution: Dict = {}, trajectories: List = None, options: Dict = {}
    ):
        """Initialize the post processor object

        args:
            :solution: Dictionary (x0, ...)
                x0 - Solution to the NLP variables (all phases)
            :trajectories: casadi function which returns (x, u, t, t0, tf) given solution
                x - scaled states
                u - scaled controls
                t - unscaled time corresponding to x, u (unscaled for simplicity in NLP transcription)
                t0 - scaled initial times
                tf - scaled final times
            :options: Dictionary
                Essential information related to OCP, post processing and nlp optimizer
                 are stored in this dictionary
        """
        self.solution = solution
        self.trajectories = trajectories
        self.options = options
        if "phases_to_plot" in self.options:
            self.phases = self.options["phases_to_plot"][0]
        else:
            self.phases = [0]
        self.nx = self.options["nx"] if "nx" in self.options else 1
        self.nu = self.options["nu"] if "nu" in self.options else 1
        self.na = self.options["na"] if "na" in self.options else 0
        self.scaling = self.options["scaling"] if "scaling" in self.options else False
        # Starting point of the grid
        self.tau0 = (
            self.options["tau0"]
            if "tau0" in self.options
            else CollocationRoots._TAU_MIN
        )

        # End point of the grid
        self.tau1 = (
            self.options["tau1"]
            if "tau1" in self.options
            else CollocationRoots._TAU_MAX
        )

        # Residuals data in options
        if "residuals" in options:
            self.residuals = options["residuals"]
        else:
            self.residuals = None

    def get_trajectories(self, phase: int = 0):
        """Get trajectories of states, controls and time vector for single phase

        args:
            :phase: index of the phase

        returns:
            Tuple : (x, u, t, a)
                x - states
                u - controls
                t - corresponding time vector
        """
        # if "seg_widths" in self.options:
        x, u, t, t0, tf, a = self.trajectories[phase](
            self.solution["x"], self.options["seg_widths"]
        )
        # else:
        # x, u, t, t0, tf, a = self.trajectories[phase](self.solution["x"])
        x_opt, u_opt, t_opt, a_opt = (x.full(), u.full(), t.full(), a.full())

        scale_t = self.options["scale_t"] if "scale_t" in self.options else 1.0
        if not self.scaling:
            scale_x = self.options["scale_x"] if "scale_x" in self.options else 1.0
            scale_u = self.options["scale_u"] if "scale_u" in self.options else 1.0
            scale_a = self.options["scale_a"] if "scale_a" in self.options else 1.0

            return (x_opt / scale_x, u_opt / scale_u, t_opt, a_opt / scale_a)

        return (x_opt, u_opt, t_opt, a_opt)

    def get_original_data(self, phases: List = []):
        """Get optimized result for multiple phases

        args:
            :phases: Optional, List of phases to retrieve the data from.

        returns:
            Tuple : (x, u, t, a)
                x - states
                u - controls
                t - corresponding time vector
        """
        if not phases:
            phases = self.phases
        x, u, t, a = self.get_trajectories(phases[0])
        if len(phases) > 1:
            for phase in phases[1:]:
                xp, up, tp, ap = self.get_trajectories(phase)
                x = np.vstack((x, xp))
                u = np.vstack((u, up))
                t = np.vstack((t, tp))
                a = np.vstack((a, ap))

        return (x, u, t, a)

    def get_interpolation_taus(
        self, n: int = 75, taus_orig: np.ndarray = None, method: str = "uniform"
    ):
        """Nodes across the normalized range [0, 1] or [-1, 1], to interpolate the
        data for post processing such as plotting

        args:
            :n: number of interpolation nodes
            :taus_orig: original grid across which interpolation is to be performed
            :method: ("uniform", "other")
                "uniform" : returns equally spaced grid points
                "other": returns mid points of the original grid recursively until 'n' elements

        returns:
            :taus: interpolation grid
        """
        if (method == "uniform") or (taus_orig is None):
            return np.linspace(self.tau0, self.tau1, n)
        else:
            return self.get_non_uniform_interpolation_grid(taus_orig, n)

    @staticmethod
    def get_non_uniform_interpolation_grid(taus_orig, n: int = 75):
        """Increase the resolution of the given taus preserving the sparsity of the given grid

        args:
            :taus_orig: original grid to be refined
            :n: max number of points in the refined grid.

        returns:
            :taus: refined grid
        """

        def get_mid_points(taus):
            points = [
                [tau, (taus[i] + taus[i + 1]) / 2.0] for i, tau in enumerate(taus[:-1])
            ]

            return np.append(np.concatenate(points), taus[-1])

        count, __max_count = 0, 5
        while len(taus_orig) < n:
            taus_orig = get_mid_points(taus_orig)
            count += 1
            if count > __max_count:
                break

        return taus_orig

    @staticmethod
    def get_interpolated_time_grid(
        t_orig, taus: np.ndarray, poly_orders: np.ndarray, tau0: float, tau1: float
    ):
        """Update the time vector with the interpolated grid across each segment of
        the original optimization problem

        args:
            :t_orig: Time grid of the original optimization problem (unscaled/scaled)
            :taus: grid of the interpolation taus across each segment of the original OCP
            :poly_orders: Order of the polynomials across each segment used in solving OCP

        returns:
            :time: Interpolated time grid
        """
        # Get the time corresponding to the segment start and end in the original opt. problem
        t_seg = [t_orig[0]] + [
            t_orig[sum(poly_orders[: (i + 1)])] for i in range(len(poly_orders))
        ]

        # Interpolate the original time grid
        time_grid = [
            t_seg[i]
            + (t_seg[i + 1] - t_seg[i])
            * (0 + (1 - 0) / (tau1 - tau0) * (taus[i] - tau0))
            for i in range(len(t_seg) - 1)
        ]

        return np.concatenate(time_grid)

    def get_interpolated_data(self, phases, taus: List = []):
        """Interpolate the original solution across given phases

        args:
            :phases: List of phase indices
            :taus: collocation grid points across which interpolation is performed

        returns:
            Tuple : (x, u, t, a)
                x - interpolated states
                u - interpolated controls
                t - interpolated time grid
        """
        scheme = (
            self.options["colloc_scheme"] if "colloc_scheme" in self.options else "LGR"
        )
        if "poly_orders" in self.options:
            poly_orders = self.options["poly_orders"]
        collocation = Collocation(poly_orders, scheme)
        if taus == []:
            taus = [
                self.get_interpolation_taus(
                    n=self._INTERPOLATION_NODES_PER_SEG,
                    taus_orig=collocation._taus_fn(p),
                    method="uniform",
                )[1:]
                for p in poly_orders
            ]
            taus[0] = np.append(self.tau0, taus[0])

        # Composite Interpolation matrix
        compI = collocation.get_composite_interpolation_matrix(taus, poly_orders)

        # Get original solution from the solution
        x_orig, u_orig, t_orig, a = self.get_original_data([phases[0]])

        # Interpolate the original solution using the composite matrix
        x, u = np.dot(compI, x_orig), np.dot(compI, u_orig)

        # Get the time corresponding to the interpolated data
        t = self.get_interpolated_time_grid(
            t_orig, taus, poly_orders, self.tau0, self.tau1
        )

        if len(phases) > 1:
            # Repeat the interpolation procedure across remaining phases
            # All phases are assumed to be having same composite matrix (as of now)
            for phase in phases[1:]:
                x_orig, u_orig, t_orig, ap = self.get_original_data([phase])
                xp, up = np.dot(compI, x_orig), np.dot(compI, u_orig)
                tp = self.get_interpolated_time_grid(
                    t_orig, taus, poly_orders, self.tau0, self.tau1
                )
                x = np.vstack((x, xp))
                u = np.vstack((u, up))
                t = np.hstack((t, tp))
                a = np.hstack((a, ap))

        return (x, u, t, a)

    def get_data(self, phases: List = [], interpolate: bool = False):
        """Get solution corresponding to given phases (original/interpolated)

        args:
            :phases: List of phase indices
            :interpolate: bool
                True - Interpolate the original grid (Refine the grid for smooth plot)
                False - Return original data

        returns:
            Tuple : (x, u, t, a)
                x - interpolated states
                u - interpolated controls
                t - interpolated time grid
        """
        if not phases:
            phases = self.phases
        (x, u, t, a) = (
            self.get_interpolated_data(phases)
            if interpolate
            else self.get_original_data(phases)
        )

        return (x, u, t, a)

    def plot_phase(self, phase: int = 0, interpolate: bool = True, fig=None, axs=None):
        """Plot states and controls across given phase

        args:
            :phase: index of phase
            :interpolate: bool
                True - Plot refined data
                False - Plot original data

        returns:
            Tuple : (fig, axs)
                fig - handle to the figure obj. of plot (matplotlib figure object)
                axs - handle to the axis of plot (matplotlib figure object)
        """
        fig, axs = self.plot_phases([phase], interpolate, fig=fig, axs=axs)

    def plot_phases(
        self,
        phases: List = None,
        interpolate: bool = True,
        residuals: bool = True,
        fig=None,
        axs=None,
        tics: List = ["-"] * 15,
        name: str = "",
    ):
        """Plot states and controls across given phases

        args:
            :phases: List of indices corresponding to phases
            :interpolate: bool
                True - Plot refined data
                False - Plot original data

        returns:
            Tuple : (fig, axs)
                fig - handle to the figure obj. of plot (matplotlib figure object)
                axs - handle to the axis of plot (matplotlib figure object)
        """
        if phases is None:
            if "phases_to_plot" in self.options:
                phases = self.options["phases_to_plot"][0]
            else:
                phase = [0]

        if residuals:
            if self.residuals is not None:
                fig = plt.figure()
                ax0 = plt.subplot(2, 2, 1)
                ax1 = plt.subplot(2, 2, 3)
                axs = [ax0, ax1]

        if interpolate:
            xi, ui, ti, _ = self.get_data(phases=phases, interpolate=interpolate)
            fig, axs = self.plot_all(
                xi, ui, ti, legend=False, fig=fig, axs=axs, tics=tics
            )
            tics = ["."] * 15
        x, u, t, _ = self.get_data(phases=phases, interpolate=False)
        fig, axs = self.plot_all(x, u, t, tics=tics, fig=fig, axs=axs, name=name)

        if residuals:
            if self.residuals is not None:
                if ("t_x" in self.residuals) and ("t_dx" in self.residuals):
                    y_data = [
                        r_seg
                        for r_p in self.residuals["t_x"][1]
                        for r_seg in r_p
                        if r_seg is not None
                    ]

                    x_data = np.hstack(np.concatenate(self.residuals["t_x"][0]))
                    y_data = np.concatenate(y_data)

                    ax2 = plt.subplot(2, 2, 2)
                    dims = y_data.shape[1]
                    self.plot_curve(ax2, y_data, x_data, "x", "States error", tics=tics)

                    ax3 = plt.subplot(2, 2, 4)
                    y_data = [
                        r_seg
                        for r_p in self.residuals["t_dx"][1]
                        for r_seg in r_p
                        if r_seg is not None
                    ]

                    x_data = np.hstack(np.concatenate(self.residuals["t_dx"][0]))
                    y_data = np.concatenate(y_data)

                    self.plot_curve(
                        ax3, y_data, x_data, "dx", "Dynamics error", tics=tics
                    )

                    axs = [ax0, ax1, ax2, ax3]
                else:
                    ax2 = plt.subplot(1, 2, 2)
                    if "t_x" in self.residuals:
                        y_data = [
                            r_seg
                            for r_p in self.residuals["t_x"][1]
                            for r_seg in r_p
                            if r_seg is not None
                        ]

                        x_data = np.hstack(np.concatenate(self.residuals["t_x"][0]))
                        y_data = np.concatenate(y_data)

                        self.plot_curve(
                            ax2, y_data, x_data, "x", "States error", tics=tics
                        )
                    else:
                        y_data = [
                            np.concatenate(
                                [r_seg for r_seg in r_p if r_seg is not None]
                            )
                            for r_p in self.residuals["t_dx"][1]
                        ]
                        y_data = np.concatenate(y_data)

                        x_data = np.concatenate([
                        r_seg
                        for r_p in self.residuals["t_dx"][0]
                        for r_seg in r_p
                        if r_seg is not None
                        ])

                        self.plot_curve(
                            ax2, y_data, x_data, "dx", "Dynamics error", tics=tics
                        )
                    ax2.set(xlabel="Time, s")
                    axs = [ax0, ax1, ax2]

        return fig, axs

    def plot_x(
        self,
        dims: List = None,
        phases: List = None,
        axis: int = 1,
        interpolate: bool = True,
        fig=None,
        axs=None,
        tics=["-"] * 15,
    ):
        """Plot given dimenstions of states across given phases stacked vertically or horizontally

        args:
            :dims: List of dimentions of the state to be plotted (List of list indicates
                    each internal list plotted in respective subplot)
            :phases: List of phases to plot
            :interpolate: bool
                True - Plot refined data
                False - Plot original data

        returns:
            Tuple : (fig, axs)
                fig - handle to the figure obj. of plot (matplotlib figure object)
                axs - handle to the axis of plot (matplotlib figure object)
        """
        if not phases:
            phases = self.phases
        if not dims:
            dims = range(self.nx)
        x, _, t, _ = self.get_data(phases, interpolate)
        fig, axs = self.plot_single_variable(
            x,
            t,
            dims,
            name="state",
            ylabel="State variables",
            axis=axis,
            fig=fig,
            axs=axs,
            tics=tics,
        )

        return fig, axs

    def plot_u(
        self,
        dims: List = None,
        phases: List = None,
        axis: int = 1,
        interpolate: bool = True,
        fig=None,
        axs=None,
        tics=None,
        name="control",
        ylabel="Control variables",
    ):
        """Plot given dimenstions of controls across given phases stacked vertically or horizontally

        args:
            :dims: List of dimentions of the control to be plotted (List of list indicates
                    each internal list plotted in respective subplot)
            :phases: List of phases to plot
            :interpolate: bool
                True - Plot refined data
                False - Plot original data

        returns:
            Tuple : (fig, axs)
                fig - handle to the figure obj. of plot (matplotlib figure object)
                axs - handle to the axis of plot (matplotlib figure object)
        """
        if not phases:
            phases = self.phases
        if not dims:
            dims = range(self.nu)
        if tics is None:
            tics = ["."] * 15
        if interpolate:
            _, u, t, _ = self.get_data(phases, interpolate=False)
            fig, axs = self.plot_single_variable(
                u,
                t,
                dims,
                name=name,
                ylabel=None,
                axis=axis,
                fig=fig,
                axs=axs,
                tics=tics,
            )

        _, u, t, _ = self.get_data(phases, interpolate)
        fig, axs = self.plot_single_variable(
            u,
            t,
            dims,
            name=None,
            ylabel=ylabel,
            axis=axis,
            fig=fig,
            axs=axs,
            tics=["-"] * 15,
        )

        return fig, axs

    @classmethod
    def plot_single_variable(
        self,
        var_data,
        t,
        dims,
        name: str = None,
        ylabel: str = None,
        axis=1,
        fig=None,
        axs=None,
        tics=["-"] * 15,
    ):
        """Plot given numpy array in various subplots

        args:
            :var_data: input data to be plotted
            :t: xdata of the plot
            :dims: List of dimentions of the data to be plotted (List of list indicates
                    each internal list plotted in respective subplot)
            :name: Variable name
            :ylabel: ylabel for the plot
            :axis: int, 0 or 1
                0 - subplots are stacked horizintally
                1 - subplots are stacked vertically


        returns:
            Tuple : (fig, axs)
                fig - handle to the figure obj. of plot (matplotlib figure object)
                axs - handle to the axis of plot (matplotlib figure object)
        """
        if tics is None:
            tics = self.__TICS
        n = len(dims) if isinstance(dims[0], list) else 1
        if (fig == None) and (axs == None):
            if axis == 1:
                fig, axs = plt.subplots(n, 1, sharex=True)
            else:
                fig, axs = plt.subplots(1, n)
        for i in range(n):
            ax = axs if n == 1 else axs[i]
            dim = dims if n == 1 else dims[i]
            self.plot_curve(
                ax,
                var_data[:, dim],
                t,
                name=name,
                ylabel=ylabel,
                tics=tics,
                legend_index=dim,
            )
            if not axis:
                ax.set(xlabel="Time, s")
        if axis == 1:
            ax.set(xlabel="Time, s")

        return fig, axs

    @staticmethod
    def plot_curve(ax, x, t, name=None, ylabel="", tics=["-"] * 15, legend_index=None):
        """2D Plot given data (x, y)

        args:
            :ax: Axis handle of matplotlib plot
            :x: y-axis data - numpy ndarray with first dimention having data
            :t: x-axis data
        """
        nx = x.shape[1]
        for i in range(nx):
            state_index = i + 1 if legend_index is None else legend_index[i]
            # state_index = ""  # if legend_index is None else legend_index[i]
            label = f"{name} {state_index}" if name is not None else name
            ax.plot(t, x[:, i], tics[i], label=label)
        ax.set(ylabel=ylabel)
        if name is not None:
            ax.legend()
        ax.grid(True)

        return ax

    @classmethod
    def plot_all(
        self,
        x,
        u,
        t,
        tics: str = None,
        fig=None,
        axs=None,
        legend: bool = True,
        name: str = "",
    ):
        """Plot states and controls

        args:
            :x: states data
            :u: controls data
            :t: time grid

        returns:
            Tuple : (fig, axs)
                fig - handle to the figure obj. of plot (matplotlib figure object)
                axs - handle to the axis of plot (matplotlib figure object)
        """
        if tics is None:
            tics = self.__TICS
        if (fig == None) and (axs == None):
            fig, axs = plt.subplots(2, 1, sharex=True)
        name_x, name_u = (name + "state", name + "control") if legend else (None, None)
        self.plot_curve(axs[0], x, t, name_x, "State variables", tics=tics)
        self.plot_curve(axs[1], u, t, name_u, "Control variables", tics=tics)
        axs[1].set(xlabel="Time, s")

        return fig, axs

    @staticmethod
    def sort_residual_data(time, residuals, phases: List = [0]):
        """Sort the given data corresponding to plases"""
        norm_residual = lambda residual: np.concatenate(
            [
                np.linalg.norm(np.array(err), 2, axis=1) if err is not None else []
                for err in residual
            ]
        )
        t, r = time[0], norm_residual(residuals[0])
        if len(phases) > 1:
            for phase in phases[1:]:
                tp, rp = time[phase], norm_residual(residuals[phase])
                r = np.vstack((r, rp))
                t = np.vstack((t, tp))

            r = np.concatenate(r)

        t = np.concatenate(t)
        r = r.reshape(t.shape[0], 1)
        return (r, t)

    @classmethod
    def plot_residuals(
        self,
        time,
        residuals,
        phases: List = [0],
        name=None,
        fig=None,
        axs=None,
        tics=["."] * 15,
    ):
        """Plot residual in dynamics"""
        if tics is None:
            tics = self.__TICS
        if (fig == None) and (axs == None):
            fig, axs = plt.subplots(1, 1)

        r, t = self.sort_residual_data(time, residuals, phases=phases)
        # print(t.shape, r.shape, time[0])

        self.plot_curve(
            axs,
            r,
            t,
            name,
            ylabel="residuals",
            tics=tics,
            legend_index=[""] * 15,
        )
        self.plot_curve(
            axs,
            r,
            t,
            ylabel="residuals",
            tics=["-"] * 15,
            legend_index=[""] * 15,
        )
        axs.set(xlabel="Time, s")

        return fig, axs


class mpopt_h_adaptive(mpopt):
    """Multi-stage Optimal control problem (OCP) solver which implements iterative
    procedure to refine the segment width in each phase adaptively while keeping the
    same number of segments

    Examples :
        >>> # Moon lander problem
        >>> from mpopt import mp
        >>> ocp = mp.OCP(n_states=2, n_controls=1, n_params=0, n_phases=1)
        >>> ocp.dynamics[0] = lambda x, u, t, a: [x[1], u[0] - 1.5]
        >>> ocp.running_costs[0] = lambda x, u, t, a: u[0]
        >>> ocp.terminal_constraints[0] = lambda xf, tf, x0, t0, a: [xf[0], xf[1]]
        >>> ocp.x00[0] = [10, -2]
        >>> ocp.lbu[0] = 0; ocp.ubu[0] = 3
        >>> ocp.lbtf[0] = 3; ocp.ubtf[0] = 5
        >>> opt = mp.mpopt_h_adaptive(ocp, n_segments=3, poly_orders=[2]*3)
        >>> solution = opt.solve()
        >>> post = opt.process_results(solution, plot=True)
    """

    _SEG_WIDTH_MIN = 1e-5
    _SEG_WIDTH_MAX = 1

    _TOL_SEG_WIDTH_CHANGE = 0.05  # < 5% change in width fraction (Converged)
    _TOL_RESIDUAL = 1e-2

    _DEFAULT_METHOD = "residual"
    _DEFAULT_SUB_METHOD = "equal_area"

    _THRESHOLD_SLOPE = 1e-1  # threshold on control slope (Adaptive scheme)

    def __init__(
        self: "mpopt_h_adaptive",
        problem: "OCP",
        n_segments: int = 1,
        poly_orders: List[int] = [9],
        scheme: str = "LGR",
        **kwargs,
    ):
        """Initialize the optimizer
        args:
            n_segments: number of segments in each phase
            poly_orders: degree of the polynomial in each segment
            problem: instance of the OCP class
        """
        super().__init__(
            problem=problem,
            n_segments=n_segments,
            poly_orders=poly_orders,
            scheme=scheme,
        )
        # Segment width bounds : default values
        self.lbh = [self._SEG_WIDTH_MIN for _ in range(self._ocp.n_phases)]
        self.ubh = [self._SEG_WIDTH_MAX for _ in range(self._ocp.n_phases)]
        self.tol_residual = [self._TOL_RESIDUAL for _ in range(self._ocp.n_phases)]
        self.fig, self.axs = None, None
        self.plot_residual_evolution = False

    def solve(
        self,
        initial_solution: Dict = None,
        reinitialize_nlp: bool = False,
        solver: str = "ipopt",
        nlp_solver_options: Dict = {},
        mpopt_options: Dict = {},
        max_iter: int = 10,
        **kwargs,
    ) -> Dict:
        """Solve the Nonlinear Programming problem

        args:
            :init_solution: Dictionary containing initial solution with keys
                x or x0 - Initional solution for the nlp variables

            :reinitialize_nlp: (True, False)
                True - Reinitialize NLP solver object
                False - Use already created object if available else create new one

            :nlp_solver_options: Options to be passed to the nlp_solver while creating
                the solver object, not while solving (like initial conditions)
            :mpopt_options: Options dict for the optimizer
                'method': 'residual' or 'control_slope'
                'sub_method': (if method is residual)
                    'merge_split'
                    'equal_area'


        returns:
            :solution: Solution as reported by the given nlp_solver object

        """
        if not self._MUTE_:
            print(f"\n *********** MPOPT H-Adaptive Summary ********** \n")

        start_time = time.monotonic()
        if (not self._nlpsolver_initialized) or (reinitialize_nlp):
            if "ipopt.print_level" not in nlp_solver_options:
                nlp_solver_options["ipopt.print_level"] = 0
            self.create_solver(solver=solver, options=nlp_solver_options)

        if mpopt_options == {}:
            mpopt_options["method"] = self._DEFAULT_METHOD
            mpopt_options["sub_method"] = self._DEFAULT_SUB_METHOD

        self.iter_count, self.iter_info = 0, dict()
        sw_old = []
        new_nlp_sw_params, max_error = self.get_segment_width_parameters(
            initial_solution, options=mpopt_options
        )
        solved = False
        if max_error is not None:
            if max_error < min(self.tol_residual):
                self.iter_info[self.iter_count] = max_error
                print(
                    f"Solved to acceptable tolerance {min(self.tol_residual)}",
                    max_error,
                )
                solution = initial_solution
                solved = True

        if not solved:
            for iter in range(max_iter):
                self._nlp_sw_params = new_nlp_sw_params
                # By default, these paramers are of equal segment width

                if self.iter_count > 0:
                    self.iter_info[self.iter_count] = max_error

                    if self.iter_count > 4:
                        mean_error = np.mean(list(self.iter_info.values())[-4:])
                        if abs(max_error - mean_error) < 0.05 * abs(max_error):
                            print(
                                "Stopping the iterations: Change in max error is < 5%"
                            )
                            self._nlp_sw_params = sw_old
                            break

                if iter > 0:
                    sw_percentage_change = np.array(
                        [
                            abs(self._nlp_sw_params[i] - sw_old[i])
                            / self._nlp_sw_params[i]
                            <= self._TOL_SEG_WIDTH_CHANGE
                            for i in range(len(self._nlp_sw_params))
                        ]
                    )
                    if sw_percentage_change.all():
                        print(
                            "Stopping the iterations: Change in width less than 5%",
                            max_error,
                        )
                        self._nlp_sw_params = sw_old
                        break

                if (iter == 0) and (initial_solution is None):
                    max_error = None

                if max_error is not None:
                    print(f"Iteration : {iter}, {max_error}")
                solver_inputs = self.get_solver_warm_start_input_parameters(
                    initial_solution
                )
                solver_inputs["p"] = self._nlp_sw_params

                solution = self.nlp_solver(**solver_inputs, **self.nlp_bounds)

                # Store the solution in intial_solution
                initial_solution = solution
                sw_old = copy.deepcopy(self._nlp_sw_params)

                new_nlp_sw_params, max_error = self.get_segment_width_parameters(
                    initial_solution, options=mpopt_options
                )

                self.iter_count += 1

                if max_error is not None:
                    if max_error < min(self.tol_residual):
                        self.iter_info[self.iter_count] = max_error
                        print(
                            f"Solved to acceptable tolerance {min(self.tol_residual)}",
                            max_error,
                        )
                        break

                if iter == max_iter - 1:
                    self.iter_info[self.iter_count] = max_error
                    print("Stopping the iterations: Iteration limit exceeded")

        end_time = time.monotonic()
        print(f"H-Adaptive Iter., max_residual : {self.iter_count}, {max_error}")

        if not self._MUTE_:
            print(" Optimal cost (J): ", solution["f"], "\n")
            print(f" Solved in {round((end_time - start_time)*1e3, 3)} ms\n")
            # print(f"\n *********** H-adaptive End ********** \n")

        return solution

    def get_segment_width_parameters(
        self,
        solution,
        options: Dict = {"method": "residual", "sub_method": "merge_split"},
    ):
        """Compute optimum segment widths in every phase based on the given solution to the NLP

        args:
            :solution: solution to the NLP
            :options: Dictionary of options if required (Computation method etc.)
                :method: Method used to refine the grid
                    'residual'
                        'merge_split'
                        'equal_area'
                    'control_slope'

        returns:
            :seg_widths: Computed segment widths in a 1-D list (each phase followed by previous)
        """
        max_error = None
        default_widths = [1 / self.n_segments] * (self.n_segments * self._ocp.n_phases)
        if self.n_segments == 1:
            return default_widths, max_error

        if solution is None:
            # Warning, Solution not found, returning default segment widths
            return default_widths, max_error

        if not hasattr(self, "_nlp_sw_params"):
            self._nlp_sw_params = default_widths

        if "method" in options:
            if options["method"] == "control_slope":
                seg_widths, max_error = self.compute_seg_width_based_on_input_slope(
                    solution
                )
            elif options["method"] == "residual":
                sub_method = (
                    options["sub_method"] if "sub_method" in options else "equal_area"
                )
                seg_widths, max_error = self.compute_seg_width_based_on_residuals(
                    solution, method=sub_method
                )
            else:
                # Method undefined, return defaults
                seg_widths = default_widths
        else:
            # Method undefined, return defaults
            seg_widths = default_widths

        return seg_widths, max_error

    def compute_seg_width_based_on_residuals(
        self,
        solution,
        method: str = "merge_split",
    ):
        """Compute the optimum segment widths based on residual of the dynamics in each segment.

        args:
            :solution: nlp solution as reported by the solver

        returns:
            :segment_widths: optimized segment widths based on present solution
        """
        segment_widths = [None] * self._ocp.n_phases
        ti, residuals = self.get_dynamics_residuals(solution)

        if self.plot_residual_evolution:
            self.fig, self.axs = post_process.plot_residuals(
                ti,
                residuals,
                phases=self._ocp.phases_to_plot,
                name=f"Iter {self.iter_count}",
                fig=self.fig,
                axs=self.axs,
                tics=["-"] * 15,
            )
            self.fig, self.axs = post_process.plot_residuals(
                ti,
                residuals,
                phases=self._ocp.phases_to_plot,
                # name=f"Iter {self.iter_count}",
                fig=self.fig,
                axs=self.axs,
                tics=["-"] * 15,
            )
        max_error = 0
        for phase in range(self._ocp.n_phases):
            max_residual = max(
                [
                    abs(np.array(err)).max() if err is not None else 0
                    for err in residuals[phase]
                ]
            )
            if max_residual > max_error:
                max_error = max_residual

            segment_widths_old = self._nlp_sw_params[
                self.n_segments * phase : self.n_segments * (phase + 1)
            ]

            if max_residual < self.tol_residual[phase]:
                segment_widths[phase] = segment_widths_old
                print(
                    f"Solved phase {phase} to acceptable tolerance {self.tol_residual[phase]}"
                )
                continue

            segment_widths[phase] = self.refine_segment_widths_based_on_residuals(
                residuals[phase],
                segment_widths_old,
                ERR_TOL=self.tol_residual[phase],
                method=method,
            )
            if method == "equal_area":
                segment_widths[phase] = 0.4 * np.array(
                    segment_widths[phase]
                ) + 0.6 * np.array(segment_widths_old)

        return np.concatenate(segment_widths), max_error

    def refine_segment_widths_based_on_residuals(
        self,
        residuals,
        segment_widths,
        ERR_TOL: float = 1e-3,
        method: str = "merge_split",
    ):
        """Refine segment widths based on residuals of dynamics

        args:
            :residuals: residual matrix of dynamics of each segment
            :segment_widths: Segment width corresponding to the residual

        returns:
            :segment_widths: Updated segment widths after refining
        """
        if method == "merge_split":
            max_residuals = [
                np.abs(np.array(err)).max() if err is not None else 0
                for err in residuals
            ]

            return self.merge_split_segments_based_on_residuals(
                max_residuals, segment_widths, ERR_TOL=ERR_TOL
            )
        elif method == "equal_area":
            residual_1D = np.concatenate(
                [
                    np.linalg.norm(np.array(err), 2, axis=1) if err is not None else [0]
                    for err in residuals
                ]
            )
            # import matplotlib.pyplot as plt
            #
            # plt.plot(residual_1D)
            # plt.show()
            segment_widths = self.get_roots_wrt_equal_area(residual_1D, self.n_segments)
            return segment_widths
        else:
            # Warning, method not defined. Return same segment widths
            return segment_widths

    @staticmethod
    def get_roots_wrt_equal_area(residuals, n_segments):
        """"""
        n_points = len(residuals)
        areas = [0.5 * (residuals[i] + residuals[i + 1]) for i in range(n_points - 1)]
        cumulative_area = np.append(0, np.cumsum(areas))
        cumulative_area = cumulative_area / cumulative_area[-1]
        seg_widths_cumulative = [None] * n_segments
        for i in range(n_segments):
            ind_j = (cumulative_area >= (i + 1) / n_segments).argmax()
            seg_widths_cumulative[i] = (
                ind_j
                - 1
                + ((i + 1) / n_segments - cumulative_area[ind_j - 1])
                / (cumulative_area[ind_j] - cumulative_area[ind_j - 1])
            ) / (n_points - 1)

        seg_widths_cumulative = np.append(0, seg_widths_cumulative)
        seg_widths = [
            seg_widths_cumulative[i + 1] - seg_widths_cumulative[i]
            for i in range(n_segments)
        ]

        return seg_widths

    @staticmethod
    def merge_split_segments_based_on_residuals(
        max_residuals, segment_widths, ERR_TOL: float = 1e-3
    ):
        """Merge/Split existing segments based on residual errors

        Merge consecutive segments with residual below tolerance

        args:
            :max_residuals: max residual in dynamics of each segment
            :segment_widths: Segment width corresponding to the residual

        returns:
            :segment_widths: Updated segment widths after merging/splitting
        """
        ns = len(segment_widths)
        data = [(max_residuals[seg], seg) for seg in range(ns)]
        groups = [
            [(key, g[1]) for g in group]
            for key, group in itertools.groupby(data, lambda x: x[0] < ERR_TOL)
        ]
        n_false = len([g[0][0] for g in groups if not g[0][0]])
        if len(groups) == ns:
            # Can not decide on merge/split
            return segment_widths
        if n_false == 0:
            # All segments meet the residual tolerance
            return segment_widths

        h_new = [sum([segment_widths[i[1]] for i in g]) for g in groups]
        len_new = len(h_new)
        n_free = ns - len_new
        n_free_per_false = [1 + int(n_free / n_false) for n in range(n_false)]
        n_free_per_false[-1] = n_free_per_false[-1] + np.mod(n_free, n_false)
        false_id, seg_id = 0, 0
        new_sw = [None] * ns
        for i, g in enumerate(groups):
            if g[0][0]:
                new_sw[seg_id] = h_new[i]
                seg_id += 1
            else:
                for j in range(n_free_per_false[false_id]):
                    new_sw[seg_id] = h_new[i] / n_free_per_false[false_id]
                    seg_id += 1
                false_id += 1

        return np.array(new_sw)

    def compute_seg_width_based_on_input_slope(self, solution):
        """Compute the optimum segment widths based on slope of the control signal.

        args:
            :solution: nlp solution as reported by the solver

        returns:
            :segment_widths: optimized segment widths based on present solution
        """
        ti, residuals = self.get_dynamics_residuals(solution)
        max_error = 0.0
        if self.plot_residual_evolution:
            tics = [".", "*", "+", "d", "^"] * 15
            self.fig, self.axs = post_process.plot_residuals(
                ti,
                residuals,
                phases=self._ocp.phases_to_plot,
                name=f"Iter {self.iter_count}",
                fig=self.fig,
                axs=self.axs,
                tics=[tics[self.iter_count]] * 15,
            )
            self.fig, self.axs = post_process.plot_residuals(
                ti,
                residuals,
                phases=self._ocp.phases_to_plot,
                # name=f"Iter {self.iter_count}",
                fig=self.fig,
                axs=self.axs,
                tics=["-"] * 15,
            )

        segment_widths = [None] * self._ocp.n_phases
        for phase in range(self._ocp.n_phases):
            max_residual = max(
                [
                    np.abs(np.array(err)).max() if err is not None else 0
                    for err in residuals[phase]
                ]
            )
            if max_residual > max_error:
                max_error = max_residual

            segment_widths_old = self._nlp_sw_params[
                self.n_segments * phase : self.n_segments * (phase + 1)
            ]

            if max_residual < self.tol_residual[phase]:
                segment_widths[phase] = segment_widths_old
                print(
                    f"Solved phase {phase} to acceptable level {self.tol_residual[phase]}, residual: {max_residual}"
                )
                continue

            trajectories = self.init_trajectories(phase)
            x, u, t, t0, tf, a = trajectories(solution["x"], self._nlp_sw_params)
            t0, tf = np.concatenate(t0.full()), np.concatenate(tf.full())
            target_nodes = self.get_residual_grid_taus(phase)
            target_grid = self.get_interpolated_time_grid(
                t, target_nodes, self.poly_orders, self.tau0, self.tau1
            )
            du_orig = ca.mtimes(self._compD, u)
            time_at_max_slope = self.compute_time_at_max_values(
                target_grid[
                    1:-1
                ],  # Excluding end points because they are already part of segment start or ends
                np.concatenate(t.full()),
                np.abs(du_orig.full()),
                threshold=self._THRESHOLD_SLOPE,
            )

            # fig, axs = post.plot_u()
            # axs = post_process.plot_curve(
            #     axs,
            #     np.zeros((time_at_max_slope.shape[0], 1)),
            #     time_at_max_slope,
            #     tics=["o"] * 15,
            # )

            if len(time_at_max_slope) == 0:
                segment_widths[phase] = segment_widths_old
            else:
                segment_widths[phase] = self.compute_segment_widths_at_times(
                    time_at_max_slope, self.n_segments, t0[0], tf[0]
                )
                # Constrain the computed segment widths between global limits
                for i, seg in enumerate(segment_widths[phase]):
                    if seg < self.lbh[phase]:
                        segment_widths[phase][i] = self.lbh[phase]
                    if seg > self.ubh[phase]:
                        segment_widths[phase][i] = self.ubh[phase]
                # Normalize to sum to 1
                segment_widths[phase] = segment_widths[phase] / sum(
                    segment_widths[phase]
                )

        return np.concatenate(segment_widths), max_error

    @staticmethod
    def compute_time_at_max_values(t_grid, t_orig, du_orig, threshold: float = 0):
        """Compute the times corresponding to max value of the given variable (du_orig)

        args:
            :t_grid: Fixed grid
            :t_orig: time corresponding to collocation nodes and variable (du_orig)
            :du_orig: Variable to decide the output times

        returns:
            :time: Time corresponding to max, slope of given variable
        """
        # Method -1:         # select max slope entries at each time
        # du_max = du_orig.max(axis=1)
        # Method-2: 2-Norm of the slope
        du_max = np.linalg.norm(du_orig, 2, axis=1)

        # Method-3: Select only one of the control curve slope
        # du_max = np.linalg.norm(du_orig, 2, axis=1)  # du_orig[:, 0]

        # Interpolation onto fixed time grid doesnt yeild good results
        # du_grid = np.interp(t_grid, t_orig, du_max)

        t_du = [i for i in zip(t_orig[1:-1], du_max[1:-1]) if i[1] >= threshold]
        t_du.sort(key=lambda t: t[1])
        if len(t_du) > 0:
            times = np.array(t_du)[:, 0]
        else:
            times = np.array([])

        # print(times[:10])

        return times

    @staticmethod
    def compute_segment_widths_at_times(times, n_segments, t0, tf):
        """Compute seg_width fractions corresponding to given times and number of segments"""
        n_points_available = len(times)
        segment_widths = [None] * n_segments
        if n_points_available > (n_segments - 2):
            times = times[:n_segments]
            times.sort()
            segment_widths[0] = times[0] - t0
            for i in range(1, n_segments - 1):
                segment_widths[i] = times[i] - times[i - 1]
            segment_widths[n_segments - 1] = tf - times[n_segments - 2]
        else:
            times.sort()
            sw0 = times[0] - t0
            sw_end = tf - times[-1]
            n_req = n_segments - (n_points_available - 1)
            # n_req < 2: # Not possible (Belongs to if loop above)
            if n_req == 2:
                n_start = n_end = 1
            else:
                n_start = 1 + int(sw0 / (sw0 + sw_end) * (n_req - 1))
                n_end = n_req - n_start
            for i in range(n_start):
                segment_widths[i] = sw0 / n_start
            for i in range(n_start, n_start + n_points_available - 1):
                segment_widths[i] = times[i - n_start + 1] - times[i - n_start]
            for i in range(n_start + n_points_available - 1, n_segments):
                segment_widths[i] = sw_end / n_end

        # Sum must be equal to 1 : TODO verify
        segment_widths = np.array(segment_widths) / (tf - t0)

        return segment_widths


class mpopt_adaptive(mpopt):
    """Multi-stage Optimal control problem (OCP) solver which implements seg-widths
    as optimization variables and solves for them along with the optimization problem.

    Examples :
        >>> # Moon lander problem
        >>> from mpopt import mp
        >>> ocp = mp.OCP(n_states=2, n_controls=1, n_phases=1)
        >>> ocp.dynamics[0] = lambda x, u, t: [x[1], u[0] - 1.5]
        >>> ocp.running_costs[0] = lambda x, u, t: u[0]
        >>> ocp.terminal_constraints[0] = lambda xf, tf, x0, t0: [xf[0], xf[1]]
        >>> ocp.x00[0] = [10, -2]
        >>> ocp.lbu[0] = 0; ocp.ubu[0] = 3
        >>> ocp.lbtf[0] = 3; ocp.ubtf[0] = 5
        >>> opt = mp.mpopt_adaptive(ocp, n_segments=3, poly_orders=[2]*3)
        >>> solution = opt.solve()
        >>> post = opt.process_results(solution, plot=True)
    """

    _SEG_WIDTH_MIN = 1e-4
    _SEG_WIDTH_MAX = 1.0
    _TOL_RESIDUAL = 1e-3

    def __init__(
        self: "mpopt_adaptive",
        problem: "OCP",
        n_segments: int = 1,
        poly_orders: List[int] = [9],
        scheme: str = "LGR",
        **kwargs,
    ):
        """Initialize the optimizer
        args:
            n_segments: number of segments in each phase
            poly_orders: degree of the polynomial in each segment
            problem: instance of the OCP class
        """
        super().__init__(
            problem=problem,
            n_segments=n_segments,
            poly_orders=poly_orders,
            scheme=scheme,
        )
        self.mid_residuals = True

        # Segment width bounds : default values
        self.lbh = [self._SEG_WIDTH_MIN for _ in range(self._ocp.n_phases)]
        self.ubh = [self._SEG_WIDTH_MAX for _ in range(self._ocp.n_phases)]
        self.tol_residual = [self._TOL_RESIDUAL for _ in range(self._ocp.n_phases)]

    def get_nlp_variables(self, phase: int = 0):
        """Retrieve optimization variables and their bounds for a given phase

        args:
            :phase: index of the phase (starting from 0)

        returns:
            Tuple : (Z, Zmin, Zmax)
                Z - Casadi SX vector containing optimization variables for the given phase (X, U, t0, tf)
                Zmin - Lower bound for the variables in 'Z'
                Zmax - Upper bound for the variables in 'Z'
        """
        if not self._variables_created:
            self.create_variables()
        Z = ca.vertcat(
            self.X[phase][:],
            self.U[phase][:],
            self.t0[phase],
            self.tf[phase],
            self.A[:, phase],
            self.seg_widths[:, phase],
        )

        # Lower and upper bounds for the states in OCP
        xmin_vec = [self._ocp.lbx[phase] * self._ocp.scale_x] * (self._Npoints)
        xmax_vec = [self._ocp.ubx[phase] * self._ocp.scale_x] * (self._Npoints)

        # Impose constraints on initial conditions (Only in phase 0)
        if phase == 0:
            xmin_vec[0] = xmax_vec[0] = self._ocp.x00[0] * self._ocp.scale_x

        Zmin = np.concatenate(
            [
                np.concatenate(np.array(xmin_vec).T),
                np.repeat(self._ocp.lbu[phase] * self._ocp.scale_u, self._Npoints),
                self._ocp.lbt0[phase] * self._ocp.scale_t,
                self._ocp.lbtf[phase] * self._ocp.scale_t,
                self._ocp.lba[phase] * self._ocp.scale_a,
                [self.lbh[phase]] * self.n_segments,
            ]
        )
        Zmax = np.concatenate(
            [
                np.concatenate(np.array(xmax_vec).T),
                np.repeat(self._ocp.ubu[phase] * self._ocp.scale_u, self._Npoints),
                self._ocp.ubt0[phase] * self._ocp.scale_t,
                self._ocp.ubtf[phase] * self._ocp.scale_t,
                self._ocp.uba[phase] * self._ocp.scale_a,
                [self.ubh[phase]] * self.n_segments,
            ]
        )

        return (Z, Zmin, Zmax)

    def init_solution_per_phase(self, phase: int) -> np.ndarray:
        """Initialize solution vector at all collocation nodes of a given phase.

        The initial solution for a given phase is estimated from the initial and
        terminal conditions defined in the OCP. Simple linear interpolation between
        initial and terminal conditions is used to estimate solution at interior collocation nodes.

        args:
            :phase: index of phase

        returns:
            solution : initialized solution for given phase
        """
        z0 = [None] * 6
        x00 = self._ocp.x00[phase] * self._ocp.scale_x
        xf0 = self._ocp.xf0[phase] * self._ocp.scale_x
        u00 = self._ocp.u00[phase] * self._ocp.scale_u
        uf0 = self._ocp.uf0[phase] * self._ocp.scale_u
        t00 = self._ocp.t00[phase] * self._ocp.scale_t
        tf0 = self._ocp.tf0[phase] * self._ocp.scale_t
        a0 = self._ocp.a0[phase] * self._ocp.scale_a

        # Linear interpolation of states
        z0[0] = np.concatenate(
            np.array(
                [
                    x00 + (xf0 - x00) / (tf0 - t00) * (t - t00)
                    for t in np.linspace(
                        t00,
                        tf0,
                        self._Npoints,
                    )
                ]
            ).T
        )
        # Linear interpolation of controls
        z0[1] = np.concatenate(
            np.array(
                [
                    u00 + (uf0 - u00) / (tf0 - t00) * (t - t00)
                    for t in np.linspace(
                        t00,
                        tf0,
                        self._Npoints,
                    )
                ]
            )
        )
        z0[2], z0[3], z0[4] = t00, tf0, a0
        z0[5] = [1.0 / self.n_segments] * self.n_segments

        return np.concatenate(z0)

    def get_nlp_constrains_for_segment_widths(self, phase: int = 0) -> Tuple:
        """Add additional constraints on segment widths to the original NLP"""
        sw, swmin, swmax = [], [], []
        # Sum equals 1 (Segment width is normalized)
        sw.append(ca.sum1(self.seg_widths[:, phase]) - 1.0)
        swmin.append([0.0])
        swmax.append([0.0])

        # Mid point control input, dynamics constraints
        mid_points = lambda tau: [
            (tau[i] + tau[i + 1]) / 2.0 for i in range(len(tau) - 1)
        ]
        taus_mid = [
            mid_points(self.collocation._taus_fn(deg)) for deg in self.poly_orders
        ]
        n_times = len(self.time_grid[phase])
        ti = ca.vertcat(
            *[
                (self.time_grid[phase][i] + self.time_grid[phase][i + 1]) / 2.0
                for i in range(n_times - 1)
            ]
        )

        comp_interpolation_I = self.collocation.get_composite_interpolation_matrix(
            taus_mid, self.poly_orders
        )
        comp_interpolation_D = self.collocation.get_composite_interpolation_Dmatrix_at(
            taus_mid, self.poly_orders
        )
        xi = ca.mtimes(comp_interpolation_I, self.X[phase])
        ui = ca.mtimes(comp_interpolation_I, self.U[phase])

        # Control input constraints
        if (self._ocp.lbu[phase] > -np.inf).any() or (
            self._ocp.ubu[phase] < np.inf
        ).any():
            sw.append(ui[:])
            n_mid = ui.size1()
            swmin.append(np.repeat(self._ocp.lbu[phase] * self._ocp.scale_u, n_mid))
            swmax.append(np.repeat(self._ocp.ubu[phase] * self._ocp.scale_u, n_mid))

        # State constraints at mid points
        if (self._ocp.lbx[phase] > -np.inf).any() or (
            self._ocp.ubx[phase] < np.inf
        ).any():
            sw.append(xi[:])
            n_mid = xi.size1()
            swmin.append(np.repeat(self._ocp.lbx[phase] * self._ocp.scale_x, n_mid))
            swmax.append(np.repeat(self._ocp.ubx[phase] * self._ocp.scale_x, n_mid))

        # Mid point residuals
        if self.mid_residuals:
            dynamics = self._ocp.get_dynamics(phase)
            Dxi = ca.mtimes(comp_interpolation_D, self.X[phase])
            index = 0
            n_taus = [len(taus) for taus in taus_mid]
            residual_phase = [None] * self.n_segments
            for seg in range(self.n_segments):
                h_seg = (
                    (self.tf[phase] - self.t0[phase])
                    / self._ocp.scale_t
                    / (self.tau1 - self.tau0)
                    * self.seg_widths[seg, phase]
                )
                taus = taus_mid[seg]
                f = [None] * n_taus[seg]
                for i, tau in enumerate(taus):
                    f[i] = (
                        h_seg
                        * ca.mtimes(
                            self._scX,
                            ca.vertcat(
                                *dynamics(
                                    xi[index, :].T / self._ocp.scale_x,
                                    ui[index, :].T / self._ocp.scale_u,
                                    ti[index],
                                    self.A[:, phase] / self._ocp.scale_a,
                                )
                            ),
                        ).T
                    )
                    index += 1
                start, end = sum(n_taus[:seg]), sum(n_taus[: (seg + 1)])
                if start == end:
                    continue
                residual_phase[seg] = (
                    self.seg_widths[seg, phase]
                    * (Dxi[start:end, :] - ca.vertcat(*f))[:]
                )

            residuals = ca.vertcat(*residual_phase)
            n_residuals = residuals.size1()
            sw.append(residuals)
            swmin.append([-self.tol_residual[phase]] * n_residuals)
            swmax.append([self.tol_residual[phase]] * n_residuals)

        (SW, SWmin, SWmax) = (
            ca.vertcat(*sw),
            np.concatenate(swmin),
            np.concatenate(swmax),
        )

        return (SW, SWmin, SWmax)

    def discretize_phase(self, phase: int) -> Tuple:
        """Discretize single phase of the Optimal Control Problem

        args:
            phase: index of the phase (starting from 0)
        returns :
            Tuple : Constraint vector (G, Gmin, Gmax) and objective function (J)
        """
        if not self._collocation_approximation_computed:
            self.compute_numerical_approximation()
        if not self._variables_created:
            self.create_variables()

        # Discretize OCP
        (f, c, q) = self.get_discretized_dynamics_constraints_and_cost_matrices(phase)
        (F, Fmin, Fmax) = self.get_nlp_constraints_for_dynamics(f, phase)
        (C, Cmin, Cmax) = self.get_nlp_constraints_for_path_contraints(c, phase)
        (
            TC,
            TCmin,
            TCmax,
            mayer_term,
        ) = self.get_nlp_constraints_for_terminal_contraints(phase)
        (DU, DUmin, DUmax) = self.get_nlp_constraints_for_control_input_slope(phase)

        # Constraints related to segment widths
        (SW, SWmin, SWmax) = self.get_nlp_constrains_for_segment_widths(phase)

        # Add running cost to the mayer term
        J = mayer_term + ca.mtimes(self._compW, ca.vertcat(*q))

        # Merge constraint vectors into sigle constraint vector
        G = ca.vertcat(*[F, C, DU, TC, SW])
        Gmin = np.concatenate([Fmin, Cmin, DUmin, TCmin, SWmin])
        Gmax = np.concatenate([Fmax, Cmax, DUmax, TCmax, SWmax])

        return (G, Gmin, Gmax, J)

    def create_solver(self, solver: str = "ipopt", options: Dict = {}) -> None:
        """Create NLP solver

        args:
            :solver: Optimization method to be used in nlp_solver (List of plugins
                    avaiable at http://casadi.sourceforge.net/v2.0.0/api/html/d6/d07/classcasadi_1_1NlpSolver.html)
            :options: Dictionary
                List of options for the optimizer (Based on CasADi documentation)

        returns:
            None

        Updates the nlpsolver object in the present optimizer class
        """
        nlp_problem, self.nlp_bounds = self.create_nlp()
        if "p" in nlp_problem:
            nlp_problem.pop("p")
        default_options = (
            {
                "ipopt.max_iter": 2000,
                "ipopt.acceptable_tol": 1e-4,
                "ipopt.print_level": 0,
                "ipopt.sb": "yes",
                "print_time": 0,
            }
            if solver == "ipopt"
            else dict()
        )
        for key in options:
            default_options[key] = options[key]
        self.nlp_solver = ca.nlpsol("solver", solver, nlp_problem, default_options)
        self._nlpsolver_initialized = True

    def solve(
        self,
        initial_solution: Dict = None,
        reinitialize_nlp: bool = False,
        solver: str = "ipopt",
        nlp_solver_options: Dict = {},
        mpopt_options: Dict = {},
        **kwargs,
    ) -> Dict:
        """Solve the Nonlinear Programming problem

        args:
            :init_solution: Dictionary containing initial solution with keys
                x or x0 - Initional solution for the nlp variables

            :reinitialize_nlp: (True, False)
                True - Reinitialize NLP solver object
                False - Use already created object if available else create new one

            :nlp_solver_options: Options to be passed to the nlp_solver while creating
                the solver object, not while solving (like initial conditions)
            :mpopt_options: Options dict for the optimizer

        returns:
            :solution: Solution as reported by the given nlp_solver object

        """
        if (not self._nlpsolver_initialized) or (reinitialize_nlp):
            self.create_solver(solver=solver, options=nlp_solver_options)

        solver_inputs = self.get_solver_warm_start_input_parameters(initial_solution)

        solution = self.nlp_solver(**solver_inputs, **self.nlp_bounds)

        sw_fn = ca.Function("sw", [self.Z], [self.seg_widths[:]], ["z"], ["h"])
        self._nlp_sw_params = np.concatenate(sw_fn(solution["x"]).full())
        print(f"Optimal segment width fractions: {self._nlp_sw_params}")

        return solution

    def init_trajectories(self, phase: int = 0) -> ca.Function:
        """Initialize trajectories of states, constrols and time variables

        args:
            :phase: index of the phase

        returns:
            :trajectories: CasADi function which returns states, controls and time
                        variable for the given phase when called with NLP solution
                        vector of all phases
                t0, tf - unscaled AND
                x, u, t - scaled trajectories
        """
        x = self.X[phase]
        u = self.U[phase]
        a = self.A[:, phase]
        t0, tf = self.t0[phase] / self._ocp.scale_t, self.tf[phase] / self._ocp.scale_t
        t = ca.vertcat(*self.time_grid[phase])
        tmp_seg = ca.SX.sym("htmp_seg", self.n_segments, self._ocp.n_phases)
        trajectories = ca.Function(
            "x_traj",
            [self.Z, tmp_seg[:]],
            [x, u, t, t0, tf, a],
            ["z", "h"],
            ["x", "u", "t", "t0", "tf", "a"],
        )

        return trajectories

    def process_results(
        self,
        solution,
        plot: bool = True,
        scaling: bool = False,
        residual_x: bool = False,
        residual_dx: bool = True,
    ):
        """Post process the solution of the NLP

        args:
            :solution: NLP solution as reported by the solver
            :plot: bool
                True - Plot states and variables in a single plot with states in a subplot and controls in another.
                False - No plot
            :scaling: bool
                True - Plot the scaled variables
                False - Plot unscaled variables meaning, original solution to the problem
            :residuals: bool
                To plot norma of the residual in the dynamics evaluated at points different from collocation nodes

        returns:
            :post: Object of post_process class (Initialized)

        """
        start_time = time.monotonic()
        trajectories = [
            self.init_trajectories(phase) for phase in range(self._ocp.n_phases)
        ]
        traj_time = time.monotonic()

        resid_value = {}
        if residual_x:
            x_int, u_int, ti, res_x = self.get_states_residuals(solution)
            resid_value["t_x"] = [ti, res_x]
            rx_time = time.monotonic()
        if residual_dx:
            tdx, res_dx = self.get_dynamics_residuals(solution)
            resid_value["t_dx"] = [tdx, res_dx]
            rdx_time = time.monotonic()
        else:
            resid_value = None

        r_time = time.monotonic()

        options = {
            "nx": self._ocp.nx,
            "nu": self._ocp.nu,
            "na": self._ocp.na,
            "nPh": self._ocp.n_phases,
            "ns": self.n_segments,
            "poly_orders": self.poly_orders,
            "N": self._Npoints,
            "phases_to_plot": self._ocp.phases_to_plot,
            "scale_x": self._ocp.scale_x,
            "scale_u": self._ocp.scale_u,
            "scale_a": self._ocp.scale_a,
            "scale_t": self._ocp.scale_t,
            "scaling": scaling,
            "colloc_scheme": self.colloc_scheme,
            "tau0": self.tau0,
            "tau1": self.tau1,
            "interpolation_depth": 3,
            "seg_widths": self._nlp_sw_params,
            "residuals": resid_value,
        }

        post = post_process(solution, trajectories, options)

        if plot:
            for phases in self._ocp.phases_to_plot:
                residuals = residual_x or residual_dx
                post.plot_phases(phases, residuals=residuals)

        post_time = time.monotonic()

        if not self._MUTE_:
            print(f"\n Post processed in {round((post_time - start_time)*1e3, 3)} ms")
            print(
                f" \t Solution retrieval            : {round((traj_time - start_time)*1e3, 3)} ms"
            )
            if residual_x:
                print(
                    f" \t Residual in states             : {round((rx_time - traj_time)*1e3, 3)} ms"
                )
                if residual_dx:
                    print(
                        f" \t Residual in dynamics          : {round((rdx_time - rx_time)*1e3, 3)} ms"
                    )
            elif residual_dx:
                print(
                    f" \t Residual in dynamics           : {round((rdx_time - traj_time)*1e3, 3)} ms"
                )
            print(
                f" \t Process solution and plot      : {round((post_time - r_time)*1e3, 3)} ms"
            )

        return post


class OCP:
    """Define Optimal Control Problem

    Optimal control problem definition in standard Bolza form.

    Examples of usage:
        >>> ocp = OCP(n_states=1, n_controls=1, n_phases=1)
        >>> ocp.dynamics[0] = lambda x, u, t, a: [u[0]]
        >>> ocp.path_constraints[0] = lambda x, u, t, a: [x[0] + u[0]]
        >>> ocp.running_costs[0] = lambda x, u, t, a: x[0]
        >>> ocp.terminal_costs[0] = lambda xf, tf, x0, t0, a: xf[0]
        >>> ocp.terminal_constraints[0] = lambda xf, tf, x0, t0, a: [xf[0] + 2]
    """

    LB_DYNAMICS = 0
    UB_DYNAMICS = 0
    LB_PATH_CONSTRAINTS = -np.inf
    UB_PATH_CONSTRAINTS = 0
    LB_TERMINAL_CONSTRAINTS = 0
    UB_TERMINAL_CONSTRAINTS = 0

    def __init__(
        self: "OCP",
        n_states: int = 1,
        n_controls: int = 1,
        n_phases: int = 1,
        n_params=0,
        **kwargs,
    ):
        """Initialize OCP object
        For all phases, number of states and controls are assumed to be same.
        It's easy to connect when the states and controls are same in multi-phase OCP.

        args:
            :n_states: number of state variables in the OCP
            :n_controls: number of control variables in the OCP
            :n_params: number of algebraic parameters in each phase
            :n_phases: number of phases in the OCP

        returns:
            OCP class object with default initialization of methods and parameters
        """
        self.nx = n_states
        self.nu = n_controls
        self.na = n_params
        self.n_phases = n_phases

        # Define OCP terms
        dynamics = lambda x, u, t, a=None: [0] * self.nx
        self.dynamics = [dynamics] * self.n_phases

        path_constraints = lambda x, u, t, a=None: None
        self.path_constraints = [path_constraints] * self.n_phases

        terminal_cost = lambda xf, tf, x0, t0, a=None: 0
        self.terminal_costs = [terminal_cost] * self.n_phases

        running_cost = lambda x, u, t, a=None: 0
        self.running_costs = [running_cost] * self.n_phases

        terminal_constraints = lambda xf, tf, x0, t0, a=None: None
        self.terminal_constraints = [terminal_constraints] * self.n_phases

        # Define defaults
        self.phase_links = [(i, i + 1) for i in range(self.n_phases - 1)]

        # Scaling
        self.scale_x = np.array([1.0] * self.nx)
        self.scale_u = np.array([1.0] * self.nu)
        self.scale_a = np.array([1.0] * self.na)
        self.scale_t = 1.0

        # Initial guess
        self.x00 = np.array([[0.0] * self.nx for _ in range(self.n_phases)])
        self.xf0 = np.array([[0.0] * self.nx for _ in range(self.n_phases)])
        self.u00 = np.array([[0.0] * self.nu for _ in range(self.n_phases)])
        self.uf0 = np.array([[0.0] * self.nu for _ in range(self.n_phases)])
        self.t00 = np.array([[0.0]] * self.n_phases)
        self.tf0 = np.array([[1.0]] * self.n_phases)
        self.a0 = np.array([[0.0] * self.na for _ in range(self.n_phases)])

        # Default bounds
        self.lbx = np.array([[-np.inf] * self.nx for _ in range(self.n_phases)])
        self.ubx = np.array([[np.inf] * self.nx for _ in range(self.n_phases)])
        self.lbu = np.array([[-np.inf] * self.nu for _ in range(self.n_phases)])
        self.ubu = np.array([[np.inf] * self.nu for _ in range(self.n_phases)])
        self.lba = np.array([[-np.inf] * self.na for _ in range(self.n_phases)])
        self.uba = np.array([[np.inf] * self.na for _ in range(self.n_phases)])
        self.lbt0 = np.array([[0.0]] * self.n_phases)
        self.ubt0 = np.array([[np.inf]] * self.n_phases)
        # First phase always starts at time zero. Hence, both lower and upper
        # bounds of t0 are set to 0.0
        self.ubt0[0] = 0.0
        self.lbtf = np.array([[0.0]] * self.n_phases)
        self.ubtf = np.array([[np.inf]] * self.n_phases)

        # Default phase continuity breaks
        self.lbe = np.array([[0.0] * self.nx for _ in range(self.n_phases - 1)])
        self.ube = np.array([[0.0] * self.nx for _ in range(self.n_phases - 1)])

        # Slope has_constraints
        self.diff_u = np.array([0] * self.n_phases)
        self.lbdu = np.array([-15 for _ in range(self.n_phases)])
        self.ubdu = np.array([15 for _ in range(self.n_phases)])

        # Constrains on control input at mid point of collocation nodes
        # Enabled by default
        self.midu = np.array([1] * self.n_phases)
        self.du_continuity = np.array([0] * self.n_phases)

        # Default post-processing settings
        self.n_figures = 1
        self.phases_to_plot = [tuple(range(self.n_phases))]
        self.plot_type = 1
        self.plot_interpolation_level = 3

    def get_dynamics(self, phase: int = 0):
        """Get dynamics function for the given phase

        args:
            :phase: index of the phase (starting from 0)

        returns:
            :dynamics: system dynamics function with arguments x, u, t, a
        """
        if self.na == 0:
            dynamics = lambda x, u, t, a: self.dynamics[phase](x, u, t)
            return dynamics

        return self.dynamics[phase]

    def get_path_constraints(self, phase: int = 0):
        """Get path constraints function for the given phase

        args:
            :phase: index of the phase (starting from 0)

        returns:
            :path_constraints: path constraints function with arguments x, u, t, a
        """
        if self.na == 0:
            path_constraints = lambda x, u, t, a: self.path_constraints[phase](x, u, t)
            return path_constraints

        return self.path_constraints[phase]

    def get_terminal_constraints(self, phase: int = 0):
        """Get terminal_constraints function for the given phase

        args:
            :phase: index of the phase (starting from 0)

        returns:
            :terminal_constraints: system terminal_constraints function with arguments x, u, t, a
        """
        if self.na == 0:
            terminal_constraints = lambda xf, tf, x0, t0, a: self.terminal_constraints[
                phase
            ](xf, tf, x0, t0)
            return terminal_constraints

        return self.terminal_constraints[phase]

    def get_running_costs(self, phase: int = 0):
        """Get running_costs function for the given phase

        args:
            :phase: index of the phase (starting from 0)

        returns:
            :running_costs: system running_costs function with arguments x, u, t, a
        """
        if self.na == 0:
            running_costs = lambda x, u, t, a: self.running_costs[phase](x, u, t)
            return running_costs

        return self.running_costs[phase]

    def get_terminal_costs(self, phase: int = 0):
        """Get terminal_costs function for the given phase

        args:
            :phase: index of the phase (starting from 0)

        returns:
            :terminal_costs: system terminal_costs function with arguments x, u, t, a
        """
        if self.na == 0:
            terminal_costs = lambda xf, tf, x0, t0, a: self.terminal_costs[phase](
                xf, tf, x0, t0
            )
            return terminal_costs

        return self.terminal_costs[phase]

    def has_path_constraints(self, phase: int = 0) -> bool:
        """Check if given phase has path constraints in given OCP

        args:
            :phase: index of phase

        return:
            :status: bool (True/False)
        """
        path_constraints = self.path_constraints[phase]

        if self.na == 0:
            return (
                path_constraints(self.x00[phase], self.u00[phase], self.t00[phase])
                is not None
            )
        return (
            path_constraints(
                self.x00[phase], self.u00[phase], self.t00[phase], self.a0[phase]
            )
            is not None
        )

    def has_terminal_constraints(self, phase: int = 0) -> bool:
        """Check if given phase has terminal equality constraints in given OCP

        args:
            :phase: index of phase

        return:
            :status: bool (True/False)
        """
        terminal_constraints = self.terminal_constraints[phase]

        if self.na == 0:
            return (
                terminal_constraints(
                    self.xf0[phase],
                    self.tf0[phase],
                    self.x00[phase],
                    self.t00[phase],
                )
                is not None
            )
        return (
            terminal_constraints(
                self.xf0[phase],
                self.tf0[phase],
                self.x00[phase],
                self.t00[phase],
                self.a0[phase],
            )
            is not None
        )

    def validate(self) -> None:
        """Validate dimensions and initialization of attributes"""
        assert self.n_phases > 0

        assert len(self.dynamics) == self.n_phases
        assert len(self.running_costs) == self.n_phases
        assert len(self.terminal_costs) == self.n_phases
        assert len(self.path_constraints) == self.n_phases
        assert len(self.terminal_constraints) == self.n_phases

        for phase in range(self.n_phases):
            x, u, t, a = (
                self.x00[phase],
                self.u00[phase],
                self.t00[phase],
                self.a0[phase],
            )
            dynamics = self.get_dynamics(phase)
            terminal_costs = self.get_terminal_costs(phase)
            running_costs = self.get_running_costs(phase)
            path_constraints = self.get_path_constraints(phase)
            terminal_constraints = self.get_terminal_constraints(phase)
            assert len(dynamics(x, u, t, a)) == self.nx
            assert terminal_costs(x, t, x, t, a) is not None
            assert running_costs(x, u, t, a) is not None
            pc = path_constraints(x, u, t, a)
            tc = terminal_constraints(x, t, x, t, a)

            if pc is not None:
                assert len(pc) > 0
            if tc is not None:
                assert len(tc) > 0

        assert len(self.scale_x) == self.nx
        assert len(self.scale_u) == self.nu
        assert len(self.scale_a) == self.na

        assert self.x00.shape == (self.n_phases, self.nx)
        assert self.xf0.shape == (self.n_phases, self.nx)
        assert self.u00.shape == (self.n_phases, self.nu)
        assert self.uf0.shape == (self.n_phases, self.nu)
        assert self.a0.shape == (self.n_phases, self.na)
        assert self.a0.shape == (self.n_phases, self.na)
        assert self.t00.shape == (self.n_phases, 1)
        assert self.tf0.shape == (self.n_phases, 1)

        assert self.lbx.shape == (self.n_phases, self.nx)
        assert self.ubx.shape == (self.n_phases, self.nx)
        assert self.lbu.shape == (self.n_phases, self.nu)
        assert self.ubu.shape == (self.n_phases, self.nu)
        assert self.lba.shape == (self.n_phases, self.na)
        assert self.uba.shape == (self.n_phases, self.na)
        assert self.lbt0.shape == (self.n_phases, 1)
        assert self.ubt0.shape == (self.n_phases, 1)
        assert self.lbtf.shape == (self.n_phases, 1)
        assert self.ubtf.shape == (self.n_phases, 1)

        assert self.lbe.shape[0] == self.n_phases - 1
        assert self.ube.shape[0] == self.n_phases - 1
        if self.n_phases > 1:
            assert self.lbe.shape[1] == self.nx
            assert self.ube.shape[1] == self.nx

        # Check if lower bound is less than upper bound
        for phase in range(self.n_phases):
            for i in range(self.nx):
                assert self.lbx[phase][i] <= self.ubx[phase][i]
            for i in range(self.nu):
                assert self.lbu[phase][i] <= self.ubu[phase][i]
            for i in range(self.na):
                assert self.lba[phase][i] <= self.uba[phase][i]
            assert self.lbt0[phase] <= self.ubt0[phase]
            assert self.lbtf[phase] <= self.ubtf[phase]
            if phase < self.n_phases - 1:
                for i in range(self.nx):
                    assert self.lbe[phase][i] <= self.ube[phase][i]


class Collocation:
    """Collocation functionality for optimizer

    Functionality related to polynomial basis, respective differential and
    integral matrices calculation is implemented here.

    Numpy polynomial modules is used for calculating differentiation and quadrature weights.
    Hence, computer precision can affect these derivative calculations

    """

    D_MATRIX_METHOD = "symbolic"  # "symbolic"
    TVAR = ca.SX.sym("t")

    def __init__(
        self,
        poly_orders: List = [],
        scheme: str = "LGR",
        polynomial_type: str = "lagrange",
    ):
        """Initialize the collocation object

        args:
            :poly_orders: List of polynomial degrees used in collocation
            :scheme: Scheme used to define collocation nodes (Possible options - "LG", "LGR", "LGL", "CGL")
                LG - Legendre-Gauss (LG)
                LGR - Legendre-Gauss-Radau (LGR)
                LGL - Legendre-Gauss-Lobatto (LGL)
                CGL - Chebyshev-Gauss-Lobatto (CGL)
            :polynomial_type: polynomials used in state and control approximation
                - lagrange
        """
        self.poly_orders = poly_orders
        colloc_roots = CollocationRoots(scheme)
        self._taus_fn = colloc_roots._taus_fn
        self.tau0 = colloc_roots._TAU_MIN
        self.tau1 = colloc_roots._TAU_MAX
        self.poly_fn = self.get_polynomial_function(polynomial_type)
        self.roots, self.polys = {}, {}
        self.unique_polys = set(self.poly_orders)
        self.init_polynomials(self.unique_polys)

        # Diff_matrix_fn takes in degree of the polynomial(n) as argument and
        # returns (n+1)x(n+1) D matrix as output
        self.diff_matrix_fn = self.get_diff_matrix_fn(polynomial_type)
        self.quad_matrix_fn = self.get_quadrature_weights_fn(polynomial_type)

    @classmethod
    def get_diff_matrix_fn(self, polynomial_type: str = "lagrange"):
        """Return a function that returns differentiation matrix

        args:
            :polynomial_type: (lagrange)

        returns:
            Diff matrix function with arguments (degree, taus_at=None)
        """
        return self.get_diff_matrix

    @classmethod
    def get_quadrature_weights_fn(self, polynomial_type: str = "lagrange"):
        """Return a function that returns quadrature weights for the cost function approx.

        args:
            :polynomial_type: (lagrange)

        returns:
            quadrature weights function with arguments (degree)
        """
        return self.get_quadrature_weights

    @classmethod
    def get_polynomial_function(self, polynomial_type: str = "lagrange"):
        """Get function which returns basis polynomials for the collocation given
        polynomial degree

        args:
            :polynomial_type: str, 'lagrange'

        returns:
            :poly_basis_fn: Function which returns basis polynomials
        """
        if polynomial_type == "lagrange":
            return self.get_lagrange_polynomials
        else:
            # Only lagrange basis polynomials are supported as of now
            return 0

    def init_polynomials(self, poly_orders) -> None:
        """Initialize roots of the polynomial and basis polynomials

        args:
            :poly_orders: List of polynomial degrees used in collocation
        """
        for degree in poly_orders:
            self.roots[degree] = self._taus_fn(degree)
            self.polys[degree] = self.poly_fn(self.roots[degree])

    def init_polynomials_with_customized_roots(self, roots_dict: Dict = None) -> None:
        """
        Initialize polynomials with predefined roots

        args:
            :roots_dict: Dictionary with a key for the roots and polys (Ideally not numbers as they are already taken by the regular polynominals)
        """
        for key in roots_dict:
            self.roots[key] = roots_dict[key]
            self.polys[key] = self.poly_fn(self.roots[key])

    def get_diff_matrix(self, key, taus: np.ndarray = None, order: int = 1):
        """Get differentiation matrix corresponding to given basis polynomial degree

        args:
            :degree: order of the polynomial used in collocation
            :taus: Diff matrix computed at these nodes if not None.

        returns:
            :D: Differentiation matrix
        """
        if (key not in self.roots) or (key not in self.polys):
            self.init_polynomials([key])
        eval_D_at = self.roots[key] if taus is None else taus
        n_i = len(eval_D_at)
        n_j = len(self.polys[key])
        D = ca.DM.zeros((n_i, n_j))
        for j, p in enumerate(self.polys[key]):
            if self.D_MATRIX_METHOD == "symbolic":
                if order == 1:
                    pder = ca.Function("pder", [self.TVAR], [ca.gradient(p, self.TVAR)])
                elif order == 2:
                    pder = ca.Function(
                        "pder1",
                        [self.TVAR],
                        [ca.gradient(ca.gradient(p, self.TVAR), self.TVAR)],
                    )
            else:
                if order == 1:
                    pder = np.polyder(p)
                elif order == 2:
                    pder = np.polyder(np.polyder(p))
            for i in range(n_i):
                D[i, j] = pder(eval_D_at[i])

        return D

    def get_quadrature_weights(self, key, tau0=None, tau1=None):
        """Get quadrature weights corresponding to given basis polynomial degree

        args:
            :degree: order of the polynomial used in collocation
        """
        if (key not in self.roots) or (key not in self.polys):
            self.init_polynomials([key])

        if tau0 is None:
            tau0 = self.tau0
        if tau1 is None:
            tau1 = self.tau1

        n = len(self.roots[key])
        quad_weights = ca.DM.zeros(n)
        for i in range(n):
            if self.D_MATRIX_METHOD == "symbolic":
                pint = ca.integrator(
                    "pint",
                    "idas",
                    {"x": ca.SX.sym("x"), "t": self.TVAR, "ode": self.polys[key][i]},
                    tau0, tau1
                )
                quad_weights[i] = pint(x0=0)[
                    "xf"
                ]  # By default integrator evaluates the final value at tf
            else:
                pint = np.polyint(self.polys[key][i])
                quad_weights[i] = pint(tau1) - pint(tau0)

        return quad_weights

    def get_interpolation_matrix(self, taus, degree):
        """Get interpolation matrix corresponsing nodes (taus) where the segment is approximated with polynomials of degree (degree)

        args:
            :taus: Points where interpolation is performed
            :degree: Order of the collocation polynomial
        """
        if (degree not in self.roots) or (degree not in self.polys):
            self.init_polynomials([degree])

        n_j = len(self.roots[degree])  # cols
        n_i = len(taus)  # rows
        C = ca.DM.zeros((n_i, n_j))
        for j, p in enumerate(self.polys[degree]):
            if self.D_MATRIX_METHOD == "symbolic":
                poly = ca.Function("p", [self.TVAR], [p])
            else:
                poly = p
            for i in range(n_i):
                C[i, j] = poly(taus[i])

        return C

    def get_diff_matrices(self, poly_orders: List = None, order: int = 1):
        """Get differentiation matrices for given collocation approximation

        args:
            :poly_orders: order of the polynomials used in collocation with each element representing one segment
        """
        unique_polys = self.unique_polys if poly_orders is None else set(poly_orders)
        diff_mat_dict = {}
        for degree in unique_polys:
            diff_mat_dict[degree] = self.diff_matrix_fn(self, degree, order=order)

        return diff_mat_dict

    def get_interpolation_Dmatrices_at(self, taus, keys: List = None, order: int = 1):
        """Get differentiation matrices at the interpolated nodes (taus), different
        from the collocation nodes.

        args:
            :taus: List of scaled taus (between 0 and 1) with length of list equal
                to length of poly_orders (= number of segments)
            :keys: keys of the roots and polys Dict element containing roots of the legendre polynomials and polynomials themselves

        returns:
            :Dict : (key, value)
                key - segment number (starting from 0)
                value - Differentiation matrix(C) such that DX_tau = D*X_colloc where
                    X_colloc is the values of states at the collocation nodes
        """
        if keys is None:
            keys = self.poly_orders
        basis_Dmats = {}
        for i, key in enumerate(keys):
            basis_Dmats[i] = self.diff_matrix_fn(self, key, taus=taus[i], order=order)

        return basis_Dmats

    def get_quad_weight_matrices(self, keys: List = None, tau0=None, tau1=None):
        """Get quadrature weights for given collocation approximation

        args:
            :keys: keys of the Dict element (roots and polys), Normally these keys are equal to the order of the polynomial

        """
        unique_keys = self.unique_polys if keys is None else set(keys)
        quad_mats = {}

        if tau0 == None:
            tau0 = self.tau0
        if tau1 == None:
            tau1 = self.tau1

        for key in unique_keys:
            quad_mats[key] = self.quad_matrix_fn(self, key, tau0=tau0, tau1=tau1)

        return quad_mats

    def get_interpolation_matrices(self, taus, poly_orders: List = None):
        """Get interpolation matrices corresponding to each poly_order at respective
        element in list of taus

        args:
            :taus: List of scaled taus (between 0 and 1) with length of list equal
                to length of poly_orders (= number of segments).
            :poly_orders: order of the polynomials used in collocation with each
                element representing one segment

        returns:
            Dict : (key, value)
                key - segment number (starting from 0)
                value - interpolation matrix(C) such that X_new = C*X

        """
        if poly_orders is None:
            poly_orders = self.poly_orders
        basis_mats = {}
        for i, degree in enumerate(poly_orders):
            basis_mats[i] = self.get_interpolation_matrix(taus[i], degree)

        return basis_mats

    @classmethod
    def get_lagrange_polynomials(self, roots):
        """Get basis polynomials given the collocation nodes

        args:
            :roots: Collocation points

        """
        n = len(roots)
        polys = [None] * n

        if self.D_MATRIX_METHOD == "symbolic":
            for j in range(n):
                p = ca.DM(1)
                for i in range(n):
                    if i != j:
                        p *= (self.TVAR - roots[i]) / (roots[j] - roots[i])
                polys[j] = p
        else:
            for j in range(n):
                p = np.poly1d([1])
                for i in range(n):
                    if i != j:
                        p *= np.poly1d([1, -roots[i]]) / (roots[j] - roots[i])
                polys[j] = p

        return polys

    def get_composite_differentiation_matrix(
        self, poly_orders: List = None, order: int = 1
    ):
        """Get composite differentiation matrix for given collocation approximation

        args:
            :poly_orders: order of the polynomials used in collocation with each
                element representing one segment

        """
        D = self.get_diff_matrices(poly_orders, order=order)
        if poly_orders is None:
            poly_orders = self.poly_orders
        n_nodes = sum(poly_orders) + 1

        comp_diff_matrix = ca.DM.zeros((n_nodes, n_nodes))
        for i, p in enumerate(poly_orders):
            if i == 0:
                comp_diff_matrix[0 : p + 1, 0 : p + 1] = D[p]
            else:
                start = sum(poly_orders[:i])
                comp_diff_matrix[
                    start + 1 : (start + 1) + p, start : start + (1 + p)
                ] = D[p][1:, :]
        return comp_diff_matrix

    def get_composite_quadrature_weights(
        self, poly_orders: List = None, tau0=None, tau1=None
    ):
        """Get composite quadrature weights for given collocation approximation

        args:
            :poly_orders: order of the polynomials used in collocation with each
                element representing one segment

        """
        if poly_orders is None:
            poly_orders = self.poly_orders

        if tau0 is None:
            tau0 = self.tau0
        if tau1 is None:
            tau1 = self.tau1

        quad_mats = self.get_quad_weight_matrices(poly_orders, tau0=tau0, tau1=tau1)
        comp_quad_weights = ca.vertcat(
            *([quad_mats[poly_orders[0]][0]] + [quad_mats[p][1:] for p in poly_orders])
        ).T

        return comp_quad_weights

    def get_composite_interpolation_matrix(self, taus, poly_orders: List = None):
        """Get interpolation matrix corresponding to given basis polynomial degree for colloation.

        args:
            :taus: List of scaled taus (between 0 and 1) with length of list equal
                to length of poly_orders (= number of segments).
                Note- taus are not assumed to have overlap between segments(end element != start of next phase)
            :poly_orders: order of the polynomials used in collocation with each
                element representing one segment

        returns:
            :I: composite interpolation matrix
        """
        C = self.get_interpolation_matrices(taus, poly_orders)
        if poly_orders is None:
            poly_orders = self.poly_orders
        n_nodes = sum(poly_orders) + 1
        n_segments = len(poly_orders)
        n_taus = [len(taus[i]) for i in range(len(taus))]
        n_points = sum(n_taus)

        comp_matrix = np.zeros((n_points, n_nodes))
        for i, p in enumerate(poly_orders):
            if n_taus[i] == 0:
                continue
            start_row, start_col = sum(n_taus[:i]), sum(poly_orders[:i])
            comp_matrix[
                start_row : start_row + n_taus[i],
                start_col : start_col + (1 + p),
            ] = C[i]
        return comp_matrix

    def get_composite_interpolation_Dmatrix_at(
        self, taus, poly_orders: List = None, order: int = 1
    ):
        """Get differentiation matrix corresponding to given basis polynomial degree
        at nodes different from collocation nodes

        args:
            :taus: List of scaled taus (between 0 and 1) with length of list equal
                to length of poly_orders (= number of segments)
            :poly_orders: order of the polynomials used in collocation with each
                element representing one segment

        returns:
            :D: Composite differentiation matrix

        """
        D = self.get_interpolation_Dmatrices_at(taus, keys=poly_orders, order=order)
        if poly_orders is None:
            poly_orders = self.poly_orders
        n_nodes = sum(poly_orders) + 1
        n_segments = len(poly_orders)
        n_taus = [len(taus[i]) for i in range(len(taus))]
        n_points = sum(n_taus)

        comp_Dmatrix = np.zeros((n_points, n_nodes))
        for i, p in enumerate(poly_orders):
            if n_taus[i] == 0:
                continue
            start_row, start_col = sum(n_taus[:i]), sum(poly_orders[:i])
            comp_Dmatrix[
                start_row : start_row + n_taus[i],
                start_col : start_col + (1 + p),
            ] = D[i]
        return comp_Dmatrix


class CollocationRoots:
    """Functionality related to commonly used gauss quadrature schemes such as

    Legendre-Gauss (LG)
    Legendre-Gauss-Radau (LGR)
    Legendre-Gauss-Lobatto (LGL)
    Chebyshev-Gauss-Lobatto (CGL)
    """

    # Min and max for the roots (Not yet implemented)
    _TAU_MIN = -1
    _TAU_MAX = 1

    def __init__(self, scheme: str = "LGR"):
        """Get differentiation matrix corresponding to given basis polynomial degree

        args:
            :degree: order of the polynomial used in collocation

        """
        self.scheme = scheme
        self._taus_fn = self.get_collocation_points(scheme)

    @classmethod
    def get_collocation_points(self, scheme: str):
        """Get function that returns collocation points for the given scheme

        args:
            :scheme: quadrature scheme to find the collocation points

        returns: Function, that retuns collocation points when called with polynomial degree
        """
        if scheme == "LG":
            return self.roots_legendre_gauss(
                tau_min=self._TAU_MIN, tau_max=self._TAU_MAX
            )
        elif scheme == "LGR":
            return self.roots_legendre_gauss_radau(
                tau_min=self._TAU_MIN, tau_max=self._TAU_MAX
            )
        elif scheme == "LGL":
            return self.roots_legendre_gauss_lobatto(
                tau_min=self._TAU_MIN, tau_max=self._TAU_MAX
            )
        elif scheme == "CGL":
            return self.roots_chebyshev_gauss_lobatto(
                tau_min=self._TAU_MIN, tau_max=self._TAU_MAX
            )
        else:
            # Unknown scheme, return equally spaced points
            return (
                lambda n_nodes: np.linspace(self._TAU_MIN, self._TAU_MAX, n_nodes)
                if n_nodes > 1
                else np.array([self._TAU_MIN, self._TAU_MAX])
            )

    @staticmethod
    def roots_legendre_gauss(tau_min=-1, tau_max=1):
        """Get legendre-gauss-radau collocation points in the interval [_TAU_MIN, _TAU_MAX)

        args: None

        returns: a function that returns collocation points given polynomial degree
        """
        # LG roots : same as scipy.special.j_roots(deg, 0, 0)[0]
        def lg_roots(deg):
            roots = np.polynomial.legendre.leggauss(deg - 1)[0]
            roots_default = np.append(-1, roots)

            return tau_min + (tau_max - tau_min) / (2) * (roots_default + 1)

        return lg_roots

    @staticmethod
    def roots_legendre_gauss_radau(tau_min=-1, tau_max=1):
        """Get legendre-gauss-radau (Left aligned) collocation points in the interval [_TAU_MIN, _TAU_MAX]

        args: None

        returns: a function that returns collocation points, given polynomial degree
        """

        def lgr_roots(deg):
            if deg > 1:
                import scipy.special

                roots = scipy.special.j_roots(deg - 1, 1.0, 0.0)[0]
                roots_minus1plus1 = np.append(np.append(-1, roots), 1.0)

                # Scale the roots to [_TAU_MIN, _TAU_MAX]
                return tau_min + (tau_max - tau_min) / (2) * (roots_minus1plus1 + 1)

            if deg == 1:
                return np.array([tau_min, tau_max])
            else:
                return np.array([0.0])

        return lgr_roots

    @staticmethod
    def roots_legendre_gauss_lobatto(tau_min=-1, tau_max=1):
        """Get legendre-gauss-lobatto collocation points in the interval [_TAU_MIN, _TAU_MAX]

        args: None

        returns: a function that returns collocation points given polynomial degree
        """

        def lgl_roots(deg):
            if deg > 1:
                import scipy.special

                roots = scipy.special.j_roots(deg - 1, 1.0, 1.0)[0]
                roots_minus1plus1 = np.append(np.append(-1, roots), 1.0)

                # Scale the roots to [_TAU_MIN, _TAU_MAX]
                return tau_min + (tau_max - tau_min) / (2) * (roots_minus1plus1 + 1)

            if deg == 1:
                return np.array([tau_min, tau_max])
            else:
                return np.array([0.0])

        # Refer https://github.com/nschloe/quadpy/blob/master/quadpy/line_segment/_gauss_lobatto.py

        return lgl_roots

    @staticmethod
    def roots_chebyshev_gauss_lobatto(tau_min=-1, tau_max=1):
        """Get Chebyshev-gauss-lobatto collocation points in the interval [_TAU_MIN, _TAU_MAX]

        args: None

        returns: a function that returns collocation points given polynomial degree
        """

        def cgl_roots(deg):
            roots = np.array([np.cos(np.pi * j / (deg)) for j in range(deg + 1)])[::-1]

            # Scale the roots to [_TAU_MIN, _TAU_MAX]
            return tau_min + (tau_max - tau_min) / (2) * (roots + 1)

        return cgl_roots


def solve(
    ocp,
    n_segments=1,
    poly_orders=9,
    scheme="LGR",
    plot=True,
    solve_dict: Dict = dict(),
    residual_x: bool = False,
    residual_dx: bool = True,
):
    """Solve OCP by creating optimizer and process results

    args:
        ocp: well defined OCP object
        n_segments
        poly_orders
        scheme : Collocation scheme (LGR, LGL, CGL)
        plot : True/False (Plot states and controls)

    returns:
        :mpo: optimizer
        :post: Post processor object
    """
    mpo = mpopt(ocp, n_segments=n_segments, poly_orders=poly_orders, scheme=scheme)
    solution = mpo.solve(**solve_dict)
    post = mpo.process_results(
        solution, plot=plot, residual_x=residual_x, residual_dx=residual_dx
    )

    return (mpo, post)


def get_segment_boundaries():
    """"""
    pass


class mpopt_ph_adaptive(mpopt):
    """Multi-stage Optimal control problem (OCP) solver which implements iterative
    procedure to refine the segment width and polynomial order in each phase adaptively

    Examples :
        >>> # Moon lander problem
        >>> from mpopt import mp
        >>> ocp = mp.OCP(n_states=2, n_controls=1, n_params=0, n_phases=1)
        >>> ocp.dynamics[0] = lambda x, u, t, a: [x[1], u[0] - 1.5]
        >>> ocp.running_costs[0] = lambda x, u, t, a: u[0]
        >>> ocp.terminal_constraints[0] = lambda xf, tf, x0, t0, a: [xf[0], xf[1]]
        >>> ocp.x00[0] = [10, -2]
        >>> ocp.lbu[0] = 0; ocp.ubu[0] = 3
        >>> ocp.lbtf[0] = 3; ocp.ubtf[0] = 5
        >>> opt = mp.mpopt_ph_adaptive(ocp, n_segments=3, poly_orders=[2]*3)
        >>> solution = opt.solve()
        >>> post = opt.process_results(solution, plot=True)
    """

    _SEG_WIDTH_MIN = 1e-5
    _SEG_WIDTH_MAX = 1

    _TOL_SEG_WIDTH_CHANGE = 0.05  # < 5% change in width fraction (Converged)
    _TOL_RESIDUAL = 1e-2

    def __init__(
        self: "mpopt_ph_adaptive",
        problem: "OCP",
        n_segments: int = 1,
        poly_orders: List[int] = [9],
        scheme: str = "LGR",
        grid_type: str = "spectral",
        max_residual: float = 1e-4,
        poly_order_min: int = 3,
        poly_order_max: int = 16,
        seg_min: int = 1,
        seg_max: int = 20,
        n_grid_points: int = 20,
        non_smooth_threshold: float = 1.05,
    ):
        """Initialize the optimizer
        args:
            n_segments: number of segments in each phase
            poly_orders: degree of the polynomial in each segment
            problem: instance of the OCP class
        """
        super().__init__(
            problem=problem,
            n_segments=n_segments,
            poly_orders=poly_orders,
            scheme=scheme,
        )

        # Check polynomial degree compatiability with default options
        if min(self.poly_orders) < poly_order_min:
            poly_order_min = min(self.poly_orders)
        if max(self.poly_orders) > poly_order_max:
            poly_order_max = max(self.poly_orders)

        if n_segments < seg_min:
            seg_min = n_segments
        if n_segments > seg_max:
            seg_max = n_segments

        self.poly_order_min = poly_order_min
        self.poly_order_max = poly_order_max
        self._MAX_GRID_POINTS = n_grid_points
        self._TOL_RESIDUAL = max_residual
        self._GRID_TYPE = grid_type
        self.max_residual = max_residual
        self.n_grid_points = n_grid_points
        self.max_segments = seg_max
        self.min_segments = seg_min
        self.non_smooth_threshold = non_smooth_threshold

        # Segment width bounds : default values
        self.lbh = [self._SEG_WIDTH_MIN for _ in range(self._ocp.n_phases)]
        self.ubh = [self._SEG_WIDTH_MAX for _ in range(self._ocp.n_phases)]
        self.tol_residual = [self._TOL_RESIDUAL for _ in range(self._ocp.n_phases)]
        self.fig, self.axs = None, None
        self.plot_residual_evolution = False
        self.reset_mpopt()

    @staticmethod
    def get_abs_max_residual(residual):
        """
        Compute the segment wise maximum residual in each phase for each state

        args:
            :residual: Values of residuals in a list of lists (Phase -> segments)

        returns:
            max_phases:
                Values of indices of maximum residuals in each segment of the phase for all phases
        """
        max_r_phases = [None] * len(residual)
        for i_phase, r_phase in enumerate(residual):
            max_r_phase = [None] * len(r_phase)
            for i_seg, r_seg in enumerate(r_phase):
                r_max_seg = abs(np.array(r_seg)).max(axis=0)
                i_max_seg = abs(np.array(r_seg)).argmax(axis=0)
                max_r_phase[i_seg] = [i_max_seg, r_max_seg]
            max_r_phases[i_phase] = max_r_phase

        return max_r_phases

    def solve_ph(self, max_iter=1, solve_dict: Dict = {}, grid_type=None):
        """Solve OCP using adaptive ph method

        args: Options for the mpopt solver

        returns:
            :Solution: Solution for the ocp

        """
        # Custom definitions
        def limit_poly_orders(poly_orders):
            # Saturate poly orders between user specified limits
            return [
                min(max(self.poly_order_min, p), self.poly_order_max)
                for p in poly_orders
            ]

        if grid_type is None:
            grid_type = self.grid_type[0]
        # ph-Adaptive algorithm (http://dx.doi.org/10.1016/j.jfranklin.2015.05.028)
        nlp_sw_params = []  # Initialize
        poly_orders = []
        for iter_no in range(max_iter):
            if iter_no > 0:
                # Update Mesh for the next iteration
                self._nlp_sw_params = np.hstack(nlp_sw_params)
                solve_dict["nlp_sw_params"] = self._nlp_sw_params
                self.poly_orders = np.hstack(poly_orders)
                self.n_segments = self._nlp_sw_params.shape[0]

            # ************************** Sparse solution ********************
            # Step 1 : Solve the problem on initial mesh (Sparse for reference)

            self.reset_mpopt()
            solution = self.solve(reinitialize_nlp=True, **solve_dict)
            # Mesh points in time in each segment of a given phase
            # taus = np.array([self.collocation._taus_fn(deg) for deg in self.poly_orders])
            # taus = [taus for phase in range(self._ocp.n_phases)]

            # Step 2 : Compute the maximum residual in the states using gaussian quadrature of dynamics (Refer Anil V. Rao http://dx.doi.org/10.1016/j.jfranklin.2015.05.028)
            _, _, time_r, residual_r = self.get_states_residuals(
                solution, residual_type="relative", plot=False
            )
            max_residual = self.get_abs_max_residual(residual_r)

            # Step-3: Check if the solution is acceptable, find segment with max residual
            refine_phase = [False] * self._ocp.n_phases
            abs_max_residual = 0

            poly_orders = copy.deepcopy(self.poly_orders)
            for i_phase, max_r_phase in enumerate(max_residual):
                # Segment wise maximum value across all states
                max_seg_residual = np.array(max_r_phase)[:, 1].max(axis=1)
                if abs_max_residual < max_seg_residual.max():
                    abs_max_residual = max_seg_residual.max()
                status = max_seg_residual > self.max_residual
                if status.any():
                    # This phase needs refinement, increase the order of the polynomial by 3 for next iteration
                    poly_orders = [
                        self.poly_orders[i] + 3 * int(st) for i, st in enumerate(status)
                    ]
                    refine_phase[i_phase] = True

            if not np.array(refine_phase).all():
                return solution

            # Compute reduction in mesh size
            # Skipped for now ---- Yet to be implemented

            # Compute nodes in next fine mesh (poly_orders + 3)
            taus_new = np.array([self.collocation._taus_fn(deg) for deg in poly_orders])
            taus_new = [taus_new for phase in range(self._ocp.n_phases)]

            # Compute second derivative at nodes of next iteration
            time_d, ddx, ddu = self.get_state_second_derivative(
                solution, nodes=taus_new
            )

            # ************************** Refined solution *******************
            print(
                f"Iteration {iter_no} : max_residual, n_segments, poly_orders, seg_widths:- ({abs_max_residual}, {self.n_segments}, {self.poly_orders}, {self._nlp_sw_params})"
            )

            # Step-1: Solve the problem by increasing the order of polynomial by 3 in segments with high error
            poly_orders_orig = copy.deepcopy(self.poly_orders)
            self.poly_orders = limit_poly_orders(poly_orders)

            # Grid M (Solution with fine mesh, same number segments as solution but with increased polynomial degree)
            self.reset_mpopt()
            solution_new = self.solve(reinitialize_nlp=True, **solve_dict)

            # Step-2  : Compute the maximum residual in the states using gaussian quadrature of dynamics (Refer Anil V. Rao http://dx.doi.org/10.1016/j.jfranklin.2015.05.028)
            _, _, time_r_new, residual_r_new = self.get_states_residuals(
                solution_new,
                residual_type="relative",
                plot=False,
            )
            max_residual_new = self.get_abs_max_residual(residual_r_new)

            # Compute second derivate (state)
            # taus_new = np.array(
            #     [self.collocation._taus_fn(deg) for deg in self.poly_orders]
            # )
            # taus_new = [taus_new for phase in range(self._ocp.n_phases)]
            time_d_new, ddx_new, ddu_new = self.get_state_second_derivative(
                solution_new, nodes=taus_new
            )

            # Step-3: Check if the solution is acceptable, find segment with max residual
            refine_phase = [False] * self._ocp.n_phases
            dd_max_phase = [None] * (self._ocp.n_phases)
            for i_phase, max_r_phase in enumerate(max_residual_new):
                max_seg_residual = np.array(max_r_phase)[:, 1].max(axis=1)
                status = max_seg_residual > self.max_residual
                if status.any():
                    # This phase needs refinement
                    dd_max_phase[i_phase] = [
                        np.array(
                            [
                                abs(ddx_new[i_phase][i_seg][:, i, 0])
                                / abs(ddx[i_phase][i_seg][:, i, 0]).max()
                                for i in range(self._ocp.nx)
                            ]
                        )
                        for i_seg, st in enumerate(status)
                    ]
                    refine_phase[i_phase] = True

            if not np.array(refine_phase).all():
                return solution_new

            abs_max_residual = 0
            # Find non-smooth segments where residual is more than the required tolerence
            for i_phase, max_r_phase in enumerate(max_residual_new):
                poly_orders = []
                nlp_sw_params = []
                max_seg_residual = np.array(max_r_phase)[:, 1].max(axis=1)
                if max_seg_residual.max() > abs_max_residual:
                    abs_max_residual = max_seg_residual.max()
                status = max_seg_residual > self.max_residual
                for i_seg, st in enumerate(status):
                    if (
                        st
                        and (
                            dd_max_phase[i_phase][i_seg] > self.non_smooth_threshold
                        ).any()
                    ):
                        # Split i_seg in additional segments based on desired residual
                        new_segments = 2
                        nlp_sw_params.append(
                            [self._nlp_sw_params[i_seg] / new_segments] * new_segments
                        )
                        poly_orders.append([self.poly_orders[i_seg]] * new_segments)

                        # max_residual_seg = residual_r[phase][seg].max()
                        # max_residual_seg_sparse = residual_rs[phase][seg].max()
                        # ratio_residual = max_residual_seg / max_residual_seg_sparse
                        # ratio_nodes = poly_orders[seg] / self.poly_orders[seg]
                        # q = np.ceil(
                        #     5 / 2 + np.log(ratio_residual) / np.log(ratio_nodes)
                        # )
                        # ratio_residual_req = max_residual_seg / max_err
                        # seg_width_new = 1 * pow(ratio_residual_req, 1.0 / q)
                        # n_splits_max = np.ceil(
                        #     np.log(ratio_residual_req, poly_orders[seg])
                        # )
                        # n_splits = min(np.ceil(1 / seg_width_new), n_splits_max)
                        # seg_refine_phase[phase][seg] = n_splits
                    else:
                        # The solution is smooth and the residual is higher than the required tolerance.
                        # Increase the order of polynomial
                        poly_orders.append(self.poly_orders[i_seg])
                        nlp_sw_params.append(self._nlp_sw_params[i_seg])

        return solution_new
