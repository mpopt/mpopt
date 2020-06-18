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

import pytest

from mpopt import mp


@pytest.mark.parametrize(
    "ocp",
    [
        mp.OCP(),
        mp.OCP(n_states=1, n_controls=0),
        mp.OCP(0, 1),
        mp.OCP(1, 1, 2),
        mp.OCP(2, 1, 2),
        mp.OCP(1, 2, 2),
        mp.OCP(5, 4, n_phases=3),
    ],
)
def test_ocp_defaults(ocp):
    assert ocp.n_phases > 0

    assert len(ocp.dynamics) == ocp.n_phases
    assert len(ocp.running_costs) == ocp.n_phases
    assert len(ocp.terminal_costs) == ocp.n_phases
    assert len(ocp.path_constraints) == ocp.n_phases
    assert len(ocp.terminal_constraints) == ocp.n_phases

    x, u, t = [0] * ocp.nx, [0] * ocp.nu, 0
    for phase in range(ocp.n_phases):
        assert len(ocp.dynamics[phase](x, u, t)) == ocp.nx
        assert ocp.terminal_costs[phase](x, t, x, t) is not None
        assert ocp.running_costs[phase](x, u, t) is not None

        pc = ocp.path_constraints[phase](x, u, t)
        tc = ocp.terminal_constraints[phase](x, t, x, t)
        if pc is not None:
            assert len(pc) > 0
        if tc is not None:
            assert len(tc) > 0

    assert len(ocp.scale_x) == ocp.nx
    assert len(ocp.scale_u) == ocp.nu

    assert ocp.x00.shape == (ocp.n_phases, ocp.nx)
    assert ocp.xf0.shape == (ocp.n_phases, ocp.nx)
    assert ocp.u00.shape == (ocp.n_phases, ocp.nu)
    assert ocp.uf0.shape == (ocp.n_phases, ocp.nu)
    assert ocp.t00.shape == (ocp.n_phases, 1)
    assert ocp.tf0.shape == (ocp.n_phases, 1)

    assert ocp.lbx.shape == (ocp.n_phases, ocp.nx)
    assert ocp.ubx.shape == (ocp.n_phases, ocp.nx)
    assert ocp.lbu.shape == (ocp.n_phases, ocp.nu)
    assert ocp.ubu.shape == (ocp.n_phases, ocp.nu)
    assert ocp.lbt0.shape == (ocp.n_phases, 1)
    assert ocp.ubt0.shape == (ocp.n_phases, 1)
    assert ocp.lbtf.shape == (ocp.n_phases, 1)
    assert ocp.ubtf.shape == (ocp.n_phases, 1)

    assert ocp.lbe.shape[0] == ocp.n_phases - 1
    assert ocp.ube.shape[0] == ocp.n_phases - 1
    if ocp.n_phases > 1:
        assert ocp.lbe.shape[1] == ocp.nx
        assert ocp.ube.shape[1] == ocp.nx


@pytest.fixture
def test_ocp():
    ocp = mp.OCP(2, 2, 2)

    dynamics = lambda x, u, t: [u[0], u[0]]
    path_constraints = lambda x, u, t: [x[0] + 1, u[0]]
    running_costs = lambda x, u, t: u[0]
    terminal_constraints = lambda xf, tf, x0, t0: [-xf[0]]
    terminal_costs = lambda xf, tf, x0, t0: tf

    ocp.dynamics = [dynamics] * ocp.n_phases
    ocp.path_constraints = [path_constraints] * ocp.n_phases
    ocp.running_costs = [running_costs] * ocp.n_phases
    ocp.terminal_constraints = [terminal_constraints] * ocp.n_phases
    ocp.terminal_costs = [terminal_costs] * ocp.n_phases

    for phase in range(ocp.n_phases):
        ocp.lbu[phase], ocp.ubu[phase] = -1.0, 1.0
        ocp.lbtf[phase], ocp.ubtf[phase] = 1.0, 1.0

    ocp.validate()

    return ocp


@pytest.fixture
def test_mpo(test_ocp):
    mpo = mp.mpopt(test_ocp)
    mpo.validate()

    return mpo


def test_mpopt_collocation_basis(test_mpo):
    test_mpo.compute_numerical_approximation()
    ocp = test_mpo._ocp
    poly_orders = test_mpo.poly_orders
    Npoints = test_mpo._Npoints
    assert len(test_mpo._taus) == len(set(poly_orders))
    assert test_mpo._compW.shape == (1, Npoints)
    for p in poly_orders:
        assert len(test_mpo._taus[p]) == p + 1
    assert test_mpo._compD.shape == (Npoints, Npoints,)
    assert test_mpo._collocation_approximation_computed


def test_mpopt_casadi_variables(test_mpo):
    test_mpo.create_variables()
    ocp = test_mpo._ocp
    assert test_mpo.X is not None
    assert test_mpo.U is not None
    assert test_mpo.t0 is not None
    assert test_mpo.tf is not None
    assert test_mpo.seg_widths is not None


def test_mpopt_ocp_discretization(test_mpo):
    ocp = test_mpo._ocp
    for phase in range(ocp.n_phases):
        G, Gmin, Gmax, J = test_mpo.discretize_phase(phase)
        assert G.shape[0] == Gmin.shape[0] == Gmax.shape[0]
        assert J.shape == (1, 1)


def test_mpopt_event_constraints(test_mpo):
    ocp = test_mpo._ocp
    E, Emin, Emax = test_mpo.get_event_constraints()

    if ocp.n_phases == 1:
        assert E == Emin == Emax == []
    assert len(E) == len(Emin) == len(Emax)


def test_mpopt_nlp_vars_init(test_mpo):
    ocp = test_mpo._ocp
    for phase in range(ocp.n_phases):
        Z, Zmin, Zmax = test_mpo.get_nlp_variables(phase)
        assert (
            Z.shape[0]
            == Zmin.shape[0]
            == Zmax.shape[0]
            == test_mpo._optimization_vars_per_phase
        )


def test_mpopt_nlp_init(test_mpo):
    nlp_prob, nlp_bounds = test_mpo.create_nlp()
    assert (
        nlp_prob["x"].shape[0]
        == nlp_bounds["lbx"].shape[0]
        == nlp_bounds["ubx"].shape[0]
    )
    assert (
        nlp_prob["g"].shape[0]
        == nlp_bounds["lbg"].shape[0]
        == nlp_bounds["ubg"].shape[0]
    )
    assert nlp_prob["f"].shape[0] == 1


def test_mpopt_init_solution(test_mpo):
    ocp = test_mpo._ocp
    test_mpo.create_variables()
    Z0 = test_mpo.initialize_solution()
    assert Z0.shape[0] == test_mpo._optimization_vars_per_phase * ocp.n_phases


def test_mpopt_solve(test_mpo):
    test_mpo.solve()
    for key in ["lbx", "lbg", "ubx", "ubg"]:
        assert key in test_mpo.nlp_bounds
