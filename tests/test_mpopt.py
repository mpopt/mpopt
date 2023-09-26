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
import numpy as np
import casadi as ca


@pytest.mark.parametrize(
    "ocp",
    [
        mp.OCP(),
        mp.OCP(n_states=1, n_controls=0),
        mp.OCP(n_states=0, n_controls=1),
        mp.OCP(n_states=1, n_controls=1, n_phases=2),
        mp.OCP(n_states=2, n_controls=1, n_phases=2),
        mp.OCP(n_states=1, n_controls=2, n_phases=2),
        mp.OCP(n_states=5, n_controls=4, n_phases=3),
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
    ocp = mp.OCP(n_states=2, n_controls=2, n_phases=2)

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
def moon_lander_ocp():
    ocp = mp.OCP(n_states=2, n_controls=1)

    def dynamics0(x, u, t):
        return [x[1], u[0] - 1.5]

    ocp.dynamics[0] = dynamics0

    def running_cost0(x, u, t):

        return u[0]

    ocp.running_costs[0] = running_cost0

    def terminal_constraints0(xf, tf, x0, t0):

        return [xf[0], xf[1]]

    ocp.terminal_constraints[0] = terminal_constraints0

    ocp.tf0[0] = 4.0
    ocp.x00[0] = [10.0, -2.0]
    ocp.lbx[0] = [-20.0, -20.0]
    ocp.ubx[0] = [20.0, 20.0]
    ocp.lbu[0] = 0
    ocp.ubu[0] = 3
    ocp.lbtf[0], ocp.ubtf[0] = 3, 5

    ocp.validate()

    return ocp


@pytest.fixture
def hyper_sensitive_ocp():
    ocp = mp.OCP(n_states=1, n_controls=1, n_phases=1)

    ocp.dynamics[0] = lambda x, u, t: [-x[0] * x[0] * x[0] + u[0]]
    ocp.running_costs[0] = lambda x, u, t: 0.5 * (x[0] * x[0] + u[0] * u[0])
    ocp.terminal_constraints[0] = lambda xf, tf, x0, t0: [xf[0] - 1.0]

    ocp.x00[0] = 1
    ocp.lbtf[0] = ocp.ubtf[0] = 1000.0
    ocp.scale_t = 1 / 1000.0

    ocp.validate()

    return ocp


@pytest.fixture
def two_phase_schwartz_ocp():
    ocp = mp.OCP(n_states=2, n_controls=1, n_phases=2)

    # Step-1 : Define dynamics
    def dynamics0(x, u, t):
        return [x[1], u[0] - 0.1 * (1.0 + 2.0 * x[0] * x[0]) * x[1]]

    ocp.dynamics = [dynamics0, dynamics0]

    def path_constraints0(x, u, t):
        return [
            1.0
            - 9.0 * (x[0] - 1) * (x[0] - 1)
            - (x[1] - 0.4) * (x[1] - 0.4) / (0.3 * 0.3)
        ]

    ocp.path_constraints[0] = path_constraints0

    def terminal_cost1(xf, tf, x0, t0):
        return 5 * (xf[0] * xf[0] + xf[1] * xf[1])

    ocp.terminal_costs[1] = terminal_cost1

    ocp.x00[0] = [1, 1]
    ocp.x00[1] = [1, 1]
    ocp.xf0[0] = [1, 1]
    ocp.xf0[1] = [0, 0]
    ocp.lbx[0][1] = -0.8

    ocp.lbu[0], ocp.ubu[0] = -1, 1

    ocp.lbt0[0], ocp.ubt0[0] = 0, 0
    ocp.lbtf[0], ocp.ubtf[0] = 1, 1
    ocp.lbtf[1], ocp.ubtf[1] = 2.9, 2.9

    ocp.validate()

    return ocp


@pytest.fixture
def van_der_pol_ocp():
    ocp = mp.OCP(n_states=2, n_controls=1)

    def dynamics(x, u, t):
        return [(1 - x[1] * x[1]) * x[0] - x[1] + u[0], x[0]]

    def running_cost(x, u, t):
        return x[0] * x[0] + x[1] * x[1] + u[0] * u[0]

    ocp.dynamics[0] = dynamics
    ocp.running_costs[0] = running_cost

    ocp.x00[0] = [0, 1]
    ocp.lbu[0] = -1.0
    ocp.ubu[0] = 1.0
    ocp.lbx[0][1] = -0.25
    ocp.lbtf[0] = 10.0
    ocp.ubtf[0] = 10.0

    ocp.validate()

    return ocp


@pytest.fixture
def test_mpo(test_ocp):
    mpo = mp.mpopt(test_ocp)
    mpo.validate()

    return mpo


@pytest.fixture
def moon_lander_mpo(moon_lander_ocp):
    mpo, post = mp.solve(
        moon_lander_ocp, n_segments=20, poly_orders=3, scheme="LGR", plot=False
    )
    mpo = mp.mpopt(moon_lander_ocp, 20, 3)
    mpo.validate()

    return mpo


@pytest.fixture
def moon_lander_mpo_h_adaptive(moon_lander_ocp):
    mpo = mp.mpopt_h_adaptive(moon_lander_ocp, 10, 4)
    mpo.validate()

    return mpo


@pytest.fixture
def moon_lander_mpo_adaptive(moon_lander_ocp):
    mpo = mp.mpopt_adaptive(moon_lander_ocp, 3, 3)
    mpo.lbh[0] = 1e-6
    mpo.mid_residuals = True
    mpo.validate()

    return mpo


@pytest.fixture
def hyper_sensitive_mpo_h_adaptive(hyper_sensitive_ocp):
    mpo = mp.mpopt_h_adaptive(hyper_sensitive_ocp, 15, 15)
    mpo.validate()

    return mpo


@pytest.fixture
def hyper_sensitive_mpo_adaptive(hyper_sensitive_ocp):
    mpo = mp.mpopt_adaptive(hyper_sensitive_ocp, 5, 15)
    mpo.lbh[0] = 1e-6
    mpo.mid_residuals = True
    mpo.validate()

    return mpo


@pytest.fixture
def hyper_sensitive_mpo(hyper_sensitive_ocp):
    mpo = mp.mpopt(hyper_sensitive_ocp, 15, 15)
    mpo.validate()

    return mpo


@pytest.fixture
def two_phase_schwartz_mpo(two_phase_schwartz_ocp):
    mpo = mp.mpopt(two_phase_schwartz_ocp, 1, 15, "LGL")
    mpo.validate()

    return mpo


@pytest.fixture
def van_der_pol_mpo(van_der_pol_ocp):
    mpo = mp.mpopt(van_der_pol_ocp, 1, 15, "LGR")
    mpo.validate()

    return mpo


@pytest.fixture
def van_der_pol_mpo_lgl(van_der_pol_ocp):
    mpo = mp.mpopt(van_der_pol_ocp, 1, 15, "LGL")
    mpo.validate()

    return mpo


@pytest.fixture
def van_der_pol_mpo_cgl(van_der_pol_ocp):
    mpo = mp.mpopt(van_der_pol_ocp, 1, 15, "CGL")
    mpo.validate()

    return mpo


@pytest.fixture
def van_der_pol_mpo_h_adaptive(van_der_pol_ocp):
    mpo = mp.mpopt_h_adaptive(van_der_pol_ocp, 5, 5, "LGR")
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
    assert test_mpo._compD.shape == (
        Npoints,
        Npoints,
    )
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


def test_moon_lander_mpopt_solve(moon_lander_mpo):
    moon_lander_mpo._ocp.diff_u[0] = 1
    moon_lander_mpo._ocp.midu[0] = 0
    moon_lander_mpo._ocp.du_continuity[0] = 1
    sol = moon_lander_mpo.solve()
    for key in ["x", "f"]:
        assert key in sol
    post = moon_lander_mpo.process_results(sol, plot=False)
    x, u, t, _ = post.get_data()
    xi, ui, ti, _ = post.get_data(interpolate=True)

    assert x.shape[0] == u.shape[0] == t.shape[0]
    assert xi.shape[0] == ui.shape[0] == ti.shape[0]


def test_moon_lander_h_adaptive_solve(moon_lander_mpo_h_adaptive):
    sol = moon_lander_mpo_h_adaptive.solve(max_iter=3)
    for key in ["x", "f"]:
        assert key in sol
    post = moon_lander_mpo_h_adaptive.process_results(sol, plot=False)
    x, u, t, _ = post.get_data()
    xi, ui, ti, _ = post.get_data(interpolate=True)

    assert x.shape[0] == u.shape[0] == t.shape[0]
    assert xi.shape[0] == ui.shape[0] == ti.shape[0]


def test_moon_lander_h_adaptive2_solve(moon_lander_mpo_h_adaptive):
    moon_lander_mpo_h_adaptive.grid_type[0] = "mid-points"
    sol = moon_lander_mpo_h_adaptive.solve(
        max_iter=2, mpopt_options={"method": "residual", "sub_method": "equal_area"}
    )
    for key in ["x", "f"]:
        assert key in sol
    post = moon_lander_mpo_h_adaptive.process_results(sol, plot=False)
    x, u, t, _ = post.get_data()
    xi, ui, ti, _ = post.get_data(interpolate=True)

    assert x.shape[0] == u.shape[0] == t.shape[0]
    assert xi.shape[0] == ui.shape[0] == ti.shape[0]


def test_moon_lander_h_adaptive3_solve(moon_lander_mpo_h_adaptive):
    moon_lander_mpo_h_adaptive.grid_type[0] = "spectral"
    sol = moon_lander_mpo_h_adaptive.solve(
        max_iter=10, mpopt_options={"method": "control_slope", "sub_method": ""}
    )
    for key in ["x", "f"]:
        assert key in sol
    post = moon_lander_mpo_h_adaptive.process_results(sol, plot=False)
    x, u, t, _ = post.get_data()
    xi, ui, ti, _ = post.get_data(interpolate=True)

    assert x.shape[0] == u.shape[0] == t.shape[0]
    assert xi.shape[0] == ui.shape[0] == ti.shape[0]


def test_moon_lander_mpopt_adaptive_solve(moon_lander_mpo_adaptive):
    moon_lander_mpo_adaptive.mid_residuals = False
    sol = moon_lander_mpo_adaptive.solve()
    for key in ["x", "f"]:
        assert key in sol
    post = moon_lander_mpo_adaptive.process_results(sol, plot=False)
    x, u, t, _ = post.get_data()
    xi, ui, ti, _ = post.get_data(interpolate=True)

    assert x.shape[0] == u.shape[0] == t.shape[0]
    assert xi.shape[0] == ui.shape[0] == ti.shape[0]


def test_hyper_sensitive_mpopt_solve(hyper_sensitive_mpo):
    sol = hyper_sensitive_mpo.solve()
    for key in ["x", "f"]:
        assert key in sol
    post = hyper_sensitive_mpo.process_results(sol, plot=False)
    x, u, t, _ = post.get_data()
    xi, ui, ti, _ = post.get_data(interpolate=True)

    assert x.shape[0] == u.shape[0] == t.shape[0]
    assert xi.shape[0] == ui.shape[0] == ti.shape[0]


def test_hyper_sensitive_h_adaptive_solve(hyper_sensitive_mpo_h_adaptive):
    sol = hyper_sensitive_mpo_h_adaptive.solve(
        max_iter=3, mpopt_options={"method": "residual", "sub_method": "merge_split"}
    )
    for key in ["x", "f"]:
        assert key in sol
    post = hyper_sensitive_mpo_h_adaptive.process_results(sol, plot=False)
    x, u, t, _ = post.get_data()
    xi, ui, ti, _ = post.get_data(interpolate=True)

    assert x.shape[0] == u.shape[0] == t.shape[0]
    assert xi.shape[0] == ui.shape[0] == ti.shape[0]


def test_hyper_sensitive_h_adaptive2_solve(hyper_sensitive_mpo_h_adaptive):
    hyper_sensitive_mpo_h_adaptive.plot_residual_evolution = True
    sol = hyper_sensitive_mpo_h_adaptive.solve(
        max_iter=2, mpopt_options={"method": "residual", "sub_method": "equal_area"}
    )
    for key in ["x", "f"]:
        assert key in sol
    post = hyper_sensitive_mpo_h_adaptive.process_results(sol, plot=False)
    x, u, t, _ = post.get_data()
    xi, ui, ti, _ = post.get_data(interpolate=True)

    assert x.shape[0] == u.shape[0] == t.shape[0]
    assert xi.shape[0] == ui.shape[0] == ti.shape[0]


def test_hyper_sensitive_h_adaptive3_solve(hyper_sensitive_mpo_h_adaptive):
    hyper_sensitive_mpo_h_adaptive.plot_residual_evolution = True
    sol = hyper_sensitive_mpo_h_adaptive.solve(
        max_iter=10, mpopt_options={"method": "control_slope", "sub_method": ""}
    )
    for key in ["x", "f"]:
        assert key in sol
    post = hyper_sensitive_mpo_h_adaptive.process_results(sol, plot=False)
    x, u, t, _ = post.get_data()
    xi, ui, ti, _ = post.get_data(interpolate=True)

    assert x.shape[0] == u.shape[0] == t.shape[0]
    assert xi.shape[0] == ui.shape[0] == ti.shape[0]


def test_hyper_sensitive_mpopt_adaptive_solve(hyper_sensitive_mpo_adaptive):
    sol = hyper_sensitive_mpo_adaptive.solve()
    for key in ["x", "f"]:
        assert key in sol
    post = hyper_sensitive_mpo_adaptive.process_results(sol, plot=False)
    x, u, t, _ = post.get_data()
    xi, ui, ti, _ = post.get_data(interpolate=True)

    assert x.shape[0] == u.shape[0] == t.shape[0]
    assert xi.shape[0] == ui.shape[0] == ti.shape[0]


def test_two_phase_schwartz_mpopt_solve(two_phase_schwartz_mpo):
    sol = two_phase_schwartz_mpo.solve()
    for key in ["x", "f"]:
        assert key in sol
    post = two_phase_schwartz_mpo.process_results(sol, plot=False)
    x, u, t, _ = post.get_data()
    xi, ui, ti, _ = post.get_data(interpolate=True)

    assert x.shape[0] == u.shape[0] == t.shape[0]
    assert xi.shape[0] == ui.shape[0] == ti.shape[0]


def test_van_der_pol_mpopt_solve(van_der_pol_mpo):
    sol = van_der_pol_mpo.solve()
    for key in ["x", "f"]:
        assert key in sol
    post = van_der_pol_mpo.process_results(sol, plot=False)
    x, u, t = post.get_data()
    xi, ui, ti = post.get_data(interpolate=True)

    assert x.shape[0] == u.shape[0] == t.shape[0]
    assert xi.shape[0] == ui.shape[0] == ti.shape[0]


def test_van_der_pol_mpopt_solve(van_der_pol_mpo_lgl):
    sol = van_der_pol_mpo_lgl.solve()
    for key in ["x", "f"]:
        assert key in sol
    post = van_der_pol_mpo_lgl.process_results(sol, plot=False)
    x, u, t, _ = post.get_data()
    xi, ui, ti, _ = post.get_data(interpolate=True)

    assert x.shape[0] == u.shape[0] == t.shape[0]
    assert xi.shape[0] == ui.shape[0] == ti.shape[0]


def test_van_der_pol_mpopt_solve(van_der_pol_mpo_cgl):
    sol = van_der_pol_mpo_cgl.solve()
    for key in ["x", "f"]:
        assert key in sol
    post = van_der_pol_mpo_cgl.process_results(sol, plot=False)
    fig, axs = post.plot_phases()
    fig, axs = post.plot_x()
    fig, axs = post.plot_u()
    x, u, t, _ = post.get_data()
    xi, ui, ti, _ = post.get_data(interpolate=True)

    assert x.shape[0] == u.shape[0] == t.shape[0]
    assert xi.shape[0] == ui.shape[0] == ti.shape[0]


@pytest.fixture
def test_collocation():
    collocation = mp.Collocation([3], "LGR")

    return collocation


def test_collocation_matrices_numerical_and_symbolic():
    mp.Collocation.D_MATRIX_METHOD = "numerical"
    collocation = mp.Collocation([3], "LGR")
    compDn = collocation.get_composite_differentiation_matrix()
    compWn = collocation.get_composite_quadrature_weights()

    mp.Collocation.D_MATRIX_METHOD = "symbolic"
    collocation_sym = mp.Collocation([3], "LGR")
    compD = collocation_sym.get_composite_differentiation_matrix()
    compW = collocation_sym.get_composite_quadrature_weights()

    assert abs(compD.full() - compDn.full()).max() < 1e-5
    assert abs(compW.full() - compWn.full()).max() < 1e-5


def test_mpopt_collocation_basis(test_collocation):
    compD = test_collocation.get_composite_differentiation_matrix()
    compW = test_collocation.get_composite_quadrature_weights()
    taus = test_collocation.roots[test_collocation.poly_orders[0]]
    tau0, tau1 = test_collocation.tau0, test_collocation.tau1

    assert tau0 == taus[0]
    assert tau1 == taus[-1]


def test_mpopt_get_residual_grid_taus(test_mpo):
    sol = test_mpo.solve()

    taus = test_mpo.get_residual_grid_taus()

    taus_1D = np.concatenate(taus)
    assert taus_1D.min() >= test_mpo.tau0
    assert taus_1D.max() <= test_mpo.tau1
    assert len(taus) == test_mpo.n_segments

    taus = test_mpo.get_residual_grid_taus(grid_type="fixed")
    taus_1D = np.concatenate(taus)
    assert taus_1D.min() >= test_mpo.tau0
    assert taus_1D.max() <= test_mpo.tau1
    assert len(taus) == test_mpo.n_segments

    taus = test_mpo.get_residual_grid_taus(grid_type="mid-points")
    taus_1D = np.concatenate(taus)
    assert taus_1D.min() >= test_mpo.tau0
    assert taus_1D.max() <= test_mpo.tau1
    assert len(taus) == test_mpo.n_segments

    taus = test_mpo.get_residual_grid_taus(grid_type="do-not-know-any")
    assert taus == None


def test_mpopt_compute_interpolation_taus_corresponding_to_original_grid():
    nodes_req, seg_widths = np.array([0, 0.5, 1]), [1]
    taus = mp.mpopt.compute_interpolation_taus_corresponding_to_original_grid(
        nodes_req, seg_widths
    )
    assert (abs(taus - nodes_req[1:]) < 1e-6).all()

    nodes_req, seg_widths = np.array([0, 0.5, 1]), [0.5, 0.5]
    taus = mp.mpopt.compute_interpolation_taus_corresponding_to_original_grid(
        nodes_req, seg_widths
    )
    assert (abs(taus[0][-1] - 1) < 1e-6).all()
    assert (abs(taus[1][-1] - 1) < 1e-6).all()


def test_mpopt_interpolate_single_phase(van_der_pol_mpo):
    sol = van_der_pol_mpo.solve()

    (
        Xi,
        Ui,
        ti,
        a,
        DXi,
        DUi,
        target_nodes,
        t0,
        tf,
    ) = van_der_pol_mpo.interpolate_single_phase(
        sol,
        phase=0,
    )

    assert Xi.size() == DXi.size()
    assert Ui.size() == DUi.size()
    assert a.size() == (van_der_pol_mpo._ocp.na, 1)
    total_nodes = sum([len(node) for node in target_nodes])
    assert ti.size() == (total_nodes, 1)

    (
        Xi,
        Ui,
        ti,
        a,
        DXi,
        DUi,
        target_nodes,
        t0,
        tf,
    ) = van_der_pol_mpo.interpolate_single_phase(
        sol,
        phase=0,
        target_nodes=np.array(
            [
                [van_der_pol_mpo.tau0, van_der_pol_mpo.tau1]
                for i in range(van_der_pol_mpo.n_segments)
            ]
        ),
    )

    assert Xi.size() == DXi.size()
    assert Ui.size() == DUi.size()
    assert a.size() == (van_der_pol_mpo._ocp.na, 1)
    total_nodes = sum([len(node) for node in target_nodes])
    assert ti.size() == (total_nodes, 1)


def test_mpopt_get_dynamics_residuals_single_phase(two_phase_schwartz_mpo):
    mpo = two_phase_schwartz_mpo
    sol = mpo.solve()
    # Get the collocation points
    taus = [mpo.collocation._taus_fn(deg)[1:-1] for deg in mpo.poly_orders]

    # Estimate residual at collocation points in first phase
    time, residual, _ = mpo.get_dynamics_residuals_single_phase(sol, 0, taus)
    max_residual = max([abs(np.array(err)).max() for err in residual])
    assert max_residual < 1e-1

    # Estimate residual at collocation points in second phase
    time, residual, _ = mpo.get_dynamics_residuals_single_phase(sol, 1, taus)
    max_residual = max([abs(np.array(err)).max() for err in residual])
    assert max_residual < 1


def test_mpopt_get_dynamics_residuals(hyper_sensitive_mpo_h_adaptive):
    mpo = hyper_sensitive_mpo_h_adaptive

    sol = mpo.solve()
    # Get the collocation points
    taus = [mpo.collocation._taus_fn(deg)[1:-1] for deg in mpo.poly_orders]

    # Estimate residual at collocation points in all phases
    nodes = [taus for phase in range(mpo._ocp.n_phases)]
    time0, residuals = mpo.get_dynamics_residuals(sol, nodes=nodes)
    max_residual = max(
        [
            abs(np.array(err)).max()
            for residual in residuals
            for err in residual
            if err is not None
        ]
    )
    assert max_residual < 1e-1

    time1, residuals = mpo.get_dynamics_residuals(sol, grid_type="fixed")
    max_residual = max(
        [
            abs(np.array(err)).max()
            for residual in residuals
            for err in residual
            if err is not None
        ]
    )
    assert max_residual < 4

    time2, residuals = mpo.get_dynamics_residuals(sol, grid_type="mid-points")
    max_residual = max(
        [
            abs(np.array(err)).max()
            for residual in residuals
            for err in residual
            if err is not None
        ]
    )
    assert max_residual < 2

    time3, residuals = mpo.get_dynamics_residuals(sol, grid_type="spectral")
    max_residual = max(
        [
            abs(np.array(err)).max()
            for residual in residuals
            for err in residual
            if err is not None
        ]
    )
    assert max_residual < 2


@pytest.fixture
def int_1D_ocp():
    ocp = mp.OCP(n_states=1, n_controls=1)

    def dynamics(x, u, t):
        return [u[0]]

    def running_cost(x, u, t):
        return u[0] * u[0]

    def terminal_cost(xf, tf, x0, t0):
        return 0.5 * xf[0] * xf[0]

    ocp.dynamics[0] = dynamics
    ocp.running_costs[0] = running_cost
    ocp.terminal_costs[0] = terminal_cost

    ocp.x00[0] = [0.2]
    ocp.lbu[0] = -1.0
    ocp.ubu[0] = 1.0
    ocp.lbx[0] = [-1]
    ocp.ubx[0] = [-1]
    ocp.lbtf[0] = 1.0
    ocp.ubtf[0] = 1.0

    ocp.validate()

    return ocp


@pytest.fixture
def int_1D_mpo(int_1D_ocp):
    mpo = mp.mpopt(int_1D_ocp)
    mpo.validate()

    return mpo


@pytest.fixture
def mine_opt_ocp():
    # https://en.wikipedia.org/wiki/Optimal_control
    ocp = mp.OCP(n_states=1, n_controls=1)

    p = 1  # Price

    def dynamics(x, u, t):
        return [-u[0]]

    def running_cost(x, u, t):
        return u[0] * u[0] / x[0] - p * u[0]

    ocp.dynamics[0] = dynamics
    ocp.running_costs[0] = running_cost

    ocp.x00[0] = [1.0]
    ocp.lbx[0] = 0
    ocp.ubx[0] = 1
    ocp.lbtf[0] = 1.0
    ocp.ubtf[0] = 1.0

    ocp.validate()

    return ocp


@pytest.fixture
def mine_opt_mpo(mine_opt_ocp):
    mpo = mp.mpopt(mine_opt_ocp, 2, 5)
    mpo.validate()

    return mpo


#
# def test_mpopt_get_state_second_derivative_single_phase(mine_opt_mpo):
#     mpo = mine_opt_mpo
#
#     sol = mpo.solve()
#     post = mpo.process_results(sol, plot=False)
#     x, u, t, _ = post.get_data()
#
#     assert (abs(x - (4 - 1 * t + 1 * 1) ** 2 / (4 + 1 * 1) ** 2 * 1) < 1e-3).all()
#
#     # Get the collocation points
#     taus = [mpo.collocation._taus_fn(deg)[1:-1] for deg in mpo.poly_orders]
#
#     # Estimate residual at collocation points in all phases
#     # nodes = [taus for phase in range(mpo._ocp.n_phases)]
#     time, ddx, ddu = mpo.get_state_second_derivative_single_phase(sol, nodes=taus)
#
#
#     print(ddx, -(4 - time[0] + 1) / 25)
#
#     assert np.array(
#         [
#             (abs(ddx_seg + 2.0 * (4 - time[i] + 1) / 25) < 1e-3).all()
#             for i, ddx_seg in enumerate(ddx)
#         ]
#     ).all()
#     # assert np.array([(abs(ddx_seg - 2.0 / 25) < 1e-3).all() for ddx_seg in ddx]).all()


@pytest.fixture
def collocation_LGR_O1():
    mp.CollocationRoots._TAU_MIN = -1
    collocation = mp.Collocation([1], "LGR")

    return collocation


@pytest.fixture
def collocation_LGL_O1():
    mp.CollocationRoots._TAU_MIN = -1
    collocation = mp.Collocation([1], "LGL")

    return collocation


@pytest.fixture
def collocation_CGL_O1():
    mp.CollocationRoots._TAU_MIN = -1
    collocation = mp.Collocation([1], "CGL")

    return collocation


def test_LGR_O1(collocation_LGR_O1):
    lgr = collocation_LGR_O1
    assert (
        np.abs(
            lgr.roots[1]
            - np.array([mp.CollocationRoots._TAU_MIN, mp.CollocationRoots._TAU_MAX])
        )
        < 1e-6
    ).all()
    h = lgr.roots[1][-1] - lgr.roots[1][0]

    assert ca.Function("a1", [lgr.TVAR], [lgr.polys[1][0]])(lgr.roots[1][0]) == 1
    assert ca.Function("a1", [lgr.TVAR], [lgr.polys[1][0]])(lgr.roots[1][-1]) == 0
    assert ca.Function("a1", [lgr.TVAR], [lgr.polys[1][1]])(lgr.roots[1][0]) == 0
    assert ca.Function("a1", [lgr.TVAR], [lgr.polys[1][1]])(lgr.roots[1][-1]) == 1

    D = lgr.get_diff_matrix(1, order=1)
    assert ((D - ca.DM([[-1 / h, 1 / h], [-1 / h, 1 / h]])).full() < 1e-6).all()

    D = lgr.get_diff_matrix(1, order=2)
    assert ((D - ca.DM.zeros(2, 2)).full() < 1e-6).all()


def test_CGL_O1(collocation_CGL_O1):
    lgr = collocation_CGL_O1
    assert (
        np.abs(
            lgr.roots[1]
            - np.array([mp.CollocationRoots._TAU_MIN, mp.CollocationRoots._TAU_MAX])
        )
        < 1e-6
    ).all()
    h = lgr.roots[1][-1] - lgr.roots[1][0]

    assert ca.Function("a1", [lgr.TVAR], [lgr.polys[1][0]])(lgr.roots[1][0]) == 1
    assert ca.Function("a1", [lgr.TVAR], [lgr.polys[1][0]])(lgr.roots[1][-1]) == 0
    assert ca.Function("a1", [lgr.TVAR], [lgr.polys[1][1]])(lgr.roots[1][0]) == 0
    assert ca.Function("a1", [lgr.TVAR], [lgr.polys[1][1]])(lgr.roots[1][-1]) == 1

    D = lgr.get_diff_matrix(1, order=1)
    assert ((D - ca.DM([[-1 / h, 1 / h], [-1 / h, 1 / h]])).full() < 1e-6).all()

    D = lgr.get_diff_matrix(1, order=2)
    assert ((D - ca.DM.zeros(2, 2)).full() < 1e-6).all()


def test_LGL_O1(collocation_LGL_O1):
    lgr = collocation_LGL_O1
    assert (
        np.abs(
            lgr.roots[1]
            - np.array([mp.CollocationRoots._TAU_MIN, mp.CollocationRoots._TAU_MAX])
        )
        < 1e-6
    ).all()
    h = lgr.roots[1][-1] - lgr.roots[1][0]

    assert ca.Function("a1", [lgr.TVAR], [lgr.polys[1][0]])(lgr.roots[1][0]) == 1
    assert ca.Function("a1", [lgr.TVAR], [lgr.polys[1][0]])(lgr.roots[1][-1]) == 0
    assert ca.Function("a1", [lgr.TVAR], [lgr.polys[1][1]])(lgr.roots[1][0]) == 0
    assert ca.Function("a1", [lgr.TVAR], [lgr.polys[1][1]])(lgr.roots[1][-1]) == 1

    D = lgr.get_diff_matrix(1, order=1)
    assert ((D - ca.DM([[-1 / h, 1 / h], [-1 / h, 1 / h]])).full() < 1e-6).all()

    D = lgr.get_diff_matrix(1, order=2)
    assert ((D - ca.DM.zeros(2, 2)).full() < 1e-6).all()


@pytest.fixture
def collocation_LGR():
    mp.CollocationRoots._TAU_MIN = 0
    collocation = mp.Collocation([1], "LGR")

    return collocation


@pytest.fixture
def collocation_LGL():
    mp.CollocationRoots._TAU_MIN = 0
    collocation = mp.Collocation([1], "LGL")

    return collocation


@pytest.fixture
def collocation_CGL():
    mp.CollocationRoots._TAU_MIN = 0
    collocation = mp.Collocation([1], "CGL")

    return collocation


def test_LGR(collocation_LGR):
    lgr = collocation_LGR
    assert (
        np.abs(
            lgr.roots[1]
            - np.array([mp.CollocationRoots._TAU_MIN, mp.CollocationRoots._TAU_MAX])
        )
        < 1e-6
    ).all()
    h = lgr.roots[1][-1] - lgr.roots[1][0]

    assert ca.Function("a1", [lgr.TVAR], [lgr.polys[1][0]])(lgr.roots[1][0]) == 1
    assert ca.Function("a1", [lgr.TVAR], [lgr.polys[1][0]])(lgr.roots[1][-1]) == 0
    assert ca.Function("a1", [lgr.TVAR], [lgr.polys[1][1]])(lgr.roots[1][0]) == 0
    assert ca.Function("a1", [lgr.TVAR], [lgr.polys[1][1]])(lgr.roots[1][-1]) == 1

    D = lgr.get_diff_matrix(1, order=1)
    assert ((D - ca.DM([[-1 / h, 1 / h], [-1 / h, 1 / h]])).full() < 1e-6).all()

    D = lgr.get_diff_matrix(1, order=2)
    assert ((D - ca.DM.zeros(2, 2)).full() < 1e-6).all()


def test_CGL(collocation_CGL):
    lgr = collocation_CGL
    assert (
        np.abs(
            lgr.roots[1]
            - np.array([mp.CollocationRoots._TAU_MIN, mp.CollocationRoots._TAU_MAX])
        )
        < 1e-6
    ).all()
    h = lgr.roots[1][-1] - lgr.roots[1][0]

    assert ca.Function("a1", [lgr.TVAR], [lgr.polys[1][0]])(lgr.roots[1][0]) == 1
    assert ca.Function("a1", [lgr.TVAR], [lgr.polys[1][0]])(lgr.roots[1][-1]) == 0
    assert ca.Function("a1", [lgr.TVAR], [lgr.polys[1][1]])(lgr.roots[1][0]) == 0
    assert ca.Function("a1", [lgr.TVAR], [lgr.polys[1][1]])(lgr.roots[1][-1]) == 1

    D = lgr.get_diff_matrix(1, order=1)
    assert ((D - ca.DM([[-1 / h, 1 / h], [-1 / h, 1 / h]])).full() < 1e-6).all()

    D = lgr.get_diff_matrix(1, order=2)
    assert ((D - ca.DM.zeros(2, 2)).full() < 1e-6).all()


def test_LGL(collocation_LGL):
    lgr = collocation_LGL
    assert (
        np.abs(
            lgr.roots[1]
            - np.array([mp.CollocationRoots._TAU_MIN, mp.CollocationRoots._TAU_MAX])
        )
        < 1e-6
    ).all()
    h = lgr.roots[1][-1] - lgr.roots[1][0]

    assert ca.Function("a1", [lgr.TVAR], [lgr.polys[1][0]])(lgr.roots[1][0]) == 1
    assert ca.Function("a1", [lgr.TVAR], [lgr.polys[1][0]])(lgr.roots[1][-1]) == 0
    assert ca.Function("a1", [lgr.TVAR], [lgr.polys[1][1]])(lgr.roots[1][0]) == 0
    assert ca.Function("a1", [lgr.TVAR], [lgr.polys[1][1]])(lgr.roots[1][-1]) == 1

    D = lgr.get_diff_matrix(1, order=1)
    assert ((D - ca.DM([[-1 / h, 1 / h], [-1 / h, 1 / h]])).full() < 1e-6).all()

    D = lgr.get_diff_matrix(1, order=2)
    assert ((D - ca.DM.zeros(2, 2)).full() < 1e-6).all()


@pytest.fixture
def ocp_with_solution():
    # Example 3.10 Nonlinear and Dynamic Optimization: From Theory to Practice. By B. Chachuat 2007 Automatic Control Laboratory, EPFL, Switzerland
    # https://www.epfl.ch/labs/la/wp-content/uploads/2018/08/ic-32_lectures-17-28.pdf
    ocp = mp.OCP(n_states=1, n_controls=1)

    def dynamics(x, u, t):
        return [2 * (1 - u[0])]

    def running_cost(x, u, t):
        return 0.5 * u[0] * u[0] - x[0]

    ocp.dynamics[0] = dynamics
    ocp.running_costs[0] = running_cost

    ocp.x00[0] = [1.0]
    # ocp.lbx[0] = 0
    # ocp.ubx[0] = 1
    ocp.lbtf[0] = 1.0
    ocp.ubtf[0] = 1.0

    ocp.validate()

    return ocp


@pytest.fixture
def mpo_with_solution(ocp_with_solution):
    mp.CollocationRoots._TAU_MIN = 0
    mpo = mp.mpopt(ocp_with_solution, 1, 5)
    mpo.validate()

    return mpo


def test_mpopt_ocp_solution_numpy_numerical(mpo_with_solution):
    mpo = mpo_with_solution
    mp.CollocationRoots._TAU_MIN = -1
    mp.Collocation.D_MATRIX_METHOD = "numerical"

    sol = mpo.solve()
    post = mpo.process_results(sol, plot=False)
    x, u, t, _ = post.get_data()
    assert (abs(x - (-2 * t * t + 6 * t + 1)) < 1e-6).all()
    assert (abs(u - 2 * (t - 1)) < 1e-6).all()


def test_mpopt_get_state_second_derivative_single_phase(mpo_with_solution):
    mpo = mpo_with_solution

    sol = mpo.solve()
    post = mpo.process_results(sol, plot=False)
    x, u, t, _ = post.get_data()

    assert (abs(x - (-2 * t * t + 6 * t + 1)) < 1e-3).all()

    # Get the collocation points
    taus = [mpo.collocation._taus_fn(deg)[1:-1] for deg in mpo.poly_orders]

    # Estimate second derivative
    # nodes = [taus for phase in range(mpo._ocp.n_phases)]
    time, ddx, ddu = mpo.get_state_second_derivative_single_phase(sol, nodes=taus)

    assert np.array(
        [(abs(ddx_seg + 4) < 1e-3).all() for i, ddx_seg in enumerate(ddx)]
    ).all()

    assert np.array(
        [(abs(ddu_seg) < 1e-3).all() for i, ddu_seg in enumerate(ddu)]
    ).all()


def test_mpopt_get_interpolated_time_grid():
    t_orig = np.array([0, 0.33, 1])
    taus = [t_orig]
    poly_orders = [2]
    tau0 = 0
    tau1 = 1

    time_grid = mp.mpopt.get_interpolated_time_grid(
        t_orig, taus, poly_orders, tau0, tau1
    ).full()

    assert (abs(np.concatenate(time_grid) - t_orig) < 1e-6).all()

    t_orig = np.array([0, 0.5, 1])
    taus = [np.array([0, 1]), np.array([1])]
    poly_orders = [1, 1]
    tau0 = 0
    tau1 = 1

    time_grid = mp.mpopt.get_interpolated_time_grid(
        t_orig, taus, poly_orders, tau0, tau1
    ).full()

    assert (abs(np.concatenate(time_grid) - t_orig) < 1e-6).all()

    t_orig = np.array([0, 0.5, 1])
    taus = [np.array([-1, 0, 1])]
    poly_orders = [2]
    tau0 = -1
    tau1 = 1

    time_grid = mp.mpopt.get_interpolated_time_grid(
        t_orig, taus, poly_orders, tau0, tau1
    ).full()

    assert (abs(np.concatenate(time_grid) - t_orig) < 1e-6).all()


def test_mpopt_compute_states_from_solution_dynamics(two_phase_schwartz_mpo):
    mpo = two_phase_schwartz_mpo
    sol = mpo.solve()
    post = mpo.process_results(sol, plot=False)

    # Get the collocation points
    taus = [mpo.collocation._taus_fn(deg)[1:-1] for deg in mpo.poly_orders]

    # Estimate residual at collocation points in first phase
    x, u, t, _ = post.get_data(phases=[0])
    states, controls, time, residuals = mpo.compute_states_from_solution_dynamics(
        sol, 0, taus
    )

    print(time[0], np.concatenate(t))
    assert (abs(time[0] - np.concatenate(t[1:-1])) < 1e-3).all()
    assert (abs(states - x[1:-1]) < 1e-3).all()

    # Estimate residual at collocation points in second phase
    x, u, t, _ = post.get_data(phases=[1])
    states, controls, time, residuals = mpo.compute_states_from_solution_dynamics(
        sol, 1, taus
    )

    assert (abs(time[0] - np.concatenate(t[1:-1])) < 1e-3).all()
    print(states, x[1:-1])
    assert (abs(states - x[1:-1]) < 1e-3).all()


def test_mpopt_compute_states_from_solution_dynamics_1P(mpo_with_solution):
    mpo = mpo_with_solution

    sol = mpo.solve()
    post = mpo.process_results(sol, plot=False)
    x, u, t, _ = post.get_data()

    assert (abs(x - (-2 * t * t + 6 * t + 1)) < 1e-3).all()

    # Get the collocation points
    taus = [mpo.collocation._taus_fn(deg)[1:-1] for deg in mpo.poly_orders]

    # Estimate residual at collocation points in first phase
    states, controls, time, residuals = mpo.compute_states_from_solution_dynamics(
        sol, 0, taus
    )

    assert (abs(time[0] - np.concatenate(t[1:-1])) < 1e-3).all()
    assert (abs(states - x[1:-1]) < 1e-3).all()

    # assert np.array(
    #     [(abs(ddx_seg + 4) < 1e-3).all() for i, ddx_seg in enumerate(ddx)]
    # ).all()
    #
    # assert np.array(
    #     [(abs(ddu_seg) < 1e-3).all() for i, ddu_seg in enumerate(ddu)]
    # ).all()


def test_mpopt_get_states_residuals(moon_lander_mpo):
    mp.mpopt._MAX_GRID_POINTS = 5
    mpo = moon_lander_mpo
    sol = mpo.solve()
    post = mpo.process_results(sol, plot=False)
    # x, u, t, _ = post.get_data(interpolate=True)
    # Spectal
    mpo.grid_type = ["spectral" for _ in range(mpo._ocp.n_phases)]
    _, _, ti, residual = mpo.get_states_residuals(sol, residual_type=None)
    for res_seg in residual[0]:
        if res_seg is not None:
            assert abs(np.array(res_seg)).max() < 1e-1

    # fixed
    # mpo = moon_lander_mpo
    # mpo.grid_type = ["fixed" for _ in range(mpo._ocp.n_phases)]
    # sol = mpo.solve()
    # post = mpo.process_results(sol, plot=False)
    #
    # _, _, ti, residual = mpo.get_states_residuals(sol, residual_type=None)
    # for res_seg in residual[0]:
    #     if res_seg is not None:
    #         assert abs(np.array(res_seg)).max() < 1e-1
    #
    # # mid-points
    # mpo = moon_lander_mpo
    # mpo.grid_type = ["mid-points" for _ in range(mpo._ocp.n_phases)]
    # sol = mpo.solve()
    # post = mpo.process_results(sol, plot=False)
    #
    # _, _, ti, residual = mpo.get_states_residuals(sol, residual_type=None)
    # for res_seg in residual[0]:
    #     if res_seg is not None:
    #         assert abs(np.array(res_seg)).max() < 1e-1
