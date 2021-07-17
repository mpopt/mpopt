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
    moon_lander_mpo_h_adaptive.grid_type[0] = "mid_points"
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
