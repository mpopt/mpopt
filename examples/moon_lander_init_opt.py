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
"""
Created: 5th May 2020
Author : Devakumar Thammisetty
"""
try:
    from mpopt import mp
except ModuleNotFoundError:
    from context import mpopt
    from mpopt import mp

ocp = mp.OCP(n_states=2, n_controls=1)


def dynamics0(x, u, t):
    return [x[1], u[0] - 1.5]


ocp.dynamics[0] = dynamics0


def running_cost0(x, u, t):

    return u[0]


ocp.running_costs[0] = running_cost0


def terminal_cost0(xf, tf, x0, t0):
    return -x0[0]


def terminal_constraints0(xf, tf, x0, t0):

    return [xf[0], xf[1]]


ocp.terminal_constraints[0] = terminal_constraints0
ocp.terminal_costs[0] = terminal_cost0

ocp.tf0[0] = 4.0
ocp.x00[0] = [10.0, -2.0]
ocp.lbx[0] = [-20.0, -20.0]
ocp.ubx[0] = [20.0, 20.0]
ocp.lbu[0] = 0
ocp.ubu[0] = 3
ocp.lbtf[0], ocp.ubtf[0] = 4, 4

ocp.validate()

if __name__ == "__main__":
    mpo = mp.mpopt(ocp, 20, 4)
    sol = mpo.solve()
    post = mpo.process_results(sol, plot=True)
    mp.plt.title(
        f"non-adaptive solution segments = {mpo.n_segments} poly={mpo.poly_orders[0]}"
    )
    t, r = mpo.get_dynamics_residuals(sol, plot=False)
    r_lgr, t_lgr = mp.post_process.sort_residual_data(t, r, phases=range(ocp.n_phases))

    mpo = mp.mpopt_h_adaptive(ocp, 10, 4)
    sol = mpo.solve(
        max_iter=10, mpopt_options={"method": "residual", "sub_method": "merge_split"}
    )
    post = mpo.process_results(sol, plot=True)
    mp.plt.title(
        f"Adaptive solution: merge_split : segments = {mpo.n_segments} poly={mpo.poly_orders[0]}"
    )
    t, r = mpo.get_dynamics_residuals(sol, plot=False)
    r_h, t_h = mp.post_process.sort_residual_data(t, r, phases=range(ocp.n_phases))

    # mpo = mp.mpopt_h_adaptive(ocp, 10, 4)
    # sol = mpo.solve(
    #     max_iter=2, mpopt_options={"method": "residual", "sub_method": "equal_area"}
    # )
    # post = mpo.process_results(sol, plot=True)
    # mp.plt.title(
    #     f"Adaptive solution: equal_residual : segments = {mpo.n_segments} poly={mpo.poly_orders[0]}"
    # )
    #
    # mpo = mp.mpopt_h_adaptive(ocp, 5, 4)
    # sol = mpo.solve(
    #     max_iter=10, mpopt_options={"method": "control_slope", "sub_method": ""}
    # )
    # post = mpo.process_results(sol, plot=True)
    # mp.plt.title(
    #     f"Adaptive solution: Control slope : segments = {mpo.n_segments} poly={mpo.poly_orders[0]}"
    # )
    #
    # mpo = mp.mpopt_adaptive(ocp, 3, 2)
    # mpo.lbh[0] = 1e-6
    # mpo.mid_residuals = True
    # sol = mpo.solve()
    # post = mpo.process_results(sol, plot=True)
    # mp.plt.title(
    #     f"Adaptive solution: Direct opt. : segments = {mpo.n_segments} poly={mpo.poly_orders[0]}"
    # )

    mp.plt.figure()
    mp.plt.plot(
        t_lgr,
        r_lgr,
        "g-",  # t_lgl, r_lgl, "b*-", t_cgl, r_cgl, "k--",
        t_h,
        r_h,
        "r-+",
    )

    mp.plt.show()
