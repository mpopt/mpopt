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
Created: 13th May 2020
Author : Devakumar Thammisetty
Description : Hyper-sensitive OCP
"""
try:
    from mpopt import mp
except ModuleNotFoundError:
    from context import mpopt
    from mpopt import mp

ocp = mp.OCP(n_states=1, n_controls=1, n_phases=1)

ocp.dynamics[0] = lambda x, u, t: [-x[0] * x[0] * x[0] + u[0]]
ocp.running_costs[0] = lambda x, u, t: 0.5 * (x[0] * x[0] + u[0] * u[0])
ocp.terminal_constraints[0] = lambda xf, tf, x0, t0: [xf[0] - 1.0]

ocp.x00[0] = 1
ocp.lbtf[0] = ocp.ubtf[0] = 1000.0
ocp.scale_t = 1 / 1000.0

ocp.validate()

if __name__ == "__main__":
    seg, p = 50, 3
    mpo = mp.mpopt(ocp, seg, p)
    sol = mpo.solve()
    post = mpo.process_results(sol, plot=True)
    mp.plt.title(
        f"non-adaptive solution segments = {mpo.n_segments} poly={mpo.poly_orders[0]}"
    )
    #
    # mpo = mp.mpopt_h_adaptive(ocp, seg, p)
    # sol = mpo.solve(
    #     max_iter=10, mpopt_options={"method": "residual", "sub_method": "merge_split"}
    # )
    # post = mpo.process_results(sol, plot=True)
    # mp.plt.title(
    #     f"Adaptive solution: merge_split : segments = {mpo.n_segments} poly={mpo.poly_orders[0]}"
    # )
    #
    # mpo = mp.mpopt_h_adaptive(ocp, seg, p)
    # sol = mpo.solve(
    #     max_iter=10, mpopt_options={"method": "residual", "sub_method": "equal_area"}
    # )
    # post = mpo.process_results(sol, plot=True)
    # mp.plt.title(
    #     f"Adaptive solution: equal_residual : segments = {mpo.n_segments} poly={mpo.poly_orders[0]}"
    # )

    # mpo = mp.mpopt_h_adaptive(ocp, seg, p)
    # sol = mpo.solve(
    #     max_iter=10, mpopt_options={"method": "control_slope", "sub_method": ""}
    # )
    # post = mpo.process_results(sol, plot=True)
    # mp.plt.title(
    #     f"Adaptive solution: Control slope : segments = {mpo.n_segments} poly={mpo.poly_orders[0]}"
    # )

    mpo = mp.mpopt_adaptive(ocp, 3, 30)
    mpo._SEG_WIDTH_MIN = 1e-6
    mpo.__init__(ocp, 3, 30)
    mpo.mid_residuals = True
    sol = mpo.solve()
    post = mpo.process_results(sol, plot=True)
    mp.plt.title(
        f"Adaptive solution: Direct opt. : segments = {mpo.n_segments} poly={mpo.poly_orders[0]}"
    )
    mp.plt.show()
