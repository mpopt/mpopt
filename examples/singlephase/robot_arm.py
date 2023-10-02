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
Created: 15th May 2020
Author : Devakumar Thammisetty
Description : OCP definition for use in NLP transcription

Robot arm example from: Journal of the Franklin Institute 352 (2015) 4081–4106

Adaptive mesh refinement method for optimal control
using nonsmoothness detection and mesh size reduction
Fengjin Liu a , William W. Hager b , Anil V. Rao a, n

http://dx.doi.org/10.1016/j.jfranklin.2015.05.028
"""
import numpy as np
import casadi as ca

try:
    from context import mpopt
    from mpopt import mp
except ModuleNotFoundError:
    from mpopt import mp

ocp = mp.OCP(n_states=6, n_controls=3)


def dynamics0(x, u, t):
    return [
        x[1],
        u[0] / 5.0,
        x[3],
        u[1] / (((5.0 - x[0]) ** 3 + x[0] ** 3) * ca.sin(x[4]) * ca.sin(x[4]) / 3.0),
        x[5],
        u[2] / (((5.0 - x[0]) ** 3 + x[0] ** 3) / 3.0),
    ]


ocp.dynamics[0] = dynamics0


def terminal_cost0(xf, tf, x0, t0):
    return tf


ocp.terminal_costs[0] = terminal_cost0


def terminal_constraints0(xf, tf, x0, t0):
    return [
        xf[0] - 4.5,
        xf[1],
        xf[2] - 2.0 * np.pi / 3.0,
        xf[3],
        xf[4] - np.pi / 4.0,
        xf[5],
    ]


ocp.terminal_constraints[0] = terminal_constraints0

ocp.x00[0] = [4.5, 0, 0, 0, np.pi / 4.0, 0.0]
ocp.xf0[0] = [4.5, 0, 2.0 * np.pi / 3.0, 0, np.pi / 4.0, 0.0]
ocp.tf0[0] = 10
ocp.lbu[0] = [-1.0, -1.0, -1.0]
ocp.ubu[0] = [1.0, 1.0, 1.0]

ocp.lbtf[0] = 10 - 3.0
ocp.ubtf[0] = 10 + 3.0

ocp.validate()

robot_arm = mp.mpopt(ocp, 20, 4, "LGR")

if __name__ == "__main__":
    # mpo = mp.mpopt(ocp, 20, 4, "LGR")
    # solution = mpo.solve()
    #
    # post = mpo.process_results(solution)

    # ocp.midu[0] = 1
    resids = dict()
    fig = mp.plt.figure()
    mp.mpopt._GRID_TYPE = "fixed"
    mp.mpopt_h_adaptive._TOL_RESIDUAL = 1e-3
    mp.mpopt_h_adaptive._TOL_SEG_WIDTH_CHANGE = 0.01

    tags = ["r-", "g.-", "ko-", "y*-", "c+-"] * 15
    for deg in range(5, 6):
        mpo = mp.mpopt_h_adaptive(ocp, deg, 6)
        mpo.plot_residual_evolution = True
        sol = mpo.solve(
            max_iter=5,
            mpopt_options={"method": "residual", "sub_method": "equal_area"},
        )

        resids[deg] = mpo.iter_info
        mp.plt.plot(
            list(mpo.iter_info.keys()),
            list(mpo.iter_info.values()),
            tags[deg],
            label=deg,
        )
    mp.plt.legend()

    post = mpo.process_results(sol)
    mp.plt.title(
        f"H-Adaptive solution : segments = {mpo.n_segments} poly={mpo.poly_orders[0]}"
    )
    mp.plt.savefig("docs/plots/tps_h_ad_merge_split.png")

    # mpo = mp.mpopt_h_adaptive(ocp, 20, 4, "LGR")
    # options = {"method": "residual", "sub_method": "merge_split"}
    # mpo.tol_residual[0] = 1e-4
    # solution = mpo.solve(max_iter=2, mpopt_options=options)
    # post = mpo.process_results(solution)
    # mp.plt.title(
    #     f"Adaptive solution: merge split : segments = {mpo.n_segments} poly={mpo.poly_orders[0]}"
    # )
    #
    # mpo = mp.mpopt_h_adaptive(ocp, 20, 4, "LGR")
    # options = {"method": "control_slope", "sub_method": ""}
    # mpo.tol_residual[0] = 1e-4
    # solution = mpo.solve(max_iter=2, mpopt_options=options)
    # post = mpo.process_results(solution)
    # mp.plt.title(
    #     f"Adaptive solution: Control slope : segments = {mpo.n_segments} poly={mpo.poly_orders[0]}"
    # )

    # mpo = mp.mpopt_adaptive(ocp, 20, 3, "LGR")
    # mpo.mid_residuals = True
    # solution = mpo.solve()
    # post = mpo.process_results(solution)
    # mp.plt.title(
    #     f"Adaptive solution: direct : segments = {mpo.n_segments} poly={mpo.poly_orders[0]}"
    # )
    mp.plt.show()
