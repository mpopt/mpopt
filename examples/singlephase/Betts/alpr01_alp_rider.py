#
# Copyright (c) 2023 Devakumar Thammisetty.
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
Created: 21st June 2023
Author : Devakumar Thammisetty
Description : Alp Rider example from "A Collection of Optimal Control Test Problems" John T. Betts 2015, page 29
"""
import numpy as np
import casadi as ca

try:
    from context import mpopt
    from mpopt import mp
except ModuleNotFoundError:
    from mpopt import mp

ocp = mp.OCP(n_states=4, n_controls=2)


def dynamics0(x, u, t):
    return [
        -10 * x[0] + u[0] + u[1],
        -2 * x[1] + u[0] + 2 * u[1],
        -3 * x[2] + 5 * x[3] + u[0] - u[1],
        5 * x[2] - 3 * x[3] + u[0] + 3 * u[1],
    ]


ocp.dynamics[0] = dynamics0


def terminal_constraints0(xf, tf, x0, t0):
    return [xf[0] - 2.0, xf[1] - 3.0, xf[2] - 1.0, xf[3] + 2]


ocp.terminal_constraints[0] = terminal_constraints0


def running_cost0(x, u, t):

    return 100 * (x[0] * x[0] + x[1] * x[1] + x[2] * x[2] + x[3] * x[3]) + 0.01 * (
        u[0] * u[0] + u[1] * u[1]
    )


ocp.running_costs[0] = running_cost0


def path_constraints(x, u, t):
    return [
        3.0 * ca.exp(-12 * (t - 3) * (t - 3))
        + 3.0 * ca.exp(-10 * (t - 6) * (t - 6))
        + 3.0 * ca.exp(-6 * (t - 10) * (t - 10))
        + 8.0 * ca.exp(-4 * (t - 15) * (t - 15))
        + 0.01
        - x[0] * x[0]
        - x[1] * x[1]
        - x[2] * x[2]
        - x[3] * x[3]
    ]


ocp.path_constraints[0] = path_constraints

ocp.x00[0] = [2.0, 1.0, 2.0, 1.0]
ocp.xf0[0] = [2.0, 3.0, 1.0, -2.0]
ocp.tf0[0] = 20

ocp.lbtf[0] = 20.0
ocp.ubtf[0] = 20.0

ocp.validate()
alpr01 = mp.mpopt(ocp, 10, 5, "LGR")

if __name__ == "__main__":
    mpo = mp.mpopt(ocp, 10, 5, "LGR")
    solution = mpo.solve()

    # mp.post_process._INTERPOLATION_NODES_PER_SEG = 200
    post = mpo.process_results(solution)

    print("Optimal cost :", solution["f"])
    #
    mpo = mp.mpopt_h_adaptive(ocp, 5, 10, "LGR")
    # options = {"method": "residual", "sub_method": "merge_split"}
    options = {"method": "residual", "sub_method": "equal_area"}
    # options = {"method": "control_slope", "sub_method": ""}
    mpo.tol_residual[0] = 1e-4
    solution = mpo.solve(max_iter=10, mpopt_options=options)
    post = mpo.process_results(solution)
    print("Optimal cost Adaptive :", solution["f"])

    mpo = mp.mpopt_adaptive(ocp, 10, 15, "LGR")
    mpo.mid_residuals = True
    ocp.midu[0] = 1
    ocp.du_continuity[0] = 1

    mpo.lbh[0] = 1e-4

    solution = mpo.solve()
    post = mpo.process_results(solution)
    print("Optimal cost Adaptive :", solution["f"])
    # post._INTERPOLATION_NODES_PER_SEG = 100
    #
    # fig, axs = post.plot_phases()
    mp.plt.show()
