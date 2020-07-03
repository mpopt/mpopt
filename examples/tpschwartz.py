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
Created: 12th Mar 2020
Author : Devakumar Thammisetty
"""
try:
    from mpopt import mp
except ModuleNotFoundError:
    from context import mpopt
    from mpopt import mp

ocp = mp.OCP(n_states=2, n_controls=1, n_phases=2)

# Step-1 : Define dynamics
def dynamics0(x, u, t):
    return [x[1], u[0] - 0.1 * (1.0 + 2.0 * x[0] * x[0]) * x[1]]


ocp.dynamics = [dynamics0, dynamics0]


def path_constraints0(x, u, t):
    return [
        1.0 - 9.0 * (x[0] - 1) * (x[0] - 1) - (x[1] - 0.4) * (x[1] - 0.4) / (0.3 * 0.3)
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
if __name__ == "__main__":
    mp.post_process._INTERPOLATION_NODES_PER_SEG = 200

    mpo, lgr = mp.solve(ocp, 1, 15, "LGR", True)
    mp.plt.title(
        f"non-adaptive solution using LGR scheme & segments = {mpo.n_segments} poly={mpo.poly_orders[0]}"
    )
    mpo, lgl = mp.solve(ocp, 1, 15, "LGL", True)
    mp.plt.title(
        f"non-adaptive solution using LGL scheme & segments = {mpo.n_segments} poly={mpo.poly_orders[0]}"
    )
    mpo, cgl = mp.solve(ocp, 1, 15, "CGL", True)
    mp.plt.title(
        f"non-adaptive solution using CGL scheme & segments = {mpo.n_segments} poly={mpo.poly_orders[0]}"
    )

    print(lgr.solution["f"], lgl.solution["f"], cgl.solution["f"])
    mp.plt.show()
