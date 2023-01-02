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

if __name__ == "__main__":
    import numpy as np

    # mp.post_process._INTERPOLATION_NODES_PER_SEG = 5
    mp.mpopt._MAX_GRID_POINTS = 5

    mpo = mp.mpopt(ocp, 20, 3)
    sol = mpo.solve()
    post = mpo.process_results(sol, plot=True)
    x, u, t, _ = post.get_data(interpolate=True)

    ti, residual = mpo.get_states_residuals(
        sol, grid_type="spectral", residual_type="relative", plot=True
    )

    ti, residual = mpo.get_dynamics_residuals(
        sol, grid_type="spectral", residual_type="relative", plot=True
    )
    mp.plt.show()
