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
"""Van der Pol oscilator OCP from https://web.casadi.org/docs/
"""
# try:
#     from mpopt import mp
# except ModuleNotFoundError:
from context import mpopt
from mpopt import mp

ocp = mp.OCP(n_states=2, n_controls=1, n_params=1)


def dynamics(x, u, t, a):
    return [(1 - x[1] * x[1]) * x[0] - x[1] + u[0], x[0]]


def running_cost(x, u, t, a):
    return x[0] * x[0] + x[1] * x[1] + u[0] * u[0]


def path_constraints(x, u, t, a):
    return [a[0] - x[1]]


ocp.dynamics[0] = dynamics
ocp.running_costs[0] = running_cost
ocp.path_constraints[0] = path_constraints

ocp.x00[0] = [0, 1]

ocp.lbu[0] = -1.0
ocp.ubu[0] = 1.0

ocp.lba[0] = 0.25
ocp.uba[0] = 0.5

ocp.lbx[0][1] = -0.25

ocp.lbtf[0] = 10.0
ocp.ubtf[0] = 10.0

ocp.validate()

if __name__ == "__main__":
    mp.post_process._INTERPOLATION_NODES_PER_SEG = 200

    seg, p = 3, 9
    mpo, lgr = mp.solve(ocp, seg, p, "LGR", plot=False)
    # mpo, lgl = mp.solve(ocp, seg, p, "LGL", plot=False)
    # mpo, cgl = mp.solve(ocp, seg, p, "CGL", plot=False)
    x, u, t, a = lgr.get_data()
    print(a)

    fig, axs = lgr.plot_phases(name="LGR")
    # fig, axs = lgl.plot_phases(fig=fig,
    # axs=axs, name="LGL")
    # fig, axs = cgl.plot_phases(fig=fig, axs=axs, name="CGL")
    mp.plt.title(
        f"non-adaptive solution segments = {mpo.n_segments} poly={mpo.poly_orders[0]}"
    )

    mph = mp.mpopt_adaptive(ocp, seg, p)
    solh = mph.solve(max_iter=10, mpopt_options={"method": "control_slope"})
    posth = mph.process_results(solh, plot=False)
    fig, axs = posth.plot_phases(fig=None, axs=None)
    mp.plt.title(
        f"Adaptive solution segments = {mph.n_segments} poly={mph.poly_orders[0]}"
    )
    x, u, t, a = posth.get_data()
    print(a)
    mp.plt.show()
