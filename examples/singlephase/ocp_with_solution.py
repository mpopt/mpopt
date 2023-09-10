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
    from context import mpopt
    from mpopt import mp
except ModuleNotFoundError:
    from mpopt import mp

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

ocpwithsolution = mp.mpopt(ocp, 2, 10)

ocp.validate()

if __name__ == "__main__":
    mp.CollocationRoots._TAU_MIN = -1
    mpo = mp.mpopt(ocp, 10, 3)
    sol = mpo.solve()
    post = mpo.process_results(sol, plot=True)
    mp.plt.title(
        f"non-adaptive solution segments = {mpo.n_segments} poly={mpo.poly_orders[0]}"
    )
    # c = mp.CollocationRoots("LGR")
    # cf = c._taus_fn
    # print(cf(2))

    mp.plt.show()
