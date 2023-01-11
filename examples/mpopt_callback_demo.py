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

import casadi as ca
import time


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


class MyCallback(ca.Callback):
    def __init__(self, name, nx, ng, np, opts={}):
        ca.Callback.__init__(self)

        self.nx = nx
        self.ng = ng
        self.np = np
        mp.plt.figure(1)
        mp.plt.subplot(111)
        mp.plt.draw()

        self.construct(name, opts)

    def get_n_in(self):
        return ca.nlpsol_n_out()

    def get_sparsity_in(self, i):
        n = ca.nlpsol_out(i)
        if n == "f":
            return ca.Sparsity.scalar()
        elif n in ("x", "lam_x"):
            return ca.Sparsity.dense(self.nx)
        elif n in ("g", "lam_g"):
            return ca.Sparsity.dense(self.ng)
        else:
            return ca.Sparsity(0, 0)

    def eval(self, arg):
        sol = {}
        for (i, s) in enumerate(ca.nlpsol_out()):
            sol[s] = arg[i]

        post = mpo.process_results(sol, plot=False)
        x, u, t, a = post.get_original_data()
        mp.plt.plot(t, x[:, 0])
        mp.plt.draw()
        mp.plt.xlabel("Time, s")
        mp.plt.ylabel("Height, m")

        time.sleep(0.25)

        return [0]


if __name__ == "__main__":
    mpo = mp.mpopt(ocp, 5, 4)

    # to get the number of variables(x), constraints(g) and parameters(p) in NLP
    nlp, _ = mpo.create_nlp()
    nx, ng, np = nlp["x"].shape[0], nlp["g"].shape[0], nlp["p"].shape[0]
    mycallback = MyCallback("mycallback", nx, ng, np)

    sol = mpo.solve(
        nlp_solver_options={
            "ipopt.acceptable_tol": 1e-6,
            "verbose": False,
            "iteration_callback": mycallback,
            "iteration_callback_step": 5,
        }
    )
    post = mpo.process_results(sol, plot=True)
    mp.plt.title(
        f"non-adaptive solution segments = {mpo.n_segments} poly={mpo.poly_orders[0]}"
    )
    mp.plt.show()
