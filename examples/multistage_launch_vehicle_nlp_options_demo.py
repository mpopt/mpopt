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
Created: 6th May 2020
Author : Devakumar Thammisetty
Description : OCP definition for use in NLP transcription
gpops2.com/resources/gpops2UsersGuide.pdf
"""
import numpy as np
import casadi as ca
import time

try:
    from mpopt import mp
except ModuleNotFoundError:
    from context import mpopt
    from mpopt import mp


ocp = mp.OCP(n_states=7, n_controls=3, n_phases=4)

# Constants
Re = 6378145.0  # m
omegaE = 7.29211585e-5
rho0 = 1.225
rhoH = 7200.0
Sa = 4 * np.pi
Cd = 0.5
muE = 3.986012e14
g0 = 9.80665

# Variable initialization
lat0 = 28.5 * np.pi / 180.0
r0 = np.array([Re * np.cos(lat0), 0.0, Re * np.sin(lat0)])
v0 = omegaE * np.array([-r0[1], r0[0], 0.0])
m0 = 301454.0
mf = 4164.0
mdrySrb = 19290.0 - 17010.0
mdryFirst = 104380.0 - 95550.0
mdrySecond = 19300.0 - 16820.0
x0 = np.array([r0[0], r0[1], r0[2], v0[0], v0[1], v0[2], m0])

# Step-1 : Define dynamics
# Thrust(N) and mass flow rate(kg/s) in each stage
Thrust = [6 * 628500.0 + 1083100.0, 3 * 628500.0 + 1083100.0, 1083100.0, 110094.0]
mdot = [
    (6 * 17010.0) / (75.2) + (95550.0) / (261.0),
    (3 * 17010.0) / (75.2) + (95550.0) / (261.0),
    (95550.0) / (261.0),
    16820.0 / 700.0,
]


def dynamics(x, u, t, param=0, T=0.0, mdot=0.0):
    r = x[:3]
    v = x[3:6]
    m = x[6]
    r_mag = ca.sqrt(r[0] * r[0] + r[1] * r[1] + r[2] * r[2])
    v_rel = ca.vertcat(v[0] + r[1] * omegaE, v[1] - r[0] * omegaE, v[2])
    v_rel_mag = ca.sqrt(v_rel[0] * v_rel[0] + v_rel[1] * v_rel[1] + v_rel[2] * v_rel[2])
    h = r_mag - Re
    rho = rho0 * ca.exp(-h / rhoH)
    D = -rho / (2 * m) * Sa * Cd * v_rel_mag * v_rel
    g = -muE / (r_mag * r_mag * r_mag) * r

    xdot = [
        x[3],
        x[4],
        x[5],
        T / m * u[0] + param * D[0] + g[0],
        T / m * u[1] + param * D[1] + g[1],
        T / m * u[2] + param * D[2] + g[2],
        -mdot,
    ]
    return xdot


def get_dynamics(param):
    dynamics0 = lambda x, u, t: dynamics(
        x, u, t, param=param, T=Thrust[0], mdot=mdot[0]
    )
    dynamics1 = lambda x, u, t: dynamics(
        x, u, t, param=param, T=Thrust[1], mdot=mdot[1]
    )
    dynamics2 = lambda x, u, t: dynamics(
        x, u, t, param=param, T=Thrust[2], mdot=mdot[2]
    )
    dynamics3 = lambda x, u, t: dynamics(
        x, u, t, param=param, T=Thrust[3], mdot=mdot[3]
    )

    return [dynamics0, dynamics1, dynamics2, dynamics3]


ocp.dynamics = get_dynamics(0)


def path_constraints0(x, u, t):
    return [
        u[0] * u[0] + u[1] * u[1] + u[2] * u[2] - 1,
        -u[0] * u[0] - u[1] * u[1] - u[2] * u[2] + 1,
        -ca.sqrt(x[0] * x[0] + x[1] * x[1] + x[2] * x[2]) / Re + 1,
    ]


ocp.path_constraints = [path_constraints0] * ocp.n_phases


def terminal_cost3(xf, tf, x0, t0):
    return -xf[-1] / m0


ocp.terminal_costs[3] = terminal_cost3


def terminal_constraints3(x, t, x0, t0):
    # https://space.stackexchange.com/questions/1904/how-to-programmatically-calculate-orbital-elements-using-position-velocity-vecto
    # http://control.asu.edu/Classes/MAE462/462Lecture06.pdf
    h = ca.vertcat(
        x[1] * x[5] - x[4] * x[2], x[3] * x[2] - x[0] * x[5], x[0] * x[4] - x[1] * x[3]
    )

    n = ca.vertcat(-h[1], h[0], 0)
    r = ca.sqrt(x[0] * x[0] + x[1] * x[1] + x[2] * x[2])

    e = ca.vertcat(
        1 / muE * (x[4] * h[2] - x[5] * h[1]) - x[0] / r,
        1 / muE * (x[5] * h[0] - x[3] * h[2]) - x[1] / r,
        1 / muE * (x[3] * h[1] - x[4] * h[0]) - x[2] / r,
    )

    e_mag = ca.sqrt(e[0] * e[0] + e[1] * e[1] + e[2] * e[2])
    h_sq = h[0] * h[0] + h[1] * h[1] + h[2] * h[2]
    v_mag = ca.sqrt(x[3] * x[3] + x[4] * x[4] + x[5] * x[5])

    a = -muE / (v_mag * v_mag - 2.0 * muE / r)
    i = ca.acos(h[2] / ca.sqrt(h_sq))
    n_mag = ca.sqrt(n[0] * n[0] + n[1] * n[1])

    node_asc = ca.acos(n[0] / n_mag)
    # if n[1] < -1e-12:
    node_asc = 2 * np.pi - node_asc

    argP = ca.acos((n[0] * e[0] + n[1] * e[1]) / (n_mag * e_mag))
    # if e[2] < 0:
    #    argP = 2*np.pi - argP

    a_req = 24361140.0
    e_req = 0.7308
    i_req = 28.5 * np.pi / 180.0
    node_asc_req = 269.8 * np.pi / 180.0
    argP_req = 130.5 * np.pi / 180.0

    return [
        (a - a_req) / (Re),
        e_mag - e_req,
        i - i_req,
        node_asc - node_asc_req,
        argP - argP_req,
    ]


ocp.terminal_constraints[3] = terminal_constraints3

ocp.scale_x = [
    1 / Re,
    1 / Re,
    1 / Re,
    1 / np.sqrt(muE / Re),
    1 / np.sqrt(muE / Re),
    1 / np.sqrt(muE / Re),
    1 / m0,
]
ocp.scale_t = np.sqrt(muE / Re) / Re


# Initial guess estimation
def ae_to_rv(a, e, i, node, argP, th):
    p = a * (1.0 - e * e)
    r = p / (1.0 + e * np.cos(th))

    r_vec = np.array([r * np.cos(th), r * np.sin(th), 0.0])
    v_vec = np.sqrt(muE / p) * np.array([-np.sin(th), e + np.cos(th), 0.0])

    cn, sn = np.cos(node), np.sin(node)
    cp, sp = np.cos(argP), np.sin(argP)
    ci, si = np.cos(i), np.sin(i)

    R = np.array(
        [
            [cn * cp - sn * sp * ci, -cn * sp - sn * cp * ci, sn * si],
            [sn * cp + cn * sp * ci, -sn * sp + cn * cp * ci, -cn * si],
            [sp * si, cp * si, ci],
        ]
    )

    r_i = np.dot(R, r_vec)
    v_i = np.dot(R, v_vec)

    return r_i, v_i


# Target conditions
a_req = 24361140.0
e_req = 0.7308
i_req = 28.5 * np.pi / 180.0
node_asc_req = 269.8 * np.pi / 180.0
argP_req = 130.5 * np.pi / 180.0
th = 0.0
rf, vf = ae_to_rv(a_req, e_req, i_req, node_asc_req, argP_req, th)

# Timings
t0, t1, t2, t3, t4 = 0.0, 75.2, 150.4, 261.0, 924.0
# Interpolate to get starting values for intermediate phases
xf = np.array([rf[0], rf[1], rf[2], vf[0], vf[1], vf[2], mf + mdrySecond])
x1 = x0 + (xf - x0) / (t4 - t0) * (t1 - t0)
x2 = x0 + (xf - x0) / (t4 - t0) * (t2 - t0)
x3 = x0 + (xf - x0) / (t4 - t0) * (t3 - t0)

# Update the state discontinuity values across phases
x0f = np.copy(x1)
x0f[-1] = x0[-1] - (6 * 17010.0 + 95550.0 / t3 * t1)
x1[-1] = x0f[-1] - 6 * mdrySrb

x1f = np.copy(x2)
x1f[-1] = x1[-1] - (3 * 17010.0 + 95550.0 / t3 * (t2 - t1))
x2[-1] = x1f[-1] - 3 * mdrySrb

x2f = np.copy(x3)
x2f[-1] = x2[-1] - (95550.0 / t3 * (t3 - t2))
x3[-1] = x2f[-1] - mdryFirst

# Step-8b: Initial guess for the states, controls and phase start and final
#             times
ocp.x00 = np.array([x0, x1, x2, x3])
ocp.xf0 = np.array([x0f, x1f, x2f, xf])
ocp.u00 = np.array([[1, 0, 0], [1, 0, 0], [0, 1, 0], [0, 1, 0]])
ocp.uf0 = np.array([[0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0]])
ocp.t00 = np.array([[t0], [t1], [t2], [t3]])
ocp.tf0 = np.array([[t1], [t2], [t3], [t4]])

# Step-8c: Bounds for states
rmin, rmax = -2 * Re, 2 * Re
vmin, vmax = -10000.0, 10000.0
rvmin = [rmin, rmin, rmin, vmin, vmin, vmin]
rvmax = [rmax, rmax, rmax, vmax, vmax, vmax]
lbx0 = rvmin + [x0f[-1]]
lbx1 = rvmin + [x1f[-1]]
lbx2 = rvmin + [x2f[-1]]
lbx3 = rvmin + [xf[-1]]
ubx0 = rvmax + [x0[-1]]
ubx1 = rvmax + [x1[-1]]
ubx2 = rvmax + [x2[-1]]
ubx3 = rvmax + [x3[-1]]
ocp.lbx = np.array([lbx0, lbx1, lbx2, lbx3])
ocp.ubx = np.array([ubx0, ubx1, ubx2, ubx3])

# Bounds for control inputs
umin = [-1, -1, -1]
umax = [1, 1, 1]
ocp.lbu = np.array([umin] * ocp.n_phases)
ocp.ubu = np.array([umax] * ocp.n_phases)

# Bounds for phase start and final times
ocp.lbt0 = np.array([[t0], [t1], [t2], [t3]])
ocp.ubt0 = np.array([[t0], [t1], [t2], [t3]])
ocp.lbtf = np.array([[t1], [t2], [t3], [t4 - 100]])
ocp.ubtf = np.array([[t1], [t2], [t3], [t4 + 100]])

# Event constraint bounds on states : State continuity/disc.
lbe0 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -6 * mdrySrb]
lbe1 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -3 * mdrySrb]
lbe2 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -mdryFirst]
ocp.lbe = np.array([lbe0, lbe1, lbe2])
ocp.ube = np.array([lbe0, lbe1, lbe2])
ocp.validate()

# Solve with drag disabled
ocp.dynamics = get_dynamics(0)


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
        x, u, t, a = post.get_data(interpolate=True)
        mp.plt.plot(t, u[:, 1])
        mp.plt.draw()

        time.sleep(0.25)

        return [0]


if __name__ == "__main__":
    mpo = mp.mpopt(ocp, 1, 11)
    sol = mpo.solve()

    # Solve with drag enabled and initial guess
    ocp.dynamics = get_dynamics(1)
    ocp.validate()

    mycallback = MyCallback(
        "mycallback", mpo.Z.shape[0], mpo.G.shape[0], mpo._nlp_sw_params
    )
    mpo._ocp = ocp
    sol = mpo.solve(
        sol,
        reinitialize_nlp=True,
        nlp_solver_options={
            "ipopt.acceptable_tol": 1e-6,
            "iteration_callback": mycallback,
            "iteration_callback_step": 5,
        },
    )
    print("Final mass : ", round(-sol["f"].full()[0, 0] * m0, 4))

    mp.post_process._INTERPOLATION_NODES_PER_SEG = 200
    # Post processing
    post = mpo.process_results(sol, plot=False, scaling=False)

    # ************** Plot height and velocity ************************
    x, u, t, _ = post.get_data(interpolate=True)
    print("Final time : ", t[-1])
    figu, axsu = post.plot_u()

    # Plot mass
    figm, axsm = post.plot_single_variable(
        x * 1e-3,
        t,
        [[-1]],
        axis=0,
        fig=None,
        axs=None,
        tics=["-"] * 15,
        name="mass",
        ylabel="Mass in tons",
    )

    # Compute and plot altitude, velocity
    r = 1e-3 * (np.sqrt(x[:, 0] ** 2 + x[:, 1] ** 2 + x[:, 2] ** 2) - Re)
    v = 1e-3 * np.sqrt(x[:, 3] ** 2 + x[:, 4] ** 2 + x[:, 5] ** 2)
    y = np.column_stack((r, v))
    fig, axs = post.plot_single_variable(y, t, [[0], [1]], axis=0)

    x, u, t, _ = post.get_data(interpolate=False)
    r = 1e-3 * (np.sqrt(x[:, 0] ** 2 + x[:, 1] ** 2 + x[:, 2] ** 2) - Re)
    v = 1e-3 * np.sqrt(x[:, 3] ** 2 + x[:, 4] ** 2 + x[:, 5] ** 2)
    y = np.column_stack((r, v))
    fig, axs = post.plot_single_variable(
        y, t, [[0], [1]], axis=0, fig=fig, axs=axs, tics=["."] * 15
    )
    axs[0].set(ylabel="Altitude, km", xlabel="Time, s")
    axs[1].set(ylabel="Velocity, km/s", xlabel="Time, s")

    # Plot mass at the collocation nodes
    figm, axsm = post.plot_single_variable(
        x * 1e-3, t, [[-1]], axis=0, fig=figm, axs=axsm, tics=["."] * 15
    )

    mp.plt.show()
