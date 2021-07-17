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
Created: 17th May 2020
Author : Devakumar Thammisetty
Description : Direct-trajectory optimization OCP taken from paper:
    Lin Ma, Kexin Wang, Zhijiang Shao, Zhengyu Song & Lorenz T. Biegler
(2019) Direct trajectory optimization framework for vertical takeoff and vertical landing reusable
rockets: case study of two-stage rockets, Engineering Optimization, 51:4, 627-645, DOI:
10.1080/0305215X.2018.1472774

https://doi.org/10.1080/0305215X.2018.1472774
"""
import casadi as ca
import numpy as np
from context import mpopt
from mpopt import mp

ocp = mp.OCP(n_states=7, n_controls=4, n_phases=3)

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
# v0     = omegaE*np.array([-r0[1], r0[0], 0.0])
v0 = omegaE * np.array([0.1, 0.1, 0.1])
m0 = 431.6e3 + 107.5e3
mf = 107.5e3 - 103.5e3
mdryBooster = 431.6e3 - 409.5e3
mdrySecond = mf
x0 = np.array([r0[0], r0[1], r0[2], v0[0], v0[1], v0[2], m0])
q_max = 80 * 1e3
x2f = np.array([5634869.50, 98357.01, 2976509.50, 0.0, 0.0, 0.0, 21000.0])

# Step-1 : Define dynamics
# Thrust(N) and mass flow rate(kg/s) in each stage
Thrust = [9 * 934.0e3, 934.0e3, 934.0e3]


def stage_dynamics(x, u, t, param=0, T=0.0):
    r = x[:3]
    v = x[3:6]
    m = x[6]
    r_mag = ca.sqrt(r[0] * r[0] + r[1] * r[1] + r[2] * r[2])
    v_rel = v  # ca.vertcat(v[0] + r[1] * omegaE, v[1] - r[0] * omegaE, v[2])
    v_rel_mag = ca.sqrt(v_rel[0] * v_rel[0] + v_rel[1] * v_rel[1] + v_rel[2] * v_rel[2])
    h = r_mag - Re
    rho = rho0 * ca.exp(-h / rhoH)
    D = -rho / (2 * m) * Sa * Cd * v_rel_mag * v_rel
    g = -muE / (r_mag * r_mag * r_mag) * r

    xdot = [
        x[3],
        x[4],
        x[5],
        T * u[3] / m * u[0] + param * D[0] + g[0],
        T * u[3] / m * u[1] + param * D[1] + g[1],
        T * u[3] / m * u[2] + param * D[2] + g[2],
        -T * u[3] / (340.0 * g0),
    ]
    return xdot


def get_dynamics(param):
    dynamics0 = lambda x, u, t: stage_dynamics(x, u, t, param=param, T=Thrust[0])
    dynamics1 = lambda x, u, t: stage_dynamics(x, u, t, param=param, T=Thrust[1])
    dynamics2 = lambda x, u, t: stage_dynamics(x, u, t, param=param, T=Thrust[2])

    return [dynamics0, dynamics1, dynamics2]


ocp.dynamics = get_dynamics(0)


def path_constraints0(x, u, t):
    return [
        u[0] * u[0] + u[1] * u[1] + u[2] * u[2] - 1,
        -u[0] * u[0] - u[1] * u[1] - u[2] * u[2] + 1,
        -ca.sqrt(x[0] * x[0] + x[1] * x[1] + x[2] * x[2]) / Re + 1,
    ]


def path_constraints2(x, u, t, dynP=0, gs=0):
    r_mag = ca.sqrt(x[0] * x[0] + x[1] * x[1] + x[2] * x[2])
    h = r_mag - Re
    rho = rho0 * ca.exp(-h / rhoH)
    v_sq = x[3] * x[3] + x[4] * x[4] + x[5] * x[5]
    r_rf = ca.vertcat(x[0] - x0[0], x[1] - x0[1], x[2] - x0[2])
    r_rf_mag = ca.sqrt(r_rf[0] * r_rf[0] + r_rf[1] * r_rf[1] + r_rf[2] * r_rf[2])
    rf_mag = np.sqrt(x0[0] * x0[0] + x0[1] * x0[1] + x0[2] * x0[2])
    glide_slope_factor = np.cos(80.0 * np.pi / 180.0)

    return [
        dynP * 0.5 * rho * v_sq / q_max - 1.0,
        u[0] * u[0] + u[1] * u[1] + u[2] * u[2] - 1,
        -u[0] * u[0] - u[1] * u[1] - u[2] * u[2] + 1,
        -ca.sqrt(x[0] * x[0] + x[1] * x[1] + x[2] * x[2]) / Re + 1,
        gs
        * (
            r_rf_mag * glide_slope_factor
            - (r_rf[0] * x0[0] + r_rf[1] * x0[1] + r_rf[2] * x0[2]) / rf_mag
        ),
    ]


ocp.path_constraints = [path_constraints0, path_constraints0, path_constraints2]


def terminal_cost1(xf, tf, x0, t0):
    return -xf[6] / m0


ocp.terminal_costs[1] = terminal_cost1


def terminal_constraints1(x, t, x0, t0):
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

    a_req = 6593145.0  # 24361140.0
    e_req = 0.0076
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


def terminal_constraints2(x, t, x_0, t_0):
    return [
        (x[0] - x2f[0]) / Re,
        (x[1] - x2f[1]) / Re,
        (x[2] - x2f[2]) / Re,
        (x[3] - x2f[3]) / np.sqrt(muE / Re),
        (x[4] - x2f[4]) / np.sqrt(muE / Re),
        (x[5] - x2f[5]) / np.sqrt(muE / Re),
    ]


ocp.terminal_constraints[1] = terminal_constraints1
ocp.terminal_constraints[2] = terminal_constraints2

# Step-7 : Define scaling for states, inputs, cost integrals, time
ocp.scale_x = np.array(
    [
        1 / Re,
        1 / Re,
        1 / Re,
        1 / np.sqrt(muE / Re),
        1 / np.sqrt(muE / Re),
        1 / np.sqrt(muE / Re),
        1 / m0,
    ]
)
ocp.scale_t = np.sqrt(muE / Re) / Re

# Initial guess estimation
# User defined functions
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
a_req = 6593145.0  # 24361140.0
e_req = 0.0076
i_req = 28.5 * np.pi / 180.0
node_asc_req = 269.8 * np.pi / 180.0
argP_req = 130.5 * np.pi / 180.0
th = 0.0
rf, vf = ae_to_rv(a_req, e_req, i_req, node_asc_req, argP_req, th)
xf = np.array([rf[0], rf[1], rf[2], vf[0], vf[1], vf[2], mf])

# Timings
t0, t1, t2, t3 = 0.0, 131.4, 453.4, 569.7

# Interpolate to get starting values for intermediate phases
x1 = x0 + (xf - x0) / (t2 - t0) * (t1 - t0)

# Update the state discontinuity values across phases
x0f = np.copy(x1)
x0f[-1] = x0[-1] - (9 * 934e3 / (340.0 * g0) * t1)
mFirst_leftout = 409.5e3 - (9 * 934e3 / (340.0 * g0) * t1)
x1[-1] = x0f[-1] - (mdryBooster + mFirst_leftout)

# Step-8b: Initial guess for the states, controls and phase start and final
#             times
ocp.x00 = np.array([x0, x1, x0f])
ocp.xf0 = np.array([x0f, xf, x2f])
ocp.u00 = np.array([[1, 0, 0, 1.0], [1, 0, 0, 1], [0, 1, 0, 1]])
ocp.uf0 = np.array([[0, 1, 0, 1.0], [0, 1, 0, 1], [1, 0, 0, 0.5]])
ocp.t00 = np.array([[t0], [t1], [t1]])
ocp.tf0 = np.array([[t1], [t2], [t3]])

# Step-8c: Bounds for states
rmin, rmax = -2 * Re, 2 * Re
vmin, vmax = -10000.0, 10000.0

lbx0 = [rmin, rmin, rmin, vmin, vmin, vmin, x0f[-1]]
lbx1 = [rmin, rmin, rmin, vmin, vmin, vmin, xf[-1]]
lbx2 = [rmin, rmin, rmin, vmin, vmin, vmin, mdryBooster]
ubx0 = [rmax, rmax, rmax, vmax, vmax, vmax, x0[-1]]
ubx1 = [rmax, rmax, rmax, vmax, vmax, vmax, 107.5e3]
ubx2 = [rmax, rmax, rmax, vmax, vmax, vmax, x0f[-1] - 107.5e3]

ocp.lbx = np.array([lbx0, lbx1, lbx2])
ocp.ubx = np.array([ubx0, ubx1, ubx2])

# Bounds for control inputs
ocp.lbu = np.array(
    [[-1.0, -1.0, -1.0, 1.0], [-1.0, -1.0, -1.0, 1.0], [-1.0, -1.0, -1.0, 0.38]]
)
ocp.ubu = np.array([[1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]])

# Bounds for phase start and final times
ocp.lbt0 = np.array([[t0], [t1], [t1]])
ocp.ubt0 = np.array([[t0], [t1], [t1]])
ocp.lbtf = np.array([[t1], [t2 - 50], [t3 - 50]])
ocp.ubtf = np.array([[t1], [t2 + 50], [t3 + 50]])

# Event constraint bounds on states : State continuity/disc.
lbe0 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -(mdryBooster + mFirst_leftout)]
lbe1 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -107.5e3]
ocp.lbe = np.array([lbe0, lbe1])
ocp.ube = np.array([lbe0, lbe1])

ocp.phase_links = [(0, 1), (0, 2)]
ocp.validate()

# Solve with drag disables
ocp.dynamics = get_dynamics(0)
# ocp.midu[2] = 1
# ocp.diff_u[2] = 1
# ocp.lbdu[2] = -5
# ocp.ubdu[2] = 5
seg = 5
p = [6] * seg
mpo = mp.mpopt_adaptive(ocp, seg, p)
sol = mpo.solve()

# Solve with drag enabled and initial guess
ocp.dynamics = get_dynamics(1)
ocp.path_constraints[2] = lambda x, u, t: path_constraints2(x, u, t, dynP=1, gs=0)

ocp.validate()
mpo = mp.mpopt_adaptive(ocp, seg, p)
sol = mpo.solve(
    sol, max_iter=1, mpopt_options={"method": "control_slope", "sub_method": ""}
)

font = {"family": "serif"}  # , "weight": "bold"}
import matplotlib

matplotlib.rc("font", **font)

SMALL_SIZE = 14
MEDIUM_SIZE = 12
BIGGER_SIZE = 12

mp.plt.rc("font", size=MEDIUM_SIZE)  # controls default text sizes
mp.plt.rc("axes", titlesize=BIGGER_SIZE)  # fontsize of the axes title
mp.plt.rc("xtick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
mp.plt.rc("axes", labelsize=BIGGER_SIZE)  # fontsize of the x and y labels
mp.plt.rc("ytick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
mp.plt.rc("legend", fontsize=MEDIUM_SIZE)  # legend fontsize
mp.plt.tight_layout()

case = "nadp"

print("Final mass : ", -sol["f"] * m0)
# Post processing
post = mpo.process_results(sol, plot=False, scaling=False)
x1, u1, t1 = post.get_data(phases=[0], interpolate=False)

# ************** Plot height and velocity ************************
for id, phase_link in enumerate(ocp.phase_links):
    x, u, t = post.get_data(phases=phase_link, interpolate=True)
    # figx, _ = post.plot_x([[0, 1, 2], [3, 4, 5], [6]])
    figu, axsu = post.plot_u(dims=[0, 1, 2], phases=phase_link)
    # figu, axsu = post.plot_u(
    #     phases=phase_link, interpolate=False, fig=figu, axs=axsu, tics=["."] * 15
    # )
    mp.plt.subplots_adjust(bottom=0.1, left=0.15)
    mp.plt.savefig(f"plots/falc_to_orbit_{case}_u{id}_s{seg}p{p[0]}.eps")
    r = 1e-3 * (np.sqrt(x[:, 0] ** 2 + x[:, 1] ** 2 + x[:, 2] ** 2) - Re)
    v = np.sqrt(x[:, 3] ** 2 + x[:, 4] ** 2 + x[:, 5] ** 2)
    y = np.column_stack((r, v))
    fig, axs = post.plot_single_variable(y, t, [[0], [1]], axis=0)

    x, u, t = post.get_data(phases=phase_link, interpolate=False)
    r = 1e-3 * (np.sqrt(x[:, 0] ** 2 + x[:, 1] ** 2 + x[:, 2] ** 2) - Re)
    v = np.sqrt(x[:, 3] ** 2 + x[:, 4] ** 2 + x[:, 5] ** 2)
    y = np.column_stack((r, v))
    fig, axs = post.plot_single_variable(
        y, t, [[0], [1]], axis=0, fig=fig, axs=axs, tics=["."] * 15
    )
    mp.plt.subplots_adjust(bottom=0.1, left=0.1)
    mp.plt.savefig(f"plots/falc_to_orbit_{case}_hv{id}_s{seg}p{p[0]}.eps")
    print("Final mass, time : ", x[-1][-1], t[-1])

x0, u0, t0 = post.get_data(phases=ocp.phase_links[0], interpolate=True)
x1, u1, t1 = post.get_data(phases=ocp.phase_links[1], interpolate=True)
x0o, u0o, t0o = post.get_data(phases=ocp.phase_links[0], interpolate=False)
x1o, u1o, t1o = post.get_data(phases=ocp.phase_links[1], interpolate=False)

r0 = 1e-3 * (np.sqrt(x0[:, 0] ** 2 + x0[:, 1] ** 2 + x0[:, 2] ** 2) - Re)
v0 = 1e-3 * np.sqrt(x0[:, 3] ** 2 + x0[:, 4] ** 2 + x0[:, 5] ** 2)
y0 = np.column_stack((r0, v0))
fig0, axs0 = post.plot_single_variable(y0, t0, [[0], [1]], axis=0)

r0o = 1e-3 * (np.sqrt(x0o[:, 0] ** 2 + x0o[:, 1] ** 2 + x0o[:, 2] ** 2) - Re)
v0o = 1e-3 * np.sqrt(x0o[:, 3] ** 2 + x0o[:, 4] ** 2 + x0o[:, 5] ** 2)
y0o = np.column_stack((r0o, v0o))
fig0o, axs0o = post.plot_single_variable(
    y0o,
    t0o,
    [[0], [1]],
    axis=0,
    fig=fig0,
    axs=axs0,
    tics=["."] * 15,
    name="Second stage",
)

r1 = 1e-3 * (np.sqrt(x1[:, 0] ** 2 + x1[:, 1] ** 2 + x1[:, 2] ** 2) - Re)
v1 = 1e-3 * np.sqrt(x1[:, 3] ** 2 + x1[:, 4] ** 2 + x1[:, 5] ** 2)
y1 = np.column_stack((r1, v1))
fig1, axs1 = post.plot_single_variable(y1, t1, [[0], [1]], axis=0, fig=fig0o, axs=axs0o)

r1o = 1e-3 * (np.sqrt(x1o[:, 0] ** 2 + x1o[:, 1] ** 2 + x1o[:, 2] ** 2) - Re)
v1o = 1e-3 * np.sqrt(x1o[:, 3] ** 2 + x1o[:, 4] ** 2 + x1o[:, 5] ** 2)
y1o = np.column_stack((r1o, v1o))
fig1o, axs1o = post.plot_single_variable(
    y1o,
    t1o,
    [[0], [1]],
    axis=0,
    fig=fig1,
    axs=axs1,
    tics=["+"] * 15,
    name="Booster stage",
)

axs1o[0].set(ylabel="Altitude, km", title="Altitude profile")
axs1o[1].set(ylabel="Velocity, km/s", title="Velocity profile")
mp.plt.subplots_adjust(bottom=0.1, left=0.1)
mp.plt.savefig(f"plots/falc_to_orbit_{case}_lgr_s{seg}p{p[0]}.eps")

x, u, t = post.get_data(phases=[0], interpolate=True)
x2, u2, t2 = post.get_data(phases=[2], interpolate=True)
u = np.vstack((Thrust[0] * u, Thrust[1] * u2))
t = np.hstack((t, t2))

fig, axs = post.plot_single_variable(u * 1e-3, t, [[3]], axis=0)
x, u, t = post.get_data(phases=[0], interpolate=False)
x2, u2, t2 = post.get_data(phases=[2], interpolate=False)
u = np.vstack((Thrust[0] * u, Thrust[1] * u2))
t = np.vstack((t, t2))
fig, axs = post.plot_single_variable(
    u * 1e-3,
    t,
    [[3]],
    axis=0,
    fig=fig,
    axs=axs,
    name="Thrust",
    ylabel="Thrust, kN",
    tics=["."] * 15,
)

mp.plt.subplots_adjust(bottom=0.1, left=0.15)
mp.plt.savefig(f"plots/falc_to_orbit_{case}_thr_lgr_s{seg}p{p[0]}.eps")
mp.plt.show()
