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

ocp = mp.OCP(n_states=7, n_controls=4, n_phases=1)

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
# m0 = 431.6e3 - 131.4 * (9 * 934e3 / (340.0 * g0))
mdryBooster = 431.6e3 - 409.5e3
# Boost stage start point
x0 = np.array(
    [
        5.66085493e06,
        8.49015331e04,
        3.07350574e06,
        1.01645801e03,
        1.87411093e03,
        5.49269958e02,
        2.07827673e05 - 107.5e3,
    ]
)
m0 = x0[-1]

xf = np.array([r0[0], r0[1], r0[2], v0[0], v0[1], v0[2], 1e3])
q_max = 80 * 1e3

# Step-1 : Define dynamics
# Thrust(N) and mass flow rate(kg/s) in each stage
Thrust = [934.0e3]


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

    return [dynamics0]


ocp.dynamics = get_dynamics(0)


def path_constraints0(x, u, t, dynP=0, gs=0):
    r_mag = ca.sqrt(x[0] * x[0] + x[1] * x[1] + x[2] * x[2])
    h = r_mag - Re
    rho = rho0 * ca.exp(-h / rhoH)
    v_sq = x[3] * x[3] + x[4] * x[4] + x[5] * x[5]
    # r_rf = ca.vertcat(x[0] - x0[0], x[1] - x0[1], x[2] - x0[2])
    # r_rf_mag = ca.sqrt(r_rf[0] * r_rf[0] + r_rf[1] * r_rf[1] + r_rf[2] * r_rf[2])
    # rf_mag = np.sqrt(x0[0] * x0[0] + x0[1] * x0[1] + x0[2] * x0[2])
    # glide_slope_factor = np.cos(80.0 * np.pi / 180.0)

    return [
        dynP * 0.5 * rho * v_sq / q_max - 1.0,
        u[0] * u[0] + u[1] * u[1] + u[2] * u[2] - 1,
        -u[0] * u[0] - u[1] * u[1] - u[2] * u[2] + 1,
        -ca.sqrt(x[0] * x[0] + x[1] * x[1] + x[2] * x[2]) / Re + 1,
        # gs
        # * (
        #     r_rf_mag * glide_slope_factor
        #     - (r_rf[0] * x0[0] + r_rf[1] * x0[1] + r_rf[2] * x0[2]) / rf_mag
        # ),
    ]


ocp.path_constraints[0] = lambda x, u, t: path_constraints0(x, u, t, dynP=0.0)


def terminal_constraints0(x, t, x_0, t_0):
    return [
        (x[0] - xf[0]),
        (x[1] - xf[1]),
        (x[2] - xf[2]),
        (x[3] - xf[3]) / np.sqrt(muE / Re),
        (x[4] - xf[4]) / np.sqrt(muE / Re),
        (x[5] - xf[5]) / np.sqrt(muE / Re),
    ]


ocp.terminal_constraints[0] = terminal_constraints0


# def terminal_cost0(x_f, tf, x0, t0):
#     return -x_f[6] / m0
#

# ocp.terminal_costs[0] = terminal_cost0

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

# Timings
t0, t1 = 0.0, 390.0  # -131.4 + 569.7

# Step-8b: Initial guess for the states, controls and phase start and final
#             times
ocp.x00 = np.array([x0])
ocp.xf0 = np.array([xf])
ocp.u00 = np.array([[0.5, 0.8, 0.2, 1.0]])
ocp.uf0 = np.array([[1, 0, 0, 1.0]])
ocp.t00 = np.array([[t0]])
ocp.tf0 = np.array([[t1]])

# Step-8c: Bounds for states
rmin, rmax = -1.1 * Re, 1.1 * Re
vmin, vmax = -5000.0, 5000.0

lbx0 = [rmin, rmin, rmin, vmin, vmin, vmin, mdryBooster]
ubx0 = [rmax, rmax, rmax, vmax, vmax, vmax, x0[-1]]

ocp.lbx[0] = lbx0
ocp.ubx[0] = ubx0

# Bounds for control inputs
ocp.lbu[0] = [-1.0, -1.0, -1.0, 0.38]
ocp.ubu[0] = [1.0, 1.0, 1.0, 1.0]

# Bounds for phase start and final times
ocp.lbt0[0] = t0
ocp.ubt0[0] = t0
ocp.lbtf[0] = t1 - 100
ocp.ubtf[0] = t1 + 100

ocp.validate()

# Solve with drag disables
ocp.dynamics = get_dynamics(0)

seg, p, max_iter = 50, 3, 1
mpo = mp.mpopt_h_adaptive(ocp, seg, p)
mpo._INTERPOLATION_NODES_PER_SEG = 100
mpo.lbh[0] = 1e-2
sol = mpo.solve(max_iter=1, mpopt_options={"method": "control_slope", "sub_method": ""})

# Solve with drag enabled and initial guess
ocp.dynamics = get_dynamics(1)
ocp.validate()
ocp.path_constraints[0] = lambda x, u, t: path_constraints0(x, u, t, dynP=1.0)

sol = mpo.solve(
    sol,
    max_iter=5,
    mpopt_options={"method": "control_slope", "sub_method": ""},
    reinitialize_nlp=True,
)
# print("Final mass : ", -sol["f"] * m0)
# Post processing
post = mpo.process_results(sol, plot=False, scaling=False)
post._INTERPOLATION_NODES_PER_SEG = 200
# ************** Plot height and velocity ************************
ocp.phase_links = [[0]]
for phase_link in ocp.phase_links:
    x, u, t, _ = post.get_data(phases=phase_link, interpolate=True)
    # figx, _ = post.plot_x([[0, 1, 2], [3, 4, 5], [6]])
    # figu, axsu = post.plot_u(phases=phase_link)
    figu, axsu = post.plot_u(phases=phase_link, fig=None, axs=None, tics=["."] * 15)
    r = 1e-3 * (np.sqrt(x[:, 0] ** 2 + x[:, 1] ** 2 + x[:, 2] ** 2) - Re)
    v = np.sqrt(x[:, 3] ** 2 + x[:, 4] ** 2 + x[:, 5] ** 2)
    y = np.column_stack((r, v))
    fig, axs = post.plot_single_variable(y, t, [[0], [1]], axis=0)

    x, u, t, _ = post.get_data(phases=phase_link, interpolate=False)
    r = 1e-3 * (np.sqrt(x[:, 0] ** 2 + x[:, 1] ** 2 + x[:, 2] ** 2) - Re)
    v = np.sqrt(x[:, 3] ** 2 + x[:, 4] ** 2 + x[:, 5] ** 2)
    y = np.column_stack((r, v))
    fig, axs = post.plot_single_variable(
        y, t, [[0], [1]], axis=0, fig=fig, axs=axs, tics=["."] * 15
    )
    mp.plt.show()
    print("Final time, mass", t[-1], t[-1] + 131.4, x[-1][-1])
