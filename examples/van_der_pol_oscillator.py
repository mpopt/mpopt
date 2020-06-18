"""Van der Pol oscilator OCP from https://web.casadi.org/docs/
"""
from context import mpopt
from mpopt import mp

ocp = mp.OCP(n_states=2, n_controls=1)


def dynamics(x, u, t):
    return [(1 - x[1] * x[1]) * x[0] - x[1] + u[0], x[0]]


def running_cost(x, u, t):
    return x[0] * x[0] + x[1] * x[1] + u[0] * u[0]


ocp.dynamics[0] = dynamics
ocp.running_costs[0] = running_cost

ocp.x00[0] = [0, 1]
ocp.lbu[0] = -1.0
ocp.ubu[0] = 1.0
ocp.lbx[0][1] = -0.25
ocp.lbtf[0] = 10.0
ocp.ubtf[0] = 10.0

ocp.validate()
mp.post_process._INTERPOLATION_NODES_PER_SEG = 200

seg, p = 1, 15
mpo, lgr = mp.solve(ocp, seg, p, "LGR", plot=False)
mpo, lgl = mp.solve(ocp, seg, p, "LGL", plot=False)
mpo, cgl = mp.solve(ocp, seg, p, "CGL", plot=False)

fig, axs = lgr.plot_phases(name="LGR")
fig, axs = lgl.plot_phases(fig=fig, axs=axs, name="LGL")
fig, axs = cgl.plot_phases(fig=fig, axs=axs, name="CGL")
mp.plt.title(
    f"non-adaptive solution segments = {mpo.n_segments} poly={mpo.poly_orders[0]}"
)

mph = mp.mpopt_h_adaptive(ocp, 5, 5)
solh = mph.solve(max_iter=10, mpopt_options={"method": "control_slope"})
posth = mph.process_results(solh, plot=False)
fig, axs = posth.plot_phases(fig=None, axs=None)
mp.plt.title(f"Adaptive solution segments = {mph.n_segments} poly={mph.poly_orders[0]}")
mp.plt.show()
