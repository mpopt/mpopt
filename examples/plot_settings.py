from mpopt import mp

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
# mp.plt.tight_layout()

mp.plt.savefig(f"plots/vdp_lgr_s{seg}p{p}.eps")
