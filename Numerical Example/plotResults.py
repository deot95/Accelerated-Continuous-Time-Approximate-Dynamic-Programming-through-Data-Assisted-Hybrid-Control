#%%
############################################
### Packages
############################################
from os import times
import pathlib
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import (zoomed_inset_axes, mark_inset)
from matplotlib.lines import Line2D
import h5py
from numpy.core.numeric import zeros_like

parentDir = pathlib.Path(__file__).parent.absolute().as_posix()
figsDir = f"{parentDir}/Figs"
pathlib.Path(figsDir).mkdir(parents=True, exist_ok=True)
#plt.style.use(["science","ieee", "grid"])

def my_rc_params(magnifier=1):
    plt.style.use("seaborn-deep")
    plt.rcParams["axes.labelsize"] = 12*magnifier
    plt.rcParams["xtick.labelsize"] = 10*magnifier
    plt.rcParams["ytick.labelsize"] = 10*magnifier
    plt.rcParams["legend.fontsize"] = 12*magnifier
    plt.rcParams["lines.linewidth"] = 4
    plt.rcParams["figure.figsize"] = 12, 6

    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Noto Serif"]
    #Ticks
    plt.rcParams["xtick.direction"] = "in"
    plt.rcParams["xtick.major.size"] = 3
    plt.rcParams["xtick.major.width"] = 0.5
    plt.rcParams["xtick.minor.size"] = 1.5
    plt.rcParams["xtick.minor.width"] = 0.5

    plt.rcParams["ytick.direction"] = "in"
    plt.rcParams["ytick.major.size"] = 3
    plt.rcParams["ytick.major.width"] = 0.5
    plt.rcParams["ytick.minor.size"] = 1.5
    plt.rcParams["ytick.minor.width"] = 0.5
    #Lws
    plt.rcParams["axes.linewidth"] = 0.5
    plt.rcParams["grid.linewidth"] = 0.5

    plt.rcParams["text.usetex"] = True
    plt.rcParams["figure.dpi"] = 600
    plt.rcParams['agg.path.chunksize'] = 10000


    plt.rcParams["text.latex.preamble"] = r"\usepackage{amsmath, mathtools}"
my_rc_params(3.3)

# %%
############################################
### Data loading
############################################
sufix = "AltParam"
fname = f"{parentDir}/Data/results{sufix}.hdf5"
data = h5py.File(fname, "r")
timeD = data["timeD"][...]
arcGradient = data["arcGradient"][...]
arcHM = data["arcHM"][...]
arcFree = data["arcFree"][...]
##########################################
#### Data
##########################################
# %%
fig, axarr = plt.subplots(1,2, figsize=(20,8))

boxstyle = dict(boxstyle="round",
                    ec="black",
                    fc="white",
                    linewidth=0.4
                    )

prop_cycle = plt.rcParams["axes.prop_cycle"]
colors = prop_cycle.by_key()["color"]
c1 = colors[0]
c2 = colors[1]
c3 = colors[2]
c4 = "#263238"
c5 = "#4a148c"
sample = 5000
thetacStar = np.array([1,0.5,0])
n = 2
lc = 3
lu = 3
m = 1

xFree = arcFree[:,:n]

xGradient = arcGradient[:, :n]
thetacGradient = arcGradient[:, n:n+lc]
thetauGradient = arcGradient[:, n+lc:n+lc+lu]

xHM = arcHM[:, :n]
thetacHM = arcHM[:, n:n+lc]
pHM = arcHM[:,n+lc:n+2*lc]
tauHM = arcHM[:, n+2*lc:n+2*lc+1]
thetauHM = arcHM[:, n+2*lc+1:n+2*lc+1+lu]

#Thetac
ax = axarr[0]
ax.plot(timeD, thetacGradient, c=c1, zorder=3)
ax.plot(timeD, thetacHM, c=c2, zorder=3)
for theta in thetacStar:
    ax.axhline(theta, ls="--", c=c4, zorder = 2)
ax.grid()
ax.set_xlabel(r"Time", fontsize=40)
ax.annotate(r"$\theta_c$", xy=(0.9, 0.8), xycoords=ax.transAxes,
                  va="center", ha="center", fontsize=50,
                  bbox=boxstyle)

#Thetau
axActor = axarr[1]
axActor.plot(timeD, thetauGradient, c=c1, zorder=3)
axActor.plot(timeD, thetauHM, c=c2, zorder=3)
for theta in thetacStar:
    axActor.axhline(theta, ls="--", c=c4, zorder = 2)
axActor.grid()
axActor.set_xlabel(r"Time", fontsize=40)
axActor.annotate(r"$\theta_u$", xy=(0.9, 0.8), xycoords=axActor.transAxes,
                  va="center", ha="center", fontsize=50,
                  bbox=boxstyle)

custom_lines = [Line2D([0], [0], color=c1, lw=6),
                Line2D([0], [0], color=c2, lw=6),
                Line2D([0], [0], color=c4, ls="--", lw=4)]
axActor.legend(custom_lines,
        [r"Critic Gradient Update", r"Critic Momentum-Based Update", r"$\theta_c^*$"],
        loc='lower left', ncol= 1, columnspacing=0.15,
        handlelength=1, fontsize = 30, bbox_to_anchor=(0.035, -0.012),
        handletextpad=0.5
        )

#Plant States
"""
axplant = axarr[2]
axplant.plot(xGradient[:,0], xGradient[:,1], c=c1, markevery=350, zorder = 5)
axplant.plot(xHM[:,0], xHM[:,1], c=c2, marker ="s", markevery=450, ms=6)
axplant.plot(xFree[:,0], xFree[:,1], c=c3, markevery=300, ms=10)

axplant.set_xlabel("Time")
axplant.set_ylabel("$x$")
axplant.grid()

custom_lines = [Line2D([0], [0], color=c1, lw=3),
                Line2D([0], [0], color=c2, lw=3),
                Line2D([0], [0], color=c4, lw=3)]
leg= ax.legend(custom_lines,
        [r"Gradient Training", r"Momentum Based Training",
             r"$\theta_c^*$"], title=r"$\theta_c(t)$",
        loc='lower left', ncol= 1,
        bbox_to_anchor=(0.275, 0.53), fontsize=18
        )

        
plt.setp(leg.get_title(),fontsize=18)

custom_lines = [Line2D([0], [0], color=c1, lw=3),
                Line2D([0], [0], color=c2, lw=3),
                Line2D([0], [0], color=c3, lw=3)]
axplant.legend(custom_lines,
        [r"Gradient Training", r"Momentum Based Training"],
        loc='lower left', ncol= 1, columnspacing=0.25,
        handlelength=1, fontsize = 22, bbox_to_anchor=(0.2, 0.01)
        )
"""        

plt.savefig(f"{figsDir}/results.pdf", bbox_inches="tight") 

# %%
fig = plt.figure(figsize=(6,6))
axPlant = plt.gca()
axPlant.plot(xGradient[:,0], xGradient[:,1], c=c1, marker ="o", markevery=350, ms=10, lw=3)
axPlant.plot(xHM[:,0], xHM[:,1], c=c2, marker ="s", markevery=300, ms=10, lw=3)
axPlant.plot(xFree[:,0], xFree[:,1], c=c3, markevery=300, ms=10, lw=3)

axPlant.set_xlabel("$x_1$")
axPlant.set_ylabel("$x_2$")
axPlant.grid()

axnorm = axPlant.inset_axes([0.4, 0.4, 0.45, 0.45])
normXGradient = np.sqrt(np.sum(xGradient**2, axis=1))
normXHM = np.sqrt(np.sum(xHM**2, axis=1))
axnorm.semilogy(timeD, normXGradient, c=c1)
axnorm.semilogy(timeD, normXHM, c=c2)
axnorm.set_xlabel("Time")


# %%
