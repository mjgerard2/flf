import os
import h5py as hf
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from matplotlib.colors import LogNorm
from matplotlib.ticker import MultipleLocator

def plot_define(fontsize=14, labelsize=16, linewidth=1):
    plt.close('all')
    font = {'family': 'sans-serif',
            'serif': 'Times New Roman',
            'weight': 'normal',
            'size': fontsize}
    mpl.rc('font', **font)
    mpl.rcParams['axes.labelsize'] = labelsize
    mpl.rcParams['lines.linewidth'] = linewidth

# Define global path to where figure will be saved. If None, figure will be shown. #
save_path = None

# define path to Lyapunov data #
lyp_path = os.path.join(os.getcwd(), '20250219_0001.h5')
with hf.File(lyp_path, 'r') as hf_:
    lyp_exp = hf_['lyapunov exponents'][()]
    R_dom = hf_['R domain'][()]
    Z_dom = hf_['Z domain'][()]
    tor_ang = hf_['tor_ang'][()]

# plot data #
plot_define()
fig, ax = plt.subplots(1, 1, tight_layout=True)
ax.set_aspect('equal')

# define colorbar parameters #
ncolors = 51
cmap = plt.get_cmap('magma_r')
cmax = 1e2  # 1e1
cmin = 1e-2  # 3e-2
clr_norm = mpl.colors.LogNorm(vmin=cmin, vmax=cmax)
R_stp, Z_stp = 0.5*(R_dom[1]-R_dom[0]), 0.5*(Z_dom[1]-Z_dom[0])
extend = [R_dom[0]-R_stp, R_dom[-1]+R_stp, Z_dom[0]-Z_stp, Z_dom[-1]+Z_stp]
args = dict(cmap=cmap, norm=clr_norm, extent=extend, origin='lower', aspect='auto', interpolation='None')
smap = ax.imshow(np.abs(lyp_exp[:,:].T), **args)

# axis limits #
ax.set_xlim(ax.get_xlim())
ax.set_ylim(ax.get_ylim())

# axis labels #
ax.set_xlabel(r'$R \ / \ \mathrm{m}$')
ax.set_ylabel(r'$Z \ / \ \mathrm{m}$')
ax.set_title('Lyapunov exponent')
cbar = fig.colorbar(smap, ax=ax)
cbar.ax.set_ylabel(r'$\lambda_{\mathrm{Lyp}}$')

# tick marks #
ax.tick_params(axis='both', which='major', direction='out', length=5)
ax.tick_params(axis='both', which='minor', direction='out', length=2.5)
ax.xaxis.set_ticks_position('default')
ax.yaxis.set_ticks_position('default')
# ax.xaxis.set_minor_locator(MultipleLocator(1))
# ax.yaxis.set_minor_locator(MultipleLocator(1))

# save/show #
if save_path is None:
    plt.show()
else:
    plt.savefig(save_path, format=save_path.split('.')[-1])
