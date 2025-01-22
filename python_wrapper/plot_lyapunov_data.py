import os
import h5py as hf
import numpy as np
import matplotlib as mpl

from matplotlib.colors import LogNorm
from matplotlib.ticker import MultipleLocator

import os, sys
WORKDIR = os.path.join('/home', 'michael', 'Desktop', 'python_repos', 'turbulence-optimization')
sys.path.append(WORKDIR)

import plot_define as pd
import flfTools.flf_operations as flfo


def plot_vessel(vess_path, ax, tor_ang):
    with open(vess_path, 'r') as f:
        lines = f.readlines()
        line_strip = lines[0].strip().split()
        tor_pnts = int(line_strip[0])
        pol_pnts = int(line_strip[1])

        R_pnts = np.empty((tor_pnts, pol_pnts))
        Z_pnts = np.empty((tor_pnts, pol_pnts))
        phi_pnts = np.empty((tor_pnts, pol_pnts))
        vess_pnts = np.empty((tor_pnts, pol_pnts, 3))
        for i in range(tor_pnts):
            for j in range(pol_pnts):
                idx = 1 + i*pol_pnts + j
                arr = np.array([float(x) for x in lines[idx].strip().split()])
                R_pnts[i,j] = np.linalg.norm(arr[0:2])
                Z_pnts[i,j] = arr[2]
                phi_pnts[i,j] = np.arctan2(arr[1], arr[0])

    tor_dom = np.mean(phi_pnts, axis=1)
    idx = np.argmin(np.abs(tor_dom-tor_ang))
    ax.plot(R_pnts[idx,:]*1e2, Z_pnts[idx,:]*1e2, c='w', linewidth=4)

def plot_poincare_surface(poin_path, ax, tor_ang):
    poinData = hf.File(poin_path, 'r')

    key = 'core'
    core = poinData[key][:]
    t_dom = core[0, :, 2]
    dt = t_dom[1] - t_dom[0]

    npts = t_dom.shape[0]
    stps = int(round(2 * np.pi / dt))
    rots = int(round(t_dom[int(npts-1)] / (2*np.pi)))
    idx_stps = np.argmin(np.abs(t_dom - t_dom[0] - tor_ang)) + [int(i*stps) for i in range(rots)]

    points = np.empty((idx_stps.shape[0], 2))
    for i, idx in enumerate(idx_stps):
        points[i] = core[-5,idx,0:2]

    ax.scatter(points[:,0], points[:,1], c='w', s=1)

    # plot vessel #
    path = os.path.join('/mnt', 'HSX_Database', 'HSX_Configs', 'coil_data', 'vessel90.h5')
    with hf.File(path,'r') as hf_file:
        vessel = hf_file['data'][:]
        v_dom = hf_file['domain'][:]

    idx = np.argmin(np.abs(v_dom - tor_ang))

    ves_x = vessel[idx,0::,0]
    ves_y = vessel[idx,0::,1]
    ves_z = vessel[idx,0::,2]
    ves_r = np.hypot(ves_x, ves_y)

    ax.plot(ves_r, ves_z, c='w', lw=3)
"""
config_id = '0-1-0'
main_id = 'main_coil_{}'.format(config_id.split('-')[0])
set_id = 'set_{}'.format(config_id.split('-')[1])
job_id = 'job_{}'.format(config_id.split('-')[2])
base_path = os.path.join('/mnt', 'HSX_Database', 'HSX_Configs', main_id, set_id, job_id)

# read in lyapunov data #
# lyp_tag = '20240408_0001'
lyp_tag = '20231017_0001'
lyp_path = os.path.join(base_path, 'lyapunov_data', '%s.h5' % lyp_tag)
"""
lyp_path = os.path.join('/home', 'michael', 'Desktop', 'for_dieter', 'large_island_lyapunov_tor0p00.h5')
with hf.File(lyp_path, 'r') as hf_:
    lyp_exp = hf_['lyapunov exponents'][()]
    R_dom = 1e2*hf_['R domain'][()]
    Z_dom = 1e2*hf_['Z domain'][()]
    tor_ang = hf_['tor_ang'][()]

# define poincare path #
"""
if config_id == '0-1-0':
    poin_path = os.path.join(base_path, 'poincare.h5')
else:
    # poin_path = os.path.join(base_path, '%s.h5' % config_id)
    poin_path = os.path.join(base_path, 'poincare_500pnts_50surf.h5')
"""
# plot data #
plot = pd.plot_define(fontSize=14, labelSize=16)
plt = plot.plt
fig, ax = plt.subplots(1, 1, tight_layout=True, figsize=(5.5, 6))
# fig, ax = plt.subplots(1, 1, tight_layout=True, figsize=(8, 6))
ax.set_aspect('equal')

ncolors = 51
cmap = plt.get_cmap('magma_r')
cmax = 1e2  # 1e1
cmin = 1e-2  # 3e-2

clr_norm = mpl.colors.LogNorm(vmin=cmin, vmax=cmax)
# bounds = np.linspace(0, 1, ncolors+1)
# clr_norm = mpl.colors.BoundaryNorm(bounds, ncolors=cmap.N)
R_stp, Z_stp = R_dom[1]-R_dom[0], Z_dom[1]-Z_dom[0]
extend = [R_dom[0]-0.5*R_stp, R_dom[-1]+0.5*R_stp, Z_dom[0]-0.5*Z_stp, Z_dom[-1]+0.5*Z_stp]
args = dict(cmap=cmap, norm=clr_norm, extent=extend, origin='lower', aspect='auto', interpolation='None')
# args = dict(cmap=cmap, vmin=0, vmax=0.5, extent=extend, origin='lower', aspect='auto', interpolation='None')
smap = ax.imshow(np.abs(lyp_exp[:,:].T), **args)
# smap = ax.pcolormesh(R_dom*1e2, Z_dom*1e2, np.abs(lyp_exp[:,:].T), cmap=cmap, norm=LogNorm(vmin=cmin, vmax=cmax))
# smap = ax.pcolormesh(R_dom*1e2, Z_dom*1e2, lyp_exp[:,:].T, cmap=cmap, vmin=0, vmax=1.)
# ax.plot([136.95, 136.95, 136.8, 136.6, 136.5, 136.4, 136.3], [23, 24, 25, 26, 27, 28, 29], c='w', ls='--', lw=2)
# ax.plot([137.2, 137, 136.8, 136.75, 136.85], [26, 27, 28, 29, 29.6], c='w', ls='--', lw=2)
# ax.plot([138, 137, 136, 135.41], [28.2, 28.5, 28.8, 28.95], c='w', ls='--', lw=2)
# ax.plot([140, 139, 138, 137, 136, 135, 134.7], [25.2, 25.6, 26.1, 26.5, 27, 27.3, 27.5], c='w', ls='--', lw=2)
# ax.plot([135.5, 135, 134.5], [26, 26.5, 28], c='w', ls='--', lw=2)
# ax.plot([137.5, 137, 136.5, 136, 135.5, 135], [25.75, 26, 26.25, 26.5, 26.75, 26.9], c='w', ls='--', lw=2)

# axis limits #
ax.set_xlim(ax.get_xlim())
ax.set_ylim(ax.get_ylim())

# flfo.plot_poincare(poin_path, tor_ang, fig, ax, show_vessel=False)
# plot_poincare_surface(poin_path, ax, tor_ang)
vess_path = os.path.join('/mnt', 'HSX_Database', 'HSX_Configs', 'coil_data', 'vessel_high_res_RF.txt')
plot_vessel(vess_path, ax, tor_ang)

# axis labels #
ax.set_xlabel('Major Radius [cm]')
ax.set_ylabel('Vertical Distance [cm]')
ax.set_title('Lyapunov exponents')
cbar = fig.colorbar(smap, ax=ax)
cbar.ax.set_ylabel(r'$\lambda_{\mathrm{lyp}}$')

# tick marks #
ax.tick_params(axis='both', which='major', direction='out', length=5)
ax.tick_params(axis='both', which='minor', direction='out', length=2.5)
ax.xaxis.set_ticks_position('default')
ax.yaxis.set_ticks_position('default')
ax.xaxis.set_minor_locator(MultipleLocator(1))
ax.yaxis.set_minor_locator(MultipleLocator(1))

# ax.set_xlim(ax.get_xlim()[0], 1.6)

# plt.show()
# save_path = os.path.join(base_path, 'lyapunov_data', '%s.png' % lyp_tag)
save_path = os.path.join('/home', 'michael', 'Desktop', 'for_dieter', 'large_island_lyapunov_tor0p00.pdf')
plt.savefig(save_path, format='pdf')
