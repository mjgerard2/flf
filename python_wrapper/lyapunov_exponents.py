import os, sys
import subprocess

import h5py as hf
import numpy as np

from sklearn.linear_model import LinearRegression
from multiprocessing import Process, Queue, cpu_count
from datetime import datetime

ModDir = os.path.join('/home', 'michael', 'Desktop', 'flf', 'python_wrapper')
sys.path.append(ModDir)
import flf_class as flfc


def maximum_lyapunov_exponent(in_que, out_que, run_dict):
    dx = run_dict['dx']
    tor_ang = run_dict['tor_ang']
    dstp = run_dict['dstp']
    rots = run_dict['rots']
    npts = run_dict['npts']
    d0 = run_dict['d0']
    run_cnt = run_dict['run_cnt']
    mod_dict = run_dict['mod_dict']
    Z_dom = run_dict['Z_dom']

    pid = os.getpid()
    file_dict = {'namelist': 'flf.namelist_%s' % pid,
                 'input': 'point_temp_%s.in' % pid}

    flf = flfc.flf_wrapper('HSX', file_dict=file_dict)
    flf.set_transit_parameters(dstp, rots)
    dphi = flf.params['points_dphi']

    pnt_ang = np.linspace(0, 2*np.pi, npts, endpoint=False)
    # x_data = np.linspace(-rots, rots, 2*flf.params['n_iter']-1).reshape((-1, 1))

    while True:
        item = in_que.get()
        if item is None:
            out_que.put(None)
            break
        else:
            run_idx, R_val = item
            lyp_exponent = np.empty(Z_dom.shape)
            for i, Z_val in enumerate(Z_dom):
                init_pnt = np.array([R_val, Z_val, tor_ang])
                flf.change_params({'points_dphi': dphi})
                pnt_for = flf.execute_flf(init_pnt)
                flf.change_params({'points_dphi': -dphi})
                pnt_bak = flf.execute_flf(init_pnt)
                if (pnt_for is None) or (pnt_bak is None):
                    lyp_exponent[i] = np.nan
                    continue
                elif np.isnan(pnt_for).any() or np.isnan(pnt_bak).any():
                    not_nan = np.isnan(pnt_for[:,0])
                    pnt_for = pnt_for[~not_nan]
                    not_nan = np.isnan(pnt_bak[:,0])
                    pnt_bak = pnt_bak[~not_nan]
                    if pnt_for.shape[0] <= 2 or pnt_bak.shape[0] <= 2:
                        lyp_exponent[i] = np.nan
                        continue
                    phi_max = min(np.max(pnt_for[:,2]), np.max(np.abs(pnt_bak[:,2])))
                    pnt_for = pnt_for[pnt_for[:,2] <= phi_max]
                    pnt_bak = pnt_bak[pnt_bak[:,2] >= -phi_max]
                else:
                    phi_max = 2*np.pi*rots

                succ = True
                lyp_exp = np.empty(npts)
                flf.set_transit_parameters(dstp, phi_max/(2*np.pi))
                flf.change_params({'points_dphi': dphi})
                pnt_for = flf.execute_flf(init_pnt)
                flf.change_params({'points_dphi': -dphi})
                pnt_bak = flf.execute_flf(init_pnt)
                for k in range(npts):
                    x_data = np.linspace(-phi_max, phi_max, 2*flf.params['n_iter']-1).reshape((-1,1))
                    d_shft = d0*np.array([np.cos(pnt_ang[k]), np.sin(pnt_ang[k]), 0])
                    flf.change_params({'points_dphi': dphi})
                    pnts_for = flf.execute_flf(init_pnt+d_shft)
                    flf.change_params({'points_dphi': -dphi})
                    pnts_bak = flf.execute_flf(init_pnt+d_shft)
                    if (pnts_for is None) or (pnts_bak is None):
                        lyp_exponent[i] = np.nan
                        succ = False
                        break
                    elif np.isnan(pnts_for).any() or np.isnan(pnts_bak).any():
                        not_nan = np.isnan(pnts_for[:,0])
                        pnts_for = pnts_for[~not_nan]
                        not_nan = np.isnan(pnts_bak[:,0])
                        pnts_bak = pnts_bak[~not_nan]
                        phi_chk = min(np.max(pnts_for[:,2]), np.max(np.abs(pnts_bak[:,2])))+.5*dphi
                        pnts_for = pnts_for[pnts_for[:,2] <= phi_chk]
                        pnts_bak = pnts_bak[pnts_bak[:,2] >= -phi_chk]
                        if pnts_for.shape[0] <= 2 or pnts_bak.shape[0] <= 2:
                            lyp_exponent[i] = np.nan
                            continue
                        pnt_for_use = pnt_for[pnt_for[:,2] <= phi_chk]
                        pnt_bak_use = pnt_bak[pnt_bak[:,2] >= -phi_chk]
                        x_data = np.linspace(-phi_chk, phi_chk, 2*pnts_for.shape[0]-1).reshape((-1,1))
                    else:
                        pnt_for_use = pnt_for
                        pnt_bak_use = pnt_bak
                    dist_for = np.linalg.norm(pnt_for_use[:, 0:2] - pnts_for[:, 0:2], axis=1)/d0
                    dist_bak = np.linalg.norm(pnt_bak_use[1::, 0:2] - pnts_bak[1::, 0:2], axis=1)/d0
                    dist = np.r_[np.flip(dist_bak), dist_for]
                    zero_idx = np.where(dist == 0)[0]
                    dist[zero_idx] = 1e-10
                    y_data = np.log(dist)
                    model = LinearRegression().fit(x_data, y_data)
                    lyp_exp[k] = model.coef_[0]
                if succ:
                    lyp_exponent[i] = np.mean(lyp_exp)

            print('({0:0.0f}|{1:0.0f})'.format(run_idx+1, run_cnt))
            out_que.put([run_idx, lyp_exponent])

    for key, name in file_dict.items():
        file_path = os.path.join('/home', 'michael', name)
        subprocess.run(['rm', file_path])

# Number of CPUs to use #
num_of_cpus = 1 # cpu_count()-1
print('Number of CPUs: {0:0.0f}'.format(num_of_cpus))

config_id = '0-1-0'
main = np.ones(6)
aux = np.zeros(6)
crnt = -10722. * np.r_[main, 14*aux]
mod_dict = {'mgrid_currents': ' '.join(['{}'.format(c) for c in crnt]), 
            'mgrid_file': os.path.join('/mnt', 'HSX_Database', 'HSX_Configs', 'coil_data', 'mgrid_res1p0mm_30pln.nc')}

date_tag = datetime.now().strftime('%Y%m%d')
run_dict = {'dx': 5e-3, # spatial separation of sample points in meters
            'tor_ang': 0, # toroidal angle of cross section
            'dstp': 5, # angular step size in field line following
            'rots': 4, # number of toroidal rotations
            'npts': 3, # number of shifted points around the sample points
            'd0': 1e-3, # distance of shift for shifted points in meters
            'mod_dict': mod_dict}

# get search domain from vessel wall #
vess_path = os.path.join('/mnt', 'HSX_Database', 'HSX_Configs', 'coil_data', 'vessel90.h5')
with hf.File(vess_path, 'r') as hf_:
    vess_data = hf_['data'][()]
    vess_dom = hf_['domain'][()]

idx = np.argmin(np.abs(vess_dom - run_dict['tor_ang']))
R_vess = np.linalg.norm(vess_data[idx][:, 0:2], axis=1)
Z_vess = vess_data[idx][:, 2]

R_min, R_max = np.min(R_vess), np.max(R_vess)
Z_min, Z_max = np.min(Z_vess), np.max(Z_vess)

Rpts = int(((R_max-R_min)/run_dict['dx'])+1)
Zpts = int(((Z_max-Z_min)/run_dict['dx'])+1)
run_dict['run_cnt'] = Rpts
print('R points: {}\nZ points: {}\n'.format(Rpts, Zpts))

R_dom = np.linspace(R_min, R_max, Rpts)
Z_dom = np.linspace(Z_min, Z_max, Zpts)
run_dict['Z_dom'] = Z_dom

# save run data #
main_id = 'main_coil_{}'.format(config_id.split('-')[0])
set_id = 'set_{}'.format(config_id.split('-')[1])
job_id = 'job_{}'.format(config_id.split('-')[2])
lyp_dir = os.path.join('/mnt', 'HSX_Database', 'HSX_Configs', main_id, set_id, job_id, 'lyapunov_data')
lyp_nums = [int(f.name.split('.')[0].split('_')[1]) for f in os.scandir(lyp_dir) if f.name.split('_')[0] == date_tag]
if len(lyp_nums) > 0:
    lyp_tag = str(max(lyp_nums)+1).zfill(4)
else:
    lyp_tag = '0001'
lyp_path = os.path.join(lyp_dir, '%s_%s.h5' % (date_tag, lyp_tag))

if os.path.isfile(lyp_path):
    with hf.File(lyp_path, 'r') as hf_:
        lyp_exponent = hf_['lyapunov exponents'][()]
else:
    lyp_exponent = np.full((Rpts, Zpts), np.nan)
    with hf.File(lyp_path, 'w') as hf_:
        hf_.create_dataset('lyapunov exponents', data=lyp_exponent)
        hf_.create_dataset('R domain', data=R_dom)
        hf_.create_dataset('Z domain', data=Z_dom)
        hf_.create_dataset('dx', data=run_dict['dx'])
        hf_.create_dataset('dstp', data=run_dict['dstp'])
        hf_.create_dataset('tor_ang', data=run_dict['tor_ang'])
        hf_.create_dataset('rots', data=run_dict['rots'])
        hf_.create_dataset('npts', data=run_dict['npts'])
        hf_.create_dataset('d0', data=run_dict['d0'])

# populate queue #
in_que = Queue()
out_que = Queue()
for R_idx, R_val in enumerate(R_dom):
    if np.isnan(lyp_exponent[R_idx, :]).all():
        in_que.put([R_idx, R_val])

for i in range(num_of_cpus):
    in_que.put(None)

# start exponent calculation processes #
for i in range(num_of_cpus):
    proc = Process(target=maximum_lyapunov_exponent, args=(in_que, out_que, run_dict))
    proc.start()

# save exponents #
done_cnt = 0
while True:
    item = out_que.get()
    if item is None:
        done_cnt+=1
        print('End Process: {0:0.0f} of {1:0.0f}'.format(done_cnt, num_of_cpus))
        if done_cnt == num_of_cpus:
            break
    else:
        R_idx, lyp_exp = item
        with hf.File(lyp_path, 'a') as hf_:
            lyp_exponent = hf_['lyapunov exponents'][()]
            lyp_exponent[R_idx] = lyp_exp
            del hf_['lyapunov exponents']
            hf_.create_dataset('lyapunov exponents', data=lyp_exponent)
