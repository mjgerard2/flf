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

    dphi = dstp * (np.pi/180)
    stps = round(2 * np.pi / dphi)
    nitr = round(rots*stps)

    pid = os.getpid()
    file_dict = {'namelist': 'flf.namelist_%s' % pid,
                 'input': 'point_temp_%s.in' % pid}

    flf = flfc.flf_wrapper('HSX', file_dict=file_dict)

    pnt_ang = np.linspace(0, 2*np.pi, nitr, endpoint=False)

    while True:
        item = in_que.get()
        if item is None:
            out_que.put(None)
            break
        else:
            run_idx, R_val = item
            lyp_exponent = np.empty((Z_dom.shape[0], 2))
            for i, Z_val in enumerate(Z_dom):
                init_pnt = np.array([R_val, Z_val, tor_ang])
                lyp_exp = flf.calc_lyapunov_exponents(init_pnt, d0, npts, dstp, rots)
                print(lyp_exp)
                if lyp_exp is None:
                    lyp_exponent[i] = [np.nan, np.nan]
                else:
                    lyp_exponent[i] = lyp_exp
                """
                mod_dict['n_iter'] = nitr
                mod_dict['points_dphi'] = dphi
                flf.change_params(mod_dict)
                pnt_for = flf.execute_flf(init_pnt)
                mod_dict['points_dphi'] = -dphi
                flf.change_params(mod_dict)
                pnt_bak = flf.execute_flf(init_pnt)
                if (pnt_for is None) or (pnt_bak is None):
                    lyp_exponent[i] = [np.nan, np.nan]
                    continue
                elif np.isnan(pnt_for).any() or np.isnan(pnt_bak).any():
                    not_nan = np.isnan(pnt_for[:,0])
                    pnt_for = pnt_for[~not_nan]
                    not_nan = np.isnan(pnt_bak[:,0])
                    pnt_bak = pnt_bak[~not_nan]
                    if pnt_for.shape[0] <= 2 or pnt_bak.shape[0] <= 2:
                        lyp_exponent[i] = [np.nan, np.nan]
                        continue
                    phi_max = min(np.max(pnt_for[:,2]-tor_ang), np.max(np.abs(tor_ang-pnt_bak[:,2])))
                    pnt_for = pnt_for[pnt_for[:,2] <= tor_ang+phi_max+.5*dphi]
                    pnt_bak = pnt_bak[pnt_bak[:,2] >= tor_ang-phi_max-.5*dphi]
                else:
                    phi_max = 2*np.pi*rots

                succ = True
                lyp_exp = np.empty((npts, 2))
                mod_dict['points_dphi'] = dphi
                mod_dict['n_iter'] = round(phi_max/dphi)
                flf.change_params(mod_dict)
                for k in range(npts):
                    x_data = np.linspace(0, phi_max, pnt_for.shape[0]).reshape((-1,1))
                    d_shft = d0*np.array([np.cos(pnt_ang[k]), np.sin(pnt_ang[k]), 0])
                    pnts_for = flf.execute_flf(init_pnt+d_shft)
                    mod_dict['points_dphi'] = -dphi
                    flf.change_params(mod_dict)
                    pnts_bak = flf.execute_flf(init_pnt+d_shft)
                    if (pnts_for is None) or (pnts_bak is None):
                        lyp_exponent[i] = [np.nan, np.nan]
                        succ = False
                        break
                    elif np.isnan(pnts_for).any() or np.isnan(pnts_bak).any():
                        not_nan = np.isnan(pnts_for[:,0])
                        pnts_for = pnts_for[~not_nan]
                        not_nan = np.isnan(pnts_bak[:,0])
                        pnts_bak = pnts_bak[~not_nan]
                        phi_chk = min(np.max(pnts_for[:,2]-tor_ang), np.max(np.abs(tor_ang-pnts_bak[:,2])))
                        pnts_for = pnts_for[pnts_for[:,2] <= tor_ang+phi_chk+.5*dphi]
                        pnts_bak = pnts_bak[pnts_bak[:,2] >= tor_ang-phi_chk-.5*dphi]
                        if pnts_for.shape[0] <= 2 or pnts_bak.shape[0] <= 2:
                            lyp_exponent[i] = [np.nan, np.nan]
                            continue
                        pnt_for_use = pnt_for[pnt_for[:,2] <= tor_ang+phi_chk+.5*dphi]
                        pnt_bak_use = pnt_bak[pnt_bak[:,2] >= tor_ang-phi_chk-.5*dphi]
                        x_data = np.linspace(0, phi_chk, pnts_for.shape[0]).reshape((-1,1))
                    else:
                        pnt_for_use = pnt_for
                        pnt_bak_use = pnt_bak

                    dist_for = np.linalg.norm(pnt_for_use[:, 0:2] - pnts_for[:, 0:2], axis=1)/d0
                    zero_for = np.where(dist_for == 0)[0]
                    dist_for[zero_for] = 1e-10
                    model_for = LinearRegression().fit(x_data, np.log(dist_for))

                    dist_bak = np.linalg.norm(pnt_bak_use[:, 0:2] - pnts_bak[:, 0:2], axis=1)/d0
                    zero_bak = np.where(dist_bak == 0)[0]
                    dist_bak[zero_bak] = 1e-10
                    model_bak = LinearRegression().fit(x_data, np.log(dist_bak))
                    lyp_exp[k] = [model_for.coef_[0], model_bak.coef_[0]]
                
                if succ:
                    lyp_exponent[i] = np.mean(lyp_exp, axis=0)
                """
            print('({0:0.0f}|{1:0.0f})'.format(run_idx+1, run_cnt))
            out_que.put([run_idx, lyp_exponent])


# Number of CPUs to use #
num_of_cpus = 1 #cpu_count()-1
print('Number of CPUs: {0:0.0f}'.format(num_of_cpus))

config_id = '0-1-0'
main = np.ones(6)
aux = np.zeros(6)
crnt = -10722. * np.r_[main, 14*aux]
mod_dict = {'mgrid_currents': ' '.join(['{}'.format(c) for c in crnt]), 
            'mgrid_file': os.path.join('/mnt', 'HSX_Database', 'HSX_Configs', 'coil_data', 'mgrid_hsx_wmain.nc')}

date_tag = 'test' # datetime.now().strftime('%Y%m%d')
run_dict = {'dx': 1e-2, # spatial separation of sample points in meters
            'tor_ang': 0.*np.pi, # toroidal angle of cross section
            'dstp': 5, # angular step size in field line following
            'rots': 5, # number of toroidal rotations
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
    lyp_exponent = np.full((Rpts, Zpts, 2), np.nan)
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
