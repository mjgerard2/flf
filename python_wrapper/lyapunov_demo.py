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
    pid = os.getpid()
    file_dict = {'namelist': 'flf.namelist_%s' % pid,
                 'input': 'point_temp_%s.in' % pid}
    flf = flfc.flf_wrapper('HSX', file_dict=file_dict)
    flf.change_params(run_dict['mod_dict'])
    while True:
        item = in_que.get()
        if item is None:
            out_que.put(None)
            break
        else:
            run_idx, R_val = item
            lyp_exponent = np.empty(Z_dom.shape[0])
            for i, Z_val in enumerate(run_dict['Z_dom']):
                init_pnt = np.array([R_val, Z_val, run_dict['tor_ang']])
                lyp_exp = flf.calc_lyapunov_exponents(init_pnt, run_dict['d0'], run_dict['npts'], run_dict['dstp'], run_dict['rots'])
                if lyp_exp is None:
                    lyp_exponent[i] = np.nan
                else:
                    lyp_exponent[i] = lyp_exp
            print('({0:0.0f}|{1:0.0f})'.format(run_idx+1, run_dict['run_cnt']))
            out_que.put([run_idx, lyp_exponent])

# Number of CPUs to use #
num_of_cpus = cpu_count()-4
print('Number of CPUs: {0:0.0f}'.format(num_of_cpus))

# specify coil currents #
crnt = np.array([-10722.0, -10507.56, -10722.0, -10614.78, -10722.0, -10722.0, -15010.8, -15010.8, -15010.8, -15010.8, -15010.8, -15010.8])
mod_dict = {'mgrid_currents': ' '.join(['{}'.format(c) for c in crnt]),
            'mgrid_file': os.path.join('/mnt', 'HSX_Database', 'HSX_Configs', 'coil_data', 'mgrid_extnd_res5p0mm_30pln.nc')}

# specify calculation parameters #
run_dict = {'dx': 1e-3, # spatial separation of sample points in meters
            'tor_ang': 0.0*np.pi, # toroidal angle of cross section
            'dstp': .25, # angular step size in field line following
            'rots': 200, # number of toroidal rotations
            'npts': 3, # number of shifted points around the sample points
            'd0': 1.44e-3, # distance of shift for shifted points in meters
            'mod_dict': mod_dict}

# define computational domain #
R_min, R_max = 1.0, 1.5
Z_min, Z_max = -0.3, 0.3
Rpts = int(((R_max-R_min)/run_dict['dx'])+1)
Zpts = int(((Z_max-Z_min)/run_dict['dx'])+1)
R_dom = np.linspace(R_min, R_max, Rpts)
Z_dom = np.linspace(Z_min, Z_max, Zpts)
run_dict['Z_dom'] = Z_dom
run_dict['run_cnt'] = Rpts
print('R points: {}\nZ points: {}\n'.format(Rpts, Zpts))

# define run name #
lyp_dir = os.getcwd()
date_tag = datetime.now().strftime('%Y%m%d')
lyp_nums = [int(f.name.split('.')[0].split('_')[1]) for f in os.scandir(lyp_dir) if f.name.split('_')[0] == date_tag]
if len(lyp_nums) > 0:
    lyp_tag = str(max(lyp_nums)+1).zfill(4)
else:
    lyp_tag = '0001'
lyp_path = os.path.join(lyp_dir, '%s_%s.h5' % (date_tag, lyp_tag))

# generate data file #
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
