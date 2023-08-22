import warnings
import numpy as np
import h5py as hf
import shlex
import subprocess

import matplotlib.path as mpltPath
from scipy.stats import chisquare

from sklearn.linear_model import LinearRegression

import os, sys
WORKDIR = os.path.join('/home', 'michael', 'Desktop', 'python_repos', 'turbulence-optimization', 'pythonTools')
sys.path.append(WORKDIR)

from databaseTools import functions
from vmecTools.wout_files import wout_read
from flfTools.exp_params import params
import fft_class

import plot_define as pd
import directory_dict as dd


class flf_wrapper:
    """ A wrapper class for the field line following fortran code titled flf.

    ...

    Attributes
    ----------
    exp : str
        Stellarator experiment acronym, used to import the default flf inputs.

    Methods
    -------
    change_params(chg_dict)
        Change flf.namelist variables.

    set_transit_parameters(self, dstp, rots)
        Define the toroidal transit parameters.

    execute_flf(init_point, quiet=False)
        Execcultes field line following code on specified input points.

    read_wout()
        Imports wout file corresponding to the current profile.

    read_out_along_line(tor_ang, ro_ang, ro_lim, nsurf=3, plt=None, return_data=False):
        Makes Poincare plot with initial points generated from a line
        extending out from the magnetic axis to the specified radial domain.

    read_out_point(init_point, plt=None, return_poin_data=False):
        Makes Poincare plot by following field line found at input point.

    read_out_domain(pnt1, pnt2, nsurf, plt=None, return_data=False):
        Makes Poincare plot with initial points generated from a line
        extending between two specified points.

    read_out_grid_sample(center_point, half_width, npts, plt=None, return_data=False, quiet=True)
        Makes Poincare plot with initial points generated as a uniform set
        from the square centered around a point.

    fit_surf(init_point, r_modes=3, z_modes=3, plt=None):
        Performs a Fourier fit to the poincare plot points in the toroidal
        domain of the initialized point.

    find_magnetic_axis(init_point, dec)
        Approximately locate the Magnetic Axis, represented as (r,z,t)
        points initialized in the flf code.

    find_boundaries(init_point, dec):
        Approximately locate the LCFS and Magnetic Axis, represented as
        (r,z,t) points initialized in the flf code.

    save_read_out_domain(data_key, pnt1, pnt2, nsurf, fileName='./poincare_set.h5'):
        Save field line coordinate data for the specified number of field
        lines, with each field line initialized along a line between two
        points.

    save_read_out_line(data_key, init_point, fileName='./poincare_set.h5'):
        Save single field line coordinate data.

    save_Bvec_data(data_key, fileName='./poincare_set.h5'):
        Save magnetic field vectors at the cylidrical points specified in
        the imported grid array.
    """

    def __init__(self, exp, file_dict=None):
        if file_dict is None:
            file_dict = {'namelist': 'flf.namelist',
                         'input': 'point_temp.in',
                         'output': 'point_temp.out',
                         'bash': 'run_flf.sh'}

        # define directories #
        self.flf_dir = os.path.join('/home', 'michael', 'Desktop', 'flf')
        self.wrapper_dir = os.path.join(self.flf_dir, 'python_wrapper')

        # define executable and file names #
        self.exe = os.path.join(self.flf_dir, 'flf')
        self.namelist = os.path.join(self.wrapper_dir, file_dict['namelist'])
        self.in_path = os.path.join(self.wrapper_dir, file_dict['input'])
        self.run_cmd = shlex.split('%s %s' % (self.exe, self.namelist))
        
        self.exp = exp
        self.params = params[exp]

        with open(self.namelist, 'w') as file:
            file.write('&flf\n' +
                       '  general_option = {}\n'.format(self.params['general_option']) +
                       '  points_file= \'{}\'\n'.format(self.in_path) +
                       '  points_number={}\n'.format(self.params['points_number']) +
                       '  follow_type={}\n'.format(self.params['follow_type']) +
                       '  points_dphi= {}\n'.format(self.params['points_dphi']) +
                       '  n_iter = {}\n'.format(self.params['n_iter']-1) +
                       '  output_coils = {}\n'.format(self.params['output_coils']) +
                       '  log_freq = {}\n'.format(self.params['log_freq']) +
                       '\n' +
                       '  num_main_coils = {}\n'.format(self.params['num_main_coils']) +
                       '  field_type = \'netcdf\'\n' +
                       '  mgrid_file = \'{}\'\n'.format(self.params['mgrid_file']) +
                       '  num_periods = {}\n'.format(self.params['num_periods']) +
                       '  is_mirrored = {}\n'.format(self.params['is_mirrored']) +
                       '  skip_value = {}\n'.format(self.params['skip_value']) +
                       '  num_aux_coils = {}\n'.format(self.params['num_aux_coils']) +
                       '\n' +
                       '  use_diffusion = {}\n'.format(self.params['use_diffusion']) +
                       '  d_perp = {}\n'.format(self.params['d_perp']) +
                       '  temperature = {}\n'.format(self.params['temperature']) +
                       '\n' +
                       '  num_divertors = {}\n'.format(self.params['num_divertors']) +
                       '  num_limiters = {}\n'.format(self.params['num_limiters']) +
                       '/\n' +
                       '&mgrid\n' +
                       '  mgrid_currents = {}\n'.format(self.params['mgrid_currents']) +
                       '/')

    def change_params(self, chg_dict):
        """ Change flf.namelist variables.

        Parameters
        ----------
        chg_dict : dict
            Dictionary of flf.namelist variables to be changed.
        """
        for key in chg_dict:
            self.params[key] = chg_dict[key]

        with open(self.namelist, 'w') as file:
            file.write('&flf\n' +
                       '  general_option = {}\n'.format(self.params['general_option']) +
                       '  points_file= \'{}\'\n'.format(self.in_path) +
                       '  points_number={}\n'.format(self.params['points_number']) +
                       '  follow_type={}\n'.format(self.params['follow_type']) +
                       '  points_dphi= {}\n'.format(self.params['points_dphi']) +
                       '  n_iter = {}\n'.format(self.params['n_iter']-1) +
                       '  output_coils = {}\n'.format(self.params['output_coils']) +
                       '  log_freq = {}\n'.format(self.params['log_freq']) +
                       '\n' +
                       '  num_main_coils = {}\n'.format(self.params['num_main_coils']) +
                       '  field_type = \'netcdf\'\n' +
                       '  mgrid_file = \'{}\'\n'.format(self.params['mgrid_file']) +
                       '  num_periods = {}\n'.format(self.params['num_periods']) +
                       '  is_mirrored = {}\n'.format(self.params['is_mirrored']) +
                       '  skip_value = {}\n'.format(self.params['skip_value']) +
                       '  num_aux_coils = {}\n'.format(self.params['num_aux_coils']) +
                       '\n' +
                       '  use_diffusion = {}\n'.format(self.params['use_diffusion']) +
                       '  d_perp = {}\n'.format(self.params['d_perp']) +
                       '  temperature = {}\n'.format(self.params['temperature']) +
                       '\n' +
                       '  num_divertors = {}\n'.format(self.params['num_divertors']) +
                       '  num_limiters = {}\n'.format(self.params['num_limiters']) +
                       '/\n' +
                       '&mgrid\n' +
                       '  mgrid_currents = {}\n'.format(self.params['mgrid_currents']) +
                       '/')

    def set_transit_parameters(self, dstp, rots):
        """ Define the toroidal transit parameters.

        Parameters
        ----------
        dstp : float
            Step size in the azimuthal coordinate, in degrees.
        rots : int
            Number of toroidal transits.
        """
        dphi = dstp * (np.pi/180)
        stps = int(2 * np.pi / dphi)
        npts = int(rots*stps)

        mod_dict = {'points_dphi': dphi,
                    'n_iter': npts}

        self.change_params(mod_dict)

    def execute_flf(self, init_point, quiet=True, clean=True):
        """ Execcultes field line following code on specified input points.

        Parameters
        ----------
        init_point : arr
            (r,z,t) points where field line following code will be initialized.
        quiet : bool, optional
            When True, flf error messages will be dumped. When False, error messages
            will be printed in terminal. Default is False.
        clean: bool, optional
            When True, the I/O files generated during execution will be deleted.
            Default is True.

        Raises
        ------
        KeyError
            General Option specified for flf code cannot currently be
            performed.

        Returns
        -------
        points : arr
            Results of flf code.
        """
        # check if namelist exists #
        if not os.path.isfile(self.namelist):
            self.change_params(self.params)

        # Read Relevant Parameters #
        genOpt = self.params['general_option']
        pNum = self.params['points_number']
        nItr = self.params['n_iter']

        # Construct flf input file #
        with open(self.in_path, 'w') as file:
            if pNum==1:
                r,z,t = init_point
                file.write('{0} {1} {2}\n'.format(r,z,t))
            else:
                for point in init_point:
                    r,z,t = point
                    file.write('{0} {1} {2}\n'.format(r,z,t))

        # Instantiate flf code through bash #
        proc = subprocess.run(self.run_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        flf_out = proc.stdout.decode('utf-8').split('\n')[0:-1]
        flf_err = proc.stderr.decode('utf-8').split('\n')[0:-1]

        # check for errors in execution #
        if len(flf_err) > 0:
            if clean:
                cmnd = shlex.split('rm %s %s %s' % (self.namelist, self.in_path, os.path.join(self.wrapper_dir, 'results.out')))
                subprocess.run(cmnd)
            if quiet:
                return None
            else:
                print('\n------------------'+
                      '\nError in FLF code:'+
                      '\n------------------')
                for line in flf_err:
                    print(line)
                return None

        # General Option 1
        if genOpt==1:
            if pNum==1:
                points = np.empty((nItr, 4))
            else:
                points = np.empty((pNum, nItr, 4))
            points[:] = np.nan
            if pNum == 1:
                try:
                    for l, line in enumerate(flf_out[1::]):
                        line = line.strip()
                        line = line.split()
                        points[l] = [float(x) for x in line]
                except ValueError:
                    if not quiet:
                        print('\n------------------------------'+
                              '\nField line failed to complete:'+
                              '\n------------------------------'+
                              '\nFailed near phi = %.5f [pi]' % (l*self.params['points_dphi']/np.pi))
            else:
                for pIdx in range(pNum):
                    try:
                        for l, line in enumerate(flf_out[1+pIdx+pIdx*nItr:1+pIdx+nItr+pIdx*nItr]):
                            line = line.strip()
                            line = line.split()
                            points[pIdx,l] = [float(x) for x in line]
                    except ValueError:
                        if not quiet:
                            print('\n------------------------------'+
                                  '\nField line failed to complete:'+
                                  '\n------------------------------'+
                                  '\nFailed near phi = %.5f [pi]' % (l*self.params['points_dphi']/np.pi))

        # General Option 2
        elif self.params['general_option']==2:
            points = np.empty((pNum, 4))
            for l, line in enumerate(flf_out):
                line = line.strip()
                line = line.split()
                points[l] = [float(x) for x in line]

        else:
            raise KeyError('general_option input {} is invalid.'.format(self.params['general_option']))

        # Delete I/O files #
        if clean:
            cmnd = shlex.split('rm %s %s %s' % (self.namelist, self.in_path, os.path.join(self.wrapper_dir, 'results.out')))
            subprocess.run(cmnd)

        # Return flf results #
        return points

    def read_wout(self):
        """ Imports wout file corresponding to the current profile.

        Raises
        ------
        IOError
            wout file could not be found.
        """
        if self.exp == 'HSX':
            crnts = self.params['mgrid_currents'].strip()
            crnts = crnts.split()
            crnts = [float(x) for x in crnts]

            main_crnt = -crnts[0:6] / 10722.
            aux_crnt = -crnts[6::] / (14 * 10722.)
            crnt = np.r_[main_crnt, aux_crnt]

            try:
                path = functions.findPathFromCrnt(crnt)
            except IOError:
                raise IOError('wout file could not found.')

            self.wout = wout_read.readWout(path)

        else:
            raise KeyError(self.exp+' experiment does not have available wout files.')


    def read_out_along_line(self, tor_ang, ro_ang, ro_lim, nsurf=3, plt=None, return_data=False):
        """ Makes Poincare plot with initial points generated from a line
        extending out from the magnetic axis to the specified radial domain.

        Parameters
        ----------
        tor_ang : float
            Angle of toroidal cross section where the Poincare plot is
            generated.
        ro_ang : float
            Angle of the generating line extending out from the magnetic axis.
        ro_lim : float
            Scalar multiple of maximum minor radius of LCFS.  Constrains the
            radial domain of generating line.
        nsurf : int, optional
            Number of initial points taken from generating line.
            The default is 3.
        plt : obj, optional
            Axis on which poincare plot is generated. The default is None.
        return_data : bool, optional
            If True, returns poincare plot points. The default is False.

        Returns
        -------
        arr
            Poincare plot points, optional
        """
        # Read Out parameters #
        stps = int( (2*np.pi) / self.params['points_dphi'] )
        rots = int( (self.params['n_iter'] * self.params['points_dphi']) / (2*np.pi) )
        idx_stps = [int(i*stps) for i in range(rots)]

        # Get magnetic axis and raidal domain from wout #
        keys = ['R', 'Z']
        pol_ang = np.linspace(0, 2*np.pi, 1001)
        self.wout.transForm_2D_vSec(pol_ang, tor_ang, keys)

        R_dom = self.wout.invFourAmps['R']
        Z_dom = self.wout.invFourAmps['Z']

        R_ma = R_dom[0,0,0]
        Z_ma = Z_dom[0,0,0]

        r_eff = np.hypot(R_dom[-1,0,0::]-R_ma, Z_dom[-1,0,0::]-Z_ma)
        r_max = np.nanmax(r_eff)
        r_dom = np.linspace(0, ro_lim*r_max, nsurf)

        # Initial Poincare Points #
        ro_cos = np.cos(ro_ang)
        ro_sin = np.sin(ro_ang)

        init_points = np.empty((nsurf, 3))
        for idx, r in enumerate(r_dom):
            x = R_ma + r * ro_cos
            y = Z_ma + r * ro_sin

            init_points[idx] = [x,y,tor_ang]

        # Run flf #
        poin_points = np.empty((nsurf, rots, 4))
        for i, init in enumerate(init_points):
            print('\nInitial Point {0} of {1} : '.format(i+1,nsurf)+'(%.4f, %.4f, %.4f)' % (init[0],init[1],init[2]))
            points = self.execute_flf(init)
            if points is None:
                raise RuntimeError('FLF code failed on execution.')
            for j, idx in enumerate(idx_stps):
                poin_points[i,j] = points[idx]

        # Make Poincare plot #
        if plt:
            cm = plt.cm.get_cmap('jet')

            B_non0 = np.nonzero(poin_points[0::,0::,3])
            Bmin = np.nanmin(poin_points[B_non0[0], B_non0[1], 3])
            Bmax = np.nanmax(poin_points[0::,0::,3])

            plt.scatter(poin_points[0::,0,0], poin_points[0::,0,1], c='k', marker='X', s=50, zorder=5)
            plt.scatter(poin_points[0::,0::,0], poin_points[0::,0::,1], c=poin_points[0::,0::,3], vmin=Bmin, vmax=Bmax, s=3, cmap=cm)
            plt.colorbar(label=r'$|\mathbf{B}|$', format='%.4f')

            plt.title(r'$\theta$ = %.4f $\pi$' % (tor_ang/np.pi))
            plt.xlabel('R')
            plt.ylabel('Z')

        if return_data:
            return poin_points


    def read_out_point(self, init_point, ax=None, return_poin_data=False, quiet=True, clean=True):
        """ Makes Poincare plot by following field line found at input point.

        Parameters
        ----------
        init_point : list
            [r,z,t] values of initial point.
        ax : obj, optional
            Axis on which poincare plot is generated. The default is None.
        return_data : bool, optional
            If True, returns poincare plot points. The default is False.

        Returns
        -------
        arr
            Poincare plot points, optional
        """
        # Read Out parameters #
        stps = int(2 * np.pi / self.params['points_dphi'] )
        rots = int( (self.params['n_iter'] * self.params['points_dphi']) / (2*np.pi) )
        idx_stps = [int(i*stps) for i in range(rots)]

        # Run flf code #
        poin_pnts = np.empty((rots, 4))

        print('\nInitial Point : (%.4f, %.4f, %.4f)' % (init_point[0],init_point[1],init_point[2]))
        points = self.execute_flf(init_point, quiet=quiet, clean=clean)
        if points is None:
            raise RuntimeError('FLF code failed on execution.')

        for i, idx in enumerate(idx_stps):
            poin_pnts[i] = points[idx]

        # Make Poincare plot #
        if ax:
            cm = 'jet'

            Bmin = np.nanmin(poin_pnts[np.nonzero(poin_pnts[0::,3]), 3])
            Bmax = np.nanmax(poin_pnts[0::,3])

            # plt.scatter([np.mean(poin_pnts[0::,0])], [np.mean(poin_pnts[0::,1])], c='k', marker='X', s=50, zorder=5)

            ax.scatter([init_point[0]], [init_point[1]], c='k', marker='X', s=50, zorder=5)
            ax.scatter(poin_pnts[0::,0], poin_pnts[0::,1], c=poin_pnts[0::,3], vmin=Bmin, vmax=Bmax, s=3)
            # ax.colorbar(label=r'$|\mathbf{B}|$', format='%.2f')

            ax.set_xlabel('R')
            ax.set_ylabel('Z')

            ax.grid()

        if return_poin_data:
            return poin_pnts


    def read_out_set(self, init_points, plt=None, return_data=False, quiet=True, clean=True):
        """ Makes Poincare plot by following field line found at input point.

        Parameters
        ----------
        init_points : arr
            Array of [r,z,t] values for initial points.
        plt : obj, optional
            Axis on which poincare plot is generated. The default is None.
        return_data : bool, optional
            If True, returns poincare plot points. The default is False.

        Returns
        -------
        arr
            Poincare plot points, optional
        """
        # Read Out parameters #
        stps = int(2 * np.pi / self.params['points_dphi'])
        rots = int( (self.params['n_iter'] * self.params['points_dphi']) / (2*np.pi) )
        idx_stps = [int(i*stps) for i in range(rots)]

        # Run flf code #
        poin_set = np.empty((init_points.shape[0], rots, 4))
        for idx, init_point in enumerate(init_points):
            poin_pnts = np.empty((rots, 4))

            print('\nInitial Point {0:0.0f} of {1:0.0f} : '.format(idx+1, init_points.shape[0])+'(%.4f, %.4f, %.4f)' % (init_point[0],init_point[1],init_point[2]))
            if idx+1 == len(init_points):
                points = self.execute_flf(init_point, quiet=quiet, clean=clean)
                if points is None:
                    raise RuntimeError('FLF code failed on execution.')
            else:
                points = self.execute_flf(init_point, quiet=quiet, clean=clean)
                if points is None:
                    raise RuntimeError('FLF code failed on execution.')

            for i, jdx in enumerate(idx_stps):
                poin_pnts[i] = points[jdx]

            poin_set[idx] = poin_pnts

        # Make Poincare plot #
        if plt:
            # cm = 'jet' # plt.cm.get_cmap('jet')

            Bmin = np.nanmin(poin_set[:,np.nonzero(poin_set[:,:,3]), 3])
            Bmax = np.nanmax(poin_set[:,:,3])

            for idx in range(poin_set.shape[0]):
                #plt.scatter(poin_set[idx,:,0], poin_set[idx,:,1], c=poin_set[idx,:,3], vmin=Bmin, vmax=Bmax, s=3, cmap=cm)
                if idx+1 == len(poin_set):
                    plt.scatter(poin_set[idx,:,0], poin_set[idx,:,1], c='tab:red', s=3, label='FLF')
                else:
                    plt.scatter(poin_set[idx,:,0], poin_set[idx,:,1], c='tab:red', s=3)
            #plt.colorbar(label=r'$|\mathbf{B}|$', format='%.2f')

            plt.xlabel('R (m)')
            plt.ylabel('Z (m)')

            plt.grid()

        if return_data:
            return poin_set


    def read_out_domain(self, pnt1, pnt2, nsurf, ax=None, return_data=False, quiet=True, clean=True):
        """ Makes Poincare plot with initial points generated from a line
        extending between two specified points.

        Parameters
        ----------
        pnt1 : arr
            [r,z,t] values of first initial point.
        pnt2 : arr
            [r,z,t] values of second initial point.
        nsurf : int
            Number of initial points taken from generating line.
        ax : obj, optional
            Axis on which poincare plot is generated. The default is None.
        return_data : bool, optional
            If True, returns poincare plot points. The default is False.

        Returns
        -------
        arr
            Poincare plot points, optional
        """
        # Read Out parameters #
        stps = int( (2*np.pi) / self.params['points_dphi'] )
        rots = int( (self.params['n_iter'] * self.params['points_dphi']) / (2*np.pi) )
        idx_stps = [int(i*stps) for i in range(rots)]

        # Poincare Points #
        m = (pnt2[1] - pnt1[1]) / (pnt2[0] - pnt1[0])
        b = -m*pnt1[0] + pnt1[1]

        r_dom = np.linspace(pnt1[0], pnt2[0], nsurf)
        z_dom = m * r_dom + b
        t_dom = np.empty(nsurf)
        t_dom[:] = pnt1[2]

        init_points = np.stack([r_dom, z_dom, t_dom], axis=1)

        # Run flf code #
        poin_points = np.empty((nsurf, rots, 4))
        for i, init in enumerate(init_points):
            print('\nInitial Point {0} of {1} : '.format(i+1,nsurf)+'(%.8f, %.8f, %.8f)' % (init[0],init[1],init[2]))
            if i+1 == init_points.shape[0]:
                points = self.execute_flf(init, quiet=quiet, clean=clean)
                if points is None:
                    raise RuntimeError('FLF code failed on execution.')
            else:
                points = self.execute_flf(init, quiet=quiet, clean=clean)
                if points is None:
                    raise RuntimeError('FLF code failed on execution.')

            for j, idx in enumerate(idx_stps):
                poin_points[i,j] = points[idx]

        # Make Poincare plot #
        if ax:
            cm ='jet'  # plt.cm.get_cmap('jet')

            B_non0 = np.nonzero(poin_points[0::,0::,3])
            Bmin = np.nanmin(poin_points[B_non0[0], B_non0[1], 3])
            Bmax = np.nanmax(poin_points[0::,0::,3])

            # plt.scatter(poin_points[:, 0, 0], poin_points[:, 0, 1], s=100, marker='x', c='k')
            ax.scatter(poin_points[0::,0::,0], poin_points[0::,0::,1], c='k', s=1)  # , vmin=Bmin, vmax=Bmax, cmap=cm)
            # plt.scatter(poin_points[0::,0::,0], poin_points[0::,0::,1], c=poin_points[0::,0::,3], s=10, vmin=Bmin, vmax=Bmax, cmap=cm)
            # plt.colorbar(label=r'$|\mathbf{B}| \ [T]$', format='%.2f')

            ax.set_title(r'$\theta$ = %.3f $\pi$' % (pnt1[2]/np.pi))
            ax.set_xlabel('R')
            ax.set_ylabel('Z')

        if return_data:
            return poin_points


    def read_out_grid_sample(self, center_point, half_width, npts, plt=None, return_data=False, quiet=True, clean=True):
        """ Makes Poincare plot with initial points generated as a uniform set
        from the square centered around a point.

        Parameters
        ----------
        center_point : arr
            [r,z,t] values of center point.
        half_width : float
            value of the half width of the square centered around the
            center_point.
        npts : int
            Number of points initialized in circle.
        plt : obj, optional
            Axis on which poincare plot is generated. The default is None.
        return_data : bool, optional
            If True, returns poincare plot points. The default is False.

        Returns
        -------
        arr
            Poincare plot points, optional
        """
        # Read Out parameters #
        stps = int((2*np.pi) / self.params['points_dphi'])
        rots = int((self.params['n_iter'] * self.params['points_dphi']) / (2*np.pi))
        idx_stps = [int(i*stps) for i in range(rots)]

        # Poincare Points #
        sqrt_pnts = int(round(np.sqrt(npts)))
        npts = sqrt_pnts**2

        r_dom = np.linspace(center_point[0]-half_width, center_point[0]+half_width, sqrt_pnts)
        z_dom = np.linspace(center_point[1]-half_width, center_point[1]+half_width, sqrt_pnts)
        t_dom = center_point[2]

        r_grid, z_grid, t_grid = np.meshgrid(r_dom, z_dom, t_dom)
        init_points = np.stack((r_grid, z_grid, t_grid), axis=3)
        init_points = init_points.flatten().reshape(npts, 3)

        # Run flf code #
        poin_points = np.empty((npts, rots, 4))
        for i, init in enumerate(init_points):
            print('\nInitial Point {0} of {1} : '.format(i+1, npts)+'(%.8f, %.8f, %.8f)' % (init[0], init[1], init[2]))
            if i+1 == init_points.shape[0]:
                points = self.execute_flf(init, quiet=quiet, clean=clean)
                if points is None:
                    raise RuntimeError('FLF code failed on execution.')
            else:
                points = self.execute_flf(init, quiet=quiet, clean=clean)
                if points is None:
                    raise RuntimeError('FLF code failed on execution.')

            for j, idx in enumerate(idx_stps):
                poin_points[i, j] = points[idx]

        # Make Poincare plot #
        if plt:
            cm = plt.cm.get_cmap('jet')

            B_non0 = np.nonzero(poin_points[:, :, 3])
            Bmin = np.nanmin(poin_points[B_non0[0], B_non0[1], 3])
            Bmax = np.nanmax(poin_points[:, :, 3])

            plt.scatter(poin_points[:, 0, 0], poin_points[:, 0, 1], s=100, marker='x', c='k')
            plt.scatter(poin_points[:, :, 0], poin_points[:, :, 1], c=poin_points[:, :, 3], s=10, vmin=Bmin, vmax=Bmax, cmap=cm)
            # plt.colorbar(label=r'$|\mathbf{B}| \ [T]$', format='%.2f')

            plt.title(r'$\theta$ = %.3f $\pi$' % (center_point[2]/np.pi))
            plt.xlabel('R')
            plt.ylabel('Z')

        if return_data:
            return poin_points


    def fit_surf(self, init_point, r_modes=3, z_modes=3, plt=None):
        """ Performs a Fourier fit to the poincare plot points in the toroidal
        domain of the initialized point.

        Parameters
        ----------
        init_point : arr
            Initial (r,z,t) point from which field line following is done.
        r_modes : int, optional
            Number of Fourier modes in r coordinates. The default is 3.
        z_modes : int, optional
            Number of Fourier modes in r coordinates. The default is 3.
        plt : obj, optional
            Axis on which poincare plot is generated. The default is None.

        Returns
        -------
        arr
            Fourier fit data to poincare plot data.
        float
            Chi-squared of Fourier fit.
        """
        points = self.read_out_point(init_point, return_poin_data=True, plt=plt)[0::, 0:2]

        # Check for flf failure #
        if np.isnan(points).any():
            print('Initial Point = ({0:0.4f}, {1:0.4f}, {2:0.4f}) : flf failure'.format(init_point[0], init_point[1], init_point[2]))
            return init_point, np.inf

        # Shift points around origin #
        avg_pnt = [np.mean(points[0::,0]), np.mean(points[0::,1])]
        points_shft = points - avg_pnt

        r_shft, z_shft = points_shft[0::,0], points_shft[0::,1]

        # Order points for FFT #
        quad_1 = np.array( sorted( points_shft[(r_shft >= 0) & (z_shft >= 0)], key=lambda x: x[0], reverse=True) )
        quad_2 = np.array( sorted( points_shft[(r_shft < 0) & (z_shft >= 0)], key=lambda x: x[0], reverse=True) )
        quad_3 = np.array( sorted( points_shft[(r_shft < 0) & (z_shft < 0)], key=lambda x: x[0]) )
        quad_4 = np.array( sorted( points_shft[(r_shft >= 0) & (z_shft < 0)], key=lambda x: x[0]) )

        r_data_base = np.r_[quad_1[0::,0], quad_2[0::,0], quad_3[0::,0], quad_4[0::,0]]
        r_data = np.r_[r_data_base, r_data_base, r_data_base, r_data_base]

        z_data_base = np.r_[quad_1[0::,1], quad_2[0::,1], quad_3[0::,1], quad_4[0::,1]]
        z_data = np.r_[z_data_base, z_data_base, z_data_base, z_data_base]

        t_dom = np.linspace(0, 1, r_data.shape[0])

        # Perform FFT #
        r_fft = fft_class.fftTools(r_data, t_dom)
        z_fft = fft_class.fftTools(z_data, t_dom)

        # Filter Peaks #
        r_fft.peak_selector(r_modes)
        z_fft.peak_selector(z_modes)

        # Perform IFFT #
        r_ifft = r_fft.ifft(return_ifft=True, renormalize=True).real
        z_ifft = z_fft.ifft(return_ifft=True, renormalize=True).real

        # Calculate Chi-squared #
        r_ifft_fit = r_ifft + np.min(r_ifft) + 1
        r_data_exp = r_data + np.min(r_data) + 1

        z_ifft_fit = z_ifft + np.min(z_ifft) + 1
        z_data_exp = z_data + np.min(z_data) + 1

        r_chi, r_pVal = chisquare(r_ifft_fit, r_data_exp)
        z_chi, z_pVal = chisquare(z_ifft_fit, z_data_exp)

        # Return flux surface points and Fourier fit chi-squared value #
        flux_surf = np.stack((r_ifft, z_ifft), axis=1) + avg_pnt
        return flux_surf, np.hypot(r_chi, z_chi)

    def flux_surface_dimensionality(self, points):
        """ Calculate the flux surface pointwise dimensional and return flux surface score.
        Ergodic field lines typically return values above 0.1

        Parameters
        ----------
        points : Arr
            Array of flux surface points.
        tor_ang : float (Optional)
            Toroidal angle at which pointwise dimension is calculated. Default is pi/4.
        """
        """
        dphi = self.params['points_dphi']
        stps = int(2*np.pi/dphi)
        rots = int(points[-1, 2] / (2*np.pi))
        idx_stps = np.argmin(np.abs(points[:, 2] - points[0, 2] - tor_ang)) + [int(i*stps) for i in range(rots)]
        print(points.shape)
        poin_data = points[idx_stps, 0:2]
        """
        # dimensionality curve for initial point #
        ma_axis = np.mean(points, axis=0)
        radius = np.mean(np.linalg.norm(points - ma_axis, axis=1))
        radii = np.logspace(np.log10(radius*(np.pi/36)), np.log10(radius), 25)

        inside = np.zeros(radii.shape)
        for idx in range(points.shape[0]):
            check_point = points[idx]
            comp_points = np.delete(points, idx, axis=0)
            norm_points = np.linalg.norm(comp_points - check_point, axis=1)

            idx_inside = np.empty(radii.shape)
            for i, radius in enumerate(radii):
                idx_inside[i] = norm_points[norm_points <= radius].shape[0]
            inside = inside + idx_inside

        inside = inside / points.shape[0]

        x_data = np.log10(radii.reshape((-1, 1)))
        y_data = np.log10(inside)

        model = LinearRegression().fit(x_data, y_data)
        r_sq = model.score(x_data, y_data)
        slope = model.coef_[0]

        # return np.hypot((1.-r_sq), (1.-slope))
        return slope-1.

    def find_lcfs(self, init_point, dec_limit, r_limits, scan_res_limit=2, high_precission=False):
        """ Approximately locate the LCFS and Magnetic Axis, represented as
        (r,z,t) points initialized in the flf code.

        Parameters
        ----------
        init_point: array
            Initial point {r,z,t} from which we move in the negative radial direction
            until a closed flux surface is found.  Initial steps are in 0.1
            increments.
        dec_limit: int
            Number of decimal points in approximation.  This does not guarantee
            accuracy to this decimal point.
        r_limits: tuple
            Radial limits of domain over which to perform scan. First and second 
            elements are the minimal and maximal R values, repsectively.
        scan_res_limit: int (optional)
            The decimal point resolution the scan will go to. Default is 2.
        high_precission: Bool (optional)
            If True, then the pointwise dimension will be calculated for the LCFS to 
            determine if the flux surface is ergodic or not. Default is False.
        """
        print('\n----------------\n'
              'Looking for LCFS\n'+
              '----------------')

        scan_res = 1
        first_contact = False
        exceed_limit = False
        r_init = init_point[0]
        flf_points = self.read_out_point(init_point, return_poin_data=True)
        while scan_res <= scan_res_limit:
            dr_dom = np.logspace(-scan_res, -dec_limit, 1+dec_limit-scan_res)
            for dr_scl in dr_dom:
                stp_cnt = -1
                if not first_contact and not exceed_limit:
                    if np.isnan(flf_points).any():
                        dr = -dr_scl
                    else:
                        dr = dr_scl

                    stp_cnt += 1
                    init_point[0] += dr
                    Dr_max = r_limits[1] - init_point[0]
                    Dr_min = init_point[0] - r_limits[0]
                    if (Dr_min <= 0) or (Dr_max <= 0):
                        exceed_limit = True
                        init_point[0] = r_init + .1*dr
                        scan_res += 1
                        break
                else:
                    dr = np.sign(dr)*dr_scl

                while True:
                    flf_points = self.read_out_point(init_point, return_poin_data=True)
                    are_nans = np.isnan(flf_points).any()
                    if are_nans and (dr > 0):
                        break
                    elif not are_nans and (dr < 0):
                        break
                    stp_cnt += 1
                    init_point[0] += dr
                    Dr_max = r_limits[1] - init_point[0]
                    Dr_min = init_point[0] - r_limits[0]
                    if (Dr_min <= 0) or (Dr_max <= 0):
                        exceed_limit = True
                        break

                    if first_contact:
                        if stp_cnt >= 8:
                            break

                if (Dr_max <= 0) or (Dr_min <= 0):
                    init_point[0] = r_init + .1*dr
                    scan_res += 1
                    break

                else:
                    first_contact = True
                    scan_res = scan_res_limit+1
                    if dr < 0:
                        r_min = init_point[0]
                        r_max = init_point[0] - dr
                        init_point[0] = r_max + .1*dr
                    else:
                        r_min = init_point[0] - dr
                        r_max = init_point[0]
                        init_point[0] = r_min + .1*dr
                    print('\n{0} < r_init < {1}'.format(round(r_min, dec_limit), round(r_max, dec_limit)))

        if (Dr_max <= 0) or (Dr_min <= 0):
            self.lcfs_point = np.full(3, np.nan)
            warnings.warn("LCFS not found within radial limits.")
        else:
            init_point[0] = r_min
            if high_precission:
                print('\n-------------------------------\n'+
                      'Beginning high precission phase\n'+
                      '-------------------------------')
                flf_points = self.read_out_point(init_point, return_poin_data=True)
                fs_check = self.flux_surface_dimensionality(flf_points)
                print('flux surface dimension = {}'.format(fs_check+1))
                while fs_check > 0.05:
                    init_point[0] -= np.abs(dr)
                    flf_points = self.read_out_point(init_point, return_poin_data=True)
                    print('flux surface dimension = {}'.format(fs_check+1))

            self.lcfs_point = np.round(init_point, decimals=dec_limit)
            print('\n---------------\n'
                  'LCFS Identified\n'+
                  '---------------\n'+
                  'FLF Point ~ ({}, {}, {})'.format(self.lcfs_point[0], self.lcfs_point[1], self.lcfs_point[2]))
        
    def find_magnetic_axis(self, init_point, dec):
        """  Approximately locate the Magnetic Axis, represented as (r,z,t)
        points initialized in the flf code.

        Parameters
        ----------
        init_point : arr
            Initial guess for magnetic axis.  This needn't be a good guess, but
            it should be on a closed flux surface for best results.
        dec : int
            Number of decimal points in approximation.  This does not guarantee
            accuracy to this decimal point.
        """
        print('\n-------------------------\n'
              'Looking for Magnetic Axis\n'+
              '-------------------------')
        points = self.read_out_point(init_point, return_poin_data=True, quiet=quiet)
        ma_point = np.r_[np.mean(points[0::,0:2], axis=0), init_point[2]]

        pnt_sep = np.max(points[0::,0]) - np.min(points[0::,0])
        pnt_scl = np.floor(np.log10(pnt_sep))

        dist = np.linalg.norm(ma_point - init_point)
        dist_scl = np.floor(np.log10(dist))

        path = mpltPath.Path(points[0::,0:2])
        inside = path.contains_point(ma_point[0:2])

        cnt = 0
        ma_points = []
        while dist_scl >= -dec:
            if not inside and dist_scl < pnt_scl:
                ma_point[0] = init_point[0] - 10**(dist_scl-pnt_scl-1)

            init_point = ma_point
            init_point[0], init_point[1] = round(init_point[0], dec), round(init_point[1], dec)

            points = self.read_out_point(init_point, return_poin_data=True, quiet=quiet)
            ma_point = np.r_[np.mean(points[0::,0:2], axis=0), init_point[2]]

            ma_points.append(ma_point)
            if len(ma_points) > 10:
                ma_point_chk = np.array(ma_points)
                resR = np.correlate(ma_point_chk[0::,0], ma_point_chk[0::,0], mode='full')

                hlf_idx = int(0.5*len(resR))
                print('   Auto Correlation = {0:0.3f}'.format(resR[hlf_idx]))
                if (resR[hlf_idx] > 0.5):
                    ma_point = np.mean(ma_point_chk[10::], axis=0)
                    ma_points = [ma_point]
                    cnt+=1

                    print('   {0} : ({1:0.4f}, {2:0.4f}, {3:0.4f})'.format(cnt, ma_point[0], ma_point[1], ma_point[2]))
                    if cnt == 3:
                        break

            pnt_sep = np.max(points[0::,0]) - np.min(points[0::,0])
            pnt_scl = np.floor(np.log10(pnt_sep))

            dist = np.linalg.norm(ma_point - init_point)
            dist_scl = np.floor(np.log10(dist))

            path = mpltPath.Path(points[0::,0:2])
            inside = path.contains_point(ma_point[0:2])

        self.ma_point = np.round(ma_point[0:3], dec)
        print('\n------------------------\n'
              'Magnetic Axis Identified\n'+
              '------------------------\n'+
              'FLF Point ~ ({}, {}, {})'.format(self.ma_point[0], self.ma_point[1], self.ma_point[2]))

    def flf_surface(self, ma_points, surf_points, mod_params, makeVTK=False, dictVTK=None):
        """ Generates a B-field surface from the flf data provided by the
        magnetic axis and the input initial field point.

        Parameters
        ----------
        ma_pnt : arr
            {r,z,t} point where magnetic axis is initialized.
        surf_pnt : arr
            {r,z,t} point where field line following will be initiated.  This
            data will define the field surface that will be returned.
        rots : int, optional
            Number of toroidal transits to be performed by the flf code for the
            field surface. The default is 500.
        makeVTK : bool, optional
            If you wish to produce a vtk file of the field surface set to True.
            The default is False.
        dictVTK : dict, optional
            Dictionary that provides the directory path and file name where the
            vtk file will be saved. The default is None.

        Returns
        -------
        arr
            Array containing the surface field data, returned in array with
            shape {360, rots, 4}.  The first index is the toroidal points, the
            second index is the poloidal points and the third index provides
            the R, Z, mod B and poloidal angles.
        """
        #stps = int(2 * np.pi / mod_params['points_dphi'] )
        rots = int( (mod_params['n_iter'] * mod_params['points_dphi']) / (2*np.pi) )
        stps = int(surf_points.shape[0] / rots)
        idx_stps = [int(i*stps) for i in range(rots)]

        v_dom = surf_points[0::,2]
        surf_ordered = np.empty((stps+1, rots, 5))
        for v_idx, v in enumerate(ma_points[0::,2]):
            idx_stps = np.argmin(np.abs(v_dom - v)) + [int(i*stps) for i in range(rots)]
            stp_idx = np.argmin(np.abs(ma_points[0::,2] - v))
            Rma, Zma = ma_points[stp_idx,0], ma_points[stp_idx,1]

            vals = np.empty((rots,5))
            for rot_idx, idx in enumerate(idx_stps):
                r_val, z_val, B_mod = surf_points[idx][0], surf_points[idx][1], surf_points[idx][3]
                pol = np.arctan2(z_val-Zma, r_val-Rma)
                if pol < 0:
                    pol = pol + 2*np.pi
                vals[rot_idx] = np.array([r_val, z_val, v, B_mod, pol])
            vals = np.array(sorted(vals, key=lambda x: x[4]))
            surf_ordered[stp_idx] = vals
        surf_ordered[-1] = surf_ordered[0]

        if makeVTK and dictVTK:
            import vtkTools.vtk_grids as vtkG

            B_mod = surf_ordered[0::,0::,3]
            z = surf_ordered[0::,0::,1]
            y = np.empty(z.shape)
            x = np.empty(z.shape)
            for idx, v in enumerate(ma_points[0::,2]):
                x[idx] = surf_ordered[idx,0::,0] * np.cos(v)
                y[idx] = surf_ordered[idx,0::,0] * np.sin(v)

            cart_coord = np.stack((x,y,z), axis=2)
            vtkG.scalar_mesh(dictVTK['savePath'], dictVTK['fileName'], cart_coord, B_mod)

        ma_ordered = np.empty(ma_points.shape)
        ma_ordered = np.stack((ma_points[0:,0], ma_points[0:,2], ma_points[0:,1]), axis=1)

        return surf_ordered, ma_ordered

    def run_descur(self, ma_point, surf_point, pol_pnts=20, tor_pnts=100, plot_data=False, save_path=os.getcwd(), quiet=True, clean=True):
        """ Generate DESCUR input data for a flux surface.

        Parameters
        ----------
        ma_point : arr
            Initial point for magnetic axis.
        surf_point : arr
            Initial point for flux surface that will be converted to a DESCUR
            input file.
        pol_pnts : int, optional
            Number of poloidal points in DESCUR input. The default is 120.
        tor_pnts : int, optional
            Number of toroidal points in DESCUR input. The default is 90.
        save_path : str, optional
            Global path to where descur input will be saved. Default is CWD.
        """
        nfp = int(self.params['num_periods'])
        u_pnts = int(pol_pnts * nfp)
        v_pnts = int(tor_pnts / nfp)
        n_pnts = int(u_pnts * v_pnts)

        if tor_pnts % nfp != 0 and pol_pnts % 10 != 0:
            raise ValueError('tor_pnts and pol_pnts must be multiples of the number of field periods ({0:0.0f}).'.format(nfp))

        dphi = (2 * np.pi) / tor_pnts
        stps = tor_pnts
        mod_dict = {'points_dphi' : dphi,
                    'n_iter' : stps}

        v_dom = np.arange(0, 2*np.pi, dphi)

        self.change_params(mod_dict)
        ma_points = self.execute_flf(ma_point, quiet=quiet, clean=clean)
        if points is None:
            raise RuntimeError('FLF code failed on execution.')

        npts = int(pol_pnts * stps)
        mod_dict = {'points_dphi' : dphi,
                    'n_iter' : npts}

        self.change_params(mod_dict)
        surf_points = self.execute_flf(surf_point, quiet=quiet, clean=clean)
        if points is None:
            raise RuntimeError('FLF code failed on execution.')

        surf, ma = self.flf_surface(ma_points, surf_points, mod_dict)

        pol_idx = np.linspace(0, u_pnts, pol_pnts, endpoint=False, dtype=int)
        fit_data = np.empty((n_pnts, 3))
        for i, v in enumerate(v_dom):
            if v >= np.pi:
                v = v - 2*np.pi

            r_vals = surf[i,0:,0]
            z_vals = surf[i,0:,1]
            v_vals = np.full(r_vals.shape, v)
            pol_vals = np.stack((r_vals, v_vals, z_vals), axis=1)

            data_idx = int((i % v_pnts) * u_pnts) + int(np.floor(i / v_pnts)) + pol_idx
            fit_data[data_idx] = pol_vals

        if plot_data == True:
            x_data = fit_data[0::,0] * np.cos(fit_data[0::,1])
            y_data = fit_data[0::,0] * np.sin(fit_data[0::,1])
            z_data = fit_data[0::,2]

            plot = pd.plot_define(proj3D=True)
            plot.ax.scatter(x_data.flatten(), y_data.flatten(), z_data.flatten(), s=1)
            plot.plt.show()

        with open(save_path, 'w') as file:
            file.write('{0} {1} {2}\n'.format(u_pnts, v_pnts, nfp))
            for n in range(n_pnts):
                file.write('{0:0.6f} {1:0.6f} {2:0.6f} \n'.format(fit_data[n,0], fit_data[n,1], fit_data[n,2]))

    def sample_circle(self, ma_pnt, lcfs_pnt, upts=100, diam=5e-2, delta=1e-3, Bidx='Bmod'):
        #from scipy.interpolate import interp1d
        dphi = (2 * np.pi) / 360
        stps = 360
        npts = int(upts * stps)

        mod_dict = {'points_dphi' : dphi,
                    'n_iter' : npts}

        nsurf = 3
        self.change_params(mod_dict)
        points = self.read_out_domain(ma_pnt, lcfs_pnt, nsurf, return_data=True)

        diam = 2 * np.max( np.linalg.norm( points[nsurf-1,0:,0:2] - ma_pnt[0:2], axis=1 ) )

        Npts = 0.25 * np.pi * ( ( diam / delta )**2 + 2 * ( diam / delta ) + 1 )
        Ypts = int( 2 * np.sqrt( Npts / np.pi ) )

        ylim = np.linspace(ma_pnt[1] - 0.5 * diam, ma_pnt[1] + 0.5 * diam, Ypts)

        xlim_left = ma_pnt[0] - np.sqrt( 0.25 * diam**2 - ( ylim - ma_pnt[1] )**2 )
        xlim_right = ma_pnt[0] + np.sqrt( 0.25 * diam**2 - ( ylim - ma_pnt[1] )**2 )

        length = 0
        for xidx, xval_l in enumerate(xlim_left):
            xval_r = xlim_right[xidx]
            length = length + xval_l - xval_r

        nspc = Npts / length

        points_in = np.array([[ xlim_left[0], ylim[0], ma_pnt[2] ]])
        for xidx, xval_l in enumerate(xlim_left[1:]):
            xidx = xidx + 1
            xval_r = xlim_right[xidx]

            lngth = xval_l - xval_r

            xpts = np.linspace(xval_l, xval_r, int( lngth * nspc ) )
            ypts = np.array([ylim[xidx]] * xpts.shape[0])
            tpts = np.array([ma_pnt[2]] * xpts.shape[0])
            pts_in = np.stack((xpts, ypts, tpts), axis=1)

            points_in = np.append(points_in, pts_in, axis=0)

        mod_dict = {'general_option' : 2,
                    'points_number' : points_in.shape[0]}

        self.change_params(mod_dict)
        points_out = self.execute_flf(points_in)
        if points is None:
            raise RuntimeError('FLF code failed on execution.')

        plot = pd.plot_define(eqAsp=True)

        if Bidx == 'Bmod':
            bidx = 3
            blab = r'$|B|$'
        elif Bidx == 'Br':
            bidx = 0
            blab = r'$B_r$'
        elif Bidx == 'Bz':
            bidx = 1
            blab = r'$B_z$'
        elif Bidx == 'Bt':
            bidx = 2
            blab = r'$B_{\theta}$'

        scl = 1e2
        plot.plt.scatter(points[0:,0:,0] * scl, points[0:,0:,1] * scl, c='k', s=5, zorder=10)
        s_map = plot.plt.scatter(points_in[0:,0] * scl, points_in[0:,1] * scl, c=points_out[0:,bidx] * 1e3, s=5, cmap='jet')

        plot.plt.xlabel('R [cm]')
        plot.plt.ylabel('Z [cm]')

        cbar = plot.fig.colorbar(s_map, ax=plot.ax)
        cbar.ax.set_ylabel(blab+' [mT]')

        plot.plt.show()

    def calc_psiEdge(self, ma_pnt, lcfs_pnt, Npts=1000, upts=100, Bidx='Bmod', plot_true=True, quiet=True, clean=True):
        """ A well documented commentary

        Parameters
        ----------
        ma_pnt : [[3,]] numpy array
            Magnetic Axis.
        lcfs_pnt : [[3,]] numpy array
            Initial point on Last Closed Flux Surface (LCFS).
        Npts : int, optional
            Number of interior points. The default is 1000.
        upts : int, optional
            Number of poloidal points. The default is 100.
        Bidx : str, optional
            Magnetic field component, options are
                'Bmod' : Magnitude of B-field
                'Br' : Radial
                'Bz' : Vertial
                'Bt' : Toroidal

        Raises
        ------
        ValueError
            DESCRIPTION.
        """
        from scipy.interpolate import interp1d

        dphi = (2 * np.pi) / 360
        stps = 360
        npts = int(upts * stps)

        mod_dict = {'points_dphi' : dphi,
                    'n_iter' : npts}
        self.change_params(mod_dict)

        points = self.read_out_point(lcfs_pnt, return_poin_data=True)[0:,0:3] - ma_pnt

        # Calculate Area #
        area_data = np.array( [ [ np.arctan2(pnt[1], pnt[0]), pnt[0]**2 + pnt[1]**2, pnt[0], pnt[1] ] for pnt in points ] )
        area_data = np.array( sorted(area_data, key=lambda x: x[0]) )

        theta = np.r_[area_data[-1,0] - 2*np.pi, area_data[0:,0], area_data[0,0] + 2*np.pi]
        r_sqrd = np.r_[area_data[-1,1], area_data[0:,1], area_data[0,1]]

        theta_ext = np.r_[area_data[0:,0] - 2*np.pi, area_data[0:,0], area_data[0:,0] + 2*np.pi]
        x_data = np.r_[area_data[0:,2], area_data[0:,2], area_data[0:,2]]
        y_data = np.r_[area_data[0:,3], area_data[0:,3], area_data[0:,3]]

        theta_dom = np.linspace(-np.pi, np.pi, upts)
        Dtheta = theta_dom[1] - theta_dom[0]

        x_data_interp = interp1d(theta_ext, x_data)
        y_data_interp = interp1d(theta_ext, y_data)

        r_sqrd_interp = interp1d(theta, r_sqrd)
        r_sqrd_dom = r_sqrd_interp(theta_dom)
        area = 0.5 * np.trapz(r_sqrd_dom, dx=Dtheta)

        # Generate Grid for B Sample #
        width = np.max(points[0:,0]) - np.min(points[0:,0])
        height = np.max(points[0:,1]) - np.min(points[0:,1])
        pref = 1 - 0.5 * (height / width)

        if pref < 0:
            hpts = int( pref * ( 1 - np.sqrt( 1 + ( (Npts * height * height) / (area * pref * pref) ) ) ) )
        elif pref > 0:
            hpts = int( pref * ( 1 + np.sqrt( 1 + ( (Npts * height * height) / (area * pref * pref) ) ) ) )
        else:
            raise ValueError("height = 2 x width -> Divide by Zero Encountered")

        y_dom = np.linspace(np.min(points[0:,1]), np.max(points[0:,1]), hpts)[0:hpts-1] + 0.5 * (height / (hpts - 2))

        y_vals = np.empty((y_dom.shape[0], 2))
        x_vals = np.empty((y_dom.shape[0], 2))

        the_init = [np.pi, 0]
        for the_idx, the_beg in enumerate(the_init):
            for idx, y_val in enumerate(y_dom):
                the_0 = the_beg
                y_0 = y_data_interp(the_0)

                y_scl_1 = y_data_interp(the_0 + 0.01 * np.pi)
                y_scl_0 = y_data_interp(the_0)

                d_scl = 0.01 * np.pi * ( ( y_scl_1 - y_scl_0 ) / np.abs( y_scl_1 - y_scl_0 ) )
                d_the = d_scl * ( ( y_val - y_0 ) / np.abs( y_val - y_0 ) )
                d_the_sign = np.sign(d_the)

                y_1 = y_data_interp(the_0 + d_the)
                the_0 = the_0 + d_the

                diff = np.abs( ( y_val - y_1 ) / y_val )
                d_the = d_the = d_scl * ( ( y_val - y_1 ) / np.abs( y_val - y_1 ) )

                cnt=0
                diff = 1
                thrsh = 1e-6
                while diff > thrsh:
                    y_1 = y_data_interp(the_0 + d_the)
                    the_0 = the_0 + d_the

                    diff = np.abs( ( y_val - y_1 ) / y_val )
                    d_the = d_scl * ( ( y_val - y_1 ) / np.abs( y_val - y_1 ) )

                    if np.sign(d_the) != d_the_sign:
                        d_the_sign = np.sign(d_the)
                        d_scl = d_scl * 0.1
                        d_the = d_the * 0.1

                    cnt+=1

                y_vals[idx, the_idx] = y_1
                x_vals[idx, the_idx] = x_data_interp(the_0)

        # Calculate Point Density #
        length = 0
        for x_set in x_vals:
            length = length + x_set[1] - x_set[0]
        Npts_len = Npts / length

        # Construct Sample Grid #
        x_set = x_vals[0]
        length = x_set[1] - x_set[0]
        x_samp = np.linspace( x_set[0], x_set[1], int( length * Npts_len ) )

        y_set = y_vals[0]
        m_y = ( y_set[1] - y_set[0] ) / ( x_set[1] - x_set[0] )
        b_y = y_set[0] - m_y * x_set[0]
        y_samp = np.array( m_y * x_samp + b_y )

        sample_points = np.stack( (x_samp, y_samp), axis=1 ).reshape(y_samp.shape[0], 2)
        for idx, x_set in enumerate(x_vals[1:]):
            length = x_set[1] - x_set[0]

            x_samp = np.linspace( x_set[0], x_set[1], int( length * Npts_len ) )

            y_set = y_vals[idx + 1]
            m_y = ( y_set[1] - y_set[0] ) / ( x_set[1] - x_set[0] )
            b_y = y_set[0] - m_y * x_set[0]
            y_samp = np.array( m_y * x_samp + b_y )

            samp = np.stack( (x_samp, y_samp), axis=1 ).reshape(y_samp.shape[0], 2)
            sample_points = np.append(sample_points, samp, axis=0)


        tor_ang = ma_pnt[2]
        t_points = np.array( [ tor_ang ] * sample_points.shape[0] )
        points_in = np.stack( ( sample_points[0:,0] + ma_pnt[0], sample_points[0:,1] + ma_pnt[1], t_points ), axis=1 )

        mod_dict = {'general_option' : 2,
                    'points_number' : sample_points.shape[0]}

        self.change_params(mod_dict)
        points_out = self.execute_flf(points_in, quiet=quiet, clean=clean)
        if points is None:
            raise RuntimeError('FLF code failed on execution.')

        if Bidx == 'Bmod':
            bidx = 3
            blab = r'$|B|$'
        elif Bidx == 'Br':
            bidx = 0
            blab = r'$B_r$'
        elif Bidx == 'Bz':
            bidx = 1
            blab = r'$B_z$'
        elif Bidx == 'Bt':
            bidx = 2
            blab = r'$B_{\theta}$'

        psi_edge = area * np.mean(points_out[0:,3])
        # print('Psi Edge : {} [T m^2]'.format( psi_edge ) )

        if plot_true:
            plot = pd.plot_define()

            plt = plot.plt
            fig, ax = plt.subplots(1, 1, tight_layout=True)

            ax.set_aspect('equal')

            ax.set_title(r'$\Psi_{edge}$'+' : {0:0.2f}'.format(1e4 * area * np.mean(points_out[0:,3]))+r' [$T\cdot cm^2$]')
            ax.scatter(points[0:,0] * 1e2, points[0:,1] * 1e2, c='k', s=5, zorder=10)

            s_map = ax.scatter( sample_points[0:,0] * 1e2, sample_points[0:,1] * 1e2, c=points_out[0:,bidx], s=10, cmap='jet')

            cbar = fig.colorbar(s_map, ax=ax)
            cbar.ax.set_ylabel(blab+' [T]')

            ax.set_xlabel('R [cm]')
            ax.set_ylabel('Z [cm]')

            ax.grid()
            plt.show()

        return psi_edge


    def z_field_drift(self, ma_init, dstp=1, quiet=True, clean=True):
        """ Calculate the magnetic field over one toroidal transit along the
        magnetic axis.

        Parameters
        ----------
        ma_init : arr
            {r,z,phi} point where magnetic axis is initialized.
        dstp : int, optional
            phi step size in degrees. The default is 1.

        Returns
        -------
        B_vec : TYPE
            DESCRIPTION.
        points : TYPE
            DESCRIPTION.

        """
        rots=1

        dphi = dstp * (np.pi/180)
        stps = int(2 * np.pi / dphi)
        npts = int(rots*stps)

        mod_dict = {'general_option' : 1,
                    'points_number' : 1,
                    'points_dphi' : dphi,
                    'n_iter' : npts}

        self.change_params(mod_dict)
        points = self.execute_flf(ma_init, quiet=quiet, clean=clean)
        if points is None:
            raise RuntimeError('FLF code failed on execution.')

        mod_dict = {'general_option' : 2,
                    'points_number' : npts}

        self.change_params(mod_dict)
        B_vec = self.execute_flf(points[0::,0:3], quiet=quiet, clean=clean)
        if points is None:
            raise RuntimeError('FLF code failed on execution.')

        return B_vec, points


    def save_read_out_domain(self, data_key, pnt1, pnt2, nsurf, fileName='./poincare_set.h5', quiet=True, clean=True):
        """ Save field line coordinate data for the specified number of field
        lines, with each field line initialized along a line between two
        points.  Makes hdf5 file.

        Parameters
        ----------
        data_key : str
            hdf5 data key to be saved.
        pnt1 : arr
            (r,z,t) point where read out line begins.
        pnt2 : arr
            (r,z,t) point where read out line ends.
        nsurf : int
            Number of field lines to be followed between pnt1 and pnt2, end
            points inclused.
        fileName : str, optional
            Full path of hdf5 file to be saved. The default is './poincare_set.h5'.
        """
        # Poincare Points #
        m = (pnt2[1] - pnt1[1]) / (pnt2[0] - pnt1[0])
        b = -m*pnt1[0] + pnt1[1]

        r_dom = np.linspace(pnt1[0], pnt2[0], nsurf)
        z_dom = m * r_dom + b
        t_dom = np.empty(nsurf)
        t_dom[:] = pnt1[2]

        init_points = np.stack([r_dom, z_dom, t_dom], axis=1)

        # Run flf code #
        points = np.empty((nsurf, self.params['n_iter'], 4))
        for i, init in enumerate(init_points):
            print('\nInitial Point {0} of {1} : '.format(i+1,nsurf)+'(%.4f, %.4f, %.4f)' % (init[0],init[1],init[2]))
            pnts = self.execute_flf(init, quiet=quiet, clean=clean)
            if pnts is None:
                points[i,:,:] = np.nan
            else:
                points[i] = pnts

        hf_file = hf.File(fileName, 'a')
        hf_file.create_dataset(data_key, data=points)
        hf_file.close()


    def save_read_out_square(self, data_key, center_point, half_width, npts, fileName='./poincare_set.h5', quiet=True, clean=True):
        """ Save field square coordinate data for the specified number of field
        lines, with each field line initialized from a square grid centered
        around the center point.  Makes hdf5 file.

        Parameters
        ----------
        data_key : str
            hdf5 data key to be saved.
        center_point : arr
            (r,z,t) point around which square grid is centered..
        half_width : float
            Half the width of the square..
        npts : int
            Approximate number of field lines to be followed
            within the grid..
        fileName : str, optional
            Full path of hdf5 file to be saved. The default is './poincare_set.h5'.
        """
        # Poincare Points #
        sqrt_pnts = int(round(np.sqrt(npts)))
        npts = sqrt_pnts**2

        r_dom = np.linspace(center_point[0]-half_width, center_point[0]+half_width, sqrt_pnts)
        z_dom = np.linspace(center_point[1]-half_width, center_point[1]+half_width, sqrt_pnts)
        t_dom = center_point[2]

        r_grid, z_grid, t_grid = np.meshgrid(r_dom, z_dom, t_dom)
        init_points = np.stack((r_grid, z_grid, t_grid), axis=3)
        init_points = init_points.flatten().reshape(npts, 3)

        # Run flf code #
        points = np.empty((npts, self.params['n_iter'], 4))
        for i, init in enumerate(init_points):
            print('\nInitial Point {0} of {1}: '.format(i+1, npts)+'(%.4f, %.4f, %.4f)' % (init[0], init[1], init[2]))
            if i+1 == init_points.shape[0]:
                points[i] = self.execute_flf(init, quiet=quiet, clean=clean)
                if points is None:
                    raise RuntimeError('FLF code failed on execution.')
            else:
                points[i] = self.execute_flf(init, quiet=quiet, clean=clean)
                if points is None:
                    raise RuntimeError('FLF code failed on execution.')

        hf_file = hf.File(fileName, 'a')
        hf_file.create_dataset(data_key, data=points)
        hf_file.close()


    def save_read_out_line(self, data_key, init_point, fileName='./poincare_set.h5', quiet=True, clean=True):
        """ Save single field line coordinate data.  Makes hdf5 file.

        Parameters
        ----------
        data_key : str
            hdf5 data key to be saved.
        init_point : arr
            (r,z,t) point from which field line following will be done.
        fileName : str, optional
            Full path of hdf5 file to be saved. The default is
            './poincare_set.h5'.
        """
        ### Run flf code ###
        print('\nInitial Point : (%.4f, %.4f, %.4f)' % (init_point[0],init_point[1],init_point[2]))
        points = self.execute_flf(init_point, quiet=quiet, clean=clean)
        if points is None:
            raise RuntimeError('FLF code failed on execution.')

        hf_file = hf.File(fileName, 'a')
        hf_file.create_dataset(data_key, data=points)
        hf_file.close()


    def save_Bvec_data(self, data_key, fileName='./poincare_set.h5', quiet=True, clean=True):
        """ Save magnetic field vectors at the cylidrical points specified in
        the imported grid array.  Makes hdf5 file.

        Parameters
        ----------
        data_key : str
            HDF5 data key of imported grid.
        fileName : str, optional
            Full path of hdf5 file to import grid data from. The default is
            './poincare_set.h5'.

        Raises
        ------
        ValueError
            Imported grid data does not have proper dimensionality.
        """
        ### Import grid points ###
        hf_file = hf.File(fileName, 'r')
        points = hf_file[data_key][:]
        hf_file.close()

        ndim = points.ndim
        if ndim == 3:
            npts = points.shape[1]
        elif ndim == 2:
            npts = points.shape[0]
        else:
            raise ValueError(fileName+' data set ('+data_key+') has dimensions {}, but should be 2 or 3'.format(ndim))

        ### Calculate B vector at grid points ###
        chg_dict = {'general_option' : 2,
                    'points_number' : npts}

        self.change_params(chg_dict)

        if ndim == 3:
            vec_points = np.empty(points.shape)
            for i in range(points.shape[0]):
                vec_points[i] = self.execute_flf(points[i,0::,0:3], quiet=quiet, clean=clean)
                if vec_points is None:
                    raise RuntimeError('FLF code failed on execution.')
        elif ndim == 2:
            vec_points = self.execute_flf(points, quiet=quiet, clean=clean)
            if vec_points is None:
                raise RuntimeError('FLF code failed on execution.')

        ### Save Vector data ###
        hf_file = hf.File(fileName, 'a')
        hf_file.create_dataset(data_key+' Bvec', data=vec_points)
        hf_file.close()


if __name__ == '__main__':
    # main, aux = functions.readCrntConfig('0-3-84')
    main = np.ones(6)
    aux = np.zeros(6)
    crnt = -10722. * np.r_[main, 14*aux]
    mod_dict = {'mgrid_currents': ' '.join(['{}'.format(c) for c in crnt])}

    flf = flf_wrapper('HSX')
    flf.change_params(mod_dict)
    flf.set_transit_parameters(45, 1)

    init_point = np.array([1.45, 0., 0.]) 
    points = flf.execute_flf(init_point, quiet=True, clean=True)
    print(points)
