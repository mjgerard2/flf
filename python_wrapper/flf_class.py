import os, sys, warnings, shlex, subprocess
import numpy as np
import h5py as hf
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.path as mpltPath

from scipy.stats import chisquare
from sklearn.linear_model import LinearRegression

wrapper_dir = os.path.dirname(os.path.realpath(__file__))
flf_dir = os.path.join(wrapper_dir, '..')
sys.path.append(wrapper_dir)
import exp_params as ep


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

    set_transit_parameters(dstp, rots)
        Define the toroidal transit parameters.

    execute_flf(init_point, quiet=True, clean=True)
        Execultes field line following code on specified input points.

    read_out_point(init_point, quiet=True, clean=True)
        Makes Poincare plot by following field line found at input point.

    read_out_set(init_points, quiet=True, clean=True)
        Makes Poincare plot by following field line found at input point.

    read_out_domain(pnt1, pnt2, nsurf, quiet=True, clean=True)
        Save field line coordinate data for the specified number of field
        lines, with each field line initialized along a line between two
        points.  Makes hdf5 file.

    read_out_square(center_point, half_width, npts, quiet=True, clean=True)
        Save field square coordinate data for the specified number of field
        lines, with each field line initialized from a square grid centered
        around the center point.  Makes hdf5 file.

    plotting(fontsize=14, labelsize=16, linewidth=2)
        Define plotting parameters.

    plot_poincare_data(save_path=None)
        Plot poincare data.

    save_poincare_data(save_path, data_key)
        Save Poincare data as hdf5 file.

    save_exe_data(save_path, data_key)
        Save data output from the flf code as hdf5 file.

    find_lcfs(init_point, dec_limit, r_limits, scan_res_limit=2, high_precission=True)
        Approximately locate the LCFS and Magnetic Axis, represented as
        (r,z,t) points initialized in the flf code.

    find_magnetic_axis(init_point, dec_limit)
        Approximately locate the Magnetic Axis, represented as (r,z,t)
        points initialized in the flf code.

    flux_surface_dimensionality(points)
        Calculate the flux surface pointwise dimensional and return flux surface score.
        Ergodic field lines typically return values above 0.1

    flf_surface(ma_points, surf_points)
        Generates a B-field surface from the flf data provided by the
        magnetic axis and the input initial field point.

    generate_descur_input(ma_point, surf_point, save_path, pol_pnts=20, tor_pnts=100, quiet=True, clean=True)
        Generate DESCUR input data for a flux surface.

    calc_psiEdge(ma_pnt, lcfs_pnt, Npts=1000, upts=100, Bidx='Bmod', plot=True, quiet=True, clean=True)
        A method for calculating the enclosed toroidal flux within the flux surface
        traced out by following the seeded field line specified in the input.

    continuous_trajectories_2D(init_point, d0, npts, dstp, rots)
        Calculate the continuous Lyapunov trajectories around the specified inital point.
        Trajectories are through a 2D phase-space, defined by the R and Z cylindrical coordinates.
        The toroidal angle is the time-like variable.

    continuous_trajectories_3D(init_point, d0, npts, darc, total_length)
        Calculate the continuous Lyapunov trajectories around the specified inital point.
        Trajectories are through a 3D phase-space, defined by the R, Z, and phi cylindrical coordinates.
        The field-line arclength is the time-like varaible.

    calc_lyapunov_exponents(init_point, d0, npts, dstp, rots, ndims=3, mode='continuous')
        Calculate both the forward and backward Lyapunov exponents in the specified
        dimensional phase-space.

    read_Bvec_data(data_key, fileName='./poincare_set.h5', quiet=True, clean=True)
        Save magnetic field vectors at the cylidrical points specified in
        the imported grid array.  Makes hdf5 file.
    """

    def __init__(self, exp, file_dict=None):
        if file_dict is None:
            file_dict = {'namelist': 'flf.namelist',
                         'input': 'point_temp.in'}

        # define directories #
        self.flf_dir = flf_dir
        self.wrapper_dir = wrapper_dir

        # define executable and file names #
        self.exe = os.path.join(self.flf_dir, 'flf')
        self.namelist = os.path.join(self.wrapper_dir, file_dict['namelist'])
        self.in_path = os.path.join(self.wrapper_dir, file_dict['input'])
        self.run_cmd = shlex.split('%s %s' % (self.exe, self.namelist))

        self.exp = exp
        self.params = ep.params[exp]

        with open(self.namelist, 'w') as file:
            file.write('&flf\n' +
                       '  general_option = {}\n'.format(self.params['general_option']) +
                       '  points_file= \'{}\'\n'.format(self.in_path) +
                       '  points_number={}\n'.format(self.params['points_number']) +
                       '  follow_type={}\n'.format(self.params['follow_type']) +
                       '  points_dphi= {}\n'.format(self.params['points_dphi']) +
                       '  n_iter = {}\n'.format(self.params['n_iter']) +
                       '  output_coils = {}\n'.format(self.params['output_coils']) +
                       '  log_freq = {}\n'.format(self.params['log_freq']) +
                       '  vessel_file = \'{}\''.format(self.params['vessel_file']) +
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
                       '  n_iter = {}\n'.format(self.params['n_iter']) +
                       '  output_coils = {}\n'.format(self.params['output_coils']) +
                       '  log_freq = {}\n'.format(self.params['log_freq']) +
                       '  vessel_file = \'{}\''.format(self.params['vessel_file']) +
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
        if self.params['follow_type'] == 1:
            dphi = dstp * (np.pi/180)
            stps = int(2 * np.pi / dphi)
            npts = int(rots*stps)
            mod_dict = {'points_dphi': dphi,
                        'n_iter': npts}
        elif self.params['follow_type'] == 2:
            mod_dict = {'points_dphi': dstp,
                        'n_iter': rots}
        self.change_params(mod_dict)

    def execute_flf(self, init_point, quiet=True, clean=True):
        """ Execultes field line following code on specified input points.

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
        nItr = self.params['n_iter']+1

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
        if len(flf_err) > 0 and len(flf_out) == 2:
            if clean:
                if os.path.isfile(self.namelist):
                    cmnd = shlex.split('rm %s' % self.namelist)
                    subprocess.run(cmnd)
                if os.path.isfile(self.in_path):
                    cmnd = shlex.split('rm %s' % self.in_path)
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
            if os.path.isfile(self.namelist):
                cmnd = shlex.split('rm %s' % self.namelist)
                subprocess.run(cmnd)
            if os.path.isfile(self.in_path):
                cmnd = shlex.split('rm %s' % self.in_path)
                subprocess.run(cmnd)

        # Return flf results #
        return points

    def read_out_point(self, init_point, quiet=True, clean=True):
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
        nitr = self.params['n_iter']
        stps = int(2 * np.pi / self.params['points_dphi'] )
        rots = int( (self.params['n_iter'] * self.params['points_dphi']) / (2*np.pi) )
        idx_stps = [int(i*stps) for i in range(rots)]

        # Run flf code #
        print('Initial Point : (%.4f, %.4f, %.4f)\n' % (init_point[0],init_point[1],init_point[2]))
        points = self.execute_flf(init_point, quiet=quiet, clean=clean)
        if points is None:
            self.exe_points = np.full((nitr, 4), np.nan)
            self.poin_points = np.full((rots, 4), np.nan)
        else:
            self.exe_points = points
            self.poin_points = self.exe_points[idx_stps]

    def read_out_set(self, init_points, quiet=True, clean=True):
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
        nitr = self.params['n_iter']
        stps = int(2 * np.pi / self.params['points_dphi'])
        rots = int( (nitr * self.params['points_dphi']) / (2*np.pi) )
        idx_stps = [int(i*stps) for i in range(rots)]

        # Run flf code #
        self.exe_points = np.full((init_points.shape[0], nitr+1, 4), np.nan)
        self.poin_points = np.full((init_points.shape[0], rots, 4), np.nan)
        for idx, init_point in enumerate(init_points):
            poin_pnts = np.empty((rots, 4))
            print('Initial Point {0:0.0f} of {1:0.0f} : ({2:0.4f}, {3:0.4f}, {4:0.4f})\n'.format(idx+1, init_points.shape[0], init_point[0],init_point[1],init_point[2]))
            points = self.execute_flf(init_point, quiet=quiet, clean=clean)
            if not points is None:
                self.exe_points[idx] = points
                self.poin_points[idx] = points[idx_stps]

    def read_out_domain(self, pnt1, pnt2, nsurf, quiet=True, clean=True):
        """ Save field line coordinate data for the specified number of field
        lines, with each field line initialized along a line between two
        points.  Makes hdf5 file.

        Parameters
        ----------
        pnt1 : arr
            (r,z,t) point where read out line begins.
        pnt2 : arr
            (r,z,t) point where read out line ends.
        nsurf : int
            Number of field lines to be followed between pnt1 and pnt2, end
            points inclused.
        """
        # Read Out parameters #
        nitr = self.params['n_iter']
        stps = int(2 * np.pi / self.params['points_dphi'])
        rots = int( (nitr * self.params['points_dphi']) / (2*np.pi) )
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
        self.exe_points = np.full((nsurf, nitr+1, 4), np.nan)
        self.poin_points = np.full((nsurf, rots, 4), np.nan)
        for i, init in enumerate(init_points):
            print('Initial Point {0} of {1} : ({2:0.4f}, {3:0.4f}, {4:0.4f})\n'.format(i+1,nsurf, init[0],init[1],init[2]))
            points = self.execute_flf(init, quiet=quiet, clean=clean)
            if not points is None:
                self.exe_points[i] = points
                self.poin_points[i] = points[idx_stps]

    def read_out_square(self, center_point, half_width, npts, quiet=True, clean=True):
        """ Save field square coordinate data for the specified number of field
        lines, with each field line initialized from a square grid centered
        around the center point.  Makes hdf5 file.

        Parameters
        ----------
        center_point : arr
            (r,z,t) point around which square grid is centered..
        half_width : float
            Half the width of the square..
        npts : int
            Approximate number of field lines to be followed within the grid.
        """
        # Read Out parameters #
        nitr = self.params['n_iter']
        stps = int(2 * np.pi / self.params['points_dphi'])
        rots = int( (npts * self.params['points_dphi']) / (2*np.pi) )
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

        # Run flf code #]
        self.exe_points = np.full((npts, nitr, 4), np.nan)
        self.poin_points = np.full((npts, rots, 4), np.nan)
        for i, init in enumerate(init_points):
            print('\nInitial Point {0} of {1} : ({2:0.4f}, {3:0.4f}, {4:0.4f})\n'.format(i+1, npts, init[0], init[1], init[2]))
            points = self.execute_flf(init, quiet=quiet, clean=clean)
            if not points is None:
                self.exe_points[i] = points
                self.poin_points[i] = points[idx_stps]

    def plotting(self, fontsize=14, labelsize=16, linewidth=2):
        """ Define plotting parameters.

        Parameters
        ----------
        fontsize: int, optional
            Size of fonts in figure. Default is 14.
        labelsize: int, optional
            Size of labels in figure. Default is 16.
        linewidth: int, optional
            Line width in figure. Default is 2.
        """
        plt.close('all')

        font = {'family': 'sans-serif',
                'weight': 'normal',
                'size': fontsize}

        mpl.rc('font', **font)

        mpl.rcParams['axes.labelsize'] = labelsize
        mpl.rcParams['lines.linewidth'] = linewidth

    def plot_poincare_data(self, save_path=None):
        """ Plot poincare data.
        """
        self.plotting()
        fig, ax = plt.subplots(tight_layout=True, figsize=(4,8))
        ax.set_aspect('equal')

        # plot data #
        poin_data = self.poin_points
        if poin_data.ndim == 3:
            for data in poin_data:
                # smap = ax.scatter(data[:, 0], data[:, 1], c=data[:, 3], s=1)
                ax.scatter(data[:, 0], data[:, 1], c='k', s=0.25)
        elif poin_data.ndim == 2:
            # smap = ax.scatter(poin_data[:, 0], poin_data[:, 1], c=poin_data[:, 3], s=1)
            ax.scatter(poin_data[:, 0], poin_data[:, 1], c='k', s=1, zorder=10)
            ax.scatter(poin_data[0, 0], poin_data[0, 1], c='k', s=100, marker='x')

        # axis labels #
        ax.set_xlabel('R/m')
        ax.set_ylabel('Z/m')

        #cbar = fig.colorbar(smap, ax=ax)
        #cbar.ax.set_ylabel('B/T')

        # axis grid #
        ax.grid()

        # save/show#
        if save_path is None:
            plt.show()
        elif save_path == 'Return':
            return plt, fig, ax
        else:
            plt.savefig(save_path)

    def save_poincare_data(self, save_path, data_key):
        """ Save Poincare data as hdf5 file.

        Parameters
        ----------
        save_path: str
            Global path to where the hdf5 file will be saved.
        data_key: str
            Hdf5 data key that the data will be saved under.
        """
        with hf.File(save_path, 'a') as hf_:
            hf_.create_dataset(data_key, data=self.poin_points)

    def save_exe_data(self, save_path, data_key):
        """ Save data output from the flf code as hdf5 file.

        Parameters
        ----------
        save_path: str
            Global path to where the hdf5 file will be saved.
        data_key: str
            Hdf5 data key that the data will be saved under.
        """
        with hf.File(save_path, 'a') as hf_:
            hf_.create_dataset(data_key, data=self.exe_points)

    def find_lcfs(self, init_point, dec_limit, r_limits, scan_res_limit=2, high_precission=True):
        """ Approximately locate the LCFS and Magnetic Axis, represented as
        (r,z,t) points initialized in the flf code.

        Parameters
        ----------
        init_point: array
            Initial point {r,z,t} from which radial steps are taken until a field-line is found
            that does not leave the mgrid domain within the specified number of toroidal transits.
            Initial steps are in 0.1 increments, but are then reduced by orders of magnitude as
            the target surface is approached.
        dec_limit: int
            Number of decimal points in terminating step size.  This does not guarantee accuracy
            to this decimal point.
        r_limits: tuple
            Radial limits of domain over which to perform scan. First and second
            elements are the minimal and maximal R values, repsectively.
        scan_res_limit: int (optional)
            The decimal point resolution the scan will go to. Default is 2.
        high_precission: Bool (optional)
            If True, then the pointwise dimension will be calculated for the LCFS to
            determine if the flux surface is ergodic or not. Default is True.
        """
        print('----------------\n'
              'Looking for LCFS\n'+
              '----------------\n')
        init_point = np.round(init_point, decimals=dec_limit)
        scan_res = 1
        first_contact = False
        exceed_limit = False
        r_init = init_point[0]
        self.read_out_point(init_point)
        while scan_res <= scan_res_limit:
            dr_dom = np.logspace(-scan_res, -dec_limit, 1+dec_limit-scan_res)
            for dr_scl in dr_dom:
                stp_cnt = -1
                if not first_contact and not exceed_limit:
                    if np.isnan(self.exe_points).any():
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
                    self.read_out_point(init_point)
                    are_nans = np.isnan(self.exe_points).any()
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
                    print('{0} < r_init < {1}\n'.format(round(r_min, dec_limit), round(r_max, dec_limit)))

        if (Dr_max <= 0) or (Dr_min <= 0):
            self.lcfs_point = np.full(3, np.nan)
            warnings.warn("LCFS not found within radial limits.")
        else:
            init_point[0] = r_min
            if high_precission:
                print('-------------------------------\n'+
                      'Beginning high precission phase\n'+
                      '-------------------------------\n')
                self.read_out_point(init_point)
                fs_check = self.flux_surface_dimensionality(self.poin_points)
                print('flux surface dimension = {}\n'.format(fs_check+1))
                while np.abs(fs_check) > 0.05:
                    init_point[0] -= np.abs(dr)
                    self.read_out_point(init_point)
                    fs_check = self.flux_surface_dimensionality(self.poin_points)
                    print('flux surface dimension = {}\n'.format(fs_check+1))

            self.lcfs_point = np.round(init_point, decimals=dec_limit)
            print('---------------\n'
                  'LCFS Identified\n'+
                  '---------------\n'+
                  'FLF Point ~ ({}, {}, {})\n'.format(self.lcfs_point[0], self.lcfs_point[1], self.lcfs_point[2]))

    def find_magnetic_axis(self, init_point, dec_limit):
        """  Approximately locate the Magnetic Axis, represented as (r,z,t)
        points initialized in the flf code.

        Parameters
        ----------
        init_point : arr
            Initial guess for magnetic axis.  This needn't be a good guess, but
            it should be on a closed flux surface for best results.
        dec_limit : int
            Number of decimal points in approximation.  This does not guarantee
            accuracy to this decimal point.
        """
        print('-------------------------\n'
              'Looking for Magnetic Axis\n'+
              '-------------------------\n')
        init_point = np.round(init_point, decimals=dec_limit)
        self.read_out_point(init_point)
        ma_point = np.round(np.r_[np.mean(self.poin_points[0::,0:2], axis=0), init_point[2]], decimals=dec_limit)

        pnt_sep = np.max(self.poin_points[0::,0]) - np.min(self.poin_points[0::,0])
        pnt_scl = np.floor(np.log10(pnt_sep))

        dist = np.linalg.norm(ma_point - init_point)
        dist_scl = np.floor(np.log10(dist))

        path = mpltPath.Path(self.poin_points[0::,0:2])
        inside = path.contains_point(ma_point[0:2])

        cnt = 0
        ma_points = []
        while dist_scl >= -dec_limit:
            if not inside and dist_scl < pnt_scl:
                ma_point[0] = init_point[0] - 10**(dist_scl-pnt_scl-1)

            init_point = ma_point
            self.read_out_point(init_point)
            ma_point = np.round(np.r_[np.mean(self.poin_points[0::,0:2], axis=0), init_point[2]], decimals=dec_limit)

            ma_points.append(ma_point)
            if len(ma_points) > 10:
                ma_point_chk = np.array(ma_points)
                resR = np.correlate(ma_point_chk[0::,0], ma_point_chk[0::,0], mode='full')

                hlf_idx = int(0.5*len(resR))
                print('   Auto Correlation = {0:0.3f}\n'.format(resR[hlf_idx]))
                if (resR[hlf_idx] > 0.5):
                    ma_point = np.round(np.mean(ma_point_chk[10::], axis=0), decimals=dec_limit)
                    ma_points = [ma_point]
                    cnt+=1

                    print('   {0} : ({1:0.4f}, {2:0.4f}, {3:0.4f})\n'.format(cnt, ma_point[0], ma_point[1], ma_point[2]))
                    if cnt == 3:
                        break

            pnt_sep = np.max(self.poin_points[0::,0]) - np.min(self.poin_points[0::,0])
            pnt_scl = np.floor(np.log10(pnt_sep))

            dist = np.linalg.norm(ma_point - init_point)
            if dist == 0:
                path = mpltPath.Path(self.poin_points[0::,0:2])
                inside = path.contains_point(ma_point[0:2])
                break
            else:
                path = mpltPath.Path(self.poin_points[0::,0:2])
                inside = path.contains_point(ma_point[0:2])
                dist_scl = np.floor(np.log10(dist))

        self.ma_point = ma_point
        print('------------------------\n'
              'Magnetic Axis Identified\n'+
              '------------------------\n'+
              'FLF Point ~ ({}, {}, {})\n'.format(self.ma_point[0], self.ma_point[1], self.ma_point[2]))

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

    def flf_surface(self, ma_points, surf_points, mod_params):
        """ Generates a B-field surface from the flf data provided by the
        magnetic axis and the input initial field point.

        Parameters
        ----------
        ma_pnt : arr
            {r,z,t} point where magnetic axis is initialized.
        surf_pnt : arr
            {r,z,t} point where field line following will be initiated.  This
            data will define the field surface that will be returned.

        Returns
        -------
        arr
            Array containing the surface field data, returned in array with
            shape {360, rots, 4}.  The first index is the toroidal points, the
            second index is the poloidal points and the third index provides
            the R, Z, mod B and poloidal angles.
        """
        # Read Out parameters #
        # nitr = self.params['n_iter']
        # stps = int(2 * np.pi / self.params['points_dphi'])
        # rots = int( (nitr * self.params['points_dphi']) / (2*np.pi) )
        # idx_stps = [int(i*stps) for i in range(rots)]
        rots = int( (mod_params['n_iter'] * mod_params['points_dphi']) / (2*np.pi) )
        stps = int(surf_points.shape[0] / rots)
        idx_base = [int(i*stps) for i in range(rots)]


        v_dom = surf_points[0::,2]
        surf_ordered = np.empty((stps+1, rots, 5))
        for v_idx, v in enumerate(ma_points[0::,2]):
            idx_stps = np.argmin(np.abs(v_dom - v)) + idx_base  # [int(i*stps) for i in range(rots)]
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

        ma_ordered = np.empty(ma_points.shape)
        ma_ordered = np.stack((ma_points[0:,0], ma_points[0:,2], ma_points[0:,1]), axis=1)

        return surf_ordered, ma_ordered

    def generate_descur_input(self, ma_point, surf_point, save_path, pol_pnts=20, tor_pnts=80, quiet=True, clean=True):
        """ Generate DESCUR input data for a flux surface.

        Parameters
        ----------
        ma_point: arr
            Initial point for magnetic axis.
        surf_point: arr
            Initial point for flux surface that will be converted to a DESCUR
            input file.
        save_path: str
            global path to where descur input will be saved.
        pol_pnts: int, optional
            Number of poloidal points in DESCUR input. The default is 20.
        tor_pnts: int, optional
            Number of toroidal points in DESCUR input. The default is 80.
        """
        nfp = int(self.params['num_periods'])
        u_pnts = int(pol_pnts * nfp)
        v_pnts = int(tor_pnts / nfp)
        n_pnts = int(u_pnts * v_pnts)

        if tor_pnts % nfp != 0 and pol_pnts % 10 != 0:
            raise ValueError('tor_pnts and pol_pnts must be multiples of the number of field periods ({0:0.0f}).'.format(nfp))

        dphi = (2*np.pi)/tor_pnts
        stps = tor_pnts
        mod_dict = {'general_option': 1,
                    'points_number': 1,
                    'points_dphi': dphi,
                    'n_iter': stps}

        v_dom = np.arange(0, 2*np.pi, dphi)

        self.change_params(mod_dict)
        ma_points = self.execute_flf(ma_point, quiet=quiet, clean=clean)
        if ma_points is None:
            raise RuntimeError('While following the magnetic axis, the FLF code failed.')

        npts = int(pol_pnts * stps)
        mod_dict = {'points_dphi' : dphi,
                    'n_iter' : npts}

        self.change_params(mod_dict)
        surf_points = self.execute_flf(surf_point, quiet=quiet, clean=clean)
        if surf_points is None:
            raise RuntimeError('While following the surface field line, the FLF code failed.')

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

        with open(save_path, 'w') as file:
            file.write('{0} {1} {2}\n'.format(u_pnts, v_pnts, nfp))
            for n in range(n_pnts):
                file.write('{0:0.6f} {1:0.6f} {2:0.6f} \n'.format(fit_data[n,0], fit_data[n,1], fit_data[n,2]))

    def calc_psiEdge(self, ma_pnt, lcfs_pnt, Npts=1000, upts=100, Bidx='Bmod', plot=True, quiet=True, clean=True):
        """ A method for calculating the enclosed toroidal flux within the flux surface
        traced out by following the seeded field line specified in the input.

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

        self.read_out_point(lcfs_pnt)
        points = self.poin_points[:,0:3] - ma_pnt

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

        if plot:
            self.plotting()
            fig, ax = plt.subplots(1, 1, tight_layout=True)

            ax.set_aspect('equal')

            # ax.set_title(r'$\Psi_{edge}$'+' : {0:0.2f}'.format(1e4 * area * np.mean(points_out[0:,3]))+r' [$T\cdot cm^2$]')
            ax.set_title(r'$\Psi_{edge}$'+' : {0:0.4f}'.format(area * np.mean(points_out[0:,3]))+r' [$T\cdot m^2$]')
            ax.scatter(points[0:,0] * 1e2, points[0:,1] * 1e2, c='k', s=5, zorder=10)

            s_map = ax.scatter( sample_points[0:,0] * 1e2, sample_points[0:,1] * 1e2, c=points_out[0:,bidx], s=10, cmap='jet')

            cbar = fig.colorbar(s_map, ax=ax)
            cbar.ax.set_ylabel(blab+' [T]')

            ax.set_xlabel('R [cm]')
            ax.set_ylabel('Z [cm]')

            ax.grid()
            plt.show()

        return psi_edge

    def continuous_trajectories_2D(self, init_point, d0, npts, dstp, rots):
        """ Calculate the continuous Lyapunov trajectories around the specified inital point.
        Trajectories are through a 2D phase-space, defined by the R and Z cylindrical coordinates.
        The toroidal angle is the time-like variable.

        Parameters
        ----------
        init_point : arr
            Inital point from which exponent will be calculated.
        d0 : float
            Initial separatial magnitude for dispacement in phase space.
        npts : int, optional
            The number of displaced points from which the exponent will be averaged.
            Default is 3.
        dstp : float
            The angular step size, in degrees.
        rots : float
            The number of toroidal rotations.

        Returns
        -------
        arr : toroidal domain over which exponent is calculated
        arr : trajectory separations over forward trajectries through phase space
        arr : trajectory separations over backward trajectories through phase space
        """
        dphi = dstp * (np.pi/180)
        stps = round(2 * np.pi / dphi)
        nitr = round(rots*stps)
        tor_ang = init_point[2]
        pnt_ang = np.linspace(0, 2*np.pi, nitr, endpoint=False)

        # follow forward #
        mod_dict = {'follow_type': 1,
                    'n_iter': nitr,
                    'points_dphi': dphi,
                    'points_number': 1}
        self.change_params(mod_dict)
        pnt_for = self.execute_flf(init_point)

        # follow backward #
        mod_dict['points_dphi'] = -dphi
        self.change_params(mod_dict)
        pnt_bak = self.execute_flf(init_point)

        # check for nans #
        if (pnt_for is None) or (pnt_bak is None):
            return None, None, None
        elif np.isnan(pnt_for).any() or np.isnan(pnt_bak).any():
            not_nan = np.isnan(pnt_for[:,0])
            pnt_for = pnt_for[~not_nan]
            not_nan = np.isnan(pnt_bak[:,0])
            pnt_bak = pnt_bak[~not_nan]
            if pnt_for.shape[0] <= 2 or pnt_bak.shape[0] <= 2:
                return None, None, None
            phi_max = min(np.max(pnt_for[:,2]-tor_ang), np.max(np.abs(tor_ang-pnt_bak[:,2])))
            pnt_for = pnt_for[pnt_for[:,2] <= tor_ang+phi_max+.5*dphi]
            pnt_bak = pnt_bak[pnt_bak[:,2] >= tor_ang-phi_max-.5*dphi]
        else:
            phi_max = 2*np.pi*rots

        mod_dict['points_dphi'] = dphi
        mod_dict['n_iter'] = round(phi_max/dphi)
        self.change_params(mod_dict)
        phi_dom = tor_ang + np.linspace(0, dphi*mod_dict['n_iter'], mod_dict['n_iter']+1)
        dist_for = np.full((npts, pnt_for.shape[0]), np.nan)
        dist_bak = np.full((npts, pnt_bak.shape[0]), np.nan)
        for k in range(npts):
            d_shft = d0*np.array([np.cos(pnt_ang[k]), np.sin(pnt_ang[k]), 0])
            pnts_for = self.execute_flf(init_point+d_shft)
            mod_dict['points_dphi'] = -dphi
            self.change_params(mod_dict)
            pnts_bak = self.execute_flf(init_point+d_shft)
            if (pnts_for is None) or (pnts_bak is None):
                continue
            elif np.isnan(pnts_for).any() or np.isnan(pnts_bak).any():
                not_nan = np.isnan(pnts_for[:,0])
                pnts_for = pnts_for[~not_nan]
                not_nan = np.isnan(pnts_bak[:,0])
                pnts_bak = pnts_bak[~not_nan]
                phi_chk = min(np.max(pnts_for[:,2]-tor_ang), np.max(np.abs(tor_ang-pnts_bak[:,2])))
                pnts_for = pnts_for[pnts_for[:,2] <= tor_ang+phi_chk+.5*dphi]
                pnts_bak = pnts_bak[pnts_bak[:,2] >= tor_ang-phi_chk-.5*dphi]
                if pnts_for.shape[0] <= 2 or pnts_bak.shape[0] <= 2:
                    continue
                pnt_for_use = pnt_for[pnt_for[:,2] <= tor_ang+phi_chk+.5*dphi]
                pnt_bak_use = pnt_bak[pnt_bak[:,2] >= tor_ang-phi_chk-.5*dphi]
            else:
                pnt_for_use = pnt_for
                pnt_bak_use = pnt_bak

            dfor = np.linalg.norm(pnt_for_use[:, 0:2] - pnts_for[:, 0:2], axis=1)/d0
            dbak = np.linalg.norm(pnt_bak_use[:, 0:2] - pnts_bak[:, 0:2], axis=1)/d0
            dist_for[k, 0:dfor.shape[0]] = dfor
            dist_bak[k, 0:dbak.shape[0]] = dbak

        return phi_dom, dist_for, dist_bak

    def continuous_trajectories_3D(self, init_point, d0, npts, darc, total_length):
        """ Calculate the continuous Lyapunov trajectories around the specified inital point.
        Trajectories are through a 3D phase-space, defined by the R, Z, and phi cylindrical coordinates.
        The field-line arclength is the time-like varaible.

        Parameters
        ----------
        init_point : arr
            Inital point from which exponent will be calculated.
        d0 : float
            Initial separatial magnitude for dispacement in phase space.
        npts : int, optional
            The number of displaced points from which the exponent will be averaged.
            Default is 3.
        darc : float
            The field-line arclength step size in meters.
        total_length : float, optional
            The distance a field line will be followed in meters. Default is 50 m.

        Returns
        -------
        arr : toroidal domain over which exponent is calculated
        arr : trajectory separations over forward trajectries through phase space
        arr : trajectory separations over backward trajectories through phase space
        """
        niter = round(float(total_length)/darc)
        pnt_ang = np.linspace(0, 2*np.pi, npts, endpoint=False)

        # follow forward #
        mod_dict = {'follow_type': 2,
                    'n_iter': niter,
                    'points_dphi': darc,
                    'points_number': 1}
        self.change_params(mod_dict)
        pnt_for = self.execute_flf(init_point)

        # follow backward #
        mod_dict['points_dphi'] = -darc
        self.change_params(mod_dict)
        pnt_bak = self.execute_flf(init_point)

        # check for nans #
        if (pnt_for is None) or (pnt_bak is None):
            return None, None, None
        elif np.isnan(pnt_for).any() or np.isnan(pnt_bak).any():
            not_nan = np.isnan(pnt_for[:,0])
            pnt_for = pnt_for[~not_nan]
            not_nan = np.isnan(pnt_bak[:,0])
            pnt_bak = pnt_bak[~not_nan]
            if pnt_for.shape[0] <= 2 or pnt_bak.shape[0] <= 2:
                return None, None, None
            else:
                arc_max = min(pnt_for[1::].shape[0]*darc, pnt_bak[1::].shape[0]*darc)
                if pnt_for.shape[0] > pnt_bak.shape[0]:
                    pnt_for = pnt_for[0:pnt_bak.shape[0]]
                elif pnt_bak.shape[0] > pnt_for.shape[0]:
                    pnt_bak = pnt_bak[0:pnt_for.shape[0]]
        else:
            arc_max = total_length

        niter = round(arc_max/darc)
        mod_dict['n_iter'] = niter
        self.change_params(mod_dict)

        dist_for = np.full((npts, niter+1), np.nan)
        dist_bak = np.full((npts, niter+1), np.nan)
        arc_dom = np.arange(0, (niter+1)*darc, darc)
        for k in range(npts):
            d_shft = d0*np.array([np.cos(pnt_ang[k]), np.sin(pnt_ang[k]), 0])
            mod_dict['points_dphi'] = darc
            self.change_params(mod_dict)
            pnt_for_shft = self.execute_flf(init_point+d_shft)
            mod_dict['points_dphi'] = -darc
            self.change_params(mod_dict)
            pnt_bak_shft = self.execute_flf(init_point+d_shft)
            if (pnt_for_shft is None) or (pnt_bak_shft is None):
                continue
            elif np.isnan(pnt_for_shft).any() or np.isnan(pnt_bak_shft).any():
                not_nan = np.isnan(pnt_for_shft[:,0])
                pnt_for_shft = pnt_for_shft[~not_nan]
                not_nan = np.isnan(pnt_bak_shft[:,0])
                pnt_bak_shft = pnt_bak_shft[~not_nan]
                arc_chk = min(pnt_for_shft[1::].shape[0]*darc, pnt_bak_shft[1::].shape[0]*darc)
                if pnt_for_shft.shape[0] < pnt_bak_shft.shape[0]:
                    pnt_for_shft = pnt_for_shft[0:pnt_bak_shft.shape[0]]
                elif pnt_bak_shft.shape[0] < pnt_for_shft.shape[0]:
                    pnt_bak_shft = pnt_bak_shft[0:pnt_for_shft.shape[0]]
                # phi_chk = min(np.max(pnt_for_shft[:,2]-tor_ang), np.max(np.abs(tor_ang-pnt_bak_shft[:,2])))
                # pnt_for_shft = pnt_for_shft[pnt_for_shft[:,2] <= tor_ang+phi_chk+.5*dphi]
                # pnt_bak_shft = pnt_bak_shft[pnt_bak_shft[:,2] >= tor_ang-phi_chk-.5*dphi]
                if pnt_for_shft.shape[0] <= 2 or pnt_bak_shft.shape[0] <= 2:
                    continue
                else:
                    pnt_for_use = pnt_for[0:pnt_for_shft.shape[0]]
                    pnt_bak_use = pnt_bak[0:pnt_bak_shft.shape[0]]
                    # pnt_for_use = pnt_for[pnt_for[:,2] <= tor_ang+phi_chk+.5*dphi]
                    # pnt_bak_use = pnt_bak[pnt_bak[:,2] >= tor_ang-phi_chk-.5*dphi]
            else:
                pnt_for_use = pnt_for
                pnt_bak_use = pnt_bak

            dfor = np.sqrt(pnt_for_use[:,0]**2+pnt_for_shft[:,0]**2-2*pnt_for_use[:,0]*pnt_for_shft[:,0]*np.cos(pnt_for_use[:,2]-pnt_for_shft[:,2])+(pnt_for_use[:,1]-pnt_for_shft[:,1])**2)/d0
            dbak = np.sqrt(pnt_bak_use[:,0]**2+pnt_bak_shft[:,0]**2-2*pnt_bak_use[:,0]*pnt_bak_shft[:,0]*np.cos(pnt_bak_use[:,2]-pnt_bak_shft[:,2])+(pnt_bak_use[:,1]-pnt_bak_shft[:,1])**2)/d0
            dist_for[k, 0:dfor.shape[0]] = dfor
            dist_bak[k, 0:dbak.shape[0]] = dbak

        return arc_dom, dist_for, dist_bak

    def discrete_trajectories_2D(self, init_point, d0, dstp, rots):
        dphi = dstp * (np.pi/180)
        stps = round(2 * np.pi / dphi)
        nitr = round(rots*stps)
        tor_ang = init_point[2]

        # follow forward #
        mod_dict = {'n_iter': nitr,
                    'points_dphi': dphi,
                    'points_number': 1}
        self.change_params(mod_dict)
        pnt_for = self.execute_flf(init_point)

        # follow backward #
        mod_dict['points_dphi'] = -dphi
        self.change_params(mod_dict)
        pnt_bak = self.execute_flf(init_point)

        # check for nans #
        if (pnt_for is None) or (pnt_bak is None):
            return None
        elif np.isnan(pnt_for).any() or np.isnan(pnt_bak).any():
            not_nan = np.isnan(pnt_for[:,0])
            pnt_for = pnt_for[~not_nan, 0:3]
            not_nan = np.isnan(pnt_bak[:,0])
            pnt_bak = pnt_bak[~not_nan, 0:3]
            if (pnt_for.shape[0] < 3) or (pnt_bak.shape[0] < 3):
                return None
        else:
            pnt_for = pnt_for[:, 0:3]
            pnt_bak = pnt_bak[:, 0:3]

        # forward parameters #
        npts = pnt_for.shape[0]-1
        mod_dict = {'n_iter': 1,
                    'points_dphi': dphi,
                    'points_number': npts}
        self.change_params(mod_dict)

        # forward radial derivative #
        shift_vec = .5*d0*np.stack((np.ones(npts), np.zeros(npts), np.zeros(npts)), axis=1)
        pnt_for_plus = self.execute_flf(pnt_for[0:npts] + shift_vec)[:,1,0:2]
        pnt_for_mins = self.execute_flf(pnt_for[0:npts] - shift_vec)[:,1,0:2]
        rad_for_deriv = np.linalg.norm(pnt_for_plus-pnt_for_mins, axis=1)/d0

        # forward zed derivative #
        shift_vec = .5*d0*np.stack((np.zeros(npts), np.ones(npts), np.zeros(npts)), axis=1)
        pnt_for_plus = self.execute_flf(pnt_for[0:npts] + shift_vec)[:,1,0:2]
        pnt_for_mins = self.execute_flf(pnt_for[0:npts] - shift_vec)[:,1,0:2]
        zed_for_deriv = np.linalg.norm(pnt_for_plus-pnt_for_mins, axis=1)/d0

        # backward parameters #
        npts = pnt_bak.shape[0]-1
        mod_dict['points_dphi'] = dphi
        mod_dict['points_number'] = npts
        self.change_params(mod_dict)

        # backward radial derivative #
        shift_vec = .5*d0*np.stack((np.ones(npts), np.zeros(npts), np.zeros(npts)), axis=1)
        pnt_bak_plus = self.execute_flf(pnt_bak[0:npts] + shift_vec)[:,1,0:2]
        pnt_bak_mins = self.execute_flf(pnt_bak[0:npts] - shift_vec)[:,1,0:2]
        rad_bak_deriv = np.linalg.norm(pnt_bak_plus-pnt_bak_mins, axis=1)/d0

        # backward zed derivative #
        shift_vec = .5*d0*np.stack((np.zeros(npts), np.ones(npts), np.zeros(npts)), axis=1)
        pnt_bak_plus = self.execute_flf(pnt_bak[0:npts] + shift_vec)[:,1,0:2]
        pnt_bak_mins = self.execute_flf(pnt_bak[0:npts] - shift_vec)[:,1,0:2]
        zed_bak_deriv = np.linalg.norm(pnt_bak_plus-pnt_bak_mins, axis=1)/d0

        forward = np.mean(np.hypot(rad_for_deriv, zed_for_deriv))
        backward = np.mean(np.hypot(rad_bak_deriv, zed_bak_deriv))
        return .5*(forward+backward)*(dphi/(2*np.pi))*np.log(10)

    def calc_lyapunov_exponents(self, init_point, d0, npts, dstp, rots, ndims=3, mode='continuous'):
        """ Calculate both the forward and backward Lyapunov exponents in the specified
        dimensional phase-space.

        Parameters
        ----------
        init_point : arr
            Point at which exponents are to be calculated.
        d0 : float
            Inital displacement vector magnitude.
        ndims : int, optional
            Phase space dimensionality. Must be either 2 or 3. Default is 2.
        mode : str, optional
            Exponents can be calculated assuming continuous or discrete mappings.
            Default is Continuous.
        kwargs : dict, optional
            Dictionary containing key word arguments for the trajectory function.
            Default is an empty dictionary.
        """
        if mode == 'continuous':
            if ndims == 2:
                lyp_norm = np.log(10)/(2*np.pi)
                stp_dom, dist_for, dist_bak = self.continuous_trajectories_2D(init_point, d0, npts, dstp, rots)
                if stp_dom is None:
                    return None
            elif ndims == 3:
                lyp_norm = 0.1*np.log(10)
                stp_dom, dist_for, dist_bak = self.continuous_trajectories_3D(init_point, d0, npts, dstp, rots)
                if stp_dom is None:
                    return None
            else:
                raise KeyError('Calculations are not developed for a {}D phase-space'.format(ndims))

            lyp_exp = np.empty((npts, 2))
            for i in range(npts):
                nan_for = np.isnan(dist_for[i])
                nan_bak = np.isnan(dist_bak[i])
                dfor = dist_for[i, ~nan_for]
                dbak = dist_bak[i, ~nan_bak]
                xfor = stp_dom[0:dfor.shape[0]].reshape((-1,1)) * lyp_norm
                xbak = stp_dom[0:dbak.shape[0]].reshape((-1,1)) * lyp_norm
                if dfor.shape[0] > 0:
                    zero_for = np.where(dfor == 0)[0]
                    dfor[zero_for] = 1e-10
                    model_for = LinearRegression().fit(xfor, np.log(dfor))
                    lyp_for = model_for.coef_[0]
                else:
                    lyp_for = np.nan
                if dbak.shape[0] > 0:
                    zero_bak = np.where(dbak == 0)[0]
                    dbak[zero_bak] = 1e-10
                    model_bak = LinearRegression().fit(xbak, np.log(dbak))
                    lyp_bak = model_bak.coef_[0]
                else:
                    lyp_bak = np.nan
                lyp_exp[i] = [lyp_for, lyp_bak]

            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                lyp_return = np.nanmax(np.nanmean(lyp_exp, axis=1))

        else:
            raise KeyError('%s mode has not been developed.')

        return lyp_return

    def read_Bvec_data(self, data_key, fileName='./poincare_set.h5', quiet=True, clean=True):
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
        # Import grid points #
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

        # Calculate B vector at grid points #
        chg_dict = {'general_option' : 2,
                    'points_number' : npts}

        self.change_params(chg_dict)

        if ndim == 3:
            vec_points = np.empty(points.shape)
            for i, point in enumerate(points):
                exe_point = self.execute_flf(point[:, 0:3], quiet=quiet, clean=clean)
                if not exe_point is None:
                    vec_points[i] = exe_point[:, 0:3]
        elif ndim == 2:
            exe_point = self.execute_flf(points, quiet=quiet, clean=clean)
            if not exe_point is None:
                vec_points[i] = exe_point

        # Save Vector data #
        hf_file = hf.File(fileName, 'a')
        hf_file.create_dataset(data_key+' Bvec', data=vec_points)
        hf_file.close()


if __name__ == '__main__':
    # instantiate flf object #
    flf = flf_wrapper('HSX')
    flf.set_transit_parameters(1, 500)

    # modify coil currents from default #
    crnt_arr = -10722 * np.append(np.array([1, 1, 1, 1, 1, 1]), np.zeros(0))
    crnt_str = ' '.join([str(x) for x in crnt_arr])
    chg_dict = {'mgrid_currents': crnt_str}
    flf.change_params(chg_dict)

    # define FLF seed points in (R, Z, \phi) coordinates #
    pnt1 = np.array([1.45, 0, 0])
    pnt2 = np.array([1.5, 0, 0])

    # plot flux surfaces #
    flf.read_out_domain(pnt1, pnt2, 3, clean=False)
    plt, fig, ax = flf.plot_poincare_data(save_path='Return')
    plt.show()
