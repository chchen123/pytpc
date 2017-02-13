from scipy.interpolate import RectBivariateSpline
import numpy as np
import random

from pytpc.constants import degrees, p_mc2, pi
from pytpc.simulation import rutherford
from pytpc.relativity import cm_to_lab_frame, find_proton_params, find_kine_vert_en
from pytpc.utilities import find_vertex_energy, find_vertex_position_from_energy, euler_matrix

import logging
logger = logging.getLogger(__name__)


def parse_dsigmaiv_output(input_path, ratio_to_rutherford=False, Z1=None, Z2=None):
    """Parse the output from the DSIGMA_IV code to load a distribution.

    The input format should be as follows:

          E[MeV]          10          15          20      ...
        b 0.350E+00   0.594E+08   0.118E+08   0.377E+07   ...
        b 0.450E+00   0.359E+08   0.714E+07   0.228E+07   ...
        b 0.550E+00   0.241E+08   0.478E+07   0.153E+07   ...
        ...

    The first row gives the CM angles in degrees. The first column lists the CM energy of the projectile in MeV.
    The remaining values are differential cross sections in mb/sr.

    Parameters
    ----------
    input_path : string
        Path to the `DSIGMAIV-ELAS.DAT` file.
    ratio_to_rutherford : bool, optional
        If True, the ratio to the Rutherford cross section will be taken.
    Z1, Z2 : int, optional
        The charge numbers of the particles for the Rutherford cross section calculation. This is only needed if
        `ratio_to_rutherford` is True.

    Returns
    -------
    ens : ndarray
        The energies, in MeV.
    angs : ndarray
        The CM angles, in radians.
    vals : ndarray
        A 2D array of the cross sections, in mb/sr. Axis 0 corresponds to energy, and axis 1 corresponds to angle.
        If `ratio_to_rutherford` was True, the output is the ratio to the Rutherford cross section instead.

    """
    with open(input_path, 'r') as f:
        eline = f.readline()
        angs = [float(a) for a in eline.strip().split()[1:]]

        vals = []
        ens = []
        for line in f:
            elem = line.strip().split()[1:]
            ens.append(float(elem[0]))
            vals.append([float(s) for s in elem[1:]])

    ens = np.array(ens)
    angs = np.array(angs) * degrees
    vals = np.array(vals)

    assert len(ens) == vals.shape[0]
    assert len(angs) == vals.shape[1]

    if ratio_to_rutherford:
        engrid, anggrid = np.meshgrid(ens, angs, indexing='ij')
        ruthgrid = rutherford(anggrid, Z1, Z2, engrid)
        vals /= ruthgrid

    return ens, angs, vals


def make_random_beam(max_beam_angle, beam_origin_z, vertex_z, window_z=1.0):
    azi = random.uniform(0, 2 * pi)

    sample_min = 0.5 * (np.cos(max_beam_angle) + 1)
    pol = pi - np.arccos(2 * random.uniform(sample_min, 1) - 1)  # sample uniformly on a sphere, but going backwards

    beam_vector = np.array([
        np.sin(pol) * np.cos(azi),
        np.sin(pol) * np.sin(azi),
        np.cos(pol),
    ])

    slopes = beam_vector / np.abs(beam_vector[2])
    beam_origin = np.array([0., 0., beam_origin_z])

    vertex = beam_origin + slopes * (beam_origin_z - vertex_z)
    window = beam_origin + slopes * (beam_origin_z - window_z)

    transform = euler_matrix(azi, pi - pol, -azi)

    return beam_vector, vertex, window, transform


def uniform_param_generator(beam_enu0, beam_mass, beam_chg, proj_mass, proj_chg, max_beam_angle, beam_origin_z, gas, num_evts):
    num_good = 0
    while num_good < num_evts:
        z0 = random.uniform(0, 1)
        beamvec, vertex, window, transform = make_random_beam(max_beam_angle, beam_origin_z, z0)
        x0, y0 = vertex[:2]

        proj_azi = random.uniform(0, 2 * pi)
        proj_pol = random.uniform(pi / 2, pi)
        proj_vec = np.array([
            np.sin(proj_pol) * np.cos(proj_azi),
            np.sin(proj_pol) * np.sin(proj_azi),
            np.cos(proj_pol)
        ])
        proj_vec_trans = transform.T @ proj_vec
        pol0 = np.arctan2(np.hypot(proj_vec_trans[0], proj_vec_trans[1]), proj_vec_trans[2])
        azi0 = np.arctan2(proj_vec_trans[1], proj_vec_trans[0]) + pi  # Change domain from [-pi, pi] to [0, 2pi]

        tracklen = np.linalg.norm(vertex - window)

        # When finding vertex energy, use (1 - len) since find_vertex_energy takes 1-z as the length
        vert_en = find_vertex_energy(1 - tracklen, beam_enu0, beam_mass, beam_chg, gas)  # the total kinetic energy
        if vert_en > beam_enu0 * beam_mass:
            vert_en = 0.0

        _, proj_total_en = find_proton_params(
            th3=pi - proj_pol,
            m1=beam_mass * p_mc2,
            m2=proj_mass * p_mc2,
            m3=proj_mass * p_mc2,
            m4=beam_mass * p_mc2,
            T=vert_en
        )
        enu0 = proj_total_en - proj_mass * p_mc2

        if enu0 >= 0:
            yield np.array([x0, y0, z0, enu0, azi0, pol0])
            num_good += 1


def distribution_param_generator(ens, angs, xsecs, beam_enu0, beam_mass, beam_chg, proj_mass, gas, num_evts,
                                 en_bounds=None, ang_bounds=None):
    spline = RectBivariateSpline(ens, angs, xsecs)
    en_min, en_max = ens.min(), ens.max()
    ang_min, ang_max = angs.min(), angs.max()

    def bounds_are_valid(old, new):
        return old[0] <= new[0] and old[1] >= new[1]

    if en_bounds is not None:
        if bounds_are_valid((en_min, en_max), en_bounds):
            en_min, en_max = en_bounds
        else:
            raise ValueError('Energy bounds were outside domain of data')

    if ang_bounds is not None:
        if bounds_are_valid((ang_min, ang_max), ang_bounds):
            ang_min, ang_max = ang_bounds
        else:
            raise ValueError('Angular bounds were outside domain of data')

    encut = np.where((ens > en_min) & (ens < en_max))[0]
    angcut = np.where((angs > ang_min) & (angs < ang_max))[0]
    ixgrid = np.ix_(encut, angcut)

    norm_factor = xsecs[ixgrid].max()

    def rejection_sample():
        while True:
            x = np.random.uniform(en_min, en_max)
            y = np.random.uniform(ang_min, ang_max)
            z = np.random.uniform(norm_factor)

            if z < spline(x, y):
                return x, y

    num_good = 0
    while num_good < num_evts:
        x0 = random.gauss(0, 0.010)
        y0 = random.gauss(0, 0.010)
        azi0 = random.uniform(0, 2 * pi)

        enu0_cm, scat_ang_cm = rejection_sample()
        en0, scat_ang = cm_to_lab_frame(
            T_recoil_cm=enu0_cm * proj_mass,
            cm_angle=scat_ang_cm,
            proj_mass=beam_mass * p_mc2,
            target_mass=proj_mass * p_mc2,
        )
        enu0 = en0 / proj_mass
        pol0 = pi - scat_ang

        vert_en = find_kine_vert_en(
            mproj=beam_mass * p_mc2,
            mtarg=proj_mass * p_mc2,
            scat_ang=scat_ang,
            recoil_ke=enu0,
        )
        z0 = find_vertex_position_from_energy(
            vertex_enu0=vert_en / beam_mass,
            beam_enu0=beam_enu0,
            beam_mass=beam_mass,
            beam_chg=beam_chg,
            gas=gas,
        )

        if vert_en / beam_mass <= beam_enu0 and z0 > 0:
            yield np.array([x0, y0, z0, enu0, azi0, pol0])
            num_good += 1
