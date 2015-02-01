from __future__ import division, print_function
import pytpc.tracking as tracking
import pytpc.simulation as sim
import numpy
from pytpc.constants import *


def chisq(meas, exp, var):
    return ((meas - exp)**2 / var).sum(axis=0) / (meas.shape[0] - meas.shape[1] - 1)


def main():

    resf = open('kf_sim_results.csv', 'w', buffering=0)
    excf = open('kf_sim_exceptions.csv', 'w', buffering=0)

    efield = [0, 0, 15e3]
    bfield = [0, 0, 0]
    g = sim.Gas(4, 2, 41.8, 150.)

    for pol in numpy.linspace(0, pi/2, 50):
        for azi in numpy.linspace(0, 2*pi, 300):
            try:
                pt = sim.Particle(mass_num=4, charge_num=2, energy_per_particle=2, azimuth=azi, polar=pol)

                pos, mom, time_, en_, azi_, pol_ = map(numpy.array, sim.track(pt, g, efield, bfield))

                sv = numpy.hstack((pos, mom))
                meas_pos = pos + numpy.random.normal(0, 1e-2, pos.shape)
                tr = tracking.Tracker(pt, g, efield, bfield, sv[0])
                tr.kfilter._dt = 5e-11
                tr.kfilter.Q = numpy.diag((1e-4, 1e-4, 1e-4, 1e0, 1e0, 1e0))**2  # Process
                tr.kfilter.R = numpy.eye(tr.kfilter._dim_z) * 1e-2  # Measurement
                tr.kfilter.P = numpy.eye(tr.kfilter._dim_x) * 10
                res, covar = tr.track(meas_pos)

                gof = chisq(meas_pos, res[:, 0:3], 1e-2**2)

                resf.write('{a}, {p}, {gof}\n'.format(a=azi, p=pol, gof=gof))
            except Exception as oops:
                excf.write('{a}, {p}, {s}\n'.format(a=azi, p=pol, s=type(oops)))
                continue


if __name__ == '__main__':
    main()