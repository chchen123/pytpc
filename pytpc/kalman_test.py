import matplotlib as mpl
mpl.use('PDF')
import matplotlib.pyplot as plt

import pytpc.tracking as tracking
import pytpc.simulation as sim
import numpy
from pytpc.constants import *

def state_vector_plots(sv, sv2=None, data=None):
    fig, ax = plt.subplots(nrows=3, ncols=2)
    fig.set_figheight(15)
    fig.set_figwidth(15)

    ax[0,0].plot(sv[:,0], label='Actual', zorder=2)
    ax[1,0].plot(sv[:,1], label='Actual', zorder=2)
    ax[2,0].plot(sv[:,2], label='Actual', zorder=2)
    ax[0,1].plot(sv[:,3], label='Actual', zorder=2)
    ax[1,1].plot(sv[:,4], label='Actual', zorder=2)
    ax[2,1].plot(sv[:,5], label='Actual', zorder=2)

    if sv2 is not None:
        ax[0,0].plot(sv2[:,0], label='Calculated', zorder=3)
        ax[1,0].plot(sv2[:,1], label='Calculated', zorder=3)
        ax[2,0].plot(sv2[:,2], label='Calculated', zorder=3)
        ax[0,1].plot(sv2[:,3], label='Calculated', zorder=3)
        ax[1,1].plot(sv2[:,4], label='Calculated', zorder=3)
        ax[2,1].plot(sv2[:,5], label='Calculated', zorder=3)

    if data is not None:
        ax[0,0].plot(data[:,0], '+', label='Data', zorder=1)
        ax[1,0].plot(data[:,1], '+', label='Data', zorder=1)
        ax[2,0].plot(data[:,2], '+', label='Data', zorder=1)
#         ax[0,1].plot(data[:,3], '+', label='Data', zorder=1)
#         ax[1,1].plot(data[:,4], '+', label='Data', zorder=1)
#         ax[2,1].plot(data[:,5], '+', label='Data', zorder=1)

    ax[0,0].set_ylabel('x [m]')
    ax[0,0].legend(loc='best')

    ax[1,0].set_ylabel('y [m]')
    ax[1,0].legend(loc='best')

    ax[2,0].set_ylabel('z [m]')
    ax[2,0].legend(loc='best')

    ax[0,1].set_ylabel('px [kg m/s]')
    ax[0,1].legend(loc='best')

    ax[1,1].set_ylabel('py [kg m/s]')
    ax[1,1].legend(loc='best')

    ax[2,1].set_ylabel('pz [kg m/s]')
    ax[2,1].legend(loc='best')
    plt.savefig('/Users/josh/Desktop/test.pdf')

def main():
    pt = sim.Particle(mass_num=4, charge_num=2, energy_per_particle=2, azimuth=pi/3, polar=pi/9)
    efield = [0, 0, 15e3]
    bfield = [0, 0, 0]
    g = sim.Gas(4, 2, 41.8, 150.)

    pos, mom, time, en = map(numpy.array, sim.track(pt, g, efield, bfield))

    sv = numpy.hstack((pos, mom))
    meas_pos = pos + numpy.random.normal(0, 1e-2, pos.shape)
    tr = tracking.Tracker(pt, g, efield, bfield, sv[0])
    tr.kfilter._dt = 5e-11
    tr.kfilter.Q = numpy.diag((1e-6, 1e-6, 1e-6, 1e-20, 1e-20, 1e-20))  # Process
    tr.kfilter.R = numpy.eye(tr.kfilter._dim_z) * 1e-2  # Measurement
    tr.kfilter.P = numpy.eye(tr.kfilter._dim_x) * 0.01
    res, covar = tr.track(meas_pos)
    sm_res, sm_covar, sm_gain = tr.kfilter.rts_smoother(res, covar)

    state_vector_plots(sv, res)

if __name__ == '__main__':
    plt.ioff()

    main()

