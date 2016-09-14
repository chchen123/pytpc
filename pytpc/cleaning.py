import numpy as np
import pandas as pd
from .constants import pi


def find_center_hough(xyzs):
    xyz_order = xyzs[np.argsort(xyzs[:, 2])]

    nbins = 200  # number of bins in r and theta for Hough space discretization
    ths = np.linspace(0, pi, nbins)

    npts = len(xyz_order) - 5
    sqdels = xyz_order[5:, :2]**2 - xyz_order[:-5, :2]**2
    deltas_tiled = np.tile(xyz_order[5:, :2] - xyz_order[:-5, :2], (nbins, 1))

    Hrad = np.empty((nbins * npts, 2))

    Hrad[:, 0] = np.repeat(ths, npts)
    Hrad[:, 1] = (np.tile(np.sum(sqdels, -1), nbins) /
                  (2 * (deltas_tiled[:, 0] * np.cos(Hrad[:, 0])
                        + deltas_tiled[:, 1] * np.sin(Hrad[:, 0]))))

    Hrad = Hrad[1:]

    countsRad, xedgesRad, yedgesRad = np.histogram2d(Hrad[:, 0], Hrad[:, 1], nbins, range=((0, pi), (-500, 500)))

    # max bin of histogram
    iRad, jRad = np.unravel_index(countsRad.argmax(), countsRad.shape)
    # convert max bin to theta, radius values
    tRad = iRad * pi / nbins
    rRad = jRad * 1000 / nbins - 500

    # convert theta, r to postions in xy space
    ax = rRad * np.cos(tRad)
    by = rRad * np.sin(tRad)

    return ax, by


def nn_remove_noise(data, radius=40, num_neighbors=10):
    if isinstance(data, pd.DataFrame):
        xyz = data.values
    else:
        xyz = data

    noise = np.sum((np.sqrt(np.sum((xyz[:, None, :3] - xyz[:, :3])**2, -1)) <= radius)
                   & ~np.eye(len(xyz), dtype='bool'), axis=0) < num_neighbors

    if isinstance(data, pd.DataFrame):
        return data.iloc[~noise]
    else:
        return data[~noise]


def apply_clean_cut(raw_xyz_full, qthresh=0.5, nthresh=10, tbthresh=505):
    qcut = raw_xyz_full[:, -1] <= qthresh
    ncut = raw_xyz_full[:, -2] >= nthresh
    tbcut = raw_xyz_full[:, 2] <= tbthresh
    cut = qcut & ncut & tbcut
    return raw_xyz_full[cut, :5]
