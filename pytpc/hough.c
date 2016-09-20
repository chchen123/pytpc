#include "hough.h"

// This indexing macro assumes C-style row-major ordering
#define INDEX(i, j, ncols) ( (j) + ( (i) * (ncols) ))

/* The type of the helper function to find a radius. I define it as a function pointer in order
   to re-use the same code to find the Hough transform for both the linear and circular functions.
 */
typedef double (*RadiusFunction)(const double *const restrict, const int, const int, const double, const double);

/* A helper function that does the actual computation. Both of the front-end Hough transform functions call
   this with an appropriate radfunc pointer to find the Hough space.

   Parameters:
   - xy: The input XYZ data. This must have at least a column for X and Y. Any other columns are ignored.
   - nrows, ncols: The number of rows and columns in xy
   - accum: An output array for the Hough transform accumulator (the Hough space map). Dimension must be nbins x nbins.
   - nbins: The number of bins to use in the computation. This should be the
   - max_val: Max radius value in Hough space. The space is symmetric about 0, so the min radius is -max_val.
   - radfunc: A pointer to a function to compute the radius.
   - firstRowIdx: Start the iteration through xy at this row.
 */
static void hough_helper(const double *restrict xy, const int nrows, const int ncols, int64_t *restrict accum,
                         const int nbins, const double max_val, const RadiusFunction radfunc, const int firstRowIdx)
{
    const double thstep = M_PI / nbins;  // Size of theta (angle) bins
    const double min_val = -max_val;

    #pragma omp parallel for
    for (int theta_idx = 0; theta_idx < nbins; theta_idx++) {
        double theta = theta_idx * thstep;
        // Precompute sin and cos here so they aren't done nrows times for each theta
        double costh = cos(theta);
        double sinth = sin(theta);

        for (int xyz_idx = firstRowIdx; xyz_idx < nrows; xyz_idx++) {
            double rad = radfunc(xy, ncols, xyz_idx, costh, sinth);
            if (rad >= min_val && rad < max_val) {
                // Find and increment histogram/accumulator bin corresponding to rad
                size_t radbin = (size_t) floor((rad + max_val) * nbins / (2 * max_val));
                accum[INDEX(theta_idx, radbin, nbins)] += 1;
            }
        }
    }
}

static inline double
houghline_find_rad(const double *const restrict xy, const int ncols, const int rowIdx,
                   const double costh, const double sinth)
{
    const double x = xy[INDEX(rowIdx, 0, ncols)];
    const double y = xy[INDEX(rowIdx, 1, ncols)];
    return x * costh + y * sinth;
}

static inline double
houghcircle_find_rad(const double *const restrict xy, const int ncols, const int rowIdx,
                     const double costh, const double sinth)
{
    const double x1 = xy[INDEX(rowIdx,     0, ncols)];
    const double x0 = xy[INDEX(rowIdx - 5, 0, ncols)];
    const double y1 = xy[INDEX(rowIdx,     1, ncols)];
    const double y0 = xy[INDEX(rowIdx - 5, 1, ncols)];

    double numer = (x1*x1 - x0*x0) + (y1*y1 - y0*y0);
    double denom = 2 * ((x1 - x0) * costh + (y1 - y0) * sinth);
    return numer / denom;
}

void houghline(const double *restrict xy, const int nrows, const int ncols, int64_t *restrict accum, const int nbins,
               const double max_val)
{
    const int firstRowIdx = 0;
    hough_helper(xy, nrows, ncols, accum, nbins, max_val, &houghline_find_rad, firstRowIdx);
}

void houghcircle(const double *restrict xy, const int nrows, const int ncols, int64_t *restrict accum, const int nbins,
                 const double max_val)
{
    const int firstRowIdx = 5;  // Must start at 5 since we compare x_i and x_(i-5) in the rad-finding function
    hough_helper(xy, nrows, ncols, accum, nbins, max_val, &houghcircle_find_rad, firstRowIdx);
}

void neighborcount(const double *restrict xy, const int nrows, const int ncols, int64_t *restrict counts, const double radius)
{
    const double rad2 = radius * radius;

    #pragma omp parallel for
    for (size_t myidx = 0; myidx < nrows; myidx++) {
        const double myX = xy[INDEX(myidx, 0, ncols)];
        const double myY = xy[INDEX(myidx, 1, ncols)];
        const double myZ = xy[INDEX(myidx, 2, ncols)];

        counts[myidx] = -1;  // Start at -1 to cancel out counting of self as neighbor

        for (size_t otheridx = 0; otheridx < nrows; otheridx++) {
            const double otherX = xy[INDEX(otheridx, 0, ncols)];
            const double otherY = xy[INDEX(otheridx, 1, ncols)];
            const double otherZ = xy[INDEX(otheridx, 2, ncols)];

            const double dx = myX - otherX;
            const double dy = myY - otherY;
            const double dz = myZ - otherZ;
            const double dist2 = dx*dx + dy*dy + dz*dz;

            if (dist2 < rad2) {
                counts[myidx] += 1;
            }
        }
    }
}
