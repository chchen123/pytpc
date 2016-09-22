/* hough.h
   This small library provides the backends for the Hough transforms. This is much faster than doing it in
   Python, and we can also use OpenMP to multithread the calculation.
 */

#ifndef HOUGH_H
#define HOUGH_H

#include <stddef.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>

#ifndef M_PI
#define M_PI 3.1415926
#endif

void houghline(const double *restrict xy, const int dim0, const int dim1, int64_t *restrict accum, const int nbins, const double max_val);
void houghcircle(const double *restrict xy, const int dim0, const int dim1, int64_t *restrict accum, const int nbins, const double max_val);

void neighborcount(const double *restrict xy, const int nrows, const int ncols, int64_t *restrict counts, const double radius);

#endif /* end of include guard: HOUGH_H */
