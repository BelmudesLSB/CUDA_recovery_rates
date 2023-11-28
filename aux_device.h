#ifndef AUX_DEVICE_H
#define AUX_DEVICE_H

#include "cuda_runtime.h"

__constant__ int d_b_grid_size_lowr;
__constant__ int d_b_grid_size_highr;
__constant__ int d_y_grid_size;
__constant__ double d_y_default;
__constant__ double d_beta;
__constant__ double d_gamma;
__constant__ double d_r;
__constant__ double d_theta;
__constant__ double d_tol;
__constant__ int d_max_iter;
__constant__ double d_b_grid_min_lowr;
__constant__ double d_b_grid_max_lowr;
__constant__ double d_b_grid_min_highr;
__constant__ double d_b_grid_max_highr;
__constant__ double d_rho;
__constant__ double d_alpha_lowr;
__constant__ double d_alpha_highr;


#endif 


