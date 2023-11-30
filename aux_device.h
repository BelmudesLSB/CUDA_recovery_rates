#ifndef AUX_DEVICE_H
#define AUX_DEVICE_H

#include "cuda_runtime.h"
#include "aux_host.h"
#include "mex.h"

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

// This class contains the memory allocated on the device for the vectors:
class Vectors_device{

    public:
        double* q_lowr;
        double* q_highr;
        double* q_lowr_new;
        double* q_highr_new;
        double* v;
        double* v_new;
        double* v_d;
        double* v_d_new;
        double* b_grid_lowr;
        double* b_grid_highr;
        double* y_grid;
        double* y_grid_default;
        int* b_policy_lowr;
        int* b_policy_highr;
        double* d_policy;
        double* v_r;
        double* v_r_new;

    Vectors_device(const Parameters_host& params){
        cudaError_t cs;
        cs =cudaMalloc((void**)&q_lowr, params.b_grid_size_lowr*params.b_grid_size_highr*params.y_grid_size*sizeof(double));
        if(cs != cudaSuccess){
            mexPrintf("Error allocating memory for q_lowr on device");
        }
        cs =cudaMalloc((void**)&q_highr, params.b_grid_size_lowr*params.b_grid_size_highr*params.y_grid_size*sizeof(double));
        if(cs != cudaSuccess){
            mexPrintf("Error allocating memory for q_highr on device");
        }
        cs =cudaMalloc((void**)&q_lowr_new, params.b_grid_size_lowr*params.b_grid_size_highr*params.y_grid_size*sizeof(double));
        if(cs != cudaSuccess){
            mexPrintf("Error allocating memory for q_lowr_new on device");
        }
        cs =cudaMalloc((void**)&q_highr_new, params.b_grid_size_lowr*params.b_grid_size_highr*params.y_grid_size*sizeof(double));
        if(cs != cudaSuccess){
            mexPrintf("Error allocating memory for q_highr_new on device");
        }
        cs =cudaMalloc((void**)&v, params.b_grid_size_lowr*params.b_grid_size_highr*params.y_grid_size*sizeof(double));
        if(cs != cudaSuccess){
            mexPrintf("Error allocating memory for v on device");
        }
        cs =cudaMalloc((void**)&v_new, params.b_grid_size_lowr*params.b_grid_size_highr*params.y_grid_size*sizeof(double));
        if(cs != cudaSuccess){
            mexPrintf("Error allocating memory for v_new on device");
        }
        cs =cudaMalloc((void**)&v_d, params.b_grid_size_lowr*params.b_grid_size_highr*params.y_grid_size*sizeof(double));
        if(cs != cudaSuccess){
            mexPrintf("Error allocating memory for v_d on device");
        }
        cs =cudaMalloc((void**)&v_d_new, params.b_grid_size_lowr*params.b_grid_size_highr*params.y_grid_size*sizeof(double));
        if(cs != cudaSuccess){
            mexPrintf("Error allocating memory for v_d_new on device");
        }
        cs =cudaMalloc((void**)&b_grid_lowr, params.b_grid_size_lowr*sizeof(double));
        if(cs != cudaSuccess){
            mexPrintf("Error allocating memory for b_grid_lowr on device");
        }
        cs =cudaMalloc((void**)&b_grid_highr, params.b_grid_size_highr*sizeof(double));
        if(cs != cudaSuccess){
            mexPrintf("Error allocating memory for b_grid_highr on device");
        }
        cs =cudaMalloc((void**)&y_grid, params.y_grid_size*sizeof(double));
        if(cs != cudaSuccess){
            mexPrintf("Error allocating memory for y_grid on device");
        }
        cs =cudaMalloc((void**)&y_grid_default, params.y_grid_size*sizeof(double));
        if(cs != cudaSuccess){
            mexPrintf("Error allocating memory for y_grid_default on device");
        }
        cs =cudaMalloc((void**)&b_policy_lowr, params.b_grid_size_lowr*params.b_grid_size_highr*params.y_grid_size*sizeof(int));
        if(cs != cudaSuccess){
            mexPrintf("Error allocating memory for b_policy_lowr on device");
        }
        cs =cudaMalloc((void**)&b_policy_highr, params.b_grid_size_lowr*params.b_grid_size_highr*params.y_grid_size*sizeof(int));
        if(cs != cudaSuccess){
            mexPrintf("Error allocating memory for b_policy_highr on device");
        }
        cs =cudaMalloc((void**)&d_policy, params.b_grid_size_lowr*params.b_grid_size_highr*params.y_grid_size*sizeof(double));
        if(cs != cudaSuccess){
            mexPrintf("Error allocating memory for d_policy on device");
        }
        cs =cudaMalloc((void**)&v_r, params.b_grid_size_lowr*params.b_grid_size_highr*params.y_grid_size*sizeof(double));
        if(cs != cudaSuccess){
            mexPrintf("Error allocating memory for v_r on device");
        }
        cs =cudaMalloc((void**)&v_r_new, params.b_grid_size_lowr*params.b_grid_size_highr*params.y_grid_size*sizeof(double));
        if(cs != cudaSuccess){
            mexPrintf("Error allocating memory for v_r_new on device");
        }
    }    
    // Free memory:
    void Free_Memory(); 
};

__global__ void fill_q(Vectors_device v_device);

#endif 