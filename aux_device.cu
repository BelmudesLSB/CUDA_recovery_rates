#include "cuda_runtime.h"
#include "aux_device.h"

// This function copies the parameters from the host to the device.
int Parameters_host::transfer_parameters_host_to_device(){
    cudaError_t cs;
    cs = cudaMemcpyToSymbol(d_b_grid_size_lowr, &b_grid_size_lowr, sizeof(int), 0, cudaMemcpyHostToDevice);
    if (cs != cudaSuccess)
    {
        mexPrintf("Error copying b_grid_size to device\n");
        return 0;
    } 
    cs = cudaMemcpyToSymbol(d_b_grid_size_highr, &b_grid_size_highr, sizeof(int), 0, cudaMemcpyHostToDevice);
    if (cs != cudaSuccess)
    {
        mexPrintf("Error copying b_grid_size to device\n", 0 , cudaMemcpyHostToDevice);
        return 0;
    } 
    cs = cudaMemcpyToSymbol(d_b_grid_min_lowr, &b_grid_min_lowr, sizeof(double), 0, cudaMemcpyHostToDevice);
    if (cs != cudaSuccess)
    {
        mexPrintf("Error copying b_grid_min to device\n");
        return 0;
    }
    cs = cudaMemcpyToSymbol(d_b_grid_min_highr, &b_grid_min_highr, sizeof(double), 0, cudaMemcpyHostToDevice);
    if (cs != cudaSuccess)
    {
        mexPrintf("Error copying b_grid_min to device\n");
        return 0;
    }
    cs = cudaMemcpyToSymbol(d_b_grid_max_lowr, &b_grid_max_lowr, sizeof(double), 0, cudaMemcpyHostToDevice);
    if (cs != cudaSuccess)
    {
        mexPrintf("Error copying b_grid_max to device\n");
        return 0;
    }
    cs = cudaMemcpyToSymbol(d_b_grid_max_highr, &b_grid_max_highr, sizeof(double), 0, cudaMemcpyHostToDevice);
    if (cs != cudaSuccess)
    {
        mexPrintf("Error copying b_grid_max to device\n");
        return 0;
    }
    cs = cudaMemcpyToSymbol(d_y_grid_size, &y_grid_size, sizeof(int), 0, cudaMemcpyHostToDevice);
    if (cs != cudaSuccess)
    {
        mexPrintf("Error copying y_grid_size to device\n");
        return 0;
    }
    cs = cudaMemcpyToSymbol(d_y_default, &y_default, sizeof(double), 0, cudaMemcpyHostToDevice);
    if (cs != cudaSuccess)
    {
        mexPrintf("Error copying y_default to device\n");
        return 0;
    }
    cs = cudaMemcpyToSymbol(d_beta, &beta, sizeof(double), 0, cudaMemcpyHostToDevice);
    if (cs != cudaSuccess)
    {
        mexPrintf("Error copying beta to device\n");
        return 0;
    }
    cs = cudaMemcpyToSymbol(d_gamma, &gamma, sizeof(double), 0, cudaMemcpyHostToDevice);
    if (cs != cudaSuccess)
    {
        mexPrintf("Error copying gamma to device\n");
        return 0;
    }
    cs = cudaMemcpyToSymbol(d_r, &r, sizeof(double), 0, cudaMemcpyHostToDevice);
    if (cs != cudaSuccess)
    {
        mexPrintf("Error copying r to device\n");
        return 0;
    }
    cs = cudaMemcpyToSymbol(d_rho, &rho, sizeof(double), 0, cudaMemcpyHostToDevice);
    if (cs != cudaSuccess)
    {
        mexPrintf("Error copying rho to device\n");
        return 0;
    }
    cs = cudaMemcpyToSymbol(d_theta, &theta, sizeof(double), 0, cudaMemcpyHostToDevice);
    if (cs != cudaSuccess)
    {
        mexPrintf("Error copying theta to device\n");
        return 0;
    }
    cs = cudaMemcpyToSymbol(d_tol, &tol, sizeof(double), 0, cudaMemcpyHostToDevice);
    if (cs != cudaSuccess)
    {
        mexPrintf("Error copying tol to device\n");
        return 0;
    }
    cs = cudaMemcpyToSymbol(d_max_iter, &max_iter, sizeof(int), 0, cudaMemcpyHostToDevice);
    if (cs != cudaSuccess)
    {
        mexPrintf("Error copying max_iter to device\n");
        return 0;
    }
    cs = cudaMemcpyToSymbol(d_b_grid_max_highr, &b_grid_max_highr, sizeof(double), 0, cudaMemcpyHostToDevice);
    if (cs != cudaSuccess)
    {
        mexPrintf("Error copying b_grid_max to device\n");
        return 0;
    }
    cs = cudaMemcpyToSymbol(d_b_grid_max_lowr, &b_grid_max_lowr, sizeof(double), 0, cudaMemcpyHostToDevice);
    if (cs != cudaSuccess)
    {
        mexPrintf("Error copying b_grid_max to device\n");
        return 0;
    }
    cs = cudaMemcpyToSymbol(d_b_grid_min_highr, &b_grid_min_highr, sizeof(double), 0, cudaMemcpyHostToDevice);
    if (cs != cudaSuccess)
    {
        mexPrintf("Error copying b_grid_max to device\n");
        return 0;
    }
    cs = cudaMemcpyToSymbol(d_b_grid_min_lowr, &b_grid_min_lowr, sizeof(double), 0, cudaMemcpyHostToDevice);
    if (cs != cudaSuccess)
    {
        mexPrintf("Error copying b_grid_max to device\n");
        return 0;
    }
    cs = cudaMemcpyToSymbol(d_alpha_lowr, &alpha_lowr, sizeof(double), 0, cudaMemcpyHostToDevice);
    if (cs != cudaSuccess)
    {
        mexPrintf("Error copying alpha_lowr to device\n");
        return 0;
    }
    cs = cudaMemcpyToSymbol(d_alpha_highr, &alpha_highr, sizeof(double), 0, cudaMemcpyHostToDevice);
    if (cs != cudaSuccess)
    {
        mexPrintf("Error copying alpha_highr to device\n");
        return 0;
    }
    mexPrintf("Parameters copied to device successfully.\n");
    return 1;
}

// This function releases memory from the device:
void Vectors_device::Free_Memory(){
    cudaFree(q_lowr);
}

// Test function:
__global__ void fill_q(Vectors_device v_device){
    int i = threadIdx.x;
    if (i % 2 == 0)
    {
        v_device.q_lowr[i] = 2 + d_b_grid_size_highr;
    } else
    {
        v_device.q_lowr[i] = 3 + d_alpha_highr;
    }
}

