#include <iostream>
#include "cuda_runtime.h"
#include "aux_device.h"

// This function releases memory from the device:
void Vectors_device::Free_Memory(){
    cudaFree(q_lowr);
}

// Test function:
__global__ void fill_q(double *q_lowr){
    int i = threadIdx.x;
    if (i % 2 == 0)
    {
        q_lowr[i] = 2.3;
    } else
    {
        q_lowr[i] = 1.2;
    }
}