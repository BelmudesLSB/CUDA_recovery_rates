#include <iostream>
#include "cuda_runtime.h"
#include "aux_device.h"
#include "aux_host.h"

// This function releases memory from the device:
void Vectors_device::Free_Memory(){
    cudaFree(q_lowr);
}

// Test function:
__global__ void fill_q(Vectors_device v_device){
    int i = threadIdx.x;
    if (i % 2 == 0)
    {
        v_device.q_lowr[i] = 2;
    } else
    {
        v_device.q_lowr[i] = 3;
    }
}
