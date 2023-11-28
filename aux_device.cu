#include <iostream>
#include "cuda_runtime.h"
#include "aux_device.h"

void Vectors_device::Free_Memory(){
    cudaFree(q_lowr);
}

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