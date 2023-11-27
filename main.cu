/*
* This code is an implementation of Foreign-vs-Domestic Bonds using MEX and CUDA.
* Lucas Belmudes & Angelo Mendes 11/27/2023.
*/

#include <iostream>
#include <cuda_runtime.h>
#include <mex.h>

/// By default all variables are in the host. Else, they will have a d_ prefix.

void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]){

    // Load and read in the parameters from matlab:

    const mxArray* parmsStruct = prhs[0];

    // Load the parameters from the struct to a class:

    Parameters_host p_host;
    p_host.read_parameters(parmsStruct);

    mexPrintf(p_host.b_grid_size_lowr);

    /// Export to matlab:
    const char* fieldNames[nfields] = {"b"};
    plhs[0] = mxCreateStructMatrix(1,1,nfields,fieldNames);
    mxSetField(plhs[0], 0, "y_grid", p_host.b_grid_size_lowr);

}



