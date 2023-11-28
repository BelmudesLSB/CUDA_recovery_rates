/*
* This code is an implementation of Foreign-vs-Domestic Bonds using MEX and CUDA.
* Lucas Belmudes & Angelo Mendes 11/27/2023.
*/

#include <iostream>
#include <mex.h>
#include "aux_host.h"
#include "aux_device.h"

/// By default all variables are in the host. Else, they will have a d_ prefix.

void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]){

    // Load the parameters from the struct to a class, and print them:
    Parameters_host p_host;
    p_host.read_parameters(prhs[0]);
    p_host.print_parameters();
    if (p_host.transfer_parameters_host_to_device()==1)
    {
        mexPrintf("Parameters copied to device successfully.\n");
    }
    // Using the parameters, create the vectors and store everything in host memory:
    Vectors_host v_host(p_host);
    fill_vectors_host(p_host, v_host);

    cudaError_t cudaStatus;
    // Solve in device:
    Vectors_device v_device(p_host);
    cudaDeviceSynchronize();

    // Kernel launch
    fill_q<<<1, p_host.b_grid_size_lowr*p_host.b_grid_size_highr*p_host.y_grid_size>>>(v_device.q_lowr);  
    cudaDeviceSynchronize();

    cudaMemcpy(v_host.q_lowr, v_device.q_lowr, p_host.b_grid_size_lowr*p_host.b_grid_size_highr*p_host.y_grid_size * sizeof(double), cudaMemcpyDeviceToHost);

    // Create the pointer in MATLAB to store the results
    mxArray* Q_m_lowr = mxCreateDoubleMatrix(p_host.y_grid_size * p_host.b_grid_size_lowr * p_host.b_grid_size_highr, 1, mxREAL);
    mxArray* Q_m_highr = mxCreateDoubleMatrix(p_host.y_grid_size * p_host.b_grid_size_lowr * p_host.b_grid_size_highr, 1, mxREAL);
    mxArray* V_m = mxCreateDoubleMatrix(p_host.y_grid_size * p_host.b_grid_size_lowr * p_host.b_grid_size_highr, 1, mxREAL);
    mxArray* V_r_m = mxCreateDoubleMatrix(p_host.y_grid_size * p_host.b_grid_size_lowr * p_host.b_grid_size_highr, 1, mxREAL);
    mxArray* V_d_m = mxCreateDoubleMatrix(p_host.y_grid_size * p_host.b_grid_size_lowr * p_host.b_grid_size_highr, 1, mxREAL);
    mxArray* B_policy_m_lowr = mxCreateDoubleMatrix(p_host.y_grid_size * p_host.b_grid_size_lowr * p_host.b_grid_size_highr, 1, mxREAL);
    mxArray* B_policy_m_highr = mxCreateDoubleMatrix(p_host.y_grid_size * p_host.b_grid_size_lowr * p_host.b_grid_size_highr, 1, mxREAL);
    mxArray* D_policy_m = mxCreateDoubleMatrix(p_host.y_grid_size * p_host.b_grid_size_lowr * p_host.b_grid_size_highr, 1, mxREAL);
    mxArray* iter_m = mxCreateDoubleMatrix(1, 1, mxREAL);
    mxArray* err_q_m = mxCreateDoubleMatrix(1, 1, mxREAL);
    mxArray* err_v_m = mxCreateDoubleMatrix(1, 1, mxREAL);
    mxArray* Y_grid_m = mxCreateDoubleMatrix(p_host.y_grid_size, 1, mxREAL);
    mxArray* Y_grid_default_m = mxCreateDoubleMatrix(p_host.y_grid_size, 1, mxREAL);
    mxArray* B_grid_lowr_m = mxCreateDoubleMatrix(p_host.b_grid_size_lowr, 1, mxREAL);
    mxArray* B_grid_highr_m = mxCreateDoubleMatrix(p_host.b_grid_size_highr, 1, mxREAL);
    mxArray* P_m = mxCreateDoubleMatrix(p_host.y_grid_size * p_host.y_grid_size, 1, mxREAL);


    // Copy from host to MATLAB:
    copy_vector(v_host.y_grid, mxGetPr(Y_grid_m), p_host.y_grid_size);
    copy_vector(v_host.y_grid_default, mxGetPr(Y_grid_default_m), p_host.y_grid_size);
    copy_vector(v_host.b_grid_lowr, mxGetPr(B_grid_lowr_m), p_host.b_grid_size_lowr);
    copy_vector(v_host.b_grid_highr, mxGetPr(B_grid_highr_m), p_host.b_grid_size_highr);
    copy_vector(v_host.prob, mxGetPr(P_m), p_host.y_grid_size * p_host.y_grid_size);
    copy_vector(v_host.q_lowr, mxGetPr(Q_m_lowr), p_host.y_grid_size * p_host.b_grid_size_lowr * p_host.b_grid_size_highr);

    // Export the objects as a MATLAB structure:
    const char* fieldNames[16] = {"Q_lowr", "Q_highr", "V", "V_r", "V_d", "B_policy_lowr", "B_policy_highr", "D_policy", "Y_grid", "Y_grid_default", "B_grid_lowr", "B_grid_highr", "P", "iter", "err_q", "err_v"};
    plhs[0] = mxCreateStructMatrix(1, 1, 16, fieldNames);
    mxSetField(plhs[0], 0, "Q_lowr", Q_m_lowr);
    mxSetField(plhs[0], 0, "Q_highr", Q_m_highr);
    mxSetField(plhs[0], 0, "V", V_m);
    mxSetField(plhs[0], 0, "V_r", V_r_m);
    mxSetField(plhs[0], 0, "V_d", V_d_m);
    mxSetField(plhs[0], 0, "B_policy_lowr", B_policy_m_lowr);
    mxSetField(plhs[0], 0, "B_policy_highr", B_policy_m_highr);
    mxSetField(plhs[0], 0, "D_policy", D_policy_m);
    mxSetField(plhs[0], 0, "Y_grid", Y_grid_m);
    mxSetField(plhs[0], 0, "Y_grid_default", Y_grid_default_m);
    mxSetField(plhs[0], 0, "B_grid_lowr", B_grid_lowr_m);
    mxSetField(plhs[0], 0, "B_grid_highr", B_grid_highr_m);
    mxSetField(plhs[0], 0, "P", P_m);
    mxSetField(plhs[0], 0, "iter", iter_m);
    mxSetField(plhs[0], 0, "err_q", err_q_m);
    mxSetField(plhs[0], 0, "err_v", err_v_m);

    // Free memory:
    v_host.Free_Memory();
    v_device.Free_Memory();
}





