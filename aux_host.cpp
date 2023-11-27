#include "aux_host.h"
#include "mex.h"
#include <iostream>

// This function reads the parameters from matlab and stores them in a class in C++ in host memory.
int Parameters_host::read_parameters(const mxArray* mxPtr){
    b_grid_size_lowr = static_cast<int>(mxGetScalar(mxGetField(mxPtr, 0, "b_grid_size_lowr")));
    b_grid_size_highr = static_cast<int>(mxGetScalar(mxGetField(mxPtr, 0, "b_grid_size_highr")));
    b_grid_min_lowr = mxGetScalar(mxGetField(mxPtr, 0, "b_grid_min_lowr"));
    b_grid_min_highr = mxGetScalar(mxGetField(mxPtr, 0, "b_grid_min_highr"));
    b_grid_max_lowr = mxGetScalar(mxGetField(mxPtr, 0, "b_grid_max_lowr"));
    b_grid_max_highr = mxGetScalar(mxGetField(mxPtr, 0, "b_grid_max_highr"));
    y_grid_size = static_cast<int>(mxGetScalar(mxGetField(mxPtr, 0, "y_grid_size")));
    y_default = mxGetScalar(mxGetField(mxPtr, 0, "y_default"));
    beta = mxGetScalar(mxGetField(mxPtr, 0, "beta"));
    gamma = mxGetScalar(mxGetField(mxPtr, 0, "gamma"));
    r = mxGetScalar(mxGetField(mxPtr, 0, "r"));
    rho = mxGetScalar(mxGetField(mxPtr, 0, "rho"));
    sigma = mxGetScalar(mxGetField(mxPtr, 0, "sigma"));
    theta = mxGetScalar(mxGetField(mxPtr, 0, "theta"));
    tol = mxGetScalar(mxGetField(mxPtr, 0, "tol"));
    max_iter = static_cast<int>(mxGetScalar(mxGetField(mxPtr, 0, "max_iter")));
    M = mxGetScalar(mxGetField(mxPtr, 0, "M"));
    alpha_lowr = mxGetScalar(mxGetField(mxPtr, 0, "alpha_lowr"));
    alpha_highr = mxGetScalar(mxGetField(mxPtr, 0, "alpha_highr"));
    return 1;
}

// This function prints the parameters in the command window.
void Parameters_host::print_parameters(){
    mexPrintf("b_grid_size_lowr: %d\n", b_grid_size_lowr);
    mexPrintf("b_grid_size_highr: %d\n", b_grid_size_highr);
    mexPrintf("b_grid_min_lowr: %f\n", b_grid_min_lowr);
    mexPrintf("b_grid_min_highr: %f\n", b_grid_min_highr);
    mexPrintf("b_grid_max_lowr: %f\n", b_grid_max_lowr);
    mexPrintf("b_grid_max_highr: %f\n", b_grid_max_highr);
    mexPrintf("y_grid_size: %d\n", y_grid_size);
    mexPrintf("y_default: %f\n", y_default);
    mexPrintf("beta: %f\n", beta);
    mexPrintf("gamma: %f\n", gamma);
    mexPrintf("r: %f\n", r);
    mexPrintf("rho: %f\n", rho);
    mexPrintf("sigma: %f\n", sigma);
    mexPrintf("theta: %f\n", theta);
    mexPrintf("tol: %f\n", tol);
    mexPrintf("max_iter: %d\n", max_iter);
    mexPrintf("M: %f\n", M);
    mexPrintf("alpha_lowr: %f\n", alpha_lowr);
    mexPrintf("alpha_highr: %f\n", alpha_highr);
}

// Create bond grids:
void create_bond_grids(double* prt_bond_grid, int Nb, double Bmax, double Bmin){
    double bstep = (Bmax - Bmin)/(Nb - 1);
    for(int i = 0; i < Nb; i++){
        prt_bond_grid[i] = Bmin + i*bstep;
    }
}

// Create the income grid and the transition matrix:
void create_income_and_prob_grids(double* prt_y_grid, double* prt_p_grid,  int Ny,  double Sigma,  double Rho,  double M){
    double sigma_y = sqrt(pow(Sigma,2)/(1-pow(Rho,2)));
    double omega = (2*M*sigma_y)/(Ny-1);
    for (int i=0; i<Ny; i++){ 
        prt_y_grid[i] = (-M*sigma_y)  + omega * i;
    }   
    for (int i=0; i<Ny; i++){
        for (int j=0; j<Ny; j++){
            if (j==0 || j==Ny-1){
                if (j==0){
                    prt_p_grid[i*Ny+j] = normalCDF((prt_y_grid[0]-Rho*prt_y_grid[i]+omega/2)/Sigma);
                }
                else {
                    prt_p_grid[i*Ny+j] = 1-normalCDF((prt_y_grid[Ny-1]-Rho*prt_y_grid[i]-omega/2)/Sigma);
                }
            } else {
                prt_p_grid[i*Ny+j] = normalCDF((prt_y_grid[j]-Rho*prt_y_grid[i]+omega/2)/Sigma)-normalCDF((prt_y_grid[j]-Rho*prt_y_grid[i]-omega/2)/Sigma);
            }
        }
    }
    for (int i=0; i<Ny; i++){
        prt_y_grid[i] = exp(prt_y_grid[i]);
    }
}

// Create the income grid for the default state:
void create_income_under_default(double* prt_y_grid_default, double* prt_y_grid,  int Ny,  double y_def){
    for (int i=0; i<Ny; i++){
        if (prt_y_grid[i]>y_def){
            prt_y_grid_default[i] = y_def;
        } else {
            prt_y_grid_default[i] = prt_y_grid[i];
        }
    }
}

// Initialize the economy:
void initialize_economy(Parameters_host p_host, Vectors_host v_host){
    create_bond_grids(v_host.b_grid_lowr, p_host.b_grid_size_lowr, p_host.b_grid_max_lowr, p_host.b_grid_min_lowr);
    create_bond_grids(v_host.b_grid_highr, p_host.b_grid_size_highr, p_host.b_grid_max_highr, p_host.b_grid_min_highr);
    create_income_and_prob_grids(v_host.y_grid, v_host.prob, p_host.y_grid_size, p_host.sigma, p_host.rho, p_host.M);
    create_income_under_default(v_host.y_grid_default, v_host.y_grid, p_host.y_grid_size, p_host.y_default);
}

// Normal cumulative distribution function:
double normalCDF(double x){
    return std::erfc(-x / std::sqrt(2)) / 2;
}

// Utility function:
double utility(double c,  double gamma, double c_lb){
    if (c>=c_lb){
        return pow(c,1-gamma)/(1-gamma);
    } else {
        return -1000000;
    }
}

// Copy vector:
void copy_vector(double* prt_vector, double* prt_vector_copy, int size){
    for (int i=0; i<size; i++){
        prt_vector_copy[i] = prt_vector[i];
    }
}

// Copy vector:
void copy_vector(int* prt_vector, int* prt_vector_copy, int size){
    for (int i=0; i<size; i++){
        prt_vector_copy[i] = (prt_vector[i]);
    }
}