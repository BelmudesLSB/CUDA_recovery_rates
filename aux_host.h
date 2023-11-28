#ifndef aux_host_h
#define aux_host_h
#include "mex.h"

class Parameters_host {
    public:
        int b_grid_size_lowr;
        int b_grid_size_highr;
        double b_grid_min_lowr;
        double b_grid_min_highr;
        double b_grid_max_lowr;
        double b_grid_max_highr;
        int y_grid_size;
        double y_default;
        double beta;
        double gamma;
        double r;
        double rho;
        double sigma;
        double theta;
        double tol;
        int max_iter;
        double M;
        double alpha_lowr;
        double alpha_highr;

        // Methods associated with this class:

        int read_parameters(const mxArray* mxPtr);
        void print_parameters();
        int transfer_parameters_host_to_device();
    };

// This class stores all the vectors in host memory:
class Vectors_host {
public:
    double *b_grid_lowr;
    double *b_grid_highr;
    double *y_grid;
    double *y_grid_default;
    double *prob;
    int *b_policy_lowr;
    int *b_policy_highr;
    double *v_r;
    double *v_default;
    double *d_policy_lowr;
    double *d_policy_highr;
    double *q_lowr;
    double *q_highr;

    Vectors_host(const Parameters_host& params){
        b_grid_lowr = new double[params.b_grid_size_lowr];
        b_grid_highr = new double[params.b_grid_size_highr];
        y_grid = new double[params.y_grid_size];
        y_grid_default = new double[params.y_grid_size];
        prob = new double[params.y_grid_size * params.y_grid_size];
        b_policy_lowr = new int[params.y_grid_size * params.b_grid_size_lowr * params.b_grid_size_lowr];
        b_policy_highr = new int[params.y_grid_size * params.b_grid_size_highr * params.b_grid_size_highr];
        v_r = new double[params.y_grid_size * params.b_grid_size_highr * params.b_grid_size_highr];
        v_default = new double[params.y_grid_size * params.b_grid_size_lowr * params.b_grid_size_lowr];
        d_policy_lowr = new double[params.y_grid_size * params.b_grid_size_lowr * params.b_grid_size_lowr];
        d_policy_highr = new double[params.y_grid_size * params.b_grid_size_highr * params.b_grid_size_highr];
        q_lowr = new double[params.y_grid_size * params.b_grid_size_highr * params.b_grid_size_lowr];
        q_highr = new double[params.y_grid_size * params.b_grid_size_highr * params.b_grid_size_lowr];
    }

    // Free memory:
    void Free_Memory();
};

// Creates bond grid:
void create_bond_grids(double* prt_bond_grid, int Nb, double Bmax, double Bmin);

// Creates income grid and transition matrix:
void create_income_and_prob_grids(double* prt_y_grid, double* prt_p_grid,  int Ny,  double Sigma,  double Rho,  double M);

// Creates income grid under default:
void create_income_under_default(double* prt_y_grid_default, double* prt_y_grid,  int Ny,  double y_def);

// Creates bond policy functions:
void fill_vectors_host(Parameters_host p_host, Vectors_host v_host);

// Normal cumulative distribution function:
double normalCDF(double x);

// copy vector:
void copy_vector(double* prt_original, double* prt_copy, int size);

// copy vector:
void copy_vector(int* prt_original, int* prt_copy, int size);

// Utility function:
double utility(double c,  double gamma);

#endif