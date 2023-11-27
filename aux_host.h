#ifndef aux_host_h
#define aux_host_h
#include <iostream>
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
        double m;
        double alpha_lowr;
        double alpha_highr;

        // Methods associated with this class:

        int read_parameters(const mxArray* mxPtr);
    };
#endif