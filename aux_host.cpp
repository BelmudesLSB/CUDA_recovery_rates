#include "aux_host.h"
#include "mex.h"

int Parameters_host::read_parameters(const mxArray* mxPtr){

    b_grid_size_lowr = static_cast<int>(mxGetScalar(mxGetField(mxPtr, 0, "b_grid_size_lowr")));
    
}
    