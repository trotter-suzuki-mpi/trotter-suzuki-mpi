%module trottersuzuki
%include <std_string.i>
%{
#define SWIG_FILE_WITH_INIT
#include "src/trottersuzuki.h"
%}

%include "numpy.i"

%init %{
import_array();
%}

class Lattice {
public:
    Lattice(int dim=100, double _length_x=20., double _length_y=20.,
            int _periods[2]=0, double omega=0.);
    double length_x, length_y;
    double delta_x, delta_y;
    int dim_x, dim_y;
    int global_dim_x, global_dim_y;
    int periods[2];

    // Computational topology
    int halo_x, halo_y;
    int start_x, end_x, inner_start_x, inner_end_x,
        start_y, end_y, inner_start_y, inner_end_y;
    int mpi_coords[2], mpi_dims[2];
    int mpi_rank;
    int mpi_procs;
#ifdef HAVE_MPI
    MPI_Comm cartcomm;
#endif
};
