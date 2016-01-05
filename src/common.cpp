/**
 * Massively Parallel Trotter-Suzuki Solver
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 */
#include <fstream>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include "trottersuzuki.h"

void my_abort(string err) {
#ifdef HAVE_MPI
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0) {
        cerr << "Error: " << err << endl;
    }
    MPI_Abort(MPI_COMM_WORLD, 1);
#else
    throw std::runtime_error(err);
#endif
}

void add_padding(double *padded_matrix, double *matrix, 
                 int padded_dim_x, int padded_dim_y, 
                 int halo_x, int halo_y, 
                 const int dim_x, const int dim_y, int *periods) {
    for (int i = 0; i < dim_y; i++) {
        for (int j = 0; j < dim_x; j++) {
            double element = matrix[j + i * dim_x];
            padded_matrix[(i + halo_y * periods[0]) * padded_dim_x + j + halo_x * periods[1]] = element;
            //Down band
            if (i < halo_y && periods[0] != 0) {
                padded_matrix[(i + padded_dim_y - halo_y) * padded_dim_x + j + halo_x * periods[1]] = element;
                //Down right corner
                if (j < halo_x && periods[1] != 0) {
                    padded_matrix[(i + padded_dim_y - halo_y) * padded_dim_x + j + padded_dim_x - halo_x] = element;
                }
                //Down left corner
                if(j >= dim_x - halo_x && periods[1] != 0) {
                    padded_matrix[(i + padded_dim_y - halo_y) * padded_dim_x + j - (dim_x - halo_x)] = element;
                }
            }
            //Upper band
            if (i >= dim_y - halo_y && periods[0] != 0) {
                padded_matrix[(i - (dim_y - halo_y)) * padded_dim_x + j + halo_x * periods[1]] = element;
                //Up right corner
                if (j < halo_x && periods[1] != 0) {
                    padded_matrix[(i - (dim_y - halo_y)) * padded_dim_x + j + padded_dim_x - halo_x] = element;
                }
                //Up left corner
                if (j >= dim_x - halo_x && periods[1] != 0) {
                    padded_matrix[(i - (dim_y - halo_y)) * padded_dim_x + j - (dim_x - halo_x)] = element;
                }
            }
            //Right band
            if (j < halo_x && periods[1] != 0) {
                padded_matrix[(i + halo_y * periods[0]) * padded_dim_x + j + padded_dim_x - halo_x] = element;
            }
            //Left band
            if (j >= dim_x - halo_x && periods[1] != 0) {
                padded_matrix[(i + halo_y * periods[0]) * padded_dim_x + j - (dim_x - halo_x)] = element;
            }
        }
    }
}

void print_complex_matrix(char * filename, double * matrix_real, double * matrix_imag, size_t stride, size_t width, size_t height) {
    ofstream out(filename, ios::out | ios::trunc);
    for (size_t i = 0; i < height; ++i) {
        for (size_t j = 0; j < width; ++j) {
            out << "(" << matrix_real[i * stride + j] << "," << matrix_imag[i * stride + j] << ") ";
        }
        out << endl;
    }
    out.close();
}

void print_matrix(const char * filename, double * matrix, size_t stride, size_t width, size_t height) {
    ofstream out(filename, ios::out | ios::trunc);
    for (size_t i = 0; i < height; ++i) {
        for (size_t j = 0; j < width; ++j) {
            out << matrix[i * stride + j] << " ";
        }
        out << endl;
    }
    out.close();
}

void memcpy2D(void * dst, size_t dstride, const void * src, size_t sstride, size_t width, size_t height) {
    char *d = reinterpret_cast<char *>(dst);
    const char *s = reinterpret_cast<const char *>(src);
    for (size_t i = 0; i < height; ++i) {
        for (size_t j = 0; j < width; ++j) {
            d[i * dstride + j] = s[i * sstride + j];
        }
    }
}

void stamp(Lattice *grid, State *state, int tag_particle, int iterations,
           int count_snap, const char * output_folder) {

    char * output_filename;
    output_filename = new char[51];
#ifdef HAVE_MPI
    // Set variables for mpi output
    char *data_as_txt;
    int count;

    MPI_File   file;
    MPI_Status status;

    // each number is represented by charspernum chars
    const int chars_per_complex_num = 30;
    MPI_Datatype complex_num_as_string;
    MPI_Type_contiguous(chars_per_complex_num, MPI_CHAR, &complex_num_as_string);
    MPI_Type_commit(&complex_num_as_string);

    const int charspernum = 14;
    MPI_Datatype num_as_string;
    MPI_Type_contiguous(charspernum, MPI_CHAR, &num_as_string);
    MPI_Type_commit(&num_as_string);

    // create a type describing our piece of the array
    int globalsizes[2] = {grid->global_dim_y - 2 * grid->periods[0] * grid->halo_y, grid->global_dim_x - 2 * grid->periods[1] * grid->halo_x};
    int localsizes [2] = {grid->inner_end_y - grid->inner_start_y, grid->inner_end_x - grid->inner_start_x};
    int starts[2]      = {grid->inner_start_y, grid->inner_start_x};
    int order          = MPI_ORDER_C;

    MPI_Datatype complex_localarray;
    MPI_Type_create_subarray(2, globalsizes, localsizes, starts, order, complex_num_as_string, &complex_localarray);
        MPI_Type_commit(&complex_localarray);
    MPI_Datatype localarray;
    MPI_Type_create_subarray(2, globalsizes, localsizes, starts, order, num_as_string, &localarray);
    MPI_Type_commit(&localarray);

    // output complex matrix
    // conversion
    data_as_txt = new char[(grid->inner_end_x - grid->inner_start_x) * (grid->inner_end_y - grid->inner_start_y) * chars_per_complex_num];
    count = 0;
    for (int j = grid->inner_start_y - grid->start_y; j < grid->inner_end_y - grid->start_y; j++) {
        for (int k = grid->inner_start_x - grid->start_x; k < grid->inner_end_x - grid->start_x - 1; k++) {
            sprintf(&data_as_txt[count * chars_per_complex_num], "(%+.5e,%+.5e)   ", state->p_real[j * grid->dim_x + k], state->p_imag[j * grid->dim_x + k]);
            count++;
        }
        if(grid->mpi_coords[1] == grid->mpi_dims[1] - 1) {
            sprintf(&data_as_txt[count * chars_per_complex_num], "(%+.5e,%+.5e)\n  ", state->p_real[j * grid->dim_x + (grid->inner_end_x - grid->start_x) - 1], state->p_imag[j * grid->dim_x + (grid->inner_end_x - grid->start_x) - 1]);
            count++;
        }
        else {
            sprintf(&data_as_txt[count * chars_per_complex_num], "(%+.5e,%+.5e)   ", state->p_real[j * grid->dim_x + (grid->inner_end_x - grid->start_x) - 1], state->p_imag[j * grid->dim_x + (grid->inner_end_x - grid->start_x) - 1]);
            count++;
        }
    }

    // open the file, and set the view
    sprintf(output_filename, "%s/%i-%i-iter-comp.dat", output_folder, tag_particle + 1, iterations * count_snap);
    MPI_File_open(grid->cartcomm, output_filename,
                  MPI_MODE_CREATE | MPI_MODE_WRONLY,
                  MPI_INFO_NULL, &file);

    MPI_File_set_view(file, 0,  MPI_CHAR, complex_localarray, "native", MPI_INFO_NULL);

    MPI_File_write_all(file, data_as_txt, (grid->inner_end_x - grid->inner_start_x) * (grid->inner_end_y - grid->inner_start_y), complex_num_as_string, &status);
    MPI_File_close(&file);
    delete [] data_as_txt;

    // output real matrix
    //conversion
    data_as_txt = new char[(grid->inner_end_x - grid->inner_start_x) * (grid->inner_end_y - grid->inner_start_y) * charspernum];
    count = 0;
    for (int j = grid->inner_start_y - grid->start_y; j < grid->inner_end_y - grid->start_y; j++) {
        for (int k = grid->inner_start_x - grid->start_x; k < grid->inner_end_x - grid->start_x - 1; k++) {
            sprintf(&data_as_txt[count * charspernum], "%+.5e  ", state->p_real[j * grid->dim_x + k]);
            count++;
        }
        if(grid->mpi_coords[1] == grid->mpi_dims[1] - 1) {
            sprintf(&data_as_txt[count * charspernum], "%+.5e\n ", state->p_real[j * grid->dim_x + (grid->inner_end_x - grid->start_x) - 1]);
            count++;
        }
        else {
            sprintf(&data_as_txt[count * charspernum], "%+.5e  ", state->p_real[j * grid->dim_x + (grid->inner_end_x - grid->start_x) - 1]);
            count++;
        }
    }

    // open the file, and set the view
    sprintf(output_filename, "%s/%i-%i-iter-real.dat", output_folder, tag_particle + 1, iterations * count_snap);
    MPI_File_open(grid->cartcomm, output_filename,
                  MPI_MODE_CREATE | MPI_MODE_WRONLY,
                  MPI_INFO_NULL, &file);

    MPI_File_set_view(file, 0,  MPI_CHAR, localarray, "native", MPI_INFO_NULL);

    MPI_File_write_all(file, data_as_txt, (grid->inner_end_x - grid->inner_start_x) * ( grid->inner_end_y - grid->inner_start_y), num_as_string, &status);
    MPI_File_close(&file);
    delete [] data_as_txt;
#else
    sprintf(output_filename, "%s/%i-%i-iter-real.dat", output_folder, tag_particle + 1, iterations * count_snap);
    print_matrix(output_filename, &(state->p_real[grid->global_dim_x * (grid->inner_start_y - grid->start_y) + grid->inner_start_x - grid->start_x]), grid->global_dim_x,
                 grid->global_dim_x - 2 * grid->periods[1]*grid->halo_x, grid->global_dim_y - 2 * grid->periods[0]*grid->halo_y);

    sprintf(output_filename, "%s/%i-%i-iter-comp.dat", output_folder, tag_particle + 1, iterations * count_snap);
    print_complex_matrix(output_filename, &(state->p_real[grid->global_dim_x * (grid->inner_start_y - grid->start_y) + grid->inner_start_x - grid->start_x]), &(state->p_imag[grid->global_dim_x * (grid->inner_start_y - grid->start_y) + grid->inner_start_x - grid->start_x]), grid->global_dim_x,
                         grid->global_dim_x - 2 * grid->periods[1]*grid->halo_x, grid->global_dim_y - 2 * grid->periods[0]*grid->halo_y);
#endif
    return;
}

void stamp_real(Lattice *grid, double *matrix, int iterations, const char * output_folder, const char * file_tag) {

    char * output_filename;
    output_filename = new char[51];
#ifdef HAVE_MPI
    // Set variables for mpi output
    char *data_as_txt;
    int count;

    MPI_File   file;
    MPI_Status status;

    // each number is represented by charspernum chars
    const int charspernum = 14;
    MPI_Datatype num_as_string;
    MPI_Type_contiguous(charspernum, MPI_CHAR, &num_as_string);
    MPI_Type_commit(&num_as_string);

    // create a type describing our piece of the array
    int globalsizes[2] = {grid->global_dim_y - 2 * grid->periods[0] * grid->halo_y, grid->global_dim_x - 2 * grid->periods[1] * grid->halo_x};
    int localsizes [2] = {grid->inner_end_y - grid->inner_start_y, grid->inner_end_x - grid->inner_start_x};
    int starts[2]      = {grid->inner_start_y, grid->inner_start_x};
    int order          = MPI_ORDER_C;

    MPI_Datatype localarray;
    MPI_Type_create_subarray(2, globalsizes, localsizes, starts, order, num_as_string, &localarray);
    MPI_Type_commit(&localarray);

    // output real matrix
    //conversion
    data_as_txt = new char[(grid->inner_end_x - grid->inner_start_x) * (grid->inner_end_y - grid->inner_start_y) * charspernum];
    count = 0;
    for (int j = grid->inner_start_y - grid->start_y; j < grid->inner_end_y - grid->start_y; j++) {
        for (int k = grid->inner_start_x - grid->start_x; k < grid->inner_end_x - grid->start_x - 1; k++) {
            sprintf(&data_as_txt[count * charspernum], "%+.5e  ", matrix[j * grid->dim_x + k]);
            count++;
        }
        if(grid->mpi_coords[1] == grid->mpi_dims[1] - 1) {
            sprintf(&data_as_txt[count * charspernum], "%+.5e\n ", matrix[j * grid->dim_x + (grid->inner_end_x - grid->start_x) - 1]);
            count++;
        }
        else {
            sprintf(&data_as_txt[count * charspernum], "%+.5e  ", matrix[j * grid->dim_x + (grid->inner_end_x - grid->start_x) - 1]);
            count++;
        }
    }

    // open the file, and set the view
    sprintf(output_filename, "%s/%i-%s", output_folder, iterations, file_tag);
    MPI_File_open(grid->cartcomm, output_filename,
                  MPI_MODE_CREATE | MPI_MODE_WRONLY,
                  MPI_INFO_NULL, &file);

    MPI_File_set_view(file, 0,  MPI_CHAR, localarray, "native", MPI_INFO_NULL);

    MPI_File_write_all(file, data_as_txt, (grid->inner_end_x - grid->inner_start_x) * (grid->inner_end_y - grid->inner_start_y), num_as_string, &status);
    MPI_File_close(&file);
    delete [] data_as_txt;
#else
    sprintf(output_filename, "%s/%i-%s", output_folder, iterations, file_tag);
    print_matrix(output_filename, &(matrix[grid->global_dim_x * (grid->inner_start_y - grid->start_y) + grid->inner_start_x - grid->start_x]), grid->global_dim_x,
                 grid->global_dim_x - 2 * grid->periods[1]*grid->halo_x, grid->global_dim_y - 2 * grid->periods[0]*grid->halo_y);
#endif
    return;
}

void expect_values(int dimx, int dimy, double delta_x, double delta_y, double delta_t, double coupling_const, int iterations, int snapshots, double * hamilt_pot, double particle_mass,
                   const char *dirname, int *periods, int halo_x, int halo_y, energy_momentum_statistics *sample) {

    int dim = dimx; //provisional
    if(snapshots == 0)
        return;

    int N_files = snapshots + 1;
    int *N_name = new int[N_files];

    N_name[0] = 0;
    for(int i = 1; i < N_files; i++) {
        N_name[i] = N_name[i - 1] + iterations;
    }

    complex<double> sum_E = 0;
    complex<double> sum_Px = 0, sum_Py = 0;
    complex<double> sum_pdi = 0;
    complex<double> sum_x2 = 0, sum_x = 0, sum_y2 = 0, sum_y = 0;
    double *energy = new double[N_files];
    double *momentum_x = new double[N_files];
    double *momentum_y = new double[N_files];

    complex<double> cost_E = -1. / (2. * particle_mass * delta_x * delta_y), cost_P_x, cost_P_y;
    cost_P_x = complex<double>(0., -0.5 / delta_x);
    cost_P_y = complex<double>(0., -0.5 / delta_y);

    stringstream filename;
    string filenames;

    filename.str("");
    filename << dirname << "/exp_val_D" << dim << "_I" << iterations << "_S" << snapshots << ".dat";
    filenames = filename.str();
    ofstream out(filenames.c_str());

    double E_before = 0, E_now = 0;

    out << "#iter\t time\tEnergy\t\tdelta_E\t\tPx\tPy\tP**2\tnorm2(psi(t))\tsigma_x\tsigma_y\t<X>\t<Y>" << endl;
    for(int i = 0; i < N_files; i++) {

        filename.str("");
        filename << dirname << "/" << "1-" << N_name[i] << "-iter-comp.dat";
        filenames = filename.str();
        ifstream up(filenames.c_str()), center(filenames.c_str()), down(filenames.c_str());
        complex<double> psi_up, psi_down, psi_center, psi_left, psi_right;

        for (int j = 0; j < dim; j++)
            center >> psi_center;
        for (int j = 0; j < 2 * dim; j++)
            down >> psi_center;
        up >> psi_up;
        down >> psi_down;
        center >> psi_left >> psi_center;
        for (int j = 1; j < dim - 1; j++) {
            if(j != 1) {
                up >> psi_up >> psi_up;
                down >> psi_down >> psi_down;
                center >> psi_left >> psi_center;
            }
            for (int k = 1; k < dim - 1; k++) {
                up >> psi_up;
                center >> psi_right;
                down >> psi_down;

                sum_E += conj(psi_center) * (cost_E * (psi_right + psi_left + psi_down + psi_up - psi_center * complex<double> (4., 0.)) + psi_center * complex<double> (hamilt_pot[j * dim + k], 0.)  + psi_center * psi_center * conj(psi_center) * complex<double> (0.5 * coupling_const, 0.)) ;
                sum_Px += conj(psi_center) * (psi_right - psi_left);
                sum_Py += conj(psi_center) * (psi_down - psi_up);
                sum_x2 += conj(psi_center) * complex<double> (k * k, 0.) * psi_center;
                sum_x += conj(psi_center) * complex<double> (k, 0.) * psi_center;
                sum_y2 += conj(psi_center) * complex<double> (j * j, 0.) * psi_center;
                sum_y += conj(psi_center) * complex<double> (j, 0.) * psi_center;
                sum_pdi += conj(psi_center) * psi_center;

                psi_left = psi_center;
                psi_center = psi_right;
            }

        }
        up.close();
        center.close();
        down.close();

        //out << N_name[i] << "\t" << real(sum_E / sum_pdi) << "\t" << real(cost_P_x * sum_Px / sum_pdi) << "\t" << real(cost_P_y * sum_Py / sum_pdi) << "\t"
          //  << real(cost_P * sum_Px / sum_pdi)*real(cost_P * sum_Px / sum_pdi) + real(cost_P * sum_Py / sum_pdi)*real(cost_P * sum_Py / sum_pdi) << "\t" << real(sum_pdi) << endl;
        E_now = real(sum_E / sum_pdi);
        out << N_name[i] << "\t" << N_name[i] * delta_t << "\t" << setw(10) << real(sum_E / sum_pdi);
        out << "\t" << setw(10) << E_before - E_now;
        out << "\t" << setw(10) << real(cost_P_x * sum_Px / sum_pdi) << "\t" << setw(10) << real(cost_P_y * sum_Py / sum_pdi) << "\t" << setw(10)
            << real(cost_P_x * sum_Px / sum_pdi)*real(cost_P_x * sum_Px / sum_pdi) + real(cost_P_y * sum_Py / sum_pdi)*real(cost_P_y * sum_Py / sum_pdi) << "\t"
            << real(sum_pdi) * delta_x * delta_y << "\t" << delta_x * sqrt(real(sum_x2 / sum_pdi - sum_x * sum_x / (sum_pdi * sum_pdi))) << "\t" << delta_y * sqrt(real(sum_y2 / sum_pdi - sum_y * sum_y / (sum_pdi * sum_pdi)))
            << setw(10) << delta_x * real(sum_x / sum_pdi) << "\t" << delta_y * real(sum_y / sum_pdi);
        out << endl;
        E_before = E_now;

        energy[i] = real(sum_E / sum_pdi);
        momentum_x[i] = real(cost_P_x * sum_Px / sum_pdi);
        momentum_y[i] = real(cost_P_y * sum_Py / sum_pdi);

        sum_E = 0;
        sum_Px = 0;
        sum_Py = 0;
        sum_pdi = 0;
        sum_x2 = 0; sum_x = 0;
        sum_y2 = 0; sum_y = 0;
    }

    //calculate sample mean and sample variance
    for(int i = 0; i < N_files; i++) {
        sample->mean_E += energy[i];
        sample->mean_Px += momentum_x[i];
        sample->mean_Py += momentum_y[i];
    }
    sample->mean_E /= N_files;
    sample->mean_Px /= N_files;
    sample->mean_Py /= N_files;

    for(int i = 0; i < N_files; i++) {
        sample->var_E += (energy[i] - sample->mean_E) * (energy[i] - sample->mean_E);
        sample->var_Px += (momentum_x[i] - sample->mean_Px) * (momentum_x[i] - sample->mean_Px);
        sample->var_Py += (momentum_y[i] - sample->mean_Py) * (momentum_y[i] - sample->mean_Py);
    }
    sample->var_E /= N_files - 1;
    sample->var_E = sqrt(sample->var_E);
    sample->var_Px /= N_files - 1;
    sample->var_Px = sqrt(sample->var_Px);
    sample->var_Py /= N_files - 1;
    sample->var_Py = sqrt(sample->var_Py);
}
