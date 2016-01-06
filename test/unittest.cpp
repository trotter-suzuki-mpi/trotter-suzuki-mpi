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

#include <cppunit/CompilerOutputter.h>
#include <cppunit/extensions/TestFactoryRegistry.h>
#include <cppunit/ui/text/TestRunner.h>
#include <string>
#include <sstream>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <complex>
#include <sys/stat.h>

#include "trottersuzuki.h"
#include "kernel.h"
#include "common.h"
#ifdef HAVE_MPI
#include <mpi.h>
#endif
#include "model.cpp"

#define LENGHT 50
#define DIM 640
#define ITERATIONS 1000
#define PARTICLES_NUM 1700000
#define KERNEL_TYPE 0
#define SNAPSHOTS 10
#define SCATTER_LENGHT_2D 5.662739242e-5

int rot_coord_x = 320, rot_coord_y = 320;
double omega = 0;

struct MAGIC_NUMBER {
    double threshold_E, threshold_P;
    double expected_E;
    double expected_Px;
    double expected_Py;
    MAGIC_NUMBER();
};

MAGIC_NUMBER::MAGIC_NUMBER() : threshold_E(3), threshold_P(3),
    expected_E((2. * M_PI / DIM) * (2. * M_PI / DIM)), expected_Px(0), expected_Py(0) {}

std::complex<double> gauss_ini_state(int m, int n, int matrix_width, int matrix_height, int * periods, int halo_x, int halo_y) {
	double delta_x = double(LENGHT)/double(DIM);
    double x = (m - matrix_width / 2.) * delta_x, y = (n - matrix_height / 2.) * delta_x;
    double w = 0.01;
    return std::complex<double>(sqrt(w * double(PARTICLES_NUM) / M_PI) * exp(-(x * x + y * y) * 0.5 * w), 0.0);
}

std::complex<double> sinus_state(int m, int n, int matrix_width, int matrix_height, int * periods, int halo_x, int halo_y) {
	double delta_x = double(LENGHT)/double(DIM);
	double x = m * delta_x, y = n * delta_x;
	return std::complex<double>(sin(2 * M_PI * x / double(LENGHT)) * sin(2 * M_PI * y / double(LENGHT)), 0.0);
}

double parabolic_potential(int m, int n, int matrix_width, int matrix_height, int * periods, int halo_x, int halo_y) {
    double delta_x = double(LENGHT)/double(DIM);
    double x = (m - matrix_width / 2.) * delta_x, y = (n - matrix_width / 2.) * delta_x;
    double w_x = 1, w_y = 1. / sqrt(2); 
    return 0.5 * (w_x * w_x * x * x + w_y * w_y * y * y);
}

/**
 * \brief Structure defining expected values calculated by expect_values().
 */
struct energy_momentum_statistics {
    double mean_E;	///< Expected total energy.
    double mean_Px; ///< Expected momentum along x axis.
    double mean_Py; ///< Expected momentum along y axis.
    double var_E;	///< Expected total energy variation.
    double var_Px;	///< Expected momentum along x axis variation.
    double var_Py;	///< Expected momentum along y axis variation.
    energy_momentum_statistics() : mean_E(0.), mean_Px(0.), mean_Py(0.),
        var_E(0.), var_Px(0.), var_Py(0.) {}
};

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


int main(int argc, char** argv) {
    int dim = DIM, iterations = ITERATIONS, snapshots = SNAPSHOTS, kernel_type = KERNEL_TYPE;
    int periods[2] = {0, 0};
    char file_name[] = "";
    char pot_name[1] = "";
    const double particle_mass = 1.;
    bool show_time_sim = false;
    bool imag_time = false;
    double norm = 1.5;
    double h_a = 0.;
    double h_b = 0.;
	
	double delta_t = 5.e-5;
	double delta_x = double(LENGHT)/double(DIM), delta_y = double(LENGHT)/double(DIM);
	
    // Get the top level suite from the registry
    CppUnit::Test *suite = CppUnit::TestFactoryRegistry::getRegistry().makeTest();
    // Adds the test to the list of test to run
    CppUnit::TextUi::TestRunner runner;
    runner.addTest( suite );
    // Change the default outputter to a compiler error format outputter
    runner.setOutputter( new CppUnit::CompilerOutputter( &runner.result(), std::cerr ) );
    // Run the tests.
    bool wasSucessful = runner.run();
    // Return error code 1 if the one of test failed.
    if(!wasSucessful)
        return 1;

    int halo_x = (kernel_type == 2 ? 3 : 4);
    int halo_y = 4;
    int matrix_width = dim + periods[1] * 2 * halo_x;
    int matrix_height = dim + periods[0] * 2 * halo_y;

#ifdef HAVE_MPI
    MPI_Init(&argc, &argv);
#endif
    //define the topology
    int coords[2], dims[2] = {0, 0};
    int rank;
    int nProcs;
#ifdef HAVE_MPI
    MPI_Comm cartcomm;
    MPI_Comm_size(MPI_COMM_WORLD, &nProcs);
    MPI_Dims_create(nProcs, 2, dims);  //partition all the processes (the size of MPI_COMM_WORLD's group) into an 2-dimensional topology
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 0, &cartcomm);
    MPI_Comm_rank(cartcomm, &rank);
    MPI_Cart_coords(cartcomm, rank, 2, coords);
#else
    nProcs = 1;
    rank = 0;
    dims[0] = dims[1] = 1;
    coords[0] = coords[1] = 0;
#endif

    //set dimension of tiles and offsets
    int start_x, end_x, inner_start_x, inner_end_x,
        start_y, end_y, inner_start_y, inner_end_y;
    calculate_borders(coords[1], dims[1], &start_x, &end_x, &inner_start_x, &inner_end_x, matrix_width - 2 * periods[1]*halo_x, halo_x, periods[1]);
    calculate_borders(coords[0], dims[0], &start_y, &end_y, &inner_start_y, &inner_end_y, matrix_height - 2 * periods[0]*halo_y, halo_y, periods[0]);
    int tile_width = end_x - start_x;
    int tile_height = end_y - start_y;
    
    //set and calculate evolution operator variables from hamiltonian
    double time_single_it;
    double coupling_const = 4. * M_PI * double(SCATTER_LENGHT_2D);
    double *external_pot_real = new double[tile_width * tile_height];
    double *external_pot_imag = new double[tile_width * tile_height];
    double (*hamiltonian_pot)(int x, int y, int matrix_width, int matrix_height, int * periods, int halo_x, int halo_y);
    hamiltonian_pot = const_potential;

    if(imag_time) {
        time_single_it = delta_t / 2.;	//second approx trotter-suzuki: time/2
        if(h_a == 0. && h_b == 0.) {
            h_a = cosh(time_single_it / (2. * particle_mass * delta_x * delta_y));
            h_b = sinh(time_single_it / (2. * particle_mass * delta_x * delta_y));
        }
    }
    else {
        time_single_it = delta_t / 2.;	//second approx trotter-suzuki: time/2
        if(h_a == 0. && h_b == 0.) {
            h_a = cos(time_single_it / (2. * particle_mass * delta_x * delta_y));
            h_b = sin(time_single_it / (2. * particle_mass * delta_x * delta_y));
        }
    }
    initialize_exp_potential(external_pot_real, external_pot_imag, pot_name, hamiltonian_pot, tile_width, tile_height, matrix_width, matrix_height,
                             start_x, start_y, periods, coords, dims, halo_x, halo_y, time_single_it, particle_mass, imag_time);

    //set initial state
    double *p_real = new double[tile_width * tile_height];
    double *p_imag = new double[tile_width * tile_height];
    std::complex<double> (*ini_state)(int x, int y, int matrix_width, int matrix_height, int * periods, int halo_x, int halo_y);
    ini_state = sinus_state;
    initialize_state(p_real, p_imag, file_name, ini_state, tile_width, tile_height, matrix_width, matrix_height, start_x, start_y,
                     periods, coords, dims, halo_x, halo_y);

    //set file output directory
    std::stringstream filename;
    std::string filenames;
    if(snapshots) {
        int status = 0;

        filename.str("");
        filename << "D" << dim << "_I" << iterations << "_S" << snapshots << "";
        filenames = filename.str();

        status = mkdir(filenames.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);

        if(status != 0 && status != -1)
            filenames = ".";
    }
    else
        filenames = ".";

    for(int count_snap = 0; count_snap <= snapshots; count_snap++) {
        stamp(p_real, p_imag, matrix_width, matrix_height, halo_x, halo_y, start_x, inner_start_x, inner_end_x, end_x,
              start_y, inner_start_y, inner_end_y, dims, coords, periods,
              0, iterations, count_snap, filenames.c_str()
#ifdef HAVE_MPI
              , cartcomm
#endif
             );

		std::stringstream filename1;
		std::string filenames1;
		filename1.str("");
        filename1 << "new/D" << dim << "_I" << iterations << "_" << count_snap << "_" << rank << "";
        filenames1 = filename1.str();
        
		print_matrix(filenames1.c_str(), p_real, tile_width, tile_width, tile_height);
        if(count_snap != snapshots) {
            trotter(h_a, h_b, coupling_const, external_pot_real, external_pot_imag, omega, rot_coord_x, rot_coord_y, p_real, p_imag, delta_x, delta_y, matrix_width, matrix_height, delta_t, iterations, kernel_type, periods, norm, imag_time);
        }
    }

    if(rank == 0) {
        MAGIC_NUMBER th_values;
        energy_momentum_statistics sample;
        double hamilt_pot[dim * dim];

        initialize_potential(hamilt_pot, hamiltonian_pot, dim, dim, periods, halo_x, halo_y);
        expect_values(dim, dim, delta_x, delta_y, delta_t, 4. * M_PI * double(SCATTER_LENGHT_2D), iterations, snapshots, hamilt_pot, particle_mass, filenames.c_str(), periods, halo_x, halo_y, &sample);

        if(std::abs(sample.mean_E - th_values.expected_E) / sample.var_E < th_values.threshold_E)
            std::cout << "Energy -> OK\tsigma: " << std::abs(sample.mean_E - th_values.expected_E) / sample.var_E << std::endl;
        else
            std::cout << "Energy value is not the one theoretically expected: sigma " << std::abs(sample.mean_E - th_values.expected_E) / sample.var_E << std::endl;
        if(std::abs(sample.mean_Px - th_values.expected_Px) / sample.var_Px < th_values.threshold_P)
            std::cout << "Momentum Px -> OK\tsigma: " << std::abs(sample.mean_Px - th_values.expected_Px) / sample.var_Px << std::endl;
        else
            std::cout << "Momentum Px value is not the one theoretically expected: sigma " << std::abs(sample.mean_Px - th_values.expected_Px) / sample.var_Px << std::endl;
        if(std::abs(sample.mean_Py - th_values.expected_Py) / sample.var_Py < th_values.threshold_P)
            std::cout << "Momentum Py -> OK\tsigma: " << std::abs(sample.mean_Py - th_values.expected_Py) / sample.var_Py << std::endl;
        else
            std::cout << "Momentum Py value is not the one theoretically expected: sigma " << std::abs(sample.mean_Py - th_values.expected_Py) / sample.var_Py << std::endl;
    }

#ifdef HAVE_MPI
    MPI_Finalize();
#endif
    return 0;
}
