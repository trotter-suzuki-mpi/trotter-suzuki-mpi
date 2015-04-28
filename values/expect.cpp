/**
 * Distributed Trotter-Suzuki solver
 * Copyright (C) 2012 Peter Wittek, 2010-2012 Carlos Bederián
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

#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <unistd.h>
#include <complex>

using namespace std;

void expect_values(int dim, int iterations, int snapshots, float * hamilt_pot, float particle_mass, const char *dirname) {

    if(snapshots == 0)
        return;

    int N_files = iterations / snapshots;
    int N_name[N_files];
    int DIM = dim;
    N_name[0] = 0;
    for(int i = 1; i < N_files; i++) {
        N_name[i] = N_name[i - 1] + snapshots;
    }

    complex<float> sum_E = 0;
    complex<float> sum_Px = 0, sum_Py = 0;
    complex<float> sum_psi = 0;

    complex<float> potential[DIM][DIM];
    complex<float> psi[DIM][DIM];
    complex<float> cost_E = -1. / (2.*particle_mass), cost_P;
    cost_P = complex<float>(0., -0.5);

    stringstream filename;
    string filenames;

    filename.str("");
    filename << dirname << "/exp_val_D" << dim << "_I" << iterations << "_S" << snapshots << ".dat";
    filenames = filename.str();
    ofstream out(filenames.c_str());

    for(int i = 0; i < DIM; i++) {
        for(int j = 0; j < DIM; j++) {
            potential[j][i] = complex<float> (hamilt_pot[i * DIM + j], 0.);
        }
    }

    out << "#time\tEnergy\t\tPx\tPy\tP**2\tnorm(psi(t))" << endl;
    for(int i = 0; i < N_files; i++) {
        filename.str("");
        filename << dirname << "/" << N_name[i] << "-iter-0-0-comp.dat";
        filenames = filename.str();
        ifstream in_compl(filenames.c_str());

        for(int j = 0; j < DIM; j++) {
            for(int k = 0; k < DIM; k++) {
                in_compl >> psi[k][j];
            }
        }
        in_compl.close();

        for(int j = 1; j < DIM - 1; j++) {
            for(int k = 1; k < DIM - 1; k++) {
                sum_E += conj(psi[k][j]) * (cost_E * (psi[k + 1][j] + psi[k - 1][j] + psi[k][j + 1] + psi[k][j - 1] - psi[k][j] * complex<float> (4., 0.)) + potential[k][j] * psi[k][j]) ;
                sum_Px += conj(psi[k][j]) * (psi[k + 1][j] - psi[k - 1][j]);
                sum_Py += conj(psi[k][j]) * (psi[k][j + 1] - psi[k][j - 1]);
                sum_psi += conj(psi[k][j]) * psi[k][j];
            }
        }

        out << N_name[i] << "\t" << real(sum_E / sum_psi) << "\t" << real(cost_P * sum_Px / sum_psi) << "\t" << real(cost_P * sum_Py / sum_psi) << "\t"
            << real(cost_P * sum_Px / sum_psi)*real(cost_P * sum_Px / sum_psi) + real(cost_P * sum_Py / sum_psi)*real(cost_P * sum_Py / sum_psi) << "\t" << real(sum_psi) << endl;
        sum_E = 0;
        sum_Px = 0;
        sum_Py = 0;
        sum_psi = 0;
    }
}

