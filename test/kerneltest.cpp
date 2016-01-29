#include <iostream>
#include "kerneltest.h"

template<class F>
void my_test<F>::free_particle_test() {
	Lattice *grid = new Lattice(100, 20, 20, true, true);
	State *state = new ExponentialState(grid);
	Hamiltonian *hamiltonian = new Hamiltonian(grid, NULL);
	Solver *solver = new Solver(grid, state, hamiltonian, 5.e-3, this->kernel_type);
	double ini_tot_energy = solver->get_total_energy();
	double ini_norm = solver->get_squared_norm();
	solver->evolve(100);
	double tot_energy = solver->get_total_energy();
	double norm = solver->get_squared_norm();
	delete solver;
  delete hamiltonian;
  delete state;
  delete grid;
  //Check
  CPPUNIT_ASSERT( std::abs(ini_tot_energy - tot_energy) < TOLERANCE );
  CPPUNIT_ASSERT( std::abs(ini_norm - norm) < NORM_TOLERANCE );
  std::cout << "TEST FUNCTION: free_particle_test -> PASSED! " << std::endl;
}

template<class F>
void my_test<F>::harmonic_oscillator_test() {
	Lattice *grid = new Lattice(100, 20, 20);
	State *state = new GaussianState(grid, 1.);
	Potential *potential = new HarmonicPotential(grid, 1., 1.);
	Hamiltonian *hamiltonian = new Hamiltonian(grid, potential);
	Solver *solver = new Solver(grid, state, hamiltonian, 5.e-3, this->kernel_type);
	double ini_tot_energy = solver->get_total_energy();
	double ini_norm = solver->get_squared_norm();
	solver->evolve(100);
	double tot_energy = solver->get_total_energy();
	double norm = solver->get_squared_norm();
	delete solver;
  delete hamiltonian;
  delete state;
  delete grid;
	//Check
	CPPUNIT_ASSERT( std::abs(ini_tot_energy - tot_energy) < TOLERANCE );
	CPPUNIT_ASSERT( std::abs(ini_norm - norm) < NORM_TOLERANCE );
    std::cout << "TEST FUNCTION: harmonic_oscillator_test -> PASSED! " << std::endl;
}

template<class F>
void my_test<F>::imaginary_harmonic_oscillator_test() {
	double std_energy = 1.00001;
	Lattice *grid = new Lattice(100, 20, 20);
	State *state = new GaussianState(grid, 0.5);
	Potential *potential = new HarmonicPotential(grid, 1., 1.);
	Hamiltonian *hamiltonian = new Hamiltonian(grid, potential);
	Solver *solver = new Solver(grid, state, hamiltonian, 5.e-3, this->kernel_type);
	double ini_norm = solver->get_squared_norm();
	solver->evolve(1000, true);
	double tot_energy = solver->get_total_energy();
	double norm = solver->get_squared_norm();
	delete solver;
  delete hamiltonian;
  delete state;
  delete grid;
	//Check
	CPPUNIT_ASSERT( std::abs(std_energy - tot_energy) < TOLERANCE );
	CPPUNIT_ASSERT( std::abs(ini_norm - norm) < NORM_TOLERANCE );
    std::cout << "TEST FUNCTION: imaginary_harmonic_oscillator_test -> PASSED! " << std::endl;
}

template<class F>
void my_test<F>::intra_particle_interaction_test() {
	double std_mean_XX = 1.05368;
	Lattice *grid = new Lattice(100, 20, 20);
	State *state = new GaussianState(grid, 1);
	Potential *potential = new HarmonicPotential(grid, 1., 1.);
	Hamiltonian *hamiltonian = new Hamiltonian(grid, potential, 1., 10);
	Solver *solver = new Solver(grid, state, hamiltonian, 1.e-3, this->kernel_type);
	double ini_tot_energy = solver->get_total_energy();
	double ini_norm = solver->get_squared_norm();
	solver->evolve(1000);
	double tot_energy = solver->get_total_energy();
	double mean_XX = state->get_mean_xx();
	double norm = solver->get_squared_norm();
	delete solver;
	delete hamiltonian;
	delete state;
	delete grid;
	//Check
	CPPUNIT_ASSERT( std::abs(ini_tot_energy - tot_energy) < TOLERANCE );
	CPPUNIT_ASSERT( std::abs(std_mean_XX - mean_XX) < TOLERANCE );
	CPPUNIT_ASSERT( std::abs(ini_norm - norm) < NORM_TOLERANCE );
	std::cout << "TEST FUNCTION: intra_particle_interaction_test -> PASSED! " << std::endl;
}

template<class F>
void my_test<F>::imaginary_intra_particle_interaction_test() {
	double std_energy = 1.59273;
	double std_mean_XX = 0.780077;
	Lattice *grid = new Lattice(100, 20, 20);
	State *state = new GaussianState(grid, 1);
	Potential *potential = new HarmonicPotential(grid, 1., 1.);
	Hamiltonian *hamiltonian = new Hamiltonian(grid, potential, 1., 10);
	Solver *solver = new Solver(grid, state, hamiltonian, 1.e-3, this->kernel_type);
	double ini_norm = solver->get_squared_norm();
	solver->evolve(1000, true);
	double tot_energy = solver->get_total_energy();
	double mean_XX = state->get_mean_xx();
	double norm = solver->get_squared_norm();
	delete solver;
	delete hamiltonian;
	delete state;
	delete grid;
	//Check
	CPPUNIT_ASSERT( std::abs(std_energy - tot_energy) < TOLERANCE );
	CPPUNIT_ASSERT( std::abs(std_mean_XX - mean_XX) < TOLERANCE );
	CPPUNIT_ASSERT( std::abs(ini_norm - norm) < NORM_TOLERANCE );
	std::cout << "TEST FUNCTION: imaginary_intra_particle_interaction_test -> PASSED! " << std::endl;
}

template<class F>
void my_test<F>::rotating_frame_of_reference_test() {
	double angular_velocity = 0.7;
	Lattice *grid = new Lattice(300, 20, 20, false, false, angular_velocity);
	State *state = new GaussianState(grid, 1);
	Potential *potential = new HarmonicPotential(grid, 1., 1.);
	Hamiltonian *hamiltonian = new Hamiltonian(grid, potential, 1., 100., angular_velocity);
	Solver *solver = new Solver(grid, state, hamiltonian, 1.e-4, this->kernel_type);
	double ini_tot_energy = solver->get_total_energy();
	double ini_norm = solver->get_squared_norm();
	solver->evolve(1000);
	double tot_energy = solver->get_total_energy();
	double norm = solver->get_squared_norm();
	delete solver;
	delete hamiltonian;
	delete state;
	delete grid;
	//Check
	CPPUNIT_ASSERT( std::abs(ini_tot_energy - tot_energy) < TOLERANCE*10. );
	CPPUNIT_ASSERT( std::abs(ini_norm - norm) < NORM_TOLERANCE );
	std::cout << "TEST FUNCTION: rotating_frame_of_reference_test -> PASSED! " << std::endl;

}

template<class F>
void my_test<F>::imaginary_rotating_frame_of_reference_test() {
	double fin_energy = 4.89895;
	double angular_velocity = 0.7;
	Lattice *grid = new Lattice(300, 20, 20, false, false, angular_velocity);
	State *state = new GaussianState(grid, 1);
	Potential *potential = new HarmonicPotential(grid, 1., 1.);
	Hamiltonian *hamiltonian = new Hamiltonian(grid, potential, 1., 100., angular_velocity);
	Solver *solver = new Solver(grid, state, hamiltonian, 1.e-4, this->kernel_type);
	double ini_norm = solver->get_squared_norm();
	solver->evolve(1000, true);
	double tot_energy = solver->get_total_energy();
	double norm = solver->get_squared_norm();
	delete solver;
	delete hamiltonian;
	delete state;
	delete grid;
	//Check
	CPPUNIT_ASSERT( std::abs(fin_energy - tot_energy) < TOLERANCE );
	CPPUNIT_ASSERT( std::abs(ini_norm - norm) < NORM_TOLERANCE );
	std::cout << "TEST FUNCTION: imaginary_rotating_frame_of_reference_test -> PASSED! " << std::endl;

}

template<class F>
void my_test<F>::mixed_BEC_test() {
	Lattice *grid = new Lattice(100, 20, 20);
	State *state1 = new GaussianState(grid, 1);
	State *state2 = new State(grid);
	Potential *potential = new HarmonicPotential(grid, 1., 1.);
	Hamiltonian2Component *hamiltonian = new Hamiltonian2Component(grid, potential, potential, 1., 1., 0., 0., 0., 2.*M_PI/10.);
	Solver *solver = new Solver(grid, state1, state2, hamiltonian, 1.e-3, this->kernel_type);
	double ini_tot_energy = solver->get_total_energy();
	double ini_norm = solver->get_squared_norm();
	double ini_norm1 = state1->get_squared_norm();
	double ini_norm2 = state2->get_squared_norm();
	solver->evolve(5000);
	double tot_energy = solver->get_total_energy();
	double norm = solver->get_squared_norm();
	double norm1 = state1->get_squared_norm();
	double norm2 = state2->get_squared_norm();
	delete solver;
	delete hamiltonian;
	delete state1;
	delete state2;
	delete grid;
	//Check
	CPPUNIT_ASSERT( std::abs(ini_tot_energy - tot_energy) < TOLERANCE );
	CPPUNIT_ASSERT( std::abs(ini_norm - norm) < NORM_TOLERANCE );
	CPPUNIT_ASSERT( std::abs(ini_norm1 - norm2) < NORM_TOLERANCE );
	CPPUNIT_ASSERT( std::abs(ini_norm2 - norm1) < NORM_TOLERANCE );
	std::cout << "TEST FUNCTION: mixed_BEC_test -> PASSED! " << std::endl;
}

template<class F>
void my_test<F>::imaginary_mixed_BEC_test() {
	double std_norm1 = 0.915292;
	double std_norm2 = 0.084708;
	Lattice *grid = new Lattice(100, 20, 20);
	State *state1 = new GaussianState(grid, 1);
	State *state2 = new State(grid);
	Potential *potential = new HarmonicPotential(grid, 1., 1.);
	Hamiltonian2Component *hamiltonian = new Hamiltonian2Component(grid, potential, potential, 1., 1., 0., 0., 0., 2.*M_PI/10.);
	Solver *solver = new Solver(grid, state1, state2, hamiltonian, 1.e-3, this->kernel_type);
	double ini_tot_energy = solver->get_total_energy();
	double ini_norm = solver->get_squared_norm();
	solver->evolve(1000, true);
	double tot_energy = solver->get_total_energy();
	double norm = solver->get_squared_norm();
	double norm1 = state1->get_squared_norm();
	double norm2 = state2->get_squared_norm();
	delete solver;
	delete hamiltonian;
	delete state1;
	delete state2;
	delete grid;
	//Check
	CPPUNIT_ASSERT( std::abs(ini_tot_energy - tot_energy) < TOLERANCE );
	CPPUNIT_ASSERT( std::abs(ini_norm - norm) < NORM_TOLERANCE );
	CPPUNIT_ASSERT( std::abs(std_norm1 - norm1) < NORM_TOLERANCE );
	CPPUNIT_ASSERT( std::abs(std_norm2 - norm2) < NORM_TOLERANCE );
	std::cout << "TEST FUNCTION: imaginary_mixed_BEC_test -> PASSED! " << std::endl;
}

void CpuKernelTest::setUp() {
    this->kernel_type = "cpu";
}

#ifdef CUDA
void GpuKernelTest::setUp() {
    this->kernel_type = "gpu";
}

void HybridKernelTest::setUp() {
    this->kernel_type = "hybrid";
}
#endif
