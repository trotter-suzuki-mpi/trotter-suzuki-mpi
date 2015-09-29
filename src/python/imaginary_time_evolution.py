import numpy as np
import trottersuzuki as ts
from matplotlib import pyplot as plt


# lattice parameters
dim = 200
delta_x = 1.
delta_y = 1.
periods = [0, 0]

# Hamiltonian parameter
particle_mass = 1
coupling_const = 0.
external_potential = np.zeros((dim, dim))

# initial state
p_real = np.ones((dim,dim))
p_imag = np.zeros((dim,dim))	

# evolution parameters
imag_time = True
delta_t = 0.1
iterations = 2000
kernel_type = 1

for i in range(0, 15):
    # launch evolution
    ts.solver(p_real, p_imag, particle_mass, coupling_const, external_potential, delta_x, delta_y, delta_t, iterations, kernel_type, periods, imag_time)
    print ts.H(p_real, p_imag, particle_mass, coupling_const, external_potential, delta_x, delta_y)
    print ts.Norm2(p_real, p_imag, delta_x, delta_y)


heatmap = plt.pcolor(p_real)
plt.show()
