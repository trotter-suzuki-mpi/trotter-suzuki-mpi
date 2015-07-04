import numpy as np
import trottersuzuki as ts
from matplotlib import pyplot as plt

imag_time = 0
order_approx = 2
dim = 640
iterations = 200
kernel_type = 0
periods = np.zeros(2, dtype=np.int32)
time = np.zeros(1, dtype=np.int32)

particle_mass = 1
time_single_it = 0.08 * particle_mass / 2

h_a = np.cos(time_single_it / (2. * particle_mass))
h_b = np.sin(time_single_it / (2. * particle_mass))

p_real = np.ones((dim,dim))
p_imag = np.zeros((dim,dim))
pot_r = np.zeros((dim, dim))
pot_i = np.zeros((dim, dim))

CONST = -1. * time_single_it * order_approx
for y in range(0, dim):
    for x in range(0, dim):
        p_real[y, x] = np.sin(2 * np.pi * x / dim) * np.sin(2 * np.pi * y / dim)
        pot_r[y, x] = np.cos(CONST * pot_r[y, x])
	pot_i[y, x] = np.sin(CONST * pot_i[y, x])

ts.trotter_func(h_a, h_b, pot_r, pot_i, p_real, p_imag, iterations, kernel_type, periods, imag_time, time)

heatmap = plt.pcolor(p_real)
plt.show()
