# lattice parameters
dim = 200;					% linear dimensione of the lattice
delta_x = 1.;				% physical resolution along the x axis
delta_y = 1.;				% physical resolution along the y axis
periods = zeros(1,2); 		% 0 for closed boundary conditions, 1 for periodic boundary conditions

# Hamiltonian parameters
particle_mass = 1;
coupling_const = 0.;
external_potential = zeros(dim, dim);

% initial state.
p_real = zeros(dim, dim);
p_imag = zeros(dim, dim);
for y = 1:dim
    for x = 1:dim
        p_real(y, x) = sin(2 * pi * x / dim) * sin(2 * pi * y / dim);
    end
end

# evolution parameters
imag_time = 0;			% 0 for real time evolution, 1 for imaginary time evolution
delta_t = 0.001;		
iterations = 200;		% number of iteration
kernel_type = 0;		% use CPU kernel

[pf_real, pf_imag] = MexTrotter(p_real, p_imag, particle_mass, coupling_const, external_potential, delta_x, delta_y, delta_t, iterations, kernel_type, periods, imag_time);

TotalEnergy = MexH(pf_real, pf_imag, particle_mass, coupling_const, external_potential, delta_x, delta_y);
KineticEnergy = MexK(pf_real, pf_imag, particle_mass, delta_x, delta_y);
Norm2 = MexNorm(pf_real, pf_imag, delta_x, delta_y);

colormap('hot');
imagesc(pf_real);
colorbar
