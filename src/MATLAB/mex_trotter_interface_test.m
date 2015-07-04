imag_time = 0;
order_approx = 2;
dim = 640;
iterations = 600;
kernel_type = 0;
periods = zeros(1,2);

particle_mass = 1;
time_single_it = 0.08 * particle_mass / 2;

h_a = cos(time_single_it / (2. * particle_mass));
h_b = sin(time_single_it / (2. * particle_mass));

p_real = zeros(dim, dim);
p_imag = zeros(dim, dim);
pot_r = zeros(dim, dim);
pot_i = zeros(dim, dim);

CONST = -1. * time_single_it * order_approx;
for y = 1:dim
    for x = 1:dim
        p_real(y, x) = sin(2 * pi * x / dim) * sin(2 * pi * y / dim);
        pot_r(y, x) = cos(CONST * pot_r(y, x));
    pot_i(y, x) = sin(CONST * pot_i(y, x));
    end
end

[pf_real, pf_imag] = MexTrotter(h_a, h_b, pot_r, pot_i, p_real, p_imag, iterations, kernel_type, periods, imag_time);

colormap('hot');
imagesc(pf_real);
colorbar
