imag_time = 0;        % 0 for real time evolution, 1 for imaginary time evolution
order_approx = 2;
dim = 640;            % linear dimensione of the lattice
iterations = 600;     % number of iteration
kernel_type = 0;      % use CPU kernel
periods = zeros(1,2); % 0 for closed boundary conditions, 1 for periodic boundary conditions
particle_mass = 1;

% External potential defined as constant.
pot_r = zeros(dim, dim);

% Initialization of the initial state.
p_real = zeros(dim, dim);
p_imag = zeros(dim, dim);
for y = 1:dim
    for x = 1:dim
        p_real(y, x) = sin(2 * pi * x / dim) * sin(2 * pi * y / dim);
    end
end

% Calculate the evolution operator.
if imag_time == 0
    pot_i = zeros(dim, dim);
    CONST = -1. * time_single_it * order_approx;
    for y = 1:dim
        for x = 1:dim
            pot_r(y, x) = cos(CONST * pot_r(y, x));
            pot_i(y, x) = sin(CONST * pot_i(y, x));
        end
    end

    time_single_it = 0.08 * particle_mass / 2;
    h_a = cos(time_single_it / (2. * particle_mass));
    h_b = sin(time_single_it / (2. * particle_mass));
else
    pot_i = zeros(dim, dim);
    CONST = -1. * time_single_it * order_approx;
    for y = 1:dim
        for x = 1:dim
            pot_r(y, x) = exp(CONST * pot_r(y, x));
            pot_i(y, x) = 0;
        end
    end

    constant = 6;
    time_single_it = 8 * particle_mass / 2.;
    h_a = cosh(time_single_it / (2. * particle_mass)) / constant;
    h_b = sinh(time_single_it / (2. * particle_mass)) / constant;
end

[pf_real, pf_imag] = MexTrotter(h_a, h_b, pot_r, pot_i, p_real, p_imag, iterations, kernel_type, periods, imag_time);

colormap('hot');
imagesc(pf_real);
colorbar
