#include <fstream>
#include <unistd.h>
#include <stdlib.h>
#include <iostream>
#include <complex>
#include "trotterkernel.h" 

#define DIM 640
#define ITERATIONS 1000
#define KERNEL_TYPE 0
#define SNAPSHOTS 0

void init_p(float *p_real, float *p_imag, int dimx, int dimy) {
    double s = 64.0; // FIXME: y esto?
    for (int y = 1; y <= dimy; y++) {
        for (int x = 1; x <= dimx; x++) {
            std::complex<float> tmp = std::complex<float>(exp(-(pow(x - 180.0, 2.0) + pow(y - 300.0, 2.0)) / (2.0 * pow(s, 2.0))), 0.0)
                                      * exp(std::complex<float>(0.0, 0.4 * (x + y - 480.0)));

            p_real[y * dimx + x] = real(tmp);
            p_imag[y * dimx + x] = imag(tmp);
        }
    }
}

void print_usage() {
    std::cout << "Usage:\n" \
              "     trotter [OPTION]\n" \
              "Arguments:\n" \
              "     -d NUMBER     Matrix dimension (default: " << DIM << ")\n" \
              "     -i NUMBER     Number of iterations (default: " << ITERATIONS << ")\n" \
              "     -k NUMBER     Kernel type (default: " << KERNEL_TYPE << "): \n" \
              "                      0: CPU, cache-optimized\n" \
              "                      1: CPU, SSE and cache-optimized\n" \
              "                      2: GPU\n" \
              "                      3: Hybrid (experimental) \n" \
              "     -s NUMBER     Snapshots are taken at every NUMBER of iterations.\n" \
              "                   Zero means no snapshots. Default: " << SNAPSHOTS << ".\n";
}

void process_command_line(int argc, char** argv, int *dim, int *iterations, int *snapshots, int *kernel_type) {
    // Setting default values
    *dim = DIM;
    *iterations = ITERATIONS;
    *snapshots = SNAPSHOTS;
    *kernel_type = KERNEL_TYPE;

    int c;
    while ((c = getopt (argc, argv, "d:hi:k:s:")) != -1) {
        switch (c) {
        case 'd':
            *dim = atoi(optarg);
            if (*dim <= 0) {
                fprintf (stderr, "The argument of option -d should be a positive integer.\n");
                abort ();
            }
            break;
        case 'i':
            *iterations = atoi(optarg);
            if (*iterations <= 0) {
                fprintf (stderr, "The argument of option -i should be a positive integer.\n");
                abort ();
            }
            break;
        case 'h':
            print_usage();
            abort ();
            break;
        case 'k':
            *kernel_type = atoi(optarg);
            if (*kernel_type < 0 || *kernel_type > 3) {
                fprintf (stderr, "The argument of option -k should be a valid kernel.\n");
                abort ();
            }
            break;
        case 's':
            *snapshots = atoi(optarg);
            if (*snapshots <= 0) {
                fprintf (stderr, "The argument of option -s should be a positive integer.\n");
                abort ();
            }
            break;
        case '?':
            if (optopt == 'd' || optopt == 'i' || optopt == 'k' || optopt == 's') {
                fprintf (stderr, "Option -%c requires an argument.\n", optopt);
                print_usage();
                abort ();
            }
            else if (isprint (optopt)) {
                fprintf (stderr, "Unknown option `-%c'.\n", optopt);
                print_usage();
                abort ();
            }
            else {
                fprintf (stderr, "Unknown option character `\\x%x'.\n",  optopt);
                print_usage();
                abort ();
            }
        default:
            abort ();
        }
    }
}

int main(int argc, char** argv){
	int dim = 0, iterations = 0, snapshots = 0, kernel_type = 0;
	float * p_real;
	float * p_imag;
   
	process_command_line(argc, argv, &dim, &iterations, &snapshots, &kernel_type);
	p_real = new float[dim*dim];
	p_imag = new float[dim*dim];
	
	init_p(p_real, p_imag, dim, dim);
	trotter(p_real, p_imag, dim, dim, iterations, snapshots, kernel_type, argc, argv);

    return 0;
}
