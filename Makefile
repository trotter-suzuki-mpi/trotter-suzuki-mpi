MC_PATH=/gpfs/apps/NVIDIA/PM/mcxx-dev/
MPI_PATH=/opt/mpi/openmpi
CUDA_PATH=/opt/cuda/4.0

OPTS=-g

#CXX=$(MPI_PATH)/bin/mpicxx
#CXXFLAGS=-O3 -march=native -ffast-math
CXX=$(MC_PATH)/bin/mcxx
CXXFLAGS=--ompss -k -O3 -ffast-math

#NVCC=$(CUDA_PATH)/bin/nvcc
#NVCCFLAGS=-O -arch=sm_20 -use_fast_math -Xptxas "-v" -Xcompiler "-O3 -march=native -ffast-math"

LDFLAGS=-lm -L$(CUDA_PATH)/lib64 -lcudart -lcublas -L $(MPI_PATH)/lib -lmpi -lmpi_cxx

INCLUDES=-I$(MPI_PATH)/include -I$(CUDA_PATH)/include

TARGETS=build/trottertest
OBJS=src/common.o src/cpublock.o src/trottertest.o

all: $(TARGETS)

build/trottertest: $(OBJS)
		$(CXX) $(CXXFLAGS) $(INCLUDES) $(OPTS) -o $@ $^ $(LDFLAGS)

%.o: %.cpp
		$(CXX) $(CXXFLAGS) $(INCLUDES) $(OPTS) -o $@ -c $<

#%.cu.co: %.cu
#		$(NVCC) $(NVCCFLAGS) $(INCLUDES) $(OPTS) -o $@ -c $<

clean:
		rm -f $(OBJS) $(TARGETS)
