MPI_PATH=/usr
CUDA_PATH=/usr/local/cuda

OPTS=-g

CXX=$(MPI_PATH)/bin/mpicxx
CXXFLAGS=-O3 -march=native -ffast-math

NVCC=$(CUDA_PATH)/bin/nvcc
NVCCFLAGS=-O -arch=sm_20 -use_fast_math -Xptxas "-v" -Xcompiler "-O3 -march=native -ffast-math"

LDFLAGS=-lm -L$(CUDA_PATH)/lib64 -lcudart

INCLUDES=-I$(MPI_PATH)/include -I$(CUDA_PATH)/include

TARGETS=build/trottertest
OBJS=src/common.o src/cpublock.o src/cpublocksse.o src/cc2kernel.cu.co src/trottertest.o

all: $(TARGETS)

build/trottertest: $(OBJS)
		$(CXX) $(CXXFLAGS) $(INCLUDES) $(OPTS) -o $@ $^ $(LDFLAGS)

%.o: %.cpp
		$(CXX) $(CXXFLAGS) $(INCLUDES) $(OPTS) -o $@ -c $<

%.cu.co: %.cu
		$(NVCC) $(NVCCFLAGS) $(INCLUDES) $(OPTS) -o $@ -c $<

clean:
		rm -f $(OBJS) $(TARGETS)
