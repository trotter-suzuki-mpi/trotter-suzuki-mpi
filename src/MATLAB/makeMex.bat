mex -D_OPENMP -DWIN32 COMPFLAGS="/openmp -I../windows $COMPFLAGS" -I../ MexSolver.cpp ../common.cpp ../cpublock.cpp  ../model.cpp ../solver.cpp
mex -D_OPENMP -DWIN32 COMPFLAGS="/openmp -I../windows $COMPFLAGS" -I../ MexH.cpp ../common.cpp
mex -D_OPENMP -DWIN32 COMPFLAGS="/openmp -I../windows $COMPFLAGS" -I../ MexK.cpp ../common.cpp
mex -D_OPENMP -DWIN32 COMPFLAGS="/openmp -I../windows $COMPFLAGS" -I../ MexNorm.cpp ../common.cpp
