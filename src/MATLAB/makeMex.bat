mex -D_OPENMP -DWIN32 COMPFLAGS="/openmp -I../windows $COMPFLAGS" -I../ MexTrotter.cpp ../cpublocksse.cpp ../common.cpp ../cpublock.cpp  ../trotter.cpp 
