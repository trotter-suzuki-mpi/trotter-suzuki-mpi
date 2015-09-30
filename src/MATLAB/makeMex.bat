mex -D_OPENMP -DWIN32 COMPFLAGS="/openmp -I../windows $COMPFLAGS" -I../ MexTrotter.cpp ../cpublocksse.cpp ../common.cpp ../cpublock.cpp  ../trotter.cpp ../solver.cpp
mex -D_OPENMP -DWIN32 COMPFLAGS="/openmp -I../windows $COMPFLAGS" -I../ MexH.cpp ../common.cpp
mex -D_OPENMP -DWIN32 COMPFLAGS="/openmp -I../windows $COMPFLAGS" -I../ MexK.cpp ../common.cpp
mex -D_OPENMP -DWIN32 COMPFLAGS="/openmp -I../windows $COMPFLAGS" -I../ MexNorm.cpp ../common.cpp
