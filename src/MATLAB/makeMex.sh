#!/bin/sh
if [ -z "$MEX_BIN" ]
    then MEX_BIN="/usr/local/MATLAB/R2015a/bin/mex"
fi
$MEX_BIN -I../ MexTrotter.cpp ../cpublocksse.o ../common.o ../cpublock.o ../trotter.o ../solver.cpp -lgomp
$MEX_BIN -I../ MexH.cpp ../common.o -lgomp
$MEX_BIN -I../ MexK.cpp ../common.o -lgomp
$MEX_BIN -I../ MexNorm.cpp ../common.o -lgomp
