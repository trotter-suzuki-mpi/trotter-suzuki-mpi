#!/bin/sh
if [ -z "$MEX_BIN" ]
    then MEX_BIN="/usr/local/MATLAB/R2015a/bin/mex"
fi
$MEX_BIN -I../ MexTrotter.cpp ../cpublocksse.o ../common.o ../cpublock.o ../trotter.o -lgomp

