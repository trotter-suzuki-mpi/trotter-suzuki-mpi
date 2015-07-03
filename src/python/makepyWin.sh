#!/bin/sh
rm -rf *.h *.i *.cpp *.cu *.cxx dist trottersuzuki.egg*
cp ../*.h .
cp ../*.i .
cp ../*.cpp .
cp ../*.cu .
python setupWin.py build_ext --inplace
