#!/bin/sh
rm -rf *.h *.i *.cpp *.cu *.cxx dist trottersuzuki.egg*
cp ../*.h .
cp ../*.i .
cp ../*.cpp .
cp ../*.cu .
#swig -c++ -python trotter.i
#python2 setup.py sdist
python setup.py build_ext --inplace
