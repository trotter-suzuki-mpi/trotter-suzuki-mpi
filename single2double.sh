#!/bin/bash
cd src
for file in `ls *.cpp *.h *.cu`; do
  cat $file|sed 's/float/double/g'|sed 's/ssyev/dsyev/g'|sed 's/slamch/dlamch/g'|\
            sed 's/sgemm/dgemm/g'|sed 's/ssymm/dsymm/g'|\
            sed 's/Sgemm/Dgemm/g'|sed 's/Ssymm/Dsymm/g'|\
            sed 's/_ps/_pd/g' | sed 's/__m128/__m128d/g'|\
            sed 's/__f/__d/g' |\
            sed 's/MPI_FLOAT/MPI_DOUBLE/g' > $file.new
  mv $file.new $file
done
cat cpublocksse.cpp|sed 's/% 4; j += 4, idx1 += 4, idx2 += 4/% 2; j += 2, idx1 += 2, idx2 += 2/' > cpublocksse.cpp.new
mv cpublocksse.cpp.new cpublocksse.cpp
cd ..
