### scalapack

## task 0

Перемножение вроде работает



запускала так:

mpicc -o program.x task1.c -ldl -lm -L$HOME/scalapack-install/lib -lscalapack -llapack -lblas -lgfortran 


mpirun -np 4 ./program.x

mpirun -np 4 ./program.x 2 5 2 c ./test.txt 1 
