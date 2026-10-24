// simulation.h
#ifndef SIMULATION_H
#define SIMULATION_H

#include <mpi.h>
#include <stdio.h>
#include <complex.h>

// === Объявления BLACS / ScaLAPACK ===
extern void Cblacs_pinfo(int *mypnum, int *nprocs);
extern void Cblacs_get(int context, int request, int *value);
extern void Cblacs_gridinit(int *context, char *order, int np_row, int np_col);
extern void Cblacs_gridinfo(int context, int *np_row, int *np_col, int *my_row, int *my_col);
extern void Cblacs_gridexit(int context);
extern void Cblacs_exit(int error_code);
extern int  numroc_(int *n, int *nb, int *iproc, int *isrcproc, int *nprocs);
extern void descinit_(int *desca, int *m, int *n, int *mb, int *nb, int *isrc, int *icsrc, int *context, int *llda, int *info);
extern void pzgemm_(char*, char*, int*, int*, int*, double complex*, double complex*, int*, int*, int*, double complex*, int*, int*, int*, double complex*, double complex*, int*, int*, int*);
extern void pzgemr2d_(int*, int*, double complex*, int*, int*, int*, double complex*, int*, int*, int*, int*);
extern void infog2l_(int *grindx, int *gcindx, int *desc, int *nprow, int *npcol, int *myrow, int *mycol, int *lrindx, int *lcindx, int *rsrc, int *csrc);
extern int  indxl2g_(int *indxloc, int *nb, int *iproc, int *isrcproc, int *nprocs);
extern void pzheev_(const char* jobz, const char* uplo, const int* n, double complex* a, const int* ia, const int* ja, const int* desca, double* w, double complex* z, const int* iz, const int* jz, const int* descz, double complex* work, const int* lwork, double* rwork, const int* lrwork, int* info);

// === Функции нашей библиотеки ===

// Чтение вектора или скаляра (mode 0 или 1)
void read_ro(FILE *f, double complex *arr, int N, int type);

// Параллельное чтение матрицы (MPI-IO)
int pz_read_matrix_binary_mpiio(const char *filename, int M, int N, int MB, int NB, double complex *local_a, int lld, int ictxt);

// Вычисление и вывод диагонали
void print_abs_diag(int N, double complex *ro, int lld, int *descRo, int step, double dT, int myrank);

// Основной цикл эволюции матрицы плотности (экспонента + шаги по времени)
void run_time_evolution(int N, int nsteps, double dT, double complex *local_H, 
                        double complex *ro_local, int lld, int *descRo, 
                        int ictxt, int myrank, int mb, int nb);

#endif // SIMULATION_H
