#include <stdio.h>
#include <stdlib.h>
#include <complex.h>
#include <mpi.h>


int print_err(const char *str, int rank) {
    if (rank == 0) {
        printf("ERROR: %s\n\n", str);
        fflush(stdout);
    }
    MPI_Abort(MPI_COMM_WORLD, 1); 
    return 1;
}

int main(int argc, char **argv)
{
	
	
    MPI_Init(&argc, &argv);
    int myrank, nprocs;
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    
    int dims[2] = {0, 0};
    int myrow, mycol, nprow, npcol;
    MPI_Dims_create(nprocs, 2, dims);
    nprow = dims[0]; npcol = dims[1];
    
    
	int N;
    double complex a, b;
    double wa, wc, Emin, Emax;
    
    
    double a_real, a_imag, b_real, b_imag;
    
    if (rank == 0) {
        printf("=== Ввод параметров модели ===\n");
        
        printf("Введите N: ");
        if (scanf("%d", &N) != 1) err_flag = 1;

        printf("Введите a (формат real img): ");
        if (scanf("%lf %lf", &a_real, &a_imag) != 2) err_flag = 1;
        
        printf("Введите b (формат real img): ");
        if (scanf("%lf %lf", &b_real, &b_imag) != 2) err_flag = 1;

        printf("Введите wa: ");
        if (scanf("%lf", &wa) != 1) err_flag = 1;

        printf("Введите wc: ");
        if (scanf("%lf", &wc) != 1) err_flag = 1;

        printf("Введите Emin и Emax: ");
        if (scanf("%lf %lf", &Emin, &Emax) != 2) err_flag = 1;
    }

    if (err_flag != 0) {
        if (rank == 0) {
            if (err_flag == 1) {
                print_err("Некорректный ввод данных (ожидались числа).", rank);
            } 
            else if (err_flag == 2) {
                print_err("N должно быть положительным и четным", rank);
            } 
            else if (err_flag == 3) {
                print_err("Emin должно быть меньше Emax.", rank);
            } 
            else if (err_flag == 4) {
                print_err("Частоты wa и wc не могут быть отрицательными.", rank);
            }
        } 
        else {
            print_err("", rank);
        }
    }
	MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&a, 1, MPI_C_DOUBLE_COMPLEX, 0, MPI_COMM_WORLD); 
	MPI_Bcast(&b, 1, MPI_C_DOUBLE_COMPLEX, 0, MPI_COMM_WORLD);
	MPI_Bcast(&wa, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(&wc, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(&Emin, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(&Emax, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
	if (rank == 0) {
        printf("N = %d, wa = %g, wc = %g, Emin = %g, Emax = %g\n", N, wa, wc, Emin, Emax);
        fflush(stdout);
    }
	
	
	MPI_Finalize();
    return 0;
}

