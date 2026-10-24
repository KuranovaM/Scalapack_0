// main.c
#include "simulation.h"
#include <stdlib.h>
#include <string.h>
#include <errno.h>

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    int myrank, nprocs;
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

    // 1. Проверка аргументов
    if (argc < 6) {
        if (myrank == 0) printf("Usage: %s N file_ro dT file_H n\n", argv[0]);
        MPI_Finalize();
        return 1;
    }

    int N = atoi(argv[1]);
    char *file_ro = argv[2];
    double dT = strtod(argv[3], NULL);
    char *file_H = argv[4];
    int n = atoi(argv[5]);

    int zero = 0;
    int dims[2] = {0, 0};
    int myrow, mycol, nprow, npcol;
    MPI_Dims_create(nprocs, 2, dims);
    nprow = dims[0]; npcol = dims[1];
    
    int context;
    Cblacs_get(-1, 0, &context);
    char order = 'R';
    Cblacs_gridinit(&context, &order, nprow, npcol);
    Cblacs_gridinfo(context, &nprow, &npcol, &myrow, &mycol);

    // Выбор режима
    int mode = 0;
    if (myrank == 0) {
        char ch;
        printf("Choose mode: [s]calar [v]ector [m]atrix: \n");
        fflush(stdout);
        if (scanf(" %c", &ch) != 1) ch = 's';
        if (ch == 'v' || ch == 'V') mode = 1;
        else if (ch == 'm' || ch == 'M') mode = 2;
        else mode = 0;
    }
    MPI_Bcast(&mode, 1, MPI_INT, 0, MPI_COMM_WORLD);

    double time1 = MPI_Wtime();
    
    // Чтение ro
    double complex *ro_arr = NULL;
    if ((myrank == 0) && (mode < 2)) {
        ro_arr = (double complex *)malloc(sizeof(double complex) * N * N);
        FILE *f = fopen(file_ro, "rb");
        if (f == NULL) {
            printf("Не удалось открыть %s: %s\n", file_ro, strerror(errno));
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        read_ro(f, ro_arr, N, mode);
        fclose(f);
    }

    int mb = 2, nb = 2;
    int i = 1;
    int descRo[9];
    int lld = numroc_(&N, &mb, &myrow, &zero, &nprow);
    lld = (lld > 1) ? lld : 1;
    descinit_(descRo, &N, &N, &mb, &nb, &zero, &zero, &context, &lld, &i);

    int local_cols = numroc_(&N, &nb, &mycol, &zero, &npcol);
    double complex *ro_local = (double complex *)calloc(lld * local_cols, sizeof(double complex));

    // Инициализация ro
    if (mode == 1) {
		ro_local = (double complex *)calloc(lld * local_cols, sizeof(double complex));
		int M_a  = N;
		int N_a = 1;
		int N_b = N;
		int M_b = 1;
		size_t elsize = sizeof(double complex);
		int bs = mb;
		
		
		
		int mloc_A = numroc_(&M_a, &bs, &myrow, &zero, &nprow);
		int nloc_A = numroc_(&N_a, &bs, &mycol, &zero, &npcol);
		int lld_A  = (mloc_A > 0) ? mloc_A : 1;

		int mloc_B = numroc_(&M_b, &bs, &myrow, &zero, &nprow);
		int nloc_B = numroc_(&N_b, &bs, &mycol, &zero, &npcol);
		int lld_B  = (mloc_B > 0) ? mloc_B : 1;

		void *A = malloc(mloc_A * nloc_A * elsize);// sizeof(double));
		void *B = malloc(mloc_B * nloc_B * elsize);// sizeof(double));
		
		int descA[9], descB[9], info;
		descinit_(descA, &M_a, &N_a, &bs, &bs, &zero, &zero, &context, &lld_A, &info);
		descinit_(descB, &M_b, &N_b, &bs, &bs, &zero, &zero, &context, &lld_B, &info);
				
		int curr0;
		Cblacs_get(-1, 0, &curr0);
		Cblacs_gridinit(&curr0, &order, 1, 1);
		int descA0[9] = {0};
		int descB0[9] = {0};
		int descC0[9] = {0};
		int info0;
		if (myrank == 0){
			printf("Grid %dx%d, A %dx%d, B %dx%d, block %dx%d (square)\n",
				   nprow, npcol, M_a, N_a, M_b, N_b, bs, bs);
			fflush(stdout);
		}

		if (curr0 >= 0) {
			descinit_(descA0, &M_a, &N_a, &M_a, &N_a, &zero, &zero, &curr0, &M_a, &info0);
			descinit_(descB0, &M_b, &N_b, &M_b, &N_b, &zero, &zero, &curr0, &M_b, &info0);
		} else {
			descA0[1] = -1;  
			descB0[1] = -1;
			descC0[1] = -1;
		}

		int i = 1;
		pzgemr2d_(&M_a, &N_a, (double complex*)ro_arr, &i, &i, descA0,
				  (double complex*)A, &i, &i, descA, &context);
		pzgemr2d_(&M_b, &N_b, (double complex*)ro_arr, &i, &i, descB0,
				  (double complex*)B, &i, &i, descB, &context);
				  
		for (int jj = 0; jj < nloc_B; ++jj) {
            for (int ii = 0; ii < mloc_B; ++ii) {
                ((double complex*)B)[ii + jj * lld_B] =
                    conj(((double complex*)B)[ii + jj * lld_B]);
            }
        }

		//if (myrank == 0) {
		//	printf("НОРМ\n\n");
		//	print_complex(ro_arr, 1, N);
		//	}
		
		if (curr0 >= 0) Cblacs_gridexit(curr0);  

		MPI_Barrier(MPI_COMM_WORLD);

		char transa = 'N', transb = 'N';

		double tm = MPI_Wtime();
		double complex alpha = 1.0 + 0.0*I, beta = 0.0 + 0.0*I;
		pzgemm_(&transa, &transb, &M_a, &N_b, &N_a, &alpha, (double complex*)A, &i, &i, descA,
				(double complex*)B, &i, &i, descB, &beta, (double complex*)ro_local, &i, &i, descRo);

		MPI_Barrier(MPI_COMM_WORLD);
		tm = MPI_Wtime() - tm;

		
		free(A); free(B);
    } else if (mode == 2) {
        int rc = pz_read_matrix_binary_mpiio(file_ro, N, N, mb, nb, ro_local, lld, context);
        if (rc != 0) {
            if (myrank == 0) fprintf(stderr, "Error reading ro matrix, code %d\n", rc);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    } else {
        int k = 0;
        if (myrank == 0) k = (int)creal(ro_arr[0]);
        MPI_Bcast(&k, 1, MPI_INT, 0, MPI_COMM_WORLD);

        if (k < 0 || k >= N) {
            if (myrank == 0) fprintf(stderr, "Error: basis index %d out of bounds\n", k);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        int i_global = k + 1, j_global = k + 1;
        int lrindx, lcindx, rsrc, csrc;
        infog2l_(&i_global, &j_global, descRo, &nprow, &npcol, &myrow, &mycol, &lrindx, &lcindx, &rsrc, &csrc);
        if (myrow == rsrc && mycol == csrc) {
            ro_local[(lrindx - 1) + (lcindx - 1) * lld] = 1.0 + 0.0 * I;
        }
    }
    if (myrank == 0) free(ro_arr);

    // Чтение H
    double complex *local_H = (double complex *)calloc(lld * local_cols, sizeof(double complex));
    int rc_H = pz_read_matrix_binary_mpiio(file_H, N, N, mb, nb, local_H, lld, context);
    if (rc_H != 0) {
        if (myrank == 0) fprintf(stderr, "Error reading H matrix, code %d\n", rc_H);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // ЗАПУСК БИБЛИОТЕЧНОЙ ФУНКЦИИ
    run_time_evolution(N, n, dT, local_H, ro_local, lld, descRo, context, myrank, mb, nb);

    MPI_Barrier(MPI_COMM_WORLD);
    time1 = MPI_Wtime() - time1;
    if (myrank == 0) printf("\nВРЕМЯ ВЫПОЛНЕНИЯ %f\n\n", time1);

    free(ro_local);
    free(local_H);
    Cblacs_gridexit(context);
    Cblacs_exit(0);
    return 0;
}
