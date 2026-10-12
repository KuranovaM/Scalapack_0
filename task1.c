#include <mpi.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>


extern void Cblacs_pinfo(int *mypnum, int *nprocs);
extern void Cblacs_get(int context, int request, int *value);
extern void Cblacs_gridinit(int *context, char *order, int np_row, int np_col);
extern void Cblacs_gridinfo(int context, int *np_row, int *np_col,
                            int *my_row, int *my_col);
extern void Cblacs_gridexit(int context);
extern void Cblacs_exit(int error_code);
extern int  numroc_(int *n, int *nb, int *iproc, int *isrcproc, int *nprocs);
extern void descinit_(int *desca, int *m, int *n, int *mb, int *nb,
                      int *isrc, int *icsrc, int *context,
                      int *llda, int *info);
extern void pdgemm_(char *transa, char *transb,
                    int *m, int *n, int *k,
                    double *alpha,
                    double *a, int *ia, int *ja, int *desca,
                    double *b, int *ib, int *jb, int *descb,
                    double *beta,
                    double *c, int *ic, int *jc, int *descc);
                    
extern void pdgemr2d_(int *m, int *n,
                      double *a, int *ia, int *ja, int *desca,
                      double *b, int *ib, int *jb, int *descb,
                      int *ictxt);


void gen(double *arr, long int n) {
    for (long int i = 0; i < n; i++) {
        arr[i] = (double)rand() / RAND_MAX * 10.0;
    }
}

void gen_ones(double *arr, int mloc, int nloc, double num) {
    for (int j = 0; j < nloc; j++) {
        for (int i = 0; i < mloc; i++) {
            if (i == j) {
                arr[i + j * mloc] = num;
            } else {
                arr[i + j * mloc] = 0.0;
            }
        }
    }
}


void print_matrix(double *arr, int mloc, int nloc){
    for (int i = 0; i < mloc; i++) {
        for (int j = 0; j < nloc; j++) {
            printf("%f ", arr[i + j*mloc]);
        }
        printf("\n");
    }
}





int main(int argc, char **argv)
{
	MPI_Init(&argc, &argv);
	int myrank, nprocs;
	MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
	MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
	srand(time(NULL) + myrank);
	int mode = 0;   // 0 = simple, 1 = random, 2 = full 
	
	if (myrank == 0){
		char ch;
		printf("Choose mode: [r]andom / [s]imple / [f]ile: \n");
		fflush(stdout);
		if (scanf(" %c", &ch) != 1) ch = 's';
		if (ch == 'r' || ch == 'R') mode = 1;
		else if (ch == 'f' || ch == 'F') mode = 2;
		else mode = 0;
	}
	MPI_Bcast(&mode, 1, MPI_INT, 0, MPI_COMM_WORLD);
	
	
	int N  = atoi(argv[1]);
    // int MB = 2;
    // int NB = 2;
    int zero = 0;
	int dims[2] = {0, 0};
	int myrow, mycol;
	int nprow, npcol;
	
	MPI_Dims_create(nprocs, 2, dims);
	nprow = dims[0];
	npcol = dims[1];
	
	int context;
    Cblacs_get(-1, 0, &context);

	char order = 'R';
	Cblacs_gridinit( &context, &order, nprow, npcol);
	Cblacs_gridinfo( context, &nprow, &npcol, &myrow, &mycol);
	
	int MB = (N + nprow - 1) / nprow;
	int NB = (N + npcol - 1) / npcol;
	if (MB < 1) MB = 1;
	if (NB < 1) NB = 1;
	if (myrank == 0){
		printf("Grid: %d x %d, N=%d, MB=%d, NB=%d\n",
			   nprow, npcol, N, MB, NB);
		fflush(stdout);
	}
		   
		   
	int mloc = numroc_(&N, &MB, &myrow, &zero, &nprow);
	int nloc = numroc_(&N, &NB, &mycol, &zero, &npcol);

	
	
	printf("$");
		fflush(stdout);

	int descA[9], descB[9], descC[9], info;
	int lld = (mloc > 0) ? mloc : 1;
	descinit_(descA, &N, &N, &MB, &NB, &zero, &zero, &context, &lld, &info );
	descinit_(descB, &N, &N, &MB, &NB, &zero, &zero, &context, &lld, &info );
	descinit_(descC, &N, &N, &MB, &NB, &zero, &zero, &context, &lld, &info );
	
	
	printf(")");
		fflush(stdout);

	
	double *A = (double*) malloc(mloc * nloc * sizeof(double));
    double *B = (double*) malloc(mloc * nloc * sizeof(double));
    double *C = (double*) malloc(mloc * nloc * sizeof(double));
    
    double *a_full = NULL;
	double *b_full = NULL;
	int curr0;
	Cblacs_get(-1, 0, &curr0);
	Cblacs_gridinit(&curr0, &order, 1, 1);
	
	//printf("*");
		//fflush(stdout);
	int descA0[9] = {0};
	int descB0[9] = {0};
	int descC0[9] = {0};
	int info0;
	int Nb = N;

	if (curr0 >= 0) {
		descinit_(descA0, &N, &N, &Nb, &Nb, &zero, &zero, &curr0, &N, &info0);
		descinit_(descB0, &N, &N, &Nb, &Nb, &zero, &zero, &curr0, &N, &info0);
		descinit_(descC0, &N, &N, &Nb, &Nb, &zero, &zero, &curr0, &N, &info0);
		
		a_full = (double*)malloc(N * N * sizeof(double));
		b_full = (double*)malloc(N * N * sizeof(double));
    
		if (mode == 0) {
			gen_ones(a_full, N, N, 1.0);
			gen_ones(b_full, N, N, 2.0);
		}
		else if (mode == 1){
			gen(a_full, (long int)N * N);
			gen(b_full, (long int)N * N);
		}
		else { 
			FILE *f = fopen(argv[2], "r");
			// fscanf(f, "%d", &N);
			for (int i = 0; i < N*N; i++) fscanf(f, "%lf", &a_full[i]);
			for (int i = 0; i < N*N; i++) fscanf(f, "%lf", &b_full[i]);
			fclose(f);
		}
	} else {
		descA0[1] = -1;  
		descB0[1] = -1;
		descC0[1] = -1;
	}

	int i = 1;
	//printf("!");
		//fflush(stdout);
	pdgemr2d_(&N, &N, a_full, &i, &i, descA0, A, &i, &i, descA, &context);
	pdgemr2d_(&N, &N, b_full, &i, &i, descB0, B, &i, &i, descB, &context);
	
	
	//printf("1!");
		//fflush(stdout);
	if (a_full) free(a_full);
	if (b_full) free(b_full);
	// if (curr0 >= 0) Cblacs_gridexit(curr0);  
	
	
	//printf("2!");
		//fflush(stdout);
	for (int p = 0; p < nprocs; p++) {
		MPI_Barrier(MPI_COMM_WORLD);
		if (myrank == p) {
			printf("=== rank %d (row=%d, col=%d) ===\n", myrank, myrow, mycol);
			printf("A:\n"); print_matrix(A, mloc, nloc);
			printf("B:\n"); print_matrix(B, mloc, nloc);
			// printf("C:\n"); print_matrix(C, mloc, nloc);
		}
	}
	
    MPI_Barrier(MPI_COMM_WORLD);
	
	printf("3!");
		fflush(stdout);
	char transa = 'N', transb = 'N';
	double alpha = 1.0, beta = 0.0;
	
	
	pdgemm_(&transa, &transb, &N, &N, &N, &alpha, A, &i, &i, descA,
			B, &i, &i, descB, &beta,
			C, &i, &i, descC);
	
    MPI_Barrier(MPI_COMM_WORLD);
	
	printf("4!");
		fflush(stdout);
	//int descC0[9] = {0};
	double *c_full = NULL;

	if (curr0 >= 0) {
		//descinit_(descC0, &N, &N, &Nb, &Nb, &zero, &zero, &curr0, &N, &info0);
		c_full = (double*) malloc(N * N * sizeof(double));
	} else {
		descC0[1] = -1; 
	}
	
	printf("5!");
		fflush(stdout);
	
    MPI_Barrier(MPI_COMM_WORLD);
	pdgemr2d_(&N, &N, C, &i, &i, descC, c_full, &i, &i, descC0, &context);
	
    MPI_Barrier(MPI_COMM_WORLD);
	printf("6!");
		fflush(stdout);
	if (curr0 >= 0) {
		printf("=== Full C ===\n");
		print_matrix(c_full, N, N);
		free(c_full);
	}

		fflush(stdout);
	printf("7!");
		fflush(stdout);
	
	free( A );
	free( B );
	free( C );
	
	printf("8!");
		fflush(stdout);
	
	if (curr0 >= 0) Cblacs_gridexit(curr0);
	Cblacs_gridexit(context);
	Cblacs_exit(0);
	// MPI_Finalize();
	return 0;
}

