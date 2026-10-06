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

void gen_ones(double *arr, int m, int n, double num) {
	for(int i = 0; i< n; i ++) {
		for (int j = 0; j < m; j ++){
			if (i == j) {
				arr[i * n + j] = num;
				}
			else
				arr[i * n + j] = 0;
			}
		}
	}

void print_matrix(double *arr, int mloc, int nloc){
    for (int li = 0; li < mloc; li++) {
        for (int lj = 0; lj < nloc; lj++) {
            printf("%f ", arr[li + lj*mloc]);
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
		printf("Choose mode: [r]andom / [s]imple / [f]ull: \n");
		fflush(stdout);
		if (scanf(" %c", &ch) != 1) ch = 's';
		if (ch == 'r' || ch == 'R') mode = 1;
		else if (ch == 'f' || ch == 'F') mode = 2;
		else mode = 0;
	}
	MPI_Bcast(&mode, 1, MPI_INT, 0, MPI_COMM_WORLD);
	
	
	int N  = 8;
    int MB = 2;
    int NB = 2;
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
	if (myrank == 0){
		printf("Grid: %d x %d, N=%d, MB=%d, NB=%d\n",
			   nprow, npcol, N, MB, NB);
		fflush(stdout);
	}
		   
		   
	int mloc = numroc_(&N, &MB, &myrow, &zero, &nprow);
	int nloc = numroc_(&N, &NB, &mycol, &zero, &npcol);

	int descA[9], descB[9], descC[9], info;
	descinit_( descA, &N, &N, &MB, &NB, &zero, &zero, &context, &mloc, &info );
	descinit_( descB, &N, &N, &MB, &NB, &zero, &zero, &context, &mloc, &info );
	descinit_( descC, &N, &N, &MB, &NB, &zero, &zero, &context, &mloc, &info );
	
	
	
	
	double *A = (double*) malloc(mloc * nloc * sizeof(double));
    double *B = (double*) malloc(mloc * nloc * sizeof(double));
    double *C = (double*) malloc(mloc * nloc * sizeof(double));
    
    
    if (mode == 0) {
		gen_ones(A, nloc, mloc, 1.0);
		gen_ones(B, nloc, mloc, 2.0);
    }
    else if (mode == 1){
		gen(A, (long int)mloc * nloc);
		gen(B, (long int)mloc * nloc);
	}
	else { 
		double *a_full = NULL;
		double *b_full = NULL;

		int curr0;
		Cblacs_get(-1, 0, &curr0);
		Cblacs_gridinit(&curr0, &order, 1, 1);   
		
		int descA0[9] = {0};
		int descB0[9] = {0};
		int info0;
		int Nb = N;

		if (curr0 >= 0) {
			descinit_(descA0, &N, &N, &Nb, &Nb, &zero, &zero, &curr0, &N, &info0);
			descinit_(descB0, &N, &N, &Nb, &Nb, &zero, &zero, &curr0, &N, &info0);
			a_full = (double*)malloc((size_t)N * N * sizeof(double));
			b_full = (double*)malloc((size_t)N * N * sizeof(double));
			gen_ones(a_full, N, N, 1.0);
			gen_ones(b_full, N, N, 2.0);
		} else {
			descA0[1] = -1;  
			descB0[1] = -1;
		}

		int ione = 1;
		pdgemr2d_(&N, &N, a_full, &ione, &ione, descA0,
						  A,      &ione, &ione, descA, &context);
		pdgemr2d_(&N, &N, b_full, &ione, &ione, descB0,
						  B,      &ione, &ione, descB, &context);

		if (a_full) free(a_full);
		if (b_full) free(b_full);
		if (curr0 >= 0) Cblacs_gridexit(curr0);  
	}
	
	
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
	
	char transa = 'N', transb = 'N';
	double alpha = 1.0, beta = 0.0;
	int i = 1;
	
	
	pdgemm_(&transa, &transb, &N, &N, &N, &alpha, A, &i, &i, descA,
			B, &i, &i, descB, &beta,
			C, &i, &i, descC);
	//
	if (myrank == 0){
		printf("\n\n\nMATRIX C\n");
		fflush(stdout);
		//print_matrix(C, nloc, mloc);
	}
	for (int p = 0; p < nprocs; p++) {
		MPI_Barrier(MPI_COMM_WORLD);
		if (myrank == p) {
			printf("--- rank %d (row=%d, col=%d) ---\n", myrank, myrow, mycol);
			printf("C:\n"); print_matrix(C, mloc, nloc);
			fflush(stdout);
		}
	}
	
	free( A );
	free( B );
	free( C );
	
	Cblacs_gridexit( context );
	Cblacs_exit( 0 );
	// MPI_Finalize();
	return 0;
}

