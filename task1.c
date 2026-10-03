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


void gen(double *arr, long int n) {
    for (long int i = 0; i < n; i++) {
        arr[i] = (double)rand() / RAND_MAX * 10.0;
    }
}

void gen_ones(double *arr, int m, int n, double num) {
	for(int i = 0; i< n; i ++) {
		for (int j = 0; j < m; j ++){
			if (i == j) {
				arr[i * m + j] = num;
				}
			else
				arr[i * m + j] = 0;
			}
		}
	}

void print_matrix(double *arr, int n, int m){
	for(int i = 0; i< n; i ++) {
		for (int j = 0; j < m; j ++){
			printf("%f ", arr[i * m + j]);
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
	
	int N  = 8;
    int MB = 2;
    int NB = 2;
    int zero = 0;
	int dims[2] = {0, 0};
	int myrow, mycol;
	int nprow, npcol;
	
	MPI_Dims_create(nprocs, 2, dims);
	nprow = dims[0]; // cartesian direction 0
	npcol = dims[1]; // cartesian direction 1
	
	int context;
    Cblacs_get(-1, 0, &context);
	// Initialize the BLACS context // 
	char order = 'R';
	Cblacs_gridinit( &context, &order, nprow, npcol);
	Cblacs_gridinfo( context, &nprow, &npcol, &myrow, &mycol);
	if (myrank == 0)
	printf("Grid: %d x %d, N=%d, MB=%d, NB=%d\n",
		   nprow, npcol, N, MB, NB);
	// Computation of local matrix size
	int mloc = numroc_(&N, &MB, &myrow, &zero, &nprow);
	int nloc = numroc_(&N, &NB, &mycol, &zero, &npcol);
	
	double *A = (double*) malloc(mloc * nloc * sizeof(double));
    double *B = (double*) malloc(mloc * nloc * sizeof(double));
    double *C = (double*) malloc(mloc * nloc * sizeof(double));
    
    gen_ones(A, nloc, mloc, 1.0);
    gen_ones(B, nloc, mloc, 2.0);
    // если рандомные, то следующие строчки
	// gen(A, (long int)mloc * nloc);
	// gen(B, (long int)mloc * nloc);
	for (int p = 0; p < nprocs; p++) {
    MPI_Barrier(MPI_COMM_WORLD);
    if (myrank == p) {
        printf("=== rank %d (row=%d, col=%d) ===\n", myrank, myrow, mycol);
        printf("A:\n"); print_matrix(A, mloc, nloc);
        printf("B:\n"); print_matrix(B, mloc, nloc);
        printf("C:\n"); print_matrix(C, mloc, nloc);
    }
}
	
    MPI_Barrier(MPI_COMM_WORLD);
	
	char transa = 'N', transb = 'N';
	double alpha = 1.0, beta = 0.0;
	int i = 1;
	
	// Descriptor
	int descA[9], descB[9], descC[9], info;
	descinit_( descA, &N, &N, &MB, &NB, &zero, &zero, &context, &mloc, &info );
	descinit_( descB, &N, &N, &MB, &NB, &zero, &zero, &context, &mloc, &info );
	descinit_( descC, &N, &N, &MB, &NB, &zero, &zero, &context, &mloc, &info );
	
	// Some operations on matrix A
	pdgemm_(&transa, &transb, &N, &N, &N, &alpha, A, &i, &i, descA,
			B, &i, &i, descB, &beta,
			C, &i, &i, descC);
	//
	if (myrank == 0){
		printf("MATRIX 0\n");
		print_matrix(C, nloc, mloc);
	}
	free( A );
	free( B );
	free( C );
	// Close BLACS environment
	Cblacs_gridexit( context );
	Cblacs_exit( 0 );
	// MPI_Finalize();
	return 0;
}

