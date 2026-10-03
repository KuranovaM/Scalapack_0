#include <iostream>

#include <stdlib.h>

void gen(double *arr, long int n) {
    for (long int i = 0; i < n; i++) {
        arr[i] = (double)rand() / RAND_MAX * 10.0;
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
	int dims[2];
	int myrow, mycol;
	
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
    
    gen(A, (long int)mloc * nloc);
	gen(B, (long int)mloc * nloc);
	
	// Descriptor
	int descA[9], descB[9], descC[9], info;
	descinit_( descA, &m, &n, &mb, &nb, &zero, &zero, &context, &mloc, &info );
	
	// Some operations on matrix A
	//
	free( A );
	free( B );
	free( C );
	// Close BLACS environment
	Cblacs_gridexit( context );
	Cblacs_exit( 0 );
	// MPI_Finalize();
	return 0;
}

