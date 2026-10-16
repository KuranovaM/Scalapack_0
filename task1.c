#include <mpi.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <complex.h>


typedef enum { TYPE_REAL, TYPE_COMPLEX } mattype_t;

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

extern void pdgemm_(char*, char*, int*, int*, int*, double*,
                    double*, int*, int*, int*, double*, int*, int*, int*,
                    double*, double*, int*, int*, int*);
extern void pzgemm_(char*, char*, int*, int*, int*, double complex*,
                    double complex*, int*, int*, int*, double complex*,
                    int*, int*, int*, double complex*,
                    double complex*, int*, int*, int*);
extern void pdgemr2d_(int*, int*, double*, int*, int*, int*,
                      double*, int*, int*, int*, int*);
extern void pzgemr2d_(int*, int*, double complex*, int*, int*, int*,
                      double complex*, int*, int*, int*, int*);



void gen(double *arr, long int n) {
    for (long int i = 0; i < n; i++) {
        arr[i] = (int)(rand() % 10);
    }
}

void gen_complex(double complex *arr, long int n) {
    for (long int i = 0; i < n; i++)
        arr[i] = (rand() % 10) + I * (rand() % 10);
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

void gen_ones_complex(double complex *arr, int m, int n, double complex v) {
    for (int j = 0; j < n; j++)
        for (int i = 0; i < m; i++)
            arr[i + j*m] = (i == j) ? v : 0.0;
}

void print_real(const double *arr, int m, int n) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) printf("%f ", arr[i + j*m]);
        printf("\n");
    }
}
void print_complex(const double complex *arr, int m, int n) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            double complex v = arr[i + j*m];
            printf("(%f,%f) ", creal(v), cimag(v));
        }
        printf("\n");
    }
}


void read_H(FILE *f, double complex *arr, int N){
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            double complex z;
            fread(&z, sizeof(double complex), 1, f);
            arr[j * N + i] = z;
        }
    }
}



void read_ro(FILE *f, double complex *arr, int N, int type){
	double x, y;
    switch (type) {
		case 2: {
			for (int i = 0; i < N; ++i) {
				for (int j = 0; j < N; ++j) {
					double x, y;
					fscanf(f, "%lf %lf", &x, &y);
					arr[j * N + i] = x + I * y;
				}
			}
			break;
		}

		case 1: {
			for (int i = 0; i < N; i ++){
				fscanf(f, "%lf %lf", &x, &y);    
				arr[i] = x + I * y;
				}

			break;
		}

		case 0: {
			int k;
			fscanf(f, "%d", &k); 
			for (int i = 0; i < N * N; i ++) {
				arr[i] = 0;
				}
			arr[k *  N + k] = 1;
			break;
		}

		default: {
			fprintf(stderr, "read_ro: неизвестный тип\n");
			exit(EXIT_FAILURE);
		}
	}
}

//void br_matr(double complex * a, int N_a, int M_a, int N_b, int N_b, int bs){
	
	
//	}

void mult_ab(double complex * A, double complex * B, double complex * C, int M_a, int N_a, int N_b, int * descA, int * descB, int * descC, int info){
	char transa = 'N', transb = 'N';
    double complex alpha = 1.0 + 0.0*I, beta = 0.0 + 0.0*I;
    int i  = 1;
    int M_b = N_a;
    
    MPI_Barrier(MPI_COMM_WORLD);
    pzgemm_(&transa, &transb, &M_a, &N_b, &N_a, &alpha, (double complex*)A, &i, &i, descA,
            (double complex*)B, &i, &i, descB, &beta, (double complex*)C, &i, &i, descC);
    MPI_Barrier(MPI_COMM_WORLD);
	
	}


int main(int argc, char **argv)
{
	MPI_Init(&argc, &argv);
	int myrank, nprocs;
	MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
	MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
	
	int N = atoi(argv[1]);
	char * file_ro = argv[2];
	char *end;
	double dT = strtod(argv[3], &end);
	char * file_H = argv[4];
	int n = atoi(argv[5]);
	
	
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
	
	
	int mode = 0;
	
	if (myrank == 0){
		char ch;
		printf("Choose mode: [s]calar [v]ector [m]atrix: \n");
		fflush(stdout);
		if (scanf(" %c", &ch) != 1) ch = 's';
		if (ch == 'v' || ch == 'V') mode = 1;
		else if (ch == 'm' || ch == 'M') mode = 2;
		else mode = 0;
	}
	MPI_Bcast(&mode, 1, MPI_INT, 0, MPI_COMM_WORLD);
	
	double complex * ro_arr = NULL;
	
	if (myrank == 0) {
		ro_arr = (double complex *) malloc(sizeof(double complex) * N * N);
		FILE *f = fopen(file_ro, "r");
		if (f == NULL) {
			printf("Не удалось открыть %s: %s\n", file_ro, strerror(errno));
			return -1;
		}
		read_ro(f, ro_arr, N, mode);
		fclose(f);
	}
	
	//double complex * local_ro_arr
	int mb = 64, nb = 64; // поменять потом!
	int i = 1, info;
	
	int descRo[9];
	int lld = numroc_(&N, &mb, &myrow, &zero, &nprow);
	lld = (lld > 1) ? lld : 1;
	descinit_(descRo, &N, &N, &mb, &nb, &zero, &zero, &context, &lld, &info);
	
	int local_rows = numroc_(&N, &mb, &myrow, &zero, &nprow);
	int local_cols = numroc_(&N, &nb, &mycol, &zero, &npcol);
	//double complex *ro_local = (double complex *)malloc(lld * local_cols * sizeof(double complex));
	double complex *ro_local = (double complex *)calloc(lld * local_cols, sizeof(double complex));

	if (mode == 1) {
		int descPsi_serial[9];
		int mb_full = N, nb_full = 1;
		int lld_full = (myrank == 0) ? N : 1;
		descinit_(descPsi_serial, &N, &i, &mb_full, &nb_full,
				  &zero, &zero, &context, &lld_full, &info);

		int descPsi_row[9];
		int lld_psi = numroc_(&N, &mb, &myrow, &zero, &nprow);
		lld_psi = (lld_psi > 1) ? lld_psi : 1;
		descinit_(descPsi_row, &N, &i, &mb, &i,
				  &zero, &zero, &context, &lld_psi, &info);
				  
		int descPsi_col[9];
		lld_psi = numroc_(&N, &nb, &mycol, &zero, &npcol);
		lld_psi = (lld_psi > 1) ? lld_psi : 1;
		descinit_(descPsi_col, &N, &i, &nb, &i,
				  &zero, &zero, &context, &lld_psi, &info);

		int nrow_local = numroc_(&N, &mb, &myrow, &zero, &nprow);
		double complex *psi_local_row = (double complex *)calloc(
			(nrow_local > 0 ? nrow_local : 1), sizeof(double complex));
		int ncol_local = numroc_(&N, &nb, &mycol, &zero, &npcol);
		double complex *psi_local_col = (double complex *)calloc(
			(ncol_local > 0 ? ncol_local : 1), sizeof(double complex));

		pzgemr2d_(&N, &i, ro_arr, &i, &i, descPsi_serial,
				  psi_local_row, &i, &i, descPsi_row, &context);
		
		pzgemr2d_(&N, &i, ro_arr, &i, &i, descPsi_serial,
				  psi_local_col, &i, &i, descPsi_col, &context);

		double complex alpha = 1.0 + 0.0 * I;
		pzgerc_(&N, &N, &alpha,
				psi_local_row, &i, &i, descPsi_row,
				psi_local_col, &i, &i, descPsi_col,
				ro_local,  &i, &i, descRo);

		free(psi_local_col);
		free(psi_local_row);
	}
	else {
		int descRo_serial[9];
		int mb_full = N, nb_full = N;
		int lld_full = (myrank == 0) ? N : 1;
		descinit_(descRo_serial, &N, &N, &mb_full, &nb_full,
				  &zero, &zero, &context, &lld_full, &info);

		pzgemr2d_(&N, &N,
				  ro_full,  &i, &i, descRo_serial,
				  ro_local, &i, &i, descRo,
				  &context);
	}
	if (myrank == 0) free(ro_arr);
	
	
	
	
	
	
	
	
	
	
	
	//srand(time(NULL) + myrank);
	c   // 0 = simple, 1 = random, 2 = full
	int pr = atoi(argv[6]);

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


	int M_a  = atoi(argv[1]);
	int N_a = atoi(argv[2]);
	int N_b = atoi(argv[3]);
	int M_b = N_a;

	char t = argv[4][0];
	mattype_t type = TYPE_REAL;
	if (t == 'c' || t == 'C') type = TYPE_COMPLEX;
	else type = TYPE_REAL;
	size_t elsize = (type == TYPE_COMPLEX) ? sizeof(double complex) : sizeof(double);

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

	int maxdim = M_a;
    if (N_a > maxdim) maxdim = N_a;
    if (N_b > maxdim) maxdim = N_b;
    int maxproc = (nprow > npcol) ? nprow : npcol;
    int bs = (maxdim + maxproc - 1) / maxproc;
    if (bs < 1) bs = 1;

    if (myrank == 0){
        printf("Grid %dx%d, A %dx%d, B %dx%d, block %dx%d (square)\n",
               nprow, npcol, M_a, N_a, M_b, N_b, bs, bs);
		fflush(stdout);
	}

    int mloc_A = numroc_(&M_a, &bs, &myrow, &zero, &nprow);
    int nloc_A = numroc_(&N_a, &bs, &mycol, &zero, &npcol);
    int lld_A  = (mloc_A > 0) ? mloc_A : 1;

    int mloc_B = numroc_(&M_b, &bs, &myrow, &zero, &nprow);
    int nloc_B = numroc_(&N_b, &bs, &mycol, &zero, &npcol);
    int lld_B  = (mloc_B > 0) ? mloc_B : 1;

    int mloc_C = numroc_(&M_a, &bs, &myrow, &zero, &nprow);
    int nloc_C = numroc_(&N_b, &bs, &mycol, &zero, &npcol);
    int lld_C  = (mloc_C > 0) ? mloc_C : 1;

    int descA[9], descB[9], descC[9], info;
    descinit_(descA, &M_a, &N_a, &bs, &bs, &zero, &zero, &context, &lld_A, &info);
    descinit_(descB, &M_b, &N_b, &bs, &bs, &zero, &zero, &context, &lld_B, &info);
    descinit_(descC, &M_a, &N_b, &bs, &bs, &zero, &zero, &context, &lld_C, &info);

    void *A = malloc(mloc_A * nloc_A * elsize);// sizeof(double));
    void *B = malloc(mloc_B * nloc_B * elsize);// sizeof(double));
    void *C = malloc(mloc_C * nloc_C * elsize);// sizeof(double));
	//NB = min(NB, MB);
	//MB = min(NB, MB);
	if (myrank == 0){
		//printf("Grid: %d x %d, M=%d, N=%d, MB=%d, NB=%d\n",
		//	   nprow, npcol,M, N, MB, NB);
		fflush(stdout);
	}

    void *a_full = NULL;
	void *b_full = NULL;
	void *c_full = NULL;
	int curr0;
	Cblacs_get(-1, 0, &curr0);
	Cblacs_gridinit(&curr0, &order, 1, 1);

	int descA0[9] = {0};
	int descB0[9] = {0};
	int descC0[9] = {0};
	int info0;
	// int Nb = N;
	// int Mb = M;

	if (curr0 >= 0) {
		descinit_(descA0, &M_a, &N_a, &M_a, &N_a, &zero, &zero, &curr0, &M_a, &info0);
        descinit_(descB0, &M_b, &N_b, &M_b, &N_b, &zero, &zero, &curr0, &M_b, &info0);
        descinit_(descC0, &M_a, &N_b, &M_a, &N_b, &zero, &zero, &curr0, &M_a, &info0);

        a_full = malloc(M_a * N_a * elsize);//  sizeof(double));
        b_full = malloc(M_b * N_b * elsize);//  sizeof(double));
        c_full = malloc(M_a * N_b * elsize);
        // c_full = malloc(M_a * N_b * sizeof(double));

		if (mode == 0) {
            if (type == TYPE_REAL) {
                gen_ones((double*)a_full, M_a, N_a, 1.0);
                gen_ones((double*)b_full, M_b, N_b, 2.0);
            }
            else {
                gen_ones_complex((double complex*)a_full, M_a, N_a, 1.0 + 0.0*I);
                gen_ones_complex((double complex*)b_full, M_b, N_b, 2.0 + 0.0*I);
            }
        }
        else if (mode == 1) {
            if (type == TYPE_REAL) {
                gen((double*)a_full, (long int)M_a * N_a);
                gen((double*)b_full, (long int)M_b * N_b);
            }
            else {
                gen_complex((double complex*)a_full, (long int)M_a * N_a);
                gen_complex((double complex*)b_full, (long int)M_b * N_b);
            }
        }
        else {
            FILE *f = fopen(argv[5], "r");
            //for (int k = 0; k < M_a*N_a; k++) fscanf(f, "%lf", &a_full[k]);
            //for (int k = 0; k < M_b*N_b; k++) fscanf(f, "%lf", &b_full[k]);
            if (type == TYPE_REAL) {
                read_real(f, (double*)a_full,(double*)b_full, (long int)M_a, (long int)N_a, (long int)N_b);
            }
            else {
                read_complex(f, (double complex*)a_full, (double complex*)b_full, (long int)M_a, (long int)N_a, (long int)N_b);
            }
            fclose(f);
        }
	} else {
		descA0[1] = -1;
		descB0[1] = -1;
		descC0[1] = -1;
	}

	int i = 1;
	if (type == TYPE_REAL) {
		pdgemr2d_(&M_a, &N_a, a_full, &i, &i, descA0, A, &i, &i, descA, &context);
		pdgemr2d_(&M_b, &N_b, b_full, &i, &i, descB0, B, &i, &i, descB, &context);
	}
	else {
        pzgemr2d_(&M_a, &N_a, (double complex*)a_full, &i, &i, descA0,
                  (double complex*)A, &i, &i, descA, &context);
        pzgemr2d_(&M_b, &N_b, (double complex*)b_full, &i, &i, descB0,
                  (double complex*)B, &i, &i, descB, &context);
    }

	if (a_full) free(a_full);
	if (b_full) free(b_full);
	// if (curr0 >= 0) Cblacs_gridexit(curr0);

	if (pr)
		for (int p = 0; p < nprocs; p++) {
			MPI_Barrier(MPI_COMM_WORLD);
			if (myrank == p) {
				printf("=== rank %d (row=%d, col=%d) ===\n", myrank, myrow, mycol);
				//printf("A:\n"); print_matrix(A, mloc_A, nloc_A);
				//printf("B:\n"); print_matrix(B, mloc_B, nloc_B);
				// printf("C:\n"); print_matrix(C, mloc, nloc);
			}
		}

    MPI_Barrier(MPI_COMM_WORLD);

	char transa = 'N', transb = 'N';
	double alpha = 1.0, beta = 0.0;

	double tm = MPI_Wtime();
	if (type == TYPE_REAL) {
		double alpha = 1.0, beta = 0.0;
        pdgemm_(&transa, &transb, &M_a, &N_b, &N_a, &alpha, (double*)A, &i, &i, descA,
                (double*)B, &i, &i, descB, &beta, (double*)C, &i, &i, descC);

	}
	else {
        double complex alpha = 1.0 + 0.0*I, beta = 0.0 + 0.0*I;
        pzgemm_(&transa, &transb, &M_a, &N_b, &N_a, &alpha, (double complex*)A, &i, &i, descA,
                (double complex*)B, &i, &i, descB, &beta, (double complex*)C, &i, &i, descC);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    tm = MPI_Wtime() - tm;
    if (myrank == 0) printf("time = %lf\n", tm);

	// double *c_full = NULL;

	if (curr0 < 0) {
		descC0[1] = -1;
	}
    MPI_Barrier(MPI_COMM_WORLD);
	if (type == TYPE_REAL) {
        pdgemr2d_(&M_a, &N_b, (double*)C, &i, &i, descC,
                  (double*)c_full, &i, &i, descC0, &context);
    } else {
        pzgemr2d_(&M_a, &N_b, (double complex*)C, &i, &i, descC,
                  (double complex*)c_full, &i, &i, descC0, &context);
    }

	//pdgemr2d_(&M_a, &N_b, C, &i, &i, descC, c_full, &i, &i, descC0, &context);

    MPI_Barrier(MPI_COMM_WORLD);

	fflush(stdout);
    if (myrank == 0 && pr) {
        printf("=== Full C ===\n");
        if (type == TYPE_REAL)
			print_real((double*)c_full, M_a, N_b);
        else
			print_complex((double complex*)c_full, M_a, N_b);
        free(c_full);
    }

	fflush(stdout);

	free( A );
	free( B );
	free( C );

	if (curr0 >= 0) Cblacs_gridexit(curr0);
	Cblacs_gridexit(context);
	Cblacs_exit(0);
	// MPI_Finalize();
	return 0;
}

