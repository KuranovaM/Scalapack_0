#include <mpi.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <complex.h>
#include <string.h>
#include <errno.h>



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
                      

extern void pzgerc_(int *m, int *n, double complex *alpha, 
                    double complex *x, int *ix, int *jx, int *descx, 
                    double complex *y, int *iy, int *jy, int *descy, 
                    double complex *a, int *ia, int *ja, int *desca);

extern void infog2l_(int *grindx, int *gcindx, int *desc, 
                     int *nprow, int *npcol, int *myrow, int *mycol, 
                     int *lrindx, int *lcindx, int *rsrc, int *csrc);
extern int  indxl2g_(int *indxloc, int *nb, int *iproc, int *isrcproc,
                     int *nprocs);
extern void pzgeadd_(const char* trans, const int* m, const int* n, 
                     const double complex* alpha, const double complex* A, 
                     const int* ia, const int* ja, const int* desca, 
                     const double complex* beta, double complex* C, 
                     const int* ic, const int* jc, const int* descc);
                     
extern void pzheev_(const char* jobz, const char* uplo, const int* n, 
                    double complex* a, const int* ia, const int* ja, const int* desca, 
                    double* w, double complex* z, const int* iz, const int* jz, 
                    const int* descz, double complex* work, const int* lwork, 
                    double* rwork, const int* lrwork, int* info);

extern void pztranc_(const int* m, const int* n, const double complex* alpha, 
                     const double complex* a, const int* ia, const int* ja, const int* desca, 
                     const double complex* beta, double complex* c, const int* ic, const int* jc, 
                     const int* descc);


void print_complex(const double complex *arr, int m, int n) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            double complex v = arr[i + j*m];
            printf("(%f,%f) ", creal(v), cimag(v));
        }
        printf("\n");
    }
}


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
	
void print_abs_diag(int N, double complex *ro, int lld, int *descRo, 
                    int step, double dT, int myrank) 
{
    int nprow, npcol, myrow, mycol;
    int ictxt = descRo[1];
    Cblacs_gridinfo(ictxt, &nprow, &npcol, &myrow, &mycol);

    double *local_diag = (double*)calloc(N, sizeof(double));
    
    for (int i = 0; i < N; ++i) {
        int i_1based = i + 1;
        int lrindx, lcindx, rsrc, csrc;
        
        infog2l_(&i_1based, &i_1based, descRo, &nprow, &npcol, &myrow, &mycol, 
                 &lrindx, &lcindx, &rsrc, &csrc); //глобальный элемент (i, i)
                 
        if (myrow == rsrc && mycol == csrc) {
            int local_idx = (lrindx - 1) + (lcindx - 1) * lld;
            // модуль
            local_diag[i] = cabs(ro[local_idx]);
        }
    }
    
    double *global_diag = NULL;
    if (myrank == 0) {
        global_diag = (double*)malloc(N * sizeof(double));
    }
    
    MPI_Reduce(local_diag, global_diag, N, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    
    if (myrank == 0) {
        printf("Step %d (t = %.4f): ", step, step * dT);
        for (int i = 0; i < N; ++i) {
            printf("%.6f ", global_diag[i]);
        }
        printf("\n");
        fflush(stdout);
        
        free(global_diag);
    }
    
    free(local_diag);
}

void read_ro(FILE *f, double complex *arr, int N, int type){
	double x, y;
    switch (type) {
		/*
		case 2: {
			for (int i = 0; i < N; ++i) {
				for (int j = 0; j < N; ++j) {
					double x, y;
					fscanf(f, "%lf %lf", &x, &y);
					arr[j * N + i] = x + I * y;
				}
			}
			break;
		}*/

		case 1: {
			for (int i = 0; i < N; i++) {
				if (fread(&arr[i], sizeof(double complex), 1, f) != 1) {
					fprintf(stderr, "read_ro: ошибка чтения вектора\n");
					exit(EXIT_FAILURE);
				}
			}
			break;
		}

		case 0: {
			int k;
			// Читаем ровно sizeof(int) байт
			if (fread(&k, sizeof(int), 1, f) != 1) {
				fprintf(stderr, "read_ro: ошибка чтения номера состояния\n");
				exit(EXIT_FAILURE);
			}
			arr[0] = k + 0.0 * I; // сохраняем в arr[0] как комплексное число
			break;
		}

		default: {
			fprintf(stderr, "read_ro: неизвестный тип\n");
			exit(EXIT_FAILURE);
		}
	}
}

	
	
int pz_read_matrix_binary(const char *filename,
                          int M, int N,
                          int MB, int NB,
                          double _Complex *local_a, int lld,
                          int ictxt)
{
    int nprow, npcol, myprow, mypcol;
    Cblacs_gridinfo(ictxt, &nprow, &npcol, &myprow, &mypcol);

    if (myprow < 0 || mypcol < 0) return 0;

    int zero = 0;
    int locr = numroc_(&M, &MB, &myprow, &zero, &nprow);
    int locc = numroc_(&N, &NB, &mypcol, &zero, &npcol);
    if (locr == 0 || locc == 0) return 0;

    FILE *fp = fopen(filename, "rb");
    if (!fp) return -1;

    const size_t esz = sizeof(double _Complex);

    for (int jl = 0; jl < locc; ++jl) {
		int jl1 = jl + 1;
        int jg = indxl2g_(&jl1, &NB, &mypcol, &zero, &npcol) - 1;

        for (int il = 0; il < locr; ++il) {
			int il1 = il + 1;
            int ig = indxl2g_(&il1, &MB, &myprow, &zero, &nprow) - 1;

            long offset = ((long)jg * (long)M + (long)ig) * (long)esz;

            if (fseek(fp, offset, SEEK_SET) != 0) { fclose(fp); return -2; }
            if (fread(&local_a[il + jl * lld], esz, 1, fp) != 1) {
                fclose(fp); return -3;
            }
        }
    }
    fclose(fp);
    return 0;
}



int pz_read_matrix_binary_mpiio(const char *filename, int M, int N, int MB, int NB,
                                 double complex *local_a, int lld, int ictxt)
{
    int nprow, npcol, myprow, mypcol;
    Cblacs_gridinfo(ictxt, &nprow, &npcol, &myprow, &mypcol);

    int in_grid = (myprow >= 0 && mypcol >= 0);
    int zero = 0;
    int locr = in_grid ? numroc_(&M, &MB, &myprow, &zero, &nprow) : 0;
    int locc = in_grid ? numroc_(&N, &NB, &mypcol, &zero, &npcol) : 0;
    int count = locr * locc;

    MPI_Datatype filetype;
    if (count == 0) {
        MPI_Type_contiguous(1, MPI_C_DOUBLE_COMPLEX, &filetype);
    } else {
        int gsizes[2]     = {M, N};
        int distribs[2]   = {MPI_DISTRIBUTE_CYCLIC, MPI_DISTRIBUTE_CYCLIC};
        int dargs[2]      = {MB, NB};
        int psizes[2]     = {nprow, npcol};
        int blacs_rank    = myprow * npcol + mypcol;

        MPI_Type_create_darray(nprow * npcol, blacs_rank, 2,
                               gsizes, distribs, dargs, psizes,
                               MPI_ORDER_FORTRAN, MPI_C_DOUBLE_COMPLEX, &filetype);
    }
    MPI_Type_commit(&filetype);

    MPI_File fh;
    if (MPI_File_open(MPI_COMM_WORLD, filename, MPI_MODE_RDONLY, MPI_INFO_NULL, &fh) != MPI_SUCCESS) {
        MPI_Type_free(&filetype);
        return -1;
    }

    MPI_File_set_view(fh, 0, MPI_C_DOUBLE_COMPLEX, filetype, "native", MPI_INFO_NULL);

    if (count == 0 || lld == locr) {
        MPI_Status status;
        MPI_File_read_at_all(fh, 0, local_a, count, MPI_C_DOUBLE_COMPLEX, &status);
    } else {
        // На случай, если lld > locr (padding)
        double complex *compact = (double complex*)malloc(count * sizeof(double complex));
        MPI_Status status;
        MPI_File_read_at_all(fh, 0, compact, count, MPI_C_DOUBLE_COMPLEX, &status);
        for (int j = 0; j < locc; ++j)
            for (int i = 0; i < locr; ++i)
                local_a[i + j*lld] = compact[i + j*locr];
        free(compact);
    }

    MPI_File_close(&fh);
    MPI_Type_free(&filetype);
    return 0;
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
	
	double time1 = MPI_Wtime();
	double complex * ro_arr = NULL;
	
	if ((myrank == 0) && (mode < 2)) {
		ro_arr = (double complex *) malloc(sizeof(double complex) * N * N);
		FILE *f = fopen(file_ro, "rb");
		if (f == NULL) {
			printf("Не удалось открыть %s: %s\n", file_ro, strerror(errno));
			return -1;
		}
		read_ro(f, ro_arr, N, mode);
		fclose(f);
	}
	
	//double complex * local_ro_arr
	int mb = 2, nb = 2; // поменять потом!
	int i = 1, info;
	
	int descRo[9];
	int lld = numroc_(&N, &mb, &myrow, &zero, &nprow);
	lld = (lld > 1) ? lld : 1;
	descinit_(descRo, &N, &N, &mb, &nb, &zero, &zero, &context, &lld, &info);
	
	int local_rows = numroc_(&N, &mb, &myrow, &zero, &nprow);
	int local_cols = numroc_(&N, &nb, &mycol, &zero, &npcol);
	//double complex *ro_local = (double complex *)malloc(lld * local_cols * sizeof(double complex));
	double complex *ro_local = NULL;

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
		
		int descA[9], descB[9], descC[9], info;
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
				  
		for (int jj = 0; jj < nloc_B; ++jj)
            for (int ii = 0; ii < mloc_B; ++ii)
                ((double complex*)B)[ii + jj * lld_B] =
                    conj(((double complex*)B)[ii + jj * lld_B]);

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

		
		
	}
	
	else if (mode == 2) { 
		//int locc = numroc_(&N, &NB, &mypcol, &zero_, &npcol);
		//double _Complex *b = malloc(sizeof(double _Complex) * lld * locc);
		if (myrank == 0)
			printf("MODE 2");
		ro_local = (double complex *)calloc(lld * local_cols, sizeof(double complex));

		int rc = pz_read_matrix_binary_mpiio(file_ro, N, N, mb, nb, ro_local, lld, context);
		// fprintf(stderr, "[rank %d] rc = %d\n", myrank, rc);
	}
	else {
		ro_local = (double complex *)calloc(lld * local_cols, sizeof(double complex));
		int k = 0;
		if (myrank == 0) {
			k = (int)creal(ro_arr[0]); 
		}
		
		MPI_Bcast(&k, 1, MPI_INT, 0, MPI_COMM_WORLD);

		int i_global = k + 1;
		int j_global = k + 1;
		int lrindx, lcindx; 
		int rsrc, csrc; 

		infog2l_(&i_global, &j_global, descRo, 
		         &nprow, &npcol, &myrow, &mycol, 
		         &lrindx, &lcindx, &rsrc, &csrc);

		if (myrow == rsrc && mycol == csrc) {
			int local_i = lrindx - 1;
			int local_j = lcindx - 1;
			
			int local_index = local_i + local_j * lld;
			ro_local[local_index] = 1.0 + 0.0 * I;
		}
		}
	if (myrank == 0) free(ro_arr);

	if (myrank == 0) {
			printf("=== Local part of ro_local on Rank 0 ===\n");
			if (local_cols > 0 && lld > 0) {
				print_complex(ro_local, lld, local_cols);
			} else {
				printf("Rank 0 has no columns allocated.\n");
			}
			printf("========================================\n");
		}
		
	double complex * local_H = (double complex *)calloc(lld * local_cols, sizeof(double complex));
	pz_read_matrix_binary_mpiio(file_H, N, N, mb, nb, local_H, lld, context);
	
	if (myrank == 0) {
		printf("=== Local part of local_H on Rank 0 ===\n");
		if (local_cols > 0 && lld > 0) {
			print_complex(local_H, lld, local_cols);
		} else {
			printf("Rank 0 has no columns allocated.\n");
		}
		printf("========================================\n");
	}
	
	
    // ==========================================================
    print_abs_diag(N, ro_local, lld, descRo, 0, dT, myrank);

    char jobz = 'V', uplo = 'L';
    int ione = 1, info_heev = 0;
    double *eigvals = (double*)malloc(N * sizeof(double));

    // pzheev: A - вход (разрушается), Q - выход (собственные векторы)
    double complex *A_heev = (double complex*)malloc(lld * local_cols * sizeof(double complex));
    memcpy(A_heev, local_H, lld * local_cols * sizeof(double complex));
    double complex *Q = (double complex*)calloc(lld * local_cols, sizeof(double complex));

    int lwork = -1, lrwork = -1;
    double complex work_query;
    double rwork_query;
    pzheev_(&jobz, &uplo, &N, A_heev, &ione, &ione, descRo, eigvals,
            Q, &ione, &ione, descRo, &work_query, &lwork,
            &rwork_query, &lrwork, &info_heev);
    if (info_heev != 0) { printf("pzheev query failed: %d\n", info_heev); return -1; }

    lwork = (int)creal(work_query);
    lrwork = (int)rwork_query;
    double complex *work = (double complex*)malloc(lwork * sizeof(double complex));
    double *rwork = (double*)malloc(lrwork * sizeof(double));

    pzheev_(&jobz, &uplo, &N, A_heev, &ione, &ione, descRo, eigvals,
            Q, &ione, &ione, descRo, work, &lwork,
            rwork, &lrwork, &info_heev);
    if (info_heev != 0) { printf("pzheev failed: %d\n", info_heev); return -1; }
    free(A_heev); free(work); free(rwork);

    double complex alpha = 1.0 + 0.0*I, beta = 0.0 + 0.0*I;

    // U_step = Q * exp(-i*Lambda*dT)
    double complex *U_step = (double complex*)malloc(lld * local_cols * sizeof(double complex));
    memcpy(U_step, Q, lld * local_cols * sizeof(double complex));
    for (int j = 0; j < local_cols; ++j) {
        int j1 = j + 1;
        int jg = indxl2g_(&j1, &nb, &mycol, &zero, &npcol) - 1;
        double complex mult = cexp(-I * eigvals[jg] * dT);
        for (int ii = 0; ii < lld; ++ii) U_step[ii + j*lld] *= mult;
    }

    // QH = Q^H
    double complex *I_mat = (double complex*)calloc(lld * local_cols, sizeof(double complex));
    for (int j = 0; j < local_cols; ++j) {
        int j1 = j + 1;
        int jg = indxl2g_(&j1, &nb, &mycol, &zero, &npcol) - 1;
        for (int ii = 0; ii < local_rows; ++ii) {
            int i1 = ii + 1;
            int ig = indxl2g_(&i1, &mb, &myrow, &zero, &nprow) - 1;
            if (ig == jg) I_mat[ii + j*lld] = 1.0;
        }
    }
    double complex *QH = (double complex*)calloc(lld * local_cols, sizeof(double complex));
    pzgemm_("N", "C", &N, &N, &N, &alpha, I_mat, &ione, &ione, descRo,
            Q, &ione, &ione, descRo, &beta, QH, &ione, &ione, descRo);
    free(I_mat);

    // U_step = U_step * QH
    double complex *temp = (double complex*)calloc(lld * local_cols, sizeof(double complex));
    pzgemm_("N", "N", &N, &N, &N, &alpha, U_step, &ione, &ione, descRo,
            QH, &ione, &ione, descRo, &beta, temp, &ione, &ione, descRo);
    memcpy(U_step, temp, lld * local_cols * sizeof(double complex));

    // rho(t+dt) = U_step * rho(t) * U_step^H
    for (int step = 0; step < n; ++step) {
        pzgemm_("N", "N", &N, &N, &N, &alpha, U_step, &ione, &ione, descRo,
                ro_local, &ione, &ione, descRo, &beta, temp, &ione, &ione, descRo);
        pzgemm_("N", "C", &N, &N, &N, &alpha, temp, &ione, &ione, descRo,
                U_step, &ione, &ione, descRo, &beta, ro_local, &ione, &ione, descRo);
        print_abs_diag(N, ro_local, lld, descRo, step + 1, dT, myrank);
    }

    free(Q); free(eigvals); free(U_step); free(QH); free(temp);
    // ==========================================================
	
	MPI_Barrier(MPI_COMM_WORLD);
	time1 = MPI_Wtime() - time1;
	if (myrank == 0){
		printf("\nВРЕМЯ ВЫПОЛНЕНИЯ %f\n\n", time1);
		fflush(stdout);
		}
	if (ro_local != NULL) free(ro_local);
	if (local_H != NULL) free(local_H);

	Cblacs_gridexit(context);
	Cblacs_exit(0);
	// MPI_Finalize();
	return 0;
}

