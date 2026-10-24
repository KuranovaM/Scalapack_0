// simulation.c
#include "simulation.h"
#include <stdlib.h>
#include <math.h>
#include <string.h>

// --- Чтение ro ---
void read_ro(FILE *f, double complex *arr, int N, int type){
    switch (type) {
        case 1: {
            if (fread(arr, sizeof(double complex), N, f) != (size_t)N) {
                fprintf(stderr, "read_ro: ошибка чтения вектора\n");
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
            break;
        }
        case 0: {
            int k;
            if (fread(&k, sizeof(int), 1, f) != 1) {
                fprintf(stderr, "read_ro: ошибка чтения номера состояния\n");
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
            arr[0] = k + 0.0 * I;
            break;
        }
        default: {
            fprintf(stderr, "read_ro: неизвестный тип\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }
}

// --- Параллельное чтение матриц ---
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

// --- Печать диагонали ---
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

// --- Основной цикл ---
void run_time_evolution(int N, int nsteps, double dT, double complex *local_H, 
                        double complex *ro_local, int lld, int *descRo, 
                        int ictxt, int myrank, int mb, int nb)
{
    int zero = 0;
    int local_cols = numroc_(&N, &nb, &descRo[2], &zero, &descRo[4]); // descRo[4] - это mycol контекста
    // Примечание: local_cols можно передать и явно, но здесь мы вычисляем через descRo.
    // descRo[2] = npcol? Нет, descRo имеет строгую структуру. 
    // Для простоты вызовем Cblacs_gridinfo:
    int nprow, npcol, myrow, mycol;
    Cblacs_gridinfo(ictxt, &nprow, &npcol, &myrow, &mycol);

    print_abs_diag(N, ro_local, lld, descRo, 0, dT, myrank);

    char jobz = 'V', uplo = 'L';
    int ione = 1, info_heev = 0;
    double *eigvals = (double*)malloc(N * sizeof(double));

    double complex *A_heev = (double complex*)malloc(lld * local_cols * sizeof(double complex));
    memcpy(A_heev, local_H, lld * local_cols * sizeof(double complex));
    double complex *Q = (double complex*)calloc(lld * local_cols, sizeof(double complex));

    int lwork = -1, lrwork = -1;
    double complex work_query;
    double rwork_query;
    pzheev_(&jobz, &uplo, &N, A_heev, &ione, &ione, descRo, eigvals,
            Q, &ione, &ione, descRo, &work_query, &lwork,
            &rwork_query, &lrwork, &info_heev);
    if (info_heev != 0) { 
        if (myrank == 0) fprintf(stderr, "pzheev query failed: %d\n", info_heev);
        MPI_Abort(MPI_COMM_WORLD, 1); 
    }

    lwork = (int)creal(work_query);
    lrwork = (int)rwork_query;
    double complex *work = (double complex*)malloc(lwork * sizeof(double complex));
    double *rwork = (double*)malloc(lrwork * sizeof(double));

    pzheev_(&jobz, &uplo, &N, A_heev, &ione, &ione, descRo, eigvals,
            Q, &ione, &ione, descRo, work, &lwork, rwork, &lrwork, &info_heev);
    if (info_heev != 0) { 
        if (myrank == 0) fprintf(stderr, "pzheev failed: Hamiltonian is not Hermitian. info = %d\n", info_heev);
        MPI_Abort(MPI_COMM_WORLD, 1); 
    }
    free(A_heev); free(work); free(rwork);

    double complex alpha = 1.0 + 0.0*I, beta = 0.0 + 0.0*I;

    double complex *U_step = (double complex*)malloc(lld * local_cols * sizeof(double complex));
    memcpy(U_step, Q, lld * local_cols * sizeof(double complex));
    for (int j = 0; j < local_cols; ++j) {
        int j1 = j + 1;
        int jg = indxl2g_(&j1, &nb, &mycol, &zero, &npcol) - 1;
        double complex mult = cexp(-I * eigvals[jg] * dT);
        for (int ii = 0; ii < lld; ++ii) U_step[ii + j*lld] *= mult;
    }

    double complex *I_mat = (double complex*)calloc(lld * local_cols, sizeof(double complex));
    for (int j = 0; j < local_cols; ++j) {
        int j1 = j + 1;
        int jg = indxl2g_(&j1, &nb, &mycol, &zero, &npcol) - 1;
        for (int ii = 0; ii < lld; ++ii) {
            int i1 = ii + 1;
            int ig = indxl2g_(&i1, &mb, &myrow, &zero, &nprow) - 1;
            if (ig == jg) I_mat[ii + j*lld] = 1.0;
        }
    }
    
    double complex *QH = (double complex*)calloc(lld * local_cols, sizeof(double complex));
    pzgemm_("N", "C", &N, &N, &N, &alpha, I_mat, &ione, &ione, descRo,
            Q, &ione, &ione, descRo, &beta, QH, &ione, &ione, descRo);
    free(I_mat);

    double complex *temp = (double complex*)calloc(lld * local_cols, sizeof(double complex));
    pzgemm_("N", "N", &N, &N, &N, &alpha, U_step, &ione, &ione, descRo,
            QH, &ione, &ione, descRo, &beta, temp, &ione, &ione, descRo);
    memcpy(U_step, temp, lld * local_cols * sizeof(double complex));

    for (int step = 0; step < nsteps; ++step) {
        pzgemm_("N", "N", &N, &N, &N, &alpha, U_step, &ione, &ione, descRo,
                ro_local, &ione, &ione, descRo, &beta, temp, &ione, &ione, descRo);
        pzgemm_("N", "C", &N, &N, &N, &alpha, temp, &ione, &ione, descRo,
                U_step, &ione, &ione, descRo, &beta, ro_local, &ione, &ione, descRo);
        print_abs_diag(N, ro_local, lld, descRo, step + 1, dT, myrank);
    }

    free(Q); free(eigvals); free(U_step); free(QH); free(temp);
}
