#include <stdio.h>
#include <stdlib.h>
#include <complex.h>
#include <mpi.h>

#include <math.h>

typedef struct {
    int n1, k1, n2, k2;
} State;

int print_err(const char *str, int rank) {
    if (rank == 0) {
        printf("ERROR: %s\n\n", str);
        fflush(stdout);
    }
    MPI_Abort(MPI_COMM_WORLD, 1); 
    return 1;
}


int count_states_global(int N, double wa, double wc, double Emin, double Emax, int rank, int size) {
    int max_photons = N;
    int max_excitations = N / 2;
    
    long long total_combinations = (long long)(max_photons + 1) * 
                                   (max_excitations + 1) * 
                                   (max_photons + 1) * 
                                   (max_excitations + 1);
    
    int local_K = 0;
    
    for (long long i = rank; i < total_combinations; i += size) {
        long long rem = i;
        
        int n1 = rem / ((max_excitations + 1) * (max_photons + 1) * (max_excitations + 1));
        rem %= ((max_excitations + 1) * (max_photons + 1) * (max_excitations + 1));
        
        int k1 = rem / ((max_photons + 1) * (max_excitations + 1));
        rem %= ((max_photons + 1) * (max_excitations + 1));
        
        int n2 = rem / (max_excitations + 1);
        int k2 = rem % (max_excitations + 1);
        
        double E = wc * (n1 + n2) + wa * (k1 + k2); // ЭНЕРГИЯ
        
        if (E >= Emin && E <= Emax) {
            local_K++;
        }
    }
    
    int global_K = 0;
    
    MPI_Allreduce(&local_K, &global_K, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    
    return global_K;
}

int fill_basis(State *basis, int N, double wa, double wc, double Emin, double Emax) {
    int max_photons     = N;
    int max_excitations = N / 2;
    int idx = 0;

    for (int n1 = 0; n1 <= max_photons; n1++) {
        for (int k1 = 0; k1 <= max_excitations; k1++) {
            for (int n2 = 0; n2 <= max_photons; n2++) {
                for (int k2 = 0; k2 <= max_excitations; k2++) {
                    double E = wc * (n1 + n2) + wa * (k1 + k2);
                    if (E >= Emin && E <= Emax) {
                        basis[idx].n1 = n1;
                        basis[idx].k1 = k1;
                        basis[idx].n2 = n2;
                        basis[idx].k2 = k2;
                        idx++;
                    }
                }
            }
        }
    }
    return idx;
}

void fill_local_H(double complex *local_H, int local_rows, int local_cols,
                  int myrow, int mycol, int nprow, int npcol, int block_size,
                  State *basis, int K_glob, double complex a, double complex b, double wa, double wc, int N){
    for (int lj = 0; lj < local_cols; lj++) {
        int gj = indxl2g_(&lj, &block_size, &mycol, &izero, &npcol); // столбец
		if (gj >= K_glob) continue;
        

        for (int li = 0; li < local_rows; li++) {
			int gi = indxl2g_(&li, &block_size, &myrow, &izero, &nprow);//строка
			if (gi >= K_glob) continue;
			
			int lrindx, lcindx, rsrc, csrc;
			infog2l_(&gi, &gj, descH, &nprow, &npcol, &myrow, &mycol, &lrindx, &lcindx, &rsrc, &csrc);
			if (rsrc != myrow || csrc != mycol) continue;
			
            double complex elem = calculate_element(basis[gi], basis[gj], a, b, wa, wc, N);

            // по столбцап (li, lj) лежит в li + lj*local_rows
            local_H[li + lj * local_rows] = elem;
        }
    }
}


double complex calculate_element(State si, State sj,
                                 double complex a, double complex b,
                                 double wa, double wc, int N)
{
    int M = N / 2;

    // si == sj
    if (si.n1 == sj.n1 && si.k1 == sj.k1 &&
        si.n2 == sj.n2 && si.k2 == sj.k2) {
        return wc * (sj.n1 + sj.n2) + wa * (sj.k1 + sj.k2) + 0.0 * I;
    }

    // перелет фотона 1 -> 2: |n1, k1, n2, k2> -> |n1-1, k1, n2+1, k2>
    if (sj.n1 >= 1) {
        State t = { sj.n1 - 1, sj.k1, sj.n2 + 1, sj.k2 };
        if (t.n1 == si.n1 && t.k1 == si.k1 &&
            t.n2 == si.n2 && t.k2 == si.k2) {
            return conj(a) * sqrt((double)sj.n1) * sqrt((double)(sj.n2 + 1));
        }
		}
    if (sj.n2 >= 1) {
		//перелет фотона 2 -> 1: |n1, k1, n2, k2> -> |n1+1, k1, n2-1, k2>
        State t = { sj.n1 + 1, sj.k1, sj.n2 - 1, sj.k2 };
        if (t.n1 == si.n1 && t.k1 == si.k1 &&
            t.n2 == si.n2 && t.k2 == si.k2) {
            return a * sqrt((double)sj.n2) * sqrt((double)(sj.n1 + 1));
        }
    }
    // поглощение в полости 1: |n1, k1> -> |n1-1, k1+1>
    if (sj.n1 >= 1 && sj.k1 < M) {
        State t = { sj.n1 - 1, sj.k1 + 1, sj.n2, sj.k2 };
        if (t.n1 == si.n1 && t.k1 == si.k1 &&
            t.n2 == si.n2 && t.k2 == si.k2) {
            return conj(b) * sqrt((double)sj.n1) * sqrt((double)((M - sj.k1) * (sj.k1 + 1)));
        }
    }
    //излучение в полости 1: |n1, k1> -> |n1+1, k1-1>
    if (sj.k1 >= 1) {
        State t = { sj.n1 + 1, sj.k1 - 1, sj.n2, sj.k2 };
        if (t.n1 == si.n1 && t.k1 == si.k1 &&
            t.n2 == si.n2 && t.k2 == si.k2) {
            return b * sqrt((double)(sj.n1 + 1)) * sqrt((double)(sj.k1 * (M - sj.k1 + 1)));
        }
    } // поглощение в полости 2: |n2, k2> -> |n2-1, k2+1>
    if (sj.n2 >= 1 && sj.k2 < M) {
        State t = { sj.n1, sj.k1, sj.n2 - 1, sj.k2 + 1 };
        if (t.n1 == si.n1 && t.k1 == si.k1 &&
            t.n2 == si.n2 && t.k2 == si.k2) {
            return conj(b) * sqrt((double)sj.n2)
                   * sqrt((double)((M - sj.k2) * (sj.k2 + 1))); } }

    // излучениев полости 2: |n2, k2> -> |n2+1, k2-1>
    if (sj.k2 >= 1) {
        State t = { sj.n1, sj.k1, sj.n2 + 1, sj.k2 - 1 };
        if (t.n1 == si.n1 && t.k1 == si.k1 &&
            t.n2 == si.n2 && t.k2 == si.k2) {
            return b * sqrt((double)(sj.n2 + 1))* sqrt((double)(sj.k2 * (M - sj.k2 + 1)));
        }
    }

    return 0.0 + 0.0 * I;
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
    
	long long K_glob_ll = count_states_global(N, wa, wc, Emin, Emax, myrank, nprocs);
	if (K_glob_ll <= 0) { print_err("No states in the given energy range", myrank);}
	if (K_glob_ll > INT_MAX) { print_err("K_glob is too large for ScaLAPACK int interface", myrank); }
	int K_glob = (int) K_glob_ll;
	
	State *basis = malloc(K_glob * sizeof(State));
	if (!basis) print_err("malloc basis failed", myrank);
	int check = fill_basis(basis, N, wa, wc, Emin, Emax);
	if (check != K_glob) { print_err("basis fill mismatch: count != fill", myrank);}
	
	
	int ictxt;
	Cblacs_get(-1, 0, &ictxt);
	Cblacs_gridinit(&ictxt, "R", nprow, npcol);
	
	int max_procs = (nprow > npcol) ? nprow : npcol;
	if (max_procs < 1) max_procs = 1;
	int block_size = K_glob / max_procs;
	if (block_size < 1) {
		block_size = 1;
	}
	if (block_size > 64) {
		block_size = 64;
	}
	
	
	int izero = 0;
	int local_rows, local_cols;

	local_rows = numroc_(&K_glob, &block_size, &myrow, &izero, &nprow);
	local_cols = numroc_(&K_glob, &block_size, &mycol, &izero, &npcol);
	
	int descH[9];
	int info;

	descinit_(descH, &K_glob, &K_glob, &block_size, &block_size, 
			  &izero, &izero, &ictxt, &local_rows, &info);
			  
	if (info != 0) {
		print_err("descinit failed", myrank);
	}
	
	double complex *local_H = (double complex*) malloc(local_rows * local_cols * sizeof(double complex));
	if (local_H == NULL) {
		print_err("Memory allocation failed", myrank);
	}
	for (int i = 0; i < local_rows * local_cols; i++) {
		local_H[i] = 0.0 + 0.0 * I;
	}
	
	
	MPI_Finalize();
    return 0;
}

