#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <string.h>
#include <math.h>
#include <chrono>
#include "mpi.h"
using namespace std;
using namespace std::chrono;

/*
Timing Intel's MKL pblas function
*/

extern "C"
{
   #include <mkl.h>
   #include <mkl_cblas.h>
   #include <mkl_pblas.h>
   #include <mkl_blacs.h>
   #include <mkl_scalapack.h>
}

/* Definitions for proper work of examples on Windows */
#define blacs_pinfo_ BLACS_PINFO
#define blacs_get_ BLACS_GET
#define blacs_gridinit_ BLACS_GRIDINIT
#define blacs_gridinfo_ BLACS_GRIDINFO
#define blacs_barrier_ BLACS_BARRIER
#define blacs_gridexit_ BLACS_GRIDEXIT
#define blacs_exit_ BLACS_EXIT
#define igebs2d_ IGEBS2D
#define igebr2d_ IGEBR2D
#define sgebs2d_ SGEBS2D
#define sgebr2d_ SGEBR2D
#define dgebs2d_ DGEBS2D
#define dgebr2d_ DGEBR2D
#define sgesd2d_ SGESD2D
#define sgerv2d_ SGERV2D
#define dgesd2d_ DGESD2D
#define dgerv2d_ DGERV2D
#define numroc_ NUMROC
#define descinit_ DESCINIT
#define psnrm2_ PSNRM2
#define pdnrm2_ PDNRM2
#define psscal_ PSSCAL
#define pdscal_ PDSCAL
#define psdot_ PSDOT
#define pddot_ PDDOT
#define pslamch_ PSLAMCH
#define pdlamch_ PDLAMCH
#define indxg2l_ INDXG2L
#define pscopy_ PSCOPY
#define pdcopy_ PDCOPY
#define pstrsv_ PSTRSV
#define pdtrsv_ PDTRSV
#define pstrmv_ PSTRMV
#define pdtrmv_ PDTRMV
#define pslange_ PSLANGE
#define pdlange_ PDLANGE
#define psgemm_ PSGEMM
#define pdgemm_ PDGEMM
#define psgeadd_ PSGEADD
#define pdgeadd_ PDGEADD


/* Pi-number */
#ifndef M_PI
#define M_PI 3.14159265358979323846264338327
#endif

/* Definition of MIN and MAX functions */
#define MAX(a,b)((a)<(b)?(b):(a))
#define MIN(a,b)((a)>(b)?(b):(a))

/* Definition of matrix descriptor */
typedef MKL_INT MDESC[ 9 ];

int main(int argc, char *argv[]) {
    //TODO, change it to make it modular
    if (argc != 1) {
        cout << "Missing argument n" << endl;
        return -1;
    }
    
    char *pEnd;
    //int n = atoi(argv[1]);
    MKL_INT n = 16384;
    int rows = n;
    int cols = n;

    double *A;
    double *B;
    double *C;

    A = (double *) mkl_malloc(n * n * sizeof(double), 32768);
    B = (double *) mkl_malloc(n * n * sizeof(double), 32768);
    C = (double *) mkl_malloc(n * n * sizeof(double), 32768);    
    
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            A[i * cols + j] = (double) rand() / RAND_MAX * 2.0 - 1.0;
        }
    }

    for(int i = 0; i < rows; i++) {
        for(int j = 0; j < cols; j++) {
            B[i * cols + j] = (double) rand() / RAND_MAX * 2.0 - 1.0;
        }
    }

    // example usage: https://github.com/4estlaine/parallel_lqr/blob/e24442fc853bd58fcab630c7b8b96116a53d151a/src/scalapack/pblasc/source/pblas3_d_example.c
    // example usage: https://github.com/GeorgiaTechMSSE/ReliableMD/blob/165774e7af9cd5d1f5d83b06432163cdfcaa8f80/cxsc-2-5-4/examples/FastPILSS-0-4-2/src/plss/cxsc_pblas.cpp

    cout << "about to do pblas" << endl;
    
    
    
    
    int desca[9], descb[9], descc[9];
    int lld = n;
    int info;
    int irsrc = 0, icsrc = 0;
    if(lld == 0) {
        lld = 1;
    } 
    
    int ic = 10;//blas context?
    int blocksize = 100;

    descinit_(desca, &n, &n, &blocksize, &blocksize, &irsrc, &icsrc, &ic, &lld, &info); 
    descinit_(descb, &n, &n, &blocksize, &blocksize, &irsrc, &icsrc, &ic, &lld, &info); 
    descinit_(descc, &n, &n, &blocksize, &blocksize, &irsrc, &icsrc, &ic, &lld, &info); 
    
    int ia=1, ja=1, ib=1, jb=1, icc=1, jc=1;
    double alpha = 1.0;
    double beta = 1.0;

    char *TRANS=(char*)"No Transpose";

    auto start = high_resolution_clock::now();
    cout << "bug?" << endl;
    MPI::Init(argc, argv);

	int p = MPI::COMM_WORLD.Get_size();
	int rank = MPI::COMM_WORLD.Get_rank();
    pdgemm_(TRANS, TRANS, &n, &n, &n, &alpha, A, &ia, &ja, desca, B, &ib, &jb, descb, &beta, C, &icc, &jc, descc);
    MPI::Finalize();
    auto end = high_resolution_clock::now();

    auto duration = duration_cast<microseconds>(end - start);

    cout << (double) duration.count() / 1000000 << endl;
    
    mkl_free(A);
    mkl_free(B);
    mkl_free(C);

    


    return 0;
}