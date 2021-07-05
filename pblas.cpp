#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <string.h>
#include <math.h>
#include <chrono>
using namespace std;
using namespace std::chrono;

/*
Timing Intel's MKL Cblas function
*/

extern "C"
{
   #include <mkl.h>
   #include <mkl_cblas.h>
   #include <mkl_pblas.h>
}

int main(int argc, char *argv[]) {
    if (argc != 2) {
        cout << "Missing argument n" << endl;
        return -1;
    }

    char *pEnd;
    //int n = atoi(argv[1]);
    int n = 100000;
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

    // example usage: https://github.com/GeorgiaTechMSSE/ReliableMD/blob/165774e7af9cd5d1f5d83b06432163cdfcaa8f80/cxsc-2-5-4/examples/FastPILSS-0-4-2/src/plss/cxsc_pblas.cpp
    int desca[9], descb[9], descc[9];
    int lld = n;
    int info;
    int irsrc = 0, icsrc = 0;
    if(lld == 0) {
        lld = 1;
    } 

    int ic = 0;//blas contxt?
    int blocksize = 100;

    //descinit_(desca, &n, &n, &blocksize, &blocksize, &irsrc, &icsrc, &ic, &lld, &info); 
    //descinit_(descb, &n, &n, &blocksize, &blocksize, &irsrc, &icsrc, &ic, &lld, &info); 
    //descinit_(descc, &n, &n, &blocksize, &blocksize, &irsrc, &icsrc, &ic, &lld, &info); 
    
    int ia=1, ja=1, ib=1, jb=1, icc=1, jc=1;
    double alpha = 1.0;
    double beta = 0.0;

    char *TRANS=(char*)"No Transpose";
    //pdgemm_(TRANS, TRANS, &r, &t, &s, &alpha, DA, &ia, &ja, descA, DB, &ib, &jb, 
    //        descB, &beta, DC, &icc, &jc, descC);




    auto start = high_resolution_clock::now();
    pdgemm_(TRANS, TRANS, &n, &n, &n, &alpha, A, &ia, &ja, desca, B, &ib, &jb, descb, &beta, C, &icc, &jc, descc);
    auto end = high_resolution_clock::now();

    auto duration = duration_cast<microseconds>(end - start);

    cout << (double) duration.count() / 1000000 << endl;

    mkl_free(A);
    mkl_free(B);
    mkl_free(C);

    return 0;
}