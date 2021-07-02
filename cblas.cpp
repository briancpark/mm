#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <string.h>
#include <math.h>
#include <chrono>
using namespace std;
using namespace std::chrono;

extern "C"
{
   #include <mkl.h>
   #include <mkl_cblas.h>
}

int main() {
    int n = 4096;
    int rows = n;
    int cols = n;

    double *A;
    double *B;
    double *C;

    A = (double *) mkl_malloc(n * n * sizeof(double), 64);
    B = (double *) mkl_malloc(n * n * sizeof(double), 64);
    C = (double *) mkl_malloc(n * n * sizeof(double), 64);    
    
    for(int i=0; i<rows; i++) {
        for(int j=0; j<cols; j++) {
            A[i*cols+j] = (double) rand() / RAND_MAX * 2.0 - 1.0;
        }
    }

    for(int i=0; i<rows; i++) {
        for(int j=0; j<cols; j++) {
            B[i*cols+j] = (double) rand() / RAND_MAX * 2.0 - 1.0;
        }
    }

    auto start = high_resolution_clock::now();
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, n, n, n, 1., A, n, B, n, 1., C, n);
    auto end = high_resolution_clock::now();

    auto duration = duration_cast<microseconds>(end - start);
  
  
    cout << (double) duration.count() / 1000000 << endl;

    return 0;
}