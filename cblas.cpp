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
   #include <cblas.h>
}

void square_dgemm(int n, double* A, double* B, double* C) {
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, n, n, n, 1., A, n, B, n, 1., C, n);
}

int main() {
    int n = 4096;
    int rows = n;
    int cols = n;

    double *A;
    double *B;
    double *C;

    A = new double[n * n];
    B = new double[n * n];
    C = new double[n * n];    
    
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
    square_dgemm(n, A, B, C);
    auto stop = high_resolution_clock::now();

    auto duration = duration_cast<microseconds>(stop - start);
  
    cout << (double) duration.count() / 1000000 << endl;

    return 0;
}