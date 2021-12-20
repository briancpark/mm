#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <string.h>
#include <math.h>
#include <chrono>
#include<vector>
#include<numeric>
#include<unistd.h>
using namespace std;
using namespace std::chrono;
using std::chrono::high_resolution_clock;
using std::chrono::high_resolution_clock;
using std::chrono::duration_cast;
using std::chrono::duration;
using std::chrono::milliseconds;

/*
Timing Intel's MKL cblas function
*/

extern "C"
{
   #include <mkl.h>
   #include <mkl_cblas.h>
}

int main(int argc, char *argv[]) {
    if (argc != 2) {
        cout << "Missing argument n" << endl;
        return -1;
    }

    char *pEnd;
    unsigned long long int n = strtoull(argv[1], &pEnd, 10);
    unsigned long long int rows = n;
    unsigned long long int cols = n;

    double *A;
    double *B;
    double *C;

    int align = 64;
    unsigned int microsecond = 1000000; //For sleep
    
    //Throwaway
    cout << "Throw away test of " << n << "x" << n << endl;
    A = (double *) mkl_malloc(n * n * sizeof(double), align);
    if (A == NULL) {
        cout << "Memory allocation failed." << endl;
        return -1;
    }
    B = (double *) mkl_malloc(n * n * sizeof(double), align);
    if (B == NULL) {
        cout << "Memory allocation failed." << endl;
        mkl_free(A);
        return -1;
    }
    C = (double *) mkl_calloc(n * n, sizeof(double), align);    
    if (C == NULL) {
        cout << "Memory allocation failed." << endl;
        mkl_free(A);
        mkl_free(B);
        return -1;
    }
    
    cout << "Initializing matrices A, B, C (throwaway)" << endl;

    for (unsigned long long int i = 0; i < rows; i++) {
        for (unsigned long long int j = 0; j < cols; j++) {
            A[i * cols + j] = (double) rand() / RAND_MAX * 2.0 - 1.0;
        }
    }

    for(unsigned long long int i = 0; i < rows; i++) {
        for(unsigned long long int j = 0; j < cols; j++) {
            B[i * cols + j] = (double) rand() / RAND_MAX * 2.0 - 1.0;
        }
    }

    usleep(3 * microsecond);//sleeps for 3 second

    auto t1 = high_resolution_clock::now();
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n, n, n, 1., A, n, B, n, 0., C, n);
    auto t2 = high_resolution_clock::now();

    usleep(3 * microsecond);//sleeps for 3 second

    /* Getting number of milliseconds as a double. */
    duration<double, std::milli>  duration = t2 - t1;

    cout << "Time for throwaway: " << duration.count() / 1000.0 << endl;
    mkl_free(A);
    mkl_free(B);
    mkl_free(C);

    cout << "Performing cblas_dgemm() 10 times" << endl;
    vector <double> times = {};

    for (int i = 0; i < 10; i++) {
        double *A;
        double *B;
        double *C;

        A = (double *) mkl_malloc(n * n * sizeof(double), align);
        if (A == NULL) {
            cout << "Memory allocation failed." << endl;
            return -1;
        }
        B = (double *) mkl_malloc(n * n * sizeof(double), align);
        if (B == NULL) {
            cout << "Memory allocation failed." << endl;
            mkl_free(A);
            return -1;
        }
        C = (double *) mkl_calloc(n * n, sizeof(double), align);    
        if (C == NULL) {
            cout << "Memory allocation failed." << endl;
            mkl_free(A);
            mkl_free(B);
            return -1;
        }

        cout << ".";

        for (unsigned long long int i = 0; i < rows; i++) {
            for (unsigned long long int j = 0; j < cols; j++) {
                A[i * cols + j] = (double) rand() / RAND_MAX * 2.0 - 1.0;
            }
        }

        for(unsigned long long int i = 0; i < rows; i++) {
            for(unsigned long long int j = 0; j < cols; j++) {
                B[i * cols + j] = (double) rand() / RAND_MAX * 2.0 - 1.0;
            }
        }

        cout << "." << endl;
        usleep(3 * microsecond);//sleeps for 3 second
        auto t1 = high_resolution_clock::now();
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n, n, n, 1., A, n, B, n, 0., C, n);
        auto t2 = high_resolution_clock::now();
        usleep(3 * microsecond);//sleeps for 3 second
        
        /* Getting number of milliseconds as a double. */
        duration = t2 - t1;

        cout << "Time: " << duration.count() / 1000.0 << endl;
        
        times.push_back(duration.count() / 1000.0);

        mkl_free(A);
        mkl_free(B);
        mkl_free(C);
    }
    
    cout << "Average time: " << accumulate(times.begin(), times.end(), 0.0) / 10.0 << " seconds" << endl;
    

    return 0;
}