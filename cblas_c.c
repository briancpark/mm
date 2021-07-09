#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <sys/time.h>
 
/*
Timing Intel's MKL cblas function
*/

#include <mkl.h>
#include <mkl_cblas.h>

int main(int argc, char *argv[]) {
    if (argc != 2) {

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

    A = (double *) mkl_malloc(n * n * sizeof(double), align);
    if (A == NULL) {

        return -1;
    }
    B = (double *) mkl_malloc(n * n * sizeof(double), align);
    if (B == NULL) {

        mkl_free(A);
        return -1;
    }
    C = (double *) mkl_calloc(n * n, sizeof(double), align);    
    if (C == NULL) {
    
        mkl_free(A);
        mkl_free(B);
        return -1;
    }
    
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




    struct timeval stop, start;
    gettimeofday(&start, NULL);
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, n, n, n, 1., A, n, B, n, 0., C, n);
    gettimeofday(&stop, NULL);
    printf("took %lu us\n", (stop.tv_sec - start.tv_sec) * 1000000 + stop.tv_usec - start.tv_usec);






    mkl_free(A);
    mkl_free(B);
    mkl_free(C);

    return 0;
}