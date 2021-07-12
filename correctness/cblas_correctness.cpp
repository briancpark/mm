#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <string.h>
#include <math.h>
#include <chrono>
#include <fstream>      
using namespace std;
using namespace std::chrono;
using std::chrono::high_resolution_clock;
using std::chrono::high_resolution_clock;
using std::chrono::duration_cast;
using std::chrono::duration;
using std::chrono::milliseconds;

extern "C"
{
   #include <mkl.h>
   #include <mkl_cblas.h>
}

void readMatrix(double *matrix, string filename, int size) {
    ifstream f(filename);   

    for (int i = 0; i < size * size; i++) {
        f >> matrix[i];
    }
}

bool compare(double *C, double *C_correct, int size) {
    bool correct_flag = true;

    for (int i = 0; i < size * size; i++) {
        
        if (C[i] != C_correct[i]) {
            return false;
        }
    }
    return correct_flag;
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
    double *C_correctness;

    int align = 64;

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

    C_correctness = (double *) mkl_calloc(n * n, sizeof(double), align);    
    if (C_correctness == NULL) {
        cout << "Memory allocation failed." << endl;
        mkl_free(A);
        mkl_free(B);
        return -1;
    }

    readMatrix(A, "A/A_128.txt", 128);
    readMatrix(B, "B/B_128.txt", 128);
    readMatrix(C_correctness, "C/C_128.txt", 128);

   
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n, n, n, 1., A, n, B, n, 0., C, n);

    if (compare(C, C_correctness, 128)) {
        cout << "check!" << endl;
    } else {
        cout << "incorrect!" << endl;
    }

    mkl_free(A);
    mkl_free(B);
    mkl_free(C);
    mkl_free(C_correctness);

    return 0;
}