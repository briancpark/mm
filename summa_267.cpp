#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <string.h>
#include <math.h>
#include "mpi.h"
#include <omp.h>


/*
    Implementation from cs267
*/



void SimpleDGEMM(double *Atemp, double *Btemp, double *C, int rows, int cols, int stride) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            for (int k = 0; k < stride; k++) {
                
            }
        }
    }
}

void SUMMA(double *mA, double *mB, double *mc, int rank, int p_c, int size) {
    int row_color = rank / p_c; // p_c = sqrt(p) for simplicity
    int col_color = rank % p_c;

    MPI::Intracomm row_comm = MPI::COMM_WORLD.Split(row_color, col_color);
	MPI::Intracomm col_comm = MPI::COMM_WORLD.Split(col_color, row_color);
    
    for (int k = 0; k < p_c; k++) {
        if (col_color == k) {
            memcpy(Atemp, mA, size);
        }
        if (row_color == k) { 
            memcpy(Btemp, mB, size);
        }
        MPI_Bcast(Atemp, size / p_c, MPI_DOUBLE, k, row_comm);
        MPI_Bcast(Btemp, size / p_c, MPI_DOUBLE, k, col_comm);
        SimpleDGEMM(Atemp, Btemp, mc, size / p_c, size / p_c, size) / p_c;
    }
}

int main(int argc, char *argv[]) {
    int size = 100;
    MPI::Init(argc, argv);

    int p = MPI::COMM_WORLD.Get_size();
    int rank = MPI::COMM_WORLD.Get_rank();
    int p_c = sqrt(p);

    if (!rank) {
        

        double *A = new double[size * size];
        double *B = new double[size * size];
        double *C = new double[size * size];

    }

   
    if (!rank) {
        SUMMA(A, B, C, rank, p_c, size);
    }

    //printf("%d", p_c);

    
    MPI::Finalize();
}
