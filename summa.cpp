//Source: https://github.com/JGU-HPC/parallelprogrammingbook/blob/master/chapter9/matrix_matrix_mult/summa.cpp

#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <string.h>
#include <math.h>
#include "mpi.h"
#include <omp.h>

void readInput(int rows, int cols, double *data){

	// Open the file pointer
	/*FILE* fp = fopen(file.c_str(), "rb");
	// Check if the file exists
	if(fp == NULL){
		std::cout << "ERROR: File " << file << " could not be opened" << std::endl;
		MPI::COMM_WORLD.Abort(1);
	}
	for(int i=0; i<rows*cols; i++){
		if(!fscanf(fp, "%f", &data[i])){
			std::cout << "ERROR: Not enough values in file " << file << std::endl;
			MPI::COMM_WORLD.Abort(1);
		}
	}*/

    // checkerboard
    for(int i=0; i<rows; i++) {
        for(int j=0; j<cols; j++) {
            data[i*cols+j] = (double) rand() / RAND_MAX * 2.0 - 1.0;
            //data[i*cols+j] = (i+j) % 2;
        }
    }
}

void printOutput(int rows, int cols, double *data){
    /*
	FILE *fp = fopen("outSUMMA.txt", "wb");
	// Check if the file was opened
	if(fp == NULL){
		std::cout << "ERROR: Output file outSUMMA.txt could not be opened" << std::endl;
		MPI::COMM_WORLD.Abort(1);
	}

    for(int i=0; i<rows; i++){
        for(int j=0; j<cols; j++)
        	fprintf(fp, "%f ", data[i*cols+j]);
        fprintf(fp, "\n");
    }

    fclose(fp);
    */
}

int main (int argc, char *argv[]){
	// Initialize MPI
	MPI::Init(argc, argv);

	// Get the number of processes
	int p = MPI::COMM_WORLD.Get_size();

	// Get the ID of the process
	int rank = MPI::COMM_WORLD.Get_rank();

	if(argc < 4){
		// Only the first process prints the output message
		if(!rank){
			std::cout << "ERROR: The syntax of the program is ./summa m k n" << std::endl;
		}
		MPI::COMM_WORLD.Abort(1);
	}

	int m = atoi(argv[1]);
	int k = atoi(argv[2]);
	int n = atoi(argv[3]);

    int p_c = sqrt(p);
    // Check if a square grid could be created
    if(p_c*p_c != p){
		// Only the first process prints the output message
		if(!rank) {
			std::cout << "ERROR: The number of processes must be square" << std::endl;
        }
		MPI::COMM_WORLD.Abort(1);
    }

	if((m%p_c) || (n%p_c) || (k%p_c)){
		// Only the first process prints the output message
		if(!rank) {
			std::cout << "ERROR: 'm', 'k' and 'n' must be multiple of sqrt(numP)" << std::endl;
        }
		MPI::COMM_WORLD.Abort(1);
	}

	if((m < 1) || (n < 1) || (k<1)){
		// Only the first process prints the output message
		if(!rank)
			std::cout << "ERROR: 'm', 'k' and 'n' must be higher than 0" << std::endl;
		
		MPI::COMM_WORLD.Abort(1);
	}

	double *A;
	double *B;
	double *C;

	// Only one process reads the data from the files and broadcasts the data
	if(!rank){
		A = new double[m*k];
		readInput(m, k, A);
		B = new double[k*n];
		readInput(k, n, B);
		C = new double[m*n];
	}

	// The computation is divided by 2D blocks
	int blockRowsA = m / p_c;
	int blockRowsB = k / p_c;
	int blockColsB = n / p_c;

	// Create the datatypes of the blocks
	MPI::Datatype blockAType = MPI::DOUBLE.Create_vector(blockRowsA, blockRowsB, k);
	MPI::Datatype blockBType = MPI::DOUBLE.Create_vector(blockRowsB, blockColsB, n);
	MPI::Datatype blockCType = MPI::DOUBLE.Create_vector(blockRowsA, blockColsB, n);
	blockAType.Commit(); blockBType.Commit(); blockCType.Commit();

	double* myA = new double[blockRowsA*blockRowsB];
	double* myB = new double[blockRowsB*blockColsB];
	double* myC = new double[blockRowsA*blockColsB]();
	double* buffA = new double[blockRowsA*blockRowsB];
	double* buffB = new double[blockRowsB*blockColsB];

	// Measure the current time
	MPI::COMM_WORLD.Barrier();
	double start = MPI::Wtime();

	MPI::Request req;

	// Scatter A and B
	if(!rank){
		for(int i = 0; i < p_c; i++){
			for(int j = 0; j < p_c; j++){
				req = MPI::COMM_WORLD.Isend(A+i*blockRowsA*k+j*blockRowsB, 1, blockAType, i*p_c+j, 0);
				req = MPI::COMM_WORLD.Isend(B+i*blockRowsB*n+j*blockColsB, 1, blockBType, i*p_c+j, 0);
			}
		}
	}

	MPI::COMM_WORLD.Recv(myA, blockRowsA*blockRowsB, MPI::DOUBLE, 0, 0);
	MPI::COMM_WORLD.Recv(myB, blockRowsB*blockColsB, MPI::DOUBLE, 0, 0);

	// Create the communicators
    //Used to be intetrcomm? --Brian
	MPI::Intracomm rowComm = MPI::COMM_WORLD.Split(rank / p_c, rank % p_c);
	MPI::Intracomm colComm = MPI::COMM_WORLD.Split(rank % p_c, rank / p_c);

	// The main loop
	for(int i=0; i<p_c; i++){
		// The owners of the block to use must copy it to the buffer
		if(rank % p_c == i){
			memcpy(buffA, myA, blockRowsA*blockRowsB*sizeof(double));
		}
		if(rank / p_c == i){
			memcpy(buffB, myB, blockRowsB*blockColsB*sizeof(double));
		}

		// Broadcast along the communicators
		rowComm.Bcast(buffA, blockRowsA*blockRowsB, MPI::DOUBLE, i);
		colComm.Bcast(buffB, blockRowsB*blockColsB, MPI::DOUBLE, i);

		// The multiplication of the submatrice
		for(int i=0; i<blockRowsA; i++){
			for(int j=0; j<blockColsB; j++){
				for(int l=0; l<blockRowsB; l++){
					myC[i*blockColsB+j] += buffA[i*blockRowsB+l]*buffB[l*blockColsB+j];
				}
			}
		}
	}

	// Only process 0 writes
	// Gather the final matrix to the memory of process 0
	if(!rank){
		for(int i=0; i<blockRowsA; i++)
			memcpy(&C[i*n], &myC[i*blockColsB], blockColsB*sizeof(double));		

		for(int i=0; i < p_c; i++)
			for(int j=0; j < p_c; j++)
				if(i || j)
					MPI::COMM_WORLD.Recv(&C[i*blockRowsA*n+j*blockColsB], 1, blockCType, i*p_c+j, 0);
	} else 
		MPI::COMM_WORLD.Send(myC, blockRowsA*blockColsB, MPI::DOUBLE, 0, 0);

	// Measure the current time
	double end = MPI::Wtime();

	if(!rank){
    	std::cout << "Time with " << p << " processes: " << end-start << " seconds" << std::endl;
    	printOutput(m, n, C);
		delete [] A;
		delete [] B;
		delete [] C;
	}

	MPI::COMM_WORLD.Barrier();

	delete [] myA;
	delete [] myB;
	delete [] myC;
	delete [] buffA;
	delete [] buffB;

	// Terminate MPI
	MPI::Finalize();
	return 0;
}