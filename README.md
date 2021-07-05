# DGEMM Benchmarks for NumS
A small case study to compare popular dgemm libraries with NumS. Here, we explore different optimizations as well as compilations with MPI, MKL, etc.

## Installation Guide

### ScaLAPACK Installation
1. Install gfortran, cmake, openmpi
```sh
sudo snap install cmake --classic
sudo apt-get install gfortran, openmpi-bin libopenmpi-dev
sudo apt-get install libblas-dev liblapack-dev libatlas-base-dev 
```
2. Build and compile
```sh
rm CMakeCache.txt
mkdir build
cd build
cmake ..
make
```

### MKL Installation
1. Install MKL. A guide for how to install it in Ubuntu Linux is [here](https://github.com/eddelbuettel/mkl4deb). With Ubuntu 20.04, simply install it with:
```sh
sudo apt install intel-mkl
```
2. Install and link MKL for Python (may require installation through conda as well)
```sh
pip3 install mkl numpy
```
Confirm that it has been installed:
```
python3
Python 3.6.9 (default, Jan 26 2021, 15:33:00) 
[GCC 8.4.0] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> import numpy as np
>>> np.show_config()
```

### NumS Installation
1. Installation is as follows from the NumS [setup guide](https://github.com/nums-project/nums):
```sh
pip install nums
# Run below to have NumPy use MKL.
conda install -fy mkl
conda install -fy numpy scipy
```
2. Also create a development environment
```sh
cd nums
conda create --name nums python=3.7 -y
conda activate nums
pip install -e ".[testing]"
```

## Running Programs

## MKL's cblas
A special Makefile to include Intel's compiler for MKL was made. A guide for how to make it was shown [here for reference](https://www.youtube.com/watch?v=PxMCthwZ8pw&t=945s).

## SUMMA implementation in C++ and OpenMPI

MPI Matrix multiplication implementation
http://www.umsl.edu/~siegelj/CS4740_5740/AlgorithmsII/mpi_mm.html


SUMMA?
https://github.com/irifed/mpi101
https://github.com/JGU-HPC/parallelprogrammingbook/tree/master/chapter9/matrix_matrix_mult

Links:
https://github.com/Schlaubischlump/cannons-algorithm.git 
https://github.com/andadiana/cannon-algorithm-mpi.git



For very large matrices in `summa.cpp`, I get the error:
```
[brian:64309] *** Process received signal ***
[brian:64309] Signal: Segmentation fault (11)
[brian:64309] Signal code: Address not mapped (1)
[brian:64309] Failing at address: 0x564a0a728000
[brian:64309] [ 0] /lib/x86_64-linux-gnu/libc.so.6(+0x3f040)[0x7fcf267c5040]
[brian:64309] [ 1] ./summa(+0xba6e)[0x564a09bdaa6e]
[brian:64309] [ 2] ./summa(+0xbf8c)[0x564a09bdaf8c]
[brian:64309] [ 3] /lib/x86_64-linux-gnu/libc.so.6(__libc_start_main+0xe7)[0x7fcf267a7bf7]
[brian:64309] [ 4] ./summa(+0xb91a)[0x564a09bda91a]
[brian:64309] *** End of error message ***
--------------------------------------------------------------------------
mpirun noticed that process rank 0 with PID 0 on node brian exited on signal 11 (Segmentation fault).
--------------------------------------------------------------------------
```




for clbas studd with mkl


limit benchmarks to n = 8GB