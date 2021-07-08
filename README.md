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

### Jax Installation
1.
```sh
pip3 install jax
```

### COSMA Installation
1. Installation instructions [here](https://github.com/eth-cscs/COSMA). 
2. Note that all MKL envrionment varialbes should be set correctly. Again, it is probably safe to enter these commands again:
```sh
export MKL_LIB_DIR=/opt/intel/compilers_and_libraries/linux/mkl/lib/intel64
export MKL_INCLUDE_DIR=/opt/intel/compilers_and_libraries/linux/mkl/include
export LD_LIBRARY_PATH=$MKL_LIB_DIR:$LD_LIBRARY_PATH
export MKLROOT=/opt/intel/mkl
```
3. Install with these commands. For some reason, cloning with teh recursive flag is VERY important!
```sh
git clone --recursive https://github.com/eth-cscs/COSMA cosma && cd cosma
mkdir build && cd build
cmake -DCOSMA_BLAS=MKL -DCOSMA_SCALAPACK=MKL -DCMAKE_INSTALL_PREFIX=../install/cosma ..
make -j
make install
```


mpirun -np 4 ./build/miniapp/pxgemm_miniapp -m 10000 -n 10000 -k 10000 --block_a=100,100 --block_b=100,100 --block_c=100,100 --p_grid=2,2 --transpose=NN --type=double --algorithm=scalapack

## Running Programs

## MKL's cblas
https://software.intel.com/content/www/us/en/develop/tools/oneapi/hpc-toolkit/download.html?operatingsystem=linux&distributions=aptpackagemanager
1. A special Makefile to include Intel's compiler for MKL was made. A guide for how to make it was shown [here for reference](https://www.youtube.com/watch?v=PxMCthwZ8pw&t=945s) and [here](https://software.intel.com/content/www/us/en/develop/documentation/mkl-tutorial-c/top/multiplying-matrices-using-dgemm.html). Requires environment variables to be written to `~/.bashrc` (if using bash):
```sh
export MKL_LIB_DIR=/opt/intel/compilers_and_libraries/linux/mkl/lib/intel64
export MKL_INCLUDE_DIR=/opt/intel/compilers_and_libraries/linux/mkl/include
export LD_LIBRARY_PATH=$MKL_LIB_DIR:$LD_LIBRARY_PATH
export MKLROOT=/opt/intel/mkl
```
2. Compile and run
```sh
make cblas
export MKL_NUM_THREADS=32
./cblas <dimension size>
```

Also make sure intel's MPI is installed:
```sh
source /opt/intel/oneapi/setvars.sh 
```

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




Notes for MKL installation
https://cirrus.readthedocs.io/en/master/software-libraries/intel_mkl.html
-DMKL_ILP64

ILP vs LP interface layer

Most applications will use 32-bit (4-byte) integers. This means the MKL 32-bit integer inteface should be selected (which gives the _lp64 extensions seen in the examples above).

For applications which require, e.g., very large array indices (greater than 2^31-1 elements), the 64-bit integer interface is required. This gives rise to _ilp64 appended to library names. This may also require -DMKL_ILP64 at the compilation stage. Check the Intel link line advisor for specific cases.

https://software.intel.com/content/www/us/en/develop/tools/oneapi/components/onemkl/link-line-advisor.html




https://stackoverflow.com/questions/10025866/parallel-linear-algebra-for-multicore-system