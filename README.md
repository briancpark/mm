# DGEMM Benchmarks for NumS
A small case study to compare popular DGEMM libraries with NumS. Here, we explore different kinds of optimizations with MPI, MKL, etc.

## Installation Guide

### ScaLAPACK (Netlib) Installation
Skip this installation if on Intel architecture. Intel has their own distribution of ScaLAPACK opimized with MKL.
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

### ScaLAPACK MKL Installation
1. Install MKL. A guide for how to install it in Ubuntu Linux is [here](https://github.com/eddelbuettel/mkl4deb). With Ubuntu 20.04, simply install it with:
```sh
sudo apt install intel-mkl
```
2. Additionally, you have to install IntelMPI for communication algorithms in ScaLAPACK MKL to work properly. Installation guide can be found [here](https://software.intel.com/content/www/us/en/develop/tools/oneapi/hpc-toolkit/download.html?operatingsystem=linux&distributions=aptpackagemanager).
3. Additionally, set environent variables in the shell. (This will be needed later if you want to compile C/C++ programs with MKL library)
```sh
export MKL_LIB_DIR=/opt/intel/compilers_and_libraries/linux/mkl/lib/intel64
export MKL_INCLUDE_DIR=/opt/intel/compilers_and_libraries/linux/mkl/include
export LD_LIBRARY_PATH=$MKL_LIB_DIR:$LD_LIBRARY_PATH
export MKLROOT=/opt/intel/mkl
```
4. Setting an envrionment variable for number of threads can also be helpful if it happens to not be running on multiple threads.
```sh
export MKL_NUM_THREADS=32
```
5. A shell configuration has to be run in order for IntelMPI to be run properly:
```sh
source /opt/intel/oneapi/setvars.sh 
```

### ScaLAPACK MKL LAPACK for NumPy Installation
1. Install and link MKL for Python (may require installation through conda as well)
```sh
pip3 install mkl numpy
```
2. Confirm that it has been installed:
```
python3
Python 3.6.9 (default, Jan 26 2021, 15:33:00) 
[GCC 8.4.0] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> import numpy as np
>>> np.show_config()
```
By default, it may be set to OpenBLAS, so always check before running benchmarks.

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
2. Note that all MKL envrionment variables should be set correctly. Again, it is probably safe to enter these commands again:
```sh
export MKL_LIB_DIR=/opt/intel/compilers_and_libraries/linux/mkl/lib/intel64
export MKL_INCLUDE_DIR=/opt/intel/compilers_and_libraries/linux/mkl/include
export LD_LIBRARY_PATH=$MKL_LIB_DIR:$LD_LIBRARY_PATH
export MKLROOT=/opt/intel/mkl
export MKL_NUM_THREADS=32
```
3. Install with these commands. For some reason, cloning with the recursive flag is **VERY** important!
```sh
git clone --recursive https://github.com/eth-cscs/COSMA cosma && cd cosma
mkdir build && cd build
cmake -DCOSMA_BLAS=MKL -DCOSMA_SCALAPACK=MKL -DCMAKE_INSTALL_PREFIX=../install/cosma ..
make -j
make install
```

## Running Programs

### NumPy with MKL
Just make sure optimizations are on and import MKL before NumPy:
```python
import mkl
import numpy as np
```
If MKL isn't installed, then Python would throw an `ImportError`.

### MKL's LAPACK `cblas_dgemm()`
1. A special Makefile to include Intel's compiler for MKL was made. A guide for how to make it was shown [here for reference](https://www.youtube.com/watch?v=PxMCthwZ8pw&t=945s) and [here](https://software.intel.com/content/www/us/en/develop/documentation/mkl-tutorial-c/top/multiplying-matrices-using-dgemm.html). Additionally, there is a link advisor that calculates what flags you need depending on what you have instaled: [link advisor](https://software.intel.com/content/www/us/en/develop/tools/oneapi/components/onemkl/link-line-advisor.html). Make sure environment variables are written to `~/.bashrc` (if using bash):
```sh
export MKL_LIB_DIR=/opt/intel/compilers_and_libraries/linux/mkl/lib/intel64
export MKL_INCLUDE_DIR=/opt/intel/compilers_and_libraries/linux/mkl/include
export LD_LIBRARY_PATH=$MKL_LIB_DIR:$LD_LIBRARY_PATH
export MKLROOT=/opt/intel/mkl
```
2. Compile and run
```sh
make cblas
export MKL_NUM_THREADS=32 #if not already set
./cblas <dimension size>
```
3. (Additional Information) [Notes for MKL installation](https://cirrus.readthedocs.io/en/master/software-libraries/intel_mkl.html). Clarification whether to use ILP or LP interface:
ILP vs LP interface layer
Most applications will use 32-bit (4-byte) integers. This means the MKL 32-bit integer inteface should be selected (which gives the _lp64 extensions seen in the examples above).
For applications which require, e.g., very large array indices (greater than 2^31-1 elements), the 64-bit integer interface is required. This gives rise to _ilp64 appended to library names. This may also require -DMKL_ILP64 at the compilation stage. Check the Intel link line advisor for specific cases.

## COSMA
```sh
mpirun -np 4 ./build/miniapp/pxgemm_miniapp -m 10000 -n 10000 -k 10000 --block_a=100,100 --block_b=100,100 --block_c=100,100 --p_grid=2,2 --transpose=NN --type=double --algorithm=scalapack
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

# Notes (Draft)

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


limit benchmarks to n = 8GB




Discussion about parallel LA packages
https://stackoverflow.com/questions/10025866/parallel-linear-algebra-for-multicore-system


NUMA nodes
https://stackoverflow.com/questions/48072530/mpi-code-only-using-one-of-two-numa-nodes


Architecture
```
lsArchitecture:        x86_64
CPU op-mode(s):      32-bit, 64-bit
Byte Order:          Little Endian
CPU(s):              64
On-line CPU(s) list: 0-63
Thread(s) per core:  2
Core(s) per socket:  16
Socket(s):           2
NUMA node(s):        2
Vendor ID:           GenuineIntel
CPU family:          6
Model:               85
Model name:          Intel(R) Xeon(R) Platinum 8272CL CPU @ 2.60GHz
Stepping:            7
CPU MHz:             2593.904
BogoMIPS:            5187.80
Virtualization:      VT-x
Hypervisor vendor:   Microsoft
Virtualization type: full
L1d cache:           32K
L1i cache:           32K
L2 cache:            1024K
L3 cache:            36608K
NUMA node0 CPU(s):   0-31
NUMA node1 CPU(s):   32-63
Flags:               fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush mmx fxsr sse sse2 ss ht syscall nx pdpe1gb rdtscp lm constant_tsc rep_good nopl xtopology cpuid pni pclmulqdq vmx ssse3 fma cx16 pcid sse4_1 sse4_2 movbe popcnt aes xsave avx f16c rdrand hypervisor lahf_lm abm 3dnowprefetch invpcid_single tpr_shadow vnmi ept vpid ept_ad fsgsbase bmi1 hle avx2 smep bmi2 erms invpcid rtm mpx avx512f avx512dq rdseed adx smap clflushopt clwb avx512cd avx512bw avx512vl xsaveopt xsavec xgetbv1 xsaves avx512_vnni md_clear arch_capabilities
```