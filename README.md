# mm

MPI Matrix multiplication implementation
http://www.umsl.edu/~siegelj/CS4740_5740/AlgorithmsII/mpi_mm.html


SUMMA?
https://github.com/irifed/mpi101


https://github.com/JGU-HPC/parallelprogrammingbook/tree/master/chapter9/matrix_matrix_mult



Links:

https://github.com/Schlaubischlump/cannons-algorithm.git 
https://github.com/andadiana/cannon-algorithm-mpi.git

## Scalapack installation
1. Install gfortran, cmake, openmpi
```
sudo snap install cmake --classic
sudo apt-get install gfortran, openmpi-bin libopenmpi-dev
sudo apt-get install libblas-dev liblapack-dev libatlas-base-dev 
```


rm CMakeCache.txt
mkdir build
cd build
cmake ..
make


install mkl
pip3 install mkl
numpy


https://github.com/eddelbuettel/mkl4deb






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