python3 benchmark.py


mpicxx -o summa summa.cpp
for i in 128, 512, 1024, 2048, 4096, 8192, 16384
    do mpirun -np 16 ./summa $i $i $i
done