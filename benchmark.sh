#python3 benchmark.py

mpicxx -o summa summa.cpp -lblas
for i in 128, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144
    do mpirun -np 64 ./summa $i $i $i
done