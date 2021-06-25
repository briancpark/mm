import numpy as np
from mpi4py import MPI

size = 2

comm = MPI.COMM_WORLD
p = comm.Get_size()
rank = comm.Get_rank()
p_c = int(p ** 0.5)


if not rank:
    A = np.random.rand(size, size)
    B = np.random.rand(size, size)
    C = np.zeros((size, size))
    print(A)

block_size = int(size / p_c)

block = MPI.DOUBLE.Create_vector(block_size, block_size, size)
block.Commit()

row_color = rank / p_c
col_color = rank % p_c

myA =  np.zeros((block_size, block_size))
myB =  np.zeros((block_size, block_size))
myC =  np.zeros((block_size, block_size))
buffA = np.zeros((block_size, block_size))
buffB = np.zeros((block_size, block_size))

row_comm = MPI.COMM_WORLD.Split(row_color, col_color)
col_comm = MPI.COMM_WORLD.Split(col_color, row_color)

MPI.COMM_WORLD.Barrier()

if not rank:
    for i in range(p_c):
        for j in range(p_c):
            req = MPI.COMM_WORLD.I_send(A)