import numpy as np

ns = [128, 512, 1024, 2048, 4096, 8192, 16384]#, 32768, 65536, 131072, 262144]
for n in ns:
    np.random.seed(n)
    A = np.random.rand(n, n)
    np.savetxt("A/A_" + str(n) + ".txt", A)
    B = np.random.rand(n, n)
    np.savetxt("B/B_" + str(n) + ".txt", B)
    C = A @ B
    np.savetxt("C/C_" + str(n) + ".txt", C)

