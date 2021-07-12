import ray
ray.init(num_cpus=32)
import nums.numpy as nps
import numpy as np

ns = [128, 512, 1024, 2048, 4096, 8192, 16384]#, 32768, 65536, 131072, 262144]
for n in ns:
    A = np.loadtxt("A/A_" + str(n) + ".txt")
    B = np.loadtxt("B/B_" + str(n) + ".txt")
    C_correct = np.loadtxt("C/C_" + str(n) + ".txt")

    A = nps.array(A)
    B = nps.array(B)

    C = A @ B

    print(np.allclose(C.get(), C_correct))
