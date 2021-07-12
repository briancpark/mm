import time
import numpy as np
import jax.numpy as jnp

ns = [128, 512, 1024, 2048, 4096, 8192, 16384]#, 32768, 65536, 131072, 262144]
for n in ns:
    A = np.loadtxt("A/A_" + str(n) + ".txt")
    B = np.loadtxt("B/B_" + str(n) + ".txt")
    C = np.loadtxt("C/C_" + str(n) + ".txt")

    A = jnp.asarray(A)
    B = jnp.asarray(B)
    C_correct = jnp.asarray(C)

    C = A @ B

    print(jnp.allclose(C, C_correct))
