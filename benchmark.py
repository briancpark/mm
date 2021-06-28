import time
import ray
ray.init()

from nums import numpy as nps
import mkl
import numpy as np
import jax.numpy as jnp
#import matplotlib.pyplot as plt

#import numc as nc

ns = [128, 512, 1024, 2048, 4096, 8192, 16384]#, 32768, 65536, 131072, 262144]


nums_times = []
jax_times = []
numpy_times = []
numc_times = []
summa_times = []



print("nums benchmarks")

for n in ns:
    A = nps.random.randn(n, n)
    B = nps.random.randn(n, n)

    start = time.time()
    C = A @ B
    C.touch()
    end = time.time()
    print(n)
    print(end - start)
    nums_times.append(end - start)


print("jax benchmarks")
for n in ns:
    A = jnp.asarray(np.random.randn(n, n))
    B = jnp.asarray(np.random.randn(n, n))

    start = time.time()
    C = A @ B
    end = time.time()
    print(n)
    print(end - start)
    jax_times.append(end - start)


print("numpy benchmarks")
for n in ns:
    A = np.random.randn(n, n)
    B = np.random.randn(n, n)

    start = time.time()
    C = A @ B
    end = time.time()
    print(n)
    print(end - start)
    numpy_times.append(end - start)



print("numc benchmarks")
for n in ns:
    A = nc.Matrix(n, n, rand=True)
    B = nc.Matrix(n, n, rand=True)

    start = time.time()
    C = A * B
    end = time.time()
    print(n)
    print(end - start)


"""
plt.figure(figsize=(10, 10))
plt.plot(ns, nums_times)
plt.plot(ns, numpy_times)
plt.plot(ns, jax_times)
#plt.plot(ns, summa_times)
#plt.plot(ns, numc_times)
plt.xlabel("n in nxn matrix")
plt.ylabel("Time in seconds")
plt.legend(["nums", "numpy", "jax"])
plt.savefig("benchmark.png")
"""