import time
import ray
ray.init()

from nums import numpy as nps
import numpy as np

ns = [128, 512, 1024, 2048, 4096, 8192, 16384]#, 32768, 65536, 131072, 262144]

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


print("numpy benchmarks")
for n in ns:
    A = np.random.randn(n, n)
    B = np.random.randn(n, n)

    start = time.time()
    C = A @ B
    end = time.time()
    print(n)
    print(end - start)
