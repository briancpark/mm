import time
import mkl
import numpy as np
import dimensions

ns = dimensions.ns
times = []
for n in ns:
    total = 0.0
    for _ in range(10):
        print("initializing")
        A = np.random.randn(n, n)
        B = np.random.randn(n, n)

        print("running cblas")
        
        time.sleep(3)
        start = time.perf_counter()
        C = A @ B
        end = time.perf_counter()
        time.sleep(3)
        del A
        del B
        del C

        total += end - start
        
    times.append(total / 10.0)
        

print(times)