import time
import mkl
import numpy as np
import dimensions

ns = dimensions.ns
times = []
for n in ns:
    A = np.random.randn(n, n)
    B = np.random.randn(n, n)

    start = time.time()
    C = A @ B
    end = time.time()
    
    print(end - start, end = ', ')
    times.append(end - start)

print()