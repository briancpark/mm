import time
import ray
import nums.numpy as nps
import dimensions

ns = dimensions.ns
times = []
for n in ns:
    A = nps.random.randn(n, n)
    B = nps.random.randn(n, n)

    start = time.time()
    C = A @ B
    C.touch()
    end = time.time()
    
    print(end - start, end = ', ')
    times.append(end - start)
print()

ray.shutdown()