import time
import numpy as np
import jax.numpy as jnp
import dimensions

ns = dimensions.ns
times = []
for n in ns:
    A = jnp.asarray(np.random.randn(n, n))
    B = jnp.asarray(np.random.randn(n, n))

    start = time.time()
    C = A @ B
    end = time.time()
    
    print(end - start, end = ', ')
    times.append(end - start)

print()