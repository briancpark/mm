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
    C = jnp.matmul(A, B).block_until_ready()
    end = time.time()
    del A
    del B
    del C
    times.append(end - start)
    time.sleep(1)

print(times)