import matplotlib.pyplot as plt

sizes = [128, 512, 1024, 2048, 4096, 8192, 16384]
#nums = [0.19564533233642578, 0.017779827117919922, 0.06365418434143066, 0.3202025890350342, 0.7260684967041016, 2.1897120475769043, 9.845763921737671]
nums = [0.02242112159729004, 0.01976799964904785, 0.0379793643951416, 0.2405102252960205, 0.5096626281738281, 1.86411714553833, 8.293015956878662] # with ray 32 threads set
jax = [0.04932808876037598, 0.008392572402954102, 0.011348962783813477, 0.024466991424560547, 0.0724184513092041, 0.2940633296966553, 2.238332509994507]
numpy = [0.015555858612060547, 0.009791851043701172, 0.009885072708129883, 0.027075767517089844, 0.1310594081878662, 0.9636118412017822, 6.108663082122803]
[0.0025182688026688994, 0.0030226021946873514, 0.005866632895777002, 0.023828013596357778, 0.13086727880872787, 0.8614165204926394, 5.022364611498778]
mkl_cblas = [0.0195606, 0.0356428, 0.0408835, 0.0366671, 0.165296, 1.17308, 5.58987]
summa = [0.102857, 0.174629, 0.188206, 0.207394, 0.675447, 2.18323, 9.60133]
cosma = [0.013, 0.025, 0.033, 0.107, 0.327, 2.731, 12.953]
scalapack = [0.009, 0.017, 0.028, 0.051, 0.195, 1.114, 7.083]
slate = [0.000328, 0.00477, 0.0271, 0.198, 0.307, 0.850, 6.305]

plt.figure(figsize=(10, 10))
plt.plot(sizes, nums)
plt.plot(sizes, numpy)
plt.plot(sizes, jax)
plt.plot(sizes, summa)
plt.plot(sizes, mkl_cblas)
plt.plot(sizes, cosma)
plt.plot(sizes, scalapack)
plt.plot(sizes, slate)
plt.title(r"$n \times n$ DGEMM Benchmarks")
plt.xlabel(r"$n$ in $n \times n$ matrix")
plt.ylabel("Time in seconds")
plt.legend(["nums", "numpy", "jax", "summa", "mkl_cblas", "cosma", "scalapack", "slate"])
plt.savefig("figures/benchmark_all.png")
plt.clf()

plt.figure(figsize=(10, 10))
plt.plot(sizes, nums)
plt.plot(sizes, summa)
plt.plot(sizes, cosma)
plt.plot(sizes, scalapack)
plt.plot(sizes, slate)
plt.title(r"$n \times n$ PDGEMM Benchmarks Running on Shared Memory System")
plt.xlabel(r"$n$ in $n \times n$ matrix")
plt.ylabel("Time in seconds")
plt.legend(["nums", "summa", "cosma", "scalapack", "slate"])
plt.savefig("figures/benchmark_distributed.png")
plt.clf()

plt.figure(figsize=(10, 10))
plt.plot(sizes, numpy)
plt.plot(sizes, jax)
plt.plot(sizes, mkl_cblas)
plt.title(r"$n \times n$ DGEMM-only Benchmarks Running on Shared Memory System")
plt.xlabel(r"$n$ in $n \times n$ matrix")
plt.ylabel("Time in seconds")
plt.legend(["numpy", "jax", "mkl_cblas"])
plt.savefig("figures/benchmark_single_node.png")
plt.clf()
