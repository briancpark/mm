import matplotlib.pyplot as plt
import numpy as np

sizes = [128, 512, 1024, 2048, 4096, 8192, 16384]
nums = [0.23119258880615234, 0.021132707595825195, 0.08403682708740234, 0.531221866607666, 0.695296049118042, 2.4768660068511963, 12.109630584716797]
numpy = [0.002043485641479492, 0.007197856903076172, 0.05278182029724121, 0.4058499336242676, 3.1809639930725098, 35.26149344444275, 281.6816885471344]
summa = [0.0794792, 0.164761, 0.349334, 1.38058, 11.6713, 92.7757, 300]

#plt.figure(figsize=(10, 10))
#plt.plot(sizes, nums)
#plt.plot(sizes, numpy)
#plt.plot(sizes, summa)
#plt.xlabel("n in nxn matrix")
#plt.ylabel("Time in seconds")
#plt.legend(["nums", "numpy", "summa"])
#plt.savefig("benchmark.png")
