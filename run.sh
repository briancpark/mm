#for n in 128, 512, 1024, 2048, 4096, 8192, 16384
#do
#	./cblas $n
#done
#printf "\n"


for n in 128, 512, 1024, 2048, 4096, 8192, 16384
do 
    mpirun -np 64 ./summa $n $n $n
done
printf "\n"