for n in 128, 512, 1024, 2048, 4096, 8192, 16384
do
	./cblas $n
    sleep 5
done
printf "\n"


#for n in 128, 512, 1024, 2048, 4096, 8192, 16384
#do 
#    mpirun -np 64 ./summa $n $n $n
#done
#printf "\n"

#Cosma
#export OMP_NUM_THREADS=16; mpirun -np 2 ./../cosma/build/miniapp/pxgemm_miniapp -m 128 -n 128 -k 128 --block_a=128,128 --block_b=128,128 --block_c=128,128 --p_grid=2,2 --transpose=NN --type=double --algorithm=cosma -r 1
#export OMP_NUM_THREADS=16; mpirun -np 2 ./../cosma/build/miniapp/pxgemm_miniapp -m 512 -n 512 -k 512 --block_a=512,512 --block_b=512,512 --block_c=512,512 --p_grid=2,2 --transpose=NN --type=double --algorithm=cosma -r 1
#export OMP_NUM_THREADS=16; mpirun -np 2 ./../cosma/build/miniapp/pxgemm_miniapp -m 1024 -n 1024 -k 1024 --block_a=1024,1024 --block_b=1024,1024 --block_c=1024,1024 --p_grid=2,2 --transpose=NN --type=double --algorithm=cosma -r 1
#export OMP_NUM_THREADS=16; mpirun -np 2 ./../cosma/build/miniapp/pxgemm_miniapp -m 2048 -n 2048 -k 2048 --block_a=2048,2048 --block_b=2048,2048 --block_c=2048,2048 --p_grid=2,2 --transpose=NN --type=double --algorithm=cosma -r 1

#export OMP_NUM_THREADS=16; mpirun -np 2 ./../cosma/build/miniapp/pxgemm_miniapp -m 4096 -n 4096 -k 4096 --block_a=512,512 --block_b=512,512 --block_c=512,512 --p_grid=2,2 --transpose=NN --type=double --algorithm=cosma -r 1
#export OMP_NUM_THREADS=16; mpirun -np 2 ./../cosma/build/miniapp/pxgemm_miniapp -m 8192 -n 8192 -k 8192 --block_a=1024,1024 --block_b=1024,1024 --block_c=1024,1024 --p_grid=2,2 --transpose=NN --type=double --algorithm=cosma -r 1
#export OMP_NUM_THREADS=16; mpirun -np 2 ./../cosma/build/miniapp/pxgemm_miniapp -m 16384 -n 16384 -k 16384 --block_a=2048,2048 --block_b=2048,2048 --block_c=2048,2048 --p_grid=2,2 --transpose=NN --type=double --algorithm=cosma -r 1

#scalapack
#export MKL_NUM_THREADS=16; mpirun -np 2 ./../cosma/build/miniapp/pxgemm_miniapp -m 128 -n 128 -k 128 --block_a=128,128 --block_b=128,128 --block_c=128,128 --transpose=NN --type=double --algorithm=scalapack -r 1
#export MKL_NUM_THREADS=16; mpirun -np 2 ./../cosma/build/miniapp/pxgemm_miniapp -m 512 -n 512 -k 512 --block_a=128,128 --block_b=128,128 --block_c=128,128 --transpose=NN --type=double --algorithm=scalapack -r 1
#export MKL_NUM_THREADS=16; mpirun -np 2 ./../cosma/build/miniapp/pxgemm_miniapp -m 1024 -n 1024 -k 1024 --block_a=128,128 --block_b=128,128 --block_c=128,128 --transpose=NN --type=double --algorithm=scalapack -r 1
#export MKL_NUM_THREADS=16; mpirun -np 2 ./../cosma/build/miniapp/pxgemm_miniapp -m 2048 -n 2048 -k 2048 --block_a=128,128 --block_b=128,128 --block_c=128,128 --transpose=NN --type=double --algorithm=scalapack -r 1

#export MKL_NUM_THREADS=16; mpirun -np 2 ./../cosma/build/miniapp/pxgemm_miniapp -m 4096 -n 4096 -k 4096 --block_a=512,512 --block_b=512,512 --block_c=512,512 --transpose=NN --type=double --algorithm=scalapack -r 1
#export MKL_NUM_THREADS=16; mpirun -np 2 ./../cosma/build/miniapp/pxgemm_miniapp -m 8192 -n 8192 -k 8192 --block_a=1024,1024 --block_b=1024,1024 --block_c=1024,1024 --transpose=NN --type=double --algorithm=scalapack -r 1
#export MKL_NUM_THREADS=16; mpirun -np 2 ./../cosma/build/miniapp/pxgemm_miniapp -m 16384 -n 16384 -k 16384 --block_a=2048,2048 --block_b=2048,2048 --block_c=2048,2048 --transpose=NN --type=double --algorithm=scalapack -r 1

#slate
#export OMP_NUM_THREADS=16; mpirun -np 2 ./tester --type d --origin s --alpha 1 --beta 0 --nb 128 --dim 128x128x128 gemm
#export OMP_NUM_THREADS=16; mpirun -np 2 ./tester --type d --origin s --alpha 1 --beta 0 --nb 512 --dim 512x512x512 gemm
#export OMP_NUM_THREADS=16; mpirun -np 2 ./tester --type d --origin s --alpha 1 --beta 0 --nb 1024 --dim 1024x1024x1024 gemm
#export OMP_NUM_THREADS=16; mpirun -np 2 ./tester --type d --origin s --alpha 1 --beta 0 --nb 2048 --dim 2048x2048x2048 gemm
#export OMP_NUM_THREADS=16; mpirun -np 2 ./tester --type d --origin s --alpha 1 --beta 0 --nb 128 --dim 4096x4096x4096 gemm
#export OMP_NUM_THREADS=16; mpirun -np 2 ./tester --type d --origin s --alpha 1 --beta 0 --nb 1024 --dim 8192x8192x8192 gemm
#export OMP_NUM_THREADS=16; mpirun -np 2 ./tester --type d --origin s --alpha 1 --beta 0 --nb 2048 --dim 16384x16384x16384 gemm