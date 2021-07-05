COMPILER_FLAGS_CBLAS = -g -fopenmp -lmkl_intel_ilp64 -lmkl_scalapack_lp64 -lmkl_core -lmkl_gnu_thread -lpthread -DMKL_ILP64 -lm -ldl
COMPILER_FLAGS_PBLAS = -g -lmkl_scalapack_ilp64 -Wl,--no-as-needed -lmkl_intel_ilp64 -lmkl_gnu_thread -lmkl_core -lmkl_blacs_intelmpi_ilp64 -lgomp -lpthread -lm -ldl

cblas: cblas.cpp
	g++ -L$(MKL_LIB_DIR) -I$(MKL_INCLUDE_DIR) cblas.cpp -o cblas $(COMPILER_FLAGS_CBLAS)

pblas: pblas.cpp
	mpicxx -L$(MKL_LIB_DIR) -I$(MKL_INCLUDE_DIR) pblas.cpp -o pblas $(COMPILER_FLAGS_PBLAS)

.PHONY : clean

clean: 
	rn cblas