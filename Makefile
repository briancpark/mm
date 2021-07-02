COMPILER_FLAGS = -fopenmp -lmkl_intel_lp64 -lmkl_core -lmkl_gnu_thread -lpthread -lm -ldl

cblas: cblas.cpp
	g++ -L$(MKL_LIB_DIR) -I$(MKL_INCLUDE_DIR) cblas.cpp -o cblas $(COMPILER_FLAGS)

.PHONY : clean

clean: 
	rn cblas