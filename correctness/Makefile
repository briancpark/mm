COMPILER_FLAGS_CBLAS = -g -fopenmp -lmkl_intel_ilp64 -lmkl_scalapack_lp64 -lmkl_core -lmkl_gnu_thread -lpthread -DMKL_ILP64 -lm -ldl
COMPILER_FLAGS_PBLAS = -lmkl_scalapack_lp64 -Wl,--no-as-needed -lmkl_intel_lp64 -lmkl_gnu_thread -lmkl_core -lmkl_blacs_intelmpi_lp64 -lmkl_blacs_intelmpi_ilp64 -lgomp -lpthread -lm -ldl

all: cblas_correctness

cblas_correctness: cblas_correctness.cpp
	g++ -L$(MKL_LIB_DIR) -I$(MKL_INCLUDE_DIR) cblas_correctness.cpp -o cblas_correctness $(COMPILER_FLAGS_CBLAS)

.PHONY : clean

clean: 
	rm cblas_correctness
