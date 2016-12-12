all: chp-night chp-laplace

CXX = g++
CFLAGS = -O3 -fopenmp 
CUDA_FLAGS = -O3 -Xcompiler -fopenmp -Xcompiler -mavx2  -arch=compute_35

image.o: image.cpp image.h
	@echo "CXX image.cpp"
	@$(CXX) $(CFLAGS) -c $< -o $@ $$(pkg-config --cflags MagickWand)

.PRECIOUS: night.cu
night.cu: night.lisp
	@echo "CM $<"
	@cm cu -i $< -o night.cu

.PRECIOUS: laplace.cu
laplace.cu: laplace.lisp
	@echo "CM $<"
	@cm cu -i $< -o laplace.cu

night.o: night.cu image.h
	@echo "NVCC night.cu"
	@nvcc $(CUDA_FLAGS) -ccbin $(CXX) -c night.cu -o $@

chp-night: night.o image.o
	@echo "LINK $^ -> $@"
	@$(CXX) $^ -o $@ -lcudart -fopenmp $$(pkg-config --libs MagickWand)

laplace.o: laplace.cu image.h
	@echo "NVCC lapla.cu"
	@nvcc $(CUDA_FLAGS) -ccbin $(CXX) -c laplace.cu -o $@

chp-laplace: laplace.o image.o
	@echo "LINK $^ -> $@"
	@$(CXX) $^ -o $@ -lcudart -fopenmp $$(pkg-config --libs MagickWand)

clean:
	rm -f chp-night chp-laplace night.o night.cu laplace.o laplace.cu image.o
	rm -f *.dot
