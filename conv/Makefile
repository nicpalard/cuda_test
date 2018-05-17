EXEC = conv
NVCC = nvcc
NVCCFLAGS = -std=c++11 -arch=sm_50

all: $(EXEC) bench

$(EXEC): conv_gpu.o main.o
	$(NVCC) $(NVCCFLAGS) $^ -o $@

bench: conv_gpu.o bench.o
	$(NVCC) $(NVCCFLAGS) $^ -o $@

main.o: src/main.cpp
	$(CC) -std=c++11 -o $@  -c -Iinclude/ $<

bench.o: src/bench.cpp
	$(CC) -std=c++11 -o $@  -c -Iinclude/ $<

conv_gpu.o: src/conv_gpu.cu
	$(NVCC) $(NVCCFLAGS) -Iinclude $^ -dc

clean:
	rm -f *.o
	rm -f *~
	rm -f $(EXEC)
	rm -f bench
	rm -f *.ppm
