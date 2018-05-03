EXEC = conv
NVCC = nvcc
NVCCFLAGS = -std=c++11 -arch=sm_50

$(EXEC): conv_gpu.o main.o
	$(NVCC) $(NVCCFLAGS) *.o -o $@

bench: conv_gpu.o bench.o
	$(NVCC) $(NVCCFLAGS) *.o -o $@

main.o: main.cpp
	$(CC) -std=c++11 -o $@  -c -I. $<

bench.o : bench.cpp
	$(CC) -std=c++11 -o $@  -c -I. $<

conv_gpu.o: conv_gpu.cu
	$(NVCC) $(NVCCFLAGS) -I. $^ -dc

clean:
	rm -f *.o
	rm -f *~
	rm -f $(EXEC)
	rm -f bench
	rm -f *.ppm
