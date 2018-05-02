EXEC = vecadd
NVCC = nvcc
NVCCFLAGS = -arch=sm_50

$(EXEC): vecadd_gpu.o main.o
	$(NVCC) $(NVCCFLAGS) *.o -o $@

main.o: main.cpp
	$(CC) -o $@  -c -I. $<

vecadd_gpu.o: vecadd_gpu.cu
	$(NVCC) $(NVCCFLAGS) -I. $^ -dc

clean:
	rm -f *.o
	rm -f *~
	rm $(EXEC)
