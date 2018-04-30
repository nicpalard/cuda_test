#include <iostream>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

__global__ void gpu_vecadd(int* a, int* b, int* c) {
    c[blockIdx.x] = a[blockIdx.x] + b[blockIdx.x];
}

void cpu_vecadd(int* a, int*b, int* c, int length)
{
    for(int i = 0 ; i < length ; ++i)
    {
        c[i] = a[i] + b[i];
    }
}

void random_ints(int* a, int N)
{
    for(int i = 0 ; i < N ; i++)
    {
        a[i] = rand() %10;
    }
}

void printVector(int* vector, int length)
{
    std::cout << "{" << " ";
    for (int i = 0; i < length; i++)
    {
        std::cout << vector[i] << " ";
    }
    std::cout << "}" << std::endl;
}

#define N 5
int main(void) {
    
    int *a, *b, *c_cpu, *c_gpu; // host copies of a, b, c
    int *d_a, *d_b, *d_c; // device copies of a, b, c
    int size = N * sizeof(int);
    
    // Alloc space for device copies of a, b, c
    gpuErrchk( cudaMalloc((void **)&d_a, size) );
    gpuErrchk( cudaMalloc((void **)&d_b, size) );
    gpuErrchk( cudaMalloc((void **)&d_c, size) );
    
    // Alloc space for host copies of a, b, c and setup input values
    a       = new int[N]; random_ints(a, N);
    b       = new int[N]; random_ints(b, N);
    c_cpu   = new int[N];
    c_gpu   = new int[N];
    
    // Copy inputs to device
    gpuErrchk( cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice) );
    
    // Launch add() kernel on GPU with N blocks
    gpu_vecadd<<<N,1>>>(d_a, d_b, d_c);
    // Copy result back to host
    gpuErrchk( cudaMemcpy(c_gpu, d_c, size, cudaMemcpyDeviceToHost) );

    cpu_vecadd(a, b, c_cpu, N);

    // Test results
    printVector(a, N);
    printVector(b, N);
    printVector(c_cpu, N);
    printVector(c_gpu, N);
    
    // Cleanup
    delete a; 
    delete b; 
    delete c_cpu;
    delete c_gpu;
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);

    return 0;
}