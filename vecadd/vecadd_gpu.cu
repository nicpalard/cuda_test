#include "vecadd.h"
#include <iostream>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
	 std::cerr << "GPUassert: " << cudaGetErrorString(code) << " " << file << " " << line << std::endl;
      if (abort) exit(code);
   }
}


__global__ void kernel_vecadd(int* vec1, int* vec2, int* result, int N)
{
  int idx = blockIdx.x;
  result[idx] = vec1[idx] + vec2[idx];
}


void gpu_vecadd(int* vec1, int* vec2, int* result, int N)
{

  int* d_vec1, *d_vec2, *d_result;
  cudaMalloc((void**) &d_vec1, sizeof(int) * N);
  cudaMalloc((void**) &d_vec2, sizeof(int) * N);
  cudaMalloc((void**) &d_result, sizeof(int) * N);

  cudaMemcpy(d_vec1, vec1, sizeof(int) * N, cudaMemcpyHostToDevice);
  cudaMemcpy(d_vec2, vec2, sizeof(int) * N, cudaMemcpyHostToDevice);
  
  kernel_vecadd<<<N, 1>>>(d_vec1, d_vec2, d_result, N);

  cudaMemcpy(result, d_result, sizeof(int) * N, cudaMemcpyDeviceToHost);

  cudaFree(d_vec1);
  cudaFree(d_vec2);
  cudaFree(d_result);
}
