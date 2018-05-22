#include "conv.h"
#include "conv_common.h"
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


__global__ void simple_conv(float* in_data, float* out_data, uint width, uint height, float* mask, uint mask_width, uint mask_height)
{
  int idx_x = blockIdx.x * blockDim.x + threadIdx.x;
  int idx_y = blockIdx.y * blockDim.y + threadIdx.y;
  if (idx_x >= width || idx_y >= height)
	return;
  
  int mask_hw = mask_width/2;
  int mask_hh = mask_height/2;

  float sum = 0.0;
  float mask_sum = 0.0;
  for (uint x = 0 ; x < mask_width ; ++x)
  {
	for (uint y = 0 ; y < mask_height ; ++y)
	{
	  int px = idx_x + (x - mask_hw);
	  int py = idx_y + (y - mask_hh);

	  if (px < 0 || px >= width || py < 0 || py >= height)
		continue;

	  float m_value = mask[x + y * mask_width];
	  sum += m_value * in_data[px + py * width];
	  mask_sum += m_value;
	}
  }
  out_data[idx_x + idx_y * width] = sum / mask_sum;
}

float* gpu_conv(float* image, uint width, uint height, float* mask, uint mask_width, uint mask_height, struct benchmark& bench)
{
  uint size = width * height;
  float* out_image = new float[size];
  
  uint mask_size = mask_width * mask_height;


  cudaEvent_t total_start, transfer_start, compute_start, total_stop, transfer_stop, compute_stop;
  cudaEventCreate(&total_start); cudaEventCreate(&transfer_start); cudaEventCreate(&compute_start);
  cudaEventCreate(&total_stop); cudaEventCreate(&transfer_stop); cudaEventCreate(&compute_stop);
  
  cudaEventRecord(total_start);
  float *d_in, *d_out, *d_mask;
  gpuErrchk( cudaMalloc((void**) &d_in, size * sizeof(float)) );
  gpuErrchk( cudaMalloc((void**) &d_out, size * sizeof(float)) );
  gpuErrchk( cudaMalloc((void**) &d_mask, mask_size * sizeof(float)) );

  cudaEventRecord(transfer_start);
  gpuErrchk( cudaMemcpy(d_in, image, size * sizeof(float), cudaMemcpyHostToDevice) );
  gpuErrchk( cudaMemcpy(d_mask, mask, mask_size * sizeof(float), cudaMemcpyHostToDevice) );

  // Determining best threads per block & block number to run the kernel
  struct cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, 0);
  int maxThreadsPerBlock = deviceProp.maxThreadsPerBlock;
  float dim = sqrt((float)maxThreadsPerBlock);
  const dim3 blockDim(dim, dim);
  const dim3 numBlocks(width/dim, height/dim);

  cudaEventRecord(compute_start);
  simple_conv<<<numBlocks, blockDim>>>(d_in, d_out, width, height, d_mask, mask_width, mask_height);
  cudaEventRecord(compute_stop);

  gpuErrchk( cudaMemcpy(out_image, d_out, size * sizeof(float), cudaMemcpyDeviceToHost) );
  cudaEventRecord(transfer_stop);
  cudaEventRecord(total_stop);

  cudaEventSynchronize(total_stop);
  float compute_time, transfer_time, total_time;
  cudaEventElapsedTime(&compute_time, compute_start, compute_stop);
  cudaEventElapsedTime(&transfer_time, transfer_start, transfer_stop);
  cudaEventElapsedTime(&total_time, total_start, total_stop);

  bench.compute_time = compute_time;
  bench.transfer_time = transfer_time;
  bench.total_time = total_time;
  
  gpuErrchk( cudaFree(d_in) );
  gpuErrchk( cudaFree(d_out) );
  gpuErrchk( cudaFree(d_mask) );

  return out_image;
}
