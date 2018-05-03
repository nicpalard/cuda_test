#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <iomanip>

#include "conv.h"
#include "conv_common.h"

float* create_gaussian_kernel(float sigma, size_t kernel_size)
{
    float* kernel = new float[kernel_size * kernel_size];
    float mean = kernel_size/2;
    float sum = 0.0;
    for (int i = 0; i < kernel_size * kernel_size; ++i) 
    {
        int x = i % kernel_size;
        int y = (i - x) / kernel_size % kernel_size;
        kernel[i] = exp( -0.5 * (pow((x-mean)/sigma, 2.0) + pow((y-mean)/sigma,2.0)) ) / (2 * M_PI * sigma * sigma);
        // Accumulate the kernel values
        sum += kernel[i];
    }
    return kernel;
}

int main(int argc, char** argv)
{

  std::cout << "** Starting benchmark **" << std::endl;
  std::cout << "Convolution using gaussian kernel of size 5x5" << std::endl;
  std::cout << "---------------------------------------------" << std::endl;
  std::cout << "Size\t\tSize (MB)\tTime (ms)\tBandwidth (MB/s)" << std::endl;
  std::cout << std::fixed << std::setprecision(3) << std::setfill('0');

  uint c_width = 5;
  uint c_height = 5;
  float* mask = create_gaussian_kernel(15, 20);

  for (long int N = 128 ; N <= 12000 ; N*= 1.2) {
	float* f_data = generate_random_image(N, N);
	float* out_image = gpu_conv(f_data, N, N, mask, c_width, c_height, true);
  }

  return EXIT_SUCCESS;
}
