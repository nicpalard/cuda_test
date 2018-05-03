#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

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

    // Normalize the kernel
    
    /*
    for (int i = 0; i < kernel_size * kernel_size; ++i)
            kernel[i] /= sum;
    */
    return kernel;
}

int main(int argc, char** argv)
{
  if (argc != 3)
  {
	return EXIT_FAILURE;
  }

  uint width;
  uint height;
  unsigned char* data = load_ppm(argv[1], width, height);
  unsigned char* data_gray = rgb_to_gray(data, width, height);
  float* f_data = uchar_to_float(data_gray, width * height);

  uint c_width = 5;
  uint c_height = 5;
  float* mask = create_gaussian_kernel(15, c_width * c_height);

  float* out_image = gpu_conv(f_data, width, height, mask, c_width, c_height, false);

  unsigned char* new_data_gray = float_to_uchar(out_image, width * height);
  unsigned char* data_color = gray_to_rgb(new_data_gray, width, height);
  write_ppm(argv[2], data_color, width, height);
}
