#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "conv.h"
#include "ImageUtils.hpp"

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

  struct benchmark benchmark;
  float* out_image = gpu_conv(f_data, width, height, mask, c_width, c_height, benchmark);

  unsigned char* new_data_gray = float_to_uchar(out_image, width * height);
  unsigned char* data_color = gray_to_rgb(new_data_gray, width, height);
  write_ppm(argv[2], data_color, width, height);
}
