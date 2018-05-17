#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <iomanip>
#include <ctype.h>

#include "conv.h"
#include "conv_common.h"

int main(int argc, char** argv)
{
  uint c_width = 5;
  uint c_height = 5;
  if (argc == 2) {
	if (isdigit(argv[1][0])) {
		c_width = atoi(argv[1]);
		c_height = c_width;
	}
	else if (std::string("-h").compare(argv[1]) == 0)
	{
	  std::cout << argv[0] << " <optional: kernel size>" << std::endl;
	  return 1;
	}
  }
  
  float* mask = create_gaussian_kernel(15, c_width * c_height);

  std::cout << "** Starting benchmark **" << std::endl;
  std::cout << "Gaussian blur with a kernel size of " << c_width << " x " << c_height << " on a single channel image" << std::endl;
  std::cout << "---------------------------------------------" << std::endl;
  std::cout << "Size\t\tSize (MB)\tTime (ms)\tBandwidth (MB/s)" << std::endl;
  std::cout << std::fixed << std::setprecision(3) << std::setfill('0');

  float exec_time = 0;
  for (long int N = 128 ; N <= 12000 ; N+= N) {
	float* f_data = generate_random_image(N, N);
	float* out_image = gpu_conv(f_data, N, N, mask, c_width, c_height, exec_time);
	std::cout << N << " x " << N << "\t" << (N * N * sizeof(float)) / 1e6 << "\t\t" << exec_time << "\t\t" << (N * N * sizeof(float)) / 1e6 / (exec_time / 1e3) << std::endl;
  }
  
  return EXIT_SUCCESS;
}
