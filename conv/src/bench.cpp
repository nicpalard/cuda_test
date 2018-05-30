#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <iomanip>
#include <ctype.h>

#include "conv.h"
#include "ImageUtils.hpp"

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

    std::fstream csvfile;
    csvfile.open("bench.csv", std::fstream::in | std::fstream::out | std::fstream::app);
    if (!csvfile.is_open())
    {
        std::cerr << "Could not opencv bench.csv, benchmark results are not going to be saved" << std::endl;
    }
  
    float* mask = create_gaussian_kernel(15, c_width * c_height);

    std::cout << "** Starting benchmark **" << std::endl;
    std::cout << "Gaussian blur with a kernel size of " << c_width << " x " << c_height << " on a single channel image" << std::endl;
    std::cout << "---------------------------------------------" << std::endl;
    std::cout << "Size\t\tSize (MB)\tCTime (ms)\tTTime (ms)\tTotal (ms)\tBandwidth (MB/s)" << std::endl;
    std::cout << std::fixed << std::setprecision(3) << std::setfill('0');

    if (csvfile.is_open())
        csvfile << "Size,Size (MB),Compute Time (ms),Transfer Time (ms),Total Time (ms),Bandwidth (MB/s)" << std::endl;

    struct benchmark benchmark;
    for (long int N = 128 ; N <= 8192 ; N+= N) {
        float* f_data = generate_random_image(N, N, 1);
        double compute_time = 0, transfer_time = 0, total_time = 0;
        unsigned int iterations = 5;
        for (int i = 0 ; i < iterations ; ++i)
        {
            float* out_image = gpu_conv(f_data, N, N, mask, c_width, c_height, benchmark);
            compute_time += benchmark.compute_time;
            transfer_time += benchmark.transfer_time;
            total_time += benchmark.total_time;
        }
        double realsize = (N * N * sizeof(float)) / 1e6;
        compute_time /= (double)iterations;
        transfer_time /= (double)iterations;
        total_time /= (double)iterations;
        double bandwidth = realsize / (total_time / 1e3);
        std::cout << N << " x " << N << "\t"
                << (N * N * sizeof(float)) / 1e6 << "\t\t"
                << compute_time << "\t\t"
                << transfer_time << "\t\t"
                << total_time << "\t\t"
                << (N * N * sizeof(float)) / 1e6 / (total_time / 1e3)
                << std::endl;

        if (csvfile.is_open())
        {
            csvfile << N << "x" << N << ","
                    << realsize << ","
                    << compute_time << ","
                    << transfer_time << ","
                    << total_time << ","
                    << bandwidth << std::endl;
        }
    }

    if (csvfile.is_open())
        csvfile.close();

    return EXIT_SUCCESS;
}
