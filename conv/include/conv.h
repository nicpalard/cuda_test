#ifndef CONV_H
#define CONV_H

struct benchmark {
    float compute_time;
    float transfer_time;
    float total_time;
};

float* gpu_conv(float* , uint, uint, float*, uint, uint, struct benchmark&);

#endif
