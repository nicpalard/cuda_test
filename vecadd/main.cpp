#include <iostream>
#include <stdlib.h>
#include <stdio.h>

#include "vecadd.h"
#include "vecadd_common.h"

int main(int argc, char** argv)
{
  int N = 10;
  int* a, *b, *c;
  a = new int[N]; random_ints(a, N);
  b = new int[N]; random_ints(b, N);
  c = new int[N];
  
  gpu_vecadd(a, b, c, N);

  print_vector(a, N);
  print_vector(b, N);
  print_vector(c, N);

  delete a;
  delete b;
  delete c;
}
