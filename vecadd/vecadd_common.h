#ifndef _VECADD_COMMON_H_
#define _VECADD_COMMON_H

#include <iostream>

inline void print_vector(int* vec, int N)
{
  for (int i = 0 ; i < N ; ++i)
  {
	std::cout << vec[i];
	if (i != N-1)
	  std::cout << ", ";
  }
  std::cout << std::endl;
}

inline void random_ints(int* vec, int N)
{
  for (int i = 0 ; i < N ; ++i)
  {
	vec[i] = rand() % 10;
  }
}

#endif
