#ifndef _BLISLAB_DGEMM_H_
#define _BLISLAB_DGEMM_H_

#define DGEMM_MC 128
#define DGEMM_NC 512
#define DGEMM_KC 512
#define DGEMM_MR 8
#define DGEMM_NR 4

#include <x86intrin.h>

typedef union
{
	__m256d v;
	double d[4];
}v4df_t;

inline int MIN(const int a, const int b)
{
	if (a < b) return a; else return b;
}

void blislab_dgemm_kernel1(MY_DGEMM_ARGS);
void blislab_dgemm_kernel2(MY_DGEMM_ARGS);

#endif
