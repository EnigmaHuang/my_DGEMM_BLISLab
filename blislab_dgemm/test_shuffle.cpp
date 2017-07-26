#include <stdio.h>
#include <x86intrin.h>

typedef union
{
	__m256d v;
	double d[4];
}v4df_t;

int main()
{
	v4df_t b0, b1;
	for (int i = 0; i < 4; i++) b0.d[i] = i;

	b1.v = _mm256_shuffle_pd(b0.v, b0.v, 0x5);
	for (int i = 0; i < 4; i++) printf("%lf ", b1.d[i]);
	printf("\n");

	b1.v = _mm256_permute2f128_pd(b1.v, b1.v, 0x1);
	for (int i = 0; i < 4; i++) printf("%lf ", b1.d[i]);
	printf("\n");

	b1.v = _mm256_shuffle_pd(b1.v, b1.v, 0x5);
	for (int i = 0; i < 4; i++) printf("%lf ", b1.d[i]);
	printf("\n");
}
