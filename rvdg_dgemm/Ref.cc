#include <omp.h>

/* Create macros so that the matrices are stored in column-major order */

#define A(i, j) A[(j) * lda + (i)]
#define B(i, j) B[(j) * ldb + (i)]
#define C(i, j) C[(j) * ldc + (i)]

extern "C"
void NaiveDGEMM(int hA, int wA, int wB,
                double* A, int lda, 
                double* B, int ldb,
                double* C, int ldc)
{
    int i, j, k;
    #pragma omp parallel for private(j, k)
    for (i = 0; i < hA; i++)
        for (j = 0; j < wB; j++) 
            for (k = 0; k < wA; k++)
                C(i, j) += A(i, k) * B(k, j);
}