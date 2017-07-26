#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <time.h>

// Under Windows 8.1 x64 and GCC 4.9.2, MM_4x4_sse needs eps >= 1e-11, 
// but such problem does not exists in Linux. 
#define eps 1e-11

#define nRepeats 4

struct TestResult
{
    double naive_time, my_time;
    double naive_GFlops, my_GFlops;
    double max_error;
    int check_passed;
};

int InitedRand = 0;

extern "C"
void NaiveDGEMM(int hA, int wA, int wB,
                double* A, int lda, 
                double* B, int ldb,
                double* C, int ldc);

extern "C"
void MyDGEMM(int hA, int wA, int wB,
             double* A, int lda, 
             double* B, int ldb,
             double* C, int ldc);

extern "C"
char* VersionStr();

void RandomMatrix(double* m, int mSize)
{
    if (!InitedRand)
    {
        InitedRand = 1;
        srand(time(NULL));
    }
    int i;
    for (i = 0; i < mSize; i++)
        m[i] = 2.0 * rand() / (double)2100000000 - 1.0;
}

double CompareMatrices(double* ref, double* dst, int mSize)
{
    int i;
    double res = 0.0;
    for (i = 0; i < mSize; i++)
        if (fabs(ref[i] - dst[i]) > res)
            res = fabs(ref[i] - dst[i]);
    return res;
}
