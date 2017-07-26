#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdlib.h>

#include "utils.c"
#include "dclock.c"

#ifdef WINDOWS_DEVCPP
#include "Ref.cc"
#include "MMult_4x4_15.cc"
#endif

void TestMM(int heightA, int widthA, int heightB, int widthB, TestResult &res)
{
    int heightC = heightA;
    int widthC = widthB;
    
    int dsize = sizeof(double);
    double *A = (double*) malloc(dsize * heightA * widthA);
    double *B = (double*) malloc(dsize * heightB * widthB);
    double *ref = (double*) malloc(dsize * heightC * widthC);
    double *dst = (double*) malloc(dsize * heightC * widthC);
    
    memset(ref, 0, dsize * heightC * widthC);
    memset(dst, 0, dsize * heightC * widthC);
    res.my_time = 9999999.9;
    
    RandomMatrix(A, heightA * widthA);
    RandomMatrix(B, heightB * widthB);
    
    double st, et;
    
    res.naive_GFlops = res.my_GFlops = 1e-9 * 2.0 * (double)widthC * (double)heightC * (double)widthA;

    st = dclock();
    NaiveDGEMM(heightA, widthA, widthB,
               A, heightA, B, heightB, ref, heightC);
    et = dclock();
    res.naive_time = et - st;
    res.naive_GFlops /= res.naive_time;
    
    for (int i = 0; i < nRepeats; i++)
    {
        memset(dst, 0, dsize * heightC * widthC);
        st = dclock();
        MyDGEMM(heightA, widthA, widthB,
                A, heightA, B, heightB, dst, heightC);
        et = dclock();
        if ((et - st) < res.my_time) res.my_time = et - st;
    }
    res.my_GFlops /= res.my_time;
    
    res.max_error = CompareMatrices(ref, dst, heightC * widthC);
    
    if (res.max_error <= eps) res.check_passed = 1;
    else res.check_passed = 0;
    
    free(A);
    free(B);
    free(ref);
    free(dst);
}

int main()
{
    TestResult res;
    int nStart, nEnd, nStep;
    
    FILE * cf = fopen("config.txt", "r");
    
    fscanf(cf, "nStart=%d\n", &nStart);
    fscanf(cf, "nEnd=%d\n", &nEnd);
    fscanf(cf, "nStep=%d\n", &nStep);
    
    fclose(cf);
    
    fprintf(stderr, "Current algorithm version : %s\n", VersionStr());
    fprintf(stderr, "N size from %d to %d with step %d\n", nStart, nEnd, nStep);
    
    printf("version = \'%s\' ;\n", VersionStr());
    printf("MY_MMult = [\n");
    
    for (int n = nStart; n <= nEnd; n += nStep)
    {
        TestMM(n, n, n, n, res);
        if (res.check_passed)
        {
            fprintf(stderr, "n size = %d\t, PASSED\n", n);
            printf("%d %le %le\n", n, res.my_GFlops, res.max_error);
        }
        else
        {
            fprintf(stderr, "n size = %d\t, FAILED\n", n);
            printf("%d %le %le\n", n, 0.0, res.max_error);
        }
    }
    
    printf("];\n");
    
    return 0;
}
