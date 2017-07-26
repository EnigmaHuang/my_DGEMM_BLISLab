#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include "sys/time.h"
#include "time.h"
#include "cblas.h"

#include "my_dgemm_kernels.h"
#include "BLISLab_dgemm_kernel.h"

double getVectorLinfNorm(double *a, double *b, const int length)
{
	register double res = 0.0;
	for (int i = 0; i < length; i++)
	{
		register double diff = fabs(a[i] - b[i]);
		if (diff > res) res = diff;
	}
	return res;
}

int main(int argc, char **argv)
{
	if (argc < 6) 
	{
		printf("Usage: ./dgemm_test <M> <N> <K> <repeats> <test_kernel>");
		return 0;
	}
	int m, n, k, repeats, test_kernel;
	m = atoi(argv[1]);
	n = atoi(argv[2]);
	k = atoi(argv[3]);
	repeats = atoi(argv[4]);
	test_kernel = atoi(argv[5]);
	printf("Matrix A : %4d rows * %4d cols\n", m, k);
	printf("Matrix B : %4d rows * %4d cols\n", k, n);
	printf("Matrix C : %4d rows * %4d cols\n", m, n);
	printf("Repeat %d times to evaluate single thread performance.\n", repeats);
	
	int sizeofa = m * k;
	int sizeofb = k * n;
	int sizeofc = m * n;
	double* A = (double*)malloc(sizeof(double) * sizeofa);
	double* B = (double*)malloc(sizeof(double) * sizeofb);
	double* C_ref = (double*)malloc(sizeof(double) * sizeofc);
	double* C_new = (double*)malloc(sizeof(double) * sizeofc);
	assert(A != NULL && B != NULL && C_ref != NULL && C_new != NULL);
	
	for (int i = 0; i < sizeofa; i++) A[i] = i % 233 + 1;
	for (int i = 0; i < sizeofb; i++) B[i] = i % 233 + 1;
	
	struct timeval start, finish;
	double ref_time = 0.0, new_time = 0.0, elapsed_time, gflops;
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1.0, A, k, B, n, 0.0, C_ref, n); // warm up
	for (int i = 0; i < repeats; i++)
	{
		gettimeofday(&start, NULL);
		cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1.0, A, k, B, n, 0.0, C_ref, n);
		gettimeofday(&finish, NULL);
		elapsed_time = ((double)(finish.tv_sec - start.tv_sec) * 1000000. + (double)(finish.tv_usec - start.tv_usec)) / 1000000.;
		ref_time += elapsed_time;
		gflops = 2.0 * m;
		gflops *= n;
		gflops *= k;
		gflops = gflops / elapsed_time * 1.0e-9;
		printf("OpenBLAS #%2d test: %lf GFlops\n", i, gflops);
	}
	elapsed_time = ref_time / (double)(repeats);
	gflops = 2.0 * m;
	gflops *= n;
	gflops *= k;
	gflops = gflops / elapsed_time * 1.0e-9;
	printf("OpenBLAS average performance = %lf GFlops\n", gflops);
	
	switch (test_kernel) // warm up
	{
		case 1: { my_dgemm_1(m, n, k, A, k, B, n, C_new, n); break; }
		case 2: { my_dgemm_2(m, n, k, A, k, B, n, C_new, n); break; }
		case 3: { my_dgemm_3(m, n, k, A, k, B, n, C_new, n); break; }
		case 4: { my_dgemm_4(m, n, k, A, k, B, n, C_new, n); break; }
		case 5: { my_dgemm_5(m, n, k, A, k, B, n, C_new, n); break; }
		case 6: { blislab_dgemm_kernel1(m, n, k, A, k, B, n, C_new, n); break; }
		case 7: { blislab_dgemm_kernel2(m, n, k, A, k, B, n, C_new, n); break; }
	}
	for (int i = 0; i < repeats; i++)
	{
		memset(C_new, 0, sizeof(double) * sizeofc);
		gettimeofday(&start, NULL);
		switch (test_kernel) // warm up
		{
			case 1: { my_dgemm_1(m, n, k, A, k, B, n, C_new, n); break; }
			case 2: { my_dgemm_2(m, n, k, A, k, B, n, C_new, n); break; }
			case 3: { my_dgemm_3(m, n, k, A, k, B, n, C_new, n); break; }
			case 4: { my_dgemm_4(m, n, k, A, k, B, n, C_new, n); break; }
			case 5: { my_dgemm_5(m, n, k, A, k, B, n, C_new, n); break; }
			case 6: { blislab_dgemm_kernel1(m, n, k, A, k, B, n, C_new, n); break; }
			case 7: { blislab_dgemm_kernel2(m, n, k, A, k, B, n, C_new, n); break; }
		}
		gettimeofday(&finish, NULL);
		elapsed_time = ((double)(finish.tv_sec - start.tv_sec) * 1000000. + (double)(finish.tv_usec - start.tv_usec)) / 1000000.;
		new_time += elapsed_time;
		gflops = 2.0 * m;
		gflops *= n;
		gflops *= k;
		gflops = gflops / elapsed_time * 1.0e-9;
		printf("My DGEMM #%2d test: %lf GFlops\n", i, gflops);
	}
	elapsed_time = new_time / (double)(repeats);
	gflops = 2.0 * m;
	gflops *= n;
	gflops *= k;
	gflops = gflops / elapsed_time * 1.0e-9;
	printf("My DGEMM average performance = %lf GFlops\n", gflops);
	
	double diff = getVectorLinfNorm(C_ref, C_new, sizeofc);
	printf("Max diff = %e \n", diff);
	
	free(A);
	free(B);
	free(C_ref);
	free(C_new);
	
	return 0;
}