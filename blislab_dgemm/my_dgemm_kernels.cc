#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <x86intrin.h>
#include "my_dgemm_kernels.h"

#define PAD_UNIT 8
#define C_TILE_ROW 8
#define C_TILE_COL 4

void my_dgemm_1(MY_DGEMM_ARGS)
{
	for (int i = 0; i < c_rows; i++)
		for (int j = 0; j < c_cols; j++)
		{
			double res = 0.0;
			for (int k = 0; k < comm_dim; k++) 
				res += A[i * lda + k] * B[k * ldb + j];
			C[i * ldc + j] = res;
		}
}

void my_dgemm_2(MY_DGEMM_ARGS)
{
	for (int i = 0; i < c_rows; i++)
		for (int k = 0; k < comm_dim; k++) 
		{
			double A_ik = A[i * lda + k];
			double *C_ptr = C + i * ldc;
			double *B_ptr = B + k * ldb;
			for (int j = 0; j < c_cols; j++)
				C_ptr[j] += A_ik * B_ptr[j];
		}
}

void pad_gemm_matrixs(
	MY_DGEMM_ARGS,
	int &pad_c_rows, int &pad_c_cols, int &pad_comm_dim,
	double *&pad_A, int &pad_lda,
	double *&pad_B, int &pad_ldb,
	double *&pad_C, int &pad_ldc
)
{
	pad_c_rows = (c_rows + PAD_UNIT - 1) / PAD_UNIT * PAD_UNIT;
	pad_c_cols = (c_cols + PAD_UNIT - 1) / PAD_UNIT * PAD_UNIT;
	pad_comm_dim = (comm_dim + PAD_UNIT - 1) / PAD_UNIT * PAD_UNIT;
	
	pad_A = (double*) _mm_malloc(sizeof(double) * pad_c_rows * pad_comm_dim, 256);
	pad_B = (double*) _mm_malloc(sizeof(double) * pad_comm_dim * pad_c_cols, 256);
	pad_C = (double*) _mm_malloc(sizeof(double) * pad_c_rows * pad_c_cols, 256);
	assert(pad_A != NULL && pad_B != NULL && pad_C != NULL);
	
	memset(pad_A, 0, sizeof(double) * pad_c_rows * pad_comm_dim);
	memset(pad_B, 0, sizeof(double) * pad_comm_dim * pad_c_cols);
	memset(pad_C, 0, sizeof(double) * pad_c_rows * pad_c_cols);
	
	for (int i = 0; i < c_rows; i++) 
		memcpy(pad_A + pad_comm_dim * i, A + lda * i, sizeof(double) * comm_dim);
	
	for (int i = 0; i < comm_dim; i++)
		memcpy(pad_B + pad_c_cols * i, B + ldb * i, sizeof(double) * c_cols);

	pad_lda = pad_comm_dim;
	pad_ldb = pad_c_cols;
	pad_ldc = pad_c_cols;
}

void unpad_gemm_matrixs(
	const int &c_rows, const int &c_cols, 
	double *C, const int &ldc,
	double *&pad_A, double *&pad_B, 
	double *&pad_C, const int &pad_ldc
)
{
	for (int i = 0; i < c_rows; i++)
		memcpy(C + ldc * i, pad_C + pad_ldc * i, sizeof(double) * c_cols);
	
	free(pad_A);
	free(pad_B);
	free(pad_C);
}

void my_dgemm_3(MY_DGEMM_ARGS)
{
	int pad_c_rows, pad_c_cols, pad_comm_dim;
	int pad_lda, pad_ldb, pad_ldc;
	double *pad_A, *pad_B, *pad_C;
	
	pad_gemm_matrixs(
		c_rows, c_cols, comm_dim, A, lda, B, ldb, C, ldc,
		pad_c_rows, pad_c_cols, pad_comm_dim, pad_A, pad_lda, pad_B, pad_ldb, pad_C, pad_ldc
	);
	
	my_dgemm_2(pad_c_rows, pad_c_cols, pad_comm_dim, pad_A, pad_lda, pad_B, pad_ldb, pad_C, pad_ldc);
	
	unpad_gemm_matrixs(c_rows, c_cols, C, ldc, pad_A, pad_B, pad_C, pad_ldc);
}

void compute4x4TileKernel(
	double *A, const int &lda, double *B, const int &ldb,
	const int &comm_dim, double *C, const int &ldc
)
{
	register double C00, C01, C02, C03, C10, C11, C12, C13,
					C20, C21, C22, C23, C30, C31, C32, C33,
					/*B00, B01, B02, B03,*/ A00, A10, A20, A30;
	C00 = 0.0; C01 = 0.0; C02 = 0.0; C03 = 0.0;
	C10 = 0.0; C11 = 0.0; C12 = 0.0; C13 = 0.0;
	C20 = 0.0; C21 = 0.0; C22 = 0.0; C23 = 0.0;
	C30 = 0.0; C31 = 0.0; C32 = 0.0; C33 = 0.0;
	
	for (int i = 0; i < comm_dim; i++)
	{
		/*
		B00 = B[i * ldb];
		B01 = B[i * ldb + 1];
		B02 = B[i * ldb + 2];
		B03 = B[i * ldb + 3];
		*/
		
		A00 = A[i];
		A10 = A[lda + i];
		A20 = A[2 * lda + i];
		A30 = A[3 * lda + i];
		
		double *B_ptr = B + i * ldb;
		
		C00 += A00 * B_ptr[0];
		C01 += A00 * B_ptr[1];
		C02 += A00 * B_ptr[2];
		C03 += A00 * B_ptr[3];
		
		C10 += A10 * B_ptr[0];
		C11 += A10 * B_ptr[1];
		C12 += A10 * B_ptr[2];
		C13 += A10 * B_ptr[3];
		
		C20 += A20 * B_ptr[0];
		C21 += A20 * B_ptr[1];
		C22 += A20 * B_ptr[2];
		C23 += A20 * B_ptr[3];
		
		C30 += A30 * B_ptr[0];
		C31 += A30 * B_ptr[1];
		C32 += A30 * B_ptr[2];
		C33 += A30 * B_ptr[3];
	}
	
	double *C_ptr = C;
	C_ptr[0] += C00; C_ptr[1] += C01; C_ptr[2] += C02; C_ptr[3] += C03;
	C_ptr = C + ldc;
	C_ptr[0] += C10; C_ptr[1] += C11; C_ptr[2] += C12; C_ptr[3] += C13;
	C_ptr = C + 2 * ldc;
	C_ptr[0] += C20; C_ptr[1] += C21; C_ptr[2] += C22; C_ptr[3] += C23;
	C_ptr = C + 3 * ldc;
	C_ptr[0] += C30; C_ptr[1] += C31; C_ptr[2] += C32; C_ptr[3] += C33;
}

void my_dgemm_4(MY_DGEMM_ARGS)
{
	int pad_c_rows, pad_c_cols, pad_comm_dim;
	int pad_lda, pad_ldb, pad_ldc;
	double *pad_A, *pad_B, *pad_C;
	
	pad_gemm_matrixs(
		c_rows, c_cols, comm_dim, A, lda, B, ldb, C, ldc,
		pad_c_rows, pad_c_cols, pad_comm_dim, pad_A, pad_lda, pad_B, pad_ldb, pad_C, pad_ldc
	);
	
	for (int i = 0; i < pad_c_rows; i += 4)
		for (int j = 0; j < pad_c_cols; j += 4)
			compute4x4TileKernel(
				pad_A + i * pad_lda, pad_lda, 
				pad_B + j, pad_ldb, pad_comm_dim, 
				pad_C + i * pad_ldc + j, pad_ldc
			);
	
	unpad_gemm_matrixs(c_rows, c_cols, C, ldc, pad_A, pad_B, pad_C, pad_ldc);
}

void InnerKernel5(MY_DGEMM_ARGS)
{
	for (int i = 0; i < c_rows; i += 4)
		for (int j = 0; j < c_cols; j += 4)
			compute4x4TileKernel(
				A + i * lda, lda, 
				B + j, ldb, comm_dim, 
				C + i * ldc + j, ldc
			);
}

#define MAX_BLOCK_ROW 128
#define MAX_BLOCK_COL 128

void my_dgemm_5(MY_DGEMM_ARGS)
{
	int pad_c_rows, pad_c_cols, pad_comm_dim;
	int pad_lda, pad_ldb, pad_ldc;
	double *pad_A, *pad_B, *pad_C;
	
	pad_gemm_matrixs(
		c_rows, c_cols, comm_dim, A, lda, B, ldb, C, ldc,
		pad_c_rows, pad_c_cols, pad_comm_dim, pad_A, pad_lda, pad_B, pad_ldb, pad_C, pad_ldc
	);
	
	for (int i = 0; i < pad_c_rows; i += MAX_BLOCK_ROW)
	{
		int block_c_row = MAX_BLOCK_ROW;
		if (pad_c_rows - i < block_c_row) block_c_row = pad_c_rows - i;
		for (int j = 0; j < pad_c_cols; j += MAX_BLOCK_COL)
		{
			int block_c_col = MAX_BLOCK_COL;
			if (pad_c_cols - j < block_c_col) block_c_col = pad_c_cols - j;
			InnerKernel5(
				block_c_row, block_c_col, pad_comm_dim,
				pad_A + i * pad_lda, pad_lda,
				pad_B + j, pad_ldb,
				pad_C + i * ldc + j, pad_ldc
			);
		}
	}
	
	unpad_gemm_matrixs(c_rows, c_cols, C, ldc, pad_A, pad_B, pad_C, pad_ldc);
}
