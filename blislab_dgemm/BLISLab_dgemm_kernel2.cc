#include <x86intrin.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <stdio.h>
#include "my_dgemm_kernels.h"
#include "BLISLab_dgemm_kernel.h"

#define DGEMM_MICRO_KERNEL_ARGS double *A_pack, const int &A_pack_rows, \
	double *B_pack, const int &B_pack_cols, \
	const int &comm_dim, double *C, const int &ldc \

void dgemm_micro_kernel_scale(DGEMM_MICRO_KERNEL_ARGS)
{
	for (int k = 0; k < comm_dim; k++) // 0-th loop, microkernel
	{
		double *A_ptr = A_pack + k * A_pack_rows;
		double *B_ptr = B_pack + k * B_pack_cols;
		
		for (int row = 0; row < A_pack_rows; row++)
			for (int col = 0; col < B_pack_cols; col++)
				C[row * ldc + col] += A_ptr[row] * B_ptr[col];
	}
}

void dgemm_micro_kernel_8x4(DGEMM_MICRO_KERNEL_ARGS)
{
	register double C00, C01, C02, C03, C10, C11, C12, C13,
					C20, C21, C22, C23, C30, C31, C32, C33,
					C40, C41, C42, C43, C50, C51, C52, C53,
					C60, C61, C62, C63, C70, C71, C72, C73;
	
	C00 = C01 = C02 = C03 = 0.0;
	C10 = C11 = C12 = C13 = 0.0;
	C20 = C21 = C22 = C23 = 0.0;
	C30 = C31 = C32 = C33 = 0.0;
	C40 = C41 = C42 = C43 = 0.0;
	C50 = C51 = C52 = C53 = 0.0;
	C60 = C61 = C62 = C63 = 0.0;
	C70 = C71 = C72 = C73 = 0.0;
	
	for (int k = 0; k < comm_dim; k++) // 0-th loop, microkernel
	{
		double *A_ptr = A_pack + k * A_pack_rows;
		double *B_ptr = B_pack + k * B_pack_cols;
		
		C00 += A_ptr[0] * B_ptr[0]; 
		C01 += A_ptr[0] * B_ptr[1]; 
		C02 += A_ptr[0] * B_ptr[2]; 
		C03 += A_ptr[0] * B_ptr[3]; 
		
		C10 += A_ptr[1] * B_ptr[0]; 
		C11 += A_ptr[1] * B_ptr[1]; 
		C12 += A_ptr[1] * B_ptr[2]; 
		C13 += A_ptr[1] * B_ptr[3]; 
		
		C20 += A_ptr[2] * B_ptr[0]; 
		C21 += A_ptr[2] * B_ptr[1]; 
		C22 += A_ptr[2] * B_ptr[2]; 
		C23 += A_ptr[2] * B_ptr[3]; 
		
		C30 += A_ptr[3] * B_ptr[0]; 
		C31 += A_ptr[3] * B_ptr[1]; 
		C32 += A_ptr[3] * B_ptr[2]; 
		C33 += A_ptr[3] * B_ptr[3]; 
		
		C40 += A_ptr[4] * B_ptr[0]; 
		C41 += A_ptr[4] * B_ptr[1]; 
		C42 += A_ptr[4] * B_ptr[2]; 
		C43 += A_ptr[4] * B_ptr[3]; 
		
		C50 += A_ptr[5] * B_ptr[0]; 
		C51 += A_ptr[5] * B_ptr[1]; 
		C52 += A_ptr[5] * B_ptr[2]; 
		C53 += A_ptr[5] * B_ptr[3]; 
		
		C60 += A_ptr[6] * B_ptr[0]; 
		C61 += A_ptr[6] * B_ptr[1]; 
		C62 += A_ptr[6] * B_ptr[2]; 
		C63 += A_ptr[6] * B_ptr[3]; 
		
		C70 += A_ptr[7] * B_ptr[0]; 
		C71 += A_ptr[7] * B_ptr[1]; 
		C72 += A_ptr[7] * B_ptr[2]; 
		C73 += A_ptr[7] * B_ptr[3]; 
	}
	
	C[0 * ldc + 0] += C00;  C[0 * ldc + 1] += C01;
	C[0 * ldc + 2] += C02;  C[0 * ldc + 3] += C03;
	
	C[1 * ldc + 0] += C10;  C[1 * ldc + 1] += C11;
	C[1 * ldc + 2] += C12;  C[1 * ldc + 3] += C13;
	
	C[2 * ldc + 0] += C20;  C[2 * ldc + 1] += C21;
	C[2 * ldc + 2] += C22;  C[2 * ldc + 3] += C23;
	
	C[3 * ldc + 0] += C30;  C[3 * ldc + 1] += C31;
	C[3 * ldc + 2] += C32;  C[3 * ldc + 3] += C33;
	
	C[4 * ldc + 0] += C40;  C[4 * ldc + 1] += C41;
	C[4 * ldc + 2] += C42;  C[4 * ldc + 3] += C43;
	
	C[5 * ldc + 0] += C50;  C[5 * ldc + 1] += C51;
	C[5 * ldc + 2] += C52;  C[5 * ldc + 3] += C53;
	
	C[6 * ldc + 0] += C60;  C[6 * ldc + 1] += C61;
	C[6 * ldc + 2] += C62;  C[6 * ldc + 3] += C63;
	
	C[7 * ldc + 0] += C70;  C[7 * ldc + 1] += C71;
	C[7 * ldc + 2] += C72;  C[7 * ldc + 3] += C73;
}

void dgemm_micro_kernel_8x4_avx_broadcast(DGEMM_MICRO_KERNEL_ARGS)
{
	register __m256d C0, C1, C2, C3, C4, C5, C6, C7;
	register __m256d B0, A0, A1, A2, A3;
	
	C0 = _mm256_load_pd(C + 0 * ldc);
	C1 = _mm256_load_pd(C + 1 * ldc);
	C2 = _mm256_load_pd(C + 2 * ldc);
	C3 = _mm256_load_pd(C + 3 * ldc);
	C4 = _mm256_load_pd(C + 4 * ldc);
	C5 = _mm256_load_pd(C + 5 * ldc);
	C6 = _mm256_load_pd(C + 6 * ldc);
	C7 = _mm256_load_pd(C + 7 * ldc);
	
	for (int k = 0; k < comm_dim; k++) // 0-th loop, microkernel
	{
		double *A_ptr = A_pack + k * A_pack_rows;
		B0  = _mm256_load_pd(B_pack + k * B_pack_cols);
		
		A0 = _mm256_broadcast_sd(A_ptr + 0);
		A1 = _mm256_broadcast_sd(A_ptr + 1);
		A2 = _mm256_broadcast_sd(A_ptr + 2);
		A3 = _mm256_broadcast_sd(A_ptr + 3);
		
		C0 += A0 * B0;
		C1 += A1 * B0;
		C2 += A2 * B0;
		C3 += A3 * B0;
		
		A0 = _mm256_broadcast_sd(A_ptr + 4);
		A1 = _mm256_broadcast_sd(A_ptr + 5);
		A2 = _mm256_broadcast_sd(A_ptr + 6);
		A3 = _mm256_broadcast_sd(A_ptr + 7);
		
		C4 += A0 * B0;
		C5 += A1 * B0;
		C6 += A2 * B0;
		C7 += A3 * B0;
	}
	
	_mm256_store_pd(C + 0 * ldc, C0);
	_mm256_store_pd(C + 1 * ldc, C1);
	_mm256_store_pd(C + 2 * ldc, C2);
	_mm256_store_pd(C + 3 * ldc, C3);
	_mm256_store_pd(C + 4 * ldc, C4);
	_mm256_store_pd(C + 5 * ldc, C5);
	_mm256_store_pd(C + 6 * ldc, C6);
	_mm256_store_pd(C + 7 * ldc, C7);
}

void dgemm_micro_kernel_8x4_avx_shuffle(DGEMM_MICRO_KERNEL_ARGS)
{
	register __m256d C0, C1, C2, C3, C4, C5, C6, C7;
	register __m256d B0, A0, A1, B1, A2, A3;
	
	v4df_t c0, c1, c2, c3, c4, c5, c6, c7;
	
	C0 = _mm256_setzero_pd();
	C1 = _mm256_setzero_pd();
	C2 = _mm256_setzero_pd();
	C3 = _mm256_setzero_pd();
	C4 = _mm256_setzero_pd();
	C5 = _mm256_setzero_pd();
	C6 = _mm256_setzero_pd();
	C7 = _mm256_setzero_pd();
	
	for (int k = 0; k < comm_dim; k += 2) // 0-th loop, microkernel
	{
		B0 = _mm256_load_pd(B_pack + k * B_pack_cols);
		A0 = _mm256_load_pd(A_pack + k * A_pack_rows);
		A1 = _mm256_load_pd(A_pack + k * A_pack_rows + 4);
		
		// prefetch
		B1 = _mm256_load_pd(B_pack + (k + 1) * B_pack_cols);
		A2 = _mm256_load_pd(A_pack + (k + 1) * A_pack_rows);
		A3 = _mm256_load_pd(A_pack + (k + 1) * A_pack_rows + 4);
		
		// k
		C0 += A0 * B0;
		C4 += A1 * B0;
		
		B0 = _mm256_shuffle_pd(B0, B0, 0x5);
		C1 += A0 * B0;
		C5 += A1 * B0;
		
		B0 = _mm256_permute2f128_pd(B0, B0, 0x1);
		C2 += A0 * B0;
		C6 += A1 * B0;
		
		B0 = _mm256_shuffle_pd(B0, B0, 0x5);
		C3 += A0 * B0;
		C7 += A1 * B0;
		
		// k + 1
		C0 += A2 * B1;
		C4 += A3 * B1;
		
		B1 = _mm256_shuffle_pd(B1, B1, 0x5);
		C1 += A2 * B1;
		C5 += A3 * B1;
		
		B1 = _mm256_permute2f128_pd(B1, B1, 0x1);
		C2 += A2 * B1;
		C6 += A3 * B1;
		
		B1 = _mm256_shuffle_pd(B1, B1, 0x5);
		C3 += A2 * B1;
		C7 += A3 * B1;
	}
	
	c0.v = C0; c1.v = C1; c2.v = C2; c3.v = C3;
	c4.v = C4; c5.v = C5; c6.v = C6; c7.v = C7;
	
	double *C_ptr = C;
	C_ptr[0] += c0.d[0];
	C_ptr[1] += c1.d[0];
	C_ptr[2] += c3.d[0];
	C_ptr[3] += c2.d[0];
	
	C_ptr = C + ldc;
	C_ptr[0] += c1.d[1];
	C_ptr[1] += c0.d[1];
	C_ptr[2] += c2.d[1];
	C_ptr[3] += c3.d[1];
	
	C_ptr = C + ldc * 2;
	C_ptr[0] += c3.d[2];
	C_ptr[1] += c2.d[2];
	C_ptr[2] += c0.d[2];
	C_ptr[3] += c1.d[2];
	
	C_ptr = C + ldc * 3;
	C_ptr[0] += c2.d[3];
	C_ptr[1] += c3.d[3];
	C_ptr[2] += c1.d[3];
	C_ptr[3] += c0.d[3];
	
	C_ptr = C + ldc * 4;
	C_ptr[0] += c4.d[0];
	C_ptr[1] += c5.d[0];
	C_ptr[2] += c7.d[0];
	C_ptr[3] += c6.d[0];
	
	C_ptr = C + ldc * 5;
	C_ptr[0] += c5.d[1];
	C_ptr[1] += c4.d[1];
	C_ptr[2] += c6.d[1];
	C_ptr[3] += c7.d[1];
	
	C_ptr = C + ldc * 6;
	C_ptr[0] += c7.d[2];
	C_ptr[1] += c6.d[2];
	C_ptr[2] += c4.d[2];
	C_ptr[3] += c5.d[2];
	
	C_ptr = C + ldc * 7;
	C_ptr[0] += c6.d[3];
	C_ptr[1] += c7.d[3];
	C_ptr[2] += c5.d[3];
	C_ptr[3] += c4.d[3];
}

void dgemm_micro_kernel_8x4_avx_shuffle_prefetch(DGEMM_MICRO_KERNEL_ARGS)
{
	register __m256d C0, C1, C2, C3, C4, C5, C6, C7;
	register __m256d B0, A0, A1, B1, A2, A3;
	
	v4df_t c0, c1, c2, c3, c4, c5, c6, c7;
	
	C0 = _mm256_setzero_pd();
	C1 = _mm256_setzero_pd();
	C2 = _mm256_setzero_pd();
	C3 = _mm256_setzero_pd();
	C4 = _mm256_setzero_pd();
	C5 = _mm256_setzero_pd();
	C6 = _mm256_setzero_pd();
	C7 = _mm256_setzero_pd();
	
	B0 = _mm256_load_pd(B_pack);
	A0 = _mm256_load_pd(A_pack);
	A1 = _mm256_load_pd(A_pack + 4);
	
	for (int k = 0; k < comm_dim; k += 2) // 0-th loop, microkernel
	{
		// prefetch for k + 1
		A2 = _mm256_load_pd(A_pack + (k + 1) * A_pack_rows);
		
		// k
		C0 += A0 * B0;
		C4 += A1 * B0;
		
		// prefetch for k + 1
		A3 = _mm256_load_pd(A_pack + (k + 1) * A_pack_rows + 4);
		
		B0 = _mm256_shuffle_pd(B0, B0, 0x5);
		C1 += A0 * B0;
		C5 += A1 * B0;
		
		// prefetch for k + 1
		B1 = _mm256_load_pd(B_pack + (k + 1) * B_pack_cols);
		
		B0 = _mm256_permute2f128_pd(B0, B0, 0x1);
		C2 += A0 * B0;
		C6 += A1 * B0;
		
		B0 = _mm256_shuffle_pd(B0, B0, 0x5);
		C3 += A0 * B0;
		C7 += A1 * B0;
		
		// k + 1
		
		// prefetch for next k
		A0 = _mm256_load_pd(A_pack + (k + 2) * A_pack_rows);
		
		C0 += A2 * B1;
		C4 += A3 * B1;
		
		B1 = _mm256_shuffle_pd(B1, B1, 0x5);
		C1 += A2 * B1;
		C5 += A3 * B1;
		
		// prefetch for next k
		A1 = _mm256_load_pd(A_pack + (k + 2) * A_pack_rows + 4);
		
		B1 = _mm256_permute2f128_pd(B1, B1, 0x1);
		C2 += A2 * B1;
		C6 += A3 * B1;
		
		// prefetch for next k
		B0 = _mm256_load_pd(B_pack + (k + 2) * B_pack_cols);
		
		B1 = _mm256_shuffle_pd(B1, B1, 0x5);
		C3 += A2 * B1;
		C7 += A3 * B1;
	}
	
	c0.v = C0; c1.v = C1; c2.v = C2; c3.v = C3;
	c4.v = C4; c5.v = C5; c6.v = C6; c7.v = C7;
	
	double *C_ptr = C;
	C_ptr[0] += c0.d[0];
	C_ptr[1] += c1.d[0];
	C_ptr[2] += c3.d[0];
	C_ptr[3] += c2.d[0];
	
	C_ptr = C + ldc;
	C_ptr[0] += c1.d[1];
	C_ptr[1] += c0.d[1];
	C_ptr[2] += c2.d[1];
	C_ptr[3] += c3.d[1];
	
	C_ptr = C + ldc * 2;
	C_ptr[0] += c3.d[2];
	C_ptr[1] += c2.d[2];
	C_ptr[2] += c0.d[2];
	C_ptr[3] += c1.d[2];
	
	C_ptr = C + ldc * 3;
	C_ptr[0] += c2.d[3];
	C_ptr[1] += c3.d[3];
	C_ptr[2] += c1.d[3];
	C_ptr[3] += c0.d[3];
	
	C_ptr = C + ldc * 4;
	C_ptr[0] += c4.d[0];
	C_ptr[1] += c5.d[0];
	C_ptr[2] += c7.d[0];
	C_ptr[3] += c6.d[0];
	
	C_ptr = C + ldc * 5;
	C_ptr[0] += c5.d[1];
	C_ptr[1] += c4.d[1];
	C_ptr[2] += c6.d[1];
	C_ptr[3] += c7.d[1];
	
	C_ptr = C + ldc * 6;
	C_ptr[0] += c7.d[2];
	C_ptr[1] += c6.d[2];
	C_ptr[2] += c4.d[2];
	C_ptr[3] += c5.d[2];
	
	C_ptr = C + ldc * 7;
	C_ptr[0] += c6.d[3];
	C_ptr[1] += c7.d[3];
	C_ptr[2] += c5.d[3];
	C_ptr[3] += c4.d[3];
}

void dgemm_marco_kernel2(
	const int &block_c_rows, const int &block_c_cols, const int &block_comm_dim,
	double *packed_A, double *packed_B, double *C, const int &ldc
)
{
	for (int j = 0; j < block_c_cols; j += DGEMM_NR)  // 2-nd loop
	{
		int curr_B_pack_cols = MIN(DGEMM_NR, block_c_cols - j);
		double *curr_B_pack = packed_B + j * block_comm_dim;
		
		for (int i = 0; i < block_c_rows; i += DGEMM_MR) // 1-st loop
		{
			int curr_A_pack_rows = MIN(DGEMM_MR, block_c_rows - i);
			double *curr_A_pack = packed_A + i * block_comm_dim;

			dgemm_micro_kernel_8x4_avx_shuffle_prefetch(
				curr_A_pack, curr_A_pack_rows,
				curr_B_pack, curr_B_pack_cols,
				block_comm_dim, C + i * ldc + j, ldc
			);
		}
	}
}

void packB_KCxNC2(
	double *B, const int &ldb, double *packed_B,
	const int &pb_rows, const int &pb_cols
)
{
	for (int iCol = 0; iCol < pb_cols; iCol += DGEMM_NR)
	{
		int copy_elem = MIN(DGEMM_NR, pb_cols - iCol);
		double *curr_B_pack = packed_B + iCol * pb_rows;
		
		for (int iRow = 0; iRow < pb_rows; iRow++)
			memcpy(
				curr_B_pack + iRow * copy_elem,
				B + iRow * ldb + iCol,
				sizeof(double) * copy_elem
			);
	}
}

void packA_MCxKC2(
	double *A, const int &lda, double *packed_A, 
	const int &pa_rows, const int &pa_cols
)
{
	for (int iRow = 0; iRow < pa_rows; iRow += DGEMM_MR)
	{
		int copy_elem = MIN(DGEMM_MR, pa_rows - iRow);
		double *curr_A_pack = packed_A + iRow * pa_cols;
		
		for (int iCol = 0; iCol < pa_cols; iCol++)
		{
			double *pA_ptr = curr_A_pack + iCol * copy_elem;
			double *A_ptr = A + iRow * lda + iCol;
			
			for (int iCopy = 0; iCopy < copy_elem; iCopy++)
				pA_ptr[iCopy] = A_ptr[iCopy * lda];
		}
	}
}

void blislab_dgemm_kernel2(MY_DGEMM_ARGS)
{
	int pad_c_rows, pad_c_cols, pad_comm_dim;
	int pad_lda, pad_ldb, pad_ldc;
	double *pad_A, *pad_B, *pad_C;
	
	pad_gemm_matrixs(
		c_rows, c_cols, comm_dim, A, lda, B, ldb, C, ldc,
		pad_c_rows, pad_c_cols, pad_comm_dim, pad_A, pad_lda, pad_B, pad_ldb, pad_C, pad_ldc
	);
	
	double *packed_A, *packed_B;
	packed_A = (double*) _mm_malloc(sizeof(double) * DGEMM_KC * (DGEMM_MC + 1), 256);
	packed_B = (double*) _mm_malloc(sizeof(double) * DGEMM_KC * (DGEMM_NC + 1), 256);
	assert(packed_A != NULL && packed_B != NULL);
	
	for (int jc = 0; jc < pad_c_cols; jc += DGEMM_NC)  // 5-th loop
	{
		int jc_blocksize = MIN(pad_c_cols - jc, DGEMM_NC);
		
		for (int kc = 0; kc < pad_comm_dim; kc += DGEMM_KC)  // 4-th loop
		{
			int kc_blocksize = MIN(pad_comm_dim - kc, DGEMM_KC);
			
			packB_KCxNC2(pad_B + kc * pad_ldb + jc, pad_ldb, packed_B, kc_blocksize, jc_blocksize);
			
			for (int ic = 0; ic < pad_c_rows; ic += DGEMM_MC)  // 3-rd loop
			{
				int ic_blocksize = MIN(pad_c_rows - ic, DGEMM_MC);
				
				packA_MCxKC2(pad_A + ic * pad_lda + kc, pad_lda, packed_A, ic_blocksize, kc_blocksize);
				
				dgemm_marco_kernel2(
					ic_blocksize, jc_blocksize, kc_blocksize,
					packed_A, packed_B, pad_C + ic * pad_ldc + jc, pad_ldc
				);
			}
		}
	}
	
	_mm_free(packed_A);
	_mm_free(packed_B);
	unpad_gemm_matrixs(c_rows, c_cols, C, ldc, pad_A, pad_B, pad_C, pad_ldc);
}