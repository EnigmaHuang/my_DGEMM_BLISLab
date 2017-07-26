#include <x86intrin.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include "my_dgemm_kernels.h"
#include "BLISLab_dgemm_kernel.h"

void compute_4x4_kernel_reg(
	double *A, const int &lda, double *B, const int &ldb,
	const int &comm_dim, double *C, const int &ldc
)
{
	register double C00, C01, C02, C03, C10, C11, C12, C13,
					C20, C21, C22, C23, C30, C31, C32, C33,
					B00, B01, B02, B03, Aij;
	C00 = 0.0; C01 = 0.0; C02 = 0.0; C03 = 0.0;
	C10 = 0.0; C11 = 0.0; C12 = 0.0; C13 = 0.0;
	C20 = 0.0; C21 = 0.0; C22 = 0.0; C23 = 0.0;
	C30 = 0.0; C31 = 0.0; C32 = 0.0; C33 = 0.0;
	
	for (int i = 0; i < comm_dim; i++)
	{
		B00 = B[i * ldb];
		B01 = B[i * ldb + 1];
		B02 = B[i * ldb + 2];
		B03 = B[i * ldb + 3];
		
		Aij = A[i];
		C00 += Aij * B00;
		C01 += Aij * B01;
		C02 += Aij * B02;
		C03 += Aij * B03;
		
		Aij = A[lda + i];
		C10 += Aij * B00;
		C11 += Aij * B01;
		C12 += Aij * B02;
		C13 += Aij * B03;
		
		Aij = A[2 * lda + i];
		C20 += Aij * B00;
		C21 += Aij * B01;
		C22 += Aij * B02;
		C23 += Aij * B03;
		
		Aij = A[3 * lda + i];
		C30 += Aij * B00;
		C31 += Aij * B01;
		C32 += Aij * B02;
		C33 += Aij * B03;
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

void compute_4x4_kernel_avx_broadcast(
	double *A, const int &lda, double *B, const int &ldb,
	const int &comm_dim, double *C, const int &ldc
)
{
	register __m256d C0, C1, C2, C3;
	register __m256d B0, A0, A1, A2, A3;
	
	C0 = _mm256_load_pd(C + 0 * ldc);
	C1 = _mm256_load_pd(C + 1 * ldc);
	C2 = _mm256_load_pd(C + 2 * ldc);
	C3 = _mm256_load_pd(C + 3 * ldc);
	
	for (int i = 0; i < comm_dim; i++)
	{
		B0 = _mm256_loadu_pd(B + i * ldb);
		A0 = _mm256_broadcast_sd(A + lda * 0 + i);
		A1 = _mm256_broadcast_sd(A + lda * 1 + i);
		A2 = _mm256_broadcast_sd(A + lda * 2 + i);
		A3 = _mm256_broadcast_sd(A + lda * 3 + i);
		
		C0 += A0 * B0;
		C1 += A1 * B0;
		C2 += A2 * B0;
		C3 += A3 * B0;
	}
	
	_mm256_store_pd(C + 0 * ldc, C0);
	_mm256_store_pd(C + 1 * ldc, C1);
	_mm256_store_pd(C + 2 * ldc, C2);
	_mm256_store_pd(C + 3 * ldc, C3);
}

void compute_4x4_kernel_avx_shuffle(
	double *A, const int &lda, double *B, const int &ldb,
	const int &comm_dim, double *C, const int &ldc
)
{
	register __m256d C0, C1, C2, C3, A0, B0;
	v4df_t A0_tmp, c0, c1, c2, c3;
	
	C0 = _mm256_setzero_pd();
	C1 = _mm256_setzero_pd();
	C2 = _mm256_setzero_pd();
	C3 = _mm256_setzero_pd();
	
	for (int i = 0; i < comm_dim; i++)
	{
		B0 = _mm256_loadu_pd(B + i * ldb);
		A0_tmp.d[0] = A[lda * 0 + i];
		A0_tmp.d[1] = A[lda * 1 + i];
		A0_tmp.d[2] = A[lda * 2 + i];
		A0_tmp.d[3] = A[lda * 3 + i];
		A0 = A0_tmp.v;

		C0 += A0 * B0;
		
		B0 = _mm256_shuffle_pd(B0, B0, 0x5);
		C1 += A0 * B0;
		
		B0 = _mm256_permute2f128_pd(B0, B0, 0x1);
		C2 += A0 * B0;
		
		B0 = _mm256_shuffle_pd(B0, B0, 0x5);
		C3 += A0 * B0;
	}
	
	c0.v = C0; c1.v = C1; c2.v = C2; c3.v = C3;
	
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
}

void dgemm_marco_kernel_ikj(MY_DGEMM_ARGS)
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

void dgemm_marco_kernel_4x4(MY_DGEMM_ARGS)
{
	for (int i = 0; i < c_rows; i += 4)
		for (int j = 0; j < c_cols; j += 4)
			compute_4x4_kernel_avx_broadcast(
				A + i * lda, lda, 
				B + j, ldb, comm_dim, 
				C + i * ldc + j, ldc
			);
}

void packB_KCxNC1(
	double *B, const int &ldb,
	double *packed_B, const int &ldpb,
	const int &pb_rows, const int &pb_cols
)
{
	for (int i = 0; i < pb_rows; i++)
		memcpy(packed_B + i * ldpb, B + i * ldb, sizeof(double) * pb_cols);
}

void packA_MCxKC1(
	double *A, const int &lda,
	double *packed_A, const int &ldpa,
	const int &pa_rows, const int &pa_cols
)
{
	for (int i = 0; i < pa_rows; i++)
		memcpy(packed_A + i * ldpa, A + i * lda, sizeof(double) * pa_cols);
}

void blislab_dgemm_kernel1(MY_DGEMM_ARGS)
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
			
			// naive one
			packB_KCxNC1(pad_B + kc * pad_ldb + jc, pad_ldb, packed_B, DGEMM_NC + 1, kc_blocksize, jc_blocksize);
			
			for (int ic = 0; ic < pad_c_rows; ic += DGEMM_MC)  // 3-rd loop
			{
				int ic_blocksize = MIN(pad_c_rows - ic, DGEMM_MC);
				
				packA_MCxKC1(pad_A + ic * pad_lda + kc, pad_lda, packed_A, DGEMM_KC + 1, ic_blocksize, kc_blocksize);
				
				dgemm_marco_kernel_4x4(
					ic_blocksize, jc_blocksize, kc_blocksize,
					packed_A, DGEMM_KC + 1,
					packed_B, DGEMM_NC + 1,
					pad_C + ic * pad_ldc + jc, pad_ldc
				);
			}
		}
	}
	
	_mm_free(packed_A);
	_mm_free(packed_B);
	unpad_gemm_matrixs(c_rows, c_cols, C, ldc, pad_A, pad_B, pad_C, pad_ldc);
}