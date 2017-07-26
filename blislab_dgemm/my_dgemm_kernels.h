#ifndef _MY_DGEMM_KERNELS_H_
#define _MY_DGEMM_KERNELS_H_

#define MY_DGEMM_ARGS const int &c_rows, const int &c_cols, const int &comm_dim, \
	double *A, const int &lda, \
	double *B, const int &ldb, \
	double *C, const int &ldc
	
void my_dgemm_1(MY_DGEMM_ARGS);
void my_dgemm_2(MY_DGEMM_ARGS);
void my_dgemm_3(MY_DGEMM_ARGS);
void my_dgemm_4(MY_DGEMM_ARGS);
void my_dgemm_5(MY_DGEMM_ARGS);

void pad_gemm_matrixs(
	MY_DGEMM_ARGS,
	int &pad_c_rows, int &pad_c_cols, int &pad_comm_dim,
	double *&pad_A, int &pad_lda,
	double *&pad_B, int &pad_ldb,
	double *&pad_C, int &pad_ldc
);

void unpad_gemm_matrixs(
	const int &c_rows, const int &c_cols, 
	double *C, const int &ldc,
	double *&pad_A, double *&pad_B, 
	double *&pad_C, const int &pad_ldc
);

#endif 
