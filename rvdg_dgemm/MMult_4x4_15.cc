#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include "x86intrin.h"

/* Create macros so that the matrices are stored in column-major order */
#define A(i, j) A[(j) * lda + (i)]
#define B(i, j) B[(j) * ldb + (i)]
#define C(i, j) C[(j) * ldc + (i)]

/* Block sizes */
#define max_hC 256
#define max_wC 128

#define min(i, j) ((i) < (j) ? (i) : (j))

char ver[] = "MMult_4x4_15";
char fatherver[] = "MMult_4x4_14";

typedef union
{
    __m128d v;
    double d[2];
}v2df_t;

typedef union
{
    __m256d v;
    double d[4];
}v4df_t;

/* 
    This routine computes a 4x4 block of matrix A
    C(0, 0), C(0, 1), C(0, 2), C(0, 3).  
    C(1, 0), C(1, 1), C(1, 2), C(1, 3).  
    C(2, 0), C(2, 1), C(2, 2), C(2, 3).  
    C(3, 0), C(3, 1), C(3, 2), C(3, 3).  
  
    Notice that this routine is called with c = C(i, j) in the
    previous routine, so these are actually the elements 
  
    C(i  , j), C(i  , j+1), C(i  , j+2), C(i  , j+3) 
    C(i+1, j), C(i+1, j+1), C(i+1, j+2), C(i+1, j+3) 
    C(i+2, j), C(i+2, j+1), C(i+2, j+2), C(i+2, j+3) 
    C(i+3, j), C(i+3, j+1), C(i+3, j+2), C(i+3, j+3) 
    in the original matrix C 
 */ 
/*
void AddDot4x4(int k, double* A, int lda, double* B, int ldb, double* C, int ldc)
{
    int p;

    v2df_t
    c_00_c_10_vreg, c_01_c_11_vreg, c_02_c_12_vreg, c_03_c_13_vreg,
    c_20_c_30_vreg, c_21_c_31_vreg, c_22_c_32_vreg, c_23_c_33_vreg,
    a_0p_a_1p_vreg, a_2p_a_3p_vreg,
    b_p0_vreg, b_p1_vreg, b_p2_vreg, b_p3_vreg; 

    c_00_c_10_vreg.v = _mm_setzero_pd();   
    c_01_c_11_vreg.v = _mm_setzero_pd();
    c_02_c_12_vreg.v = _mm_setzero_pd(); 
    c_03_c_13_vreg.v = _mm_setzero_pd(); 
    c_20_c_30_vreg.v = _mm_setzero_pd();   
    c_21_c_31_vreg.v = _mm_setzero_pd();  
    c_22_c_32_vreg.v = _mm_setzero_pd();   
    c_23_c_33_vreg.v = _mm_setzero_pd(); 

    for (p = 0; p < k; p++)
    {
        a_0p_a_1p_vreg.v = _mm_load_pd((double*) A);
        a_2p_a_3p_vreg.v = _mm_load_pd((double*) A + 2);
        A += 4;

        b_p0_vreg.v = _mm_loaddup_pd((double*) B); 
        b_p1_vreg.v = _mm_loaddup_pd((double*) B + 1); 
        b_p2_vreg.v = _mm_loaddup_pd((double*) B + 2); 
        b_p3_vreg.v = _mm_loaddup_pd((double*) B + 3); 
        B += 4;

        // First row and second rows 
        c_00_c_10_vreg.v += a_0p_a_1p_vreg.v * b_p0_vreg.v;
        c_01_c_11_vreg.v += a_0p_a_1p_vreg.v * b_p1_vreg.v;
        c_02_c_12_vreg.v += a_0p_a_1p_vreg.v * b_p2_vreg.v;
        c_03_c_13_vreg.v += a_0p_a_1p_vreg.v * b_p3_vreg.v;

        // Third and fourth rows 
        c_20_c_30_vreg.v += a_2p_a_3p_vreg.v * b_p0_vreg.v;
        c_21_c_31_vreg.v += a_2p_a_3p_vreg.v * b_p1_vreg.v;
        c_22_c_32_vreg.v += a_2p_a_3p_vreg.v * b_p2_vreg.v;
        c_23_c_33_vreg.v += a_2p_a_3p_vreg.v * b_p3_vreg.v;
    }

    C(0, 0) += c_00_c_10_vreg.d[0];  C(0, 1) += c_01_c_11_vreg.d[0];  
    C(0, 2) += c_02_c_12_vreg.d[0];  C(0, 3) += c_03_c_13_vreg.d[0]; 

    C(1, 0) += c_00_c_10_vreg.d[1];  C(1, 1) += c_01_c_11_vreg.d[1];  
    C(1, 2) += c_02_c_12_vreg.d[1];  C(1, 3) += c_03_c_13_vreg.d[1]; 

    C(2, 0) += c_20_c_30_vreg.d[0];  C(2, 1) += c_21_c_31_vreg.d[0];  
    C(2, 2) += c_22_c_32_vreg.d[0];  C(2, 3) += c_23_c_33_vreg.d[0]; 

    C(3, 0) += c_20_c_30_vreg.d[1];  C(3, 1) += c_21_c_31_vreg.d[1];  
    C(3, 2) += c_22_c_32_vreg.d[1];  C(3, 3) += c_23_c_33_vreg.d[1]; 
}
*/
void AddDot4x4(int k, double* A, int lda, double* B, int ldb, double* C, int ldc)
{
    int p;

	register __m256d C0, C1, C2, C3, A0, B0, B1, B2, B3;
	v4df_t c0, c1, c2, c3;

    C0 = _mm256_setzero_pd();
	C1 = _mm256_setzero_pd();
	C2 = _mm256_setzero_pd();
	C3 = _mm256_setzero_pd();

    for (p = 0; p < k; p++)
    {
		A0 = _mm256_loadu_pd((double*) A);
        A += 4;

		B0 = _mm256_broadcast_sd((double*) B);
		B1 = _mm256_broadcast_sd((double*) B + 1);
		B2 = _mm256_broadcast_sd((double*) B + 2);
		B3 = _mm256_broadcast_sd((double*) B + 3);
		B += 4;
		
		C0 += A0 * B0;
		C1 += A0 * B1;
		C2 += A0 * B2;
		C3 += A0 * B3;
    }
	
	c0.v = C0; c1.v = C1; c2.v = C2; c3.v = C3;
	
	C(0, 0) += c0.d[0];
	C(1, 0) += c0.d[1];
	C(2, 0) += c0.d[2];
	C(3, 0) += c0.d[3];
	
	C(0, 1) += c1.d[0];
	C(1, 1) += c1.d[1];
	C(2, 1) += c1.d[2];
	C(3, 1) += c1.d[3];

	C(0, 2) += c2.d[0];
	C(1, 2) += c2.d[1];
	C(2, 2) += c2.d[2];
	C(3, 2) += c2.d[3];
	
	C(0, 3) += c3.d[0];
	C(1, 3) += c3.d[1];
	C(2, 3) += c3.d[2];
	C(3, 3) += c3.d[3];
}

/*
  A(CM)        ->      packedA(RM)
0 10 20 ...            0  1  2  3
1 11 21 ...           10 11 12 13
2 12 22 ...           20 21 22 23
3 13 23 ...           30 31 32 33
4 14 24 ...           40 41 42 43
... ... ...           ... ... ...
so the new lda shoule be 4.
*/
void PackMatrixA(int wA, double* A, int lda, double* pA)
{
    int j;
    double *a_ij_ptr;
    for (j = 0; j < wA; j++)
    {
        a_ij_ptr = &A(0, j);
        *pA = *a_ij_ptr;
        *(pA + 1) = *(a_ij_ptr + 1);
        *(pA + 2) = *(a_ij_ptr + 2);
        *(pA + 3) = *(a_ij_ptr + 3);
        pA += 4;
    }
}


/*
  B(CM)           ->    pB(RM)
0 10 20 30 40...      0 10 20 30    
1 11 21 31 41...      1 11 21 31
2 12 22 32 42...      2 12 22 32
3 13 23 33 43...      3 13 23 33
4 14 24 34 44...      4 14 24 34
... ... ... ...       ... ... ...
*/
void PackMatrixB(int wA, double *B, int ldb, double *pB)
{
    int i;
    double *b_i0_ptr = &B(0, 0);
    double *b_i1_ptr = &B(0, 1);
    double *b_i2_ptr = &B(0, 2);
    double *b_i3_ptr = &B(0, 3);

    for (i = 0; i < wA; i++) // loop over rows of B 
    {  
        *pB++ = *b_i0_ptr++;
        *pB++ = *b_i1_ptr++;
        *pB++ = *b_i2_ptr++;
        *pB++ = *b_i3_ptr++;
    }
}

void InnerKernel(int hA, int wA, int wB, 
                 double *A, int lda, 
                 double *B, int ldb,
                 double *C, int ldc, 
                 int is_first_time)
{
    int i, j;
    double* pA = (double*)malloc(sizeof(double) * hA * wA);
    //Note: using a static buffer is not thread safe
    static double pB[max_wC * 2100];
    
    for (j = 0; j < wB; j += 4)
    {
        if (is_first_time) PackMatrixB(wA, &B(0, j), ldb, &pB[j * wA]);
        for (i = 0; i < hA; i += 4)
        {      
            if (j == 0)  PackMatrixA(wA, &A(i, 0), lda, &pA[i * wA]);
            AddDot4x4(wA, &pA[i * wA], 4, &pB[j * wA], wA, &C(i, j), ldc);
        }
    }
    
    free(pA);
}


void MM_Kernel(int hA, int wA, int wB,
               double* A, int lda, 
               double* B, int ldb,
               double* C, int ldc)
{
    int i, p, block_wC, block_hC;
    // This time, we compute a max_hC x wB(original C width) block 
    // of C by a call to the InnerKernel
    for (p = 0; p < wA; p += max_wC)
    {
        block_wC = min(wA - p, max_wC);
        for (i = 0; i < hA; i += max_hC)
        {
            block_hC = min(hA - i, max_hC);
            InnerKernel(block_hC, block_wC, wB, &A(i, p), lda, &B(p, 0), ldb, &C(i, 0), ldc, i == 0);
        }
    }
}

int Align_K(int x, int k)
{
    int res = x / k * k;
    if (res < x) res += k;
    return res;
}

extern "C"
void MyDGEMM(int hA, int wA, int wB,
             double* A, int lda, 
             double* B, int ldb,
             double* C, int ldc)
{
    //MM_Kernel(hA, wA, wB, A, lda, B, ldb, C, ldc);
    
    int i;
    int dsize = sizeof(double);
    int hB = wA;
    int hC = hA;
    int wC = wB;
    int nhA = Align_K(hA, 4);
    int nwA = Align_K(wA, 4);
    int nhB = nwA;
    int nwB = Align_K(wB, 4);
    int nhC = nhA;
    int nwC = nwB;
    int nlda = nhA;
    int nldb = nhB;
    int nldc = nhC;
    
    double* nA = (double*) malloc(dsize * nhA * nwA);
    double* nB = (double*) malloc(dsize * nhB * nwB);
    double* nC = (double*) malloc(dsize * nhC * nwC);
    
    memset(nA, 0, dsize * nhA * nwA);
    memset(nB, 0, dsize * nhB * nwB);
    memset(nC, 0, dsize * nhC * nwC);
    
    // Notice : matrices are stored in Coloum-Major
    for (i = 0; i < wA; i++)
        memcpy(nA + i * nhA, A + i * lda, dsize * hA);
    
    for (i = 0; i < wB; i++)
        memcpy(nB + i * nhB, B + i * ldb, dsize * hB);
    
    MM_Kernel(nhA, nwA, nwB, nA, nlda, nB, nldb, nC, nldc);
    
    // Copy back result to original C matrix
    for (i = 0; i < wC; i++)
        memcpy(C + i * ldc, nC + i * nhC, dsize * hC);
    
    free(nA);
    free(nB);
    free(nC);
}

extern "C"
char* VersionStr()
{
    return ver;
}