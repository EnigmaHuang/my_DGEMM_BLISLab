/* Create macros so that the matrices are stored in column-major order */

#define A(i, j) A[(j) * lda + (i)]
#define B(i, j) B[(j) * ldb + (i)]
#define C(i, j) C[(j) * ldc + (i)]

char ver[] = "MMult_4x4_6";
char fatherver[] = "MMult_4x4_5";

/* 
Compute gamma += x' * y with vectors x and y of length n.

Here x starts at location x with increment (stride) incx and y starts at location y and has (implicit) stride of 1.

Create macro to let X(i ) equal the ith element of x 
*/

#define X(i) x[(i) * incx]

void AddDot(int k, double* x, int incx, double *y, double *gamma)
{
    int p;
    for (p = 0; p < k; p++)
        *gamma += X(p) * y[p];
}

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
void AddDot4x4(int k, double* A, int lda, double* B, int ldb, double* C, int ldc)
{
    int p;
    register double 
        a0p, a1p, a2p, a3p, 
        c00, c01, c02, c03, c10, c11, c12, c13,
        c20, c21, c22, c23, c30, c31, c32, c33;
        
    c00 = c01 = c02 = c03 = 0.0;
    c10 = c11 = c12 = c13 = 0.0;
    c20 = c21 = c22 = c23 = 0.0;
    c30 = c31 = c32 = c33 = 0.0;
        
    for (p = 0; p < k; p++)
    {
        a0p = A(0, p); a1p = A(1, p); a2p = A(2, p); a3p = A(3, p);
        
        c00 += a0p * B(p, 0);
        c01 += a0p * B(p, 1);
        c02 += a0p * B(p, 2);
        c03 += a0p * B(p, 3);
        
        c10 += a1p * B(p, 0);
        c11 += a1p * B(p, 1);
        c12 += a1p * B(p, 2);
        c13 += a1p * B(p, 3);
        
        c20 += a2p * B(p, 0);
        c21 += a2p * B(p, 1);
        c22 += a2p * B(p, 2);
        c23 += a2p * B(p, 3);
        
        c30 += a3p * B(p, 0);
        c31 += a3p * B(p, 1);
        c32 += a3p * B(p, 2);
        c33 += a3p * B(p, 3);
    }
    
    C(0, 0) += c00; C(0, 1) += c01; C(0, 2) += c02; C(0, 3) += c03;
    C(1, 0) += c10; C(1, 1) += c11; C(1, 2) += c12; C(1, 3) += c13;
    C(2, 0) += c20; C(2, 1) += c21; C(2, 2) += c22; C(2, 3) += c23;
    C(3, 0) += c30; C(3, 1) += c31; C(3, 2) += c32; C(3, 3) += c33;
}


extern "C"
void MyDGEMM(int hA, int wA, int wB,
                double* A, int lda, 
                double* B, int ldb,
                double* C, int ldc)
{
    int i, j;
    for (j = 0; j < wB; j += 4)
        for (i = 0; i < hA; i += 4)
            AddDot4x4(wA, &A(i, 0), lda, &B(0, j), ldb, &C(i, j), ldc);
}

extern "C"
char* VersionStr()
{
    return ver;
}