## Brief Introduction

My (re)implementation of x64 CPU DGEMM with AVX / AVX2.

Reference:

1. [How To Optimize GEMM](https://github.com/flame/how-to-optimize-gemm)
2. [BLISLab: A Sandbox for Optimizing GEMM](https://github.com/flame/blislab)

File `rvdg_dgemm/MMult_4x4_15.cc` is the counter part of Ref. 1's `src/MMult_4x4_15.c` , I use AVX Intrinsic.

File `blislab_dgemm/BLISLab_dgemm_kernel2.cc` is the counter part of Ref. 2's `step3/dgemm/my_dgemm.c` and `step3/kernels/bl_dgemm_int_8x4.c` . 

## Performance Test Results

Reference performance comes from [OpenBLAS](https://github.com/xianyi/OpenBLAS) v0.2.19. Test single thread performance. The unit of performance is GFlops.

Core i7-4702MQ @ 2.2GHz, AVX2 (theoretically 16 DP Flops / cycle), CentOS 7.3 x64 with GCC 4.8.5-11:

| n=m=k | Theory Peak | OpenBLAS | My MMult\_4x4\_15 | BLISLab_Step3_Int | My BLISLab Kernel |
| ----- | ----------- | -------- | ----------------- | ----------------- | ----------------- |
| 200   | 35.2        | 24.72    | 10.51             | 14.44             | 12.65             |
| 400   | 35.2        | 28.52    | 12.84             | 15.52             | 13.16             |
| 600   | 35.2        | 33.93    | 13.83             | 19.01             | 17.36             |
| 800   | 35.2        | 34.23    | 14.35             | 19.41             | 19.53             |
| 1000  | 35.2        | 35.18    | 14.93             | 19.69             | 20.57             |

Xeon E5-2620v2 @ 2.1GHz, AVX (theoretically 8 DP Flops / cycle), CentOS 7.3 x64 with GCC 4.8.5-11:

| n=m=k | Theory Peak | OpenBLAS | My MMult\_4x4\_15 | BLISLab_Step3_Int | My BLISLab Kernel |
| ----- | ----------- | -------- | ----------------- | ----------------- | ----------------- |
| 200   | 16.8        | 9.63     | 6.58              | 9.51              | 10.05             |
| 400   | 16.8        | 14.62    | 7.47              | 14.56             | 9.54              |
| 600   | 16.8        | 15.15    | 9.51              | 14.42             | 10.28             |
| 800   | 16.8        | 15.34    | 9.51              | 14.47             | 10.48             |
| 1000  | 16.8        | 15.48    | 9.86              | 14.54             | 10.87             |

For more information, please refer to [this page](http://enigmahuang.github.io/2017/07/26/my-DGEMM-BLISLab/).