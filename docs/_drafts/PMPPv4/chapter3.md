## Chapter 3 Exercise

1. 
a.
```c++
__global__ void MatrixMulKernel(float* M, float* N,
                                float* P, int Width) {
    int row = blockIdx.x*blockDim.x+threadIdx.x;
    if (row < Width) {
        for (int col = 0; col < Width; ++col) {
            float Pvalue = 0;
            for (int k = 0; k < Width; ++k) {
                Pvalue += M[row*Width+k] * N[k*Width+col];
            }
            P[row*Width+col] = Pvalue;
        }
    }
}
MatrixMulKernel<<<ceil_div(Width, THREADS_PER_BLOCK), THREADS_PER_BLOCK>>>();
```
b.

```c++
__global__ void MatrixMulKernel(float* M, float* N,
                                float* P, int Width) {
    int col = blockIdx.x*blockDim.x+threadIdx.x;
    if (col < Width) {
        for(int row = 0; row < Width; ++row) {
            float Pvalue = 0;
            for (int k = 0; k < Width; ++k) {
                Pvalue += M[row*Width+k] * N[k*Width+col];
            }
            P[row*Width+col] = Pvalue;
        }
    }
}
MatrixMulKernel<<<ceil_div(Width, THREADS_PER_BLOCK), THREADS_PER_BLOCK>>>();
```
c. 
for a., the access to N is not consecutive 
for b., the access to M is consecutive

2. 

```c++
__global__ void MatrixVectorMulKernel(float* A, float* B,
                                float* C, int Width) {
    int i = blockIdx.x*blockDim.x+threadIdx.x;
    if (i < Width) {
        float Pvalue = 0;
        for (int k = 0; k < Width; ++k) {
            Pvalue += B[i*Width+k] * C[k];
        }
        A[i] = Pvalue;
    }
}
MatrixVectorMulKernel<<<ceil_div(Width, THREADS_PER_BLOCK), THREADS_PER_BLOCK>>>();
```

3. 
a. 16*32 = 512
b. 512 * 95 = 48640
c. ((300-1)/16 + 1) * ((150-1)/32 + 1) = 19 * 5 = 95
d. 150 * 300 = 45000

4. 
a. 20*400+10 = 8010
b. 10*500+20 = 5020

5.
5*(400*500) + 20*400 + 10 = 1008010


