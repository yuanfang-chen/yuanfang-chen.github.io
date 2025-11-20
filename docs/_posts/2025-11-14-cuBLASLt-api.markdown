---
layout: post
title:  "cuBLASLt API 简介"
date:   2025-11-14 11:18:26 -0800
categories: CUDA
---

* TOC
{:toc}

<style>
  table {
    border-collapse: collapse; /* Ensures borders are collapsed for a cleaner look */
  }
</style>

## 什么是cuBLASLt
CUBLASLt（cuBLAS Light）是 NVIDIA cuBLAS 库的一个轻量级、灵活且高性能的扩展，专为深度学习中的混合精度矩阵乘法（GEMM）优化而设计。它支持 Tensor Core、自定义数据布局、融合操作（如 GELU、ReLU）、稀疏计算等高级特性。cuBLASLt支持自动调优。

## 核心概念
1. cuBLASLt是Host API，所有API都在Host侧调用。提供GEMM API（`cublasLtMatmul()`）和矩阵Element-wise转换能力（`cublasLtMatrixTransform()`）
1. 相比于cuBLAS的GEMM，cuBLASLt的GEMM API中可以传入更加详细的两个信息: GEMM计算的逻辑描述（`cublasLtMatmulDesc_t`）,A/B/C的布局描述（`cublasLtMatrixLayout_t`）
1. GEMM计算的逻辑描述（`cublasLtMatmulDesc_t`）包括：

   | Value | Description |
   |----------|---------------|
   | `x` | 。 |
   | `CUBLASLT_MATMUL_DESC_COMPUTE_TYPE` | 指定计算精度。注意这个和A/B/C的数据精度是解耦的。 |
   | `CUBLASLT_MATMUL_DESC_SCALE_TYPE` | 指定`alpha`和`beta`的数据精度。默认值同`CUBLASLT_MATMUL_DESC_COMPUTE_TYPE` |
   | `CUBLASLT_MATMUL_DESC_POINTER_MODE` | 指定`alpha`和`beta`是在Host侧内存中，Device侧内存中，还是一个Device侧向量。默认在Host侧。 |
   | `CUBLASLT_MATMUL_DESC_TRANSA` | 。 |
   | `CUBLASLT_MATMUL_DESC_A_SCALE_POINTER` | 以指针的形式指定一个scale用于将矩阵 A 中的数据转换到计算数据类型`CUBLASLT_MATMUL_DESC_COMPUTE_TYPE`的数值范围内。scale的数据类型必须与计算类型``CUBLASLT_MATMUL_DESC_COMPUTE_TYPE``相同。如果未指定或设为 NULL，则默认缩放因子为 1。 |
   | `CUBLASLT_MATMUL_DESC_B_SCALE_POINTER` | 与`CUBLASLT_MATMUL_DESC_A_SCALE_POINTER`等价 |
   | `CUBLASLT_MATMUL_DESC_C_SCALE_POINTER` | 与`CUBLASLT_MATMUL_DESC_A_SCALE_POINTER`等价 |
   | `CUBLASLT_MATMUL_DESC_D_SCALE_POINTER` | 与`CUBLASLT_MATMUL_DESC_A_SCALE_POINTER`等价 |
    TODO:

1. A/B/C的布局描述（`cublasLtMatrixLayout_t`）包括：

   | Value | Description |
   |----------|---------------|
   | `x` | 。 |
1. 开发者指定优化设置（`cublasLtMatmulPreference_t`），这些设置用于指导 cuBLASLt 在矩阵乘法（matmul）操作中进行算法选择（`cublasLtMatmulAlgo_t`）。算法的详细信息由`cublasLtMatmulAlgoConfigAttributes_t`表示，`cublasLtMatmulAlgoConfigSetAttribute`/`cublasLtMatmulAlgoConfigGetAttribute()`可以读写`cublasLtMatmulAlgoConfigAttributes_t`的每个属性。

    选择最优算法需要三个步骤：

    1. 创建一个 cublasLtMatmulPreference_t 对象；
    1. 设置你的偏好（如最大 workspace 大小、是否允许非确定性等）；
    1. 调用 cublasLtMatmulAlgoGetHeuristic()，传入该 preference，获取推荐的算法列表。

1. 开发者也可以通过`cublasLtMatmulAlgoInit`创建`cublasLtMatmulAlgo_t`。`cublasLtMatmulAlgoInit`的`algoId`入参可以通过`cublasLtMatmulAlgoGetIds()`来获取。

    ```cpp
    cublasLtMatmulAlgo_t algo = {};
    const int32_t algoId = 10;
    const cublasLtMatmulTile_t tileId = CUBLASLT_MATMUL_TILE_16x16; // 5
    const cublasLtReductionScheme_t reductionMode = CUBLASLT_REDUCTION_SCHEME_INPLACE; // 1
    const int32_t splitKFactor = 256;

    cublasLtMatmulAlgoInit(ltHandle,  //
                           CUBLAS_COMPUTE_64F,   // compute
                           CUDA_R_64F,   // scale
                           CUDA_R_64F,   // A
                           CUDA_R_64F,   // B
                           CUDA_R_64F,   // C
                           CUDA_R_64F,   // D
                           algoId,
                           &algo);

    cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_TILE_ID, &tileId, sizeof(tileId));
    cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME, &reductionMode, sizeof(reductionMode));
    cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_SPLITK_NUM, &splitKFactor, sizeof(splitKFactor));
    ```

## GEMM调用步骤
1. 创建GEMM的逻辑描述（`cublasLtMatmulDesc_t`）

    ```cpp
    // create operation desciriptor; see cublasLtMatmulDescAttributes_t for details about defaults; here we just need to
    // set the transforms for A and B
    cublasLtMatmulDescCreate(&operationDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F);
    cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa));
    cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transb));
    ```
1. 创建矩阵的布局描述（`cublasLtMatrixLayout_t`）

    ```cpp
    cublasLtMatrixLayoutCreate(&Adesc, CUDA_R_32F, m, k, lda);
    cublasLtMatrixLayoutCreate(&Bdesc, CUDA_R_32F, k, n, ldb);
    cublasLtMatrixLayoutCreate(&Cdesc, CUDA_R_32F, m, n, ldc);
    ```
1. 创建优化偏好（`cublasLtMatmulPreference_t`）

    ```cpp
    // create preference handle; here we could use extra attributes to disable tensor ops or to make sure algo selected
    // will work with badly aligned A, B, C; here for simplicity we just assume A,B,C are always well aligned (e.g.
    // directly come from cudaMalloc)
    cublasLtMatmulPreferenceCreate(&preference);
    cublasLtMatmulPreferenceSetAttribute(preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspaceSize, sizeof(workspaceSize));
    ```
1. 调用`cublasLtMatmulAlgoGetHeuristic`算出适用的GEMM算法。通过`requestedAlgoCount`入参可以指定可用的算法数量。

    ```cpp
    // we just need the best available heuristic to try and run matmul. There is no guarantee this will work, e.g. if A
    // is badly aligned, you can request more (e.g. 32) algos and try to run them one by one until something works
    cublasLtMatmulAlgoGetHeuristic(ltHandle, operationDesc, Adesc, Bdesc, Cdesc, Cdesc, preference, 1, &heuristicResult, &returnedResults);
    if (returnedResults == 0) {
        checkCublasStatus(CUBLAS_STATUS_NOT_SUPPORTED);
    }
    ```

1. 调用GEMM`cublasLtMatmul`

    ```cpp
    cublasLtMatmul(ltHandle,
                   operationDesc,
                   alpha,
                   A,
                   Adesc,
                   B,
                   Bdesc,
                   beta,
                   C,
                   Cdesc,
                   C,
                   Cdesc,
                   &heuristicResult.algo,
                   workspace,
                   workspaceSize,
                   0);
    ```

## 特性
## Epilogue融合
支持 BIAS, RELU, GELU, SCALE, RESIDUAL 等，减少 kernel launch 次数

下面是一个 使用 cuBLASLt 实现带 Epilogue 融合的 GEMM 示例，具体实现：
> D = ReLU( A × B + bias )

这个例子展示了如何使用 cuBLASLt 的 epilogue 融合功能，将 GEMM + Bias + ReLU 在一个 kernel 中完成，避免多次 kernel launch 和中间结果写回显存。

```cpp
int main() {
    // ----------------------------
    // 定义GEMM的形状
    // ----------------------------
    const int M = 4;
    const int N = 8;
    const int K = 16;

    float *A, *B, *C, *D, *bias;
    CHECK_CUDA(cudaMalloc(&A, M*K*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&B, K*N*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&C, M*N*sizeof(float)));  // used if beta != 0
    CHECK_CUDA(cudaMalloc(&D, M*N*sizeof(float)));  // output
    CHECK_CUDA(cudaMalloc(&bias, N*sizeof(float))); // bias vector size N

    float alpha = 1.0f;
    float beta  = 0.0f;

    // ----------------------------
    // 创建cuBLASLt句柄
    // ----------------------------
    cublasLtHandle_t lt;
    CHECK_CUBLAS(cublasLtCreate(&lt));

    // ----------------------------
    // 创建 Matmul 描述符
    // ----------------------------
    cublasLtMatmulDesc_t opDesc;
    CHECK_CUBLAS(
        cublasLtMatmulDescCreate(
            &opDesc,
            CUBLAS_COMPUTE_F32,
            CUDA_R_32F
        )
    );

    // 并设置 epilogue 为 BIAS + ReLU
    cublasLtEpilogue_t epi = CUBLASLT_EPILOGUE_RELU_BIAS;
    CHECK_CUBLAS(
        cublasLtMatmulDescSetAttribute(
            opDesc,
            CUBLASLT_MATMUL_DESC_EPILOGUE,
            &epi,
            sizeof(epi)
        )
    );

    // 设置bias数据的pointer
    CHECK_CUBLAS(
        cublasLtMatmulDescSetAttribute(
            opDesc,
            CUBLASLT_MATMUL_DESC_BIAS_POINTER,
            &bias,
            sizeof(bias)
        )
    );

    // ----------------------------
    // 创建矩阵布局
    // ----------------------------
    cublasLtMatrixLayout_t layoutA, layoutB, layoutC, layoutD;

    CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&layoutA, CUDA_R_32F, M, K, K));
    CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&layoutB, CUDA_R_32F, K, N, N));
    CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&layoutC, CUDA_R_32F, M, N, N));
    CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&layoutD, CUDA_R_32F, M, N, N));

    // ----------------------------
    // 分配 workspace
    // ----------------------------
    size_t workspaceSize = 8 * 1024 * 1024;
    void* workspace = nullptr;
    CHECK_CUDA(cudaMalloc(&workspace, workspaceSize));

    // ----------------------------
    // 创建 preference
    // ----------------------------
    cublasLtMatmulPreference_t pref;
    CHECK_CUBLAS(cublasLtMatmulPreferenceCreate(&pref));
    CHECK_CUBLAS(
        cublasLtMatmulPreferenceSetAttribute(
            pref,
            CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
            &workspaceSize,
            sizeof(workspaceSize)
        )
    );

    // ----------------------------
    // 执行 (GEMM + bias + ReLU)
    // ----------------------------
    CHECK_CUBLAS(
        cublasLtMatmul(
            lt,
            opDesc,
            &alpha,
            A, layoutA,
            B, layoutB,
            &beta,
            C, layoutC,
            D, layoutD,
            nullptr,
            workspace,
            workspaceSize,
            0  // stream
        )
    );

    CHECK_CUDA(cudaDeviceSynchronize());

    // ----------------------------
    // 清理资源
    // ----------------------------
    cudaFree(workspace);
    cublasLtMatrixLayoutDestroy(layoutA);
    cublasLtMatrixLayoutDestroy(layoutB);
    cublasLtMatrixLayoutDestroy(layoutC);
    cublasLtMatrixLayoutDestroy(layoutD);
    cublasLtMatmulDescDestroy(opDesc);
    cublasLtMatmulPreferenceDestroy(pref);
    cublasLtDestroy(lt);

    return 0;
}
```

## Python Bindings
https://docs.nvidia.com/cuda/nvmath-python/0.5.0/bindings/cublasLt.html
