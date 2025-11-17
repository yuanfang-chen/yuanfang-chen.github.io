---
layout: post
title:  "cuBLASLt API 简介"
# date:   2021-11-28 11:18:26 -0800
categories: CUDA
---

# 什么是cuBLASLt
CUBLASLt（cuBLAS Light）是 NVIDIA cuBLAS 库的一个轻量级、灵活且高性能的扩展，专为深度学习中的混合精度矩阵乘法（GEMM）优化而设计。它支持 Tensor Core、自定义数据布局、融合操作（如 GELU、ReLU）、稀疏计算等高级特性。cuBLASLt支持自动调优。

# 特性
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

## 算法选择
`cublasLtMatmulPreference_t` 是 cuBLASLt（cuBLAS Light）库中的一个关键数据结构，用于在执行混合精度矩阵乘法（GEMM）时 配置算法选择的偏好策略。它决定了 cuBLASLt 在搜索可用算法（algorithms）时的行为，比如是否允许非确定性结果、最大工作空间大小、数学模式等。

选择最优算法需要三个步骤：

1. 创建一个 cublasLtMatmulPreference_t 对象；
2. 设置你的偏好（如最大 workspace 大小、是否允许非确定性等）；
3. 调用 cublasLtMatmulAlgoGetHeuristic()，传入该 preference，获取推荐的算法列表。

cublasLtMatmulPreference_t的定义
```cpp
/** Algo search preference to fine tune the heuristic function. */
typedef enum {
  /** Search mode, see cublasLtMatmulSearch_t.
   *
   * uint32_t, default: CUBLASLT_SEARCH_BEST_FIT
   */
  CUBLASLT_MATMUL_PREF_SEARCH_MODE = 0,

  /** Maximum allowed workspace size in bytes.
   *
   * uint64_t, default: 0 - no workspace allowed
   */
  CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES = 1,

  /** Reduction scheme mask, see cublasLtReductionScheme_t. Filters heuristic result to only include algo configs that se one of the required modes.
   *
   * E.g. mask value of 0x03 will allow only INPLACE and COMPUTE_TYPE reduction schemes.
   *
   * uint32_t, default: CUBLASLT_REDUCTION_SCHEME_MASK (allows all reduction schemes)
   */
  CUBLASLT_MATMUL_PREF_REDUCTION_SCHEME_MASK = 3,

  /** Minimum buffer alignment for matrix A (in bytes).
   *
   * Selecting a smaller value will exclude algorithms that can not work with matrix A that is not as strictly aligned
   * as they need.
   *
   * uint32_t, default: 256
   */
  CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_A_BYTES = 5,

  /** Minimum buffer alignment for matrix B (in bytes).
   *
   * Selecting a smaller value will exclude algorithms that can not work with matrix B that is not as strictly aligned
   * as they need.
   *
   * uint32_t, default: 256
   */
  CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_B_BYTES = 6,

  /** Minimum buffer alignment for matrix C (in bytes).
   *
   * Selecting a smaller value will exclude algorithms that can not work with matrix C that is not as strictly aligned
   * as they need.
   *
   * uint32_t, default: 256
   */
  CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_C_BYTES = 7,

  /** Minimum buffer alignment for matrix D (in bytes).
   *
   * Selecting a smaller value will exclude algorithms that can not work with matrix D that is not as strictly aligned
   * as they need.
   *
   * uint32_t, default: 256
   */
  CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_D_BYTES = 8,

  /** Maximum wave count.
   *
   * See cublasLtMatmulHeuristicResult_t::wavesCount.
   *
   * Selecting a non-zero value will exclude algorithms that report device utilization higher than specified.
   *
   * float, default: 0.0f
   */
  CUBLASLT_MATMUL_PREF_MAX_WAVES_COUNT = 9,

  /** Numerical implementation details mask, see cublasLtNumericalImplFlags_t. Filters heuristic result to only include
   * algorithms that use the allowed implementations.
   *
   * uint64_t, default: uint64_t(-1) (allow everything)
   */
  CUBLASLT_MATMUL_PREF_IMPL_MASK = 12,
} cublasLtMatmulPreferenceAttributes_t;

```

```cpp
// 1. 创建 preference 对象
cublasLtMatmulPreference_t preference;
cublasLtMatmulPreferenceInit(ltHandle, &preference);

// 2. 设置最大 workspace（例如 32MB）
size_t workspace_size = 32 * 1024 * 1024; // 32 MiB
cublasLtMatmulPreferenceSetAttribute(
    preference,
    CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
    &workspace_size,
    sizeof(workspace_size)
);

// 3. 可选：设置只允许确定性算法（避免非确定性结果）
uint32_t isDeterministic = CUBLASLT_REDUCTION_SCHEME_INPLACE;
cublasLtMatmulPreferenceSetAttribute(
    preference,
    CUBLASLT_MATMUL_PREF_REDUCTION_SCHEME_MASK,
    &isDeterministic,
    sizeof(isDeterministic)
);

// 4. 获取启发式推荐的算法
cublasLtMatmulHeuristicResult_t heuristic_results[8];
int returned_results = 0;
cublasLtMatmulAlgoGetHeuristic(
    ltHandle,
    matmul_desc,
    A_layout, B_layout, C_layout, D_layout,
    preference,
    8, // 最多返回 8 个候选
    heuristic_results,
    &returned_results
);

// 5. 使用第一个推荐算法执行 matmul
cublasLtMatmul(
    ltHandle,
    matmul_desc,
    &alpha, d_A, A_layout,
            d_B, B_layout,
    &beta,  d_C, C_layout,
                    d_D, D_layout,
    &heuristic_results[0].algo,
    d_workspace, workspace_size,
    stream
);
```

## Python Bindings
https://docs.nvidia.com/cuda/nvmath-python/0.5.0/bindings/cublasLt.html
