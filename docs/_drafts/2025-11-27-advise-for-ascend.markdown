---
layout: post
title:  "Advise for Ascend"
date:   2025-11-27 23:44:54 +0800
categories: ascend
typora-root-url: ..
---

## Architecture Advise

### TMA Store Reduce

All of these operations — namely reduce sum, reduce max, and reduce min — are fairly common in tensor programs. In particular, reduce sum is an inevitable subroutine in Split-K GEMM, while reduce max and reduce min are often used in attention. As simple as these operations look, implementing them in CUDA kernels is not very straightforward. We invite readers to briefly think through how many rounds of data movements between GMEM and SMEM must be carried out to achieve these goals before reading the next paragraph.

The vanilla implementation of a reduce operation that “accumulates” values from a CTA’s SMEM into a tile in a GMEM tensor consists of one GMEM read, one processing block, and one GMEM write. First, the original value from the GMEM is loaded into the CTA’s SMEM or register, then the reduce operation happens, and finally the result is written back out. This process is slow.

```cpp
// original: create a TMA store object
auto tma_store = make_tma_copy(SM90_TMA_STORE{}, gmem_tensor, smem_layout);
 
// to create a TMA reduce sum object
auto tma_reduce_sum = make_tma_copy(SM90_TMA_REDUCE_ADD{}, gmem_tensor, smem_layout);
```

### [Stream-k lesson](https://research.colfax-intl.com/cutlass-tutorial-persistent-kernels-and-stream-k/)

![img](https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2024/12/hybrid.png?resize=1906%2C1202&ssl=1)

L2 cache for subTile A and subTile B





![img](https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2024/12/m1024-w-heuristic.png?resize=720%2C540&ssl=1)

As you can see, the CUTLASS Heuristic mode does a very good job predicting the best performing decomposition mode. It selects the DataParallel mode when the quantization effect is low, and selects Stream-K when it is high. As the Heuristic mode is the default, you are generally better off not specifying the decomposition mode and letting CUTLASS decide.

昇腾是否有这种Heuristic，不需要开发者介入的。关键在于性能的可预测性。



昇腾是否可以借鉴B100 UMMA的**instruction descriptor**

// The A and B matrices that are read from SMEM need to be provided to MMA instructions as SMEM Descriptors.

//   These are the A and B fragments of the tcgen05.mma in CuTe terminology.



## 硬件架构建议

https://old.hotchips.org/wp-content/uploads/hc_archives/hc29/HC29.21-Monday-Pub/HC29.21.10-GPU-Gaming-Pub/HC29.21.132-Volta-Choquette-NVIDIA-Final3.pdf



![image-20251208201156943](/assets/images/image-20251208201156943.png)

![image-20251208201444420](/assets/images/image-20251208201444420.png)

- Trade TLP for ILP because ILP is what make CPU world is good at.
- "ILP + massive parallism"  means synchronization needs to be fast.



* use TN layout MMA https://leimao.github.io/blog/NVIDIA-Tensor-Core-MMA-Instruction-TN-Layout/





## 编程架构

第一层：cutlass/catlass cute
第二层：cuBLASDx
第三层：cuBLASLt
第四层：cuTile

跨平台编程框架如Triton、TileLang也要支持，但不如原生的重要



## cuBLAS

Matmul API 优化细节过多，类似imperative programming

cublaslt是declarative programming，有一定的kernel fusion能力
cublasdx可以完全定制kernel fusion



建议Matmul API拆分成cublaslt和cublasdx






cublaslt的 input/output 是都在host
    checkCublasStatus(cublasLtMatmul(ltHandle,
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
                                     0));


"模板参数MatmulCallBackFunc支持用户定制化Matmul的A矩阵、B矩阵及C矩阵的搬入搬出功能，如非连续搬入或针对搬出设置不同的数据片段间隔等。"
这个能力cute Layout可以解决

https://chatgpt.com/s/t_6902394f732481918af7f92f3b0a25da    Why cuBLASLt omits fill mode
