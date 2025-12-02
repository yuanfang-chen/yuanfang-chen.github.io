---
layout: post
title: CUTLASS/cute resources
categories: CUDA
typora-root-url: ..
---

## Concepts
### Epilogue
The above code focuses only on the matrix multiply computation C = AB whose result is held in the registers of each thread within the threadblock. The mapping of logical elements in the output tile to each thread is chosen to maximize performance of the matrix multiply computation but does not result in efficient, coalesced loads and stores to global memory.

The epilogue is a separate phase in which threads exchange data through shared memory then cooperatively access global memory using efficient striped access patterns. It is also the phase in which linear scaling and other elementwise operations may be conveniently computed using the matrix product results as inputs.

CUTLASS defines several typical epilogue operations such as linear scaling and clamping, but other device-side function call operators may be used to perform custom operations.

## Optimizations
### Parallelized Reductions
- Split K - reduction across threadblocks (TODO: add detail)
- Sliced K - reduction across warps (TODO: add detail)

### threadblock swizzle
To maximize reuse of data held in the last level cache, CUTLASS defines several functions to affect the mapping of threadblocks to logical partitions of the GEMM problem. These map consecutively launched threadblocks to packed two-dimensional regions of the partitioned GEMM problem to increase the probability that these will access the same tiles of global memory at approximately the same time.

Several functions are defined in cutlass/gemm/threadblock_swizzle.h.


## GTC Talks
- [GTC'18 CUTLASS: Software Primitives for Dense Linear Algebra at All Levels and Scales within CUDA](https://www.nvidia.com/en-us/on-demand/session/gtcsiliconvalley2018-s8854/)
- [GTC'19 PROGRAMMING TENSOR CORES: NATIVE VOLTA TENSOR CORES WITH CUTLASS](https://developer.download.nvidia.com/video/gputechconf/gtc/2019/presentation/s9593-cutensor-high-performance-tensor-operations-in-cuda-v2.pdf)
- [GTC'20 Developing CUDA kernels to push Tensor Cores to the Absolute Limit on NVIDIA A100](https://www.nvidia.com/en-us/on-demand/session/gtcsj20-s21745/)
- [GTC'21 Accelerating Convolution with Tensor Cores in CUTLASS](https://www.nvidia.com/en-us/on-demand/session/gtcspring21-s31883/)
- [GTC'22 Accelerating Backward Data Gradient by Increasing Tensor Core Utilization in CUTLASS](https://www.nvidia.com/en-us/on-demand/session/gtcspring22-s41996/)
- [GTC'22 CUTLASS: Python API, Enhancements, and NVIDIA Hopper](https://www.nvidia.com/en-us/on-demand/session/gtcfall22-a41131/)
- [GTC'23 Developing Optimal CUDA Kernels on Hopper Tensor Cores](https://www.nvidia.com/en-us/on-demand/session/gtcspring23-s51413/)
- [GTC'24 CUTLASS: A Performant, Flexible, and Portable Way to Target Hopper Tensor Cores](https://www.nvidia.com/en-us/on-demand/session/gtc24-s61198/)
- [GTC'25 Programming Blackwell Tensor Cores with CUTLASS](https://www.nvidia.com/en-us/on-demand/session/gtc25-s72720/)
- [GTC'25 Enable Tensor Core Programming in Python with CUTLASS 4.0](https://www.nvidia.com/en-us/on-demand/session/gtc25-s74639/)


## Official Blog
- [Feb 2013 - An Efficient Matrix Transpose in CUDA C/C++](https://developer.nvidia.com/blog/efficient-matrix-transpose-cuda-cc/)
- [Oct 2017 - Volta - Programming Tensor Cores in CUDA 9](https://developer.nvidia.com/blog/programming-tensor-cores-cuda-9/)
- [Dec 2017 - Volta - CUTLASS: Fast Linear Algebra in CUDA C++](https://developer.nvidia.com/blog/cutlass-linear-algebra-cuda/)
- [Sep 2020 - Controlling Data Movement to Boost Performance on the NVIDIA Ampere Architecture](https://developer.nvidia.com/blog/controlling-data-movement-to-boost-performance-on-ampere-architecture/)
- [Mar 2022 - NVIDIA Hopper Architecture In-Depth](https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/)
- [July 2024 - Next Generation of FlashAttention](https://developer.nvidia.com/blog/next-generation-of-flashattention/)
- [July 2025 - CUTLASS 3.x: Orthogonal, Reusable, and Composable Abstractions for GEMM Kernel Design ](https://developer.nvidia.com/blog/cutlass-3-x-orthogonal-reusable-and-composable-abstractions-for-gemm-kernel-design/)
- [July 2025 -  CUTLASS: Principled Abstractions for Handling Multidimensional Data Through Tensors and Spatial Microkernels](https://developer.nvidia.com/blog/cutlass-principled-abstractions-for-handling-multidimensional-data-through-tensors-and-spatial-microkernels/)
- [Nov 2025 - Achieve CUTLASS C++ Performance with Python APIs Using CuTe DSL](https://developer.nvidia.com/blog/achieve-cutlass-c-performance-with-python-apis-using-cute-dsl/)
- [GPU Performance Background User's Guide](https://docs.nvidia.com/deeplearning/performance/dl-performance-gpu-background/index.html#)
- [Matrix Multiplication Background User's Guide](https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html)


## Non-official Blogs
- [NVIDIA Tensor Core Evolution: From Volta To Blackwell](https://newsletter.semianalysis.com/p/nvidia-tensor-core-evolution-from-volta-to-blackwell)
- [All Posts on Colfax](https://research.colfax-intl.com/)
- 
- [How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance: a Worklog (without using Tensor Core)](https://siboehm.com/articles/22/CUDA-MMM)
- [Outperforming cuBLAS on H100: a Worklog (using Tensor Core)](https://cudaforfun.substack.com/p/outperforming-cublas-on-h100-a-worklog)
- [Matrix Multiplication on Blackwell](https://www.modular.com/matrix-multiplication-on-blackwell)
- [Volta: Performance and Programmability](https://ieeexplore.ieee.org/document/8344474)
- [Inside NVIDIA Blackwell Ultra: The Chip Powering the AI Factory Era](https://developer.nvidia.com/blog/inside-nvidia-blackwell-ultra-the-chip-powering-the-ai-factory-era/)
- [Blackwell GPGPU架构新特性概览](https://zhuanlan.zhihu.com/p/32148105488)
- [Optimization Techniques for GPU Programming](https://dl.acm.org/doi/fullHtml/10.1145/3570638)
- [Deep Dive on CUTLASS Ping-Pong GEMM Kernel](https://pytorch.org/blog/cutlass-ping-pong-gemm-kernel/)


## Deep Learning Concepts
- [Deep Learning in a Nutshell: Core Concepts](https://developer.nvidia.com/blog/deep-learning-nutshell-core-concepts/)
- [Understanding Convolution in Deep Learning](https://timdettmers.com/2015/03/26/convolution-deep-learning/)
