---
layout: post
title:  "Advise for Ascend"
date:   2025-11-27 23:44:54 +0800
categories: ascend
typora-root-url: ..
typora-copy-images-to: ../assets/images
---

 HW的生态位置
	支撑自己的AI需求
	拓展生态，多卖卡

CANN相关的文档全部使用Sphinx



学习Groq LPU, Deterministic execution

> ## Execution Model: Static Scheduling
>
> GPU architectures rely on dynamic scheduling - hardware queues, runtime arbitration, and software kernels that introduce non-deterministic latency. During collective operations, when hundreds of cores must synchronize activation tensors, any delay propagates through the entire system.

compiler-defined (co-design)

> Our compiler pre-computes the entire execution graph, including inter-chip communication patterns, down to the individual clock cycles. This static scheduling eliminates:

- Cache coherency protocols
- Reorder buffers
- Speculative execution overhead
- Runtime coordination delays

> Software-Scheduled Network: RealScale Chip-to-Chip Interconnect
> Groq uses a plesiosynchronous, chip-to-chip protocol to cancel natural clock drift and align hundreds of LPUs to act as a single core. The SW compiler then can predict exactly when data will arrive, so developers can reason about timing. Periodic software sync adjusts for crystal-based drift, enabling not just compute scheduling but also network scheduling. This lets Groq operate like a single-core supercluster, sidestepping complex coordination problems found in traditional architectures by starting with the compiler.

![img](./assets/images/df308eef891cc2f8811be4e05b10561b81247747-1280x720.gif)

## Python-first

- easy to learn
- no Compilation
- better integrated with pytorch etc. (DL framework)
- better debug (no screens and screens of template errors like C++ does)
- must have same perf as C++ 

## ML architecture & NPU codesign

Using ML TO TAILOR THE DNN TO
THE TPU AND THE TPU TOPOLOGY TO
THE DNN
To enable Pareto-optimizations over quality and performance for
DNN models, we developed platform-aware neural architecture
search (PA-NAS) at scale to tailor DNN models for TPU v4
supercomputers automatically [32]. A PA-NAS designed CNN1
achieves ~1.6X better performance (QPS and latency) than the
baseline designed by generic NAS, with comparable accuracy [32].
Unlike [33], here we show how PA-NAS improved the
performance of DLRM0 on TPU v4.



"TPU v4: An Optically Reconfigurable Supercomputer for Machine Learning with Hardware Support for Embeddings"



"The first concern of a DSA after its compiler is the memory system" 

Dally, W.J., Turakhia, Y. and Han, S., 2020. Domain-specific
hardware accelerators. Communications of the ACM, 63(7), 48-57.



https://medium.com/@harishsingh8529/latency-vs-predictability-the-hidden-tradeoff-powering-modern-systems-98f4952b7415



## why tile-based programmming is succesful

## use agile hardware development

## performance-counters are important

Pitfall: Performance counters added as an afterthought for DSA hardware.

The TPU has 106 performance counters, and the designers wanted even more (see
Figure 7.45). The raison d’^ etre for DSAs is performance, and it is way too early in
their evolution to have a good idea about what is going on.



Workload analysis features. Building upon lessons
from TPUv1 [21], TPUv4i includes extensive tracing and
performance counter hardware features, particularly in the
uncore. They are used by the software stack to measure and analyze system-level bottlenecks in user workloads and
guide continuous compiler-level and application-level
optimizations (Figure 2). These features increase design
time and area, but are worthwhile because we aim for Perf/
TCO, not Perf/CapEx ③. The features enable significant
system-level performance improvements and boost
developer productivity over the lifetime of the product as
DNN workloads grow and evolve (see Table 4)



## CCCL for SIMT

write parallel code like CPU code

## CANN编程必须是一个体系

需要成体系的库，让内部开发者优化的库，固化成可以复用的库
https://www.nvidia.com/en-us/on-demand/session/gtc25-s72383/

比喻：建一个漂亮的高楼大厦，即使地基暂时不稳，但是也可以

## task parallism

> Task Parallelism vs. Data Parallelism
> Data parallelism is not the only type of parallelism used in parallel pro-
> gramming. Task parallelism has also been used extensively in parallel
> programming. Task parallelism is typically exposed through task decom-
> position of applications. For example, a simple application may need
> to do a vector addition and a matrix-vector multiplication. Each of these
> would be a task. Task parallelism exists if the two tasks can be done
> independently. I/O and data transfers are also common sources of tasks.
> In large applications, there are usually a larger number of independent
> tasks and therefore larger amount of task parallelism. For example, in a
> molecular dynamics simulator, the list of natural tasks includes vibrational
> forces, rotational forces, neighbor identification for non-bonding forces,
> non-bonding forces, velocity and position, and other physical properties
> based on velocity and position.
> In general, data parallelism is the main source of scalability for par-
> allel programs. With large datasets, one can often find abundant data
> parallelism to be able to utilize massively parallel processors and allow
> application performance to grow with each generation of hardware that
> has more execution resources. Nevertheless, task parallelism can also
> play an important role in achieving performance goals. We will be cover-
> ing task parallelism later when we introduce streams.

提高AI Core利用率需要使能task parallism

## Green Context

To address the needs of "AI Factories," CUDA 13 introduced Green Contexts.

This allows developers to partition a single GPU into deterministic, isolated resource pools at the hardware level (Streaming Multiprocessor groups).

This is critical for latency-sensitive applications where you need to guarantee that a high-priority task isn't "starved" by a massive background batch process.

## Find the right problem

## analysis parallism and data reuse
do a simialr analyais to [MTIA](https://dl.acm.org/doi/abs/10.1145/3579371.3589348) Chapter 3.5 for Ascend

## flexibility
Architecture and programmming stack should demonstrate sufficient flexibility to support a wide range of models.

DSA不可能灵活到可以在所有模型上都有最优的性能。Meta MTIA 2i采取的方式是牺牲一部分灵活性，保证一部分最重要的模型有最优的性能。

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



借鉴 “Linear Layouts: Robust Code Generation of Efficient Tensor Computation Using F2” 看如何给昇腾亲和的layout做定义



[Ascend needs automatic warp specialization](https://github.com/triton-lang/triton/blob/b5a27fc1ab5fa22c0adad7409a50ef3e6271ada6/docs/meetups/11-05-2025/notes.md) 



Other critera folks use when deciding on what language/framework to pick include: feature completeness and maturity. Is the language/framework in startup phase, are there teams using/supporting it, is it still evolving.



Hongtao Yu, Meta - It depends on how the hardware is designed. If scheduling is better on chip, we won’t need to do it in software. Nvidia HW is super configurable but the HW can’t schedule efficiently. Nvidia needs to invest more in hardware scheduling. We'll be keeping an eye on this.



Learn from Apple AMX / [Arm SME2](https://www.arm.com/technologies/sme2) to compute small matrix operation in CPU instead of on GPU. So ascend should be able to target CPU.



need advanced auto-tuning techniques like below, because there is hardware-assisted pipelining/scheduling in hardware. 

TritonForge: Profiling-Guided Framework for Automated Triton Kernel Optimization https://arxiv.org/abs/2512.09196

> ​     High-performance GPU kernel optimization remains a critical yet labor-intensive task in modern machine learning workloads. Although Triton, a domain-specific language for GPU programming, enables developers to write efficient kernels with concise code, achieving expert-level performance still requires deep understanding of GPU architectures and low-level performance trade-offs. We present TritonForge, a profiling-guided framework for automated Triton kernel optimization. TritonForge integrates kernel analysis, runtime profiling, and iterative code transformation to streamline the optimization process. By incorporating data-driven feedback from profiling results, the system identifies performance bottlenecks, proposes targeted code modifications, and evaluates their impact automatically. While our prototype leverages large language models (LLMs) to assist in code reasoning and transformation, the framework remains modular and model-agnostic. Across diverse kernel types and GPU architectures, TritonForge achieves up to 5x performance improvement over baseline implementations and on average 1.76x of the cases are successful, providing a foundation for future research in automated GPU performance optimization.



https://jlebar.com/2024/2/4/completeness.html

**Upshot 2:** If you're designing an incomplete IR, I'd say you probably should design a programming language to go with it. This way you can enforce your IR's limitations at the level of user code, instead of having performance and functionality cliffs that are unpredictable to your users and that depend on which compiler backend they're using. I think this tight coupling between language and IR may be one of the reasons JAX and Triton have been so successful.

A corollary of this is that your incomplete IR probably won't be relevant unless its associated programming language is successful. At which point, you should probably think about the programming language first and the IR second.



triton need good debugging

![Screenshot 2025-12-24 at 6.57.49 PM](/assets/images/Screenshot 2025-12-24 at 6.57.49 PM.png)







## 2. The New Frontier: AI-Agentic Kernel Generation

As of 2024–2025, the industry is shifting toward using **LLMs (like DeepSeek-R1 or GPT-4o) as compiler engineers**. This is often called "Agentic Compilation."

Instead of a human writing a kernel for a new "long-tail" operation, an AI agent system (like **KernelFalcon** or **PRAGMA**) takes over:

1. **Drafting:** The AI agent writes a candidate GPU kernel (often in Triton or CUDA).
2. **Verification:** A "Verifier" agent runs the code on a real GPU to see if it’s correct and fast.
3. **Iterative Refinement:** If the code is slow or buggy, the error logs and performance profiles are fed back to the AI. The AI "thinks" and rewrites the code.
4. **Deployment:** After 10–15 minutes of "reasoning," the system produces a kernel that often beats human-written code.

https://pytorch.org/blog/kernelfalcon-autonomous-gpu-kernel-generation-via-deep-agents/









![Screenshot 2025-12-24 at 8.18.22 PM](/assets/images/Screenshot 2025-12-24 at 8.18.22 PM.png)



## Lessons from MTIA
MTIA v1
MTIA 2i

## Lessons from TPU

https://cloud.google.com/blog/products/ai-machine-learning/an-in-depth-look-at-googles-first-tensor-processing-unit-tpu

like TPU, use CISC instead of RISC?



More importantly, despite having many more arithmetic units and large on-chip memory, the TPU chip is half the size of the other chips. Since the cost of a chip is a function of the area3 — more [smaller chips per silicon wafer](http://anysilicon.com/die-per-wafer-formula-free-calculators/) and higher yield for small chips since they're less likely to have manufacturing defects***** — halving chip size reduces chip cost by roughly a factor of 8 (23).


## Little's law

tensor core 必须依赖流水并行，硬件优化流水并行能力

simt core依赖高度并行，需要类似warp scheduler



[S72683 - CUDA Techniques to Maximize Memory Bandwidth and Hide Latency](https://www.nvidia.com/en-us/on-demand/session/gtc25-s72683/)



[Bill Dally - Trends in Deep Learning Hardware](https://www.youtube.com/watch?v=4u8iMr3iXR4)

