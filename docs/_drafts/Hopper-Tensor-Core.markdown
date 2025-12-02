---
layout: post
title:  "Hopper Tensor Core"
#date:   2025-11-27 23:44:54 +0800
categories: CUDA
typora-root-url: ..
---

All hardware features except WGMMA are inherited by Blackwell.

## Thread Block Cluster

**distributed shared memory (DSMEM)**

GPC



## Tensor Memory Accelerator



### TMA Multicast

As a brief review, TMA multicast places the data loaded by the TMA in the SMEM of multiple CTAs in the same cluster. Using this feature, a set of CTAs in a cluster can collaboratively and simultaneously load a tile of data into each of their shared memories, decreasing the global memory traffic in cases when multiple CTAs need to load the same data. Each CTA loads a portion of the data that is multicast into the SMEM of the other participating CTAs. For example, if the number of participating CTAs is 4, each CTA loads a quarter of the data, thereby reducing the total amount of data loaded with TMA by a factor of 4. Technically, this collaborative partial loading is a programming paradigm and is not intrinsic to TMA multicast feature, but in this article we will treat them as synonymous.

## Warpgroup MMA (WGMMA)

- 异步操作
- 连续128个线程完成，首个warp的rank必须是4的整数倍
- PTX `wgmma.mma_async`
- `A`可以在SMEM或者RMEM；`B`必须在SMEM；`C`必须在RMEM

## format
wgmma.mma_async.sync.aligned.m64n64k16.f16.f16.f16
The CUTLASS notation is such that one can immediately read off the relationship between the wrapped PTX instruction and the MMA atom. Firstly, SM90 is a different name for the Hopper architecture. SM90 MMA atoms are then labeled as SM90_MxNxK_XYZ_SS or SM90_MxNxK_XYZ_RS, with two template parameters that can be either GMMA::Major::MN or GMMA::Major::K. Their meanings are as follows:

X and Y are the datatypes of the operands.
Z is the datatype of the accumulator.
MxNxK are the tile sizes that the wgmma instruction computes with — the “wgmma atom”. Not all values of MxNxK are possible. Here is the list of allowed shapes: M is always 64, N is a multiple of 8 from 8 to 256, and for 16-bit operand datatype, K is 16 (more generally, K is fixed to be 32 bytes).
The suffix RS or SS indicates whether operand A is sourced from registers (R) or shared memory (S). Operand B is always sourced from shared memory, hence the S.
The two template parameters indicate whether operands A and B are memory-contiguous in the MN mode or K mode. For example, in BLAS notations, the operands both being K-major would correspond to a TN gemm (cf. this table). Note that for 16-bit operand datatypes, one has flexibility with the memory layouts being either MN-major or K-major. However, for non 16-bit operand datatypes, the layout must always be K-major.


```cpp
ThrMMA thr_mma = mma.get_slice(threadIdx.x);
Tensor tCsA = thr_mma.partition_A(sA);        // (MMA,MMA_M,MMA_K)
Tensor tCsB = thr_mma.partition_B(sB);        // (MMA,MMA_N,MMA_K)
Tensor tCgC = thr_mma.partition_C(gC);        // (MMA,MMA_M,MMA_N)
// Allocate the accumulators -- same size as the projected data
Tensor tCrC = thr_mma.make_fragment_C(tCgC);  // (MMA,MMA_M,MMA_N)
```



## setmaxnreg



## mainloop

 
