---
layout: post
title:  "CUDA Performance Analysis"
# date:   2025-11-21 11:18:26 -0800
categories: CUDA
---

## XX
The big picture: “Feeding the beast”
There are 2 main actions in a GEMM kernel: copying the numbers to the correct memory addresses, and multiply-accumulating them. The former action is handled by copy instructions: TMA in Hopper, cp.async in Ampere, and vanilla copy in earlier architectures. The latter action, since the Volta architecture in 2017, has become the exclusive business of the tensor cores.

Through many generations, the tensor cores have become a beast at consuming the numbers fed to them. For instance, the H200 SXM GPU’s tensor cores can deliver up to 3,958 TFLOPS (TeraFLOPs per second). On the other hand, the memory bandwidth of the same H200 SXM GPU is only 4.8 TB/s (TeraBytes per second). This data transferring speed is much slower than the tensor cores’ speed, and oftentimes is not trivial to fully utilize! As such, a common theme of CUDA programming — and GEMM kernel design in particular — is to figure out how to copy numbers fast enough to keep the tensor cores busy. We call this process “feeding the beast.”

In general, there are two overarching strategies to “feed the beast,” which are complementary and function at different scopes (grid vs. block). 

- The first strategy is effective threadblock scheduling, which entails distributing the computation among the CTAs to obtain good load balancing and a higher rate of L2 cache hits. We will discuss this in a later blog post, but for now, we refer curious readers to the techniques of threadblock rasterization and persistent kernels, for instance as implemented in CUTLASS. 
- The second strategy, which we focus on in this tutorial, is to overlap copying with math operations. In particular, while the tensor cores are busy multiplying a batch of numbers that they receive, we should tell the copying units to copy the next batch of numbers. That way, we effectively hide part of the copying latency. This is the goal of pipelining.

https://developer.nvidia.com/blog/accelerating-hpc-applications-with-nsight-compute-roofline-analysis/


