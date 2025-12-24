---
layout: post
title:  "Flash Attention"
# date:   2025-12-08 11:18:26 -0800
categories: CUDA
typora-root-url: ..
---



[A Case Study in CUDA Kernel Fusion: Implementing FlashAttention-2 on NVIDIA Hopper Architecture using the CUTLASS Library](https://arxiv.org/abs/2312.11918)



FlashAttention v3 manually optimizes data movement
using byte permute and warp shuffle instructions to bypass
shared memory in layout conversions—an approach that has
not yet been implemented in DL compilers.



## Flex Attention

https://pytorch.org/blog/flexattention/
