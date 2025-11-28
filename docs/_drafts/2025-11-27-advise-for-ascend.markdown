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

