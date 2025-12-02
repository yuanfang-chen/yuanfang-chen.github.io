---
layout: post
title:  "CUTLASS cute for user"
#date:   2025-11-11 11:18:26 -0800
categories: CUDA
typora-root-url: ..
---

## variable naming convention in Cute
Optimizing tool selection...Short answer: it's a naming convention. tCgA means "the tiled/partitioned tensor (tC) produced from the global tensor gA".

Explanation and how to read these names
- Prefixes:
  - `g` = global/gmem slice (e.g. `gA`, `gB`, `gC`, `gD` are the GMEM sub-tensors produced by `local_tile`).
  - `t` = tiled / partitioned tensor (result of applying a tiler/partitioner).
  - `s` = smem (a SMEM-backed tensor or smem view).
  - `r` = fragment/ register-like descriptor (a fragment or descriptor used by the MMA).
  - other letters may appear to indicate other transformations (e.g., `tCtAcc` is a tiled TMEM accumulator).
- Middle letter(s) after `t` identify the partitioning pattern or which partition API was used:
  - in this file the code uses methods like `cta_mma.partition_A`, `partition_B`, `partition_C`. The name `tC` is the token used by the authors to indicate that partitioning pattern (the exact token is just a human-readable label used consistently in the example).
- Suffix is the source tensor the partition was applied to:
  - `gA`, `gB`, `gC`, `gD` are the global tensors (slices) the kernel works on.

Concrete mapping from the example
- `gA`, `gB`, `gC`, `gD` — global (GMEM) tile slices returned by `local_tile`.
- `tCgA = cta_mma.partition_A(gA)` — apply the `tC` partitioning pattern (CTA/MMA partitioning) to `gA`; result is a tensor partitioned for the MMA/CTA usage (shape printed in the example).
- `tCgB = cta_mma.partition_B(gB)` — same, for `gB`.
- `tCgC`, `tCgD` — partitioned versions of `gC`/`gD` (the C/D GMEM tiles).
- Related names in the file:
  - `tCsA`, `tCsB` — SMEM-backed tensors (SMEM views) for the same partitioning pattern.
  - `tCrA`, `tCrB` — fragment/descriptor views prepared for the MMA (register/fragment descriptors).
  - `tCtAcc` — tiled TMEM accumulator produced by `make_fragment_C`.

Why this is useful
- The compact mnemonic `tXgY` documents both what transform was done (`tX`) and what it was applied to (`gY`), so you can quickly see the provenance and role of each tensor in the kernel (GMEM → partition → SMEM → fragment → TMEM, etc.).

If you want, I can:
- Add a short comment block in 02_mma_tma_sm100.cu that documents this naming convention (makes examples easier to scan).
- Produce a one-line table of the common name tokens (`g`, `t`, `s`, `r`, `CtAcc`, etc.) and their meanings and insert it near the top of the example. Which would you prefer?

## PermutationMNK in AtomLayoutMNK
https://github.com/NVIDIA/cutlass/discussions/1345



## Tiled MMA and Copy

The Tiled MMA or Copy are tilings of MMA atoms resp. Copy atoms across threads and data, with possible permutations applied to the resulting tiling. This layer is most analogous to the warp level tiling of MMA instructions in CUTLASS 2.x. However, it views the tiling from the perspective of all threads participating in the operation and generalizes the concept to copy operations as well. The purpose of this layer is to build composable GPU micro-kernels out of a plethora of hardware accelerated math and data movement operations, each with their unit layouts in threads and data. The tiled MMA and Copy types present all these various hardware accelerated CuTe Atoms with a single, consistent API.

The resulting tiled operation acts as a single MMA or copy operation that users can invoke in the “inner” loop of the three-nested-loops pseudocode at the top of this document using `cute::gemm()` or `cute::copy()`.

We call this API “tiled” because it constructs larger operations out of the Atoms provided by CuTe, as if fitting together individual tiles to build a reusable component of a mosaic. For example, CuTe might provide an MMA Atom that users can call on a single warp, for fixed M, N, and K dimensions. CUTLASS can then use CuTe operations like `make_tiled_mma` to turn this Atom into an operation that works on an entire thread block, for larger M, N, and K dimensions.



Here we use a specialized function, `make_tmem_copy`, to deduce a TV-layout from the copy atom and TMEM tensor and create the TiledCopy. One important thing to know about this function is that *it is hardcoded to use 4 warps, or 1 warpgroup.* As mentioned in the earlier section, certain regions of TMEM are only accessible by a corresponding warp in a warpgroup, based on the warp index mod 4. This [diagram from the PTX manual](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#layout-d-m-128-cta-group-1) shows how the data is assigned to warps for our example:

![img](https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2025/04/tcgen05-data-path-layout-d1.png?resize=960%2C386&ssl=1)

