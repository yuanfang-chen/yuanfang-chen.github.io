---
layout: post
title:  "Blackwell Tensor Core"
#date:   2025-11-27 23:44:54 +0800
categories: CUDA
typora-root-url: ..
---

## 架构特性总览

1. **Blackwell Tensor Cores – tcgen05**

   	* 2x throughput vs Hopper for all Hoper types: FP16, BF16, TF32, INT8, FP8
		
   	* New block-scaled type support with mixed-inputs
		
   	* MXFP8 / MXFP6 - 2x throughput vs Hopper FP8
		
   	* MXFP4 - 4x throughput vs Hopper FP8
		
   	* Expanding Tensor Core execution to two SMs
		
   	* Fully asynchronous Tensor Core programming model

2. **Tensor Memory (TMEM)**

   * New memory on each SM with same capacity as the Register File

   * Used for Tensor Core inputs and outputs

3. **New Scheduling Capabilities**
   * Preferred Thread Block Clusters - CUDA grid with two Cluster configurations
   * Runtime Persistent Scheduling – dynamic mapping of output tiles to SMs. (using `clusterlaunchcontrol` PTX)

## Blackwell Tensor Cores – tcgen05

### UMMA `tcgen05.mma`

- Operand A can be in TMEM or SMEM
- Operand B must be in SMEM
- Accumulator must be in TMEM
- MMA instructions are available in shapes 64 x N x K with N a multiple of 8 and 128 x N x K with N a multiple of 16, where in both cases N is at most 256, K is expected to be 32 bytes wide for dense GEMM
- the data must be copied out to registers before storing or post-processing, and that each warp can only access ¼ of TMEM. This means that an entire warpgroup is required for the epilogue



> `tcgen05.mma.cta_group.kind  [d-tmem], a-desc, b-desc, idesc, enable-input-d;`
>
> `tcgen05.mma.cta_group.kind  [d-tmem], [a-tmem], b-desc, idesc, enable-input-d;`
>
> `.kind   = { .kind::f16, .kind::tf32, .kind::f8f6f4 }`
>
> `.cta_group = { .cta_group::1, .cta_group::2 }`

- The operands `a-desc` and `b-desc` are [shared memory descriptors](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#shared-memory-descriptor) (If A is sourced from TMEM, its descriptor is replaced by its TMEM address.) 
- The argument `idesc` is **instruction descriptor**
- The argument `enable-input-d` switches between zeroing out the accumulators before executing MMA (the operation D = A * B) and retaining the accumulators (D = A * B + D)

### CTA Pair

Two CTAs in a thread block cluster form a **CTA pair** if their CTA ranks in their thread block cluster differ by the last bit, e.g. 0 and 1, 4 and 5. A CTA pair maps to a Texture Processing Cluster (TPC), which consists of two SMs and combines with other TPCs to form a GPC. When Blackwell Tensor Core operations perform at a CTA pair granularity, the two CTAs are able to share input operands. This sharing reduces both SMEM capacity and bandwidth requirements.

### Block Scaling



## Tensor Memory

TMEM is 256KB per SM in size, and is organized 2-dimensionally in 512 columns and 128 rows, or **lanes**, of 32-bit cells. This inherent 2-D structure is reflected in the 32-bit addresses as well, where bits 31-16 denote the lane ID while 15-0 denote the column. This image from the [PTX documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#tensor-memory-addressing) shows the layout:

![img](https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2025/04/tensor-memory-layout.png?resize=960%2C540&ssl=1)

- TMEM is allocated dynamically using the `tcgen05.alloc` instruction
- TMEM must be explicitly deallocated with `tcgen05.dealloc`
- allocation is in units of columns
- The number of columns allocated must be a power of 2 and at least 32
- each warp can only access 32 of the 128 TMEM lanes
- TMEM is only for Tensor Core; SIMT operations are not supported on TMEM



The [shared memory descriptor](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#tcgen05-shared-memory-descriptor) describes the properties of multiplicand matrix in shared memory including its location in the shared memory of the current CTA. It is a 64-bit value contained in a register with the following layout:


The [instruction descriptor](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#tcgen05-instruction-descriptor) describes the shapes, types and other details of all the matrices and the matrix-multiplication-and-accumulation operation. It is a 32-bit value in registers and the exact layout is dependent on the MMA-Kind:

Typically, data gets into TMEM via UMMA operations, and is explicitly moved out to registers using tcgen05.ld for post-processing. It’s also possible for threads to manually load data into TMEM, either from SMEM through tcgen05.cp or from registers through tcgen05.st. However, TMEM access patterns for explicit load and store are very restricted. Each warp within a warpgroup can only access 32 lanes (with warp 0 associated to lanes 0-31, warp 1 to lanes 32-63, and so forth). Additionally, both the UMMA operation and the data movement operations expect certain data layouts. Luckily for us, CUTLASS provides utility functions that we’ll cover later that simplify the process of organizing data via swizzling. That said, those interested can find the layout information in the PTX guide.

Finally, besides UMMA operations and these data movement instructions, no other operations access data from TMEM. In other words, all pre-processing must happen before the data is loaded onto TMEM, and all post-processing must happen after the data is retrieved out of TMEM.



https://docs.nvidia.com/cuda/parallel-thread-execution/index.html?highlight=tcgen05%2520ld#tcgen05-memory-consistency-model



## Prefered Thread Block Cluster



## Swizzling



## References

- [[Blackwell Part1] CUTLASS Tutorial: Writing GEMM Kernels Using Tensor Memory For NVIDIA® Blackwell GPUs](https://research.colfax-intl.com/cutlass-tutorial-writing-gemm-kernels-using-tensor-memory-for-nvidia-blackwell-gpus/)
- [[Blackwell Part2] CUTLASS Tutorial: GEMM with Thread Block Clusters on NVIDIA® Blackwell GPUs](https://research.colfax-intl.com/cutlass-tutorial-gemm-with-thread-block-clusters-on-nvidia-blackwell-gpus/)
- [[Blackwell Part3] CUTLASS Tutorial: Sub-byte GEMM on NVIDIA® Blackwell GPUs](https://research.colfax-intl.com/cutlass-tutorial-sub-byte-gemm-on-nvidia-blackwell-gpus/)
- [Blackwell GPGPU架构新特性概览](https://zhuanlan.zhihu.com/p/32148105488)
