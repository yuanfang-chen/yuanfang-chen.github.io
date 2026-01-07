---
fylayout: post
title:  "CUDA Performance Analysis"
# date:   2025-11-21 11:18:26 -0800
categories: CUDA
typora-root-url: ..
mathjax: true
---

## 数学

### [幂定律](https://zh.wikipedia.org/wiki/%E5%86%AA%E5%AE%9A%E5%BE%8B)



## 定律

### Amdahl's law

- 该定律由 Gene Amdahl 提出，用于预测并行计算的理论最大加速比。
- **假设：问题大小不变，增加计算资源**
- 它指出，程序的加速比受到其**串行部分**所占比例的限制。即使可以无限增加并行处理器的数量，串行部分的处理时间也无法缩短，从而限制了整体性能的提升。

The fact that the level of speedup that one can achieve through

parallel execution can be severely limited by the parallelizable portion of the appli-

假设一个应用只有30%的执行时间可以并行执行，即使计算机有无限并行度，最多也只能优化30%的执行时间。

lesson learned: 应用的并行度有决定性作用。

### Gustafson's law

- 作为对 Amdahl 定律的补充，该定律由 John Gustafson 和 Edwin Barsis 提出。
- **假设：随着计算资源的增加，问题尺寸会变大**
- 它关注随着问题规模（而非固定的工作负载）的增大，并行计算的性能提升。它假设当处理器数量增加时，工作负载的总大小也会按比例增加，这更贴合实际应用中处理更大、更复杂问题的场景。
- lesson learned: 随着可用处理器数量的增加，用户倾向于扩大问题规模，以充分利用提升的计算能力，从而保持执行时间不变

### Little's law

- 虽然 Little 定律最初源于排队论，但它在并行和分布式系统中被广泛应用。

- 它描述了在一个稳定的系统中，平均客户数量（\\(L\\)）等于客户的平均到达率（\\(\lambda\\)）乘以客户在系统中花费的平均时间（\\(W\\)），即\\(L=\lambda W\\)。在计算领域，它有助于分析系统吞吐量、延迟和并行度之间的关系。

-  lesson learned: maximizing **instructions in flight (concurrency)** to hide memory/compute latency, ensuring GPU units (warps/cores) are always busy, especially by having enough active threads to saturate memory bandwidth. 但是单独通路的tensor core，导致little law适用SIMT core，不适用tensor core，tensor core只能使用流水掩盖

![img](/assets/images/little-law1.png)

![img](/assets/images/little-law2.png)

### Scaling law

- strong scaling
- weak scaling

### huang’s law



## 指标

### TOPS (On-Device AI)

**TOPS (Trillions of Operations Per Second)** measures **Inference**. This is how your laptop or phone runs an existing AI model (like live translation, background blur, or a local LLM like Llama 3) efficiently without draining your battery.

| **Chip Family**         | **NPU TOPS** | **Standout Feature**                                         |
| ----------------------- | ------------ | ------------------------------------------------------------ |
| **Snapdragon X2 Elite** | **80 TOPS**  | The efficiency king; leads in battery life and pure NPU throughput. |
| **Apple M5**            | **~40-50+**  | Not just the NPU; Apple added "Neural Accelerators" to every GPU core, giving it a **4x boost** in AI tasks over the M4. |
| **Intel Lunar Lake**    | **48 TOPS**  | Best for "Total Platform TOPS" (combining CPU + GPU + NPU) for complex agentic tasks. |
| **AMD Ryzen AI 300**    | **50 TOPS**  | The favorite for local gaming AI and heavy multi-threaded creative work. |

### FLOPS

**FLOPS (Floating Point Operations Per Second)** measures **Precision**. This is used by NVIDIA and Google to train the next generation of "Frontier Models" (GPT-5, Gemini 2, etc.). Since training requires high math accuracy, we measure it in **PetaFLOPS** ($10^{15}$ operations).

### The 2025 Heavyweights

- **NVIDIA Blackwell Ultra (B200):** The gold standard. It delivers roughly **20 PetaFLOPS** of FP8 performance. Its new **FP4 precision** mode allows it to run inference on massive trillion-parameter models twice as fast as the original Blackwell.
- **Google TPU v7 (Ironwood):** Google’s custom "scalpel." It hits about **4.6 PetaFLOPS**, but it is significantly more power-efficient than NVIDIA’s GPUs, allowing Google to scale "pods" of 9,000+ chips for massive training runs.

## The Big Picture

The big picture: “Feeding the beast”
There are 2 main actions in a GEMM kernel: copying the numbers to the correct memory addresses, and multiply-accumulating them. The former action is handled by copy instructions: TMA in Hopper, cp.async in Ampere, and vanilla copy in earlier architectures. The latter action, since the Volta architecture in 2017, has become the exclusive business of the tensor cores.

Through many generations, the tensor cores have become a beast at consuming the numbers fed to them. For instance, the H200 SXM GPU’s tensor cores can deliver up to 3,958 TFLOPS (TeraFLOPs per second). On the other hand, the memory bandwidth of the same H200 SXM GPU is only 4.8 TB/s (TeraBytes per second). This data transferring speed is much slower than the tensor cores’ speed, and oftentimes is not trivial to fully utilize! As such, a common theme of CUDA programming — and GEMM kernel design in particular — is to figure out how to copy numbers fast enough to keep the tensor cores busy. We call this process “feeding the beast.”

In general, there are two overarching strategies to “feed the beast,” which are complementary and function at different scopes (grid vs. block). 

- **The first strategy** is effective threadblock scheduling, which entails distributing the computation among the CTAs to obtain good load balancing and a higher rate of L2 cache hits. We will discuss this in a later blog post, but for now, we refer curious readers to the techniques of threadblock rasterization and persistent kernels, for instance as implemented in CUTLASS. 
- **The second strategy**, which we focus on in this tutorial, is to overlap copying with math operations. In particular, while the tensor cores are busy multiplying a batch of numbers that they receive, we should tell the copying units to copy the next batch of numbers. That way, we effectively hide part of the copying latency. This is the goal of pipelining.

![GEMM tiles are evenly divided among available SMs](/assets/images/gemm-hierarchy-with-epilogue.png)

大方向：先解决负载均衡的问题（且降低其overhead），再解决单核（SM）利用率。

## Persistent Kernel

What is **Wave**: In a GPU, a wave refers to a batch of thread blocks that can be assigned to all available streaming multiprocessors (SMs) at once. 

PS: 事实上，在同一时间，GPU上可能有多个Kernel在跑。所以一个Kernel所能使用的SM数量是无法静态确定的。

**without persistent kernel**

![GEMM tiles are evenly divided among available SMs](/assets/images/non_persistent.png)

**with persistent kernel**

![GEMM tiles are unevenly divided among available SMs, leading to workload imbalance](/assets/images/persistent_static.png)

```c++
// Non-persistent kernel
__device__ non_persistent_kernel(...) {
  setup_common_data_structures();
  dim3 workCoordinates = blockIdx;
  coordinate_specific_compute(workCoordinates);
}

// Static Persistent Kernel
__device__ static_persistent_kernel(...) {
  setup_common_data_structures(...);
  dim3 workCoordinates = blockIdx;
  bool isValidId;
  do {
    coordinate_specific_compute(workCoordinates);
    std::tie(isValidId, workCoordinates) = staticTileScheduler.fetch_next_work();
  } while (isValidId);
}
```


### [Quantization](https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html#dim-quantization) (SM之间切分)

#### Wave Quantization

An NVIDIA GPU consists of a number of streaming multiprocessors (SMs): each SM has its own shared memory, register file, Tensor Cores, etc., and they operate independently from each other. An ideal workload takes maximal advantage of parallelism between the SMs by evenly distributing work among the SMs, so that all SMs are kept busy for the entire duration of the kernel. However, if some SMs complete their portion quicker than others, then they will sit idle waiting for the rest of the SMs to complete. This is an example of **load imbalance**.

Consider a computation that is divisible into equally-sized **work units**, where each work unit can be completed by a single SM in the same amount of time. For example, GEMM is generally partitioned into work units that each compute a single bM x bN output tile. These work units are then assigned to threadblocks (CTAs), and each CTA computes its assigned work unit on an available SM. We will call the assignment of work units to SMs **scheduling**.

If the number of work units exceeds the number of available SMs, then the work units will be processed in multiple **waves**, where 1 wave is the completion of a single work unit by every available SM.

**Wave quantization** then arises when the number of work units isn’t evenly divisible by the number of available SMs. For example, consider a case where there are 10 work units, and 4 SMs. Then the work unit execution timeline looks like:

![img](https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2024/12/quantization.png?resize=590%2C540&ssl=1)



The impact of wave quantization decreases as the number of waves increases (AKA. as the number of threadblocks increase). However, increasing the number of waves can be difficult, especially considering that the number of SMs on NVIDIA GPUs continues to grow with newer architectures. So it is important that we come up with strategies to mitigate the impact of wave quantization without making assumptions on the problem size.

##### 增加更多的Work unit

增加更多work unit的方式之一是把bN减半。**缺点是这样会降低[arithmetic intensity](https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html#math-mem)**，从而导致受益不够。

![img](https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2024/12/split-mn-more-tiles.png?resize=2053%2C1243&ssl=1)

##### sliced-k 

> **reduction across warps on shared memory in CTA**

优点：提升单个 SM 的资源利用率来改善性能

缺点：解决不了M/N小，K大的问题

![img](https://picx.zhimg.com/v2-c7adbe37ee93405af0c0917fafcd87b5_1440w.jpg)

##### split-k

> **reduction across CTAs**



![img](https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2024/12/split-k.png?resize=2006%2C1227&ssl=1)



##### **stream-k **

stream-k是一种负载均衡的手段，其实现利用了persistent kernel。但是persistent kernel本身和负载均衡无关。

[Our code sample on GitHub](https://github.com/ColfaxResearch/cfx-article-src/tree/master/streamk) provides three examples of schedulers: a trivial non-persistent scheduler that assigns 1 worktile to each CTA over a grid determined by the problem shape; a data-parallel persistent scheduler; and a Stream-K hybrid scheduler which incorporates a some but not all of CUTLASS’s optimizations. In practice, we found that many of CUTLASS’s optimizations were necessary to get reasonable performance: notably, the additional GMEM accesses and smaller tile sizes caused by reduction are a real cost, and the boundaries of Stream-K work assignments need to be carefully tweaked to minimize this cost.







![img](https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2024/12/stream-k.png?resize=1768%2C1104&ssl=1)

##### Hybrid Stream-K

naive stream-k is bad for cache

![img](https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2024/12/hybrid.png?resize=1906%2C1202&ssl=1)

![img](https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2024/12/m1024-no-heuristic.png?resize=720%2C540&ssl=1)

The vertical dotted lines denote the wave boundaries. As expected, there is a sharp drop in performance for the DataParallel mode when going over wave boundaries. This is the wave quantization effect. The DataParallel mode matches or outperforms all other modes when the last wave is mostly full (tiles-per-SM is just under a whole integer), and underperforms when it is nearly empty (tiles-per-SM is just over a whole integer). Finally, we can see that the wave quantization effect is the most pronounced when the total number of waves is low.



## [Blackwell Cluster Launch Control (CLC)](https://docs.nvidia.com/cutlass/media/docs/cpp/blackwell_cluster_launch_control.html)

persistent kernel限制在于：在运行时（realtime）无法确切知道可以利用的 SM 数量。某些 SM 可能已被其他内核占用，因此其资源不可用。这使得在各个 SM 之间实现负载均衡变得十分困难。

```c++
// Dynamic Persistent Kernel
__device__ clc_dynamic_persistent_kernel(...) {
  setup_common_data_structures(...);
  dim3 workCoordinates = blockIdx;
  dim3 newClcID;
  bool isValidId;
  do {
    coordinate_specific_compute(workCoordinates);
    std::tie(isValidId, newClcID) = clcTileScheduler.fetch_next_work();
    workCoordinates = newClcID;
  } while (isValidId);
}
```

https://www.modular.com/blog/matrix-multiplication-on-blackwell-part-4---breaking-sota





## Threadblock Rasterization (6 SMs)

An advantage of persistent kernels independent of the wave quantization issue is the ability to choose the order in which worktiles are launched. 

**Rasterization along M**

![rasterization along M](https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2024/12/rasterization-2.png?resize=503%2C540&ssl=1)

**Rasterization along M&N**

Left; rasterization along M with swizzle 2. Right; rasterization along M with swizzle 1.

<img src="https://i0.wp.com/research.colfax-intl.com/wp-content/uploads/2024/12/swizzle-2.png?resize=960%2C506&ssl=1" alt="img" style="zoom: 67%;" />



## Pipelining

### Warp-specialization
Specializing warps into producers (data transfer) and consumers (compute), and having them run concurrently.

Warp Specialization is introduced in Hopper.

A standard GPU program executes the same logic on each warp, while a warp specialized program uses different warps to execute different components of the overall program. Let’s take a look at some of these warp specialization strategies in the aforementioned contexts.

在一个warp内，warp divergence

- **[CUDA-DMA](https://lightsighter.org/pdfs/cudadma-sc11.pdf)**: separated the warps into memory loading (GMEM->SMEM) warps and compute warps; the loader warps issue loads and signal the compute warps when the loaded data is available.
- **[Singe compiler](https://cs.stanford.edu/~sjt/pubs/ppopp14.pdf)**: 
- **CUDA Tensor Core**: Specialized warps are used on Hopper and Blackwell to issue either TMA copies or Tensor Core matrix-multiplies. The TMA warp issue copies and notifies the Tensor Core warps when data is ready to be multiplied, and the Tensor Core warps notify the TMA warp when data has been consumed and the memory is free to use for more copies.
- **[high performance Flash Attention implementation on Blackwell](https://github.com/NVIDIA/cutlass/tree/a49a78ffefc86a87160dfe0ccc3a3a2d1622c918/examples/77_blackwell_fmha)**: uses at least 5 different kinds of specialized warps! In this Flash Attention implementation, there are warps for loading data, issuing matrix multiplication, computing softmax, scaling intermediate results, and storing data. As a result, the code is complex; the strategy itself is carefully constructed to yield high performance, and there is abundant cross-warp data movement and synchronization. Imagine the code above with 5 different warp cases and each cases signaling the others to proceed at different times! (多级流水：类似Asend NPU；有点像CPU流水线了？？)

With warp-specialization, some warps are dedicated to memory fetches (producers), while others are dedicated to compute (consumers), and named barriers are used for synchronization between them. The idea is that the warp schedulers can then more easily hide the latency of copy operations within compute (and vice-versa).

```cpp
if warpid() == LOAD:
  for i, tile in enumerate(tiles):
    if i > 0:
      wait_for_tile_release()
    async_tma_load(tile)
    wait_for_tma_load()
    signal_tile_loaded()
else:
  for tile in enumerate(tiles):
    wait_for_tile_loaded()
    tile_data = get_loaded_tile(tile)
    async_mma(tile_data)
    wait_for_async_mma()
    signal_tile_released()
```

![Figure 2: An overview of the Ping-Pong Kernel pipeline. Time moves left to right.](https://pytorch.org/wp-content/uploads/2024/11/image-9.png)

![Figure 3: An overview of the full async pipeline for Ping-Pong](https://pytorch.org/wp-content/uploads/2024/11/image-5.png)

![img](https://picx.zhimg.com/v2-8633e0b9f54f88a1a4eb7053c463980f_r.jpg)

### Multistage

Masking data transfer by using asynchronous copy (TMA on Hopper or `cp.async` on Ampere) to load the next set of data, while computing on the current set. Warps take on both producer and consumer roles.

![img](https://i0.wp.com/github.com/NVIDIA/cutlass/raw/main/media/images/software-pipeline.png?ssl=1)



## [Grouped Kernel Schedulers](https://docs.nvidia.com/cutlass/media/docs/cpp/grouped_scheduler.html)

CUTLASS’s grouped kernel is a persistent kernel which launches multiple problems (e.g., GEMMs, SYR2Ks) within a single CUDA kernel launch.

8 threadblocks

![ALT](/assets/images/grouped-gemm-schedule-2x2.png)

![ALT](/assets/images/grouped-gemm-schedule-varied.png)

## Preferred Thread Block Clusters

- cudaLaunchAttributePreferredClusterDimension = 11

  Valid for graph nodes and launches. Set [cudaLaunchAttributeValue::preferredClusterDim](https://docs.nvidia.com/cuda/cuda-runtime-api/unioncudaLaunchAttributeValue.html#unioncudaLaunchAttributeValue_1bf53f6cb9ba3e18833d99c51a2568df5) to allow the kernel launch to specify a preferred substitute cluster dimension. Blocks may be grouped according to either the dimensions specified with this attribute (grouped into a "preferred substitute cluster"), or the one specified with [cudaLaunchAttributeClusterDimension](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1ggfc5ed48085f05863b1aeebb14934b0563de2ec80489ee6a8010335db69abc3ff) attribute (grouped into a "regular cluster"). The cluster dimensions of a "preferred substitute cluster" shall be an integer multiple greater than zero of the regular cluster dimensions. The device will attempt - on a best-effort basis - to group thread blocks into preferred clusters over grouping them into regular clusters. When it deems necessary (primarily when the device temporarily runs out of physical resources to launch the larger preferred clusters), the device may switch to launch the regular clusters instead to attempt to utilize as much of the physical device resources as possible. Each type of cluster will have its enumeration / coordinate setup as if the grid consists solely of its type of cluster. For example, if the preferred substitute cluster dimensions double the regular cluster dimensions, there might be simultaneously a regular cluster indexed at (1,0,0), and a preferred cluster indexed at (1,0,0). In this example, the preferred substitute cluster (1,0,0) replaces regular clusters (2,0,0) and (3,0,0) and groups their blocks. This attribute will only take effect when a regular cluster dimension has been specified. The preferred substitute cluster dimension must be an integer multiple greater than zero of the regular cluster dimension and must divide the grid. It must also be no more than `maxBlocksPerCluster`, if it is set in the kernel's `__launch_bounds__`. Otherwise it must be less than the maximum value the driver can support. Otherwise, setting this attribute to a value physically unable to fit on any particular device is permitted.

![img](https://pic1.zhimg.com/v2-e0e5e0946fe8709bd777464df7e948d8_1440w.jpg)

## [Dependent kernel launches](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#programmatic-dependent-launch-and-synchronization)

The *Programmatic Dependent Launch* mechanism allows for a dependent *secondary* kernel to launch before the *primary* kernel it depends on in the same CUDA stream has finished executing. Available starting with devices of compute capability 9.0, this technique can provide performance benefits when the *secondary* kernel can complete significant work that does not depend on the results of the *primary* kernel.

![GPU activity timeline](/assets/images/gpu-activity.png)

![Concurrent execution of ``primary_kernel`` and ``secondary_kernel``](/assets/images/preamble-overlap.png)

## NCCL



## 优化手段

### [vectorized load/store](https://developer.nvidia.com/blog/cuda-pro-tip-increase-performance-with-vectorized-memory-access/)

SASS中`LDG.E` and `STG.E` 指令从GMEM load and store 32 bits数据，不是vectorized load/store.

`LDG.E.{64,128}` and `STG.E.{64,128}`. 指令从GMEM load and store 64/128 bits数据，是vectorized load/store

这个例子没有用到vectorized load/store

```c++
__global__ void device_copy_scalar_kernel(int* d_in, int* d_out, int N) { 
  int idx = blockIdx.x * blockDim.x + threadIdx.x; 
  for (int i = idx; i < N; i += blockDim.x * gridDim.x) { 
    // 先从GMEM load到RMEM，再从RMEM store到GMEM
    d_out[i] = d_in[i];  // <<=== LDG.E R3, desc[UR6][R2.64] ; STG.E desc[UR6][R4.64], R3 ; 
  } 
} 
 
void device_copy_scalar(int* d_in, int* d_out, int N) 
{ 
  int threads = 256; 
  int blocks = min((N + threads-1) / threads, MAX_BLOCKS);  
  device_copy_scalar_kernel<<<blocks, threads>>>(d_in, d_out, N); 
}
```

这个例子用到了vectorized load/store: `LDG.E.64` and `STG.E.64`.

```c++
__global__ void device_copy_vector2_kernel(int* d_in, int* d_out, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  for (int i = idx; i < N/2; i += blockDim.x * gridDim.x) {
    reinterpret_cast<int2*>(d_out)[i] = reinterpret_cast<int2*>(d_in)[i];
  }
 
  // in only one thread, process final element (if there is one)
  if (idx==N/2 && N%2==1)
    d_out[N-1] = d_in[N-1];
}
 
void device_copy_vector2(int* d_in, int* d_out, int n) {
  threads = 256; 
  blocks = min((N/2 + threads-1) / threads, MAX_BLOCKS); 
 
  device_copy_vector2_kernel<<<blocks, threads>>>(d_in, d_out, N);
}
```

这个例子用到了vectorized load/store: `LDG.E.128` and `STG.E.128`.

```c++
__global__ void device_copy_vector4_kernel(int* d_in, int* d_out, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  for(int i = idx; i < N/4; i += blockDim.x * gridDim.x) {
    reinterpret_cast<int4*>(d_out)[i] = reinterpret_cast<int4*>(d_in)[i];
  }
 
  // in only one thread, process final elements (if there are any)
  int remainder = N%4;
  if (idx==N/4 && remainder!=0) {
    while(remainder) {
      int idx = N - remainder--;
      d_out[idx] = d_in[idx];
    }
  }
}
 
void device_copy_vector4(int* d_in, int* d_out, int N) {
  int threads = 256;
  int blocks = min((N/4 + threads-1) / threads, MAX_BLOCKS);
 
  device_copy_vector4_kernel<<<blocks, threads>>>(d_in, d_out, N);
}
```

### Occupancy

CUDA occupancy is a metric that represents the ratio of **active warps** per multiprocessor (SM) to the **maximum possible number of active warps**.1 Higher occupancy generally allows the GPU to better hide "latency" (the time spent waiting for memory or math operations to finish) by switching to other warps that are ready to work.2



Occupancy is calculated based on four primary hardware constraints.

------

#### 1. The Core Calculation

The formula is straightforward, but the variables are determined by hardware limits:

\\(\text{Occupancy} = \frac{\text{Active Warps Per SM}}{\text{Maximum Warps Per SM}}\\)

To find the "Active Warps," you must determine which of the following three resources runs out first:

1. **Registers** (per thread)
2. **Shared Memory** (per block)
3. **Thread Block Limit** (per SM)

------

#### 2. The Three Determining Factors

##### A. Register Pressure

Each SM has a fixed pool of registers (e.g., 64K registers). If your kernel uses many registers per thread, the GPU cannot fit as many threads onto the SM at once.

- **Limit:** \\(\text{Active Blocks} \le \frac{\text{Total Registers per SM}}{\text{Registers per Thread} \times \text{Threads per Block}}\\)

##### B. Shared Memory (SMem)

Shared memory is allocated per thread block. If a kernel requires a large amount of shared memory, fewer blocks can reside on the SM.3



- **Limit:** \\(\text{Active Blocks} \le \frac{\text{Total Shared Memory per SM}}{\text{Shared Memory per Block}}\\)

##### C. Max Threads/Blocks per SM

Every GPU architecture (Pascal, Ampere, Hopper, etc.) has hard limits regardless of how small your kernel is. For example, a common limit is **32 blocks** or **2048 threads** per SM.

------

#### 3. How to Calculate It (Tools)

Calculating this by hand is difficult because hardware specs change between GPU generations. Instead, use these three methods:

##### Method 1: The CUDA Occupancy Calculator (Excel)

NVIDIA provides an [Excel Spreadsheet](https://www.google.com/search?q=https://docs.nvidia.com/cuda/cuda-occupancy-calculator/index.html) where you input your GPU's Compute Capability (e.g., 8.6 for an RTX 30-series) and your kernel's resource usage (found via `nvcc --ptxas-options=-v`).

##### Method 2: Nsight Compute (Recommended)

This is the modern way to profile. Run your code through Nsight Compute, and it will give you a "Theoretical Occupancy" vs. "Achieved Occupancy" graph.

- **Theoretical:** The max occupancy possible based on your code's resource usage.
- **Achieved:** What actually happened during the run (often lower due to tail-end effects or uneven workloads).

##### Method 3: API-based Calculation

You can query the occupancy directly in your C++ code to dynamically choose a block size:

C++

```c++
int minGridSize;
int blockSize;
cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, myKernel, 0, 0);
```

------

#### 4. Is Higher Occupancy Always Better?

**Not necessarily.** * **The "Cliff" Effect:** Increasing occupancy from 10% to 50% usually gives a massive speedup.

- **Diminishing Returns:** Moving from 50% to 100% often provides little benefit, and can sometimes slow down the kernel if it causes cache thrashing.
- **ILP (Instruction Level Parallelism):** Some kernels are faster with low occupancy but high work-per-thread (using more registers to do more work in a single thread).

## Perf/W

Perf/W, or Performance per Watt (PPW), is a key energy efficiency metric in computing, measuring how much computational work (performance) a system achieves for each unit of electrical power (watt) it consumes, crucial for data centers, mobile devices, and sustainable computing to balance speed with energy cost and environmental impact. It's often used with Linux's perf tool, which can track performance events and power usage (like Intel RAPL) to calculate this ratio, helping optimize hardware for maximum efficiency, notes Supermicro, Wikipedia, and Medium. 
How it works
Performance: Measured by benchmarks like LINPACK (for supercomputers) or application-specific metrics (e.g., instructions per cycle, operations per second).
Wattage: Measured by power monitoring features in the hardware, often accessed via Linux's perf tool using events like power/energy-pkg/ or power/energy-cores/.
The Ratio: Performance (e.g., GFLOPS) / Power (Watts) = Performance per Watt (e.g., GFLOPS/W). 
Why it's important
Cost & Environment: Lowers energy bills and carbon footprint in large-scale computing.
Device Constraints: Critical for spaceflight and mobile devices with limited power budgets.
Hardware Selection: Guides designers to choose CPUs/GPUs that offer more speed for less power.

## GF/sample
In Deep Learning (DL), GF/sample stands for GigaFLOPs per sample.

## Model FLOPs Utilization


## References

- [NVIDIA Deep Learning Performance](https://docs.nvidia.com/deeplearning/performance/index.html)

- [GPU Glossary - Performance](https://modal.com/gpu-glossary/perf)

- [Roofline: an insightful visual performance model for multicore architectures](https://people.eecs.berkeley.edu/~kubitron/cs252/handouts/papers/RooflineVyNoYellow.pdf)

- [Quantitative System Performance](https://homes.cs.washington.edu/~lazowska/qsp/)

- [How to Scale Your Model](https://jax-ml.github.io/scaling-book/)

- [cutlass GEMM——sliced-K、split-K & stream-K 分析 （一）](https://zhuanlan.zhihu.com/p/713411778)

- [Latency Numbers Every Programmer Should Know](https://colin-scott.github.io/personal_website/research/interactive_latency.html)

- [Building Machine Learning Systems for a Trillion Trillion Floating Point Operations](https://www.youtube.com/watch?v=139UPjoq7Kw&t=1229s)

- [Strangely, Matrix Multiplications on GPUs Run Faster When Given "Predictable" Data! [short]](https://www.thonking.ai/p/strangely-matrix-multiplications)

- [CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide)

- [Deep Dive on CUTLASS Ping-Pong GEMM Kernel](https://pytorch.org/blog/cutlass-ping-pong-gemm-kernel/)

- [Inside NVIDIA GPUs: Anatomy of high performance matmul kernels](https://www.aleksagordic.com/blog/matmul)

- [Accelerating HPC Applications with NVIDIA Nsight Compute Roofline Analysis](https://developer.nvidia.com/blog/accelerating-hpc-applications-with-nsight-compute-roofline-analysis/)

- [Techniques for training large neural networks](https://openai.com/index/techniques-for-training-large-neural-networks/)

- [The Technology Behind BLOOM Training](https://huggingface.co/blog/bloom-megatron-deepspeed)

- [Optimization Techniques for GPU Programming](https://dl.acm.org/doi/10.1145/3570638)

- https://d2l.ai/chapter_computational-performance/index.html

- [CUDA Pro Tip: Increase Performance with Vectorized Memory Access](https://developer.nvidia.com/blog/cuda-pro-tip-increase-performance-with-vectorized-memory-access/)

- [S72683 - CUDA Techniques to Maximize Memory Bandwidth and Hide Latency](https://www.nvidia.com/en-us/on-demand/session/gtc25-s72683/)

- [S72685 - CUDA Techniques to Maximize Compute and Instruction Throughput](https://www.nvidia.com/en-us/on-demand/session/gtc25-s72685/)

- [S72686 - CUDA Techniques to Maximize Concurrency and System Utilization](https://www.nvidia.com/en-us/on-demand/session/gtc25-s72686/)

- [S51413 - Developing Optimal CUDA Kernels on Hopper Tensor Cores](https://www.nvidia.com/en-us/on-demand/session/gtcspring23-s51413/)

- [S61198 - CUTLASS: A Performant, Flexible, and Portable Way to Target Hopper Tensor Cores](https://www.nvidia.com/en-us/on-demand/session/gtc24-s61198/)

- https://github.com/stas00/ml-engineering/blob/master/training/performance/README.md

- https://hanlab.mit.edu/courses/2023-fall-65940

- 

  
