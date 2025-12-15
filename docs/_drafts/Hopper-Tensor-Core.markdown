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



## Tensor Memory Accelerator (TMA) (PTX `cp.async.bulk.tensor`)

The NVIDIA Hopper Architecture provides new features that improve asynchronous execution and enable further overlap of memory copies with computation and other independent work, while also minimizing synchronization points. We describe the new async memory copy unit called the Tensor Memory Accelerator (TMA) and a new asynchronous transaction barrier.

> In historical context, these developments continue a trend of replacing general-purpose computational resources by specialized hardware resources, to both remove bottlenecks and free up those general-purpose resources for other operations. Starting with the Volta architecture, the Tensor Cores divorced GEMM arithmetic operations from the general computational pipeline. Ampere’s asynchronous copy instructions allowed for true pipelining of GEMM mainloops. On Hopper GPUs, the asynchronous, single-threaded TMA and the ability to reallocate registers between warpgroups dramatically reduced the register and thread cost of data movement, and the asynchronous WGMMA allowed for pipelining of MMA with other compute operations. Now, Tensor Memory and UMMA do for MMA just what TMA did for copy, making it a single-threaded, asynchronous operation that does not consume registers. As a result, registers can primarily be used for other tasks like scheduling and fused epilogue operations.



### why need TMA

Many applications require movement of large amounts of data from and to global memory. Often, the data is laid out in global memory as a multi-dimensional array with non-sequential data acess patterns. To reduce global memory usage, sub-tiles of such arrays are copied to shared memory before use in computations. The loading and storing involves address-calculations that can be error-prone and repetitive. To offload these computations, Compute Capability 9.0 introduces the Tensor Memory Accelerator (TMA). The primary goal of TMA is to provide an efficient data transfer mechanism from global memory to shared memory for multi-dimensional arrays.

Dimensions. TMA supports copying both one-dimensional and multi-dimensional arrays (up to 5-dimensional). The programming model for bulk-asynchronous copies of one-dimensional contiguous arrays is different from the programming model for bulk tensor asynchronous copies of multi-dimensional arrays. To perform a bulk tensor asynchronous copy of a multi-dimensional array, the hardware requires a tensor map. This object describes the layout of the multi-dimensional array in global and shared memory. A tensor map is typically created on the host using the cuTensorMapEncode API and then transferred from host to device as a const kernel parameter annotated with __grid_constant__. The tensor map is transferred from host to device as a const kernel parameter annotated with __grid_constant__, and can be used on the device to copy a tile of data between shared and global memory. In contrast, performing a bulk-asynchronous copy of a contiguous one-dimensional array does not require a tensor map: it can be performed on-device with a pointer and size parameter.


- GPU的效率最大化依赖充分利用core的算力和内存带宽
- 如果算力（即MMA）需要绑定寄存器，那可能带来SM occupyancy不足的问题，进一步导致无法launch足够的warp来掩盖memory latency

### How to choose

As the PTX instructions name suggests:

- Use `cp.async` for transferring small amount of data
- Use `cp.async.bulk` for transferring large amount of data



### 访问方法

TMA 使用**张量描述符**（Tensor Descriptor，或 Tensor Map）替代显式地址计算，开发者只需定义“搬什么”，而无需关心“怎么搬”，避免与复杂的物理地址接触。描述符是储存在 GMEM 中的一个数据结构，包含张量的所有属性。如果输入布局发生更改，修改 GMEM 中的描述符即可，无需重新修改编译整个 kernel。

TMA 以 **Bounding Box** 为单位搬运数据，一个 thread 发起一次 TMA 搬运操作，搬运一个 Box。Box 是张量中抠出的一个块，维度与张量相同。TMA 支持 1~5 维张量。

如下图所示，蓝色立方体是原始张量，绿色立方体是一次 TMA 操作搬运的 Box。因为访问 Box 地址必须 16Byte 对齐，但张量的边长可能是非对齐的，所以要求张量加上 padding，让每个方向上 tensor 第一个元素都是 16B 对齐的，透明立方体描述的就是张量做 16B 对齐 padding 后。

![img](https://pic3.zhimg.com/v2-3b8790b6a802d9b2ba48973bb5fde344_1440w.jpg)

因此，在描述符中指定张量的起始地址、数据类型、搬运模式，以及蓝色、绿色、透明立方体的参数，就可以指定 TMA 如何访问张量。具体有以下参数：

- **tensorDataType**: 元素的数据类型。
- **tensorRank**: 张量维度数，支持 1D~5D。
- **globalAddress**: 张量的内存起始地址，需 16B 对齐。
- **globalDim**: 一个数组，存放张量在每个维度的长度（图中的 TensorSize）。
- **globalStrides**: 一个数组，存放张量在每个维度 padding 后的长度，即在每个维度上连续两行张量首地址之间的间隔（图中的 TensorStride），需要 16B 对齐。
- **boxDim**: 一个数组，存放 box 在每个维度的长度（图中的 AccessBoxSize），需要 16B 对齐。
- **elementStride**: 一个数组，存放 box 在每个维度的间隔，即在每个维度上连续两个 box 首地址之间的间隔（图中的 TraversalStride），box 之间是可以重叠的。需要 16B 对齐。
- **interleave、swizzle**: 用于改变数据排列的顺序，下文介绍。
- **L2Promotion**: 用于 L2 cache 取数据优化。
- **oobFill**: 当访问的 box 超过张量的范围时，会填充为 0 或 NaN。如下图黄色区域。

![img](https://pica.zhimg.com/v2-8bea81533ab7feea5deb508fda7d906c_1440w.jpg)





### Tensor Map

The primary difference between the one-dimensional and multi-dimensional case is that a tensor map must be created on the host and passed to the CUDA kernel. 

[cuTensorMap](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html#group__CUDA__TENSOR__MEMORY)

| Method                                                       | Description                                                  |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| [cuTensorMapEncodeIm2col](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html#group__CUDA__TENSOR__MEMORY_1gb14d707a18d23fc0c3e22a67ceedc15a) | Create a tensor map descriptor object representing im2col memory region. |
| [cuTensorMapEncodeIm2colWide](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html#group__CUDA__TENSOR__MEMORY_1g6c1be81856c4e311f085e33a42403444) | Create a tensor map descriptor object representing im2col memory region, but where the elements are exclusively loaded along the W dimension. |
| [cuTensorMapEncodeTiled](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html#group__CUDA__TENSOR__MEMORY_1ga7c7d2aaac9e49294304e755e6f341d7) | Create a tensor map descriptor object representing tiled memory region. |
| [cuTensorMapReplaceAddress](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html#group__CUDA__TENSOR__MEMORY_1g8d54c0ff5c49b1b1a9baaac6fc796db3) | Modify an existing tensor map descriptor with an updated global address. |

### PTX

PTX 是英伟达开放给用户的虚拟汇编语言，可以理解为是 Cuda 代码转换为真正的 GPU 硬件指令的中间层。CUTLASS 中对 TMA 的操作其实就是调用相应的 PTX 指令。

与 TMA 相关的 PTX 指令有以下几类：

1. **Bulk Copy / Reduce / Prefetch**（非张量的搬运）

   ```text
   cp.async.bulk
   ```

   ```text
   cp.reduce.async.bulk
   ```

   ```text
   cp.async.bulk.prefetch
   ```

2. **[Tensor Copy](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-bulk-tensor) / Reduce / Prefetch**（多维张量搬运）

   ```text
   cp.async.bulk.tensor
   ```

   ```text
   cp.reduce.async.bulk.tensor
   ```

   ```text
   cp.async.bulk.prefetch.tensor
   ```

3. **同步管理指令**

   ```text
   cp.async.bulk.commit_group
   ```

   ```text
   cp.async.bulk.wait_group
   ```

4. **修改已存在的 Tensor map (描述符)字段**

   ```text
   tensormap.replace
   ```

### TMA Load

**The two-step process.** To perform this task, we use TMA load. In CuTe, a TMA load operation is implemented in two steps. The first step is the construction of the TMA copy descriptor in the *host code*, while the second step is the execution of the actual TMA load using this descriptor inside the *kernel code.*

https://research.colfax-intl.com/tutorial-hopper-tma/

#### Host代码

```c++
template <int TILE_M = 128, int TILE_N = 128, int THREADS = 32>
int host_fn(int M, int N, int iterations = 1) {
  using namespace cute;

  using Element = float;

  auto tensor_shape = make_shape(M, N);

  // Allocate and initialize host and device tensors
  thrust::host_vector<Element> h_S(size(tensor_shape)); // (M, N)
  thrust::host_vector<Element> h_D(size(tensor_shape)); // (M, N)

  for (size_t i = 0; i < h_S.size(); ++i)
    h_S[i] = static_cast<Element>(float(i));

  thrust::device_vector<Element> d_S = h_S;
  thrust::device_vector<Element> d_D = h_D;

  //
  // Make tensors
  //

  auto gmemLayoutS = make_layout(tensor_shape, LayoutRight{});
  auto gmemLayoutD = make_layout(tensor_shape, LayoutRight{});
  Tensor tensor_S = make_tensor(
      make_gmem_ptr(thrust::raw_pointer_cast(d_S.data())), gmemLayoutS);
  Tensor tensor_D = make_tensor(
      make_gmem_ptr(thrust::raw_pointer_cast(d_D.data())), gmemLayoutD);

  using bM = Int<TILE_M>;
  using bN = Int<TILE_N>;

  auto tileShape = make_shape(bM{}, bN{});
  // NOTE: same smem layout for TMA load and store
  auto smemLayout = make_layout(tileShape, LayoutRight{});
  auto tma_load = make_tma_copy(SM90_TMA_LOAD{}, tensor_S, smemLayout);
  auto tma_store = make_tma_copy(SM90_TMA_STORE{}, tensor_D, smemLayout);

  Params params(tma_load, tma_store, gmemLayoutS, smemLayout, tileShape);

  dim3 gridDim(ceil_div(M, TILE_M), ceil_div(N, TILE_N));
  dim3 blockDim(THREADS);

  int smem_size = int(sizeof(SharedStorageTMA<Element, decltype(smemLayout)>));

  void const *kernel =
      (void const *)copyTMAKernel<THREADS, Element, decltype(params)>;
  cfk::utils::set_smem_size(smem_size, kernel);

  dim3 cluster_dims(1);

  // Define the cluster launch parameter structure.
  cutlass::ClusterLaunchParams launch_params{gridDim, blockDim, cluster_dims,
                                             smem_size};
  cutlass::Status status =
        cutlass::launch_kernel_on_cluster(launch_params, kernel, params);
}
```



#### Kernel代码 (load)

```c++
template <typename T, int CTA_M, int CTA_N, class TmaLoad, class GmemTensor>
void tma_load_kernel(__grid_constant__ const TmaLoad tma_load, GmemTensor gmem_tensor) {
  using namespace cute;
  constexpr int tma_transaction_bytes = CTA_M * CTA_N * sizeof(T);
 
  __shared__ T smem_data[CTA_M * CTA_N];
  __shared__ uint64_t tma_load_mbar;
 
  auto smem_layout = make_layout(make_shape(CTA_M, CTA_N), LayoutRight{});
  auto smem_tensor = make_tensor(make_smem_ptr(smem_data), smem_layout);
 
  if (threadIdx.x == 0) {
    auto gmem_tensor_coord = tma_load.get_tma_tensor(shape(gmem_tensor));
 
    auto gmem_tensor_coord_cta = local_tile(
        gmem_tensor_coord,
        Tile<Int<CTA_M>, Int<CTA_N>>{},
        make_coord(blockIdx.x, blockIdx.y));
 
    initialize_barrier(tma_load_mbar, /* arrival count */ 1);
 
    set_barrier_transaction_bytes(tma_load_mbar, tma_transaction_bytes);
 
    auto tma_load_per_cta = tma_load.get_slice(0);
    copy(tma_load.with(tma_load_mbar),
         tma_load_per_cta.partition_S(gmem_tensor_coord_cta),
         tma_load_per_cta.partition_D(smem_tensor));
  }
  __syncthreads();
  wait_barrier(tma_load_mbar, /* phase */ 0);
 
  // after this line, the TMA load is finished
}
```



#### Kernel代码（store）

```c++
```

#### Kernel代码 （store reduce）

典型的使用场景是Split-K GEMM

```c++
// original: create a TMA store object
auto tma_store = make_tma_copy(SM90_TMA_STORE{}, gmem_tensor, smem_layout);
 
// to create a TMA reduce sum object
auto tma_reduce_sum = make_tma_copy(SM90_TMA_REDUCE_ADD{}, gmem_tensor, smem_layout);
```





### [Multicast Support](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#multicast-support)

As a brief review, TMA multicast places the data loaded by the TMA in the SMEM of multiple CTAs in the same cluster. Using this feature, a set of CTAs in a cluster can collaboratively and simultaneously load a tile of data into each of their shared memories, decreasing the global memory traffic in cases when multiple CTAs need to load the same data. Each CTA loads a portion of the data that is multicast into the SMEM of the other participating CTAs. For example, if the number of participating CTAs is 4, each CTA loads a quarter of the data, thereby reducing the total amount of data loaded with TMA by a factor of 4. Technically, this collaborative partial loading is a programming paradigm and is not intrinsic to TMA multicast feature, but in this article we will treat them as synonymous.

## Warpgroup MMA (WGMMA)

- 异步操作
- 连续128个线程完成，首个warp的rank必须是4的整数倍
- PTX `wgmma.mma_async`
- `A`可以在SMEM或者RMEM；`B`必须在SMEM；`C`必须在RMEM

### format
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

- pre-Hopper MMAs operates on threads
- Blackwell MMAs operates on CTAs



PTX:

1. **wgmma.fence** : Enforce an ordering of register accesses between `wgmma.mma_async` and other operations.
2. **wgmma.commit_group** : Commits all prior uncommitted `wgmma.mma_async` operations into a *wgmma-group*.
3. **wgmma.wait_group** : Signal the completion of a preceding warpgroup operation.

cutlass:

1. **warpgroup_fence_operand**: It forces the compiler to treat the given register(s) (or a fragment made of registers) as a live read/write operand at a synchronization point so the compiler does not optimize-away or reorder those registers across the GMMA warpgroup fence/commit/wait sequence.





We explain these points in order. First, a `wgmma.fence` instruction ensures that `wgmma.mma_async` only accesses certain RMEM addresses after all prior accesses to such addresses have finished. Without the `wgmma.fence`, the behavior is undefined. An exception to this rule is that Hopper allows *multiple* `wgmma.mma_async` instructions to be in flight simultaneously. As long as these `wgmma.mma_async` instructions have the same accumulator shape, they can share the same accumulator tensor, i.e., write to the same register memory addresses. In that case, a fence is not required. For example, we don’t need to insert a `wgmma.fence` within the loop over `MMA_K` done as part of the `cute::gemm` call.



```cpp
cute::warpgroup_arrive();
cute::gemm(tiled_mma, tCrA(_,_,_,read_pipe), tCrB(_,_,_,read_pipe), tCrC);
cute::warpgroup_commit_batch();
cute::warpgroup_wait<0>();
```



## setmaxnreg



## mainloop

 



