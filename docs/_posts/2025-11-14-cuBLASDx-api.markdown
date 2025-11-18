---
layout: post
title:  "cuBLASDx总结"
mathjax: true
# date:   2021-11-28 11:18:26 -0800
categories: CUDA
---

* TOC
{:toc}

<style>
  table {
    border-collapse: collapse; /* Ensures borders are collapsed for a cleaner look */
  }
  /* table, th, td {
    border: 2px solid yellow; /* Adjust '2px' for desired thickness and 'black' for color */
  } */
</style>

## cuBLASDx简介
cuBLAS 设备扩展（cuBLASDx）库使您能够在自己的 CUDA kernel 内部执行 cuBLAS 中提供的部分线性代数函数。目前该功能仅限于通用矩阵乘法（GEMM）。将线性代数与其他操作融合，可以降低延迟并提升应用程序的整体性能。

cuBLASDx 库目前提供以下特性：

* 可嵌入 CUDA kernel 的 BLAS GEMM 例程。
* 高性能，避免了不必要的全局内存数据搬运。
* 高度可定制，支持根据不同需求调整 GEMM 例程（如矩阵尺寸、精度、数据类型、目标 CUDA 架构等）。
* 灵活的累加与融合方式，可在共享内存（shared memory）或寄存器（registers）中进行计算。
* 支持将 BLAS 计算与其他操作融合，从而减少访问全局内存的次数。
* 与未来版本的 CUDA Toolkit 兼容。

cuBLASDx（cuBLAS Device Extensions）是 NVIDIA 在 CUDA Toolkit 11.0+ 中引入的一个轻量级、高性能库扩展，它允许开发者直接在 CUDA kernel 内部调用高度优化的 BLAS 函数，而无需像传统 cuBLAS 那样通过主机端 API 启动单独的 kernel。

>  简单说：cuBLASDx = 可嵌入 CUDA kernel 的 cuBLAS GEMM。

<!-- 可以把cuBLASDx理解为基于cute的和平台无关的device端GEMM抽象层** -->
<!-- 1. 一些cuBLASDx API有`get_xxx()`和`suggest_xxx()`两个版本，比如`get_layout_smem_a`/`get_layout_smem_b`/`get_layout_smem_c`和`suggest_layout_smem_a`/`suggest_layout_smem_b`/`suggest_layout_smem_c` -->
<!-- cuBLASDx和核心抽象是自动把开发者对GEMM的描述转换成 -->

## 核心概念
1. cuBLASDx目前只支持BLAS的GEMM函数
1. 有3种GEMM可以调用
    - Shared memory API: \\(\mathbf{C}\_{m\times n} = {\alpha} \times \mathbf{A}\_{m\times k} \times \mathbf{B}\_{k\times n} + {\beta} \times \mathbf{C}\_{m\times n}\\)
    - Register API with accumulator: \\(\mathbf{C}\_{m\times n} = \mathbf{A}\_{m\times k} \times \mathbf{B}\_{k\times n} + \mathbf{C}\_{m\times n}\\)
    - Register API without accumulator: \\(\mathbf{C}\_{m\times n} = \mathbf{A}\_{m\times k} \times \mathbf{B}\_{k\times n}\\)
    <a id="reg-acc"></a>
    - **说明：**对Shared memory API，\\(\mathbf{A}/\mathbf{B}/\mathbf{C}\\)必须在SMEM中；对Register API，\\(\mathbf{A}/\mathbf{B}\\)必须在SMEM中，\\(\mathbf{C}\\)必须在RMEM中
1. GEMM的输入输出精度（\\(\mathbf{A}/\mathbf{B}/\mathbf{C}\\)的精度）和GEMM中的计算精度（\\(\times\\)和\\(+\\)）解耦；在做GEMM计算之前，\\(\mathbf{A}/\mathbf{B}/\mathbf{C}\\)要做类型转换，转换到定义的GEMM计算精度（可以通过[`GEMM::execute()`](https://docs.nvidia.com/cuda/cublasdx/api/methods.html#shared-memory-api)的入参指定转换函数，不指定的话，类型转换必须可以通过implicit conversion进行，否则会有编译时报错）
1. \\(\mathbf{A}/\mathbf{B}/\mathbf{C}\\)可以以cute Layout格式或者以指针的方式传给GEMM API；cute Layout已经包含了内存布局信息，使用指针时，内存布局信息来自对GEMM的描述。TODO: 增加一个用例表示如果Layout信息和GEMM描述信息有冲突，会发生什么
1. 开发者提供: • **对GEMM的逻辑描述(Description Operators)** • **对GEMM的执行描述(Execution Operators)**。每个细节的描述称为一个operator。Description Operators和Execution Operators加在一起称为function descriptor，也称为BLAS。代码示例：

    ```cpp
    #include <cublasdx.hpp>

    using BLAS = decltype(cublasdx::Size<8, 16, 32>()
        + cublasdx::Precision<double>()
        + cublasdx::Type<cublasdx::type::complex>()
        + cublasdx::Arrangement<cublasdx::col_major, cublasdx::col_major>()
        + cublasdx::Function<cublasdx::function::MM>()
        + cublasdx::SM<700>());
    ```

1. **对GEMM的逻辑描述(Description Operators)**包括：

   | Operator | 默认值 | 描述 |
   |----------|---------------|-------------|
   | `Size <M, N, K>` | 无 | GEMM的大小。 |
   | `Arrangement<ArrA, ArrB, ArrC>` | `row_major`, `col_major`, `col_major` | \\(\mathbf{A}/\mathbf{B}/\mathbf{C}\\)的majorness。 |
   | `Precision<PA, PB, PC>` | `float`, `float`, `float` | \\(\mathbf{A}/\mathbf{B}/\mathbf{C}\\)的**计算精度**；必须全是浮点数或者全是整数。 |
   | `Type<type>` | `type::real` | \\(\mathbf{A}/\mathbf{B}/\mathbf{C}\\)的类型，实数或是复数(`type::real` or `type::complex`). |
   | `LeadingDimension<LDA, LDB, LDC>` | 由`Size`和`Arrangement`定义 | \\(\mathbf{A}/\mathbf{B}/\mathbf{C}\\)的Leading Dimensions。 |
   | `Alignment <AlignA, AlignB, AlignC>` | `alignof(BLAS::a_value_type)`, … | \\(\mathbf{A}/\mathbf{B}/\mathbf{C}\\)的Alignments(以bytes为单位)。 |
   | `SM<CC>` | 无 | 目标CUDA架构的SM。|
1. **对GEMM的执行描述(Execution Operators)**包括：

   | Operator | 默认值 | 描述 |
   |----------|---------------|-------------|
   | `Block` | 无 | 创建在CUDA block中执行的BLAS函数。 |
   | `BlockDim<X, Y, Z>` | `BLAS::suggested_block_dim()`的返回值 | 配置执行BLAS函数的线程数。`X*Y*Z`必须大于等于32，最好是32的整数倍。|
1. cuBLASDx根据function descriptor计算
    - GEMM需要的SMEM总大小（`get_shared_storage_size()`, `get_shared_storage_size_ab()`），以及\\(\mathbf{A}/\mathbf{B}/\mathbf{C}\\)各自占用的SMEM大小（`slice_shared_memory()`, `slice_shared_memory_ab()`）
    - \\(\mathbf{A}/\mathbf{B}/\mathbf{C}\\)在GMEM和SMEM中的cute Layout（`BLAS::get_layout_<gmem/smem>_<a/b/c>()`根据矩阵Layout信息计算，不带优化； `BLAS::suggest_layout_<gmem/smem>_<a/b/c>()`根据矩阵Layout信息和具体的SM计算，带MMA优化和copy优化）
    - 选择合适的MMA指令
    - MMA指令的tiling（矩阵计算的Shape）
    - 以及参与GEMM计算的thread的register fragment
1. 创建function descriptor后，可以通过[`Traits`](https://docs.nvidia.com/cuda/cublasdx/api/traits.html)返回其中包含的GEMM相关信息，比如：
    - 如果function descriptor包含Description Operators，则`cublasdx::is_blas<BLAS>::value`为真
    - 如果function descriptor里有且只有一个`Size + Function + SM`operator，则`cublasdx::is_complete_blas<BLAS>::value`为真
    - 如果function descriptor包含Description Operators和Execution Operators，则`cublasdx::is_blas_execution<BLAS>::value`为真    TODO：写个例子，只有Block
    - 如果`cublasdx::is_complete_blas<BLAS>::value`和`cublasdx::is_blas_execution<BLAS>::value`同时为真，则`cublasdx::is_complete_blas_execution<BLAS>::value`为真
1. RMEM是每个thread独占的且十分有限，GEMM的输入输出必须分布于参与GEMM的一组thread中所有RMEM中。把tensor数据分布于一组thread的RMEM的方式称为**partitioning**。cuBLASDx中[partitioner](https://docs.nvidia.com/cuda/cublasdx/api/other_tensors.html#partitioner-and-register-fragment-tensors)包含了和partitioning有关的所有信息。获取partitioner有三种方式：

    ```cpp
    // #1a 默认的partitioner
    auto partitioner = BLAS::get_partitioner();

    // #1b 带优化的partitioner
    auto partitioner = BLAS::suggest_partitioner();

    // "Register API without accumulator"方式的GEMM返回值包括partitioner
    auto [c_register_fragment, partitioner]
                    = BLAS().execute(a_shared_tensor, b_shared_tensor);
    ```
    partitioner的选择要和使用的Layout的对应，否则对性能有影响：
    - `BLAS::get_partitioner()`配合`get_layout_smem_*()`使用
    - `BLAS::suggest_partitioner()`配合`suggest_layout_smem_*()`使用

1. partitioner有如下API：

    ```cpp
    // Partitioning properties
    __device__ bool is_predicated();
    __device__ bool is_thread_active();

    // Accumulator creation, creates a register cublasdx::tensor
    __device__ constexpr auto make_accumulator_fragment();

    // This method will return a non-owning view of its argument’s subtensor assigned to the calling thread, corresponding to its local register fragment.
    template<class CTensor>
    __forceinline__ __device__
    auto partition_like_C(CTensor && ctensor) const;

    // These 2 functions extend functionality of is_predicated() allowing to map local register fragment index to its source (global or shared) tensor index, as well as check if this index is in bounds.
    template<class ... Coords>
    __forceinline__ __device__
    auto map_fragment_index(Coords&& ... coords) const;
    template<class ... Coords>
    __forceinline__ __device__
    bool is_index_in_bounds(Coords&& ... coords) const;
    ```
    说明：
    - xx
    - xx

1. cuBLASDx支持两种tensor拷贝操作：• GMEM和SMEM的双向拷贝 • SMEM和RMEM的双向拷贝。
1. GMEM和SMEM的双向拷贝：该拷贝操作是协同完成的。所有线程（由 NumThreads 或 BLAS::block_dim 指定）都将参与此次拷贝。该函数会考虑给定的内存对齐方式，并在可能的情况下尝试vectorized load/store。
    ```cpp
    template<uint32_t NumThreads,       // Number of threads performing copy operation
            uint32_t AlignmentInBytes, // Pointer alignment of src and dst tensor (minimum of them if they are different)
            class SrcEngine,
            class SrcLayout,
            class DstEngine,
            class DstLayout>
    __forceinline__ __device__
    void copy(const unsigned int                            tid, // Thread index in CUDA block
            const cublasdx::tensor<SrcEngine, SrcLayout>& src,
            cublasdx::tensor<DstEngine, DstLayout>&       dst)

    // Assumes pointers in both dst and src tensors are not extra aligned
    template<uint32_t NumThreads, // Number of threads performing copy operation
            class SrcEngine,
            class SrcLayout,
            class DstEngine,
            class DstLayout>
    __forceinline__ __device__
    void copy(const unsigned int                            tid, // Thread index in CUDA block
            const cublasdx::tensor<SrcEngine, SrcLayout>& src,
            cublasdx::tensor<DstEngine, DstLayout>&       dst)

    template<class BLAS,                // BLAS description which provides the number of threads
            uint32_t AlignmentInBytes, // Pointer alignment of src and dst tensor (minimum of them if they are different)
            class SrcEngine,
            class SrcLayout,
            class DstEngine,
            class DstLayout>
    __forceinline__ __device__
    void copy(const cublasdx::tensor<SrcEngine, SrcLayout>& src,
            cublasdx::tensor<DstEngine, DstLayout>&       dst)
    ```
1. SMEM和RMEM的双向拷贝

    ```cpp
    // #1 Store fragment: partition and copy from register fragment to global / shared memory tensor
    template<unsigned AlignmentInBytes,    // Alignment of source tensor pointer
            class TRC, class CFragLayout, // Register Memory Fragment Tensor
            class TC, class CLayout,      // Global or Shared Memory tensor
            class Partitioner>
    __forceinline__ __device__
    copy_fragment(tensor<TRC, CFragLayout> const& tS, // Entire non-partitioned global / shared tensor
                tensor<TC, CLayout>           & tD, // Calling thread's register fragment tensor
                Partitioner              const& p);

    // #2 Load fragment: partition and copy from global / shared memory tensor to register fragment
    template<unsigned AlignmentInBytes,    // Alignment of source tensor pointer
            class TRC, class CFragLayout, // Register Memory Fragment Tensor
            class TC, class CLayout,      // Global or Shared Memory tensor
            class Partitioner>
    __forceinline__ __device__
    copy_fragment(tensor<TC, CLayout>      const& tS,
                tensor<TRC, CFragLayout>      & tD,
                Partitioner              const& p);
    ```


## GEMM的调用步骤
1. 用`Operator`定义
1. 

# 例子

```cpp
template<class GEMM>
__global__ void gemm_kernel_shared(const typename GEMM::c_value_type  alpha,
                                   const typename GEMM::a_value_type* a,
                                   const typename GEMM::b_value_type* b,
                                   const typename GEMM::c_value_type  beta,
                                   typename GEMM::c_value_type* c) {
    extern __shared__ __align__(16) char smem[];

    // Make global memory tensor
    auto a_global_tensor = cublasdx::make_tensor(a, GEMM::get_layout_gmem_a());
    auto b_global_tensor = cublasdx::make_tensor(b, GEMM::get_layout_gmem_b());
    auto c_global_tensor = cublasdx::make_tensor(c, GEMM::get_layout_gmem_c());

    // Make shared memory tensor
    auto [smem_a, smem_b, smem_c] = cublasdx::slice_shared_memory<GEMM>(smem);
    auto a_shared_tensor = cublasdx::make_tensor(smem_a, GEMM::get_layout_smem_a());
    auto b_shared_tensor = cublasdx::make_tensor(smem_b, GEMM::get_layout_smem_b());
    auto c_shared_tensor = cublasdx::make_tensor(smem_c, GEMM::get_layout_smem_c());

    // Load data from global memory tensor to shared memory tensor
    using alignment = cublasdx::alignment_of<GEMM>;
    cublasdx::copy<GEMM, alignment::a>(a_global_tensor, a_shared_tensor);
    cublasdx::copy<GEMM, alignment::b>(b_global_tensor, b_shared_tensor);
    cublasdx::copy<GEMM, alignment::c>(c_global_tensor, c_shared_tensor);
    cublasdx::copy_wait();

    // Execute GEMM
    GEMM().execute(alpha, a_shared_tensor, b_shared_tensor, beta, c_shared_tensor);
    __syncthreads();

    // Store data from shared memory tensor to global memory tensor
    cublasdx::copy<GEMM, alignment::c>(c_shared_tensor, c_global_tensor);
}

template<class GEMM>
__global__ void gemm_kernel_registers_accumulation(const typename GEMM::a_value_type* a,
                                                   const typename GEMM::b_value_type* b,
                                                   typename GEMM::c_value_type* c) {
    extern __shared__ __align__(16) char smem[];

    // Make global memory tensor
    auto a_global_tensor = cublasdx::make_tensor(a, GEMM::get_layout_gmem_a());
    auto b_global_tensor = cublasdx::make_tensor(b, GEMM::get_layout_gmem_b());
    auto c_global_tensor = cublasdx::make_tensor(c, GEMM::get_layout_gmem_c());

    // Make shared memory tensor
    auto [smem_a, smem_b] = cublasdx::slice_shared_memory_ab<GEMM>(smem);
    auto a_shared_tensor = cublasdx::make_tensor(smem_a, GEMM::get_layout_smem_a());
    auto b_shared_tensor = cublasdx::make_tensor(smem_b, GEMM::get_layout_smem_b());

    // Load data from global memory tensor to shared memory tensor
    using alignment = cublasdx::alignment_of<GEMM>;
    cublasdx::copy<GEMM, alignment::a>(a_global_tensor, a_shared_tensor);
    cublasdx::copy<GEMM, alignment::b>(b_global_tensor, b_shared_tensor);
    cublasdx::copy_wait();

    // Get default partitioner
    auto partitioner = GEMM::get_partitioner();
    // Create Register Fragment Accumulator
    auto c_register_fragment = partitioner.make_accumulator_fragment();
    // Partition Global C for GEMM and load appropriate elements into register fragment
    cublasdx::copy_fragment<alignment::c>(c_global_tensor, c_register_fragment, partitioner);

    // Execute GEMM with accumulation
    GEMM().execute(a_shared_tensor, b_shared_tensor, c_register_fragment);

    // Partition Global C for GEMM and store appropriate elements to global memory
    cublasdx::copy_fragment<alignment::c>(c_register_fragment, c_global_tensor, partitioner);
}

template<class GEMM>
__global__ void gemm_kernel_registers(const typename GEMM::a_value_type* a,
                                      const typename GEMM::b_value_type* b,
                                      typename GEMM::c_value_type* c) {
    extern __shared__ __align__(16) char smem[];

    // Make global memory tensor
    auto a_global_tensor = cublasdx::make_tensor(a, GEMM::get_layout_gmem_a());
    auto b_global_tensor = cublasdx::make_tensor(b, GEMM::get_layout_gmem_b());
    auto c_global_tensor = cublasdx::make_tensor(c, GEMM::get_layout_gmem_c());

    // Make shared memory tensor
    auto [smem_a, smem_b] = cublasdx::slice_shared_memory_ab<GEMM>(smem);
    auto a_shared_tensor = cublasdx::make_tensor(smem_a, GEMM::get_layout_smem_a());
    auto b_shared_tensor = cublasdx::make_tensor(smem_b, GEMM::get_layout_smem_b());

    // Load data from global memory tensor to shared memory tensor
    using alignment = cublasdx::alignment_of<GEMM>;
    cublasdx::copy<GEMM, alignment::a>(a_global_tensor, a_shared_tensor);
    cublasdx::copy<GEMM, alignment::b>(b_global_tensor, b_shared_tensor);
    cublasdx::copy_wait();

    // Execute GEMM and get register fragment results and data partitioner in return
    auto [c_register_fragment, partitioner] = GEMM().execute(a_shared_tensor, b_shared_tensor);

    // Partition Global C for GEMM and store appropriate elements to global memory
    cublasdx::copy_fragment<alignment::c>(c_register_fragment, c_global_tensor, partitioner);
}

template<unsigned int Arch>
int introduction_example() {
    using GEMM = decltype(cublasdx::Size<32, 32, 32>()
                  + cublasdx::Precision<double>()
                  + cublasdx::Type<cublasdx::type::real>()
                  + cublasdx::Arrangement<cublasdx::row_major, cublasdx::col_major>()
                  + cublasdx::Function<cublasdx::function::MM>()
                  + cublasdx::SM<Arch>()
                  + cublasdx::Block()
                  + cublasdx::BlockDim<256>());

    using value_type = typename example::uniform_value_type_t<GEMM>;

    constexpr auto global_a_size = example::global_memory_size_of<GEMM>::a_size;
    constexpr auto global_b_size = example::global_memory_size_of<GEMM>::b_size;
    constexpr auto global_c_size = example::global_memory_size_of<GEMM>::c_size;

    // Allocate managed memory for A, B, C matrices in one go
    value_type* abc;
    auto        size       = global_a_size + global_b_size + global_c_size;
    auto        size_bytes = size * sizeof(value_type);
    CUDA_CHECK_AND_EXIT(cudaMallocManaged(&abc, size_bytes));
    // Generate data
    for (size_t i = 0; i < size; i++) {
        abc[i] = double(i / size);
    }

    value_type* a = abc;
    value_type* b = abc + global_a_size;
    value_type* c = abc + global_a_size + global_b_size;


    // Shared Memory API: C = alpha * A * B + beta * C
    // Invokes kernel with GEMM::block_dim threads in CUDA block
    gemm_kernel_shared<GEMM><<<1, GEMM::block_dim, cublasdx::get_shared_storage_size<GEMM>()>>>(1.0, a, b, 1.0, c);
    gemm_kernel_shared<GEMM><<<1, GEMM::block_dim, cublasdx::get_shared_storage_size<GEMM>()>>>(1.0, a, b, 1.0, c);
    CUDA_CHECK_AND_EXIT(cudaPeekAtLastError());
    // Register Fragment Accumulation API: C = A * B + C
    // Invokes kernel with GEMM::block_dim threads in CUDA block
    gemm_kernel_registers_accumulation<GEMM><<<1, GEMM::block_dim, cublasdx::get_shared_storage_size_ab<GEMM>()>>>(a, b, c);
    CUDA_CHECK_AND_EXIT(cudaPeekAtLastError());
    // Register Fragment API: C = A * B
    // Invokes kernel with GEMM::block_dim threads in CUDA block
    gemm_kernel_registers<GEMM><<<1, GEMM::block_dim, cublasdx::get_shared_storage_size_ab<GEMM>()>>>(a, b, c);

    CUDA_CHECK_AND_EXIT(cudaPeekAtLastError());
    CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());

    CUDA_CHECK_AND_EXIT(cudaFree(abc));
    std::cout << "Success" << std::endl;
    return 0;
}

struct introduction_example_functor {
    template<int Arch>
    int operator()(std::integral_constant<int, Arch>) {
        return introduction_example<Arch>();
    }
};

int main(int, char**) {
    return example::sm_runner(introduction_example_functor{});
}

```

## References
- https://docs.nvidia.com/cuda/cublasdx/index.html
- https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html#dim-quantization
