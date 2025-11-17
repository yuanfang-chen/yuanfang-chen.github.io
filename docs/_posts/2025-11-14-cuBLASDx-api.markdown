---
layout: post
title:  "cuBLASDx总结"
mathjax: true
# date:   2021-11-28 11:18:26 -0800
categories: CUDA
---

* TOC
{:toc}

# cuBLASDx简介
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

一些核心概念
1. cuBLASDx目前只支持GEMM
1. 有3种GEMM可以调用
    1. \$$\mathbf{C}_{m\times n} = {\alpha} \times \mathbf{A}_{m\times k} \times \mathbf{B}_{k\times n} + {\beta} \times \mathbf{C}_{m\times n}$$
    1. \$$\mathbf{C}_{m\times n} = \mathbf{A}_{m\times k} \times \mathbf{B}_{k\times n} + \mathbf{C}_{m\times n}$$
    1. \$$\mathbf{C}_{m\times n} = \mathbf{A}_{m\times k} \times \mathbf{B}_{k\times n}$$ 
1. cuBLASDx GEMM的operands必须在SMEM或者RMEM



cuBLASDx和核心抽象是自动把开发者对GEMM的描述转换成

## 使用cuBLASDx GEMM的步骤
1. 
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
