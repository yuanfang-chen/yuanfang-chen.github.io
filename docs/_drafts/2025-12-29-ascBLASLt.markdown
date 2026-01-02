---
layout: post
title:  "Ascend ascBLASLt"
# date:   2025-12-08 11:18:26 -0800
categories: ascend
typora-root-url: ..
mathjax: true
---

## 什么是ascBLASLt

ascBLASLt是一套Host侧API，提供GEMM计算能力。它比BLAS API更加灵活并且提供更多的GEMM定制能力。GEMM操作由[ascblasLtMatmul()]()实现，用数学公式表达：

$$
\begin{split}Aux_{temp} & = \alpha \cdot scale_A \cdot scale_B \cdot \text{op}(A) \text{op}(B) + \beta \cdot scale_C \cdot C + bias, \\
D_{temp} & = \mathop{Epilogue}(Aux_{temp}), \\
amax_{D} & = \mathop{absmax}(D_{temp}), \\
amax_{Aux} & = \mathop{absmax}(Aux_{temp}), \\
D & = scale_D * D_{temp}, \\
Aux & = scale_{Aux} * Aux_{temp}. \\\end{split}
$$

其中$$op(A)/op(B)$$是指in-place操作，比如transpose；$$scale_A/scale_B/scale_C$$用于narrow precision；$$alpha$$和$$beta$$是常量；$$Epilogue$$函数支持xxx；$$bias$$矢量长度和M一样，会在列维广播。

ascBLASLt和BLAS对比

| API       | API复杂度 | 调用侧 | 是否支持Fusion                         | 适用的矩阵大小 |
| --------- | --------- | ------ | -------------------------------------- | -------------- |
| BLAS      | Low       | Host   | 不支持                                 | 大矩阵         |
| ascBLASLt | Medium    | Host   | 支持一些特定的Fusion，不支持通用Fusion | 中等矩阵       |

## Matmul算法选择

Matmul的具体算法和输入数据，目标设备等有关。有两种方式来选择最终使用的算法：`cublasLtMatmulAlgoGetIds()`返回所有适用的算法，但是需要开发者自行配置算法的参数才能最终调用Matmul；`cublasLtMatmulAlgoGetHeuristic()`使用启发式算法和缓存，cuBLAS自动选择并配置多种算法并按预估性能排序后，由开发者选择最终的算法。两种选择算法的方式对比（`cublasLtMatmulAlgoGetIds()`[例子](https://github.com/NVIDIA/CUDALibrarySamples/blob/f5460f18f2badb55c754afbb89d253e1a678ee65/cuBLASLt/Common/LtMatmulCustomFind.h)，`cublasLtMatmulAlgoGetHeuristic()`[例子](https://github.com/NVIDIA/CUDALibrarySamples/blob/70ecf4f7ebaccf03c84698257253e1b424427c76/cuBLASLt/LtSgemmSimpleAutoTuning/sample_cublasLt_LtSgemmSimpleAutoTuning.cu)）：

| 特性         | `cublasLtMatmulAlgoGetIds()`                                 | `cublasLtMatmulAlgoGetHeuristic()`                           |
| ------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 用途         | 针对指定计算 / 数据类型组合，列出所有可用的算法 ID           | 查询启发式算法，选择并配置适用于特定矩阵乘法问题的候选算法   |
| 输入         | `cublasLtHandle_t`（cuBLASLt 句柄）、`computeType`（计算类型）、`scaleType`（缩放类型）、`Atype`（矩阵A数据类型）、`Btype`（矩阵B数据类型）、`Ctype`（矩阵C数据类型）、`Dtype`（矩阵D数据类型）、`requestedAlgoCount`（请求返回的算法数量） | `cublasLtHandle_t`（cuBLASLt 句柄）、`operationDesc`（运算描述符）、`Adesc`（矩阵A描述符）、`Bdesc`（矩阵B描述符）、`Cdesc`（矩阵C描述符）、`Ddesc`（矩阵D描述符）、`preference`（偏好设置）、`requestedAlgoCount`（请求返回的算法数量） |
| 输出         | `int algoId[]`（算法 ID 数组）和 `returnAlgoCount`（实际返回的算法数量） | `cublasLtMatmulHeuristicResult_t` 数组（每个元素包含配置完成的 `cublasLtMatmulAlgo_t`（算法描述符）、`workspaceSize`（工作空间大小）、`wavesCount`（波数）、`state`（状态）等）和 `returnAlgoCount`（实际返回的算法数量） |
| 形状感知能力 | 无 — 仅依赖数据类型、计算类型及设备支持情况，不感知矩阵形状 / 布局 | 有 — 返回候选算法时，会考虑矩阵形状、布局、属性及偏好设置    |
| 返回内容     | 仅返回算法 ID（无调优 / 配置信息，无运行时预估数据）         | 可直接使用 / 已校验的算法描述符，以及运行时 / 性能预估数据（按预估时间递增排序） |
| 适用场景     | 枚举支持的算法、实现手动基准测试循环，或通过 `cublasLtMatmulAlgoInit()` 初始化特定算法 | 需库为实际问题选择 / 调优候选算法，并获取预估信息（工作空间、利用率），且无需枚举原始算法 ID 时 |

### 启发式算法缓存 Heuristics Cache

[cublasLtHeuristicsCacheGetCapacity()](https://docs.nvidia.com/cuda/cublas/#cublasltheuristicscachegetcapacity), [cublasLtHeuristicsCacheSetCapacity()](https://docs.nvidia.com/cuda/cublas/#cublasltheuristicscachesetcapacity).

CUBLASLT_HEURISTICS_CACHE_CAPACITY

## Epilogue

- 为了兼容cuBLAS/cuBLASLt，支持RELU,GELU
- 支持后向传播（训练场景）：
  - ReLuBias and GeluBias epilogues that produce an auxiliary output which is used on backward propagation to compute the corresponding gradients.
  - DReLuBGrad and DGeluBGrad epilogues that compute the backpropagation of the corresponding activation function on matrix C, and produce bias gradient as a separate output. These epilogues require auxiliary input mentioned in the bullet above.
  - Support fusion in DL training: CUBLASLT_EPILOGUE_{DRELU,DGELU} which are similar to CUBLASLT_EPILOGUE_{DRELU,DGELU}_BGRAD but don’t compute bias gradient.
  - Support fusion in DLtraining: CUBLASLT_EPILOGUE_BGRADA and CUBLASLT_EPILOGUE_BGRADB which compute bias gradients based on matrices A and B respectively.



## Matrix Transfrom

```c++
cublasStatus_t cublasLtMatrixTransform(
      cublasLtHandle_t lightHandle,
      cublasLtMatrixTransformDesc_t transformDesc,
      const void *alpha,
      const void *A,
      cublasLtMatrixLayout_t Adesc,
      const void *beta,
      const void *B,
      cublasLtMatrixLayout_t Bdesc,
      void *C,
      cublasLtMatrixLayout_t Cdesc,
      cudaStream_t stream);
```
Transfrom可以scale/shift输入矩阵的元素，也可以改变数据的内存排序。它由`cublasLtMatrixTransform()`实现，具体操作用数学公式表示为：

$$
C = alpha*transformation(A) + beta*transformation(B),
$$


$$A/B$$是输入矩阵，$$\alpha/ \beta$$是常量，$$transformation$$操作由`transformDesc`来定义。


## [AMAX ](https://ww2.mathworks.cn/matlabcentral/fileexchange/41115-absmax)(Absolute Maximum)

返回绝对值最大的数


$$
absmax([-5 3 2 3; 3 2 1 4]) = -5 \\
absmax([643, 10]) = 643
$$


## [narrow precision data types](https://docs.nvidia.com/cuda/cublas/#narrow-precision-data-types-usage)

TODO

## [Floating Point Emulation](https://docs.nvidia.com/cuda/cublas/index.html#floating-point-emulation)

TODO

cublasLtEmulationDesc_t

cublasLtEmulationDescAttributes_t

## 日志



## 一个例子

```c++
```





## ascBLASLt 环境变量



## ascBLASLt 类型定义

### 句柄

#### cublasLtHandle_t

[cublasLtHandle_t](https://docs.nvidia.com/cuda/cublas/#cublaslthandle-t) 类型是指向包含 cuBLASLt 库上下文的不透明结构的指针类型。使用 cublasLtCreate()</b1 初始化 cuBLASLt 库上下文，并返回指向包含 cuBLASLt 库上下文的不透明结构的句柄；使用 cublasLtDestroy()</b2 销毁先前创建的 cuBLASLt 库上下文描述符并释放资源。

### Matmul描述

#### cublasLtMatmulDesc_t

[cublasLtMatmulDesc_t](https://docs.nvidia.com/cuda/cublas/#cublasltmatmuldesc-t)是一个指向不透明结构的指针，该结构包含矩阵乘法运算[cublasLtMatmul()](https://docs.nvidia.com/cuda/cublas/#cublasltmatmul)的描述。可以通过调用[cublasLtMatmulDescCreate()](https://docs.nvidia.com/cuda/cublas/#cublasltmatmuldesccreate)创建描述符，并通过调用[cublasLtMatmulDescDestroy()](https://docs.nvidia.com/cuda/cublas/#cublasltmatmuldescdestroy)销毁描述符。

#### cublasLtMatmulDescAttributes_t

[cublasLtMatmulDescAttributes_t](https://docs.nvidia.com/cuda/cublas/#cublasltmatmuldescattributes-t)是矩阵乘法描述符，定义矩阵乘法运算具体细节。使用[cublasLtMatmulDescGetAttribute()](https://docs.nvidia.com/cuda/cublas/#cublasltmatmuldescgetattribute)和[cublasLtMatmulDescSetAttribute()](https://docs.nvidia.com/cuda/cublas/#cublasltmatmuldescsetattribute)来获取和设置矩阵乘法描述符的属性值。

| 值                                               | 描述                                                         | 数据类型                                                     |
| ------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| `CUBLASLT_MATMUL_DESC_COMPUTE_TYPE`              | 计算类型。定义用于乘法和累加运算的数据类型，以及矩阵乘法过程中的累加器。参见[cublasComputeType_t](https://docs.nvidia.com/cuda/cublas/#cublascomputetype-t)。 | `int32_t`                                                    |
| `CUBLASLT_MATMUL_DESC_SCALE_TYPE`                | 缩放类型。定义缩放因子`alpha`和`beta`的数据类型。累加器值和来自矩阵`C`的值通常在最终缩放前转换为缩放类型。然后，该值在存储到内存前从缩放类型转换为矩阵`D`的类型。默认值取决于`CUBLASLT_MATMUL_DESC_COMPUTE_TYPE`。请参见[cudaDataType_t](https://docs.nvidia.com/cuda/cublas/#cudadatatype-t)。 | `int32_t`                                                    |
| `CUBLASLT_MATMUL_DESC_POINTER_MODE`              | 指定`alpha`和`beta`通过引用传递，无论它们是主机端或设备端的标量，还是设备端向量。默认值为：`CUBLASLT_POINTER_MODE_HOST`（即位于主机端）。请参见[cublasLtPointerMode_t](https://docs.nvidia.com/cuda/cublas/#cublasltpointermode-t)。 | `int32_t`                                                    |
| `CUBLASLT_MATMUL_DESC_TRANSA`                    | 指定应对矩阵A执行的变换操作类型。默认值为：`CUBLAS_OP_N`（即非转置操作）。请参见[cublasOperation_t](https://docs.nvidia.com/cuda/cublas/#cublasoperation-t)。 | `int32_t`                                                    |
| `CUBLASLT_MATMUL_DESC_TRANSB`                    | 指定应对矩阵B执行的变换操作类型。默认值为：`CUBLAS_OP_N`（即非转置操作）。请参见[cublasOperation_t](https://docs.nvidia.com/cuda/cublas/#cublasoperation-t)。 | `int32_t`                                                    |
| `CUBLASLT_MATMUL_DESC_TRANSC`                    | 指定应对矩阵C执行的变换操作类型。目前仅支持`CUBLAS_OP_N`。默认值为：`CUBLAS_OP_N`（即非转置操作）。请参见[cublasOperation_t](https://docs.nvidia.com/cuda/cublas/#cublasoperation-t)。 | `int32_t`                                                    |
| `CUBLASLT_MATMUL_DESC_EPILOGUE`                  | 尾声函数。参见[cublasLtEpilogue_t](https://docs.nvidia.com/cuda/cublas/#cublasltepilogue-t)。默认值为：`CUBLASLT_EPILOGUE_DEFAULT`。 | `uint32_t`                                                   |
| `CUBLASLT_MATMUL_DESC_BIAS_POINTER`              | 设备内存中的偏置或偏置梯度向量指针。当使用以下结尾之一时，输入向量的长度应与矩阵D的行数匹配：`CUBLASLT_EPILOGUE_BIAS`、`CUBLASLT_EPILOGUE_RELU_BIAS`、`CUBLASLT_EPILOGUE_RELU_AUX_BIAS`、`CUBLASLT_EPILOGUE_GELU_BIAS`、`CUBLASLT_EPILOGUE_GELU_AUX_BIAS`。当使用以下尾声之一时，输出向量的长度与矩阵D的行数匹配：`CUBLASLT_EPILOGUE_DRELU_BGRAD`、`CUBLASLT_EPILOGUE_DGELU_BGRAD`、`CUBLASLT_EPILOGUE_BGRADA`。当使用以下收尾操作之一时，输出向量的长度与矩阵D的列数匹配：`CUBLASLT_EPILOGUE_BGRADB`。当矩阵D的数据类型为`CUDA_R_8I`时，偏置向量元素与`alpha`和`beta`的类型相同（参见此表中的`CUBLASLT_MATMUL_DESC_SCALE_TYPE`），否则与矩阵D的数据类型相同。有关详细映射，请参见[cublasLtMatmul()](https://docs.nvidia.com/cuda/cublas/#cublasltmatmul)下的数据类型表。默认值为：NULL。 | `void *` / `const void *`                                    |
| `CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER`      | 尾声辅助缓冲区的指针。当使用`CUBLASLT_EPILOGUE_RELU_AUX`或`CUBLASLT_EPILOGUE_RELU_AUX_BIAS`收尾操作时，前向传播中ReLU位掩码的输出向量。当使用`CUBLASLT_EPILOGUE_DRELU`或`CUBLASLT_EPILOGUE_DRELU_BGRAD`结尾时，反向传播中ReLu位掩码的输入向量。使用`CUBLASLT_EPILOGUE_GELU_AUX_BIAS`尾声时，前向传播中GELU输入矩阵的输出。当使用`CUBLASLT_EPILOGUE_DGELU`或`CUBLASLT_EPILOGUE_DGELU_BGRAD`尾声时，反向传播的GELU输入矩阵的输入。关于辅助数据类型，请参见`CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_DATA_TYPE`。不间接引用此指针的例程（如[cublasLtMatmulAlgoGetHeuristic()](https://docs.nvidia.com/cuda/cublas/#cublasltmatmulalgogetheuristic)）会根据其值来确定预期的指针对齐方式。需要设置`CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_LD`属性。 | `void *` / `const void *`                                    |
| `CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_LD`           | 尾声辅助缓冲区的主维度。当使用`CUBLASLT_EPILOGUE_RELU_AUX`、`CUBLASLT_EPILOGUE_RELU_AUX_BIAS`、`CUBLASLT_EPILOGUE_DRELU_BGRAD`或`CUBLASLT_EPILOGUE_DRELU_BGRAD`结尾时，ReLu位掩码矩阵的主维度（以元素即位为单位）。该维度必须能被128整除，且不小于输出矩阵的行数。当使用`CUBLASLT_EPILOGUE_GELU_AUX_BIAS`、`CUBLASLT_EPILOGUE_DGELU`或`CUBLASLT_EPILOGUE_DGELU_BGRAD`结尾时，GELU输入矩阵的主维度（以元素计）。该维度必须能被8整除，且不小于输出矩阵的行数。 | `int64_t`                                                    |
| `CUBLASLT_MATMUL_DESC_SM_COUNT_TARGET`           | 用于并行执行的目标SM数量。当用户期望并发流使用部分设备资源时，该参数会优化针对不同SM数量的执行启发式算法。默认值：0。 | `int32_t`                                                    |
| `CUBLASLT_MATMUL_DESC_AMAX_D_POINTER`            | 指向内存位置的设备指针，完成后该内存位置将被设置为输出矩阵中绝对值的最大值。计算出的值与计算类型具有相同的类型。如果未指定或设置为NULL，则不计算最大绝对值。如果为不受支持的矩阵数据、缩放和计算类型组合设置了此指针，调用[cublasLtMatmul()](https://docs.nvidia.com/cuda/cublas/#cublasltmatmul)将返回`CUBLAS_INVALID_VALUE`。默认值：NULL | `void *`                                                     |
| `CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_DATA_TYPE`    | 将存储在`CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER`中的数据类型。如果未设置（或设置为默认值-1），则数据类型将被设置为输出矩阵元素的数据类型（DType），但有一些例外情况：ReLu使用位掩码。对于输出类型（DType）为`CUDA_R_8F_E4M3`的FP8内核，在以下情况下可以将数据类型设置为非默认值：AType 和 BType 均为 `CUDA_R_8F_E4M3`。偏差类型为`CUDA_R_16F`。CType为`CUDA_R_16BF`或`CUDA_R_16F``CUBLASLT_MATMUL_DESC_EPILOGUE` 被设置为 `CUBLASLT_EPILOGUE_GELU_AUX`当CType为`CUDA_R_16F`时，数据类型可设置为`CUDA_R_16F`或`CUDA_R_8F_E4M3`。当CType为`CUDA_R_16BF`时，数据类型可设置为`CUDA_R_16BF`。其他情况下，数据类型应保持未设置状态或设为默认值-1。如果为不受支持的矩阵数据、规模和计算类型组合进行设置，调用[cublasLtMatmul()](https://docs.nvidia.com/cuda/cublas/#cublasltmatmul)将返回`CUBLAS_INVALID_VALUE`。默认值：-1 | `int32_t`（[cudaDataType_t](https://docs.nvidia.com/cuda/cublas/#cudadatatype-t)） |
| `CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_AMAX_POINTER` | 指向内存位置的设备指针，完成后该内存位置将被设置为通过`CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER`设置的缓冲区中绝对值的最大值。计算出的值与计算类型具有相同的类型。如果未指定或设置为NULL，则不计算最大绝对值。如果为不受支持的矩阵数据、缩放和计算类型组合进行了设置，调用[cublasLtMatmul()](https://docs.nvidia.com/cuda/cublas/#cublasltmatmul)将返回`CUBLAS_INVALID_VALUE`。默认值：NULL | `void *`                                                     |
| `CUBLASLT_MATMUL_DESC_FAST_ACCUM`                | 用于管理FP8快速累加模式的标志。启用后，在某些GPU上，问题执行速度可能会更快，但精度会降低，因为中间结果不会定期提升到更高的精度。目前，此标志对以下GPU有效：Ada、Hopper。默认值：0 - 快速累加模式已禁用 | `int8_t`                                                     |
| `CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE`            | 设备内存中偏置或偏置梯度向量的类型。偏置情况：参见`CUBLASLT_EPILOGUE_BIAS`。如果未设置（或设置为默认值-1），则偏置向量元素与输出矩阵（Dtype）的元素类型相同，但有以下例外情况：IMMA内核的computeType=`CUDA_R_32I`且`Ctype=CUDA_R_8I`，其中偏置向量元素与alpha、beta的类型相同（`CUBLASLT_MATMUL_DESC_SCALE_TYPE=CUDA_R_32F`）对于输出类型为`CUDA_R_32F`、`CUDA_R_8F_E4M3`或`CUDA_R_8F_E5M2`的FP8内核。有关更多详细信息，请参见[cublasLtMatmul()](https://docs.nvidia.com/cuda/cublas/#cublasltmatmul)。默认值：-1 | `int32_t`（[cudaDataType_t](https://docs.nvidia.com/cuda/cublas/#cudadatatype-t)） |

### Matrix Layout

#### cublasLtMatrixLayout_t

#### cublasLtMatrixLayoutAttribute_t

### Matmul算法选择

#### cublasLtMatmulPreference_t

#### cublasLtMatmulPreferenceAttributes_t

#### cublasLtMatmulAlgo_t

#### cublasLtMatmulAlgoCapAttributes_t

#### cublasLtMatmulAlgoConfigAttributes_t

#### cublasLtMatmulHeuristicResult_t

### Matrix Transform

#### cublasLtMatrixTransformDesc_t

#### cublasLtMatrixTransformDescAttributes_t

### cublasLtEpilogue_t
[cublasLtEpilogue_t](https://docs.nvidia.com/cuda/cublas/#cublasltepilogue-t)是一种枚举类型，用于设置后处理。

| 值                                                           | 描述                                                         |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| `CUBLASLT_EPILOGUE_DEFAULT = 1`                              | 无需特殊的后处理，必要时只需对结果进行缩放和量化即可。       |
| `CUBLASLT_EPILOGUE_RELU = 2`                                 | 对结果应用ReLU逐点变换（`x := max(x, 0)`）。                 |
| `CUBLASLT_EPILOGUE_RELU_AUX = CUBLASLT_EPILOGUE_RELU | 128`  | 对结果应用ReLU逐点变换（`x := max(x, 0)`）。这种收尾模式会产生一个额外输出，详见[cublasLtMatmulDescAttributes_t](https://docs.nvidia.com/cuda/cublas/#cublasltmatmuldescattributes-t)的`CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER`。 |
| `CUBLASLT_EPILOGUE_BIAS = 4`                                 | 应用（广播）来自偏置向量的偏置。偏置向量的长度必须与矩阵D的行数匹配，并且必须是压缩的（例如向量元素之间的步长为1）。偏置向量会广播到所有列，并在应用最终的后处理之前相加。 |
| `CUBLASLT_EPILOGUE_RELU_BIAS = CUBLASLT_EPILOGUE_RELU = CUBLASLT_EPILOGUE_BIAS` | 应用偏置，然后进行ReLU变换。                                 |
| `CUBLASLT_EPILOGUE_RELU_AUX_BIAS = CUBLASLT_EPILOGUE_RELU_AUX = CUBLASLT_EPILOGUE_BIAS` | 应用偏置，然后进行ReLU变换。这种收尾模式会产生一个额外输出，参见<cublasLtMatmulDescAttributes_t>的<CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER>。 |
| `CUBLASLT_EPILOGUE_GELU = 32`                                | 对结果应用GELU逐点变换（`x := GELU(x)`）。                   |
| `CUBLASLT_EPILOGUE_GELU_AUX = CUBLASLT_EPILOGUE_GELU | 128`  | 对结果应用GELU逐点变换（`x := GELU(x)`）。这种收尾模式将GELU输入作为单独的矩阵输出（在训练中很有用）。请参见[cublasLtMatmulDescAttributes_t](https://docs.nvidia.com/cuda/cublas/#cublasltmatmuldescattributes-t)的`CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER`。 |
| `CUBLASLT_EPILOGUE_GELU_BIAS = CUBLASLT_EPILOGUE_GELU = CUBLASLT_EPILOGUE_BIAS` | 先应用偏置，然后进行GELU变换[5](https://docs.nvidia.com/cuda/cublas/#gelu)。 |
| `CUBLASLT_EPILOGUE_GELU_AUX_BIAS = CUBLASLT_EPILOGUE_GELU_AUX = CUBLASLT_EPILOGUE_BIAS` | 先应用偏置，然后执行GELU变换[5](https://docs.nvidia.com/cuda/cublas/#gelu)。这种收尾模式将GELU输入作为单独的矩阵输出（在训练中很有用）。参见[cublasLtMatmulDescAttributes_t](https://docs.nvidia.com/cuda/cublas/#cublasltmatmuldescattributes-t)的`CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER`。 |

### 日志

#### cublasLtLoggerCallback_t

## ascBLASLt API列表

### Handler

#### cublasLtCreate()

```c++
cublasStatus_t cublasLtCreate(cublasLtHandle_t *lighthandle)
```

此函数用于初始化cuBLASLt库，并创建一个指向包含cuBLASLt库上下文的不透明结构的句柄。

#### cublasLtDestroy()

```c++
cublasStatus_t cublasLtDestroy(cublasLtHandle_t lightHandle)
```

此函数会释放cuBLASLt库所使用的硬件资源。

### Matmul运算

#### cublasLtMatmul()

该函数用于计算矩阵 A 与矩阵 B 的矩阵乘积，得到输出矩阵 D，计算规则遵循如下公式：
$$D = \alpha \times (A \times B) + \beta \times (C)$$

其中，A、B、C 为输入矩阵，$$\alpha$$ 和 $$\beta$$ 为输入标量。

注意：该函数同时支持原地矩阵乘法和非原地矩阵乘法两种模式：

1. 原地矩阵乘法：要求满足 $$C = D$$ 且 Cdesc = Ddesc（即矩阵C和D指向同一内存空间，且描述信息完全一致）。

2. 非原地矩阵乘法：要求满足 $$C \neq D$$，且两个矩阵必须数据类型相同、行数相同、列数相同、批次大小相同、内存存储顺序相同。

在非原地模式下，矩阵 C 的主维度可以与矩阵 D 的主维度不同。特别地，当矩阵 C 的主维度取值为 0 时，可实现按行或按列的广播运算。

若参数 Cdesc（矩阵C的描述信息）被省略，则该函数默认其与 Ddesc（矩阵D的描述信息）保持一致。

#### cublasLtMatmulDescInit()

#### cublasLtMatmulDescCreate()

#### cublasLtMatmulDescDestroy()

### Matmul Layout

描述A/B/C/D的内存布局

#### cublasLtMatrixLayoutCreate()

#### cublasLtMatrixLayoutInit()

#### 

#### cublasLtMatrixLayoutDestroy()

#### cublasLtMatrixLayoutGetAttribute()

#### cublasLtMatrixLayoutSetAttribute()

### Matmul Preference

heuristic search preferences descriptor.

#### cublasLtMatmulPreferenceInit()

#### cublasLtMatmulPreferenceCreate()

#### cublasLtMatmulPreferenceDestroy()

#### cublasLtMatmulPreferenceGetAttribute()

#### cublasLtMatmulPreferenceSetAttribute()

### Matmul算法选择
#### cublasLtMatmulAlgoGetHeuristic()

```c++
cublasStatus_t cublasLtMatmulAlgoGetHeuristic(
      cublasLtHandle_t lightHandle,
      cublasLtMatmulDesc_t operationDesc,
      cublasLtMatrixLayout_t Adesc,
      cublasLtMatrixLayout_t Bdesc,
      cublasLtMatrixLayout_t Cdesc,
      cublasLtMatrixLayout_t Ddesc,
      cublasLtMatmulPreference_t preference,
      int requestedAlgoCount,
      cublasLtMatmulHeuristicResult_t heuristicResultsArray[],
      int *returnAlgoCount);
```

This function retrieves the possible algorithms for the matrix multiply operation [cublasLtMatmul()](https://docs.nvidia.com/cuda/cublas/#cublasltmatmul) function with the given input matrices A, B and C, and the output matrix D. The output is placed in `heuristicResultsArray[]` in the order of increasing estimated compute time.

#### cublasLtMatmulAlgoGetIds()

```c++
cublasStatus_t cublasLtMatmulAlgoGetIds(
      cublasLtHandle_t lightHandle,
      cublasComputeType_t computeType,
      cudaDataType_t scaleType,
      cudaDataType_t Atype,
      cudaDataType_t Btype,
      cudaDataType_t Ctype,
      cudaDataType_t Dtype,
      int requestedAlgoCount,
      int algoIdsArray[],
      int *returnAlgoCount);
```

This function retrieves the IDs of all the matrix multiply algorithms that are valid, and can potentially be run by the [cublasLtMatmul()](https://docs.nvidia.com/cuda/cublas/#cublasltmatmul) function, for given types of the input matrices A, B and C, and of the output matrix D.

#### cublasLtHeuristicsCacheGetCapacity()

#### cublasLtHeuristicsCacheSetCapacity()

#### cublasLtMatmulAlgoCapGetAttribute()

This function returns the value of the queried capability attribute for an initialized [cublasLtMatmulAlgo_t](https://docs.nvidia.com/cuda/cublas/#cublasltmatmulalgo-t) descriptor structure. The capability attribute value is retrieved from the enumerated type [cublasLtMatmulAlgoCapAttributes_t](https://docs.nvidia.com/cuda/cublas/#cublasltmatmulalgocapattributes-t).

#### cublasLtMatmulAlgoInit()

#### cublasLtMatmulAlgoCheck()

#### cublasLtMatmulAlgoConfigGetAttribute()

#### cublasLtMatmulAlgoConfigSetAttribute()

### Transform运算

#### cublasLtMatrixTransform()

#### cublasLtMatrixTransformDescCreate()

#### cublasLtMatrixTransformDescInit()

#### cublasLtMatrixTransformDescDestroy()

#### cublasLtMatrixTransformDescGetAttribute()

#### cublasLtMatrixTransformDescSetAttribute()

### 日志

#### cublasLtLoggerSetCallback()

#### cublasLtLoggerSetFile()

#### cublasLtLoggerOpenFile()

#### cublasLtLoggerSetLevel()

#### cublasLtLoggerSetMask()

#### cublasLtLoggerForceDisable()


## TODO

- Epilogue support
- Remove CUBLASLT_MATMUL_DESC_FILL_MODE
- CUBLASLT_ALGO_CONFIG_INNER_SHAPE_ID
- CUBLASLT_ALGO_CONFIG_STAGES_ID
- Scale support (cublasLtMatmulMatrixScale_t)
- batch GEMM support (cublasLtBatchMode_t)
- grouped GEMM support:  `cublasLtGroupedMatrixLayoutCreate()`，`cublasLtGroupedMatrixLayoutInit()`
  - **Mechanism:** It addresses the "Mixture of Experts" (MoE) bottleneck where small, varied batch sizes would otherwise require multiple high-overhead kernel launches.
- In epilogue, support backpropagation
- support scaling factor




## Question

(关于cublasLtMatmulTile_t). 昇腾 tensor core支持的matrix大小？
