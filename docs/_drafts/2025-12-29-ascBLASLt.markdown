---
layout: post
title:  "Ascend ascBLASLt"
# date:   2025-12-08 11:18:26 -0800
categories: ascend
typora-root-url: ..
mathjax: true
---


![img](/assets/images/v2-1d1e8468ba44adb7b3ec6659589b173d_1440w.jpg)

<img src="/assets/images/image-20260107114142883.png" alt="image-20260107114142883" style="zoom:50%;" />

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

ascBLASLt支持一些特定的Fusion，不支持通用Fusion；BLAS不支持Fusion。

## 矩阵大小的限制

TODO

## Matmul算法选择

Matmul的具体算法和输入数据，目标设备等有关。有**手动和自动**两种方式来选择最终使用的算法：`ascblasLtMatmulAlgoGetIds()`返回所有适用的算法，但是需要开发者自行配置算法的参数才能最终调用Matmul；`ascblasLtMatmulAlgoGetHeuristic()`使用启发式算法和缓存，ascblas自动选择并配置多种算法并按预估性能排序后，由开发者选择最终的算法。两种选择算法的方式对比（`ascblasLtMatmulAlgoGetIds()`[例子](https://github.com/NVIDIA/CUDALibrarySamples/blob/f5460f18f2badb55c754afbb89d253e1a678ee65/ascblasLt/Common/LtMatmulCustomFind.h)，`ascblasLtMatmulAlgoGetHeuristic()`[例子](https://github.com/NVIDIA/CUDALibrarySamples/blob/70ecf4f7ebaccf03c84698257253e1b424427c76/ascblasLt/LtSgemmSimpleAutoTuning/sample_ascblasLt_LtSgemmSimpleAutoTuning.cu)）：

| 特性         | `ascblasLtMatmulAlgoGetIds()`                                 | `ascblasLtMatmulAlgoGetHeuristic()`                           |
| ------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 用途         | 针对指定计算 / 数据类型组合，列出所有可用的算法 ID           | 查询启发式算法，选择并配置适用于特定矩阵乘法问题的候选算法   |
| 输入         | `ascblasLtHandle_t`（ascblasLt 句柄）、`computeType`（计算类型）、`scaleType`（缩放类型）、`Atype`（矩阵A数据类型）、`Btype`（矩阵B数据类型）、`Ctype`（矩阵C数据类型）、`Dtype`（矩阵D数据类型）、`requestedAlgoCount`（请求返回的算法数量） | `ascblasLtHandle_t`（ascblasLt 句柄）、`operationDesc`（运算描述符）、`Adesc`（矩阵A描述符）、`Bdesc`（矩阵B描述符）、`Cdesc`（矩阵C描述符）、`Ddesc`（矩阵D描述符）、`preference`（偏好设置）、`requestedAlgoCount`（请求返回的算法数量） |
| 输出         | `int algoId[]`（算法 ID 数组）和 `returnAlgoCount`（实际返回的算法数量） | `ascblasLtMatmulHeuristicResult_t` 数组（每个元素包含配置完成的 `ascblasLtMatmulAlgo_t`（算法描述符）、`workspaceSize`（工作空间大小）、`wavesCount`（波数）、`state`（状态）等）和 `returnAlgoCount`（实际返回的算法数量） |
| 形状感知能力 | 无 — 仅依赖数据类型、计算类型及设备支持情况，不感知矩阵形状 / 布局 | 有 — 返回候选算法时，会考虑矩阵形状、布局、属性及偏好设置    |
| 返回内容     | 仅返回算法 ID（无调优 / 配置信息，无运行时预估数据）         | 可直接使用 / 已校验的算法描述符，以及运行时 / 性能预估数据（按预估时间递增排序） |
| 适用场景     | 枚举支持的算法、实现手动基准测试循环，或通过 `ascblasLtMatmulAlgoInit()` 初始化特定算法 | 需库为实际问题选择 / 调优候选算法，并获取预估信息（工作空间、利用率），且无需枚举原始算法 ID 时 |

### 启发式算法缓存 Heuristics Cache

[ascblasLtHeuristicsCacheGetCapacity()](https://docs.nvidia.com/cuda/ascblas/#ascblasltheuristicscachegetcapacity), [ascblasLtHeuristicsCacheSetCapacity()](https://docs.nvidia.com/cuda/ascblas/#ascblasltheuristicscachesetcapacity).

ascblasLT_HEURISTICS_CACHE_CAPACITY

## Epilogue

- 支持RELU,GELU（兼容cublas/cublasLt）
- 支持ReLuBias/GeluBias/DReLuBGrad/DGeluBGrad支持后向传播（训练场景）：
  - ReLuBias and GeluBias epilogues that produce an auxiliary output which is used on backward propagation to compute the corresponding gradients.
  - DReLuBGrad and DGeluBGrad epilogues that compute the backpropagation of the corresponding activation function on matrix C, and produce bias gradient as a separate output. These epilogues require auxiliary input mentioned in the bullet above.
  - Support fusion in DL training: ascblasLT_EPILOGUE_{DRELU,DGELU} which are similar to ascblasLT_EPILOGUE_{DRELU,DGELU}_BGRAD but don’t compute bias gradient.
  - Support fusion in DLtraining: ascblasLT_EPILOGUE_BGRADA and ascblasLT_EPILOGUE_BGRADB which compute bias gradients based on matrices A and B respectively.



## Matrix Transfrom

```c++
ascblasStatus_t ascblasLtMatrixTransform(
      ascblasLtHandle_t lightHandle,
      ascblasLtMatrixTransformDesc_t transformDesc,
      const void *alpha,
      const void *A,
      ascblasLtMatrixLayout_t Adesc,
      const void *beta,
      const void *B,
      ascblasLtMatrixLayout_t Bdesc,
      void *C,
      ascblasLtMatrixLayout_t Cdesc,
      cudaStream_t stream);
```
Transfrom可以scale/shift输入矩阵的元素，也可以改变数据的内存排序。它由`ascblasLtMatrixTransform()`实现，具体操作用数学公式表示为：

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


## [narrow precision data types](https://docs.nvidia.com/cuda/ascblas/#narrow-precision-data-types-usage)

TODO

## [Floating Point Emulation](https://docs.nvidia.com/cuda/ascblas/index.html#floating-point-emulation)

TODO

ascblasLtEmulationDesc_t

ascblasLtEmulationDescAttributes_t

## 日志



## 一个例子

```c++
```



## ascBLASLt 环境变量



## ascBLAS类型定义

### ascDataType_t

DataType::DT_FLOAT/DataType::DT_FLOAT16/DataType::DT_BFLOAT16/DataType::DT_INT8/DataType::DT_INT4。

| 值            | 含义                                 |
| ------------- | ------------------------------------ |
| `CUDA_R_16F`  | 该数据类型是16位实型半精度浮点数。   |
| `CUDA_R_16BF` | 该数据类型是16位实型bfloat16浮点数。 |
| `CUDA_R_32F`  | 数据类型为32位实型单精度浮点数       |
| `CUDA_R_8I`   | 数据类型是8位实型有符号整数          |
| `CUDA_R_4I`   | 数据类型是8位实型无符号整数          |

### ascblasOperation_t

[ascblasOperation_t](https://docs.nvidia.com/cuda/ascblas/#ascblasoperation-t) 类型指示需要对稠密矩阵执行哪种操作。其值对应于 Fortran 字符 `‘N’` 或 `‘n’`（非转置）、`‘T’` 或 `‘t’`（转置）以及 `‘C’` 或 `‘c’`（共轭转置），这些字符常被用作传统 BLAS 实现的参数。

| 值            | 含义               |
| ------------- | ------------------ |
| `ascblas_OP_N` | 已选择非转置操作。 |
| `ascblas_OP_T` | 已选择转置操作。   |

### ascblasComputeType_t

[ascblasComputeType_t](https://docs.nvidia.com/cuda/ascblas/#ascblascomputetype-t)枚举类型用于[ascblasGemmEx()](https://docs.nvidia.com/cuda/ascblas/#ascblasgemmex)和[ascblasLtMatmul()](https://docs.nvidia.com/cuda/ascblas/#ascblasltmatmul)（包括所有批处理和跨步批处理变体）中，以选择如下定义的计算精度模式。

| 值 \| 含义                    |                                                              |
| ----------------------------- | ------------------------------------------------------------ |
| `ascblas_COMPUTE_16F`          | 这是16位半精度浮点数以及所有计算和中间存储精度至少为16位半精度的默认且性能最高的模式。只要有可能，就会使用张量核心。 |
| `ascblas_COMPUTE_32F_FAST_16F` | 允许该库使用张量核心，通过自动下转换和16位半精度计算来处理32位输入和输出矩阵。 |
| `ascblas_COMPUTE_64F`          | 这是默认的64位双精度浮点数，并且使用至少64位的计算精度和中间存储精度。 |
| `ascblas_COMPUTE_32I`          | 这是默认的32位整数模式，使用至少32位的计算精度和中间存储精度。 |

## ascBLASLt 类型定义

### 句柄

#### ascblasLtHandle_t

[ascblasLtHandle_t](https://docs.nvidia.com/cuda/ascblas/#ascblaslthandle-t) 类型是指向包含 ascblasLt 库上下文的不透明结构的指针类型。使用 ascblasLtCreate()</b1 初始化 ascblasLt 库上下文，并返回指向包含 ascblasLt 库上下文的不透明结构的句柄；使用 ascblasLtDestroy()</b2 销毁先前创建的 ascblasLt 库上下文描述符并释放资源。

### Matmul描述符

#### ascblasLtMatmulDesc_t

[ascblasLtMatmulDesc_t](https://docs.nvidia.com/cuda/ascblas/#ascblasltmatmuldesc-t)是一个指向不透明结构的指针，该结构包含矩阵乘法运算[ascblasLtMatmul()](https://docs.nvidia.com/cuda/ascblas/#ascblasltmatmul)的描述。可以通过调用[ascblasLtMatmulDescCreate()](https://docs.nvidia.com/cuda/ascblas/#ascblasltmatmuldesccreate)创建描述符，并通过调用[ascblasLtMatmulDescDestroy()](https://docs.nvidia.com/cuda/ascblas/#ascblasltmatmuldescdestroy)销毁描述符。

#### ascblasLtMatmulDescAttributes_t

[ascblasLtMatmulDescAttributes_t](https://docs.nvidia.com/cuda/ascblas/#ascblasltmatmuldescattributes-t)是矩阵乘法描述符，定义矩阵乘法运算具体细节。使用[ascblasLtMatmulDescGetAttribute()](https://docs.nvidia.com/cuda/ascblas/#ascblasltmatmuldescgetattribute)和[ascblasLtMatmulDescSetAttribute()](https://docs.nvidia.com/cuda/ascblas/#ascblasltmatmuldescsetattribute)来获取和设置矩阵乘法描述符的属性值。

| 值                                                | 描述                                                         | 数据类型                                                     |
| ------------------------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| `ascblasLT_MATMUL_DESC_COMPUTE_TYPE`              | 计算类型。定义用于乘法和累加运算的数据类型，以及矩阵乘法过程中的累加器。参见[ascblasComputeType_t](https://docs.nvidia.com/cuda/ascblas/#ascblascomputetype-t)。 | `int32_t`                                                    |
| `ascblasLT_MATMUL_DESC_SCALE_TYPE`                | 缩放类型。定义缩放因子`alpha`和`beta`的数据类型。累加器值和来自矩阵`C`的值通常在最终缩放前转换为缩放类型。然后，该值在存储到内存前从缩放类型转换为矩阵`D`的类型。默认值取决于`ascblasLT_MATMUL_DESC_COMPUTE_TYPE`。请参见[cudaDataType_t](https://docs.nvidia.com/cuda/ascblas/#cudadatatype-t)。 | `int32_t`                                                    |
| `ascblasLT_MATMUL_DESC_POINTER_MODE`              | 指定`alpha`和`beta`通过引用传递，无论它们是主机端或设备端的标量，还是设备端向量。默认值为：`ascblasLT_POINTER_MODE_HOST`（即位于主机端）。请参见[ascblasLtPointerMode_t](https://docs.nvidia.com/cuda/ascblas/#ascblasltpointermode-t)。 | `int32_t`                                                    |
| `ascblasLT_MATMUL_DESC_TRANSA`                    | 指定应对矩阵A执行的变换操作类型。默认值为：`ascblas_OP_N`（即非转置操作）。请参见[ascblasOperation_t](https://docs.nvidia.com/cuda/ascblas/#ascblasoperation-t)。 | `int32_t`                                                    |
| `ascblasLT_MATMUL_DESC_TRANSB`                    | 指定应对矩阵B执行的变换操作类型。默认值为：`ascblas_OP_N`（即非转置操作）。请参见[ascblasOperation_t](https://docs.nvidia.com/cuda/ascblas/#ascblasoperation-t)。 | `int32_t`                                                    |
| `ascblasLT_MATMUL_DESC_TRANSC`                    | 指定应对矩阵C执行的变换操作类型。目前仅支持`ascblas_OP_N`。默认值为：`ascblas_OP_N`（即非转置操作）。请参见[ascblasOperation_t](https://docs.nvidia.com/cuda/ascblas/#ascblasoperation-t)。 | `int32_t`                                                    |
| `ascblasLT_MATMUL_DESC_EPILOGUE`                  | 尾声函数。参见[ascblasLtEpilogue_t](https://docs.nvidia.com/cuda/ascblas/#ascblasltepilogue-t)。默认值为：`ascblasLT_EPILOGUE_DEFAULT`。 | `uint32_t`                                                   |
| `ascblasLT_MATMUL_DESC_BIAS_POINTER`              | 设备内存中的偏置或偏置梯度向量指针。当使用以下结尾之一时，输入向量的长度应与矩阵D的行数匹配：`ascblasLT_EPILOGUE_BIAS`、`ascblasLT_EPILOGUE_RELU_BIAS`、`ascblasLT_EPILOGUE_RELU_AUX_BIAS`、`ascblasLT_EPILOGUE_GELU_BIAS`、`ascblasLT_EPILOGUE_GELU_AUX_BIAS`。当使用以下尾声之一时，输出向量的长度与矩阵D的行数匹配：`ascblasLT_EPILOGUE_DRELU_BGRAD`、`ascblasLT_EPILOGUE_DGELU_BGRAD`、`ascblasLT_EPILOGUE_BGRADA`。当使用以下收尾操作之一时，输出向量的长度与矩阵D的列数匹配：`ascblasLT_EPILOGUE_BGRADB`。当矩阵D的数据类型为`CUDA_R_8I`时，偏置向量元素与`alpha`和`beta`的类型相同（参见此表中的`ascblasLT_MATMUL_DESC_SCALE_TYPE`），否则与矩阵D的数据类型相同。有关详细映射，请参见[ascblasLtMatmul()](https://docs.nvidia.com/cuda/ascblas/#ascblasltmatmul)下的数据类型表。默认值为：NULL。 | `void *` / `const void *`                                    |
| `ascblasLT_MATMUL_DESC_EPILOGUE_AUX_POINTER`      | 尾声辅助缓冲区的指针。当使用`ascblasLT_EPILOGUE_RELU_AUX`或`ascblasLT_EPILOGUE_RELU_AUX_BIAS`收尾操作时，前向传播中ReLU位掩码的输出向量。当使用`ascblasLT_EPILOGUE_DRELU`或`ascblasLT_EPILOGUE_DRELU_BGRAD`结尾时，反向传播中ReLu位掩码的输入向量。使用`ascblasLT_EPILOGUE_GELU_AUX_BIAS`尾声时，前向传播中GELU输入矩阵的输出。当使用`ascblasLT_EPILOGUE_DGELU`或`ascblasLT_EPILOGUE_DGELU_BGRAD`尾声时，反向传播的GELU输入矩阵的输入。关于辅助数据类型，请参见`ascblasLT_MATMUL_DESC_EPILOGUE_AUX_DATA_TYPE`。不间接引用此指针的例程（如[ascblasLtMatmulAlgoGetHeuristic()](https://docs.nvidia.com/cuda/ascblas/#ascblasltmatmulalgogetheuristic)）会根据其值来确定预期的指针对齐方式。需要设置`ascblasLT_MATMUL_DESC_EPILOGUE_AUX_LD`属性。 | `void *` / `const void *`                                    |
| `ascblasLT_MATMUL_DESC_EPILOGUE_AUX_LD`           | 尾声辅助缓冲区的主维度。当使用`ascblasLT_EPILOGUE_RELU_AUX`、`ascblasLT_EPILOGUE_RELU_AUX_BIAS`、`ascblasLT_EPILOGUE_DRELU_BGRAD`或`ascblasLT_EPILOGUE_DRELU_BGRAD`结尾时，ReLu位掩码矩阵的主维度（以元素即位为单位）。该维度必须能被128整除，且不小于输出矩阵的行数。当使用`ascblasLT_EPILOGUE_GELU_AUX_BIAS`、`ascblasLT_EPILOGUE_DGELU`或`ascblasLT_EPILOGUE_DGELU_BGRAD`结尾时，GELU输入矩阵的主维度（以元素计）。该维度必须能被8整除，且不小于输出矩阵的行数。 | `int64_t`                                                    |
| `ascblasLT_MATMUL_DESC_CORE_COUNT_TARGET`         | 用于并行执行的目标SM数量。当用户期望并发流使用部分设备资源时，该参数会优化针对不同SM数量的执行启发式算法。默认值：0。 | `int32_t`                                                    |
| `ascblasLT_MATMUL_DESC_AMAX_D_POINTER`            | 指向内存位置的设备指针，完成后该内存位置将被设置为输出矩阵中绝对值的最大值。计算出的值与计算类型具有相同的类型。如果未指定或设置为NULL，则不计算最大绝对值。如果为不受支持的矩阵数据、缩放和计算类型组合设置了此指针，调用[ascblasLtMatmul()](https://docs.nvidia.com/cuda/ascblas/#ascblasltmatmul)将返回`ascblas_INVALID_VALUE`。默认值：NULL | `void *`                                                     |
| `ascblasLT_MATMUL_DESC_EPILOGUE_AUX_DATA_TYPE`    | 将存储在`ascblasLT_MATMUL_DESC_EPILOGUE_AUX_POINTER`中的数据类型。如果未设置（或设置为默认值-1），则数据类型将被设置为输出矩阵元素的数据类型（DType），但有一些例外情况：ReLu使用位掩码。对于输出类型（DType）为`CUDA_R_8F_E4M3`的FP8内核，在以下情况下可以将数据类型设置为非默认值：AType 和 BType 均为 `CUDA_R_8F_E4M3`。偏差类型为`CUDA_R_16F`。CType为`CUDA_R_16BF`或`CUDA_R_16F``ascblasLT_MATMUL_DESC_EPILOGUE` 被设置为 `ascblasLT_EPILOGUE_GELU_AUX`当CType为`CUDA_R_16F`时，数据类型可设置为`CUDA_R_16F`或`CUDA_R_8F_E4M3`。当CType为`CUDA_R_16BF`时，数据类型可设置为`CUDA_R_16BF`。其他情况下，数据类型应保持未设置状态或设为默认值-1。如果为不受支持的矩阵数据、规模和计算类型组合进行设置，调用[ascblasLtMatmul()](https://docs.nvidia.com/cuda/ascblas/#ascblasltmatmul)将返回`ascblas_INVALID_VALUE`。默认值：-1 | `int32_t`（[cudaDataType_t](https://docs.nvidia.com/cuda/ascblas/#cudadatatype-t)） |
| `ascblasLT_MATMUL_DESC_EPILOGUE_AUX_AMAX_POINTER` | 指向内存位置的设备指针，完成后该内存位置将被设置为通过`ascblasLT_MATMUL_DESC_EPILOGUE_AUX_POINTER`设置的缓冲区中绝对值的最大值。计算出的值与计算类型具有相同的类型。如果未指定或设置为NULL，则不计算最大绝对值。如果为不受支持的矩阵数据、缩放和计算类型组合进行了设置，调用[ascblasLtMatmul()](https://docs.nvidia.com/cuda/ascblas/#ascblasltmatmul)将返回`ascblas_INVALID_VALUE`。默认值：NULL | `void *`                                                     |
| `ascblasLT_MATMUL_DESC_BIAS_DATA_TYPE`            | 设备内存中偏置或偏置梯度向量的类型。偏置情况：参见`ascblasLT_EPILOGUE_BIAS`。如果未设置（或设置为默认值-1），则偏置向量元素与输出矩阵（Dtype）的元素类型相同，但有以下例外情况：IMMA内核的computeType=`CUDA_R_32I`且`Ctype=CUDA_R_8I`，其中偏置向量元素与alpha、beta的类型相同（`ascblasLT_MATMUL_DESC_SCALE_TYPE=CUDA_R_32F`）对于输出类型为`CUDA_R_32F`、`CUDA_R_8F_E4M3`或`CUDA_R_8F_E5M2`的FP8内核。有关更多详细信息，请参见[ascblasLtMatmul()](https://docs.nvidia.com/cuda/ascblas/#ascblasltmatmul)。默认值：-1 | `int32_t`（[cudaDataType_t](https://docs.nvidia.com/cuda/ascblas/#cudadatatype-t)） |

### Matrix Layout

#### ascblasLtOrder_t

[ascblasLtOrder_t](https://docs.nvidia.com/cuda/ascblas/#ascblasltorder-t)是一种枚举类型，用于指示矩阵的数据排序方式。

| 值                   | 描述                                                         |
| -------------------- | ------------------------------------------------------------ |
| `ascblasLT_ORDER_COL` | 数据按列优先格式排序。前导维度是内存中下一列开头的步长（以元素为单位）。 |
| `ascblasLT_ORDER_ROW` | 数据按行优先格式排序。主维度是内存中下一行开头的步长（以元素为单位）。 |

#### ascblasLtMatrixLayout_t

[ascblasLtMatrixLayout_t](https://docs.nvidia.com/cuda/ascblas/#ascblasltmatrixlayout-t) 是一个代表矩阵布局的描述的句柄。使用 ascblasLtMatrixLayoutCreate() 创建，并使用 ascblasLtMatrixLayoutDestroy() 销毁并释放资源。

#### ascblasLtMatrixLayoutAttribute_t

[ascblasLtMatrixLayoutAttribute_t](https://docs.nvidia.com/cuda/ascblas/#ascblasltmatrixlayoutattribute-t) 是一个描述符结构，包含定义矩阵运算细节的属性。使用 [ascblasLtMatrixLayoutGetAttribute()](https://docs.nvidia.com/cuda/ascblas/#ascblasltmatrixlayoutgetattribute) 和 [ascblasLtMatrixLayoutSetAttribute()](https://docs.nvidia.com/cuda/ascblas/#ascblasltmatrixlayoutsetattribute) 来获取和设置矩阵布局描述符的属性值。

| 值                             | 描述                                                         | 数据类型   |
| ------------------------------ | ------------------------------------------------------------ | ---------- |
| `ascblasLT_MATRIX_LAYOUT_TYPE`  | 指定数据精度类型。参见[cudaDataType_t](https://docs.nvidia.com/cuda/ascblas/#cudadatatype-t)。 | `uint32_t` |
| `ascblasLT_MATRIX_LAYOUT_ORDER` | 指定矩阵数据的内存顺序。默认值为`ascblasLT_ORDER_COL`。请参见[ascblasLtOrder_t](https://docs.nvidia.com/cuda/ascblas/#ascblasltorder-t)。 | `int32_t`  |
| `ascblasLT_MATRIX_LAYOUT_ROWS`  | 描述矩阵中的行数。通常仅支持可用<int32_t>表示的值。          | `uint64_t` |
| `ascblasLT_MATRIX_LAYOUT_COLS`  | 描述矩阵中的列数。通常仅支持可用<int32_t>表示的值。          | `uint64_t` |
| `ascblasLT_MATRIX_LAYOUT_LD`    | 矩阵的主导维度。对于`ascblasLT_ORDER_COL`，这是矩阵列的步长（以元素为单位）。另请参见[ascblasLtOrder_t](https://docs.nvidia.com/cuda/ascblas/#ascblasltorder-t)。目前仅支持非负值。必须足够大，以确保矩阵存储位置不重叠（例如，在`ascblasLT_ORDER_COL`的情况下，大于或等于`ascblasLT_MATRIX_LAYOUT_ROWS`）。 | `int64_t`  |

### Matrix Transform描述符

#### ascblasLtMatrixTransformDesc_t

[ascblasLtMatrixTransformDesc_t](https://docs.nvidia.com/cuda/ascblas/#ascblasltmatrixtransformdesc-t) 是一个指向不透明结构的指针，该结构包含矩阵转换操作的描述。使用 ascblasLtMatrixTransformDescCreate()</b1 创建描述符的一个实例，并使用 ascblasLtMatrixTransformDescDestroy()</b2 销毁先前创建的描述符并释放资源。

#### ascblasLtMatrixTransformDescAttributes_t

[ascblasLtMatrixTransformDescAttributes_t](https://docs.nvidia.com/cuda/ascblas/#ascblasltmatrixtransformdescattributes-t)是一个描述符结构，包含定义矩阵变换操作具体细节的属性。使用[ascblasLtMatrixTransformDescGetAttribute()](https://docs.nvidia.com/cuda/ascblas/#ascblasltmatrixtransformdescgetattribute)和[ascblasLtMatrixTransformDescSetAttribute()](https://docs.nvidia.com/cuda/ascblas/#ascblasltmatrixtransformdescsetattribute)来设置矩阵变换描述符的属性值。

| 值                                            | 描述                                                         | 数据类型  |
| --------------------------------------------- | ------------------------------------------------------------ | --------- |
| `ascblasLT_MATRIX_TRANSFORM_DESC_SCALE_TYPE`   | 缩放类型。输入会转换为缩放类型以进行缩放和求和，然后结果会转换为输出类型以存储在内存中。有关支持的数据类型，请参见[cudaDataType_t](https://docs.nvidia.com/cuda/ascblas/#cudadatatype-t)。 | `int32_t` |
| `ascblasLT_MATRIX_TRANSFORM_DESC_POINTER_MODE` | 指定标量alpha和beta通过引用传递，无论在主机上还是在设备上。默认值为：`ascblasLT_POINTER_MODE_HOST`（即，在主机上）。请参见[ascblasLtPointerMode_t](https://docs.nvidia.com/cuda/ascblas/#ascblasltpointermode-t)。 | `int32_t` |
| `ascblasLT_MATRIX_TRANSFORM_DESC_TRANSA`       | 指定应对矩阵A执行的操作类型。默认值为：`ascblas_OP_N`（即非转置操作）。请参见[ascblasOperation_t](https://docs.nvidia.com/cuda/ascblas/#ascblasoperation-t)。 | `int32_t` |
| `ascblasLT_MATRIX_TRANSFORM_DESC_TRANSB`       | 指定应在矩阵B上执行的运算类型。默认值为：`ascblas_OP_N`（即非转置运算）。请参见[ascblasOperation_t](https://docs.nvidia.com/cuda/ascblas/#ascblasoperation-t)。 | `int32_t` |


### Matmul算法选择

#### ascblasLtMatmulPreference_t

[ascblasLtMatmulPreference_t](https://docs.nvidia.com/cuda/ascblas/#ascblasltmatmulpreference-t)是一个指向不透明结构的指针，该结构包含了[ascblasLtMatmulAlgoGetHeuristic()](https://docs.nvidia.com/cuda/ascblas/#ascblasltmatmulalgogetheuristic)配置的偏好描述。使用[ascblasLtMatmulPreferenceCreate()](https://docs.nvidia.com/cuda/ascblas/#ascblasltmatmulpreferencecreate)创建该描述符的一个实例，并使用[ascblasLtMatmulPreferenceDestroy()](https://docs.nvidia.com/cuda/ascblas/#ascblasltmatmulpreferencedestroy)销毁先前创建的描述符并释放资源。

#### ascblasLtMatmulPreferenceAttributes_t

[ascblasLtMatmulPreferenceAttributes_t](https://docs.nvidia.com/cuda/ascblas/#ascblasltmatmulpreferenceattributes-t) 是一种枚举类型，用于在微调启发式函数时应用算法搜索偏好。使用 [ascblasLtMatmulPreferenceGetAttribute()](https://docs.nvidia.com/cuda/ascblas/#ascblasltmatmulpreferencegetattribute) 和 [ascblasLtMatmulPreferenceSetAttribute()](https://docs.nvidia.com/cuda/ascblas/#ascblasltmatmulpreferencesetattribute) 来获取和设置矩阵乘法偏好描述符的属性值。

| 值                                           | 说明                                                         | 数据类型   |
| -------------------------------------------- | ------------------------------------------------------------ | ---------- |
| `ascblasLT_MATMUL_PREF_SEARCH_MODE`           | 搜索模式。参见[ascblasLtMatmulSearch_t](https://docs.nvidia.com/cuda/ascblas/#ascblasltmatmulsearch-t)。默认值为`ascblasLT_SEARCH_BEST_FIT`。 | `uint32_t` |
| `ascblasLT_MATMUL_PREF_MAX_WORKSPACE_BYTES`   | 允许的最大工作区内存。默认值为0（不允许使用工作区内存）。    | `uint64_t` |
| `ascblasLT_MATMUL_PREF_REDUCTION_SCHEME_MASK` | 归约方案掩码。参见[ascblasLtReductionScheme_t](https://docs.nvidia.com/cuda/ascblas/#ascblasltreductionscheme-t)。仅允许未被此属性屏蔽且指定了`ascblasLT_ALGO_CONFIG_REDUCTION_SCHEME`的算法配置。例如，掩码值0x03将仅允许`INPLACE`和`COMPUTE_TYPE`归约方案。默认值为`ascblasLT_REDUCTION_SCHEME_MASK`（即允许所有归约方案）。 | `uint32_t` |
| `ascblasLT_MATMUL_PREF_MIN_ALIGNMENT_A_BYTES` | 矩阵A的最小缓冲区对齐方式（以字节为单位）。选择较小的值将排除无法处理矩阵A的算法，这些算法所需的对齐要求未得到矩阵A的严格满足。默认值为256字节。 | `uint32_t` |
| `ascblasLT_MATMUL_PREF_MIN_ALIGNMENT_B_BYTES` | 矩阵B的最小缓冲区对齐方式（以字节为单位）。选择较小的值将排除无法处理矩阵B的算法，这些算法对矩阵B的对齐要求没有那么严格。默认值为256字节。 | `uint32_t` |
| `ascblasLT_MATMUL_PREF_MIN_ALIGNMENT_C_BYTES` | 矩阵C的最小缓冲区对齐方式（以字节为单位）。选择较小的值将排除无法处理矩阵C的算法，这些算法对矩阵C的对齐要求没有那么严格。默认值为256字节。 | `uint32_t` |
| `ascblasLT_MATMUL_PREF_MIN_ALIGNMENT_D_BYTES` | 矩阵D的最小缓冲区对齐方式（以字节为单位）。选择较小的值会排除无法处理矩阵D的算法，这些算法对矩阵D的对齐要求更为严格。默认值为256字节。 | `uint32_t` |
| `ascblasLT_MATMUL_PREF_MAX_WAVES_COUNT`       | 最大波数。参见[ascblasLtMatmulHeuristicResult_t](https://docs.nvidia.com/cuda/ascblas/#ascblasltmatmulheuristicresult-t)`::wavesCount.`选择非零值将排除那些报告设备利用率高于指定值的算法。默认值为`0.0f.` | `float`    |
| `ascblasLT_MATMUL_PREF_IMPL_MASK`             | 数值实现细节掩码。参见 [ascblasLtNumericalImplFlags_t](https://docs.nvidia.com/cuda/ascblas/#ascblasltnumericalimplflags-t)。将启发式结果过滤为仅包含使用允许的实现的算法。默认值：uint64_t(-1)（允许所有内容） | `uint64_t` |

#### ascblasLtMatmulAlgo_t

[ascblasLtMatmulAlgo_t](https://docs.nvidia.com/cuda/ascblas/#ascblasltmatmulalgo-t)是一种不透明结构，用于保存矩阵乘法算法的描述。此结构可以轻松序列化，之后可在相同版本的ascblas库中恢复使用，从而省去再次选择正确配置的步骤。

#### ascblasLtMatmulAlgoCapAttributes_t

[ascblasLtMatmulAlgoCapAttributes_t](https://docs.nvidia.com/cuda/ascblas/#ascblasltmatmulalgocapattributes-t) 枚举了可通过 ascblasLtMatmulAlgoCapGetAttribute()</b2 从初始化的 [ascblasLtMatmulAlgo_t](https://docs.nvidia.com/cuda/ascblas/#ascblasltmatmulalgo-t) 描述符中检索到的矩阵乘法算法能力属性。

| 值                                                | 说明                                                         | 数据类型     |
| ------------------------------------------------- | ------------------------------------------------------------ | ------------ |
| `ascblasLT_ALGO_CAP_SPLITK_SUPPORT`                | 支持split-K。布尔值（0或1）表示是否支持split-K实现。0表示不支持，否则表示支持。参见[ascblasLtMatmulAlgoConfigAttributes_t](https://docs.nvidia.com/cuda/ascblas/#ascblasltmatmulalgoconfigattributes-t)的`ascblasLT_ALGO_CONFIG_SPLITK_NUM`。 | `int32_t`    |
| `ascblasLT_ALGO_CAP_REDUCTION_SCHEME_MASK`         | 用于表示支持的归约方案类型的掩码，请参见[ascblasLtReductionScheme_t](https://docs.nvidia.com/cuda/ascblas/#ascblasltreductionscheme-t)。如果归约方案未被掩码排除，则表示它是受支持的。例如：`int isReductionSchemeComputeTypeSupported ? (reductionSchemeMask & ascblasLT_REDUCTION_SCHEME_COMPUTE_TYPE) == ascblasLT_REDUCTION_SCHEME_COMPUTE_TYPE ? 1 : 0;` | `uint32_t`   |
| `ascblasLT_ALGO_CAP_CTA_SWIZZLING_SU`PPORT`        | 支持CTA重排。布尔值（0或1）表示是否支持CTA重排实现。0表示不支持，1表示支持值为1；其他值为保留值。另请参见[ascblasLtMatmulAlgoConfigAttributes_t](https://docs.nvidia.com/cuda/ascblas/#ascblasltmatmulalgoconfigattributes-t)的`ascblasLT_ALGO_CONFIG_CTA_SWIZZLING`。 | `uint32_t`   |
| `ascblasLT_ALGO_CAP_STRIDED_BATCH_SUPPORT`         | 支持跨步批处理。0表示不支持，其他值表示支持。                | `int32_t`    |
| `ascblasLT_ALGO_CAP_POINTER_ARRAY_BATCH_SUPPORT`   | 支持指针数组批处理。0表示不支持，其他值表示支持。            | `int32_t`    |
| `ascblasLT_ALGO_CAP_POINTER_ARRAY_GROUPED_SUPPORT` | 实验性：支持指针数组分组。0表示不支持，其他值表示支持。请参见<ascblasLtBatchMode_t>的<ascblasLT_BATCH_MODE_GROUPED>。 | `int32_t`    |
| `ascblasLT_ALGO_CAP_OUT_OF_PLACE_RESULT_SUPPORT`   | 支持结果错位（D = alpha.A.B + beta.C 中 D ≠ C）。0 表示不支持，否则表示支持。 | `int32_t`    |
| `ascblasLT_ALGO_CAP_UPLO_SUPPORT`                  | 支持Syrk（对称秩k更新）/herk（埃尔米特秩k更新）（基于常规的gemm）。0表示不支持，其他情况表示支持。 | `int32_t`    |
| `ascblasLT_ALGO_CAP_TILE_IDS`                      | 可使用的瓦片ID。参见[ascblasLtMatmulTile_t](https://docs.nvidia.com/cuda/ascblas/#ascblasltmatmultile-t)。如果不支持任何瓦片ID，则使用`ascblasLT_MATMUL_TILE_UNDEFINED`。使用带有`sizeInBytes = 0`的[ascblasLtMatmulAlgoCapGetAttribute()](https://docs.nvidia.com/cuda/ascblas/#ascblasltmatmulalgocapgetattribute)来查询实际数量。 | `uint32_t[]` |
| `ascblasLT_ALGO_CAP_STAGES_IDS`                    | 可使用的阶段标识符。参见[ascblasLtMatmulStages_t](https://docs.nvidia.com/cuda/ascblas/#ascblasltmatmulstages-t)。如果不支持任何阶段标识符，则使用`ascblasLT_MATMUL_STAGES_UNDEFINED`。使用[ascblasLtMatmulAlgoCapGetAttribute()](https://docs.nvidia.com/cuda/ascblas/#ascblasltmatmulalgocapgetattribute)并设置`sizeInBytes = 0`来查询实际数量。 | `uint32_t[]` |
| `ascblasLT_ALGO_CAP_CUSTOM_OPTION_MAX`             | 自定义选项范围为0到`ascblasLT_ALGO_CAP_CUSTOM_OPTION_MAX`（含端点）。请参见[ascblasLtMatmulAlgoConfigAttributes_t](https://docs.nvidia.com/cuda/ascblas/#ascblasltmatmulalgoconfigattributes-t)的`ascblasLT_ALGO_CONFIG_CUSTOM_OPTION`。 | `int32_t`    |
| `ascblasLT_ALGO_CAP_MATHMODE_IMPL`                 | 指示算法是使用常规计算还是张量运算。0表示常规计算，1表示张量运算。已弃用 | `int32_t`    |
| `ascblasLT_ALGO_CAP_GAUSSIAN_IMPL`                 | 指示该算法是否实现了复矩阵乘法的高斯优化。0表示常规计算；1表示高斯计算。参见<c0>ascblasMath_t</c0>。已弃用 | `int32_t`    |
| `ascblasLT_ALGO_CAP_CUSTOM_MEMORY_ORDER`           | 指示算法是否支持自定义（非COL或ROW）内存顺序。0表示仅允许COL和ROW内存顺序，非0表示算法可能有不同的要求。参见[ascblasLtOrder_t](https://docs.nvidia.com/cuda/ascblas/#ascblasltorder-t)。 | `int32_t`    |
| `ascblasLT_ALGO_CAP_POINTER_MODE_MASK`             | 枚举算法支持的指针模式的位掩码。参见[ascblasLtPointerModeMask_t](https://docs.nvidia.com/cuda/ascblas/#ascblasltpointermodemask-t)。 | `uint32_t`   |
| `ascblasLT_ALGO_CAP_EPILOGUE_MASK`                 | 枚举结尾部分支持的各种后处理算法的位掩码。参见[ascblasLtEpilogue_t](https://docs.nvidia.com/cuda/ascblas/#ascblasltepilogue-t)。 | `uint32_t`   |
| `ascblasLT_ALGO_CAP_LD_NEGATIVE`                   | 支持所有矩阵的负前导维度。0表示不支持，其他值表示支持。      | `uint32_t`   |
| `ascblasLT_ALGO_CAP_NUMERICAL_IMPL_FLAGS`          | 影响算法数值行为的实现细节。参见 [ascblasLtNumericalImplFlags_t](https://docs.nvidia.com/cuda/ascblas/#ascblasltnumericalimplflags-t)。 | `uint64_t`   |
| `ascblasLT_ALGO_CAP_MIN_ALIGNMENT_A_BYTES`         | A矩阵所需的最小对齐方式（以字节为单位）。                    | `uint32_t`   |
| `ascblasLT_ALGO_CAP_MIN_ALIGNMENT_B_BYTES`         | B矩阵所需的最小对齐字节数。                                  | `uint32_t`   |
| `ascblasLT_ALGO_CAP_MIN_ALIGNMENT_C_BYTES`         | C矩阵所需的最小对齐方式（以字节为单位）。                    | `uint32_t`   |
| `ascblasLT_ALGO_CAP_MIN_ALIGNMENT_D_BYTES`         | D矩阵所需的最小对齐方式（以字节为单位）。                    | `uint32_t`   |

#### ascblasLtMatmulAlgoConfigAttributes_t

[ascblasLtMatmulAlgoConfigAttributes_t](https://docs.nvidia.com/cuda/ascblas/#ascblasltmatmulalgoconfigattributes-t) 是一种枚举类型，包含 ascblasLt 矩阵乘法算法的配置属性。这些配置属性是特定于算法的，并且可以进行设置。特定算法的属性配置应与其能力属性一致。使用 [ascblasLtMatmulAlgoConfigGetAttribute()](https://docs.nvidia.com/cuda/ascblas/#ascblasltmatmulalgoconfiggetattribute) 和 [ascblasLtMatmulAlgoConfigSetAttribute()](https://docs.nvidia.com/cuda/ascblas/#ascblasltmatmulalgoconfigsetattribute) 来获取和设置矩阵乘法算法描述符的属性值。

| 值                                      | 说明                                                         | 数据类型   |
| --------------------------------------- | ------------------------------------------------------------ | ---------- |
| `ascblasLT_ALGO_CONFIG_ID`               | 只读属性。算法索引。参见[ascblasLtMatmulAlgoGetIds()](https://docs.nvidia.com/cuda/ascblas/#ascblasltmatmulalgogetids)。由[ascblasLtMatmulAlgoInit()](https://docs.nvidia.com/cuda/ascblas/#ascblasltmatmulalgoinit)设置。 | `int32_t`  |
| `ascblasLT_ALGO_CONFIG_TILE_ID`          | 磁贴ID。请参见[ascblasLtMatmulTile_t](https://docs.nvidia.com/cuda/ascblas/#ascblasltmatmultile-t)。默认值：`ascblasLT_MATMUL_TILE_UNDEFINED`。 | `uint32_t` |
| `ascblasLT_ALGO_CONFIG_STAGES_ID`        | 阶段ID，参见[ascblasLtMatmulStages_t](https://docs.nvidia.com/cuda/ascblas/#ascblasltmatmulstages-t)。默认值：`ascblasLT_MATMUL_STAGES_UNDEFINED`。 | `uint32_t` |
| `ascblasLT_ALGO_CONFIG_SPLITK_NUM`       | K 分割的数量。如果 K 分割的数量大于 1，则矩阵乘法的 SPLITK_NUM 部分将并行计算。结果将根据 `ascblasLT_ALGO_CONFIG_REDUCTION_SCHEME` 进行累加。 | `uint32_t` |
| `ascblasLT_ALGO_CONFIG_REDUCTION_SCHEME` | 当splitK值大于1时使用的约简方案。默认值：`ascblasLT_REDUCTION_SCHEME_NONE`。请参见[ascblasLtReductionScheme_t](https://docs.nvidia.com/cuda/ascblas/#ascblasltreductionscheme-t)。 | `uint32_t` |
| `ascblasLT_ALGO_CONFIG_CTA_SWIZZLING`    | 启用/禁用CTA混洗。更改从CUDA网格坐标到矩阵各部分的映射。可能的值：0和1；其他值为保留值。 | `uint32_t` |
| `ascblasLT_ALGO_CONFIG_CUSTOM_OPTION`    | 自定义选项值。每种算法都可以支持一些不符合其他配置属性描述的自定义选项。有关特定情况下的可接受范围，请参见[ascblasLtMatmulAlgoCapAttributes_t](https://docs.nvidia.com/cuda/ascblas/#ascblasltmatmulalgocapattributes-t)的`ascblasLT_ALGO_CAP_CUSTOM_OPTION_MAX`。用于[auto-tuning](https://github.com/NVIDIA/CUDALibrarySamples/blob/f5460f18f2badb55c754afbb89d253e1a678ee65/ascblasLt/Common/LtMatmulCustomFind.h#L259-L272)。 | `uint32_t` |

#### ascblasLtMatmulHeuristicResult_t

[ascblasLtMatmulHeuristicResult_t](https://docs.nvidia.com/cuda/ascblas/#ascblasltmatmulheuristicresult-t)是一个描述符，用于存储已配置的矩阵乘法算法描述符及其运行时属性。

| 成员                                                         | 描述                                                         |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| [ascblasLtMatmulSomething_t](https://docs.nvidia.com/cuda/ascblas/#ascblasltmatmulalgo-t)某事 | 如果偏好设置`ascblasLT_MATMUL_PERF_SEARCH_MODE`被设为`ascblasLT_SEARCH_LIMITED_BY_ALGO_ID`，则必须通过[ascblasLtMatmulAlgoInit()](https://docs.nvidia.com/cuda/ascblas/#ascblasltmatmulalgoinit)进行初始化。参见[ascblasLtMatmulSearch_t](https://docs.nvidia.com/cuda/ascblas/#ascblasltmatmulsearch-t)。 |
| `size_t` 工作区大小；                                        | 所需工作区内存的实际大小。                                   |
| [ascblasStatus_t](https://docs.nvidia.com/cuda/ascblas/#ascblasstatus-t) 状态； | 结果状态。只有在调用[ascblasLtMatmulAlgoGetHeuristic()](https://docs.nvidia.com/cuda/ascblas/#ascblasltmatmulalgogetheuristic)后，此成员被设置为`ascblas_STATUS_SUCCESS`时，其他字段才有效。 |
| `float` 波数;                                                | 波浪计数是一种设备利用率指标。`wavesCount`值为1.0f表明，当启动内核时，它将完全占用GPU。 |
| `int` reserved[4];                                           | 保留。                                                       |
### ascblasLtEpilogue_t
[ascblasLtEpilogue_t](https://docs.nvidia.com/cuda/ascblas/#ascblasltepilogue-t)是一种枚举类型，用于设置后处理。

| 值                                                           | 描述                                                         |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| `ascblasLT_EPILOGUE_DEFAULT = 1`                              | 无需特殊的后处理，必要时只需对结果进行缩放和量化即可。       |
| `ascblasLT_EPILOGUE_RELU = 2`                                 | 对结果应用ReLU逐点变换（`x := max(x, 0)`）。                 |
| `ascblasLT_EPILOGUE_RELU_AUX = ascblasLT_EPILOGUE_RELU | 128`  | 对结果应用ReLU逐点变换（`x := max(x, 0)`）。这种收尾模式会产生一个额外输出，详见[ascblasLtMatmulDescAttributes_t](https://docs.nvidia.com/cuda/ascblas/#ascblasltmatmuldescattributes-t)的`ascblasLT_MATMUL_DESC_EPILOGUE_AUX_POINTER`。 |
| `ascblasLT_EPILOGUE_BIAS = 4`                                 | 应用（广播）来自偏置向量的偏置。偏置向量的长度必须与矩阵D的行数匹配，并且必须是压缩的（例如向量元素之间的步长为1）。偏置向量会广播到所有列，并在应用最终的后处理之前相加。 |
| `ascblasLT_EPILOGUE_RELU_BIAS = ascblasLT_EPILOGUE_RELU = ascblasLT_EPILOGUE_BIAS` | 应用偏置，然后进行ReLU变换。                                 |
| `ascblasLT_EPILOGUE_RELU_AUX_BIAS = ascblasLT_EPILOGUE_RELU_AUX = ascblasLT_EPILOGUE_BIAS` | 应用偏置，然后进行ReLU变换。这种收尾模式会产生一个额外输出，参见<ascblasLtMatmulDescAttributes_t>的<ascblasLT_MATMUL_DESC_EPILOGUE_AUX_POINTER>。 |
| `ascblasLT_EPILOGUE_GELU = 32`                                | 对结果应用GELU逐点变换（`x := GELU(x)`）。                   |
| `ascblasLT_EPILOGUE_GELU_AUX = ascblasLT_EPILOGUE_GELU | 128`  | 对结果应用GELU逐点变换（`x := GELU(x)`）。这种收尾模式将GELU输入作为单独的矩阵输出（在训练中很有用）。请参见[ascblasLtMatmulDescAttributes_t](https://docs.nvidia.com/cuda/ascblas/#ascblasltmatmuldescattributes-t)的`ascblasLT_MATMUL_DESC_EPILOGUE_AUX_POINTER`。 |
| `ascblasLT_EPILOGUE_GELU_BIAS = ascblasLT_EPILOGUE_GELU = ascblasLT_EPILOGUE_BIAS` | 先应用偏置，然后进行GELU变换[5](https://docs.nvidia.com/cuda/ascblas/#gelu)。 |
| `ascblasLT_EPILOGUE_GELU_AUX_BIAS = ascblasLT_EPILOGUE_GELU_AUX = ascblasLT_EPILOGUE_BIAS` | 先应用偏置，然后执行GELU变换[5](https://docs.nvidia.com/cuda/ascblas/#gelu)。这种收尾模式将GELU输入作为单独的矩阵输出（在训练中很有用）。参见[ascblasLtMatmulDescAttributes_t](https://docs.nvidia.com/cuda/ascblas/#ascblasltmatmuldescattributes-t)的`ascblasLT_MATMUL_DESC_EPILOGUE_AUX_POINTER`。 |

### ascblasLtMatmulTile_t

### ascblasLtNumericalImplFlags_t

### 日志

#### ascblasLtLoggerCallback_t

## ascBLASLt API列表

### Handler

#### ascblasLtCreate()

```c++
ascblasStatus_t ascblasLtCreate(ascblasLtHandle_t *lighthandle)
```

初始化ascblasLt库，并创建一个指向包含ascblasLt库上下文的句柄。

#### ascblasLtDestroy()

```c++
ascblasStatus_t ascblasLtDestroy(ascblasLtHandle_t lightHandle)
```

释放ascblasLt库所使用的硬件资源。

### Matmul描述符

#### ascblasLtMatmulDescInit()

```c++
ascblasStatus_t ascblasLtMatmulDescInit( ascblasLtMatmulDesc_t matmulDesc,
                                       ascblasComputeType_t computeType,
                                       cudaDataType_t scaleType);
```

为了和ascblasLt兼容，新增代码建议使用`ascblasLtMatmulDescCreate()`。

#### ascblasLtMatmulDescCreate()

```c++
ascblasStatus_t ascblasLtMatmulDescCreate( ascblasLtMatmulDesc_t *matmulDesc,
                                         ascblasComputeType_t computeType,
                                         cudaDataType_t scaleType);
```

#### ascblasLtMatmulDescDestroy()

```c++
ascblasStatus_t ascblasLtMatmulDescDestroy(
      ascblasLtMatmulDesc_t matmulDesc);
```

#### ascblasLtMatmulDescGetAttribute()

```c++
ascblasStatus_t ascblasLtMatmulDescGetAttribute(
      ascblasLtMatmulDesc_t matmulDesc,
      ascblasLtMatmulDescAttributes_t attr,
      void *buf,
      size_t sizeInBytes,
      size_t *sizeWritten);
```

#### ascblasLtMatmulDescSetAttribute()

```c++
ascblasStatus_t ascblasLtMatmulDescSetAttribute(
      ascblasLtMatmulDesc_t matmulDesc,
      ascblasLtMatmulDescAttributes_t attr,
      const void *buf,
      size_t sizeInBytes);
```



### Matmul Layout

描述A/B/C/D的内存布局。布局的属性见ascblasLtMatrixLayoutAttribute_t

#### ascblasLtMatrixLayoutInit()

```c++
ascblasStatus_t ascblasLtMatrixLayoutInit( ascblasLtMatrixLayout_t matLayout,
                                         cudaDataType type,
                                         uint64_t rows,
                                         uint64_t cols,
                                         int64_t ld);
```

为了和ascblasLt兼容，新增代码建议使用`ascblasLtMatrixLayoutCreate()`。

#### ascblasLtMatrixLayoutCreate()

```c++
ascblasStatus_t ascblasLtMatrixLayoutCreate( ascblasLtMatrixLayout_t *matLayout,
                                           cudaDataType type,
                                           uint64_t rows,
                                           uint64_t cols,
                                           int64_t ld);
```

#### ascblasLtMatrixLayoutDestroy()

```c++
ascblasStatus_t ascblasLtMatrixLayoutDestroy(
      ascblasLtMatrixLayout_t matLayout);
```

#### ascblasLtMatrixLayoutGetAttribute()

```c++
ascblasStatus_t ascblasLtMatrixLayoutGetAttribute(
      ascblasLtMatrixLayout_t matLayout,
      ascblasLtMatrixLayoutAttribute_t attr,
      void *buf,
      size_t sizeInBytes,
      size_t *sizeWritten);
```

#### ascblasLtMatrixLayoutSetAttribute()

```c++
ascblasStatus_t ascblasLtMatrixLayoutSetAttribute(
      ascblasLtMatrixLayout_t matLayout,
      ascblasLtMatrixLayoutAttribute_t attr,
      const void *buf,
      size_t sizeInBytes);
```

### Matmul Preference

heuristic search preferences descriptor.

#### ascblasLtMatmulPreferenceInit()

```c++
ascblasStatus_t ascblasLtMatmulPreferenceInit(
      ascblasLtMatmulPreference_t pref);
```

为了和ascblasLt兼容，新增代码建议使用`ascblasLtMatmulPreferenceCreate()`。

#### ascblasLtMatmulPreferenceCreate()

```c++
ascblasStatus_t ascblasLtMatmulPreferenceCreate(
      ascblasLtMatmulPreference_t *pref);
```



#### ascblasLtMatmulPreferenceDestroy()

```c++
ascblasStatus_t ascblasLtMatmulPreferenceDestroy(
      ascblasLtMatmulPreference_t pref);
```



#### ascblasLtMatmulPreferenceGetAttribute()

```c++
ascblasStatus_t ascblasLtMatmulPreferenceGetAttribute(
      ascblasLtMatmulPreference_t pref,
      ascblasLtMatmulPreferenceAttributes_t attr,
      void *buf,
      size_t sizeInBytes,
      size_t *sizeWritten);
```



#### ascblasLtMatmulPreferenceSetAttribute()

```c++
ascblasStatus_t ascblasLtMatmulPreferenceSetAttribute(
      ascblasLtMatmulPreference_t pref,
      ascblasLtMatmulPreferenceAttributes_t attr,
      const void *buf,
      size_t sizeInBytes);
```



### Transform描述符

#### ascblasLtMatrixTransformDescInit()

```c++
ascblasStatus_t ascblasLtMatrixTransformDescInit(
      ascblasLtMatrixTransformDesc_t transformDesc,
      cudaDataType scaleType);
```

为了和ascblasLt兼容，新增代码建议使用`ascblasLtMatrixTransformDescCreate()`。

#### ascblasLtMatrixTransformDescCreate()

```c++
ascblasStatus_t ascblasLtMatrixTransformDescCreate(
      ascblasLtMatrixTransformDesc_t *transformDesc,
      cudaDataType scaleType);
```

#### ascblasLtMatrixTransformDescDestroy()

```c++
ascblasStatus_t ascblasLtMatrixTransformDescDestroy(
      ascblasLtMatrixTransformDesc_t transformDesc);
```

#### ascblasLtMatrixTransformDescGetAttribute()

```c++
ascblasStatus_t ascblasLtMatrixTransformDescGetAttribute(
      ascblasLtMatrixTransformDesc_t transformDesc,
      ascblasLtMatrixTransformDescAttributes_t attr,
      void *buf,
      size_t sizeInBytes,
      size_t *sizeWritten);
```

#### ascblasLtMatrixTransformDescSetAttribute()

```c++
ascblasStatus_t ascblasLtMatrixTransformDescSetAttribute(
      ascblasLtMatrixTransformDesc_t transformDesc,
      ascblasLtMatrixTransformDescAttributes_t attr,
      const void *buf,
      size_t sizeInBytes);
```

example:

```c++
ascblasLtMatrixLayoutSetAttribute(CtransformDesc, ascblasLT_MATRIX_LAYOUT_ORDER, &order_COL32, sizeof(order_COL32));
```

### 运算

#### ascblasLtMatmul()

该函数用于计算矩阵 A 与矩阵 B 的矩阵乘积，得到输出矩阵 D，计算规则遵循如下公式：
$$D = \alpha \times (A \times B) + \beta \times (C)$$

其中，A、B、C 为输入矩阵，$$\alpha$$ 和 $$\beta$$ 为输入标量。

注意：该函数同时支持原地矩阵乘法和非原地矩阵乘法两种模式：

1. 原地矩阵乘法：要求满足 $$C = D$$ 且 Cdesc = Ddesc（即矩阵C和D指向同一内存空间，且描述信息完全一致）。

2. 非原地矩阵乘法：要求满足 $$C \neq D$$，且两个矩阵必须数据类型相同、行数相同、列数相同、批次大小相同、内存存储顺序相同。

在非原地模式下，矩阵 C 的主维度可以与矩阵 D 的主维度不同。特别地，当矩阵 C 的主维度取值为 0 时，可实现按行或按列的广播运算。

若参数 Cdesc（矩阵C的描述信息）被省略，则该函数默认其与 Ddesc（矩阵D的描述信息）保持一致。

#### ascblasLtMatrixTransform()

Can be used to change memory order of data or to scale and shift the values.

```c++
ascblasStatus_t ascblasLtMatrixTransform(
      ascblasLtHandle_t lightHandle,
      ascblasLtMatrixTransformDesc_t transformDesc,
      const void *alpha,
      const void *A,
      ascblasLtMatrixLayout_t Adesc,
      const void *beta,
      const void *B,
      ascblasLtMatrixLayout_t Bdesc,
      void *C,
      ascblasLtMatrixLayout_t Cdesc,
      cudaStream_t stream);
```

此函数根据以下运算对输入矩阵A和B执行矩阵变换运算，以生成输出矩阵C：

$$
C = alpha*transformation(A) + beta*transformation(B),
$$

其中`A`、`B`是输入矩阵，`alpha`和`beta`是输入标量。变换操作由`transformDesc`指针定义。此函数可用于更改数据的存储顺序，或对值进行缩放和偏移。

**参数**:

| 参数                      | 内存       | 输入/输出 | 描述                                                         |
| ------------------------- | ---------- | --------- | ------------------------------------------------------------ |
| `lightHandle`             |            | 输入      | 指向为ascblasLt上下文分配的ascblasLt句柄的指针。参见[ascblasLtHandle_t](https://docs.nvidia.com/cuda/ascblas/#ascblaslthandle-t)。 |
| `transformDesc`           |            | 输入      | 指向包含矩阵变换操作的不透明描述符的指针。参见[ascblasLtMatrixTransformDesc_t](https://docs.nvidia.com/cuda/ascblas/#ascblasltmatrixtransformdesc-t)。 |
| `alpha`，`beta`           | 设备或主机 | 输入      | 乘法中使用的标量的指针。                                     |
| `A`、`B`                  | 设备       | 输入      | 指向与相应描述符`Adesc`和`Bdesc`相关联的GPU内存的指针。      |
| `C`                       | 设备       | 输出      | 与`Cdesc`描述符相关联的GPU内存指针。                         |
| `Adesc`、`Bdesc`和`Cdesc` |            | 输入      | 先前创建的类型为[ascblasLtMatrixLayout_t](https://docs.nvidia.com/cuda/ascblas/#ascblasltmatrixlayout-t)的描述符的句柄。如果相应的指针为NULL且相应的标量为零，则`Adesc`或`Bdesc`可以为NULL。 |
| `stream`                  | 主机       | 输入      | 所有GPU工作将提交到的CUDA流。                                |

#### 

### Matmul算法选择

#### ascblasLtMatmulAlgoGetHeuristic()

```c++
ascblasStatus_t ascblasLtMatmulAlgoGetHeuristic(
      ascblasLtHandle_t lightHandle,
      ascblasLtMatmulDesc_t operationDesc,
      ascblasLtMatrixLayout_t Adesc,
      ascblasLtMatrixLayout_t Bdesc,
      ascblasLtMatrixLayout_t Cdesc,
      ascblasLtMatrixLayout_t Ddesc,
      ascblasLtMatmulPreference_t preference,
      int requestedAlgoCount,
      ascblasLtMatmulHeuristicResult_t heuristicResultsArray[],
      int *returnAlgoCount);
```

根据给定的输入矩阵A、B、C和输出矩阵D，获取矩阵乘法运算[ascblasLtMatmul()](https://docs.nvidia.com/cuda/ascblas/#ascblasltmatmul)函数可能的算法。输出结果按估计计算时间从短到长的顺序存储在`heuristicResultsArray[]`中。

#### ascblasLtMatmulAlgoGetIds()

```c++
ascblasStatus_t ascblasLtMatmulAlgoGetIds(
      ascblasLtHandle_t lightHandle,
      ascblasComputeType_t computeType,
      cudaDataType_t scaleType,
      cudaDataType_t Atype,
      cudaDataType_t Btype,
      cudaDataType_t Ctype,
      cudaDataType_t Dtype,
      int requestedAlgoCount,
      int algoIdsArray[],
      int *returnAlgoCount);
```

对于给定的输入矩阵A、B、C和输出矩阵D的类型，此函数会检索所有有效的矩阵乘法算法的ID，这些算法有可能由[ascblasLtMatmul()](https://docs.nvidia.com/cuda/ascblas/#ascblasltmatmul)函数运行。

#### ascblasLtHeuristicsCacheGetCapacity()

```c++
ascblasStatus_t ascblasLtHeuristicsCacheGetCapacity(size_t* capacity);
```

返回[启发式缓存](https://docs.nvidia.com/cuda/ascblas/#heuristics-cache)容量。

#### ascblasLtHeuristicsCacheSetCapacity()

```c++
ascblasStatus_t ascblasLtHeuristicsCacheSetCapacity(size_t capacity);
```

设置[启发式缓存](https://docs.nvidia.com/cuda/ascblas/#heuristics-cache)的容量。将容量设置为0可禁用启发式缓存。

此函数的优先级高于`ascblasLT_HEURISTICS_CACHE_CAPACITY`环境变量。

#### ascblasLtMatmulAlgoCapGetAttribute()
```c++
ascblasStatus_t ascblasLtMatmulAlgoCapGetAttribute(
      const ascblasLtMatmulAlgo_t *algo,
      ascblasLtMatmulAlgoCapAttributes_t attr,
      void *buf,
      size_t sizeInBytes,
      size_t *sizeWritten);
```

#### ascblasLtMatmulAlgoInit()

```c++
ascblasStatus_t ascblasLtMatmulAlgoInit(
      ascblasLtHandle_t lightHandle,
      ascblasComputeType_t computeType,
      cudaDataType_t scaleType,
      cudaDataType_t Atype,
      cudaDataType_t Btype,
      cudaDataType_t Ctype,
      cudaDataType_t Dtype,
      int algoId,
      ascblasLtMatmulAlgo_t *algo);
```

为指定的矩阵乘法算法以及输入矩阵A、B、C和输出矩阵D初始化用于<c0>ascblasLtMatmul()</c0>的矩阵乘法算法结构。

#### ascblasLtMatmulAlgoCheck()

```c++
ascblasStatus_t ascblasLtMatmulAlgoCheck(
      ascblasLtHandle_t lightHandle,
      ascblasLtMatmulDesc_t operationDesc,
      ascblasLtMatrixLayout_t Adesc,
      ascblasLtMatrixLayout_t Bdesc,
      ascblasLtMatrixLayout_t Cdesc,
      ascblasLtMatrixLayout_t Ddesc,
      const ascblasLtMatmulAlgo_t *algo,
      ascblasLtMatmulHeuristicResult_t *result);
```

针对矩阵乘法运算[ascblasLtMatmul()](https://docs.nvidia.com/cuda/ascblas/#ascblasltmatmul)的矩阵乘法算法描述符进行正确性检查，涉及给定的输入矩阵A、B、C以及输出矩阵D。它会检查该描述符是否在当前设备上受支持，并返回包含所需工作空间和计算出的波数的结果

#### ascblasLtMatmulAlgoConfigGetAttribute()
```c++
ascblasStatus_t ascblasLtMatmulAlgoConfigSetAttribute(
      ascblasLtMatmulAlgo_t *algo,
      ascblasLtMatmulAlgoConfigAttributes_t attr,
      const void *buf,
      size_t sizeInBytes);
```

#### ascblasLtMatmulAlgoConfigSetAttribute()
```c++
ascblasStatus_t ascblasLtMatmulAlgoConfigSetAttribute(
      ascblasLtMatmulAlgo_t *algo,
      ascblasLtMatmulAlgoConfigAttributes_t attr,
      const void *buf,
      size_t sizeInBytes);
```

### 日志

#### ascblasLtLoggerSetCallback()

#### ascblasLtLoggerSetFile()

#### ascblasLtLoggerOpenFile()

#### ascblasLtLoggerSetLevel()

#### ascblasLtLoggerSetMask()

#### ascblasLtLoggerForceDisable()


## Future Work

- Remove ascblasLT_MATMUL_DESC_FILL_MODE
- ascblasLT_ALGO_CONFIG_INNER_SHAPE_ID
- ascblasLT_ALGO_CONFIG_STAGES_ID
- Scale support (cublasLtMatmulMatrixScale_t); 对应[矩阵乘输出的量化/反量化](https://www.hiascend.com/document/detail/zh/canncommercial/83RC1/opdevg/Ascendcopdevg/atlas_ascendc_10_10017.html) 
- batch GEMM support (cublasLtBatchMode_t)
- grouped GEMM support:  `ascblasLtGroupedMatrixLayoutCreate()`，`ascblasLtGroupedMatrixLayoutInit()`
  - **Mechanism:** It addresses the "Mixture of Experts" (MoE) bottleneck where small, varied batch sizes would otherwise require multiple high-overhead kernel launches.
- In epilogue, support backpropagation
- support scaling factor
- support float-point emulation: for accelerate matrix multiplication for higher precision data types; works by first transforming the inputs into multiple lower precision values, then leverages lower precision hardware units to compute partial results, and finally recombines the results back into full precision
- support stream-k (like hipBLASLt)
- support GEMV? [矩阵向量乘](https://www.hiascend.com/document/detail/zh/canncommercial/83RC1/opdevg/Ascendcopdevg/atlas_ascendc_10_10019.html)  


## Question

(关于ascblasLtMatmulTile_t). 昇腾 tensor core支持的matrix大小？

https://leimao.github.io/blog/NVIDIA-Tensor-Core-MMA-Instruction-TN-Layout/

