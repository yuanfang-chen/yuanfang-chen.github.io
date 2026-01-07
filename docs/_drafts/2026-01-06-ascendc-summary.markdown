---
layout: post
title:  "AscendC总结"
# date:   2025-12-08 11:18:26 -0800
categories: ascend
typora-root-url: ..
mathjax: true
---



## Ascend C API全览

Ascend C提供一组类库API，开发者使用标准C++语法和类库API进行编程。Ascend C编程类库API示意图如下所示，分为：

- Kernel API：用于实现算子核函数的API接口。包括：
  - **基本数据结构：**kernel API中使用到的基本数据结构，比如GlobalTensor和LocalTensor。
  - **基础API：**实现对硬件能力的抽象，开放芯片的能力，保证完备性和兼容性。标注为ISASI（Instruction Set Architecture Special Interface，硬件体系结构相关的接口）类别的API，不能保证跨硬件版本兼容。
  - **高阶API：**实现一些常用的计算算法，用于提高编程开发效率，通常会调用多种基础API实现。高阶API包括数学库、Matmul、Softmax等API。高阶API可以保证兼容性。
- **Host API**：
  - 高阶API配套的Tiling API：kernel侧高阶API配套的Tiling API，方便开发者获取kernel计算时所需的Tiling参数。
  - Ascend C算子原型注册与管理API：用于Ascend C算子原型定义和注册的API。
  - Tiling数据结构注册API：用于Ascend C算子TilingData数据结构定义和注册的API。
  - 平台信息获取API：在实现Host侧的Tiling函数时，可能需要获取一些硬件平台的信息，来支撑Tiling的计算，比如获取硬件平台的核数等信息。平台信息获取API提供获取这些平台信息的功能。
- **算子调测API**：用于算子调测的API，包括孪生调试，性能调测等。

进行Ascend C算子Host侧编程时，需要使用基础数据结构和API，请参考[基础数据结构与接口](https://www.hiascend.com/document/detail/zh/canncommercial/83RC1/API/basicdataapi/atlasopapi_07_00002.html)；完成算子开发后，需要使用Runtime API完成算子的调用，请参考“[acl API（C&C++）](https://www.hiascend.com/document/detail/zh/canncommercial/83RC1/API/appdevgapi/aclcppdevg_03_0004.html)”。

![img](/assets/images/zh-cn_image_0000002446718504.png)