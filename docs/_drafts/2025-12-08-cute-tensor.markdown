---
layout: post
title:  "cute Tensor"
# date:   2025-12-08 11:18:26 -0800
categories: CUDA
typora-root-url: ..
---

# **Partitioning is tiling and/or composition followed by slicing.**



A `Tensor` is represented by two template parameters: `Engine` and `Layout`. 

The `Engine` concept is a wrapper for an iterator or array of data. It uses a stripped-down interface of `std::array` to present the iterator.

### **Tagged Iterators**

用来表示iterator指向的内存空间，比如，是指向SMEM还是GMEM，等





- `Tensor` is defined as an `Engine` and a `Layout`.
  - `Engine` is an iterator that can be offset and dereferenced.
  - `Layout` defines the logical domain of the tensor and maps coordinates to offsets.
- Tile a `Tensor` using the same methods for tiling `Layout`s.
- Slice a `Tensor` to retrieve subtensors.
- Partitioning is tiling and/or composition followed by slicing.

