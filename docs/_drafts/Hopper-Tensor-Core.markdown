---
layout: post
title:  "Hopper Tensor Core"
#date:   2025-11-27 23:44:54 +0800
categories: CUDA
typora-root-url: ..
---

## Warpgroup MMA (WGMMA)

- 异步操作
- 连续128个线程完成，首个warp的rank必须是4的整数倍
- PTX `wgmma.mma_async`
- `B`必须在SMEM；`A`可以在SMEM或者RMEM；`C`必须在RMEM

## mainloop

 
