# ascBLASLt架构设计



## 为什么要做ascBLASLt

 - 作为优化算子的数据库
 - 昇腾平台上GEMM性能优化的基线
 - 固化NPU各个代际的GEMM优化算子



## 什么是ascBLASLt? 

 ascBLASLt = 算子数据库 + 启发式算法/cost model

```
┌─────────────────────────────────────────────────────────┐
│           Kernel 数据库                                  │
│  libcublasLt.so 中约 500 个预编译 kernel                  │
│  每个 kernel 都能处理任意形状                               │
│  问题从来不是"能不能跑"，而是"跑多快"                         │
└─────────────────────────┬───────────────────────────────┘
                          │
                          │  GetHeuristic 从中选择
                          │
┌─────────────────────────▼───────────────────────────────┐
│           启发式算法 / cost model                          │
│  是一个函数，不是查找表                                     │
│  接受任意 (M,N,K) → 为每个 kernel 打分 → 排序               │
│  基于 profiling 数据校准，但能泛化到未见过的形状              │
│  对从未出现过的形状同样有效                                  │
└─────────────────────────┬───────────────────────────────┘
                          │
                          │  如果启发式不够好
                          │
┌─────────────────────────▼───────────────────────────────┐
│           用户侧自动调优  （类比cublasLtMatmulAlgoGetIds）   │
│  框架自行枚举算法 + 实测基准性能                              │
│  在模型加载时对每个形状缓存最优选择                            │
│  这是启发式算法出错时的"逃生通道"                             │
└─────────────────────────────────────────────────────────┘
```



## ascBLASLt运行流程

开发者告诉ascBLASLt要做的GEMM的配置（M,N,K,batch size,dtype,compute type,是否做融合等)；同时，开发者指定算子的部分参数，ascBLASLt拿到这些信息后输出GEMM要调用的算子的算法和参数，并以预估性能对这些算子进行排序，最终把排序后的算子列表返回给开发者，开发者可以直接选取第一个算子并调用GEMM（ascBLASLt自动调优），也可以自行选择算子列表中的任意算子并调用GEMM（开发者手动调优）。



- 持续收集高性能算子
- 持续收集高频，高价值shape
- 持续profiling新增的算子和shape



## 算子的算法和参数

```text
Level 1 （算法）: Which kernel FAMILY / algorithm?  (cuBLASLt algo IDs)
  "Use the 128x128 tile Tensor Core HMMA kernel"  vs
  "Use the split-K 64x64 tile kernel"  vs
  "Use the specialized skinny-M kernel"

Level 2	（算法参数）: Within that kernel, what CONFIGURATION?
  Given "128x128 tile Tensor Core kernel", choose:
  - Exactly which warp arrangement? (2x2 vs 4x1 vs 1x4)
  - Pipeline depth? (2-stage vs 4-stage vs 6-stage)
  - L2 cache swizzle pattern?
  - Software prefetch distance?
  - Split-K factor? (1, 2, 4, 8?)
  - Which TMA mode? (bulk vs tiled)
```

**澄清：**算子一般可以处理所有GEMM shape，不是只能处理特定GEMM shape。但是对同一个GEMM shape，不同的算子性能有差异。

### 编译期参数 （使用CUTLASS模版库生成二进制算子）

```c++
// CUTLASS example — each unique combination = different .cubin
using GemmKernel = cutlass::gemm::kernel::GemmUniversal
    cutlass::half_t,                          // ElementA
    cutlass::layout::RowMajor,                // LayoutA
    cutlass::half_t,                          // ElementB
    cutlass::layout::ColumnMajor,             // LayoutB
    cutlass::half_t,                          // ElementC
    cutlass::layout::RowMajor,                // LayoutC
    float,                                     // ElementAccumulator
    cutlass::arch::OpClassTensorOp,           // Use Tensor Cores
    cutlass::arch::Sm80,                      // Target Ampere
    cutlass::gemm::GemmShape<128, 128, 32>,   // ThreadBlock tile (M, N, K)
    cutlass::gemm::GemmShape<64, 64, 32>,     // Warp tile
    cutlass::gemm::GemmShape<16, 8, 16>,      // MMA instruction shape
    cutlass::epilogue::thread::LinearCombination
        cutlass::half_t, 8, float, float>,    // Epilogue
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,  // Swizzle
    4,                                         // Pipeline stages
    8,                                         // Alignment A
    8,                                         // Alignment B
>;

// ThreadBlock tile shape:  {64×64, 64×128, 128×64, 128×128, 128×256, 256×128, 256×256}
//                           × K-tile {16, 32, 64}
// Warp tile shape:         derived from threadblock tile ÷ warp arrangement
// MMA instruction:         {16×8×8, 16×8×16, 16×8×32} — hardware-dependent
// Pipeline stages:         {2, 3, 4, 5, 6} — deeper = more latency hiding
// Alignment:               {4, 8, 16} — elements per vectorized load
// Epilogue type:           {none, bias, bias+relu, bias+gelu, ...}
// Swizzle pattern:         {identity, swizzle_1, swizzle_2, swizzle_4, swizzle_8}
```



### 算子下发期参数 (chosen per GEMM call)

```
// cuBLASLt exposes these through cublasLtMatmulAlgo_t attributes
cublasLtMatmulAlgoConfigSetAttribute(
    &algo,
    CUBLASLT_ALGO_CONFIG_TILE_ID,       // which tile size variant
    &tileId, sizeof(tileId));

cublasLtMatmulAlgoConfigSetAttribute(
    &algo,
    CUBLASLT_ALGO_CONFIG_SPLITK_NUM,    // split-K factor
    &splitKNum, sizeof(splitKNum));

cublasLtMatmulAlgoConfigSetAttribute(
    &algo,
    CUBLASLT_ALGO_CONFIG_STAGES_ID,     // pipeline depth
    &stagesId, sizeof(stagesId));

cublasLtMatmulAlgoConfigSetAttribute(
    &algo,
    CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING, // CTA scheduling order
    &swizzle, sizeof(swizzle));

cublasLtMatmulAlgoConfigSetAttribute(
    &algo,
    CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME, // for split-K: how to reduce
    &reductionScheme, sizeof(reductionScheme));
```

## 算子二进制库 (kernel binary database)

- Contains ~500 pre-compiled kernel binaries
- 包含深度调优算子（非CUTLASS）和模版库生成算子 （CUTLASS）
- Each kernel binary can run ANY shape (with some constraints) 
- A kernel compiled with tile_128x128 can handle M=1 or M=100000 
- The kernel code has runtime logic for partial tiles, edge cases, etc.

极端情况下，GEMM shape无法被算子数据库中的算子覆盖，可以（1）在算子数据库中放一个通用算子（2）JIT产生算子

长期策略：重点调优模版库性能，从而覆盖80%的场景，剩余场景可以手动优化

## 实现GetHeuristic

```
                    GetHeuristic(M=1234, N=5678, K=2048, FP16, BIAS_GELU)
                                        │
                                        ▼
                    ┌───────────────────────────────────┐
                    │  1. FILTER: which kernels are      │
                    │     structurally valid?             │
                    │                                     │
                    │  80 kernels → alignment check       │
                    │            → pipeline depth check   │
                    │            → epilogue compatibility  │
                    │            → 40 candidates remain    │
                    └──────────────┬────────────────────┘
                                   │
                                   ▼
                    ┌───────────────────────────────────┐
                    │  2. For each candidate, compute     │
                    │     SPLIT-K options                 │
                    │                                     │
                    │  Each kernel × {1,2,4,8,16} splitK  │
                    │  Filter invalid combos              │
                    │  → ~100 (kernel, splitK) pairs      │
                    └──────────────┬────────────────────┘
                                   │
                                   ▼
                    ┌───────────────────────────────────┐
                    │  3. SCORE each pair with cost model │
                    │                                     │
                    │  For each (kernel, splitK):          │
                    │    → compute wave efficiency         │
                    │    → compute tile efficiency         │
                    │    → estimate compute time           │
                    │    → estimate memory time            │
                    │    → account for pipeline overlap    │
                    │    → add split-K overhead            │
                    │    → total score                     │
                    │                                     │
                    │  Pure CPU arithmetic, ~microseconds  │
                    └──────────────┬────────────────────┘
                                   │
                                   ▼
                    ┌───────────────────────────────────┐
                    │  4. RANK by estimated performance   │
                    │                                     │
                    │  Sort all candidates by score       │
                    │  Return top-K to the user            │
                    │                                     │
                    │  User gets:                          │
                    │   #1: tile_128x128, stages=4, sk=1  │
                    │   #2: tile_256x128, stages=3, sk=1  │
                    │   #3: tile_64x64,   stages=4, sk=2  │
                    └───────────────────────────────────┘
```



## Cost Model算法

```python
def estimate_latency(M, N, K, batch, kernel_config, chip_info):
    """
    Analytical cost model that estimates kernel latency from shape
    and config WITHOUT running the kernel.

    This is what GetHeuristic evaluates for each candidate.
    """

    tile_M = kernel_config.tile_M     # e.g., 128
    tile_N = kernel_config.tile_N     # e.g., 128
    tile_K = kernel_config.tile_K     # e.g., 32
    stages = kernel_config.stages     # e.g., 4
    split_k = kernel_config.split_k   # e.g., 1

    # ─── Grid geometry ───────────────────────────────────────
    tiles_M = ceil_div(M, tile_M)
    tiles_N = ceil_div(N, tile_N)
    tiles_K = ceil_div(K, tile_K * split_k)  # K iterations per split-K slice
    total_tiles = tiles_M * tiles_N * batch * split_k

    # ─── Wave analysis (critical) ────────────────────────────
    # How many "waves" of tiles across the chip's compute units?
    num_SMs = chip_info.num_SMs  # or num_cube_units for Ascend
    num_waves = ceil_div(total_tiles, num_SMs)

    # Tiles actually utilized in the last wave
    tiles_last_wave = total_tiles % num_SMs
    if tiles_last_wave == 0:
        tiles_last_wave = num_SMs

    # Wave efficiency: what fraction of SMs are busy in the last wave?
    wave_efficiency = tiles_last_wave / num_SMs
    # Overall efficiency considering all waves
    overall_wave_efficiency = total_tiles / (num_waves * num_SMs)

    # ─── Compute cost ────────────────────────────────────────
    # FLOPs per tile
    flops_per_tile = 2 * tile_M * tile_N * tile_K * tiles_K  # K-loop iterations

    # But partial tiles (at edges) do less useful work:
    # Last tile in M does min(M % tile_M, tile_M) useful rows
    partial_M = M % tile_M if M % tile_M != 0 else tile_M
    partial_N = N % tile_N if N % tile_N != 0 else tile_N

    # Fraction of compute that's "wasted" on padding in partial tiles
    m_efficiency = (((tiles_M - 1) * tile_M + partial_M) / (tiles_M * tile_M))
    n_efficiency = (((tiles_N - 1) * tile_N + partial_N) / (tiles_N * tile_N))
    tile_efficiency = m_efficiency * n_efficiency

    # Time spent on compute (assuming full utilization of math units)
    # peak_flops_per_sm is the Tensor Core throughput per SM per second
    compute_time = (flops_per_tile * tiles_K) / chip_info.peak_flops_per_sm

    # ─── Memory cost ─────────────────────────────────────────
    # Each tile loads: tile_M × tile_K of A + tile_K × tile_N of B
    # Over tiles_K iterations of the K-loop
    bytes_A_per_tile = tile_M * tile_K * dtype_size * tiles_K
    bytes_B_per_tile = tile_K * tile_N * dtype_size * tiles_K
    bytes_C_per_tile = tile_M * tile_N * dtype_size  # write output once

    total_bytes_per_tile = bytes_A_per_tile + bytes_B_per_tile + bytes_C_per_tile

    # L2 cache reuse: if the working set fits in L2, effective bandwidth is higher
    working_set = tile_M * tile_K * dtype_size + tile_K * tile_N * dtype_size
    if working_set * stages < chip_info.l2_per_sm:
        effective_bandwidth = chip_info.l2_bandwidth
    else:
        effective_bandwidth = chip_info.hbm_bandwidth

    memory_time = total_bytes_per_tile / effective_bandwidth

    # ─── Pipeline overlap ────────────────────────────────────
    # With N pipeline stages, compute and memory access overlap
    # The dominant cost wins (max of compute and memory per iteration)
    per_iteration_time = max(
        compute_time / tiles_K,      # compute per K-iteration
        memory_time / tiles_K,       # memory per K-iteration
    )

    # First few iterations fill the pipeline, last few drain it
    pipeline_fill_drain = stages * per_iteration_time
    steady_state = (tiles_K - stages) * per_iteration_time
    time_per_tile = pipeline_fill_drain + steady_state

    # ─── Swizzle / L2 locality bonus ────────────────────────
    # CTA swizzling reorders tile scheduling to improve L2 hit rate
    # when multiple tiles share rows of A or columns of B
    if kernel_config.swizzle > 1:
        # Empirical factor: good swizzle can improve memory time by 10-30%
        # The benefit depends on whether tiles share data in L2
        l2_reuse_factor = estimate_l2_reuse(tiles_M, tiles_N, kernel_config.swizzle)
        time_per_tile *= (1.0 - 0.2 * l2_reuse_factor)  # rough adjustment

    # ─── Split-K overhead ────────────────────────────────────
    split_k_overhead = 0
    if split_k > 1:
        # Split-K adds a reduction kernel after the main GEMM
        reduction_bytes = M * N * sizeof(float) * split_k  # accumulator size
        split_k_overhead = reduction_bytes / chip_info.hbm_bandwidth

    # ─── Total estimate ──────────────────────────────────────
    total_time = (
        time_per_tile * num_waves     # all waves of tiles
        / tile_efficiency             # penalty for partial tiles
        / overall_wave_efficiency     # penalty for last-wave underutilization
        + split_k_overhead            # split-K reduction cost
    )

    return total_time
```

算法的目标是算出近似值，只要它在*方向上*是正确的，不用在意是否是最终的算子计算延迟。这样做的目的是为了更快的cost model运行速度 — 只有纯粹的算术运算，没有内核启动，启发式方法可以在微秒级别评估很多GEMM shape。 



## 如何更新Cost Model

### 调优数据库 (profiling database) & Shape数据库

- Contains actual measured latency for ~10K-50K shapes × all kernel configs
- Used ONLY during heuristic model training
- NOT shipped with the library
- Users never interact with this



### 更新cost model的算法

```python
def calibrate_cost_model(chip_id):
    """
    Run on a set of anchor shapes, compare model predictions
    to actual measured latency, fit the fudge factors.
    """
    anchor_shapes = load_anchor_shapes()  # ~1000 shapes
    all_configs = load_kernel_configs()

    # Collect ground truth
    ground_truth = []
    for shape in anchor_shapes:
        for config in all_configs:
            actual_time = profile_on_hardware(shape, config, chip_id)
            predicted_time = estimate_latency(shape, config, chip_info[chip_id])
            ground_truth.append({
                "shape": shape,
                "config": config,
                "actual": actual_time,
                "predicted": predicted_time,
            })

    # Fit correction factors to minimize prediction error
    # The goal isn't accurate absolute prediction — it's correct RANKING
    # We need: if actual(config_A) < actual(config_B),
    #          then predicted(config_A) < predicted(config_B)

    # Ranking accuracy metric:
    correct_rankings = 0
    total_pairs = 0
    for shape in anchor_shapes:
        shape_results = [r for r in ground_truth if r["shape"] == shape]
        for a, b in itertools.combinations(shape_results, 2):
            total_pairs += 1
            actual_order = a["actual"] < b["actual"]
            predicted_order = a["predicted"] < b["predicted"]
            if actual_order == predicted_order:
                correct_rankings += 1

    ranking_accuracy = correct_rankings / total_pairs
    # Target: >90% pairwise ranking accuracy
```

洞察：cost model不需要准确预测绝对延迟。它只需要正确地对配置进行排序。



## TODO

算子优化Agent，持续研究和输出最优化的GEMM算子

[刚刚，英伟达革了自己的命：智能体自主进化7天，干掉所有算子工程师、GPU专家](https://mp.weixin.qq.com/s/JGqF9Z1up9nRq_owePisbQ?scene=1)





