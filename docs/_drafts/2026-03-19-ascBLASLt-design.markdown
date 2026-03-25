# cuBLAS/cuBLASLt分析 与 ascBLAS/ascBLASLt设计


## cuBLAS (GEMM部分) / cuBLASLt 在PyTorch中的使用
[CUDABlas.cpp](https://github1s.com/pytorch/pytorch/blob/main/aten/src/ATen/cuda/CUDABlas.cpp)

https://github.com/vllm-project/vllm/issues/35467

Here's the breakdown for each mode, specifically for GEMM/matmul operations on GPU:

---

### 1. `mode="default"` (or `mode=None`)

**GEMM strategy: extern call to ATen → cuBLAS. No autotuning.**

When Inductor encounters a matmul, it emits an **extern kernel call** like:
```python
extern_kernels.addmm(bias, input, weight, alpha=1, beta=1, out=buf0)
```

The default setting of flag `max_autotune` is False, which generates `extern_kernels.addmm(...)`. This is a cuBLAS op.

This calls directly into PyTorch's ATen library, which routes to cuBLAS (and cuBLAS internally uses its recommender system to pick a kernel). Inductor does **no** kernel selection, **no** Triton GEMM templates, **no** CUTLASS, **no** benchmarking. It trusts cuBLAS entirely. The only Inductor optimizations applied are operator fusion of non-GEMM ops (pointwise, reductions) into Triton kernels, but the GEMM itself is a black-box cuBLAS call.

**Compile time impact:** Fast — no GEMM autotuning overhead.

---

### 2. `mode="reduce-overhead"`

**GEMM strategy: Same as default (extern cuBLAS), but wrapped in CUDA graphs.**

This mode focuses on reducing Python dispatch and kernel launch overhead, not on finding better GEMM kernels. The matmul path is identical to `default` — it still emits `extern_kernels.addmm(...)` calling cuBLAS. The key difference is that the entire compiled graph (including the cuBLAS calls) is captured into a **CUDA graph**, which eliminates per-call CPU→GPU launch overhead on subsequent runs.

`max_autotune` remains False, so no Triton/CUTLASS GEMM alternatives are considered.

**Compile time impact:** Slightly slower than default (CUDA graph capture), but no GEMM autotuning.

---

## 3. `mode="max-autotune"`

**GEMM strategy: Race multiple backends, benchmark each, pick the winner. CUDA graphs enabled.**

This is where it gets interesting. `max-autotune` is a mode that leverages Triton or template-based matrix multiplications on supported devices. It sets `max_autotune=True`, which triggers the full GEMM autotuning pipeline:

**Step 1 — Candidate generation from multiple backends:**

Backends that are considered are set by `max_autotune_gemm_backends`, defaulting to `"ATEN,TRITON,CPP"`. Each kernel has implementations for the different backends which are added to possible choices.

On GPU, the effective backends are:

- **ATEN** — the cuBLAS extern call (same as default mode). Always included as a baseline.
- **TRITON** — Inductor generates Triton matmul template kernels with a **static hardcoded list** of configs (`BLOCK_M/N/K`, `num_warps`, `num_stages`). This benchmarks a static list of Triton configurations and uses the fastest for each shape.
- **CUTLASS** — if available and the shape qualifies, CUTLASS3xGemmTemplate adds CUTLASS GEMM choices to the candidate pool.
- **NVGEMM** — NVIDIA's universal GEMM backend. The maximum number of NVGEMM configs to profile is 5 by default, to keep compile time reasonable.

**Step 2 — Compile-time benchmarking:**

All candidates are **actually compiled and run** on the GPU with the concrete shapes from tracing. `_inductor/select_algorithm.py` benchmarks each candidate and records wall-clock times.

**Step 3 — Winner selection:**

The fastest candidate wins and is baked into the generated code. This might be cuBLAS, a Triton template, or a CUTLASS kernel — whatever was fastest for that specific (M, N, K) shape.

**Step 4 — Epilogue fusion opportunity:**

A key advantage of Triton/CUTLASS winning over cuBLAS: if the matmul is followed by pointwise ops (bias add, ReLU, GELU), these can be **fused into the GEMM kernel** via Triton templates or CUTLASS epilogues. cuBLAS can't do this (it would require a separate kernel launch). So `max-autotune` doesn't just pick the fastest standalone GEMM — it also evaluates fused GEMM+epilogue variants.

You also need a hefty enough GPU to get the benefit: `is_big_gpu` gates it on the number of SMs, so it works best on a 3090, V100 or above.

**Compile time impact:** Significantly slower — every GEMM shape triggers compilation and benchmarking of potentially dozens of candidates.

---

### 4. `mode="max-autotune-no-cudagraphs"`

**GEMM strategy: Identical to `max-autotune`, but without CUDA graph capture.**

The GEMM autotuning pipeline is exactly the same — same backends, same benchmarking, same winner selection. The only difference is that the final compiled graph is **not** wrapped in a CUDA graph. This is useful when CUDA graphs are incompatible with your workload (e.g., dynamic shapes, operations that allocate memory, or multi-stream usage).

---

### Summary table

| Mode | GEMM kernel source | Autotuning? | Backends raced | CUDA graphs | Epilogue fusion |
|---|---|---|---|---|---|
| **`default`** | cuBLAS (extern call) | No | None — cuBLAS only | No | No |
| **`reduce-overhead`** | cuBLAS (extern call) | No | None — cuBLAS only | **Yes** | No |
| **`max-autotune`** | Best of cuBLAS / Triton / CUTLASS / NVGEMM | **Yes** (compile-time benchmarking) | ATEN, TRITON, CUTLASS, NVGEMM | **Yes** | **Yes** (Triton/CUTLASS templates) |
| **`max-autotune-no-cudagraphs`** | Best of cuBLAS / Triton / CUTLASS / NVGEMM | **Yes** (compile-time benchmarking) | ATEN, TRITON, CUTLASS, NVGEMM | No | **Yes** (Triton/CUTLASS templates) |

The critical takeaway: only `max-autotune` variants do any GEMM kernel selection work. In all other modes, every matmul is a direct cuBLAS call, and NVIDIA's internal cuBLASLt recommender system is the only thing choosing the kernel.




## cuBLAS and cuBLASLt relationship

Great question. Here's a practical breakdown of when each API makes sense:

### cuBLAS (the standard API)

**Use when** you want simplicity and broad BLAS coverage. cuBLAS provides the full BLAS specification (Level 1, 2, and 3 — vector ops, matrix-vector ops, and matrix-matrix ops), with a straightforward API. For general-purpose linear algebra operations, cuBLAS is often the simpler choice.

It's the right pick when:

- You need more than just GEMM (e.g., `axpy`, `dot`, `trsm`, `symm`, etc.)
- You want a quick, low-effort integration — one function call like `cublasSgemm()` with minimal boilerplate
- You're prototyping or your GEMM workload isn't performance-critical enough to justify tuning
- You're on older GPU architectures where the cuBLASLt tuning options are a no-op anyway

### cuBLASLt (the "Lightweight" / advanced GEMM API)

cuBLASLt is a lightweight library dedicated to GEMM operations with a new flexible API that adds flexibility in matrix data layouts, input types, compute types, and also in choosing algorithmic implementations and heuristics through parameter programmability.

**Use cuBLASLt when** you need any of the following:

**1. Performance auto-tuning.** This is the biggest reason. Unlike cuBLAS's black-box heuristics that hide algorithm selection, cuBLASLt exposes all available algorithm variants to developers and allows explicitly enumerating and evaluating them. Via `cublasLtMatmulAlgoGetHeuristic`, you can benchmark multiple kernel candidates for your specific problem size and lock in the fastest one. On Ampere and newer, the `cublasGemmAlgo_t` parameter in the standard cuBLAS API is essentially ignored, so cuBLASLt is the *only* way to do manual kernel selection on modern hardware.

**2. Fused epilogues.** cuBLASLt lets you fuse operations like bias addition, ReLU, GELU, or auxiliary matrix outputs directly into the GEMM kernel — avoiding extra kernel launches and memory round-trips. This is critical for deep learning inference and training.

**3. Mixed-precision and narrow types (FP8, INT8, BF16).** cuBLASLt has broader and earlier support for exotic precision combinations, especially FP8 on Hopper/Ada and increasing mixed-precision options.

**4. Per-call stream and workspace control.** cuBLASLt lets you specify the CUDA stream and workspace memory on a per-function-call basis, rather than tying them to a global handle. This is important for complex multi-stream pipelines.

**5. Flexible matrix layouts.** cuBLASLt supports non-standard data layouts beyond simple column-major/row-major, which matters for certain DL tensor formats.

**6. Plan-and-reuse pattern.** Once a set of options for the intended GEMM operation are identified, they can be reused repeatedly for different inputs, similar to cuFFT plans. This amortizes setup cost when you're running the same shaped GEMM millions of times (e.g., during training).

## The tradeoff

There's one big reason for not using cuBLASLt: it is significantly more complicated to use than cublasGemmEx. You need to create and configure multiple descriptor objects (`cublasLtMatmulDesc_t`, `cublasLtMatrixLayout_t`, `cublasLtMatmulPreference_t`) before you can even issue a single multiply. It's substantially more code.

## The practical recommendation

It is recommended that advanced users of NVIDIA Ampere architecture and newer GPUs migrate from cublasGemmEx to cublasLtMatmul. NVIDIA themselves suggest this because cuBLASLt is where all the new performance work, fused epilogues, and mixed-precision support are landing. The standard `cublas<t>gemm` functions internally route to the same kernel pool, but you lose the ability to tune or fuse.

In short: if GEMM performance matters to your application and you're on Ampere or newer, use cuBLASLt. If you just need quick BLAS operations with minimal code, the standard cuBLAS API is fine.



## cuBLAS/cuBLASLt 二进制文件

```bash
(venv) root ➜ /usr/…/targets/x86_64-linux/lib $ ll libcublas.so.13.2.1.1 libcublasLt.so.13.2.1.1
-rw-r--r-- 1 root root 480M Dec 19 10:40 libcublasLt.so.13.2.1.1
-rw-r--r-- 1 root root  52M Dec 19 10:40 libcublas.so.13.2.1.1
(venv) root ➜ /usr/…/targets/x86_64-linux/lib $ ldd libcublas.so.13.2.1.1
        linux-vdso.so.1 (0x00007b45de290000)
        libcublasLt.so.13 => /usr/local/cuda/lib64/libcublasLt.so.13 (0x00007b45b7200000)
        librt.so.1 => /lib/x86_64-linux-gnu/librt.so.1 (0x00007b45de27b000)
        libpthread.so.0 => /lib/x86_64-linux-gnu/libpthread.so.0 (0x00007b45de276000)
        libdl.so.2 => /lib/x86_64-linux-gnu/libdl.so.2 (0x00007b45de271000)
        libm.so.6 => /lib/x86_64-linux-gnu/libm.so.6 (0x00007b45b7117000)
        libgcc_s.so.1 => /lib/x86_64-linux-gnu/libgcc_s.so.1 (0x00007b45de241000)
        libc.so.6 => /lib/x86_64-linux-gnu/libc.so.6 (0x00007b45b6f05000)
        /lib64/ld-linux-x86-64.so.2 (0x00007b45de292000)
```

### What lives where

**`libcublasLt.so`** (the larger library) contains:

- All the GEMM kernel binaries (this is the bulk of the size)
- The heuristic/recommender system for kernel selection
- The cuBLASLt API surface (matmul descriptors, layout descriptors, algorithm selection, etc.)

**`libcublas.so`** (the smaller library) contains:

- The standard BLAS API surface (all 152 routines across Level 1, 2, and 3)
- BLAS Level 1 kernels (vector-vector: `axpy`, `dot`, `scal`, `nrm2`, etc.)
- BLAS Level 2 kernels (matrix-vector: `gemv`, `trsv`, `symv`, etc.)
- The convenience wrapper logic that translates `cublasSgemm()` / `cublasGemmEx()` calls into the appropriate cuBLASLt descriptor setup and dispatch
- cuBLASXt multi-GPU logic
- Legacy API compatibility layer



## GEMM配置推荐系统

https://docs.nvidia.com/cuda/nvidia-matmul-heuristics/index.html#

```
推荐系统输入输出
Given:
  - Problem shape: (M, N, K, batch_size)
  - Data types: (dtype_A, dtype_B, dtype_C, compute_type)
  - Epilogue: (bias, activation, residual)
  - Layout: (transA, transB, row/col major)
  - Hardware: (chip_model, num_cubes, HBM_bandwidth, ...)

Select:
  - Kernel configuration that minimizes latency
    (tile_M, tile_N, tile_K, pipeline_depth, split_K,
     warp_layout, memory_access_pattern, ...)
```



### 算子来源

1. CUTLASS生成
2. 非CUTLASS生成：手写SASS, 或者任何被证明高性能的算子（NV内部等）



### 方案一（Nvidia）

决策树，查表，经验分析法等

This is essentially a hand-crafted decision tree. The splits are determined by domain expertise (a kernel engineer knows that M=1 GEMMs are fundamentally different from M=4096 GEMMs). The leaves are determined by profiling data. 

**Pros:** Interpretable, fast, captures the major regime transitions well. 

**Cons:** Boundaries are somewhat arbitrary (why split at M=128 and not M=96?), hard to maintain as kernel count grows, doesn't handle multi-dimensional interactions well (the best tile might depend on M×K jointly, not M and K separately).

```python
# 决策树例子
# Simplified reconstruction of what the heuristic probably looks like
def select_algorithm(M, N, K, batch, dtype, epilogue):
    # Classify the problem regime
    if M <= 4:
        regime = "decode"  # memory-bound, single-token
    elif M <= 128:
        regime = "small_batch"  # mixed compute/memory bound
    elif M * N * K < 2**20:
        regime = "small_problem"  # launch overhead matters
    else:
        regime = "compute_bound"  # large GEMM, maximize FLOPS

    # Within each regime, further split
    if regime == "decode":
        if K <= 4096:
            return ALGO_SKINNY_SMALL_K  # specialized vector kernel
        elif K <= 16384:
            return ALGO_SKINNY_MEDIUM_K
        else:
            return ALGO_SKINNY_SPLIT_K  # split-K to get parallelism

    elif regime == "compute_bound":
        # Tile size selection based on shape ratios
        if M >= 4096 and N >= 4096:
            return ALGO_LARGE_TILE_256x128  # maximize compute density
        elif M / N > 4:
            return ALGO_TALL_TILE_256x64  # tall problem, wide tile in M
        elif N / M > 4:
            return ALGO_WIDE_TILE_64x256  # wide problem
        else:
            return ALGO_SQUARE_TILE_128x128  # balanced
    # ... etc
```

### 方案二（AMD）

ML训练

早期：类似穷尽搜索，并收集数据

后期：逐渐使用收集的数据训练推荐系统

This is the frontier, and there's active research here. The idea is to train a model that predicts kernel performance from problem features.

```
┌─────────────────────────────────────────────────┐
│              Offline Training Pipeline            │
│                                                   │
│  ┌───────────┐    ┌──────────────┐               │
│  │ Shape      │    │ Kernel       │               │
│  │ Database   │───▶│ Profiler     │               │
│  │ (10K+      │    │ (exhaustive) │               │
│  │  shapes)   │    └──────┬───────┘               │
│  └───────────┘           │                        │
│                    ┌─────▼────────┐               │
│                    │ Training Data │               │
│                    │ (shape, cfg,  │               │
│                    │  latency)     │               │
│                    └─────┬────────┘               │
│                          │                        │
│              ┌───────────▼───────────┐            │
│              │                       │            │
│         ┌────▼─────┐          ┌─────▼──────┐     │
│         │ Perf     │          │ Top-K      │     │
│         │ Predictor│          │ Classifier │     │
│         │ (MLP)    │          │ (XGBoost)  │     │
│         └──────────┘          └────────────┘     │
│                                                   │
└─────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────┐
│             Online Inference (GetHeuristic)       │
│                                                   │
│  Input: (M,N,K,batch,dtype,epilogue)             │
│                │                                  │
│         ┌──────▼───────┐                          │
│         │  Feature     │                          │
│         │  Engineering │                          │
│         └──────┬───────┘                          │
│                │                                  │
│         ┌──────▼───────┐                          │
│         │  Top-K       │  → returns 5 candidate   │
│         │  Classifier  │    kernel configs        │
│         └──────┬───────┘                          │
│                │                                  │
│         ┌──────▼───────┐                          │
│         │  Perf        │  → ranks candidates by   │
│         │  Predictor   │    predicted latency      │
│         └──────┬───────┘                          │
│                │                                  │
│         Output: ranked list of algorithms         │
└─────────────────────────────────────────────────┘
```









## 推荐缓存

cuBLAS 12.0 引入了启发式缓存机制，用于存储矩阵乘法问题到先前由启发式算法选定的内核之间的映射关系。这有助于减少重复矩阵乘法问题的主机端开销。

由于深度学习工作负载会对相同形状的 GEMM 调用数百万次，cuBLAS 会缓存启发式查找的结果。对于给定的 (M, N, K, 数据类型, 布局) 组合，第一次调用需要承担查找开销；后续调用则直接命中缓存，跳过启发式计算，直接进入内核启动阶段。NVIDIA 还针对高驱逐率的工作负载（即包含大量不同 GEMM 形状的场景）对该缓存进行了优化。



## cuBLAS/cuBLASLt整体流程

```
┌─────────────────────────────────────────┐
│         Kernel Development               │
│  (CUTLASS-like templates, hand-tuned     │
│   kernels for special cases)             │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│      Exhaustive Offline Profiling        │
│  - For each (arch, dtype, epilogue):     │
│    - Enumerate all valid tile configs    │
│    - Profile on ~10K+ shapes             │
│    - Record perf for every (shape,algo)  │
│  - Runs on dedicated GPU farms           │
│  - Takes days/weeks per architecture     │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│       Heuristic Model Training           │
│  - Input: (M, N, K, batch, dtype, ...)  │
│  - Output: ranked algorithm list         │
│  - Likely decision tree / lookup table   │
│  - Validated against held-out shapes     │
│  - "Top-1 accuracy" = % where heuristic │
│    picks algo within 5% of true best     │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│         Regression Testing (CI)          │
│  - Per-commit: fast subset (~100 shapes) │
│  - Nightly: full suite (~10K shapes)     │
│  - Per-arch: test on V100, A100, H100,   │
│    H200, B100, B200, ...                 │
│  - Compare against previous release      │
│  - Both kernel perf AND heuristic quality │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│       Release Qualification              │
│  - End-to-end model benchmarks           │
│    (not just GEMMs — actual PyTorch      │
│     models using cuBLAS underneath)      │
│  - Framework integration tests           │
│  - Customer shape databases              │
└─────────────────────────────────────────┘
```

### 反馈闭环

**1. NVIDIA 拥有海量的客户遥测数据。** 他们知道哪些形状是真正重要的，因为他们能看到实际在 GPU 上运行的工作负载是什么样的（通过 Nsight 等性能分析工具、bug 报告以及与合作伙伴的深度合作）。

**2. 他们拥有专用的 GPU 集群用于 profiling。** 当他们新增一个 kernel 或修改一个 tiling 策略时，他们可以在所有架构上对整个形状空间重新进行性能分析。这是一笔巨大的算力投资。

**3. CUTLASS ↔ cuBLAS 的流水线**意味着研究创新（新的 tiling 策略、新的 warp 级原语）从开源的探索阶段流入闭源的生产库，经过加固和性能分析后正式上线。

**4. 多代积累的经验**——他们从 Kepler 架构就开始做这件事了。他们的 kernel 数据库、启发式训练流水线和回归测试基础设施已经迭代了 10 年以上。



## ascBLAS分阶段推进

### 第一阶段：决策树启发式，手工调优

手工编写决策树启发式算法。重点把**问题域分类**做对——decode（解码）vs. prefill（预填充） vs. 小规模问题。每个域内部，根据工程经验选择 tile 大小和 split-K 策略。

这能让你快速拥有一个可用的产品。不需要完美——只要在大多数情况下选到前3名的 kernel 就够了。

```python
# 第一阶段：简单但有效
def select_algorithm(M, N, K, batch, dtype, epilogue):
    if M <= 4:
        regime = "decode"        # 访存受限，单 token 推理
    elif M <= 128:
        regime = "small_batch"   # 计算和访存混合受限
    elif M * N * K < 2**20:
        regime = "small_problem" # kernel 启动开销占主导
    else:
        regime = "compute_bound" # 大 GEMM，最大化算力利用
    
    # 在每个 regime 内部，进一步根据形状特征细分
    # 边界值通过性能分析数据确定
    ...
```

**关键交付物：**

- 可用的 `GetHeuristic` 接口
- 覆盖核心 LLM 推理形状（decode M=1, prefill M=2048, 常见 FFN 宽度）
- 用已有的 profiling 数据验证决策树的选择质量

------

### 第二阶段：建设 Profiling 基础设施和形状数据库

这是**最关键的基础设施投资**。在这一阶段：

1. **搭建自动化 profiling 流水线**——专用 NPU 集群，时钟锁定，热稳定
2. **开始收集穷举式 profiling 数据**——对约 10,000 个锚点形状 × 所有有效 kernel 配置进行性能测量
3. **训练 XGBoost 分类器**，作为手工决策树的替代品（可直接插入现有接口）
4. **建立启发式质量度量体系**：
   - **Top-1 准确率**：启发式返回的第一选择是否在真实最优的 5% 以内？
   - **Top-3 准确率**：真实最优是否出现在返回的前3个候选中？
   - **达到 roofline 的百分比**：在关键形状上达到理论峰值的多少？

```python
# 第二阶段核心代码：特征工程
def compute_features(M, N, K, batch, chip_info):
    total_flops = 2 * M * N * K * batch
    total_bytes = (M * K + K * N) * dtype_size * batch
    
    return {
        "log_M": log2(M),
        "log_N": log2(N),
        "log_K": log2(K),
        # 算术强度——最重要的单一特征
        # 决定了计算受限还是访存受限
        "arithmetic_intensity": total_flops / total_bytes,
        # 形状比例——捕捉问题的"几何形态"
        "M_over_N": log2(M / N),
        # 对齐特征——对 tiling 效率至关重要
        "M_mod_256": M % 256,
        # wave 量化效率——捕捉性能悬崖
        "wave_efficiency_tile128": compute_wave_efficiency(M, N, 128, 128, chip_info),
    }
```

**关键交付物：**

- 完整的 profiling 数据库（PostgreSQL 或类似存储）
- XGBoost 模型，pairwise ranking 准确率 > 85%
- 每夜自动运行的 profiling 回归检测

------

### 第三阶段：MLP 性能预测器 + 主动学习

在这一阶段，你从"查表"升级为"理解"——用神经网络来**预测** kernel 在任意形状上的性能：

1. **MLP 性能预测器**：输入 (shape_features, kernel_config_features)，输出预测延迟。用于对 Top-K 候选进行排序
2. **主动学习（Active Learning）**：智能选择下一步要 profile 哪些 (shape, config) 对，用模型不确定性来引导采样。可以将 profiling 成本降低 3-5 倍
3. **迁移学习基础设施**：为芯片代际切换做准备——在 910B 上训练的基础模型，只需 10% 的 profiling 数据就能适配 920

```python
# 预测 log-latency（延迟呈对数正态分布）
class PerfPredictor(nn.Module):
    def __init__(self, shape_feat_dim, config_feat_dim):
        super().__init__()
        self.shape_encoder = nn.Sequential(
            nn.Linear(shape_feat_dim, 128), nn.ReLU(), nn.Linear(128, 64))
        self.config_encoder = nn.Sequential(
            nn.Linear(config_feat_dim, 64), nn.ReLU(), nn.Linear(64, 64))
        self.predictor = nn.Sequential(
            nn.Linear(128, 128), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))
    
    def forward(self, shape_feats, config_feats):
        s = self.shape_encoder(shape_feats)
        c = self.config_encoder(config_feats)
        return self.predictor(torch.cat([s, c], dim=-1))
```

**关键洞察：\**性能预测模型\**不需要准确预测绝对延迟**——它只需要**正确排序**。如果模型说配置 A 比配置 B 快，这个判断正确就够了。绝对延迟值可以偏差 2 倍都没关系，只要排序正确。

**关键交付物：**

- MLP 模型，ranking 准确率 > 90%
- 主动学习流水线，新 kernel 上线时自动 profile
- 跨芯片迁移学习的概念验证
