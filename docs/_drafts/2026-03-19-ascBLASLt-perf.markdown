This is a critical piece of infrastructure to get right early — performance regressions in a BLAS library are silent killers that propagate to every workload on top. Let me lay out a comprehensive design.

## 1. Benchmark Shape Matrix

The most important decision is **which problem shapes to benchmark**. You need coverage across the shapes that actually matter in production workloads.

**Shape categories:**

- **LLM prefill** — large square-ish GEMMs: (M=2048, N=4096, K=4096), (M=4096, N=11008, K=4096) — these are the FFN shapes from LLaMA-style models
- **LLM decode** — tall-skinny: (M=1, N=4096, K=4096), (M=1, N=11008, K=4096) — single-token autoregressive, extremely memory-bound
- **Batched decode** — (M=32..256, N=4096, K=4096) — continuous batching scenarios
- **Attention projection** — (M=seq_len, N=head_dim, K=head_dim) with batch = num_heads
- **Embedding/output layers** — (M=batch, N=vocab_size, K=hidden) — vocab is often 32k-150k, very wide
- **Classic HPC shapes** — powers of 2, nice alignment: 512×512, 1024×1024, 2048×2048, 4096×4096
- **Pathological shapes** — primes, non-aligned: (M=17, N=4093, K=511) — these catch tiling bugs and padding inefficiencies
- **Batch GEMM** — same shapes above with batch counts of 1, 12, 32, 64, 96 (typical head counts)

**Data type matrix:**

| Compute type | Input A | Input B | Output | Accumulator | Notes |
|---|---|---|---|---|---|
| FP16×FP16 | FP16 | FP16 | FP16 | FP32 | Baseline |
| BF16×BF16 | BF16 | BF16 | BF16 | FP32 | Training-common |
| FP16 mixed | FP16 | FP16 | FP32 | FP32 | High-precision |
| INT8×INT8 | INT8 | INT8 | INT32 | INT32 | W8A8 quantized |
| INT8×INT4 | INT8 | INT4 | INT16 | INT32 | W4A8 (your OKR) |
| FP32×FP32 | FP32 | FP32 | FP32 | FP32 | Reference/fallback |


**Epilogue combinations** (test each with representative shapes):

- No epilogue (raw GEMM)
- Bias add
- Bias + ReLU
- Bias + GELU
- Bias + SiLU (important for LLaMA-style)
- Bias + residual add (important for Transformer blocks)

## 2. Benchmark Harness Design

```
benchmark_suite/
├── shapes/
│   ├── llm_shapes.json          # extracted from real model configs
│   ├── hpc_shapes.json          # classical BLAS benchmarks
│   ├── pathological_shapes.json # stress tests
│   └── custom_shapes.json       # user-defined
├── harness/
│   ├── benchmark_runner.py      # orchestrates runs
│   ├── warmup.py                # NPU warmup protocol
│   ├── timer.py                 # high-precision timing
│   ├── memory_tracker.py        # workspace/HBM tracking
│   └── result_collector.py      # structured output
├── analysis/
│   ├── regression_detector.py   # statistical comparison
│   ├── roofline.py              # roofline model analysis
│   └── report_generator.py      # HTML/dashboard output
├── baselines/
│   ├── v0.1.0/                  # stored baseline results per version
│   └── golden/                  # cuBLAS reference numbers (where available)
└── ci/
    ├── nightly.yaml             # full suite
    ├── precommit.yaml           # fast subset
    └── release.yaml             # comprehensive + comparison
```

**Key harness principles:**

```python
class GEMMBenchmark:
    def __init__(self, shape, dtype_config, epilogue, num_warmup=50, num_iters=200):
        self.shape = shape  # (M, N, K, batch)
        self.dtype_config = dtype_config
        self.epilogue = epilogue
        self.num_warmup = num_warmup
        self.num_iters = num_iters

    def run(self):
        # 1. Allocate inputs (random, but SEEDED for reproducibility)
        A, B, bias = self.allocate_inputs()
        C = self.allocate_output()
        workspace = self.allocate_workspace()

        # 2. Create descriptors (your cuBLASLt-equivalent API)
        desc = create_matmul_desc(self.dtype_config, self.epilogue)
        layout_a = create_layout(A, ...)
        layout_b = create_layout(B, ...)

        # 3. Algorithm selection — benchmark ALL returned algorithms
        algos = get_heuristic(desc, layout_a, layout_b, preference)

        results = []
        for algo in algos:
            # Warmup: critical for NPU — first runs include
            # compilation, memory allocation, cache warming
            for _ in range(self.num_warmup):
                matmul(desc, A, B, C, algo, workspace, stream)
            sync_stream(stream)

            # Timed runs
            times = []
            for _ in range(self.num_iters):
                start = npu_event()
                matmul(desc, A, B, C, algo, workspace, stream)
                end = npu_event()
                sync_stream(stream)
                times.append(event_elapsed(start, end))

            results.append({
                "algo_id": algo.id,
                "times_us": times,
                "median_us": median(times),
                "p95_us": percentile(times, 95),
                "p99_us": percentile(times, 99),
                "tflops": self.compute_tflops(median(times)),
                "hw_utilization": self.compute_utilization(median(times)),
                "workspace_bytes": workspace.size,
            })

        return BenchmarkResult(
            shape=self.shape,
            dtype=self.dtype_config,
            epilogue=self.epilogue,
            algo_results=results,
            best_algo=min(results, key=lambda r: r["median_us"]),
            env=collect_env_info(),  # chip model, driver version, CANN version
        )
```

## 3. Metrics to Track

For each (shape, dtype, epilogue, algorithm) tuple:

| Metric                       | Why                                 | Regression threshold |
| ---------------------------- | ----------------------------------- | -------------------- |
| **Median latency (µs)**      | Primary performance indicator       | >5% regression       |
| **P95 latency**              | Tail latency matters for serving    | >10% regression      |
| **P99 latency**              | Outlier detection                   | >15% regression      |
| **TFLOPS achieved**          | Hardware efficiency                 | >5% drop             |
| **% of roofline**            | Normalized against theoretical peak | >3% drop             |
| **Workspace size**           | Memory regression                   | >20% increase        |
| **Coefficient of variation** | Benchmark stability                 | >5% = noisy, re-run  |

**TFLOPS calculation:**

```
TFLOPS = (2 * M * N * K * batch) / (median_time_seconds * 1e12)
```

**Roofline utilization:**

```
theoretical_peak = chip_cube_tflops[dtype]  # e.g., Ascend 910B FP16 peak
utilization = achieved_tflops / theoretical_peak * 100
```

## 4. Regression Detection — Statistical Rigor

Don't use naive "is it X% slower" — NPU timings are noisy. Use proper statistical methods:

```python
class RegressionDetector:
    def compare(self, baseline_times, current_times):
        # 1. Remove outliers (IQR method)
        baseline_clean = remove_outliers(baseline_times)
        current_clean = remove_outliers(current_times)

        # 2. Check distribution normality
        _, p_normal = shapiro(current_clean)

        # 3. Two-sample test
        if p_normal > 0.05:
            # Mann-Whitney U (non-parametric, safer for skewed latency data)
            stat, p_value = mannwhitneyu(baseline_clean, current_clean,
                                          alternative='less')
        else:
            stat, p_value = mannwhitneyu(baseline_clean, current_clean,
                                          alternative='less')

        # 4. Effect size (Cohen's d)
        effect = cohens_d(baseline_clean, current_clean)

        # 5. Practical significance check
        median_change_pct = (
            (median(current_clean) - median(baseline_clean))
            / median(baseline_clean) * 100
        )

        return RegressionResult(
            is_regression=(p_value < 0.01 and median_change_pct > 5.0),
            is_improvement=(p_value < 0.01 and median_change_pct < -5.0),
            p_value=p_value,
            effect_size=effect,
            median_change_pct=median_change_pct,
            confidence="high" if len(current_clean) >= 100 else "medium",
        )
```

**Key insight:** require *both* statistical significance (p < 0.01) *and* practical significance (>5% change). Small statistically-significant changes are noise; large changes with high p-values need more samples.

## 5. CI/CD Integration — Three Tiers

### Tier 1: Pre-commit (fast, blocking)

- **~20 representative shapes** covering each category
- FP16 and INT8 only
- No epilogue (raw GEMM)
- **Runtime:** ~10 minutes
- **Gate:** blocks merge if any >10% regression
- Runs on a **dedicated NPU** to minimize interference

### Tier 2: Nightly (comprehensive)

- **Full shape matrix** (~200-500 shapes)
- All dtype combinations
- All epilogues
- Algorithm heuristic quality check (is the top-1 returned algo actually the fastest?)
- **Runtime:** ~2-4 hours
- **Gate:** sends alerts, doesn't block — creates tracking issues automatically
- Also runs **correctness validation** against reference CPU implementation

### Tier 3: Release qualification

- Everything in nightly
- **Cross-chip testing** (910A, 910B, 910C if available)
- **Comparison against previous release** baseline
- **Comparison against competitor** (cuBLAS on equivalent NVIDIA hardware, where possible)
- **Long-running stability** — run the same shape 10,000 times to check for thermal throttling, memory leaks
- **Multi-process contention** — run benchmarks while other NPU workloads are active
- Generates the release performance report

## 6. Infrastructure Requirements

```
┌──────────────────────────────────────────────┐
│              CI Orchestrator                  │
│         (Jenkins / GitLab CI / custom)        │
└──────────┬───────────────┬───────────────────┘
           │               │
    ┌──────▼──────┐  ┌─────▼──────┐
    │  NPU Pool   │  │  Baseline  │
    │  (dedicated, │  │  Storage   │
    │  no sharing) │  │  (S3/NAS)  │
    └──────┬──────┘  └─────┬──────┘
           │               │
    ┌──────▼───────────────▼──────┐
    │      Analysis Pipeline      │
    │  ┌─────────────────────┐    │
    │  │ Regression Detector  │    │
    │  │ Roofline Analyzer    │    │
    │  │ Report Generator     │    │
    │  └─────────────────────┘    │
    └──────────────┬──────────────┘
                   │
    ┌──────────────▼──────────────┐
    │     Dashboard / Alerting     │
    │  (Grafana + time-series DB)  │
    └─────────────────────────────┘
```

**Critical infrastructure rules:**

1. **Dedicated NPUs** — never share benchmark machines with other workloads. Even background processes cause variance.
2. **Fixed clock frequencies** — disable DVFS/power management during benchmarks. You want to measure algorithm quality, not the power governor's mood.
3. **Pin CANN driver + firmware versions** in baseline metadata. A driver update can silently change numbers.
4. **Store raw timing arrays**, not just summaries. You'll want to re-analyze later.
5. **Environment fingerprint** every run: chip model, driver version, CANN version, firmware version, HBM temperature, ambient temp if available.

## 7. Dashboard & Reporting

Track these over time in a Grafana-style dashboard:

- **Performance timeline** — TFLOPS for key shapes over commits/dates
- **Roofline plot** — arithmetic intensity vs. achieved bandwidth, per shape category
- **Heuristic quality** — % of shapes where the top-1 heuristic-returned algorithm is actually within 5% of the true best
- **Dtype parity** — performance ratio between your library and cuBLAS for equivalent shapes (if you have access to NVIDIA hardware)
- **Regression tracker** — open regressions, age, severity

## 8. Shape Extraction from Real Models

Don't just invent shapes — extract them from real models:

```python
# Instrument your matmul wrapper to log shapes during model execution
def extract_gemm_shapes(model_name):
    """Run model inference and collect all GEMM shapes invoked."""
    shapes = []
    original_matmul = cann_blas.matmul

    def logging_matmul(desc, A, B, C, *args):
        shapes.append({
            "M": A.shape[0], "N": B.shape[1], "K": A.shape[1],
            "batch": desc.batch_count,
            "dtype_a": A.dtype, "dtype_b": B.dtype,
            "epilogue": desc.epilogue,
            "model": model_name,
            "layer": get_current_layer_name(),
        })
        return original_matmul(desc, A, B, C, *args)

    # Monkey-patch and run
    cann_blas.matmul = logging_matmul
    run_inference(model_name, sample_input)
    cann_blas.matmul = original_matmul

    return deduplicate(shapes)

# Run on your target models
for model in ["llama2-7b", "llama2-70b", "qwen2-72b", "chatglm3-6b"]:
    shapes = extract_gemm_shapes(model)
    save_json(f"shapes/{model}.json", shapes)
```

This gives you a **ground-truth shape database** tied to real workloads, which is far more valuable than synthetic shapes.

------

Want me to go deeper on any specific aspect? The heuristic quality testing (ensuring your algorithm selection actually returns good algorithms) is a particularly interesting sub-problem, and the roofline analysis methodology for Ascend's Cube Units has some nuances worth discussing.





## TODO

- Collect real shapes from your customers and internal teams
  - **massive telemetry** from customers
- Start building the profiling database
- Invest in the offline profiling farm
- The heuristic is a product

