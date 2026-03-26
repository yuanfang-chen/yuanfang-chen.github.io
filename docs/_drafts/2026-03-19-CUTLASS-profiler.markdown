## CUTLASS Profiler Use Case

The CUTLASS Profiler (`cutlass_profiler`) is essentially a **command-line benchmarking and kernel selection tool** for the CUTLASS library. Its primary use cases are:

**1. Kernel Performance Benchmarking**

It's a command-line driven test and profiling environment for CUTLASS computations defined in the CUTLASS Instance Library, capable of executing each GEMM, Sparse GEMM, Conv2d, and Conv3d kernel. For each problem it reports runtime, memory bandwidth (GiB/s), and compute throughput (GFLOP/s).

**2. Best-Kernel Selection for a Given Problem Shape**

This is probably the most important use case, especially relevant to your CANN library work. You can sweep over all compiled kernel variants for a given set of problem dimensions (M, N, K) and find the fastest one. The `--enable-best-kernel-for-fixed-shape` flag does exactly this — it performs an exhaustive performance search across kernel configurations, sorting results by FLOPS/sec. This is how teams build heuristic lookup tables that map problem shapes → best kernel config.

**3. Correctness Verification**

The profiler supports verification modes including comparing output against cuBLAS as a reference implementation, comparing against CUTLASS device reference kernels, or disabling verification entirely. So it doubles as a functional correctness tool, not just a perf tool.

**4. Tuning Kernel Parameters**

The profiler lets you sweep over tile sizes, cluster shapes, pipeline stages, swizzle patterns, data distributions, and more. For example, finding the best swizzle number for a workload or the right tile size for a particular problem may require trial and error using the CUTLASS Profiler.

**5. Feeding Heuristic Generation**

You can set `CUTLASS_LIBRARY_HEURISTICS_PROBLEMS_FILE` to a file containing problem specifications, and the build system queries heuristics to determine optimal kernel configurations. This is how the profiler data flows into cuBLASLt-style automatic kernel selection heuristics.

**Why this matters for your CANN library design:** The CUTLASS Profiler is basically the offline data-collection infrastructure that feeds into cuBLASLt's runtime kernel selection heuristics. It's the mechanism by which NVIDIA figures out "for this (M, N, K, dtype, layout) tuple, which tile size / cluster shape / schedule produces the best throughput." If you're building the CANN equivalent, you'd need an analogous profiling harness — something that can enumerate your compiled kernel variants, sweep problem shapes, measure performance, verify correctness against a reference, and export the results into whatever heuristic format your runtime dispatch layer consumes.

## Using CUTLASS profiler

Here's a walkthrough of using the CUTLASS Profiler, from building it to running various profiling commands.

### 1. Building the Profiler

First, configure and build. You target specific GPU architectures and optionally filter which kernels get compiled:

```bash
# Build for Ampere (SM80) with all kernels
cmake .. -DCUTLASS_NVCC_ARCHS=80 -DCUTLASS_LIBRARY_KERNELS=all -DCUTLASS_UNITY_BUILD_ENABLED=ON
make cutlass_profiler -j16
```

To compile a subset of kernels (reducing build time), you can use wildcard-based filtering:

```bash
# Only Tensor Core FP16 GEMM kernels for Ampere+Turing
cmake .. -DCUTLASS_NVCC_ARCHS='75;80' \
  -DCUTLASS_LIBRARY_KERNELS=cutlass_tensorop_s*gemm_f16_*_nt_align8
make cutlass_profiler -j16
```

### 2. Profiling a Single SGEMM Kernel

The simplest example profiles a single-precision GEMM:

```bash
./tools/profiler/cutlass_profiler \
  --kernels=sgemm \
  --m=3456 --n=4096 --k=4096
```

This produces output like:

```
=============================
  Problem ID: 1
  Provider: CUTLASS
  OperationKind: gemm
  Operation: cutlass_simt_sgemm_128x128_8x2_nn_align1

  Status:       Success
  Verification: ON
  Disposition:  Passed
  cuBLAS:       Passed

  Arguments:  --m=3456 --n=4096 --k=4096 --A=f32:column --B=f32:column
              --C=f32:column --alpha=1 --beta=0 --split_k_slices=1 ...

  Bytes:   180355072  bytes
  FLOPs:   115992428544 flops
  Runtime: 6.73655  ms
  Memory:  24.934   GiB/s
  Math:    17218.4  GFLOP/s
=============================
```

The profiler automatically verifies results against cuBLAS and reports throughput metrics.

### 3. Profiling Tensor Core GEMM Kernels

```bash
./tools/profiler/cutlass_profiler \
  --kernels=cutlass_tensorop_s*gemm_f16_*_nt_align8 \
  --m=3456 --n=4096 --k=4096
```

This runs all matching Tensor Core FP16 GEMM kernels and reports performance for each.

### 4. Profiling Convolutions

For a 2D convolution forward-propagation kernel:

```bash
./tools/profiler/cutlass_profiler \
  --kernels=s1688fprop \
  --n=8 --h=224 --w=224 --c=128 --k=128 \
  --r=3 --s=3 --pad_h=1 --pad_w=1
```

### 5. Sweeping ("Schmoo") Over Problem Sizes

You can sweep ranges of M, N, K using `start:end:step` syntax:

```bash
./tools/profiler/cutlass_profiler \
  --operation=blockwise_gemm \
  --m=1024:4096:256 \
  --n=1024:4096:256 \
  --k=128:8192:128 \
  --beta=0,1,2.5
```

This profiles every combination in those ranges, which is useful for finding the best kernel for your workload.

### 6. Useful Flags

Some handy options to know:

- `--mode=profile` — default, runs verification + profiling
- `--mode=dry_run` — no kernels launched, useful for checking configuration
- `--mode=enumerate` — lists all available operations and kernels
- `--providers=cutlass` — restrict to CUTLASS only (skip cuBLAS comparison)
- `--output=results.csv` — save results to CSV for analysis
- `--profiling-iterations=N` — control how many timing iterations to run
- `--save-workspace=incorrect` — dump workspace data if verification fails (great for debugging)
- `--split_k_mode=parallel --split_k_slices=2` — test split-K configurations

### 7. Grouped GEMM (Newer Feature)

Grouped GEMM is also supported in the profiler:

```bash
./cutlass_profiler --operation=GroupedGemm --help
```

This gives you a full end-to-end workflow: build only the kernels you care about, profile them across your target problem sizes, and export the results for analysis.