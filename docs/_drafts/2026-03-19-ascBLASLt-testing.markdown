## 正确性测试

## The Core Problem

GEMM correctness is subtle because you're dealing with **floating-point non-associativity**. The mathematically correct answer `C = α·A·B + β·C` has a single real-number result, but the actual computed result depends on:

- Accumulation order (which changes with tiling strategy)
- Intermediate precision (FP32 accumulator vs FP16 accumulator)
- Fused vs. unfused epilogue (fused bias+activation may round differently)
- Split-K reduction order
- Hardware-specific rounding behavior of the Cube Units

So "correct" doesn't mean "bit-exact" — it means "within acceptable numerical tolerance for this dtype and algorithm class."

## 1. Reference Implementation — The Golden Truth

You need a **single, trusted, maximally-precise reference** to compare against:

```python
import numpy as np
from decimal import Decimal

class GEMMReference:
    """
    CPU-based, high-precision GEMM reference.
    This is the ground truth. It must be SIMPLE and OBVIOUSLY CORRECT,
    not fast. Never optimize this code.
    """

    @staticmethod
    def gemm_fp64(A, B, C, alpha, beta, transA, transB, epilogue=None, bias=None):
        """
        Everything upcast to FP64 for computation, regardless of input type.
        This gives us ~15 digits of precision as reference.
        """
        A_64 = A.astype(np.float64)
        B_64 = B.astype(np.float64)
        C_64 = C.astype(np.float64)

        if transA:
            A_64 = A_64.T
        if transB:
            B_64 = B_64.T

        # Naive matmul in FP64 — NOT np.matmul which may use optimized BLAS
        # For absolute reference, use pure Python loops (slow but unambiguous)
        result = float(alpha) * np.dot(A_64, B_64) + float(beta) * C_64

        # Apply epilogue in FP64
        if epilogue:
            if bias is not None:
                result = result + bias.astype(np.float64)
            if epilogue.activation == "relu":
                result = np.maximum(result, 0.0)
            elif epilogue.activation == "gelu":
                result = result * 0.5 * (1.0 + np.vectorize(math.erf)(result / math.sqrt(2.0)))
            elif epilogue.activation == "silu":
                result = result / (1.0 + np.exp(-result))

        return result

    @staticmethod
    def gemm_int_reference(A_int8, B_int8, bias_int32=None):
        """
        Integer GEMM reference. This one IS exact —
        int8 × int8 accumulated in int32 has no rounding error
        (as long as K < 2^15, which avoids int32 overflow).
        """
        A_32 = A_int8.astype(np.int32)
        B_32 = B_int8.astype(np.int32)
        result = A_32 @ B_32
        if bias_int32 is not None:
            result += bias_int32
        return result
```

**Critical rule:** The reference implementation must be so simple that its correctness is *obvious by inspection*. No optimizations, no clever tricks. If someone asks "is the reference correct?", the answer should be self-evident from reading the code.

**For extra paranoia**, maintain a second independent reference (e.g., use `mpmath` arbitrary-precision library) and cross-validate the two references against each other:

```python
import mpmath

def gemm_mpmath(A, B, alpha=1.0, beta=0.0, C=None, precision=50):
    """Arbitrary precision reference using mpmath. Glacially slow but inarguable."""
    mpmath.mp.dps = precision  # 50 decimal digits
    M, K = A.shape
    _, N = B.shape
    result = mpmath.matrix(M, N)
    for i in range(M):
        for j in range(N):
            s = mpmath.mpf(0)
            for k in range(K):
                s += mpmath.mpf(float(A[i, k])) * mpmath.mpf(float(B[k, j]))
            result[i, j] = mpmath.mpf(alpha) * s
            if C is not None:
                result[i, j] += mpmath.mpf(beta) * mpmath.mpf(float(C[i, j]))
    return result
```

You only need this for small shapes (M,N,K < 64) to validate the FP64 reference itself. It's a reference-for-the-reference.

## 2. Error Metrics — What "Close Enough" Means

Different metrics catch different kinds of bugs:

```python
class NumericalValidator:
    def __init__(self, dtype_config):
        self.tolerances = self.get_tolerances(dtype_config)

    def validate(self, result_npu, reference_fp64, input_dtype):
        """
        Run ALL metrics. A bug might show up in one metric but not others.
        """
        # Cast reference down to the output dtype for fair comparison
        # (the reference in FP64 may have precision that the output dtype can't represent)
        ref_in_output_dtype = reference_fp64.astype(result_npu.dtype)

        return {
            "element_wise": self.check_element_wise(result_npu, ref_in_output_dtype),
            "matrix_norm":  self.check_matrix_norms(result_npu, reference_fp64),
            "ulp":          self.check_ulp_error(result_npu, ref_in_output_dtype),
            "outliers":     self.check_outlier_elements(result_npu, ref_in_output_dtype),
            "statistics":   self.check_error_distribution(result_npu, ref_in_output_dtype),
        }

    def check_element_wise(self, result, reference):
        """
        The standard atol + rtol check, but done carefully.
        allclose(a, b) checks: |a - b| <= atol + rtol * |b|
        """
        atol = self.tolerances["atol"]
        rtol = self.tolerances["rtol"]

        abs_diff = np.abs(result.astype(np.float64) - reference.astype(np.float64))
        threshold = atol + rtol * np.abs(reference.astype(np.float64))
        failures = abs_diff > threshold

        return {
            "pass": not np.any(failures),
            "num_failures": int(np.sum(failures)),
            "failure_rate": float(np.mean(failures)),
            "max_abs_error": float(np.max(abs_diff)),
            "max_rel_error": float(np.max(
                abs_diff / (np.abs(reference.astype(np.float64)) + 1e-30)
            )),
            # WHERE the failures are — critical for debugging tiling bugs
            "failure_indices": np.argwhere(failures)[:20].tolist(),
        }

    def check_matrix_norms(self, result, reference):
        """
        Matrix-level error norms. These catch systematic bias
        that element-wise checks might miss if every element is
        slightly off but within tolerance.
        """
        diff = result.astype(np.float64) - reference
        ref_norm = np.linalg.norm(reference, ord='fro')

        return {
            "frobenius_abs": float(np.linalg.norm(diff, ord='fro')),
            "frobenius_rel": float(np.linalg.norm(diff, ord='fro') / (ref_norm + 1e-30)),
            "inf_norm_abs": float(np.linalg.norm(diff, ord=np.inf)),
            "one_norm_abs": float(np.linalg.norm(diff, ord=1)),
            # Relative Frobenius should be O(sqrt(K) * eps) for well-behaved GEMM
            "expected_frobenius_rel": float(np.sqrt(result.shape[1]) * self.tolerances["eps"]),
        }

    def check_ulp_error(self, result, reference):
        """
        ULP (Unit in the Last Place) error.
        This is the most precise way to measure floating-point error.
        1 ULP = the smallest representable difference at that magnitude.
        """
        def ulp_distance(a, b):
            """Number of representable floats between a and b."""
            # For FP16: reinterpret as int16, compute integer distance
            a_bits = a.view(np.int16 if a.dtype == np.float16 else np.int32)
            b_bits = b.view(np.int16 if b.dtype == np.float16 else np.int32)
            return np.abs(a_bits.astype(np.int64) - b_bits.astype(np.int64))

        if result.dtype in (np.float16, np.float32):
            ulp_errors = ulp_distance(result, reference)
            return {
                "max_ulp": int(np.max(ulp_errors)),
                "mean_ulp": float(np.mean(ulp_errors)),
                "p99_ulp": float(np.percentile(ulp_errors, 99)),
                # For GEMM, max ULP should be O(K) in the worst case
                "acceptable_max_ulp": result.shape[1] * 2,  # 2*K ULP is generous
            }
        return {"skipped": "ULP not applicable for this dtype"}

    def check_outlier_elements(self, result, reference):
        """
        Look for individual elements that are WAY off.
        These typically indicate a bug, not accumulated rounding error.
        A correct GEMM with rounding error has smooth error distribution.
        A buggy GEMM (wrong tiling, off-by-one in index) has outlier spikes.
        """
        abs_diff = np.abs(result.astype(np.float64) - reference.astype(np.float64))
        mean_error = np.mean(abs_diff)
        std_error = np.std(abs_diff)

        # Elements more than 6 sigma from mean error are suspicious
        outlier_threshold = mean_error + 6 * std_error
        outliers = abs_diff > outlier_threshold

        return {
            "num_outliers": int(np.sum(outliers)),
            "outlier_indices": np.argwhere(outliers)[:20].tolist(),
            "outlier_values_result": result[outliers][:10].tolist() if np.any(outliers) else [],
            "outlier_values_reference": reference[outliers][:10].tolist() if np.any(outliers) else [],
            # Spatial pattern analysis: are outliers clustered?
            # Clustered outliers → tiling boundary bug
            # Random outliers → hardware issue or accumulation order
            "spatial_pattern": self.analyze_outlier_pattern(outliers, result.shape),
        }

    def analyze_outlier_pattern(self, outlier_mask, shape):
        """
        Check if outliers cluster at tile boundaries.
        This is the #1 most useful debug signal for tiling bugs.
        """
        if not np.any(outlier_mask):
            return "no_outliers"

        indices = np.argwhere(outlier_mask)
        rows = indices[:, 0]
        cols = indices[:, 1] if len(indices.shape) > 1 else np.zeros_like(rows)

        patterns = []
        for tile_size in [16, 32, 64, 128, 256]:
            # Check if outliers cluster at multiples of tile_size
            row_at_boundary = np.mean(rows % tile_size < 2) > 0.5  # within 2 of boundary
            col_at_boundary = np.mean(cols % tile_size < 2) > 0.5
            if row_at_boundary or col_at_boundary:
                patterns.append(f"clustered_at_tile_{tile_size}_boundary")

        # Check last-row / last-column pattern (partial tile bug)
        if np.mean(rows > shape[0] - 16) > 0.5:
            patterns.append("last_rows_affected")
        if len(shape) > 1 and np.mean(cols > shape[1] - 16) > 0.5:
            patterns.append("last_cols_affected")

        return patterns if patterns else "random_distribution"

    @staticmethod
    def get_tolerances(dtype_config):
        """
        Tolerance table. These are calibrated from empirical observation
        of what "correct but differently-rounded" GEMM results look like.
        """
        tables = {
            "fp16_fp16_fp16": {
                "eps": 9.77e-4,   # FP16 machine epsilon
                "atol": 1e-2,     # Absolute tolerance
                "rtol": 1e-2,     # Relative tolerance (1%)
            },
            "fp16_fp16_fp32": {
                "eps": 9.77e-4,
                "atol": 1e-3,
                "rtol": 5e-3,
            },
            "bf16_bf16_bf16": {
                "eps": 3.91e-3,   # BF16 has less mantissa precision than FP16
                "atol": 5e-2,
                "rtol": 2e-2,
            },
            "fp32_fp32_fp32": {
                "eps": 1.19e-7,
                "atol": 1e-5,
                "rtol": 1e-5,
            },
            "int8_int8_int32": {
                "eps": 0,         # Integer arithmetic is exact
                "atol": 0,
                "rtol": 0,
            },
        }
        return tables.get(dtype_config, tables["fp32_fp32_fp32"])
```

**Why so many metrics?** Because different bugs manifest in different metrics:

| Bug type                    | Element-wise               | Matrix norm       | ULP               | Outlier pattern            |
| --------------------------- | -------------------------- | ----------------- | ----------------- | -------------------------- |
| Wrong accumulation order    | may pass                   | slightly elevated | elevated mean ULP | no outliers                |
| Off-by-one in tile index    | few failures               | slightly elevated | huge max ULP      | clustered at tile boundary |
| Partial tile not handled    | failures in last rows/cols | moderate          | huge in corners   | last_rows / last_cols      |
| Split-K reduction bug       | random failures            | elevated          | varies            | random                     |
| Epilogue applied twice      | many failures              | very elevated     | huge              | uniform across matrix      |
| Transpose flag ignored      | many failures              | very elevated     | huge              | structured pattern         |
| Wrong dtype conversion      | many failures              | huge              | huge              | uniform                    |
| Uninitialized output memory | sparse wild values         | moderate to huge  | huge              | random, values are garbage |

## 3. Test Input Generation — Not Just Random

The inputs you test with matter enormously:

```python
class TestInputGenerator:
    """
    Generate inputs that stress different numerical regimes.
    Random uniform inputs are the LEAST likely to catch bugs.
    """

    @staticmethod
    def generate_suite(M, N, K, dtype, seed=42):
        rng = np.random.RandomState(seed)
        inputs = {}

        # 1. Standard random — baseline
        inputs["random_uniform"] = {
            "A": rng.uniform(-1, 1, (M, K)).astype(dtype),
            "B": rng.uniform(-1, 1, (K, N)).astype(dtype),
        }

        # 2. Random normal — more realistic for neural network weights
        inputs["random_normal"] = {
            "A": rng.randn(M, K).astype(dtype) * 0.02,  # Xavier-like scale
            "B": rng.randn(K, N).astype(dtype) * 0.02,
        }

        # 3. Large values — test overflow handling
        if dtype == np.float16:
            max_safe = 100.0  # FP16 max is 65504, but K accumulations can overflow
            inputs["large_values"] = {
                "A": rng.uniform(-max_safe, max_safe, (M, K)).astype(dtype),
                "B": rng.uniform(-max_safe, max_safe, (K, N)).astype(dtype),
            }

        # 4. Tiny values — test underflow / denormals
        inputs["tiny_values"] = {
            "A": (rng.randn(M, K) * 1e-5).astype(dtype),
            "B": (rng.randn(K, N) * 1e-5).astype(dtype),
        }

        # 5. Mixed magnitude — the GEMM accumulator sees huge + tiny terms
        # This is where catastrophic cancellation happens
        inputs["mixed_magnitude"] = {
            "A": generate_mixed_magnitude(rng, M, K, dtype),
            "B": rng.randn(K, N).astype(dtype),
        }

        # 6. Sparse inputs — many zeros, stress edge handling
        inputs["sparse_90pct"] = {
            "A": apply_sparsity(rng.randn(M, K), 0.9).astype(dtype),
            "B": apply_sparsity(rng.randn(K, N), 0.9).astype(dtype),
        }

        # 7. Identity-like — result should ≈ input, easy to visually verify
        if M == K:
            inputs["A_identity"] = {
                "A": np.eye(M, K).astype(dtype),
                "B": rng.randn(K, N).astype(dtype),
            }

        # 8. Structured inputs — catch index/transpose bugs
        # A[i,k] = i * K + k — each element is unique and position-dependent
        inputs["structured_sequential"] = {
            "A": np.arange(M * K).reshape(M, K).astype(dtype) / (M * K),
            "B": np.arange(K * N).reshape(K, N).astype(dtype) / (K * N),
        }

        # 9. Single non-zero row/column — isolates specific accumulation paths
        inputs["single_row_A"] = {
            "A": single_nonzero_row(rng, M, K, dtype, row=M//2),
            "B": rng.randn(K, N).astype(dtype),
        }

        # 10. Adversarial for specific tile sizes
        # Put special values right at tile boundaries
        for tile_size in [64, 128, 256]:
            if M > tile_size and N > tile_size:
                inputs[f"boundary_stress_tile{tile_size}"] = {
                    "A": boundary_stress_input(rng, M, K, dtype, tile_size),
                    "B": boundary_stress_input(rng, K, N, dtype, tile_size),
                }

        return inputs

def generate_mixed_magnitude(rng, M, K, dtype):
    """
    Matrix where some rows are O(1) and others are O(1e-4).
    When accumulated together, this causes catastrophic cancellation.
    """
    A = np.zeros((M, K), dtype=np.float64)
    for i in range(M):
        if i % 4 == 0:
            A[i, :] = rng.randn(K) * 100.0
        else:
            A[i, :] = rng.randn(K) * 0.001
    return A.astype(dtype)

def boundary_stress_input(rng, rows, cols, dtype, tile_size):
    """
    Put NaN-adjacent values (very large, very small) at tile boundaries.
    If the tiling is wrong, these propagate visibly into wrong output positions.
    """
    A = rng.randn(rows, cols).astype(np.float64)
    for boundary in range(tile_size, rows, tile_size):
        if boundary < rows:
            A[boundary, :] = 999.0  # distinctive value
            A[boundary - 1, :] = -999.0
    return A.astype(dtype)
```

## 4. Edge Case Matrix — The Bug Catchers

These shapes and configurations specifically target common GEMM implementation bugs:

```python
EDGE_CASE_SHAPES = [
    # === Degenerate dimensions ===
    {"M": 1, "N": 1, "K": 1, "tag": "scalar_gemm"},
    {"M": 1, "N": 1, "K": 4096, "tag": "dot_product"},
    {"M": 1, "N": 4096, "K": 4096, "tag": "single_row_output"},
    {"M": 4096, "N": 1, "K": 4096, "tag": "single_col_output"},
    {"M": 1, "N": 1, "K": 1, "batch": 1000, "tag": "many_scalar_gemms"},

    # === Partial tile cases ===
    # If tile_M=128, these force a partial last tile
    {"M": 129, "N": 4096, "K": 4096, "tag": "one_past_tile_M128"},
    {"M": 127, "N": 4096, "K": 4096, "tag": "one_before_tile_M128"},
    {"M": 4096, "N": 129, "K": 4096, "tag": "one_past_tile_N128"},
    {"M": 4096, "N": 4096, "K": 65, "tag": "partial_K_tile"},

    # === Prime dimensions (worst case for all tiling strategies) ===
    {"M": 17, "N": 19, "K": 23, "tag": "small_primes"},
    {"M": 127, "N": 131, "K": 137, "tag": "medium_primes"},
    {"M": 1021, "N": 1031, "K": 1033, "tag": "large_primes"},
    {"M": 4093, "N": 4099, "K": 4111, "tag": "very_large_primes"},

    # === Powers of 2 (should be the best case — if not, something is wrong) ===
    {"M": 256, "N": 256, "K": 256, "tag": "perfect_aligned_small"},
    {"M": 2048, "N": 2048, "K": 2048, "tag": "perfect_aligned_medium"},

    # === Very tall/wide (extreme aspect ratios) ===
    {"M": 1, "N": 32000, "K": 4096, "tag": "decode_vocab_projection"},
    {"M": 100000, "N": 1, "K": 128, "tag": "extremely_tall"},
    {"M": 16, "N": 16, "K": 131072, "tag": "very_deep_K"},

    # === K=1 (degenerate — outer product, not inner product) ===
    {"M": 1024, "N": 1024, "K": 1, "tag": "outer_product"},

    # === Batch edge cases ===
    {"M": 64, "N": 64, "K": 64, "batch": 1, "tag": "batch_1"},
    {"M": 64, "N": 64, "K": 64, "batch": 1000, "tag": "many_small_batches"},
    {"M": 1, "N": 128, "K": 128, "batch": 96, "tag": "attention_heads_decode"},

    # === Split-K edge cases ===
    # When K is large but M*N is small, split-K is needed for parallelism
    # The reduction step is a common bug source
    {"M": 16, "N": 16, "K": 65536, "tag": "extreme_split_k"},

    # === Zero dimensions (should return immediately, not crash) ===
    {"M": 0, "N": 1024, "K": 1024, "tag": "zero_M"},
    {"M": 1024, "N": 0, "K": 1024, "tag": "zero_N"},
    {"M": 1024, "N": 1024, "K": 0, "tag": "zero_K"},
]

EDGE_CASE_PARAMS = [
    # === Alpha/beta edge cases ===
    {"alpha": 0.0, "beta": 1.0, "tag": "alpha_zero_passthrough_C"},
    {"alpha": 1.0, "beta": 0.0, "tag": "beta_zero_ignore_C"},
    {"alpha": 0.0, "beta": 0.0, "tag": "both_zero_should_be_zero_output"},
    {"alpha": -1.0, "beta": 1.0, "tag": "negative_alpha"},
    {"alpha": 1e-7, "beta": 1.0, "tag": "tiny_alpha"},
    {"alpha": 1e6, "beta": 1.0, "tag": "huge_alpha"},

    # === Transpose combinations ===
    {"transA": False, "transB": False, "tag": "NN"},
    {"transA": True,  "transB": False, "tag": "TN"},
    {"transA": False, "transB": True,  "tag": "NT"},
    {"transA": True,  "transB": True,  "tag": "TT"},

    # === Layout combinations ===
    {"layoutA": "row_major", "layoutB": "row_major", "tag": "RR"},
    {"layoutA": "row_major", "layoutB": "col_major", "tag": "RC"},
    {"layoutA": "col_major", "layoutB": "row_major", "tag": "CR"},
    {"layoutA": "col_major", "layoutB": "col_major", "tag": "CC"},

    # === In-place operation (C is also an input) ===
    {"in_place": True, "tag": "output_aliases_input_C"},

    # === Non-contiguous strides ===
    {"ldA": "padded_to_256", "tag": "non_minimal_leading_dim"},
]
```

## 5. Epilogue Correctness — Test the Fusions

Epilogues are a massive bug surface because they fuse multiple operations:

```python
class EpilogueCorrectnessTest:
    """
    Test each epilogue by comparing:
    fused_result = GEMM_with_epilogue(A, B, bias, ...)
    reference    = apply_epilogue_cpu(GEMM_reference(A, B), bias, ...)

    The reference applies GEMM and epilogue SEPARATELY in FP64.
    """

    EPILOGUE_CONFIGS = [
        # (epilogue_type, has_bias, activation)
        ("none",           False, None),
        ("bias",           True,  None),
        ("bias_relu",      True,  "relu"),
        ("bias_gelu",      True,  "gelu"),
        ("bias_gelu_aux",  True,  "gelu"),  # stores pre-activation for backward
        ("bias_silu",      True,  "silu"),
        ("relu",           False, "relu"),   # activation without bias
        ("residual_add",   False, None),     # D = matmul(A,B) + C_residual
        ("bias_residual",  True,  None),     # D = matmul(A,B) + bias + residual
    ]

    def test_epilogue(self, shape, dtype, epilogue_config):
        A, B = generate_inputs(shape, dtype)
        epilogue_type, has_bias, activation = epilogue_config

        bias = generate_bias(shape.N, dtype) if has_bias else None
        residual = generate_residual(shape.M, shape.N, dtype) if "residual" in epilogue_type else None

        # NPU result: everything fused in one kernel
        npu_result = cann_blas.matmul(A, B,
                                       epilogue=epilogue_type,
                                       bias=bias,
                                       residual=residual)

        # CPU reference: step by step in FP64
        ref = GEMMReference.gemm_fp64(A, B, alpha=1.0)
        if has_bias:
            ref = ref + bias.astype(np.float64)
        if residual is not None:
            ref = ref + residual.astype(np.float64)
        if activation == "relu":
            ref = np.maximum(ref, 0.0)
        elif activation == "gelu":
            ref = gelu_reference_fp64(ref)
        elif activation == "silu":
            ref = silu_reference_fp64(ref)

        return validate(npu_result, ref, dtype)

    def test_epilogue_numerical_edge_cases(self):
        """
        GELU and SiLU have numerically tricky regions.
        Test inputs that land right at the inflection points.
        """
        # GELU: the transition around x=0 is where approximation errors peak
        # Pre-compute GEMM outputs that land in [-3, 3] range
        # (this is where GELU's derivative changes most rapidly)

        # SiLU: x * sigmoid(x) has issues near x = -10
        # (sigmoid underflows, but x * 0 should still be ~0)

        for activation in ["gelu", "silu"]:
            for scale in [0.01, 0.1, 1.0, 10.0, 100.0]:
                A = np.randn(128, 128).astype(np.float16) * scale
                B = np.eye(128).astype(np.float16)  # Identity B → output ≈ A
                # Now the activation sees inputs at scale `scale`
                self.test_epilogue(
                    Shape(128, 128, 128), "fp16",
                    ("bias_" + activation, False, activation)
                )
```

## 6. Determinism Testing

This catches race conditions and non-deterministic reduction:

```python
class DeterminismTest:
    """
    Same inputs → same outputs, every time.
    If not, you have a race condition in your kernel.
    """

    def test_bitwise_determinism(self, shape, dtype, algo, num_runs=20):
        """
        Run the exact same GEMM `num_runs` times.
        All results must be BIT-IDENTICAL.
        """
        A, B = generate_inputs(shape, dtype, seed=12345)

        results = []
        for _ in range(num_runs):
            C = cann_blas.matmul(A, B, algo=algo)
            results.append(C.tobytes())  # raw bytes for exact comparison

        unique_results = set(results)

        return {
            "deterministic": len(unique_results) == 1,
            "num_unique_results": len(unique_results),
            "shape": shape,
            "algo": algo,
        }

    def test_cross_algorithm_consistency(self, shape, dtype):
        """
        Different algorithms may produce slightly different results
        (different accumulation order), but they should all be
        within tolerance of each other and the reference.
        """
        A, B = generate_inputs(shape, dtype, seed=42)
        algos = get_all_algorithms(shape, dtype)
        reference = GEMMReference.gemm_fp64(A, B)

        algo_results = {}
        for algo in algos:
            C = cann_blas.matmul(A, B, algo=algo)
            algo_results[algo.id] = {
                "result": C,
                "error_vs_reference": max_relative_error(C, reference),
            }

        # Every algorithm should be within tolerance of reference
        for algo_id, data in algo_results.items():
            assert data["error_vs_reference"] < tolerance[dtype], \
                f"Algorithm {algo_id} exceeds tolerance"

        # Pairwise: algorithms shouldn't diverge wildly from each other
        algo_ids = list(algo_results.keys())
        for i, j in itertools.combinations(algo_ids, 2):
            inter_algo_error = max_relative_error(
                algo_results[i]["result"],
                algo_results[j]["result"]
            )
            # Inter-algorithm divergence should be same order as
            # individual-to-reference error (not worse)
            assert inter_algo_error < 2 * tolerance[dtype], \
                f"Algorithms {i} and {j} diverge excessively"

    def test_order_independence(self, shape, dtype):
        """
        Swapping A rows or B columns should produce correspondingly
        swapped output rows/columns. This catches index computation bugs.
        """
        A, B = generate_inputs(shape, dtype, seed=42)
        C_original = cann_blas.matmul(A, B)

        # Swap rows 0 and 1 of A → rows 0 and 1 of C should swap
        A_swapped = A.copy()
        A_swapped[0], A_swapped[1] = A[1].copy(), A[0].copy()
        C_swapped = cann_blas.matmul(A_swapped, B)

        assert_bitwise_equal(C_swapped[0], C_original[1])
        assert_bitwise_equal(C_swapped[1], C_original[0])
        assert_bitwise_equal(C_swapped[2:], C_original[2:])
```

## 7. CI Integration — Putting It Together

```
┌───────────────────────────────────────────────────────┐
│                 Functional Test Tiers                   │
├───────────────────────────────────────────────────────┤
│                                                        │
│  PRE-COMMIT (blocking, ~5 min)                        │
│  ├── Core shapes: 20 shapes × 4 dtypes                │
│  ├── All transpose/layout combos on 3 shapes           │
│  ├── All epilogues on 3 shapes                         │
│  ├── Determinism check on 5 shapes                     │
│  ├── Zero-dimension / alpha-beta edge cases            │
│  └── Input types: random_uniform + structured only     │
│                                                        │
│  NIGHTLY (blocking for release branch, ~2 hrs)        │
│  ├── Full edge case shape matrix (~60 shapes)          │
│  ├── All dtype × epilogue × layout combinations        │
│  ├── Full input generation suite (10 input types)      │
│  ├── All algorithms tested (not just heuristic pick)   │
│  ├── Cross-algorithm consistency                       │
│  ├── Determinism: 50 runs per shape                    │
│  ├── Numerical precision tracking (store error stats)  │
│  └── Tolerance regression (error shouldn't GROW        │
│       across versions even if within tolerance)        │
│                                                        │
│  RELEASE (blocking, ~8 hrs)                            │
│  ├── Everything in nightly                             │
│  ├── Exhaustive shape sweep: ~2000 shapes              │
│  ├── Cross-chip consistency (same algo on 910A vs 910B │
│  │   should produce results within tolerance)          │
│  ├── Stress test: 10K iterations per critical shape    │
│  ├── Memory leak check (run GEMM in loop, watch RSS)   │
│  ├── Concurrent execution (multiple streams/processes) │
│  └── End-to-end model accuracy check:                  │
│       run LLaMA inference, verify output token matches │
└───────────────────────────────────────────────────────┘
```

### The End-to-End Model Accuracy Test

This is the ultimate functional test — does the BLAS library produce correct model outputs?

```python
def test_model_accuracy(model_name="llama2-7b"):
    """
    Run inference on a fixed prompt with fixed weights.
    Compare output logits (not just tokens) against a
    CPU FP32 reference run.

    This catches errors that are within per-GEMM tolerance
    but accumulate across 32+ layers to produce wrong outputs.
    """
    model = load_model(model_name)
    prompt = FIXED_TEST_PROMPT  # deterministic, version-controlled
    input_ids = tokenize(prompt)

    # NPU forward pass
    npu_logits = model.forward_npu(input_ids)

    # CPU FP32 reference (slow but trusted)
    cpu_logits = model.forward_cpu_fp32(input_ids)

    # Check logit-level agreement
    max_logit_diff = np.max(np.abs(npu_logits - cpu_logits))
    assert max_logit_diff < 0.5, \
        f"Logit divergence too large: {max_logit_diff}"

    # Check top-1 token agreement (must match for at least 95% of positions)
    npu_tokens = np.argmax(npu_logits, axis=-1)
    cpu_tokens = np.argmax(cpu_logits, axis=-1)
    agreement = np.mean(npu_tokens == cpu_tokens)
    assert agreement > 0.95, \
        f"Top-1 token agreement too low: {agreement:.1%}"

    # Check that error doesn't grow across layers
    # (measure per-layer output divergence)
    per_layer_errors = model.forward_with_layer_comparison(input_ids)
    for layer_idx, error in enumerate(per_layer_errors):
        # Error should grow sub-linearly with depth
        expected_max = 0.01 * np.sqrt(layer_idx + 1)
        assert error < expected_max, \
            f"Layer {layer_idx} error {error} exceeds expected {expected_max}"
```

### Tolerance Regression Tracking

This is a subtle but important concept — even if all results are "within tolerance," the *magnitude* of the error shouldn't increase across versions:

```python
class ToleranceRegressionTracker:
    """
    Track error statistics over time. Alert if errors are growing,
    even if they're still passing.

    Why? Because a growing error trend means you're approaching the
    tolerance boundary. Eventually you'll cross it and have a hard failure.
    Better to catch the trend early.
    """

    def check_error_trend(self, shape, dtype, current_errors, history_db):
        historical = history_db.get_last_n_runs(shape, dtype, n=20)

        current_median_error = np.median(current_errors)
        historical_median = np.median([h.median_error for h in historical])

        # Alert if median error increased by >50%, even if within tolerance
        if current_median_error > 1.5 * historical_median:
            return Warning(
                f"Error trend increasing for {shape}/{dtype}: "
                f"{historical_median:.2e} → {current_median_error:.2e} "
                f"(still within tolerance but trending wrong direction)"
            )
```

## Summary: The Full Correctness Test Taxonomy

| Test category                       | What it catches                | When to run  |
| ----------------------------------- | ------------------------------ | ------------ |
| Element-wise against FP64 reference | Gross computation errors       | Every commit |
| ULP error analysis                  | Subtle precision loss          | Nightly      |
| Outlier spatial analysis            | Tiling boundary bugs           | Every commit |
| Input diversity (10 types)          | Numerical corner cases         | Nightly      |
| Edge case shapes                    | Partial tiles, degenerate dims | Every commit |
| Alpha/beta/transpose combos         | Parameter handling bugs        | Every commit |
| Epilogue correctness                | Fusion bugs                    | Every commit |
| Determinism (N runs)                | Race conditions                | Nightly      |
| Cross-algorithm consistency         | Algorithm-specific bugs        | Nightly      |
| Order independence                  | Index computation bugs         | Nightly      |
| Tolerance regression tracking       | Creeping precision loss        | Nightly      |
| End-to-end model accuracy           | Accumulated layer errors       | Release      |
| Cross-chip consistency              | Hardware-specific bugs         | Release      |

The functional correctness suite and the performance benchmark suite should share the same shape database but run independently — a shape that regresses in performance might still be correct, and vice versa. Both gates must pass for a release.

Want me to go deeper on any particular area — the tolerance calibration methodology, the spatial outlier analysis for debugging tiling bugs, or how to structure the test result database for long-term trend analysis?