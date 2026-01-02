





## [CUDA Toolkit 8.0 GA1](https://docs.nvidia.com/cuda/archive/8.0/cuda-toolkit-release-notes/index.html) (Sept 2016)

- CUDA Libraries

  **cuBLAS.** The cuBLAS library added a new function cublasGemmEx(), which is an extension of cublas<t/>gemm(). It allows the user to specify the algorithm, as well as the precision of the computation and of the input and output matrices. The function can be used to perform matrix-matrix multiplication at lower precision.

### [2.3.1. cuBLAS Library](https://docs.nvidia.com/cuda/archive/8.0/cuda-toolkit-release-notes/index.html#cublas-new-features)

- The cuBLAS library added a new function cublasGemmEx(), which is an extension of cublas<t/>gemm(). It allows the user to specify the algorithm, as well as the precision of the computation and of the input and output matrices. The function can be used to perform matrix-matrix multiplication at lower precision.
- The cuBLAS library now supports a Gaussian implementation for the GEMM, SYRK, and HERK operations on complex matrices.
- New routines for batched GEMMs, cublas<T>gemmStridedBatch(), have been added. These routines implement a new batch API for GEMMs that is easier to set up. The routines are optimized for performance on GPU architectures sm_5x or greater.
- The cublasXt API now accepts matrices that are resident in GPU memory.

### [5.2.1. cuBLAS Library](https://docs.nvidia.com/cuda/archive/8.0/cuda-toolkit-release-notes/index.html#cublas-performance-improvements)

- The cuBLAS library now supports high-performance SGEMM routines on Maxwell for handling problem sizes where m and n are not necessarily a multiple of the computation tile size. This leads to much smoother and more predictable performance.

### [6.3.1. cuBLAS Library](https://docs.nvidia.com/cuda/archive/8.0/cuda-toolkit-release-notes/index.html#cublas-resolved-issues)

- Fixed GEMM performance issues on Kepler and Pascal for different matrix sizes, including small batches. Note that this fix is available **only** in the cuBLAS packages on the CUDA network repository.
- Updated the cuBLAS headers to use comments that are in compliance with ANSI C standards.
- Made optimizations for mixed-precision (FP16, INT8) matrix-matrix multiplication of matrices with a small number of columns (n).
- Fixed an issue with the trsm() function for large-sized matrices.

## [CUDA Toolkit 8.0 GA2](https://developer.nvidia.com/cuda-80-ga2-download-archive) (Feb 2017)

Same as above

## [CUDA Toolkit 9.0](https://developer.nvidia.com/cuda-90-download-archive) (Sept 2017)

### [2.3.1. cuBLAS Library](https://docs.nvidia.com/cuda/archive/9.0/cuda-toolkit-release-notes/index.html#cublas-new-features)

- cuBLAS 9.0.333 is an update to CUDA Toolkit 9 that improves GEMM computation performance on Tesla V100 systems and includes bug fixes aimed at deep learning and scientific computing applications. The update includes optimized performance of the cublasGemmEx() API for GEMM input sizes used in deep learning applications, such as convolutional sequence to sequence (seq2seq) models, when the CUBLAS_GEMM_DEFAULT_TENSOR_OP and CUBLAS_GEMM_DEFAULT algorithm types are used.
- cuBLAS 9.0.282 is an update to CUDA Toolkit 9 that includes GEMM performance enhancements on Tesla V100 and several bug fixes targeted for both deep learning and scientific computing applications. Key highlights of the update include:
  - Overall performance enhancements across key input sizes that are used in recurrent neural networks (RNNs) and speech models
  - Optimized performance for small-tiled GEMMs with support for new HMMA and FFMA GEMM kernels
  - Improved heuristics to speed up GEMMs across various input sizes
- The Volta architecture is supported, including optimized single-precision and mixed-precision Generalized Matrix-Matrix Multiply (GEMM) matrices, namely: SGEMM and SGEMMEx with FP16 input and FP32 computation for Tesla V100 Tensor Cores.
- Performance enhancements have been made to GEMM matrices that are primarily used in deep learning applications based on Recurrent Neural Networks (RNNs) and Fully Connected Networks (FCNs).
- GEMM heuristics are improved to choose the most optimized GEMM kernel for the input matrices. Heuristics for batched GEMM matrices are also fixed.
- OpenAI GEMM kernels and optimizations in GEMM for small matrices and batch sizes have been integrated. These improvements are transparent with no API changes.

#### Limitations on New Features of the cuBLAS Library in CUDA 9

- Batching GEMM matrices for Tesla V100 Tensor Cores is not supported. You may not be able to use cublas<t>gemmBatched() APIs on Tesla V100 GPUs with Tensor Cores, but these functions will use the legacy FMA and HFMA instructions.

- Some GEMM heuristic optimizations and OpenAI GEMM kernels for small matrices are not available on Tesla V100 Tensor Cores.

- For cublasSetMathMode(), when set to CUBLAS_TENSOR_OP_MATH, cublasSgemm(), cublasGemmEx(), and cublasSgemmEx() will allow the Tensor Cores to be used when A/B types are set to CUDA_R_32F.

  - In the CUDA 9 RC build, the behavior was to perform a down conversion from FP32 to FP16 with Round to Zero.
  - In the production release of CUDA 9, the behavior is to perform a down conversion with Round to Nearest instead.

- To use single-precision and mixed-precision GEMM operations on Tensor Cores, you must do the following:

  1. Set the math mode to Tensor_OP by using the following function call:

     ```
     cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);
     ```

  2. Set the algorithm to CUBLAS_GEMM_DFALT_TENSOR_OP in the cublasGemmEx API.

  The following example shows how to use the GEMMEx API for Tensor Cores.

  ```
  cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);
  
  status = cublasGemmEx(handle,
                        CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, d_A,
                        CUDA_R_32F ,N, d_B,CUDA_R_32F, N, &beta, d_C, 
                        CUDA_R_32F, N, CUDA_R_32F,
                        CUBLAS_GEMM_DFALT_TENSOR_OP);
  ```

## [CUDA Toolkit 9.1](https://developer.nvidia.com/cuda-91-download-archive-new) (Dec 2017)

### [2.3.1. cuBLAS Library](https://docs.nvidia.com/cuda/archive/9.1/cuda-toolkit-release-notes/index.html#cublas-new-features)

- cuBLAS 9.1.181 is an update to CUDA Toolkit 9.1 that improves GEMM computation performance on Tesla V100 systems and includes bug fixes aimed at deep learning and scientific computing applications. The update includes optimized performance of the cublasGemmEx() API for GEMM input sizes used in deep learning applications, such as convolutional sequence to sequence (seq2seq) models, when the CUBLAS_GEMM_DEFAULT_TENSOR_OP and CUBLAS_GEMM_DEFAULT algorithm types are used.
- cuBLAS 9.1.128 is an update to CUDA Toolkit 9.1 that includes GEMM performance enhancements on Tesla V100 and several bug fixes targeted for both deep learning and scientific computing applications. Key highlights of the update include:
  - Overall performance enhancements across key input sizes that are used in recurrent neural networks (RNNs) and speech models
  - Optimized performance for small-tiled GEMMs with support for new HMMA and FFMA GEMM kernels
  - Improved heuristics to speed up GEMMs across various input sizes
- Two functions have been added to improve deep learning performance on GPUs based on the Volta architecture. These functions perform matrix-matrix multiplication of a series of matrices with mixed-precision input, output, and compute formats. They are an extension to the existing batched GEMM API, which now includes the ability to specify mixed-precision formats. You can now take advantage of Tensor Cores on Tesla V100 GPUs to perform batched GEMM computation on 16-bit floating point input and output formats, and use 32-bit floating format for computation. Note that these new functions are available only on GPUs with compute capability >=5.0. For details of these new functions, refer to [cuBLAS Library User Guide (http://docs.nvidia.com/cuda/cublas/)](http://docs.nvidia.com/cuda/cublas/).

### [5.3.1. cuBLAS Library](https://docs.nvidia.com/cuda/archive/9.1/cuda-toolkit-release-notes/index.html#cublas-known-issues)

- The following functions are not supported on the device cuBLAS API library (cublas_device.a):

  - cublas<t>gemmBatched()
  - cublasBatchedGemmEx
  - cublasGemmExStridedBatched

  Any attempt to use these functions with the device API library will result in a link error. For more details about cuBLAS library functions or limitations, refer to

   

  cuBLAS Library User Guide (http://docs.nvidia.com/cuda/cublas/)

## [CUDA Toolkit 9.2](https://developer.nvidia.com/cuda-92-download-archive) (May 2018)

### [2.3.1. cuBLAS Library](https://docs.nvidia.com/cuda/archive/9.2/cuda-toolkit-release-notes/index.html#cublas-new-features)

- Improved performance for a range of small and large tile size matrices that are extensively used in RNN based speech and NLP models, Convolutional seq2seq (Fairseq) models, OpenAI research and other emerging DL applications. These sizes are optimized on the Tesla V100 architecture to deliver enhanced out-of-the-box performance.
- Added GEMM API logging for developers to trace the algorithm and dataset used during the last BLAS API call.
- Improved GEMM performance on Tesla V100 for single and double precision inputs.

- CUDA Libraries


Since CUDA 5.0, the cuBLAS library has supported the ability to call the same cuBLAS APIs from within device routines, i.e. kernels. These routines are implemented using the Dynamic Parallelism feature, which is available starting with the Kepler generation of GPUs.The [device library ](https://docs.nvidia.com/cuda/cublas/index.html#device-api)(cublas_device) that enables this feature, is deprecated in this release and will be dropped starting next release. NOTE: none of the main cuBLAS library functionality and the APIs that can be called from the host, is impacted.

### [5.3.1. cuBLAS Library](https://docs.nvidia.com/cuda/archive/9.2/cuda-toolkit-release-notes/index.html#cublas-known-issues)

- The previously documented behavior of cuBLAS allowed the same handle to be used simultaneously from multiple host threads. However, there are multiple known issues with this, including in application crashes in some instances, and performance degradations in other situations. To avoid this issue, each host thread should use a separate cuBLAS handle to call the APIs. The documentation for the cuBLAS library has also been changed to indicate that simultaneous use of the same handle from multiple host threads is disallowed, as the functionality and performance issues will not be addressed.
- A silent error might occur in cublasGemmEx when called with CUBLAS_GEMM_ALGO2_TENSOR_OP. We recommend applications to not use CUBLAS_GEMM_ALGO2_TENSOR_OP until the issue is fixed.

## [CUDA Toolkit 10.0](https://developer.nvidia.com/cuda-10.0-download-archive) (Sept 2018)

### [2.3.3. cuBLAS Library](https://docs.nvidia.com/cuda/archive/10.0/cuda-toolkit-release-notes/index.html#cublas-new-features)

- Includes Turing architecture-optimized mixed-precision GEMMs for Deep Learning applications.
- Added batched GEMV (General Matrix Vector Multiplication) support for mixed precision (FP16 input and output, and FP32 accumulation) to enable deep learning RNNs using attention models.
- Several improvements made to API logging, including logging the following formats:
  - Append print,
  - Print local time,
  - Append printing version,
  - Append synchronziation via mutex use.
  - Added different levels of logging where detailed information can be printed about gemm, such as:
    - Tensor-Core vs. Non-Tensor Core
    - Tile sizes and other performance options that are used internally, and
    - Grid dimensions and kernel name.

- CUDA Libraries

  The cuBLAS library, to support the ability to call the same cuBLAS APIs from within the device routines (cublas_device), is dropped starting with CUDA 10.0.

## [CUDA Toolkit 10.1 ](https://developer.nvidia.com/cuda-10.1-download-archive-base)(Feb 2019)

### [3.3.1. cuBLAS Library](https://docs.nvidia.com/cuda/archive/10.1/cuda-toolkit-release-notes/index.html#cublas-u1-new-features)

This release features cuBLAS version 10.2.0 which adds the following features and enhancements to cuBLASLt API:

- cuBLASLt added the Tensor Core-accelerated IMMA kernels for Turing architecture GPUs with CUDA_R_8I outputs, float (and optionally per-row) scaling and optional ReLU and/or bias during epilogue phase of matrix multiplication.
- cuBLASLt extended the mixed precision Tensor Core-accelerated complex coverage to support both half (FP16) and single (FP32) precision outputs.
- cuBLASLt out-of-place reduction modes now work with arbitrary number of splits in the K dimension of the matrix multiply operation (i.e., split-K algorithm).
- Several performance improvements are made in this release. These enhancements are targeted for Tensor Core-accelerated mixed and single precision matrix multiplications on Volta and Turing GPU architectures.

### [4.3.1. cuBLAS Library](https://docs.nvidia.com/cuda/archive/10.1/cuda-toolkit-release-notes/index.html#cublas-new-features)

- With this release, on Linux systems, the cuBLAS libraries listed below are now installed in the /usr/lib/<arch>-linux-gnu/ or /usr/lib64/ directories as shared and static libraries. Their interfaces are available in the /usr/include directory:
  - cublas (BLAS)
  - cublasLt (new Matrix Multiply library)
- A new library, the cuBLASLt, is added. The cuBLASLt is a new lightweight library dedicated to GEneral Matrix-to-matrix Multiply (GEMM) operations with a new flexible API. This new library adds flexibility in matrix data layouts, input types, compute types, and also in choosing the algorithmic implementations and heuristics through parameter programmability. Read more at:[ http://docs.nvidia.com/cuda/cublas/index.html#using-the-cublasLt-api](http://docs.nvidia.com/cuda/cublas/index.html#using-the-cublasLt-api).
- The new cuBLASLt library is packaged as a separate binary and a header file. Also, the cuBLASLt now adds support for:
  - Utilization of IMMA tensor core operations on Turing GPUs for int8 input matrices.
  - FP16 half-precision CGEMM split-complex matrix multiplies using tensor cores on Volta and Turing GPUs.

## [CUDA Toolkit 10.2 ](https://developer.nvidia.com/cuda-10.2-download-archive)(Nov 2019)

- CUDA Libraries - cuBLAS

  This patch fixes a bug in cuBLAS that caused silent corruption of results on Volta and Turing architecture GPUs when the following three conditions were met:Batched GEMM APIs (cublasGemmStridedBatchedEx() or cublasGemmBatchedEx()) were called with a batch count above 65535.Mixed precision or fast math was turned on via the CUBLAS_TENSOR_OP_MATH math mode option (https://docs.nvidia.com/cuda/archive/10.2/cublas/index.html#cublassetmathmode).The problem dimensions or pointer alignment did not allow cuBLAS to use tensor core accelerated kernels despite being requested and the fall back occurred to non-tensor core kernels (see https://docs.nvidia.com/cuda/archive/10.2/cublas/index.html#tensorop-restrictions for details).Resolved an issue where CUDA Graph capture with cuBLAS routines on multiple concurrent streams would have caused hangs or data corruption in some cases.Resolved an issue where strided batched GEMM routines can cause misaligned read errors.Resolved an issue where calls to cublasLtMatmul() with non-square and row-major matrices within the cuBLASLt API caused silent corruption of data or failed to execute.Resolved an issue where calls to cublasCgemm() or cublasZgemm() with single-row matrices (dimension M=1), also resulted in a silent corruption of data.Resolved an issue where calls to cublas?gemm() with matrices A,B, or C having CUBLAS_OP_CONJ transposition flag, failed to report as not supported and similarly resulted in silent corruption of data.

## [CUDA Toolkit 11.0.3](https://developer.nvidia.com/cuda-11-0-3-download-archive) (August 2020)

- **cuBLAS**
  - The cuBLAS API was extended with a new function: cublasSetWorkspace(), which allows the user to set the cuBLAS library workspace to a user-owned device buffer, which will be used by cuBLAS to execute all subsequent calls to the library on the currently set stream.
  - The cuBLASLt experimental logging mechanism can be enabled in two ways:
    - By setting the following environment variables before launching the target application:
      - CUBLASLT_LOG_LEVEL=<level> - where level is one of the following levels:
        - "0" - Off - logging is disabled (default)
        - "1" - Error - only errors will be logged
        - "2" - Trace - API calls that launch CUDA kernels will log their parameters and important information
        - "3" - Hints - hints that can potentially improve the application's performance
        - "4" - Heuristics - heuristics log that may help users to tune their parameters
        - "5" - API Trace - API calls will log their parameter and important information
      - CUBLASLT_LOG_MASK=<mask> - while mask is a combination of the following masks:
        - "0" - Off
        - "1" - Error
        - "2" - Trace
        - "4" - Hints
        - "8" - Heuristics
        - "16" - API Trace
      - CUBLASLT_LOG_FILE=<value> - where value is a file name in the format of "<file_name>.%i"; %i will be replaced with the process ID. If CUBLASLT_LOG_FILE is not defined, the log messages are printed to stdout.
    - By using the runtime API functions defined in the cublasLt header:
      - typedef void(*cublasLtLoggerCallback_t)(int logLevel, const char* functionName, const char* message) - A type of callback function pointer.
      - cublasStatus_t cublasLtLoggerSetCallback(cublasLtLoggerCallback_t callback) - Allows to set a call back functions that will be called for every message that is logged by the library.
      - cublasStatus_t cublasLtLoggerSetFile(FILE* file) - Allows to set the output file for the logger. The file must be open and have write permissions.
      - cublasStatus_t cublasLtLoggerOpenFile(const char* logFile) - Allows to give a path in which the logger should create the log file.
      - cublasStatus_t cublasLtLoggerSetLevel(int level) - Allows to set the log level to one of the above mentioned levels.
      - cublasStatus_t cublasLtLoggerSetMask(int mask) - Allows to set the log mask to a combination of the above mentioned masks.
      - cublasStatus_t cublasLtLoggerForceDisable() - Allows to disable to logger for the entire session. Once this API is being called, the logger cannot be reactivated in the current session.

### [2.6.1. cuBLAS Library](https://docs.nvidia.com/cuda/archive/11.0/cuda-toolkit-release-notes/index.html#cublas-new-features)

- cuBLASLt Matrix Multiplication adds support for fused ReLU and bias operations for all floating point types except double precision (FP64).
- Improved batched TRSM performance for matrices larger than 256.
- Many performance improvements have been implemented for the NVIDIA Ampere, Volta, and Turing Architecture based GPUs.
- With this release, on Linux systems, the cuBLAS libraries listed below are now installed in the /usr/local/cuda-11.0 (./lib64/ for lib and ./include/ for headers) directories as shared and static libraries.
- The cuBLASLt logging mechanism can be enabled by setting the following environment variables before launching the target application:
  - CUBLASLT_LOG_LEVEL=<level> - while level is one of the following levels:
    - "0" - Off - logging is disabled (default)
    - "1" - Error - only errors will be logged
    - "2" - Trace - API calls will be logged with their parameters and important information
  - CUBLASLT_LOG_FILE=<value> - while value is a file name in the format of "<file_name>.%i", %i will be replaced with the process id. If CUBLASLT_LOG_FILE is not defined, the log messages are printed to stdout.
- For matrix multiplication APIs:
  - cublasGemmEx, cublasGemmBatchedEx, cublasGemmStridedBatchedEx and cublasLtMatmul has new data type support for BFLOAT16 (CUDA_R_16BF).
  - The newly introduced computeType_t changes function prototypes on the API: cublasGemmEx, cublasGemmBatchedEx, and cublasGemmStridedBatchedEx have a new signature that uses cublasComputeType_t for the computeType parameter. Backward compatibility is ensured with internal mapping for C users and with added overload for C++ users.
  - cublasLtMatmulDescCreate, cublasLtMatmulAlgoGetIds, and cublasLtMatmulAlgoInit have new signatures that use cublasComputeType_t.
  - A new compute type TensorFloat32 (TF32) has been added to provide tensor core acceleration for FP32 matrix multiplication routines with full dynamic range and increased precision compared to BFLOAT16.
  - New compute modes Default, Pedantic, and Fast have been introduced to offer more control over compute precision used.
  - *Init versions of *Create functions are introduced in cublasLt to allow for simple wrappers that hold all descriptors on stack.
  - Experimental feature of cuBLASLt API logging is introduced.
  - Tensor cores are now enabled by default for half-, and mixed-precision- matrix multiplications.
  - Double precision tensor cores (DMMA) are used automatically.
  - Tensor cores can now be used for all sizes and data alignments and for all GPU architectures:
    - Selection of these kernels through cuBLAS heuristics is automatic and will depend on factors such as math mode setting as well as whether it will run faster than the non-tensor core kernels.
    - Users should note that while these new kernels that use tensor cores for all unaligned cases are expected to perform faster than non-tensor core based kernels but slower than kernels that can be run when all buffers are well aligned.

## [CUDA Toolkit 11.1.1](https://developer.nvidia.com/cuda-11-1-1-download-archive) (October 2020)

### [2.7.3. cuBLAS Library](https://docs.nvidia.com/cuda/archive/11.1.1/cuda-toolkit-release-notes/index.html#cublas-resolved-issues)

- A performance regression in the cublasCgetrfBatched and cublasCgetriBatched routines has been fixed.
- The IMMA kernels do not support padding in matrix C and may corrupt the data when matrix C with padding is supplied to cublasLtMatmul. A suggested work around is to supply matrix C with leading dimension equal to 32 times the number of rows when targeting the IMMA kernels: computeType = CUDA_R_32I and CUBLASLT_ORDER_COL32 for matrices A,C,D, and CUBLASLT_ORDER_COL4_4R2_8C (on NVIDIA Ampere GPU architecture or Turing architecture) or CUBLASLT_ORDER_COL32_2R_4R4 (on NVIDIA Ampere GPU architecture) for matrix B. Matmul descriptor must specify CUBLAS_OP_T on matrix B and CUBLAS_OP_N (default) on matrix A and C. The data corruption behavior was fixed so that CUBLAS_STATUS_NOT_SUPPORTED is returned instead.
- Fixed an issue that caused an Address out of bounds error when calling cublasSgemm().
- A performance regression in the cublasCgetrfBatched and cublasCgetriBatched routines has been fixed.

## [CUDA Toolkit 11.3.1](https://developer.nvidia.com/cuda-11-3-1-download-archive) (May 2021)

**cuBLAS**

- New Features:
  - Some new kernels have been added for improved performance but have the limitation that only host pointers are supported for scalars (for example, alpha and beta parameters). This limitation is expected to be resolved in a future release.
  - New epilogues have been added to support fusion in ML training. This includes:
    - ReLuBias and GeluBias epilogues that produce an auxiliary output which is used on backward propagation to compute the corresponding gradients.
    - DReLuBGrad and DGeluBGrad epilogues that compute the backpropagation of the corresponding activation function on matrix C, and produce bias gradient as a separate output. These epilogues require auxiliary input mentioned in the bullet above.
- Deprecations:
  - Linking with static cublas and cublasLt libraries on Linux now requires using gcc-5.2 and compatible or higher due to C++11 requirements in these libraries.
- Known Issues:
  - To be able to access the fastest possible kernels through cublasLtMatmulAlgoGetHeuristic() you need to set CUBLASLT_MATMUL_PREF_POINTER_MODE_MASK in search preferences to CUBLASLT_POINTER_MODE_MASK_HOST or CUBLASLT_POINTER_MODE_MASK_NO_FILTERING. By default, heuristics query assumes the pointer mode may change later and only returns algo configurations that support both _HOST and _DEVICE modes. Without this, newly added kernels will be excluded and it will likely lead to a performance penalty on some problem sizes.

### [1.8.1. cuBLAS Library](https://docs.nvidia.com/cuda/archive/11.3.1/cuda-toolkit-release-notes/index.html#cublas-known-issues)

- The planar complex matrix descriptor for batched matmul has inconsistent interpretation of batch offset.
- Mixed precision operations with reduction scheme CUBLASLT_REDUCTION_SCHEME_OUTPUT_TYPE (might be automatically selected based on problem size by cublasSgemmEx() or cublasGemmEx() too, unless CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION math mode bit is set) not only stores intermediate results in output type but also accumulates them internally in the same precision, which may result in lower than expected accuracy. Please use CUBLASLT_MATMUL_PREF_REDUCTION_SCHEME_MASK or CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION if this results in numerical precision issues in your application.

## [CUDA Toolkit 11.8.0](https://developer.nvidia.com/cuda-11-8-0-download-archive) (October 2022)

### [2.1. cuBLAS Library](https://docs.nvidia.com/cuda/archive/11.8.0/cuda-toolkit-release-notes/index.html#title-cublas-library)



### [2.1.1. cuBLAS: Release 11.8](https://docs.nvidia.com/cuda/archive/11.8.0/cuda-toolkit-release-notes/index.html#cublas-11.8.0)

- **New Features**
  - Improved performance for Hopper by adding Hopper specific kernel.
  - Extended API to support FP8 (8-bit floating point) mixed-precision tensor core accelerated matrix multiplication for compute capability 9.0 (Hopper) and higher (refer to https://docs.nvidia.com/cuda/cublas/index.htmlindex.html#fp8-usage for more details). This includes:
    - Support for two new FP8 data types (CUDA_R_8F_E4M3 and CUDA_R_8F_E5M2) with different dynamic ranges and precisions.
    - New FP8 specific matmul description attributes that allow more control over the computation by allowing configuration of non-default bias types, scaling factors, auxiliary storage data types, and an additional output to store the maximum of absolute values of the output matrix or epilogue output.
  - Allow a new configuration, inner shape, for cublasLtMatmul which impacts non-grid size internal kernel design. This is only supported for compute capability 9.0 and higher.
  - Allow configuration of thread block cluster dimensions for cublasLtMatmul. This is only supported for compute capability 9.0 and higher.
  - Introduced cuBlasLt heuristics cache that stores the mapping of matmul problems to kernels previously selected by heuristics. That helps reduce the host-side overhead for repeating matmul problems. Refer to https://docs.nvidia.com/cuda/cublas/index.html#cublasLt-heuristics-cache.
- **Known Issues**
  - Some H100 specific kernels, including FP8 matrix multiplication, are only supported on the x86 architecture for Windows and Linux.
  - H100 kernels have increased need for workspace size, when running on H100 it’s recommended to provide at least 32 MiB (33554432 B) of workspace (for cuBLASLt calls or if using cublasSetWorkspace()).
- **Deprecations**
  - CUBLASLT_MATMUL_PREF_POINTER_MODE_MASK, CUBLASLT_MATMUL_PREF_EPILOGUE_MASK and cublasLt3mMode_t will be removed in the next major version of the library, along with the already deprecated elements of cuBLASLt API. In the case of both _MASK properties, the filtering functionality is removed and heuristics query will only be restricted by currently requested options as specified within cublasLtMatmulDesc_t.



### [2.1.2. cuBLAS: Release 11.6 Update 2](https://docs.nvidia.com/cuda/archive/11.8.0/cuda-toolkit-release-notes/index.html#cublas-11.6.2)

- **New Features**
  - Performance improvements for batched GEMV.
  - Performance improvements for the following BLAS Level 3 routines on NVIDIA Ampere GPU architecture (SM80): {D,Z}{SYRK,SYMM,TRMM}, Z{HERK,HEMM}.
- **Known Issues**
  - The cublasGetVersion() API return value was updated due to cuBLAS minor version >= 10 and therefore, depending on how the API is used, version checks based on this API can lead to warnings or errors. Use cases such as cublasGetVersion() >= CUBLAS_VERSION will not break based on how the API was updated. The cublasGetProperty() API still returns correct values.
- **Resolved Issues**
  - Fixed incorrect bias gradient computations for CUBLASLT_EPILOGUE_BGRAD{A,B} when the corresponding matrix (A or B) size is greater than 231.
  - Fixed a potential cuBLAS hang when cuBLAS API is called with different CUDA streams but which are the same value-wise (e.g. this could happen in a loop that creates CUDA stream, calls cuBLAS with it, and then deletes the stream).
  - If cuBLAS uses internal CUDA streams, their priority now matches the priority of the stream with which cuBLAS API was called.



### [2.1.3. cuBLAS: Release 11.6](https://docs.nvidia.com/cuda/archive/11.8.0/cuda-toolkit-release-notes/index.html#cublas-11.6.0)

- **New Features**
  - New epilogue options have been added to support fusion in DLtraining: CUBLASLT_EPILOGUE_{DRELU,DGELU} which are similar to CUBLASLT_EPILOGUE_{DRELU,DGELU}_BGRAD but don’t compute bias gradient.
- **Resolved Issues**
  - Some syrk-related functions (cublas{D,Z}syrk, cublas{D,Z}syr2k, cublas{D,Z}syrkx) may fail for matrices which size is greater than 2^31.



### [2.1.4. cuBLAS: Release 11.4 Update 3](https://docs.nvidia.com/cuda/archive/11.8.0/cuda-toolkit-release-notes/index.html#cublas-11.4.3)

- **Resolved Issues**
  - Some cublas and cublasLt functions sometimes returned CUBLAS_STATUS_EXECUTION_FAILED if the dynamic library was loaded and unloaded several times during application lifetime within the same CUDA context. This issue has been resolved.



### [2.1.5. cuBLAS: Release 11.4 Update 2](https://docs.nvidia.com/cuda/archive/11.8.0/cuda-toolkit-release-notes/index.html#cublas-11.4.2)

- **New Features**
  - Vector (and batched) alpha support for per-row scaling in TN int32 math Matmul with int8 output. See CUBLASLT_POINTER_MODE_ALPHA_DEVICE_VECTOR_BETA_HOST and CUBLASLT_MATMUL_DESC_ALPHA_VECTOR_BATCH_STRIDE.
  - New epilogue options have been added to support fusion in DLtraining: CUBLASLT_EPILOGUE_BGRADA and CUBLASLT_EPILOGUE_BGRADB which compute bias gradients based on matrices A and B respectively.
  - New auxiliary functions cublasGetStatusName(), cublasGetStatusString() have been added to cuBLAS that return the string representation and the description of the cuBLAS status (cublasStatus_t) respectively. Similarly, cublasLtGetStatusName(), cublasLtGetStatusString() have been added to cuBlasLt.
- **Known Issues**
  - [cublasGemmBatchedEx()](https://docs.nvidia.com/cuda/cublas/index.html#cublas-GemmBatchedEx) and [cublasgemmBatched()](https://docs.nvidia.com/cuda/cublas/index.html#cublas-lt-t-gt-gemmbatched) check the alignment of the input/output arrays of the pointers like they were the pointers to the actual matrices. These checks are irrelevant and will be disabled in future releases. This mostly affects half-precision inputGEMMs which might require 16-byte alignment, and array of pointers could only be aligned by 8-byte boundary.
- **Resolved Issues**
  - cublasLtMatrixTransform can now operate on matrices with dimensions greater than 65535.
  - Fixed out-of-bound access in GEMM and Matmul functions, when split K or non-default epilogue is used and leading dimension of the output matrix exceeds int32_t limit.
  - NVBLAS now uses lazy loading of the CPU BLAS library on Linux to avoid issues caused by preloading libnvblas.so in complex applications that use fork and similar APIs.
  - Resolved symbols name conflict when using cuBlasLt static library with static TensorRT or cuDNN libraries.



### [2.1.6. cuBLAS: Release 11.4](https://docs.nvidia.com/cuda/archive/11.8.0/cuda-toolkit-release-notes/index.html#cublas-11.4.0)

- **Resolved Issues**
  - Some gemv cases were producing incorrect results if the matrix dimension (n or m) was large, for example 2^20.



### [2.1.7. cuBLAS: Release 11.3 Update 1](https://docs.nvidia.com/cuda/archive/11.8.0/cuda-toolkit-release-notes/index.html#cublas-11.3.1)

- **New Features**
  - Some new kernels have been added for improved performance but have the limitation that only host pointers are supported for scalars (for example, alpha and beta parameters). This limitation is expected to be resolved in a future release.
  - New epilogues have been added to support fusion in ML training. These include:
    - ReLuBias and GeluBias epilogues that produce an auxiliary output which is used on backward propagation to compute the corresponding gradients.
    - DReLuBGrad and DGeluBGrad epilogues that compute the backpropagation of the corresponding activation function on matrix C, and produce bias gradient as a separate output. These epilogues require auxiliary input mentioned in the bullet above.
- **Resolved Issues**
  - Some tensor core accelerated strided batched GEMM routines would result in misaligned memory access exceptions when batch stride wasn't a multiple of 8.
  - Tensor core accelerated cublasGemmBatchedEx (pointer-array) routines would use slower variants of kernels assuming bad alignment of the pointers in the pointer array. Now it assumes that pointers are well aligned, as noted in the documentation.
- **Known Issues**
  - To be able to access the fastest possible kernels through cublasLtMatmulAlgoGetHeuristic() you need to set CUBLASLT_MATMUL_PREF_POINTER_MODE_MASK in search preferences to CUBLASLT_POINTER_MODE_MASK_HOST or CUBLASLT_POINTER_MODE_MASK_NO_FILTERING. By default, heuristics query assumes the pointer mode may change later and only returns algo configurations that support both _HOST and _DEVICE modes. Without this, newly added kernels will be excluded and it will likely lead to a performance penalty on some problem sizes.
- **Deprecated Features**
  - Linking with static cublas and cublasLt libraries on Linux now requires using gcc-5.2 and compatible or higher due to C++11 requirements in these libraries.



### [2.1.8. cuBLAS: Release 11.3](https://docs.nvidia.com/cuda/archive/11.8.0/cuda-toolkit-release-notes/index.html#cublas-11.3.0)

- **Known Issues**
  - The planar complex matrix descriptor for batched matmul has inconsistent interpretation of batch offset.
  - Mixed precision operations with reduction scheme CUBLASLT_REDUCTION_SCHEME_OUTPUT_TYPE (might be automatically selected based on problem size by cublasSgemmEx() or cublasGemmEx() too, unless CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION math mode bit is set) not only stores intermediate results in output type but also accumulates them internally in the same precision, which may result in lower than expected accuracy. Please use CUBLASLT_MATMUL_PREF_REDUCTION_SCHEME_MASK or CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION if this results in numerical precision issues in your application.



### [2.1.9. cuBLAS: Release 11.2](https://docs.nvidia.com/cuda/archive/11.8.0/cuda-toolkit-release-notes/index.html#cublas-11.2.2)

- **Known Issues**
  - cublas<s/d/c/z>Gemm() with very large n and m=k=1 may fail on Pascal devices.



### [2.1.10. cuBLAS: Release 11.1 Update 1](https://docs.nvidia.com/cuda/archive/11.8.0/cuda-toolkit-release-notes/index.html#cublas-11.1.1)

- **New Features**
  - cuBLASLt Logging is officially stable and no longer experimental. cuBLASLt Logging APIs are still experimental and may change in future releases.
- **Resolved Issues**
  - cublasLt Matmul fails on Volta architecture GPUs with CUBLAS_STATUS_EXECUTION_FAILED when n dimension > 262,137 and epilogue bias feature is being used. This issue exists in 11.0 and 11.1 releases but has been corrected in 11.1 Update 1



### [2.1.11. cuBLAS: Release 11.1](https://docs.nvidia.com/cuda/archive/11.8.0/cuda-toolkit-release-notes/index.html#cublas-11.1.0)

- **Resolved Issues**
  - A performance regression in the cublasCgetrfBatched and cublasCgetriBatched routines has been fixed.
  - The IMMA kernels do not support padding in matrix C and may corrupt the data when matrix C with padding is supplied to cublasLtMatmul. A suggested work around is to supply matrix C with leading dimension equal to 32 times the number of rows when targeting the IMMA kernels: computeType = CUDA_R_32I and CUBLASLT_ORDER_COL32 for matrices A,C,D, and CUBLASLT_ORDER_COL4_4R2_8C (on NVIDIA Ampere GPU architecture or Turing architecture) or CUBLASLT_ORDER_COL32_2R_4R4 (on NVIDIA Ampere GPU architecture) for matrix B. Matmul descriptor must specify CUBLAS_OP_T on matrix B and CUBLAS_OP_N (default) on matrix A and C. The data corruption behavior was fixed so that CUBLAS_STATUS_NOT_SUPPORTED is returned instead.
  - Fixed an issue that caused an Address out of bounds error when calling cublasSgemm().
  - A performance regression in the cublasCgetrfBatched and cublasCgetriBatched routines has been fixed.



### [2.1.12. cuBLAS: Release 11.0 Update 1](https://docs.nvidia.com/cuda/archive/11.8.0/cuda-toolkit-release-notes/index.html#cublas-11.0.3)

- **New Features**
  - The cuBLAS API was extended with a new function, cublasSetWorkspace(), which allows the user to set the cuBLAS library workspace to a user-owned device buffer, which will be used by cuBLAS to execute all subsequent calls to the library on the currently set stream.
  - cuBLASLt experimental logging mechanism can be enabled in two ways:
    - By setting the following environment variables before launching the target application:
      - CUBLASLT_LOG_LEVEL=<level> -- where level is one of the following levels:
        - "0" - Off - logging is disabled (default)
        - "1" - Error - only errors will be logged
        - "2" - Trace - API calls that launch CUDA kernels will log their parameters and important information
        - "3" - Hints - hints that can potentially improve the application's performance
        - "4" - Heuristics - heuristics log that may help users to tune their parameters
        - "5" - API Trace - API calls will log their parameter and important information
      - CUBLASLT_LOG_MASK=<mask> -- where mask is a combination of the following masks:
        - "0" - Off
        - "1" - Error
        - "2" - Trace
        - "4" - Hints
        - "8" - Heuristics
        - "16" - API Trace
      - CUBLASLT_LOG_FILE=<value> -- where value is a file name in the format of "<file_name>.%i", %i will be replaced with process id.If CUBLASLT_LOG_FILE is not defined, the log messages are printed to stdout.
    - By using the runtime API functions defined in the cublasLt header:
      - typedef void(*cublasLtLoggerCallback_t)(int logLevel, const char* functionName, const char* message) -- A type of callback function pointer.
      - cublasStatus_t cublasLtLoggerSetCallback(cublasLtLoggerCallback_t callback) -- Allows to set a call back functions that will be called for every message that is logged by the library.
      - cublasStatus_t cublasLtLoggerSetFile(FILE* file) -- Allows to set the output file for the logger. The file must be open and have write permissions.
      - cublasStatus_t cublasLtLoggerOpenFile(const char* logFile) -- Allows to give a path in which the logger should create the log file.
      - cublasStatus_t cublasLtLoggerSetLevel(int level) -- Allows to set the log level to one of the above mentioned levels.
      - cublasStatus_t cublasLtLoggerSetMask(int mask) -- Allows to set the log mask to a combination of the above mentioned masks.
      - cublasStatus_t cublasLtLoggerForceDisable() -- Allows to disable to logger for the entire session. Once this API is being called, the logger cannot be reactivated in the current session.



### [2.1.13. cuBLAS: Release 11.0](https://docs.nvidia.com/cuda/archive/11.8.0/cuda-toolkit-release-notes/index.html#cublas-11.0.2)

- **New Features**
  - cuBLASLt Matrix Multiplication adds support for fused ReLU and bias operations for all floating point types except double precision (FP64).
  - Improved batched TRSM performance for matrices larger than 256.



### [2.1.14. cuBLAS: Release 11.0 RC](https://docs.nvidia.com/cuda/archive/11.8.0/cuda-toolkit-release-notes/index.html#cublas-11.0-RC)

- **New Features**
  - Many performance improvements have been implemented for NVIDIA Ampere, Volta, and Turing Architecture based GPUs.
  - The cuBLASLt logging mechanism can be enabled by setting the following environment variables before launching the target application:
    - CUBLASLT_LOG_LEVEL=<level> - while level is one of the following levels:
      - "0" - Off - logging is disabled (default)
      - "1" - Error - only errors will be logged
      - "2" - Trace - API calls will be logged with their parameters and important information
    - CUBLAS**LT**_LOG_FILE=<value> - while value is a file name in the format of "<file_name>.%i", %i will be replaced with process id. If CUBLAS**LT**_LOG_FILE is not defined, the log messages are printed to stdout.
  - For matrix multiplication APIs:
    - cublasGemmEx, cublasGemmBatchedEx, cublasGemmStridedBatchedEx and cublasLtMatmul added new data type support for __nv_bfloat16 (CUDA_R_16BF).
    - A new compute type TensorFloat32 (TF32) has been added to provide tensor core acceleration for FP32 matrix multiplication routines with full dynamic range and increased precision compared to BFLOAT16.
    - New compute modes Default, Pedantic, and Fast have been introduced to offer more control over compute precision used.
    - Tensor cores are now enabled by default for half-, and mixed-precision- matrix multiplications.
    - Double precision tensor cores (DMMA) are used automatically.
    - Tensor cores can now be used for all sizes and data alignments and for all GPU architectures:
      - Selection of these kernels through cuBLAS heuristics is automatic and will depend on factors such as math mode setting as well as whether it will run faster than the non-tensor core kernels.
      - Users should note that while these new kernels that use tensor cores for all unaligned cases are expected to perform faster than non-tensor core based kernels but slower than kernels that can be run when all buffers are well aligned.
- **Deprecated Features**
  - Algorithm selection in cublasGemmEx APIs (including batched variants) is non-functional for NVIDIA Ampere Architecture GPUs. Regardless of selection it will default to a heuristics selection. Users are encouraged to use the cublasLt APIs for algorithm selection functionality.
  - The matrix multiply math mode CUBLAS_TENSOR_OP_MATH is being deprecated and will be removed in a future release. Users are encouraged to use the new cublasComputeType_t enumeration to define compute precision.

## [CUDA Toolkit 12.9.1](https://developer.nvidia.com/cuda-12-9-1-download-archive) (June 2025)

### 3.1.1. cuBLAS: Release 12.9 Update 1[](https://docs.nvidia.com/cuda/archive/12.9.1/cuda-toolkit-release-notes/index.html#cublas-release-12-9-update-1)

- **New Features**

  - Improved performance for 128×128-element 2D block scaling on NVIDIA Hopper GPUs.

- **Resolved Issues**

  - cuBLAS now enforces the 256-byte alignment requirement for workspace memory.

- **Deprecations**

  - Starting in a future release, cuBLAS will change the order of applying scaling factors for 128-element 1D block scaling and 128x128-element 2D block scaling from:

  > `(scale_a * block_accumulator) * scale_b` to `(scale_a * scale_b) * block_accumulator`
  >
  > so bitwise differences are expected between the new and the old ordering.

### 3.1.2. cuBLAS: Release 12.9[](https://docs.nvidia.com/cuda/archive/12.9.1/cuda-toolkit-release-notes/index.html#cublas-release-12-9)

- **New Features**

  - We have introduced support for independent batch pointers in Matrix Multiplication operations within the cuBLASLt API. This feature, previously available only in the cuBLAS gemmEx API, now enables pointer array batch support for low precision data types. Note that there is currently limited support for fused epilogues.
  - We have added support for new scaling modes on Hopper (`sm_90`), including outer vector (per-channel/per-row), per-128-element, and per-128x128-block. Note that there is currently limited support for fused epilogues.
  - We have enabled up to a 3x speedup and improved energy efficiency in compute-bound instances of FP32 matrix multiplication by using emulated FP32 with the BF16x9 algorithm. This feature is available on a subset of Blackwell GPUs. For more details, click [here](https://docs.nvidia.com/cuda/cublas/index.html#floating-point-emulation-support-overview). To enable FP32 emulation, refer to the [CUDA library samples](https://github.com/NVIDIA/CUDALibrarySamples/tree/master/cuBLAS/Emulation). Note that non-numbers (NaNs, Infs, etc.) are treated as interchangeable error indicators.

- **Known Issues**

  - `cublasLtMatmul` ignores user-specified Aux data types for ReLU epilogues and defaults to using a bitmask. The correct behavior is to return an error if an invalid Aux data type is specified by the user for ReLU epilogues. [*CUB-7984*]

- **Deprecations**

  - In a future release, cuBLAS will enforce 256-byte alignment for workspace memory.

  - In an upcoming release, cuBLAS will return `CUBLAS_STATUS_NOT_SUPPORTED` if any of the following descriptor attributes are set but the corresponding scale is not supported:

    - `CUBLASLT_MATMUL_DESC_A_SCALE_POINTER`
    - `CUBLASLT_MATMUL_DESC_B_SCALE_POINTER`
    - `CUBLASLT_MATMUL_DESC_D_SCALE_POINTER`
    - `CUBLASLT_MATMUL_DESC_D_OUT_SCALE_POINTER`
    - `CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_SCALE_POINTER`

    This behavior is already enforced for non-narrow precision matmuls, and will soon apply to narrow precision matmuls when a scale is set for a non-narrow precision tensor.

  - Share feedback on upcoming deprecations by posting on the [NVIDIA Developer Forums](https://forums.developer.nvidia.com/) or by emailing us at: `Math-Libs-Feedback@nvidia.com`.

### 3.1.3. cuBLAS: Release 12.8 Update 1[](https://docs.nvidia.com/cuda/archive/12.9.1/cuda-toolkit-release-notes/index.html#cublas-release-12-8-update-1)

- **New Features**
  - Performance Improvements on Nvidia Blackwell GPU Architecure:
    - Matrix Multiplication (Matmuls): Enhanced performance for FP8 (both block-scaled and tensor-wide scaled), FP4, and FP16/BF16.
    - BLAS Level 3: Optimized SSYRK, CSYRK, and CHERK operations, especially for unaligned problems.
    - Batched Operations: Improved efficiency for batched GEMMs and batched GEMVs.
  - Added support for block-scaled FP8 and FP4 datatypes on Blackwell GeForce-class GPUs.
  - Improved performance on Blackwell GeForce-class GPUs.
- **Resolved Issues**
  - Using `cublasLtMatmul` with m or n equal to 1 and leading dimensions that cause the input or output matrices to exceed 2^31 elements may result in illegal memory access. [*5113092, 4959900*]
  - Using `cublasLtMatmul` with m or n equal to 1 and the `CUBLASLT_EPILOGUE_BIAS` epilogue may produce incorrect results. [*5104822*]
  - Under rare circumstances, `cublasLtMatmul` running FP8, FP16, or BF16 on a Blackwell GPU may result in a “CUDA Exception: Cluster target block not present” or a “CUDA Error 719: Unspecified launch failure”. [*5124406*]

### 3.1.4. cuBLAS: Release 12.8[](https://docs.nvidia.com/cuda/archive/12.9.1/cuda-toolkit-release-notes/index.html#cublas-release-12-8)

- **New Features**
  - Added support for NVIDIA Blackwell GPU architecture.
  - Extended the cuBLASLt API to support micro-scaled 4-bit and 8-bit floating-point mixed-precision tensor core-accelerated matrix multiplication for compute capability 10.0 (Blackwell) and higher. Extensions include:
    - CUDA_R_4F_E2M1: Integration with `CUDA_R_UE4M3` scales and 16-element scaling blocks.
    - CUDA_R_8F variants: Compatibility with `CUDA_R_UE8` scales and 32-element scaling blocks.
    - [FP8 Matmul Attribute extensions](https://docs.nvidia.com/cuda/cublas/index.html#d-block-scaling-for-fp8-and-fp4-data-types)
      - Support for block-scaled use cases with scaling factor tensors instead of scalars.
      - Ability to compute scaling factors dynamically for output tensors when the output is a 4-bit or 8-bit floating-point data type.
  - Introduced initial support for CUDA in Graphics (CIG) on Windows x64 for NVIDIA Ampere GPU architecture and Blackwell GeForce-class GPUs. CIG contexts are now auto-detected, and cuBLAS selects kernels that comply with CIG shared memory usage limits.
  - Performance improvement on all Hopper GPUs for non-aligned INT8 matmuls.
- **Resolved Issues**
  - The use of `cublasLtMatmul` with `CUBLASLT_EPILOGUE_BGRAD{A,B}` epilogue allowed the output matrix to be in `CUBLASLT_ORDER_ROW` layout, which led to incorrectly computed bias gradients. This layout is now disallowed when using `CUBLASLT_EPILOGUE_BGRAD{A,B}` epilogue. [*4910924*]
- **Deprecations**
  - The experimental feature for [Atomics Synchronization](https://docs.nvidia.com/cuda/cublas/#atomics-synchronization) along rows (`CUBLASLT_MATMUL_DESC_ATOMIC_SYNC_NUM_CHUNKS_D_ROWS`) or columns (`CUBLASLT_MATMUL_DESC_ATOMIC_SYNC_NUM_CHUNKS_D_COLS`) of the output matrix is now deprecated. The functional implementation is still available but not performant and will be removed in a future release.

### 3.1.5. cuBLAS: Release 12.6 Update 2[](https://docs.nvidia.com/cuda/archive/12.9.1/cuda-toolkit-release-notes/index.html#cublas-release-12-6-update-2)

- **New Features**
  - Broad performance improvement on all Hopper GPUs for FP8, FP16 and BF16 matmuls. This improvement also includes the following fused epilogues `CUBLASLT_EPILOGUE_BIAS`, `CUBLASLT_EPILOGUE_RELU`, `CUBLASLT_EPILOGUE_RELU_BIAS`, `CUBLASLT_EPILOGUE_RELU_AUX`, `CUBLASLT_EPILOGUE_RELU_AUX_BIAS`, `CUBLASLT_EPILOGUE_GELU`, and `CUBLASLT_EPILOGUE_GELU_BIAS`.
- **Known Issues**
  - cuBLAS in multi context scenarios may hang with R535 Driver for version below <535.91. [*CUB-7024*]
  - Users may observe suboptimal performance on Hopper GPUs for FP64 GEMMs. A potential workaround is to conditionally turn on swizzling. To do this, users can take the algo returned via `cublasLtMatmulAlgoGetHeuristic` and query if swizzling can be enabled by calling `cublasLtMatmulAlgoCapGetAttribute` with `CUBLASLT_ALGO_CAP_CTA_SWIZZLING_SUPPORT`. If swizzling is supported, you can enable swizzling by calling `cublasLtMatmulAlgoConfigSetAttribute` with `CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING`. [*4872420*]
- **Resolved Issues**
  - `cublasLtMatmul` could ignore the user specified Bias or Aux data types (`CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE` and `CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_DATA_TYPE`) for FP8 matmul operations if these data types do not match the documented limitations in cublasLtMatmulDescAttributes_t <https://docs.nvidia.com/cuda/cublas/#cublasltmatmuldescattributes-t>__. [*44750343, 4801528*]
  - Setting `CUDA_MODULE_LOADING` to `EAGER` could lead to longer library load times on Hopper GPUs due to JIT compilation of PTX kernels. This can be mitigated by setting this environment variable to `LAZY`. [*4720601*]
  - `cublasLtMatmul` with INT8 inputs, INT32 accumulation, INT8 outputs, and FP32 scaling factors could have produced numerical inaccuracies when a `splitk` reduction was used. [*4751576*]

### 3.1.6. cuBLAS: Release 12.6 Update 1[](https://docs.nvidia.com/cuda/archive/12.9.1/cuda-toolkit-release-notes/index.html#cublas-release-12-6-update-1)

- **Known Issues**
  - `cublasLtMatmul` could ignore the user specified Bias or Aux data types (`CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE` and `CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_DATA_TYPE`) for FP8 matmul operations if these data types do not match the documented limitations in [cublasLtMatmulDescAttributes_t](https://docs.nvidia.com/cuda/cublas/#cublasltmatmuldescattributes-t). [*4750343*]
  - Setting `CUDA_MODULE_LOADING` to `EAGER` could lead to longer library load times on Hopper GPUs due to JIT compilation of PTX kernels. This can be mitigated by setting this environment variable to `LAZY`. [*4720601*]
  - `cublasLtMatmul` with INT8 inputs, INT32 accumulation, INT8 outputs, and FP32 scaling factors may produce accuracy issues when a `splitk` reduction is used. To workaround this issue, you can use `cublasLtMatmulAlgoConfigSetAttribute` to set the reduction scheme to none and set the `splitk` value to 1. [*4751576*]

### 3.1.7. cuBLAS: Release 12.6[](https://docs.nvidia.com/cuda/archive/12.9.1/cuda-toolkit-release-notes/index.html#cublas-release-12-6)

- **Known Issues**
  - Computing matrix multiplication and an epilogue with INT8 inputs, INT8 outputs, and FP32 scaling factors can have numerical errors in cases when a second kernel is used to compute the epilogue. This happens because the first GEMM kernel converts the intermediate result from FP32 into INT8 and stores it for the subsequent epilogue kernel to use. If a value is outside of the range of INT8 before the epilogue and the epilogue would bring it into the range of INT8, there will be numerical errors. This issue has existed since before CUDA 12 and there is no known workaround. [*CUB-6831*]
  - `cublasLtMatmul` could ignore the user specified Bias or Aux data types (`CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE` and `CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_DATA_TYPE`) for FP8 matmul operations if these data types do not match the documented limitations in [cublasLtMatmulDescAttributes_t](https://docs.nvidia.com/cuda/cublas/#cublasltmatmuldescattributes-t). [*4750343*]
- **Resolved Issues**
  - `cublasLtMatmul` produced incorrect results when data types of matrices `A` and `B` were different FP8 (for example, `A` is `CUDA_R_8F_E4M3` and `B` is `CUDA_R_8F_E5M2`) and matrix `D` layout was `CUBLASLT_ORDER_ROW`. [*4640468*]
  - `cublasLt` may return not supported on Hopper GPUs in some cases when `A`, `B`, and `C` are of type `CUDA_R_8I` and the compute type is `CUBLAS_COMPUTE_32I`. [*4381102*]
  - cuBLAS could produce floating point exceptions when running GEMM with `K` equal to 0. [*4614629*]

### 3.1.8. cuBLAS: Release 12.5 Update 1[](https://docs.nvidia.com/cuda/archive/12.9.1/cuda-toolkit-release-notes/index.html#cublas-release-12-5-update-1)

- **New Features**
  - Performance improvement to matrix multiplication targeting large language models, specifically for small batch sizes on Hopper GPUs.
- **Known Issues**
  - The bias epilogue (without ReLU or GeLU) may be not supported on Hopper GPUs for strided batch cases. A workaround is to implement batching manually. This will be fixed in a future release.
  - `cublasGemmGroupedBatchedEx` and `cublas<t>gemmGroupedBatched` have large CPU overheads. This will be addressed in an upcoming release.
- **Resolved Issues**
  - Under rare circumstances, executing SYMM/HEMM concurrently with GEMM on Hopper GPUs might have caused race conditions in the host code, which could lead to an Illegal Memory Access CUDA error. [*4403010*]
  - `cublasLtMatmul` could produce an Illegal Instruction CUDA error on Pascal GPUs under the following conditions: batch is greater than 1, and beta is not equal to 0, and the computations are out-of-place (C != D). [*4566993*]

### 3.1.9. cuBLAS: Release 12.5[](https://docs.nvidia.com/cuda/archive/12.9.1/cuda-toolkit-release-notes/index.html#cublas-release-12-5)

- **New Features**
  - cuBLAS adds an experimental API to support mixed precision grouped batched GEMMs. This enables grouped batched GEMMs with FP16 or BF16 inputs/outputs with the FP32 compute type. Refer to [cublasGemmGroupedBatchedEx](https://docs.nvidia.com/cuda/cublas/index.html#cublasgemmgroupedbatchedex) for more details.
- **Known Issues**
  - `cublasLtMatmul` ignores inputs to `CUBLASLT_MATMUL_DESC_D_SCALE_POINTER` and `CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_SCALE_POINTER` if the elements of the respective matrix are not of FP8 types.
- **Resolved Issues**
  - `cublasLtMatmul` ignored the mismatch between the provided scale type and the implied by the documentation, assuming the latter. For instance, an unsupported configuration of `cublasLtMatmul` with the scale type being FP32 and all other types being FP16 would run with the implicit assumption that the scale type is FP16 and produce incorrect results.
  - cuBLAS SYMV failed for large n dimension: 131072 and above for ssymv, 92673 and above for csymv and dsymv, and 65536 and above for zsymv.

### 3.1.10. cuBLAS: Release 12.4 Update 1[](https://docs.nvidia.com/cuda/archive/12.9.1/cuda-toolkit-release-notes/index.html#cublas-release-12-4-update-1)

- **Known Issues**
  - Setting a cuBLAS handle stream to `cudaStreamPerThread` and setting the workspace via `cublasSetWorkspace` will cause any subsequent `cublasSetWorkspace` calls to fail. This will be fixed in an upcoming release.
  - `cublasLtMatmul` ignores mismatches between the provided scale type and the scale type implied by the documentation and assumes the latter. For example, an unsupported configuration of `cublasLtMatmul` with the scale type being FP32 and all other types being FP16 would run with the implicit assumption that the scale type is FP16 which can produce incorrect results. This will be fixed in an upcoming release.
- **Resolved Issues**
  - `cublasLtMatmul` ignored the `CUBLASLT_MATMUL_DESC_AMAX_D_POINTER` for unsupported configurations instead of returning an error. In particular, computing absolute maximum of D is currently supported only for FP8 Matmul when the output data type is also FP8 (`CUDA_R_8F_E4M3` or `CUDA_R_8F_E5M2`).
  - Reduced host-side overheads for some of the cuBLASLt APIs: `cublasLtMatmul()`, `cublasLtMatmulAlgoCheck()`, and `cublasLtMatmulAlgoGetHeuristic()`. The issue was introduced in CUDA Toolkit 12.4.
  - `cublasLtMatmul()` and `cublasLtMatmulAlgoGetHeuristic()` could have resulted in floating point exceptions (FPE) on some Hopper-based GPUs, including Multi-Instance GPU (MIG). The issue was introduced in cuBLAS 11.8.

### 3.1.11. cuBLAS: Release 12.4[](https://docs.nvidia.com/cuda/archive/12.9.1/cuda-toolkit-release-notes/index.html#cublas-release-12-4)

- **New Features**
  - cuBLAS adds experimental APIs to support grouped batched GEMM for single precision and double precision. Single precision also supports the math mode, `CUBLAS_TF32_TENSOR_OP_MATH`. Grouped batch mode allows you to concurrently solve GEMMs of different dimensions (m, n, k), leading dimensions (lda, ldb, ldc), transpositions (transa, transb), and scaling factors (alpha, beta). Please see [gemmGroupedBatched](https://docs.nvidia.com/cuda/cublas/index.html#cublas-t-gemmgroupedbatched) for more details.
- **Known Issues**
  - When the current context has been created using `cuGreenCtxCreate()`, cuBLAS does not properly detect the number of SMs available. The user may provide the corrected SM count to cuBLAS using an API such as `cublasSetSmCountTarget()`.
  - BLAS level 2 and 3 functions might not treat alpha in a BLAS compliant manner when alpha is zero and the pointer mode is set to `CUBLAS_POINTER_MODE_DEVICE`. This is the same known issue documented in cuBLAS 12.3 Update 1.
  - `cublasLtMatmul` with K equals 1 and epilogue `CUBLASLT_EPILOGUE_D{RELU,GELU}_BGRAD` could out-of-bound access the workspace. The issue exists since cuBLAS 11.3 Update 1.
  - `cublasLtMatmul` with K equals 1 and epilogue `CUBLASLT_EPILOGUE_D{RELU,GELU}` could produce illegal memory access if no workspace is provided. The issue exists since cuBLAS 11.6.
  - When captured in CUDA Graph stream capture, cuBLAS routines can create [memory nodes](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#graph-memory-nodes) through the use of stream-ordered allocation APIs, `cudaMallocAsync` and `cudaFreeAsync`. However, as there is currently no support for memory nodes in [child graphs](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#node-types) or graphs launched [from the device](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#device-graph-launch), attempts to capture cuBLAS routines in such scenarios may fail. To avoid this issue, use the [cublasSetWorkspace()](https://docs.nvidia.com/cuda/cublas/index.html#cublassetworkspace) function to provide user-owned workspace memory.

### 3.1.12. cuBLAS: Release 12.3 Update 1[](https://docs.nvidia.com/cuda/archive/12.9.1/cuda-toolkit-release-notes/index.html#cublas-release-12-3-update-1)

- **New Features**
  - Improved performance of heuristics cache for workloads that have a high eviction rate.
- **Known Issues**
  - BLAS level 2 and 3 functions might not treat alpha in a BLAS compliant manner when alpha is zero and the pointer mode is set to `CUBLAS_POINTER_MODE_DEVICE`. The expected behavior is that the corresponding computations would be skipped. You may encounter the following issues: (1) HER{,2,X,K,2K} may zero the imaginary part on the diagonal elements of the output matrix; and (2) HER{,2,X,K,2K}, SYR{,2,X,K,2K} and others may produce NaN resulting from performing computation on matrices A and B which would otherwise be skipped. If strict compliance with BLAS is required, the user may manually check for alpha value before invoking the functions or switch to `CUBLAS_POINTER_MODE_HOST`.
- **Resolved Issues**
  - cuBLASLt matmul operations might have computed the output incorrectly under the following conditions: the data type of matrices A and B is FP8, the data type of matrices C and D is FP32, FP16, or BF16, the beta value is 1.0, the C and D matrices are the same, the epilogue contains GELU activation function.
  - When an application compiled with cuBLASLt from CUDA Toolkit 12.2 update 1 or earlier runs with cuBLASLt from CUDA Toolkit 12.2 update 2 or CUDA Toolkit 12.3, matrix multiply descriptors initialized using `cublasLtMatmulDescInit()` sometimes did not respect attribute changes using `cublasLtMatmulDescSetAttribute()`.
  - Fixed creation of cuBLAS or cuBLASLt handles on Hopper GPUs under the Multi-Process Service (MPS).
  - `cublasLtMatmul` with K equals 1 and epilogue `CUBLASLT_EPILOGUE_BGRAD{A,B}` might have returned incorrect results for the bias gradient.

### 3.1.13. cuBLAS: Release 12.3[](https://docs.nvidia.com/cuda/archive/12.9.1/cuda-toolkit-release-notes/index.html#cublas-release-12-3)

- **New Features**
  - Improved performance on NVIDIA L40S Ada GPUs.
- **Known Issues**
  - cuBLASLt matmul operations may compute the output incorrectly under the following conditions: the data type of matrices A and B is FP8, the data type of matrices C and D is FP32, FP16, or BF16, the beta value is 1.0, the C and D matrices are the same, the epilogue contains GELU activation function.
  - When an application compiled with cuBLASLt from CUDA Toolkit 12.2 update 1 or earlier runs with cuBLASLt from CUDA Toolkit 12.2 update 2 or later, matrix multiply descriptors initialized using `cublasLtMatmulDescInit()` may not respect attribute changes using `cublasLtMatmulDescSetAttribute()`. To workaround this issue, create the matrix multiply descriptor using `cublasLtMatmulDescCreate()` instead of `cublasLtMatmulDescInit()`. This will be fixed in an upcoming release.

### 3.1.14. cuBLAS: Release 12.2 Update 2[](https://docs.nvidia.com/cuda/archive/12.9.1/cuda-toolkit-release-notes/index.html#cublas-release-12-2-update-2)

- **New Features**
  - cuBLASLt will now attempt to decompose problems that cannot be run by a single gemm kernel. It does this by partitioning the problem into smaller chunks and executing the gemm kernel multiple times. This improves functional coverage for very large m, n, or batch size cases and makes the transition from the cuBLAS API to the cuBLASLt API more reliable.
- **Known Issues**
  - cuBLASLt matmul operations may compute the output incorrectly under the following conditions: the data type of matrices A and B is FP8, the data type of matrices C and D is FP32, FP16, or BF16, the beta value is 1.0, the C and D matrices are the same, the epilogue contains GELU activation function.

### 3.1.15. cuBLAS: Release 12.2[](https://docs.nvidia.com/cuda/archive/12.9.1/cuda-toolkit-release-notes/index.html#cublas-release-12-2)

- **Known Issues**
  - cuBLAS initialization fails on Hopper architecture GPUs when MPS is in use with `CUDA_MPS_ACTIVE_THREAD_PERCENTAGE` set to a value less than 100%. There is currently no workaround for this issue.
  - Some Hopper kernels produce incorrect results for batched matmuls with `CUBLASLT_EPILOGUE_RELU_BIAS` or `CUBLASLT_EPILOGUE_GELU_BIAS` and a non-zero `CUBLASLT_MATMUL_DESC_BIAS_BATCH_STRIDE`. The kernels apply the first batch’s bias vector to all batches. This will be fixed in a future release.

### 3.1.16. cuBLAS: Release 12.1 Update 1[](https://docs.nvidia.com/cuda/archive/12.9.1/cuda-toolkit-release-notes/index.html#cublas-release-12-1-update-1)

- **New Features**
  - Support for FP8 on NVIDIA Ada GPUs.
  - Improved performance on NVIDIA L4 Ada GPUs.
  - Introduced an API that instructs the cuBLASLt library to not use some CPU instructions. This is useful in some rare cases where certain CPU instructions used by cuBLASLt heuristics negatively impact CPU performance. Refer to https://docs.nvidia.com/cuda/cublas/index.html#disabling-cpu-instructions.
- **Known Issues**
  - When creating a matrix layout using the `cublasLtMatrixLayoutCreate()` function, the object pointed at by `cublasLtMatrixLayout_t` is smaller than `cublasLtMatrixLayoutOpaque_t` (but enough to hold the internal structure). As a result, the object should not be dereferenced or copied explicitly, as this might lead to out of bound accesses. If one needs to serialize the layout or copy it, it is recommended to manually allocate an object of size `sizeof(cublasLtMatrixLayoutOpaque_t)` bytes, and initialize it using `cublasLtMatrixLayoutInit()` function. The same applies to `cublasLtMatmulDesc_t` and `cublasLtMatrixTransformDesc_t`. The issue will be fixed in future releases by ensuring that `cublasLtMatrixLayoutCreate()` allocates at least `sizeof(cublasLtMatrixLayoutOpaque_t)` bytes.

### 3.1.17. cuBLAS: Release 12.0 Update 1[](https://docs.nvidia.com/cuda/archive/12.9.1/cuda-toolkit-release-notes/index.html#cublas-release-12-0-update-1)

- **New Features**
  - Improved performance on NVIDIA H100 SXM and NVIDIA H100 PCIe GPUs.
- **Known Issues**
  - For optimal performance on NVIDIA Hopper architecture, cuBLAS needs to allocate a bigger internal workspace (64 MiB) than on the previous architectures (8 MiB). In the current and previous releases, cuBLAS allocates 256 MiB. This will be addressed in a future release. A possible workaround is to set the `CUBLAS_WORKSPACE_CONFIG` environment variable to :32768:2 when running cuBLAS on NVIDIA Hopper architecture.
- **Resolved Issues**
  - Reduced cuBLAS host-side overheads caused by not using the cublasLt heuristics cache. This began in the CUDA Toolkit 12.0 release.
  - Added forward compatible single precision complex GEMM that does not require workspace.

### 3.1.18. cuBLAS: Release 12.0[](https://docs.nvidia.com/cuda/archive/12.9.1/cuda-toolkit-release-notes/index.html#cublas-release-12-0)

- **New Features**

  - `cublasLtMatmul` now supports FP8 with a non-zero beta.
  - Added `int64` APIs to enable larger problem sizes; refer to [64-bit integer interface](https://docs.nvidia.com/cuda/cublas/index.html#int64-interface).
  - Added more Hopper-specific kernels for `cublasLtMatmul` with epilogues:
    - `CUBLASLT_EPILOGUE_BGRAD{A,B}`
    - `CUBLASLT_EPILOGUE_{RELU,GELU}_AUX`
    - `CUBLASLT_EPILOGUE_D{RELU,GELU}`
  - Improved Hopper performance on arm64-sbsa by adding Hopper kernels that were previously supported only on the x86_64 architecture for Windows and Linux.

- **Known Issues**

  - There are no forward compatible kernels for single precision complex gemms that do not require workspace. Support will be added in a later release.

- **Resolved Issues**

  - Fixed an issue on NVIDIA Ampere architecture and newer GPUs where `cublasLtMatmul` with epilogue `CUBLASLT_EPILOGUE_BGRAD{A,B}` and a nontrivial reduction scheme (that is, not `CUBLASLT_REDUCTION_SCHEME_NONE`) could return incorrect results for the bias gradient.
  - `cublasLtMatmul` for gemv-like cases (that is, m or n equals 1) might ignore bias with the `CUBLASLT_EPILOGUE_RELU_BIAS` and `CUBLASLT_EPILOGUE_BIAS` epilogues.

  **Deprecations**

  - Disallow including `cublas.h` and `cublas_v2.h` in the same translation unit.
  - Removed:
    - `CUBLAS_MATMUL_STAGES_16x80` and `CUBLAS_MATMUL_STAGES_64x80` from `cublasLtMatmulStages_t`. No kernels utilize these stages anymore.
    - `cublasLt3mMode_t`, `CUBLASLT_MATMUL_PREF_MATH_MODE_MASK`, and `CUBLASLT_MATMUL_PREF_GAUSSIAN_MODE_MASK` from `cublasLtMatmulPreferenceAttributes_t`. Instead, use the corresponding flags from `cublasLtNumericalImplFlags_t`.
    - `CUBLASLT_MATMUL_PREF_POINTER_MODE_MASK`, `CUBLASLT_MATMUL_PREF_EPILOGUE_MASK`, and `CUBLASLT_MATMUL_PREF_SM_COUNT_TARGET` from `cublasLtMatmulPreferenceAttributes_t`. The corresponding parameters are taken directly from `cublasLtMatmulDesc_t`.
    - `CUBLASLT_POINTER_MODE_MASK_NO_FILTERING` from `cublasLtPointerModeMask_t`. This mask was only applicable to `CUBLASLT_MATMUL_PREF_MATH_MODE_MASK` which was removed.

## [CUDA Toolkit 13.1.0 ](https://developer.nvidia.com/cuda-downloads)(December 2025)

### 3.1.1. cuBLAS: Release 13.1[](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html#cublas-release-13-1)

- **New Features**

  - Introduced experimental support for grouped GEMM in cuBLASLt. Users can create a matrix with grouped layout using `cublasLtGroupedMatrixLayoutCreate` or `cublasLtGroupedMatrixLayoutInit`, where matrix shapes are passed as device arrays. `cublasLtMatmul` now accepts matrices with grouped layout, in which case matrices are passed as a device array of pointers, where each pointer is a separate matrix that represents a group with its own shapes. Initial support covers A/B types FP8 (E4M3/E5M2), FP16, and BF16, with C/D types FP16, BF16, and FP32; column-major only, default epilogue, 16-byte alignment; requires GPUs with compute capability 10.x or 11.0.

    In addition, the following experimental features were added as part of grouped GEMM:

    - Per-batch tensor-wide scaling for FP8 inputs, enabled by the new `cublasLtMatmulDescAttributes_t` entry `CUBLASLT_MATMUL_MATRIX_SCALE_PER_BATCH_SCALAR_32F`.
    - Per-batch device-side alpha and beta, enabled by the new `cublasLtMatmulDescAttributes_t` entries `CUBLASLT_MATMUL_DESC_ALPHA_BATCH_STRIDE` and `CUBLASLT_MATMUL_DESC_BETA_BATCH_STRIDE`.

  - Improved performance on NVIDIA DGX Spark for CFP32 GEMMs. [*5514146*]

  - Added SM121 DriveOS support.

  - Improved performance on Blackwell (`sm_100` and `sm_103`) via heuristics tuning for FP32 GEMMs whose shapes satisfy `M, N >> K`. [*CUB-8572*]

  - Improved performance of FP16, FP32, and CFP32 GEMMs on Blackwell Thor.

- **Resolved Issues**

  - Fixed missing memory initialization in `cublasCreate()` that could result in emulation environment variables being ignored. [*CUB-9302*]
  - Removed unnecessary overhead related to loading kernels on GPUs with compute capability 10.3. [*5547886*]
  - Fixed FP8 matmuls potentially failing to launch on multi-device Blackwell GeForce systems. [*CUB-9487*]
  - Added stricter checks for in-place matmul to prevent invalid use cases (`C == D` is allowed if and only if `Cdesc == Ddesc`). As a side effect, users are no longer able to use `D` as a dummy pointer for `C` when using `CUBLASLT_POINTER_MODE_DEVICE` with `beta = 0`. However, a distinct dummy pointer may still be passed. The stricter checking was added in CUDA Toolkit 13.0 Update 2. [*5471880*]
  - Fixed `cublasLtMatmul` with `INT8` inputs, `INT32` accumulation, and `INT32` outputs potentially returning `CUBLAS_STATUS_NOT_SUPPORTED` when dimension `N` is larger than 65,536 or when batch count is larger than 1. [*5541380*]

- **Known Issues**

  - The `Grouped GEMM` cuBLASLt API ignores groups with `k = 0`, which can lead to incorrect results. As a workaround, initialize each output matrix `D` with `beta * C` for all groups before the call, then compute Grouped GEMM as `D += A * B` so that the result for groups with `k = 0` is preserved. This issue applies to the experimental Grouped GEMM cuBLASLt API released in CUDA 13.1. [*CUB-9529*]

### 3.1.2. cuBLAS: Release 13.0 Update 2[](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html#cublas-release-13-0-update-2)

- **New Features**
  - Enabled opt-in fixed-point emulation for FP64 matmuls (D/ZGEMM) which improves performance and power-efficiency. The implementation follows the [Ozaki-1 Scheme](https://doi.org/10.1177/10943420241239588) and leverages an automatic dynamic precision framework to ensure FP64-level accuracy. See [here](https://docs.nvidia.com/cuda/cublas/index.html#fixed-point) for more details on fixed-point emulation along with the [table](https://docs.nvidia.com/cuda/cublas/index.html#floating-point-emulation-support-overview) of supported compute-capabilities and the [CUDA library samples](https://github.com/NVIDIA/CUDALibrarySamples/tree/master/cuBLAS/Emulation) for example usages.
  - Improved performance on NVIDIA [DGX Spark](https://www.nvidia.com/en-us/products/workstations/dgx-spark/) for FP16/BF16 and FP8 GEMMs.
  - Added support for [BF16x9 FP32 emulation](https://docs.nvidia.com/cuda/cublas/#bf16x9) to `cublas[SC]syr[2]k` and `cublasCher[2]k` routines. With the math mode set to `CUBLAS_FP32_EMULATED_BF16X9_MATH`, for large enough problems, cuBLAS will automatically dispatch SYRK and HERK to BF16x9-accelerated algorithms.
- **Resolved Issues**
  - Fixed undefined behavior caused by dereferencing a `nullptr` when passing an uninitialized matrix layout descriptor for `Cdesc` in `cublasLtMatmul`. [*CUB-8911*]
  - Improved performance of `cublas[SCDZ]syr[2]k` and `cublas[CZ]her[2]k` on Hopper GPUs when dimension `N` is large. [*CUB-8293*, *5384826*]
- **Known Issues**
  - `cublasLtMatmul` with INT8 inputs, INT32 accumulation, and INT32 outputs might return `CUBLAS_STATUS_NOT_SUPPORTED` when dimension `N` is larger than 65,536 or when the batch count is larger than 1. The issue has existed since CUDA Toolkit 13.0 Update 1 and will be fixed in a later release. [*5541380*]

### 3.1.3. cuBLAS: Release 13.0 Update 1[](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html#cublas-release-13-0-update-1)

- **New Features**
  - Improved performance:
    - Block-scaled FP4 GEMMs on NVIDIA Blackwell and Blackwell Ultra GPUs
    - `SYMV` on NVIDIA Blackwell GPUs [*5171345*]
    - `cublasLtMatmul` for small cases when run concurrently with other CUDA kernels [*5238629*]
    - TF32 GEMMs on Thor GPUs [*5313616*]
    - [Programmatic Dependent Launch (PDL)](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#programmatic-dependent-launch-and-synchronization) is now supported in some cuBLAS kernels for architectures `sm_90` and above, decreasing kernel launch latencies when executed alongside other PDL kernels.
- **Resolved Issues**
  - Fixed an issue where some `cublasSsyrkx` kernels produced incorrect results when `beta = 0` on NVIDIA Blackwell GPUs. [*CUB-8846*]
  - Resolved issues in `cublasLtMatmul` with INT8 inputs, INT32 accumulation, and INT32 outputs where:
    - `cublasLtMatmul` could have produced incorrect results when A and B matrices used regular ordering (CUBLASLT_ORDER_COL or CUBLASLT_ORDER_ROW). [*CUB-8874*]
    - `cublasLtMatmul` could have been run with unsupported configurations of `alpha`/ `beta`, which must be 0 or 1. [*CUB-8873*]

### 3.1.4. cuBLAS: Release 13.0[](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html#cublas-release-13-0)

- **New Features**

  - The `cublasGemmEx`, `cublasGemmBatchedEx`, and `cublasGemmStridedBatchedEx` functions now accept `CUBLAS_GEMM_AUTOTUNE` as a valid value for the `algo` parameter. When this option is used, the library benchmarks a selection of available algorithms internally and chooses the optimal one based on the given problem configuration. The selected algorithm is cached within the current `cublasHandle_t`, so subsequent calls with the same problem descriptor will reuse the cached configuration for improved performance.

    This is an experimental feature. Users are encouraged to transition to the cuBLASLt API, which provides fine-grained control over algorithm selection through the heuristics API and includes support for additional data types such as FP8 and block-scaled formats, as well as kernel fusion. (see autotuning example in [cuBLASLt](https://github.com/NVIDIA/CUDALibrarySamples/tree/master/cuBLASLt/LtSgemmSimpleAutoTuning)).

  - Improved performance of BLAS Level 3 non-GEMM kernels (SYRK, HERK, TRMM, SYMM, HEMM) for FP32 and CF32 precisions on NVIDIA Blackwell GPUs.

  - This release adds support of SM110 GPUs for arm64-sbsa on Linux.

- **Known Issues**

  - `cublasLtMatmul` previously ignored user-specified auxiliary (Aux) data types for ReLU epilogues and defaulted to using a bitmask. The correct behavior is now enforced: an error is returned if an invalid Aux data type is specified for ReLU epilogues. *[CUB-7984]*

- **Deprecations**

  - The experimental feature for atomic synchronization along the rows (`CUBLASLT_MATMUL_DESC_ATOMIC_SYNC_NUM_CHUNKS_D_ROWS`) and columns (`CUBLASLT_MATMUL_DESC_ATOMIC_SYNC_NUM_CHUNKS_D_COLS`) of the output matrix which was deprecated in 12.8 has now been **removed**.
  - Starting with this release, cuBLAS will return `CUBLAS_STATUS_NOT_SUPPORTED` if any of the following descriptor attributes are set but the corresponding scale is not supported:
    - `CUBLASLT_MATMUL_DESC_A_SCALE_POINTER`
    - `CUBLASLT_MATMUL_DESC_B_SCALE_POINTER`
    - `CUBLASLT_MATMUL_DESC_D_SCALE_POINTER`
    - `CUBLASLT_MATMUL_DESC_D_OUT_SCALE_POINTER`
    - `CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_SCALE_POINTER`
  - Previously, this restriction applied only to [non-narrow precision](https://docs.nvidia.com/cuda/cublas/#narrow-precision-data-types-usage) matmuls. It now also applies to narrow precision matmuls when a scale is set for a non-narrow precision tensor.