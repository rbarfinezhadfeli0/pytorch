# Documentation: `aten/src/ATen/native/cuda/linalg/BatchLinearAlgebraLib.h`

## File Metadata

- **Path**: `aten/src/ATen/native/cuda/linalg/BatchLinearAlgebraLib.h`
- **Size**: 4,318 bytes (4.22 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
#pragma once

#include <ATen/core/Tensor.h>
#include <ATen/Context.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDACachingAllocator.h>

#include <ATen/native/TransposeType.h>
#include <ATen/native/cuda/MiscUtils.h>

#if (defined(CUDART_VERSION) && defined(CUSOLVER_VERSION)) || defined(USE_ROCM)
#define USE_LINALG_SOLVER
#endif

// cusolverDn<T>potrfBatched may have numerical issue before cuda 11.3 release,
// (which is cusolver version 11101 in the header), so we only use cusolver potrf batched
// if cuda version is >= 11.3
#if CUSOLVER_VERSION >= 11101
  constexpr bool use_cusolver_potrf_batched_ = true;
#else
  constexpr bool use_cusolver_potrf_batched_ = false;
#endif

// cusolverDn<T>syevjBatched may have numerical issue before cuda 11.3.1 release,
// (which is cusolver version 11102 in the header), so we only use cusolver syevj batched
// if cuda version is >= 11.3.1
// See https://github.com/pytorch/pytorch/pull/53040#issuecomment-793626268 and https://github.com/cupy/cupy/issues/4847
#if CUSOLVER_VERSION >= 11102
  constexpr bool use_cusolver_syevj_batched_ = true;
#else
  constexpr bool use_cusolver_syevj_batched_ = false;
#endif

// From cuSOLVER doc: Jacobi method has quadratic convergence, so the accuracy is not proportional to number of sweeps.
//   To guarantee certain accuracy, the user should configure tolerance only.
// The current pytorch implementation sets gesvdj tolerance to epsilon of a C++ data type to target the best possible precision.
constexpr int cusolver_gesvdj_max_sweeps = 400;


namespace at::native {

void geqrf_batched_cublas(const Tensor& input, const Tensor& tau);
void triangular_solve_cublas(const Tensor& A, const Tensor& B, bool left, bool upper, TransposeType transpose, bool unitriangular);
void triangular_solve_batched_cublas(const Tensor& A, const Tensor& B, bool left, bool upper, TransposeType transpose, bool unitriangular);
void gels_batched_cublas(const Tensor& a, Tensor& b, Tensor& infos);
void ldl_factor_cusolver(
    const Tensor& LD,
    const Tensor& pivots,
    const Tensor& info,
    bool upper,
    bool hermitian);
void ldl_solve_cusolver(
    const Tensor& LD,
    const Tensor& pivots,
    const Tensor& B,
    bool upper);
void lu_factor_batched_cublas(const Tensor& A, const Tensor& pivots, const Tensor& infos, bool get_pivots);
void lu_solve_batched_cublas(const Tensor& LU, const Tensor& pivots, const Tensor& B, TransposeType transpose);

#if defined(USE_LINALG_SOLVER)

// entrance of calculations of `svd` using cusolver gesvdj and gesvdjBatched
void svd_cusolver(const Tensor& A, const bool full_matrices, const bool compute_uv,
  const std::optional<std::string_view>& driver, const Tensor& U, const Tensor& S, const Tensor& V, const Tensor& info);

// entrance of calculations of `cholesky` using cusolver potrf and potrfBatched
void cholesky_helper_cusolver(const Tensor& input, bool upper, const Tensor& info);
Tensor _cholesky_solve_helper_cuda_cusolver(const Tensor& self, const Tensor& A, bool upper);
Tensor& cholesky_inverse_kernel_impl_cusolver(Tensor &result, Tensor& infos, bool upper);

void geqrf_cusolver(const Tensor& input, const Tensor& tau);
void ormqr_cusolver(const Tensor& input, const Tensor& tau, const Tensor& other, bool left, bool transpose);
Tensor& orgqr_helper_cusolver(Tensor& result, const Tensor& tau);

void linalg_eigh_cusolver(const Tensor& eigenvalues, const Tensor& eigenvectors, const Tensor& infos, bool upper, bool compute_eigenvectors);

void linalg_eig_cusolver_xgeev(const Tensor& eigenvalues, const Tensor& eigenvectors, const Tensor& input, const Tensor& infos, bool compute_eigenvectors);



void lu_solve_looped_cusolver(const Tensor& LU, const Tensor& pivots, const Tensor& B, TransposeType transpose);

void lu_factor_looped_cusolver(const Tensor& self, const Tensor& pivots, const Tensor& infos, bool get_pivots);

#endif  // USE_LINALG_SOLVER

#if defined(BUILD_LAZY_CUDA_LINALG)
namespace cuda { namespace detail {
// This is only used for an old-style dispatches
// Please do not add any new entries to it
struct LinalgDispatch {
   Tensor (*cholesky_solve_helper)(const Tensor& self, const Tensor& A, bool upper);
};
C10_EXPORT void registerLinalgDispatch(const LinalgDispatch& /*disp_*/);
}} // namespace cuda::detail
#endif

} // namespace at::native

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 20 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `cuda`, `detail`, `at`

**Classes/Structs**: `LinalgDispatch`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `aten/src/ATen/native/cuda/linalg`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



## Dependencies

### Import Dependencies

This file includes:

- `ATen/core/Tensor.h`
- `ATen/Context.h`
- `ATen/cuda/CUDAContext.h`
- `c10/cuda/CUDACachingAllocator.h`
- `ATen/native/TransposeType.h`
- `ATen/native/cuda/MiscUtils.h`


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.

*Detailed performance analysis requires profiling and benchmarking.*


## Security & Safety

### Security Considerations

- No obvious security concerns detected in automated analysis.

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

Test files for this module may be located in the `test/` directory.

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`aten/src/ATen/native/cuda/linalg`):

- [`MagmaUtils.h_docs.md`](./MagmaUtils.h_docs.md)
- [`BatchLinearAlgebraLibBlas.cpp_docs.md`](./BatchLinearAlgebraLibBlas.cpp_docs.md)
- [`CudssHandlePool.cpp_docs.md`](./CudssHandlePool.cpp_docs.md)
- [`CUDASolver.h_docs.md`](./CUDASolver.h_docs.md)
- [`BatchLinearAlgebraLib.cpp_docs.md`](./BatchLinearAlgebraLib.cpp_docs.md)
- [`CUDASolver.cpp_docs.md`](./CUDASolver.cpp_docs.md)
- [`CusolverDnHandlePool.cpp_docs.md`](./CusolverDnHandlePool.cpp_docs.md)
- [`BatchLinearAlgebra.cpp_docs.md`](./BatchLinearAlgebra.cpp_docs.md)


## Cross-References

- **File Documentation**: `BatchLinearAlgebraLib.h_docs.md`
- **Keyword Index**: `BatchLinearAlgebraLib.h_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
