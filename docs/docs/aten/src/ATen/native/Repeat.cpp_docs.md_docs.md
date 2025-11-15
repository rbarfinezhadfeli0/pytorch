# Documentation: `docs/aten/src/ATen/native/Repeat.cpp_docs.md`

## File Metadata

- **Path**: `docs/aten/src/ATen/native/Repeat.cpp_docs.md`
- **Size**: 6,408 bytes (6.26 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `aten/src/ATen/native/Repeat.cpp`

## File Metadata

- **Path**: `aten/src/ATen/native/Repeat.cpp`
- **Size**: 3,756 bytes (3.67 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <ATen/native/Repeat.h>
#include <c10/util/irange.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/empty.h>
#include <ATen/ops/repeat_interleave.h>
#include <ATen/ops/repeat_interleave_native.h>
#endif

template <typename index_t>
static void compute_cpu(
    const index_t* repeat_ptr,
    const int64_t* cumsum_ptr,
    index_t* result_ptr,
    int64_t size,
    int64_t result_size) {
  TORCH_CHECK(
      (result_size == cumsum_ptr[size - 1]),
      "allocated size does not match required size");
  at::parallel_for(0, size, 1, [&](int64_t i_begin, int64_t i_end) {
    for (const auto i : c10::irange(i_begin, i_end)) {
      int64_t end = cumsum_ptr[i];
      index_t size = repeat_ptr[i];
      TORCH_CHECK((size >= 0), "repeats can not be negative");
      int64_t start = end - size;
      for (const auto j : c10::irange(start, end)) {
        result_ptr[j] = i;
      }
    }
  });
}

namespace at::native {

Tensor repeat_interleave_cpu(
    const Tensor& repeat,
    std::optional<int64_t> output_size) {
  Tensor output;
  AT_DISPATCH_INDEX_TYPES(repeat.scalar_type(), "repeat_interleave_cpu", [&]() {
    output = repeat_interleave_common<index_t, compute_cpu<index_t>>(
        repeat, output_size);
  });

  return output;
}

Tensor repeat_interleave_symint(
    const Tensor& self,
    const Tensor& repeats,
    std::optional<int64_t> dim,
    std::optional<SymInt> output_size) {
  Tensor input = self;

  // Store conj and neg bits
  const auto conj = input.is_conj();
  if (conj) {
    input = input.conj();
  }
  const auto neg = input.is_neg();
  if (neg) {
    input = input._neg_view();
  }

  if (!dim) {
    input = input.flatten();
    dim = 0;
  }

  Tensor repeats_ = repeats;
  if (repeats.dim() == 0 || (repeats.dim() == 1 && TORCH_GUARD_OR_FALSE(repeats.sym_size(0).sym_eq(1)))) {
    repeats_ = repeats.reshape({1}).expand_symint({input.sym_size(dim.value())});
  } else if (repeats.dim() == 1) {
    TORCH_CHECK(
        repeats.sym_size(0) == input.sym_size(dim.value()),
        "repeats must have the same size as input along dim, but got repeats.size(0) = ",
        repeats.sym_size(0), " and input.size(", dim.value(), ") = ", input.sym_size(dim.value())
    );
  } else {
    TORCH_CHECK(false, "repeats must be 0-dim or 1-dim tensor");
  }

  auto ret = input.index_select(
      dim.value(), at::repeat_interleave_symint(repeats_, std::move(output_size)));
  // Restore conj and neg bits
  if (conj) {
    ret = ret.conj();
  }
  if (neg) {
    ret = ret._neg_view();
  }
  return ret;
}

Tensor repeat_interleave_symint(
    const Tensor& self,
    c10::SymInt repeats,
    std::optional<int64_t> dim_opt,
    std::optional<SymInt> output_size) {
  Tensor input = dim_opt ? self : self.flatten();
  int64_t dim = c10::maybe_wrap_dim(dim_opt.value_or(0), self.dim());
  TORCH_CHECK(repeats >= 0, "Repeats must be non-negative");

  input = input.unsqueeze(dim + 1);
  auto expand_shape = input.sym_sizes().vec();
  expand_shape[dim + 1] = repeats;
  input = input.expand_symint(expand_shape);

  // This argument doesn't really make sense for the scalar overload, but exists
  // for consistency with the tensor overload
  if (output_size) {
    auto calculated_size = (repeats * expand_shape[dim]).guard_int(__FILE__, __LINE__);
    TORCH_CHECK(*output_size == calculated_size, "repeat_interleave: Invalid output_size, expected ",
                calculated_size, " but got ", *output_size);
  }

  return input.clone(at::MemoryFormat::Contiguous).flatten(dim, dim + 1);
}

} // namespace at::native

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 7 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `at`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `aten/src/ATen/native`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



## Dependencies

### Import Dependencies

This file includes:

- `ATen/core/Tensor.h`
- `ATen/Dispatch.h`
- `ATen/Parallel.h`
- `ATen/native/Repeat.h`
- `c10/util/irange.h`
- `ATen/Functions.h`
- `ATen/NativeFunctions.h`
- `ATen/ops/empty.h`
- `ATen/ops/repeat_interleave.h`
- `ATen/ops/repeat_interleave_native.h`


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

Files in the same folder (`aten/src/ATen/native`):

- [`LossMulti.h_docs.md`](./LossMulti.h_docs.md)
- [`NaiveConvolutionTranspose3d.cpp_docs.md`](./NaiveConvolutionTranspose3d.cpp_docs.md)
- [`UnaryOps.cpp_docs.md`](./UnaryOps.cpp_docs.md)
- [`ResizeCommon.h_docs.md`](./ResizeCommon.h_docs.md)
- [`FusedAdagrad.cpp_docs.md`](./FusedAdagrad.cpp_docs.md)
- [`SharedReduceOps.h_docs.md`](./SharedReduceOps.h_docs.md)
- [`SpectralOpsUtils.h_docs.md`](./SpectralOpsUtils.h_docs.md)
- [`FractionalMaxPooling.h_docs.md`](./FractionalMaxPooling.h_docs.md)
- [`TensorDimApply.h_docs.md`](./TensorDimApply.h_docs.md)
- [`Lerp.cpp_docs.md`](./Lerp.cpp_docs.md)


## Cross-References

- **File Documentation**: `Repeat.cpp_docs.md`
- **Keyword Index**: `Repeat.cpp_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/aten/src/ATen/native`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/aten/src/ATen/native`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
- Contains **benchmarking** code or performance tests.

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

Files in the same folder (`docs/aten/src/ATen/native`):

- [`AdaptiveMaxPooling2d.cpp_docs.md_docs.md`](./AdaptiveMaxPooling2d.cpp_docs.md_docs.md)
- [`Distributions.cpp_docs.md_docs.md`](./Distributions.cpp_docs.md_docs.md)
- [`im2col_shape_check.h_docs.md_docs.md`](./im2col_shape_check.h_docs.md_docs.md)
- [`ReduceOps.cpp_kw.md_docs.md`](./ReduceOps.cpp_kw.md_docs.md)
- [`Lerp.cpp_kw.md_docs.md`](./Lerp.cpp_kw.md_docs.md)
- [`CPUFallback.h_docs.md_docs.md`](./CPUFallback.h_docs.md_docs.md)
- [`MetaTensor.cpp_docs.md_docs.md`](./MetaTensor.cpp_docs.md_docs.md)
- [`Correlation.cpp_kw.md_docs.md`](./Correlation.cpp_kw.md_docs.md)
- [`im2col_shape_check.h_kw.md_docs.md`](./im2col_shape_check.h_kw.md_docs.md)
- [`UpSampleNearest2d.cpp_kw.md_docs.md`](./UpSampleNearest2d.cpp_kw.md_docs.md)


## Cross-References

- **File Documentation**: `Repeat.cpp_docs.md_docs.md`
- **Keyword Index**: `Repeat.cpp_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
