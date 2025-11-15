# Documentation: `docs/aten/src/ATen/native/Onehot.cpp_docs.md`

## File Metadata

- **Path**: `docs/aten/src/ATen/native/Onehot.cpp_docs.md`
- **Size**: 5,566 bytes (5.44 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `aten/src/ATen/native/Onehot.cpp`

## File Metadata

- **Path**: `aten/src/ATen/native/Onehot.cpp`
- **Size**: 2,895 bytes (2.83 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/DTensorState.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/arange.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/eq.h>
#include <ATen/ops/one_hot_native.h>
#include <ATen/ops/zeros.h>
#endif

namespace at::native {

Tensor one_hot(const Tensor &self, int64_t num_classes) {
    TORCH_CHECK(self.dtype() == kLong, "one_hot is only applicable to index tensor of type LongTensor.");

    // using meta bit test to catch Fake Tensor as well until __torch_function__
    if (self.key_set().has_all(DispatchKeySet(BackendComponent::MetaBit)) ||
            self.key_set().has_all(DispatchKeySet(DispatchKey::Python))) {
        // functional version that torch.compiles better and works with dynamic shapes
        if (num_classes == -1) {
          num_classes = self.max().item().toLong() + 1;
        }
        {
          // If `self` is a DTensor, then allow implicit replication
          // of the `index` Tensor.
          at::DTensorAllowImplicitReplication guard;
          at::Tensor index = at::arange(num_classes, self.options());
          return at::eq(self.unsqueeze(-1), index).to(kLong);
        }
    }

    auto shape = self.sym_sizes().vec();

    // empty tensor could be converted to one hot representation,
    // but shape inference is not possible.
    if (self.sym_numel() == 0) {
        if (num_classes <= 0) {
            TORCH_CHECK(false, "Can not infer total number of classes from empty tensor.");
        } else {
            shape.emplace_back(num_classes);
            return at::empty_symint(shape, self.options());
        }
    }

    // non-empty tensor
    if (self.device().type() != at::kCUDA && self.device().type() != at::kMPS &&
        self.device().type() != at::kPrivateUse1 && self.device().type() != at::kXLA) {
      // for cuda, rely on device assert thrown by scatter
      TORCH_CHECK(self.min().item().toLong() >= 0, "Class values must be non-negative.");
    }
    if (num_classes == -1) {
        num_classes = self.max().item().toLong() + 1;
    } else {
        if (self.device().type() != at::kCUDA && self.device().type() != at::kMPS &&
            self.device().type() != at::kPrivateUse1 && self.device().type() != at::kXLA) {
          // rely on device asserts from scatter to avoid sync here
          TORCH_CHECK(num_classes > self.max().item().toLong(), "Class values must be smaller than num_classes.");
        } else {
            //for cuda, assert that num_classes is at least 1
            TORCH_CHECK(num_classes >= 1, "num_classes should be positive");
        }
    }

    shape.emplace_back(num_classes);
    Tensor ret = at::zeros_symint(shape, self.options());
    ret.scatter_(-1, self.unsqueeze(-1), 1);
    return ret;
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
- `ATen/DTensorState.h`
- `ATen/Functions.h`
- `ATen/NativeFunctions.h`
- `ATen/ops/arange.h`
- `ATen/ops/empty.h`
- `ATen/ops/eq.h`
- `ATen/ops/one_hot_native.h`
- `ATen/ops/zeros.h`


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
- May involve **JIT compilation** or compilation optimizations.

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

- **File Documentation**: `Onehot.cpp_docs.md`
- **Keyword Index**: `Onehot.cpp_kw.md`
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
- May involve **JIT compilation** or compilation optimizations.
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

- **File Documentation**: `Onehot.cpp_docs.md_docs.md`
- **Keyword Index**: `Onehot.cpp_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
