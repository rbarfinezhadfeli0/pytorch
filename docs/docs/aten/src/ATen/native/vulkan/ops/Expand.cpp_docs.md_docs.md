# Documentation: `docs/aten/src/ATen/native/vulkan/ops/Expand.cpp_docs.md`

## File Metadata

- **Path**: `docs/aten/src/ATen/native/vulkan/ops/Expand.cpp_docs.md`
- **Size**: 4,824 bytes (4.71 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `aten/src/ATen/native/vulkan/ops/Expand.cpp`

## File Metadata

- **Path**: `aten/src/ATen/native/vulkan/ops/Expand.cpp`
- **Size**: 2,378 bytes (2.32 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#include <ATen/native/vulkan/ops/Common.h>
#include <ATen/native/vulkan/ops/Utils.h>
#include <torch/library.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#else
#include <ATen/ops/repeat.h>
#endif

#include <ATen/native/vulkan/ops/Utils.h>

namespace at {
namespace native {
namespace vulkan {
namespace ops {
namespace {

using namespace api::utils;

Tensor expand(
    const at::Tensor& self,
    const IntArrayRef output_size,
    bool implicit = false) {
  TORCH_CHECK(
      self.dim() > 0 && self.dim() <= 4,
      "Vulkan expand supports up to 4d tensors");
  TORCH_CHECK(
      static_cast<size_t>(self.dim()) <= output_size.size(),
      "Vulkan expand: the number of sizes provided (",
      output_size.size(),
      ") must be greater or equal to the number of dimensions in the tensor (",
      self.dim(),
      ").");

  std::vector<int64_t> repeat_size = std::vector<int64_t>(output_size.size());
  std::vector<int64_t> input_size = self.sizes().vec();

  int in_idx = input_size.size() - 1;
  for (int i = output_size.size() - 1; i >= 0; --i) {
    if (in_idx >= 0) {
      TORCH_CHECK(
          input_size[in_idx] == output_size[i] || input_size[in_idx] == 1 ||
              output_size[i] == -1,
          "Vulkan expand: the expanded size of the tensor (",
          output_size[i],
          ") must match the existing size (",
          input_size[in_idx],
          ") at non-singleton dimension ",
          i);

      if (input_size[in_idx] == output_size[i] || output_size[i] == -1) {
        repeat_size[i] = 1;
      } else if (input_size[in_idx] == 1) {
        repeat_size[i] = output_size[i];
      }
      --in_idx;
    } else {
      TORCH_CHECK(
          output_size[i] != -1,
          "Vulkan expand: the expanded size of the tensor (-1) is not allowed in a leading, non-existing dimension 0.");

      repeat_size[i] = output_size[i];
    }
  }

  return self.repeat(repeat_size);
}

Tensor expand_as(const at::Tensor& self, const at::Tensor& other) {
  return expand(self, other.sizes());
}

#ifdef USE_VULKAN_API

TORCH_LIBRARY_IMPL(aten, Vulkan, m) {
  m.impl(TORCH_SELECTIVE_NAME("aten::expand"), TORCH_FN(expand));
  m.impl(TORCH_SELECTIVE_NAME("aten::expand_as"), TORCH_FN(expand_as));
}

#endif /* USE_VULKAN_API */

} // namespace
} // namespace ops
} // namespace vulkan
} // namespace native
} // namespace at

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 10 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `vulkan`, `ops`, `api`, `native`, `at`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `aten/src/ATen/native/vulkan/ops`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



## Dependencies

### Import Dependencies

This file includes:

- `ATen/native/vulkan/ops/Common.h`
- `ATen/native/vulkan/ops/Utils.h`
- `torch/library.h`
- `ATen/Functions.h`
- `ATen/ops/repeat.h`
- `ATen/native/vulkan/ops/Utils.h`


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes


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

Files in the same folder (`aten/src/ATen/native/vulkan/ops`):

- [`Convert.h_docs.md`](./Convert.h_docs.md)
- [`Batchnorm.cpp_docs.md`](./Batchnorm.cpp_docs.md)
- [`Slice.cpp_docs.md`](./Slice.cpp_docs.md)
- [`Lerp.cpp_docs.md`](./Lerp.cpp_docs.md)
- [`Shape.cpp_docs.md`](./Shape.cpp_docs.md)
- [`Mean.cpp_docs.md`](./Mean.cpp_docs.md)
- [`UnaryOp.cpp_docs.md`](./UnaryOp.cpp_docs.md)
- [`Permute.cpp_docs.md`](./Permute.cpp_docs.md)
- [`Unsqueeze.cpp_docs.md`](./Unsqueeze.cpp_docs.md)
- [`Stack.cpp_docs.md`](./Stack.cpp_docs.md)


## Cross-References

- **File Documentation**: `Expand.cpp_docs.md`
- **Keyword Index**: `Expand.cpp_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/aten/src/ATen/native/vulkan/ops`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/aten/src/ATen/native/vulkan/ops`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

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

Files in the same folder (`docs/aten/src/ATen/native/vulkan/ops`):

- [`Lerp.cpp_kw.md_docs.md`](./Lerp.cpp_kw.md_docs.md)
- [`Select.cpp_docs.md_docs.md`](./Select.cpp_docs.md_docs.md)
- [`Batchnorm.h_docs.md_docs.md`](./Batchnorm.h_docs.md_docs.md)
- [`Lstm.cpp_kw.md_docs.md`](./Lstm.cpp_kw.md_docs.md)
- [`Concat.cpp_kw.md_docs.md`](./Concat.cpp_kw.md_docs.md)
- [`Convolution.cpp_docs.md_docs.md`](./Convolution.cpp_docs.md_docs.md)
- [`Zero.cpp_kw.md_docs.md`](./Zero.cpp_kw.md_docs.md)
- [`Gru.h_kw.md_docs.md`](./Gru.h_kw.md_docs.md)
- [`Repeat.cpp_kw.md_docs.md`](./Repeat.cpp_kw.md_docs.md)
- [`Register.cpp_docs.md_docs.md`](./Register.cpp_docs.md_docs.md)


## Cross-References

- **File Documentation**: `Expand.cpp_docs.md_docs.md`
- **Keyword Index**: `Expand.cpp_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
