# Documentation: `docs/aten/src/ATen/native/xnnpack/Common.h_docs.md`

## File Metadata

- **Path**: `docs/aten/src/ATen/native/xnnpack/Common.h_docs.md`
- **Size**: 5,990 bytes (5.85 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `aten/src/ATen/native/xnnpack/Common.h`

## File Metadata

- **Path**: `aten/src/ATen/native/xnnpack/Common.h`
- **Size**: 3,434 bytes (3.35 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
#pragma once

#ifdef USE_XNNPACK

#include <xnnpack.h>
#include <caffe2/utils/threadpool/pthreadpool-cpp.h>
#include <c10/util/ArrayRef.h>
#include <limits>
#include <memory>

namespace at::native::xnnpack {

struct Deleter final {
  void operator()(const xnn_operator_t op) const {
    xnn_delete_operator(op);
  }
};

using Operator = std::unique_ptr<xnn_operator, Deleter>;

struct ContextLinear final {
  Operator op;
  int64_t output_channels;

  ContextLinear() = delete;

  ContextLinear(Operator&& o, int64_t o_channels) : op(std::move(o)), output_channels(o_channels) {}
  static constexpr float kMin = -std::numeric_limits<float>::infinity();
  static constexpr float kMax = std::numeric_limits<float>::infinity();
};

// This contains information for both the transpose and non-transpose cases.
struct ContextConv2D final {
  Operator op;
  std::array<int64_t, 4> weight_size_;
  std::array<int64_t, 2> padding_;
  std::array<int64_t, 2> output_padding_;
  std::array<int64_t, 2> stride_;
  std::array<int64_t, 2> dilation_;
  bool transposed_;
  int64_t groups_;

  ContextConv2D() = delete;

  ContextConv2D(
      Operator&& o,
      std::array<int64_t, 4> weight_size,
      std::array<int64_t, 2> padding,
      std::array<int64_t, 2> output_padding,
      std::array<int64_t, 2> stride,
      std::array<int64_t, 2> dilation,
      bool transposed,
      int64_t groups)
      :  op(std::move(o)),
         weight_size_(weight_size),
         padding_(padding),
         output_padding_(output_padding),
         stride_(stride),
         dilation_(dilation),
         transposed_(transposed),
         groups_(groups) {}
  static constexpr float kMin = -std::numeric_limits<float>::infinity();
  static constexpr float kMax = std::numeric_limits<float>::infinity();
};


namespace internal {

struct Layout final {
  // 4D Activation Maps
  struct Activation4D final {
    static constexpr size_t batch = 0u;
    static constexpr size_t channels = 1u;
    static constexpr size_t height = 2u;
    static constexpr size_t width = 3u;
  };

  // ND Activation Maps
  struct ActivationND final {
    // Some operators may not be limited to 4 dimensional tensors. In that scenario,
    // XNNPACK denotes that operator with an _nc suffix and expects all dimensions,
    // except channels, to be flattened into one argument: batch_size.
    static int64_t batch(const IntArrayRef tensor) {
      if (C10_UNLIKELY(tensor.empty())) {
        return -1;
      }

      // Handle the case where batch size is zero.
      int64_t batch = tensor[0];

      for (size_t index = 1u; index < (tensor.size() - 1u); ++index) {
        batch *= tensor[index];
      }

      return batch;
    }

    static int64_t channel(const IntArrayRef tensor) {
      if (C10_UNLIKELY(tensor.empty())) {
        return -1;
      }

      return tensor.back();
    }
  };

  // Convolution Filters
  struct Filter final {
    static constexpr size_t output = 0u;
    static constexpr size_t input = 1u;
    static constexpr size_t height = 2u;
    static constexpr size_t width = 3u;
  };

  // Parameters (Pooling Kernels, Dilation, Padding, Stride, etc.)
  struct Parameter final {
    static constexpr size_t height = 0u;
    static constexpr size_t width = 1u;
  };
};
} // namespace internal
} // namespace at::native::xnnpack

#endif /* USE_XNNPACK */

namespace at::native::xnnpack {
bool available();
} // namespace at::native::xnnpack

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 4 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `internal`, `at`

**Classes/Structs**: `Deleter`, `ContextLinear`, `ContextConv2D`, `Layout`, `Activation4D`, `ActivationND`, `Filter`, `Parameter`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `aten/src/ATen/native/xnnpack`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



## Dependencies

### Import Dependencies

This file includes:

- `xnnpack.h`
- `caffe2/utils/threadpool/pthreadpool-cpp.h`
- `c10/util/ArrayRef.h`
- `limits`
- `memory`


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

Files in the same folder (`aten/src/ATen/native/xnnpack`):

- [`Engine.h_docs.md`](./Engine.h_docs.md)
- [`Linear.cpp_docs.md`](./Linear.cpp_docs.md)
- [`ChannelShuffle.cpp_docs.md`](./ChannelShuffle.cpp_docs.md)
- [`Convolution.h_docs.md`](./Convolution.h_docs.md)
- [`RegisterOpContextClass.cpp_docs.md`](./RegisterOpContextClass.cpp_docs.md)
- [`Convolution.cpp_docs.md`](./Convolution.cpp_docs.md)
- [`Activation.cpp_docs.md`](./Activation.cpp_docs.md)
- [`Linear.h_docs.md`](./Linear.h_docs.md)
- [`Shim.cpp_docs.md`](./Shim.cpp_docs.md)


## Cross-References

- **File Documentation**: `Common.h_docs.md`
- **Keyword Index**: `Common.h_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/aten/src/ATen/native/xnnpack`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/aten/src/ATen/native/xnnpack`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



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

Files in the same folder (`docs/aten/src/ATen/native/xnnpack`):

- [`MaxPooling.cpp_docs.md_docs.md`](./MaxPooling.cpp_docs.md_docs.md)
- [`Convolution.cpp_docs.md_docs.md`](./Convolution.cpp_docs.md_docs.md)
- [`Common.h_kw.md_docs.md`](./Common.h_kw.md_docs.md)
- [`Pooling.h_docs.md_docs.md`](./Pooling.h_docs.md_docs.md)
- [`RegisterOpContextClass.cpp_kw.md_docs.md`](./RegisterOpContextClass.cpp_kw.md_docs.md)
- [`AveragePooling.cpp_kw.md_docs.md`](./AveragePooling.cpp_kw.md_docs.md)
- [`OpContext.cpp_kw.md_docs.md`](./OpContext.cpp_kw.md_docs.md)
- [`ChannelShuffle.cpp_docs.md_docs.md`](./ChannelShuffle.cpp_docs.md_docs.md)
- [`MaxPooling.cpp_kw.md_docs.md`](./MaxPooling.cpp_kw.md_docs.md)


## Cross-References

- **File Documentation**: `Common.h_docs.md_docs.md`
- **Keyword Index**: `Common.h_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
