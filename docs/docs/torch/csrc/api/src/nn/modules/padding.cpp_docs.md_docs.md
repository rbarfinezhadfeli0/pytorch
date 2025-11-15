# Documentation: `docs/torch/csrc/api/src/nn/modules/padding.cpp_docs.md`

## File Metadata

- **Path**: `docs/torch/csrc/api/src/nn/modules/padding.cpp_docs.md`
- **Size**: 6,181 bytes (6.04 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/csrc/api/src/nn/modules/padding.cpp`

## File Metadata

- **Path**: `torch/csrc/api/src/nn/modules/padding.cpp`
- **Size**: 3,683 bytes (3.60 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#include <torch/nn/modules/padding.h>

#include <torch/expanding_array.h>

namespace F = torch::nn::functional;

namespace torch::nn {

template <size_t D, typename Derived>
ReflectionPadImpl<D, Derived>::ReflectionPadImpl(
    const ReflectionPadOptions<D>& options_)
    : options(options_) {}

template <size_t D, typename Derived>
void ReflectionPadImpl<D, Derived>::reset() {}

template <size_t D, typename Derived>
Tensor ReflectionPadImpl<D, Derived>::forward(const Tensor& input) {
  return F::detail::pad(input, options.padding(), torch::kReflect, 0);
}

template <size_t D, typename Derived>
void ReflectionPadImpl<D, Derived>::pretty_print(std::ostream& stream) const {
  stream << "torch::nn::ReflectionPad" << D << "d"
         << "(padding=" << options.padding() << ")";
}

template class ReflectionPadImpl<1, ReflectionPad1dImpl>;
template class ReflectionPadImpl<2, ReflectionPad2dImpl>;
template class ReflectionPadImpl<3, ReflectionPad3dImpl>;

// ============================================================================

template <size_t D, typename Derived>
ReplicationPadImpl<D, Derived>::ReplicationPadImpl(
    const ReplicationPadOptions<D>& options_)
    : options(options_) {}

template <size_t D, typename Derived>
void ReplicationPadImpl<D, Derived>::reset() {}

template <size_t D, typename Derived>
Tensor ReplicationPadImpl<D, Derived>::forward(const Tensor& input) {
  return F::detail::pad(input, options.padding(), torch::kReplicate, 0);
}

template <size_t D, typename Derived>
void ReplicationPadImpl<D, Derived>::pretty_print(std::ostream& stream) const {
  stream << "torch::nn::ReplicationPad" << D << "d"
         << "(padding=" << options.padding() << ")";
}

template class ReplicationPadImpl<1, ReplicationPad1dImpl>;
template class ReplicationPadImpl<2, ReplicationPad2dImpl>;
template class ReplicationPadImpl<3, ReplicationPad3dImpl>;

// ============================================================================

template <size_t D, typename Derived>
ZeroPadImpl<D, Derived>::ZeroPadImpl(const ZeroPadOptions<D>& options_)
    : options(options_) {}

template <size_t D, typename Derived>
void ZeroPadImpl<D, Derived>::reset() {}

template <size_t D, typename Derived>
Tensor ZeroPadImpl<D, Derived>::forward(const Tensor& input) {
  return F::detail::pad(input, options.padding(), torch::kConstant, 0);
}

template <size_t D, typename Derived>
void ZeroPadImpl<D, Derived>::pretty_print(std::ostream& stream) const {
  stream << "torch::nn::ZeroPad" << D << "d"
         << "(padding=" << options.padding() << ")";
}

template class ZeroPadImpl<1, ZeroPad1dImpl>;
template class ZeroPadImpl<2, ZeroPad2dImpl>;
template class ZeroPadImpl<3, ZeroPad3dImpl>;

// ============================================================================

template <size_t D, typename Derived>
ConstantPadImpl<D, Derived>::ConstantPadImpl(
    const ConstantPadOptions<D>& options_)
    : options(options_) {}

template <size_t D, typename Derived>
void ConstantPadImpl<D, Derived>::reset() {}

template <size_t D, typename Derived>
Tensor ConstantPadImpl<D, Derived>::forward(const Tensor& input) {
  return F::detail::pad(
      input, options.padding(), torch::kConstant, options.value());
}

template <size_t D, typename Derived>
void ConstantPadImpl<D, Derived>::pretty_print(std::ostream& stream) const {
  stream << "torch::nn::ConstantPad" << D << "d"
         << "(padding=" << options.padding() << ", value=" << options.value()
         << ")";
}

template class ConstantPadImpl<1, ConstantPad1dImpl>;
template class ConstantPadImpl<2, ConstantPad2dImpl>;
template class ConstantPadImpl<3, ConstantPad3dImpl>;

} // namespace torch::nn

```



## High-Level Overview


This C++ file contains approximately 12 class(es)/struct(s) and 0 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `torch`, `F`

**Classes/Structs**: `ReflectionPadImpl`, `ReflectionPadImpl`, `ReflectionPadImpl`, `ReplicationPadImpl`, `ReplicationPadImpl`, `ReplicationPadImpl`, `ZeroPadImpl`, `ZeroPadImpl`, `ZeroPadImpl`, `ConstantPadImpl`, `ConstantPadImpl`, `ConstantPadImpl`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/csrc/api/src/nn/modules`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `torch/nn/modules/padding.h`
- `torch/expanding_array.h`


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

Files in the same folder (`torch/csrc/api/src/nn/modules`):

- [`pooling.cpp_docs.md`](./pooling.cpp_docs.md)
- [`linear.cpp_docs.md`](./linear.cpp_docs.md)
- [`rnn.cpp_docs.md`](./rnn.cpp_docs.md)
- [`upsampling.cpp_docs.md`](./upsampling.cpp_docs.md)
- [`embedding.cpp_docs.md`](./embedding.cpp_docs.md)
- [`dropout.cpp_docs.md`](./dropout.cpp_docs.md)
- [`pixelshuffle.cpp_docs.md`](./pixelshuffle.cpp_docs.md)
- [`loss.cpp_docs.md`](./loss.cpp_docs.md)
- [`fold.cpp_docs.md`](./fold.cpp_docs.md)


## Cross-References

- **File Documentation**: `padding.cpp_docs.md`
- **Keyword Index**: `padding.cpp_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/csrc/api/src/nn/modules`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/csrc/api/src/nn/modules`, which is part of the **core PyTorch library**.



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

Files in the same folder (`docs/torch/csrc/api/src/nn/modules`):

- [`dropout.cpp_docs.md_docs.md`](./dropout.cpp_docs.md_docs.md)
- [`dropout.cpp_kw.md_docs.md`](./dropout.cpp_kw.md_docs.md)
- [`pooling.cpp_kw.md_docs.md`](./pooling.cpp_kw.md_docs.md)
- [`rnn.cpp_docs.md_docs.md`](./rnn.cpp_docs.md_docs.md)
- [`linear.cpp_docs.md_docs.md`](./linear.cpp_docs.md_docs.md)
- [`normalization.cpp_kw.md_docs.md`](./normalization.cpp_kw.md_docs.md)
- [`_functions.cpp_kw.md_docs.md`](./_functions.cpp_kw.md_docs.md)
- [`upsampling.cpp_docs.md_docs.md`](./upsampling.cpp_docs.md_docs.md)
- [`pixelshuffle.cpp_docs.md_docs.md`](./pixelshuffle.cpp_docs.md_docs.md)
- [`adaptive.cpp_kw.md_docs.md`](./adaptive.cpp_kw.md_docs.md)


## Cross-References

- **File Documentation**: `padding.cpp_docs.md_docs.md`
- **Keyword Index**: `padding.cpp_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
