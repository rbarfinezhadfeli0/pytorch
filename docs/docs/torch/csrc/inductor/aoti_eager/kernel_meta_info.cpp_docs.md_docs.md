# Documentation: `docs/torch/csrc/inductor/aoti_eager/kernel_meta_info.cpp_docs.md`

## File Metadata

- **Path**: `docs/torch/csrc/inductor/aoti_eager/kernel_meta_info.cpp_docs.md`
- **Size**: 9,062 bytes (8.85 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/csrc/inductor/aoti_eager/kernel_meta_info.cpp`

## File Metadata

- **Path**: `torch/csrc/inductor/aoti_eager/kernel_meta_info.cpp`
- **Size**: 7,020 bytes (6.86 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#if !defined(C10_MOBILE) && !defined(ANDROID)
#include <torch/csrc/inductor/aoti_eager/kernel_meta_info.h>
#include <iostream>
#include <utility>

namespace torch::inductor {

TensorMetadata::TensorMetadata(const at::Tensor& src_tensor)
    : is_symbolic_(false),
      dtype_(src_tensor.scalar_type()),
      device_(src_tensor.device()),
      dispatch_key_set_(src_tensor.key_set()),
      sizes_(src_tensor.sizes().vec()),
      strides_(src_tensor.strides().vec()),
      requires_grad_(src_tensor.requires_grad()) {}

TensorMetadata::TensorMetadata(
    bool is_symbolic,
    c10::ScalarType dtype,
    c10::Device device,
    c10::DispatchKeySet dispatch_key_set,
    std::vector<int64_t> sizes,
    std::vector<int64_t> strides,
    bool requires_grad)
    : is_symbolic_(is_symbolic),
      dtype_(dtype),
      device_(device),
      dispatch_key_set_(dispatch_key_set),
      sizes_(std::move(sizes)),
      strides_(std::move(strides)),
      requires_grad_(requires_grad) {
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
      !is_symbolic_, "Not support symbolic shape now");
}

void TensorMetadata::build_guard(const torch::dynamo::LocalState& local_state) {
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
      !is_symbolic_, "Not support symbolic shape now");
  std::vector<std::optional<c10::SymInt>> sym_sizes;
  std::vector<std::optional<c10::SymInt>> sym_strides;
  std::transform(
      sizes_.begin(),
      sizes_.end(),
      std::back_inserter(sym_sizes),
      [](int64_t size) { return std::optional<c10::SymInt>(size); });
  std::transform(
      strides_.begin(),
      strides_.end(),
      std::back_inserter(sym_strides),
      [](int64_t stride) { return std::optional<c10::SymInt>(stride); });
  tensor_check_ = torch::dynamo::TensorCheck(
      local_state,
      nullptr,
      dispatch_key_set_,
      dtype_,
      device_.index(),
      requires_grad_,
      sym_sizes,
      sym_strides);
}

bool TensorMetadata::operator==(const TensorMetadata& other) const {
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
      !is_symbolic_, "Not support symbolic shape now");

  if (tensor_check_.has_value()) {
    auto sizes_ = c10::IntArrayRef(other.sizes_);
    auto strides_ = c10::IntArrayRef(other.strides_);
    auto sym_sizes = c10::SymIntArrayRef(
        reinterpret_cast<const c10::SymInt*>(sizes_.data()), sizes_.size());
    auto sym_strides = c10::SymIntArrayRef(
        reinterpret_cast<const c10::SymInt*>(strides_.data()), strides_.size());

    torch::dynamo::LocalState local_state;
    local_state.overrideDispatchKeySet(dispatch_key_set_);
    auto _tensor_check = tensor_check_.value();
    auto res = _tensor_check.check(
        local_state,
        other.dispatch_key_set_,
        other.dtype_,
        other.device_,
        sym_sizes,
        sym_strides,
        other.requires_grad_ /* Should we need to care about grad requirement?*/);
    return res;
  } else {
    return this->is_symbolic_ == other.is_symbolic_ &&
        this->dtype_ == other.dtype_ && this->device_ == other.device_ &&
        this->dispatch_key_set_ == other.dispatch_key_set_ &&
        this->requires_grad_ == other.requires_grad_ &&
        this->sizes_ == other.sizes_ && this->strides_ == other.strides_;
  }
}

std::ostream& operator<<(
    std::ostream& stream,
    const TensorMetadata& tensor_metadata) {
  stream << "is_symbolic_: " << tensor_metadata.is_symbolic_ << '\n';
  stream << "dtype_: " << tensor_metadata.dtype_ << '\n';
  stream << "device_: " << tensor_metadata.device_ << '\n';
  stream << "sizes_: ";
  for (const auto& size : tensor_metadata.sizes_) {
    stream << size << " ";
  }
  stream << '\n';
  stream << "strides_: ";
  for (const auto& stride : tensor_metadata.strides_) {
    stream << stride << " ";
  }

  stream << "requires_grad_: " << tensor_metadata.requires_grad_ << '\n';
  stream << "dispatch_key_set_: " << tensor_metadata.dispatch_key_set_ << '\n';
  stream << "tensor_check_: " << tensor_metadata.tensor_check_.has_value()
         << '\n';
  stream << '\n';
  return stream;
}

ParameterMetadata::ParameterMetadata(
    TensorMetadata tensor_metadata,
    uint64_t input_order)
    : tag_(TENSOR), value_(tensor_metadata), order_(input_order) {}

ParameterMetadata::ParameterMetadata(
    const at::Tensor& tensor,
    uint64_t input_order)
    : tag_(TENSOR), order_(input_order) {
  value_ = TensorMetadata(tensor);
}

ParameterMetadata::ParameterMetadata(
    const std::vector<TensorMetadata>& tensor_metadata_list,
    uint64_t input_order)
    : tag_(TENSOR_LIST), value_(tensor_metadata_list), order_(input_order) {}

ParameterMetadata::ParameterMetadata(
    const std::vector<at::Tensor>& tensor_list,
    uint64_t input_order)
    : tag_(TENSOR_LIST), order_(input_order) {
  std::vector<TensorMetadata> tensor_metadata_list;
  tensor_metadata_list.reserve(tensor_list.size());
  for (const auto& tensor : tensor_list) {
    tensor_metadata_list.emplace_back(tensor);
  }
  value_ = tensor_metadata_list;
}

ParameterMetadata::ParameterMetadata(
    const c10::Scalar& scalar,
    uint64_t input_order)
    : tag_(SCALAR), value_(scalar), order_(input_order) {}

ParameterMetadata::ParameterMetadata(
    const std::string& str,
    uint64_t input_order)
    : tag_(STRING), value_(str), order_(input_order) {}

ParameterMetadata::ParameterMetadata(
    const c10::Device& device,
    uint64_t input_order)
    : tag_(DEVICE), value_(device), order_(input_order) {}

bool ParameterMetadata::operator==(const ParameterMetadata& other) const {
  // Same type
  if (tag_ != other.tag_) {
    return false;
  }

  // Same order of the input parameters
  if (order_ != other.order_) {
    return false;
  }

  switch (tag_) {
    case TENSOR:
      return std::get<TensorMetadata>(value_) ==
          std::get<TensorMetadata>(other.value_);
    case TENSOR_LIST:
      return std::get<std::vector<TensorMetadata>>(value_) ==
          std::get<std::vector<TensorMetadata>>(other.value_);
    case SCALAR:
      TORCH_INTERNAL_ASSERT(
          std::get<c10::Scalar>(other.value_).isFloatingPoint() ||
          std::get<c10::Scalar>(other.value_).isIntegral(true /*includeBool*/));
      return equal_to(std::get<c10::Scalar>(other.value_));
    case STRING:
      return std::get<std::string>(value_) ==
          std::get<std::string>(other.value_);
    case DEVICE:
      return std::get<c10::Device>(value_) ==
          std::get<c10::Device>(other.value_);
    default:
      return false;
  }
}

bool ParameterMetadata::equal_to(const c10::Scalar& scalar) const {
  TORCH_INTERNAL_ASSERT(scalar.isFloatingPoint() || scalar.isIntegral(true));
  if (tag_ != SCALAR) {
    return false;
  }

  const auto& self_scalar = std::get<c10::Scalar>(value_);
  if (scalar.isFloatingPoint() && self_scalar.isFloatingPoint()) {
    return self_scalar.toDouble() == scalar.toDouble();
  } else if (scalar.isIntegral(true) && self_scalar.isIntegral(true)) {
    return self_scalar.toInt() == scalar.toInt();
  }

  return false;
}

} // namespace torch::inductor
#endif

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 4 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `torch`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/csrc/inductor/aoti_eager`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `torch/csrc/inductor/aoti_eager/kernel_meta_info.h`
- `iostream`
- `utility`


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

Files in the same folder (`torch/csrc/inductor/aoti_eager`):

- [`kernel_holder.cpp_docs.md`](./kernel_holder.cpp_docs.md)
- [`kernel_meta_info.h_docs.md`](./kernel_meta_info.h_docs.md)
- [`kernel_holder.h_docs.md`](./kernel_holder.h_docs.md)


## Cross-References

- **File Documentation**: `kernel_meta_info.cpp_docs.md`
- **Keyword Index**: `kernel_meta_info.cpp_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/csrc/inductor/aoti_eager`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/csrc/inductor/aoti_eager`, which is part of the **core PyTorch library**.



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

Files in the same folder (`docs/torch/csrc/inductor/aoti_eager`):

- [`kernel_meta_info.cpp_kw.md_docs.md`](./kernel_meta_info.cpp_kw.md_docs.md)
- [`kernel_holder.h_docs.md_docs.md`](./kernel_holder.h_docs.md_docs.md)
- [`kernel_holder.cpp_docs.md_docs.md`](./kernel_holder.cpp_docs.md_docs.md)
- [`kernel_meta_info.h_kw.md_docs.md`](./kernel_meta_info.h_kw.md_docs.md)
- [`kernel_holder.h_kw.md_docs.md`](./kernel_holder.h_kw.md_docs.md)
- [`kernel_holder.cpp_kw.md_docs.md`](./kernel_holder.cpp_kw.md_docs.md)
- [`kernel_meta_info.h_docs.md_docs.md`](./kernel_meta_info.h_docs.md_docs.md)


## Cross-References

- **File Documentation**: `kernel_meta_info.cpp_docs.md_docs.md`
- **Keyword Index**: `kernel_meta_info.cpp_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
