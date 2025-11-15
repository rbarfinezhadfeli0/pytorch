# Documentation: `torch/csrc/lazy/core/shape.cpp`

## File Metadata

- **Path**: `torch/csrc/lazy/core/shape.cpp`
- **Size**: 4,021 bytes (3.93 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#include <c10/util/env.h>
#include <c10/util/irange.h>
#include <torch/csrc/lazy/core/shape.h>
#include <torch/csrc/lazy/core/tensor.h>

#include <utility>

C10_DEFINE_bool(
    ltc_enable_symbolic_shapes,
    false,
    "Enables calculation of if dims are symbolic")

namespace torch::lazy {

Shape::Shape(
    at::ScalarType scalar_type,
    c10::ArrayRef<int64_t> sizes,
    std::optional<std::vector<bool>> is_symbolic)
    : scalar_type_(scalar_type),
      sizes_(sizes.begin(), sizes.end()),
      is_symbolic_(std::move(is_symbolic)) {}

std::string Shape::to_string() const {
  return c10::str(toString(scalar_type_), "[", c10::Join(",", sizes_), "]");
}

bool Shape::operator==(const Shape& other) const {
  return scalar_type_ == other.scalar_type_ && sizes_ == other.sizes_;
}

std::ostream& operator<<(std::ostream& out, const Shape& shape) {
  return out << shape.to_string();
}

size_t Shape::numel() const {
  size_t elts = 1;
  for (auto size : sizes_) {
    elts *= size;
  }
  return elts;
}

hash_t Shape::hash(bool bakeInSizes) const {
  if (bakeInSizes) {
    return HashCombine(
        Hash(scalar_type_),
        DataHash(sizes_.data(), sizes_.size() * sizeof(int64_t)));
  } else {
    return HashCombine(Hash(scalar_type_), Hash(sizes_.size()));
  }
}

Shape Shape::with_symbolic_dims(
    std::optional<std::vector<bool>> symbolic_dims) const {
  Shape copy = *this;
  copy.is_symbolic_ = std::move(symbolic_dims);
  return copy;
}

bool symbolicShapeEnabled() {
  static bool enabled = c10::utils::has_env("LTC_ENABLE_SYMBOLIC_SHAPES");
  return enabled || FLAGS_ltc_enable_symbolic_shapes;
}

static c10::SymbolicShape get_symbolic_shape(at::Tensor& tensor) {
  auto ltc_tensor = TryGetLtcTensor(tensor);
  if (!ltc_tensor) {
    // Set Concrete sizes for Concrete tensors
    return c10::SymbolicShape(tensor.sizes());
  }
  const Shape& input_shape = ltc_tensor->GetIrValue()->shape();
  auto& is_symbolic = input_shape.is_symbolic();
  if (!is_symbolic.has_value()) {
    return c10::SymbolicShape();
  }
  auto sizes = input_shape.sizes();
  TORCH_INTERNAL_ASSERT(
      sizes.size() == is_symbolic->size(),
      "Dims of two values are not consistent");
  std::vector<std::optional<int64_t>> symbolic_dims;
  for (size_t i = 0; i < sizes.size(); i++) {
    if (is_symbolic->at(i)) {
      symbolic_dims.emplace_back(std::nullopt);
    } else {
      symbolic_dims.emplace_back(sizes.at(i));
    }
  }
  return c10::SymbolicShape(symbolic_dims);
}

void applySymbolicShapesOnLT(
    const char* schema_str,
    std::vector<c10::IValue> args,
    std::vector<Shape>& result_shapes) {
  std::vector<jit::SSAInput> converted_args;
  // TODO: Determine if there are any unknown values in LazyTensor
  const c10::FunctionSchema& schema =
      jit::getOperatorForLiteral(schema_str)->schema();

  for (auto& arg : args) {
    // Handle list of tensors
    if (arg.isTensorList()) {
      at::List<at::Tensor> tensor_list = arg.toTensorList();
      for (at::Tensor tensor : tensor_list) {
        converted_args.emplace_back(get_symbolic_shape(tensor));
      }
    } else if (arg.isTensor()) {
      auto ss = get_symbolic_shape(arg.toTensor());
      converted_args.emplace_back(ss);
    } else {
      // If we need to support symbolic ints, here is the place
      // to add it.
      converted_args.emplace_back(arg);
    }
  }
  auto res_symbolic = jit::calculateSymbolicShapesOnOp(&schema, converted_args);
  if (!res_symbolic) {
    for (auto& result_shape : result_shapes) {
      result_shape = result_shape.with_symbolic_dims(std::nullopt);
    }
  } else {
    TORCH_INTERNAL_ASSERT(
        res_symbolic->size() == result_shapes.size(),
        "Result shape size is not consistent");
    for (size_t i = 0; i < res_symbolic->size(); i++) {
      auto sym_dims = res_symbolic->at(i).symbolicDims();
      if (sym_dims.has_value()) {
        result_shapes[i] =
            result_shapes[i].with_symbolic_dims(std::move(sym_dims));
      }
    }
  }
}

} // namespace torch::lazy

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 7 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `torch`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/csrc/lazy/core`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `c10/util/env.h`
- `c10/util/irange.h`
- `torch/csrc/lazy/core/shape.h`
- `torch/csrc/lazy/core/tensor.h`
- `utility`


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

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

Files in the same folder (`torch/csrc/lazy/core`):

- [`hash.cpp_docs.md`](./hash.cpp_docs.md)
- [`shape_inference.cpp_docs.md`](./shape_inference.cpp_docs.md)
- [`tensor_impl.h_docs.md`](./tensor_impl.h_docs.md)
- [`helpers.h_docs.md`](./helpers.h_docs.md)
- [`tensor_impl.cpp_docs.md`](./tensor_impl.cpp_docs.md)
- [`ir_metadata.cpp_docs.md`](./ir_metadata.cpp_docs.md)
- [`ir_metadata.h_docs.md`](./ir_metadata.h_docs.md)
- [`trie.cpp_docs.md`](./trie.cpp_docs.md)
- [`cache.h_docs.md`](./cache.h_docs.md)
- [`config.cpp_docs.md`](./config.cpp_docs.md)


## Cross-References

- **File Documentation**: `shape.cpp_docs.md`
- **Keyword Index**: `shape.cpp_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
