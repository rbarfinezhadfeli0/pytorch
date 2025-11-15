# Documentation: `docs/torch/csrc/lazy/core/hash.h_docs.md`

## File Metadata

- **Path**: `docs/torch/csrc/lazy/core/hash.h_docs.md`
- **Size**: 9,983 bytes (9.75 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/csrc/lazy/core/hash.h`

## File Metadata

- **Path**: `torch/csrc/lazy/core/hash.h`
- **Size**: 7,599 bytes (7.42 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
/**
 * Hash utils in this file is adapted from PyTorch/XLA
 * https://github.com/pytorch/xla/blob/e0e5f937a0ba8d904f9608137dc8c51ba439df2d/third_party/xla_client/util.h
 */
#pragma once

#include <ATen/Tensor.h>
#include <c10/core/Scalar.h>
#include <c10/util/int128.h>
#include <torch/csrc/Export.h>
#include <cstring>
#include <set>
#include <string>
#include <string_view>
#include <vector>

namespace torch::lazy {

using size_t = std::size_t;

class TORCH_API hash_t : public c10::uint128 {
 public:
  // Switch from typedef hash_t = uint128 to provide explicit casters
  hash_t(int8_t val) : uint128(static_cast<uint32_t>(val)) {}
  hash_t(int16_t val) : uint128(static_cast<uint32_t>(val)) {}
  hash_t(int32_t val) : uint128(static_cast<uint32_t>(val)) {}
  hash_t(int64_t val) : uint128(static_cast<uint64_t>(val)) {}
  hash_t(uint32_t val) : uint128(val) {}
  hash_t(uint64_t val) : uint128(val) {}
  hash_t(uint128 val) : uint128(val) {}
  hash_t(uint64_t top, uint64_t bottom) : uint128(top, bottom) {}
  hash_t() = default;
};

// Std* functions use 64-bit hash
size_t TORCH_API StdDataHash(const void* data, size_t size);

size_t TORCH_API StdHashCombine(uintmax_t a, uintmax_t b);

// Other functions are all 128-bit
hash_t TORCH_API HashBlock(const void* data, size_t n, const hash_t& seed);

hash_t TORCH_API DataHash(const void* data, size_t size);

hash_t TORCH_API HashCombine(const hash_t& a, const hash_t& b);

size_t TORCH_API HashReduce(const hash_t& a);

// Returns a string representation of a hash
std::string TORCH_API HashToString(const hash_t& a);

struct HashReducer {
  size_t operator()(const hash_t& value) const {
    return HashReduce(value);
  }
};

static inline hash_t StringHash(const char* data) {
  return DataHash(data, std::strlen(data));
}

// Automatic templated implementation for 'arithmetic' types
template <typename T, std::enable_if_t<std::is_arithmetic_v<T>>* = nullptr>
hash_t Hash(const T& value) {
  return DataHash(&value, sizeof(value));
}

// added because on macos builds the vector<bool> specialization
// breaks falling through to the templated arithmetic types above
hash_t TORCH_API Hash(const std::vector<bool>& value);

// Specialized implementations for proprietary types
static inline hash_t Hash(const c10::ScalarType& value) {
  return DataHash(&value, sizeof(value));
}

static inline hash_t Hash(const c10::MemoryFormat& value) {
  return DataHash(&value, sizeof(value));
}

static inline hash_t Hash(const c10::DeviceType& value) {
  return DataHash(&value, sizeof(value));
}

static inline hash_t Hash(const c10::Device& value) {
  return HashCombine(Hash(value.type()), Hash(value.index()));
}

static inline hash_t Hash(const c10::Layout& value) {
  return DataHash(&value, sizeof(value));
}

static inline hash_t Hash(const c10::Scalar& value) {
  switch (value.type()) {
    case c10::ScalarType::ComplexDouble:
      return Hash(value.toComplexDouble());
    case c10::ScalarType::Double:
      return Hash(value.toDouble());
    case c10::ScalarType::Long:
      return Hash(value.toLong());
    case c10::ScalarType::Bool:
      return Hash(value.toBool());
    default:
      TORCH_INTERNAL_ASSERT(false, "Unknown scalar type.", value.type());
  }
}

static inline hash_t TensorHash(const at::Tensor& tensor) {
  at::Tensor ctensor = tensor.contiguous();
  int64_t size = ctensor.numel() * ctensor.element_size();
  switch (ctensor.scalar_type()) {
    case at::ScalarType::Bool:
      return DataHash(ctensor.const_data_ptr<bool>(), size);
    case at::ScalarType::Byte:
      return DataHash(ctensor.const_data_ptr<uint8_t>(), size);
    case at::ScalarType::Char:
      return DataHash(ctensor.const_data_ptr<int8_t>(), size);
    case at::ScalarType::Short:
      return DataHash(ctensor.const_data_ptr<int16_t>(), size);
    case at::ScalarType::Int:
      return DataHash(ctensor.const_data_ptr<int32_t>(), size);
    case at::ScalarType::Long:
      return DataHash(ctensor.const_data_ptr<int64_t>(), size);
    case at::ScalarType::Float:
      return DataHash(ctensor.const_data_ptr<float>(), size);
    case at::ScalarType::Double:
      return DataHash(ctensor.const_data_ptr<double>(), size);
    case at::ScalarType::BFloat16:
      return DataHash(ctensor.const_data_ptr<at::BFloat16>(), size);
    case at::ScalarType::Half:
      return DataHash(ctensor.const_data_ptr<at::Half>(), size);
    case at::ScalarType::ComplexFloat:
      return DataHash(ctensor.const_data_ptr<c10::complex<float>>(), size);
    case at::ScalarType::ComplexDouble:
      return DataHash(ctensor.const_data_ptr<c10::complex<double>>(), size);
    case at::ScalarType::UInt16:
      return DataHash(ctensor.const_data_ptr<uint16_t>(), size);
    case at::ScalarType::UInt32:
      return DataHash(ctensor.const_data_ptr<uint32_t>(), size);
    case at::ScalarType::UInt64:
      return DataHash(ctensor.const_data_ptr<uint64_t>(), size);
    default:
      TORCH_INTERNAL_ASSERT(
          false, "Unsupported scalar type:", ctensor.scalar_type());
  }
}

static inline hash_t Hash(const std::string& value) {
  return DataHash(value.data(), value.size());
}

static inline hash_t Hash(const std::string_view& value) {
  return DataHash(value.data(), value.size());
}

static inline hash_t Hash(const at::Generator& value) {
  return TensorHash(value.get_state());
}

// Taken from glibc's implementation of hashing optionals,
// we want to include a contribution to the hash to distinguish
// cases where one or another option was null, but we hope it doesn't
// collide with an actually scalar value.
//
// Use an arbitrary randomly-selected 64-bit integer rather than a
// small constant that we then hash at runtime so we don't have to
// repeatedly hash a constant at runtime.
// NOLINTNEXTLINE(*-narrowing-conversions)
static const int64_t kNullOpt = 0x8655d738f3678dda;

// Hashing for std::optional types contributes to hash
// for optionals with null value, important to distinguish
// between <nullopt, non-nullopt> and <non-nullopt, nullopt> cases
template <typename T>
hash_t Hash(const std::optional<T>& value) {
  if (value.has_value()) {
    return Hash(value.value());
  } else {
    return kNullOpt;
  }
}

// Hashing of containers
// Forward declare to allow hashes of vectors of vectors to work.
template <typename T>
hash_t ContainerHash(const T& values);

template <typename T>
hash_t Hash(const std::vector<T>& values) {
  return ContainerHash(values);
}

// Need a special case for std::optional<container>?
template <typename T>
hash_t Hash(const std::optional<std::vector<T>>& value) {
  if (value.has_value()) {
    return ContainerHash(value.value());
  } else {
    return kNullOpt;
  }
}

template <typename T>
hash_t Hash(const std::set<T>& values) {
  return ContainerHash(values);
}

template <typename T, typename S>
hash_t Hash(const std::pair<T, S>& values) {
  return HashCombine(Hash(values.first), Hash(values.second));
}

static inline hash_t Hash(const hash_t& value) {
  return value;
}

template <typename T>
hash_t Hash(c10::ArrayRef<T> values) {
  return ContainerHash(values);
}

template <typename T>
hash_t ContainerHash(const T& values) {
  hash_t h(static_cast<uint64_t>(0x85ebca77c2b2ae63));
  for (const auto& value : values) {
    h = HashCombine(h, Hash(value));
  }
  return h;
}

// Varargs hashing
template <typename T = void>
hash_t MHash() {
  return hash_t(static_cast<uint64_t>(0x165667b19e3779f9));
}

template <typename T, typename... Targs>
hash_t MHash(T value, Targs... Fargs) {
  return HashCombine(Hash(value), MHash(Fargs...));
}

} // namespace torch::lazy

```



## High-Level Overview


This C++ file contains approximately 1 class(es)/struct(s) and 72 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `torch`

**Classes/Structs**: `TORCH_API`, `HashReducer`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/csrc/lazy/core`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `ATen/Tensor.h`
- `c10/core/Scalar.h`
- `c10/util/int128.h`
- `torch/csrc/Export.h`
- `cstring`
- `set`
- `string`
- `string_view`
- `vector`


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

- **File Documentation**: `hash.h_docs.md`
- **Keyword Index**: `hash.h_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/csrc/lazy/core`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/csrc/lazy/core`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- Implements or uses **caching** mechanisms.
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

Files in the same folder (`docs/torch/csrc/lazy/core`):

- [`helpers.cpp_docs.md_docs.md`](./helpers.cpp_docs.md_docs.md)
- [`tensor_util.h_kw.md_docs.md`](./tensor_util.h_kw.md_docs.md)
- [`permutation_util.h_kw.md_docs.md`](./permutation_util.h_kw.md_docs.md)
- [`ir_util.cpp_kw.md_docs.md`](./ir_util.cpp_kw.md_docs.md)
- [`shape_inference.h_kw.md_docs.md`](./shape_inference.h_kw.md_docs.md)
- [`ir_builder.h_docs.md_docs.md`](./ir_builder.h_docs.md_docs.md)
- [`shape_inference.cpp_kw.md_docs.md`](./shape_inference.cpp_kw.md_docs.md)
- [`hash.h_kw.md_docs.md`](./hash.h_kw.md_docs.md)
- [`multi_wait.cpp_kw.md_docs.md`](./multi_wait.cpp_kw.md_docs.md)
- [`lazy_graph_executor.cpp_docs.md_docs.md`](./lazy_graph_executor.cpp_docs.md_docs.md)


## Cross-References

- **File Documentation**: `hash.h_docs.md_docs.md`
- **Keyword Index**: `hash.h_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
