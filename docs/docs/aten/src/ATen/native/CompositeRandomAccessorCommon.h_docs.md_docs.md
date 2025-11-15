# Documentation: `docs/aten/src/ATen/native/CompositeRandomAccessorCommon.h_docs.md`

## File Metadata

- **Path**: `docs/aten/src/ATen/native/CompositeRandomAccessorCommon.h_docs.md`
- **Size**: 9,254 bytes (9.04 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `aten/src/ATen/native/CompositeRandomAccessorCommon.h`

## File Metadata

- **Path**: `aten/src/ATen/native/CompositeRandomAccessorCommon.h`
- **Size**: 6,733 bytes (6.58 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
#include <utility>

#pragma once

namespace at::native {

namespace {

// operator_brackets_proxy is used in
// CompositeRandomAccessor in place of operator[].
// For some iterators, references returned by operator[]
// could become invalid, operator_brackets_proxy tries to
// resolve that by making accessor[n] to be equivalent to
// *(accessor + n).
template <typename Accessor>
class operator_brackets_proxy {
  using reference = typename std::iterator_traits<Accessor>::reference;
  using value_type = typename std::iterator_traits<Accessor>::value_type;

public:
  C10_HOST_DEVICE
  operator_brackets_proxy(Accessor const& accessor)
    : accessor(accessor)
  {}

  C10_HOST_DEVICE
  operator reference() {
    return *accessor;
  }

  C10_HOST_DEVICE
  reference operator*() {
    return *accessor;
  }

  C10_HOST_DEVICE
  operator_brackets_proxy& operator=(value_type const& val) {
    *accessor = val;
    return *this;
  }

private:
  Accessor accessor;
};

}

// references_holder is used as a surrogate for the
// references type from std::iterator_traits in CompositeRandomAccessor.
// It is assumed in CompositeRandomAccessor that
// References = tuple<Types&...>,
// Values = tuple<Types...> by default,
// but they could be anything as long as References could be
// cast to Values.
// If you plan to use it with STL, for example, you will need to
// define 'swap` and `get`(aka std::get) methods.
template <typename Values, typename References>
class references_holder {
public:
  using values = Values;
  using references = References;

  C10_HOST_DEVICE
  references_holder(references refs)
    : refs{std::move(refs)}
  {}

  C10_HOST_DEVICE
  operator references() {
    return refs;
  }

  C10_HOST_DEVICE
  operator values() {
    return refs;
  }

  C10_HOST_DEVICE
  references_holder& operator=(values vals) {
    refs = vals;
    return *this;
  }

  C10_HOST_DEVICE
  references& data() {
    return refs;
  }

protected:
  references refs;
};

// CompositeRandomAccessor is essentially a simplified version of
// a random access iterator over two random access iterators.
// TupleInfo should contain a variadic type `tuple`, and a method `tie`,
// which constructs a tuple of references from a variadic list of arguments.
template <typename KeyAccessor, typename ValueAccessor, typename TupleInfo>
class CompositeRandomAccessor {
  using self_type = CompositeRandomAccessor<KeyAccessor, ValueAccessor, TupleInfo>;

  using key_accessor_value_type =
    typename std::iterator_traits<KeyAccessor>::value_type;
  using value_accessor_value_type =
    typename std::iterator_traits<ValueAccessor>::value_type;
  using key_accessor_reference_type =
    typename std::iterator_traits<KeyAccessor>::reference;
  using value_accessor_reference_type =
    typename std::iterator_traits<ValueAccessor>::reference;

  using composite_value_type = typename TupleInfo::template tuple<
    key_accessor_value_type,
    value_accessor_value_type>;
  using composite_reference = typename TupleInfo::template tuple<
    key_accessor_reference_type,
    value_accessor_reference_type>;

public:
  using value_type = composite_value_type;
  using reference = references_holder<composite_value_type, composite_reference>;
  // Note that CompositeRandomAccessor does not hold key and values
  // in a specific datastructure, which means that a pointer to a (key, value)
  // is not defined. Hence we just use a pointer type of the KeyAccessor.
  using pointer = typename std::iterator_traits<KeyAccessor>::pointer;
  using difference_type = typename std::iterator_traits<KeyAccessor>::difference_type;
  using iterator_category = std::random_access_iterator_tag;

  C10_HOST_DEVICE
  CompositeRandomAccessor() = default;

  C10_HOST_DEVICE
  CompositeRandomAccessor(KeyAccessor keys, ValueAccessor values)
    : keys(keys), values(values)
  {}

  // Pointer-like operations {
  C10_HOST_DEVICE
  reference operator*() const {
    return TupleInfo::tie(*keys, *values);
  }

  // operator->() is supposed to return a pointer type.
  // Since CompositeRandomAccessor does not hold pointers to pairs,
  // we just return a pointer to a key.
  C10_HOST_DEVICE
  auto* operator->() const {
    return keys.operator->();
  }

  C10_HOST_DEVICE
  reference operator[](difference_type idx) {
    return operator_brackets_proxy<self_type>(
      CompositeRandomAccessor(keys + idx, values + idx)
    );
  }
  // }

  // Prefix/postfix increment/decrement {
  C10_HOST_DEVICE
  CompositeRandomAccessor& operator++() {
    ++keys;
    ++values;
    return *this;
  }

  C10_HOST_DEVICE
  CompositeRandomAccessor operator++(int) {
    CompositeRandomAccessor copy(*this);
    ++*this;
    return copy;
  }

  C10_HOST_DEVICE
  CompositeRandomAccessor& operator--() {
    --keys;
    --values;
    return *this;
  }

  C10_HOST_DEVICE
  CompositeRandomAccessor operator--(int) {
    CompositeRandomAccessor copy(*this);
    --*this;
    return copy;
  }
  // }

  // Arithmetic operations {
  C10_HOST_DEVICE
  CompositeRandomAccessor& operator+=(difference_type offset) {
    keys += offset;
    values += offset;
    return *this;
  }

  C10_HOST_DEVICE
  CompositeRandomAccessor operator+(difference_type offset) const {
    return CompositeRandomAccessor(keys + offset, values + offset);
  }

  C10_HOST_DEVICE
  friend CompositeRandomAccessor operator+(
    difference_type offset,
    const CompositeRandomAccessor& accessor
  ) {
    return accessor + offset;
  }

  C10_HOST_DEVICE
  CompositeRandomAccessor& operator-=(difference_type offset) {
    keys -= offset;
    values -= offset;
    return *this;
  }

  C10_HOST_DEVICE
  CompositeRandomAccessor operator-(difference_type offset) const {
    return CompositeRandomAccessor(keys - offset, values - offset);
  }

  C10_HOST_DEVICE
  difference_type operator-(const CompositeRandomAccessor& other) const {
    return keys - other.keys;
  }
  // }

  // Comparison operators {
  C10_HOST_DEVICE
  bool operator==(const CompositeRandomAccessor& other) const {
    return keys == other.keys;
  }

  C10_HOST_DEVICE
  bool operator!=(const CompositeRandomAccessor& other) const {
    return keys != other.keys;
  }

  C10_HOST_DEVICE
  bool operator<(const CompositeRandomAccessor& other) const {
    return keys < other.keys;
  }

  C10_HOST_DEVICE
  bool operator<=(const CompositeRandomAccessor& other) const {
    return keys <= other.keys;
  }

  C10_HOST_DEVICE
  bool operator>(const CompositeRandomAccessor& other) const {
    return keys > other.keys;
  }

  C10_HOST_DEVICE
  bool operator>=(const CompositeRandomAccessor& other) const {
    return keys >= other.keys;
  }
  // }

protected:
  KeyAccessor keys;
  ValueAccessor values;
};

} // namespace at::native

```



## High-Level Overview


This C++ file contains approximately 3 class(es)/struct(s) and 12 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `at`

**Classes/Structs**: `operator_brackets_proxy`, `references_holder`, `CompositeRandomAccessor`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `aten/src/ATen/native`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



## Dependencies

### Import Dependencies

This file includes:

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

- **File Documentation**: `CompositeRandomAccessorCommon.h_docs.md`
- **Keyword Index**: `CompositeRandomAccessorCommon.h_kw.md`
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

- **File Documentation**: `CompositeRandomAccessorCommon.h_docs.md_docs.md`
- **Keyword Index**: `CompositeRandomAccessorCommon.h_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
