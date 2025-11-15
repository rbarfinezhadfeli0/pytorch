# Documentation: `docs/aten/src/ATen/quantized/QTensorImpl.h_docs.md`

## File Metadata

- **Path**: `docs/aten/src/ATen/quantized/QTensorImpl.h_docs.md`
- **Size**: 6,103 bytes (5.96 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `aten/src/ATen/quantized/QTensorImpl.h`

## File Metadata

- **Path**: `aten/src/ATen/quantized/QTensorImpl.h`
- **Size**: 4,021 bytes (3.93 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
#pragma once

#include <ATen/quantized/Quantizer.h>
#include <c10/core/TensorImpl.h>
#include <c10/util/Exception.h>

namespace at {

/**
 * QTensorImpl is a TensorImpl for Quantized Tensors, it stores Quantizer which
 * specifies the quantization scheme and parameters, for more information please
 * see ATen/quantized/Quantizer.h
 *
 * We'll use QTensor in code or documentation to refer to a Tensor with QTensorImpl.
 */
struct TORCH_API QTensorImpl : public c10::TensorImpl {
 public:
  QTensorImpl(
      Storage&& storage,
      DispatchKeySet key_set,
      const caffe2::TypeMeta data_type,
      QuantizerPtr quantizer);

  // See Note [Enum ImplType]
  QTensorImpl(
      ImplType type,
      Storage&& storage,
      DispatchKeySet key_set,
      const caffe2::TypeMeta data_type,
      QuantizerPtr quantizer);


  // TODO: Expose in PyTorch Frontend
  QuantizerPtr quantizer() {
    return quantizer_;
  }

  void set_quantizer_(QuantizerPtr quantizer) {
    quantizer_ = quantizer;
  }

  /**
   * Return a TensorImpl that is a shallow-copy of this TensorImpl.
   *
   * For usage of `version_counter` and `allow_tensor_metadata_change`,
   * see NOTE [ TensorImpl Shallow-Copying ].
   */
  c10::intrusive_ptr<TensorImpl> shallow_copy_and_detach(
      const c10::VariableVersion& version_counter,
      bool allow_tensor_metadata_change) const override {
    auto impl = c10::make_intrusive<QTensorImpl>(
        Storage(storage()), key_set(), data_type_, quantizer_);
    copy_tensor_metadata(
      /*src_q_impl=*/this,
      /*dest_q_impl=*/impl.get(),
      /*version_counter=*/version_counter,
      /*allow_tensor_metadata_change=*/allow_tensor_metadata_change);
    impl->refresh_numel();
    impl->refresh_contiguous();
    return impl;
  }

  /**
   * Return a TensorImpl that is a shallow-copy of this TensorImpl.
   *
   * For usage of `version_counter` and `allow_tensor_metadata_change`,
   * see NOTE [ TensorImpl Shallow-Copying ].
   */
  c10::intrusive_ptr<TensorImpl> shallow_copy_and_detach(
      c10::VariableVersion&& version_counter,
      bool allow_tensor_metadata_change) const override {
    auto impl = c10::make_intrusive<QTensorImpl>(
        Storage(storage()), key_set(), data_type_, quantizer_);
    copy_tensor_metadata(
      /*src_q_impl=*/this,
      /*dest_q_impl=*/impl.get(),
      /*version_counter=*/std::move(version_counter),
      /*allow_tensor_metadata_change=*/allow_tensor_metadata_change);
    impl->refresh_numel();
    impl->refresh_contiguous();
    return impl;
  }

  /**
   * Shallow-copies data from another TensorImpl into this TensorImpl.
   *
   * For why this function doesn't check this TensorImpl's `allow_tensor_metadata_change_`,
   * see NOTE [ TensorImpl Shallow-Copying ].
   */
  void shallow_copy_from(const c10::intrusive_ptr<TensorImpl>& impl) override {
    AT_ASSERT(has_compatible_shallow_copy_type(impl->key_set()));
    auto q_impl = static_cast<const QTensorImpl*>(impl.get());
    copy_tensor_metadata(
      /*src_q_impl=*/q_impl,
      /*dest_q_impl=*/this,
      /*version_counter=*/version_counter(),
      /*allow_tensor_metadata_change=*/allow_tensor_metadata_change());
    refresh_numel();
    refresh_contiguous();
  }

 private:
  QuantizerPtr quantizer_;

  const char* tensorimpl_type_name() const override;

  /**
   * Copy the tensor metadata fields (e.g. sizes / strides / storage pointer / storage_offset)
   * from one TensorImpl to another TensorImpl.
   *
   * For usage of `version_counter` and `allow_tensor_metadata_change`, see NOTE [ TensorImpl Shallow-Copying ].
   */
  static void copy_tensor_metadata(
      const QTensorImpl* src_q_impl,
      QTensorImpl* dest_q_impl,
      const c10::VariableVersion& version_counter,
      bool allow_tensor_metadata_change) {
    TensorImpl::copy_tensor_metadata(src_q_impl, dest_q_impl, version_counter, allow_tensor_metadata_change);

    // OpaqueTensorImpl-specific fields.
    dest_q_impl->quantizer_ = src_q_impl->quantizer_;
  }
};

} // namespace at

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 5 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `at`

**Classes/Structs**: `TORCH_API`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `aten/src/ATen/quantized`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



## Dependencies

### Import Dependencies

This file includes:

- `ATen/quantized/Quantizer.h`
- `c10/core/TensorImpl.h`
- `c10/util/Exception.h`


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

Files in the same folder (`aten/src/ATen/quantized`):

- [`CMakeLists.txt_docs.md`](./CMakeLists.txt_docs.md)
- [`Quantizer.cpp_docs.md`](./Quantizer.cpp_docs.md)
- [`Quantizer.h_docs.md`](./Quantizer.h_docs.md)
- [`QTensorImpl.cpp_docs.md`](./QTensorImpl.cpp_docs.md)


## Cross-References

- **File Documentation**: `QTensorImpl.h_docs.md`
- **Keyword Index**: `QTensorImpl.h_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/aten/src/ATen/quantized`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/aten/src/ATen/quantized`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



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

Files in the same folder (`docs/aten/src/ATen/quantized`):

- [`CMakeLists.txt_docs.md_docs.md`](./CMakeLists.txt_docs.md_docs.md)
- [`QTensorImpl.h_kw.md_docs.md`](./QTensorImpl.h_kw.md_docs.md)
- [`QTensorImpl.cpp_docs.md_docs.md`](./QTensorImpl.cpp_docs.md_docs.md)
- [`Quantizer.cpp_kw.md_docs.md`](./Quantizer.cpp_kw.md_docs.md)
- [`Quantizer.h_kw.md_docs.md`](./Quantizer.h_kw.md_docs.md)
- [`QTensorImpl.cpp_kw.md_docs.md`](./QTensorImpl.cpp_kw.md_docs.md)
- [`Quantizer.cpp_docs.md_docs.md`](./Quantizer.cpp_docs.md_docs.md)
- [`CMakeLists.txt_kw.md_docs.md`](./CMakeLists.txt_kw.md_docs.md)
- [`Quantizer.h_docs.md_docs.md`](./Quantizer.h_docs.md_docs.md)


## Cross-References

- **File Documentation**: `QTensorImpl.h_docs.md_docs.md`
- **Keyword Index**: `QTensorImpl.h_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
