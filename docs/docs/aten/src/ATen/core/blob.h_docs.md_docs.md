# Documentation: `docs/aten/src/ATen/core/blob.h_docs.md`

## File Metadata

- **Path**: `docs/aten/src/ATen/core/blob.h_docs.md`
- **Size**: 7,690 bytes (7.51 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `aten/src/ATen/core/blob.h`

## File Metadata

- **Path**: `aten/src/ATen/core/blob.h`
- **Size**: 5,244 bytes (5.12 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
#pragma once

#include <type_traits>

#include <c10/util/intrusive_ptr.h>
#include <c10/util/typeid.h>
#include <c10/macros/Macros.h>

namespace caffe2 {

class Tensor;

/**
 * @brief Blob is a general container that hosts a typed pointer.
 *
 * A Blob hosts a pointer as well as its type, and takes charge of deleting it
 * properly when the blob is deallocated or re-allocated with a new type. A blob
 * could contain anything, although the most common case is to contain a Tensor.
 */
class TORCH_API Blob final : public c10::intrusive_ptr_target {
 public:
  /**
   * Initializes an empty Blob.
   */
  Blob() noexcept = default;
  ~Blob() override {
    Reset();
  }

  Blob(Blob&& other) noexcept : Blob() {
    swap(other);
  }

  Blob& operator=(Blob&& other) noexcept {
    Blob(std::move(other)).swap(*this);
    return *this;
  }

  /**
   * Checks if the content stored in the blob is of type T.
   */
  template <class T>
  bool IsType() const noexcept {
    return meta_.Match<T>();
  }

  /**
   * Returns the meta info of the blob.
   */
  const TypeMeta meta() const noexcept {
    return meta_;
  }

  /**
   * Returns a printable typename of the blob.
   */
  std::string_view TypeName() const noexcept {
    return meta_.name();
  }

  /**
   * @brief Gets the const reference of the stored object. The code checks if
   * the stored object is of the desired type.
   */
  // TODO(jerryzh): add a Get(c10::DeviceType) function?
  template <class T>
  const T& Get() const {
    TORCH_INTERNAL_ASSERT(
        IsType<T>(),
        "wrong type for the Blob instance. Blob contains ",
        meta_.name(),
        " while caller expects ",
        TypeMeta::TypeName<T>());
    // TODO: after we add Get<Tensor>(c10::DeviceType)
    // and changed all the callsites, we can add
    // a static assert here to enforce T != Tensor
    return *static_cast<const T*>(pointer_);
  }

  const void* GetRaw() const noexcept {
    return pointer_;
  }
  void* GetRaw() noexcept {
    return pointer_;
  }

  /**
   * @brief Gets a mutable pointer to the stored object.
   *
   * If the current object is not of the right type, a new object is created
   * and the old object is freed. Note that type T should have a default
   * constructor. Otherwise, create the object yourself first, and use
   * Reset().
   */
  template <class T>
  T* GetMutable() {
    static_assert(
        std::is_default_constructible_v<T>,
        "GetMutable can't be called with non-default-constructible types. "
        "Try using specialized methods");
    if (IsType<T>()) {
      return static_cast<T*>(pointer_);
    } else {
      // TODO Re-enable logging
      // VLOG(1) << "Create new mutable object " << TypeMeta::TypeName<T>();
      return Reset<T>(new T());
    }
  }

  template <class T>
  T* GetMutableOrNull() {
    if (IsType<T>()) {
      return static_cast<T*>(pointer_);
    } else {
      return nullptr;
    }
  }

  /**
   * Sets the underlying object to the allocated one. The Blob then takes over
   * the ownership of the passed in pointer. If there is already an object in
   * the Blob, the old object is freed.
   *
   * This is used when the underlying class T does not have a default ctor, or
   * complex initializations needs to be done outside the blob.
   */
  template <class T>
  T* Reset(T* allocated) {
    free_();
    meta_ = TypeMeta::Make<T>();
    pointer_ = static_cast<void*>(allocated);
    has_ownership_ = true;
    return allocated;
  }

  /**
   * Sets the underlying object to the allocated one, but does not take over
   * the ownership of the passed in pointer. If there is already an object in
   * the Blob, the old object is freed.
   *
   * Unlike Reset, this does not take over the ownership of the pointer and the
   * caller is responsible for making sure that the lifetime of the allocated
   * blob outlasts the lifetime of any access to this blob, until another Reset
   * call is made or the blob is destructed.
   */
  template <class T>
  std::remove_const_t<T>* ShareExternal(
      std::remove_const_t<T>* allocated) {
    return static_cast<T*>(ShareExternal(
        static_cast<void*>(allocated),
        TypeMeta::Make<std::remove_const_t<T>>()));
  }

  void* ShareExternal(void* allocated, const TypeMeta meta) {
    free_();
    meta_ = meta;
    pointer_ = allocated;
    has_ownership_ = false;
    return allocated;
  }

  /**
   * Resets the Blob to an empty one.
   */
  void Reset() {
    free_();
    pointer_ = nullptr;
    meta_ = TypeMeta();
    has_ownership_ = false;
  }

  /**
   * @brief Swaps the underlying storage of two blobs.
   */
  void swap(Blob& rhs)  noexcept {
    using std::swap;
    swap(meta_, rhs.meta_);
    swap(pointer_, rhs.pointer_);
    swap(has_ownership_, rhs.has_ownership_);
  }

 private:
  void free_() {
    if (has_ownership_ && pointer_ != nullptr) {
      (*meta_.deleteFn())(pointer_);
    }
  }

  TypeMeta meta_;
  void* pointer_{nullptr};
  bool has_ownership_{false};

  C10_DISABLE_COPY_AND_ASSIGN(Blob);
};

inline void swap(Blob& lhs, Blob& rhs)  noexcept {
  lhs.swap(rhs);
}

inline std::ostream& operator<<(std::ostream& out, const Blob& v) {
  return out << "Blob[" << v.TypeName() << "]";
}

} // namespace caffe2

```



## High-Level Overview


This C++ file contains approximately 9 class(es)/struct(s) and 9 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `caffe2`

**Classes/Structs**: `Tensor`, `TORCH_API`, `T`, `T`, `T`, `T`, `T`, `T`, `T`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `aten/src/ATen/core`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



## Dependencies

### Import Dependencies

This file includes:

- `type_traits`
- `c10/util/intrusive_ptr.h`
- `c10/util/typeid.h`
- `c10/macros/Macros.h`


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

Files in the same folder (`aten/src/ATen/core`):

- [`DistributionsHelper.h_docs.md`](./DistributionsHelper.h_docs.md)
- [`rref_interface.h_docs.md`](./rref_interface.h_docs.md)
- [`Generator.h_docs.md`](./Generator.h_docs.md)
- [`enum_type.h_docs.md`](./enum_type.h_docs.md)
- [`QuantizerBase.h_docs.md`](./QuantizerBase.h_docs.md)
- [`Array.h_docs.md`](./Array.h_docs.md)
- [`MetaFallbackKernel.cpp_docs.md`](./MetaFallbackKernel.cpp_docs.md)
- [`ATenOpList.h_docs.md`](./ATenOpList.h_docs.md)
- [`ivalue_inl.h_docs.md`](./ivalue_inl.h_docs.md)
- [`TransformationHelper.h_docs.md`](./TransformationHelper.h_docs.md)


## Cross-References

- **File Documentation**: `blob.h_docs.md`
- **Keyword Index**: `blob.h_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/aten/src/ATen/core`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/aten/src/ATen/core`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



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

Files in the same folder (`docs/aten/src/ATen/core`):

- [`operator_name.cpp_docs.md_docs.md`](./operator_name.cpp_docs.md_docs.md)
- [`builtin_function.h_kw.md_docs.md`](./builtin_function.h_kw.md_docs.md)
- [`QuantizerBase.h_docs.md_docs.md`](./QuantizerBase.h_docs.md_docs.md)
- [`MT19937RNGEngine.h_docs.md_docs.md`](./MT19937RNGEngine.h_docs.md_docs.md)
- [`UndefinedTensorImpl.h_docs.md_docs.md`](./UndefinedTensorImpl.h_docs.md_docs.md)
- [`IListRef_test.cpp_docs.md_docs.md`](./IListRef_test.cpp_docs.md_docs.md)
- [`CheckMemoryFormat.h_docs.md_docs.md`](./CheckMemoryFormat.h_docs.md_docs.md)
- [`Tensor.cpp_kw.md_docs.md`](./Tensor.cpp_kw.md_docs.md)
- [`PythonFallbackKernel.cpp_docs.md_docs.md`](./PythonFallbackKernel.cpp_docs.md_docs.md)
- [`Dict.h_kw.md_docs.md`](./Dict.h_kw.md_docs.md)


## Cross-References

- **File Documentation**: `blob.h_docs.md_docs.md`
- **Keyword Index**: `blob.h_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
