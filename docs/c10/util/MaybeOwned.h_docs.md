# Documentation: `c10/util/MaybeOwned.h`

## File Metadata

- **Path**: `c10/util/MaybeOwned.h`
- **Size**: 7,177 bytes (7.01 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
#pragma once

#include <c10/macros/Macros.h>
#include <c10/util/Exception.h>

#include <memory>
#include <type_traits>
#include <utility>

namespace c10 {

/// MaybeOwnedTraits<T> describes how to borrow from T.  Here is how we
/// can implement borrowing from an arbitrary type T using a raw
/// pointer to const:
template <typename T>
struct MaybeOwnedTraitsGenericImpl {
  using owned_type = T;
  using borrow_type = const T*;

  static borrow_type createBorrow(const owned_type& from) {
    return &from;
  }

  static void assignBorrow(borrow_type& lhs, borrow_type rhs) {
    lhs = rhs;
  }

  static void destroyBorrow(borrow_type& /*toDestroy*/) {}

  static const owned_type& referenceFromBorrow(const borrow_type& borrow) {
    return *borrow;
  }

  static const owned_type* pointerFromBorrow(const borrow_type& borrow) {
    return borrow;
  }

  static bool debugBorrowIsValid(const borrow_type& borrow) {
    return borrow != nullptr;
  }
};

/// It is possible to eliminate the extra layer of indirection for
/// borrows for some types that we control. For examples, see
/// intrusive_ptr.h and TensorBody.h.

template <typename T>
struct MaybeOwnedTraits;

// Explicitly enable MaybeOwned<shared_ptr<T>>, rather than allowing
// MaybeOwned to be used for any type right away.
template <typename T>
struct MaybeOwnedTraits<std::shared_ptr<T>>
    : public MaybeOwnedTraitsGenericImpl<std::shared_ptr<T>> {};

/// A smart pointer around either a borrowed or owned T. When
/// constructed with borrowed(), the caller MUST ensure that the
/// borrowed-from argument outlives this MaybeOwned<T>. Compare to
/// Rust's std::borrow::Cow
/// (https://doc.rust-lang.org/std/borrow/enum.Cow.html), but note
/// that it is probably not suitable for general use because C++ has
/// no borrow checking. Included here to support
/// Tensor::expect_contiguous.
template <typename T>
class MaybeOwned final {
  using borrow_type = typename MaybeOwnedTraits<T>::borrow_type;
  using owned_type = typename MaybeOwnedTraits<T>::owned_type;

  bool isBorrowed_;
  union {
    borrow_type borrow_;
    owned_type own_;
  };

  /// Don't use this; use borrowed() instead.
  explicit MaybeOwned(const owned_type& t)
      : isBorrowed_(true), borrow_(MaybeOwnedTraits<T>::createBorrow(t)) {}

  /// Don't use this; use owned() instead.
  explicit MaybeOwned(T&& t) noexcept(std::is_nothrow_move_constructible_v<T>)
      : isBorrowed_(false), own_(std::move(t)) {}

  /// Don't use this; use owned() instead.
  template <class... Args>
  explicit MaybeOwned(std::in_place_t /*unused*/, Args&&... args)
      : isBorrowed_(false), own_(std::forward<Args>(args)...) {}

 public:
  explicit MaybeOwned() : isBorrowed_(true), borrow_() {}

  // Copying a borrow yields another borrow of the original, as with a
  // T*. Copying an owned T yields another owned T for safety: no
  // chains of borrowing by default! (Note you could get that behavior
  // with MaybeOwned<T>::borrowed(*rhs) if you wanted it.)
  MaybeOwned(const MaybeOwned& rhs) : isBorrowed_(rhs.isBorrowed_) {
    if (C10_LIKELY(rhs.isBorrowed_)) {
      MaybeOwnedTraits<T>::assignBorrow(borrow_, rhs.borrow_);
    } else {
      new (&own_) T(rhs.own_);
    }
  }

  MaybeOwned& operator=(const MaybeOwned& rhs) {
    if (this == &rhs) {
      return *this;
    }
    if (C10_UNLIKELY(!isBorrowed_)) {
      if (rhs.isBorrowed_) {
        own_.~T();
        MaybeOwnedTraits<T>::assignBorrow(borrow_, rhs.borrow_);
        isBorrowed_ = true;
      } else {
        own_ = rhs.own_;
      }
    } else {
      if (C10_LIKELY(rhs.isBorrowed_)) {
        MaybeOwnedTraits<T>::assignBorrow(borrow_, rhs.borrow_);
      } else {
        MaybeOwnedTraits<T>::destroyBorrow(borrow_);
        new (&own_) T(rhs.own_);
        isBorrowed_ = false;
      }
    }
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(isBorrowed_ == rhs.isBorrowed_);
    return *this;
  }

  MaybeOwned(MaybeOwned&& rhs) noexcept(
      // NOLINTNEXTLINE(*-noexcept-move-*)
      std::is_nothrow_move_constructible_v<T> &&
      std::is_nothrow_move_assignable_v<borrow_type>)
      : isBorrowed_(rhs.isBorrowed_) {
    if (C10_LIKELY(rhs.isBorrowed_)) {
      MaybeOwnedTraits<T>::assignBorrow(borrow_, rhs.borrow_);
    } else {
      new (&own_) T(std::move(rhs.own_));
    }
  }

  MaybeOwned& operator=(MaybeOwned&& rhs) noexcept(
      std::is_nothrow_move_assignable_v<T> &&
      std::is_nothrow_move_assignable_v<borrow_type> &&
      std::is_nothrow_move_constructible_v<T> &&
      // NOLINTNEXTLINE(*-noexcept-move-*)
      std::is_nothrow_destructible_v<T> &&
      std::is_nothrow_destructible_v<borrow_type>) {
    if (this == &rhs) {
      return *this;
    }
    if (C10_UNLIKELY(!isBorrowed_)) {
      if (rhs.isBorrowed_) {
        own_.~T();
        MaybeOwnedTraits<T>::assignBorrow(borrow_, rhs.borrow_);
        isBorrowed_ = true;
      } else {
        own_ = std::move(rhs.own_);
      }
    } else {
      if (C10_LIKELY(rhs.isBorrowed_)) {
        MaybeOwnedTraits<T>::assignBorrow(borrow_, rhs.borrow_);
      } else {
        MaybeOwnedTraits<T>::destroyBorrow(borrow_);
        new (&own_) T(std::move(rhs.own_));
        isBorrowed_ = false;
      }
    }
    return *this;
  }

  static MaybeOwned borrowed(const T& t) {
    return MaybeOwned(t);
  }

  static MaybeOwned owned(T&& t) noexcept(
      std::is_nothrow_move_constructible_v<T>) {
    return MaybeOwned(std::move(t));
  }

  template <class... Args>
  static MaybeOwned owned(std::in_place_t /*unused*/, Args&&... args) {
    return MaybeOwned(std::in_place, std::forward<Args>(args)...);
  }

  ~MaybeOwned() noexcept(
      // NOLINTNEXTLINE(*-noexcept-destructor)
      std::is_nothrow_destructible_v<T> &&
      std::is_nothrow_destructible_v<borrow_type>) {
    if (C10_UNLIKELY(!isBorrowed_)) {
      own_.~T();
    } else {
      MaybeOwnedTraits<T>::destroyBorrow(borrow_);
    }
  }

  // This is an implementation detail!  You should know what you're doing
  // if you are testing this.  If you just want to guarantee ownership move
  // this into a T
  bool unsafeIsBorrowed() const {
    return isBorrowed_;
  }

  const T& operator*() const& {
    if (isBorrowed_) {
      TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
          MaybeOwnedTraits<T>::debugBorrowIsValid(borrow_));
    }
    return C10_LIKELY(isBorrowed_)
        ? MaybeOwnedTraits<T>::referenceFromBorrow(borrow_)
        : own_;
  }

  const T* operator->() const {
    if (isBorrowed_) {
      TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
          MaybeOwnedTraits<T>::debugBorrowIsValid(borrow_));
    }
    return C10_LIKELY(isBorrowed_)
        ? MaybeOwnedTraits<T>::pointerFromBorrow(borrow_)
        : &own_;
  }

  // If borrowed, copy the underlying T. If owned, move from
  // it. borrowed/owned state remains the same, and either we
  // reference the same borrow as before or we are an owned moved-from
  // T.
  T operator*() && {
    if (isBorrowed_) {
      TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
          MaybeOwnedTraits<T>::debugBorrowIsValid(borrow_));
      return MaybeOwnedTraits<T>::referenceFromBorrow(borrow_);
    } else {
      return std::move(own_);
    }
  }
};

} // namespace c10

```



## High-Level Overview


This C++ file contains approximately 1 class(es)/struct(s) and 21 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `c10`

**Classes/Structs**: `MaybeOwnedTraitsGenericImpl`, `MaybeOwnedTraits`, `MaybeOwnedTraits`, `MaybeOwned`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `c10/util`, which is part of **C10** (Caffe2 Core), the core library providing fundamental abstractions.



## Dependencies

### Import Dependencies

This file includes:

- `c10/macros/Macros.h`
- `c10/util/Exception.h`
- `memory`
- `type_traits`
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

Files in the same folder (`c10/util`):

- [`CallOnce.h_docs.md`](./CallOnce.h_docs.md)
- [`Unicode.cpp_docs.md`](./Unicode.cpp_docs.md)
- [`logging_is_not_google_glog.h_docs.md`](./logging_is_not_google_glog.h_docs.md)
- [`Array.h_docs.md`](./Array.h_docs.md)
- [`complex_math.h_docs.md`](./complex_math.h_docs.md)
- [`order_preserving_flat_hash_map.h_docs.md`](./order_preserving_flat_hash_map.h_docs.md)
- [`flags_use_gflags.cpp_docs.md`](./flags_use_gflags.cpp_docs.md)
- [`flags_use_no_gflags.cpp_docs.md`](./flags_use_no_gflags.cpp_docs.md)
- [`Float8_e4m3fnuz.h_docs.md`](./Float8_e4m3fnuz.h_docs.md)
- [`typeid.cpp_docs.md`](./typeid.cpp_docs.md)


## Cross-References

- **File Documentation**: `MaybeOwned.h_docs.md`
- **Keyword Index**: `MaybeOwned.h_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
