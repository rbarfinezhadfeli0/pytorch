# Documentation: `docs/aten/src/ATen/test/MaybeOwned_test.cpp_docs.md`

## File Metadata

- **Path**: `docs/aten/src/ATen/test/MaybeOwned_test.cpp_docs.md`
- **Size**: 12,263 bytes (11.98 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**. This appears to be a **test file**.

## Original Source

```markdown
# Documentation: `aten/src/ATen/test/MaybeOwned_test.cpp`

## File Metadata

- **Path**: `aten/src/ATen/test/MaybeOwned_test.cpp`
- **Size**: 9,556 bytes (9.33 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**.

## Original Source

```cpp
#include <gtest/gtest.h>

#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#include <ATen/Tensor.h>
#include <ATen/core/ivalue.h>
#include <c10/util/intrusive_ptr.h>
#include <c10/util/MaybeOwned.h>

#include <memory>
#include <string>

namespace {

using at::Tensor;
using c10::IValue;

struct MyString : public c10::intrusive_ptr_target, public std::string {
  using std::string::string;
};

template <typename T>
class MaybeOwnedTest : public ::testing::Test {
 public:
  T borrowFrom;
  T ownCopy;
  T ownCopy2;
  c10::MaybeOwned<T> borrowed;
  c10::MaybeOwned<T> owned;
  c10::MaybeOwned<T> owned2;

 protected:
  void SetUp() override; // defined below helpers
  void TearDown() override {
    // Release everything to try to trigger ASAN violations in the
    // test that broke things.
    borrowFrom = T();
    ownCopy = T();
    ownCopy2 = T();

    borrowed = c10::MaybeOwned<T>();
    owned = c10::MaybeOwned<T>();
    owned2 = c10::MaybeOwned<T>();
  }

};


//////////////////// Helpers that differ per tested type. ////////////////////

template <typename T>
T getSampleValue();

template <typename T>
T getSampleValue2();

template <typename T>
void assertBorrow(const c10::MaybeOwned<T>&, const T&);

template <typename T>
void assertOwn(const c10::MaybeOwned<T>&, const T&, size_t useCount = 2);

////////////////// Helper implementations for intrusive_ptr. //////////////////
template<>
c10::intrusive_ptr<MyString> getSampleValue() {
  return c10::make_intrusive<MyString>("hello");
}

template<>
c10::intrusive_ptr<MyString> getSampleValue2() {
  return c10::make_intrusive<MyString>("goodbye");
}

bool are_equal(const c10::intrusive_ptr<MyString>& lhs, const c10::intrusive_ptr<MyString>& rhs) {
  if (!lhs || !rhs) {
    return !lhs && !rhs;
  }
  return *lhs == *rhs;
}

template <>
void assertBorrow(
    const c10::MaybeOwned<c10::intrusive_ptr<MyString>>& mo,
    const c10::intrusive_ptr<MyString>& borrowedFrom) {
  EXPECT_EQ(*mo, borrowedFrom);
  EXPECT_EQ(mo->get(), borrowedFrom.get());
  EXPECT_EQ(borrowedFrom.use_count(), 1);
}

template <>
void assertOwn(
    const c10::MaybeOwned<c10::intrusive_ptr<MyString>>& mo,
    const c10::intrusive_ptr<MyString>& original,
    size_t useCount) {
  EXPECT_EQ(*mo, original);
  EXPECT_EQ(mo->get(), original.get());
  EXPECT_NE(&*mo, &original);
  EXPECT_EQ(original.use_count(), useCount);
}

//////////////////// Helper implementations for Tensor. ////////////////////

template<>
Tensor getSampleValue() {
  return at::zeros({2, 2}).to(at::kCPU);
}

template<>
Tensor getSampleValue2() {
  return at::native::ones({2, 2}).to(at::kCPU);
}

bool are_equal(const Tensor& lhs, const Tensor& rhs) {
  if (!lhs.defined() || !rhs.defined()) {
    return !lhs.defined() && !rhs.defined();
  }
  return at::native::cpu_equal(lhs, rhs);
}

template <>
void assertBorrow(
    const c10::MaybeOwned<Tensor>& mo,
    const Tensor& borrowedFrom) {
  EXPECT_TRUE(mo->is_same(borrowedFrom));
  EXPECT_EQ(borrowedFrom.use_count(), 1);
}

template <>
void assertOwn(
    const c10::MaybeOwned<Tensor>& mo,
    const Tensor& original,
    size_t useCount) {
  EXPECT_TRUE(mo->is_same(original));
  EXPECT_EQ(original.use_count(), useCount);
}

//////////////////// Helper implementations for IValue. ////////////////////

template<>
IValue getSampleValue() {
  return IValue(getSampleValue<Tensor>());
}

template<>
IValue getSampleValue2() {
  return IValue("hello");
}

bool are_equal(const IValue& lhs, const IValue& rhs) {
  if (lhs.isTensor() != rhs.isTensor()) {
    return false;
  }
  if (lhs.isTensor() && rhs.isTensor()) {
    return lhs.toTensor().equal(rhs.toTensor());
  }
  return lhs == rhs;
}

template <>
void assertBorrow(
    const c10::MaybeOwned<IValue>& mo,
    const IValue& borrowedFrom) {
  if (!borrowedFrom.isPtrType()) {
    EXPECT_EQ(*mo, borrowedFrom);
  } else {
    EXPECT_EQ(mo->internalToPointer(), borrowedFrom.internalToPointer());
    EXPECT_EQ(borrowedFrom.use_count(), 1);
  }
}

template <>
void assertOwn(
    const c10::MaybeOwned<IValue>& mo,
    const IValue& original,
    size_t useCount) {
  if (!original.isPtrType()) {
    EXPECT_EQ(*mo, original);
  } else {
    EXPECT_EQ(mo->internalToPointer(), original.internalToPointer());
    EXPECT_EQ(original.use_count(), useCount);
  }
}

template <typename T>
void MaybeOwnedTest<T>::SetUp() {
  borrowFrom = getSampleValue<T>();
  ownCopy = getSampleValue<T>();
  ownCopy2 = getSampleValue<T>();
  borrowed = c10::MaybeOwned<T>::borrowed(borrowFrom);
  owned = c10::MaybeOwned<T>::owned(std::in_place, ownCopy);
  owned2 = c10::MaybeOwned<T>::owned(T(ownCopy2));
}

using MaybeOwnedTypes = ::testing::Types<
  c10::intrusive_ptr<MyString>,
  at::Tensor,
  c10::IValue
  >;

TYPED_TEST_SUITE(MaybeOwnedTest, MaybeOwnedTypes);

TYPED_TEST(MaybeOwnedTest, SimpleDereferencingString) {
  assertBorrow(this->borrowed, this->borrowFrom);
  assertOwn(this->owned, this->ownCopy);
  assertOwn(this->owned2, this->ownCopy2);
}

TYPED_TEST(MaybeOwnedTest, DefaultCtor) {
  c10::MaybeOwned<TypeParam> borrowed, owned;
  // Don't leave the fixture versions around messing up reference counts.
  this->borrowed = c10::MaybeOwned<TypeParam>();
  this->owned = c10::MaybeOwned<TypeParam>();
  borrowed = c10::MaybeOwned<TypeParam>::borrowed(this->borrowFrom);
  owned = c10::MaybeOwned<TypeParam>::owned(std::in_place, this->ownCopy);

  assertBorrow(borrowed, this->borrowFrom);
  assertOwn(owned, this->ownCopy);
}

TYPED_TEST(MaybeOwnedTest, CopyConstructor) {

  auto copiedBorrowed(this->borrowed);
  auto copiedOwned(this->owned);
  auto copiedOwned2(this->owned2);

  assertBorrow(this->borrowed, this->borrowFrom);
  assertBorrow(copiedBorrowed, this->borrowFrom);

  assertOwn(this->owned, this->ownCopy, 3);
  assertOwn(copiedOwned, this->ownCopy, 3);
  assertOwn(this->owned2, this->ownCopy2, 3);
  assertOwn(copiedOwned2, this->ownCopy2, 3);
}

TYPED_TEST(MaybeOwnedTest, MoveDereferencing) {
  // Need a different value.
  this->owned = c10::MaybeOwned<TypeParam>::owned(std::in_place, getSampleValue2<TypeParam>());

  EXPECT_TRUE(are_equal(*std::move(this->borrowed), getSampleValue<TypeParam>()));
  EXPECT_TRUE(are_equal(*std::move(this->owned), getSampleValue2<TypeParam>()));

  // Borrowed is unaffected.
  assertBorrow(this->borrowed, this->borrowFrom);

  // Owned is a null c10::intrusive_ptr / empty Tensor.
  EXPECT_TRUE(are_equal(*this->owned, TypeParam()));
}

TYPED_TEST(MaybeOwnedTest, MoveConstructor) {
  auto movedBorrowed(std::move(this->borrowed));
  auto movedOwned(std::move(this->owned));
  auto movedOwned2(std::move(this->owned2));

  assertBorrow(movedBorrowed, this->borrowFrom);
  assertOwn(movedOwned, this->ownCopy);
  assertOwn(movedOwned2, this->ownCopy2);
}

TYPED_TEST(MaybeOwnedTest, CopyAssignmentIntoOwned) {
  auto copiedBorrowed = c10::MaybeOwned<TypeParam>::owned(std::in_place);
  auto copiedOwned = c10::MaybeOwned<TypeParam>::owned(std::in_place);
  auto copiedOwned2 = c10::MaybeOwned<TypeParam>::owned(std::in_place);

  copiedBorrowed = this->borrowed;
  copiedOwned = this->owned;
  copiedOwned2 = this->owned2;

  assertBorrow(this->borrowed, this->borrowFrom);
  assertBorrow(copiedBorrowed, this->borrowFrom);
  assertOwn(this->owned, this->ownCopy, 3);
  assertOwn(copiedOwned, this->ownCopy, 3);
  assertOwn(this->owned2, this->ownCopy2, 3);
  assertOwn(copiedOwned2, this->ownCopy2, 3);
}

TYPED_TEST(MaybeOwnedTest, CopyAssignmentIntoBorrowed) {
  auto otherBorrowFrom = getSampleValue2<TypeParam>();
  auto otherOwnCopy = getSampleValue2<TypeParam>();
  auto copiedBorrowed = c10::MaybeOwned<TypeParam>::borrowed(otherBorrowFrom);
  auto copiedOwned = c10::MaybeOwned<TypeParam>::borrowed(otherOwnCopy);
  auto copiedOwned2 = c10::MaybeOwned<TypeParam>::borrowed(otherOwnCopy);

  copiedBorrowed = this->borrowed;
  copiedOwned = this->owned;
  copiedOwned2 = this->owned2;

  assertBorrow(this->borrowed, this->borrowFrom);
  assertBorrow(copiedBorrowed, this->borrowFrom);

  assertOwn(this->owned, this->ownCopy, 3);
  assertOwn(this->owned2, this->ownCopy2, 3);
  assertOwn(copiedOwned, this->ownCopy, 3);
  assertOwn(copiedOwned2, this->ownCopy2, 3);
}


TYPED_TEST(MaybeOwnedTest, MoveAssignmentIntoOwned) {

  auto movedBorrowed = c10::MaybeOwned<TypeParam>::owned(std::in_place);
  auto movedOwned = c10::MaybeOwned<TypeParam>::owned(std::in_place);
  auto movedOwned2 = c10::MaybeOwned<TypeParam>::owned(std::in_place);

  movedBorrowed = std::move(this->borrowed);
  movedOwned = std::move(this->owned);
  movedOwned2 = std::move(this->owned2);

  assertBorrow(movedBorrowed, this->borrowFrom);
  assertOwn(movedOwned, this->ownCopy);
  assertOwn(movedOwned2, this->ownCopy2);
}


TYPED_TEST(MaybeOwnedTest, MoveAssignmentIntoBorrowed) {
  auto y = getSampleValue2<TypeParam>();
  auto movedBorrowed = c10::MaybeOwned<TypeParam>::borrowed(y);
  auto movedOwned = c10::MaybeOwned<TypeParam>::borrowed(y);
  auto movedOwned2 = c10::MaybeOwned<TypeParam>::borrowed(y);

  movedBorrowed = std::move(this->borrowed);
  movedOwned = std::move(this->owned);
  movedOwned2 = std::move(this->owned2);

  assertBorrow(movedBorrowed, this->borrowFrom);
  assertOwn(movedOwned, this->ownCopy);
  assertOwn(movedOwned2, this->ownCopy2);
}

TYPED_TEST(MaybeOwnedTest, SelfAssignment) {
  this->borrowed = this->borrowed;
  this->owned = this->owned;
  this->owned2 = this->owned2;

  assertBorrow(this->borrowed, this->borrowFrom);
  assertOwn(this->owned, this->ownCopy);
  assertOwn(this->owned2, this->ownCopy2);
}

} // namespace

```



## High-Level Overview


This C++ file contains approximately 1 class(es)/struct(s) and 27 function(s).

## Detailed Analysis

### Code Structure

**Classes/Structs**: `MyString`, `MaybeOwnedTest`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `aten/src/ATen/test`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



## Dependencies

### Import Dependencies

This file includes:

- `gtest/gtest.h`
- `ATen/Functions.h`
- `ATen/NativeFunctions.h`
- `ATen/Tensor.h`
- `ATen/core/ivalue.h`
- `c10/util/intrusive_ptr.h`
- `c10/util/MaybeOwned.h`
- `memory`
- `string`


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

This is a test file. Run it with:

```bash
python aten/src/ATen/test/MaybeOwned_test.cpp
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`aten/src/ATen/test`):

- [`operators_test.cpp_docs.md`](./operators_test.cpp_docs.md)
- [`xpu_generator_test.cpp_docs.md`](./xpu_generator_test.cpp_docs.md)
- [`native_test.cpp_docs.md`](./native_test.cpp_docs.md)
- [`reportMemoryUsage.h_docs.md`](./reportMemoryUsage.h_docs.md)
- [`tensor_iterator_test.cpp_docs.md`](./tensor_iterator_test.cpp_docs.md)
- [`memory_overlapping_test.cpp_docs.md`](./memory_overlapping_test.cpp_docs.md)
- [`operator_name_test.cpp_docs.md`](./operator_name_test.cpp_docs.md)
- [`cuda_distributions_test.cu_docs.md`](./cuda_distributions_test.cu_docs.md)
- [`type_test.cpp_docs.md`](./type_test.cpp_docs.md)
- [`allocator_clone_test.h_docs.md`](./allocator_clone_test.h_docs.md)


## Cross-References

- **File Documentation**: `MaybeOwned_test.cpp_docs.md`
- **Keyword Index**: `MaybeOwned_test.cpp_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/aten/src/ATen/test`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/aten/src/ATen/test`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



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

This is a test file. Run it with:

```bash
python docs/aten/src/ATen/test/MaybeOwned_test.cpp_docs.md
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`docs/aten/src/ATen/test`):

- [`cuda_dlconvertor_test.cpp_kw.md_docs.md`](./cuda_dlconvertor_test.cpp_kw.md_docs.md)
- [`cuda_atomic_ops_test.cu_kw.md_docs.md`](./cuda_atomic_ops_test.cu_kw.md_docs.md)
- [`ivalue_test.cpp_kw.md_docs.md`](./ivalue_test.cpp_kw.md_docs.md)
- [`mobile_memory_cleanup.cpp_kw.md_docs.md`](./mobile_memory_cleanup.cpp_kw.md_docs.md)
- [`reportMemoryUsage_test.cpp_docs.md_docs.md`](./reportMemoryUsage_test.cpp_docs.md_docs.md)
- [`cpu_rng_test.cpp_kw.md_docs.md`](./cpu_rng_test.cpp_kw.md_docs.md)
- [`lazy_tensor_test.cpp_kw.md_docs.md`](./lazy_tensor_test.cpp_kw.md_docs.md)
- [`cuda_allocator_test.cpp_docs.md_docs.md`](./cuda_allocator_test.cpp_docs.md_docs.md)
- [`dlconvertor_test.cpp_kw.md_docs.md`](./dlconvertor_test.cpp_kw.md_docs.md)


## Cross-References

- **File Documentation**: `MaybeOwned_test.cpp_docs.md_docs.md`
- **Keyword Index**: `MaybeOwned_test.cpp_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
