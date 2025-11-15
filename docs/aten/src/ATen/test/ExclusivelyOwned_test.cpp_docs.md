# Documentation: `aten/src/ATen/test/ExclusivelyOwned_test.cpp`

## File Metadata

- **Path**: `aten/src/ATen/test/ExclusivelyOwned_test.cpp`
- **Size**: 2,983 bytes (2.91 KB)
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
#include <caffe2/core/tensor.h>
#include <c10/util/ExclusivelyOwned.h>
#include <c10/util/intrusive_ptr.h>

#include <string>

namespace {

template <typename T>
class ExclusivelyOwnedTest : public ::testing::Test {
 public:
  c10::ExclusivelyOwned<T> defaultConstructed;
  c10::ExclusivelyOwned<T> sample;
 protected:
  void SetUp() override; // defined below helpers
  void TearDown() override {
    defaultConstructed = c10::ExclusivelyOwned<T>();
    sample = c10::ExclusivelyOwned<T>();
  }
};

template <typename T>
T getSampleValue();

template <>
at::Tensor getSampleValue() {
  return at::zeros({2, 2}).to(at::kCPU);
}

template <>
caffe2::Tensor getSampleValue() {
  return caffe2::Tensor(getSampleValue<at::Tensor>());
}

template <typename T>
void assertIsSampleObject(const T& eo);

template <>
void assertIsSampleObject<at::Tensor>(const at::Tensor& t) {
  EXPECT_EQ(t.sizes(), (c10::IntArrayRef{2, 2}));
  EXPECT_EQ(t.strides(), (c10::IntArrayRef{2, 1}));
  ASSERT_EQ(t.scalar_type(), at::ScalarType::Float);
  static const float zeros[4] = {0};
  EXPECT_EQ(memcmp(zeros, t.data_ptr(), 4 * sizeof(float)), 0);
}

template <>
void assertIsSampleObject<caffe2::Tensor>(const caffe2::Tensor& t) {
  assertIsSampleObject<at::Tensor>(at::Tensor(t));
}


template <typename T>
void ExclusivelyOwnedTest<T>::SetUp() {
  defaultConstructed = c10::ExclusivelyOwned<T>();
  sample = c10::ExclusivelyOwned<T>(getSampleValue<T>());
}

using ExclusivelyOwnedTypes = ::testing::Types<
  at::Tensor,
  caffe2::Tensor
  >;

TYPED_TEST_SUITE(ExclusivelyOwnedTest, ExclusivelyOwnedTypes);

TYPED_TEST(ExclusivelyOwnedTest, DefaultConstructor) {
  c10::ExclusivelyOwned<TypeParam> defaultConstructed;
}

TYPED_TEST(ExclusivelyOwnedTest, MoveConstructor) {
  auto movedDefault = std::move(this->defaultConstructed);
  auto movedSample = std::move(this->sample);

  assertIsSampleObject(*movedSample);
}

TYPED_TEST(ExclusivelyOwnedTest, MoveAssignment) {
  // Move assignment from a default-constructed ExclusivelyOwned is handled in
  // TearDown at the end of every test!
  c10::ExclusivelyOwned<TypeParam> anotherSample = c10::ExclusivelyOwned<TypeParam>(getSampleValue<TypeParam>());
  anotherSample = std::move(this->sample);
  assertIsSampleObject(*anotherSample);
}

TYPED_TEST(ExclusivelyOwnedTest, MoveAssignmentFromContainedType) {
  c10::ExclusivelyOwned<TypeParam> anotherSample = c10::ExclusivelyOwned<TypeParam>(getSampleValue<TypeParam>());
  anotherSample = getSampleValue<TypeParam>();
  assertIsSampleObject(*anotherSample);
}

TYPED_TEST(ExclusivelyOwnedTest, Take) {
  auto x = std::move(this->sample).take();
  assertIsSampleObject(x);
}

} // namespace

extern "C" void inspectTensor() {
  auto t = getSampleValue<at::Tensor>();
}

extern "C" void inspectExclusivelyOwnedTensor() {
  c10::ExclusivelyOwned<at::Tensor> t(getSampleValue<at::Tensor>());
}

```



## High-Level Overview


This C++ file contains approximately 1 class(es)/struct(s) and 8 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `extern`

**Classes/Structs**: `ExclusivelyOwnedTest`


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
- `caffe2/core/tensor.h`
- `c10/util/ExclusivelyOwned.h`
- `c10/util/intrusive_ptr.h`
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
python aten/src/ATen/test/ExclusivelyOwned_test.cpp
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

- **File Documentation**: `ExclusivelyOwned_test.cpp_docs.md`
- **Keyword Index**: `ExclusivelyOwned_test.cpp_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
