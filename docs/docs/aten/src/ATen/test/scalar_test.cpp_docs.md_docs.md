# Documentation: `docs/aten/src/ATen/test/scalar_test.cpp_docs.md`

## File Metadata

- **Path**: `docs/aten/src/ATen/test/scalar_test.cpp_docs.md`
- **Size**: 8,705 bytes (8.50 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**. This appears to be a **test file**.

## Original Source

```markdown
# Documentation: `aten/src/ATen/test/scalar_test.cpp`

## File Metadata

- **Path**: `aten/src/ATen/test/scalar_test.cpp`
- **Size**: 6,014 bytes (5.87 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**.

## Original Source

```cpp
#include <gtest/gtest.h>

#include <iostream>
#include <random>
#include <c10/core/SymInt.h>
// define constants like M_PI and C keywords for MSVC
#ifdef _MSC_VER
#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES
#endif
#include <math.h>
#endif
#include <ATen/ATen.h>
#include <ATen/Dispatch.h>

// We intentionally test self assignment/move in this file, suppress warnings
// on them
#ifndef _MSC_VER
#pragma GCC diagnostic ignored "-Wpragmas"
#pragma GCC diagnostic ignored "-Wunknown-warning-option"
#pragma GCC diagnostic ignored "-Wself-move"
#endif

#ifdef __clang__
#pragma clang diagnostic ignored "-Wself-assign-overloaded"
#endif

using std::cout;
using namespace at;

template<typename scalar_type>
struct Foo {
  static void apply(Tensor a, Tensor b) {
    scalar_type s = 1;
    std::stringstream ss;
    ss << "hello, dispatch: " << a.toString() << s << "\n";
    auto data = (scalar_type*)a.data_ptr();
    (void)data;
  }
};
template<>
struct Foo<Half> {
  static void apply(Tensor a, Tensor b) {}
};

void test_overflow() {
  auto s1 = Scalar(M_PI);
  ASSERT_EQ(s1.toFloat(), static_cast<float>(M_PI));
  s1.toHalf();

  s1 = Scalar(100000);
  ASSERT_EQ(s1.toFloat(), 100000.0);
  ASSERT_EQ(s1.toInt(), 100000);

  // NOLINTNEXTLINE(cppcoreguidelines-avoid-goto,hicpp-avoid-goto)
  ASSERT_THROW(s1.toHalf(), std::runtime_error);

  s1 = Scalar(NAN);
  ASSERT_TRUE(std::isnan(s1.toFloat()));
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-goto,hicpp-avoid-goto)
  ASSERT_THROW(s1.toInt(), std::runtime_error);

  s1 = Scalar(INFINITY);
  ASSERT_TRUE(std::isinf(s1.toFloat()));
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-goto,hicpp-avoid-goto)
  ASSERT_THROW(s1.toInt(), std::runtime_error);
}

TEST(TestScalar, TestScalar) {
  manual_seed(123);

  Scalar what = 257;
  Scalar bar = 3.0;
  Half h = bar.toHalf();
  Scalar h2 = h;
  cout << "H2: " << h2.toDouble() << " " << what.toFloat() << " "
       << bar.toDouble() << " " << what.isIntegral(false) << "\n";
  auto gen = at::detail::getDefaultCPUGenerator();
  {
    // See Note [Acquire lock when using random generators]
    std::lock_guard<std::mutex> lock(gen.mutex());
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-goto,hicpp-avoid-goto)
    ASSERT_NO_THROW(gen.set_current_seed(std::random_device()()));
  }
  if (at::hasCUDA()) {
    auto t2 = zeros({4, 4}, at::kCUDA);
    cout << &t2 << "\n";
  }
  auto t = ones({4, 4});

  auto wha2 = zeros({4, 4}).add(t).sum();
  ASSERT_EQ(wha2.item<double>(), 16.0);

  ASSERT_EQ(t.sizes()[0], 4);
  ASSERT_EQ(t.sizes()[1], 4);
  ASSERT_EQ(t.strides()[0], 4);
  ASSERT_EQ(t.strides()[1], 1);

  TensorOptions options = dtype(kFloat);
  Tensor x = randn({1, 10}, options);
  Tensor prev_h = randn({1, 20}, options);
  Tensor W_h = randn({20, 20}, options);
  Tensor W_x = randn({20, 10}, options);
  Tensor i2h = at::mm(W_x, x.t());
  Tensor h2h = at::mm(W_h, prev_h.t());
  Tensor next_h = i2h.add(h2h);
  next_h = next_h.tanh();

  // NOLINTNEXTLINE(cppcoreguidelines-avoid-goto,hicpp-avoid-goto)
  ASSERT_ANY_THROW(Tensor{}.item());

  test_overflow();

  if (at::hasCUDA()) {
    auto r = next_h.to(at::Device(kCUDA), kFloat, /*non_blocking=*/ false, /*copy=*/ true);
    ASSERT_TRUE(r.to(at::Device(kCPU), kFloat, /*non_blocking=*/ false, /*copy=*/ true).equal(next_h));
  }
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-goto,hicpp-avoid-goto)
  ASSERT_NO_THROW(randn({10, 10, 2}, options));

  // check Scalar.toTensor on Scalars backed by different data types
  ASSERT_EQ(scalar_to_tensor(bar).scalar_type(), kDouble);
  ASSERT_EQ(scalar_to_tensor(what).scalar_type(), kLong);
  ASSERT_EQ(scalar_to_tensor(ones({}).item()).scalar_type(), kDouble);

  if (x.scalar_type() != ScalarType::Half) {
    AT_DISPATCH_ALL_TYPES(x.scalar_type(), "foo", [&] {
      scalar_t s = 1;
      std::stringstream ss;
      // NOLINTNEXTLINE(cppcoreguidelines-avoid-goto,hicpp-avoid-goto)
      ASSERT_NO_THROW(
          ss << "hello, dispatch" << x.toString() << s << "\n");
      auto data = (scalar_t*)x.data_ptr();
      (void)data;
    });
  }

  // test direct C-scalar type conversions
  {
    auto x = ones({1, 2}, options);
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-goto,hicpp-avoid-goto)
    ASSERT_ANY_THROW(x.item<float>());
  }
  auto float_one = ones({}, options);
  ASSERT_EQ(float_one.item<float>(), 1);
  ASSERT_EQ(float_one.item<int32_t>(), 1);
  ASSERT_EQ(float_one.item<at::Half>(), 1);
}

TEST(TestScalar, TestConj) {
  Scalar int_scalar = 257;
  Scalar float_scalar = 3.0;
  Scalar complex_scalar = c10::complex<double>(2.3, 3.5);

  ASSERT_EQ(int_scalar.conj().toInt(), 257);
  ASSERT_EQ(float_scalar.conj().toDouble(), 3.0);
  ASSERT_EQ(complex_scalar.conj().toComplexDouble(), c10::complex<double>(2.3, -3.5));
}

TEST(TestScalar, TestEqual) {
  ASSERT_FALSE(Scalar(1.0).equal(false));
  ASSERT_FALSE(Scalar(1.0).equal(true));
  ASSERT_FALSE(Scalar(true).equal(1.0));
  ASSERT_TRUE(Scalar(true).equal(true));

  ASSERT_TRUE(Scalar(c10::complex<double>{2.0, 5.0}).equal(c10::complex<double>{2.0, 5.0}));
  ASSERT_TRUE(Scalar(c10::complex<double>{2.0, 0}).equal(2.0));
  ASSERT_TRUE(Scalar(c10::complex<double>{2.0, 0}).equal(2));

  ASSERT_TRUE(Scalar(2.0).equal(c10::complex<double>{2.0, 0.0}));
  ASSERT_FALSE(Scalar(2.0).equal(c10::complex<double>{2.0, 4.0}));
  ASSERT_FALSE(Scalar(2.0).equal(3.0));
  ASSERT_TRUE(Scalar(2.0).equal(2));

  ASSERT_TRUE(Scalar(2).equal(c10::complex<double>{2.0, 0}));
  ASSERT_TRUE(Scalar(2).equal(2));
  ASSERT_TRUE(Scalar(2).equal(2.0));
}

TEST(TestScalar, TestFormatting) {
  auto format = [] (Scalar a) {
    std::ostringstream str;
    str << a;
    return str.str();
  };
  ASSERT_EQ("3", format(Scalar(3)));
  ASSERT_EQ("3.1", format(Scalar(3.1)));
  ASSERT_EQ("true", format(Scalar(true)));
  ASSERT_EQ("false", format(Scalar(false)));
  ASSERT_EQ("(2,3.1)", format(Scalar(c10::complex<double>(2.0, 3.1))));
  ASSERT_EQ("(2,3.1)", format(Scalar(c10::complex<float>(2.0, 3.1))));
  ASSERT_EQ("4", format(Scalar(Scalar(4).toSymInt())));
}

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 4 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `at`

**Classes/Structs**: `Foo`, `Foo`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `aten/src/ATen/test`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



## Dependencies

### Import Dependencies

This file includes:

- `gtest/gtest.h`
- `iostream`
- `random`
- `c10/core/SymInt.h`
- `math.h`
- `ATen/ATen.h`
- `ATen/Dispatch.h`


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.

*Detailed performance analysis requires profiling and benchmarking.*


## Security & Safety

### Security Considerations

- No obvious security concerns detected in automated analysis.

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

This is a test file. Run it with:

```bash
python aten/src/ATen/test/scalar_test.cpp
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

- **File Documentation**: `scalar_test.cpp_docs.md`
- **Keyword Index**: `scalar_test.cpp_kw.md`
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
python docs/aten/src/ATen/test/scalar_test.cpp_docs.md
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
- [`MaybeOwned_test.cpp_docs.md_docs.md`](./MaybeOwned_test.cpp_docs.md_docs.md)
- [`dlconvertor_test.cpp_kw.md_docs.md`](./dlconvertor_test.cpp_kw.md_docs.md)


## Cross-References

- **File Documentation**: `scalar_test.cpp_docs.md_docs.md`
- **Keyword Index**: `scalar_test.cpp_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
