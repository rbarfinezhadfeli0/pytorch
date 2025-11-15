# Documentation: `test/cpp/aoti_abi_check/test_dtype.cpp`

## File Metadata

- **Path**: `test/cpp/aoti_abi_check/test_dtype.cpp`
- **Size**: 6,448 bytes (6.30 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**.

## Original Source

```cpp
#include <gtest/gtest.h>

#include <torch/headeronly/core/ScalarType.h>
#include <torch/headeronly/util/BFloat16.h>
#include <torch/headeronly/util/Float4_e2m1fn_x2.h>
#include <torch/headeronly/util/Float8_e4m3fn.h>
#include <torch/headeronly/util/Float8_e4m3fnuz.h>
#include <torch/headeronly/util/Float8_e5m2.h>
#include <torch/headeronly/util/Float8_e5m2fnuz.h>
#include <torch/headeronly/util/Float8_e8m0fnu.h>
#include <torch/headeronly/util/Half.h>
#include <torch/headeronly/util/bits.h>
#include <torch/headeronly/util/complex.h>
#include <torch/headeronly/util/qint32.h>
#include <torch/headeronly/util/qint8.h>
#include <torch/headeronly/util/quint2x4.h>
#include <torch/headeronly/util/quint4x2.h>
#include <torch/headeronly/util/quint8.h>

TEST(TestDtype, TestBFloat16) {
  torch::headeronly::BFloat16 a = 1.0f;
  torch::headeronly::BFloat16 b = 2.0f;
  torch::headeronly::BFloat16 add = 3.0f;
  torch::headeronly::BFloat16 sub = -1.0f;
  torch::headeronly::BFloat16 mul = 2.0f;
  torch::headeronly::BFloat16 div = 0.5f;

  EXPECT_EQ(a + b, add);
  EXPECT_EQ(a - b, sub);
  EXPECT_EQ(a * b, mul);
  EXPECT_EQ(a / b, div);
}

TEST(TestDtype, TestFloat8_e4m3fn) {
  torch::headeronly::Float8_e4m3fn a = 1.0f;
  torch::headeronly::Float8_e4m3fn b = 2.0f;
  torch::headeronly::Float8_e4m3fn add = 3.0f;
  torch::headeronly::Float8_e4m3fn sub = -1.0f;
  torch::headeronly::Float8_e4m3fn mul = 2.0f;
  torch::headeronly::Float8_e4m3fn div = 0.5f;

  EXPECT_EQ(a + b, add);
  EXPECT_EQ(a - b, sub);
  EXPECT_EQ(a * b, mul);
  EXPECT_EQ(a / b, div);
}

TEST(TestDtype, TestFloat8_e4m3fuz) {
  torch::headeronly::Float8_e4m3fnuz a = 1.0f;
  torch::headeronly::Float8_e4m3fnuz b = 2.0f;
  torch::headeronly::Float8_e4m3fnuz add = 3.0f;
  torch::headeronly::Float8_e4m3fnuz sub = -1.0f;
  torch::headeronly::Float8_e4m3fnuz mul = 2.0f;
  torch::headeronly::Float8_e4m3fnuz div = 0.5f;

  EXPECT_EQ(a + b, add);
  EXPECT_EQ(a - b, sub);
  EXPECT_EQ(a * b, mul);
  EXPECT_EQ(a / b, div);
}

TEST(TestDtype, TestFloat8_e5m2) {
  torch::headeronly::Float8_e5m2 a = 1.0f;
  torch::headeronly::Float8_e5m2 b = 2.0f;
  torch::headeronly::Float8_e5m2 add = 3.0f;
  torch::headeronly::Float8_e5m2 sub = -1.0f;
  torch::headeronly::Float8_e5m2 mul = 2.0f;
  torch::headeronly::Float8_e5m2 div = 0.5f;

  EXPECT_EQ(a + b, add);
  EXPECT_EQ(a - b, sub);
  EXPECT_EQ(a * b, mul);
  EXPECT_EQ(a / b, div);
}

TEST(TestDtype, TestFloat8_e5m2fnuz) {
  torch::headeronly::Float8_e5m2fnuz a = 1.0f;
  torch::headeronly::Float8_e5m2fnuz b = 2.0f;
  torch::headeronly::Float8_e5m2fnuz add = 3.0f;
  torch::headeronly::Float8_e5m2fnuz sub = -1.0f;
  torch::headeronly::Float8_e5m2fnuz mul = 2.0f;
  torch::headeronly::Float8_e5m2fnuz div = 0.5f;

  EXPECT_EQ(a + b, add);
  EXPECT_EQ(a - b, sub);
  EXPECT_EQ(a * b, mul);
  EXPECT_EQ(a / b, div);
}

TEST(TestDtype, TestFloat8_e8m0fnu) {
  torch::headeronly::Float8_e8m0fnu a = 1.0f;
  ASSERT_FALSE(a.isnan());
}

TEST(TestDtype, TestFloat4) {
  // not much you can do with this type, just make sure it compiles
  torch::headeronly::Float4_e2m1fn_x2 a(5);
}

TEST(TestDtype, TestHalf) {
  torch::headeronly::Half a = 1.0f;
  torch::headeronly::Half b = 2.0f;
  torch::headeronly::Half add = 3.0f;
  torch::headeronly::Half sub = -1.0f;
  torch::headeronly::Half mul = 2.0f;
  torch::headeronly::Half div = 0.5f;

  EXPECT_EQ(a + b, add);
  EXPECT_EQ(a - b, sub);
  EXPECT_EQ(a * b, mul);
  EXPECT_EQ(a / b, div);
  EXPECT_EQ(a += b, add);
  EXPECT_EQ(a -= b, add - b);
  EXPECT_EQ(a *= b, b);
  EXPECT_EQ(a /= b, mul * div);

#if defined(__aarch64__) && !defined(__CUDACC__)
  EXPECT_EQ(
      torch::headeronly::detail::fp16_to_bits(
          torch::headeronly::detail::fp16_from_bits(32)),
      32);
#endif
}

TEST(TestDtype, TestComplexFloat) {
  torch::headeronly::complex<float> a(std::complex<float>(1.0f, 2.0f));
  torch::headeronly::complex<float> b(std::complex<float>(3.0f, 4.0f));
  torch::headeronly::complex<float> add(std::complex<float>(4.0f, 6.0f));
  torch::headeronly::complex<float> sub(std::complex<float>(-2.0f, -2.0f));
  torch::headeronly::complex<float> mul(std::complex<float>(-5.0f, 10.0f));
  torch::headeronly::complex<float> div(std::complex<float>(0.44f, 0.08f));

  EXPECT_EQ(a + b, add);
  EXPECT_EQ(a - b, sub);
  EXPECT_EQ(a * b, mul);
  EXPECT_EQ(a / b, div);
}

TEST(TestDtype, TestQuintsQintsAndBits) {
  // There's not much you can do with these dtypes...
  // so we'll just check that it compiles
  auto a = torch::headeronly::quint8(0);
  auto b = torch::headeronly::quint4x2(5);
  auto c = torch::headeronly::quint2x4(1);
  auto d = torch::headeronly::qint32(5);
  auto e = torch::headeronly::qint8(1);
  auto f = torch::headeronly::bits1x8(9);
  auto g = torch::headeronly::bits2x4(9);
  auto h = torch::headeronly::bits4x2(9);
  auto i = torch::headeronly::bits8(2);
  auto j = torch::headeronly::bits16(6);
}

TEST(TestDtype, TestScalarType) {
  using torch::headeronly::ScalarType;
  constexpr ScalarType expected_scalar_types[] = {
      ScalarType::Byte,
      ScalarType::Char,
      ScalarType::Short,
      ScalarType::Int,
      ScalarType::Long,
      ScalarType::Half,
      ScalarType::Float,
      ScalarType::Double,
      ScalarType::ComplexHalf,
      ScalarType::ComplexFloat,
      ScalarType::ComplexDouble,
      ScalarType::Bool,
      ScalarType::QInt8,
      ScalarType::QUInt8,
      ScalarType::QInt32,
      ScalarType::BFloat16,
      ScalarType::QUInt4x2,
      ScalarType::QUInt2x4,
      ScalarType::Bits1x8,
      ScalarType::Bits2x4,
      ScalarType::Bits4x2,
      ScalarType::Bits8,
      ScalarType::Bits16,
      ScalarType::Float8_e5m2,
      ScalarType::Float8_e4m3fn,
      ScalarType::Float8_e5m2fnuz,
      ScalarType::Float8_e4m3fnuz,
      ScalarType::UInt16,
      ScalarType::UInt32,
      ScalarType::UInt64,
      ScalarType::UInt1,
      ScalarType::UInt2,
      ScalarType::UInt3,
      ScalarType::UInt4,
      ScalarType::UInt5,
      ScalarType::UInt6,
      ScalarType::UInt7,
      ScalarType::Int1,
      ScalarType::Int2,
      ScalarType::Int3,
      ScalarType::Int4,
      ScalarType::Int5,
      ScalarType::Int6,
      ScalarType::Int7,
      ScalarType::Float8_e8m0fnu,
      ScalarType::Float4_e2m1fn_x2,
      ScalarType::Undefined,
  };
  for (int8_t i = 0; i < static_cast<int8_t>(torch::headeronly::NumScalarTypes);
       i++) {
    EXPECT_EQ(static_cast<ScalarType>(i), expected_scalar_types[i]);
  }
}

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 2 function(s).

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/cpp/aoti_abi_check`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file includes:

- `gtest/gtest.h`
- `torch/headeronly/core/ScalarType.h`
- `torch/headeronly/util/BFloat16.h`
- `torch/headeronly/util/Float4_e2m1fn_x2.h`
- `torch/headeronly/util/Float8_e4m3fn.h`
- `torch/headeronly/util/Float8_e4m3fnuz.h`
- `torch/headeronly/util/Float8_e5m2.h`
- `torch/headeronly/util/Float8_e5m2fnuz.h`
- `torch/headeronly/util/Float8_e8m0fnu.h`
- `torch/headeronly/util/Half.h`
- `torch/headeronly/util/bits.h`
- `torch/headeronly/util/complex.h`
- `torch/headeronly/util/qint32.h`
- `torch/headeronly/util/qint8.h`
- `torch/headeronly/util/quint2x4.h`
- `torch/headeronly/util/quint4x2.h`
- `torch/headeronly/util/quint8.h`


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
- May involve **JIT compilation** or compilation optimizations.

*Detailed performance analysis requires profiling and benchmarking.*


## Security & Safety

### Security Considerations

- No obvious security concerns detected in automated analysis.

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

This is a test file. Run it with:

```bash
python test/cpp/aoti_abi_check/test_dtype.cpp
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`test/cpp/aoti_abi_check`):

- [`test_metaprogramming.cpp_docs.md`](./test_metaprogramming.cpp_docs.md)
- [`test_headeronlyarrayref.cpp_docs.md`](./test_headeronlyarrayref.cpp_docs.md)
- [`test_cast.cpp_docs.md`](./test_cast.cpp_docs.md)
- [`test_scalartype.cpp_docs.md`](./test_scalartype.cpp_docs.md)
- [`CMakeLists.txt_docs.md`](./CMakeLists.txt_docs.md)
- [`test_typetraits.cpp_docs.md`](./test_typetraits.cpp_docs.md)
- [`test_math.cpp_docs.md`](./test_math.cpp_docs.md)
- [`test_dispatch.cpp_docs.md`](./test_dispatch.cpp_docs.md)
- [`test_exception.cpp_docs.md`](./test_exception.cpp_docs.md)


## Cross-References

- **File Documentation**: `test_dtype.cpp_docs.md`
- **Keyword Index**: `test_dtype.cpp_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
