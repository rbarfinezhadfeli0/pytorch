# Documentation: `docs/test/cpp/aoti_abi_check/test_scalartype.cpp_docs.md`

## File Metadata

- **Path**: `docs/test/cpp/aoti_abi_check/test_scalartype.cpp_docs.md`
- **Size**: 5,968 bytes (5.83 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**. This appears to be a **test file**.

## Original Source

```markdown
# Documentation: `test/cpp/aoti_abi_check/test_scalartype.cpp`

## File Metadata

- **Path**: `test/cpp/aoti_abi_check/test_scalartype.cpp`
- **Size**: 3,567 bytes (3.48 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**.

## Original Source

```cpp
#include <gtest/gtest.h>

#include <torch/headeronly/core/ScalarType.h>

TEST(TestScalarType, ScalarTypeToCPPTypeT) {
  using torch::headeronly::ScalarType;
  using torch::headeronly::impl::ScalarTypeToCPPTypeT;

#define DEFINE_CHECK(TYPE, SCALARTYPE) \
  EXPECT_EQ(typeid(ScalarTypeToCPPTypeT<ScalarType::SCALARTYPE>), typeid(TYPE));

  AT_FORALL_SCALAR_TYPES_WITH_COMPLEX_AND_QINTS(DEFINE_CHECK);
#undef DEFINE_CHECK
}

TEST(TestScalarType, CppTypeToScalarType) {
  using torch::headeronly::CppTypeToScalarType;
  using torch::headeronly::ScalarType;

#define DEFINE_CHECK(TYPE, SCALARTYPE) \
  EXPECT_EQ(CppTypeToScalarType<TYPE>::value, ScalarType::SCALARTYPE);

  AT_FORALL_SCALAR_TYPES_WITH_COMPLEX_AND_QINTS(DEFINE_CHECK);
#undef DEFINE_CHECK
}

#define DEFINE_CHECK(TYPE, SCALARTYPE)                                       \
  {                                                                          \
    EXPECT_EQ(                                                               \
        typeid(ScalarTypeToCPPTypeT<ScalarType::SCALARTYPE>), typeid(TYPE)); \
    count++;                                                                 \
  }

#define TEST_FORALL(M, EXPECTEDCOUNT, ...)               \
  TEST(TestScalarType, M) {                              \
    using torch::headeronly::ScalarType;                 \
    using torch::headeronly::impl::ScalarTypeToCPPTypeT; \
    int8_t count = 0;                                    \
    M(__VA_ARGS__ DEFINE_CHECK);                         \
    EXPECT_EQ(count, EXPECTEDCOUNT);                     \
  }

TEST_FORALL(AT_FORALL_SCALAR_TYPES_WITH_COMPLEX_EXCEPT_COMPLEX_HALF_F8NZ, 14)
TEST_FORALL(AT_FORALL_SCALAR_TYPES_WITH_COMPLEX, 18)
TEST_FORALL(AT_FORALL_SCALAR_TYPES_WITH_COMPLEX_AND_QINTS, 46)
TEST_FORALL(AT_FORALL_INT_TYPES, 5)
TEST_FORALL(AT_FORALL_SCALAR_TYPES, 7)
TEST_FORALL(AT_FORALL_SCALAR_TYPES_AND, 8, Bool, )
TEST_FORALL(AT_FORALL_SCALAR_TYPES_AND2, 9, Bool, Half, )
TEST_FORALL(AT_FORALL_SCALAR_TYPES_AND3, 10, Bool, Half, ComplexFloat, )
TEST_FORALL(
    AT_FORALL_SCALAR_TYPES_AND7,
    14,
    Bool,
    Half,
    ComplexHalf,
    ComplexFloat,
    ComplexDouble,
    UInt16,
    UInt32, )
TEST_FORALL(AT_FORALL_QINT_TYPES, 5)
TEST_FORALL(AT_FORALL_FLOAT8_TYPES, 5)
TEST_FORALL(AT_FORALL_COMPLEX_TYPES, 2)

#undef DEFINE_CHECK
#undef TEST_FORALL

TEST(TestScalarType, toString) {
  using torch::headeronly::ScalarType;

#define DEFINE_CHECK(_, name) EXPECT_EQ(toString(ScalarType::name), #name);
  AT_FORALL_SCALAR_TYPES_WITH_COMPLEX_AND_QINTS(DEFINE_CHECK);
#undef DEFINE_CHECK
}

TEST(TestScalarType, operator_left_shift) {
  using torch::headeronly::ScalarType;

#define DEFINE_CHECK(_, name)   \
  {                             \
    std::stringstream ss;       \
    ss << ScalarType::name;     \
    EXPECT_EQ(ss.str(), #name); \
  }
  AT_FORALL_SCALAR_TYPES_WITH_COMPLEX_AND_QINTS(DEFINE_CHECK);
#undef DEFINE_CHECK
}

TEST(TestScalarType, toUnderlying) {
  using torch::headeronly::ScalarType;
  using torch::headeronly::toUnderlying;

  EXPECT_EQ(toUnderlying(ScalarType::QUInt8), ScalarType::Byte);
  EXPECT_EQ(toUnderlying(ScalarType::QUInt4x2), ScalarType::Byte);
  EXPECT_EQ(toUnderlying(ScalarType::QUInt2x4), ScalarType::Byte);
  EXPECT_EQ(toUnderlying(ScalarType::QInt8), ScalarType::Char);
  EXPECT_EQ(toUnderlying(ScalarType::QInt32), ScalarType::Int);
#define DEFINE_CHECK(_, name) \
  EXPECT_EQ(toUnderlying(ScalarType::name), ScalarType::name);
  AT_FORALL_SCALAR_TYPES_WITH_COMPLEX(DEFINE_CHECK);
  AT_FORALL_FLOAT8_TYPES(DEFINE_CHECK);
#undef DEFINE_CHECK
}

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 8 function(s).

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
python test/cpp/aoti_abi_check/test_scalartype.cpp
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`test/cpp/aoti_abi_check`):

- [`test_metaprogramming.cpp_docs.md`](./test_metaprogramming.cpp_docs.md)
- [`test_headeronlyarrayref.cpp_docs.md`](./test_headeronlyarrayref.cpp_docs.md)
- [`test_cast.cpp_docs.md`](./test_cast.cpp_docs.md)
- [`CMakeLists.txt_docs.md`](./CMakeLists.txt_docs.md)
- [`test_typetraits.cpp_docs.md`](./test_typetraits.cpp_docs.md)
- [`test_dtype.cpp_docs.md`](./test_dtype.cpp_docs.md)
- [`test_math.cpp_docs.md`](./test_math.cpp_docs.md)
- [`test_dispatch.cpp_docs.md`](./test_dispatch.cpp_docs.md)
- [`test_exception.cpp_docs.md`](./test_exception.cpp_docs.md)


## Cross-References

- **File Documentation**: `test_scalartype.cpp_docs.md`
- **Keyword Index**: `test_scalartype.cpp_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/test/cpp/aoti_abi_check`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/test/cpp/aoti_abi_check`, which is part of the **testing infrastructure**.



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

This is a test file. Run it with:

```bash
python docs/test/cpp/aoti_abi_check/test_scalartype.cpp_docs.md
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`docs/test/cpp/aoti_abi_check`):

- [`test_exception.cpp_kw.md_docs.md`](./test_exception.cpp_kw.md_docs.md)
- [`test_metaprogramming.cpp_docs.md_docs.md`](./test_metaprogramming.cpp_docs.md_docs.md)
- [`CMakeLists.txt_docs.md_docs.md`](./CMakeLists.txt_docs.md_docs.md)
- [`test_math.cpp_docs.md_docs.md`](./test_math.cpp_docs.md_docs.md)
- [`test_dtype.cpp_kw.md_docs.md`](./test_dtype.cpp_kw.md_docs.md)
- [`test_typetraits.cpp_docs.md_docs.md`](./test_typetraits.cpp_docs.md_docs.md)
- [`README.md_docs.md_docs.md`](./README.md_docs.md_docs.md)
- [`test_exception.cpp_docs.md_docs.md`](./test_exception.cpp_docs.md_docs.md)
- [`test_headeronlyarrayref.cpp_kw.md_docs.md`](./test_headeronlyarrayref.cpp_kw.md_docs.md)
- [`test_rand.cpp_kw.md_docs.md`](./test_rand.cpp_kw.md_docs.md)


## Cross-References

- **File Documentation**: `test_scalartype.cpp_docs.md_docs.md`
- **Keyword Index**: `test_scalartype.cpp_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
