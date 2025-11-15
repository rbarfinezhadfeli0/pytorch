# Documentation: `docs/test/cpp/jit/test_qualified_name.cpp_docs.md`

## File Metadata

- **Path**: `docs/test/cpp/jit/test_qualified_name.cpp_docs.md`
- **Size**: 4,960 bytes (4.84 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**. This appears to be a **test file**.

## Original Source

```markdown
# Documentation: `test/cpp/jit/test_qualified_name.cpp`

## File Metadata

- **Path**: `test/cpp/jit/test_qualified_name.cpp`
- **Size**: 2,332 bytes (2.28 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**.

## Original Source

```cpp
#include <gtest/gtest.h>

#include <ATen/core/qualified_name.h>
#include <c10/util/Exception.h>

using c10::QualifiedName;

namespace torch {
namespace jit {
TEST(QualifiedNameTest, PrefixConstruction) {
  // Test prefix construction
  auto foo = QualifiedName("foo");
  auto bar = QualifiedName(foo, "bar");
  auto baz = QualifiedName(bar, "baz");
  ASSERT_EQ(baz.qualifiedName(), "foo.bar.baz");
  ASSERT_EQ(baz.prefix(), "foo.bar");
  ASSERT_EQ(baz.name(), "baz");
  auto nullstate = QualifiedName();
  ASSERT_EQ(nullstate.qualifiedName(), "");
  ASSERT_EQ(nullstate.prefix(), "");
  ASSERT_EQ(nullstate.name(), "");
}

TEST(QualifiedNameTest, DottedConstruction) {
  // Test dotted construction
  auto foo = QualifiedName("foo.bar.baz");
  ASSERT_EQ(foo.qualifiedName(), "foo.bar.baz");
  ASSERT_EQ(foo.prefix(), "foo.bar");
  ASSERT_EQ(foo.name(), "baz");

  auto bar = QualifiedName("bar");
  ASSERT_EQ(bar.qualifiedName(), "bar");
  ASSERT_EQ(bar.prefix(), "");
  ASSERT_EQ(bar.name(), "bar");
}

TEST(QualifiedNameTest, BadInputRaises) {
  // throw some bad inputs at it
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-goto,hicpp-avoid-goto)
  ASSERT_ANY_THROW(QualifiedName("foo..bar"));
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-goto,hicpp-avoid-goto)
  ASSERT_ANY_THROW(QualifiedName(".foo.bar"));
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-goto,hicpp-avoid-goto)
  ASSERT_ANY_THROW(QualifiedName("foo.bar."));
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-goto,hicpp-avoid-goto)
  ASSERT_ANY_THROW(QualifiedName(""));
}

TEST(QualifiedNameTest, Equality) {
  // test equality api
  auto foo1 = QualifiedName("foo.bar.baz");
  auto foo2 = QualifiedName("foo.bar.baz");
  auto foo3 = QualifiedName("bar.bar.baz");
  ASSERT_EQ(foo1, foo2);
  ASSERT_NE(foo1, foo3);
  auto bar1 = QualifiedName("sup");
  auto bar2 = QualifiedName("sup");
  ASSERT_EQ(foo1, foo2);
}

TEST(QualifiedNameTest, IsPrefixOf) {
  // test prefix api
  auto foo1 = QualifiedName("foo.bar.baz");
  auto foo2 = QualifiedName("foo.bar");
  auto foo3 = QualifiedName("bar.bar.baz");
  auto foo4 = QualifiedName("foo.bar");
  ASSERT_TRUE(foo2.isPrefixOf(foo1));
  ASSERT_TRUE(foo2.isPrefixOf(foo4));
  ASSERT_TRUE(foo4.isPrefixOf(foo2));
  ASSERT_FALSE(foo1.isPrefixOf(foo2));
  ASSERT_FALSE(foo2.isPrefixOf(foo3));
}
} // namespace jit
} // namespace torch

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 0 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `jit`, `torch`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/cpp/jit`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file includes:

- `gtest/gtest.h`
- `ATen/core/qualified_name.h`
- `c10/util/Exception.h`


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

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
python test/cpp/jit/test_qualified_name.cpp
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`test/cpp/jit`):

- [`test_code_template.cpp_docs.md`](./test_code_template.cpp_docs.md)
- [`test_memory_dag.cpp_docs.md`](./test_memory_dag.cpp_docs.md)
- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`test_cleanup_passes.cpp_docs.md`](./test_cleanup_passes.cpp_docs.md)
- [`test_union.cpp_docs.md`](./test_union.cpp_docs.md)
- [`test_subgraph_rewriter.cpp_docs.md`](./test_subgraph_rewriter.cpp_docs.md)
- [`test_backend_compiler_preprocess.cpp_docs.md`](./test_backend_compiler_preprocess.cpp_docs.md)
- [`test_lite_interpreter_direct.cpp_docs.md`](./test_lite_interpreter_direct.cpp_docs.md)
- [`test_save_load.cpp_docs.md`](./test_save_load.cpp_docs.md)
- [`test_module_api.cpp_docs.md`](./test_module_api.cpp_docs.md)


## Cross-References

- **File Documentation**: `test_qualified_name.cpp_docs.md`
- **Keyword Index**: `test_qualified_name.cpp_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/test/cpp/jit`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/test/cpp/jit`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- May involve **JIT compilation** or compilation optimizations.
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
python docs/test/cpp/jit/test_qualified_name.cpp_docs.md
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`docs/test/cpp/jit`):

- [`test_graph_iterator.cpp_kw.md_docs.md`](./test_graph_iterator.cpp_kw.md_docs.md)
- [`test_fuser.cpp_kw.md_docs.md`](./test_fuser.cpp_kw.md_docs.md)
- [`test_utils.cpp_docs.md_docs.md`](./test_utils.cpp_docs.md_docs.md)
- [`test_custom_class_registrations.h_docs.md_docs.md`](./test_custom_class_registrations.h_docs.md_docs.md)
- [`tests_setup.py_docs.md_docs.md`](./tests_setup.py_docs.md_docs.md)
- [`test_exception.cpp_kw.md_docs.md`](./test_exception.cpp_kw.md_docs.md)
- [`test_cs_debug_info_serialization.cpp_docs.md_docs.md`](./test_cs_debug_info_serialization.cpp_docs.md_docs.md)
- [`torch_python_test.cpp_docs.md_docs.md`](./torch_python_test.cpp_docs.md_docs.md)
- [`test_backend_compiler_preprocess.cpp_docs.md_docs.md`](./test_backend_compiler_preprocess.cpp_docs.md_docs.md)


## Cross-References

- **File Documentation**: `test_qualified_name.cpp_docs.md_docs.md`
- **Keyword Index**: `test_qualified_name.cpp_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
