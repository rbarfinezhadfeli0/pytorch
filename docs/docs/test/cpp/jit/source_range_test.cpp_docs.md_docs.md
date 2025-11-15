# Documentation: `docs/test/cpp/jit/source_range_test.cpp_docs.md`

## File Metadata

- **Path**: `docs/test/cpp/jit/source_range_test.cpp_docs.md`
- **Size**: 4,849 bytes (4.74 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**. This appears to be a **test file**.

## Original Source

```markdown
# Documentation: `test/cpp/jit/source_range_test.cpp`

## File Metadata

- **Path**: `test/cpp/jit/source_range_test.cpp`
- **Size**: 2,254 bytes (2.20 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**.

## Original Source

```cpp
#include <gtest/gtest.h>

#include <c10/util/irange.h>
#include <torch/csrc/jit/frontend/source_range.h>

using namespace ::testing;
using namespace ::torch::jit;

std::vector<StringCordView> sampleStringCordViews() {
  std::vector<StringCordView> result;

  std::vector<std::shared_ptr<std::string>> strings;
  strings.push_back(std::make_shared<std::string>("hello world"));
  strings.push_back(std::make_shared<std::string>("nihaoma"));

  std::vector<std::string_view> pieces{*strings[0], *strings[1]};

  result.emplace_back(std::move(pieces), std::move(strings));
  pieces = {"hello worldnihaoma"};
  strings.clear();
  result.emplace_back(std::move(pieces), std::move(strings));
  return result;
}

TEST(SourceRangeTest, test_find) {
  for (const auto& view : sampleStringCordViews()) {
    auto x = view.find("rldni", 0);
    EXPECT_EQ(x, 8) << view.str();
    EXPECT_EQ(view.find("ello", 0), 1);
  }
}

TEST(SourceRangeTest, test_substr) {
  for (const auto& view : sampleStringCordViews()) {
    auto x = view.substr(4, 10).str();
    EXPECT_EQ(x, view.str().substr(4, 10));
    EXPECT_EQ(view.substr(0, view.size()).str(), view.str());
    for (const auto start : c10::irange(view.size())) {
      for (const auto size : c10::irange(view.size())) {
        EXPECT_EQ(
            view.substr(start, size).str(), view.str().substr(start, size));
      }
    }
  }
}

TEST(SourceRangeTest, test_iter_simple) {
  for (const auto& view : sampleStringCordViews()) {
    EXPECT_NE(view.begin(), view.end());
    EXPECT_TRUE(view.begin().has_next());
    EXPECT_EQ(view.str(), std::string(view.begin(), view.end()));
  }
}

TEST(SourceRangeTest, test_iter) {
  int idx = 0;
  for (const auto& view : sampleStringCordViews()) {
    auto iter = view.iter_for_pos(5);
    EXPECT_EQ(*iter, ' ');
    if (idx++ == 0) {
      EXPECT_EQ(iter.rest_line(), " world");
    } else {
      EXPECT_EQ(iter.rest_line(), " worldnihaoma");
    }
    EXPECT_EQ(*iter.next_iter(), 'w');
    EXPECT_EQ(iter.pos(), 5);
    iter = view.iter_for_pos(13);
    EXPECT_EQ(iter.pos(), 13);
  }
}

TEST(SourceRangeTest, SimpleString) {
  Source src("hello");
  EXPECT_EQ(src.num_lines(), 1);
  EXPECT_EQ(src.get_line(0), "hello");
  EXPECT_EQ(src.text_str().str(), "hello");
}

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 1 function(s).

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/cpp/jit`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file includes:

- `gtest/gtest.h`
- `c10/util/irange.h`
- `torch/csrc/jit/frontend/source_range.h`


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
python test/cpp/jit/source_range_test.cpp
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

- **File Documentation**: `source_range_test.cpp_docs.md`
- **Keyword Index**: `source_range_test.cpp_kw.md`
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
python docs/test/cpp/jit/source_range_test.cpp_docs.md
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`docs/test/cpp/jit`):

- [`test_graph_iterator.cpp_kw.md_docs.md`](./test_graph_iterator.cpp_kw.md_docs.md)
- [`test_qualified_name.cpp_docs.md_docs.md`](./test_qualified_name.cpp_docs.md_docs.md)
- [`test_fuser.cpp_kw.md_docs.md`](./test_fuser.cpp_kw.md_docs.md)
- [`test_utils.cpp_docs.md_docs.md`](./test_utils.cpp_docs.md_docs.md)
- [`test_custom_class_registrations.h_docs.md_docs.md`](./test_custom_class_registrations.h_docs.md_docs.md)
- [`tests_setup.py_docs.md_docs.md`](./tests_setup.py_docs.md_docs.md)
- [`test_exception.cpp_kw.md_docs.md`](./test_exception.cpp_kw.md_docs.md)
- [`test_cs_debug_info_serialization.cpp_docs.md_docs.md`](./test_cs_debug_info_serialization.cpp_docs.md_docs.md)
- [`torch_python_test.cpp_docs.md_docs.md`](./torch_python_test.cpp_docs.md_docs.md)
- [`test_backend_compiler_preprocess.cpp_docs.md_docs.md`](./test_backend_compiler_preprocess.cpp_docs.md_docs.md)


## Cross-References

- **File Documentation**: `source_range_test.cpp_docs.md_docs.md`
- **Keyword Index**: `source_range_test.cpp_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
