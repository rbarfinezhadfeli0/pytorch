# Documentation: `docs/test/cpp/jit/test_jit_logging_levels.cpp_docs.md`

## File Metadata

- **Path**: `docs/test/cpp/jit/test_jit_logging_levels.cpp_docs.md`
- **Size**: 4,630 bytes (4.52 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**. This appears to be a **test file**.

## Original Source

```markdown
# Documentation: `test/cpp/jit/test_jit_logging_levels.cpp`

## File Metadata

- **Path**: `test/cpp/jit/test_jit_logging_levels.cpp`
- **Size**: 1,967 bytes (1.92 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**.

## Original Source

```cpp
#include <gtest/gtest.h>
#include <test/cpp/jit/test_utils.h>

#include <torch/csrc/jit/jit_log.h>
#include <sstream>

namespace torch {
namespace jit {

TEST(JitLoggingTest, CheckSetLoggingLevel) {
  ::torch::jit::set_jit_logging_levels("file_to_test");
  ASSERT_TRUE(::torch::jit::is_enabled(
      "file_to_test.cpp", JitLoggingLevels::GRAPH_DUMP));
}

TEST(JitLoggingTest, CheckSetMultipleLogLevels) {
  ::torch::jit::set_jit_logging_levels("f1:>f2:>>f3");
  ASSERT_TRUE(::torch::jit::is_enabled("f1.cpp", JitLoggingLevels::GRAPH_DUMP));
  ASSERT_TRUE(
      ::torch::jit::is_enabled("f2.cpp", JitLoggingLevels::GRAPH_UPDATE));
  ASSERT_TRUE(
      ::torch::jit::is_enabled("f3.cpp", JitLoggingLevels::GRAPH_DEBUG));
}

TEST(JitLoggingTest, CheckLoggingLevelAfterUnset) {
  ::torch::jit::set_jit_logging_levels("f1");
  ASSERT_EQ("f1", ::torch::jit::get_jit_logging_levels());
  ::torch::jit::set_jit_logging_levels("invalid");
  ASSERT_FALSE(
      ::torch::jit::is_enabled("f1.cpp", JitLoggingLevels::GRAPH_DUMP));
}

TEST(JitLoggingTest, CheckAfterChangingLevel) {
  ::torch::jit::set_jit_logging_levels("f1");
  ::torch::jit::set_jit_logging_levels(">f1");
  ASSERT_TRUE(
      ::torch::jit::is_enabled("f1.cpp", JitLoggingLevels::GRAPH_UPDATE));
}

TEST(JitLoggingTest, CheckOutputStreamSetting) {
  ::torch::jit::set_jit_logging_levels("test_jit_logging_levels");
  std::ostringstream test_stream;
  ::torch::jit::set_jit_logging_output_stream(test_stream);
  /* Using JIT_LOG checks if this file has logging enabled with
    is_enabled(__FILE__, level) making the test fail. since we are only testing
    the OutputStreamSetting we can forcefully output to it directly.
  */
  ::torch::jit::get_jit_logging_output_stream() << ::torch::jit::jit_log_prefix(
      ::torch::jit::JitLoggingLevels::GRAPH_DUMP,
      __FILE__,
      __LINE__,
      ::c10::str("Message"));
  ASSERT_TRUE(test_stream.str().size() > 0);
}

} // namespace jit
} // namespace torch

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 1 function(s).

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
- `test/cpp/jit/test_utils.h`
- `torch/csrc/jit/jit_log.h`
- `sstream`


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
python test/cpp/jit/test_jit_logging_levels.cpp
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

- **File Documentation**: `test_jit_logging_levels.cpp_docs.md`
- **Keyword Index**: `test_jit_logging_levels.cpp_kw.md`
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
python docs/test/cpp/jit/test_jit_logging_levels.cpp_docs.md
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

- **File Documentation**: `test_jit_logging_levels.cpp_docs.md_docs.md`
- **Keyword Index**: `test_jit_logging_levels.cpp_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
