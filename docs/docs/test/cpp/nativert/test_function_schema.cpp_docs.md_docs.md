# Documentation: `docs/test/cpp/nativert/test_function_schema.cpp_docs.md`

## File Metadata

- **Path**: `docs/test/cpp/nativert/test_function_schema.cpp_docs.md`
- **Size**: 4,897 bytes (4.78 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**. This appears to be a **test file**.

## Original Source

```markdown
# Documentation: `test/cpp/nativert/test_function_schema.cpp`

## File Metadata

- **Path**: `test/cpp/nativert/test_function_schema.cpp`
- **Size**: 2,272 bytes (2.22 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**.

## Original Source

```cpp
#include <gtest/gtest.h>

#include <ATen/core/dispatch/Dispatcher.h>
#include <ATen/core/op_registration/op_registration.h>
#include <torch/nativert/executor/memory/FunctionSchema.h>

using namespace ::testing;

int64_t increment_kernel(const at::Tensor& tensor, int64_t input) {
  return input + 1;
}

at::Tensor slice_kernel(const at::Tensor& tensor, int64_t dim) {
  return tensor.slice(dim);
}

TEST(TestFunctionSchema, testNoAlias) {
  auto registrar = c10::RegisterOperators().op(
      "_test::my_op(Tensor dummy, int input) -> int", &increment_kernel);
  auto handle = c10::Dispatcher::singleton().findSchema({"_test::my_op", ""});

  EXPECT_TRUE(handle.has_value());
  EXPECT_TRUE(handle->hasSchema());

  auto nativert_schema = torch::nativert::FunctionSchema(handle->schema());

  EXPECT_FALSE(nativert_schema.alias(0, 0));
  EXPECT_FALSE(nativert_schema.alias(1, 0));

  // bounds check
  EXPECT_THROW(nativert_schema.alias(2, 0), c10::Error);
  EXPECT_THROW(nativert_schema.alias(1, 1), c10::Error);
}

TEST(TestFunctionSchema, testAliasOverride) {
  auto registrar = c10::RegisterOperators().op(
      "_test::my_op(Tensor dummy, int input) -> int", &increment_kernel);
  auto handle = c10::Dispatcher::singleton().findSchema({"_test::my_op", ""});

  EXPECT_TRUE(handle.has_value());
  EXPECT_TRUE(handle->hasSchema());

  auto nativert_schema =
      torch::nativert::FunctionSchema(handle->schema(), {{0, 0}});

  EXPECT_TRUE(nativert_schema.alias(0, 0));
  EXPECT_FALSE(nativert_schema.alias(1, 0));

  // bounds check
  EXPECT_THROW(nativert_schema.alias(2, 0), c10::Error);
  EXPECT_THROW(nativert_schema.alias(1, 1), c10::Error);
}

TEST(TestFunctionSchema, testAlias) {
  auto registrar = c10::RegisterOperators().op(
      "_test::my_op(Tensor(a) dummy, int input) -> Tensor(a)", &slice_kernel);
  auto handle = c10::Dispatcher::singleton().findSchema({"_test::my_op", ""});

  EXPECT_TRUE(handle.has_value());
  EXPECT_TRUE(handle->hasSchema());

  auto nativert_schema = torch::nativert::FunctionSchema(handle->schema());

  EXPECT_TRUE(nativert_schema.alias(0, 0));
  EXPECT_FALSE(nativert_schema.alias(1, 0));

  // bounds check
  EXPECT_THROW(nativert_schema.alias(2, 0), c10::Error);
  EXPECT_THROW(nativert_schema.alias(1, 1), c10::Error);
}

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 5 function(s).

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/cpp/nativert`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file includes:

- `gtest/gtest.h`
- `ATen/core/dispatch/Dispatcher.h`
- `ATen/core/op_registration/op_registration.h`
- `torch/nativert/executor/memory/FunctionSchema.h`


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
python test/cpp/nativert/test_function_schema.cpp
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`test/cpp/nativert`):

- [`test_alias_analyzer.cpp_docs.md`](./test_alias_analyzer.cpp_docs.md)
- [`test_placement.cpp_docs.md`](./test_placement.cpp_docs.md)
- [`test_static_kernel_ops.cpp_docs.md`](./test_static_kernel_ops.cpp_docs.md)
- [`test_static_dispatch_kernel_registration.cpp_docs.md`](./test_static_dispatch_kernel_registration.cpp_docs.md)
- [`test_graph.cpp_docs.md`](./test_graph.cpp_docs.md)
- [`test_c10_kernel.cpp_docs.md`](./test_c10_kernel.cpp_docs.md)
- [`test_execution_frame.cpp_docs.md`](./test_execution_frame.cpp_docs.md)
- [`test_triton_kernel_manager_registration.cpp_docs.md`](./test_triton_kernel_manager_registration.cpp_docs.md)
- [`CMakeLists.txt_docs.md`](./CMakeLists.txt_docs.md)


## Cross-References

- **File Documentation**: `test_function_schema.cpp_docs.md`
- **Keyword Index**: `test_function_schema.cpp_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/test/cpp/nativert`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/test/cpp/nativert`, which is part of the **testing infrastructure**.



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
python docs/test/cpp/nativert/test_function_schema.cpp_docs.md
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`docs/test/cpp/nativert`):

- [`test_execution_frame.cpp_kw.md_docs.md`](./test_execution_frame.cpp_kw.md_docs.md)
- [`test_tensor_meta.cpp_kw.md_docs.md`](./test_tensor_meta.cpp_kw.md_docs.md)
- [`test_graph_signature.cpp_kw.md_docs.md`](./test_graph_signature.cpp_kw.md_docs.md)
- [`CMakeLists.txt_docs.md_docs.md`](./CMakeLists.txt_docs.md_docs.md)
- [`test_static_kernel_ops.cpp_kw.md_docs.md`](./test_static_kernel_ops.cpp_kw.md_docs.md)
- [`test_layout_planner_algorithm.cpp_docs.md_docs.md`](./test_layout_planner_algorithm.cpp_docs.md_docs.md)
- [`test_pass_manager.cpp_docs.md_docs.md`](./test_pass_manager.cpp_docs.md_docs.md)
- [`test_static_dispatch_kernel_registration.cpp_kw.md_docs.md`](./test_static_dispatch_kernel_registration.cpp_kw.md_docs.md)
- [`test_placement.cpp_kw.md_docs.md`](./test_placement.cpp_kw.md_docs.md)
- [`test_static_kernel_ops.cpp_docs.md_docs.md`](./test_static_kernel_ops.cpp_docs.md_docs.md)


## Cross-References

- **File Documentation**: `test_function_schema.cpp_docs.md_docs.md`
- **Keyword Index**: `test_function_schema.cpp_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
