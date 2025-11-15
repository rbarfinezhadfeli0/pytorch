# Documentation: `test/cpp/jit/test_schema_matching.cpp`

## File Metadata

- **Path**: `test/cpp/jit/test_schema_matching.cpp`
- **Size**: 2,319 bytes (2.26 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**.

## Original Source

```cpp
#include <gtest/gtest.h>

#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/runtime/custom_operator.h>
#include <torch/csrc/jit/testing/file_check.h>
#include <torch/jit.h>

#include <sstream>
#include <string>

namespace torch {
namespace jit {

TEST(SchemaMatchingTest, VarType) {
  RegisterOperators reg({
      Operator(
          "aten::test_vartype(t[] a, t b) -> (t)",
          [](Stack& stack) {
            c10::List<double> list;
            // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
            double a;
            pop(stack, list, a);
            push(stack, a);
          },
          c10::AliasAnalysisKind::FROM_SCHEMA),
  });
  Module m("m");
  m.define(R"(
      def test(self):
        a = (1.0, 2.0)
        return torch.test_vartype(a, 2.0)
    )");
  auto result = m.run_method("test");
  TORCH_INTERNAL_ASSERT(result.toDouble() == 2.0);

  const std::string error_example = R"JIT(
      def test_2(self):
          a = (1.0, 2.0)
          non_float = (1, 1)
          return torch.test_vartype(a, non_float)
    )JIT";

  std::string err = "";
  try {
    m.define(error_example);
  } catch (const std::exception& e) {
    err = e.what();
  }
  TORCH_INTERNAL_ASSERT(
      err.find("previously matched to type") != std::string::npos);
}

TEST(SchemaMatchingTest, VarType2) {
  RegisterOperators reg({
      Operator(
          "aten::test_vartype2(t a, t[] b) -> (t[])",
          [](Stack& stack) {
            // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
            double a;
            c10::List<double> list;
            pop(stack, a, list);
            push(stack, a);
          },
          AliasAnalysisKind::FROM_SCHEMA),
  });
  Module m("m");
  m.define(R"JIT(
      def test(self):
          a = (1.0, 2.0)
          return torch.test_vartype2(3.0, a)
    )JIT");
  auto result = m.run_method("test");
  TORCH_INTERNAL_ASSERT(result.toDouble() == 3.0);

  static const auto error_exam2 = R"JIT(
      def test_2(self):
          a = (1, 2)
          return torch.test_vartype2(3.0, a)
    )JIT";

  std::string err = "";
  try {
    m.define(error_exam2);
  } catch (const std::exception& e) {
    err = e.what();
  }
  TORCH_INTERNAL_ASSERT(
      err.find("previously matched to type") != std::string::npos);
}
} // namespace jit
} // namespace torch

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 8 function(s).

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
- `torch/csrc/jit/ir/ir.h`
- `torch/csrc/jit/runtime/custom_operator.h`
- `torch/csrc/jit/testing/file_check.h`
- `torch/jit.h`
- `sstream`
- `string`


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
python test/cpp/jit/test_schema_matching.cpp
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

- **File Documentation**: `test_schema_matching.cpp_docs.md`
- **Keyword Index**: `test_schema_matching.cpp_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
