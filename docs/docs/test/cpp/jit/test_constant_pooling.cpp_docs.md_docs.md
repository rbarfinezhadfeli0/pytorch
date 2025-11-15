# Documentation: `docs/test/cpp/jit/test_constant_pooling.cpp_docs.md`

## File Metadata

- **Path**: `docs/test/cpp/jit/test_constant_pooling.cpp_docs.md`
- **Size**: 5,987 bytes (5.85 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**. This appears to be a **test file**.

## Original Source

```markdown
# Documentation: `test/cpp/jit/test_constant_pooling.cpp`

## File Metadata

- **Path**: `test/cpp/jit/test_constant_pooling.cpp`
- **Size**: 3,188 bytes (3.11 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**.

## Original Source

```cpp
#include <gtest/gtest.h>

#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/ir/irparser.h>
#include <torch/csrc/jit/passes/constant_pooling.h>
#include <torch/csrc/jit/passes/constant_propagation.h>
#include <torch/csrc/jit/testing/file_check.h>

#include <sstream>
#include <string>

namespace torch {
namespace jit {

TEST(ConstantPoolingTest, Int) {
  auto graph = std::make_shared<Graph>();
  parseIR(
      R"IR(
graph():
  %8 : int = prim::Constant[value=1]()
  %10 : int = prim::Constant[value=1]()
  return (%8, %10)
  )IR",
      &*graph);
  ConstantPooling(graph);
  testing::FileCheck()
      .check_count("prim::Constant", 1, /*exactly*/ true)
      ->run(*graph);
}

TEST(ConstantPoolingTest, PoolingAcrossBlocks) {
  auto graph = std::make_shared<Graph>();
  parseIR(
      R"IR(
graph(%cond : Tensor):
  %a : str = prim::Constant[value="bcd"]()
  %3 : bool = aten::Bool(%cond)
  %b : str = prim::If(%3)
    block0():
      %b.1 : str = prim::Constant[value="abc"]()
      -> (%b.1)
    block1():
      %b.2 : str = prim::Constant[value="abc"]()
      -> (%b.2)
  %7 : (str, str) = prim::TupleConstruct(%a, %b)
  return (%7)
  )IR",
      &*graph);
  ConstantPooling(graph);
  testing::FileCheck()
      .check_count("prim::Constant[value=\"abc\"]", 1, /*exactly*/ true)
      ->check_count("prim::Constant[value=\"bcd\"]", 1, /*exactly*/ true)
      ->run(*graph);
}

TEST(ConstantPoolingTest, PoolingDifferentDevices) {
  auto graph = std::make_shared<Graph>();
  parseIR(
      R"IR(
graph():
  %2 : int = prim::Constant[value=2]()
  %1 : int = prim::Constant[value=1]()
  %5 : int? = prim::Constant()
  %7 : Device? = prim::Constant()
  %15: bool = prim::Constant[value=0]()
  %10 : int = prim::Constant[value=6]()
  %3 : int[] = prim::ListConstruct(%1, %2)
  %x : Tensor = aten::tensor(%3, %5, %7, %15)
  %y : Tensor = aten::tensor(%3, %10, %7, %15)
  %9 : int[] = prim::ListConstruct(%1, %2)
  %z : Tensor = aten::tensor(%9, %10, %7, %15)
  prim::Print(%x, %y, %z)
  return (%1)
  )IR",
      &*graph);
  // three tensors created - two different devices among the three
  // don't have good support for parsing tensor constants
  ConstantPropagation(graph);
  ConstantPooling(graph);
  testing::FileCheck()
      .check_count(
          "Float(2, strides=[1], requires_grad=0, device=cpu) = prim::Constant",
          1,
          /*exactly*/ true)
      ->check_count(
          "Long(2, strides=[1], requires_grad=0, device=cpu) = prim::Constant",
          1,
          /*exactly*/ true)
      ->run(*graph);
}

TEST(ConstantPoolingTest, DictConstantPooling) {
  auto graph = std::make_shared<Graph>();
  parseIR(
      R"IR(
graph():
  %0 : int = prim::Constant[value=1]() # test/elias.py:6:9
  %1 : int = prim::Constant[value=2]() # test/elias.py:6:12
  %a.1 : Dict(int, int) = prim::DictConstruct(%0, %1)
  %b.1 : Dict(int, int) = prim::DictConstruct(%1, %1)
  return (%a.1, %b.1)
  )IR",
      &*graph);
  ConstantPropagation(graph);
  ConstantPooling(graph);
  testing::FileCheck()
      .check_count(
          "Dict(int, int) = prim::Constant",
          2,
          /*exactly*/ true)
      ->run(*graph);
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
- `torch/csrc/jit/ir/ir.h`
- `torch/csrc/jit/ir/irparser.h`
- `torch/csrc/jit/passes/constant_pooling.h`
- `torch/csrc/jit/passes/constant_propagation.h`
- `torch/csrc/jit/testing/file_check.h`
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
python test/cpp/jit/test_constant_pooling.cpp
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

- **File Documentation**: `test_constant_pooling.cpp_docs.md`
- **Keyword Index**: `test_constant_pooling.cpp_kw.md`
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
python docs/test/cpp/jit/test_constant_pooling.cpp_docs.md
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

- **File Documentation**: `test_constant_pooling.cpp_docs.md_docs.md`
- **Keyword Index**: `test_constant_pooling.cpp_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
