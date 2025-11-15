# Documentation: `docs/test/cpp/jit/test_class_type.cpp_docs.md`

## File Metadata

- **Path**: `docs/test/cpp/jit/test_class_type.cpp_docs.md`
- **Size**: 6,047 bytes (5.91 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**. This appears to be a **test file**.

## Original Source

```markdown
# Documentation: `test/cpp/jit/test_class_type.cpp`

## File Metadata

- **Path**: `test/cpp/jit/test_class_type.cpp`
- **Size**: 3,407 bytes (3.33 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**.

## Original Source

```cpp
#include <gtest/gtest.h>

#include <test/cpp/jit/test_utils.h>
#include <torch/csrc/jit/testing/file_check.h>
#include <torch/torch.h>

namespace torch {
namespace jit {

TEST(ClassTypeTest, AddRemoveAttr) {
  auto cu = std::make_shared<CompilationUnit>();
  auto cls = ClassType::create("foo.bar", cu, true);
  cls->addAttribute("attr1", TensorType::get(), true);
  cls->addAttribute("attr2", TensorType::get());
  cls->addAttribute("attr3", TensorType::get());
  ASSERT_TRUE(cls->hasAttribute("attr1"));
  ASSERT_TRUE(cls->hasAttribute("attr2"));
  ASSERT_TRUE(cls->hasAttribute("attr3"));

  // removing attribute attr2
  cls->unsafeRemoveAttribute("attr2");
  ASSERT_TRUE(cls->hasAttribute("attr1"));
  ASSERT_FALSE(cls->hasAttribute("attr2"));
  ASSERT_TRUE(cls->hasAttribute("attr3"));

  // removing parameter attr1
  cls->unsafeRemoveAttribute("attr1");
  ASSERT_FALSE(cls->hasAttribute("attr1"));
  ASSERT_FALSE(cls->hasAttribute("attr2"));
  ASSERT_TRUE(cls->hasAttribute("attr3"));

  // check that we can still add a non-parameter attr1 with
  // different type
  cls->addAttribute("attr1", IntType::get());
}

TEST(ClassTypeTest, AddRemoveConstant) {
  auto cu = std::make_shared<CompilationUnit>();
  auto cls = ClassType::create("foo.bar", cu);
  cls->addConstant("const1", IValue(1));
  cls->addConstant("const2", IValue(2));
  cls->addConstant("const3", IValue(3));
  ASSERT_EQ(cls->numConstants(), 3);
  ASSERT_TRUE(cls->hasConstant("const1"));
  ASSERT_TRUE(cls->hasConstant("const2"));
  ASSERT_TRUE(cls->hasConstant("const3"));
  ASSERT_FALSE(cls->hasConstant("const4"));

  ASSERT_EQ(cls->getConstant("const1").toInt(), 1);
  ASSERT_EQ(cls->getConstant("const2").toInt(), 2);
  ASSERT_EQ(cls->getConstant("const3").toInt(), 3);

  cls->unsafeRemoveConstant("const2");
  ASSERT_TRUE(cls->hasConstant("const1"));
  ASSERT_FALSE(cls->hasConstant("const2"));
  ASSERT_TRUE(cls->hasConstant("const3"));
}

TEST(ClassTypeTest, IdenticalTypesDifferentCus) {
  auto cu1 = std::make_shared<CompilationUnit>();
  auto cu2 = std::make_shared<CompilationUnit>();

  // Create two identically named ClassTypes and put them
  // in separate compilation units.
  auto cls1 = ClassType::create("foo", cu1);
  auto cls2 = ClassType::create("foo", cu2);

  // Create a function that accepts "foo" (cls1) as input.
  Argument arg("arg", cls1);
  Argument ret("ret", IntType::get());

  FunctionSchema schema("fn", "", {arg}, {ret});

  jit::BuiltinOpFunction method(
      "method",
      std::move(schema),
      [](jit::Stack& stack) mutable -> void {
        pop(stack);
        push(stack, 0);
      },
      "");

  // Create an object of type cls2.
  Object obj(cu2, cls2);

  // Call method with the above object; this should
  // throw an error because the types have identical
  // names but are in different compilation units.
  Stack stack;
  push(stack, obj._ivalue());
  try {
    method(stack, {});
  } catch (const std::exception& e) {
    // Check that the exception contains the address of the compilation unit
    // in addition to the ClassType's name.
    testing::FileCheck()
        .check("foo (of Python compilation unit at: 0x")
        ->check_same(")")
        ->check("foo (of Python compilation unit at: 0x")
        ->check_same(")")
        ->run(e.what());

    return;
  }

  // This should never execute.
  ASSERT_TRUE(false);
}

} // namespace jit
} // namespace torch

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 5 function(s).

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
- `torch/csrc/jit/testing/file_check.h`
- `torch/torch.h`


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
python test/cpp/jit/test_class_type.cpp
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

- **File Documentation**: `test_class_type.cpp_docs.md`
- **Keyword Index**: `test_class_type.cpp_kw.md`
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
python docs/test/cpp/jit/test_class_type.cpp_docs.md
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

- **File Documentation**: `test_class_type.cpp_docs.md_docs.md`
- **Keyword Index**: `test_class_type.cpp_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
