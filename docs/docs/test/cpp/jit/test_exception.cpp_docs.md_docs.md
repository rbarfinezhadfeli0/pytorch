# Documentation: `docs/test/cpp/jit/test_exception.cpp_docs.md`

## File Metadata

- **Path**: `docs/test/cpp/jit/test_exception.cpp_docs.md`
- **Size**: 8,026 bytes (7.84 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**. This appears to be a **test file**.

## Original Source

```markdown
# Documentation: `test/cpp/jit/test_exception.cpp`

## File Metadata

- **Path**: `test/cpp/jit/test_exception.cpp`
- **Size**: 5,130 bytes (5.01 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**.

## Original Source

```cpp
/*
 * We have a python unit test for exceptions in test/jit/test_exception.py .
 * Add a CPP version here to verify that excepted exception types thrown from
 * C++. This is hard to test in python code since C++ exceptions will be
 * translated to python exceptions.
 */
#include <gtest/gtest.h>
#include <pybind11/embed.h>
#include <torch/csrc/jit/frontend/parser.h>
#include <torch/csrc/jit/frontend/resolver.h>
#include <torch/csrc/jit/runtime/jit_exception.h>
#include <torch/csrc/utils/pybind.h>
#include <torch/jit.h>
#include <iostream>
#include <stdexcept>

namespace torch {
namespace jit {

namespace py = pybind11;

TEST(TestException, TestAssertion) {
  std::string pythonCode = R"PY(
  def foo():
    raise AssertionError("An assertion failed")
  )PY";
  auto cu_ptr = torch::jit::compile(pythonCode);
  torch::jit::GraphFunction* gf =
      (torch::jit::GraphFunction*)&cu_ptr->get_function("foo");
  std::cerr << "Graph is\n" << *gf->graph() << std::endl;

  bool is_jit_exception = false;
  std::string message;
  std::optional<std::string> exception_class;
  try {
    cu_ptr->run_method("foo");
  } catch (JITException& e) {
    is_jit_exception = true;
    message = e.what();
    exception_class = e.getPythonClassName();
  }
  EXPECT_TRUE(is_jit_exception);
  EXPECT_FALSE(exception_class);
  EXPECT_TRUE(
      message.find("RuntimeError: AssertionError: An assertion failed") !=
      std::string::npos);
}

struct MyPythonExceptionValue : public torch::jit::SugaredValue {
  explicit MyPythonExceptionValue(const py::object& exception_class) {
    qualified_name_ =
        (py::str(py::getattr(exception_class, "__module__", py::str(""))) +
         py::str(".") +
         py::str(py::getattr(exception_class, "__name__", py::str(""))))
            .cast<std::string>();
  }

  std::string kind() const override {
    return "My Python exception";
  }

  // Simplified from PythonExceptionValue::call
  std::shared_ptr<torch::jit::SugaredValue> call(
      const torch::jit::SourceRange& loc,
      torch::jit::GraphFunction& caller,
      at::ArrayRef<torch::jit::NamedValue> args,
      at::ArrayRef<torch::jit::NamedValue> kwargs,
      size_t n_binders) override {
    TORCH_CHECK(args.size() == 1);
    Value* error_message = args.at(0).value(*caller.graph());
    Value* qualified_class_name =
        insertConstant(*caller.graph(), qualified_name_, loc);
    return std::make_shared<ExceptionMessageValue>(
        error_message, qualified_class_name);
  }

 private:
  std::string qualified_name_;
};

class SimpleResolver : public torch::jit::Resolver {
 public:
  explicit SimpleResolver() {}

  std::shared_ptr<torch::jit::SugaredValue> resolveValue(
      const std::string& name,
      torch::jit::GraphFunction& m,
      const torch::jit::SourceRange& loc) override {
    // follows toSugaredValue (toSugaredValue is defined in caffe2:_C which is
    // a python extension. We can not add that as a cpp_binary's dep)
    if (name == "SimpleValueError") {
      py::object obj = py::globals()["SimpleValueError"];
      return std::make_shared<MyPythonExceptionValue>(obj);
    }
    TORCH_CHECK(false, "resolveValue: can not resolve '", name, "{}'");
  }

  torch::jit::TypePtr resolveType(
      const std::string& name,
      const torch::jit::SourceRange& loc) override {
    return nullptr;
  }
};

/*
 * - The python source code parsing for TorchScript here is learned from
 * torch::jit::compile.
 * - The code only parses one Def. If there are multiple in the code, those
 * except the first one are skipped.
 */
TEST(TestException, TestCustomException) {
  py::scoped_interpreter guard{};
  py::exec(R"PY(
  class SimpleValueError(ValueError):
    def __init__(self, message):
      super().__init__(message)
  )PY");

  std::string pythonCode = R"PY(
  def foo():
    raise SimpleValueError("An assertion failed")
  )PY";

  torch::jit::Parser p(
      std::make_shared<torch::jit::Source>(pythonCode, "<string>", 1));
  auto def = torch::jit::Def(p.parseFunction(/*is_method=*/false));
  std::cerr << "Def is:\n" << def << std::endl;
  auto cu = std::make_shared<torch::jit::CompilationUnit>();
  (void)cu->define(
      std::nullopt,
      {},
      {},
      {def},
      // class PythonResolver is defined in
      // torch/csrc/jit/python/script_init.cpp. It's not in a header file so I
      // can not use it. Create a SimpleResolver instead
      {std::make_shared<SimpleResolver>()},
      nullptr);
  torch::jit::GraphFunction* gf =
      (torch::jit::GraphFunction*)&cu->get_function("foo");
  std::cerr << "Graph is\n" << *gf->graph() << std::endl;
  bool is_jit_exception = false;
  std::optional<std::string> exception_class;
  std::string message;
  try {
    cu->run_method("foo");
  } catch (JITException& e) {
    is_jit_exception = true;
    exception_class = e.getPythonClassName();
    message = e.what();
  }
  EXPECT_TRUE(is_jit_exception);
  EXPECT_EQ("__main__.SimpleValueError", *exception_class);
  EXPECT_TRUE(
      message.find("__main__.SimpleValueError: An assertion failed") !=
      std::string::npos);
}

} // namespace jit
} // namespace torch

```



## High-Level Overview


This C++ file contains approximately 3 class(es)/struct(s) and 12 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `jit`, `torch`, `py`

**Classes/Structs**: `MyPythonExceptionValue`, `SimpleResolver`, `SimpleValueError`, `PythonResolver`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/cpp/jit`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file includes:

- `gtest/gtest.h`
- `pybind11/embed.h`
- `torch/csrc/jit/frontend/parser.h`
- `torch/csrc/jit/frontend/resolver.h`
- `torch/csrc/jit/runtime/jit_exception.h`
- `torch/csrc/utils/pybind.h`
- `torch/jit.h`
- `iostream`
- `stdexcept`


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors


## Performance Considerations

### Performance Notes

- May involve **JIT compilation** or compilation optimizations.

*Detailed performance analysis requires profiling and benchmarking.*


## Security & Safety

### Security Considerations

- **Code Execution**: Uses `eval()` or `exec()` - ensure input is sanitized

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

This is a test file. Run it with:

```bash
python test/cpp/jit/test_exception.cpp
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

- **File Documentation**: `test_exception.cpp_docs.md`
- **Keyword Index**: `test_exception.cpp_kw.md`
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

- **Object-Oriented Design**: Uses classes and constructors


## Performance Considerations

### Performance Notes

- May involve **JIT compilation** or compilation optimizations.
- Contains **benchmarking** code or performance tests.

*Detailed performance analysis requires profiling and benchmarking.*


## Security & Safety

### Security Considerations

- **Code Execution**: Uses `eval()` or `exec()` - ensure input is sanitized

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

This is a test file. Run it with:

```bash
python docs/test/cpp/jit/test_exception.cpp_docs.md
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

- **File Documentation**: `test_exception.cpp_docs.md_docs.md`
- **Keyword Index**: `test_exception.cpp_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
