# Documentation: `docs/test/cpp/jit/test_custom_class.cpp_docs.md`

## File Metadata

- **Path**: `docs/test/cpp/jit/test_custom_class.cpp_docs.md`
- **Size**: 7,321 bytes (7.15 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**. This appears to be a **test file**.

## Original Source

```markdown
# Documentation: `test/cpp/jit/test_custom_class.cpp`

## File Metadata

- **Path**: `test/cpp/jit/test_custom_class.cpp`
- **Size**: 4,522 bytes (4.42 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**.

## Original Source

```cpp
#include <gtest/gtest.h>

#include <test/cpp/jit/test_custom_class_registrations.h>
#include <torch/csrc/jit/passes/freeze_module.h>
#include <torch/custom_class.h>
#include <torch/script.h>

#include <iostream>
#include <string>
#include <vector>

namespace torch {
namespace jit {

TEST(CustomClassTest, TorchbindIValueAPI) {
  script::Module m("m");

  // test make_custom_class API
  auto custom_class_obj = make_custom_class<MyStackClass<std::string>>(
      std::vector<std::string>{"foo", "bar"});
  m.define(R"(
    def forward(self, s : __torch__.torch.classes._TorchScriptTesting._StackString):
      return s.pop(), s
  )");

  auto test_with_obj = [&m](IValue obj, std::string expected) {
    auto res = m.run_method("forward", obj);
    auto tup = res.toTuple();
    AT_ASSERT(tup->elements().size() == 2);
    auto str = tup->elements()[0].toStringRef();
    auto other_obj =
        tup->elements()[1].toCustomClass<MyStackClass<std::string>>();
    AT_ASSERT(str == expected);
    auto ref_obj = obj.toCustomClass<MyStackClass<std::string>>();
    AT_ASSERT(other_obj.get() == ref_obj.get());
  };

  test_with_obj(custom_class_obj, "bar");

  // test IValue() API
  auto my_new_stack = c10::make_intrusive<MyStackClass<std::string>>(
      std::vector<std::string>{"baz", "boo"});
  auto new_stack_ivalue = c10::IValue(my_new_stack);

  test_with_obj(new_stack_ivalue, "boo");
}

TEST(CustomClassTest, ScalarTypeClass) {
  script::Module m("m");

  // test make_custom_class API
  auto cc = make_custom_class<ScalarTypeClass>(at::kFloat);
  m.register_attribute("s", cc.type(), cc, false);

  std::ostringstream oss;
  m.save(oss);
  std::istringstream iss(oss.str());
  caffe2::serialize::IStreamAdapter adapter{&iss};
  auto loaded_module = torch::jit::load(iss, torch::kCPU);
}

class TorchBindTestClass : public torch::jit::CustomClassHolder {
 public:
  std::string get() {
    return "Hello, I am your test custom class";
  }
};

// NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
constexpr char class_doc_string[] = R"(
  I am docstring for TorchBindTestClass
  Args:
      What is an argument? Oh never mind, I don't take any.

  Return:
      How would I know? I am just a holder of some meaningless test methods.
  )";
// NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
constexpr char method_doc_string[] =
    "I am docstring for TorchBindTestClass get_with_docstring method";

namespace {
static auto reg =
    torch::class_<TorchBindTestClass>(
        "_TorchBindTest",
        "_TorchBindTestClass",
        class_doc_string)
        .def("get", &TorchBindTestClass::get)
        .def("get_with_docstring", &TorchBindTestClass::get, method_doc_string);

} // namespace

// Tests DocString is properly propagated when defining CustomClasses.
TEST(CustomClassTest, TestDocString) {
  auto class_type = getCustomClass(
      "__torch__.torch.classes._TorchBindTest._TorchBindTestClass");
  AT_ASSERT(class_type);
  AT_ASSERT(class_type->doc_string() == class_doc_string);

  AT_ASSERT(class_type->getMethod("get").doc_string().empty());
  AT_ASSERT(
      class_type->getMethod("get_with_docstring").doc_string() ==
      method_doc_string);
}

TEST(CustomClassTest, Serialization) {
  script::Module m("m");

  // test make_custom_class API
  auto custom_class_obj = make_custom_class<MyStackClass<std::string>>(
      std::vector<std::string>{"foo", "bar"});
  m.register_attribute(
      "s",
      custom_class_obj.type(),
      custom_class_obj,
      // NOLINTNEXTLINE(bugprone-argument-comment)
      /*is_parameter=*/false);
  m.define(R"(
    def forward(self):
      return self.s.return_a_tuple()
  )");

  auto test_with_obj = [](script::Module& mod) {
    auto res = mod.run_method("forward");
    auto tup = res.toTuple();
    AT_ASSERT(tup->elements().size() == 2);
    auto i = tup->elements()[1].toInt();
    AT_ASSERT(i == 123);
  };

  auto frozen_m = torch::jit::freeze_module(m.clone());

  test_with_obj(m);
  test_with_obj(frozen_m);

  std::ostringstream oss;
  m.save(oss);
  std::istringstream iss(oss.str());
  caffe2::serialize::IStreamAdapter adapter{&iss};
  auto loaded_module = torch::jit::load(iss, torch::kCPU);

  std::ostringstream oss_frozen;
  frozen_m.save(oss_frozen);
  std::istringstream iss_frozen(oss_frozen.str());
  caffe2::serialize::IStreamAdapter adapter_frozen{&iss_frozen};
  auto loaded_frozen_module = torch::jit::load(iss_frozen, torch::kCPU);
}

} // namespace jit
} // namespace torch

```



## High-Level Overview


This C++ file contains approximately 4 class(es)/struct(s) and 10 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `jit`, `torch`

**Classes/Structs**: `API`, `API`, `TorchBindTestClass`, `API`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/cpp/jit`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file includes:

- `gtest/gtest.h`
- `test/cpp/jit/test_custom_class_registrations.h`
- `torch/csrc/jit/passes/freeze_module.h`
- `torch/custom_class.h`
- `torch/script.h`
- `iostream`
- `string`
- `vector`


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
python test/cpp/jit/test_custom_class.cpp
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

- **File Documentation**: `test_custom_class.cpp_docs.md`
- **Keyword Index**: `test_custom_class.cpp_kw.md`
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
python docs/test/cpp/jit/test_custom_class.cpp_docs.md
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

- **File Documentation**: `test_custom_class.cpp_docs.md_docs.md`
- **Keyword Index**: `test_custom_class.cpp_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
