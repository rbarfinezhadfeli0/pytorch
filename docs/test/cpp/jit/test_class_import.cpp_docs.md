# Documentation: `test/cpp/jit/test_class_import.cpp`

## File Metadata

- **Path**: `test/cpp/jit/test_class_import.cpp`
- **Size**: 4,890 bytes (4.78 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**.

## Original Source

```cpp
#include <gtest/gtest.h>

#include <ATen/core/qualified_name.h>
#include <test/cpp/jit/test_utils.h>
#include <torch/csrc/jit/frontend/resolver.h>
#include <torch/csrc/jit/serialization/import_source.h>
#include <torch/torch.h>

namespace torch {
namespace jit {

static constexpr std::string_view classSrcs1 = R"JIT(
class FooNestedTest:
    def __init__(self, y):
        self.y = y

class FooNestedTest2:
    def __init__(self, y):
        self.y = y
        self.nested = __torch__.FooNestedTest(y)

class FooTest:
    def __init__(self, x):
        self.class_attr = __torch__.FooNestedTest(x)
        self.class_attr2 = __torch__.FooNestedTest2(x)
        self.x = self.class_attr.y + self.class_attr2.y
)JIT";

static constexpr std::string_view classSrcs2 = R"JIT(
class FooTest:
    def __init__(self, x):
      self.dx = x
)JIT";

static void import_libs(
    std::shared_ptr<CompilationUnit> cu,
    const std::string& class_name,
    const std::shared_ptr<Source>& src,
    const std::vector<at::IValue>& tensor_table) {
  SourceImporter si(
      cu,
      &tensor_table,
      [&](const std::string& name) -> std::shared_ptr<Source> { return src; },
      /*version=*/2);
  si.loadType(QualifiedName(class_name));
}

TEST(ClassImportTest, Basic) {
  auto cu1 = std::make_shared<CompilationUnit>();
  auto cu2 = std::make_shared<CompilationUnit>();
  std::vector<at::IValue> constantTable;
  // Import different versions of FooTest into two namespaces.
  import_libs(
      cu1,
      "__torch__.FooTest",
      std::make_shared<Source>(classSrcs1),
      constantTable);
  import_libs(
      cu2,
      "__torch__.FooTest",
      std::make_shared<Source>(classSrcs2),
      constantTable);

  // We should get the correct version of `FooTest` for whichever namespace we
  // are referencing
  c10::QualifiedName base("__torch__");
  auto classType1 = cu1->get_class(c10::QualifiedName(base, "FooTest"));
  ASSERT_TRUE(classType1->hasAttribute("x"));
  ASSERT_FALSE(classType1->hasAttribute("dx"));

  auto classType2 = cu2->get_class(c10::QualifiedName(base, "FooTest"));
  ASSERT_TRUE(classType2->hasAttribute("dx"));
  ASSERT_FALSE(classType2->hasAttribute("x"));

  // We should only see FooNestedTest in the first namespace
  auto c = cu1->get_class(c10::QualifiedName(base, "FooNestedTest"));
  ASSERT_TRUE(c);

  c = cu2->get_class(c10::QualifiedName(base, "FooNestedTest"));
  ASSERT_FALSE(c);
}

TEST(ClassImportTest, ScriptObject) {
  Module m1("m1");
  Module m2("m2");
  std::vector<at::IValue> constantTable;
  import_libs(
      m1._ivalue()->compilation_unit(),
      "__torch__.FooTest",
      std::make_shared<Source>(classSrcs1),
      constantTable);
  import_libs(
      m2._ivalue()->compilation_unit(),
      "__torch__.FooTest",
      std::make_shared<Source>(classSrcs2),
      constantTable);

  // Incorrect arguments for constructor should throw
  c10::QualifiedName base("__torch__");
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-goto,hicpp-avoid-goto)
  ASSERT_ANY_THROW(m1.create_class(c10::QualifiedName(base, "FooTest"), {1}));
  auto x = torch::ones({2, 3});
  auto obj = m2.create_class(c10::QualifiedName(base, "FooTest"), x).toObject();
  auto dx = obj->getAttr("dx");
  ASSERT_TRUE(almostEqual(x, dx.toTensor()));

  auto new_x = torch::rand({2, 3});
  obj->setAttr("dx", new_x);
  auto new_dx = obj->getAttr("dx");
  ASSERT_TRUE(almostEqual(new_x, new_dx.toTensor()));
}

static const auto methodSrc = R"JIT(
def __init__(self, x):
    return x
)JIT";

TEST(ClassImportTest, ClassDerive) {
  auto cu = std::make_shared<CompilationUnit>();
  auto cls = ClassType::create("foo.bar", cu);
  const auto self = SimpleSelf(cls);
  auto methods = cu->define("foo.bar", methodSrc, nativeResolver(), &self);
  auto method = methods[0];
  cls->addAttribute("attr", TensorType::get());
  ASSERT_TRUE(cls->findMethod(method->name()));

  // Refining a new class should retain attributes and methods
  auto newCls = cls->refine({TensorType::get()});
  ASSERT_TRUE(newCls->hasAttribute("attr"));
  ASSERT_TRUE(newCls->findMethod(method->name()));

  auto newCls2 = cls->withContained({TensorType::get()})->expect<ClassType>();
  ASSERT_TRUE(newCls2->hasAttribute("attr"));
  ASSERT_TRUE(newCls2->findMethod(method->name()));
}

static constexpr std::string_view torchbindSrc = R"JIT(
class FooBar1234(Module):
  __parameters__ = []
  f : __torch__.torch.classes._TorchScriptTesting._StackString
  training : bool
  def forward(self: __torch__.FooBar1234) -> str:
    return (self.f).top()
)JIT";

TEST(ClassImportTest, CustomClass) {
  auto cu1 = std::make_shared<CompilationUnit>();
  std::vector<at::IValue> constantTable;
  // Import different versions of FooTest into two namespaces.
  import_libs(
      cu1,
      "__torch__.FooBar1234",
      std::make_shared<Source>(torchbindSrc),
      constantTable);
}

} // namespace jit
} // namespace torch

```



## High-Level Overview


This C++ file contains approximately 6 class(es)/struct(s) and 13 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `jit`, `torch`, `we`, `auto`

**Classes/Structs**: `FooNestedTest`, `FooNestedTest2`, `FooTest`, `FooTest`, `should`, `FooBar1234`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/cpp/jit`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file includes:

- `gtest/gtest.h`
- `ATen/core/qualified_name.h`
- `test/cpp/jit/test_utils.h`
- `torch/csrc/jit/frontend/resolver.h`
- `torch/csrc/jit/serialization/import_source.h`
- `torch/torch.h`


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors


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
python test/cpp/jit/test_class_import.cpp
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

- **File Documentation**: `test_class_import.cpp_docs.md`
- **Keyword Index**: `test_class_import.cpp_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
