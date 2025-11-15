# Documentation: `aten/src/ATen/core/custom_class.cpp`

## File Metadata

- **Path**: `aten/src/ATen/core/custom_class.cpp`
- **Size**: 5,492 bytes (5.36 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#include <ATen/core/function_schema.h>
#include <ATen/core/functional.h>
#include <ATen/core/jit_type.h>
#include <ATen/core/type_factory.h>
#include <ATen/record_function.h>
#include <c10/util/flat_hash_map.h>
#include <torch/custom_class.h>
#include <torch/custom_class_detail.h>

#include <unordered_map>

namespace c10 {

static ska::flat_hash_map<std::type_index, c10::ClassTypePtr>&
getCustomClassTypeMap() {
  static ska::flat_hash_map<std::type_index, c10::ClassTypePtr> tmap;
  return tmap;
}

c10::ClassTypePtr getCustomClassTypeImpl(const std::type_index& tindex) {
  auto& tmap = c10::getCustomClassTypeMap();
  auto res = tmap.find(tindex);
  if (C10_UNLIKELY(res == tmap.end())) {
    // type_index is not guaranteed to be unique across shared libraries on some
    // platforms For example see
    // https://github.com/llvm-mirror/libcxx/blob/78d6a7767ed57b50122a161b91f59f19c9bd0d19/include/typeinfo#L133
    // Also, this is not the case if RTLD_LOCAL option is used, see
    // https://github.com/pybind/pybind11/blob/f791dc8648e1f6ec33f402d679b6b116a76d4e1b/include/pybind11/detail/internals.h#L101-L106
    // Take a slow path of iterating over all registered types and compare their
    // names
    auto class_name = std::string(tindex.name());
    for (const auto& it : tmap) {
      if (class_name == it.first.name()) {
        // Do not modify existing type map here as this template is supposed to
        // be called only once per type from getCustomClassTypeImpl()
        return it.second;
      }
    }
    TORCH_CHECK(
        false,
        "Can't find class id in custom class type map for ",
        tindex.name());
  }
  return res->second;
}

} // namespace c10

namespace torch {

namespace detail {

#if defined ENABLE_RECORD_KERNEL_FUNCTION_DTYPE
void record_custom_class(std::string name) {
  RECORD_FUNCTION_WITH_SCOPE(
      at::RecordScope::CUSTOM_CLASS,
      std::move(name),
      c10::ArrayRef<const c10::IValue>{});
}
#endif

} // namespace detail

static std::unordered_map<std::string, at::ClassTypePtr>& customClasses() {
  static std::unordered_map<std::string, at::ClassTypePtr> customClasses;
  return customClasses;
}

void registerCustomClass(at::ClassTypePtr class_type) {
  TORCH_INTERNAL_ASSERT(class_type->name());
  // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
  auto name = class_type->name()->qualifiedName();
  TORCH_CHECK(
      !customClasses().count(name),
      "Custom class with name ",
      name,
      " is already registered. Ensure that registration with torch::class_ is only called once.");
  customClasses()[name] = std::move(class_type);
}

at::ClassTypePtr getCustomClass(const std::string& class_name) {
  auto ret =
      customClasses().count(class_name) ? customClasses()[class_name] : nullptr;
  if (ret) {
    RECORD_CUSTOM_CLASS(class_name);
  }
  return ret;
}

const std::unordered_set<std::string> getAllCustomClassesNames() {
  std::unordered_set<std::string> ret;
  for (const auto& kv : customClasses()) {
    ret.insert(kv.first);
  }
  return ret;
}

bool isCustomClass(const c10::IValue& v) {
  return v.isObject() && v.toObject()->type()->name() &&
      // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
      getCustomClass(v.toObject()->type()->name()->qualifiedName());
}

static std::vector<std::unique_ptr<jit::Function>>& customClassMethods() {
  static std::vector<std::unique_ptr<jit::Function>> customClassMethods;
  return customClassMethods;
}

void registerCustomClassMethod(std::unique_ptr<jit::Function> fn) {
  customClassMethods().emplace_back(std::move(fn));
}

std::vector<c10::FunctionSchema> customClassSchemasForBCCheck() {
  auto& methods = customClassMethods();
  return c10::fmap(methods, [](const std::unique_ptr<jit::Function>& fn) {
    return fn->getSchema();
  });
}

namespace detail {
class_base::class_base(
    const std::string& namespaceName,
    const std::string& className,
    std::string doc_string,
    const std::type_info& intrusivePtrClassTypeid,
    const std::type_info& taggedCapsuleClassTypeid)
    : qualClassName(
          "__torch__.torch.classes." + namespaceName + '.' + className),
      classTypePtr(at::ClassType::create(
          c10::QualifiedName(qualClassName),
          std::weak_ptr<jit::CompilationUnit>(),
          /*is_module=*/false,
          std::move(doc_string))) {
  detail::checkValidIdent(namespaceName, "Namespace name");
  detail::checkValidIdent(className, "Class name");
  classTypePtr->addAttribute(
      "capsule", c10::TypeFactory::get<c10::CapsuleType>());
  c10::getCustomClassTypeMap().insert(
      {std::type_index(intrusivePtrClassTypeid), classTypePtr});
  c10::getCustomClassTypeMap().insert(
      {std::type_index(taggedCapsuleClassTypeid), classTypePtr});

  registerCustomClass(classTypePtr);
}

c10::FunctionSchema class_base::withNewArguments(
    const c10::FunctionSchema& schema,
    std::initializer_list<arg> default_args) {
  const auto& old_args = schema.arguments();
  std::vector<c10::Argument> new_args;
  new_args.reserve(old_args.size());

  new_args.emplace_back(old_args[0]);
  // Skip self.
  size_t argIdx = 1;
  for (const auto& default_arg : default_args) {
    auto& old_arg = old_args[argIdx++];
    new_args.emplace_back(
        default_arg.name_,
        old_arg.type(),
        old_arg.real_type(),
        old_arg.N(),
        default_arg.value_);
  }
  return schema.cloneWithArguments(std::move(new_args));
}

} // namespace detail
} // namespace torch

```



## High-Level Overview


This C++ file contains approximately 3 class(es)/struct(s) and 7 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `torch`, `detail`, `c10`

**Classes/Structs**: `id`, `type`, `with`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `aten/src/ATen/core`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



## Dependencies

### Import Dependencies

This file includes:

- `ATen/core/function_schema.h`
- `ATen/core/functional.h`
- `ATen/core/jit_type.h`
- `ATen/core/type_factory.h`
- `ATen/record_function.h`
- `c10/util/flat_hash_map.h`
- `torch/custom_class.h`
- `torch/custom_class_detail.h`
- `unordered_map`


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

Test files for this module may be located in the `test/` directory.

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`aten/src/ATen/core`):

- [`DistributionsHelper.h_docs.md`](./DistributionsHelper.h_docs.md)
- [`rref_interface.h_docs.md`](./rref_interface.h_docs.md)
- [`Generator.h_docs.md`](./Generator.h_docs.md)
- [`enum_type.h_docs.md`](./enum_type.h_docs.md)
- [`QuantizerBase.h_docs.md`](./QuantizerBase.h_docs.md)
- [`Array.h_docs.md`](./Array.h_docs.md)
- [`MetaFallbackKernel.cpp_docs.md`](./MetaFallbackKernel.cpp_docs.md)
- [`ATenOpList.h_docs.md`](./ATenOpList.h_docs.md)
- [`ivalue_inl.h_docs.md`](./ivalue_inl.h_docs.md)
- [`TransformationHelper.h_docs.md`](./TransformationHelper.h_docs.md)


## Cross-References

- **File Documentation**: `custom_class.cpp_docs.md`
- **Keyword Index**: `custom_class.cpp_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
