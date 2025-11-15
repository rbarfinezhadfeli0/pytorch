# Documentation: `docs/torch/csrc/jit/api/object.h_docs.md`

## File Metadata

- **Path**: `docs/torch/csrc/jit/api/object.h_docs.md`
- **Size**: 8,459 bytes (8.26 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/csrc/jit/api/object.h`

## File Metadata

- **Path**: `torch/csrc/jit/api/object.h`
- **Size**: 6,083 bytes (5.94 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
#pragma once

#include <ATen/core/functional.h>
#include <ATen/core/ivalue.h>
#include <torch/csrc/jit/api/method.h>
#include <optional>

#include <utility>

namespace torch::jit {

struct Resolver;
using ResolverPtr = std::shared_ptr<Resolver>;

using ObjectPtr = c10::intrusive_ptr<c10::ivalue::Object>;

// Throw this in C++ land if `attr` fails. This will be converted to a Python
// AttributeError by the Python binding code
class ObjectAttributeError : public std::runtime_error {
 public:
  ObjectAttributeError(const std::string& what) : std::runtime_error(what) {}
};

struct TORCH_API Object {
  Object() = default;
  Object(const Object&) = default;
  Object& operator=(const Object&) = default;
  Object(Object&&) noexcept = default;
  Object& operator=(Object&&) noexcept = default;
  Object(ObjectPtr _ivalue) : _ivalue_(std::move(_ivalue)) {}
  Object(std::shared_ptr<CompilationUnit> cu, const c10::ClassTypePtr& type);
  Object(
      c10::QualifiedName,
      std::shared_ptr<CompilationUnit> cu,
      bool shouldMangle = false);

  ObjectPtr _ivalue() const {
    TORCH_INTERNAL_ASSERT(_ivalue_);
    return _ivalue_;
  }

  c10::ClassTypePtr type() const {
    return _ivalue()->type();
  }

  struct Property {
    std::string name;
    Method getter_func;
    std::optional<Method> setter_func;
  };

  void setattr(const std::string& name, c10::IValue v) {
    if (_ivalue()->type()->hasConstant(name)) {
      TORCH_CHECK(
          false,
          "Can't set constant '",
          name,
          "' which has value:",
          _ivalue()->type()->getConstant(name));
    } else if (auto slot = _ivalue()->type()->findAttributeSlot(name)) {
      const c10::TypePtr& expected = _ivalue()->type()->getAttribute(*slot);
      TORCH_CHECK(
          v.type()->isSubtypeOf(*expected),
          "Expected a value of type '",
          expected->repr_str(),
          "' for field '",
          name,
          "', but found '",
          v.type()->repr_str(),
          "'");
      _ivalue()->setSlot(*slot, std::move(v));
    } else {
      TORCH_CHECK(false, "Module has no attribute '", name, "'");
    }
  }

  c10::IValue attr(const std::string& name) const {
    if (auto r = _ivalue()->type()->findAttributeSlot(name)) {
      return _ivalue()->getSlot(*r);
    }
    if (auto r = _ivalue()->type()->findConstantSlot(name)) {
      return _ivalue()->type()->getConstant(*r);
    }
    std::stringstream err;
    err << _ivalue()->type()->repr_str() << " does not have a field with name '"
        << name.c_str() << "'";
    throw ObjectAttributeError(err.str());
  }

  c10::IValue attr(const std::string& name, c10::IValue or_else) const {
    if (auto r = _ivalue()->type()->findAttributeSlot(name)) {
      return _ivalue()->getSlot(*r);
    }
    if (auto r = _ivalue()->type()->findConstantSlot(name)) {
      return _ivalue()->type()->getConstant(*r);
    }
    return or_else;
  }

  bool hasattr(const std::string& name) const {
    return _ivalue()->type()->hasAttribute(name) ||
        _ivalue()->type()->hasConstant(name);
  }

  // each object owns its methods. The reference returned here
  // is guaranteed to stay valid until this module has been destroyed
  Method get_method(const std::string& name) const {
    if (auto method = find_method(name)) {
      return *method;
    }
    TORCH_CHECK(false, "Method '", name, "' is not defined.");
  }

  const std::vector<Method> get_methods() const {
    return c10::fmap(type()->methods(), [&](Function* func) {
      return Method(_ivalue(), func);
    });
  }

  bool has_property(const std::string& name) const {
    for (const auto& prop : type()->properties()) {
      if (prop.name == name) {
        return true;
      }
    }
    return false;
  }

  const Property get_property(const std::string& name) const {
    for (const auto& prop : type()->properties()) {
      if (prop.name == name) {
        std::optional<Method> setter = std::nullopt;
        if (prop.setter) {
          setter = Method(_ivalue(), prop.setter);
        }
        return Property{
            prop.name, Method(_ivalue(), prop.getter), std::move(setter)};
      }
    }
    TORCH_CHECK(false, "Property '", name, "' is not defined.");
  }

  const std::vector<Property> get_properties() const {
    return c10::fmap(type()->properties(), [&](ClassType::Property prop) {
      std::optional<Method> setter = std::nullopt;
      if (prop.setter) {
        setter = Method(_ivalue(), prop.setter);
      }
      return Property{
          std::move(prop.name),
          Method(_ivalue(), prop.getter),
          std::move(setter)};
    });
  }

  std::optional<Method> find_method(const std::string& basename) const;

  /// Run a method from this module.
  ///
  /// For example:
  /// @code
  ///   IValue output = module->run("relu_script", a, b);
  /// @endcode
  ///
  /// To get a compile a module from a source string, see torch::jit::compile
  ///
  /// @param method_name The name of the method to run
  /// @param args Arguments to be passed to the method
  /// @return An IValue containing the return value (or values if it is a tuple)
  /// from the method
  template <typename... Types>
  IValue run_method(const std::string& method_name, Types&&... args) {
    return get_method(method_name)({IValue(std::forward<Types>(args))...});
  }

  // so that C++ users can easily add methods
  void define(const std::string& src, const ResolverPtr& resolver = nullptr);

  size_t num_slots() const {
    return _ivalue()->slots().size();
  }

  // shallow copy the object
  Object copy() const;

  // Copies all the attributes of the object recursively without creating new
  // `ClassType`, including deepcopy of Tensors
  Object deepcopy() const;

 private:
  // mutable be we lazily initialize in module_object.
  mutable ObjectPtr _ivalue_;
};

namespace script {
// We once had a `script::` namespace that was deleted. This is for backcompat
// of the public API; new code should not use this type alias.
using Object = ::torch::jit::Object;
} // namespace script
} // namespace torch::jit

```



## High-Level Overview


This C++ file contains approximately 1 class(es)/struct(s) and 26 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `script`, `torch`, `that`

**Classes/Structs**: `Resolver`, `ObjectAttributeError`, `TORCH_API`, `Property`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/csrc/jit/api`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `ATen/core/functional.h`
- `ATen/core/ivalue.h`
- `torch/csrc/jit/api/method.h`
- `optional`
- `utility`


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

Files in the same folder (`torch/csrc/jit/api`):

- [`function_impl.h_docs.md`](./function_impl.h_docs.md)
- [`module.h_docs.md`](./module.h_docs.md)
- [`module.cpp_docs.md`](./module.cpp_docs.md)
- [`module_save.cpp_docs.md`](./module_save.cpp_docs.md)
- [`compilation_unit.h_docs.md`](./compilation_unit.h_docs.md)
- [`object.cpp_docs.md`](./object.cpp_docs.md)
- [`function_impl.cpp_docs.md`](./function_impl.cpp_docs.md)
- [`method.h_docs.md`](./method.h_docs.md)


## Cross-References

- **File Documentation**: `object.h_docs.md`
- **Keyword Index**: `object.h_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/csrc/jit/api`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/csrc/jit/api`, which is part of the **core PyTorch library**.



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

Test files for this module may be located in the `test/` directory.

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`docs/torch/csrc/jit/api`):

- [`compilation_unit.h_docs.md_docs.md`](./compilation_unit.h_docs.md_docs.md)
- [`object.cpp_docs.md_docs.md`](./object.cpp_docs.md_docs.md)
- [`compilation_unit.h_kw.md_docs.md`](./compilation_unit.h_kw.md_docs.md)
- [`function_impl.cpp_docs.md_docs.md`](./function_impl.cpp_docs.md_docs.md)
- [`object.h_kw.md_docs.md`](./object.h_kw.md_docs.md)
- [`module_save.cpp_kw.md_docs.md`](./module_save.cpp_kw.md_docs.md)
- [`module.h_kw.md_docs.md`](./module.h_kw.md_docs.md)
- [`method.h_kw.md_docs.md`](./method.h_kw.md_docs.md)
- [`module.h_docs.md_docs.md`](./module.h_docs.md_docs.md)
- [`method.h_docs.md_docs.md`](./method.h_docs.md_docs.md)


## Cross-References

- **File Documentation**: `object.h_docs.md_docs.md`
- **Keyword Index**: `object.h_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
