# Documentation: `torch/csrc/jit/python/python_dict.h`

## File Metadata

- **Path**: `torch/csrc/jit/python/python_dict.h`
- **Size**: 3,383 bytes (3.30 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
#pragma once

#include <ATen/core/Dict.h>
#include <ATen/core/ivalue.h>
#include <ATen/core/jit_type.h>
#include <torch/csrc/utils/pybind.h>

namespace torch::jit {

void initScriptDictBindings(PyObject* module);

/// An iterator over the keys of ScriptDict. This is used to support
/// .keys() and iteration.
class ScriptDictKeyIterator final {
 public:
  ScriptDictKeyIterator(
      c10::impl::GenericDict::iterator iter,
      c10::impl::GenericDict::iterator end)
      : iter_(std::move(iter)), end_(std::move(end)) {}
  at::IValue next();

 private:
  c10::impl::GenericDict::iterator iter_;
  c10::impl::GenericDict::iterator end_;
};

/// An iterator over the key-value pairs of ScriptDict. This is used to support
/// .items().
class ScriptDictIterator final {
 public:
  ScriptDictIterator(
      c10::impl::GenericDict::iterator iter,
      c10::impl::GenericDict::iterator end)
      : iter_(std::move(iter)), end_(std::move(end)) {}
  at::IValue next();

 private:
  c10::impl::GenericDict::iterator iter_;
  c10::impl::GenericDict::iterator end_;
};

/// A wrapper around c10::Dict that can be exposed in Python via pybind
/// with an API identical to the Python dictionary class. This allows
/// dictionaries to have reference semantics across the Python/TorchScript
/// boundary.
class ScriptDict final {
 public:
  // Constructor.
  ScriptDict(const at::IValue& data)
      : dict_(at::AnyType::get(), at::AnyType::get()) {
    TORCH_INTERNAL_ASSERT(data.isGenericDict());
    dict_ = data.toGenericDict();
  }

  // Get the type of the dictionary.
  at::DictTypePtr type() const {
    return at::DictType::create(dict_.keyType(), dict_.valueType());
  }

  // Return a string representation that can be used
  // to reconstruct the instance.
  std::string repr() const {
    std::ostringstream s;
    s << '{';
    bool f = false;
    for (auto const& kv : dict_) {
      if (f) {
        s << ", ";
      }
      s << kv.key() << ": " << kv.value();
      f = true;
    }
    s << '}';
    return s.str();
  }

  // Return an iterator over the keys of the dictionary.
  ScriptDictKeyIterator iter() const {
    auto begin = dict_.begin();
    auto end = dict_.end();
    return ScriptDictKeyIterator(begin, end);
  }

  // Return an iterator over the key-value pairs of the dictionary.
  ScriptDictIterator items() const {
    auto begin = dict_.begin();
    auto end = dict_.end();
    return ScriptDictIterator(begin, end);
  }

  // Interpret the dictionary as a boolean; empty means false, non-empty means
  // true.
  bool toBool() const {
    return !(dict_.empty());
  }

  // Get the value for the given key. Throws std::out_of_range if the key does
  // not exist.
  at::IValue getItem(const at::IValue& key) {
    return dict_.at(key);
  }

  // Set the value for the given key.
  void setItem(const at::IValue& key, const at::IValue& value) {
    dict_.insert_or_assign(key, value);
  }

  // Check whether the dictionary contains the given key.
  bool contains(const at::IValue& key) {
    return dict_.contains(key);
  }

  // Delete the given key from the dictionary.
  bool delItem(const at::IValue& key) {
    return dict_.erase(key);
  }

  // Get the size of the dictionary.
  int64_t len() const {
    return dict_.size();
  }

  // A c10::Dict instance that holds the actual data.
  c10::impl::GenericDict dict_;
};

} // namespace torch::jit

```



## High-Level Overview


This C++ file contains approximately 3 class(es)/struct(s) and 15 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `torch`

**Classes/Structs**: `ScriptDictKeyIterator`, `ScriptDictIterator`, `ScriptDict`, `the`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/csrc/jit/python`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `ATen/core/Dict.h`
- `ATen/core/ivalue.h`
- `ATen/core/jit_type.h`
- `torch/csrc/utils/pybind.h`


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

Files in the same folder (`torch/csrc/jit/python`):

- [`python_ir.cpp_docs.md`](./python_ir.cpp_docs.md)
- [`python_custom_class.cpp_docs.md`](./python_custom_class.cpp_docs.md)
- [`python_ivalue.h_docs.md`](./python_ivalue.h_docs.md)
- [`pybind_utils.cpp_docs.md`](./pybind_utils.cpp_docs.md)
- [`opaque_obj.h_docs.md`](./opaque_obj.h_docs.md)
- [`update_graph_executor_opt.h_docs.md`](./update_graph_executor_opt.h_docs.md)
- [`python_tree_views.h_docs.md`](./python_tree_views.h_docs.md)
- [`python_ir.h_docs.md`](./python_ir.h_docs.md)
- [`python_tracer.h_docs.md`](./python_tracer.h_docs.md)
- [`python_arg_flatten.h_docs.md`](./python_arg_flatten.h_docs.md)


## Cross-References

- **File Documentation**: `python_dict.h_docs.md`
- **Keyword Index**: `python_dict.h_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
