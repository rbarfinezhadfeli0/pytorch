# Documentation: `docs/torch/csrc/jit/mobile/module.h_docs.md`

## File Metadata

- **Path**: `docs/torch/csrc/jit/mobile/module.h_docs.md`
- **Size**: 8,489 bytes (8.29 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/csrc/jit/mobile/module.h`

## File Metadata

- **Path**: `torch/csrc/jit/mobile/module.h`
- **Size**: 5,921 bytes (5.78 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
#pragma once
#include <ATen/core/jit_type.h>
#include <torch/csrc/jit/mobile/debug_info.h>
#include <torch/csrc/jit/mobile/function.h>
#include <torch/csrc/jit/mobile/method.h>
#include <torch/csrc/jit/mobile/quantization.h>

#include <utility>

namespace torch::jit::mobile {
using Stack = std::vector<c10::IValue>;

// A CompilationUnit object is the one that gets executed by the lite
// interpreter.
//
// A CompilationUnit object contains a list of Method Objects. These are methods
// that appear in the original PyTorch Model. These method correspond to Python
// member functions of the Model class.
//
// Methods in turn contain a Function, and a back-pointer to the Module that
// owns this Method instance.
//
// A Function contains a Code Object (code_) which is defined in interpreter.h
//
// A Code object contains the following:
//
// std::vector<Instruction> instructions_;
// std::vector<c10::OperatorName> op_names_;
// std::vector<std::function<void(Stack&)>> operators_;
// std::vector<c10::IValue> constants_;
// std::vector<c10::TypePtr> types_;
// size_t register_size_; // Aggregated output size.
//
class CompilationUnit {
 public:
  void register_function(std::unique_ptr<Function> fn);
  std::vector<std::unique_ptr<Function>>& methods() {
    return methods_;
  }
  const std::vector<std::unique_ptr<Function>>& methods() const {
    return methods_;
  }
  Function* find_function(const c10::QualifiedName& qn);
  const Function* find_function(const c10::QualifiedName& qn) const;

  void unsafeRemoveFunction(const int64_t index) {
    methods_.erase(methods_.begin() + index);
  }

 private:
  std::vector<std::unique_ptr<Function>> methods_;
};

// A Torch Mobile Module is a representation of the model (trained in case
// of inference). A Mobile Module contains
//
// 1. data (object_)
// 2. metadata (optional) about the model (metadata_ from the metadata.pkl
//    file added after training)
// 3. Compilation Unit (cu_)
//
class TORCH_API Module {
 public:
  Module(
      c10::intrusive_ptr<c10::ivalue::Object> object,
      std::shared_ptr<CompilationUnit> cu)
      : object_(std::move(object)), cu_(std::move(cu)) {}
  Module() = default;
  Method get_method(const std::string& method_name) const;
  template <typename... Types>
  c10::IValue run_method(const std::string& method_name, Types&&... args) {
    return get_method(method_name)({IValue(std::forward<Types>(args))...});
  }
  c10::IValue forward(std::vector<c10::IValue> inputs) {
    return get_method("forward")(std::move(inputs));
  }
  std::optional<Method> find_method(const std::string& basename) const;

  const std::string name() const {
    return object_->name();
  }
  const std::vector<at::IValue>& slots() const {
    return object_->slots();
  }
  const c10::intrusive_ptr<c10::ivalue::Object> _ivalue() const {
    return object_;
  }
  const std::vector<at::Tensor> parameters() const;
  const std::map<std::string, at::Tensor> named_parameters() const;
  std::string get_forward_method_debug_info(int64_t debug_handle) const;
  std::string getModuleHierarchy(const int64_t debug_handle) const;
  std::string getCallStack(const int64_t debug_handle) const;
  /// Enables "training" mode.
  void train(bool on = true);
  /// Calls train(false) to enable "eval" mode.
  void eval() {
    train(/*on=*/false);
  }
  /// True if the module is in training mode.
  bool is_training() const;
  const std::unordered_map<std::string, std::string> getMetadata() const {
    return metadata_;
  }
  void setMetadata(
      const std::unordered_map<std::string, std::string>& metadata) {
    metadata_ = metadata;
  }
  const std::vector<Method> get_methods() const;

  c10::IValue attr(const std::string& name, c10::IValue or_else) const {
    if (auto r = object_->type()->findAttributeSlot(name)) {
      return object_->getSlot(*r);
    }
    if (auto r = object_->type()->findConstantSlot(name)) {
      return object_->type()->getConstant(*r);
    }
    return or_else;
  }

  void setDebugTable(MobileDebugTable&& debug_table) {
    debug_table_ = std::move(debug_table);
  }
  const MobileDebugTable& getDebugTable() const {
    return debug_table_;
  }

  void setHasDebugHandles(bool has_debug_handles) {
    has_debug_handles_ = has_debug_handles;
  }

  bool hasDebugHandles() const {
    return has_debug_handles_;
  }

  const CompilationUnit& compilation_unit() const {
    return *cu_;
  }

  void set_delete_memory(std::shared_ptr<char> delete_mem) {
    mem_to_delete_ = std::move(delete_mem);
  }

  void set_min_operator_version(int64_t version) {
    min_operator_version_ = version;
  }

  int64_t min_operator_version() const {
    return min_operator_version_;
  }

  void set_bytecode_version(int64_t version) {
    bytecode_version_ = version;
  }

  int64_t bytecode_version() const {
    return bytecode_version_;
  }

 private:
  friend class quantization::PTQQuanizationHelper;

  bool compareMethodSchemas(
      const std::string& name_1,
      const std::string& name_2);

  void unsafeRemoveMethod(const std::string& basename);

  void unsafeCopyMethod(
      const std::string& new_method_name,
      const Function& to_be_copied);

  c10::intrusive_ptr<c10::ivalue::Object> object_;
  std::unordered_map<std::string, std::string> metadata_;
  std::shared_ptr<CompilationUnit> cu_;
  MobileDebugTable debug_table_;
  bool has_debug_handles_ = false;
  int64_t min_operator_version_ = 4;
  int64_t bytecode_version_ = 4;

  // Extra handle for the module to delete when itself is deleted
  std::shared_ptr<char> mem_to_delete_;
};

struct TORCH_API ModuleInfo {
  uint64_t bytecode_version;
  uint64_t operator_version;
  std::unordered_map<std::string, int> opname_to_num_args;
  std::unordered_set<std::string> function_names;
  std::unordered_set<std::string> type_names;
};
TORCH_API ModuleInfo get_module_info(const mobile::Module& module);

} // namespace torch::jit::mobile

```



## High-Level Overview


This C++ file contains approximately 3 class(es)/struct(s) and 33 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `torch`

**Classes/Structs**: `CompilationUnit`, `TORCH_API`, `quantization`, `TORCH_API`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/csrc/jit/mobile`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `ATen/core/jit_type.h`
- `torch/csrc/jit/mobile/debug_info.h`
- `torch/csrc/jit/mobile/function.h`
- `torch/csrc/jit/mobile/method.h`
- `torch/csrc/jit/mobile/quantization.h`
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

- **Code Execution**: Uses `eval()` or `exec()` - ensure input is sanitized

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

Test files for this module may be located in the `test/` directory.

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`torch/csrc/jit/mobile`):

- [`register_ops_common_utils.cpp_docs.md`](./register_ops_common_utils.cpp_docs.md)
- [`import.h_docs.md`](./import.h_docs.md)
- [`prim_ops_registery.h_docs.md`](./prim_ops_registery.h_docs.md)
- [`profiler_edge.h_docs.md`](./profiler_edge.h_docs.md)
- [`interpreter.h_docs.md`](./interpreter.h_docs.md)
- [`file_format.h_docs.md`](./file_format.h_docs.md)
- [`observer.h_docs.md`](./observer.h_docs.md)
- [`module.cpp_docs.md`](./module.cpp_docs.md)
- [`flatbuffer_loader.cpp_docs.md`](./flatbuffer_loader.cpp_docs.md)


## Cross-References

- **File Documentation**: `module.h_docs.md`
- **Keyword Index**: `module.h_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/csrc/jit/mobile`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/csrc/jit/mobile`, which is part of the **core PyTorch library**.



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

- **Code Execution**: Uses `eval()` or `exec()` - ensure input is sanitized

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

Test files for this module may be located in the `test/` directory.

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`docs/torch/csrc/jit/mobile`):

- [`code.h_docs.md_docs.md`](./code.h_docs.md_docs.md)
- [`register_ops_common_utils.cpp_docs.md_docs.md`](./register_ops_common_utils.cpp_docs.md_docs.md)
- [`observer.h_kw.md_docs.md`](./observer.h_kw.md_docs.md)
- [`prim_ops_registery.cpp_kw.md_docs.md`](./prim_ops_registery.cpp_kw.md_docs.md)
- [`quantization.h_docs.md_docs.md`](./quantization.h_docs.md_docs.md)
- [`debug_info.cpp_kw.md_docs.md`](./debug_info.cpp_kw.md_docs.md)
- [`interpreter.cpp_kw.md_docs.md`](./interpreter.cpp_kw.md_docs.md)
- [`debug_info.h_docs.md_docs.md`](./debug_info.h_docs.md_docs.md)
- [`interpreter.cpp_docs.md_docs.md`](./interpreter.cpp_docs.md_docs.md)
- [`promoted_prim_ops.cpp_docs.md_docs.md`](./promoted_prim_ops.cpp_docs.md_docs.md)


## Cross-References

- **File Documentation**: `module.h_docs.md_docs.md`
- **Keyword Index**: `module.h_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
