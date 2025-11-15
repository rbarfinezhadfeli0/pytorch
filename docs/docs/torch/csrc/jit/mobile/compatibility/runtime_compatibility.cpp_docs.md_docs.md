# Documentation: `docs/torch/csrc/jit/mobile/compatibility/runtime_compatibility.cpp_docs.md`

## File Metadata

- **Path**: `docs/torch/csrc/jit/mobile/compatibility/runtime_compatibility.cpp_docs.md`
- **Size**: 5,661 bytes (5.53 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/csrc/jit/mobile/compatibility/runtime_compatibility.cpp`

## File Metadata

- **Path**: `torch/csrc/jit/mobile/compatibility/runtime_compatibility.cpp`
- **Size**: 3,042 bytes (2.97 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#include <ATen/core/dispatch/Dispatcher.h>
#include <ATen/core/type_factory.h>
#include <caffe2/serialize/inline_container.h>
#include <torch/csrc/jit/mobile/compatibility/runtime_compatibility.h>
#include <torch/csrc/jit/mobile/type_parser.h>
#include <torch/csrc/jit/runtime/operator.h>
#include <torch/custom_class.h>
#include <unordered_map>

namespace torch::jit {

uint64_t _get_runtime_bytecode_version() {
  return caffe2::serialize::kMaxSupportedBytecodeVersion;
}

std::pair<uint64_t, uint64_t> _get_runtime_bytecode_min_max_versions() {
  return std::pair<uint64_t, uint64_t>(
      caffe2::serialize::kMinSupportedBytecodeVersion,
      caffe2::serialize::kMaxSupportedBytecodeVersion);
}

std::pair<uint64_t, uint64_t> _get_runtime_operators_min_max_versions() {
  return std::pair<uint64_t, uint64_t>(
      caffe2::serialize::kMinSupportedFileFormatVersion,
      caffe2::serialize::kMaxSupportedFileFormatVersion);
}

/*
 * Returns all registered PyTorch ops and their versioning
 */
std::unordered_map<std::string, OperatorInfo> _get_runtime_ops_and_info() {
  std::unordered_map<std::string, OperatorInfo> result;

  // Grab the jit operators
  auto nonDispatcherOperators = torch::jit::getAllOperators();
  for (const auto& full_op : nonDispatcherOperators) {
    auto op = full_op->schema();
    auto num_schema_args = op.arguments().size();
    auto op_name = op.name();
    if (!op.overload_name().empty()) {
      op_name += ("." + op.overload_name());
    }
    result.emplace(op_name, OperatorInfo{num_schema_args});
  }

  // Grab the dispatcher operators
  auto dispatcherOperators = c10::Dispatcher::singleton().getAllOpNames();
  for (auto& op : dispatcherOperators) {
    // grab schema
    const auto op_handle = c10::Dispatcher::singleton().findOp(op);
    std::optional<int> num_schema_args;
    if (op_handle->hasSchema()) {
      num_schema_args = op_handle->schema().arguments().size();
    }
    auto op_name = op.name;
    if (!op.overload_name.empty()) {
      op_name += ("." + op.overload_name);
    }
    result.emplace(op_name, OperatorInfo{num_schema_args});
  }

  return result;
}

RuntimeCompatibilityInfo RuntimeCompatibilityInfo::get() {
  return RuntimeCompatibilityInfo{
      _get_runtime_bytecode_min_max_versions(),
      _get_runtime_ops_and_info(),
      _get_mobile_supported_types(),
      _get_runtime_operators_min_max_versions()};
}

std::unordered_set<std::string> _get_mobile_supported_types() {
  std::unordered_set<std::string> supported_types;
  for (const auto& it : c10::DynamicTypeFactory::basePythonTypes()) {
    supported_types.insert(it.first);
  }
  supported_types.insert(
      at::TypeParser::getNonSimpleType().begin(),
      at::TypeParser::getNonSimpleType().end());
  supported_types.insert(
      at::TypeParser::getCustomType().begin(),
      at::TypeParser::getCustomType().end());

  return supported_types;
}

TORCH_API std::unordered_set<std::string> _get_loaded_custom_classes() {
  return torch::getAllCustomClassesNames();
}

} // namespace torch::jit

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 1 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `torch`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/csrc/jit/mobile/compatibility`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `ATen/core/dispatch/Dispatcher.h`
- `ATen/core/type_factory.h`
- `caffe2/serialize/inline_container.h`
- `torch/csrc/jit/mobile/compatibility/runtime_compatibility.h`
- `torch/csrc/jit/mobile/type_parser.h`
- `torch/csrc/jit/runtime/operator.h`
- `torch/custom_class.h`
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

Files in the same folder (`torch/csrc/jit/mobile/compatibility`):

- [`model_compatibility.h_docs.md`](./model_compatibility.h_docs.md)
- [`runtime_compatibility.h_docs.md`](./runtime_compatibility.h_docs.md)
- [`backport_manager.cpp_docs.md`](./backport_manager.cpp_docs.md)
- [`model_compatibility.cpp_docs.md`](./model_compatibility.cpp_docs.md)
- [`backport_manager.h_docs.md`](./backport_manager.h_docs.md)
- [`backport.h_docs.md`](./backport.h_docs.md)
- [`backport.cpp_docs.md`](./backport.cpp_docs.md)


## Cross-References

- **File Documentation**: `runtime_compatibility.cpp_docs.md`
- **Keyword Index**: `runtime_compatibility.cpp_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/csrc/jit/mobile/compatibility`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/csrc/jit/mobile/compatibility`, which is part of the **core PyTorch library**.



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

Files in the same folder (`docs/torch/csrc/jit/mobile/compatibility`):

- [`model_compatibility.cpp_docs.md_docs.md`](./model_compatibility.cpp_docs.md_docs.md)
- [`runtime_compatibility.cpp_kw.md_docs.md`](./runtime_compatibility.cpp_kw.md_docs.md)
- [`backport_manager.cpp_docs.md_docs.md`](./backport_manager.cpp_docs.md_docs.md)
- [`backport.cpp_kw.md_docs.md`](./backport.cpp_kw.md_docs.md)
- [`runtime_compatibility.h_kw.md_docs.md`](./runtime_compatibility.h_kw.md_docs.md)
- [`backport.h_kw.md_docs.md`](./backport.h_kw.md_docs.md)
- [`backport_manager.h_kw.md_docs.md`](./backport_manager.h_kw.md_docs.md)
- [`backport_manager.h_docs.md_docs.md`](./backport_manager.h_docs.md_docs.md)
- [`model_compatibility.h_kw.md_docs.md`](./model_compatibility.h_kw.md_docs.md)
- [`model_compatibility.cpp_kw.md_docs.md`](./model_compatibility.cpp_kw.md_docs.md)


## Cross-References

- **File Documentation**: `runtime_compatibility.cpp_docs.md_docs.md`
- **Keyword Index**: `runtime_compatibility.cpp_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
