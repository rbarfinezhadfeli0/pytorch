# Documentation: `torch/csrc/jit/mobile/compatibility/model_compatibility.h`

## File Metadata

- **Path**: `torch/csrc/jit/mobile/compatibility/model_compatibility.h`
- **Size**: 3,610 bytes (3.53 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
#pragma once

#include <ATen/core/ivalue.h>
#include <c10/macros/Export.h>
#include <torch/csrc/jit/mobile/compatibility/runtime_compatibility.h>

#include <istream>
#include <memory>
#include <unordered_map>
#include <vector>

namespace caffe2::serialize {
class PyTorchStreamReader;
class ReadAdapterInterface;
} // namespace caffe2::serialize

namespace torch::jit {

// The family of methods below to get bytecode version from a model
// Throws if not passed in a well formed model
TORCH_API uint64_t _get_model_bytecode_version(std::istream& in);

TORCH_API uint64_t _get_model_bytecode_version(const std::string& filename);

TORCH_API uint64_t _get_model_bytecode_version(
    const std::shared_ptr<caffe2::serialize::ReadAdapterInterface>& rai);

uint64_t _get_model_bytecode_version(
    const std::vector<c10::IValue>& bytecode_ivalues);

// The family of methods below to get the operator version from a model
// Throws if not passed in a well formed model
TORCH_API uint64_t _get_model_operator_version(std::istream& in);

TORCH_API uint64_t _get_model_operator_version(const std::string& filename);

TORCH_API uint64_t _get_model_operator_version(
    std::shared_ptr<caffe2::serialize::ReadAdapterInterface> rai);

// Utility Functions
std::vector<c10::IValue> get_bytecode_ivalues(
    caffe2::serialize::PyTorchStreamReader& reader);

c10::IValue readArchive(
    const std::string& archive_name,
    caffe2::serialize::PyTorchStreamReader& stream_reader);

bool check_zip_file(
    const std::shared_ptr<caffe2::serialize::ReadAdapterInterface>& rai);

// The family of methods below to get the root ops and information from a model
TORCH_API std::unordered_map<std::string, OperatorInfo> _get_model_ops_and_info(
    std::istream& in);

TORCH_API std::unordered_map<std::string, OperatorInfo> _get_model_ops_and_info(
    const std::string& filename);

TORCH_API std::unordered_map<std::string, OperatorInfo> _get_model_ops_and_info(
    std::shared_ptr<caffe2::serialize::ReadAdapterInterface> rai);

// The family of methods below to get contained types from a model
// Throws if not passed in a well formed model
TORCH_API std::unordered_set<std::string> _get_mobile_model_contained_types(
    std::istream& in);

TORCH_API std::unordered_set<std::string> _get_mobile_model_contained_types(
    const std::string& filename);

TORCH_API std::unordered_set<std::string> _get_mobile_model_contained_types(
    std::shared_ptr<caffe2::serialize::ReadAdapterInterface> rai);

std::unordered_set<std::string> _get_mobile_model_contained_types(
    const std::vector<c10::IValue>& bytecode_ivalues);

// The family of methods below return the compatibility information of a model
struct ModelCompatibilityInfo {
  uint64_t bytecode_version;
  std::unordered_map<std::string, OperatorInfo> operator_info;
  std::unordered_set<std::string> type_table;
  uint64_t operator_version;

  // Factory Methods
  static TORCH_API ModelCompatibilityInfo get(std::istream& in);
  static TORCH_API ModelCompatibilityInfo get(const std::string& filename);
  static TORCH_API ModelCompatibilityInfo
  get(std::shared_ptr<caffe2::serialize::ReadAdapterInterface> rai);
};

enum ModelCompatibilityStatus {
  OK = 1,
  ERROR = 2,
};

struct ModelCompatCheckResult {
  ModelCompatibilityStatus status;
  std::vector<std::string> errors;
};
// Takes in information about a runtime and a model and returns if the two are
// compatible with one another.
TORCH_API ModelCompatCheckResult is_compatible(
    RuntimeCompatibilityInfo runtime_info,
    const ModelCompatibilityInfo& model_info);

} // namespace torch::jit

```



## High-Level Overview


This C++ file contains approximately 2 class(es)/struct(s) and 13 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `caffe2`, `torch`

**Classes/Structs**: `PyTorchStreamReader`, `ReadAdapterInterface`, `ModelCompatibilityInfo`, `ModelCompatCheckResult`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/csrc/jit/mobile/compatibility`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `ATen/core/ivalue.h`
- `c10/macros/Export.h`
- `torch/csrc/jit/mobile/compatibility/runtime_compatibility.h`
- `istream`
- `memory`
- `unordered_map`
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

Test files for this module may be located in the `test/` directory.

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`torch/csrc/jit/mobile/compatibility`):

- [`runtime_compatibility.cpp_docs.md`](./runtime_compatibility.cpp_docs.md)
- [`runtime_compatibility.h_docs.md`](./runtime_compatibility.h_docs.md)
- [`backport_manager.cpp_docs.md`](./backport_manager.cpp_docs.md)
- [`model_compatibility.cpp_docs.md`](./model_compatibility.cpp_docs.md)
- [`backport_manager.h_docs.md`](./backport_manager.h_docs.md)
- [`backport.h_docs.md`](./backport.h_docs.md)
- [`backport.cpp_docs.md`](./backport.cpp_docs.md)


## Cross-References

- **File Documentation**: `model_compatibility.h_docs.md`
- **Keyword Index**: `model_compatibility.h_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
