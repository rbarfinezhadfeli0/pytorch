# Documentation: `docs/torch/csrc/utils/tensor_types.cpp_docs.md`

## File Metadata

- **Path**: `docs/torch/csrc/utils/tensor_types.cpp_docs.md`
- **Size**: 8,541 bytes (8.34 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/csrc/utils/tensor_types.cpp`

## File Metadata

- **Path**: `torch/csrc/utils/tensor_types.cpp`
- **Size**: 5,955 bytes (5.82 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#include <Python.h>

#include <torch/csrc/utils/tensor_types.h>

#include <ATen/Context.h>
#include <ATen/Formatting.h>
#include <torch/csrc/Exceptions.h>
#include <torch/csrc/autograd/generated/VariableType.h>
#include <torch/csrc/tensor/python_tensor.h>

#include <algorithm>
#include <sstream>
#include <unordered_map>

using namespace at;

namespace torch::utils {

static const char* parse_privateuseone_backend(bool is_sparse = false) {
  static std::string backend_name = "torch." + get_privateuse1_backend();
  static std::string sparse_backend_name = backend_name + ".sparse";
  return is_sparse == false ? backend_name.c_str()
                            : sparse_backend_name.c_str();
}

const char* backend_to_string(const at::Backend& backend) {
  switch (backend) {
    case at::Backend::CPU:
      return "torch";
    case at::Backend::CUDA:
      return "torch.cuda";
    case at::Backend::XPU:
      return "torch.xpu";
    case at::Backend::IPU:
      return "torch.ipu";
    case at::Backend::SparseCPU:
      return "torch.sparse";
    case at::Backend::SparseCUDA:
      return "torch.cuda.sparse";
    case at::Backend::SparseXPU:
      return "torch.xpu.sparse";
    case at::Backend::SparseMPS:
      return "torch.mps.sparse";
    case at::Backend::QuantizedCPU:
      return "torch.quantized";
    case at::Backend::HPU:
      return "torch.hpu";
    case at::Backend::MPS:
      return "torch.mps";
    case at::Backend::MTIA:
      return "torch.mtia";
    case at::Backend::PrivateUse1:
      return parse_privateuseone_backend();
    case at::Backend::SparsePrivateUse1:
      return parse_privateuseone_backend(true);
    case at::Backend::Lazy:
      return "torch.lazy";
    case at::Backend::XLA:
      return "torch.xla";
    case at::Backend::Meta:
      return "torch.meta";
    default:
      TORCH_CHECK(false, "Unimplemented backend ", backend);
  }
}

std::string options_to_string(const at::TensorOptions& options) {
  std::ostringstream ss;
  ss << backend_to_string(options.backend()) << "."
     << toString(at::typeMetaToScalarType(options.dtype())) << "Tensor";
  return ss.str();
}

std::string type_to_string(const at::DeprecatedTypeProperties& type) {
  std::ostringstream ss;
  ss << backend_to_string(type.backend()) << "." << toString(type.scalarType())
     << "Tensor";
  return ss.str();
}

at::TensorOptions options_from_string(const std::string& str) {
  static std::string cuda_prefix("torch.cuda.");
  static std::string xpu_prefix("torch.xpu.");
  static std::string privateUser_prefix(
      std::string(parse_privateuseone_backend()) + ".");
  static std::unordered_map<std::string, at::DeprecatedTypeProperties*> cpu_map;
  static std::unordered_map<std::string, at::DeprecatedTypeProperties*> xpu_map;
  static std::unordered_map<std::string, at::DeprecatedTypeProperties*>
      cuda_map;
  static std::unordered_map<std::string, at::DeprecatedTypeProperties*>
      privateUser1_map;

  const std::unordered_map<std::string, at::DeprecatedTypeProperties*>* map =
      nullptr;

  if (str == "torch.Tensor") {
    auto backend =
        dispatchKeyToBackend(torch::tensors::get_default_dispatch_key());
    auto scalar_type = torch::tensors::get_default_scalar_type();
    return getDeprecatedTypeProperties(backend, scalar_type).options();
  }

  if (std::mismatch(cuda_prefix.begin(), cuda_prefix.end(), str.begin())
          .first == cuda_prefix.end()) {
    // torch.cuda. is prefix of str
    static bool cuda_once [[maybe_unused]] = []() {
      for (auto type : autograd::VariableType::allCUDATypes()) {
        cuda_map.emplace(type_to_string(*type), type);
      }
      return true;
    }();
    map = &cuda_map;
  } else if (
      std::mismatch(xpu_prefix.begin(), xpu_prefix.end(), str.begin()).first ==
      xpu_prefix.end()) {
    // torch.xpu. is prefix of str
    static bool xpu_once [[maybe_unused]] = []() {
      for (auto type : autograd::VariableType::allXPUTypes()) {
        xpu_map.emplace(type_to_string(*type), type);
      }
      return true;
    }();
    map = &xpu_map;
  } else if (
      std::mismatch(
          privateUser_prefix.begin(), privateUser_prefix.end(), str.begin())
          .first == privateUser_prefix.end()) {
    // torch.foo. foo is privateUser1 name
    static bool privateUser1_once [[maybe_unused]] = []() {
      for (auto type : autograd::VariableType::allPrivateUser1Types()) {
        privateUser1_map.emplace(type_to_string(*type), type);
      }
      return true;
    }();
    map = &privateUser1_map;
  } else {
    static bool cpu_once [[maybe_unused]] = []() {
      for (auto type : autograd::VariableType::allCPUTypes()) {
        cpu_map.emplace(type_to_string(*type), type);
      }
      return true;
    }();
    map = &cpu_map;
  }

  auto it = map->find(str);
  TORCH_CHECK_VALUE(it != map->end(), "invalid type: '", str, "'");
  return it->second->options();
}

std::vector<std::pair<Backend, ScalarType>> all_declared_types() {
  std::vector<std::pair<Backend, ScalarType>> ret;

  // NOTE: Do not add more types here. This list controls the creation
  // of legacy tensor types e.g. torch.cuda.FloatTensor which are
  // maintained for backwards-compatibility only.
  auto backends = {
      Backend::CPU, Backend::CUDA, Backend::SparseCPU, Backend::SparseCUDA};
  auto scalar_types = {
      ScalarType::Byte,
      ScalarType::Char,
      ScalarType::Double,
      ScalarType::Float,
      ScalarType::Int,
      ScalarType::Long,
      ScalarType::Short,
      ScalarType::Half,
      ScalarType::Bool,
      ScalarType::BFloat16};

  for (auto& backend : backends) {
    for (auto& scalar_type : scalar_types) {
      // there is no sparse bool type.
      if (scalar_type == ScalarType::Bool &&
          (backend == Backend::SparseCUDA || backend == Backend::SparseCPU)) {
        continue;
      }
      ret.emplace_back(backend, scalar_type);
    }
  }

  return ret;
}

} // namespace torch::utils

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 11 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `torch`, `at`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/csrc/utils`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `Python.h`
- `torch/csrc/utils/tensor_types.h`
- `ATen/Context.h`
- `ATen/Formatting.h`
- `torch/csrc/Exceptions.h`
- `torch/csrc/autograd/generated/VariableType.h`
- `torch/csrc/tensor/python_tensor.h`
- `algorithm`
- `sstream`
- `unordered_map`


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.

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

Files in the same folder (`torch/csrc/utils`):

- [`tensor_list.h_docs.md`](./tensor_list.h_docs.md)
- [`disable_torch_function.cpp_docs.md`](./disable_torch_function.cpp_docs.md)
- [`tensor_new.cpp_docs.md`](./tensor_new.cpp_docs.md)
- [`tensor_apply.cpp_docs.md`](./tensor_apply.cpp_docs.md)
- [`cpp_stacktraces.cpp_docs.md`](./cpp_stacktraces.cpp_docs.md)
- [`numpy_stub.h_docs.md`](./numpy_stub.h_docs.md)
- [`nested.h_docs.md`](./nested.h_docs.md)
- [`nested.cpp_docs.md`](./nested.cpp_docs.md)
- [`six.h_docs.md`](./six.h_docs.md)
- [`python_scalars.h_docs.md`](./python_scalars.h_docs.md)


## Cross-References

- **File Documentation**: `tensor_types.cpp_docs.md`
- **Keyword Index**: `tensor_types.cpp_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/csrc/utils`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/csrc/utils`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
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

Files in the same folder (`docs/torch/csrc/utils`):

- [`python_tuples.h_kw.md_docs.md`](./python_tuples.h_kw.md_docs.md)
- [`six.h_kw.md_docs.md`](./six.h_kw.md_docs.md)
- [`tensor_list.h_kw.md_docs.md`](./tensor_list.h_kw.md_docs.md)
- [`verbose.h_kw.md_docs.md`](./verbose.h_kw.md_docs.md)
- [`invalid_arguments.cpp_kw.md_docs.md`](./invalid_arguments.cpp_kw.md_docs.md)
- [`tensor_apply.h_kw.md_docs.md`](./tensor_apply.h_kw.md_docs.md)
- [`cuda_enabled.h_docs.md_docs.md`](./cuda_enabled.h_docs.md_docs.md)
- [`tensor_layouts.h_docs.md_docs.md`](./tensor_layouts.h_docs.md_docs.md)
- [`variadic.h_kw.md_docs.md`](./variadic.h_kw.md_docs.md)


## Cross-References

- **File Documentation**: `tensor_types.cpp_docs.md_docs.md`
- **Keyword Index**: `tensor_types.cpp_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
