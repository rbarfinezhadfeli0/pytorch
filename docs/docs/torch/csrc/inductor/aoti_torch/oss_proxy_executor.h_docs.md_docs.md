# Documentation: `docs/torch/csrc/inductor/aoti_torch/oss_proxy_executor.h_docs.md`

## File Metadata

- **Path**: `docs/torch/csrc/inductor/aoti_torch/oss_proxy_executor.h_docs.md`
- **Size**: 7,462 bytes (7.29 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/csrc/inductor/aoti_torch/oss_proxy_executor.h`

## File Metadata

- **Path**: `torch/csrc/inductor/aoti_torch/oss_proxy_executor.h`
- **Size**: 4,639 bytes (4.53 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
#pragma once

#include <ATen/core/dispatch/Dispatcher.h>
#include <ATen/core/ivalue.h>
#include <c10/macros/Export.h>
#include <nlohmann/json.hpp>
#include <torch/csrc/inductor/aoti_torch/c/shim.h>
#include <torch/csrc/inductor/aoti_torch/proxy_executor.h>
#include <torch/csrc/jit/api/function_impl.h> // @manual
#include <iostream>
#include <utility>

namespace torch::aot_inductor {

inline std::ostream& operator<<(std::ostream& os, DynamicArgType arg_type) {
  os << static_cast<int>(arg_type);
  return os;
}

struct OSSDynamicArg {
  OSSDynamicArg(
      int arg_index,
      DynamicArgType arg_type,
      int length,
      std::optional<std::vector<std::string>> list_item_types = std::nullopt)
      : arg_index(arg_index),
        arg_type(arg_type),
        length(length),
        list_item_types(std::move(list_item_types)) {}
  int arg_index;
  DynamicArgType arg_type;
  int length;
  std::optional<std::vector<std::string>>
      list_item_types; // only used for parsing list of optional tensors
};

struct OSSTorchBindArg {
  OSSTorchBindArg(int arg_index, std::string arg_name)
      : arg_index(arg_index), arg_name(std::move(arg_name)) {}
  int arg_index;
  // arg_name is used to find the corresponding IValue in customObjs_
  std::string arg_name;
};

struct OSSOpKernel {
  explicit OSSOpKernel(std::string target) : target_(std::move(target)) {}
  // Explicitly declare copy and move constructors
  OSSOpKernel(const OSSOpKernel&) = default;
  OSSOpKernel(OSSOpKernel&&) = default;
  // Explicitly declare copy and move assignment operators
  OSSOpKernel& operator=(const OSSOpKernel&) = default;
  OSSOpKernel& operator=(OSSOpKernel&&) = default;

  std::string target_;
  std::vector<OSSDynamicArg> dynamic_args_;
  std::vector<OSSTorchBindArg> torchbind_args_;
  std::vector<OSSDynamicArg> outputs_;
  std::vector<c10::IValue> stack_;

  int num_output_tensors() const {
    int num_output_tensors = 0;
    for (const auto& output : outputs_) {
      if (isTensorType(output.arg_type)) {
        num_output_tensors += output.length;
      }
    }
    return num_output_tensors;
  }

  int num_output_ints() const {
    int num_output_ints = 0;
    for (const auto& output : outputs_) {
      if (output.arg_type == DynamicArgType::IntType) {
        num_output_ints += output.length;
      }
    }
    return num_output_ints;
  }

  virtual void run(std::vector<c10::IValue>& stack) = 0;
  virtual c10::FunctionSchema schema() const = 0;
  virtual ~OSSOpKernel() = default;
};

struct OSSOpKernelOperator : public OSSOpKernel {
  OSSOpKernelOperator(std::string target, c10::OperatorHandle op_handle)
      : OSSOpKernel(std::move(target)), op_handle_(std::move(op_handle)) {}

  c10::OperatorHandle op_handle_;
  void run(std::vector<c10::IValue>& stack) override {
    op_handle_.callBoxed(stack);
  }

  c10::FunctionSchema schema() const override {
    return op_handle_.schema();
  }
};

struct OSSCallTorchBindKernel : public OSSOpKernel {
  OSSCallTorchBindKernel(std::string target, torch::jit::Function* method)
      : OSSOpKernel(std::move(target)), method_(method) {}
  torch::jit::Function* method_;
  void run(std::vector<c10::IValue>& stack) override {
    method_->run(stack);
  }

  c10::FunctionSchema schema() const override {
    return method_->getSchema();
  }
};

class OSSProxyExecutor : public ProxyExecutor {
 public:
  explicit OSSProxyExecutor(
      const std::string& json_path,
      bool is_cpu,
      std::optional<std::unordered_map<std::string, c10::IValue>> custom_objs =
          std::nullopt);

  void call_function(
      int extern_node_index,
      int num_ints,
      int64_t* flatten_int_args,
      int num_tensors,
      AtenTensorHandle* flatten_tensor_args) override;

 private:
  void prefill_stack_with_static_arguments(
      size_t index,
      const at::TypePtr& schema_arg_type,
      const nlohmann::json& serialized_arg,
      OSSOpKernel* op_kernel,
      const std::string& torchbind_arg_name);

  void get_input_info_from_serialized(
      const std::vector<c10::Argument>& schema_args,
      const nlohmann::json& serialized_node,
      OSSOpKernel& op_kernel);

  void get_output_info_from_serialized(
      const std::vector<c10::Argument>& schema_returns,
      const nlohmann::json& serialized_node,
      OSSOpKernel& op_kernel);

  std::unique_ptr<OSSCallTorchBindKernel> get_call_torch_bind_kernel(
      const nlohmann::json& serialized_node);

  std::vector<std::unique_ptr<OSSOpKernel>> op_kernels_;
  std::unique_ptr<c10::Device> device_;
  std::unordered_map<std::string, c10::IValue> custom_objs_;
};

} // namespace torch::aot_inductor

```



## High-Level Overview


This C++ file contains approximately 1 class(es)/struct(s) and 15 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `torch`

**Classes/Structs**: `OSSDynamicArg`, `OSSTorchBindArg`, `OSSOpKernel`, `OSSOpKernelOperator`, `OSSCallTorchBindKernel`, `OSSProxyExecutor`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/csrc/inductor/aoti_torch`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `ATen/core/dispatch/Dispatcher.h`
- `ATen/core/ivalue.h`
- `c10/macros/Export.h`
- `nlohmann/json.hpp`
- `torch/csrc/inductor/aoti_torch/c/shim.h`
- `torch/csrc/inductor/aoti_torch/proxy_executor.h`
- `torch/csrc/jit/api/function_impl.h`
- `iostream`
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

Files in the same folder (`torch/csrc/inductor/aoti_torch`):

- [`shim_xpu.cpp_docs.md`](./shim_xpu.cpp_docs.md)
- [`tensor_converter.h_docs.md`](./tensor_converter.h_docs.md)
- [`tensor_converter.cpp_docs.md`](./tensor_converter.cpp_docs.md)
- [`utils.h_docs.md`](./utils.h_docs.md)
- [`shim_mps.cpp_docs.md`](./shim_mps.cpp_docs.md)
- [`mkldnn_tensor.h_docs.md`](./mkldnn_tensor.h_docs.md)
- [`proxy_executor.h_docs.md`](./proxy_executor.h_docs.md)
- [`shim_cuda.cpp_docs.md`](./shim_cuda.cpp_docs.md)
- [`shim_common.cpp_docs.md`](./shim_common.cpp_docs.md)
- [`oss_proxy_executor.cpp_docs.md`](./oss_proxy_executor.cpp_docs.md)


## Cross-References

- **File Documentation**: `oss_proxy_executor.h_docs.md`
- **Keyword Index**: `oss_proxy_executor.h_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/csrc/inductor/aoti_torch`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/csrc/inductor/aoti_torch`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
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

Files in the same folder (`docs/torch/csrc/inductor/aoti_torch`):

- [`oss_proxy_executor.h_kw.md_docs.md`](./oss_proxy_executor.h_kw.md_docs.md)
- [`shim_cuda.cpp_kw.md_docs.md`](./shim_cuda.cpp_kw.md_docs.md)
- [`shim_cpu.cpp_kw.md_docs.md`](./shim_cpu.cpp_kw.md_docs.md)
- [`oss_proxy_executor.cpp_kw.md_docs.md`](./oss_proxy_executor.cpp_kw.md_docs.md)
- [`shim_mps.cpp_docs.md_docs.md`](./shim_mps.cpp_docs.md_docs.md)
- [`shim_common.cpp_kw.md_docs.md`](./shim_common.cpp_kw.md_docs.md)
- [`utils.h_docs.md_docs.md`](./utils.h_docs.md_docs.md)
- [`shim_xpu.cpp_docs.md_docs.md`](./shim_xpu.cpp_docs.md_docs.md)
- [`tensor_converter.cpp_docs.md_docs.md`](./tensor_converter.cpp_docs.md_docs.md)
- [`shim_common.cpp_docs.md_docs.md`](./shim_common.cpp_docs.md_docs.md)


## Cross-References

- **File Documentation**: `oss_proxy_executor.h_docs.md_docs.md`
- **Keyword Index**: `oss_proxy_executor.h_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
