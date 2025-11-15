# Documentation: `docs/torch/csrc/inductor/aoti_runner/model_container_runner.h_docs.md`

## File Metadata

- **Path**: `docs/torch/csrc/inductor/aoti_runner/model_container_runner.h_docs.md`
- **Size**: 8,236 bytes (8.04 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/csrc/inductor/aoti_runner/model_container_runner.h`

## File Metadata

- **Path**: `torch/csrc/inductor/aoti_runner/model_container_runner.h`
- **Size**: 5,449 bytes (5.32 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
#if !defined(C10_MOBILE) && !defined(ANDROID)
#pragma once

#include <ATen/Tensor.h>
#include <torch/csrc/inductor/aoti_runtime/interface.h>
#include <torch/csrc/inductor/aoti_torch/proxy_executor.h>

// Forward declare DynamicLibrary
namespace at {
struct DynamicLibrary;
}

namespace torch::inductor {
using TensorConstantMap = std::unordered_map<std::string, at::Tensor*>;

class TORCH_API AOTIModelContainerRunner {
 public:
  AOTIModelContainerRunner() = delete;
  AOTIModelContainerRunner(const AOTIModelContainerRunner& other) = delete;
  AOTIModelContainerRunner(AOTIModelContainerRunner&& other) = delete;
  AOTIModelContainerRunner& operator=(const AOTIModelContainerRunner& other) =
      delete;
  AOTIModelContainerRunner& operator=(AOTIModelContainerRunner&& other) =
      delete;
  virtual ~AOTIModelContainerRunner();

  std::vector<at::Tensor> run(
      const std::vector<at::Tensor>& inputs,
      void* stream_handle = nullptr);

  // boxed_run will steal the ownership of the input tensors
  std::vector<at::Tensor> boxed_run(
      std::vector<at::Tensor>&& inputs,
      void* stream_handle = nullptr);

  std::unordered_map<std::string, std::string> getConstantNamesToOriginalFQNs()
      const;
  std::unordered_map<std::string, int32_t> getConstantNamesToDtypes() const;

  const std::unordered_map<std::string, at::Tensor> extract_constants_map(
      bool use_inactive) const;
  void update_inactive_constant_buffer(const TensorConstantMap& const_map);
  void update_constant_buffer(
      std::unordered_map<std::string, at::Tensor>& tensor_map,
      bool use_inactive,
      bool validate_full_updates,
      bool user_managed = false);
  void update_constant_buffer(
      const TensorConstantMap& const_map,
      bool use_inactive,
      bool validate_full_updates,
      bool user_managed = false);
  void run_const_fold(
      bool use_inactive,
      AOTInductorStreamHandle cuda_stream_handle = nullptr);
  void swap_constant_buffer();
  void free_inactive_constant_buffer();
  void update_constant_buffer_from_blob(const std::string& weights_path);

  std::vector<std::string> get_call_spec();

 protected:
  AOTIModelContainerRunner(
      const std::string& model_so_path,
      size_t num_models,
      const std::string& device_str,
      const std::string& cubin_dir,
      const bool run_single_threaded);

  virtual std::vector<at::Tensor> run_impl(
      std::vector<AtenTensorHandle>& input_handles,
      void* stream_handle);

  std::unique_ptr<at::DynamicLibrary> model_so_;
  decltype(&AOTInductorModelContainerCreateWithDevice) create_func_{nullptr};
  decltype(&AOTInductorModelContainerDelete) delete_func_{nullptr};
  decltype(&AOTInductorModelContainerGetNumOutputs) get_num_outputs_func_{
      nullptr};
  decltype(&AOTInductorModelContainerRun) run_func_{nullptr};
  decltype(&AOTInductorModelContainerGetNumConstants) get_num_constants_func_{
      nullptr};
  decltype(&AOTInductorModelContainerGetConstantName) get_constant_name_func_{
      nullptr};
  decltype(&AOTInductorModelContainerGetConstantOriginalFQN)
      get_constant_original_fqn_func_{nullptr};
  decltype(&AOTInductorModelContainerGetConstantDtype) get_constant_dtype_func_{
      nullptr};
  decltype(&AOTInductorModelContainerExtractConstantsMap)
      extract_constants_map_func_{nullptr};
  decltype(&AOTInductorModelContainerUpdateUserManagedConstantBuffer)
      update_user_managed_constant_buffer_func_{nullptr};
  decltype(&AOTInductorModelContainerUpdateConstantBuffer)
      update_constant_buffer_func_{nullptr};
  decltype(&AOTInductorModelContainerUpdateInactiveConstantBuffer)
      update_inactive_constant_buffer_func_{nullptr};
  decltype(&AOTInductorModelContainerRunConstantFolding) run_const_fold_func_{
      nullptr};
  decltype(&AOTInductorModelContainerSwapConstantBuffer)
      swap_constant_buffer_func_{nullptr};
  decltype(&AOTInductorModelContainerFreeInactiveConstantBuffer)
      free_inactive_constant_buffer_func_{nullptr};
  decltype(&AOTInductorModelContainerGetCallSpec) get_call_spec_func_{nullptr};
  decltype(&AOTInductorModelContainerGetConstantsBlobSize)
      get_constants_blob_size_func_{nullptr};
  decltype(&AOTInductorModelUpdateConstantsFromBlob)
      update_constants_from_blob_func_{nullptr};

  AOTInductorModelContainerHandle container_handle_ = nullptr;

  AOTIProxyExecutorHandle proxy_executor_handle_;

 private:
  std::unique_ptr<torch::aot_inductor::ProxyExecutor> proxy_executor_;
};

using CreateAOTIModelRunnerFunc = std::unique_ptr<AOTIModelContainerRunner> (*)(
    const std::string& model_so_path,
    size_t num_models,
    const std::string& device_str,
    const std::string& bin_dir,
    const bool run_single_threaded);

// Return a global map "device name" -> "aoti model runner create function" for
// all registered in AOTI external backends
TORCH_API std::unordered_map<std::string, CreateAOTIModelRunnerFunc>&
getAOTIModelRunnerRegistry();

// To register a new external backend in AOTI one needs to create an instance of
// this struct. It is not thread-safe. Because it is expected to be called
// during the initialization of the program.
struct TORCH_API RegisterAOTIModelRunner{RegisterAOTIModelRunner(
    const std::string& name,
    CreateAOTIModelRunnerFunc create_aoti_model_runner_fn){
    getAOTIModelRunnerRegistry()[name] = create_aoti_model_runner_fn;
} // namespace torch::inductor
}
;

} // namespace torch::inductor
#endif

```



## High-Level Overview


This C++ file contains approximately 1 class(es)/struct(s) and 7 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `torch`, `at`

**Classes/Structs**: `DynamicLibrary`, `TORCH_API`, `TORCH_API`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/csrc/inductor/aoti_runner`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `ATen/Tensor.h`
- `torch/csrc/inductor/aoti_runtime/interface.h`
- `torch/csrc/inductor/aoti_torch/proxy_executor.h`


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

Files in the same folder (`torch/csrc/inductor/aoti_runner`):

- [`model_container_runner_cuda.cpp_docs.md`](./model_container_runner_cuda.cpp_docs.md)
- [`model_container_runner_xpu.cpp_docs.md`](./model_container_runner_xpu.cpp_docs.md)
- [`model_container_runner_mps.cpp_docs.md`](./model_container_runner_mps.cpp_docs.md)
- [`model_container_runner_cpu.h_docs.md`](./model_container_runner_cpu.h_docs.md)
- [`model_container_runner_cuda.h_docs.md`](./model_container_runner_cuda.h_docs.md)
- [`pybind.cpp_docs.md`](./pybind.cpp_docs.md)
- [`model_container_runner_xpu.h_docs.md`](./model_container_runner_xpu.h_docs.md)
- [`model_container_runner_mps.h_docs.md`](./model_container_runner_mps.h_docs.md)
- [`model_container_runner.cpp_docs.md`](./model_container_runner.cpp_docs.md)


## Cross-References

- **File Documentation**: `model_container_runner.h_docs.md`
- **Keyword Index**: `model_container_runner.h_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/csrc/inductor/aoti_runner`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/csrc/inductor/aoti_runner`, which is part of the **core PyTorch library**.



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

Files in the same folder (`docs/torch/csrc/inductor/aoti_runner`):

- [`model_container_runner_xpu.cpp_kw.md_docs.md`](./model_container_runner_xpu.cpp_kw.md_docs.md)
- [`model_container_runner.cpp_kw.md_docs.md`](./model_container_runner.cpp_kw.md_docs.md)
- [`model_container_runner_cuda.h_docs.md_docs.md`](./model_container_runner_cuda.h_docs.md_docs.md)
- [`model_container_runner_xpu.h_kw.md_docs.md`](./model_container_runner_xpu.h_kw.md_docs.md)
- [`model_container_runner_cuda.h_kw.md_docs.md`](./model_container_runner_cuda.h_kw.md_docs.md)
- [`model_container_runner.cpp_docs.md_docs.md`](./model_container_runner.cpp_docs.md_docs.md)
- [`model_container_runner_cpu.cpp_kw.md_docs.md`](./model_container_runner_cpu.cpp_kw.md_docs.md)
- [`model_container_runner_cuda.cpp_kw.md_docs.md`](./model_container_runner_cuda.cpp_kw.md_docs.md)
- [`model_container_runner_cuda.cpp_docs.md_docs.md`](./model_container_runner_cuda.cpp_docs.md_docs.md)
- [`pybind.h_docs.md_docs.md`](./pybind.h_docs.md_docs.md)


## Cross-References

- **File Documentation**: `model_container_runner.h_docs.md_docs.md`
- **Keyword Index**: `model_container_runner.h_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
