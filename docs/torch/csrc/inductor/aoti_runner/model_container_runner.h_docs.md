# Documentation: model_container_runner.h

## File Metadata
- **Path**: `torch/csrc/inductor/aoti_runner/model_container_runner.h`
- **Size**: 5449 bytes
- **Lines**: 140
- **Extension**: .h
- **Type**: Regular file

## Original Source

```h
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

This file is part of the PyTorch repository. It is a C++/CUDA source/header file that may contain implementations, declarations, or kernel code.

## Detailed Walkthrough

### Classes
This file defines 1 class(es): TORCH_API

### Structures
This file defines 2 struct(s): DynamicLibrary, TORCH_API


## Key Components

The file contains 329 words across 140 lines of code/text.

## Usage & Examples

This file is part of the larger PyTorch codebase. For usage examples, refer to related test files and documentation.

## Performance & Security Notes

- File size: 5449 bytes
- Complexity: Standard

## Related Files

See the folder index for related files in the same directory.

## Testing

Refer to the PyTorch test suite for test coverage of this file.

---
*Generated by Repo Book Generator v1.0*
