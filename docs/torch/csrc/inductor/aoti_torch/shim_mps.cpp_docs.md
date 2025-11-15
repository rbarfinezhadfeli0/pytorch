# Documentation: `torch/csrc/inductor/aoti_torch/shim_mps.cpp`

## File Metadata

- **Path**: `torch/csrc/inductor/aoti_torch/shim_mps.cpp`
- **Size**: 4,790 bytes (4.68 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#include <ATen/native/mps/MetalShaderLibrary.h>
#include <torch/csrc/inductor/aoti_torch/c/shim_mps.h>
#include <torch/csrc/inductor/aoti_torch/utils.h>

using namespace torch::aot_inductor;

AOTITorchError aoti_torch_mps_set_arg_tensor(
    AOTIMetalKernelFunctionHandle handle,
    unsigned idx,
    AtenTensorHandle tensor) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    auto t = tensor_handle_to_tensor_pointer(tensor);
    if (t == nullptr) {
      throw std::runtime_error("Tensor is null.");
    }
    auto func = reinterpret_cast<at::native::mps::MetalKernelFunction*>(handle);
    func->setArg(idx, *t);
  });
}

AOTITorchError aoti_torch_mps_set_arg_int(
    AOTIMetalKernelFunctionHandle handle,
    unsigned idx,
    int64_t val) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    auto func = reinterpret_cast<at::native::mps::MetalKernelFunction*>(handle);
    func->setArg(idx, val);
  });
}

AOTITorchError aoti_torch_mps_create_shader_library(
    const char* metal_shader_source,
    AOTIMetalShaderLibraryHandle* library_handle) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    auto* library = new at::native::mps::DynamicMetalShaderLibrary(
        std::string(metal_shader_source));
    *library_handle = reinterpret_cast<AOTIMetalShaderLibraryHandle>(library);
  });
}

AOTITorchError aoti_torch_mps_delete_shader_library(
    AOTIMetalShaderLibraryHandle library_handle) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    auto* library =
        reinterpret_cast<at::native::mps::MetalShaderLibrary*>(library_handle);
    delete library;
  });
}

AOTITorchError aoti_torch_mps_get_kernel_function(
    AOTIMetalShaderLibraryHandle library_handle,
    const char* kernel_name,
    AOTIMetalKernelFunctionHandle* function_handle) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    auto* library =
        reinterpret_cast<at::native::mps::MetalShaderLibrary*>(library_handle);
    auto* function =
        library->getCachedKernelFunctionPtr(std::string(kernel_name));
    *function_handle =
        reinterpret_cast<AOTIMetalKernelFunctionHandle>(function);
  });
}

AOTITorchError aoti_torch_mps_start_encoding(
    AOTIMetalKernelFunctionHandle func) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    auto* function_ptr =
        reinterpret_cast<at::native::mps::MetalKernelFunction*>(func);
    function_ptr->startEncoding();
  });
}

AOTITorchError aoti_torch_mps_dispatch_single(
    AOTIMetalKernelFunctionHandle func,
    uint64_t length) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    auto* function_ptr =
        reinterpret_cast<at::native::mps::MetalKernelFunction*>(func);
    function_ptr->dispatch(length);
  });
}

AOTITorchError aoti_torch_mps_dispatch_single_with_group_size(
    AOTIMetalKernelFunctionHandle func,
    uint64_t length,
    uint64_t group_size) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    auto* function_ptr =
        reinterpret_cast<at::native::mps::MetalKernelFunction*>(func);
    function_ptr->dispatch(length, group_size);
  });
}

AOTITorchError aoti_torch_mps_dispatch_array(
    AOTIMetalKernelFunctionHandle func,
    const uint64_t* length,
    size_t length_size) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    auto* function_ptr =
        reinterpret_cast<at::native::mps::MetalKernelFunction*>(func);
    c10::ArrayRef<uint64_t> length_ref(length, length_size);
    function_ptr->dispatch(length_ref);
  });
}

AOTITorchError aoti_torch_mps_dispatch_array_with_group_size(
    AOTIMetalKernelFunctionHandle func,
    const uint64_t* length,
    size_t length_size,
    const uint64_t* group_size,
    size_t group_size_size) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    auto* function_ptr =
        reinterpret_cast<at::native::mps::MetalKernelFunction*>(func);
    c10::ArrayRef<uint64_t> length_ref(length, length_size);
    c10::ArrayRef<uint64_t> group_size_ref(group_size, group_size_size);
    function_ptr->dispatch(length_ref, group_size_ref);
  });
}

// Shared callback function for std::function trampoline
void aoti_torch_mps_shared_callback(
    AOTIMetalKernelFunctionHandle func,
    void* user_data) {
  auto* function_wrapper =
      static_cast<std::function<void(AOTIMetalKernelFunctionHandle)>*>(
          user_data);
  (*function_wrapper)(func);
}

// Pure C version using function pointer and user data for trampoline pattern
AOTITorchError aoti_torch_mps_run_command_block(
    AOTIMetalKernelFunctionHandle func,
    aoti_torch_mps_command_block_callback_t callback,
    void* user_data) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    auto* function_ptr =
        reinterpret_cast<at::native::mps::MetalKernelFunction*>(func);
    function_ptr->runCommandBlock(
        [callback, func, user_data]() { callback(func, user_data); });
  });
}

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 12 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `torch`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/csrc/inductor/aoti_torch`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `ATen/native/mps/MetalShaderLibrary.h`
- `torch/csrc/inductor/aoti_torch/c/shim_mps.h`
- `torch/csrc/inductor/aoti_torch/utils.h`


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- Implements or uses **caching** mechanisms.

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
- [`mkldnn_tensor.h_docs.md`](./mkldnn_tensor.h_docs.md)
- [`proxy_executor.h_docs.md`](./proxy_executor.h_docs.md)
- [`shim_cuda.cpp_docs.md`](./shim_cuda.cpp_docs.md)
- [`shim_common.cpp_docs.md`](./shim_common.cpp_docs.md)
- [`oss_proxy_executor.cpp_docs.md`](./oss_proxy_executor.cpp_docs.md)


## Cross-References

- **File Documentation**: `shim_mps.cpp_docs.md`
- **Keyword Index**: `shim_mps.cpp_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
