# Documentation: `torch/csrc/inductor/aoti_eager/kernel_holder.h`

## File Metadata

- **Path**: `torch/csrc/inductor/aoti_eager/kernel_holder.h`
- **Size**: 4,125 bytes (4.03 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
#if !defined(C10_MOBILE) && !defined(ANDROID)
#pragma once

#include <ATen/ATen.h>
#include <ATen/core/boxing/KernelFunction.h>
#include <ATen/core/function_schema.h>

#include <torch/csrc/dynamo/guards.h>
#include <torch/csrc/inductor/aoti_eager/kernel_meta_info.h>
#include <torch/csrc/inductor/aoti_runner/model_container_runner.h>
#include <torch/csrc/utils/pybind.h>

#include <string>

namespace torch::inductor {

// Represent AOTI kernel. It contains all the parameter metadata of the kernel
// and the AOTI model runner.
struct AOTIKernelMetadata {
  // Represent all the parameters of AOTI kernel
  std::vector<ParameterMetadata> parameter_metadata_list_;
  // AOTI model runner to run the AOTI kernel
  std::shared_ptr<AOTIModelContainerRunner> kernel_runner_;
  AOTIKernelMetadata() : kernel_runner_(nullptr) {}

  // Check whether the given parameter metadata list is the same as the
  // parameter metadata list of the AOTI kernel.
  bool check(
      const std::vector<ParameterMetadata>& parameter_metadata_list) const {
    if (parameter_metadata_list_.size() != parameter_metadata_list.size()) {
      return false;
    }

    for (size_t i = 0; i < parameter_metadata_list_.size(); ++i) {
      if (parameter_metadata_list_[i] == parameter_metadata_list[i]) {
        continue;
      } else {
        return false;
      }
    }

    return true;
  }
};

// The AOTIPythonKernelHolder class uses the AOT Inductor to generate a kernel
// for a specified operation. To speed up this process, the generated kernel
// library is cached on disk. Detailed information from the input tensors is
// used as the key for caching the kernel library. On subsequent runs, these
// input tensors are used to search the cache. If a cache hit occurs, the cached
// kernel library is loaded and executed. If a cache miss occurs, the AOT
// Inductor is called again to generate the kernel library.
class AOTIPythonKernelHolder : public c10::OperatorKernel {
  // A DispatchKey object that represents the dispatch key for the kernel.
  c10::DispatchKey dispatch_key_;
  // Namespace of the kernel.
  std::string ns_;
  // Name of the operation the kernel performs.
  std::string op_name_with_overload_;
  // The device on which the kernel is to be executed.
  c10::Device device_;
  // The Python interpreter to get OpOverload object with the given op_name and
  // op_overload_name.
  c10::impl::PyInterpreter* pyinterpreter_;
  // Cache the produced kernels by AOTI and its metadata
  std::vector<AOTIKernelMetadata> aoti_kernel_cache_;

 public:
  AOTIPythonKernelHolder(
      c10::DispatchKey dispatch_key,
      std::string_view ns,
      std::string_view op_name_with_overload);

  void operator()(
      const c10::OperatorHandle& op,
      c10::DispatchKeySet keyset,
      torch::jit::Stack* stack);

 private:
  bool cache_lookup(
      const c10::OperatorHandle& op,
      const c10::DispatchKeySet& keyset,
      const torch::jit::Stack* stack,
      AOTIKernelMetadata& aoti_kernel_metadata);
  void cache_miss(
      const c10::OperatorHandle& op,
      const c10::DispatchKeySet& keyset,
      torch::jit::Stack* stack);
  void cache_hit(
      const AOTIKernelMetadata& aoti_kernel_metadata,
      const c10::OperatorHandle& op,
      const c10::DispatchKeySet& keyset,
      torch::jit::Stack* stack);
  // Invoke python utility function on the Inductor side to produce AOTI kernel
  // for the given operation.
  //   Inductor utility function -
  //   torch._inductor.utils.aoti_compile_with_persistent_cache
  std::string produce_aoti_kernel_lib(
      const c10::OperatorHandle& op,
      const c10::DispatchKeySet& keyset,
      const torch::jit::Stack* stack);
  // Invoke python utility function on the Inductor side to load AOTI kernel for
  // the given operation.
  //   Inductor utility function - torch._inductor.utils.load_aoti_eager_cache
  void init_aoti_kernel_cache();
  // Load the AOTIModelContainerRunner object from the given file path.
  std::shared_ptr<AOTIModelContainerRunner> load_aoti_model_runner(
      const std::string& /*so_path*/);
};

} // namespace torch::inductor
#endif

```



## High-Level Overview


This C++ file contains approximately 2 class(es)/struct(s) and 7 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `torch`

**Classes/Structs**: `AOTIKernelMetadata`, `uses`, `AOTIPythonKernelHolder`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/csrc/inductor/aoti_eager`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `ATen/ATen.h`
- `ATen/core/boxing/KernelFunction.h`
- `ATen/core/function_schema.h`
- `torch/csrc/dynamo/guards.h`
- `torch/csrc/inductor/aoti_eager/kernel_meta_info.h`
- `torch/csrc/inductor/aoti_runner/model_container_runner.h`
- `torch/csrc/utils/pybind.h`
- `string`


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- Implements or uses **caching** mechanisms.
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

Files in the same folder (`torch/csrc/inductor/aoti_eager`):

- [`kernel_holder.cpp_docs.md`](./kernel_holder.cpp_docs.md)
- [`kernel_meta_info.h_docs.md`](./kernel_meta_info.h_docs.md)
- [`kernel_meta_info.cpp_docs.md`](./kernel_meta_info.cpp_docs.md)


## Cross-References

- **File Documentation**: `kernel_holder.h_docs.md`
- **Keyword Index**: `kernel_holder.h_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
