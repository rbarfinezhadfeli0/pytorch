# Documentation: `aten/src/ATen/native/mps/MetalShaderLibrary.h`

## File Metadata

- **Path**: `aten/src/ATen/native/mps/MetalShaderLibrary.h`
- **Size**: 6,101 bytes (5.96 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
#pragma once
#ifdef __OBJC__
#include <Metal/Metal.h>
typedef id<MTLLibrary> MTLLibrary_t;
typedef id<MTLFunction> MTLFunction_t;
typedef id<MTLComputePipelineState> MTLComputePipelineState_t;
typedef id<MTLComputeCommandEncoder> MTLComputeCommandEncoder_t;
#else
typedef void MTLCompileOptions;
typedef void* MTLLibrary_t;
typedef void* MTLFunction_t;
typedef void* MTLComputePipelineState_t;
typedef void* MTLComputeCommandEncoder_t;
#endif

#include <c10/core/Scalar.h>
#include <c10/util/OptionalArrayRef.h>
#include <functional>
#include <optional>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

// Forward declaration of TensorBase and TensorIteratorBase
namespace at {
class TensorBase;
struct TensorIteratorBase;
} // namespace at

namespace at::native::mps {

namespace detail {
template <typename T>
class has_size_type {
  template <typename U>
  static constexpr std::true_type check(typename U::size_type*);
  template <typename>
  static constexpr std::false_type check(...);

 public:
  static constexpr bool value = decltype(check<T>(nullptr))::value;
};

template <typename T>
constexpr bool has_size_type_v = has_size_type<T>::value;

} // namespace detail

// Returns `gpuAddress` of respective `id<MTLBuffer>` plus storage offset
void* get_tensor_gpu_address(const at::TensorBase&);

class MetalKernelFunction {
 public:
  MetalKernelFunction(MTLComputePipelineState_t cps_, MTLFunction_t f_);
  ~MetalKernelFunction();
  MetalKernelFunction(MetalKernelFunction&) = delete;
  // Shader properties
  uint64_t getMaxThreadsPerThreadgroup() const;
  uint64_t getThreadExecutionWidth() const;
  uint64_t getStaticThreadGroupMemoryLength() const;
  void runCommandBlock(std::function<void(void)> f);
  // Methods below should be called from runCommandBlock function
  void startEncoding();
  void setArg(unsigned idx, const at::TensorBase& t);
  void setArg(unsigned idx, const void* ptr, uint64_t size);
  template <
      typename T,
      typename = std::enable_if_t<
          std::is_integral_v<T> || std::is_same_v<T, float> ||
          (std::is_class_v<T> && std::is_trivially_copyable_v<T> &&
           !detail::has_size_type_v<T>)>>
  inline void setArg(unsigned idx, const T val) {
    setArg(idx, &val, sizeof(T));
  }

  template <
      typename Container,
      typename = std::enable_if_t<detail::has_size_type_v<Container>>>
  inline void setArg(unsigned idx, const Container& values) {
    setArg(
        idx,
        values.data(),
        values.size() * sizeof(typename Container::value_type));
  }
  void dispatch(
      uint64_t length,
      std::optional<uint64_t> groupSize = std::nullopt);
  void dispatch(
      c10::ArrayRef<uint64_t> length,
      c10::OptionalArrayRef<uint64_t> groupSize = std::nullopt);

 private:
  MTLComputePipelineState_t cps;
  MTLFunction_t func;
  MTLComputeCommandEncoder_t encoder = nullptr;
};

class MetalShaderLibrary {
 public:
  MetalShaderLibrary(std::string src)
      : shaderSource(std::move(src)), nparams(0), compile_options(nullptr) {}
  MetalShaderLibrary(std::string src, unsigned nparams_)
      : shaderSource(std::move(src)),
        nparams(nparams_),
        compile_options(nullptr) {}
  MetalShaderLibrary(
      std::string src,
      unsigned nparams_,
      MTLCompileOptions* compile_options_)
      : shaderSource(std::move(src)),
        nparams(nparams_),
        compile_options(compile_options_) {}
  MetalShaderLibrary(const MetalShaderLibrary&) = delete;
  virtual ~MetalShaderLibrary();
  std::vector<std::string> getFunctionNames();
  std::shared_ptr<MetalKernelFunction> getKernelFunction(
      const std::string& name);
  // Returns a raw pointer to the kernel function for use in C APIs
  MetalKernelFunction* getCachedKernelFunctionPtr(const std::string& name);
  inline MTLComputePipelineState_t getPipelineStateForFunc(
      const std::string& fname) {
    return getLibraryPipelineState(getLibrary(), fname).first;
  }
  MTLComputePipelineState_t getPipelineStateForFunc(
      const std::string& fname,
      const std::initializer_list<std::string>& params) {
    return getLibraryPipelineState(getLibrary(params), fname).first;
  }
  inline MTLFunction_t getMTLFunction(const std::string& fname) {
    return getLibraryPipelineState(getLibrary(), fname).second;
  }
  MTLFunction_t getMTLFunction(
      const std::string& fname,
      const std::initializer_list<std::string>& params) {
    return getLibraryPipelineState(getLibrary(params), fname).second;
  }
  static MetalShaderLibrary& getBundledLibrary();
  void exec_unary_kernel(
      TensorIteratorBase& iter,
      const std::string& name,
      const std::optional<c10::Scalar> alpha = std::nullopt,
      const std::optional<c10::ScalarType> scalar_arg_type = std::nullopt);
  void exec_binary_kernel(
      TensorIteratorBase& iter,
      const std::string& name,
      const std::optional<c10::Scalar> alpha = std::nullopt,
      const std::optional<c10::ScalarType> scalar_arg_type = std::nullopt);

 protected:
  virtual MTLLibrary_t getLibrary();
  virtual MTLLibrary_t getLibrary(
      const std::initializer_list<std::string>& params);
  MTLLibrary_t library = nullptr;

 private:
  std::pair<MTLComputePipelineState_t, MTLFunction_t> getLibraryPipelineState(
      MTLLibrary_t lib,
      const std::string& fname);
  MTLLibrary_t compileLibrary(const std::string& src);
  std::string shaderSource;
  unsigned nparams;
  MTLCompileOptions* compile_options;
  std::unordered_map<std::string, MTLLibrary_t> libMap;
  std::unordered_map<
      std::string,
      std::pair<MTLComputePipelineState_t, MTLFunction_t>>
      cplMap;
  // Cache for kernel functions returned by getCachedKernelFunctionPtr
  std::unordered_map<std::string, std::unique_ptr<MetalKernelFunction>>
      kernelCache;
};

class DynamicMetalShaderLibrary : public MetalShaderLibrary {
 public:
  DynamicMetalShaderLibrary(const std::string& src) : MetalShaderLibrary(src) {
    // Compile right away
    getLibrary();
  }
  ~DynamicMetalShaderLibrary() override;
};

} // namespace at::native::mps

```



## High-Level Overview


This C++ file contains approximately 5 class(es)/struct(s) and 27 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `detail`, `at`

**Classes/Structs**: `TensorBase`, `TensorIteratorBase`, `has_size_type`, `MetalKernelFunction`, `MetalShaderLibrary`, `DynamicMetalShaderLibrary`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `aten/src/ATen/native/mps`, which is part of **ATen** (A Tensor Library), PyTorch's C++ tensor library.



## Dependencies

### Import Dependencies

This file includes:

- `Metal/Metal.h`
- `c10/core/Scalar.h`
- `c10/util/OptionalArrayRef.h`
- `functional`
- `optional`
- `type_traits`
- `unordered_map`
- `utility`
- `vector`


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
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

Files in the same folder (`aten/src/ATen/native/mps`):

- [`OperationUtils.h_docs.md`](./OperationUtils.h_docs.md)
- [`TensorFactory.h_docs.md`](./TensorFactory.h_docs.md)
- [`Copy.h_docs.md`](./Copy.h_docs.md)
- [`MPSGraphSequoiaOps.h_docs.md`](./MPSGraphSequoiaOps.h_docs.md)
- [`TensorFactory.cpp_docs.md`](./TensorFactory.cpp_docs.md)


## Cross-References

- **File Documentation**: `MetalShaderLibrary.h_docs.md`
- **Keyword Index**: `MetalShaderLibrary.h_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
