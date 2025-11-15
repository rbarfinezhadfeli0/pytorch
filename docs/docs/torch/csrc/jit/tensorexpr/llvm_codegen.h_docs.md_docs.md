# Documentation: `docs/torch/csrc/jit/tensorexpr/llvm_codegen.h_docs.md`

## File Metadata

- **Path**: `docs/torch/csrc/jit/tensorexpr/llvm_codegen.h_docs.md`
- **Size**: 6,478 bytes (6.33 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/csrc/jit/tensorexpr/llvm_codegen.h`

## File Metadata

- **Path**: `torch/csrc/jit/tensorexpr/llvm_codegen.h`
- **Size**: 3,839 bytes (3.75 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
#pragma once

#ifdef TORCH_ENABLE_LLVM
#include <torch/csrc/Export.h>

#include <torch/csrc/jit/tensorexpr/codegen.h>
#include <torch/csrc/jit/tensorexpr/ir.h>
#include <torch/csrc/jit/tensorexpr/ir_visitor.h>

#include <optional>

#include <unordered_map>
#include <vector>

namespace torch {
namespace jit {
namespace tensorexpr {

class LLVMCodeGenImpl;
class LLVMCodeGenCallee;

class TORCH_API LLVMCodeGen : public CodeGen {
 public:
  explicit LLVMCodeGen(
      StmtPtr stmt,
      const std::vector<BufferArg>& args,
      at::Device device = at::kCPU,
      const std::string& kernel_func_name = "func",
      Dtype dtype = kInt,
      std::optional<std::string> triple = std::nullopt,
      std::optional<std::string> cpu = std::nullopt,
      std::optional<std::string> attrs = std::nullopt);
  explicit LLVMCodeGen(StmtPtr stmt);

  LLVMCodeGen() = delete;
  ~LLVMCodeGen() override;

  // Cleans up all the memory used during LLVM code generation pass except
  // the generated kernel. After calling this method, users should not call
  // methods like `getCodeText` that require the LLVMCodeGenImpl data. However,
  // users can continue to call this kernel using `call` and `call_raw`.
  void cleanup_memory();

  TORCH_API void call(const std::vector<CallArg>& args) override;
  TORCH_API void call_raw(const std::vector<void*>& args) override;
  TORCH_API void call_with_numel(void** args, int64_t numel) override;

  at::Tensor empty_strided(
      c10::IntArrayRef size,
      c10::IntArrayRef stride,
      std::optional<c10::ScalarType> dtype_opt,
      std::optional<c10::Layout> layout_opt,
      std::optional<c10::Device> device_opt,
      std::optional<bool> pin_memory_opt) override;

  template <typename T>
  T value() {
    return value<T>(nullptr);
  }

  template <typename T>
  T value(std::vector<void*>& args) {
    return value<T>(args.data());
  }

  template <typename T>
  T value(void** args) {
    T (*fp)(void**) = (T(*)(void**))getKernelAddress(callee_.get());
    T rv = fp(args);
    return rv;
  }

  std::string getCodeText(const std::string& attr = "") override;

 private:
  void* getKernelAddress(LLVMCodeGenCallee* callee);

  std::unique_ptr<LLVMCodeGenCallee> callee_;
  std::unique_ptr<LLVMCodeGenImpl> impl_;
};

struct TORCH_API LLVMCodeGenBuilder {
  using BufferArg = CodeGen::BufferArg;

  LLVMCodeGenBuilder(StmtPtr stmt, std::vector<BufferArg> args)
      : stmt_(stmt), args_(std::move(args)) {}

  LLVMCodeGenBuilder& device(at::Device device) {
    device_ = device;
    return *this;
  }

  LLVMCodeGenBuilder& kernelFuncName(std::string name) {
    kernelFuncName_ = std::move(name);
    return *this;
  }

  LLVMCodeGenBuilder& dtype(Dtype d) {
    dtype_ = d;
    return *this;
  }

  LLVMCodeGenBuilder& triple(std::string triple) {
    triple_ = std::move(triple);
    return *this;
  }

  LLVMCodeGenBuilder& cpu(std::string cpu) {
    cpu_ = std::move(cpu);
    return *this;
  }

  LLVMCodeGenBuilder& attrs(std::string attrs) {
    attrs_ = std::move(attrs);
    return *this;
  }

  std::unique_ptr<LLVMCodeGen> build() {
    return std::make_unique<LLVMCodeGen>(
        stmt_, args_, device_, kernelFuncName_, dtype_, triple_, cpu_, attrs_);
  }

 private:
  StmtPtr stmt_;
  std::vector<BufferArg> args_;
  at::Device device_ = at::kCPU;
  std::string kernelFuncName_ = "func";
  Dtype dtype_ = kInt;
  std::optional<std::string> triple_ = std::nullopt;
  std::optional<std::string> cpu_ = std::nullopt;
  std::optional<std::string> attrs_ = std::nullopt;
};

TORCH_API std::optional<std::string>& LLVMTargetTriple();
TORCH_API std::optional<std::string>& LLVMTargetCPU();
TORCH_API std::optional<std::string>& LLVMTargetAttrs();
TORCH_API bool& LLVMAOTWorkflow();

} // namespace tensorexpr
} // namespace jit
} // namespace torch

#endif // TORCH_ENABLE_LLVM

```



## High-Level Overview


This C++ file contains approximately 3 class(es)/struct(s) and 11 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `tensorexpr`, `jit`, `torch`

**Classes/Structs**: `LLVMCodeGenImpl`, `LLVMCodeGenCallee`, `TORCH_API`, `TORCH_API`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/csrc/jit/tensorexpr`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `torch/csrc/Export.h`
- `torch/csrc/jit/tensorexpr/codegen.h`
- `torch/csrc/jit/tensorexpr/ir.h`
- `torch/csrc/jit/tensorexpr/ir_visitor.h`
- `optional`
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

Files in the same folder (`torch/csrc/jit/tensorexpr`):

- [`codegen.cpp_docs.md`](./codegen.cpp_docs.md)
- [`bounds_overlap.h_docs.md`](./bounds_overlap.h_docs.md)
- [`eval.h_docs.md`](./eval.h_docs.md)
- [`cuda_codegen.cpp_docs.md`](./cuda_codegen.cpp_docs.md)
- [`ir_simplifier.h_docs.md`](./ir_simplifier.h_docs.md)
- [`reduction.h_docs.md`](./reduction.h_docs.md)
- [`kernel.cpp_docs.md`](./kernel.cpp_docs.md)
- [`cuda_codegen.h_docs.md`](./cuda_codegen.h_docs.md)
- [`external_functions_core.cpp_docs.md`](./external_functions_core.cpp_docs.md)
- [`graph_opt.h_docs.md`](./graph_opt.h_docs.md)


## Cross-References

- **File Documentation**: `llvm_codegen.h_docs.md`
- **Keyword Index**: `llvm_codegen.h_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/csrc/jit/tensorexpr`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/csrc/jit/tensorexpr`, which is part of the **core PyTorch library**.



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

Files in the same folder (`docs/torch/csrc/jit/tensorexpr`):

- [`loopnest.h_kw.md_docs.md`](./loopnest.h_kw.md_docs.md)
- [`expr.h_docs.md_docs.md`](./expr.h_docs.md_docs.md)
- [`block_codegen.h_kw.md_docs.md`](./block_codegen.h_kw.md_docs.md)
- [`ir_cloner.cpp_kw.md_docs.md`](./ir_cloner.cpp_kw.md_docs.md)
- [`types.cpp_docs.md_docs.md`](./types.cpp_docs.md_docs.md)
- [`tensorexpr_init.h_docs.md_docs.md`](./tensorexpr_init.h_docs.md_docs.md)
- [`lowerings.cpp_kw.md_docs.md`](./lowerings.cpp_kw.md_docs.md)
- [`graph_opt.h_kw.md_docs.md`](./graph_opt.h_kw.md_docs.md)
- [`eval.h_kw.md_docs.md`](./eval.h_kw.md_docs.md)
- [`kernel.cpp_docs.md_docs.md`](./kernel.cpp_docs.md_docs.md)


## Cross-References

- **File Documentation**: `llvm_codegen.h_docs.md_docs.md`
- **Keyword Index**: `llvm_codegen.h_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
