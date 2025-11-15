# Documentation: `docs/torch/csrc/jit/runtime/register_cuda_ops.cpp_docs.md`

## File Metadata

- **Path**: `docs/torch/csrc/jit/runtime/register_cuda_ops.cpp_docs.md`
- **Size**: 10,265 bytes (10.02 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/csrc/jit/runtime/register_cuda_ops.cpp`

## File Metadata

- **Path**: `torch/csrc/jit/runtime/register_cuda_ops.cpp`
- **Size**: 7,476 bytes (7.30 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
// This file registers special JIT operators used to implement the PyTorch CUDA
// API in TorchScript.
#include <torch/csrc/api/include/torch/utils.h>
#include <torch/csrc/jit/cuda/cuda.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/runtime/custom_operator.h>
#include <torch/csrc/jit/runtime/operator.h>

namespace torch::jit {

namespace {

c10::AliasAnalysisKind aliasAnalysisFromSchema() {
  return c10::AliasAnalysisKind::FROM_SCHEMA;
}

void _device_synchronize(int64_t device_index) {
  // This is a helper API which synchronizes the device identified
  // by the device index. The device index of the device is passed as an
  // argument to this API.
  auto current_device_index = c10::cuda::current_device();
  // If the current_device and the device to synchronize are not
  // the same, set the device to the device_index of the device
  // to synchronize.
  if (current_device_index != device_index) {
    c10::cuda::set_device(device_index);
  }
  c10::cuda::device_synchronize();

  // Reset the device to current_device before synchronizing.
  if (current_device_index != device_index) {
    c10::cuda::set_device(current_device_index);
  }
}

RegisterOperators const reg({
    Operator(
        "cuda::current_stream.device(Device? device) -> __torch__.torch.classes.cuda.Stream",
        [](Stack& stack) {
          auto device = pop(stack).toOptional<c10::Device>();
          c10::DeviceIndex device_index = device.has_value()
              ? device->index()
              : c10::cuda::current_device();
          auto s = c10::cuda::getCurrentCUDAStream(device_index);
          auto st = make_custom_class<torch::jit::CUDAStream>(s);
          push(stack, IValue(st));
        },
        aliasAnalysisFromSchema()),
    Operator(
        "cuda::current_stream.int(int? val) -> __torch__.torch.classes.cuda.Stream",
        [](Stack& stack) {
          auto idx = pop(stack).toOptional<c10::DeviceIndex>();
          c10::DeviceIndex device_index =
              idx.has_value() ? idx.value() : c10::cuda::current_device();
          auto s = c10::cuda::getCurrentCUDAStream(device_index);
          auto st = make_custom_class<torch::jit::CUDAStream>(s);
          push(stack, IValue(st));
        },
        aliasAnalysisFromSchema()),
    Operator(
        "cuda::default_stream.device(Device? device) -> __torch__.torch.classes.cuda.Stream",
        [](Stack& stack) {
          auto device = pop(stack).toOptional<c10::Device>();
          c10::DeviceIndex device_index = device.has_value()
              ? device->index()
              : c10::cuda::current_device();
          auto s = c10::cuda::getDefaultCUDAStream(device_index);
          auto st = make_custom_class<torch::jit::CUDAStream>(s);
          push(stack, IValue(st));
        },
        aliasAnalysisFromSchema()),
    Operator(
        "cuda::default_stream.int(int? val) -> __torch__.torch.classes.cuda.Stream",
        [](Stack& stack) {
          auto idx = pop(stack).toOptional<c10::DeviceIndex>();
          c10::DeviceIndex device_index =
              idx.has_value() ? idx.value() : c10::cuda::current_device();
          auto s = c10::cuda::getDefaultCUDAStream(device_index);
          auto st = make_custom_class<torch::jit::CUDAStream>(s);
          push(stack, IValue(st));
        },
        aliasAnalysisFromSchema()),
    Operator(
        "cuda::_current_device() -> int",
        [](Stack& stack) {
          auto v = c10::cuda::current_device();
          push(stack, static_cast<int>(v));
        },
        aliasAnalysisFromSchema()),
    Operator(
        "cuda::_exchange_device(int64_t index) -> int",
        [](Stack& stack) {
          int64_t idx = -1;
          pop(stack, idx);
          if (idx < 0) {
            push(stack, -1);
            return;
          }
          auto prev_idx = c10::cuda::current_device();
          c10::cuda::set_device(static_cast<c10::DeviceIndex>(idx));
          push(stack, static_cast<int>(prev_idx));
        },
        // cuda::set_device has side effects.
        c10::AliasAnalysisKind::CONSERVATIVE),
    Operator(
        "cuda::_maybe_exchange_device(int64_t index) -> int",
        [](Stack& stack) {
          int64_t idx = -1;
          pop(stack, idx);
          if (idx < 0) {
            push(stack, -1);
            return;
          }
          int prev_idx = c10::cuda::MaybeExchangeDevice(static_cast<int>(idx));
          push(stack, prev_idx);
        },
        c10::AliasAnalysisKind::CONSERVATIVE),
    Operator(
        "cuda::_set_device(int64_t val) -> ()",
        [](Stack& stack) {
          int64_t idx = -1;
          pop(stack, idx);
          c10::cuda::set_device(static_cast<c10::DeviceIndex>(idx));
        },
        aliasAnalysisFromSchema()),
    Operator(
        "cuda::device_index(Device device) -> int",
        [](Stack& stack) {
          auto device = pop(stack);
          auto idx = device.toDevice().index();
          push(stack, idx);
        },
        aliasAnalysisFromSchema()),
    Operator(
        "cuda::device_count() -> int",
        [](Stack& stack) { push(stack, at::cuda::device_count()); },
        aliasAnalysisFromSchema()),
    Operator(
        "cuda::set_stream(__torch__.torch.classes.cuda.Stream stream) -> ()",
        [](Stack& stack) {
          auto v = pop(stack);
          auto s = v.toCustomClass<torch::jit::CUDAStream>();
          auto stream_device_idx = s->device_index();
          auto cur_device_idx = c10::cuda::current_device();
          // If the stream is not on the current device, change the
          // device to the device of the stream.
          if (cur_device_idx != stream_device_idx) {
            c10::cuda::set_device(stream_device_idx);
          }
          // To set the current CUDA stream using
          // c10::cuda::setCurrentCUDAStream, the jit::CUDAStream object needs
          // to be converted to c10::cuda::CUDAStream. Since the latter cannot
          // be returned from a class registered via TorchBind, this can only be
          // achieved by packing the c10::cuda::CUDAStream instance contained
          // inside the jit::CUDAStream object to a struct representation, and
          // unpacking it inside this operator. The unpacked stream is then used
          // to set the current CUDA stream.
          auto unpacked = c10::cuda::CUDAStream::unpack3(
              s->id(), stream_device_idx, c10::DeviceType::CUDA);
          c10::cuda::setCurrentCUDAStream(unpacked);
        },
        aliasAnalysisFromSchema()),
    Operator(
        "cuda::synchronize() -> ()",
        [](Stack& stack) { c10::cuda::device_synchronize(); },
        aliasAnalysisFromSchema()),
    Operator(
        "cuda::synchronize.device(Device? device) -> ()",
        [](Stack& stack) {
          auto device = pop(stack).toOptional<c10::Device>();
          c10::DeviceIndex device_index = device.has_value()
              ? device->index()
              : c10::cuda::current_device();
          _device_synchronize(device_index);
        },
        aliasAnalysisFromSchema()),
    Operator(
        "cuda::synchronize.int(int? val) -> ()",
        [](Stack& stack) {
          auto idx = pop(stack).toOptional<c10::DeviceIndex>();
          c10::DeviceIndex device_index =
              idx.has_value() ? idx.value() : c10::cuda::current_device();
          _device_synchronize(device_index);
        },
        aliasAnalysisFromSchema()),
});
} // namespace
} // namespace torch::jit

```



## High-Level Overview


This C++ file contains approximately 1 class(es)/struct(s) and 3 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `torch`

**Classes/Structs**: `registered`, `representation`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/csrc/jit/runtime`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `torch/csrc/api/include/torch/utils.h`
- `torch/csrc/jit/cuda/cuda.h`
- `torch/csrc/jit/ir/ir.h`
- `torch/csrc/jit/runtime/custom_operator.h`
- `torch/csrc/jit/runtime/operator.h`


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

- This file appears to involve **GPU/parallel computing** capabilities.
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

Files in the same folder (`torch/csrc/jit/runtime`):

- [`decomposition_registry.h_docs.md`](./decomposition_registry.h_docs.md)
- [`register_distributed_ops.cpp_docs.md`](./register_distributed_ops.cpp_docs.md)
- [`instruction.h_docs.md`](./instruction.h_docs.md)
- [`argument_spec.cpp_docs.md`](./argument_spec.cpp_docs.md)
- [`instruction.cpp_docs.md`](./instruction.cpp_docs.md)
- [`symbolic_script.h_docs.md`](./symbolic_script.h_docs.md)
- [`register_prim_ops_fulljit.cpp_docs.md`](./register_prim_ops_fulljit.cpp_docs.md)
- [`symbolic_shape_registry_util.cpp_docs.md`](./symbolic_shape_registry_util.cpp_docs.md)
- [`interpreter.h_docs.md`](./interpreter.h_docs.md)
- [`logging.h_docs.md`](./logging.h_docs.md)


## Cross-References

- **File Documentation**: `register_cuda_ops.cpp_docs.md`
- **Keyword Index**: `register_cuda_ops.cpp_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/csrc/jit/runtime`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/csrc/jit/runtime`, which is part of the **core PyTorch library**.



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

Files in the same folder (`docs/torch/csrc/jit/runtime`):

- [`register_ops_utils.h_docs.md_docs.md`](./register_ops_utils.h_docs.md_docs.md)
- [`register_c10_ops.cpp_docs.md_docs.md`](./register_c10_ops.cpp_docs.md_docs.md)
- [`exception_message.h_kw.md_docs.md`](./exception_message.h_kw.md_docs.md)
- [`register_prim_ops.cpp_kw.md_docs.md`](./register_prim_ops.cpp_kw.md_docs.md)
- [`autodiff.cpp_kw.md_docs.md`](./autodiff.cpp_kw.md_docs.md)
- [`decomposition_registry_util.h_docs.md_docs.md`](./decomposition_registry_util.h_docs.md_docs.md)
- [`slice_indices_adjust.cpp_docs.md_docs.md`](./slice_indices_adjust.cpp_docs.md_docs.md)
- [`graph_iterator.h_kw.md_docs.md`](./graph_iterator.h_kw.md_docs.md)
- [`shape_function_registry.h_docs.md_docs.md`](./shape_function_registry.h_docs.md_docs.md)
- [`symbolic_script.cpp_docs.md_docs.md`](./symbolic_script.cpp_docs.md_docs.md)


## Cross-References

- **File Documentation**: `register_cuda_ops.cpp_docs.md_docs.md`
- **Keyword Index**: `register_cuda_ops.cpp_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
