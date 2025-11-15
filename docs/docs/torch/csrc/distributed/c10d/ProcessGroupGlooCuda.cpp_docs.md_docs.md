# Documentation: `docs/torch/csrc/distributed/c10d/ProcessGroupGlooCuda.cpp_docs.md`

## File Metadata

- **Path**: `docs/torch/csrc/distributed/c10d/ProcessGroupGlooCuda.cpp_docs.md`
- **Size**: 9,154 bytes (8.94 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/csrc/distributed/c10d/ProcessGroupGlooCuda.cpp`

## File Metadata

- **Path**: `torch/csrc/distributed/c10d/ProcessGroupGlooCuda.cpp`
- **Size**: 6,509 bytes (6.36 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#ifdef USE_C10D_GLOO
#include <torch/csrc/distributed/c10d/ProcessGroupGloo.hpp>
#include <torch/csrc/distributed/c10d/ProcessGroupGlooDetail.hpp>
#include <utility>

#include <gloo/cuda_allreduce_ring_chunked.h>

namespace c10d {

class AsyncAllreduceCUDADeviceWork : public ProcessGroupGloo::AsyncWork {
 public:
  AsyncAllreduceCUDADeviceWork(
      std::shared_ptr<gloo::Context> context,
      std::vector<at::Tensor>& inputs,
      ReduceOp reduceOp,
      uint32_t tag,
      uint64_t seq,
      std::chrono::milliseconds timeout)
      : ProcessGroupGloo::AsyncWork(
            std::move(context),
            {inputs},
            OpType::ALLREDUCE,
            seq,
            timeout,
            "gloo:all_reduce",
            inputs),
        inputs_(inputs),
        reduceOp_(std::move(reduceOp)) {}

  template <typename T>
  void createAlgorithm(std::unique_ptr<gloo::Algorithm>& algo) {
    auto count = inputs_.at(0).numel();
    std::vector<T*> ptrs;
    for (const auto& tensor : inputs_) {
      TORCH_CHECK_EQ(tensor.numel(), count);
      ptrs.push_back(static_cast<T*>(tensor.data_ptr()));
    }
    algo = std::make_unique<
        gloo::CudaAllreduceRingChunked<T, gloo::CudaDeviceWorkspace<T>>>(
        context_, ptrs, count);
  }

  void run() override {
    const auto& scalarType = inputs_.at(0).scalar_type();

    std::unique_ptr<gloo::Algorithm> algo;
    GENERATE_ALL_TYPES(scalarType, createAlgorithm, algo);
    algo->run();

    // Gloo doesn't support AVG so we use SUM + division.
    if (reduceOp_ == ReduceOp::AVG) {
      inputs_[0] /= context_->size;
    } else {
      TORCH_CHECK_EQ(reduceOp_, ReduceOp::SUM);
    }
  }

  const std::vector<at::Tensor> getInputTensors() override {
    return inputs_;
  }

  const std::vector<at::Tensor> getOutputTensors() override {
    return inputs_;
  }

  void synchronize() override {
    // TODO: is synchronization needed?
  }

 private:
  std::vector<at::Tensor> inputs_;
  const ReduceOp reduceOp_;
};

class AsyncAllreduceCUDAHostWork : public AsyncAllreduceWork {
 public:
  AsyncAllreduceCUDAHostWork(
      const std::shared_ptr<gloo::Context>& context,
      std::vector<at::Tensor>& inputs,
      ReduceOp reduceOp,
      uint32_t tag,
      uint64_t seq,
      std::chrono::milliseconds timeout)
      : AsyncAllreduceWork(
            context,
            inputs,
            std::move(reduceOp),
            tag,
            seq,
            timeout) {
    initializeStreamsEvents(inputs, streams, events);

    // Kick off copy from CUDA tensors to pinned CPU tensors.
    tmp.reserve(inputs.size());
    c10::OptionalStreamGuard guard;
    for (const auto i : c10::irange(inputs.size())) {
      guard.reset_stream(streams[i]);
      tmp.push_back(pinnedLike(inputs[i]).copy_(inputs[i], true));
    }
  }

  void run() override {
    // Synchronize with copy operations.
    for (const auto i : c10::irange(inputs.size())) {
      streams[i].synchronize();
    }

    // Run allreduce on host side tensors.
    allreduce(tmp);

    c10::OptionalStreamGuard guard;
    for (const auto i : c10::irange(inputs.size())) {
      guard.reset_stream(streams[i]);
      inputs[i].copy_(tmp[i], /* non_blocking */ true);
      events[i].record(streams[i]);
    }
  }

  void synchronize() override {
    // Synchronize with the copy back to CUDA tensors.
    for (const auto i : c10::irange(inputs.size())) {
      c10::Device device = inputs[i].device();
      events[i].block(
          c10::impl::VirtualGuardImpl(device.type()).getStream(device));
    }
  }

  std::vector<at::Tensor> tmp;
  std::vector<c10::Stream> streams;
  std::vector<c10::Event> events;
};

class AsyncSparseAllreduceCUDAWork : public AsyncSparseAllreduceWork {
 public:
  AsyncSparseAllreduceCUDAWork(
      const std::shared_ptr<gloo::Context>& context,
      std::vector<at::Tensor>& inputs,
      uint32_t tag,
      uint64_t seq,
      std::chrono::milliseconds timeout)
      : AsyncSparseAllreduceWork(context, inputs, tag, seq, timeout) {
    initializeStreamsEvents(inputs, streams, events);

    // Kick off copy from CUDA tensors to CPU tensors.
    // Note that both coalescing the sparse tensor and copying it to CPU
    // memory must be performed asynchronously, or we block the caller.
    tmp.reserve(inputs.size());
    c10::OptionalStreamGuard guard;
    for (const auto i : c10::irange(inputs.size())) {
      guard.reset_stream(streams[i]);
      tmp.push_back(
          inputs[i].coalesce().to(at::DeviceType::CPU, /*non_blocking=*/true));
    }
  }

  void run() override {
    // Synchronize with copy operations.
    for (const auto i : c10::irange(inputs.size())) {
      streams[i].synchronize();
    }

    // Run allreduce on host side tensors.
    auto output = allreduce(tmp);

    // Kick off copy back to the CUDA tensors.
    c10::OptionalStreamGuard guard;
    for (const auto i : c10::irange(inputs.size())) {
      guard.reset_stream(streams[i]);
      inputs[i].copy_(output, /*non_blocking=*/true);
      events[i].record(streams[i]);
    }
  }

  void synchronize() override {
    // Synchronize with the copy back to CUDA tensors.
    for (const auto i : c10::irange(inputs.size())) {
      c10::Device device = inputs[i].device();
      events[i].block(
          c10::impl::VirtualGuardImpl(device.type()).getStream(device));
    }
  }

  std::vector<at::Tensor> tmp;
  std::vector<c10::Stream> streams;
  std::vector<c10::Event> events;
};

static c10::intrusive_ptr<ProcessGroupGloo::AsyncWork> makeAllreduceCUDAWork(
    std::shared_ptr<gloo::Context> context,
    std::vector<at::Tensor>& inputs,
    ReduceOp reduceOp,
    uint32_t tag,
    uint64_t seq,
    std::chrono::milliseconds timeout) {
  auto layout = inputs[0].layout();

  if (layout == c10::kStrided) {
    if (context->getDevice()->hasGPUDirect()) {
      return c10::make_intrusive<AsyncAllreduceCUDADeviceWork>(
          std::move(context), inputs, reduceOp, tag, seq, timeout);
    } else {
      return c10::make_intrusive<AsyncAllreduceCUDAHostWork>(
          std::move(context), inputs, reduceOp, tag, seq, timeout);
    }
  } else if (layout == c10::kSparse) {
    return c10::make_intrusive<AsyncSparseAllreduceCUDAWork>(
        std::move(context), inputs, tag, seq, timeout);
  } else {
    TORCH_CHECK(false, "ProcessGroupGloo::allreduce: unsupported layout");
  }
}

C10_REGISTER_TYPED_CREATOR(
    GlooAllreduceRegistry,
    at::kCUDA,
    makeAllreduceCUDAWork)
} // namespace c10d

#endif // USE_C10D_GLOO

```



## High-Level Overview


This C++ file contains approximately 3 class(es)/struct(s) and 8 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `c10d`

**Classes/Structs**: `AsyncAllreduceCUDADeviceWork`, `AsyncAllreduceCUDAHostWork`, `AsyncSparseAllreduceCUDAWork`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/csrc/distributed/c10d`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `torch/csrc/distributed/c10d/ProcessGroupGloo.hpp`
- `torch/csrc/distributed/c10d/ProcessGroupGlooDetail.hpp`
- `utility`
- `gloo/cuda_allreduce_ring_chunked.h`


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

Files in the same folder (`torch/csrc/distributed/c10d`):

- [`Utils.hpp_docs.md`](./Utils.hpp_docs.md)
- [`Ops.cpp_docs.md`](./Ops.cpp_docs.md)
- [`Store.hpp_docs.md`](./Store.hpp_docs.md)
- [`WinSockUtils.hpp_docs.md`](./WinSockUtils.hpp_docs.md)
- [`FakeProcessGroup.hpp_docs.md`](./FakeProcessGroup.hpp_docs.md)
- [`Work.cpp_docs.md`](./Work.cpp_docs.md)
- [`PrefixStore.hpp_docs.md`](./PrefixStore.hpp_docs.md)
- [`PyProcessGroup.hpp_docs.md`](./PyProcessGroup.hpp_docs.md)
- [`debug.h_docs.md`](./debug.h_docs.md)
- [`exception.h_docs.md`](./exception.h_docs.md)


## Cross-References

- **File Documentation**: `ProcessGroupGlooCuda.cpp_docs.md`
- **Keyword Index**: `ProcessGroupGlooCuda.cpp_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/csrc/distributed/c10d`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/csrc/distributed/c10d`, which is part of the **core PyTorch library**.



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

Files in the same folder (`docs/torch/csrc/distributed/c10d`):

- [`ProcessGroupWrapper.cpp_docs.md_docs.md`](./ProcessGroupWrapper.cpp_docs.md_docs.md)
- [`c10d.h_kw.md_docs.md`](./c10d.h_kw.md_docs.md)
- [`TCPStoreLibUvBackend.cpp_kw.md_docs.md`](./TCPStoreLibUvBackend.cpp_kw.md_docs.md)
- [`NanCheck.cu_docs.md_docs.md`](./NanCheck.cu_docs.md_docs.md)
- [`python_callback_work.hpp_kw.md_docs.md`](./python_callback_work.hpp_kw.md_docs.md)
- [`sequence_num.hpp_kw.md_docs.md`](./sequence_num.hpp_kw.md_docs.md)
- [`Functional.hpp_kw.md_docs.md`](./Functional.hpp_kw.md_docs.md)
- [`TCPStoreBackend.cpp_kw.md_docs.md`](./TCPStoreBackend.cpp_kw.md_docs.md)
- [`ProcessGroupUCC.cpp_kw.md_docs.md`](./ProcessGroupUCC.cpp_kw.md_docs.md)


## Cross-References

- **File Documentation**: `ProcessGroupGlooCuda.cpp_docs.md_docs.md`
- **Keyword Index**: `ProcessGroupGlooCuda.cpp_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
