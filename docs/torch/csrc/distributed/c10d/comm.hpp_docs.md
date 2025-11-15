# Documentation: `torch/csrc/distributed/c10d/comm.hpp`

## File Metadata

- **Path**: `torch/csrc/distributed/c10d/comm.hpp`
- **Size**: 4,423 bytes (4.32 KB)
- **Type**: C++ Header File
- **Extension**: `.hpp`

## File Purpose

This is a c++ header file that is part of the PyTorch project.

## Original Source

```cpp
#pragma once

#include <ATen/ATen.h>
#include <ATen/core/ivalue.h>
#include <torch/csrc/Export.h>
#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>
#include <utility>

namespace c10d {

// Broadcast many tensors to all processes in the process group.
TORCH_API void broadcast_coalesced(
    const c10::intrusive_ptr<c10d::ProcessGroup>& process_group,
    at::TensorList tensors,
    size_t buffer_size,
    int rank = 0);

// This class passes bucket contents tensor to DDP communication hook.
class TORCH_API GradBucket {
 public:
  explicit GradBucket(
      size_t index,
      size_t bucket_count,
      at::Tensor tensor,
      std::vector<size_t> offsets,
      std::vector<size_t> lengths,
      std::vector<c10::IntArrayRef> sizes_vec,
      std::vector<at::Tensor> parameters,
      std::optional<at::Tensor> sparse_grad_indices)
      : index_(index),
        bucket_count_(bucket_count),
        buffer_(std::move(tensor)),
        offsets_(std::move(offsets)),
        lengths_(std::move(lengths)),
        sizes_vec_(std::move(sizes_vec)),
        parameters_(std::move(parameters)),
        sparse_grad_indices_(std::move(sparse_grad_indices)) {}

  // Returns the index of the bucket, which is unique across all the buckets.
  size_t getIndex() const {
    return index_;
  }

  const at::Tensor& getBuffer() const {
    return buffer_;
  }

  // Returns a mutable buffer compared with the above method.
  at::Tensor& getBufferRef() {
    return buffer_;
  }

  // Overwrites the buffer at a specific index.
  void setBuffer(at::Tensor& buffer) {
    buffer_ = buffer;
  }

  // Each tensor in the list that getGradients corresponds to a
  // parameter.
  std::vector<at::Tensor> getGradients() const;

  // Returns model parameters belonging to this bucket. They are returned in the
  // same order as gradient tensors via getGradients(). For example,
  // getParameters[i] will have its gradient stored in
  // getGradients[i]
  const std::vector<at::Tensor> getParameters() const {
    return parameters_;
  }

  // Returns whether this bucket is the last bucket to allreduce in an
  // iteration.
  bool isLast() const {
    return index_ == bucket_count_ - 1;
  }

  std::optional<at::Tensor>& getSparseGradIndices() {
    return sparse_grad_indices_;
  }

 private:
  size_t index_;
  size_t bucket_count_;
  at::Tensor buffer_;

  // Per-variable info in buffer_.
  std::vector<size_t> offsets_;
  std::vector<size_t> lengths_;
  std::vector<c10::IntArrayRef> sizes_vec_;

  // Model parameters for this bucket.
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-const-or-ref-data-members)
  const std::vector<at::Tensor> parameters_;

  // Predefined sparse indices for this bucket (only used for sparse tensors).
  // The gradients will be updated to have indices with these tensor values
  std::optional<at::Tensor> sparse_grad_indices_;
};

// Base class of both `PythonCommHook` and `CppCommHook`.
// Requires implementing 1) `runHook` method that communicates gradients
// asynchronously, and 2) `parseHookResult` method that converts the hook
// result into a tensor.
class TORCH_API CommHookInterface {
 public:
  virtual ~CommHookInterface() = default;

  // Passes the input grad bucket to the registered communication hook.
  // Once the tensor in the bucket are ready, kicks off the hook asynchronously
  // and returns a future that holds the communication results.
  virtual c10::intrusive_ptr<c10::ivalue::Future> runHook(
      GradBucket& bucket) = 0;

  // Returns the resulting tensor once the communication hook result is
  // ready. The resulting tensor will then be copied to the grads of
  // individual parameters.
  virtual at::Tensor parseHookResult(const c10::IValue& result) = 0;
};

namespace detail {
// This helper function is called both by CppCommHookInterface below and inside
// reducer.
TORCH_API at::Tensor parseCppCommHookResult(const c10::IValue& result);
} // namespace detail

// This CppCommHook interface only requires implementing runHook method that
// potentially uses a state.
template <typename T>
class CppCommHookInterface : public CommHookInterface {
 public:
  explicit CppCommHookInterface(T state) : state_(std::move(state)) {}

  ~CppCommHookInterface() override = default;

  at::Tensor parseHookResult(const c10::IValue& result) override {
    return detail::parseCppCommHookResult(result);
  }

 protected:
  T state_;
};

} // namespace c10d

```



## High-Level Overview


This C++ file contains approximately 5 class(es)/struct(s) and 11 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `detail`, `c10d`

**Classes/Structs**: `passes`, `TORCH_API`, `of`, `TORCH_API`, `CppCommHookInterface`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/csrc/distributed/c10d`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `ATen/ATen.h`
- `ATen/core/ivalue.h`
- `torch/csrc/Export.h`
- `torch/csrc/distributed/c10d/ProcessGroup.hpp`
- `utility`


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes


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

- **File Documentation**: `comm.hpp_docs.md`
- **Keyword Index**: `comm.hpp_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
