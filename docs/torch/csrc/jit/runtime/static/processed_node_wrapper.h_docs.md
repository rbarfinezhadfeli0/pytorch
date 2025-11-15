# Documentation: `torch/csrc/jit/runtime/static/processed_node_wrapper.h`

## File Metadata

- **Path**: `torch/csrc/jit/runtime/static/processed_node_wrapper.h`
- **Size**: 6,595 bytes (6.44 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
#pragma once

#include <ATen/ATen.h>
#include <torch/csrc/jit/runtime/static/impl.h>

namespace torch::jit {

// The following class facilitates code reuse between ProcessedNodeInputWrapper
// and ProcessedNodeOutputWrapper via CRTP
template <typename DerivedWrapper>
class ProcessedNodeWrapperBase {
 public:
  class ProcessedNodeWrapperBaseIter {
   public:
    using iterator_category = std::forward_iterator_tag;
    using value_type = at::Tensor;
    using difference_type = size_t;
    using pointer = const at::Tensor*;
    using reference = const at::Tensor&;

    ProcessedNodeWrapperBaseIter() = default;

    ProcessedNodeWrapperBaseIter(
        const DerivedWrapper* container,
        size_t start_idx)
        : container_(container), idx_(start_idx) {}

    ProcessedNodeWrapperBaseIter& operator++() {
      TORCH_DCHECK_NE(idx_, container_->size());
      ++idx_;
      return *this;
    }

    ProcessedNodeWrapperBaseIter operator++(int) {
      ProcessedNodeWrapperBaseIter old = *this;
      ++(*this);
      return old;
    }

    reference operator*() const {
      TORCH_CHECK(container_ != nullptr);
      return (*container_)[idx_];
    }

    pointer operator->() const {
      TORCH_CHECK(container_ != nullptr);
      return &(*container_)[idx_];
    }

    friend bool operator==(
        ProcessedNodeWrapperBaseIter lhs,
        ProcessedNodeWrapperBaseIter rhs) {
      TORCH_DCHECK_EQ(lhs.container_, rhs.container_);
      return lhs.idx_ == rhs.idx_;
    }

    friend bool operator!=(
        ProcessedNodeWrapperBaseIter lhs,
        ProcessedNodeWrapperBaseIter rhs) {
      return !(lhs == rhs);
    }

   private:
    const DerivedWrapper* container_ = nullptr;
    size_t idx_ = 0;
  };

  // NB: to mimic the behavior of at::ArrayRef, both iterators are
  // the const version.
  using iterator = ProcessedNodeWrapperBaseIter;
  using const_iterator = ProcessedNodeWrapperBaseIter;
  using size_type = size_t;
  using value_type = at::Tensor;

  explicit ProcessedNodeWrapperBase(ProcessedNode& pnode) : pnode_(pnode) {}

  iterator begin() {
    return ProcessedNodeWrapperBaseIter(static_cast<DerivedWrapper*>(this), 0);
  }
  iterator end() {
    return ProcessedNodeWrapperBaseIter(
        static_cast<DerivedWrapper*>(this),
        static_cast<DerivedWrapper*>(this)->size());
  }

  const_iterator begin() const {
    return ProcessedNodeWrapperBaseIter(
        static_cast<const DerivedWrapper*>(this), 0);
  }
  const_iterator end() const {
    return ProcessedNodeWrapperBaseIter(
        static_cast<const DerivedWrapper*>(this),
        static_cast<const DerivedWrapper*>(this)->size());
  }

  const_iterator cbegin() const {
    return ProcessedNodeWrapperBaseIter(
        static_cast<const DerivedWrapper*>(this), 0);
  }
  const_iterator cend() const {
    return ProcessedNodeWrapperBaseIter(
        static_cast<const DerivedWrapper*>(this),
        static_cast<const DerivedWrapper*>(this)->size());
  }

  bool empty() const {
    return static_cast<const DerivedWrapper*>(this)->size() == 0;
  }

 protected:
  ProcessedNode& pnode_;
};

// A ProcessedNodeWrapperBase lets us use ProcessedNode directly in a context
// where a container of IValues is expected. This trick is handy for avoiding
// refcount bumps in perf-sensitive native ops. For example, suppose we have an
// op that takes a list of tensors as an argument and we've turned the op into a
// variadic variant in static runtime. To use the PyTorch library implementation
// of the op, we would have to pack the variadic arguments into a list:
//   std::vector<Tensor> tensor_list;
//   tensor_list.reserve(pnode->num_outputs());
//   for (const auto i : c10::irange(pnode->num_inputs())
//     tensor_list.push_back(pnode->Input(i).toTensor());
//   op_impl(tensor_list);
// Using ProcessedNodeWrapperBase, we can avoid this round of refcount bumps.
// All we need to do is turn `op_impl` into a template and pass it
// ProcessedNodeInputWrapper(*pnode)!
class ProcessedNodeInputWrapper
    : public ProcessedNodeWrapperBase<ProcessedNodeInputWrapper> {
 public:
  // The last `back_elements_ignored` elements are not considered.
  // Same for the first `front_elements_ignored` elements.
  // This is useful for ops where
  // only the first N elements are tensors (N < inputs.size()).
  // For instance, the last argument to VarStack is an integer dimension.
  explicit ProcessedNodeInputWrapper(
      ProcessedNode& pnode,
      size_t front_elements_ignored = 0,
      size_t back_elements_ignored = 1)
      : ProcessedNodeWrapperBase<ProcessedNodeInputWrapper>(pnode),
        front_elements_ignored_(front_elements_ignored),
        back_elements_ignored_(back_elements_ignored) {
    TORCH_CHECK(front_elements_ignored_ <= pnode_.num_inputs());
    TORCH_CHECK(
        back_elements_ignored_ <=
        pnode_.num_inputs() - front_elements_ignored_);
  }

  size_t size() const {
    return pnode_.num_inputs() - back_elements_ignored_ -
        front_elements_ignored_;
  }

  const at::Tensor& operator[](size_t idx) const {
    TORCH_CHECK(idx < size());
    return pnode_.Input(front_elements_ignored_ + idx).toTensor();
  }

  const at::Tensor& front() const {
    TORCH_CHECK(
        !empty(),
        "Attempted to access front() of empty ProcessedNodeInputWrapper");
    return pnode_.Input(front_elements_ignored_).toTensor();
  }

  const at::Tensor& back() const {
    TORCH_CHECK(
        !empty(),
        "Attempted to access back() of empty ProcessedNodeInputWrapper");
    return pnode_.Input(pnode_.num_inputs() - back_elements_ignored_ - 1)
        .toTensor();
  }

 private:
  size_t front_elements_ignored_;
  size_t back_elements_ignored_;
};

// Similar to ProcessedNodeInputWrapper, but wraps outputs and allows for
// writing.
class ProcessedNodeOutputWrapper
    : public ProcessedNodeWrapperBase<ProcessedNodeOutputWrapper> {
 public:
  using ProcessedNodeWrapperBase<
      ProcessedNodeOutputWrapper>::ProcessedNodeWrapperBase;

  size_t size() const {
    return pnode_.num_outputs();
  }

  at::Tensor& operator[](size_t idx) const {
    TORCH_CHECK(idx < size());
    return pnode_.Output(idx).toTensor();
  }

  at::Tensor& front() const {
    TORCH_CHECK(
        !empty(),
        "Attempted to access front() of empty ProcessedNodeOutputWrapper");
    return pnode_.Output(0).toTensor();
  }

  at::Tensor& back() const {
    TORCH_CHECK(
        !empty(),
        "Attempted to access back() of empty ProcessedNodeOutputWrapper");
    return pnode_.Output(size() - 1).toTensor();
  }
};

} // namespace torch::jit

```



## High-Level Overview


This C++ file contains approximately 5 class(es)/struct(s) and 22 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `torch`

**Classes/Structs**: `facilitates`, `ProcessedNodeWrapperBase`, `ProcessedNodeWrapperBaseIter`, `ProcessedNodeInputWrapper`, `ProcessedNodeOutputWrapper`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/csrc/jit/runtime/static`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `ATen/ATen.h`
- `torch/csrc/jit/runtime/static/impl.h`


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

Files in the same folder (`torch/csrc/jit/runtime/static`):

- [`memory_planner.h_docs.md`](./memory_planner.h_docs.md)
- [`ops.h_docs.md`](./ops.h_docs.md)
- [`fusion.h_docs.md`](./fusion.h_docs.md)
- [`fusion.cpp_docs.md`](./fusion.cpp_docs.md)
- [`memory_planner.cpp_docs.md`](./memory_planner.cpp_docs.md)
- [`generated_ops.cpp_docs.md`](./generated_ops.cpp_docs.md)
- [`init.h_docs.md`](./init.h_docs.md)
- [`passes.cpp_docs.md`](./passes.cpp_docs.md)
- [`passes.h_docs.md`](./passes.h_docs.md)
- [`impl.h_docs.md`](./impl.h_docs.md)


## Cross-References

- **File Documentation**: `processed_node_wrapper.h_docs.md`
- **Keyword Index**: `processed_node_wrapper.h_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
