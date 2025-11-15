# Documentation: `docs/torch/csrc/autograd/functions/comm.cpp_docs.md`

## File Metadata

- **Path**: `docs/torch/csrc/autograd/functions/comm.cpp_docs.md`
- **Size**: 6,541 bytes (6.39 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/csrc/autograd/functions/comm.cpp`

## File Metadata

- **Path**: `torch/csrc/autograd/functions/comm.cpp`
- **Size**: 4,045 bytes (3.95 KB)
- **Type**: C++ Source Code
- **Extension**: `.cpp`

## File Purpose

This is a c++ source code that is part of the PyTorch project.

## Original Source

```cpp
#include <torch/csrc/autograd/functions/comm.h>

#include <ATen/core/functional.h>
#include <torch/csrc/autograd/function.h>
#include <torch/csrc/autograd/functions/utils.h>
#include <torch/csrc/autograd/variable.h>
#include <torch/csrc/cuda/comm.h>

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

#include <memory>
#include <vector>

namespace torch::autograd {
Scatter::Scatter(
    std::vector<at::Device> devices,
    std::optional<std::vector<int64_t>> chunk_sizes,
    int64_t dim,
    std::optional<std::vector<std::optional<at::cuda::CUDAStream>>> streams,
    bool unsqueeze_scalars)
    : devices_(std::move(devices)),
      chunk_sizes_(std::move(chunk_sizes)),
      dim_(dim),
      streams_(std::move(streams)),
      unsqueeze_scalars_(unsqueeze_scalars) {}

Scatter::~Scatter() = default;

variable_list Scatter::apply(variable_list&& inputs) {
  AT_ASSERT(inputs.size() == 1);
  auto& input = inputs.front();

  std::shared_ptr<Node> grad_fn;
  if (compute_requires_grad(input)) {
    grad_fn =
        std::make_shared<Gather>(/*destination_device=*/input.device(), dim_);
    grad_fn->set_next_edges(collect_next_edges(input));
  }

  auto device_indices = fmap(devices_, [](const at::Device& device) -> int64_t {
    return device.index();
  });
  auto tensors =
      torch::cuda::scatter(input, device_indices, chunk_sizes_, dim_, streams_);

  std::vector<Variable> variables;
  variables.reserve(tensors.size());
  for (auto& tensor : tensors) {
    AT_ASSERT(tensor.defined());
    if (unsqueeze_scalars_) {
      AT_ASSERT(tensor.dim() == 1 && tensor.numel() == 1);
      variables.push_back(tensor[0]);
    } else {
      variables.push_back(std::move(tensor));
    }
  }

  if (grad_fn) {
    set_history(variables, grad_fn);
  }

  return variables;
}

Gather::Gather(const at::Device& destination_device, int64_t dim)
    : destination_device_(destination_device), dim_(dim) {}

Gather::~Gather() = default;

variable_list Gather::apply(variable_list&& inputs) {
  bool all_are_zero_dim = true;
  for (const auto& input : inputs) {
    TORCH_CHECK(
        input.is_cuda(),
        "All inputs to Gather must be CUDA tensors, got ",
        input.toString());
    if (input.dim() > 0) {
      all_are_zero_dim = false;
    }
  }

  const bool unsqueeze_scalars = all_are_zero_dim && dim_ == 0;
  if (unsqueeze_scalars) {
    TORCH_WARN(
        "Was asked to gather along dimension 0, but all "
        "input tensors were scalars; will instead unsqueeze "
        "and return a vector.");
  }

  std::shared_ptr<Node> grad_fn;
  // compute this before moving variables from `inputs`
  if (compute_requires_grad(inputs)) {
    std::vector<at::Device> source_devices;
    source_devices.reserve(inputs.size());
    std::vector<int64_t> input_sizes;
    input_sizes.reserve(inputs.size());
    for (auto& input : inputs) {
      source_devices.push_back(input.device());
      input_sizes.push_back(input.size(dim_));
    }
    grad_fn = std::make_shared<Scatter>(
        std::move(source_devices),
        std::move(input_sizes),
        dim_,
        /*streams=*/std::nullopt,
        /*unsqueeze_scalars=*/unsqueeze_scalars);
    grad_fn->set_next_edges(collect_next_edges(inputs));
  }

  std::vector<at::Tensor> tensors;
  if (unsqueeze_scalars) {
    tensors.reserve(inputs.size());
    for (auto& variable : inputs) {
      tensors.push_back(variable.view(1));
    }
  } else {
    tensors = std::move(inputs);
  }

  // Disable the autograd during the actual computation
  // torch::cuda::gather does not return a view or change things inplace
  // so no need for extra logic here
  at::Tensor variable;
  {
    at::AutoDispatchBelowAutograd mode;
    // This is special logic for torch::cuda::gather!
    const auto destination_index =
        destination_device_.is_cpu() ? -1 : destination_device_.index();
    variable = torch::cuda::gather(tensors, dim_, destination_index);
  }
  if (grad_fn) {
    set_history(variable, grad_fn);
  }
  return {variable};
}

} // namespace torch::autograd

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 0 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `torch`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/csrc/autograd/functions`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `torch/csrc/autograd/functions/comm.h`
- `ATen/core/functional.h`
- `torch/csrc/autograd/function.h`
- `torch/csrc/autograd/functions/utils.h`
- `torch/csrc/autograd/variable.h`
- `torch/csrc/cuda/comm.h`
- `ATen/ATen.h`
- `ATen/cuda/CUDAContext.h`
- `memory`
- `vector`


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

Files in the same folder (`torch/csrc/autograd/functions`):

- [`basic_ops.cpp_docs.md`](./basic_ops.cpp_docs.md)
- [`utils.cpp_docs.md`](./utils.cpp_docs.md)
- [`utils.h_docs.md`](./utils.h_docs.md)
- [`tensor.cpp_docs.md`](./tensor.cpp_docs.md)
- [`tensor.h_docs.md`](./tensor.h_docs.md)
- [`accumulate_grad.h_docs.md`](./accumulate_grad.h_docs.md)
- [`init.cpp_docs.md`](./init.cpp_docs.md)
- [`comm.h_docs.md`](./comm.h_docs.md)
- [`basic_ops.h_docs.md`](./basic_ops.h_docs.md)


## Cross-References

- **File Documentation**: `comm.cpp_docs.md`
- **Keyword Index**: `comm.cpp_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/csrc/autograd/functions`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/csrc/autograd/functions`, which is part of the **core PyTorch library**.



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

Files in the same folder (`docs/torch/csrc/autograd/functions`):

- [`utils.cpp_docs.md_docs.md`](./utils.cpp_docs.md_docs.md)
- [`basic_ops.cpp_kw.md_docs.md`](./basic_ops.cpp_kw.md_docs.md)
- [`accumulate_grad.h_kw.md_docs.md`](./accumulate_grad.h_kw.md_docs.md)
- [`utils.h_docs.md_docs.md`](./utils.h_docs.md_docs.md)
- [`pybind.h_docs.md_docs.md`](./pybind.h_docs.md_docs.md)
- [`utils.h_kw.md_docs.md`](./utils.h_kw.md_docs.md)
- [`accumulate_grad.h_docs.md_docs.md`](./accumulate_grad.h_docs.md_docs.md)
- [`init.cpp_kw.md_docs.md`](./init.cpp_kw.md_docs.md)
- [`basic_ops.h_kw.md_docs.md`](./basic_ops.h_kw.md_docs.md)
- [`init.cpp_docs.md_docs.md`](./init.cpp_docs.md_docs.md)


## Cross-References

- **File Documentation**: `comm.cpp_docs.md_docs.md`
- **Keyword Index**: `comm.cpp_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
