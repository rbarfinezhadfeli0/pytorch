# Documentation: `docs/torch/csrc/api/include/torch/utils.h_docs.md`

## File Metadata

- **Path**: `docs/torch/csrc/api/include/torch/utils.h_docs.md`
- **Size**: 5,947 bytes (5.81 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/csrc/api/include/torch/utils.h`

## File Metadata

- **Path**: `torch/csrc/api/include/torch/utils.h`
- **Size**: 3,585 bytes (3.50 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
#pragma once

#include <ATen/Parallel.h>
#include <ATen/record_function.h>
#include <torch/csrc/api/include/torch/types.h>
#include <torch/csrc/autograd/grad_mode.h>
#include <torch/csrc/autograd/profiler.h>

// NOLINTBEGIN(misc-unused-using-decls)
namespace torch {

/// A RAII, thread-local guard that disabled gradient calculation.
///
/// Disabling gradient calculation is useful for inference, when you are sure
/// that you will not call `at::Tensor::backward`. It will reduce memory
/// consumption for computations that would otherwise have `requires_grad() ==
/// true`.
///
/// In this mode, the result of every computation will have
/// `requires_grad() == false`, even when the inputs have `requires_grad() ==
/// true`.
///
/// This context manager is thread-local; it will not affect computation
/// in other threads.
///
/// Example:
/// @code
/// auto x = torch::tensor({1.}, torch::requires_grad());
/// {
///   torch::NoGradGuard no_grad;
///   auto y = x * 2;
///   std::cout << y.requires_grad() << std::endl; // prints `false`
/// }
/// {
///   auto doubler = [](torch::Tensor x) {
///     torch::NoGradGuard no_grad;
///     return x * 2;
///   };
///   auto z = doubler(x);
///   std::cout << z.requires_grad() << std::endl; // prints `false`
/// }
/// @endcode
using NoGradGuard = at::NoGradGuard;

/// A RAII, thread-local guard that sets gradient calculation to on or off.
///
/// ``AutoGradMode`` will enable or disable grads based on its argument
/// `enabled`.
///
/// This context manager is thread-local; it will not affect computation
/// in other threads.
///
/// \param enabled: Flag whether to enable grad (``true``), or disable
///              (``false``). This can be used to conditionally enable
///              gradients.
///
/// Example:
/// @code
/// auto x = torch::tensor({1.}, torch::requires_grad());
/// {
///   torch::AutoGradMode enable_grad(true);
///   auto y = x * 2;
///   std::cout << y.requires_grad() << std::endl; // prints `true`
/// }
/// {
///   torch::AutoGradMode enable_grad(false);
///   auto y = x * 2;
///   std::cout << y.requires_grad() << std::endl; // prints `false`
/// }
/// @endcode
using AutoGradMode = at::AutoGradMode;

/// Sets the global random seed for all newly created CPU and CUDA tensors.
using at::manual_seed;

// Called during new thread initialization
using at::init_num_threads;

// Returns the number of threads used in parallel region.
using at::get_num_threads;

// Sets the number of threads to be used in parallel region.
using at::set_num_threads;

// Returns the number of threads used for inter-op parallelism.
using at::get_num_interop_threads;

// Sets the number of threads to be used for inter-op parallelism.
using at::set_num_interop_threads;

// Returns true if both t1, t2 are undefined or both are defined and equal
inline bool equal_if_defined(const Tensor& t1, const Tensor& t2) {
  return (
      (!t1.defined() && !t2.defined()) ||
      (t1.defined() && t2.defined() && torch::equal(t1, t2)));
}

// RecordFunction API
using at::addGlobalCallback;
using at::addThreadLocalCallback;
using at::CallbackHandle;
using at::clearCallbacks;
using at::clearGlobalCallbacks;
using at::clearThreadLocalCallbacks;
using at::DisableRecordFunctionGuard;
using at::enableRecordFunction;
using at::hasCallbacks;
using at::hasGlobalCallbacks;
using at::hasThreadLocalCallbacks;
using at::isRecordFunctionEnabled;
using at::RecordFunction;
using at::RecordFunctionCallback;
using at::RecordFunctionGuard;
using at::removeCallback;

} // namespace torch
// NOLINTEND(misc-unused-using-decls)

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 4 function(s).

## Detailed Analysis

### Code Structure

**Namespaces**: `torch`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/csrc/api/include/torch`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

This file includes:

- `ATen/Parallel.h`
- `ATen/record_function.h`
- `torch/csrc/api/include/torch/types.h`
- `torch/csrc/autograd/grad_mode.h`
- `torch/csrc/autograd/profiler.h`


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

Files in the same folder (`torch/csrc/api/include/torch`):

- [`ordered_dict.h_docs.md`](./ordered_dict.h_docs.md)
- [`fft.h_docs.md`](./fft.h_docs.md)
- [`nested.h_docs.md`](./nested.h_docs.md)
- [`serialize.h_docs.md`](./serialize.h_docs.md)
- [`nn.h_docs.md`](./nn.h_docs.md)
- [`special.h_docs.md`](./special.h_docs.md)
- [`expanding_array.h_docs.md`](./expanding_array.h_docs.md)
- [`data.h_docs.md`](./data.h_docs.md)
- [`version.h_docs.md`](./version.h_docs.md)


## Cross-References

- **File Documentation**: `utils.h_docs.md`
- **Keyword Index**: `utils.h_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/csrc/api/include/torch`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/csrc/api/include/torch`, which is part of the **core PyTorch library**.



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

Files in the same folder (`docs/torch/csrc/api/include/torch`):

- [`expanding_array.h_docs.md_docs.md`](./expanding_array.h_docs.md_docs.md)
- [`nn.h_kw.md_docs.md`](./nn.h_kw.md_docs.md)
- [`serialize.h_docs.md_docs.md`](./serialize.h_docs.md_docs.md)
- [`sparse.h_kw.md_docs.md`](./sparse.h_kw.md_docs.md)
- [`nested.h_docs.md_docs.md`](./nested.h_docs.md_docs.md)
- [`types.h_docs.md_docs.md`](./types.h_docs.md_docs.md)
- [`enum.h_docs.md_docs.md`](./enum.h_docs.md_docs.md)
- [`special.h_kw.md_docs.md`](./special.h_kw.md_docs.md)
- [`nn.h_docs.md_docs.md`](./nn.h_docs.md_docs.md)


## Cross-References

- **File Documentation**: `utils.h_docs.md_docs.md`
- **Keyword Index**: `utils.h_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
