# Documentation: `docs/torch/csrc/api/include/torch/nn/modules/common.h_docs.md`

## File Metadata

- **Path**: `docs/torch/csrc/api/include/torch/nn/modules/common.h_docs.md`
- **Size**: 6,570 bytes (6.42 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `torch/csrc/api/include/torch/nn/modules/common.h`

## File Metadata

- **Path**: `torch/csrc/api/include/torch/nn/modules/common.h`
- **Size**: 4,348 bytes (4.25 KB)
- **Type**: C/C++ Header File
- **Extension**: `.h`

## File Purpose

This is a c/c++ header file that is part of the PyTorch project.

## Original Source

```c
#pragma once

/// This macro enables a module with default arguments in its forward method
/// to be used in a Sequential module.
///
/// Example usage:
///
/// Let's say we have a module declared like this:
/// ```
/// struct MImpl : torch::nn::Module {
///  public:
///   explicit MImpl(int value_) : value(value_) {}
///   torch::Tensor forward(int a, int b = 2, double c = 3.0) {
///     return torch::tensor(a + b + c);
///   }
///  private:
///   int value;
/// };
/// TORCH_MODULE(M);
/// ```
///
/// If we try to use it in a Sequential module and run forward:
/// ```
/// torch::nn::Sequential seq(M(1));
/// seq->forward(1);
/// ```
///
/// We will receive the following error message:
/// ```
/// MImpl's forward() method expects 3 argument(s), but received 1.
/// If MImpl's forward() method has default arguments, please make sure
/// the forward() method is declared with a corresponding
/// `FORWARD_HAS_DEFAULT_ARGS` macro.
/// ```
///
/// The right way to fix this error is to use the `FORWARD_HAS_DEFAULT_ARGS`
/// macro when declaring the module:
/// ```
/// struct MImpl : torch::nn::Module {
///  public:
///   explicit MImpl(int value_) : value(value_) {}
///   torch::Tensor forward(int a, int b = 2, double c = 3.0) {
///     return torch::tensor(a + b + c);
///   }
///  protected:
///   /*
///   NOTE: looking at the argument list of `forward`:
///   `forward(int a, int b = 2, double c = 3.0)`
///   we saw the following default arguments:
///   ----------------------------------------------------------------
///   0-based index of default |         Default value of arg
///   arg in forward arg list  |  (wrapped by `torch::nn::AnyValue()`)
///   ----------------------------------------------------------------
///               1            |       torch::nn::AnyValue(2)
///               2            |       torch::nn::AnyValue(3.0)
///   ----------------------------------------------------------------
///   Thus we pass the following arguments to the `FORWARD_HAS_DEFAULT_ARGS`
///   macro:
///   */
///   FORWARD_HAS_DEFAULT_ARGS({1, torch::nn::AnyValue(2)}, {2,
///   torch::nn::AnyValue(3.0)})
///  private:
///   int value;
/// };
/// TORCH_MODULE(M);
/// ```
/// Now, running the following would work:
/// ```
/// torch::nn::Sequential seq(M(1));
/// seq->forward(1);  // This correctly populates the default arguments for
/// `MImpl::forward`
/// ```
#define FORWARD_HAS_DEFAULT_ARGS(...)                                    \
  template <typename ModuleType, typename... ArgumentTypes>              \
  friend struct torch::nn::AnyModuleHolder;                              \
  bool _forward_has_default_args() override {                            \
    return true;                                                         \
  }                                                                      \
  unsigned int _forward_num_required_args() override {                   \
    std::vector<std::pair<unsigned int, torch::nn::AnyValue>> args_info{ \
        __VA_ARGS__};                                                    \
    return std::begin(args_info)->first;                                 \
  }                                                                      \
  std::vector<torch::nn::AnyValue> _forward_populate_default_args(       \
      std::vector<torch::nn::AnyValue>&& arguments) override {           \
    std::vector<std::pair<unsigned int, torch::nn::AnyValue>> args_info{ \
        __VA_ARGS__};                                                    \
    unsigned int num_all_args = std::rbegin(args_info)->first + 1;       \
    TORCH_INTERNAL_ASSERT(                                               \
        arguments.size() >= _forward_num_required_args() &&              \
        arguments.size() <= num_all_args);                               \
    std::vector<torch::nn::AnyValue> ret = std::move(arguments);         \
    ret.reserve(num_all_args);                                           \
    for (auto& arg_info : args_info) {                                   \
      if (arg_info.first > ret.size() - 1)                               \
        ret.emplace_back(std::move(arg_info.second));                    \
    }                                                                    \
    return ret;                                                          \
  }

```



## High-Level Overview


This C++ file contains approximately 0 class(es)/struct(s) and 13 function(s).

## Detailed Analysis

### Code Structure

**Classes/Structs**: `MImpl`, `MImpl`, `torch`


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `torch/csrc/api/include/torch/nn/modules`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

*No includes detected.*


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

Files in the same folder (`torch/csrc/api/include/torch/nn/modules`):

- [`embedding.h_docs.md`](./embedding.h_docs.md)
- [`normalization.h_docs.md`](./normalization.h_docs.md)
- [`loss.h_docs.md`](./loss.h_docs.md)
- [`distance.h_docs.md`](./distance.h_docs.md)
- [`batchnorm.h_docs.md`](./batchnorm.h_docs.md)
- [`adaptive.h_docs.md`](./adaptive.h_docs.md)
- [`activation.h_docs.md`](./activation.h_docs.md)
- [`padding.h_docs.md`](./padding.h_docs.md)
- [`utils.h_docs.md`](./utils.h_docs.md)


## Cross-References

- **File Documentation**: `common.h_docs.md`
- **Keyword Index**: `common.h_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/torch/csrc/api/include/torch/nn/modules`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/torch/csrc/api/include/torch/nn/modules`, which is part of the **core PyTorch library**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

*No specific patterns automatically detected.*


## Performance Considerations

### Performance Notes

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

Files in the same folder (`docs/torch/csrc/api/include/torch/nn/modules`):

- [`transformerlayer.h_docs.md_docs.md`](./transformerlayer.h_docs.md_docs.md)
- [`linear.h_kw.md_docs.md`](./linear.h_kw.md_docs.md)
- [`transformercoder.h_docs.md_docs.md`](./transformercoder.h_docs.md_docs.md)
- [`loss.h_kw.md_docs.md`](./loss.h_kw.md_docs.md)
- [`batchnorm.h_docs.md_docs.md`](./batchnorm.h_docs.md_docs.md)
- [`normalization.h_kw.md_docs.md`](./normalization.h_kw.md_docs.md)
- [`fold.h_docs.md_docs.md`](./fold.h_docs.md_docs.md)
- [`distance.h_kw.md_docs.md`](./distance.h_kw.md_docs.md)
- [`utils.h_docs.md_docs.md`](./utils.h_docs.md_docs.md)
- [`transformercoder.h_kw.md_docs.md`](./transformercoder.h_kw.md_docs.md)


## Cross-References

- **File Documentation**: `common.h_docs.md_docs.md`
- **Keyword Index**: `common.h_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
