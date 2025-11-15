# Documentation: `test/cpp_api_parity/sample_module.py`

## File Metadata

- **Path**: `test/cpp_api_parity/sample_module.py`
- **Size**: 3,462 bytes (3.38 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**.

## Original Source

```python
import torch


"""
`SampleModule` is used by `test_cpp_api_parity.py` to test that Python / C++ API
parity test harness works for `torch.nn.Module` subclasses.

When `SampleModule.has_parity` is true, behavior of `forward` / `backward`
is the same as the C++ equivalent.

When `SampleModule.has_parity` is false, behavior of `forward` / `backward`
is different from the C++ equivalent.
"""


class SampleModule(torch.nn.Module):
    def __init__(self, has_parity, has_submodule):
        super().__init__()
        self.has_parity = has_parity
        if has_submodule:
            self.submodule = SampleModule(self.has_parity, False)

        self.has_submodule = has_submodule
        self.register_parameter("param", torch.nn.Parameter(torch.empty(3, 4)))

        self.reset_parameters()

    def reset_parameters(self):
        with torch.no_grad():
            self.param.fill_(1)

    def forward(self, x):
        submodule_forward_result = (
            self.submodule(x) if hasattr(self, "submodule") else 0
        )
        if self.has_parity:
            return x + self.param * 2 + submodule_forward_result
        else:
            return x + self.param * 4 + submodule_forward_result + 3


torch.nn.SampleModule = SampleModule

SAMPLE_MODULE_CPP_SOURCE = """\n
namespace torch {
namespace nn {
struct C10_EXPORT SampleModuleOptions {
  SampleModuleOptions(bool has_parity, bool has_submodule) : has_parity_(has_parity), has_submodule_(has_submodule) {}

  TORCH_ARG(bool, has_parity);
  TORCH_ARG(bool, has_submodule);
};

struct C10_EXPORT SampleModuleImpl : public torch::nn::Cloneable<SampleModuleImpl> {
  explicit SampleModuleImpl(SampleModuleOptions options) : options(std::move(options)) {
    if (options.has_submodule()) {
      submodule = register_module(
        "submodule",
        std::make_shared<SampleModuleImpl>(SampleModuleOptions(options.has_parity(), false)));
    }
    reset();
  }
  void reset() {
    param = register_parameter("param", torch::ones({3, 4}));
  }
  torch::Tensor forward(torch::Tensor x) {
    return x + param * 2 + (submodule ? submodule->forward(x) : torch::zeros_like(x));
  }
  SampleModuleOptions options;
  torch::Tensor param;
  std::shared_ptr<SampleModuleImpl> submodule{nullptr};
};

TORCH_MODULE(SampleModule);
} // namespace nn
} // namespace torch
"""

module_tests = [
    dict(
        module_name="SampleModule",
        desc="has_parity",
        constructor_args=(True, True),
        cpp_constructor_args="torch::nn::SampleModuleOptions(true, true)",
        input_size=(3, 4),
        cpp_input_args=["torch::randn({3, 4})"],
        has_parity=True,
    ),
    dict(
        fullname="SampleModule_no_parity",
        constructor=lambda: SampleModule(has_parity=False, has_submodule=True),
        cpp_constructor_args="torch::nn::SampleModuleOptions(false, true)",
        input_size=(3, 4),
        cpp_input_args=["torch::randn({3, 4})"],
        has_parity=False,
    ),
    # This is to test that setting the `test_cpp_api_parity=False` flag skips
    # the C++ API parity test accordingly (otherwise this test would run and
    # throw a parity error).
    dict(
        fullname="SampleModule_THIS_TEST_SHOULD_BE_SKIPPED",
        constructor=lambda: SampleModule(False, True),
        cpp_constructor_args="torch::nn::SampleModuleOptions(false, true)",
        input_size=(3, 4),
        cpp_input_args=["torch::randn({3, 4})"],
        test_cpp_api_parity=False,
    ),
]

```



## High-Level Overview

"""`SampleModule` is used by `test_cpp_api_parity.py` to test that Python / C++ APIparity test harness works for `torch.nn.Module` subclasses.When `SampleModule.has_parity` is true, behavior of `forward` / `backward`is the same as the C++ equivalent.When `SampleModule.has_parity` is false, behavior of `forward` / `backward`is different from the C++ equivalent.

This Python file contains 1 class(es) and 3 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `SampleModule`

**Functions defined**: `__init__`, `reset_parameters`, `forward`

**Key imports**: torch


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/cpp_api_parity`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `torch`


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors
- **Neural Network**: Defines or uses PyTorch neural network components


## Performance Considerations

### Performance Notes


*Detailed performance analysis requires profiling and benchmarking.*


## Security & Safety

### Security Considerations

- No obvious security concerns detected in automated analysis.

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

This is a test file. Run it with:

```bash
python test/cpp_api_parity/sample_module.py
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`test/cpp_api_parity`):

- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`sample_functional.py_docs.md`](./sample_functional.py_docs.md)
- [`module_impl_check.py_docs.md`](./module_impl_check.py_docs.md)
- [`utils.py_docs.md`](./utils.py_docs.md)
- [`parity-tracker.md_docs.md`](./parity-tracker.md_docs.md)
- [`parity_table_parser.py_docs.md`](./parity_table_parser.py_docs.md)
- [`functional_impl_check.py_docs.md`](./functional_impl_check.py_docs.md)


## Cross-References

- **File Documentation**: `sample_module.py_docs.md`
- **Keyword Index**: `sample_module.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
