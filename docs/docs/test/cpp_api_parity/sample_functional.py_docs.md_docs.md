# Documentation: `docs/test/cpp_api_parity/sample_functional.py_docs.md`

## File Metadata

- **Path**: `docs/test/cpp_api_parity/sample_functional.py_docs.md`
- **Size**: 4,901 bytes (4.79 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**.

## Original Source

```markdown
# Documentation: `test/cpp_api_parity/sample_functional.py`

## File Metadata

- **Path**: `test/cpp_api_parity/sample_functional.py`
- **Size**: 2,132 bytes (2.08 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**.

## Original Source

```python
import torch
import torch.nn.functional as F
from torch.testing._internal.common_nn import wrap_functional


"""
`sample_functional` is used by `test_cpp_api_parity.py` to test that Python / C++ API
parity test harness works for `torch.nn.functional` functions.

When `has_parity=true` is passed to `sample_functional`, behavior of `sample_functional`
is the same as the C++ equivalent.

When `has_parity=false` is passed to `sample_functional`, behavior of `sample_functional`
is different from the C++ equivalent.
"""


def sample_functional(x, has_parity):
    if has_parity:
        return x * 2
    else:
        return x * 4


torch.nn.functional.sample_functional = sample_functional

SAMPLE_FUNCTIONAL_CPP_SOURCE = """\n
namespace torch {
namespace nn {
namespace functional {

struct C10_EXPORT SampleFunctionalFuncOptions {
  SampleFunctionalFuncOptions(bool has_parity) : has_parity_(has_parity) {}

  TORCH_ARG(bool, has_parity);
};

Tensor sample_functional(Tensor x, SampleFunctionalFuncOptions options) {
    return x * 2;
}

} // namespace functional
} // namespace nn
} // namespace torch
"""

functional_tests = [
    dict(
        constructor=wrap_functional(F.sample_functional, has_parity=True),
        cpp_options_args="F::SampleFunctionalFuncOptions(true)",
        input_size=(1, 2, 3),
        fullname="sample_functional_has_parity",
        has_parity=True,
    ),
    dict(
        constructor=wrap_functional(F.sample_functional, has_parity=False),
        cpp_options_args="F::SampleFunctionalFuncOptions(false)",
        input_size=(1, 2, 3),
        fullname="sample_functional_no_parity",
        has_parity=False,
    ),
    # This is to test that setting the `test_cpp_api_parity=False` flag skips
    # the C++ API parity test accordingly (otherwise this test would run and
    # throw a parity error).
    dict(
        constructor=wrap_functional(F.sample_functional, has_parity=False),
        cpp_options_args="F::SampleFunctionalFuncOptions(false)",
        input_size=(1, 2, 3),
        fullname="sample_functional_THIS_TEST_SHOULD_BE_SKIPPED",
        test_cpp_api_parity=False,
    ),
]

```



## High-Level Overview

"""`sample_functional` is used by `test_cpp_api_parity.py` to test that Python / C++ APIparity test harness works for `torch.nn.functional` functions.When `has_parity=true` is passed to `sample_functional`, behavior of `sample_functional`is the same as the C++ equivalent.When `has_parity=false` is passed to `sample_functional`, behavior of `sample_functional`is different from the C++ equivalent.

This Python file contains 0 class(es) and 1 function(s).

## Detailed Analysis

### Code Structure

**Functions defined**: `sample_functional`

**Key imports**: torch, torch.nn.functional as F, wrap_functional


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/cpp_api_parity`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `torch`
- `torch.nn.functional as F`
- `torch.testing._internal.common_nn`: wrap_functional


## Code Patterns & Idioms

### Common Patterns

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
python test/cpp_api_parity/sample_functional.py
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`test/cpp_api_parity`):

- [`__init__.py_docs.md`](./__init__.py_docs.md)
- [`module_impl_check.py_docs.md`](./module_impl_check.py_docs.md)
- [`sample_module.py_docs.md`](./sample_module.py_docs.md)
- [`utils.py_docs.md`](./utils.py_docs.md)
- [`parity-tracker.md_docs.md`](./parity-tracker.md_docs.md)
- [`parity_table_parser.py_docs.md`](./parity_table_parser.py_docs.md)
- [`functional_impl_check.py_docs.md`](./functional_impl_check.py_docs.md)


## Cross-References

- **File Documentation**: `sample_functional.py_docs.md`
- **Keyword Index**: `sample_functional.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/test/cpp_api_parity`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/test/cpp_api_parity`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

*Dependency analysis not applicable for this file type.*


## Code Patterns & Idioms

### Common Patterns

- **Neural Network**: Defines or uses PyTorch neural network components


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

This is a test file. Run it with:

```bash
python docs/test/cpp_api_parity/sample_functional.py_docs.md
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`docs/test/cpp_api_parity`):

- [`parity_table_parser.py_docs.md_docs.md`](./parity_table_parser.py_docs.md_docs.md)
- [`module_impl_check.py_docs.md_docs.md`](./module_impl_check.py_docs.md_docs.md)
- [`module_impl_check.py_kw.md_docs.md`](./module_impl_check.py_kw.md_docs.md)
- [`parity-tracker.md_kw.md_docs.md`](./parity-tracker.md_kw.md_docs.md)
- [`sample_module.py_kw.md_docs.md`](./sample_module.py_kw.md_docs.md)
- [`parity_table_parser.py_kw.md_docs.md`](./parity_table_parser.py_kw.md_docs.md)
- [`utils.py_docs.md_docs.md`](./utils.py_docs.md_docs.md)
- [`functional_impl_check.py_docs.md_docs.md`](./functional_impl_check.py_docs.md_docs.md)
- [`sample_module.py_docs.md_docs.md`](./sample_module.py_docs.md_docs.md)


## Cross-References

- **File Documentation**: `sample_functional.py_docs.md_docs.md`
- **Keyword Index**: `sample_functional.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
