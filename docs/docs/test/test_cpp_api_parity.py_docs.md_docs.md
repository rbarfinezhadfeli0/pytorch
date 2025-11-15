# Documentation: `docs/test/test_cpp_api_parity.py_docs.md`

## File Metadata

- **Path**: `docs/test/test_cpp_api_parity.py_docs.md`
- **Size**: 5,944 bytes (5.80 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**. This appears to be a **test file**.

## Original Source

```markdown
# Documentation: `test/test_cpp_api_parity.py`

## File Metadata

- **Path**: `test/test_cpp_api_parity.py`
- **Size**: 3,026 bytes (2.96 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Can be **executed as a standalone script**.

## Original Source

```python
# Owner(s): ["module: cpp"]


import os

from cpp_api_parity import (
    functional_impl_check,
    module_impl_check,
    sample_functional,
    sample_module,
)
from cpp_api_parity.parity_table_parser import parse_parity_tracker_table
from cpp_api_parity.utils import is_torch_nn_functional_test

import torch
import torch.testing._internal.common_nn as common_nn
import torch.testing._internal.common_utils as common


# NOTE: turn this on if you want to print source code of all C++ tests (e.g. for debugging purpose)
PRINT_CPP_SOURCE = False

devices = ["cpu", "cuda"]

PARITY_TABLE_PATH = os.path.join(
    os.path.dirname(__file__), "cpp_api_parity", "parity-tracker.md"
)

parity_table = parse_parity_tracker_table(PARITY_TABLE_PATH)


@torch.testing._internal.common_utils.markDynamoStrictTest
class TestCppApiParity(common.TestCase):
    module_test_params_map = {}
    functional_test_params_map = {}


expected_test_params_dicts = []

for test_params_dicts, test_instance_class in [
    (sample_module.module_tests, common_nn.NewModuleTest),
    (sample_functional.functional_tests, common_nn.NewModuleTest),
    (common_nn.module_tests, common_nn.NewModuleTest),
    (common_nn.get_new_module_tests(), common_nn.NewModuleTest),
    (common_nn.criterion_tests, common_nn.CriterionTest),
]:
    for test_params_dict in test_params_dicts:
        if test_params_dict.get("test_cpp_api_parity", True):
            if is_torch_nn_functional_test(test_params_dict):
                functional_impl_check.write_test_to_test_class(
                    TestCppApiParity,
                    test_params_dict,
                    test_instance_class,
                    parity_table,
                    devices,
                )
            else:
                module_impl_check.write_test_to_test_class(
                    TestCppApiParity,
                    test_params_dict,
                    test_instance_class,
                    parity_table,
                    devices,
                )
            expected_test_params_dicts.append(test_params_dict)

# Assert that all NN module/functional test dicts appear in the parity test
assert len(
    [name for name in TestCppApiParity.__dict__ if "test_torch_nn_" in name]
) == len(expected_test_params_dicts) * len(devices)

# Assert that there exists auto-generated tests for `SampleModule` and `sample_functional`.
# 4 == 2 (number of test dicts that are not skipped) * 2 (number of devices)
assert len([name for name in TestCppApiParity.__dict__ if "SampleModule" in name]) == 4
# 4 == 2 (number of test dicts that are not skipped) * 2 (number of devices)
assert (
    len([name for name in TestCppApiParity.__dict__ if "sample_functional" in name])
    == 4
)

module_impl_check.build_cpp_tests(TestCppApiParity, print_cpp_source=PRINT_CPP_SOURCE)
functional_impl_check.build_cpp_tests(
    TestCppApiParity, print_cpp_source=PRINT_CPP_SOURCE
)

if __name__ == "__main__":
    common.TestCase._default_dtype_check_enabled = True
    common.run_tests()

```



## High-Level Overview


This Python file contains 2 class(es) and 0 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `TestCppApiParity`

**Key imports**: os, parse_parity_tracker_table, is_torch_nn_functional_test, torch, torch.testing._internal.common_nn as common_nn, torch.testing._internal.common_utils as common


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `os`
- `cpp_api_parity.parity_table_parser`: parse_parity_tracker_table
- `cpp_api_parity.utils`: is_torch_nn_functional_test
- `torch`
- `torch.testing._internal.common_nn as common_nn`
- `torch.testing._internal.common_utils as common`


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

This is a test file. Run it with:

```bash
python test/test_cpp_api_parity.py
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`test`):

- [`test_file_check.py_docs.md`](./test_file_check.py_docs.md)
- [`test_jit_simple.py_docs.md`](./test_jit_simple.py_docs.md)
- [`test_mkldnn.py_docs.md`](./test_mkldnn.py_docs.md)
- [`test_expanded_weights.py_docs.md`](./test_expanded_weights.py_docs.md)
- [`test_overrides.py_docs.md`](./test_overrides.py_docs.md)
- [`test_decomp.py_docs.md`](./test_decomp.py_docs.md)
- [`test_show_pickle.py_docs.md`](./test_show_pickle.py_docs.md)
- [`test_utils_config_module.py_docs.md`](./test_utils_config_module.py_docs.md)
- [`test_mobile_optimizer.py_docs.md`](./test_mobile_optimizer.py_docs.md)
- [`test_type_info.py_docs.md`](./test_type_info.py_docs.md)


## Cross-References

- **File Documentation**: `test_cpp_api_parity.py_docs.md`
- **Keyword Index**: `test_cpp_api_parity.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/test`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/test`, which is part of the **testing infrastructure**.



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

- **Serialization**: Uses pickle - be cautious with untrusted data

*Manual security review is recommended for production code.*


## Testing & Usage

### Testing

This is a test file. Run it with:

```bash
python docs/test/test_cpp_api_parity.py_docs.md
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`docs/test`):

- [`test_ops.py_docs.md_docs.md`](./test_ops.py_docs.md_docs.md)
- [`test_tensorexpr.py_docs.md_docs.md`](./test_tensorexpr.py_docs.md_docs.md)
- [`pytest_shard_custom.py_docs.md_docs.md`](./pytest_shard_custom.py_docs.md_docs.md)
- [`test_weak.py_kw.md_docs.md`](./test_weak.py_kw.md_docs.md)
- [`test_view_ops.py_kw.md_docs.md`](./test_view_ops.py_kw.md_docs.md)
- [`test_varlen_attention.py_kw.md_docs.md`](./test_varlen_attention.py_kw.md_docs.md)
- [`test_namedtensor.py_docs.md_docs.md`](./test_namedtensor.py_docs.md_docs.md)
- [`test_binary_ufuncs.py_docs.md_docs.md`](./test_binary_ufuncs.py_docs.md_docs.md)
- [`test_ops_gradients.py_kw.md_docs.md`](./test_ops_gradients.py_kw.md_docs.md)
- [`test_torchfuzz_repros.py_docs.md_docs.md`](./test_torchfuzz_repros.py_docs.md_docs.md)


## Cross-References

- **File Documentation**: `test_cpp_api_parity.py_docs.md_docs.md`
- **Keyword Index**: `test_cpp_api_parity.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
