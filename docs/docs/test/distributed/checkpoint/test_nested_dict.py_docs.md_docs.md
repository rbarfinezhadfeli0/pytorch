# Documentation: `docs/test/distributed/checkpoint/test_nested_dict.py_docs.md`

## File Metadata

- **Path**: `docs/test/distributed/checkpoint/test_nested_dict.py_docs.md`
- **Size**: 5,120 bytes (5.00 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**. This appears to be a **test file**.

## Original Source

```markdown
# Documentation: `test/distributed/checkpoint/test_nested_dict.py`

## File Metadata

- **Path**: `test/distributed/checkpoint/test_nested_dict.py`
- **Size**: 2,065 bytes (2.02 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Can be **executed as a standalone script**.

## Original Source

```python
# Owner(s): ["oncall: distributed"]

import torch
from torch.distributed.checkpoint._nested_dict import (
    flatten_state_dict,
    unflatten_state_dict,
)
from torch.testing._internal.common_utils import run_tests, TestCase


class TestFlattening(TestCase):
    def test_flattening_round_trip(self) -> None:
        state_dict = {
            "key0": 1,
            "key1": [1, 2],
            "key2": {"1": 2, "2": 3},
            "key3": torch.tensor([1]),
            "key4": [[torch.tensor(2), "x"], [1, 2, 3], {"key6": [44]}],
        }

        flatten_dict, mapping = flatten_state_dict(state_dict)
        """
        flatten_dict:
            {
                'key0': 1,
                'key1': [1, 2],
                'key2': {'1': 2, '2': 3},
                'key3': tensor([1]),
                'key4.0.0': tensor(2),
                'key4.0.1': 'x',
                'key4.1': [1, 2, 3],
                'key4.2': {'key6': [44]}
            }
        """
        restored = unflatten_state_dict(flatten_dict, mapping)

        self.assertEqual(state_dict, restored)

    def test_mapping(self) -> None:
        state_dict = {
            "k0": [1],
            "k2": [torch.tensor([1]), 99, [{"k3": torch.tensor(1)}]],
            "k3": ["x", 99, [{"k3": "y"}]],
        }

        _, mapping = flatten_state_dict(state_dict)
        """
        flatten_dict:
        {'k0': [1], 'k2.0': tensor([1]), 'k2.1': 99, 'k2.2.0.k3': tensor(1), 'k3': ['x', 99, [{'k3': 'y'}]]}
        mapping:
        {'k0': ('k0',), 'k2.0': ('k2', 0), 'k2.1': ('k2', 1), 'k2.2.0.k3': ('k2', 2, 0, 'k3'), 'k3': ('k3',)}
        """

        self.assertEqual(("k0",), mapping["k0"])
        self.assertEqual(("k2", 0), mapping["k2.0"])
        self.assertEqual(("k2", 1), mapping["k2.1"])
        self.assertEqual(("k2", 2, 0, "k3"), mapping["k2.2.0.k3"])
        self.assertEqual(("k3", 0), mapping["k3.0"])
        self.assertEqual(("k3", 1), mapping["k3.1"])
        self.assertEqual(("k3", 2, 0, "k3"), mapping["k3.2.0.k3"])


if __name__ == "__main__":
    run_tests()

```



## High-Level Overview

"""        flatten_dict:            {                'key0': 1,                'key1': [1, 2],                'key2': {'1': 2, '2': 3},                'key3': tensor([1]),                'key4.0.0': tensor(2),                'key4.0.1': 'x',                'key4.1': [1, 2, 3],                'key4.2': {'key6': [44]}            }

This Python file contains 1 class(es) and 2 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `TestFlattening`

**Functions defined**: `test_flattening_round_trip`, `test_mapping`

**Key imports**: torch, run_tests, TestCase


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/distributed/checkpoint`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `torch`
- `torch.testing._internal.common_utils`: run_tests, TestCase


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

This is a test file. Run it with:

```bash
python test/distributed/checkpoint/test_nested_dict.py
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`test/distributed/checkpoint`):

- [`test_format_utils.py_docs.md`](./test_format_utils.py_docs.md)
- [`test_save_load_api.py_docs.md`](./test_save_load_api.py_docs.md)
- [`test_pg_transport.py_docs.md`](./test_pg_transport.py_docs.md)
- [`test_async_process_executor.py_docs.md`](./test_async_process_executor.py_docs.md)
- [`test_file_system_checkpoint.py_docs.md`](./test_file_system_checkpoint.py_docs.md)
- [`test_hf_storage.py_docs.md`](./test_hf_storage.py_docs.md)
- [`test_hf_safetensor_e2e.py_docs.md`](./test_hf_safetensor_e2e.py_docs.md)
- [`test_fsdp_optim_state.py_docs.md`](./test_fsdp_optim_state.py_docs.md)
- [`test_state_dict_stager.py_docs.md`](./test_state_dict_stager.py_docs.md)


## Cross-References

- **File Documentation**: `test_nested_dict.py_docs.md`
- **Keyword Index**: `test_nested_dict.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/test/distributed/checkpoint`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/test/distributed/checkpoint`, which is part of the **testing infrastructure**.



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

This is a test file. Run it with:

```bash
python docs/test/distributed/checkpoint/test_nested_dict.py_docs.md
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`docs/test/distributed/checkpoint`):

- [`test_state_dict.py_docs.md_docs.md`](./test_state_dict.py_docs.md_docs.md)
- [`test_checkpoint.py_kw.md_docs.md`](./test_checkpoint.py_kw.md_docs.md)
- [`test_dtensor_checkpoint.py_docs.md_docs.md`](./test_dtensor_checkpoint.py_docs.md_docs.md)
- [`test_file_system_checkpoint_cpu.py_docs.md_docs.md`](./test_file_system_checkpoint_cpu.py_docs.md_docs.md)
- [`test_dedup_tensors.py_docs.md_docs.md`](./test_dedup_tensors.py_docs.md_docs.md)
- [`test_fsspec.py_kw.md_docs.md`](./test_fsspec.py_kw.md_docs.md)
- [`test_quantized_hf_storage.py_kw.md_docs.md`](./test_quantized_hf_storage.py_kw.md_docs.md)
- [`test_pg_transport.py_kw.md_docs.md`](./test_pg_transport.py_kw.md_docs.md)
- [`test_dedup_tensors.py_kw.md_docs.md`](./test_dedup_tensors.py_kw.md_docs.md)
- [`test_hsdp_checkpoint.py_docs.md_docs.md`](./test_hsdp_checkpoint.py_docs.md_docs.md)


## Cross-References

- **File Documentation**: `test_nested_dict.py_docs.md_docs.md`
- **Keyword Index**: `test_nested_dict.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
