# Documentation: `docs/test/distributed/checkpoint/test_dedup_tensors.py_docs.md`

## File Metadata

- **Path**: `docs/test/distributed/checkpoint/test_dedup_tensors.py_docs.md`
- **Size**: 5,353 bytes (5.23 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**. This appears to be a **test file**.

## Original Source

```markdown
# Documentation: `test/distributed/checkpoint/test_dedup_tensors.py`

## File Metadata

- **Path**: `test/distributed/checkpoint/test_dedup_tensors.py`
- **Size**: 1,897 bytes (1.85 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Can be **executed as a standalone script**.

## Original Source

```python
# Owner(s): ["oncall: distributed"]

import dataclasses

import torch
from torch.distributed.checkpoint._dedup_tensors import dedup_tensors
from torch.distributed.checkpoint.planner import SavePlan, WriteItemType
from torch.distributed.checkpoint.planner_helpers import _create_write_item_for_tensor
from torch.testing._internal.common_utils import run_tests, TestCase


def create_plan(second_fqn: str) -> SavePlan:
    """
    Creates a SavePlan with two write items:

    1. A write item representing a shard of a tensor named "tensor_0".
    2. A write item representing another tensor identified by the provided second_fqn.

    Args:
        second_fqn (str): The fully qualified name for the second tensor.

    Returns:
        SavePlan: A plan that includes the two write items.
    """
    # the first write item is for a duplicated shard (that covers the whole tensor)
    write_item_1 = _create_write_item_for_tensor("tensor_0", torch.rand(4))
    write_item_1 = dataclasses.replace(write_item_1, type=WriteItemType.SHARD)

    # the second write item has different keys
    write_item_2 = _create_write_item_for_tensor(second_fqn, torch.rand(10))

    return SavePlan([write_item_1, write_item_2])


class TestDedupTensor(TestCase):
    """
    Test class for deduplication of tensor write items across different ranks.
    """

    def test_dedup_shards(self):
        rank0 = create_plan("r0")
        rank1 = create_plan("r1")

        dedup_plans = dedup_tensors([rank0, rank1])

        self.assertEqual(2, len(dedup_plans[0].items))
        self.assertEqual(1, len(dedup_plans[1].items))

        self.assertIn("tensor_0", (item.index.fqn for item in dedup_plans[0].items))
        self.assertIn("r0", (item.index.fqn for item in dedup_plans[0].items))

        self.assertIn("r1", (item.index.fqn for item in dedup_plans[1].items))


if __name__ == "__main__":
    run_tests()

```



## High-Level Overview

"""    Creates a SavePlan with two write items:    1. A write item representing a shard of a tensor named "tensor_0".    2. A write item representing another tensor identified by the provided second_fqn.    Args:        second_fqn (str): The fully qualified name for the second tensor.    Returns:        SavePlan: A plan that includes the two write items.

This Python file contains 2 class(es) and 2 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `TestDedupTensor`

**Functions defined**: `create_plan`, `test_dedup_shards`

**Key imports**: dataclasses, torch, dedup_tensors, SavePlan, WriteItemType, _create_write_item_for_tensor, run_tests, TestCase


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/distributed/checkpoint`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `dataclasses`
- `torch`
- `torch.distributed.checkpoint._dedup_tensors`: dedup_tensors
- `torch.distributed.checkpoint.planner`: SavePlan, WriteItemType
- `torch.distributed.checkpoint.planner_helpers`: _create_write_item_for_tensor
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
python test/distributed/checkpoint/test_dedup_tensors.py
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
- [`test_nested_dict.py_docs.md`](./test_nested_dict.py_docs.md)
- [`test_hf_storage.py_docs.md`](./test_hf_storage.py_docs.md)
- [`test_hf_safetensor_e2e.py_docs.md`](./test_hf_safetensor_e2e.py_docs.md)
- [`test_fsdp_optim_state.py_docs.md`](./test_fsdp_optim_state.py_docs.md)
- [`test_state_dict_stager.py_docs.md`](./test_state_dict_stager.py_docs.md)


## Cross-References

- **File Documentation**: `test_dedup_tensors.py_docs.md`
- **Keyword Index**: `test_dedup_tensors.py_kw.md`
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
python docs/test/distributed/checkpoint/test_dedup_tensors.py_docs.md
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
- [`test_fsspec.py_kw.md_docs.md`](./test_fsspec.py_kw.md_docs.md)
- [`test_quantized_hf_storage.py_kw.md_docs.md`](./test_quantized_hf_storage.py_kw.md_docs.md)
- [`test_pg_transport.py_kw.md_docs.md`](./test_pg_transport.py_kw.md_docs.md)
- [`test_dedup_tensors.py_kw.md_docs.md`](./test_dedup_tensors.py_kw.md_docs.md)
- [`test_hsdp_checkpoint.py_docs.md_docs.md`](./test_hsdp_checkpoint.py_docs.md_docs.md)


## Cross-References

- **File Documentation**: `test_dedup_tensors.py_docs.md_docs.md`
- **Keyword Index**: `test_dedup_tensors.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
