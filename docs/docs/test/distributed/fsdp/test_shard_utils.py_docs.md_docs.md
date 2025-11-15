# Documentation: `docs/test/distributed/fsdp/test_shard_utils.py_docs.md`

## File Metadata

- **Path**: `docs/test/distributed/fsdp/test_shard_utils.py_docs.md`
- **Size**: 5,634 bytes (5.50 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**. This appears to be a **test file**.

## Original Source

```markdown
# Documentation: `test/distributed/fsdp/test_shard_utils.py`

## File Metadata

- **Path**: `test/distributed/fsdp/test_shard_utils.py`
- **Size**: 2,490 bytes (2.43 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Can be **executed as a standalone script**.

## Original Source

```python
# Owner(s): ["oncall: distributed"]

import torch
from torch.distributed.distributed_c10d import _get_default_group
from torch.distributed.fsdp._shard_utils import (
    _create_chunk_dtensor,
    _create_chunk_sharded_tensor,
)
from torch.testing._internal.common_fsdp import FSDPTest
from torch.testing._internal.common_utils import run_tests
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    skip_if_lt_x_gpu,
    with_comms,
)


device_type = acc.type if (acc := torch.accelerator.current_accelerator()) else "cpu"


class TestShardUtilsDistributed(FSDPTest):
    @property
    def world_size(self):
        return 2

    def _create_tensor(self, *size):
        # Keep everything deterministic.
        torch.manual_seed(0)
        return torch.rand(*size).to(device=device_type)

    @skip_if_lt_x_gpu(2)
    def test_create_chunk_sharded_tensor(self):
        for size in ((1,), (1, 6), (12,), (12, 6), (25,), (25, 6)):
            tensor = self._create_tensor(*size)

            sharded_tensor = _create_chunk_sharded_tensor(
                tensor,
                self.rank,
                self.world_size,
                torch.accelerator.device_count(),
                _get_default_group(),
            )
            output = (
                torch.empty(*size).to(device=device_type) if self.rank == 0 else None
            )
            sharded_tensor.gather(0, output)
            if self.rank == 0:
                self.assertEqual(tensor, output)


class TestShardUtilsDistributedDTensor(DTensorTestBase):
    @property
    def world_size(self):
        return 2

    def _create_tensor(self, *size):
        # Keep everything deterministic.
        torch.manual_seed(0)
        return torch.rand(*size).to(device=device_type)

    @with_comms
    @skip_if_lt_x_gpu(2)
    def test_create_chunk_dtensor(self):
        device_mesh = self.build_device_mesh()

        for size in ((1,), (1, 6), (12,), (12, 6), (25,), (25, 6)):
            tensor = self._create_tensor(*size)
            tensor_chunks = torch.chunk(tensor, self.world_size, dim=0)

            dtensor = _create_chunk_dtensor(tensor, self.rank, device_mesh)
            local_tensor = dtensor.to_local()

            if local_tensor.numel() != 0:
                self.assertEqual(local_tensor, tensor_chunks[self.rank])
            else:
                self.assertEqual(self.rank >= len(tensor_chunks), True)


if __name__ == "__main__":
    run_tests()

```



## High-Level Overview


This Python file contains 2 class(es) and 6 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `TestShardUtilsDistributed`, `TestShardUtilsDistributedDTensor`

**Functions defined**: `world_size`, `_create_tensor`, `test_create_chunk_sharded_tensor`, `world_size`, `_create_tensor`, `test_create_chunk_dtensor`

**Key imports**: torch, _get_default_group, FSDPTest, run_tests


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/distributed/fsdp`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `torch`
- `torch.distributed.distributed_c10d`: _get_default_group
- `torch.testing._internal.common_fsdp`: FSDPTest
- `torch.testing._internal.common_utils`: run_tests


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
python test/distributed/fsdp/test_shard_utils.py
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`test/distributed/fsdp`):

- [`test_fsdp_memory.py_docs.md`](./test_fsdp_memory.py_docs.md)
- [`test_fsdp_mixed_precision.py_docs.md`](./test_fsdp_mixed_precision.py_docs.md)
- [`test_fsdp_uneven.py_docs.md`](./test_fsdp_uneven.py_docs.md)
- [`test_fsdp_dtensor_state_dict.py_docs.md`](./test_fsdp_dtensor_state_dict.py_docs.md)
- [`test_fsdp_tp_integration.py_docs.md`](./test_fsdp_tp_integration.py_docs.md)
- [`test_distributed_checkpoint.py_docs.md`](./test_distributed_checkpoint.py_docs.md)
- [`test_fsdp_multiple_forward.py_docs.md`](./test_fsdp_multiple_forward.py_docs.md)
- [`test_checkpoint_wrapper.py_docs.md`](./test_checkpoint_wrapper.py_docs.md)
- [`test_fsdp_clip_grad_norm.py_docs.md`](./test_fsdp_clip_grad_norm.py_docs.md)
- [`test_fsdp_use_orig_params.py_docs.md`](./test_fsdp_use_orig_params.py_docs.md)


## Cross-References

- **File Documentation**: `test_shard_utils.py_docs.md`
- **Keyword Index**: `test_shard_utils.py_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*

```



## High-Level Overview

This file is part of the PyTorch framework located at `docs/test/distributed/fsdp`.

## Detailed Analysis

### Code Structure


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `docs/test/distributed/fsdp`, which is part of the **testing infrastructure**.



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

This is a test file. Run it with:

```bash
python docs/test/distributed/fsdp/test_shard_utils.py_docs.md
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`docs/test/distributed/fsdp`):

- [`test_fsdp_grad_acc.py_docs.md_docs.md`](./test_fsdp_grad_acc.py_docs.md_docs.md)
- [`test_fsdp_ignored_modules.py_kw.md_docs.md`](./test_fsdp_ignored_modules.py_kw.md_docs.md)
- [`test_fsdp_meta.py_kw.md_docs.md`](./test_fsdp_meta.py_kw.md_docs.md)
- [`test_fsdp_apply.py_docs.md_docs.md`](./test_fsdp_apply.py_docs.md_docs.md)
- [`test_fsdp_tp_integration.py_kw.md_docs.md`](./test_fsdp_tp_integration.py_kw.md_docs.md)
- [`test_fsdp_fx.py_docs.md_docs.md`](./test_fsdp_fx.py_docs.md_docs.md)
- [`test_fsdp_memory.py_kw.md_docs.md`](./test_fsdp_memory.py_kw.md_docs.md)
- [`test_fsdp_apply.py_kw.md_docs.md`](./test_fsdp_apply.py_kw.md_docs.md)
- [`test_fsdp_tp_integration.py_docs.md_docs.md`](./test_fsdp_tp_integration.py_docs.md_docs.md)
- [`test_fsdp_multiple_forward.py_kw.md_docs.md`](./test_fsdp_multiple_forward.py_kw.md_docs.md)


## Cross-References

- **File Documentation**: `test_shard_utils.py_docs.md_docs.md`
- **Keyword Index**: `test_shard_utils.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
