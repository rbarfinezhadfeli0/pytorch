# Documentation: `docs/test/distributed/checkpoint/test_save_load_api.py_docs.md`

## File Metadata

- **Path**: `docs/test/distributed/checkpoint/test_save_load_api.py_docs.md`
- **Size**: 6,323 bytes (6.17 KB)
- **Type**: Markdown Documentation
- **Extension**: `.md`

## File Purpose

This file is part of the **testing infrastructure**. This file is part of the **documentation**. This appears to be a **test file**.

## Original Source

```markdown
# Documentation: `test/distributed/checkpoint/test_save_load_api.py`

## File Metadata

- **Path**: `test/distributed/checkpoint/test_save_load_api.py`
- **Size**: 2,993 bytes (2.92 KB)
- **Type**: Python Source Code
- **Extension**: `.py`

## File Purpose

This file is part of the **testing infrastructure**. This appears to be a **test file**. Can be **executed as a standalone script**.

## Original Source

```python
# Owner(s): ["oncall: distributed"]
import os
from unittest.mock import patch

import torch
import torch.distributed.checkpoint as dcp
import torch.nn as nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.testing._internal.common_utils import run_tests
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    skip_if_lt_x_gpu,
    with_comms,
)
from torch.testing._internal.distributed.checkpoint_utils import with_temp_dir


class MyTestModule(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net1 = nn.Sequential(nn.Linear(8, 16), nn.ReLU())
        self.net2 = nn.Sequential(nn.Linear(16, 32), nn.ReLU())
        self.net3 = nn.Linear(32, 64)
        self.net4 = nn.Sequential(nn.ReLU(), nn.Linear(64, 8))

    def forward(self, x):
        return self.net4(self.net3(self.net2(self.net1(x))))


class TestSaveAndLoadAPI(DTensorTestBase):
    @property
    def world_size(self) -> int:
        return 2

    @with_comms
    @skip_if_lt_x_gpu(4)
    @with_temp_dir
    def test_auto_detect(self):
        model = FSDP(MyTestModule().to(self.device_type))
        device_mesh = init_device_mesh(self.device_type, (self.world_size,))
        model = FSDP(model, device_mesh=device_mesh)
        dcp.save(model.state_dict(), checkpoint_id=os.path.join(self.temp_dir, "first"))
        dcp.load(model.state_dict(), checkpoint_id=os.path.join(self.temp_dir, "first"))

        with patch.object(
            dcp.FileSystemReader, "validate_checkpoint_id", return_value=False
        ):
            with patch.object(
                dcp.FileSystemWriter, "validate_checkpoint_id", return_value=False
            ):
                dcp.save(
                    model.state_dict(),
                    checkpoint_id=os.path.join(self.temp_dir, "second"),
                )
                dcp.load(
                    model.state_dict(),
                    checkpoint_id=os.path.join(self.temp_dir, "second"),
                )

        with self.assertRaisesRegex(RuntimeError, "Cannot detect"):
            dcp.save(model.state_dict(), checkpoint_id="abc://abc.abc")
        with self.assertRaisesRegex(RuntimeError, "Cannot detect"):
            dcp.load(model.state_dict(), checkpoint_id="abc://abc.abc")

    @with_comms
    @skip_if_lt_x_gpu(2)
    def test_assert_same_keys(self):
        """Test the `_assert_same_keys` function."""
        model = MyTestModule()
        state_dict = model.state_dict()
        # Check across ranks; expect true
        dcp.utils._assert_same_keys(state_dict)

        # Introduces difference; expect false
        if self.rank == 0:
            state_dict["abc"] = torch.rand(1)
        else:
            state_dict["def"] = torch.rand(1)

        with self.assertRaises(AssertionError):
            dcp.utils._assert_same_keys(state_dict)


if __name__ == "__main__":
    run_tests()

```



## High-Level Overview


This Python file contains 2 class(es) and 5 function(s).

## Detailed Analysis

### Code Structure

**Classes defined**: `MyTestModule`, `TestSaveAndLoadAPI`

**Functions defined**: `__init__`, `forward`, `world_size`, `test_auto_detect`, `test_assert_same_keys`

**Key imports**: os, patch, torch, torch.distributed.checkpoint as dcp, torch.nn as nn, init_device_mesh, FullyShardedDataParallel as FSDP, run_tests, with_temp_dir


*For complete code details, see the Original Source section above.*


## Architecture & Design

### Role in PyTorch Architecture

This file is located in `test/distributed/checkpoint`, which is part of the **testing infrastructure**.



## Dependencies

### Import Dependencies

This file imports:

- `os`
- `unittest.mock`: patch
- `torch`
- `torch.distributed.checkpoint as dcp`
- `torch.nn as nn`
- `torch.distributed.device_mesh`: init_device_mesh
- `torch.distributed.fsdp`: FullyShardedDataParallel as FSDP
- `torch.testing._internal.common_utils`: run_tests
- `torch.testing._internal.distributed.checkpoint_utils`: with_temp_dir


## Code Patterns & Idioms

### Common Patterns

- **Object-Oriented Design**: Uses classes and constructors
- **Neural Network**: Defines or uses PyTorch neural network components


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
python test/distributed/checkpoint/test_save_load_api.py
```

### Usage Examples

*See the source code and related test files for usage examples.*


## Related Files

### Related Files

Files in the same folder (`test/distributed/checkpoint`):

- [`test_format_utils.py_docs.md`](./test_format_utils.py_docs.md)
- [`test_pg_transport.py_docs.md`](./test_pg_transport.py_docs.md)
- [`test_async_process_executor.py_docs.md`](./test_async_process_executor.py_docs.md)
- [`test_file_system_checkpoint.py_docs.md`](./test_file_system_checkpoint.py_docs.md)
- [`test_nested_dict.py_docs.md`](./test_nested_dict.py_docs.md)
- [`test_hf_storage.py_docs.md`](./test_hf_storage.py_docs.md)
- [`test_hf_safetensor_e2e.py_docs.md`](./test_hf_safetensor_e2e.py_docs.md)
- [`test_fsdp_optim_state.py_docs.md`](./test_fsdp_optim_state.py_docs.md)
- [`test_state_dict_stager.py_docs.md`](./test_state_dict_stager.py_docs.md)


## Cross-References

- **File Documentation**: `test_save_load_api.py_docs.md`
- **Keyword Index**: `test_save_load_api.py_kw.md`
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

- **Object-Oriented Design**: Uses classes and constructors
- **Neural Network**: Defines or uses PyTorch neural network components


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
python docs/test/distributed/checkpoint/test_save_load_api.py_docs.md
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

- **File Documentation**: `test_save_load_api.py_docs.md_docs.md`
- **Keyword Index**: `test_save_load_api.py_docs.md_kw.md`
- **Folder Index**: `index.md`
- **Folder Documentation**: `doc.md`

---

*Generated by PyTorch Repository Documentation System*
